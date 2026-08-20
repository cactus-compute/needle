import os
import pickle
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np

from .tokenizer import get_tokenizer, BOS_ID, EOS_ID, PAD_ID
from .architecture import SimpleAttentionNetwork, TransformerConfig

CHECKPOINT_FORMAT_VERSION = 2
BUF_BUCKET = 128
_decode_fn_cache = {}


def average_checkpoints(paths, out):
    acc, meta, steps = None, None, []

    def add(dst, src, scale):
        for k, v in src.items():
            if isinstance(v, dict):
                add(dst.setdefault(k, {}), v, scale)
            else:
                arr = np.asarray(v, np.float32) * scale
                dst[k] = dst[k] + arr if k in dst else arr

    scale = 1.0 / len(paths)
    for p in paths:
        with open(p, "rb") as f:
            ckpt = pickle.load(f)
        if ckpt.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"{p} is not a format-v{CHECKPOINT_FORMAT_VERSION} checkpoint")
        acc = acc if acc is not None else {}
        add(acc, ckpt["params"], scale)
        meta = ckpt
        steps.append(ckpt.get("step"))

    def to_fp16(t):
        return {k: to_fp16(v) if isinstance(v, dict) else v.astype(np.float16)
                for k, v in t.items()}

    with open(out, "wb") as f:
        pickle.dump({
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "params": to_fp16(acc),
            "config": meta["config"],
            "step": meta.get("step"),
            "run": {**(meta.get("run") or {}), "averaged_from": steps},
        }, f)
    print(f"Averaged {len(paths)} checkpoints (steps {steps}) -> {out}")


def head_output_name(checkpoint, suffix):
    stem = os.path.splitext(os.path.basename(checkpoint))[0]
    stem = re.sub(r"_step_\d+$", "", stem)
    return f"{stem}_{suffix}"


def load_checkpoint(path, return_run=False):
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        from .tokenizer import HF_REPO
        print(f"  {'fetch':<9} {path}  downloading from Hugging Face", flush=True)
        rel = os.path.normpath(path).replace(os.sep, "/")
        names = []
        if not os.path.isabs(path):
            names.append(rel if rel.startswith("checkpoints/") else "checkpoints/" + rel)
        flat = "checkpoints/" + os.path.basename(path)
        if flat not in names:
            names.append(flat)
        err = None
        for name in names:
            try:
                path = hf_hub_download(HF_REPO, name, repo_type="model", local_dir=".")
                break
            except EntryNotFoundError as e:
                err = e
        else:
            raise err
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    version = ckpt.get("format_version") if isinstance(ckpt, dict) else None
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"{path} is not a format-v{CHECKPOINT_FORMAT_VERSION} checkpoint "
            f"(got format_version={version!r}). Old encoder-decoder/tool-calling "
            f"checkpoints are incompatible with this branch."
        )
    config = ckpt["config"]
    if isinstance(config, dict):
        config = TransformerConfig(**config)
    if return_run:
        return ckpt["params"], config, ckpt.get("run") or {}
    return ckpt["params"], config


def _get_decode_fn(model, buf_len):
    key = (id(model), buf_len)
    if key not in _decode_fn_cache:
        @jax.jit
        def decode_fn(params, tokens):
            return model.apply({"params": params}, tokens)
        _decode_fn_cache[key] = decode_fn
    return _decode_fn_cache[key]


def _step_stats(step_logits):
    logp = jax.nn.log_softmax(step_logits.astype(jnp.float32), axis=-1)
    p = jnp.exp(logp)
    chosen = jnp.max(p, axis=-1)
    entropy = -jnp.sum(p * logp, axis=-1)
    return np.asarray(chosen), np.asarray(entropy)


def batch_generate(model, params, tokenizer, prompts, max_new_tokens=96,
                   return_signals=False):
    enc = [([BOS_ID] + tokenizer.encode(p))[:model.config.max_seq_len - 1] for p in prompts]
    plens = [len(e) for e in enc]
    buf_len = min(model.config.max_seq_len,
                  -(-(max(plens) + max_new_tokens) // BUF_BUCKET) * BUF_BUCKET)
    B = len(prompts)
    buffer = np.full((B, buf_len), PAD_ID, dtype=np.int32)
    for i, e in enumerate(enc):
        buffer[i, :plens[i]] = e[:buf_len]
    decode_fn = _get_decode_fn(model, buf_len)
    params = jax.device_put(params)

    done = [plens[i] >= buf_len for i in range(B)]
    gen = [[] for _ in range(B)]
    probs = [[] for _ in range(B)]
    ents = [[] for _ in range(B)]
    for pos in range(min(plens) - 1, buf_len - 1):
        logits = decode_fn(params, jnp.asarray(buffer))
        step = logits[:, pos]
        nxt = np.asarray(jnp.argmax(step, axis=-1))
        chosen_p, chosen_h = _step_stats(step) if return_signals else (None, None)
        for i in range(B):
            if done[i] or pos < plens[i] - 1:
                continue
            t = int(nxt[i])
            if t == EOS_ID:
                done[i] = True
                continue
            gen[i].append(t)
            if return_signals:
                probs[i].append(float(chosen_p[i]))
                ents[i].append(float(chosen_h[i]))
            buffer[i, pos + 1] = t
            if len(gen[i]) >= max_new_tokens:
                done[i] = True
        if all(done):
            break

    texts = [tokenizer.decode(g) for g in gen]
    if not return_signals:
        return texts

    out = []
    for i in range(B):
        pr = probs[i]
        out.append({
            "text": texts[i],
            "tokens": gen[i],
            "token_probs": pr,
            "token_entropies": ents[i],
            "min_prob": min(pr) if pr else 0.0,
            "mean_logprob": float(np.mean(np.log(np.clip(pr, 1e-12, 1.0)))) if pr else float("-inf"),
            "max_entropy": max(ents[i]) if ents[i] else 0.0,
        })
    return out


def generate(model, params, tokenizer, prompt, max_new_tokens=256, temperature=0.0,
             seed=0, stream=True):
    prompt_ids = [BOS_ID] + tokenizer.encode(prompt)
    buf_len = min(model.config.max_seq_len, len(prompt_ids) + max_new_tokens)
    if len(prompt_ids) >= buf_len:
        raise ValueError(f"Prompt ({len(prompt_ids)} tokens) does not fit in max_seq_len={model.config.max_seq_len}")

    buffer = jnp.full((1, buf_len), PAD_ID, dtype=jnp.int32)
    buffer = buffer.at[0, :len(prompt_ids)].set(jnp.array(prompt_ids, dtype=jnp.int32))
    decode_fn = _get_decode_fn(model, buf_len)
    rng = jax.random.PRNGKey(seed)

    generated = []
    printed = ""
    for pos in range(len(prompt_ids) - 1, buf_len - 1):
        logits = decode_fn(params, buffer)[0, pos]
        if temperature <= 0.0:
            next_token = int(jnp.argmax(logits))
        else:
            rng, sample_rng = jax.random.split(rng)
            next_token = int(jax.random.categorical(sample_rng, logits / temperature))
        if next_token == EOS_ID:
            break
        generated.append(next_token)
        buffer = buffer.at[0, pos + 1].set(next_token)
        if stream:
            text = tokenizer.decode(generated)
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text

    text = tokenizer.decode(generated)
    if stream:
        sys.stdout.write(text[len(printed):] + "\n")
        sys.stdout.flush()
    return text


def main(args):
    params, config = load_checkpoint(args.checkpoint)
    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer(config.vocab_size)

    prompt = args.query or "The most surprising thing about"
    print(f"prompt: {prompt!r}")
    generate(
        model, params, tokenizer, prompt,
        max_new_tokens=args.max_len,
        temperature=getattr(args, "temperature", 0.0),
        seed=args.seed,
    )
