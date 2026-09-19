import os
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from .tokenizer import get_tokenizer, BOS_ID, EOS_ID, PAD_ID
from .architecture import SimpleAttentionNetwork, TransformerConfig
from .checkpoints import read_checkpoint

CHECKPOINT_FORMAT_VERSION = 2
BUF_BUCKET = 128
_decode_fn_cache = {}



def load_checkpoint(path, return_run=False):
    if not os.path.exists(path):
        from ..agent import fetch
        print(f"  {'fetch':<9} {path}  downloading from Hugging Face", flush=True)
        dest_dir = os.path.dirname(path) or fetch.CHECKPOINT_PREFIX
        path = fetch.fetch_checkpoint(os.path.basename(path), dest_dir, generation=3)
    ckpt = read_checkpoint(path)
    version = ckpt.get("format_version") if isinstance(ckpt, dict) else None
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"{path} is not a format-v{CHECKPOINT_FORMAT_VERSION} checkpoint "
            f"(got format_version={version!r}). Old encoder-decoder/tool-calling "
            f"checkpoints are incompatible with this branch."
        )
    saved = ckpt["config"] if isinstance(ckpt["config"], dict) else dict(vars(ckpt["config"]))
    if "attn_dim" in saved and "qk_head_dim" not in saved:
        raise ValueError(
            f"{path} is a Needle 2 checkpoint; this package fine-tunes and builds Needle 3. "
            "Use cactus-needle 2.x for it: pip install 'cactus-needle<3'")
    config = TransformerConfig.from_saved(saved)
    params = {k: v for k, v in ckpt["params"].items() if not k.startswith("mtp_")}
    if return_run:
        return params, config, ckpt.get("run") or {}
    return params, config


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


def build_prompt(query, tools=None):
    if not tools:
        return query
    from .finetune import render_example
    prompt, _ = render_example({"query": query, "tools": tools})
    return prompt


def main(args):
    params, config = load_checkpoint(args.checkpoint)
    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer(config.vocab_size)

    prompt = args.query or "The most surprising thing about"
    if getattr(args, "tools", None):
        with open(args.tools) as handle:
            prompt = build_prompt(prompt, json.load(handle))
    print(f"prompt: {prompt!r}")
    generate(
        model, params, tokenizer, prompt,
        max_new_tokens=args.max_len,
        temperature=args.temperature,
        seed=args.seed,
    )
