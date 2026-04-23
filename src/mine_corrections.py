"""Mine self-correction training pairs from the SFT checkpoint.

For each training prompt, sample K completions at temperature T from the
frozen SFT model, classify each against the gold, and write
(query, tools, draft, gold, category) tuples to JSONL for downstream
tokenization.

Categories written: value_wrong, args_wrong, correct.
Categories dropped: tool_wrong, unparseable. They are counted in the
summary so you can see the distribution, but a corrector can't recover
tool-selection or parse errors from a draft, so we don't train on them.

Resumability: the script writes progress to <output>.cursor. On restart
it reads the cursor, reopens the output in append mode, and resumes from
the next unprocessed prompt. Crashing mid-chunk can cause up to one
batch of duplicates — acceptable for mining, downstream dedup is cheap.

Usage:
    python -m src.mine_corrections \\
        --checkpoint checkpoints/needle.pkl \\
        --output data/corrections.jsonl \\
        --max-samples 100000 \\
        --k 1 --temperature 0.7
"""

import argparse
import json
import os
import time
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .data import get_tokenizer, load_tool_calls
from .distributed import _replicate
from .model import (
    EncoderDecoderTransformer,
    make_causal_mask,
    make_padding_mask,
)
from .run import _build_encoder_input, load_checkpoint
from .tokenizer import EOS_ID, PAD_ID, to_snake_case


_HF_SFT_REPO = "Cactus-Compute/needle"


def _maybe_download_ckpt(path):
    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    local_dir = os.path.dirname(path) or "checkpoints"
    os.makedirs(local_dir, exist_ok=True)
    print(f"  Downloading {os.path.basename(path)} from {_HF_SFT_REPO} ...")
    return hf_hub_download(
        repo_id=_HF_SFT_REPO,
        filename=os.path.basename(path),
        repo_type="model",
        local_dir=local_dir,
    )


def _call_key_set(calls):
    keys = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        name = to_snake_case(c.get("name", ""))
        args = c.get("arguments", {}) or {}
        keys.append(json.dumps({"name": name, "arguments": args}, sort_keys=True))
    return sorted(keys)


def _classify(pred_text, ref_text):
    """Return one of: correct, value_wrong, args_wrong, tool_wrong, unparseable."""
    if pred_text.startswith("<tool_call>"):
        pred_text = pred_text[len("<tool_call>"):]
    try:
        pred_calls = json.loads(pred_text)
        if isinstance(pred_calls, dict):
            pred_calls = [pred_calls]
        elif not isinstance(pred_calls, list):
            return "unparseable"
    except (json.JSONDecodeError, TypeError):
        return "unparseable"
    try:
        ref_calls = json.loads(ref_text)
        if not isinstance(ref_calls, list):
            return "unparseable"
    except (json.JSONDecodeError, TypeError):
        return "unparseable"

    for c in pred_calls:
        if isinstance(c, dict) and isinstance(c.get("name"), str):
            c["name"] = to_snake_case(c["name"])
        else:
            return "unparseable"
    for c in ref_calls:
        if isinstance(c, dict) and isinstance(c.get("name"), str):
            c["name"] = to_snake_case(c["name"])

    if _call_key_set(pred_calls) == _call_key_set(ref_calls):
        return "correct"

    ref_names = sorted(c["name"] for c in ref_calls if isinstance(c, dict) and "name" in c)
    pred_names = sorted(c["name"] for c in pred_calls if isinstance(c, dict) and "name" in c)
    if ref_names != pred_names:
        return "tool_wrong"

    # Tool names match — check whether arg keys or just values differ
    ref_by_name, pred_by_name = {}, {}
    for c in ref_calls:
        if isinstance(c, dict) and "name" in c:
            ref_by_name.setdefault(c["name"], []).append(c.get("arguments", {}) or {})
    for c in pred_calls:
        if isinstance(c, dict) and "name" in c:
            pred_by_name.setdefault(c["name"], []).append(c.get("arguments", {}) or {})

    for name, ref_list in ref_by_name.items():
        pred_list = pred_by_name.get(name, [])
        for ra, pa in zip(ref_list, pred_list):
            if set((ra or {}).keys()) != set((pa or {}).keys()):
                return "args_wrong"
    return "value_wrong"


def _make_sampling_fns(model, max_gen_len, temperature):
    """Build pmapped encode + sample-step functions with temperature baked in."""
    inv_t = 1.0 / max(temperature, 1e-5)

    @jax.pmap
    def p_encode(params, enc_input):
        src_mask = make_padding_mask(enc_input, PAD_ID)
        return model.apply(
            {"params": params}, enc_input,
            src_mask=src_mask, method="encode",
        )

    @jax.pmap
    def p_sample_step(params, dec_buffer, encoder_out, cross_mask,
                      rng, step_idx, finished):
        sm = make_causal_mask(max_gen_len)
        logits = model.apply(
            {"params": params}, dec_buffer, encoder_out,
            self_mask=sm, cross_mask=cross_mask, method="decode",
        )
        next_logits = jax.lax.dynamic_index_in_dim(
            logits, step_idx, axis=1, keepdims=False,
        ) * inv_t
        next_tokens = jax.random.categorical(rng, next_logits, axis=-1)
        next_tokens = jnp.where(finished, PAD_ID, next_tokens)
        new_finished = finished | (next_tokens == EOS_ID)
        return next_tokens, new_finished

    return p_encode, p_sample_step


def _sample_completions(p_encode, p_sample_step, r_params, enc_sharded,
                         num_devices, max_gen_len, rng):
    encoder_out, cross_mask = p_encode(r_params, enc_sharded)
    per_device = enc_sharded.shape[1]

    dec_buffer = np.full((num_devices, per_device, max_gen_len), PAD_ID, dtype=np.int32)
    dec_buffer[:, :, 0] = EOS_ID
    dec_buffer = jnp.asarray(dec_buffer)
    finished = jnp.zeros((num_devices, per_device), dtype=jnp.bool_)

    for step in range(max_gen_len - 1):
        rng, sub = jax.random.split(rng)
        sub_per = jax.random.split(sub, num_devices)
        step_idx = jnp.full((num_devices,), step, dtype=jnp.int32)
        next_tokens, finished = p_sample_step(
            r_params, dec_buffer, encoder_out, cross_mask,
            sub_per, step_idx, finished,
        )
        dec_buffer = dec_buffer.at[:, :, step + 1].set(next_tokens)

    return np.asarray(dec_buffer), rng


def _decode_sample(tokenizer, token_row):
    """Decode a sampled row and strip the leading <tool_call> task marker.

    Matches run.py's `generate_batch` post-processing: strip `<tool_call>`
    if present so the stored draft is the raw JSON. Without this the draft
    re-encodes to a double TOOL_CALL_ID at train time while inference (which
    strips it) sees a single one — a train/inference format mismatch.
    """
    sampled = token_row[1:]
    eos_positions = np.where(sampled == EOS_ID)[0]
    end = int(eos_positions[0]) if len(eos_positions) > 0 else len(sampled)
    ids = [int(t) for t in sampled[:end] if t != PAD_ID]
    text = tokenizer.decode(ids)
    if text.startswith("<tool_call>"):
        text = text[len("<tool_call>"):]
    return text


def _read_cursor(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        try:
            return int(f.read().strip())
        except ValueError:
            return 0


def _write_cursor(path, value):
    with open(path, "w") as f:
        f.write(str(value))


def mine(args):
    if os.path.exists("/dev/accel0"):
        jax.distributed.initialize()

    num_devices = jax.local_device_count()
    print(f"Devices: {num_devices} local, {jax.device_count()} total")

    ckpt_path = _maybe_download_ckpt(args.checkpoint)
    print(f"Loading checkpoint {ckpt_path} ...")
    params, config = load_checkpoint(ckpt_path)
    model = EncoderDecoderTransformer(config)
    tokenizer = get_tokenizer()

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Parameters: {param_count:,}")
    print(f"  Config: d={config.d_model}, heads={config.num_heads}, "
          f"layers={config.num_encoder_layers}/{config.num_decoder_layers}")

    r_params = _replicate(params)
    del params

    p_encode, p_sample_step = _make_sampling_fns(
        model, args.max_gen_len, args.temperature)

    print("Loading train split ...")
    train_ds = load_tool_calls("train")
    n_total = len(train_ds)
    print(f"  {n_total:,} rows (empty-answer examples skipped inline)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cursor_path = args.output + ".cursor"
    start_idx = _read_cursor(cursor_path)
    if start_idx > 0:
        print(f"  Resuming from dataset index {start_idx}")

    B = args.batch_size
    K = args.k
    BK = B * K
    if BK % num_devices != 0:
        raise ValueError(
            f"B*K ({BK}) must be divisible by num_devices ({num_devices}). "
            f"Try --batch-size={num_devices * (BK // num_devices) // K}")
    per_device = BK // num_devices

    print(f"  B={B} prompts, K={K} samples/prompt, BK={BK} ({per_device}/device)")
    print(f"  temperature={args.temperature}, max_gen_len={args.max_gen_len}")

    rng = jax.random.PRNGKey(args.seed)
    stats = Counter()

    out_f = open(args.output, "a", encoding="utf-8", buffering=1)
    pbar_total = args.max_samples if args.max_samples else (n_total - start_idx)
    pbar = tqdm(total=pbar_total, desc="Mining", unit="prompts", initial=0)
    t0 = time.perf_counter()
    processed = 0
    last_ds_idx = start_idx - 1

    def _iter_prompts():
        for i in range(start_idx, n_total):
            ex = train_ds[int(i)]
            if ex["answers"].strip() in ("", "[]"):
                continue
            yield i, ex

    from itertools import islice
    prompt_iter = _iter_prompts()

    try:
        while True:
            pairs = list(islice(prompt_iter, B))
            if not pairs:
                break
            batch_ds_idx = [p[0] for p in pairs]
            batch = [p[1] for p in pairs]
            actual_B = len(batch)
            last_ds_idx = batch_ds_idx[-1]

            # Pad batch to full B by repeating last example (outputs ignored)
            while len(batch) < B:
                batch.append(batch[-1])

            enc_lists = [
                _build_encoder_input(tokenizer, ex["query"], ex["tools"],
                                     args.max_enc_len)
                for ex in batch
            ]
            enc_np = np.full((B, args.max_enc_len), PAD_ID, dtype=np.int32)
            for i, toks in enumerate(enc_lists):
                enc_np[i, :len(toks)] = toks

            enc_rep = np.repeat(enc_np, K, axis=0)
            enc_sharded = jnp.asarray(
                enc_rep.reshape(num_devices, per_device, args.max_enc_len))

            dec_buffer_np, rng = _sample_completions(
                p_encode, p_sample_step, r_params, enc_sharded,
                num_devices, args.max_gen_len, rng,
            )
            dec_buffer_flat = dec_buffer_np.reshape(BK, args.max_gen_len)

            for prompt_idx in range(actual_B):
                prompt = batch[prompt_idx]
                kept_correct = False
                for k in range(K):
                    i = prompt_idx * K + k
                    draft = _decode_sample(tokenizer, dec_buffer_flat[i])
                    category = _classify(draft, prompt["answers"])
                    stats[category] += 1

                    if category in ("value_wrong", "args_wrong"):
                        out_f.write(json.dumps({
                            "query": prompt["query"],
                            "tools": prompt["tools"],
                            "draft": draft,
                            "answers": prompt["answers"],
                            "category": category,
                            "source_idx": int(batch_ds_idx[prompt_idx]),
                        }, ensure_ascii=False) + "\n")
                    elif category == "correct" and not kept_correct:
                        kept_correct = True
                        out_f.write(json.dumps({
                            "query": prompt["query"],
                            "tools": prompt["tools"],
                            "draft": draft,
                            "answers": prompt["answers"],
                            "category": "correct",
                            "source_idx": int(batch_ds_idx[prompt_idx]),
                        }, ensure_ascii=False) + "\n")

            _write_cursor(cursor_path, last_ds_idx + 1)
            pbar.update(actual_B)
            processed += actual_B

            if processed % (B * 10) == 0:
                elapsed = time.perf_counter() - t0
                rate = processed / max(elapsed, 1e-6)
                total_gen = max(sum(stats.values()), 1)
                pbar.set_postfix(
                    val=f"{stats['value_wrong']} ({100*stats['value_wrong']/total_gen:.0f}%)",
                    args=f"{stats['args_wrong']} ({100*stats['args_wrong']/total_gen:.0f}%)",
                    corr=f"{stats['correct']} ({100*stats['correct']/total_gen:.0f}%)",
                    tool=f"{stats['tool_wrong']} ({100*stats['tool_wrong']/total_gen:.0f}%)",
                    rate=f"{rate:.1f}/s",
                )

            if args.max_samples and processed >= args.max_samples:
                break
    finally:
        out_f.close()
        pbar.close()
        total = sum(stats.values())
        written = stats["value_wrong"] + stats["args_wrong"] + stats["correct"]
        print("\n=== Final stats ===")
        for k in ("correct", "value_wrong", "args_wrong", "tool_wrong",
                  "unparseable"):
            v = stats.get(k, 0)
            pct = 100 * v / max(total, 1)
            print(f"  {k:<14} {v:>10}  ({pct:>5.1f}%)")
        print(f"  total generations  {total:>10}")
        print(f"  rows written       {written:>10}  -> {args.output}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Mine self-correction pairs from an SFT checkpoint")
    p.add_argument("--checkpoint", type=str, default="checkpoints/needle.pkl")
    p.add_argument("--output", type=str, default="data/corrections.jsonl")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit number of training prompts (default: all)")
    p.add_argument("--k", type=int, default=1,
                   help="Samples per prompt (default: 1)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Prompts per forward pass; B*K must divide num_devices")
    p.add_argument("--max-enc-len", type=int, default=1024)
    p.add_argument("--max-gen-len", type=int, default=256,
                   help="Generation cap — most tool calls fit in 256; "
                        "raise if you see truncations")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    mine(parse_args())
