"""Categorize tool-call argument-value failures to quantify what a
draft-correction second pass could actually fix.

Runs unconstrained generation on a stratified single+multi-call val sample,
then sorts each argument-value mismatch into one of these buckets:

  correctable      ref value appears as a substring of the query (a second
                   pass can re-extract it)
  canonicalization ref and pred match after lowercase + whitespace collapse
                   + stripping a small set of prefixes ("at ", "today at ",
                   "on "). Data-inconsistency noise, not model error.
  cross_lingual    ref and pred use different scripts (Latin vs Cyrillic,
                   etc.) — translation mismatch, usually annotation noise.
  unrecoverable    neither ref nor pred appear in the query — a draft
                   corrector can't fix these without external info.
  missing_key      pred omitted a key the ref requires
  extra_key        pred emitted a key the ref doesn't require

Usage:
    python -m src.categorize_failures --checkpoint checkpoints/needle.pkl
    python -m src.categorize_failures --checkpoint needle.pkl --num-samples 500
"""

import argparse
import json
import re
import unicodedata
from collections import Counter

import jax
import numpy as np

from .data import get_tokenizer, load_tool_calls
from .model import EncoderDecoderTransformer
from .run import generate_batch, load_checkpoint
from .tokenizer import to_snake_case


_CANONICAL_PREFIXES = ("today at ", "tomorrow at ", "tonight at ",
                       "this evening at ", "at ", "on ", "the ")


def _strip_canonical(s):
    """Lowercase, collapse whitespace, strip one canonical prefix."""
    if not isinstance(s, str):
        return s
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    for p in _CANONICAL_PREFIXES:
        if s.startswith(p):
            s = s[len(p):].strip()
            break
    return s


def _script_of(s):
    """Return dominant script of alphabetic chars, or 'other' / 'empty'."""
    if not isinstance(s, str) or not s:
        return "empty"
    scripts = set()
    for ch in s:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            continue
        if "LATIN" in name:
            scripts.add("latin")
        elif "CYRILLIC" in name:
            scripts.add("cyrillic")
        elif "GREEK" in name:
            scripts.add("greek")
        elif "ARABIC" in name:
            scripts.add("arabic")
        elif "HEBREW" in name:
            scripts.add("hebrew")
        elif "HANGUL" in name:
            scripts.add("hangul")
        elif "DEVANAGARI" in name:
            scripts.add("devanagari")
        elif "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name:
            scripts.add("cjk")
        else:
            scripts.add("other")
    if not scripts:
        return "empty"
    if len(scripts) == 1:
        return next(iter(scripts))
    return "mixed"


def _value_in_query(value, query):
    """True if value appears in query (case-insensitive for strings)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return str(value) in query
    if isinstance(value, str):
        v = value.strip().lower()
        if len(v) < 2:
            return False
        return v in query.lower()
    return False


def categorize_value_mismatch(query, ref_value, pred_value):
    if isinstance(ref_value, str) and isinstance(pred_value, str):
        if _strip_canonical(ref_value) == _strip_canonical(pred_value):
            return "canonicalization"
        rs, ps = _script_of(ref_value), _script_of(pred_value)
        if rs != ps and rs not in ("empty", "other", "mixed") and ps not in ("empty", "other", "mixed"):
            return "cross_lingual"
    if _value_in_query(ref_value, query):
        return "correctable"
    return "unrecoverable"


def categorize_pool(model, params, tokenizer, examples, max_gen_len, max_enc_len,
                    batch_size=32):
    """Generate predictions for a pool and categorize each value mismatch.

    Returns (bucket_counts, bucket_examples).
    """
    queries = [ex["query"] for ex in examples]
    tools_list = [ex["tools"] for ex in examples]

    all_preds = []
    for i in range(0, len(examples), batch_size):
        chunk_q = queries[i:i + batch_size]
        chunk_t = tools_list[i:i + batch_size]
        preds = generate_batch(
            model, params, tokenizer, chunk_q, chunk_t,
            max_gen_len=max_gen_len, max_enc_len=max_enc_len,
            normalize=True, constrained=False,
        )
        all_preds.extend(preds)

    bucket_counts = Counter()
    bucket_examples = {b: [] for b in
                       ("correctable", "canonicalization", "cross_lingual",
                        "unrecoverable", "missing_key", "extra_key")}

    for ex, pred_text in zip(examples, all_preds):
        query = ex["query"]
        try:
            ref_calls = json.loads(ex["answers"])
        except (ValueError, TypeError):
            continue
        try:
            pred_calls = json.loads(pred_text)
            if isinstance(pred_calls, dict):
                pred_calls = [pred_calls]
            elif not isinstance(pred_calls, list):
                continue
        except (ValueError, TypeError):
            continue
        if not isinstance(ref_calls, list):
            continue

        for c in ref_calls:
            if isinstance(c, dict) and "name" in c:
                c["name"] = to_snake_case(c["name"])
        for c in pred_calls:
            if isinstance(c, dict) and "name" in c:
                c["name"] = to_snake_case(c["name"])

        ref_by_name = {}
        for c in ref_calls:
            if isinstance(c, dict) and "name" in c:
                ref_by_name.setdefault(c["name"], []).append(c.get("arguments", {}) or {})

        for c in pred_calls:
            if not isinstance(c, dict) or "name" not in c:
                continue
            name = c["name"]
            if name not in ref_by_name:
                continue  # tool-level mismatch — outside this script's scope
            pred_args = c.get("arguments", {}) or {}
            ref_args = ref_by_name[name][0]
            if not isinstance(ref_args, dict):
                continue

            all_keys = set(pred_args.keys()) | set(ref_args.keys())
            for key in all_keys:
                has_pred = key in pred_args
                has_ref = key in ref_args

                if has_ref and not has_pred:
                    bucket_counts["missing_key"] += 1
                    if len(bucket_examples["missing_key"]) < 5:
                        bucket_examples["missing_key"].append(
                            {"query": query, "tool": name, "key": key,
                             "ref": ref_args[key], "pred": "<MISSING>"})
                    continue
                if has_pred and not has_ref:
                    bucket_counts["extra_key"] += 1
                    if len(bucket_examples["extra_key"]) < 5:
                        bucket_examples["extra_key"].append(
                            {"query": query, "tool": name, "key": key,
                             "ref": "<MISSING>", "pred": pred_args[key]})
                    continue

                rv, pv = ref_args[key], pred_args[key]
                if json.dumps(pv, sort_keys=True) == json.dumps(rv, sort_keys=True):
                    continue
                cat = categorize_value_mismatch(query, rv, pv)
                bucket_counts[cat] += 1
                if len(bucket_examples[cat]) < 5:
                    bucket_examples[cat].append(
                        {"query": query, "tool": name, "key": key,
                         "ref": rv, "pred": pv})

    return bucket_counts, bucket_examples


def _print_block(label, counts, examples):
    total = sum(counts.values())
    print(f"\n=== {label} ===")
    print(f"  Total value-level mismatches: {total}")
    if total == 0:
        return
    print()
    order = ("correctable", "canonicalization", "cross_lingual",
             "unrecoverable", "missing_key", "extra_key")
    for b in order:
        n = counts.get(b, 0)
        pct = 100 * n / total
        bar = "█" * int(pct / 2)
        print(f"  {b:<18} {n:>5}  ({pct:>5.1f}%)  {bar}")

    for b in order:
        if not examples.get(b):
            continue
        print(f"\n  -- {b} examples --")
        for e in examples[b]:
            q = e["query"][:110]
            print(f"    Q: {q}")
            print(f"    {e['tool']}.{e['key']}")
            print(f"      ref : {e['ref']!r}")
            print(f"      pred: {e['pred']!r}")


def main(args):
    print(f"Loading checkpoint {args.checkpoint} ...")
    params, config = load_checkpoint(args.checkpoint)
    model = EncoderDecoderTransformer(config)
    tokenizer = get_tokenizer()

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Parameters: {param_count:,}")

    val_ds = load_tool_calls("validation")
    rng = np.random.RandomState(42)

    pools = {"single": [], "multi": []}
    for k in range(len(val_ds)):
        ex = val_ds[k]
        try:
            answers = json.loads(ex["answers"])
        except (ValueError, TypeError):
            continue
        if not isinstance(answers, list) or not answers:
            continue
        pools["single" if len(answers) == 1 else "multi"].append(ex)

    for name in pools:
        rng.shuffle(pools[name])
        pools[name] = pools[name][:args.num_samples]
        print(f"  {name}-call sample: {len(pools[name])}")

    combined_counts = Counter()
    for name in ("single", "multi"):
        print(f"\n--- Running generation on {len(pools[name])} {name}-call examples ...")
        counts, examples = categorize_pool(
            model, params, tokenizer, pools[name],
            max_gen_len=args.max_gen_len,
            max_enc_len=args.max_enc_len,
            batch_size=args.batch_size,
        )
        _print_block(f"{name.upper()}-CALL", counts, examples)
        combined_counts += counts

    total = sum(combined_counts.values())
    if total == 0:
        return
    print(f"\n=== COMBINED ({total} value mismatches) ===")
    order = ("correctable", "canonicalization", "cross_lingual",
             "unrecoverable", "missing_key", "extra_key")
    for b in order:
        n = combined_counts.get(b, 0)
        pct = 100 * n / total
        bar = "█" * int(pct / 2)
        print(f"  {b:<18} {n:>5}  ({pct:>5.1f}%)  {bar}")

    correctable = combined_counts.get("correctable", 0)
    missing = combined_counts.get("missing_key", 0)
    print(f"\n  Upper bound for a perfect draft-corrector: "
          f"{100 * (correctable + missing) / total:.1f}% of value errors "
          f"({correctable + missing}/{total})")
    noise = combined_counts.get("canonicalization", 0) + combined_counts.get("cross_lingual", 0)
    print(f"  Fixable by cleaning labels (no model change): "
          f"{100 * noise / total:.1f}% ({noise}/{total})")
    unrecov = combined_counts.get("unrecoverable", 0)
    print(f"  Floor a corrector can't beat: "
          f"{100 * unrecov / total:.1f}% ({unrecov}/{total})")


def parse_args():
    p = argparse.ArgumentParser(
        description="Categorize tool-call value failures by correctability")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=500,
                   help="Samples per pool (single-call + multi-call)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-enc-len", type=int, default=1024)
    p.add_argument("--max-gen-len", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
