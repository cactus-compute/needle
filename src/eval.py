import argparse
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import get_batches, get_tokenizer, load_tool_calls, prepare_tool_call_pairs, DEFAULT_MAX_ENC_LEN, DEFAULT_MAX_DEC_LEN, DEFAULT_MAX_GEN_LEN
from .model import (
    EncoderDecoderTransformer,
    TransformerConfig,
    make_causal_mask,
    make_padding_mask,
)
from .run import load_checkpoint, _get_decode_fn


def compute_perplexity(model, params, enc_inputs, dec_inputs, dec_targets, batch_size, pad_id, loss_mask=None):
    total_loss = 0.0
    total_tokens = 0

    for batch in get_batches(enc_inputs, dec_inputs, dec_targets, batch_size, shuffle=False, loss_mask=loss_mask):
        if loss_mask is not None:
            src, tgt_in, tgt_out, lm = batch
            src, tgt_in, tgt_out, lm = jnp.array(src), jnp.array(tgt_in), jnp.array(tgt_out), jnp.array(lm)
            mask = lm
        else:
            src, tgt_in, tgt_out = batch
            src, tgt_in, tgt_out = jnp.array(src), jnp.array(tgt_in), jnp.array(tgt_out)
            mask = (tgt_out != pad_id).astype(jnp.float32)

        src_mask = make_padding_mask(src, pad_id)
        tgt_mask = make_causal_mask(tgt_in.shape[1]) & make_padding_mask(tgt_in, pad_id)

        logits = model.apply(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask,
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, tgt_out)
        total_loss += float(jnp.sum(loss * mask))
        total_tokens += int(jnp.sum(mask))

    avg_nll = total_loss / max(total_tokens, 1)
    return math.exp(avg_nll)


def measure_throughput(model, params, tokenizer, num_runs=10, prompt='What is the weather?', max_gen_len=64):
    enc_tokens = tokenizer.encode(prompt)
    enc_input = jnp.array([enc_tokens])
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    src_mask = make_padding_mask(enc_input, pad_id)
    decode_fn = _get_decode_fn(model, max_gen_len)

    # Warmup JIT
    encoder_out, enc_mask = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, method="encode"
    )
    dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
    dec_buffer = dec_buffer.at[0, 0].set(eos_id)
    decode_fn(params, dec_buffer, encoder_out, enc_mask)

    tokens_generated = []
    latencies = []

    for run in range(num_runs):
        rng = jax.random.PRNGKey(run)
        dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
        dec_buffer = dec_buffer.at[0, 0].set(eos_id)

        start = time.perf_counter()
        encoder_out, enc_mask = model.apply(
            {"params": params}, enc_input, src_mask=src_mask, method="encode"
        )
        logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)

        num_tokens = 0
        for i in range(max_gen_len - 1):
            rng, sample_rng = jax.random.split(rng)
            next_token = jax.random.categorical(sample_rng, logits[0, i]).item()

            if next_token == eos_id:
                break

            num_tokens += 1
            dec_buffer = dec_buffer.at[0, i + 1].set(next_token)
            logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)

        elapsed = time.perf_counter() - start
        tokens_generated.append(num_tokens)
        latencies.append(elapsed)

    avg_tokens = np.mean(tokens_generated)
    avg_latency = np.mean(latencies)
    tokens_per_sec = sum(tokens_generated) / sum(latencies)

    return {
        "avg_tokens_generated": avg_tokens,
        "avg_latency_s": avg_latency,
        "tokens_per_second": tokens_per_sec,
    }


def compute_repetition_rate(texts):
    bigram_rep_rates = []
    for text in texts:
        words = text.lower().split()
        if len(words) < 2:
            bigram_rep_rates.append(0.0)
            continue
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        unique = len(set(bigrams))
        bigram_rep_rates.append(1.0 - unique / len(bigrams))
    return float(np.mean(bigram_rep_rates))


def benchmark_generation_quality(model, params, tokenizer, prompts, max_gen_len=128, temperature=0.8):
    from .run import generate

    generations = []
    for i, prompt in enumerate(prompts):
        text = generate(model, params, tokenizer, prompt, max_gen_len, temperature, seed=i, stream=False)
        generations.append(text)

    lengths = [len(tokenizer.encode(t)) for t in generations]
    rep_rate = compute_repetition_rate(generations)

    return {
        "avg_generation_length": float(np.mean(lengths)),
        "min_generation_length": int(np.min(lengths)),
        "max_generation_length": int(np.max(lengths)),
        "bigram_repetition_rate": rep_rate,
        "generations": list(zip(prompts, generations)),
    }


def compute_wer(hypotheses, references):
    """Compute word error rate using edit distance."""
    total_edits = 0
    total_ref_words = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_words = hyp.lower().split()
        ref_words = ref.lower().split()
        n = len(ref_words)
        m = len(hyp_words)

        # DP edit distance
        d = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            d[i][0] = i
        for j in range(m + 1):
            d[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

        total_edits += d[n][m]
        total_ref_words += n

    return total_edits / max(total_ref_words, 1)


def benchmark_tool_calls(model, params, tokenizer, num_samples=200, max_gen_len=DEFAULT_MAX_GEN_LEN, max_enc_len=DEFAULT_MAX_ENC_LEN, constrained=True):
    """Generate tool-call predictions and compute structured metrics."""
    import json
    from .run import generate_batch, normalize_tools, restore_tool_names
    from .data import load_tool_calls, to_snake_case

    ds = load_tool_calls("validation", max_samples=num_samples)

    queries = [ex["query"] for ex in ds]
    tools_list = [ex["tools"] for ex in ds]
    all_preds = generate_batch(model, params, tokenizer, queries, tools_list, max_gen_len=max_gen_len, max_enc_len=max_enc_len, normalize=True, constrained=constrained)

    total = 0
    exact_match = 0
    tp_names = 0
    fp_names = 0
    fn_names = 0
    tp_calls = 0
    fp_calls = 0
    fn_calls = 0
    json_parse_errors = 0
    empty_ref = 0
    empty_pred = 0
    args_correct = 0
    args_total = 0
    halluc_params = 0
    total_pred_params = 0
    missing_params = 0
    total_ref_params = 0
    correct_values = 0
    matched_params = 0
    samples = []

    def call_key(c):
        if not isinstance(c, dict):
            return None
        return json.dumps({"name": c.get("name"), "arguments": c.get("arguments")}, sort_keys=True)

    for i, (ex, pred_text) in enumerate(zip(ds, all_preds)):
        ref_text = ex["answers"]
        pred_text = pred_text.strip()

        try:
            ref_calls = json.loads(ref_text)
        except (json.JSONDecodeError, TypeError):
            ref_calls = []

        # Normalize ref tool names to snake_case for consistent comparison
        for rc in ref_calls:
            if isinstance(rc, dict) and "name" in rc:
                rc["name"] = to_snake_case(rc["name"])

        # Normalize pred tool names to snake_case too (restore_tool_names in
        # generate_batch maps back to originals, but we want snake_case comparison)
        try:
            pred_calls_raw = json.loads(pred_text)
            if isinstance(pred_calls_raw, list):
                for pc in pred_calls_raw:
                    if isinstance(pc, dict) and "name" in pc:
                        pc["name"] = to_snake_case(pc["name"])
            elif isinstance(pred_calls_raw, dict) and "name" in pred_calls_raw:
                pred_calls_raw["name"] = to_snake_case(pred_calls_raw["name"])
            pred_text = json.dumps(pred_calls_raw)
        except (json.JSONDecodeError, TypeError):
            pass

        # Also normalize tool definitions for param validation
        try:
            tool_defs_raw = json.loads(ex["tools"])
            for td in tool_defs_raw:
                if isinstance(td, dict) and "name" in td:
                    td["name"] = to_snake_case(td["name"])
            tools_normalized = json.dumps(tool_defs_raw)
        except (json.JSONDecodeError, TypeError):
            tools_normalized = ex["tools"]

        try:
            pred_calls = json.loads(pred_text)
            if not isinstance(pred_calls, list):
                pred_calls = [pred_calls] if isinstance(pred_calls, dict) else []
        except (json.JSONDecodeError, TypeError):
            json_parse_errors += 1
            pred_calls = []

        total += 1
        if i < 10:
            samples.append((ex["query"][:80], ref_text[:120], pred_text[:120]))

        ref_is_empty = ref_text.strip() in ("", "[]")
        pred_is_empty = pred_text.strip() in ("", "[]")
        if ref_is_empty:
            empty_ref += 1
        if pred_is_empty:
            empty_pred += 1

        if ref_is_empty and pred_is_empty:
            exact_match += 1
        elif not ref_is_empty and not pred_is_empty and json.dumps(pred_calls, sort_keys=True) == json.dumps(ref_calls, sort_keys=True):
            exact_match += 1

        ref_name_set = {c["name"] for c in ref_calls if isinstance(c, dict) and "name" in c}
        pred_name_set = {c["name"] for c in pred_calls if isinstance(c, dict) and "name" in c}
        tp_names += len(pred_name_set & ref_name_set)
        fp_names += len(pred_name_set - ref_name_set)
        fn_names += len(ref_name_set - pred_name_set)

        ref_keys = {call_key(c) for c in ref_calls} - {None}
        pred_keys = {call_key(c) for c in pred_calls} - {None}
        tp_calls += len(pred_keys & ref_keys)
        fp_calls += len(pred_keys - ref_keys)
        fn_calls += len(ref_keys - pred_keys)

        ref_by_name = {}
        for c in ref_calls:
            if isinstance(c, dict) and "name" in c:
                ref_by_name.setdefault(c["name"], []).append(c.get("arguments", {}))
        for c in pred_calls:
            if isinstance(c, dict) and "name" in c and c["name"] in ref_by_name:
                args_total += 1
                pa = json.dumps(c.get("arguments", {}), sort_keys=True)
                if any(pa == json.dumps(ra, sort_keys=True) for ra in ref_by_name[c["name"]]):
                    args_correct += 1

        try:
            tool_defs = json.loads(tools_normalized)
            tool_param_map = {t["name"]: set((t.get("parameters") or {}).keys()) for t in tool_defs if isinstance(t, dict) and "name" in t}
        except (json.JSONDecodeError, TypeError):
            tool_param_map = {}
        for c in pred_calls:
            if not isinstance(c, dict) or "name" not in c:
                continue
            cname = c["name"]
            if cname not in tool_param_map:
                continue
            schema_keys = tool_param_map[cname]
            p_keys = set((c.get("arguments") or {}).keys())
            total_pred_params += len(p_keys)
            halluc_params += len(p_keys - schema_keys)
            if cname in ref_by_name:
                ref_args = ref_by_name[cname][0]
                r_keys = set((ref_args if isinstance(ref_args, dict) else {}).keys())
                total_ref_params += len(r_keys)
                missing_params += len(r_keys - p_keys)
                m_keys = p_keys & r_keys
                matched_params += len(m_keys)
                for k in m_keys:
                    if json.dumps(c.get("arguments", {})[k], sort_keys=True) == json.dumps(ref_args[k], sort_keys=True):
                        correct_values += 1

    name_precision = tp_names / max(tp_names + fp_names, 1)
    name_recall = tp_names / max(tp_names + fn_names, 1)
    name_f1 = 2 * name_precision * name_recall / max(name_precision + name_recall, 1e-9)

    call_precision = tp_calls / max(tp_calls + fp_calls, 1)
    call_recall = tp_calls / max(tp_calls + fn_calls, 1)
    call_f1 = 2 * call_precision * call_recall / max(call_precision + call_recall, 1e-9)

    return {
        "num_samples": total,
        "exact_match": exact_match / max(total, 1),
        "json_parse_rate": 1.0 - json_parse_errors / max(total, 1),
        "name_precision": name_precision,
        "name_recall": name_recall,
        "name_f1": name_f1,
        "args_acc": args_correct / max(args_total, 1),
        "param_haluc": halluc_params / max(total_pred_params, 1),
        "param_miss": missing_params / max(total_ref_params, 1),
        "value_acc": correct_values / max(matched_params, 1),
        "call_precision": call_precision,
        "call_recall": call_recall,
        "call_f1": call_f1,
        "empty_ref_pct": empty_ref / max(total, 1),
        "empty_pred_pct": empty_pred / max(total, 1),
        "samples": samples,
    }


def benchmark_retrieval(model, params, tokenizer, num_samples=500, max_len=256, ks=(1, 3, 5, 10), ds=None):
    """Benchmark contrastive retrieval: Recall@k and MRR over validation set.

    For each query, ranks all tools from that example by cosine similarity
    and checks if the positive (called) tools appear in top-k.
    """
    import json
    from .run import encode_for_retrieval

    if ds is None:
        ds = load_tool_calls("validation", max_samples=num_samples)
    elif num_samples and len(ds) > num_samples:
        ds = ds.select(range(num_samples))

    queries = []
    all_tool_strs = []
    tool_groups = []  # (start_idx, count, set_of_positive_indices_within_group)

    for ex in ds:
        query = ex["query"]
        try:
            tools = json.loads(ex["tools"])
        except (ValueError, TypeError):
            continue
        if not isinstance(tools, list) or len(tools) == 0:
            continue
        try:
            calls = json.loads(ex["answers"])
        except (ValueError, TypeError):
            calls = []
        pos_names = set()
        if isinstance(calls, list):
            for c in calls:
                if isinstance(c, dict) and "name" in c:
                    pos_names.add(c["name"])
        if not pos_names:
            continue

        start = len(all_tool_strs)
        pos_indices = set()
        for j, tool in enumerate(tools):
            if not isinstance(tool, dict):
                continue
            all_tool_strs.append(json.dumps(tool, separators=(",", ":")))
            if tool.get("name", "") in pos_names:
                pos_indices.add(len(all_tool_strs) - 1 - start)

        if pos_indices:
            queries.append(query)
            tool_groups.append((start, len(all_tool_strs) - start, pos_indices))

    if not queries:
        return {"recall@k": {}, "mrr": 0.0, "num_queries": 0}

    # Encode all queries and tools
    q_embs = encode_for_retrieval(model, params, tokenizer, queries, max_len=max_len)
    t_embs = encode_for_retrieval(model, params, tokenizer, all_tool_strs, max_len=max_len)

    max_k = max(ks)
    recall = {k: 0 for k in ks}
    mrr_sum = 0.0

    for i, (start, count, pos_set) in enumerate(tool_groups):
        group_embs = t_embs[start:start + count]  # (count, dim)
        scores = q_embs[i:i+1] @ group_embs.T  # (1, count)
        ranked = np.argsort(-scores[0])

        # MRR: reciprocal rank of first positive
        for rank, idx in enumerate(ranked):
            if idx in pos_set:
                mrr_sum += 1.0 / (rank + 1)
                break

        # Recall@k
        for k in ks:
            top_k_set = set(ranked[:k].tolist())
            if top_k_set & pos_set:
                recall[k] += 1

    n = len(queries)
    return {
        "recall@k": {k: recall[k] / n for k in ks},
        "mrr": mrr_sum / n,
        "num_queries": n,
    }


def main(args):
    params, config = load_checkpoint(args.checkpoint)
    model = EncoderDecoderTransformer(config)
    tokenizer = get_tokenizer()

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"\ncheckpoint:  {args.checkpoint}")
    print(f"parameters:  {param_count:,}")
    print(f"config:      d={config.d_model}, heads={config.num_heads}, layers={config.num_encoder_layers}/{config.num_decoder_layers}")

    print(f"\nevaluating tool-call perplexity ({args.max_eval_samples} samples)...")
    ds = load_tool_calls("validation", max_samples=args.max_eval_samples)
    enc_inputs, dec_inputs, dec_targets, loss_mask_arr, _ = prepare_tool_call_pairs(
        ds, tokenizer, max_enc_len=args.max_enc_len, max_dec_len=args.max_dec_len
    )
    ppl = compute_perplexity(model, params, enc_inputs, dec_inputs, dec_targets, args.batch_size, config.pad_token_id, loss_mask=loss_mask_arr)

    tc_samples = getattr(args, "tool_call_samples", 200)
    tc = None
    if tc_samples > 0:
        print(f"\nevaluating tool-call accuracy ({tc_samples} samples)...")
        tc = benchmark_tool_calls(model, params, tokenizer, num_samples=tc_samples, max_gen_len=args.max_gen_len, max_enc_len=args.max_enc_len, constrained=not getattr(args, "no_constrained", False))

    print(f"\n  ─────────────────────────────────────")
    print(f"  Tool-Call Metrics")
    print(f"  ─────────────────────────────────────")
    print(f"  Perplexity       {ppl:>10.2f}  ({args.max_eval_samples} samples)")
    if tc:
        print(f"  JSON parse rate  {tc['json_parse_rate']:>10.1%}")
        print(f"  Exact match      {tc['exact_match']:>10.1%}")
        print(f"  ─── Function Name ───────────────────")
        print(f"  Precision        {tc['name_precision']:>10.3f}")
        print(f"  Recall           {tc['name_recall']:>10.3f}")
        print(f"  F1               {tc['name_f1']:>10.3f}")
        print(f"  ─── Parameter-Level ────────────────")
        print(f"  Param haluc      {tc['param_haluc']:>10.1%}")
        print(f"  Param miss       {tc['param_miss']:>10.1%}")
        print(f"  Value acc        {tc['value_acc']:>10.1%}")
        print(f"  Args acc         {tc['args_acc']:>10.1%}")
        print(f"  ─── Full Call (name+args) ───────────")
        print(f"  Precision        {tc['call_precision']:>10.3f}")
        print(f"  Recall           {tc['call_recall']:>10.3f}")
        print(f"  F1               {tc['call_f1']:>10.3f}")
        print(f"  ─────────────────────────────────────")
        print(f"  Defer ref/pred   {tc['empty_ref_pct']:.1%} / {tc['empty_pred_pct']:.1%}")
        if tc["samples"]:
            print(f"\n  samples:")
            for query, ref, pred in tc["samples"][:5]:
                print(f"    Q: {query}")
                print(f"    R: {ref}")
                print(f"    P: {pred}")
                print()

    print(f"\nmeasuring throughput ({args.throughput_runs} runs)...")
    throughput = measure_throughput(model, params, tokenizer, num_runs=args.throughput_runs)
    print(f"avg tokens:  {throughput['avg_tokens_generated']:.1f}")
    print(f"avg latency: {throughput['avg_latency_s']:.3f}s")
    print(f"tokens/sec:  {throughput['tokens_per_second']:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark transformer on generation tasks")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch3.pkl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-eval-samples", type=int, default=1000)
    parser.add_argument("--max-enc-len", type=int, default=DEFAULT_MAX_ENC_LEN)
    parser.add_argument("--max-dec-len", type=int, default=DEFAULT_MAX_DEC_LEN)
    parser.add_argument("--max-gen-len", type=int, default=DEFAULT_MAX_GEN_LEN)
    parser.add_argument("--throughput-runs", type=int, default=10)
    parser.add_argument("--tool-call-samples", type=int, default=200,
                        help="Samples for tool-call accuracy eval (default: 200)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
