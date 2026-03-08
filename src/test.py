import argparse
import math
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import get_batches, get_tokenizer, load_tool_calls, prepare_tool_call_pairs, load_librispeech
from .model import (
    EncoderDecoderTransformer,
    TransformerConfig,
    make_causal_mask,
    make_padding_mask,
)


def load_checkpoint(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    params = jax.tree.map(jnp.array, data["params"])
    config = TransformerConfig(**data["config"])
    return params, config


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
        cross_mask = make_padding_mask(src, pad_id)

        logits = model.apply(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask,
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
    tgt_mask = make_causal_mask(max_gen_len)

    @jax.jit
    def decode_step(dec_buffer, encoder_out, src_mask):
        logits = model.apply(
            {"params": params}, dec_buffer, encoder_out,
            self_mask=tgt_mask, cross_mask=src_mask, method="decode",
        )
        return logits

    # Warmup — compile once
    encoder_out = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, method="encode"
    )
    dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
    dec_buffer = dec_buffer.at[0, 0].set(eos_id)
    decode_step(dec_buffer, encoder_out, src_mask)

    # Benchmark generation
    tokens_generated = []
    latencies = []

    for run in range(num_runs):
        rng = jax.random.PRNGKey(run)
        dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
        dec_buffer = dec_buffer.at[0, 0].set(eos_id)

        start = time.perf_counter()
        encoder_out = model.apply(
            {"params": params}, enc_input, src_mask=src_mask, method="encode"
        )
        logits = decode_step(dec_buffer, encoder_out, src_mask)

        num_tokens = 0
        for i in range(max_gen_len - 1):
            rng, sample_rng = jax.random.split(rng)
            next_token = jax.random.categorical(sample_rng, logits[0, i]).item()

            if next_token == eos_id:
                break

            num_tokens += 1
            dec_buffer = dec_buffer.at[0, i + 1].set(next_token)
            logits = decode_step(dec_buffer, encoder_out, src_mask)

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


def benchmark_tool_calls(model, params, tokenizer, num_samples=200, max_gen_len=512):
    """Generate tool-call predictions and compute structured metrics."""
    import json
    from .run import generate
    from .data import load_tool_calls

    ds = load_tool_calls("validation", max_samples=num_samples)

    total = 0
    exact_match = 0
    # For function-name-level P/R/F1
    tp_names = 0  # predicted name in reference
    fp_names = 0  # predicted name not in reference
    fn_names = 0  # reference name not predicted
    # For full call-level (name + args) P/R/F1
    tp_calls = 0
    fp_calls = 0
    fn_calls = 0
    json_parse_errors = 0
    empty_ref = 0
    empty_pred = 0

    samples = []

    for i, ex in enumerate(ds):
        ref_text = ex["answers"]

        pred_text = generate(
            model, params, tokenizer, ex["query"],
            tools=ex["tools"], max_gen_len=max_gen_len, seed=i, stream=False,
        ).strip()

        # Parse reference
        try:
            ref_calls = json.loads(ref_text)
        except (json.JSONDecodeError, TypeError):
            ref_calls = []

        # Parse prediction
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

        # Track empty
        if len(ref_calls) == 0:
            empty_ref += 1
        if len(pred_calls) == 0:
            empty_pred += 1

        # Exact match (normalized)
        if json.dumps(pred_calls, sort_keys=True) == json.dumps(ref_calls, sort_keys=True):
            exact_match += 1

        # Function name P/R/F1
        ref_names = [c["name"] for c in ref_calls if isinstance(c, dict) and "name" in c]
        pred_names = [c["name"] for c in pred_calls if isinstance(c, dict) and "name" in c]
        ref_name_set = set(ref_names)
        pred_name_set = set(pred_names)
        tp_names += len(pred_name_set & ref_name_set)
        fp_names += len(pred_name_set - ref_name_set)
        fn_names += len(ref_name_set - pred_name_set)

        # Full call P/R/F1 (name + args must match)
        def call_key(c):
            if not isinstance(c, dict):
                return None
            return json.dumps({"name": c.get("name"), "arguments": c.get("arguments")}, sort_keys=True)

        ref_keys = set(call_key(c) for c in ref_calls)
        pred_keys = set(call_key(c) for c in pred_calls)
        ref_keys.discard(None)
        pred_keys.discard(None)
        tp_calls += len(pred_keys & ref_keys)
        fp_calls += len(pred_keys - ref_keys)
        fn_calls += len(ref_keys - pred_keys)

    # Compute metrics
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
        "call_precision": call_precision,
        "call_recall": call_recall,
        "call_f1": call_f1,
        "empty_ref_pct": empty_ref / max(total, 1),
        "empty_pred_pct": empty_pred / max(total, 1),
        "samples": samples,
    }


def benchmark_asr(model, params, tokenizer, num_samples=100):
    """Transcribe LibriSpeech test-clean samples and report WER."""
    from .run import transcribe
    from .data import load_librispeech

    ds = load_librispeech("test.clean", max_samples=num_samples)

    hypotheses = []
    references = []

    for example in ds:
        audio_data = example["audio"]
        audio = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]
        ref_text = example["text"].strip().lower()

        hyp_text = transcribe(
            model, params, tokenizer, audio, sr=sr,
            max_gen_len=128, temperature=0.0, seed=0, stream=False,
        ).strip().lower()

        hypotheses.append(hyp_text)
        references.append(ref_text)

    wer = compute_wer(hypotheses, references)
    return {"wer": wer, "num_samples": len(hypotheses)}


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
    enc_inputs, dec_inputs, dec_targets, loss_mask_arr = prepare_tool_call_pairs(
        ds, tokenizer, max_enc_len=args.max_enc_len, max_dec_len=args.max_dec_len
    )
    ppl = compute_perplexity(model, params, enc_inputs, dec_inputs, dec_targets, args.batch_size, config.pad_token_id, loss_mask=loss_mask_arr)

    tc_samples = getattr(args, "tool_call_samples", 200)
    tc = None
    if tc_samples > 0:
        print(f"\nevaluating tool-call accuracy ({tc_samples} samples)...")
        tc = benchmark_tool_calls(model, params, tokenizer, num_samples=tc_samples, max_gen_len=args.max_gen_len)

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
        print(f"  ─── Full Call (name+args) ───────────")
        print(f"  Precision        {tc['call_precision']:>10.3f}")
        print(f"  Recall           {tc['call_recall']:>10.3f}")
        print(f"  F1               {tc['call_f1']:>10.3f}")
        print(f"  ─────────────────────────────────────")
        print(f"  Empty ref/pred   {tc['empty_ref_pct']:.1%} / {tc['empty_pred_pct']:.1%}")
        if tc["samples"]:
            print(f"\n  samples:")
            for query, ref, pred in tc["samples"][:5]:
                print(f"    Q: {query}")
                print(f"    R: {ref}")
                print(f"    P: {pred}")
                print()

    asr_samples = getattr(args, "asr_samples", 50)
    if asr_samples > 0:
        print(f"\nevaluating ASR WER ({asr_samples} samples)...")
        asr_results = benchmark_asr(model, params, tokenizer, num_samples=asr_samples)
        print(f"ASR WER:     {asr_results['wer']:.4f} ({asr_results['num_samples']} samples)")

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
    parser.add_argument("--max-enc-len", type=int, default=256)
    parser.add_argument("--max-dec-len", type=int, default=1024)
    parser.add_argument("--max-gen-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--throughput-runs", type=int, default=10)
    parser.add_argument("--tool-call-samples", type=int, default=200,
                        help="Samples for tool-call accuracy eval (default: 200)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
