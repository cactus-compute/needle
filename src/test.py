import argparse
import math
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import get_batches, get_nar_batches, get_tokenizer, load_tinystories, prepare_encoder_decoder_pairs, prepare_nar_pairs
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


def compute_perplexity(model, params, enc_inputs, dec_inputs, dec_targets, batch_size, pad_id):
    total_loss = 0.0
    total_tokens = 0

    for src, tgt_in, tgt_out in get_batches(enc_inputs, dec_inputs, dec_targets, batch_size, shuffle=False):
        src, tgt_in, tgt_out = jnp.array(src), jnp.array(tgt_in), jnp.array(tgt_out)

        src_mask = make_padding_mask(src, pad_id)
        tgt_mask = make_causal_mask(tgt_in.shape[1]) & make_padding_mask(tgt_in, pad_id)
        cross_mask = make_padding_mask(src, pad_id)

        logits = model.apply(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask,
                   )

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, tgt_out)
        mask = (tgt_out != pad_id).astype(jnp.float32)
        total_loss += float(jnp.sum(loss * mask))
        total_tokens += int(jnp.sum(mask))

    avg_nll = total_loss / max(total_tokens, 1)
    return math.exp(avg_nll)


def measure_throughput(model, params, tokenizer, num_runs=10, prompt="Once upon a time", max_gen_len=64):
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


def distinct_n(texts, n=2):
    """Fraction of unique n-grams across all generated texts. Higher = more diverse."""
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        if len(tokens) >= n:
            all_ngrams.extend(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def corpus_bleu4(references, hypotheses):
    """Corpus-level BLEU-4 between lists of token-id sequences.

    Simple implementation without external dependencies. Uses uniform weights
    (1/4 each for 1-gram through 4-gram) with brevity penalty.
    """
    from collections import Counter

    clipped_counts = [Counter() for _ in range(4)]
    total_counts = [0] * 4
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)

        ref_ngrams = [Counter() for _ in range(4)]
        hyp_ngrams = [Counter() for _ in range(4)]

        for n in range(4):
            for i in range(len(ref) - n):
                ref_ngrams[n][tuple(ref[i:i+n+1])] += 1
            for i in range(len(hyp) - n):
                ng = tuple(hyp[i:i+n+1])
                hyp_ngrams[n][ng] += 1
                total_counts[n] += 1

            for ng, count in hyp_ngrams[n].items():
                clipped_counts[n][ng] += min(count, ref_ngrams[n].get(ng, 0))

    # Precision for each n-gram order
    precisions = []
    for n in range(4):
        clipped = sum(clipped_counts[n].values())
        total = total_counts[n]
        if total == 0:
            return 0.0
        precisions.append(clipped / total)

    # Log-average precision (avoid log(0))
    if any(p == 0 for p in precisions):
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / 4

    # Brevity penalty
    if hyp_len >= ref_len:
        bp = 1.0
    elif hyp_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)

    return bp * math.exp(log_avg)


def evaluate_bleu4(model, params, tokenizer, enc_inputs, dec_targets,
                   num_samples=200, temperature=0.8, pad_id=0, nar_mode=False):
    """BLEU-4 of model generations against ground truth decoder targets.

    Works for both AR and NAR models — the primary cross-comparable quality metric.
    Uses fixed-shape padded arrays to avoid JAX recompilation per sample.
    """
    from .run import ctc_collapse

    n = min(num_samples, len(enc_inputs))
    references = []
    hypotheses = []

    if nar_mode:
        # Batched NAR eval: enc_inputs are already padded to fixed shape.
        # Run forward_nar in batches to avoid per-sample JIT recompilation.
        blank_id = model.config.blank_token_id
        batch_size = 32

        @jax.jit
        def nar_batch_forward(src):
            src_mask = make_padding_mask(src, pad_id)
            logits, _ = model.apply(
                {"params": params}, src, src_mask=src_mask,
                method="forward_nar",
            )
            return logits

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            src = jnp.array(enc_inputs[start:end])
            logits = nar_batch_forward(src)  # (B, N, V+1)
            pred_all = jnp.argmax(logits, axis=-1)  # (B, N)

            for j in range(end - start):
                ref = [int(t) for t in dec_targets[start + j] if t != pad_id]
                if len(ref) < 2:
                    continue
                hyp = ctc_collapse(pred_all[j], blank_id)
                hyp = [t for t in hyp if t != pad_id and t != tokenizer.eos_token_id]
                references.append(ref)
                hypotheses.append(hyp)
    else:
        # AR eval: generate one at a time (unavoidable for autoregressive)
        from .run import generate
        for i in range(n):
            ref = [int(t) for t in dec_targets[i] if t != pad_id]
            if len(ref) < 2:
                continue
            prompt_text = tokenizer.decode([int(t) for t in enc_inputs[i] if t != pad_id])
            gen_text = generate(model, params, tokenizer, prompt_text,
                                max_gen_len=len(ref) + 16, temperature=temperature,
                                seed=i, stream=False)
            hyp = tokenizer.encode(gen_text)
            references.append(ref)
            hypotheses.append(hyp)

    bleu = corpus_bleu4(references, hypotheses)

    # Length accuracy
    ratios = [len(h) / max(len(r), 1) for h, r in zip(hypotheses, references)]
    length_stats = {
        "mean_length_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "std_length_ratio": float(np.std(ratios)) if ratios else 0.0,
    }

    return {"bleu4": bleu, "num_samples": len(references), **length_stats}


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
        "distinct_1": distinct_n(generations, n=1),
        "distinct_2": distinct_n(generations, n=2),
        "generations": list(zip(prompts, generations)),
    }


def measure_nar_throughput(model, params, tokenizer, num_runs=10, prompt="Once upon a time"):
    """Measure NAR inference throughput (single forward pass)."""
    enc_tokens = tokenizer.encode(prompt)
    enc_input = jnp.array([enc_tokens])
    src_mask = make_padding_mask(enc_input, tokenizer.pad_token_id)

    @jax.jit
    def nar_forward(enc_input, src_mask):
        return model.apply(
            {"params": params}, enc_input, src_mask=src_mask,
            method="forward_nar",
        )

    # Warmup — compile once
    nar_forward(enc_input, src_mask)

    from .run import ctc_collapse
    latencies = []
    tokens_generated = []

    for _ in range(num_runs):
        start = time.perf_counter()
        logits, _ = nar_forward(enc_input, src_mask)
        logits.block_until_ready()
        elapsed = time.perf_counter() - start

        pred_tokens = jnp.argmax(logits[0], axis=-1)
        collapsed = ctc_collapse(pred_tokens, model.config.blank_token_id)
        collapsed = [t for t in collapsed if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]

        latencies.append(elapsed)
        tokens_generated.append(len(collapsed))

    avg_latency = np.mean(latencies)
    tokens_per_sec = sum(tokens_generated) / sum(latencies) if sum(latencies) > 0 else 0

    return {
        "avg_tokens_generated": float(np.mean(tokens_generated)),
        "avg_latency_s": float(avg_latency),
        "tokens_per_second": float(tokens_per_sec),
    }


def benchmark_nar_generation_quality(model, params, tokenizer, prompts):
    """Benchmark NAR generation quality using CTC collapse."""
    from .run import generate_nar

    generations = []
    for prompt in prompts:
        text = generate_nar(model, params, tokenizer, prompt, stream=False)
        generations.append(text)

    lengths = [len(tokenizer.encode(t)) for t in generations]
    rep_rate = compute_repetition_rate(generations)

    return {
        "avg_generation_length": float(np.mean(lengths)),
        "min_generation_length": int(np.min(lengths)),
        "max_generation_length": int(np.max(lengths)),
        "bigram_repetition_rate": rep_rate,
        "distinct_1": distinct_n(generations, n=1),
        "distinct_2": distinct_n(generations, n=2),
        "generations": list(zip(prompts, generations)),
    }


def compute_nar_ctc_loss(model, params, enc_inputs, dec_targets, batch_size, pad_id, blank_id=8192):
    """Compute average CTC loss for NAR model on validation data."""
    total_loss = 0.0
    total_samples = 0

    @jax.jit
    def nar_loss_batch(params, src, tgt_out):
        src_mask = make_padding_mask(src, pad_id)
        logits, _ = model.apply(
            {"params": params}, src, src_mask=src_mask,
            method="forward_nar",
        )
        B, N = logits.shape[0], logits.shape[1]
        logit_paddings = jnp.zeros((B, N), dtype=jnp.float32)
        label_paddings = (tgt_out == pad_id).astype(jnp.float32)
        loss = optax.ctc_loss(logits, logit_paddings, tgt_out, label_paddings,
                              blank_id=blank_id)
        # zero_infinity: clamp infeasible samples
        loss = jnp.where(jnp.isfinite(loss), loss, 0.0)
        loss = jnp.minimum(loss, 1e4)
        valid = loss > 0
        return jnp.sum(loss), jnp.sum(valid).astype(jnp.float32)

    for src, tgt_out in get_nar_batches(enc_inputs, dec_targets, batch_size, shuffle=False):
        src, tgt_out = jnp.array(src), jnp.array(tgt_out)
        loss, count = nar_loss_batch(params, src, tgt_out)
        total_loss += float(loss)
        total_samples += float(count)

    return total_loss / max(total_samples, 1)


def main(args):
    params, config = load_checkpoint(args.checkpoint)
    model = EncoderDecoderTransformer(config)
    tokenizer = get_tokenizer()

    nar_mode = config.num_queries > 0
    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"\ncheckpoint:  {args.checkpoint}")
    print(f"mode:        {'NAR (Q-CTC)' if nar_mode else 'AR'}")
    print(f"parameters:  {param_count:,}")
    print(f"config:      d={config.d_model}, heads={config.num_heads}, layers={config.num_encoder_layers}/{config.num_decoder_layers}")
    if nar_mode:
        print(f"queries:     {config.num_queries}")

    ds = load_tinystories("validation", max_samples=args.max_eval_samples)

    if nar_mode:
        print(f"\nevaluating CTC loss ({args.max_eval_samples} samples)...")
        enc_inputs, dec_targets = prepare_nar_pairs(
            ds, tokenizer, max_enc_len=args.max_enc_len, max_dec_len=args.max_dec_len,
            max_target_len=config.num_queries,
        )
        ctc_val = compute_nar_ctc_loss(model, params, enc_inputs, dec_targets, args.batch_size, config.pad_token_id)
        print(f"CTC loss:    {ctc_val:.4f}")

        print(f"\nmeasuring NAR throughput ({args.throughput_runs} runs)...")
        throughput = measure_nar_throughput(model, params, tokenizer, num_runs=args.throughput_runs)
    else:
        print(f"\nevaluating perplexity ({args.max_eval_samples} samples)...")
        enc_inputs, dec_inputs, dec_targets = prepare_encoder_decoder_pairs(
            ds, tokenizer, max_enc_len=args.max_enc_len, max_dec_len=args.max_dec_len
        )
        ppl = compute_perplexity(model, params, enc_inputs, dec_inputs, dec_targets, args.batch_size, config.pad_token_id)
        print(f"perplexity:  {ppl:.2f}")

        print(f"\nmeasuring throughput ({args.throughput_runs} runs)...")
        throughput = measure_throughput(model, params, tokenizer, num_runs=args.throughput_runs)

    print(f"avg tokens:  {throughput['avg_tokens_generated']:.1f}")
    print(f"avg latency: {throughput['avg_latency_s']:.3f}s")
    print(f"tokens/sec:  {throughput['tokens_per_second']:.1f}")

    prompts = [
        "Once upon a time",
        "The little dog",
        "She was very happy because",
        "One day, a boy named",
        "The cat and the mouse",
    ]

    if nar_mode:
        print(f"\ngenerating NAR samples...")
        quality = benchmark_nar_generation_quality(model, params, tokenizer, prompts)
    else:
        print(f"\ngenerating samples (t={args.temperature})...")
        quality = benchmark_generation_quality(
            model, params, tokenizer, prompts, max_gen_len=args.max_gen_len, temperature=args.temperature
        )

    print(f"avg length:  {quality['avg_generation_length']:.1f} tokens")
    print(f"repetition:  {quality['bigram_repetition_rate']:.3f}")
    print(f"distinct-1:  {quality['distinct_1']:.3f}")
    print(f"distinct-2:  {quality['distinct_2']:.3f}")

    # BLEU-4 against ground truth
    print(f"\nevaluating BLEU-4 (200 samples)...")
    bleu_results = evaluate_bleu4(model, params, tokenizer,
                                   enc_inputs, dec_targets, num_samples=200,
                                   temperature=args.temperature,
                                   pad_id=config.pad_token_id, nar_mode=nar_mode)
    print(f"BLEU-4:      {bleu_results['bleu4']:.4f}")
    print(f"length ratio:{bleu_results['mean_length_ratio']:>7.3f}  (ideal=1.0)")

    print("\nsamples:")
    for prompt, gen in quality["generations"]:
        print(f"  [{prompt}]")
        print(f"  {gen[:120]}")
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark transformer on generation tasks")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch3.pkl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-eval-samples", type=int, default=1000)
    parser.add_argument("--max-enc-len", type=int, default=128)
    parser.add_argument("--max-dec-len", type=int, default=128)
    parser.add_argument("--max-gen-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--throughput-runs", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
