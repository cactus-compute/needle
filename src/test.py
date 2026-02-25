import argparse
import math
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import get_batches, get_tokenizer, load_tinystories, prepare_encoder_decoder_pairs
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
            deterministic=True,
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
            self_mask=tgt_mask, cross_mask=src_mask, deterministic=True, method="decode",
        )
        return logits

    # Warmup — compile once
    encoder_out = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, deterministic=True, method="encode"
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
            {"params": params}, enc_input, src_mask=src_mask, deterministic=True, method="encode"
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


def main(args):
    params, config = load_checkpoint(args.checkpoint)
    model = EncoderDecoderTransformer(config)
    tokenizer = get_tokenizer()

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"\ncheckpoint:  {args.checkpoint}")
    print(f"parameters:  {param_count:,}")
    print(f"config:      d={config.d_model}, heads={config.num_heads}, layers={config.num_encoder_layers}/{config.num_decoder_layers}")

    print(f"\nevaluating perplexity ({args.max_eval_samples} samples)...")
    ds = load_tinystories("validation", max_samples=args.max_eval_samples)
    texts = [example["text"] for example in ds]
    enc_inputs, dec_inputs, dec_targets = prepare_encoder_decoder_pairs(
        texts, tokenizer, max_enc_len=args.max_enc_len, max_dec_len=args.max_dec_len
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
    print(f"\ngenerating samples (t={args.temperature})...")
    quality = benchmark_generation_quality(
        model, params, tokenizer, prompts, max_gen_len=args.max_gen_len, temperature=args.temperature
    )
    print(f"avg length:  {quality['avg_generation_length']:.1f} tokens")
    print(f"repetition:  {quality['bigram_repetition_rate']:.3f}")

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
