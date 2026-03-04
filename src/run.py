import argparse
import pickle
import sys

import jax
import jax.numpy as jnp

from .data import get_tokenizer
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


def generate(model, params, tokenizer, prompt, max_gen_len=128, temperature=0.8, seed=0, stream=True):
    enc_tokens = tokenizer.encode(prompt)
    enc_input = jnp.array([enc_tokens])

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    src_mask = make_padding_mask(enc_input, pad_id)
    encoder_out = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, method="encode"
    )

    # Fixed-size buffer so JAX compiles once
    tgt_mask = make_causal_mask(max_gen_len)
    dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
    dec_buffer = dec_buffer.at[0, 0].set(eos_id)

    @jax.jit
    def decode_step(dec_buffer, encoder_out, src_mask):
        logits = model.apply(
            {"params": params},
            dec_buffer,
            encoder_out,
            self_mask=tgt_mask,
            cross_mask=src_mask,
                       method="decode",
        )
        return logits

    rng = jax.random.PRNGKey(seed)
    generated_tokens = []

    if stream:
        sys.stdout.write(f"\n{prompt}")
        sys.stdout.flush()

    # Compile once on first call
    logits = decode_step(dec_buffer, encoder_out, src_mask)

    for i in range(max_gen_len - 1):
        next_logits = logits[0, i] / temperature
        rng, sample_rng = jax.random.split(rng)
        next_token = jax.random.categorical(sample_rng, next_logits).item()

        if next_token == eos_id:
            break

        generated_tokens.append(next_token)
        dec_buffer = dec_buffer.at[0, i + 1].set(next_token)

        if stream:
            sys.stdout.write(tokenizer.decode([next_token]))
            sys.stdout.flush()

        logits = decode_step(dec_buffer, encoder_out, src_mask)

    if stream:
        sys.stdout.write("\n")

    return tokenizer.decode(generated_tokens)


def ctc_collapse(tokens, blank_id):
    """Remove blanks and merge consecutive duplicate tokens."""
    result = []
    prev = -1
    for t in tokens:
        t = int(t)
        if t == blank_id:
            prev = -1  # blank resets dedup so same token can appear again
            continue
        if t == prev:
            continue
        result.append(t)
        prev = t
    return result


def generate_nar(model, params, tokenizer, prompt, stream=True, max_enc_len=256, max_passes=10):
    """Chained NAR generation: repeat forward passes until EOS or max_passes.

    Each pass: encode(context) → decode queries → collapse → append to context.
    Produces up to num_queries tokens per pass, chained for longer output.
    """
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    blank_id = model.config.blank_token_id

    enc_tokens = tokenizer.encode(prompt)
    all_generated = []

    @jax.jit
    def nar_forward(enc_input, src_mask):
        return model.apply(
            {"params": params}, enc_input, src_mask=src_mask,
            method="forward_nar",
        )

    for pass_idx in range(max_passes):
        # Pad encoder input to fixed length for JIT reuse
        context = enc_tokens + all_generated
        if len(context) > max_enc_len:
            context = context[-max_enc_len:]  # keep most recent context
        padded = context + [pad_id] * (max_enc_len - len(context))
        enc_input = jnp.array([padded])
        src_mask = make_padding_mask(enc_input, pad_id)

        logits, _ = nar_forward(enc_input, src_mask)
        pred_tokens = jnp.argmax(logits[0], axis=-1)
        collapsed = ctc_collapse(pred_tokens, blank_id)

        # Check for EOS
        hit_eos = False
        new_tokens = []
        for t in collapsed:
            if t == eos_id:
                hit_eos = True
                break
            if t != pad_id:
                new_tokens.append(t)

        if not new_tokens:
            break  # model produced nothing — stop

        all_generated.extend(new_tokens)

        if stream:
            chunk_text = tokenizer.decode(new_tokens)
            if pass_idx == 0:
                sys.stdout.write(f"\n{prompt}")
            sys.stdout.write(chunk_text)
            sys.stdout.flush()

        if hit_eos:
            break

    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return tokenizer.decode(all_generated)


def main(args):
    print(f"Loading checkpoint: {args.checkpoint}")
    params, config = load_checkpoint(args.checkpoint)

    model = EncoderDecoderTransformer(config)
    tokenizer = get_tokenizer()

    nar_mode = getattr(args, "nar", False) or config.num_queries > 0
    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameters: {param_count:,}")
    if nar_mode:
        print(f"Mode: NAR (num_queries={config.num_queries})")

    prompts = args.prompts or ["Once upon a time", "The little dog", "She was very happy because"]

    for i, prompt in enumerate(prompts):
        if nar_mode:
            generate_nar(model, params, tokenizer, prompt, stream=True)
        else:
            generate(
                model,
                params,
                tokenizer,
                prompt,
                max_gen_len=args.max_len,
                temperature=args.temperature,
                seed=args.seed + i,
                stream=True,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate stories with trained transformer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch3.pkl")
    parser.add_argument("--prompts", type=str, nargs="*", help="Prompts to continue")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
