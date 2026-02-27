import argparse
import math
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from flax import jax_utils

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


def _make_p_encode(model):
    """Create a pmap'd encode function."""
    def _encode(params, src, src_mask):
        return model.apply(
            {"params": params}, src, src_mask=src_mask, deterministic=True, method="encode",
        )
    return jax.pmap(_encode, axis_name="batch")


def _make_p_decode(model):
    """Create a pmap'd decode function."""
    def _decode(params, dec_input, encoder_out, tgt_mask, cross_mask):
        return model.apply(
            {"params": params}, dec_input, encoder_out,
            self_mask=tgt_mask, cross_mask=cross_mask, deterministic=True, method="decode",
        )
    return jax.pmap(_decode, axis_name="batch")


def _make_p_forward(model):
    """Create a pmap'd full forward function."""
    def _forward(params, src, tgt_in, src_mask, tgt_mask, cross_mask):
        return model.apply(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask,
            deterministic=True,
        )
    return jax.pmap(_forward, axis_name="batch")


def _shard_single(x, num_devices):
    """Replicate a single-sample batch across all devices for pmap."""
    return jnp.broadcast_to(x, (num_devices, *x.shape[1:]))


def _shard_batch(x, num_devices):
    """Reshape batch for pmap: (N, ...) -> (num_devices, per_device, ...)."""
    return x.reshape(num_devices, -1, *x.shape[1:])


def score_sequence(model, params, enc_tokens, dec_tokens, pad_id, p_encode=None, p_decode=None, num_devices=1):
    """Compute average negative log-likelihood of dec_tokens given enc_tokens."""
    enc_input = jnp.array([enc_tokens])
    src_mask = make_padding_mask(enc_input, pad_id)

    if p_encode is not None and num_devices > 1:
        # Replicate single sample across devices, run pmap, take first result
        enc_s = _shard_single(enc_input, num_devices)
        src_mask_s = _shard_single(src_mask, num_devices)
        encoder_out = p_encode(params, enc_s, src_mask_s)[0:1]
    else:
        p = params if num_devices <= 1 else jax_utils.unreplicate(params)
        encoder_out = model.apply(
            {"params": p}, enc_input, src_mask=src_mask, deterministic=True, method="encode",
        )

    dec_in = [pad_id] + list(dec_tokens[:-1])
    dec_input = jnp.array([dec_in])
    tgt_mask = make_causal_mask(len(dec_in))
    cross_mask = src_mask

    if p_decode is not None and num_devices > 1:
        dec_s = _shard_single(dec_input, num_devices)
        tgt_mask_s = jnp.broadcast_to(tgt_mask, (num_devices, *tgt_mask.shape[1:]))
        cross_mask_s = _shard_single(cross_mask, num_devices)
        enc_out_s = _shard_single(encoder_out, num_devices)
        logits = p_decode(params, dec_s, enc_out_s, tgt_mask_s, cross_mask_s)[0]
    else:
        p = params if num_devices <= 1 else jax_utils.unreplicate(params)
        logits = model.apply(
            {"params": p}, dec_input, encoder_out,
            self_mask=tgt_mask, cross_mask=cross_mask, deterministic=True, method="decode",
        )[0]

    log_probs = jax.nn.log_softmax(logits if logits.ndim == 2 else logits[0])
    target_ids = jnp.array(dec_tokens)
    token_lls = log_probs[jnp.arange(len(dec_tokens)), target_ids]
    return float(jnp.mean(token_lls))


def eval_wikitext2(model, params, tokenizer, max_samples=500, max_len=256,
                   num_devices=1, p_encode=None, p_decode=None):
    """Perplexity on WikiText-2 test split."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    pad_id = tokenizer.pad_token_id
    total_nll = 0.0
    total_tokens = 0
    evaluated = 0

    single_params = jax_utils.unreplicate(params) if num_devices > 1 else params

    for example in ds:
        text = example["text"].strip()
        if len(text) < 20:
            continue

        tokens = tokenizer.encode(text)
        if len(tokens) < 6:
            continue

        tokens = tokens[:max_len]
        split = max(2, len(tokens) // 3)
        enc_tokens = tokens[:split]
        dec_tokens = tokens[split:]

        enc_input = jnp.array([enc_tokens])
        src_mask = make_padding_mask(enc_input, pad_id)

        if p_encode is not None and num_devices > 1:
            enc_s = _shard_single(enc_input, num_devices)
            src_mask_s = _shard_single(src_mask, num_devices)
            encoder_out = p_encode(params, enc_s, src_mask_s)[0:1]
        else:
            encoder_out = model.apply(
                {"params": single_params}, enc_input, src_mask=src_mask, deterministic=True, method="encode",
            )

        dec_in = [pad_id] + list(dec_tokens[:-1])
        dec_input = jnp.array([dec_in])
        tgt_mask = make_causal_mask(len(dec_in))

        if p_decode is not None and num_devices > 1:
            dec_s = _shard_single(dec_input, num_devices)
            tgt_mask_s = jnp.broadcast_to(tgt_mask, (num_devices, *tgt_mask.shape[1:]))
            cross_mask_s = _shard_single(src_mask, num_devices)
            enc_out_s = _shard_single(encoder_out, num_devices)
            logits = p_decode(params, dec_s, enc_out_s, tgt_mask_s, cross_mask_s)[0]
        else:
            logits = model.apply(
                {"params": single_params}, dec_input, encoder_out,
                self_mask=tgt_mask, cross_mask=src_mask, deterministic=True, method="decode",
            )

        log_probs = jax.nn.log_softmax(logits[0] if logits.ndim == 3 else logits[0])
        target_ids = jnp.array(dec_tokens)
        token_lls = log_probs[jnp.arange(len(dec_tokens)), target_ids]

        total_nll += -float(jnp.sum(token_lls))
        total_tokens += len(dec_tokens)
        evaluated += 1

        if evaluated >= max_samples:
            break

    ppl = math.exp(total_nll / max(total_tokens, 1))
    return {"perplexity": ppl, "samples": evaluated, "tokens": total_tokens}


def eval_lambada(model, params, tokenizer, max_samples=500,
                 num_devices=1, p_encode=None, p_decode=None):
    """Accuracy of predicting the final word on LAMBADA."""
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")

    pad_id = tokenizer.pad_token_id
    correct = 0
    total = 0

    single_params = jax_utils.unreplicate(params) if num_devices > 1 else params

    for example in ds:
        text = example["text"].strip()
        words = text.rsplit(" ", 1)
        if len(words) < 2:
            continue

        context, last_word = words
        context_tokens = tokenizer.encode(context)
        target_tokens = tokenizer.encode(" " + last_word)

        if not context_tokens or not target_tokens:
            continue

        enc_input = jnp.array([context_tokens])
        src_mask = make_padding_mask(enc_input, pad_id)

        if p_encode is not None and num_devices > 1:
            enc_s = _shard_single(enc_input, num_devices)
            src_mask_s = _shard_single(src_mask, num_devices)
            encoder_out = p_encode(params, enc_s, src_mask_s)[0:1]
        else:
            encoder_out = model.apply(
                {"params": single_params}, enc_input, src_mask=src_mask, deterministic=True, method="encode",
            )

        dec_in = jnp.array([[pad_id]])
        tgt_mask = make_causal_mask(1)

        if p_decode is not None and num_devices > 1:
            dec_s = _shard_single(dec_in, num_devices)
            tgt_mask_s = jnp.broadcast_to(tgt_mask, (num_devices, *tgt_mask.shape[1:]))
            cross_mask_s = _shard_single(src_mask, num_devices)
            enc_out_s = _shard_single(encoder_out, num_devices)
            logits = p_decode(params, dec_s, enc_out_s, tgt_mask_s, cross_mask_s)[0]
        else:
            logits = model.apply(
                {"params": single_params}, dec_in, encoder_out,
                self_mask=tgt_mask, cross_mask=src_mask, deterministic=True, method="decode",
            )

        predicted = int(jnp.argmax(logits[0, 0] if logits.ndim == 3 else logits[0, 0]))
        if predicted == target_tokens[0]:
            correct += 1
        total += 1

        if total >= max_samples:
            break

    acc = correct / max(total, 1)
    return {"accuracy": acc, "correct": correct, "total": total}


def eval_hellaswag(model, params, tokenizer, max_samples=500,
                   num_devices=1, p_encode=None, p_decode=None):
    """Accuracy on HellaSwag by scoring each candidate ending."""
    ds = load_dataset("Rowan/hellaswag", split="validation")

    pad_id = tokenizer.pad_token_id
    correct = 0
    total = 0

    for example in ds:
        ctx = example["ctx"].strip()
        endings = example["endings"]
        label = int(example["label"])

        enc_tokens = tokenizer.encode(ctx)
        if not enc_tokens:
            continue

        # Truncate context to keep it manageable
        enc_tokens = enc_tokens[:192]

        scores = []
        for ending in endings:
            dec_tokens = tokenizer.encode(" " + ending.strip())
            if not dec_tokens:
                scores.append(float("-inf"))
                continue
            dec_tokens = dec_tokens[:64]
            score = score_sequence(model, params, enc_tokens, dec_tokens, pad_id,
                                   p_encode=p_encode, p_decode=p_decode, num_devices=num_devices)
            scores.append(score)

        predicted = int(np.argmax(scores))
        if predicted == label:
            correct += 1
        total += 1

        if total >= max_samples:
            break

    acc = correct / max(total, 1)
    return {"accuracy": acc, "correct": correct, "total": total}


def eval_arc_easy(model, params, tokenizer, max_samples=500,
                  num_devices=1, p_encode=None, p_decode=None):
    """Accuracy on ARC-Easy by scoring each answer choice."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

    pad_id = tokenizer.pad_token_id
    correct = 0
    total = 0

    for example in ds:
        question = example["question"].strip()
        choices = example["choices"]
        answer_key = example["answerKey"]

        labels = choices["label"]
        texts = choices["text"]

        enc_tokens = tokenizer.encode(question)[:128]
        if not enc_tokens:
            continue

        scores = []
        for choice_text in texts:
            dec_tokens = tokenizer.encode(" " + choice_text.strip())
            if not dec_tokens:
                scores.append(float("-inf"))
                continue
            dec_tokens = dec_tokens[:64]
            score = score_sequence(model, params, enc_tokens, dec_tokens, pad_id,
                                   p_encode=p_encode, p_decode=p_decode, num_devices=num_devices)
            scores.append(score)

        predicted_idx = int(np.argmax(scores))
        predicted_label = labels[predicted_idx]
        if predicted_label == answer_key:
            correct += 1
        total += 1

        if total >= max_samples:
            break

    acc = correct / max(total, 1)
    return {"accuracy": acc, "correct": correct, "total": total}


BENCHMARKS = {
    "wikitext2": eval_wikitext2,
    "lambada": eval_lambada,
    "hellaswag": eval_hellaswag,
    "arc_easy": eval_arc_easy,
}


def main(args):
    num_devices = jax.local_device_count()
    print(f"Detected {num_devices} device(s) for data-parallel evaluation")

    print(f"Loading checkpoint: {args.checkpoint}")
    params, config = load_checkpoint(args.checkpoint)
    model = EncoderDecoderTransformer(config)
    tokenizer = get_tokenizer()

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameters: {param_count:,}")

    # Replicate params across devices for pmap
    if num_devices > 1:
        params = jax_utils.replicate(params)
        p_encode = _make_p_encode(model)
        p_decode = _make_p_decode(model)
        print(f"Params replicated across {num_devices} devices")
    else:
        p_encode = None
        p_decode = None

    benchmarks = args.benchmarks or list(BENCHMARKS.keys())

    results = {}
    for name in benchmarks:
        if name not in BENCHMARKS:
            print(f"Unknown benchmark: {name}, skipping")
            continue

        print(f"\n{'=' * 50}")
        print(f"Benchmark: {name}")
        print("=" * 50)

        result = BENCHMARKS[name](
            model, params, tokenizer, max_samples=args.max_samples,
            num_devices=num_devices, p_encode=p_encode, p_decode=p_decode,
        )
        results[name] = result

        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    print(f"\n{'=' * 50}")
    print("Summary")
    print("=" * 50)
    for name, result in results.items():
        metric = "perplexity" if "perplexity" in result else "accuracy"
        print(f"  {name:>12s}: {result[metric]:.4f} ({metric})")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on established NLP benchmarks")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch3.pkl")
    parser.add_argument("--benchmarks", type=str, nargs="*", choices=list(BENCHMARKS.keys()),
                        help="Benchmarks to run (default: all)")
    parser.add_argument("--max-samples", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
