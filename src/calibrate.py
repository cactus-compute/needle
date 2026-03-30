"""Post-training calibration of the confidence head.

Freezes the encoder and decoder, computes per-example decoder perplexity
on the training data, maps perplexity to [0, 1] confidence labels via a
sigmoid calibration curve, then trains only the confidence head (2 Dense layers)
to predict these labels from encoder output.

Usage:
    needle calibrate --checkpoint checkpoints/needle_12_512_50000.pkl
    needle calibrate --checkpoint checkpoints/needle_12_512_50000.pkl --num-samples 10000
"""

import math
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from .data import (
    get_tokenizer, load_prepared_data, count_batches, get_batches,
)
from .model import (
    EncoderDecoderTransformer, TransformerConfig,
    make_packing_mask, make_causal_packing_mask, make_cross_packing_mask,
)


def compute_per_example_perplexity(model, params, val_data, batch_size, max_samples=None):
    """Compute per-example perplexity using packed validation data.

    Returns:
        encoder_outputs: list of (encoder_out, enc_mask) tuples
        perplexities: numpy array of per-example perplexity values
    """
    enc = val_data["packed_enc"]
    dec_in = val_data["packed_dec_in"]
    dec_tgt = val_data["packed_dec_tgt"]
    enc_seg = val_data["packed_enc_seg"]
    dec_seg = val_data["packed_dec_seg"]

    n = len(enc)
    if max_samples:
        n = min(n, max_samples)

    all_ppl = []
    all_enc_out = []
    all_enc_mask = []
    all_src = []
    all_enc_seg_ids = []

    @jax.jit
    def forward_step(params, src, tgt_in, tgt_out, enc_seg_ids, dec_seg_ids):
        src_mask = make_packing_mask(enc_seg_ids)
        tgt_mask = make_causal_packing_mask(dec_seg_ids)
        cross_mask = make_cross_packing_mask(enc_seg_ids, dec_seg_ids)

        # Get encoder output
        encoder_out, enc_mask = model.apply(
            {"params": params}, src, src_mask=src_mask, method="encode_text",
        )

        # Get decoder logits
        logits = model.apply(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask,
        )

        # Per-token cross entropy
        ce = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), tgt_out
        )
        # Per-example: average over non-padding tokens
        padding_mask = (dec_seg_ids > 0).astype(jnp.float32)
        per_example_loss = jnp.sum(ce * padding_mask, axis=-1) / jnp.maximum(
            jnp.sum(padding_mask, axis=-1), 1.0
        )
        per_example_ppl = jnp.exp(jnp.minimum(per_example_loss, 20.0))

        return per_example_ppl, encoder_out, enc_mask

    print(f"Computing per-example perplexity ({n} examples, batch_size={batch_size})...")
    for i in tqdm(range(0, n, batch_size), desc="PPL forward"):
        end = min(i + batch_size, n)
        src = jnp.asarray(enc[i:end], dtype=jnp.int32)
        tgt_in = jnp.asarray(dec_in[i:end], dtype=jnp.int32)
        tgt_out = jnp.asarray(dec_tgt[i:end], dtype=jnp.int32)
        es = jnp.asarray(enc_seg[i:end], dtype=jnp.int32)
        ds = jnp.asarray(dec_seg[i:end], dtype=jnp.int32)

        ppl, enc_out, enc_m = forward_step(params, src, tgt_in, tgt_out, es, ds)
        all_ppl.append(np.array(ppl))
        all_enc_out.append(np.array(enc_out))
        all_enc_mask.append(np.array(enc_m))
        all_src.append(np.array(src))
        all_enc_seg_ids.append(np.array(es))

    perplexities = np.concatenate(all_ppl)
    print(f"  Perplexity stats: min={perplexities.min():.2f}, median={np.median(perplexities):.2f}, "
          f"mean={perplexities.mean():.2f}, max={perplexities.max():.2f}")

    return all_enc_out, all_enc_mask, all_src, all_enc_seg_ids, perplexities


def perplexity_to_confidence(perplexities, mu=None, k=3.0):
    """Map perplexity values to [0, 1] confidence via sigmoid calibration.

    confidence = sigmoid(-k * (log(ppl) - log(mu)))

    Args:
        perplexities: array of perplexity values
        mu: center point (default: median perplexity)
        k: steepness of the sigmoid curve

    Returns:
        confidence labels in [0, 1]
    """
    if mu is None:
        mu = float(np.median(perplexities))

    log_ppl = np.log(np.maximum(perplexities, 1.0))
    log_mu = np.log(mu)

    # sigmoid(-k * (log_ppl - log_mu))
    # PPL << mu → large positive → confidence ≈ 1
    # PPL >> mu → large negative → confidence ≈ 0
    confidence = 1.0 / (1.0 + np.exp(k * (log_ppl - log_mu)))

    return confidence.astype(np.float32), mu


def train_confidence_head(model, params, enc_outputs, enc_masks, confidence_labels,
                          epochs=10, lr=1e-3, batch_size=64):
    """Train only the confidence head parameters, freezing everything else.

    Args:
        model: EncoderDecoderTransformer instance
        params: full model params (only confidence_hidden and confidence_out will be updated)
        enc_outputs: list of (B, T, d_model) encoder output arrays
        enc_masks: list of (B, 1, 1, T) mask arrays
        confidence_labels: (N,) float32 array of target confidence values

    Returns:
        Updated params with trained confidence head.
    """
    # Flatten batched arrays
    all_enc = np.concatenate(enc_outputs, axis=0)
    all_mask = np.concatenate(enc_masks, axis=0)
    n = len(confidence_labels)

    # Extract confidence head params
    conf_param_keys = {"confidence_hidden", "confidence_out"}

    def is_conf_param(path):
        return any(k in str(path) for k in conf_param_keys)

    # Create optimizer for confidence params only
    tx = optax.adam(lr)

    # We need to separate confidence params from frozen params
    conf_params = jax.tree_util.tree_map_with_path(
        lambda path, x: x if is_conf_param(path) else jax.lax.stop_gradient(x),
        params,
    )
    opt_state = tx.init(params)

    @jax.jit
    def confidence_loss(params, enc_out, enc_mask, targets):
        """MSE loss between predicted and target confidence."""
        pred = model.apply(
            {"params": params}, enc_out, enc_mask,
            method="predict_confidence",
        )
        return jnp.mean((pred - targets) ** 2)

    @jax.jit
    def train_step(params, opt_state, enc_out, enc_mask, targets):
        loss, grads = jax.value_and_grad(confidence_loss)(params, enc_out, enc_mask, targets)

        # Zero out gradients for non-confidence params
        grads = jax.tree_util.tree_map_with_path(
            lambda path, g: g if is_conf_param(path) else jnp.zeros_like(g),
            grads,
        )

        updates, opt_state_new = tx.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    print(f"\nTraining confidence head ({n} examples, {epochs} epochs, lr={lr})...")

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_losses = []

        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            enc_batch = jnp.asarray(all_enc[idx])
            mask_batch = jnp.asarray(all_mask[idx])
            target_batch = jnp.asarray(confidence_labels[idx])

            params, opt_state, loss = train_step(params, opt_state, enc_batch, mask_batch, target_batch)
            epoch_losses.append(float(loss))

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        print(f"  Epoch {epoch + 1}/{epochs}: MSE={avg_loss:.6f}")

    return params


def evaluate_confidence(model, params, enc_outputs, enc_masks, confidence_labels, perplexities):
    """Evaluate calibrated confidence head and print diagnostic stats."""
    all_enc = np.concatenate(enc_outputs, axis=0)
    all_mask = np.concatenate(enc_masks, axis=0)
    n = len(confidence_labels)

    # Predict in batches
    predictions = []
    batch_size = 128
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        pred = model.apply(
            {"params": params},
            jnp.asarray(all_enc[i:end]),
            jnp.asarray(all_mask[i:end]),
            method="predict_confidence",
        )
        predictions.append(np.array(pred))
    predictions = np.concatenate(predictions)

    # Stats
    mse = np.mean((predictions - confidence_labels) ** 2)
    mae = np.mean(np.abs(predictions - confidence_labels))
    correlation = np.corrcoef(predictions, confidence_labels)[0, 1]

    print(f"\nConfidence head evaluation:")
    print(f"  MSE:         {mse:.6f}")
    print(f"  MAE:         {mae:.4f}")
    print(f"  Correlation: {correlation:.4f}")

    # Show confidence vs perplexity buckets
    print(f"\n  {'PPL Range':<15s} {'Count':>6s} {'Avg Conf':>10s} {'Avg Pred':>10s}")
    print(f"  {'─' * 45}")
    buckets = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 50), (50, 1000)]
    for lo, hi in buckets:
        mask = (perplexities >= lo) & (perplexities < hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidence_labels[mask].mean()
        avg_pred = predictions[mask].mean()
        print(f"  {f'{lo}-{hi}':<15s} {mask.sum():>6d} {avg_conf:>10.3f} {avg_pred:>10.3f}")

    return predictions


def main(args):
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        data = pickle.load(f)
    params = jax.tree.map(lambda x: jnp.array(x, dtype=jnp.bfloat16), data["params"])
    config = TransformerConfig(**data["config"])
    model = EncoderDecoderTransformer(config)

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Parameters: {param_count:,}")
    print(f"  Config: d={config.d_model}, heads={config.num_heads}, "
          f"layers={config.num_encoder_layers}/{config.num_decoder_layers}")

    # Check if confidence head params exist; if not, initialize them
    flat_keys = str(jax.tree.map(lambda x: x.shape, params))
    if "confidence_hidden" not in flat_keys:
        print("  Confidence head not found in checkpoint — initializing...")
        rng = jax.random.PRNGKey(0)
        dummy_src = jnp.ones((1, 128), dtype=jnp.int32)
        dummy_tgt = jnp.ones((1, 128), dtype=jnp.int32)
        init_vars = model.init({"params": rng}, dummy_src, dummy_tgt, method="init_all")
        # Merge: keep existing params, add new confidence head params
        init_params = init_vars["params"]
        params = {**params, "confidence_hidden": init_params["confidence_hidden"],
                  "confidence_out": init_params["confidence_out"]}
        print("  Confidence head initialized.")

    # Load training data for perplexity computation
    print(f"\nLoading prepared data...")
    train_data = load_prepared_data("train", mmap=True)
    print(f"  {len(train_data['packed_enc']):,} packed bins")

    # Step 1: Compute per-example perplexity
    enc_outputs, enc_masks, srcs, enc_seg_ids, perplexities = compute_per_example_perplexity(
        model, params, train_data, args.batch_size, max_samples=args.num_samples,
    )

    # Step 2: Map perplexity → confidence labels
    confidence_labels, mu = perplexity_to_confidence(perplexities, k=args.k)
    print(f"\nCalibration: mu (median PPL) = {mu:.2f}, k = {args.k}")
    print(f"  Confidence stats: min={confidence_labels.min():.3f}, "
          f"median={np.median(confidence_labels):.3f}, "
          f"mean={confidence_labels.mean():.3f}, max={confidence_labels.max():.3f}")

    # Step 3: Train the confidence head
    params = train_confidence_head(
        model, params, enc_outputs, enc_masks, confidence_labels,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )

    # Step 4: Evaluate
    evaluate_confidence(model, params, enc_outputs, enc_masks, confidence_labels, perplexities)

    # Step 5: Save updated checkpoint
    output_path = args.output or args.checkpoint
    params_np = jax.tree.map(lambda x: np.array(x).astype(np.float16), params)

    # Store calibration metadata alongside config
    save_config = dict(data["config"])
    save_config["confidence_mu"] = float(mu)
    save_config["confidence_k"] = float(args.k)

    with open(output_path, "wb") as f:
        pickle.dump({"params": params_np, "config": save_config}, f)
    print(f"\nSaved calibrated checkpoint: {output_path}")
    print(f"  Calibration params: mu={mu:.2f}, k={args.k}")
    print(f"\n  Usage at inference:")
    print(f"    confidence, encoder_out, enc_mask = model.apply(")
    print(f'        {{"params": params}}, src, src_mask=src_mask, method="forward_confidence")')
    print(f"    if confidence > 0.6:  # user-configurable threshold")
    print(f"        # decode locally")
    print(f"    else:")
    print(f"        # route to cloud")
