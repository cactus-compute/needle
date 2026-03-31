"""Post-training calibration of the confidence head.

Freezes the encoder and decoder, computes per-example decoder perplexity
on the training data, maps perplexity to [0, 1] confidence labels via a
sigmoid calibration curve, then trains only the confidence head (2 Dense layers)
to predict these labels from encoder output.

Two-pass design for memory efficiency:
  Pass 1: Forward through encoder+decoder to compute per-example PPL (stores only scalars)
  Pass 2: Re-encode through encoder only, train confidence head on the fly

All compute is distributed across all available devices via pmap.

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


def _replicate(tree):
    """Replicate params/state across all devices."""
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (jax.local_device_count(),) + x.shape), tree)


def _unreplicate(tree):
    """Get single copy from replicated params/state."""
    return jax.tree.map(lambda x: x[0], tree)


def compute_per_example_perplexity(model, params, train_data, batch_size, max_samples=None):
    """Compute per-example perplexity using packed training data.

    Memory-efficient: only stores scalar PPL per example, not encoder outputs.
    Distributed across all devices via pmap.

    Returns:
        perplexities: numpy array of per-example perplexity values
    """
    enc = train_data["packed_enc"]
    dec_in = train_data["packed_dec_in"]
    dec_tgt = train_data["packed_dec_tgt"]
    enc_seg = train_data["packed_enc_seg"]
    dec_seg = train_data["packed_dec_seg"]

    n = len(enc)
    if max_samples:
        n = min(n, max_samples)

    num_devices = jax.local_device_count()
    per_device_batch = batch_size // num_devices
    assert batch_size % num_devices == 0, f"batch_size {batch_size} must be divisible by {num_devices} devices"

    all_ppl = []

    @jax.pmap
    def forward_step(params, src, tgt_in, tgt_out, enc_seg_ids, dec_seg_ids):
        src_mask = make_packing_mask(enc_seg_ids)
        tgt_mask = make_causal_packing_mask(dec_seg_ids)
        cross_mask = make_cross_packing_mask(enc_seg_ids, dec_seg_ids)

        logits = model.apply(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask,
        )

        ce = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), tgt_out
        )
        padding_mask = (dec_seg_ids > 0).astype(jnp.float32)
        per_example_loss = jnp.sum(ce * padding_mask, axis=-1) / jnp.maximum(
            jnp.sum(padding_mask, axis=-1), 1.0
        )
        per_example_ppl = jnp.exp(jnp.minimum(per_example_loss, 20.0))
        return per_example_ppl

    r_params = _replicate(params)

    print(f"Computing per-example perplexity ({n} examples, batch_size={batch_size}, {num_devices} devices)...")
    for i in tqdm(range(0, n, batch_size), desc="PPL forward"):
        end = min(i + batch_size, n)
        actual = end - i

        # Pad to full batch if needed
        def _load(arr, i, end, actual):
            batch = np.array(arr[i:end], dtype=np.int32)
            if actual < batch_size:
                pad = np.zeros((batch_size - actual, batch.shape[1]), dtype=np.int32)
                batch = np.concatenate([batch, pad], axis=0)
            return batch.reshape(num_devices, per_device_batch, -1)

        src = _load(enc, i, end, actual)
        tgt_in = _load(dec_in, i, end, actual)
        tgt_out = _load(dec_tgt, i, end, actual)
        es = _load(enc_seg, i, end, actual)
        ds = _load(dec_seg, i, end, actual)

        ppl = forward_step(r_params, src, tgt_in, tgt_out, es, ds)
        # ppl shape: (num_devices, per_device_batch) → flatten and trim
        ppl_flat = np.array(ppl).reshape(-1)[:actual]
        all_ppl.append(ppl_flat)

    perplexities = np.concatenate(all_ppl)
    print(f"  Perplexity stats: min={perplexities.min():.2f}, median={np.median(perplexities):.2f}, "
          f"mean={perplexities.mean():.2f}, max={perplexities.max():.2f}")

    return perplexities


def perplexity_to_confidence(perplexities, mu=None, k=3.0):
    """Map perplexity values to [0, 1] confidence via sigmoid calibration.

    confidence = sigmoid(-k * (log(ppl) - log(mu)))
    """
    if mu is None:
        mu = float(np.median(perplexities))

    log_ppl = np.log(np.maximum(perplexities, 1.0))
    log_mu = np.log(mu)

    confidence = 1.0 / (1.0 + np.exp(k * (log_ppl - log_mu)))

    return confidence.astype(np.float32), mu


def train_confidence_head(model, params, train_data, confidence_labels,
                          epochs=1, lr=1e-3, batch_size=128):
    """Train only the confidence head parameters, re-encoding on the fly.

    Distributed across all devices via pmap. Memory-efficient: re-runs encoder
    forward pass during training instead of caching all encoder outputs.
    """
    enc = train_data["packed_enc"]
    enc_seg = train_data["packed_enc_seg"]
    n = len(confidence_labels)

    num_devices = jax.local_device_count()
    per_device_batch = batch_size // num_devices
    assert batch_size % num_devices == 0, f"batch_size {batch_size} must be divisible by {num_devices} devices"

    conf_param_keys = {"confidence_hidden", "confidence_out"}

    def is_conf_param(path):
        return any(k in str(path) for k in conf_param_keys)

    tx = optax.adam(lr)
    opt_state = tx.init(params)

    train_step = jax.pmap(
        lambda params, opt_state, src, enc_seg_ids, targets: _train_step_inner(
            model, tx, is_conf_param, params, opt_state, src, enc_seg_ids, targets
        ),
        axis_name="devices",
    )

    r_params = _replicate(params)
    r_opt_state = _replicate(opt_state)

    print(f"\nTraining confidence head ({n} examples, {epochs} epoch(s), lr={lr}, "
          f"batch_size={batch_size}, {num_devices} devices)...")

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_losses = []

        for i in tqdm(range(0, n - batch_size + 1, batch_size),
                      desc=f"Calibrating epoch {epoch + 1}/{epochs}"):
            idx = perm[i:i + batch_size]
            src = np.array(enc[idx], dtype=np.int32).reshape(num_devices, per_device_batch, -1)
            es = np.array(enc_seg[idx], dtype=np.int32).reshape(num_devices, per_device_batch, -1)
            tgt = np.array(confidence_labels[idx]).reshape(num_devices, per_device_batch)

            r_params, r_opt_state, loss = train_step(r_params, r_opt_state, src, es, tgt)
            epoch_losses.append(float(loss[0]))  # loss is replicated, take first

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        print(f"  Epoch {epoch + 1}/{epochs}: MSE={avg_loss:.6f}")

    params = _unreplicate(r_params)
    return params


def _train_step_inner(model, tx, is_conf_param, params, opt_state, src, enc_seg_ids, targets):
    """Inner training step — separated for pmap with axis_name."""
    def loss_fn(params):
        src_mask = make_packing_mask(enc_seg_ids)
        encoder_out, enc_mask = model.apply(
            {"params": params}, src, src_mask=src_mask, method="encode_text",
        )
        pred = model.apply(
            {"params": params}, encoder_out, enc_mask,
            method="predict_confidence",
        )
        return jnp.mean((pred - targets) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name="devices")

    # Zero out gradients for non-confidence params
    grads = jax.tree_util.tree_map_with_path(
        lambda path, g: g if is_conf_param(path) else jnp.zeros_like(g),
        grads,
    )

    updates, opt_state_new = tx.update(grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)
    return params_new, opt_state_new, loss


def evaluate_confidence(model, params, train_data, confidence_labels, perplexities, max_eval=10000):
    """Evaluate calibrated confidence head and print diagnostic stats."""
    enc = train_data["packed_enc"]
    enc_seg = train_data["packed_enc_seg"]
    n = min(len(confidence_labels), max_eval)

    num_devices = jax.local_device_count()
    batch_size = 128
    # Round down to multiple of num_devices
    batch_size = (batch_size // num_devices) * num_devices
    per_device_batch = batch_size // num_devices

    @jax.pmap
    def predict_step(params, src, enc_seg_ids):
        src_mask = make_packing_mask(enc_seg_ids)
        encoder_out, enc_mask = model.apply(
            {"params": params}, src, src_mask=src_mask, method="encode_text",
        )
        return model.apply(
            {"params": params}, encoder_out, enc_mask,
            method="predict_confidence",
        )

    r_params = _replicate(params)

    predictions = []
    for i in tqdm(range(0, n, batch_size), desc="Evaluating confidence"):
        end = min(i + batch_size, n)
        actual = end - i

        src = np.array(enc[i:end], dtype=np.int32)
        es = np.array(enc_seg[i:end], dtype=np.int32)
        if actual < batch_size:
            pad_src = np.zeros((batch_size - actual, src.shape[1]), dtype=np.int32)
            src = np.concatenate([src, pad_src], axis=0)
            pad_es = np.zeros((batch_size - actual, es.shape[1]), dtype=np.int32)
            es = np.concatenate([es, pad_es], axis=0)

        src = src.reshape(num_devices, per_device_batch, -1)
        es = es.reshape(num_devices, per_device_batch, -1)

        pred = predict_step(r_params, src, es)
        pred_flat = np.array(pred).reshape(-1)[:actual]
        predictions.append(pred_flat)

    predictions = np.concatenate(predictions)

    labels = confidence_labels[:n]
    ppl = perplexities[:n]

    # Stats
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    correlation = np.corrcoef(predictions, labels)[0, 1]

    print(f"\nConfidence head evaluation:")
    print(f"  MSE:         {mse:.6f}")
    print(f"  MAE:         {mae:.4f}")
    print(f"  Correlation: {correlation:.4f}")

    print(f"\n  {'PPL Range':<15s} {'Count':>6s} {'Avg Conf':>10s} {'Avg Pred':>10s}")
    print(f"  {'─' * 45}")
    buckets = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 50), (50, 1000)]
    for lo, hi in buckets:
        mask = (ppl >= lo) & (ppl < hi)
        if mask.sum() == 0:
            continue
        avg_conf = labels[mask].mean()
        avg_pred = predictions[mask].mean()
        print(f"  {f'{lo}-{hi}':<15s} {mask.sum():>6d} {avg_conf:>10.3f} {avg_pred:>10.3f}")

    return predictions


def main(args):
    num_devices = jax.local_device_count()
    print(f"Devices: {num_devices}")

    # Ensure batch_size is divisible by num_devices
    if args.batch_size % num_devices != 0:
        args.batch_size = (args.batch_size // num_devices) * num_devices
        print(f"  Adjusted batch_size to {args.batch_size} (divisible by {num_devices})")

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
        init_params = init_vars["params"]
        params = {**params, "confidence_hidden": init_params["confidence_hidden"],
                  "confidence_out": init_params["confidence_out"]}
        print("  Confidence head initialized.")

    # Load training data
    print(f"\nLoading prepared data...")
    train_data = load_prepared_data("train", mmap=True)
    print(f"  {len(train_data['packed_enc']):,} packed bins")

    # Pass 1: Compute per-example perplexity (forward-only, no gradients → large batch)
    ppl_batch_size = num_devices * 128  # 128 per device, forward-only no gradients
    perplexities = compute_per_example_perplexity(
        model, params, train_data, ppl_batch_size, max_samples=args.num_samples,
    )

    # Map perplexity → confidence labels
    confidence_labels, mu = perplexity_to_confidence(perplexities, k=args.k)
    print(f"\nCalibration: mu (median PPL) = {mu:.2f}, k = {args.k}")
    print(f"  Confidence stats: min={confidence_labels.min():.3f}, "
          f"median={np.median(confidence_labels):.3f}, "
          f"mean={confidence_labels.mean():.3f}, max={confidence_labels.max():.3f}")

    # Pass 2: Train confidence head (has gradients → use args.batch_size)
    params = train_confidence_head(
        model, params, train_data, confidence_labels,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )

    # Evaluate
    evaluate_confidence(model, params, train_data, confidence_labels, perplexities)

    # Save updated checkpoint
    output_path = args.output or args.checkpoint
    params_np = jax.tree.map(lambda x: np.array(x).astype(np.float16), params)

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
