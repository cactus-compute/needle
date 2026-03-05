"""Export MRL sub-models by slicing a full checkpoint to a target dimension."""

import os
import pickle
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .model import TransformerConfig


def export_submodel(checkpoint_path, d_prime, output_path):
    """Export an MRL sub-model from a full checkpoint.

    For slice-trained models: prefix-slices all weights to create a genuinely smaller
    architecture (d_model=d_prime, fewer heads, smaller FFN).

    For topk-trained models: saves full model weights + binary mask indices. The mask
    operates at the logit stage only — internal decoder weights (attention heads, FFN)
    must stay full-width to preserve learned interactions. Size savings come from the
    masked logit projection (only d_prime embedding columns participate).
    """

    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    params = data["params"]
    config = TransformerConfig(**data["config"])

    # Check if this is a topk-trained model
    mrl_method = data.get("mrl_method", "slice")
    use_topk = mrl_method == "topk" and "mask_logits" in data

    if use_topk:
        return _export_topk(data, params, config, d_prime, output_path)
    else:
        return _export_slice(params, config, d_prime, output_path)


def _export_topk(data, params, config, d_prime, output_path):
    """TopK export: full model + binary mask indices for logit-stage masking."""
    mask_logits = np.asarray(data["mask_logits"])
    mrl_dims = data["mrl_dims"]

    # Find mask index for this d_prime
    found = False
    for i, d in enumerate(mrl_dims):
        if d == d_prime:
            logits = mask_logits[i]
            indices = np.sort(np.argsort(-logits)[:d_prime])
            found = True
            break
    if not found:
        raise ValueError(f"d_prime={d_prime} not found in mrl_dims={mrl_dims}")

    # Save full model + mask indices (no weight slicing)
    params_np = jax.tree.map(
        lambda x: np.asarray(x) if isinstance(x, jnp.ndarray) else x, params
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "params": params_np,
            "config": config.__dict__,
            "mrl_mask_indices": indices,
            "mrl_d_prime": d_prime,
        }, f)

    orig_count = sum(x.size for x in jax.tree.leaves(params))
    orig_bytes = sum(x.nbytes for x in jax.tree.leaves(params))

    print(f"\n  TopK export: {output_path}")
    print(f"  ─────────────────────────────────────")
    print(f"  d_model (full)    {config.d_model:>12d}")
    print(f"  d_prime (mask)    {d_prime:>12d}")
    print(f"  mask indices      {len(indices):>12d} dims selected")
    print(f"  params (full)     {orig_count:>12,d}")
    print(f"  size (MB)         {orig_bytes / 1e6:>12.1f}")
    print(f"  Note: decoder weights kept full-width;")
    print(f"        mask applied at logit projection only")
    print()


def _export_slice(params, config, d_prime, output_path):
    """Slice export: prefix-slice all weights to a smaller architecture."""
    head_dim = config.d_model // config.num_heads
    if d_prime % head_dim != 0:
        raise ValueError(
            f"d_prime={d_prime} must be divisible by head_dim={head_dim} "
            f"(d_model={config.d_model} // num_heads={config.num_heads})"
        )

    n_h = d_prime // head_dim
    n_kv = min(config.num_kv_heads, n_h)
    d_ff_prime = config.d_ff * d_prime // config.d_model
    kv_dim = config.num_kv_heads * head_dim
    kv_dim_prime = n_kv * head_dim

    d_model = config.d_model
    d_ff = config.d_ff

    def slice_leaf(key_path, leaf):
        arr = np.asarray(leaf)

        if arr.ndim == 1:
            if arr.shape[0] == d_model:
                return arr[:d_prime]
            return arr

        if arr.ndim == 3:
            if arr.shape[2] == d_model:
                return arr[:, :, :d_prime]
            return arr

        if arr.ndim == 2:
            rows, cols = arr.shape
            def _target(size):
                if size == d_model:
                    return d_prime
                if size == d_ff:
                    return d_ff_prime
                if size == kv_dim:
                    return kv_dim_prime
                return size

            r = _target(rows)
            c = _target(cols)
            return arr[:r, :c]

        return arr

    sliced = jax.tree_util.tree_map_with_path(slice_leaf, params)

    new_config = replace(
        config,
        d_model=d_prime,
        d_ff=d_ff_prime,
        num_heads=n_h,
        num_kv_heads=n_kv,
    )

    sliced_np = jax.tree.map(
        lambda x: np.asarray(x) if isinstance(x, jnp.ndarray) else x, sliced
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"params": sliced_np, "config": new_config.__dict__}, f)

    orig_count = sum(x.size for x in jax.tree.leaves(params))
    new_count = sum(x.size for x in jax.tree.leaves(sliced_np))
    orig_bytes = sum(x.nbytes for x in jax.tree.leaves(params))
    new_bytes = sum(x.nbytes for x in jax.tree.leaves(sliced_np))

    print(f"\n  Export complete: {output_path}")
    print(f"  ─────────────────────────────────────")
    print(f"  {'':>20s} {'Original':>12s} {'Exported':>12s}")
    print(f"  {'d_model':>20s} {config.d_model:>12d} {d_prime:>12d}")
    print(f"  {'d_ff':>20s} {config.d_ff:>12d} {d_ff_prime:>12d}")
    print(f"  {'num_heads':>20s} {config.num_heads:>12d} {n_h:>12d}")
    print(f"  {'num_kv_heads':>20s} {config.num_kv_heads:>12d} {n_kv:>12d}")
    print(f"  {'params':>20s} {orig_count:>12,d} {new_count:>12,d}")
    print(f"  {'size (MB)':>20s} {orig_bytes / 1e6:>12.1f} {new_bytes / 1e6:>12.1f}")
    print()


def main(args):
    checkpoint = args.checkpoint
    d_prime = args.dim
    output = args.output

    if output is None:
        stem = Path(checkpoint).stem
        parent = Path(checkpoint).parent
        output = str(parent / f"{stem}_d{d_prime}.pkl")

    export_submodel(checkpoint, d_prime, output)
