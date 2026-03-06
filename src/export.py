"""Export matryoshka sub-models by slicing FFN weights by a shrink factor.

With FFN interior matryoshka, d_model stays constant — only FFN intermediate
dimensions (gate_proj, up_proj, down_proj) are sliced.

For topk-trained models, exports full model + mask indices per factor.
"""

import os
import pickle
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .model import TransformerConfig

# FFN kernel names that should be sliced in the d_ff dimension
_FFN_KERNEL_NAMES = {"gate_proj", "up_proj", "down_proj"}


def export_submodel(checkpoint_path, factor, output_path):
    """Export a matryoshka sub-model from a full checkpoint.

    For prefix-trained models: slices FFN weights to create a smaller d_ff.
    For topk-trained models: saves full model + binary mask indices per factor.
    """

    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    params = data["params"]
    config = TransformerConfig(**data["config"])

    mat_method = data.get("mat_method", "static-prefix")
    if mat_method == "topk" and "mask_logits" in data:
        return _export_topk(data, params, config, factor, output_path)
    else:
        return _export_prefix(params, config, factor, output_path)


def _export_topk(data, params, config, factor, output_path):
    """TopK export: full model + per-layer binary mask indices for FFN masking."""
    mask_logits = np.asarray(data["mask_logits"])  # (n_mat, n_blocks, d_ff)
    mat_factors = data.get("mat_factors", [])

    found = False
    for i, f in enumerate(mat_factors):
        if f == factor:
            factor_logits = mask_logits[i]  # (n_blocks, d_ff)
            ff_w = config.d_ff // factor
            # Per-layer indices
            per_layer_indices = []
            for b in range(factor_logits.shape[0]):
                indices = np.sort(np.argsort(-factor_logits[b])[:ff_w])
                per_layer_indices.append(indices)
            found = True
            break
    if not found:
        raise ValueError(f"factor={factor} not found in mat_factors={mat_factors}")

    params_np = jax.tree.map(
        lambda x: np.asarray(x) if isinstance(x, jnp.ndarray) else x, params
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "params": params_np,
            "config": config.__dict__,
            "mat_mask_indices": per_layer_indices,
            "mat_factor": factor,
            "mat_ff_width": ff_w,
        }, f)

    n_blocks = len(per_layer_indices)
    orig_count = sum(x.size for x in jax.tree.leaves(params))
    orig_bytes = sum(x.nbytes for x in jax.tree.leaves(params))

    print(f"\n  TopK export: {output_path}")
    print(f"  ─────────────────────────────────────")
    print(f"  d_ff (full)       {config.d_ff:>12d}")
    print(f"  d_ff (masked)     {ff_w:>12d}")
    print(f"  factor            {str(factor)+'x':>12s}")
    print(f"  blocks            {n_blocks:>12d} (per-layer masks)")
    print(f"  neurons/layer     {ff_w:>12d}")
    print(f"  params (full)     {orig_count:>12,d}")
    print(f"  size (MB)         {orig_bytes / 1e6:>12.1f}")
    print(f"  Note: full weights kept; per-layer mask applied at FFN level")
    print()


def _export_prefix(params, config, factor, output_path):
    """Prefix export: slice FFN weights to a smaller d_ff."""
    d_ff_new = config.d_ff // factor
    if d_ff_new == 0:
        raise ValueError(f"factor={factor} too large: would give d_ff=0")

    d_ff = config.d_ff

    def slice_leaf(key_path, leaf):
        arr = np.asarray(leaf)
        if arr.ndim != 2:
            return arr

        parent_name = None
        for part in key_path:
            name = part.key if hasattr(part, "key") else str(part)
            if name in _FFN_KERNEL_NAMES:
                parent_name = name
                break

        if parent_name is None:
            return arr

        rows, cols = arr.shape
        if parent_name in ("gate_proj", "up_proj"):
            if cols == d_ff:
                return arr[:, :d_ff_new]
        elif parent_name == "down_proj":
            if rows == d_ff:
                return arr[:d_ff_new, :]

        return arr

    sliced = jax.tree_util.tree_map_with_path(slice_leaf, params)

    new_config = replace(config, d_ff=d_ff_new)

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
    print(f"  {'d_model':>20s} {config.d_model:>12d} {config.d_model:>12d}")
    print(f"  {'d_ff':>20s} {config.d_ff:>12d} {d_ff_new:>12d}")
    print(f"  {'factor':>20s} {'1x':>12s} {str(factor)+'x':>12s}")
    print(f"  {'num_heads':>20s} {config.num_heads:>12d} {config.num_heads:>12d}")
    print(f"  {'num_kv_heads':>20s} {config.num_kv_heads:>12d} {config.num_kv_heads:>12d}")
    print(f"  {'params':>20s} {orig_count:>12,d} {new_count:>12,d}")
    print(f"  {'size (MB)':>20s} {orig_bytes / 1e6:>12.1f} {new_bytes / 1e6:>12.1f}")
    print()


def main(args):
    checkpoint = args.checkpoint
    factor = args.factor
    output = args.output

    if output is None:
        stem = Path(checkpoint).stem
        parent = Path(checkpoint).parent
        output = str(parent / f"{stem}_{factor}x.pkl")

    export_submodel(checkpoint, factor, output)
