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
    """Slice a full MRL checkpoint to a self-contained sub-model at dimension d_prime."""

    # Load checkpoint
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    params = data["params"]
    config = TransformerConfig(**data["config"])

    head_dim = config.d_model // config.num_heads
    if d_prime % head_dim != 0:
        raise ValueError(
            f"d_prime={d_prime} must be divisible by head_dim={head_dim} "
            f"(d_model={config.d_model} // num_heads={config.num_heads})"
        )

    # Derived dims
    n_h = d_prime // head_dim
    n_kv = min(config.num_kv_heads, n_h)
    d_ff_prime = config.d_ff * d_prime // config.d_model
    kv_dim = config.num_kv_heads * head_dim
    kv_dim_prime = n_kv * head_dim

    d_model = config.d_model
    d_ff = config.d_ff

    def _target(size):
        """Map original dimension to sliced dimension."""
        if size == d_model:
            return d_prime
        if size == d_ff:
            return d_ff_prime
        if size == kv_dim:
            return kv_dim_prime
        return size  

    def slice_leaf(key_path, leaf):
        arr = np.asarray(leaf)

        if arr.ndim == 1:
            return arr[:_target(arr.shape[0])]

        if arr.ndim == 3:
            if arr.shape[0] > 1:
                _, in_ch, out_ch = arr.shape
                r_in = in_ch if in_ch == config.n_mels else _target(in_ch)
                r_out = _target(out_ch)
                return arr[:, :r_in, :r_out]
            return arr[:, :, :d_prime]

        if arr.ndim == 2:
            rows, cols = arr.shape
            r = _target(rows)
            c = _target(cols)
            return arr[:r, :c]

        return arr

    sliced = jax.tree_util.tree_map_with_path(slice_leaf, params)

    # Build new config
    new_config = replace(
        config,
        d_model=d_prime,
        d_ff=d_ff_prime,
        num_heads=n_h,
        num_kv_heads=n_kv,
    )

    # Convert any JAX arrays to numpy for pickling
    sliced_np = jax.tree.map(
        lambda x: np.asarray(x) if isinstance(x, jnp.ndarray) else x, sliced
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"params": sliced_np, "config": new_config.__dict__}, f)

    # Print summary
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
