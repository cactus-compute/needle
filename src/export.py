"""Export matryoshka sub-models by slicing FFN weights by a shrink factor.

With FFN interior matryoshka, d_model stays constant — only FFN intermediate
dimensions (gate_proj, up_proj, down_proj) are sliced.
"""

import os
import pickle
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .model import TransformerConfig

_FFN_KERNEL_NAMES = {"gate_proj", "up_proj", "down_proj"}


def export_submodel(checkpoint_path, factor, output_path):
    """Slice a full matryoshka checkpoint to a sub-model at given shrink factor.

    factor: how many times smaller the FFN width (e.g. 2 = half, 4 = quarter).
    Attention, embeddings, and norms are unchanged.
    """

    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    params = data["params"]
    config = TransformerConfig(**data["config"])

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
