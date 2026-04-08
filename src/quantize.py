import jax
import jax.numpy as jnp


def _fake_quantize_int4(w, group_size=32):
    """Symmetric group-wise INT4 fake quantization with STE.

    Divides the input dimension (axis 0) into groups of `group_size` elements,
    each with its own scale factor. Falls back to per-channel if in_features < group_size.
    """
    in_feat, out_feat = w.shape
    gs = min(group_size, in_feat)

    pad = (gs - in_feat % gs) % gs
    if pad > 0:
        w_padded = jnp.pad(w, ((0, pad), (0, 0)))
    else:
        w_padded = w

    num_groups = w_padded.shape[0] // gs
    w_grouped = w_padded.reshape(num_groups, gs, out_feat)

    scale = jnp.max(jnp.abs(w_grouped), axis=1, keepdims=True) / 7.0
    scale = jnp.maximum(scale, 1e-8)
    w_q = jnp.clip(jnp.round(w_grouped / scale), -8, 7) * scale

    w_q = w_q.reshape(-1, out_feat)[:in_feat]

    return w + jax.lax.stop_gradient(w_q - w)


def _fake_quantize_int8(w, group_size=32):
    """Symmetric group-wise INT8 fake quantization with STE.

    Same structure as INT4 but with [-128, 127] range (8-bit signed).
    """
    in_feat, out_feat = w.shape
    gs = min(group_size, in_feat)

    pad = (gs - in_feat % gs) % gs
    if pad > 0:
        w_padded = jnp.pad(w, ((0, pad), (0, 0)))
    else:
        w_padded = w

    num_groups = w_padded.shape[0] // gs
    w_grouped = w_padded.reshape(num_groups, gs, out_feat)

    scale = jnp.max(jnp.abs(w_grouped), axis=1, keepdims=True) / 127.0
    scale = jnp.maximum(scale, 1e-8)
    w_q = jnp.clip(jnp.round(w_grouped / scale), -128, 127) * scale

    w_q = w_q.reshape(-1, out_feat)[:in_feat]

    return w + jax.lax.stop_gradient(w_q - w)


def _quantize_params(params, group_size=32, precision="int4"):
    """Fake-quantize all Dense kernels in the param tree."""
    qfn = _fake_quantize_int8 if precision == "int8" else _fake_quantize_int4
    def _maybe_quantize(path, leaf):
        name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        if name == "kernel" and leaf.ndim == 3:
            return jax.vmap(lambda w: qfn(w, group_size=group_size), in_axes=(0,))(leaf)
        if name == "kernel" and leaf.ndim == 2:
            return qfn(leaf, group_size=group_size)
        return leaf
    return jax.tree_util.tree_map_with_path(_maybe_quantize, params)


def _cubic_sparsity_schedule(step, t_start, t_end, s_final):
    """Cubic sparsity ramp (Zhu & Gupta 2017). Returns target sparsity at *step*."""
    if step < t_start:
        return 0.0
    if step >= t_end:
        return s_final
    frac = (step - t_start) / (t_end - t_start)
    return s_final * (1.0 - (1.0 - frac) ** 3)


def _make_prune_mask(params, sparsity, group_size):
    """Compute block-prune mask entirely on-device (no numpy round-trips).

    Returns binary mask tree matching param shapes and dtypes.
    Handles both 2D (in, out) and 3D (num_layers, in, out) kernels from nn.scan.
    """
    all_scores = []
    for _, leaf in jax.tree_util.tree_leaves_with_path(params):
        if leaf.ndim == 2:
            in_feat, out_feat = leaf.shape
            gs = min(group_size, in_feat)
            pad = (gs - in_feat % gs) % gs
            w = jnp.pad(leaf, ((0, pad), (0, 0))) if pad else leaf
            scores = jnp.sum(jnp.abs(w.reshape(-1, gs, out_feat)), axis=1).ravel()
            all_scores.append(scores)
        elif leaf.ndim == 3:
            num_layers, in_feat, out_feat = leaf.shape
            gs = min(group_size, in_feat)
            pad = (gs - in_feat % gs) % gs
            for li in range(num_layers):
                w = jnp.pad(leaf[li], ((0, pad), (0, 0))) if pad else leaf[li]
                scores = jnp.sum(jnp.abs(w.reshape(-1, gs, out_feat)), axis=1).ravel()
                all_scores.append(scores)

    if not all_scores:
        return jax.tree.map(jnp.ones_like, params)

    threshold = jnp.percentile(jnp.concatenate(all_scores), sparsity * 100)

    def _leaf_mask(path, leaf):
        if leaf.ndim == 3:
            num_layers, in_feat, out_feat = leaf.shape
            gs = min(group_size, in_feat)
            pad = (gs - in_feat % gs) % gs
            masks = []
            for li in range(num_layers):
                w = jnp.pad(leaf[li], ((0, pad), (0, 0))) if pad else leaf[li]
                w_grouped = w.reshape(-1, gs, out_feat)
                block_keep = (jnp.sum(jnp.abs(w_grouped), axis=1, keepdims=True) > threshold)
                m = jnp.broadcast_to(block_keep, w_grouped.shape).reshape(-1, out_feat)[:in_feat]
                masks.append(m)
            return jnp.stack(masks, axis=0).astype(leaf.dtype)
        if leaf.ndim != 2:
            return jnp.ones_like(leaf)
        in_feat, out_feat = leaf.shape
        gs = min(group_size, in_feat)
        pad = (gs - in_feat % gs) % gs
        w = jnp.pad(leaf, ((0, pad), (0, 0))) if pad else leaf
        w_grouped = w.reshape(-1, gs, out_feat)
        block_keep = (jnp.sum(jnp.abs(w_grouped), axis=1, keepdims=True) > threshold)
        return jnp.broadcast_to(block_keep, w_grouped.shape).reshape(-1, out_feat)[:in_feat].astype(leaf.dtype)

    return jax.tree_util.tree_map_with_path(_leaf_mask, params)
