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
