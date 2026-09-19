import functools
import math

import numpy as np
import jax
import jax.numpy as jnp


def _leaf_key(path):
    """Return the owning parameter key, skipping Flax metadata wrappers."""
    for entry in reversed(path):
        if hasattr(entry, "key"):
            return str(entry.key)
    return ""


def fake_quant(w, group_size=128, bits=4):
    qmax = 2 ** (bits - 1) - 1    
    D = w.shape[-1]
    pad = (-D) % group_size
    wp = jnp.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, pad)]) if pad else w
    g = wp.reshape(*wp.shape[:-1], -1, group_size).astype(jnp.float32)
    absmax = jnp.max(jnp.abs(g), axis=-1, keepdims=True)
    scale = jnp.where(absmax > 0, absmax / qmax, 1.0)
    q = jnp.clip(jnp.round(g / scale), -qmax - 1, qmax) * scale
    q = q.reshape(wp.shape).astype(w.dtype)
    if pad:
        q = q[..., :D]
    return w + jax.lax.stop_gradient(q - w)  


def fake_quant_act(x):
    return fake_quant(x, x.shape[-1], ACT_BITS)


def cq_fake_quant_kv(x, bits, group=64):
    return x + jax.lax.stop_gradient(cq_quantize(x, bits, group) - x)


def a8_fake_quant_kv(x):
    """Per-head symmetric A8 KV quantization matching the native cache."""
    return fake_quant(x, x.shape[-1], 8)


def maybe_quant_query(x, quant):
    """Per-head A8 query quantization matching the native attention kernel."""
    if not ACT_BITS:
        return x
    return jax.lax.cond(quant, a8_fake_quant_kv, lambda t: t, x)


def maybe_quant_kv(x, quant):
    if not KV_BITS:
        return x
    quantize = (a8_fake_quant_kv if KV_BITS >= 8 else
                lambda t: cq_fake_quant_kv(t, KV_BITS, _KV_GROUP))
    return jax.lax.cond(
        quant, quantize, lambda t: t, x)


WEIGHT_BITS = 4
ACT_BITS = 8
KV_BITS = 0
_KV_GROUP = 64



def configure_deploy(act_bits=8, kv_bits=8, kv_group=64):
    global ACT_BITS, KV_BITS, _KV_GROUP
    kv_bits = int(kv_bits)
    changed = (ACT_BITS, KV_BITS, _KV_GROUP) != (int(act_bits), int(kv_bits), int(kv_group))
    ACT_BITS, KV_BITS, _KV_GROUP = int(act_bits), int(kv_bits), int(kv_group)
    if changed:
        jax.clear_caches()



@functools.lru_cache(maxsize=None)
def _lloyd_max_gaussian(bits, iters=200, samples=400000, seed=0):
    if bits == 1:
        c = math.sqrt(2.0 / math.pi)
        return np.array([-c, c])
    levels = 1 << bits
    x = np.sort(np.random.RandomState(seed).randn(samples))
    c = x[((np.arange(levels) + 0.5) / levels * samples).astype(int)].astype(np.float64)
    for _ in range(iters):
        bnd = (c[:-1] + c[1:]) / 2.0
        idx = np.searchsorted(bnd, x)
        for k in range(levels):
            m = idx == k
            if m.any():
                c[k] = x[m].mean()
    return np.sort(c)


TERNARY_BITS = 1.58
_TERNARY_CB = np.array([-1.2240064, 0.0, 1.2240064])


@functools.lru_cache(maxsize=None)
def _cq_codebook_np(bits, group_size):
    cb = _TERNARY_CB if bits == TERNARY_BITS else _lloyd_max_gaussian(bits)
    return (cb / np.sqrt(group_size)).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _cq_hadamard_np(group_size):
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < group_size:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(group_size)).astype(np.float32)


def _cq_nearest(x, cb):
    flat = x.reshape(-1)
    pos = jnp.clip(jnp.searchsorted(cb, flat), 1, cb.shape[0] - 1)
    left, right = cb[pos - 1], cb[pos]
    idx = jnp.where(jnp.abs(flat - left) <= jnp.abs(flat - right), pos - 1, pos)
    return cb[idx].reshape(x.shape)


def cq_quantize(w, bits, group_size=128, codebook=None):
    cb = codebook if codebook is not None else jnp.asarray(_cq_codebook_np(bits, group_size))
    D, g = w.shape[-1], group_size
    pad = (-D) % g
    wp = jnp.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, pad)]) if pad else w
    groups = wp.reshape(*wp.shape[:-1], -1, g).astype(jnp.float32)
    H = jnp.asarray(_cq_hadamard_np(g))
    rot = groups @ H
    norm = jnp.sqrt(jnp.sum(rot ** 2, axis=-1, keepdims=True))
    unit = rot / jnp.maximum(norm, 1e-12)
    norm = norm.astype(jnp.float16).astype(jnp.float32)
    deq = (_cq_nearest(unit, cb) * norm) @ H
    deq = deq.reshape(wp.shape).astype(w.dtype)
    return deq[..., :D] if pad else deq


def _is_quant_leaf(path, leaf):
    key = _leaf_key(path)
    return ((key in ("kernel", "embedding") or key.startswith("mhc_phi"))
            and getattr(leaf, "ndim", 0) >= 2)


def _reduces_second_last(path):
    key = _leaf_key(path)
    return key == "kernel" or key.startswith("mhc_phi")


def cq_quantize_params(params, bits, group_size=128):
    return _map_quant_leaves(
        params, lambda w, i: cq_quantize(w, bits, group_size))


AB_KEY = "ab_scales"


def _ab_of(params):
    return params.get(AB_KEY) if hasattr(params, "get") else None



CQ_GROUP_SIZE = 128



def _map_quant_leaves(params, fn):
    """fn(leaf_with_reduction_axis_last, index) over the CQ-quantized leaves;
    AB scales, when present, sandwich fn: leaf -> a * fn(b * leaf)."""
    ab = _ab_of(params)
    counter = [0]

    def q(path, leaf):
        if not _is_quant_leaf(path, leaf):
            return leaf
        i = counter[0]
        counter[0] += 1
        s = ab.get(leaf_name(path)) if ab else None

        def apply(w):
            out = fn(w * s["b"] if s else w, i)
            return out * s["a"] if s else out

        if _reduces_second_last(path):
            return jnp.swapaxes(apply(jnp.swapaxes(leaf, -1, -2)), -1, -2)
        return apply(leaf)

    return jax.tree_util.tree_map_with_path(q, params)



def leaf_name(path):
    # Partitioned/AxisMetadata leaves add a trailing GetAttrKey("value").
    # That is container metadata, not part of the checkpoint parameter name.
    return "/".join(
        str(p.key) if hasattr(p, "key") else str(p.idx)
        for p in path
        if hasattr(p, "key") or hasattr(p, "idx")
    )


def quant_leaf_names(params):
    out = []

    def visit(path, leaf):
        if _is_quant_leaf(path, leaf):
            out.append((leaf_name(path), int(np.prod(leaf.shape))))
        return leaf

    jax.tree_util.tree_map_with_path(visit, params)
    return out



def cq_ste(w, bits, group_size=CQ_GROUP_SIZE):
    return w + jax.lax.stop_gradient(cq_quantize(w, bits, group_size) - w)


HEAD_BITS = 4


def head_weight(w, quant, reduce_second_last=False, group_size=CQ_GROUP_SIZE):
    """A probe-head matrix as the engine sees it: CQ at HEAD_BITS under QAT,
    straight-through for the gradient; kernels quantize along their input axis."""
    def fake(x):
        if reduce_second_last:
            return jnp.swapaxes(
                cq_ste(jnp.swapaxes(x, -1, -2), HEAD_BITS, group_size), -1, -2)
        return cq_ste(x, HEAD_BITS, group_size)
    return jax.lax.cond(quant, fake, lambda x: x, w)


def cq_ste_params(params, bits, group_size=CQ_GROUP_SIZE):
    return _map_quant_leaves(params, lambda w, i: cq_ste(w, bits, group_size))


