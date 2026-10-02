import mlx.core as mx

from . import quantize as _quantize

KV_GROUP = 64


def fake_quant(w, group_size=128, bits=4):
    qmax = 2 ** (bits - 1) - 1
    D = w.shape[-1]
    pad = (-D) % group_size
    wp = mx.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, pad)]) if pad else w
    g = wp.reshape(*wp.shape[:-1], -1, group_size).astype(mx.float32)
    absmax = mx.max(mx.abs(g), axis=-1, keepdims=True)
    scale = mx.where(absmax > 0, absmax / qmax, 1.0)
    q = mx.clip(mx.round(g / scale), -qmax - 1, qmax) * scale
    q = q.reshape(wp.shape).astype(w.dtype)
    if pad:
        q = q[..., :D]
    return w + mx.stop_gradient(q - w)


def _cq_nearest(x, cb):
    # Counts the codebook steps x is strictly closer to, which is the left-on-tie
    # choice quantize._cq_nearest makes with searchsorted.
    idx = mx.zeros(x.shape, mx.int32)
    for k in range(cb.shape[0] - 1):
        idx = idx + (mx.abs(x - cb[k + 1]) < mx.abs(x - cb[k])).astype(mx.int32)
    return cb[idx]


def cq_quantize(w, bits, group_size=128):
    cb = mx.array(_quantize._cq_codebook_np(bits, group_size))
    D, g = w.shape[-1], group_size
    pad = (-D) % g
    wp = mx.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, pad)]) if pad else w
    groups = wp.reshape(*wp.shape[:-1], -1, g).astype(mx.float32)
    H = mx.array(_quantize._cq_hadamard_np(g))
    rot = groups @ H
    norm = mx.sqrt(mx.sum(rot ** 2, axis=-1, keepdims=True))
    unit = rot / mx.maximum(norm, 1e-12)
    norm = norm.astype(mx.float16).astype(mx.float32)
    deq = (_cq_nearest(unit, cb) * norm) @ H
    deq = deq.reshape(wp.shape).astype(w.dtype)
    return deq[..., :D] if pad else deq


def cq_ste(w, bits, group_size=_quantize.CQ_GROUP_SIZE):
    return w + mx.stop_gradient(cq_quantize(w, bits, group_size) - w)


def is_quant_leaf(name, leaf):
    key = name.rsplit("/", 1)[-1]
    return (key in ("kernel", "embedding") or key.startswith("mhc_phi")) and leaf.ndim >= 2


def cq_ste_leaf(name, w, bits):
    """quantize._map_quant_leaves for one leaf: kernels and mhc_phi reduce along their input axis."""
    key = name.rsplit("/", 1)[-1]
    if key == "kernel" or key.startswith("mhc_phi"):
        return mx.swapaxes(cq_ste(mx.swapaxes(w, -1, -2), bits), -1, -2)
    return cq_ste(w, bits)


def cq_ste_params(params, bits):
    """quantize.cq_ste_params over the flat dict architecture_mlx.to_mlx returns."""
    if any(k.split("/", 1)[0] == _quantize.AB_KEY for k in params):
        raise NotImplementedError("the MLX backend does not support ab_scales checkpoints")
    return {k: cq_ste_leaf(k, v, bits) if is_quant_leaf(k, v) else v for k, v in params.items()}


class Numerics:
    """The fake-quant a forward applies: none, or the export's A8 activations and KV."""

    def __init__(self, config, quant):
        self.quant = bool(quant)
        self.act_bits = int(getattr(config, "act_bits", 8))
        self.kv_bits = int(getattr(config, "kv_bits", 8))

    def act(self, x):
        return fake_quant(x, x.shape[-1], self.act_bits) if self.quant else x

    def query(self, x):
        return fake_quant(x, x.shape[-1], 8) if self.quant and self.act_bits else x

    def kv(self, x):
        if not (self.quant and self.kv_bits):
            return x
        if self.kv_bits >= 8:
            return fake_quant(x, x.shape[-1], 8)
        return x + mx.stop_gradient(cq_quantize(x, self.kv_bits, KV_GROUP) - x)
