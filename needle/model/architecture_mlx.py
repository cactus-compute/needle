import functools
import math

import numpy as np
import mlx.core as mx

from .architecture import (
    ENGRAM_CONV_TAPS, HEAD_KEYS, _ENGRAM_PRIME, _ENGRAM_SEED, _hada_perms,
    engram_geometry, head_dims, ladder_layer_ranks,
)
from .checkpoints import flatten
from .quantize_mlx import Numerics

MX_DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}
BLOCK = "stack/layers/block/"


def to_mlx(params):
    """A checkpoint param tree as the flat {"a/b/c": array} dict the MLX forward reads."""
    return {k: mx.array(np.asarray(v, np.float32)) for k, v in flatten(params).items()
            if k.split("/", 1)[0] not in HEAD_KEYS}


def make_causal_mask(seq_len):
    return mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))[None, None]


def _shift_right(x, offset):
    if offset == 0:
        return x
    pad = [(0, 0)] * x.ndim
    pad[1] = (offset, 0)
    return mx.pad(x, pad)[:, : x.shape[1]]


def _mask_diag(mask, offset):
    m = mask[:, 0]
    T = m.shape[-1]
    if offset >= T:
        return mx.zeros(m.shape[:-2] + (T,), m.dtype)
    d = mx.diagonal(m, offset=-offset, axis1=-2, axis2=-1)
    if offset == 0:
        return d
    return mx.pad(d, [(0, 0), (offset, 0)])


def _rms_unit(x, epsilon=1e-6):
    xf = x.astype(mx.float32)
    return xf * mx.rsqrt(mx.mean(xf ** 2, axis=-1, keepdims=True) + epsilon)


def zc_rms_norm(x, scale, dtype, epsilon=1e-6):
    rms = mx.sqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + epsilon)
    return ((1 + scale) * x / rms).astype(dtype)


def _sinkhorn(logits, iters=20):
    log_K = logits
    for _ in range(iters):
        log_K = log_K - mx.logsumexp(log_K, axis=-1, keepdims=True)
        log_K = log_K - mx.logsumexp(log_K, axis=-2, keepdims=True)
    return mx.exp(log_K)


def _dense(x, kernel, dtype):
    return x.astype(dtype) @ kernel.astype(dtype)


def precompute_rope_freqs(head_dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    angles = mx.arange(seq_len).astype(mx.float32)[:, None] * freqs[None, :]
    return mx.cos(angles), mx.sin(angles)


def apply_rope(x, cos, sin):
    T = x.shape[2]
    half = x.shape[-1] // 2
    cos = cos[:T][None, None, :, :]
    sin = sin[:T][None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def engram_indices(tokens, orders, heads, slots, seed_heads=0):
    u = tokens.astype(mx.uint32)
    stride = seed_heads or heads
    prime = mx.array(_ENGRAM_PRIME, mx.uint32)
    idx = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (_ENGRAM_SEED * (oi * stride + h + 1)) & 0xFFFFFFFF
            acc = mx.full(u.shape, seed, mx.uint32)
            for j in range(order):
                acc = (acc ^ _shift_right(u, j)) * prime
            acc = acc ^ (acc >> mx.array(15, mx.uint32))
            idx.append((acc % mx.array(slots, mx.uint32)).astype(mx.int32))
    return mx.stack(idx, axis=-1)


def engram_kv(p, cfg, tokens, mask, num, dtype):
    if not cfg.engram_layers:
        return None
    orders, heads, _ = engram_geometry(cfg)
    dilation = max(orders)
    indices = engram_indices(tokens, orders, heads, cfg.engram_slots,
                             getattr(cfg, "engram_seed_heads", 0))
    ngram_ok = mx.stack([_mask_diag(mask, o - 1) for o in orders for _ in range(heads)], axis=-1)
    tap_ok = [_mask_diag(mask, j * dilation) for j in range(ENGRAM_CONV_TAPS)]
    ks, vs = [], []
    for site in range(len(cfg.engram_layers)):
        name = f"engrams_{site}/"
        tables = p[name + "embedding"]
        num_tables, slots, sub_dim = tables.shape
        flat = indices + mx.arange(num_tables, dtype=mx.int32) * slots
        fetched = tables.reshape(num_tables * slots, sub_dim)[flat] * ngram_ok[..., None]
        e = num.act(fetched.reshape(*indices.shape[:2], num_tables * sub_dim).astype(dtype))
        k = _dense(e, p[name + "key_proj/kernel"], dtype)
        v = _dense(e, p[name + "value_proj/kernel"], dtype)
        taps = p[name + "taps"].astype(dtype)
        v = sum(taps[j] * _shift_right(v, j * dilation) * tap_ok[j][..., None]
                for j in range(ENGRAM_CONV_TAPS))
        ks.append(k)
        vs.append(v)
    return ks, vs


def attention(lp, cfg, x, mask, rope, num, dtype):
    qk_hd, v_hd = head_dims(cfg)
    H, KV = cfg.num_heads, cfg.num_kv_heads
    B, T, _ = x.shape
    x = num.act(x)
    q = _dense(x, lp["self_attn/q_proj/kernel"], dtype)
    k = _dense(x, lp["self_attn/k_proj/kernel"], dtype)
    v = _dense(x, lp["self_attn/v_proj/kernel"], dtype)

    n_taps = getattr(cfg, "qkv_conv_taps", 0)
    if n_taps:
        tap_ok = [None] + [_mask_diag(mask, j)[..., None].astype(dtype) for j in range(1, n_taps)]

        def conv(z, taps):
            taps = taps.astype(dtype)
            return sum(taps[j] * (z if j == 0 else _shift_right(z, j) * tap_ok[j])
                       for j in range(n_taps))

        q = conv(q, lp["self_attn/q_taps"])
        k = conv(k, lp["self_attn/k_taps"])
        v = conv(v, lp["self_attn/v_taps"])

    q = q.reshape(B, T, H, qk_hd).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, KV, qk_hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, KV, v_hd).transpose(0, 2, 1, 3)
    q = zc_rms_norm(q, lp["self_attn/q_norm/scale"], dtype)
    k = zc_rms_norm(k, lp["self_attn/k_norm/scale"], dtype)
    cos, sin = rope
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    q, k, v = num.query(q), num.kv(k), num.kv(v)

    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(qk_hd), mask=mask)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, H * v_hd)
    out = out * mx.sigmoid(_dense(x, lp["self_attn/gate_proj/kernel"], dtype))
    return _dense(num.act(out), lp["self_attn/out_proj/kernel"], dtype)


def _kron_apply(z, a, b):
    lead = z.shape[:-1]
    z = z.reshape(*lead, a.shape[0], b.shape[0])
    return (a.T @ z @ b).reshape(*lead, a.shape[0] * b.shape[0])


@functools.lru_cache(maxsize=None)
def _perms(n, split):
    return tuple(mx.array(np.asarray(p)) for p in _hada_perms(n, split))


def _silu(x):
    return x * mx.sigmoid(x)


def hadamard_mlp(lp, cfg, x, dtype):
    d_model = cfg.d_model
    n = 1 << (d_model - 1).bit_length()
    p1, p2 = _perms(n, bool(getattr(cfg, "ladder_widths", ())))
    w = {k: lp["hadamard_mlp/" + k].astype(dtype)
         for k in ("w1a", "w1b", "w2a", "w2b", "w3a", "w3b",
                   "d1", "d2", "b2", "d3", "d4", "cond_v", "cond_u")}
    cond = 1 + mx.softmax(x @ w["cond_v"], axis=-1) @ w["cond_u"]
    pad = n - d_model
    z = mx.pad(x, [(0, 0), (0, 0), (0, pad)]) if pad else x
    z = _kron_apply(w["d1"] * z, w["w1a"], w["w1b"])[..., p1]
    z = _kron_apply(_silu(w["d2"] * cond * z + w["b2"]), w["w2a"], w["w2b"])[..., p2]
    z = _kron_apply(w["d3"] * z, w["w3a"], w["w3b"])
    return (w["d4"] * z)[..., :d_model]


def block(lp, cfg, x, mask, rope, num, dtype, site_kv):
    if site_kv is not None:
        ek, ev = site_kv
        alpha = mx.sigmoid(mx.sum(_rms_unit(x) * _rms_unit(ek), axis=-1) / math.sqrt(cfg.d_model))
        x = x + (alpha[..., None] * ev.astype(mx.float32)).astype(x.dtype)

    skip = x
    x = zc_rms_norm(x, lp["ZCRMSNorm_0/scale"], dtype)
    x = attention(lp, cfg, x, mask, rope, num, dtype)
    x = zc_rms_norm(x, lp["post_attn_norm/scale"], dtype)
    x = skip + mx.sigmoid(lp["attn_gate"]).astype(dtype) * x

    skip = x
    x = zc_rms_norm(x, lp["pre_hada_norm/scale"], dtype)
    return skip + hadamard_mlp(lp, cfg, x, dtype)


def _advance(cfg, num, dtype, stream, lp, pre_off, post_off, mask, rope, site_kv):
    B, T, n, C = stream.shape
    xf = stream.astype(mx.float32)
    nx = num.act(_rms_unit(stream.reshape(B, T, n * C)))
    hpre = mx.sigmoid(lp["a_pre"] * (nx @ lp["phi_pre"]) + lp["b_pre"] + pre_off)
    u = mx.sum(hpre[..., None] * xf, axis=2).astype(dtype)
    y = block(lp, cfg, u, mask, rope, num, dtype, site_kv) - u
    hpost = 2 * mx.sigmoid(lp["a_post"] * (nx @ lp["phi_post"]) + lp["b_post"] + post_off)
    hres = _sinkhorn(lp["a_res"] * (nx @ lp["phi_res"]).reshape(B, T, n, n) + lp["b_res"])
    return (hres @ xf + hpost[..., None] * y.astype(mx.float32)[:, :, None, :]).astype(dtype)


def _layer_params(p, layer):
    lp = {k[len(BLOCK):]: v[layer] for k, v in p.items() if k.startswith(BLOCK)}
    for name in ("phi_pre", "phi_post", "phi_res", "b_pre", "b_post", "b_res",
                 "a_pre", "a_post", "a_res"):
        lp[name] = p["stack/mhc_" + name][layer]
    return lp


def _lane_offsets(positions, n):
    lane = mx.array(np.eye(n, dtype=np.float32)[np.asarray(positions) % n])
    return 8 * lane - 4, -4 * (1 - lane)


def stack(p, cfg, x, mask, rope, ekv, num, dtype, exit_depth=None, subnetwork_only=False):
    if subnetwork_only and exit_depth is None:
        raise ValueError("subnetwork_only requires exit_depth")
    L, n = cfg.num_layers, cfg.mhc_lanes
    x = x.astype(dtype)
    local_mask = None
    if cfg.sliding_window:
        pos = mx.arange(x.shape[1])
        local_mask = mask & ((pos[:, None] - pos[None, :]) < cfg.sliding_window)[None, None]
    sites = {layer: s for s, layer in enumerate(cfg.engram_layers)}

    ranks = ladder_layer_ranks(cfg)
    pre_off, post_off = _lane_offsets(np.arange(L), n)
    if exit_depth is not None:
        active = np.asarray(ranks) < exit_depth
        sub_pre_off, sub_post_off = _lane_offsets(np.cumsum(active) - 1, n)
    capture = exit_depth is not None and not subnetwork_only

    x = mx.broadcast_to(x[:, :, None, :], (*x.shape[:2], n, x.shape[-1]))
    sub_x = x
    for layer in range(L):
        lp = _layer_params(p, layer)
        layer_mask = mask if local_mask is None or layer in cfg.global_layers else local_mask
        s = sites.get(layer)
        site_kv = (ekv[0][s], ekv[1][s]) if ekv is not None and s is not None else None
        step = functools.partial(_advance, cfg, num, dtype)
        if cfg.remat:
            step = mx.checkpoint(step)
        in_sub = exit_depth is not None and ranks[layer] < exit_depth
        if subnetwork_only:
            if in_sub:
                x = step(x, lp, sub_pre_off[layer], sub_post_off[layer], layer_mask, rope, site_kv)
            continue
        x = step(x, lp, pre_off[layer], post_off[layer], layer_mask, rope, site_kv)
        if capture and in_sub:
            sub_x = step(sub_x, lp, sub_pre_off[layer], sub_post_off[layer], layer_mask, rope,
                         site_kv)

    final = p["stack/final_norm/scale"]
    x = zc_rms_norm(mx.mean(x, axis=2), final, dtype)
    if capture:
        return x, zc_rms_norm(mx.mean(sub_x, axis=2), final, dtype)
    return x, None


def forward(p, cfg, tokens, mask=None, quant=False, exit_depth=None, exit_only=False,
            subnetwork_only=False):
    """SimpleAttentionNetwork.__call__ on MLX: logits, or (logits, exit_logits) with exit_depth.

    `p` comes from to_mlx; under quant it must already hold the CQ weights, as the
    JAX path passes cq_ste_params(...) to model.apply.
    """
    dtype = MX_DTYPES[cfg.dtype]
    num = Numerics(cfg, quant)
    if mask is None:
        mask = make_causal_mask(tokens.shape[1])
    embedding = p["embedding/embedding"]
    x = embedding[tokens] * math.sqrt(cfg.d_model)
    rope = precompute_rope_freqs(head_dims(cfg)[0], tokens.shape[1], cfg.rope_theta)
    ekv = engram_kv(p, cfg, tokens, mask, num, dtype)
    x, exit_x = stack(p, cfg, x, mask, rope, ekv, num, dtype, exit_depth=exit_depth,
                      subnetwork_only=subnetwork_only)
    head = embedding[: cfg.out_vocab] if cfg.out_vocab else embedding

    def logits(h):
        return num.act(h).astype(mx.float32) @ head.T

    if exit_x is None:
        return logits(x)
    if exit_only:
        return logits(exit_x)
    return logits(x), logits(exit_x)
