import math
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn.initializers as jinit
import flax.linen as nn

from . import quantize as _quantize
from .quantize import fake_quant_act, head_weight
def _aq(x, quant):
    if quant is False:
        return x
    return jax.lax.cond(quant, fake_quant_act, lambda t: t, x)


def default_init():
    return jinit.normal(stddev=0.02)


def residual_init(num_layers):
    return jinit.normal(stddev=0.02 / math.sqrt(2 * num_layers))


DTYPE_MAP = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}

class ZCRMSNorm(nn.Module):
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", jinit.zeros, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + self.epsilon)
        return ((1 + scale) * x / rms).astype(self.dtype)



@dataclass
class TransformerConfig:
    vocab_size: int = 16384
    d_model: int = 768
    num_heads: int = 12
    num_kv_heads: int = 2
    num_layers: int = 20
    qk_head_dim: int = 48
    v_head_dim: int = 64
    max_seq_len: int = 4096
    pad_token_id: int = 0
    embedding_dim: int = 128
    embedding_probes: int = 4
    embedding_queries: int = 4
    confidence_probes: int = 4
    confidence_queries: int = 4
    router_probes: int = 4
    router_queries: int = 4
    rope_theta: float = 100000.0
    dtype: str = "bfloat16"
    flash: bool = True
    engram_orders: tuple = (2, 3)
    engram_heads: int = 0
    engram_slots: int = 18432
    engram_seed_heads: int = 0
    engram_layers: tuple = (3, 7, 11, 15, 19)
    global_layers: tuple = (4, 9, 14, 19)
    sliding_window: int = 1024
    ladder_depths: tuple = ()
    ladder_sample: bool = False
    ladder_widths: tuple = ()
    ladder_order: tuple = ()
    mhc_lanes: int = 4
    qkv_conv_taps: int = 3
    out_vocab: int = 0
    kv_window: int = 0
    kv_bits: int = 8
    act_bits: int = 8
    weight_bits: str = ""
    remat: bool = True
    scan_unroll: int = 1

    def __init__(self, **kwargs):
        valid = {f.name for f in self.__dataclass_fields__.values()}
        for k, v in kwargs.items():
            if k in valid:
                setattr(self, k, v)
        legacy_attn = kwargs.get("attn_dim", 0)
        if legacy_attn:
            legacy_head = legacy_attn // self.num_heads
            if "qk_head_dim" not in kwargs:
                self.qk_head_dim = legacy_head
            if "v_head_dim" not in kwargs:
                self.v_head_dim = legacy_head
            if "sliding_window" not in kwargs:
                self.sliding_window = 0
            if "global_layers" not in kwargs:
                self.global_layers = ()
        self.engram_layers = tuple(self.engram_layers)
        self.global_layers = tuple(self.global_layers)
        self.ladder_depths = tuple(self.ladder_depths)
        self.ladder_widths = tuple(self.ladder_widths)
        self.ladder_order = tuple(self.ladder_order)

    @classmethod
    def from_saved(cls, saved):
        if not isinstance(saved, dict):
            saved = dict(saved.__dict__)
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        saved = {k: v for k, v in saved.items() if k in allowed}
        for key, off in (("qk_head_dim", 0), ("v_head_dim", 0),
                         ("sliding_window", 0), ("global_layers", ()),
                         ("ladder_depths", ()), ("ladder_sample", False),
                         ("ladder_widths", ()), ("ladder_order", ()),
                         ("engram_seed_heads", 0),
                         ("qkv_conv_taps", 0), ("out_vocab", 0)):
            saved.setdefault(key, off)
        return cls(**saved)

    @property
    def jax_dtype(self):
        return DTYPE_MAP[self.dtype]


def head_dims(config):
    legacy = (getattr(config, "attn_dim", 0) or config.d_model) // config.num_heads
    qk = getattr(config, "qk_head_dim", 0) or legacy
    v = getattr(config, "v_head_dim", 0) or legacy
    return qk, v


def _ladder_layer_order(num_layers):
    if num_layers < 1:
        raise ValueError("models require at least one layer")
    if num_layers == 1:
        return (0,)
    selected = [0, num_layers - 1]
    order = list(selected)
    while len(order) < num_layers:
        selected.sort()
        _gap, left, right = max(
            ((right - left, left, right)
             for left, right in zip(selected, selected[1:])
             if right - left > 1),
            key=lambda item: (item[0], -item[1]),
        )
        candidate = (left + right) // 2
        selected.append(candidate)
        order.append(candidate)
    return tuple(order)


def ladder_order(spec):
    """Block selection order for a model: the bisection order of its depth, or
    the parent's order carried by a sliced rung so nested slices stay trained."""
    if isinstance(spec, int):
        return _ladder_layer_order(spec)
    saved = tuple(getattr(spec, "ladder_order", ()) or ())
    if saved:
        if sorted(saved) != list(range(spec.num_layers)):
            raise ValueError(f"ladder_order {saved} is not a permutation of "
                             f"{spec.num_layers} blocks")
        return saved
    return _ladder_layer_order(spec.num_layers)



def ladder_layer_indices(spec, depth):
    """Return the stable, nested original-layer indices for a depth rung.

    Every deployable rung keeps both endpoint blocks. Interior blocks are
    added one at a time by bisecting the largest remaining gap, producing
    deterministic nested subnetworks with spatially balanced coverage.
    `spec` is a layer count or a config (whose ladder_order wins).
    """
    num_layers = spec if isinstance(spec, int) else spec.num_layers
    if not 2 <= depth <= num_layers:
        raise ValueError(
            f"ladder depth must be in [2, {num_layers}], got {depth}")
    return tuple(sorted(ladder_order(spec)[:depth]))


def ladder_config(config, depth):
    assert depth < config.num_layers
    assert tuple(config.engram_layers) == tuple(sorted(config.engram_layers))
    selected = ladder_layer_indices(config, depth)
    remap = {layer: i for i, layer in enumerate(selected)}
    fields = dict(config.__dict__)
    fields["num_layers"] = depth
    fields["ladder_order"] = tuple(remap[l] for l in ladder_order(config) if l in remap)
    fields["global_layers"] = tuple(
        remap[l] for l in config.global_layers if l in remap)
    fields["engram_layers"] = tuple(
        remap[l] for l in config.engram_layers if l in remap)
    fields["ladder_depths"] = ()
    fields["ladder_sample"] = False
    return TransformerConfig(**fields)


def ladder_slice(params, config, depth):
    selected = ladder_layer_indices(config, depth)
    selected_set = set(selected)
    selected_array = jnp.asarray(selected, dtype=jnp.int32)
    rows = jnp.asarray((0, *(layer + 1 for layer in selected)), jnp.int32)
    out = {}
    for k, v in params.items():
        if k == "stack":
            out[k] = {sk: (sv if sk == "final_norm"
                           else jax.tree.map(
                               lambda a: jnp.take(a, selected_array, axis=0), sv))
                      for sk, sv in v.items()}
        elif k in HEAD_KEYS:
            def take_rows(name, axis):
                return jax.tree.map(lambda a: jnp.take(a, rows, axis=axis), v[name])
            out[k] = {**v, "probes": take_rows("probes", 0),
                      "gain": take_rows("gain", 0), "row_bias": take_rows("row_bias", 1)}
        elif k.startswith("engrams_"):
            site = int(k.rsplit("_", 1)[1])
            if config.engram_layers[site] in selected_set:
                new_site = sum(
                    layer in selected_set for layer in config.engram_layers[:site])
                out[f"engrams_{new_site}"] = v
        else:
            out[k] = v
    return out


def ladder_layer_ranks(spec):
    """Map each original block to its selection rank in the nested ladder."""
    order = ladder_order(spec)
    ranks = [0] * len(order)
    for rank, layer in enumerate(order):
        ranks[layer] = rank
    return tuple(ranks)



def ladder_row_keep(config, exit_depth=None):
    if exit_depth is None:
        return None
    ranks = jnp.asarray(ladder_layer_ranks(config), jnp.int32)
    return jnp.concatenate([jnp.ones((1,), jnp.float32),
                            (ranks < exit_depth).astype(jnp.float32)])



def width_config(config, w):
    assert 2 * w == config.d_model
    assert config.d_model & (config.d_model - 1) == 0
    assert config.num_heads % 2 == 0
    assert _hada_blocks(w)[1] == _hada_blocks(config.d_model)[1]
    orders, heads, _ = engram_geometry(config)
    if config.engram_layers:
        assert heads % 2 == 0
    fields = dict(config.__dict__)
    fields.update(
        d_model=w,
        num_heads=config.num_heads // 2,
        engram_seed_heads=getattr(config, "engram_seed_heads", 0) or heads,
        ladder_depths=(),
        ladder_sample=False,
        ladder_widths=(),
    )
    return TransformerConfig(**fields)


def _parameter_path(path):
    """Return a stable parameter path through optional Flax metadata nodes."""
    parts = []
    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif getattr(entry, "name", None) == "value":
            continue
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        else:
            parts.append(str(entry))
    return "/".join(parts)


def width_slice(params, config, w):
    D = config.d_model
    orders, heads, sub_dim = engram_geometry(config)
    nh = config.num_heads // 2
    qk_hd, v_hd = head_dims(config)
    q_cols, o_rows = nh * qk_hd, nh * v_hd
    ba_c = _hada_blocks(1 << (w - 1).bit_length())[0]
    kept = (np.concatenate([oi * heads + np.arange(heads // 2)
                            for oi in range(len(orders))])
            if config.engram_layers else None)
    kv_rows = ((kept[:, None] * sub_dim + np.arange(sub_dim)).ravel()
               if config.engram_layers else None)
    lane_rows = (np.arange(config.mhc_lanes)[:, None] * D + np.arange(w)).ravel()

    def cut(path, a):
        p = _parameter_path(path)
        if p.startswith("ab_scales"):
            return a
        if "engrams_" in p:
            if p.endswith("/embedding"):
                return a[kept]
            if p.endswith("key_proj/kernel") or p.endswith("value_proj/kernel"):
                return a[kv_rows][:, :w]
            if p.endswith("/taps"):
                return a[..., :w]
            return a
        if ("embedding_head" in p or "confidence_head" in p
                or "router_head" in p):
            if p.endswith("/probes") or p.endswith("/query"):
                return a[..., :w]
            if p.endswith("proj/kernel"):
                return a.reshape(-1, D, a.shape[-1])[:, :w].reshape(-1, a.shape[-1])
            return a
        if "mhc_phi" in p:
            return a[:, lane_rows]
        if "hadamard_mlp" in p:
            last = p.rsplit("/", 1)[1]
            if last in ("w1a", "w2a", "w3a"):
                return a[..., :ba_c, :ba_c]
            if last in ("d1", "d2", "b2", "d3", "d4"):
                return a[..., :w]
            if last == "cond_v":
                return a[..., :w, :]
            if last == "cond_u":
                return a[..., :w]
            return a
        if p.endswith("q_proj/kernel"):
            return a[..., :w, :q_cols]
        if p.endswith("gate_proj/kernel"):
            return a[..., :w, :o_rows]
        if p.endswith("k_proj/kernel") or p.endswith("v_proj/kernel"):
            return a[..., :w, :]
        if p.endswith("out_proj/kernel"):
            return a[..., :o_rows, :w]
        if p.endswith("/q_taps"):
            return a[..., :q_cols]
        if p.endswith("/k_taps") or p.endswith("/v_taps"):
            return a
        if p.endswith("q_norm/scale") or p.endswith("k_norm/scale"):
            return a
        if p.endswith("/scale"):
            return a[..., :w]
        if p == "embedding/embedding":
            return a[..., :w]
        return a

    return jax.tree_util.tree_map_with_path(cut, params)



def precompute_rope_freqs(head_dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2).astype(jnp.float32) / head_dim))
    t = jnp.arange(seq_len).astype(jnp.float32)
    angles = jnp.outer(t, freqs)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
    T = x.shape[2]
    half = x.shape[-1] // 2
    cos = cos[:T][None, None, :, :]
    sin = sin[:T][None, None, :, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


ENGRAM_SUB_DIM = 128
ENGRAM_CONV_TAPS = 4
_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193


def engram_geometry(config):
    orders = tuple(config.engram_orders)
    heads = config.engram_heads or max(1, config.d_model // (len(orders) * ENGRAM_SUB_DIM))
    sub_dim = config.d_model // (len(orders) * heads)
    return orders, heads, sub_dim


def _shift_right(x, offset):
    if offset == 0:
        return x
    pad = [(0, 0)] * x.ndim
    pad[1] = (offset, 0)
    return jnp.pad(x, pad)[:, : x.shape[1]]


def _mask_diag(mask, offset):
    m = mask[:, 0]
    T = m.shape[-1]
    if offset >= T:
        return jnp.zeros(m.shape[:-2] + (T,), m.dtype)
    d = jnp.diagonal(m, offset=-offset, axis1=-2, axis2=-1)
    if offset == 0:
        return d
    return jnp.pad(d, ((0, 0), (offset, 0)))


def engram_indices(tokens, orders, heads, slots, seed_heads=0):
    u = tokens.astype(jnp.uint32)
    stride = seed_heads or heads
    idx = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (_ENGRAM_SEED * (oi * stride + h + 1)) & 0xFFFFFFFF
            acc = jnp.full_like(u, jnp.uint32(seed))
            for j in range(order):
                acc = (acc ^ _shift_right(u, j)) * jnp.uint32(_ENGRAM_PRIME)
            acc = acc ^ (acc >> jnp.uint32(15))
            idx.append((acc % jnp.uint32(slots)).astype(jnp.int32))
    return jnp.stack(idx, axis=-1)


def _rms_unit(x, epsilon=1e-6):
    xf = x.astype(jnp.float32)
    return xf * jax.lax.rsqrt(jnp.mean(xf ** 2, axis=-1, keepdims=True) + epsilon)


def _conv_identity_init(key, shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype).at[0].set(1.0)


def _sinkhorn(logits, iters=20):
    log_K = logits
    for _ in range(iters):
        log_K = log_K - jax.nn.logsumexp(log_K, axis=-1, keepdims=True)
        log_K = log_K - jax.nn.logsumexp(log_K, axis=-2, keepdims=True)
    return jnp.exp(log_K)


def _res_identity_init(key, shape, dtype=jnp.float32):
    return jnp.broadcast_to(4.0 * jnp.eye(shape[-1], dtype=dtype), shape)


class Engram(nn.Module):
    d_model: int
    num_tables: int
    slots: int
    sub_dim: int
    num_layers: int
    conv_dilation: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, indices, ngram_ok, tap_ok, quant=False):
        tables = self.param("embedding", default_init(),
                            (self.num_tables, self.slots, self.sub_dim))
        fetched = tables[jnp.arange(self.num_tables), indices]
        fetched = fetched * ngram_ok[..., None]
        e = fetched.reshape(*indices.shape[:2], self.num_tables * self.sub_dim)
        e = _aq(e.astype(self.dtype), quant)
        k = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False,
                     kernel_init=default_init(), name="key_proj")(e)
        v = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False,
                     kernel_init=residual_init(self.num_layers), name="value_proj")(e)
        taps = self.param("taps", _conv_identity_init,
                          (ENGRAM_CONV_TAPS, self.d_model)).astype(self.dtype)
        v = sum(taps[j] * _shift_right(v, j * self.conv_dilation) * tap_ok[j][..., None]
                for j in range(ENGRAM_CONV_TAPS))
        return k, v


class MultiHeadAttention(nn.Module):
    num_heads: int
    num_kv_heads: int
    d_model: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    flash: bool = True
    qk_head_dim: int = 0
    v_head_dim: int = 0
    qkv_conv_taps: int = 0

    @nn.compact
    def __call__(self, x, mask=None, rope=None, quant=False):
        qk_hd = self.qk_head_dim or self.d_model // self.num_heads
        v_hd = self.v_head_dim or self.d_model // self.num_heads
        out_dim = self.num_heads * v_hd
        B = x.shape[0]

        x = _aq(x, quant)
        q = nn.Dense(self.num_heads * qk_hd, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="q_proj")(x)
        k = nn.Dense(self.num_kv_heads * qk_hd, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="k_proj")(x)
        v = nn.Dense(self.num_kv_heads * v_hd, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="v_proj")(x)

        if self.qkv_conv_taps:
            n_taps = self.qkv_conv_taps
            qt = self.param("q_taps", _conv_identity_init,
                            (n_taps, self.num_heads * qk_hd)).astype(self.dtype)
            kt = self.param("k_taps", _conv_identity_init,
                            (n_taps, self.num_kv_heads * qk_hd)).astype(self.dtype)
            vt = self.param("v_taps", _conv_identity_init,
                            (n_taps, self.num_kv_heads * v_hd)).astype(self.dtype)
            tap_ok = [None] + [
                (_mask_diag(mask, j)[..., None].astype(q.dtype)
                 if mask is not None else None)
                for j in range(1, n_taps)]

            def tapped(z, j):
                shifted = _shift_right(z, j)
                return shifted if tap_ok[j] is None else shifted * tap_ok[j]

            q = sum(qt[j] * tapped(q, j) for j in range(n_taps))
            k = sum(kt[j] * tapped(k, j) for j in range(n_taps))
            v = sum(vt[j] * tapped(v, j) for j in range(n_taps))

        q = q.reshape(B, -1, self.num_heads, qk_hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_kv_heads, qk_hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_kv_heads, v_hd).transpose(0, 2, 1, 3)

        q = ZCRMSNorm(dtype=self.dtype, name="q_norm")(q)
        k = ZCRMSNorm(dtype=self.dtype, name="k_norm")(k)

        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        q = _quantize.maybe_quant_query(q, quant)
        k = _quantize.maybe_quant_kv(k, quant)
        v = _quantize.maybe_quant_kv(v, quant)

        if self.flash:
            fused_hd = max(qk_hd, v_hd)
            qf = q * jnp.asarray(math.sqrt(fused_hd / qk_hd), q.dtype)
            pad_qk = fused_hd - qk_hd
            if pad_qk:
                qf = jnp.pad(qf, ((0, 0), (0, 0), (0, 0), (0, pad_qk)))
                k = jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, pad_qk)))
            pad_v = fused_hd - v_hd
            vf = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad_v))) if pad_v else v
            impl = ("cudnn" if jax.default_backend() == "gpu"
                    and q.dtype in (jnp.bfloat16, jnp.float16) else None)
            out = jax.nn.dot_product_attention(
                qf.transpose(0, 2, 1, 3),
                k.transpose(0, 2, 1, 3),
                vf.transpose(0, 2, 1, 3),
                mask=mask,
                implementation=impl,
            )
            out = out[..., :v_hd].reshape(B, -1, out_dim)
        else:
            repeats = self.num_heads // self.num_kv_heads
            if repeats > 1:
                k = jnp.repeat(k, repeats, axis=1)
                v = jnp.repeat(v, repeats, axis=1)

            scale = jnp.sqrt(jnp.float32(qk_hd))
            attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale

            if mask is not None:
                attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)

            attn_weights = nn.softmax(attn_weights, axis=-1)

            out = jnp.matmul(attn_weights, v)
            out = out.transpose(0, 2, 1, 3).reshape(B, -1, out_dim)

        out = out * nn.sigmoid(
            nn.Dense(out_dim, dtype=self.dtype, use_bias=False,
                     kernel_init=default_init(), name="gate_proj")(x))
        out = _aq(out, quant)
        return nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=residual_init(self.num_layers), name="out_proj")(out)


def _walsh_matrix(n):
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return jnp.asarray(H / np.sqrt(n))


HADA_COND_RANK = 8
_HADA_PERM_SEEDS = (11, 13)


def _hada_blocks(n):
    b = 1 << ((n - 1).bit_length() // 2)
    return b, n // b


def _hada_perms(n, split=False):
    if split:
        h = n // 2
        return tuple(jnp.asarray(np.concatenate(
            [np.random.RandomState(s).permutation(h),
             h + np.random.RandomState(s + 977).permutation(h)]))
            for s in _HADA_PERM_SEEDS)
    return tuple(jnp.asarray(np.random.RandomState(s).permutation(n))
                 for s in _HADA_PERM_SEEDS)


def _kron_apply(z, a, b):
    lead = z.shape[:-1]
    z = z.reshape(*lead, a.shape[0], b.shape[0])
    z = jnp.einsum("...ij,ik,jl->...kl", z, a, b)
    return z.reshape(*lead, a.shape[0] * b.shape[0])


def _walsh_init(size):
    w = _walsh_matrix(size)
    return lambda key, shape: jnp.asarray(w)


class HadamardMLP(nn.Module):
    d_model: int
    dtype: jnp.dtype = jnp.bfloat16
    split_perms: bool = False

    @nn.compact
    def __call__(self, x):
        n = 1 << (self.d_model - 1).bit_length()
        ba, bb = _hada_blocks(n)
        p1, p2 = _hada_perms(n, self.split_perms)
        factors = [
            (self.param(f"w{s}a", _walsh_init(ba), (ba, ba)).astype(self.dtype),
             self.param(f"w{s}b", _walsh_init(bb), (bb, bb)).astype(self.dtype))
            for s in (1, 2, 3)
        ]
        d1 = self.param("d1", jinit.ones, (n,)).astype(self.dtype)
        d2 = self.param("d2", jinit.ones, (n,)).astype(self.dtype)
        b2 = self.param("b2", jinit.zeros, (n,)).astype(self.dtype)
        d3 = self.param("d3", jinit.ones, (n,)).astype(self.dtype)
        d4 = self.param("d4", jinit.constant(0.02), (n,)).astype(self.dtype)
        cond_v = self.param("cond_v", default_init(),
                            (self.d_model, HADA_COND_RANK)).astype(self.dtype)
        cond_u = self.param("cond_u", jinit.zeros,
                            (HADA_COND_RANK, n)).astype(self.dtype)

        cond = 1 + nn.softmax(x @ cond_v, axis=-1) @ cond_u
        pad = n - self.d_model
        z = jnp.pad(x, ((0, 0), (0, 0), (0, pad))) if pad else x
        z = _kron_apply(d1 * z, *factors[0])[..., p1]
        z = _kron_apply(nn.silu(d2 * cond * z + b2), *factors[1])[..., p2]
        z = _kron_apply(d3 * z, *factors[2])
        return (d4 * z)[..., : self.d_model]


class Block(nn.Module):
    num_heads: int
    num_kv_heads: int
    d_model: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    flash: bool = True
    qk_head_dim: int = 0
    v_head_dim: int = 0
    qkv_conv_taps: int = 0
    hada_split: bool = False

    def _gate(self, name):
        return nn.sigmoid(self.param(name, jinit.zeros, ())).astype(self.dtype)

    @nn.compact
    def __call__(self, x, mask=None, rope=None, quant=False, engram_kv=None, site_flags=None):
        if engram_kv is not None:
            ek, ev = engram_kv
            alpha = nn.sigmoid(jnp.einsum("btd,sbtd->sbt", _rms_unit(x), _rms_unit(ek))
                               / math.sqrt(self.d_model))
            x = x + jnp.einsum("s,sbt,sbtd->btd", site_flags.astype(jnp.float32),
                               alpha, ev.astype(jnp.float32)).astype(x.dtype)

        skip = x
        x = ZCRMSNorm(dtype=self.dtype)(x)
        x = MultiHeadAttention(self.num_heads, self.num_kv_heads, self.d_model, self.num_layers,
                               self.dtype, self.flash, qk_head_dim=self.qk_head_dim,
                               v_head_dim=self.v_head_dim,
                               qkv_conv_taps=self.qkv_conv_taps,
                               name="self_attn")(x, mask=mask, rope=rope, quant=quant)
        x = ZCRMSNorm(dtype=self.dtype, name="post_attn_norm")(x)
        x = skip + self._gate("attn_gate") * x

        skip = x
        x = ZCRMSNorm(dtype=self.dtype, name="pre_hada_norm")(x)
        x = HadamardMLP(self.d_model, self.dtype, split_perms=self.hada_split,
                        name="hadamard_mlp")(x)
        return skip + x


class _ScanBody(nn.Module):
    num_heads: int
    num_kv_heads: int
    d_model: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    flash: bool = True
    collect_hidden: bool = False
    qk_head_dim: int = 0
    v_head_dim: int = 0
    qkv_conv_taps: int = 0
    hada_split: bool = False
    capture_subnetwork: bool = False
    subnetwork_only: bool = False

    @nn.compact
    def __call__(self, carry, xs, mask, local_mask, rope, quant, engram_kv,
                 exit_depth):
        if self.capture_subnetwork:
            x, sub_x = carry
        else:
            x = carry
        site_flags, gflag, hc, layer_rank = xs
        if local_mask is not None:
            mask = jnp.where(gflag > 0, mask, local_mask)
        block = Block(
            self.num_heads, self.num_kv_heads, self.d_model, self.num_layers,
            self.dtype, self.flash, qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim, qkv_conv_taps=self.qkv_conv_taps,
            hada_split=self.hada_split, name="block",
        )
        def advance(stream, pre_off, post_off, block_module):
            B, T, n, C = stream.shape
            xf = stream.astype(jnp.float32)
            nx = _aq(_rms_unit(stream.reshape(B, T, n * C)), quant)
            hpre = nn.sigmoid(
                hc["a_pre"] * (nx @ hc["phi_pre"].astype(jnp.float32))
                + hc["b_pre"] + pre_off)
            u = jnp.einsum("btn,btnc->btc", hpre, xf).astype(self.dtype)
            y = block_module(u, mask=mask, rope=rope, quant=quant,
                             engram_kv=engram_kv, site_flags=site_flags) - u
            hpost = 2 * nn.sigmoid(
                hc["a_post"] * (nx @ hc["phi_post"].astype(jnp.float32))
                + hc["b_post"] + post_off)
            res = nx @ hc["phi_res"].astype(jnp.float32)
            hres = _sinkhorn(
                hc["a_res"] * res.reshape(B, T, n, n) + hc["b_res"])
            return (jnp.einsum("btij,btjc->btic", hres, xf)
                    + hpost[..., None]
                    * y.astype(jnp.float32)[:, :, None, :]).astype(self.dtype)

        if self.subnetwork_only:
            new_x = nn.cond(
                layer_rank < exit_depth,
                lambda mdl, stream: advance(
                    stream, hc["sub_pre_off"], hc["sub_post_off"], mdl),
                lambda _mdl, stream: stream,
                block,
                x,
            )
        else:
            new_x = advance(x, hc["pre_off"], hc["post_off"], block)
        if self.capture_subnetwork:
            sub_x = nn.cond(
                layer_rank < exit_depth,
                lambda mdl, stream: advance(
                    stream, hc["sub_pre_off"], hc["sub_post_off"], mdl),
                lambda _mdl, stream: stream,
                block,
                sub_x,
            )
            out = jnp.mean(sub_x, axis=2) if self.collect_hidden else None
            return (new_x, sub_x), out
        out = jnp.mean(new_x, axis=2) if self.collect_hidden else None
        return new_x, out


class Stack(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, mask=None, rope=None, engram_kv=None,
                 collect_hidden=False, quant=False, exit_depth=None,
                 subnetwork_only=False):
        cfg = self.config
        dt = cfg.jax_dtype
        x = x.astype(dt)

        site_flags = None
        if engram_kv is not None:
            flags = np.zeros((cfg.num_layers, len(cfg.engram_layers)), np.float32)
            for s, layer in enumerate(cfg.engram_layers):
                flags[layer, s] = 1.0
            site_flags = jnp.asarray(flags)

        local_mask, gflags = None, jnp.zeros((cfg.num_layers,), jnp.float32)
        if cfg.sliding_window and mask is not None:
            T = x.shape[1]
            pos = jnp.arange(T)
            band = ((pos[:, None] - pos[None, :]) < cfg.sliding_window)[None, None]
            local_mask = mask & band
            gflags = jnp.asarray([1.0 if i in cfg.global_layers else 0.0
                                  for i in range(cfg.num_layers)])

        n, L, nC = cfg.mhc_lanes, cfg.num_layers, cfg.mhc_lanes * cfg.d_model
        lane = np.eye(n, dtype=np.float32)[np.arange(L) % n]
        hc = {
            "phi_pre": self.param("mhc_phi_pre", default_init(), (L, nC, n)),
            "phi_post": self.param("mhc_phi_post", default_init(), (L, nC, n)),
            "phi_res": self.param("mhc_phi_res", default_init(), (L, nC, n * n)),
            "b_pre": self.param("mhc_b_pre", jinit.zeros, (L, n)),
            "b_post": self.param("mhc_b_post", jinit.zeros, (L, n)),
            "b_res": self.param("mhc_b_res", _res_identity_init, (L, n, n)),
            "a_pre": self.param("mhc_a_pre", jinit.constant(0.01), (L,)),
            "a_post": self.param("mhc_a_post", jinit.constant(0.01), (L,)),
            "a_res": self.param("mhc_a_res", jinit.constant(0.01), (L,)),
            "pre_off": jnp.asarray(8 * lane - 4),
            "post_off": jnp.asarray(-4 * (1 - lane)),
        }
        x = jnp.broadcast_to(x[:, :, None, :], (*x.shape[:2], n, x.shape[-1]))
        if subnetwork_only and exit_depth is None:
            raise ValueError("subnetwork_only requires exit_depth")
        capture_subnetwork = exit_depth is not None and not subnetwork_only
        if capture_subnetwork:
            carry = (x, x)
        else:
            carry = x

        layer_ranks = jnp.asarray(ladder_layer_ranks(cfg), dtype=jnp.int32)
        if exit_depth is not None:
            active = layer_ranks < exit_depth
            sub_positions = jnp.cumsum(active.astype(jnp.int32)) - 1
            sub_lane = jax.nn.one_hot(
                jnp.mod(sub_positions, n), n, dtype=jnp.float32)
            hc["sub_pre_off"] = 8 * sub_lane - 4
            hc["sub_post_off"] = -4 * (1 - sub_lane)
        else:
            sub_positions = jnp.zeros((L,), dtype=jnp.int32)
            hc["sub_pre_off"] = hc["pre_off"]
            hc["sub_post_off"] = hc["post_off"]

        ScanBlock = nn.scan(
            nn.remat(_ScanBody) if cfg.remat else _ScanBody,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            metadata_params={nn.PARTITION_NAME: "layers"},
            length=cfg.num_layers,
            unroll=cfg.scan_unroll,
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast,
                     nn.broadcast, nn.broadcast),
        )
        carry, hidden = ScanBlock(
            cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.num_layers, dt,
            cfg.flash, collect_hidden, qk_head_dim=cfg.qk_head_dim,
            v_head_dim=cfg.v_head_dim,
            qkv_conv_taps=getattr(cfg, "qkv_conv_taps", 0),
            hada_split=bool(getattr(cfg, "ladder_widths", ())),
            capture_subnetwork=capture_subnetwork,
            subnetwork_only=subnetwork_only,
            name="layers",
        )(carry, (site_flags, gflags, hc, layer_ranks),
          mask, local_mask, rope, quant, engram_kv,
          jnp.int32(0) if exit_depth is None else exit_depth)

        if capture_subnetwork:
            x, sub_x = carry
        else:
            x = carry
        x = jnp.mean(x, axis=2)
        final_norm = ZCRMSNorm(dtype=dt, name="final_norm")
        if capture_subnetwork:
            sub_x = jnp.mean(sub_x, axis=2)
            both = final_norm(jnp.concatenate((x, sub_x), axis=0))
            x, sub_x = jnp.split(both, 2, axis=0)
            return x, hidden, sub_x
        x = final_norm(x)
        return x, hidden


def probe_pool(cells, keep, probes, gain, query, bias, row_keep=None,
               dtype=jnp.bfloat16):
    cf = cells.astype(jnp.float32)
    b, t, l1, d = cf.shape
    k, q = probes.shape[1], query.shape[0]
    scores = jnp.einsum("btld,lkd->blkt", cf,
                        probes.astype(jnp.float32)) / math.sqrt(d)
    if keep is not None:
        scores = jnp.where(keep[:, None, None, :] > 0, scores, -jnp.inf)
    r = jnp.einsum("blkt,btld->blkd", jax.nn.softmax(scores, axis=-1), cf)
    r = _rms_unit(r) * gain.astype(jnp.float32)[None, :, :, None]
    u = (jnp.einsum("blkd,qd->bqlk", r, query.astype(jnp.float32))
         / math.sqrt(d) + bias.astype(jnp.float32)[None])
    if row_keep is not None:
        u = jnp.where(row_keep[None, None, :, None] > 0, u, -jnp.inf)
    w = jax.nn.softmax(u.reshape(b, q, l1 * k), axis=-1)
    return jnp.einsum("bqm,bmd->bqd", w, r.reshape(b, l1 * k, d)
                      ).reshape(b, q * d).astype(dtype)



class HeadProjection(nn.Module):
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = None

    @nn.compact
    def __call__(self, x, quant=False):
        kernel = head_weight(
            self.param("kernel", default_init(), (x.shape[-1], self.features)),
            quant, reduce_second_last=True)
        if self.dtype is not None:
            x, kernel = x.astype(self.dtype), kernel.astype(self.dtype)
        y = x @ kernel
        if self.use_bias:
            bias = self.param("bias", jinit.zeros, (self.features,))
            y = y + (bias.astype(self.dtype) if self.dtype is not None else bias)
        return y


class ProbeHead(nn.Module):
    config: TransformerConfig
    dtype: jnp.dtype = None
    key = ""
    code = 0
    out_dim = 0

    @classmethod
    def export(cls, head):
        kernel = np.asarray(head["proj"]["kernel"]).T
        bias = (np.asarray(head["proj"]["bias"]) if "bias" in head["proj"]
                else np.zeros(kernel.shape[0], np.float16))
        return [(f"{cls.key}.probes", np.asarray(head["probes"])),
                (f"{cls.key}.gain", np.asarray(head["gain"])),
                (f"{cls.key}.query", np.asarray(head["query"])),
                (f"{cls.key}.row_bias", np.asarray(head["row_bias"])),
                (f"{cls.key}.proj", kernel),
                (f"{cls.key}.bias", bias)]

    def pooled(self, cells, keep, row_keep, quant=False):
        name = self.key.removesuffix("_head")
        l1, d = cells.shape[2], cells.shape[3]
        k, q = getattr(self.config, f"{name}_probes"), getattr(self.config, f"{name}_queries")
        return probe_pool(cells, keep,
                          head_weight(self.param("probes", default_init(), (l1, k, d)), quant),
                          self.param("gain", jinit.ones, (l1, k)),
                          head_weight(self.param("query", default_init(), (q, d)), quant),
                          self.param("row_bias", jinit.zeros, (q, l1, k)),
                          row_keep, self.dtype or self.config.jax_dtype)

    @nn.compact
    def __call__(self, cells, keep=None, row_keep=None, quant=False):
        logits = HeadProjection(self.out_dim, dtype=self.dtype or self.config.jax_dtype,
                                use_bias=True, name="proj")(
            self.pooled(cells, keep, row_keep, quant), quant)
        return logits.astype(jnp.float32)


class ConfidenceHead(ProbeHead):
    key, code, out_dim = "confidence_head", 2, 1


class RouterHead(ProbeHead):
    key, code, out_dim = "router_head", 3, 3
    calibration_init = (0.90, 0.00, 0.60)

    @classmethod
    def export(cls, head):
        calibration = np.asarray(head.get("calibration", cls.calibration_init), np.float16)
        return super().export(head) + [(f"{cls.key}.calibration", calibration)]


class EmbeddingHead(ProbeHead):
    key, code = "embedding_head", 1
    temp_init: float = 10.0
    bias_init: float = -10.0

    @nn.compact
    def __call__(self, cells, keep=None, row_keep=None, quant=False):
        p = HeadProjection(self.config.embedding_dim, dtype=self.dtype or self.config.jax_dtype,
                           use_bias=False, name="proj")(
            self.pooled(cells, keep, row_keep, quant), quant)
        log_temp = self.param("log_temp", jinit.constant(math.log(self.temp_init)), ())
        bias = self.param("bias", jinit.constant(self.bias_init), ())
        denom = jnp.sqrt(jnp.sum(p.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-12)
        return p / denom.astype(p.dtype), log_temp, bias


HEADS = (EmbeddingHead, ConfidenceHead, RouterHead)
HEAD_KEYS = tuple(head.key for head in HEADS)



class SimpleAttentionNetwork(nn.Module):
    config: TransformerConfig

    def setup(self):
        cfg = self.config
        self.embedding = nn.Embed(cfg.vocab_size, cfg.d_model, embedding_init=jinit.normal(stddev=0.02))
        self.embed_scale = math.sqrt(cfg.d_model)
        self.stack = Stack(cfg)
        self.embedding_head = EmbeddingHead(cfg)
        self.confidence_head = ConfidenceHead(cfg)
        self.router_head = RouterHead(cfg)
        assert all(l < cfg.num_layers for l in cfg.engram_layers)
        orders, heads, sub_dim = engram_geometry(cfg)
        self.engrams = [
            Engram(cfg.d_model, len(orders) * heads, cfg.engram_slots,
                   sub_dim, cfg.num_layers, max(orders), cfg.jax_dtype)
            for _ in cfg.engram_layers
        ]

    def _rope(self, seq_len):
        cfg = self.config
        qk_hd, _ = head_dims(cfg)
        return precompute_rope_freqs(qk_hd, seq_len, cfg.rope_theta)

    def _engram_kv(self, tokens, mask, quant):
        if not self.engrams:
            return None
        orders, heads, _ = engram_geometry(self.config)
        indices = engram_indices(tokens, orders, heads, self.config.engram_slots,
                                 getattr(self.config, "engram_seed_heads", 0))
        ngram_ok = jnp.stack([_mask_diag(mask, o - 1) for o in orders for _ in range(heads)],
                             axis=-1)
        tap_ok = jnp.stack([_mask_diag(mask, j * max(orders)) for j in range(ENGRAM_CONV_TAPS)])
        pairs = [e(indices, ngram_ok, tap_ok, quant=quant) for e in self.engrams]
        return jnp.stack([k for k, _ in pairs]), jnp.stack([v for _, v in pairs])

    def _input_embeddings(self, tokens):
        return self.embedding(tokens) * self.embed_scale

    def __call__(self, tokens, mask=None, quant=False,
                 exit_depth=None, exit_only=False, subnetwork_only=False):
        if mask is None:
            mask = make_causal_mask(tokens.shape[1])
        x = self._input_embeddings(tokens)
        rope = self._rope(tokens.shape[1])
        engram_kv = self._engram_kv(tokens, mask, quant)
        stack_out = self.stack(
            x, mask=mask, rope=rope, engram_kv=engram_kv, quant=quant,
            exit_depth=exit_depth, subnetwork_only=subnetwork_only)
        if subnetwork_only:
            x, _ = stack_out
            exit_x = None
        elif exit_depth is None:
            x, _ = stack_out
            exit_x = None
        else:
            x, _, exit_x = stack_out
        head = self.embedding.embedding
        if self.config.out_vocab:
            head = head[: self.config.out_vocab]
        if subnetwork_only:
            return _aq(x, quant).astype(jnp.float32) @ head.T
        if exit_x is not None:
            exit_logits = _aq(exit_x, quant).astype(jnp.float32) @ head.T
            if exit_only:
                return exit_logits
        logits = _aq(x, quant).astype(jnp.float32) @ head.T
        return (logits, exit_logits) if exit_x is not None else logits

    def hidden_cells(self, tokens, quant=False, window=0, sink=None, exit_depth=None):
        cfg = self.config
        mask = (make_causal_mask(tokens.shape[1])
                & make_padding_mask(tokens, cfg.pad_token_id))
        if window:
            pos = jnp.arange(tokens.shape[1])
            recent = ((pos[:, None] - pos[None, :]) < window)[None, None, :, :]
            keep = recent if sink is None else (recent | sink[:, None, None, :])
            mask = mask & keep
        x0 = self._input_embeddings(tokens)
        rope = self._rope(tokens.shape[1])
        engram_kv = self._engram_kv(tokens, mask, quant)
        hidden = self.stack(
            x0, mask=mask, rope=rope, engram_kv=engram_kv,
            quant=quant, collect_hidden=True, exit_depth=exit_depth)[1]
        return jnp.stack([x0, *hidden], axis=2)

    def _head_cells(self, tokens, quant=False, window=0, sink=None,
                    exit_depth=None):
        cells = jax.lax.stop_gradient(
            self.hidden_cells(tokens, quant=quant, window=window, sink=sink,
                              exit_depth=exit_depth))
        keep = (tokens != self.config.pad_token_id).astype(jnp.float32)
        return cells, keep, ladder_row_keep(self.config, exit_depth)

    def forward_embedding(self, tokens, quant=False, window=0, sink=None,
                           exit_depth=None):
        return self.embedding_head(
            *self._head_cells(tokens, quant, window, sink, exit_depth), quant=quant)[0]

    def forward_confidence(self, tokens, quant=False, window=0, sink=None,
                           exit_depth=None):
        return self.confidence_head(
            *self._head_cells(tokens, quant, window, sink, exit_depth), quant=quant)[..., 0]

    def forward_router(self, tokens, quant=False, window=0, sink=None,
                       exit_depth=None):
        return self.router_head(
            *self._head_cells(tokens, quant, window, sink, exit_depth), quant=quant)

    def hidden_states(self, tokens, mask=None):
        if mask is None:
            mask = make_causal_mask(tokens.shape[1])
        x = self._input_embeddings(tokens)
        rope = self._rope(tokens.shape[1])
        engram_kv = self._engram_kv(tokens, mask, False)
        _, hidden = self.stack(x, mask=mask, rope=rope, engram_kv=engram_kv,
                               collect_hidden=True)
        return hidden.astype(jnp.float32)


def make_causal_mask(seq_len):
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask[None, None, :, :]


def make_padding_mask(tokens, pad_token_id):
    mask = tokens != pad_token_id
    return mask[:, None, None, :]


KV_BUDGET_BYTES = 11 * 1024 * 1024 + 512 * 1024
KV_GROUP = 32
KV_WINDOW_MIN = 160


def kv_budget_window(config):
    qk_hd, v_hd = head_dims(config)
    kd = config.num_kv_heads * qk_hd
    vd = config.num_kv_heads * v_hd
    d, L = config.d_model, config.num_layers
    sites = len(tuple(getattr(config, "engram_layers", (2, 15))))
    per_layer = kd + vd + (kd // KV_GROUP + vd // KV_GROUP) * 4
    per_site = d + (d // KV_GROUP) * 4
    sw = getattr(config, "sliding_window", 0)
    if sw:
        n_global = len(tuple(getattr(config, "global_layers", ())))
        local = min(sw, config.max_seq_len)
        fixed = ((L - n_global) * per_layer + sites * per_site) * local
        if n_global == 0:
            return config.max_seq_len if fixed <= KV_BUDGET_BYTES else \
                max(KV_WINDOW_MIN, local)
        ctx = ((KV_BUDGET_BYTES - fixed) // (n_global * per_layer)
               // KV_GROUP * KV_GROUP)
        return max(KV_WINDOW_MIN, min(ctx, config.max_seq_len))
    per_pos = L * per_layer + sites * per_site
    window = (KV_BUDGET_BYTES // per_pos) // KV_GROUP * KV_GROUP
    return max(KV_WINDOW_MIN, min(window, config.max_seq_len))


def effective_kv_window(config):
    budget = kv_budget_window(config)
    return min(budget, config.kv_window) if config.kv_window else budget


