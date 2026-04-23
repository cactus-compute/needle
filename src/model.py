import math
from dataclasses import dataclass

import jax.numpy as jnp
import jax.nn.initializers as jinit
import flax.linen as nn


def default_init():
    return jinit.normal(stddev=0.02)


def residual_init(num_layers):
    return jinit.normal(stddev=0.02 / math.sqrt(2 * num_layers))


DTYPE_MAP = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


class ZCRMSNorm(nn.Module):
    """Zero-centred RMSNorm: scale initialized to 0, applied as (1 + γ) * x / RMS(x)."""
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", jinit.zeros, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + self.epsilon)
        return ((1 + scale) * x / rms).astype(self.dtype)


@dataclass
class TransformerConfig:
    vocab_size: int = 8192
    d_model: int = 128
    num_heads: int = 4
    num_kv_heads: int = 2
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_ff: int = 512
    max_seq_len: int = 128
    pad_token_id: int = 0
    rope_theta: float = 10000.0
    dtype: str = "bfloat16"
    activation: str = "drelu"
    num_memory_slots: int = 64
    encoder_type: str = "memory_mixer"  # "memory_mixer" | "vanilla"
    slot_mrl_dims: tuple = ()  # e.g. (64, 32, 16, 8); empty = disabled

    @property
    def jax_dtype(self):
        return DTYPE_MAP[self.dtype]

    @property
    def total_layers(self):
        return self.num_encoder_layers + self.num_decoder_layers


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
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class MultiHeadAttention(nn.Module):
    num_heads: int
    num_kv_heads: int
    d_model: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    rope_keys_only: bool = False

    @nn.compact
    def __call__(self, q_input, kv_input, mask=None, rope=None):
        head_dim = self.d_model // self.num_heads
        kv_dim = self.num_kv_heads * head_dim
        B = q_input.shape[0]

        q = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="q_proj")(q_input)
        k = nn.Dense(kv_dim, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="k_proj")(kv_input)
        v = nn.Dense(kv_dim, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="v_proj")(kv_input)

        q = q.reshape(B, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q = ZCRMSNorm(dtype=self.dtype, name="q_norm")(q)
        k = ZCRMSNorm(dtype=self.dtype, name="k_norm")(k)

        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            k = jnp.repeat(k, repeats, axis=1)
            v = jnp.repeat(v, repeats, axis=1)

        if rope is not None:
            cos, sin = rope
            if not self.rope_keys_only:
                q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        scale = jnp.sqrt(jnp.float32(head_dim))
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale

        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)

        attn_weights = nn.softmax(attn_weights, axis=-1)

        out = jnp.matmul(attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, -1, self.d_model)
        return nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=residual_init(self.num_layers), name="out_proj")(out)


class FeedForward(nn.Module):
    d_model: int
    d_ff: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    activation: str = "drelu"

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.d_ff, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="gate_proj")(x)
        up = nn.Dense(self.d_ff, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="up_proj")(x)
        if self.activation == "swiglu":
            x = nn.silu(gate) * up
        elif self.activation == "geglu":
            x = nn.gelu(gate) * up
        else:  # drelu
            x = nn.relu(gate) * nn.relu(up)
        return nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=residual_init(self.num_layers), name="down_proj")(x)


class MLPMixer(nn.Module):
    """MLP-Mixer operating on fixed-size memory slots (B, M, d)."""
    num_slots: int
    d_model: int
    d_ff: int
    dtype: jnp.dtype = jnp.bfloat16
    activation: str = "drelu"

    @nn.compact
    def __call__(self, s):
        residual = s
        s = ZCRMSNorm(dtype=self.dtype, name="token_mix_norm")(s)
        s = s.transpose(0, 2, 1)
        gate = nn.Dense(self.d_ff, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="token_mix_gate")(s)
        up = nn.Dense(self.d_ff, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="token_mix_up")(s)
        if self.activation == "swiglu":
            s = nn.silu(gate) * up
        elif self.activation == "geglu":
            s = nn.gelu(gate) * up
        else:
            s = nn.relu(gate) * nn.relu(up)
        s = nn.Dense(self.num_slots, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="token_mix_down")(s)
        s = s.transpose(0, 2, 1)
        s = s + residual

        residual = s
        s = ZCRMSNorm(dtype=self.dtype, name="channel_mix_norm")(s)
        gate = nn.Dense(self.d_ff, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="channel_mix_gate")(s)
        up = nn.Dense(self.d_ff, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="channel_mix_up")(s)
        if self.activation == "swiglu":
            s = nn.silu(gate) * up
        elif self.activation == "geglu":
            s = nn.gelu(gate) * up
        else:
            s = nn.relu(gate) * nn.relu(up)
        s = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="channel_mix_down")(s)
        s = s + residual

        return s


class EncoderBlock(nn.Module):
    num_heads: int
    num_kv_heads: int
    d_model: int
    d_ff: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    activation: str = "drelu"

    @nn.compact
    def __call__(self, x, mask=None, rope=None):
        residual = x
        x = ZCRMSNorm(dtype=self.dtype)(x)
        x = MultiHeadAttention(self.num_heads, self.num_kv_heads, self.d_model, self.num_layers, self.dtype)(
            x, x, mask=mask, rope=rope
        )
        x = x + residual

        residual = x
        x = ZCRMSNorm(dtype=self.dtype)(x)
        x = FeedForward(self.d_model, self.d_ff, self.num_layers, self.dtype, self.activation)(x)
        x = x + residual

        return x


class VanillaEncoder(nn.Module):
    """Plain self-attention transformer encoder (pre-MemoryMixer baseline)."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, mask=None, rope=None):
        cfg = self.config
        dt = cfg.jax_dtype
        x = x.astype(dt)

        # Self-attention uses square (B,1,T,T) mask. Incoming mask is padding (B,1,1,T).
        if mask is not None:
            mask = mask & mask.transpose(0, 1, 3, 2)

        for i in range(cfg.num_encoder_layers):
            x = EncoderBlock(
                cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.d_ff, cfg.total_layers, dt, cfg.activation, name=f"block_{i}"
            )(x, mask=mask, rope=rope)

        x = ZCRMSNorm(dtype=dt, name="final_norm")(x)
        return x


class MemoryMixerBlock(nn.Module):
    """Block 4a: Pack (cross-attn S←X with RoPE on keys) → Mix (MLP-Mixer) → Local Update (MLP)."""
    num_heads: int
    num_kv_heads: int
    d_model: int
    d_ff: int
    num_slots: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    activation: str = "drelu"

    @nn.compact
    def __call__(self, x, s, mask=None, rope=None):
        s_norm = ZCRMSNorm(dtype=self.dtype, name="pack_s_norm")(s)
        x_norm = ZCRMSNorm(dtype=self.dtype, name="pack_x_norm")(x)
        s = s + MultiHeadAttention(
            self.num_heads, self.num_kv_heads, self.d_model, self.num_layers, self.dtype,
            rope_keys_only=True, name="pack_attn"
        )(s_norm, x_norm, mask=mask, rope=rope)

        s = s + MLPMixer(self.num_slots, self.d_model, self.d_ff, self.dtype, self.activation, name="mixer")(
            ZCRMSNorm(dtype=self.dtype, name="mix_norm")(s)
        )

        residual = x
        x = ZCRMSNorm(dtype=self.dtype, name="local_norm")(x)
        x = FeedForward(self.d_model, self.d_ff, self.num_layers, self.dtype, self.activation, name="local_ffn")(x)
        x = x + residual

        return x, s


class MemoryMixerEncoder(nn.Module):
    """Encoder using MemoryMixer blocks. Output is the final memory slots S."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, mask=None, rope=None):
        cfg = self.config
        dt = cfg.jax_dtype
        x = x.astype(dt)

        s_base = self.param("memory_slots", jinit.normal(stddev=0.02), (1, cfg.num_memory_slots, cfg.d_model))
        x_pool = jnp.mean(x, axis=1)  # (B, d)
        s_bias = nn.Dense(cfg.d_model, dtype=dt, use_bias=False, kernel_init=jinit.zeros, name="slot_init")(x_pool)
        s = jnp.broadcast_to(s_base.astype(dt), (x.shape[0], cfg.num_memory_slots, cfg.d_model)) + s_bias[:, None, :]

        for i in range(cfg.num_encoder_layers):
            x, s = MemoryMixerBlock(
                cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.d_ff,
                cfg.num_memory_slots, cfg.total_layers, dt, cfg.activation, name=f"block_{i}"
            )(x, s, mask=mask, rope=rope)

        s = ZCRMSNorm(dtype=dt, name="final_norm")(s)
        return s


class DecoderBlock(nn.Module):
    num_heads: int
    num_kv_heads: int
    d_model: int
    d_ff: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    activation: str = "drelu"

    @nn.compact
    def __call__(self, x, encoder_out, self_mask=None, cross_mask=None, rope=None):
        residual = x
        x = ZCRMSNorm(dtype=self.dtype)(x)
        x = MultiHeadAttention(self.num_heads, self.num_kv_heads, self.d_model, self.num_layers, self.dtype, name="self_attn")(
            x, x, mask=self_mask, rope=rope
        )
        x = x + residual

        residual = x
        x = ZCRMSNorm(dtype=self.dtype)(x)
        x = MultiHeadAttention(self.num_heads, self.num_kv_heads, self.d_model, self.num_layers, self.dtype, name="cross_attn")(
            x, encoder_out, mask=cross_mask
        )
        x = x + residual

        residual = x
        x = ZCRMSNorm(dtype=self.dtype)(x)
        x = FeedForward(self.d_model, self.d_ff, self.num_layers, self.dtype, self.activation)(x)
        x = x + residual

        return x



class Decoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, encoder_out, self_mask=None, cross_mask=None, rope=None):
        cfg = self.config
        dt = cfg.jax_dtype
        x = x.astype(dt)

        for i in range(cfg.num_decoder_layers):
            x = DecoderBlock(
                cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.d_ff, cfg.total_layers, dt, cfg.activation, name=f"block_{i}"
            )(x, encoder_out, self_mask=self_mask, cross_mask=cross_mask, rope=rope)

        x = ZCRMSNorm(dtype=dt)(x)
        return x


class EncoderDecoderTransformer(nn.Module):
    """Encoder-decoder transformer with shared embeddings, tied output, RoPE, and bfloat16."""
    config: TransformerConfig

    def setup(self):
        self.embedding = nn.Embed(self.config.vocab_size, self.config.d_model, embedding_init=jinit.normal(stddev=0.02))
        self.embed_scale = math.sqrt(self.config.d_model)
        if self.config.encoder_type == "vanilla":
            self.encoder = VanillaEncoder(self.config)
        else:
            self.encoder = MemoryMixerEncoder(self.config)
        self.decoder = Decoder(self.config)

    def _rope(self, seq_len):
        head_dim = self.config.d_model // self.config.num_heads
        return precompute_rope_freqs(head_dim, seq_len, self.config.rope_theta)

    def encode(self, src, src_mask=None):
        x = self.embedding(src) * self.embed_scale
        rope = self._rope(src.shape[1])
        return self.encoder(x, mask=src_mask, rope=rope)

    def decode(self, tgt, encoder_out, self_mask=None, cross_mask=None):
        x = self.embedding(tgt) * self.embed_scale
        rope = self._rope(tgt.shape[1])
        x = self.decoder(
            x, encoder_out, self_mask=self_mask, cross_mask=cross_mask, rope=rope
        )
        logits = x.astype(jnp.float32) @ self.embedding.embedding.T
        return logits

    def _apply_slot_mask(self, encoder_out, slot_keep):
        """Zero out inactive slots (MemoryMixer only). slot_keep: (M,) float {0,1}."""
        if slot_keep is None or self.config.encoder_type == "vanilla":
            return encoder_out
        m = slot_keep.astype(encoder_out.dtype)[None, :, None]
        return encoder_out * m

    def _slot_cross_mask(self, tgt_len, slot_keep):
        """Build (1,1,T_tgt,M) cross mask from a (M,) slot-keep vector."""
        if slot_keep is None:
            return None
        return jnp.broadcast_to(slot_keep.astype(jnp.bool_)[None, None, None, :],
                                (1, 1, tgt_len, slot_keep.shape[0]))

    def __call__(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None, slot_keep=None):
        encoder_out = self.encode(src, src_mask=src_mask)
        if self.config.encoder_type == "vanilla":
            dec_cross_mask = cross_mask
        else:
            encoder_out = self._apply_slot_mask(encoder_out, slot_keep)
            dec_cross_mask = self._slot_cross_mask(tgt.shape[1], slot_keep)
        logits = self.decode(
            tgt, encoder_out, self_mask=tgt_mask, cross_mask=dec_cross_mask
        )
        return logits

    def forward_with_aux(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None, mrl_dims=None, slot_keep=None):
        encoder_out = self.encode(src, src_mask=src_mask)
        if self.config.encoder_type == "vanilla":
            dec_cross_mask = cross_mask
        else:
            encoder_out = self._apply_slot_mask(encoder_out, slot_keep)
            dec_cross_mask = self._slot_cross_mask(tgt.shape[1], slot_keep)
        x = self.embedding(tgt) * self.embed_scale
        rope = self._rope(tgt.shape[1])
        x = self.decoder(x, encoder_out, self_mask=tgt_mask, cross_mask=dec_cross_mask, rope=rope)
        x_f32 = x.astype(jnp.float32)
        emb = self.embedding.embedding
        logits = x_f32 @ emb.T

        mrl_logits = []
        if mrl_dims is not None:
            for d in mrl_dims:
                if d < self.config.d_model:
                    mrl_logits.append(x_f32[..., :d] @ emb[:, :d].T)

        if self.config.encoder_type == "vanilla":
            slot_div = jnp.float32(0.0)
        else:
            s = encoder_out.astype(jnp.float32)  # already masked
            gram = jnp.matmul(s, s.transpose(0, 2, 1))
            diag_sq = jnp.sum(jnp.diagonal(gram, axis1=1, axis2=2) ** 2)
            if slot_keep is None:
                denom = s.shape[0]
            else:
                active = jnp.sum(slot_keep)
                denom = s.shape[0] * jnp.maximum(active * active, 1.0)
                denom = denom / jnp.maximum((s.shape[1] * s.shape[1]), 1.0)
                denom = jnp.maximum(denom, 1.0)
            slot_div = (jnp.sum(gram ** 2) - diag_sq) / denom
        return logits, slot_div, mrl_logits


def make_causal_mask(seq_len):
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask[None, None, :, :]


def make_padding_mask(tokens, pad_token_id):
    mask = tokens != pad_token_id
    return mask[:, None, None, :]
