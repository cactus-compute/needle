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


@dataclass
class TransformerConfig:
    vocab_size: int = 50257
    d_model: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_ff: int = 512
    max_seq_len: int = 128
    dropout_rate: float = 0.1
    pad_token_id: int = 50256
    rope_theta: float = 10000.0
    dtype: str = "bfloat16"

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
    d_model: int
    num_layers: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, q_input, kv_input, mask=None, rope=None, deterministic=True):
        head_dim = self.d_model // self.num_heads
        B = q_input.shape[0]

        q = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="q_proj")(q_input)
        k = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="k_proj")(kv_input)
        v = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="v_proj")(kv_input)

        q = q.reshape(B, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        scale = jnp.sqrt(jnp.float32(head_dim))
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale

        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)

        attn_weights = nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(rate=self.dropout_rate)(attn_weights, deterministic=deterministic)

        out = jnp.matmul(attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, -1, self.d_model)
        return nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=residual_init(self.num_layers), name="out_proj")(out)


class FeedForward(nn.Module):
    d_model: int
    d_ff: int
    num_layers: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, deterministic=True):
        gate = nn.Dense(self.d_ff, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="gate_proj")(x)
        up = nn.Dense(self.d_ff, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="up_proj")(x)
        x = nn.silu(gate) * up
        x = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=residual_init(self.num_layers), name="down_proj")(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x


class EncoderBlock(nn.Module):
    num_heads: int
    d_model: int
    d_ff: int
    num_layers: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, mask=None, rope=None, deterministic=True):
        residual = x
        x = nn.RMSNorm(dtype=self.dtype)(x)
        x = MultiHeadAttention(self.num_heads, self.d_model, self.num_layers, self.dropout_rate, self.dtype)(
            x, x, mask=mask, rope=rope, deterministic=deterministic
        )
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + residual

        residual = x
        x = nn.RMSNorm(dtype=self.dtype)(x)
        x = FeedForward(self.d_model, self.d_ff, self.num_layers, self.dropout_rate, self.dtype)(x, deterministic=deterministic)
        x = x + residual

        return x


class DecoderBlock(nn.Module):
    num_heads: int
    d_model: int
    d_ff: int
    num_layers: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, encoder_out, self_mask=None, cross_mask=None, rope=None, deterministic=True):
        # Self-attention with RoPE
        residual = x
        x = nn.RMSNorm(dtype=self.dtype)(x)
        x = MultiHeadAttention(self.num_heads, self.d_model, self.num_layers, self.dropout_rate, self.dtype, name="self_attn")(
            x, x, mask=self_mask, rope=rope, deterministic=deterministic
        )
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + residual

        # Cross-attention (no RoPE)
        residual = x
        x = nn.RMSNorm(dtype=self.dtype)(x)
        x = MultiHeadAttention(self.num_heads, self.d_model, self.num_layers, self.dropout_rate, self.dtype, name="cross_attn")(
            x, encoder_out, mask=cross_mask, deterministic=deterministic
        )
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + residual

        residual = x
        x = nn.RMSNorm(dtype=self.dtype)(x)
        x = FeedForward(self.d_model, self.d_ff, self.num_layers, self.dropout_rate, self.dtype)(x, deterministic=deterministic)
        x = x + residual

        return x


class Encoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, mask=None, rope=None, deterministic=True):
        cfg = self.config
        dt = cfg.jax_dtype
        x = x.astype(dt)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)

        for i in range(cfg.num_encoder_layers):
            x = EncoderBlock(
                cfg.num_heads, cfg.d_model, cfg.d_ff, cfg.total_layers, cfg.dropout_rate, dt, name=f"block_{i}"
            )(x, mask=mask, rope=rope, deterministic=deterministic)

        x = nn.RMSNorm(dtype=dt)(x)
        return x


class Decoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, encoder_out, self_mask=None, cross_mask=None, rope=None, deterministic=True):
        cfg = self.config
        dt = cfg.jax_dtype
        x = x.astype(dt)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)

        for i in range(cfg.num_decoder_layers):
            x = DecoderBlock(
                cfg.num_heads, cfg.d_model, cfg.d_ff, cfg.total_layers, cfg.dropout_rate, dt, name=f"block_{i}"
            )(x, encoder_out, self_mask=self_mask, cross_mask=cross_mask, rope=rope, deterministic=deterministic)

        x = nn.RMSNorm(dtype=dt)(x)
        return x


class EncoderDecoderTransformer(nn.Module):
    """Encoder-decoder transformer with shared embeddings, tied output, RoPE, and bfloat16."""
    config: TransformerConfig

    def setup(self):
        self.embedding = nn.Embed(self.config.vocab_size, self.config.d_model, embedding_init=jinit.normal(stddev=0.02))
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def _rope(self, seq_len):
        head_dim = self.config.d_model // self.config.num_heads
        return precompute_rope_freqs(head_dim, seq_len, self.config.rope_theta)

    def encode(self, src, src_mask=None, deterministic=True):
        x = self.embedding(src)
        rope = self._rope(src.shape[1])
        return self.encoder(x, mask=src_mask, rope=rope, deterministic=deterministic)

    def decode(self, tgt, encoder_out, self_mask=None, cross_mask=None, deterministic=True):
        x = self.embedding(tgt)
        rope = self._rope(tgt.shape[1])
        x = self.decoder(
            x, encoder_out, self_mask=self_mask, cross_mask=cross_mask, rope=rope, deterministic=deterministic
        )
        logits = x.astype(jnp.float32) @ self.embedding.embedding.T
        return logits

    def __call__(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None, deterministic=True):
        encoder_out = self.encode(src, src_mask=src_mask, deterministic=deterministic)
        logits = self.decode(
            tgt, encoder_out, self_mask=tgt_mask, cross_mask=cross_mask, deterministic=deterministic
        )
        return logits


def make_causal_mask(seq_len):
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask[None, None, :, :]


def make_padding_mask(tokens, pad_token_id):
    mask = tokens != pad_token_id
    return mask[:, None, None, :]
