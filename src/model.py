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
    n_mels: int = 80

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
    def __call__(self, x, s, mask=None, rope=None, dim_mask=None):
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

        if dim_mask is not None:
            dm = dim_mask[None, None, :]
            x = x * dm
            s = s * dm

        return x, s


class MemoryMixerEncoder(nn.Module):
    """Encoder using MemoryMixer blocks. Output is the final memory slots S."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, mask=None, rope=None, dim_mask=None):
        cfg = self.config
        dt = cfg.jax_dtype
        x = x.astype(dt)

        s_base = self.param("memory_slots", jinit.normal(stddev=0.02), (1, cfg.num_memory_slots, cfg.d_model))
        x_pool = jnp.mean(x, axis=1)  # (B, d)
        s_bias = nn.Dense(cfg.d_model, dtype=dt, use_bias=False, kernel_init=jinit.zeros, name="slot_init")(x_pool)
        s = jnp.broadcast_to(s_base.astype(dt), (x.shape[0], cfg.num_memory_slots, cfg.d_model)) + s_bias[:, None, :]

        x = nn.Conv(features=cfg.d_model, kernel_size=(4,), strides=(2,),
                    padding='SAME', feature_group_count=cfg.d_model,
                    dtype=dt, use_bias=False, name="downsample_dw")(x)
        x = nn.Dense(cfg.d_model, dtype=dt, use_bias=False,
                     kernel_init=default_init(), name="downsample_pw")(x)

        if dim_mask is not None:
            dm = dim_mask[None, None, :].astype(dt)
            x = x * dm
            s = s * dm

        if mask is not None:
            T_new = x.shape[1]
            if mask.shape[-1] % 2:
                mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 0), (0, 1)))
            mask = mask.reshape(mask.shape[0], 1, 1, -1, 2).any(axis=-1)
            mask = mask[..., :T_new]

        for i in range(cfg.num_encoder_layers):
            x, s = nn.remat(MemoryMixerBlock)(
                cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.d_ff,
                cfg.num_memory_slots, cfg.total_layers, dt, cfg.activation, name=f"block_{i}"
            )(x, s, mask=mask, rope=rope, dim_mask=dim_mask)

        s = ZCRMSNorm(dtype=dt, name="final_norm")(s)
        if dim_mask is not None:
            s = s * dim_mask[None, None, :].astype(dt)
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
    def __call__(self, x, encoder_out, self_mask=None, cross_mask=None, rope=None, dim_mask=None):
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

        if dim_mask is not None:
            x = x * dim_mask[None, None, :]

        return x



class Decoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, encoder_out, self_mask=None, cross_mask=None, rope=None, dim_mask=None):
        cfg = self.config
        dt = cfg.jax_dtype
        x = x.astype(dt)

        for i in range(cfg.num_decoder_layers):
            x = nn.remat(DecoderBlock)(
                cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.d_ff, cfg.total_layers, dt, cfg.activation, name=f"block_{i}"
            )(x, encoder_out, self_mask=self_mask, cross_mask=cross_mask, rope=rope, dim_mask=dim_mask)

        x = ZCRMSNorm(dtype=dt)(x)
        if dim_mask is not None:
            x = x * dim_mask[None, None, :]
        return x


class MelProjection(nn.Module):
    """Project mel spectrogram features to model dimension."""
    d_model: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, mel):
        return nn.Dense(self.d_model, dtype=self.dtype, use_bias=True,
                        kernel_init=default_init(), name="proj")(mel)


class EncoderDecoderTransformer(nn.Module):
    """Encoder-decoder transformer with shared embeddings, tied output, RoPE, and bfloat16."""
    config: TransformerConfig

    def setup(self):
        self.embedding = nn.Embed(self.config.vocab_size, self.config.d_model, embedding_init=jinit.normal(stddev=0.02))
        self.embed_scale = math.sqrt(self.config.d_model)
        self.mel_proj = MelProjection(self.config.d_model, self.config.jax_dtype)
        self.encoder = MemoryMixerEncoder(self.config)
        self.decoder = Decoder(self.config)

    def _rope(self, seq_len):
        head_dim = self.config.d_model // self.config.num_heads
        return precompute_rope_freqs(head_dim, seq_len, self.config.rope_theta)

    def encode_text(self, src, src_mask=None, dim_mask=None):
        x = self.embedding(src) * self.embed_scale
        if dim_mask is not None:
            x = x * dim_mask[None, None, :]
        rope = self._rope(src.shape[1])
        return self.encoder(x, mask=src_mask, rope=rope, dim_mask=dim_mask)

    def encode(self, src, src_mask=None):
        """Backward-compatible alias for encode_text."""
        return self.encode_text(src, src_mask=src_mask)

    def encode_speech(self, mel, src_mask=None, dim_mask=None):
        x = self.mel_proj(mel) * self.embed_scale
        if dim_mask is not None:
            x = x * dim_mask[None, None, :]
        rope = self._rope(mel.shape[1])
        return self.encoder(x, mask=src_mask, rope=rope, dim_mask=dim_mask)

    def decode(self, tgt, encoder_out, self_mask=None, cross_mask=None):
        x = self.embedding(tgt) * self.embed_scale
        rope = self._rope(tgt.shape[1])
        x = self.decoder(
            x, encoder_out, self_mask=self_mask, cross_mask=None, rope=rope
        )
        logits = x.astype(jnp.float32) @ self.embedding.embedding.T
        return logits

    def __call__(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        encoder_out = self.encode_text(src, src_mask=src_mask)
        logits = self.decode(
            tgt, encoder_out, self_mask=tgt_mask
        )
        return logits

    def _run_decoder(self, encoder_out, tgt, tgt_mask=None, dim_mask=None):
        """Run decoder and return float32 hidden states."""
        x = self.embedding(tgt) * self.embed_scale
        if dim_mask is not None:
            x = x * dim_mask[None, None, :]
        rope = self._rope(tgt.shape[1])
        x = self.decoder(x, encoder_out, self_mask=tgt_mask, cross_mask=None, rope=rope, dim_mask=dim_mask)
        return x.astype(jnp.float32)

    def _slot_diversity(self, encoder_out):
        s = encoder_out.astype(jnp.float32)
        gram = jnp.matmul(s, s.transpose(0, 2, 1))
        diag_sq = jnp.sum(jnp.diagonal(gram, axis1=1, axis2=2) ** 2)
        return (jnp.sum(gram ** 2) - diag_sq) / s.shape[0]

    def forward_with_aux(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None, mrl_dims=None, mrl_masks=None):
        emb = self.embedding.embedding

        # Full forward (no masking)
        encoder_out = self.encode_text(src, src_mask=src_mask)
        x_f32 = self._run_decoder(encoder_out, tgt, tgt_mask=tgt_mask)
        logits = x_f32 @ emb.T
        slot_div = self._slot_diversity(encoder_out)

        # MRL sub-model forwards
        mrl_logits = []
        if mrl_dims is not None:
            for i, d in enumerate(mrl_dims):
                if d < self.config.d_model:
                    if mrl_masks is not None:
                        # Interior masking: full re-encode + re-decode with mask
                        dm = mrl_masks[i]
                        enc_m = self.encode_text(src, src_mask=src_mask, dim_mask=dm)
                        x_m = self._run_decoder(enc_m, tgt, tgt_mask=tgt_mask, dim_mask=dm)
                        mrl_logits.append(x_m @ emb.T)
                    else:
                        # Slice mode: output-only prefix slicing
                        mrl_logits.append(x_f32[..., :d] @ emb[:, :d].T)

        return logits, slot_div, mrl_logits

    def forward_speech_with_aux(self, mel, tgt, src_mask=None, tgt_mask=None, mrl_dims=None, mrl_masks=None):
        emb = self.embedding.embedding

        # Full forward (no masking)
        encoder_out = self.encode_speech(mel, src_mask=src_mask)
        x_f32 = self._run_decoder(encoder_out, tgt, tgt_mask=tgt_mask)
        logits = x_f32 @ emb.T
        slot_div = self._slot_diversity(encoder_out)

        # MRL sub-model forwards
        mrl_logits = []
        if mrl_dims is not None:
            for i, d in enumerate(mrl_dims):
                if d < self.config.d_model:
                    if mrl_masks is not None:
                        dm = mrl_masks[i]
                        enc_m = self.encode_speech(mel, src_mask=src_mask, dim_mask=dm)
                        x_m = self._run_decoder(enc_m, tgt, tgt_mask=tgt_mask, dim_mask=dm)
                        mrl_logits.append(x_m @ emb.T)
                    else:
                        mrl_logits.append(x_f32[..., :d] @ emb[:, :d].T)

        return logits, slot_div, mrl_logits

    def init_all(self, src, tgt, mel):
        """Dummy forward through both text and speech pathways to initialize all params."""
        src_mask = make_padding_mask(src, self.config.pad_token_id)
        tgt_mask = make_causal_mask(tgt.shape[1]) & make_padding_mask(tgt, self.config.pad_token_id)
        text_out = self.encode_text(src, src_mask=src_mask)
        mel_mask = make_mel_padding_mask(mel)
        speech_out = self.encode_speech(mel, src_mask=mel_mask)
        _ = self._run_decoder(text_out, tgt, tgt_mask=tgt_mask)
        _ = self._run_decoder(speech_out, tgt, tgt_mask=tgt_mask)
        return jnp.zeros(())


def make_causal_mask(seq_len):
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask[None, None, :, :]


def make_padding_mask(tokens, pad_token_id):
    mask = tokens != pad_token_id
    return mask[:, None, None, :]


def make_mel_padding_mask(mel):
    """Create padding mask from mel spectrogram: non-zero frames are valid."""
    mask = jnp.any(mel != 0, axis=-1)  # (B, T)
    return mask[:, None, None, :]  # (B, 1, 1, T)
