import math
from dataclasses import dataclass

import jax
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
    def __call__(self, x, ffn_mask=None):
        gate = nn.Dense(self.d_ff, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="gate_proj")(x)
        up = nn.Dense(self.d_ff, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="up_proj")(x)
        if self.activation == "swiglu":
            h = nn.silu(gate) * up
        elif self.activation == "geglu":
            h = nn.gelu(gate) * up
        else:  # drelu
            h = nn.relu(gate) * nn.relu(up)
        if ffn_mask is not None:
            h = h * ffn_mask[:, None, :]  # (batch, 1, d_ff)
        return nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=residual_init(self.num_layers), name="down_proj")(h)


class MLPMixer(nn.Module):
    """MLP-Mixer operating on fixed-size memory slots (B, M, d)."""
    num_slots: int
    d_model: int
    d_ff: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    activation: str = "drelu"

    @nn.compact
    def __call__(self, s, ffn_mask=None):
        residual = s
        s = ZCRMSNorm(dtype=self.dtype, name="token_mix_norm")(s)
        s = s.transpose(0, 2, 1)
        s = FeedForward(self.num_slots, self.d_ff, self.num_layers, self.dtype, self.activation, name="token_mix")(s, ffn_mask=ffn_mask)
        s = s.transpose(0, 2, 1)
        s = s + residual

        residual = s
        s = ZCRMSNorm(dtype=self.dtype, name="channel_mix_norm")(s)
        s = FeedForward(self.d_model, self.d_ff, self.num_layers, self.dtype, self.activation, name="channel_mix")(s, ffn_mask=ffn_mask)
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
    def __call__(self, x, s, mask=None, rope=None, ffn_mask=None):
        s_norm = ZCRMSNorm(dtype=self.dtype, name="pack_s_norm")(s)
        x_norm = ZCRMSNorm(dtype=self.dtype, name="pack_x_norm")(x)
        s = s + MultiHeadAttention(
            self.num_heads, self.num_kv_heads, self.d_model, self.num_layers, self.dtype,
            rope_keys_only=True, name="pack_attn"
        )(s_norm, x_norm, mask=mask, rope=rope)

        s = s + MLPMixer(self.num_slots, self.d_model, self.d_ff, self.num_layers, self.dtype, self.activation, name="mixer")(
            ZCRMSNorm(dtype=self.dtype, name="mix_norm")(s), ffn_mask=ffn_mask
        )

        residual = x
        x = ZCRMSNorm(dtype=self.dtype, name="local_norm")(x)
        x = FeedForward(self.d_model, self.d_ff, self.num_layers, self.dtype, self.activation, name="local_ffn")(x, ffn_mask=ffn_mask)
        x = x + residual

        return x, s


def _get_block_mask(ffn_mask, block_idx):
    """Extract per-block FFN mask: index into first dim if 3D, otherwise pass through."""
    if ffn_mask is not None and ffn_mask.ndim == 3:
        return ffn_mask[block_idx]
    return ffn_mask


class MemoryMixerEncoder(nn.Module):
    """Encoder using MemoryMixer blocks. Output is the final memory slots S."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, mask=None, rope=None, ffn_mask=None):
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

        if mask is not None:
            T_new = x.shape[1]
            if mask.shape[-1] % 2:
                mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 0), (0, 1)))
            mask = mask.reshape(mask.shape[0], 1, 1, -1, 2).any(axis=-1)
            mask = mask[..., :T_new]

        for i in range(cfg.num_encoder_layers):
            block_mask = _get_block_mask(ffn_mask, i)
            x, s = nn.remat(MemoryMixerBlock)(
                cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.d_ff,
                cfg.num_memory_slots, cfg.total_layers, dt, cfg.activation, name=f"block_{i}"
            )(x, s, mask=mask, rope=rope, ffn_mask=block_mask)

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
    def __call__(self, x, encoder_out, self_mask=None, cross_mask=None, rope=None, ffn_mask=None):
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
        x = FeedForward(self.d_model, self.d_ff, self.num_layers, self.dtype, self.activation)(x, ffn_mask=ffn_mask)
        x = x + residual

        return x



class Decoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, encoder_out, self_mask=None, cross_mask=None, rope=None, ffn_mask=None):
        cfg = self.config
        dt = cfg.jax_dtype
        x = x.astype(dt)

        for i in range(cfg.num_decoder_layers):
            block_mask = _get_block_mask(ffn_mask, i)
            x = nn.remat(DecoderBlock)(
                cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.d_ff, cfg.total_layers, dt, cfg.activation, name=f"block_{i}"
            )(x, encoder_out, self_mask=self_mask, cross_mask=cross_mask, rope=rope, ffn_mask=block_mask)

        x = ZCRMSNorm(dtype=dt)(x)
        return x


class SpecAugment(nn.Module):
    """SpecAugment: time and frequency masking for speech regularization."""
    num_time_masks: int = 2
    max_time_width: int = 100
    num_freq_masks: int = 1
    max_freq_width: int = 27

    @nn.compact
    def __call__(self, mel, deterministic=True):
        if deterministic:
            return mel
        # mel: (B, T, F)
        B, T, F = mel.shape
        rng = self.make_rng('specaugment')

        x = mel
        # Time masking
        for i in range(self.num_time_masks):
            rng, k1, k2 = jax.random.split(rng, 3)
            width = jax.random.randint(k1, (), 0, jnp.minimum(self.max_time_width, T))
            start = jax.random.randint(k2, (), 0, jnp.maximum(T - width, 1))
            time_mask = (jnp.arange(T) >= start) & (jnp.arange(T) < start + width)
            x = x * (1.0 - time_mask[None, :, None].astype(x.dtype))

        # Frequency masking
        for i in range(self.num_freq_masks):
            rng, k1, k2 = jax.random.split(rng, 3)
            width = jax.random.randint(k1, (), 0, jnp.minimum(self.max_freq_width, F))
            start = jax.random.randint(k2, (), 0, jnp.maximum(F - width, 1))
            freq_mask = (jnp.arange(F) >= start) & (jnp.arange(F) < start + width)
            x = x * (1.0 - freq_mask[None, None, :].astype(x.dtype))

        return x


class EncoderDecoderTransformer(nn.Module):
    """Encoder-decoder transformer with shared embeddings, tied output, RoPE, and bfloat16."""
    config: TransformerConfig

    def setup(self):
        self.embedding = nn.Embed(self.config.vocab_size, self.config.d_model, embedding_init=jinit.normal(stddev=0.02))
        self.embed_scale = math.sqrt(self.config.d_model)
        self.mel_proj = nn.Dense(self.config.d_model, dtype=self.config.jax_dtype,
                                 use_bias=True, kernel_init=default_init(), name="mel_proj")
        self.spec_augment = SpecAugment()
        self.encoder = MemoryMixerEncoder(self.config)
        self.decoder = Decoder(self.config)

    def _rope(self, seq_len):
        head_dim = self.config.d_model // self.config.num_heads
        return precompute_rope_freqs(head_dim, seq_len, self.config.rope_theta)

    def encode_text(self, src, src_mask=None, ffn_mask=None):
        x = self.embedding(src) * self.embed_scale
        rope = self._rope(src.shape[1])
        return self.encoder(x, mask=src_mask, rope=rope, ffn_mask=ffn_mask)

    def encode(self, src, src_mask=None, ffn_mask=None):
        return self.encode_text(src, src_mask=src_mask, ffn_mask=ffn_mask)

    def encode_speech(self, mel, src_mask=None, ffn_mask=None, deterministic=True):
        mel = self.spec_augment(mel, deterministic=deterministic)
        x = self.mel_proj(mel) * self.embed_scale
        rope = self._rope(x.shape[1])
        return self.encoder(x, mask=src_mask, rope=rope, ffn_mask=ffn_mask)

    def decode(self, tgt, encoder_out, self_mask=None, cross_mask=None, ffn_mask=None):
        """Decode from encoder memory slots. No cross_mask needed (fixed-size slots)."""
        x = self.embedding(tgt) * self.embed_scale
        rope = self._rope(tgt.shape[1])
        x = self.decoder(
            x, encoder_out, self_mask=self_mask, cross_mask=None, rope=rope, ffn_mask=ffn_mask
        )
        logits = x.astype(jnp.float32) @ self.embedding.embedding.T
        return logits

    def __call__(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_out = self.encode_text(src, src_mask=src_mask)
        logits = self.decode(tgt, encoder_out, self_mask=tgt_mask)
        return logits

    def _run_decoder(self, encoder_out, tgt, tgt_mask=None, ffn_mask=None):
        """Run decoder and return float32 hidden states."""
        x = self.embedding(tgt) * self.embed_scale
        rope = self._rope(tgt.shape[1])
        x = self.decoder(x, encoder_out, self_mask=tgt_mask, cross_mask=None, rope=rope, ffn_mask=ffn_mask)
        return x.astype(jnp.float32)

    def _slot_diversity(self, encoder_out):
        s = encoder_out.astype(jnp.float32)
        gram = jnp.matmul(s, s.transpose(0, 2, 1))
        diag_sq = jnp.sum(jnp.diagonal(gram, axis1=1, axis2=2) ** 2)
        return (jnp.sum(gram ** 2) - diag_sq) / s.shape[0]

    def _split_ffn_mask(self, ffn_mask):
        """Split a (n_blocks, batch, d_ff) mask into encoder and decoder portions."""
        if ffn_mask is not None and ffn_mask.ndim == 3:
            n_enc = self.config.num_encoder_layers
            return ffn_mask[:n_enc], ffn_mask[n_enc:]
        return ffn_mask, ffn_mask

    def _forward_masked_impl(self, encoder_out, tgt, tgt_mask=None, dec_mask=None):
        """Shared masked forward: decoder + logits + slot diversity."""
        x_f32 = self._run_decoder(encoder_out, tgt, tgt_mask=tgt_mask, ffn_mask=dec_mask)
        logits = x_f32 @ self.embedding.embedding.T
        slot_div = self._slot_diversity(encoder_out)
        return logits, slot_div

    def forward_masked(self, src, tgt, src_mask=None, tgt_mask=None, ffn_mask=None):
        """Single forward with per-batch-item FFN masking. Returns (logits, slot_div)."""
        enc_mask, dec_mask = self._split_ffn_mask(ffn_mask)
        encoder_out = self.encode_text(src, src_mask=src_mask, ffn_mask=enc_mask)
        return self._forward_masked_impl(encoder_out, tgt, tgt_mask=tgt_mask, dec_mask=dec_mask)

    def forward_speech_masked(self, mel, tgt, src_mask=None, tgt_mask=None, ffn_mask=None, deterministic=True):
        """Single speech forward with per-batch-item FFN masking. Returns (logits, slot_div)."""
        enc_mask, dec_mask = self._split_ffn_mask(ffn_mask)
        encoder_out = self.encode_speech(mel, src_mask=src_mask, ffn_mask=enc_mask, deterministic=deterministic)
        return self._forward_masked_impl(encoder_out, tgt, tgt_mask=tgt_mask, dec_mask=dec_mask)

    def _make_eval_ffn_mask(self, ff_width, B, dtype):
        """Default prefix FFN mask for eval: first ff_width neurons active."""
        mask = (jnp.arange(self.config.d_ff) < ff_width).astype(dtype)
        return jnp.broadcast_to(mask[None, :], (B, self.config.d_ff))

    def _eval_sub_models(self, encode_fn, src, tgt, src_mask, tgt_mask, B, dtype, mat_ff_widths, mat_ffn_masks):
        """Run per-width forwards for matryoshka eval. Returns list of logit tensors."""
        emb = self.embedding.embedding
        n_enc = self.config.num_encoder_layers
        d_ff = self.config.d_ff
        mat_logits = []
        if mat_ffn_masks is not None:
            for m in mat_ffn_masks:
                if m.ndim == 2:
                    mask = jnp.broadcast_to(m[:, None, :], (m.shape[0], B, d_ff))
                    enc_m, dec_m = mask[:n_enc], mask[n_enc:]
                else:
                    enc_m = dec_m = jnp.broadcast_to(m[None, :], (B, d_ff))
                x_m = self._run_decoder(encode_fn(src, src_mask=src_mask, ffn_mask=enc_m), tgt, tgt_mask=tgt_mask, ffn_mask=dec_m)
                mat_logits.append(x_m @ emb.T)
        elif mat_ff_widths is not None:
            for ff_w in mat_ff_widths:
                mask = self._make_eval_ffn_mask(ff_w, B, dtype)
                x_m = self._run_decoder(encode_fn(src, src_mask=src_mask, ffn_mask=mask), tgt, tgt_mask=tgt_mask, ffn_mask=mask)
                mat_logits.append(x_m @ emb.T)
        return mat_logits

    def _forward_with_aux_impl(self, encode_fn, src, tgt, src_mask=None, tgt_mask=None, mat_ff_widths=None, mat_ffn_masks=None):
        """Eval-only: full forward + per-width sub-model forwards for reporting per-width PPL."""
        encoder_out = encode_fn(src, src_mask=src_mask)
        x_f32 = self._run_decoder(encoder_out, tgt, tgt_mask=tgt_mask)
        logits = x_f32 @ self.embedding.embedding.T
        slot_div = self._slot_diversity(encoder_out)
        mat_logits = self._eval_sub_models(encode_fn, src, tgt, src_mask, tgt_mask, src.shape[0], x_f32.dtype, mat_ff_widths, mat_ffn_masks)
        return logits, slot_div, mat_logits

    def forward_with_aux(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None, mat_ff_widths=None, mat_ffn_masks=None):
        return self._forward_with_aux_impl(self.encode_text, src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                           mat_ff_widths=mat_ff_widths, mat_ffn_masks=mat_ffn_masks)

    def forward_speech_with_aux(self, mel, tgt, src_mask=None, tgt_mask=None, mat_ff_widths=None, mat_ffn_masks=None, deterministic=True):
        from functools import partial
        encode_fn = partial(self.encode_speech, deterministic=deterministic)
        return self._forward_with_aux_impl(encode_fn, mel, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                           mat_ff_widths=mat_ff_widths, mat_ffn_masks=mat_ffn_masks)

    def init_all(self, src, tgt, mel):
        """Dummy forward through both text and speech pathways to initialize all params."""
        src_mask = make_padding_mask(src, self.config.pad_token_id)
        tgt_mask = make_causal_mask(tgt.shape[1]) & make_padding_mask(tgt, self.config.pad_token_id)
        text_out = self.encode_text(src, src_mask=src_mask)
        mel_mask = make_mel_padding_mask(mel)
        speech_out = self.encode_speech(mel, src_mask=mel_mask, deterministic=True)
        _ = self._run_decoder(text_out, tgt, tgt_mask=tgt_mask)
        _ = self._run_decoder(speech_out, tgt, tgt_mask=tgt_mask)
        return jnp.zeros(())


def count_params(params):
    """Count total number of parameters in a pytree."""
    return sum(x.size for x in jax.tree.leaves(params))


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
