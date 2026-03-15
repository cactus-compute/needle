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
    dropout_rate: float = 0.1
    contrastive_dim: int = 128
    conv_kernel_size: int = 0

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
            h = h * ffn_mask[:, None, :] 
        return nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=residual_init(self.num_layers), name="down_proj")(h)


class SpeechSubsampling(nn.Module):
    """FastConformer-style 8x depthwise-separable convolutional subsampling.

    Input: (B, T, n_mels) raw mel spectrogram.
    Output: (B, T//8, d_model) subsampled features.

    Architecture (3 stages of stride-2 = 8x total):
      Conv2D(1 -> C, k=3, s=2) + ReLU
      DW-Conv2D(C, k=3, s=2) + PW-Conv2D(C -> C, k=1) + ReLU
      DW-Conv2D(C, k=3, s=2) + PW-Conv2D(C -> C, k=1) + ReLU
      Reshape(C * F') + Linear -> d_model
    """
    d_model: int
    conv_channels: int = 256
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, mel):
        x = mel[..., None] 

        x = nn.Conv(features=self.conv_channels, kernel_size=(3, 3), strides=(2, 2),
                     padding='SAME', dtype=self.dtype, use_bias=False, name="conv1")(x)
        x = nn.relu(x)

        x = nn.Conv(features=self.conv_channels, kernel_size=(3, 3), strides=(2, 2),
                     padding='SAME', feature_group_count=self.conv_channels,
                     dtype=self.dtype, use_bias=False, name="dw_conv2")(x)
        x = nn.Conv(features=self.conv_channels, kernel_size=(1, 1),
                     dtype=self.dtype, use_bias=False, name="pw_conv2")(x)
        x = nn.relu(x)

        x = nn.Conv(features=self.conv_channels, kernel_size=(3, 3), strides=(2, 2),
                     padding='SAME', feature_group_count=self.conv_channels,
                     dtype=self.dtype, use_bias=False, name="dw_conv3")(x)
        x = nn.Conv(features=self.conv_channels, kernel_size=(1, 1),
                     dtype=self.dtype, use_bias=False, name="pw_conv3")(x)
        x = nn.relu(x)

        B, T, F, C = x.shape
        x = x.reshape(B, T, F * C)
        x = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False,
                      kernel_init=default_init(), name="out_proj")(x)
        return x


class ConvModule(nn.Module):
    """Conformer-style convolution module: pointwise up -> GLU -> depthwise conv -> norm -> pointwise down."""
    d_model: int
    kernel_size: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        
        x = nn.Dense(self.d_model * 2, dtype=self.dtype, use_bias=False,
                      kernel_init=default_init(), name="pw_up")(x)
        
        x1, x2 = jnp.split(x, 2, axis=-1)
        x = x1 * nn.sigmoid(x2)
        
        x = nn.Conv(features=self.d_model, kernel_size=(self.kernel_size,),
                     padding='SAME', feature_group_count=self.d_model,
                     dtype=self.dtype, use_bias=False, name="dw_conv")(x)

        x = ZCRMSNorm(dtype=self.dtype, name="conv_norm")(x)
        
        x = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False,
                      kernel_init=residual_init(self.num_layers), name="pw_down")(x)
        return x


class EncoderBlock(nn.Module):
    """Pre-norm self-attention + pre-norm FFN with residual connections."""
    num_heads: int
    num_kv_heads: int
    d_model: int
    d_ff: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    activation: str = "drelu"
    dropout_rate: float = 0.0
    conv_kernel_size: int = 0

    @nn.compact
    def __call__(self, x, mask=None, rope=None, ffn_mask=None, deterministic=True):
        residual = x
        x = ZCRMSNorm(dtype=self.dtype)(x)
        x = MultiHeadAttention(self.num_heads, self.num_kv_heads, self.d_model, self.num_layers, self.dtype, name="self_attn")(
            x, x, mask=mask, rope=rope
        )
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic) + residual

        if self.conv_kernel_size > 0:
            residual = x
            x = ZCRMSNorm(dtype=self.dtype)(x)
            x = ConvModule(self.d_model, self.conv_kernel_size, self.num_layers,
                           self.dtype, name="conv_module")(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic) + residual

        residual = x
        x = ZCRMSNorm(dtype=self.dtype)(x)
        x = FeedForward(self.d_model, self.d_ff, self.num_layers, self.dtype, self.activation)(x, ffn_mask=ffn_mask)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic) + residual

        return x


class _EncoderScanBody(nn.Module):
    """Wraps EncoderBlock for nn.scan: carry = (x, mask, rope, ffn_mask)."""
    num_heads: int
    num_kv_heads: int
    d_model: int
    d_ff: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    activation: str = "drelu"
    dropout_rate: float = 0.0
    conv_kernel_size: int = 0
    deterministic: bool = True

    @nn.compact
    def __call__(self, carry, _):
        x, mask, rope, ffn_mask = carry
        x = EncoderBlock(
            self.num_heads, self.num_kv_heads, self.d_model, self.d_ff,
            self.num_layers, self.dtype, self.activation, self.dropout_rate,
            self.conv_kernel_size,
        )(x, mask, rope, ffn_mask, self.deterministic)
        return (x, mask, rope, ffn_mask), None


class Encoder(nn.Module):
    """Self-attention encoder. Returns (output, mask) tuple."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, mask=None, rope=None, ffn_mask=None, deterministic=True):
        cfg = self.config
        dt = cfg.jax_dtype
        x = x.astype(dt)

        ScanBlock = nn.scan(
            nn.remat(_EncoderScanBody),
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=cfg.num_encoder_layers,
        )
        (x, _, _, _), _ = ScanBlock(
            cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.d_ff,
            cfg.total_layers, dt, cfg.activation, cfg.dropout_rate,
            cfg.conv_kernel_size, deterministic, name="layers",
        )((x, mask, rope, ffn_mask), None)

        x = ZCRMSNorm(dtype=dt, name="final_norm")(x)
        return x, mask


class DecoderBlock(nn.Module):
    num_heads: int
    num_kv_heads: int
    d_model: int
    d_ff: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    activation: str = "drelu"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, encoder_out, self_mask=None, cross_mask=None, rope=None, ffn_mask=None, deterministic=True):
        residual = x
        x = ZCRMSNorm(dtype=self.dtype)(x)
        x = MultiHeadAttention(self.num_heads, self.num_kv_heads, self.d_model, self.num_layers, self.dtype, name="self_attn")(
            x, x, mask=self_mask, rope=rope
        )
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic) + residual

        residual = x
        x = ZCRMSNorm(dtype=self.dtype)(x)
        x = MultiHeadAttention(self.num_heads, self.num_kv_heads, self.d_model, self.num_layers, self.dtype, name="cross_attn")(
            x, encoder_out, mask=cross_mask
        )
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic) + residual

        residual = x
        x = ZCRMSNorm(dtype=self.dtype)(x)
        x = FeedForward(self.d_model, self.d_ff, self.num_layers, self.dtype, self.activation)(x, ffn_mask=ffn_mask)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic) + residual

        return x



class _DecoderScanBody(nn.Module):
    """Wraps DecoderBlock for nn.scan: carry = (x, encoder_out, self_mask, cross_mask, rope, ffn_mask)."""
    num_heads: int
    num_kv_heads: int
    d_model: int
    d_ff: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    activation: str = "drelu"
    dropout_rate: float = 0.0
    deterministic: bool = True

    @nn.compact
    def __call__(self, carry, _):
        x, encoder_out, self_mask, cross_mask, rope, ffn_mask = carry
        x = DecoderBlock(
            self.num_heads, self.num_kv_heads, self.d_model, self.d_ff,
            self.num_layers, self.dtype, self.activation, self.dropout_rate,
        )(x, encoder_out, self_mask, cross_mask, rope, ffn_mask, self.deterministic)
        return (x, encoder_out, self_mask, cross_mask, rope, ffn_mask), None


class Decoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, encoder_out, self_mask=None, cross_mask=None, rope=None, ffn_mask=None, deterministic=True):
        cfg = self.config
        dt = cfg.jax_dtype
        x = x.astype(dt)

        ScanBlock = nn.scan(
            nn.remat(_DecoderScanBody),
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=cfg.num_decoder_layers,
        )
        (x, _, _, _, _, _), _ = ScanBlock(
            cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.d_ff,
            cfg.total_layers, dt, cfg.activation, cfg.dropout_rate,
            deterministic, name="layers",
        )((x, encoder_out, self_mask, cross_mask, rope, ffn_mask), None)

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
            
        B, T, F = mel.shape
        rng = self.make_rng('specaugment')

        x = mel
        
        for i in range(self.num_time_masks):
            rng, k1, k2 = jax.random.split(rng, 3)
            width = jax.random.randint(k1, (), 0, jnp.minimum(self.max_time_width, T))
            start = jax.random.randint(k2, (), 0, jnp.maximum(T - width, 1))
            time_mask = (jnp.arange(T) >= start) & (jnp.arange(T) < start + width)
            x = x * (1.0 - time_mask[None, :, None].astype(x.dtype))

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
        self.spec_augment = SpecAugment()
        self.speech_subsample = SpeechSubsampling(
            self.config.d_model, dtype=self.config.jax_dtype, name="speech_subsample")
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.contrastive_proj = nn.Dense(
            self.config.contrastive_dim, dtype=self.config.jax_dtype,
            use_bias=False, kernel_init=default_init(), name="contrastive_proj",
        )
        self.log_temp = self.param("log_temp", jinit.zeros, ())

    def _rope(self, seq_len):
        head_dim = self.config.d_model // self.config.num_heads
        return precompute_rope_freqs(head_dim, seq_len, self.config.rope_theta)

    def encode_text(self, src, src_mask=None, ffn_mask=None, deterministic=True):
        x = self.embedding(src) * self.embed_scale
        rope = self._rope(src.shape[1])
        return self.encoder(x, mask=src_mask, rope=rope, ffn_mask=ffn_mask, deterministic=deterministic)

    def encode(self, src, src_mask=None):
        """Backward-compatible alias for encode_text."""
        return self.encode_text(src, src_mask=src_mask)

    def encode_speech(self, mel, src_mask=None, ffn_mask=None, deterministic=True):
        mel = self.spec_augment(mel, deterministic=deterministic)
        # FastConformer-style 8x subsampling: (B, T, 80) -> (B, T//8, d_model)
        x = self.speech_subsample(mel) * self.embed_scale
        if src_mask is not None:
            T_new = x.shape[1]
            # Downsample mask by 8x: group frames and keep valid if any in group
            T_mask = src_mask.shape[-1]
            pad_needed = (8 - T_mask % 8) % 8
            if pad_needed > 0:
                src_mask = jnp.pad(src_mask, ((0, 0), (0, 0), (0, 0), (0, pad_needed)))
            src_mask = src_mask.reshape(src_mask.shape[0], 1, 1, -1, 8).any(axis=-1)
            src_mask = src_mask[..., :T_new]
        rope = self._rope(x.shape[1])
        return self.encoder(x, mask=src_mask, rope=rope, ffn_mask=ffn_mask, deterministic=deterministic)

    def decode(self, tgt, encoder_out, self_mask=None, cross_mask=None, deterministic=True):
        """Decode from encoder output with cross_mask for variable-length encoder output."""
        x = self.embedding(tgt) * self.embed_scale
        rope = self._rope(tgt.shape[1])
        x = self.decoder(x, encoder_out, self_mask=self_mask, cross_mask=cross_mask, rope=rope, deterministic=deterministic)
        logits = x.astype(jnp.float32) @ self.embedding.embedding.T
        return logits

    def _mean_pool(self, encoder_out, enc_mask):
        """Mean-pool encoder output over non-padded positions. Returns (B, d_model)."""
        if enc_mask is not None:
            mask_2d = enc_mask[:, 0, 0, :]
        else:
            mask_2d = jnp.ones(encoder_out.shape[:2], dtype=encoder_out.dtype)
        mask_3d = mask_2d[:, :, None].astype(encoder_out.dtype) 
        summed = jnp.sum(encoder_out * mask_3d, axis=1) 
        counts = jnp.maximum(jnp.sum(mask_2d, axis=1, keepdims=True), 1.0)
        return summed / counts

    def encode_contrastive(self, tokens, deterministic=True):
        """Encode tokens and project to L2-normalized contrastive space. Returns (B, contrastive_dim)."""
        src_mask = make_padding_mask(tokens, self.config.pad_token_id)
        encoder_out, enc_mask = self.encode_text(tokens, src_mask=src_mask, deterministic=deterministic)
        pooled = self._mean_pool(encoder_out, enc_mask)
        projected = self.contrastive_proj(pooled)
        return projected / jnp.maximum(jnp.linalg.norm(projected, axis=-1, keepdims=True), 1e-8)

    def forward_contrastive(self, query_tokens, tool_tokens, deterministic=True):
        """Encode query and tool tokens, return (q_emb, t_emb, log_temp)."""
        q_emb = self.encode_contrastive(query_tokens, deterministic=deterministic)
        t_emb = self.encode_contrastive(tool_tokens, deterministic=deterministic)
        return q_emb, t_emb, self.log_temp

    def __call__(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_out, enc_mask = self.encode_text(src, src_mask=src_mask)
        logits = self.decode(tgt, encoder_out, self_mask=tgt_mask, cross_mask=enc_mask)
        return logits

    def _run_decoder(self, encoder_out, tgt, tgt_mask=None, cross_mask=None, ffn_mask=None, deterministic=True):
        """Run decoder and return float32 hidden states."""
        x = self.embedding(tgt) * self.embed_scale
        rope = self._rope(tgt.shape[1])
        x = self.decoder(x, encoder_out, self_mask=tgt_mask, cross_mask=cross_mask, rope=rope, ffn_mask=ffn_mask, deterministic=deterministic)
        return x.astype(jnp.float32)

    def forward_masked(self, src, tgt, src_mask=None, tgt_mask=None, ffn_mask=None, deterministic=True):
        """Single forward with per-batch-item FFN masking. Returns (logits, slot_div)."""
        encoder_out, enc_mask = self.encode_text(src, src_mask=src_mask, ffn_mask=ffn_mask, deterministic=deterministic)
        x_f32 = self._run_decoder(encoder_out, tgt, tgt_mask=tgt_mask, cross_mask=enc_mask, ffn_mask=ffn_mask, deterministic=deterministic)
        logits = x_f32 @ self.embedding.embedding.T
        return logits, 0.0

    def forward_speech_masked(self, mel, tgt, src_mask=None, tgt_mask=None, ffn_mask=None, deterministic=True):
        """Single speech forward with per-batch-item FFN masking. Returns (logits, slot_div)."""
        encoder_out, enc_mask = self.encode_speech(mel, src_mask=src_mask, ffn_mask=ffn_mask, deterministic=deterministic)
        x_f32 = self._run_decoder(encoder_out, tgt, tgt_mask=tgt_mask, cross_mask=enc_mask, ffn_mask=ffn_mask, deterministic=deterministic)
        logits = x_f32 @ self.embedding.embedding.T
        return logits, 0.0

    def _make_eval_ffn_mask(self, ff_width, B, dtype):
        """Default prefix FFN mask for eval: first ff_width neurons active."""
        mask = (jnp.arange(self.config.d_ff) < ff_width).astype(dtype)
        return jnp.broadcast_to(mask[None, :], (B, self.config.d_ff))

    def forward_with_aux(self, src, tgt, src_mask=None, tgt_mask=None, mat_ff_widths=None):
        """Eval-only: separate per-width forwards for reporting per-width PPL.

        mat_ff_widths: list of FFN widths to evaluate (e.g. [1024, 512, 256]).
        """
        emb = self.embedding.embedding
        B = src.shape[0]

        encoder_out, enc_mask = self.encode_text(src, src_mask=src_mask)
        x_f32 = self._run_decoder(encoder_out, tgt, tgt_mask=tgt_mask, cross_mask=enc_mask)
        logits = x_f32 @ emb.T

        mat_logits = []
        if mat_ff_widths is not None:
            for ff_w in mat_ff_widths:
                mask = self._make_eval_ffn_mask(ff_w, B, x_f32.dtype)
                enc_m, enc_m_mask = self.encode_text(src, src_mask=src_mask, ffn_mask=mask)
                x_m = self._run_decoder(enc_m, tgt, tgt_mask=tgt_mask, cross_mask=enc_m_mask, ffn_mask=mask)
                mat_logits.append(x_m @ emb.T)

        return logits, 0.0, mat_logits

    def forward_speech_with_aux(self, mel, tgt, src_mask=None, tgt_mask=None, mat_ff_widths=None, deterministic=True):
        """Eval-only: separate per-width speech forwards for reporting per-width PPL."""
        emb = self.embedding.embedding
        B = mel.shape[0]

        encoder_out, enc_mask = self.encode_speech(mel, src_mask=src_mask, deterministic=deterministic)
        x_f32 = self._run_decoder(encoder_out, tgt, tgt_mask=tgt_mask, cross_mask=enc_mask)
        logits = x_f32 @ emb.T

        mat_logits = []
        if mat_ff_widths is not None:
            for ff_w in mat_ff_widths:
                mask = self._make_eval_ffn_mask(ff_w, B, x_f32.dtype)
                enc_m, enc_m_mask = self.encode_speech(mel, src_mask=src_mask, ffn_mask=mask, deterministic=deterministic)
                x_m = self._run_decoder(enc_m, tgt, tgt_mask=tgt_mask, cross_mask=enc_m_mask, ffn_mask=mask)
                mat_logits.append(x_m @ emb.T)

        return logits, 0.0, mat_logits

    def init_all(self, src, tgt, mel):
        """Dummy forward through both text and speech pathways to initialize all params."""
        src_mask = make_padding_mask(src, self.config.pad_token_id)
        tgt_mask = make_causal_mask(tgt.shape[1]) & make_padding_mask(tgt, self.config.pad_token_id)
        text_out, text_enc_mask = self.encode_text(src, src_mask=src_mask)
        mel_mask = make_mel_padding_mask(mel)
        speech_out, speech_enc_mask = self.encode_speech(mel, src_mask=mel_mask, deterministic=True)
        _ = self._run_decoder(text_out, tgt, tgt_mask=tgt_mask, cross_mask=text_enc_mask)
        _ = self._run_decoder(speech_out, tgt, tgt_mask=tgt_mask, cross_mask=speech_enc_mask)
        _ = self.encode_contrastive(src)
        return jnp.zeros(())


def make_causal_mask(seq_len):
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask[None, None, :, :]


def make_padding_mask(tokens, pad_token_id):
    mask = tokens != pad_token_id
    return mask[:, None, None, :]


def make_mel_padding_mask(mel):
    """Create padding mask from mel spectrogram: non-zero frames are valid."""
    mask = jnp.any(mel != 0, axis=-1)  
    return mask[:, None, None, :] 
