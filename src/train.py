import argparse
import math
import os
import pickle
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
from flax import jax_utils
from flax.training import train_state

from .data import (
    get_batches, get_tokenizer, load_tinystories, prepare_text_pairs,
    load_librispeech, prepare_speech_pairs, get_speech_batches,
)
from .model import (
    EncoderDecoderTransformer,
    TransformerConfig,
    make_causal_mask,
    make_padding_mask,
    make_mel_padding_mask,
)

def _newton_schulz(G, steps=5):
    """Approximate polar decomposition via Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    orig_dtype = G.dtype
    G = G.astype(jnp.float32)
    X = G / (jnp.linalg.norm(G) + 1e-7)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.astype(orig_dtype)


class MuonState(NamedTuple):
    mu: optax.Updates


def scale_by_muon(momentum=0.95, ns_steps=5):
    """Muon gradient transform: orthogonalize 2D+ grads, then Nesterov momentum."""

    def init_fn(params):
        return MuonState(mu=jax.tree.map(jnp.zeros_like, params))

    def update_fn(updates, state, params=None):
        del params

        def ortho(g):
            if g.ndim == 2:
                return _newton_schulz(g, steps=ns_steps)
            return g

        ortho_g = jax.tree.map(ortho, updates)
        new_mu = jax.tree.map(lambda m, g: momentum * m + g, state.mu, ortho_g)
        new_updates = jax.tree.map(
            lambda g, m: g + momentum * m, ortho_g, new_mu
        )
        return new_updates, MuonState(mu=new_mu)

    return optax.GradientTransformation(init_fn, update_fn)


def _param_labels(params):
    """Label each param: 'muon' for Dense kernels, 'adam' for the rest."""

    def _label(path, leaf):
        name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        if name == "kernel" and leaf.ndim == 2:
            return "muon"
        return "adam"

    return jax.tree_util.tree_map_with_path(_label, params)


def _wsd_schedule(peak_value, total_steps, warmup_steps, decay_ratio=0.15):
    """Warmup-Stable-Decay schedule: linear warmup, hold peak, linear decay."""
    decay_steps = max(1, int(total_steps * decay_ratio))
    stable_steps = total_steps - warmup_steps - decay_steps
    return optax.join_schedules(
        [
            optax.linear_schedule(0.0, peak_value, warmup_steps),
            optax.constant_schedule(peak_value),
            optax.linear_schedule(peak_value, peak_value * 0.1, decay_steps),
        ],
        boundaries=[warmup_steps, warmup_steps + stable_steps],
    )


def create_train_state(rng, config, learning_rate, muon_lr, total_steps, warmup_steps):
    model = EncoderDecoderTransformer(config)

    rng, init_rng = jax.random.split(rng)
    dummy_src = jnp.ones((1, 128), dtype=jnp.int32)
    dummy_tgt = jnp.ones((1, 128), dtype=jnp.int32)
    dummy_mel = jnp.ones((1, 128, config.n_mels), dtype=jnp.float32)
    variables = model.init(
        {"params": init_rng},
        dummy_src,
        dummy_tgt,
        dummy_mel,
        method="init_all",
    )

    adam_schedule = _wsd_schedule(learning_rate, total_steps, warmup_steps)
    muon_schedule = _wsd_schedule(muon_lr, total_steps, warmup_steps)

    muon_opt = optax.chain(
        scale_by_muon(momentum=0.95, ns_steps=5),
        optax.add_decayed_weights(weight_decay=0.01),
        optax.scale_by_schedule(muon_schedule),
        optax.scale(-1.0),
    )
    adam_opt = optax.chain(
        optax.adamw(adam_schedule, b2=0.95, weight_decay=0.0),
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {"muon": muon_opt, "adam": adam_opt},
            _param_labels,
        ),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )


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


def _cubic_sparsity_schedule(step, t_start, t_end, s_final):
    """Cubic sparsity ramp (Zhu & Gupta 2017). Returns target sparsity at *step*."""
    if step < t_start:
        return 0.0
    if step >= t_end:
        return s_final
    frac = (step - t_start) / (t_end - t_start)
    return s_final * (1.0 - (1.0 - frac) ** 3)


def _make_prune_mask(params, sparsity, group_size):
    """Compute block-prune mask entirely on-device (no numpy round-trips).

    Returns binary mask tree matching param shapes and dtypes.
    """
    # Pass 1: collect block L1 scores as JAX arrays
    all_scores = []
    for _, leaf in jax.tree_util.tree_leaves_with_path(params):
        if leaf.ndim != 2:
            continue
        in_feat, out_feat = leaf.shape
        gs = min(group_size, in_feat)
        pad = (gs - in_feat % gs) % gs
        w = jnp.pad(leaf, ((0, pad), (0, 0))) if pad else leaf
        scores = jnp.sum(jnp.abs(w.reshape(-1, gs, out_feat)), axis=1).ravel()
        all_scores.append(scores)

    if not all_scores:
        return jax.tree.map(jnp.ones_like, params)

    threshold = jnp.percentile(jnp.concatenate(all_scores), sparsity * 100)

    # Pass 2: build per-leaf binary mask (in param dtype to save memory)
    def _leaf_mask(path, leaf):
        if leaf.ndim != 2:
            return jnp.ones_like(leaf)
        in_feat, out_feat = leaf.shape
        gs = min(group_size, in_feat)
        pad = (gs - in_feat % gs) % gs
        w = jnp.pad(leaf, ((0, pad), (0, 0))) if pad else leaf
        w_grouped = w.reshape(-1, gs, out_feat)
        block_keep = (jnp.sum(jnp.abs(w_grouped), axis=1, keepdims=True) > threshold)
        return jnp.broadcast_to(block_keep, w_grouped.shape).reshape(-1, out_feat)[:in_feat].astype(leaf.dtype)

    return jax.tree_util.tree_map_with_path(_leaf_mask, params)


def _quantize_params(params, group_size=32):
    """Fake-quantize all Dense kernels in the param tree."""
    def _maybe_quantize(path, leaf):
        name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        if name == "kernel" and leaf.ndim == 2:
            return _fake_quantize_int4(leaf, group_size=group_size)
        return leaf
    return jax.tree_util.tree_map_with_path(_maybe_quantize, params)


_GROUP_SIZE = 32
_MAT_FACTORS = ()
_MAT_FF_WIDTHS = ()  # precomputed d_ff widths per factor
_D_FF = 2048  # set in train()


def _text_loss_fn(state, params, src, tgt_in, tgt_out, causal_mask, ffn_mask):
    pad_id = 0
    src_mask = make_padding_mask(src, pad_id)
    tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
    logits, slot_div = state.apply_fn(
        {"params": _quantize_params(params, group_size=_GROUP_SIZE)},
        src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask,
        ffn_mask=ffn_mask,
        method="forward_masked",
    )
    logits_f32 = logits.astype(jnp.float32)
    mask = (tgt_out != pad_id).astype(jnp.float32)
    ce_loss = jnp.sum(
        optax.softmax_cross_entropy_with_integer_labels(logits_f32, tgt_out) * mask
    ) / jnp.maximum(jnp.sum(mask), 1.0)
    z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits_f32, axis=-1) ** 2)
    div_loss = 1e-4 * slot_div
    return ce_loss + z_loss + div_loss


def _speech_loss_fn(state, params, mel, tgt_in, tgt_out, causal_mask, ffn_mask, rng):
    pad_id = 0
    src_mask = make_mel_padding_mask(mel)
    tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
    logits, slot_div = state.apply_fn(
        {"params": _quantize_params(params, group_size=_GROUP_SIZE)},
        mel, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask,
        ffn_mask=ffn_mask,
        deterministic=False,
        method="forward_speech_masked",
        rngs={"specaugment": rng},
    )
    logits_f32 = logits.astype(jnp.float32)
    mask = (tgt_out != pad_id).astype(jnp.float32)
    ce_loss = jnp.sum(
        optax.softmax_cross_entropy_with_integer_labels(logits_f32, tgt_out) * mask
    ) / jnp.maximum(jnp.sum(mask), 1.0)
    z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits_f32, axis=-1) ** 2)
    div_loss = 1e-4 * slot_div
    return ce_loss + z_loss + div_loss


def _make_ffn_mask(batch_size, d_ff, mat_ff_widths):
    """Build (batch, d_ff) prefix mask: batch split equally across widths.

    First 1/N = full width (all ones), remaining 1/N sections = mat widths.
    """
    n_widths = 1 + len(mat_ff_widths)  # full + mat widths
    per_width = batch_size // n_widths
    arange = jnp.arange(d_ff)
    rows = [jnp.ones((per_width, d_ff), dtype=jnp.bfloat16)]  # full width
    for k in mat_ff_widths:
        rows.append((arange[None, :] < k).astype(jnp.bfloat16).repeat(per_width, axis=0))
    # Handle remainder (assign to full width)
    remainder = batch_size - per_width * n_widths
    if remainder > 0:
        rows.append(jnp.ones((remainder, d_ff), dtype=jnp.bfloat16))
    return jnp.concatenate(rows, axis=0)


def _train_step_text(state, ema_params, src, tgt_in, tgt_out, causal_mask, ffn_mask):
    ema_decay = 0.999
    loss, grads = jax.value_and_grad(
        lambda p: _text_loss_fn(state, p, src, tgt_in, tgt_out, causal_mask, ffn_mask)
    )(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    ema_params = jax.tree.map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema_params, state.params)
    return state, ema_params, loss, grad_norm


def _train_step_text_masked(state, ema_params, src, tgt_in, tgt_out, causal_mask, prune_mask, ffn_mask):
    """Text training step with fused prune mask application."""
    ema_decay = 0.999
    loss, grads = jax.value_and_grad(
        lambda p: _text_loss_fn(state, p, src, tgt_in, tgt_out, causal_mask, ffn_mask)
    )(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    masked_params = jax.tree.map(lambda w, m: w * m, state.params, prune_mask)
    state = state.replace(params=masked_params)
    ema_params = jax.tree.map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema_params, masked_params)
    return state, ema_params, loss, grad_norm


def _train_step_speech(state, ema_params, mel, tgt_in, tgt_out, causal_mask, ffn_mask, rng):
    ema_decay = 0.999
    loss, grads = jax.value_and_grad(
        lambda p: _speech_loss_fn(state, p, mel, tgt_in, tgt_out, causal_mask, ffn_mask, rng)
    )(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    ema_params = jax.tree.map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema_params, state.params)
    return state, ema_params, loss, grad_norm


def _train_step_speech_masked(state, ema_params, mel, tgt_in, tgt_out, causal_mask, prune_mask, ffn_mask, rng):
    """Speech training step with fused prune mask application."""
    ema_decay = 0.999
    loss, grads = jax.value_and_grad(
        lambda p: _speech_loss_fn(state, p, mel, tgt_in, tgt_out, causal_mask, ffn_mask, rng)
    )(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    masked_params = jax.tree.map(lambda w, m: w * m, state.params, prune_mask)
    state = state.replace(params=masked_params)
    ema_params = jax.tree.map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema_params, masked_params)
    return state, ema_params, loss, grad_norm


def _make_p_train_step():
    return jax.pmap(_train_step_text, axis_name="batch", donate_argnums=(0, 1))


def _make_p_train_step_masked():
    return jax.pmap(_train_step_text_masked, axis_name="batch", donate_argnums=(0, 1))


def _make_p_train_step_speech():
    return jax.pmap(_train_step_speech, axis_name="batch", donate_argnums=(0, 1))


def _make_p_train_step_speech_masked():
    return jax.pmap(_train_step_speech_masked, axis_name="batch", donate_argnums=(0, 1))


def _make_val_loss_fn(apply_fn):
    @jax.jit
    def val_loss_batch(params, src, tgt_in, tgt_out, causal_mask):
        pad_id = 0
        src_mask = make_padding_mask(src, pad_id)
        tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
        logits = apply_fn(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=src_mask,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tgt_out)
        mask = (tgt_out != pad_id).astype(jnp.float32)
        return jnp.sum(loss * mask), jnp.sum(mask)
    return val_loss_batch


def _make_mat_val_loss_fn(apply_fn, ff_width):
    """Val loss for matryoshka sub-model at given FFN width."""
    @jax.jit
    def val_loss_batch(params, src, tgt_in, tgt_out, causal_mask):
        pad_id = 0
        src_mask = make_padding_mask(src, pad_id)
        tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
        logits, _, mat_logits = apply_fn(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=src_mask,
            mat_ff_widths=(ff_width,),
            method="forward_with_aux",
        )
        trunc_logits = mat_logits[0].astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(trunc_logits, tgt_out)
        mask = (tgt_out != pad_id).astype(jnp.float32)
        return jnp.sum(loss * mask), jnp.sum(mask)
    return val_loss_batch


def _make_speech_val_loss_fn(apply_fn):
    @jax.jit
    def val_loss_batch(params, mel, tgt_in, tgt_out, causal_mask):
        pad_id = 0
        src_mask = make_mel_padding_mask(mel)
        tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
        logits, _, _ = apply_fn(
            {"params": params}, mel, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask,
            deterministic=True,
            method="forward_speech_with_aux",
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tgt_out)
        mask = (tgt_out != pad_id).astype(jnp.float32)
        return jnp.sum(loss * mask), jnp.sum(mask)
    return val_loss_batch


def _estimate_mat_params(config, matryoshka_factor):
    """Estimate parameter count of a sub-model at a given matryoshka factor.

    matryoshka_factor: how many times smaller the FFN widths are (e.g. 2 = half width).
    Embedding and attention weights are unchanged.
    """
    d = config.d_model
    v = config.vocab_size
    n_enc = config.num_encoder_layers
    n_dec = config.num_decoder_layers
    kv_dim = config.num_kv_heads * (d // config.num_heads)
    m = config.num_memory_slots
    d_ff = config.d_ff // matryoshka_factor

    emb = v * d
    attn = d * d + d * kv_dim * 2 + d * d
    ffn = d * d_ff * 3
    mixer_token = m * d_ff * 3
    mixer_channel = d * d_ff * 3
    enc_block = attn + ffn + mixer_token + mixer_channel
    dec_block = attn * 2 + ffn
    total = emb + n_enc * enc_block + n_dec * dec_block
    return int(total)


def shard_batch(batch, num_devices):
    """Reshape a batch array so leading dim is (num_devices, per_device_batch, ...)."""
    return batch.reshape(num_devices, -1, *batch.shape[1:])


def train(args):
    import itertools

    num_devices = jax.local_device_count()
    no_speech = getattr(args, "no_speech", False)
    speech_every_override = getattr(args, "speech_every", None)
    n_mels = getattr(args, "n_mels", 80)
    max_mel_len = getattr(args, "max_mel_len", 1024)

    use_wandb = getattr(args, "wandb", False)
    if use_wandb:
        import wandb
        if wandb.run is None:
            wandb.init(project="needle-v1", config=vars(args))

    total_data_steps = 6 if not no_speech else 4
    step_idx = 0

    step_idx += 1
    print(f"\n[{step_idx}/{total_data_steps}] Detecting devices...")
    print(f"      {num_devices} device(s) for data-parallel training")

    step_idx += 1
    print(f"\n[{step_idx}/{total_data_steps}] Loading tokenizer...")
    tokenizer = get_tokenizer(max_samples=args.max_samples)

    step_idx += 1
    print(f"\n[{step_idx}/{total_data_steps}] Loading TinyStories dataset...")
    ds = load_tinystories("train", max_samples=args.max_samples)
    val_ds = load_tinystories("validation", max_samples=getattr(args, "max_eval_samples", None))

    step_idx += 1
    print(f"\n[{step_idx}/{total_data_steps}] Preparing text pairs...")
    enc_inputs, dec_inputs, dec_targets = prepare_text_pairs(
        ds, tokenizer, max_enc_len=args.max_enc_len, max_dec_len=args.max_dec_len
    )
    val_enc, val_dec_in, val_dec_tgt = prepare_text_pairs(
        val_ds, tokenizer, max_enc_len=args.max_enc_len, max_dec_len=args.max_dec_len
    )
    val_enc = np.array(val_enc)
    val_dec_in = np.array(val_dec_in)
    val_dec_tgt = np.array(val_dec_tgt)
    print(f"      {len(enc_inputs):,} train / {len(val_enc):,} val text pairs")

    # --- Speech data ---
    speech_mel = speech_dec_in = speech_dec_tgt = None
    val_speech_mel = val_speech_dec_in = val_speech_dec_tgt = None
    if not no_speech:
        step_idx += 1
        max_speech_samples = getattr(args, "max_speech_samples", None)
        print(f"\n[{step_idx}/{total_data_steps}] Loading LibriSpeech...")
        speech_ds = load_librispeech("train.clean.100", max_samples=max_speech_samples)
        speech_val_ds = load_librispeech("validation.clean", max_samples=getattr(args, "max_eval_samples", None))

        step_idx += 1
        print(f"\n[{step_idx}/{total_data_steps}] Preparing speech pairs...")
        speech_mel, speech_dec_in, speech_dec_tgt = prepare_speech_pairs(
            speech_ds, tokenizer, n_mels=n_mels, max_mel_len=max_mel_len, max_dec_len=args.max_dec_len
        )
        val_speech_mel, val_speech_dec_in, val_speech_dec_tgt = prepare_speech_pairs(
            speech_val_ds, tokenizer, n_mels=n_mels, max_mel_len=max_mel_len, max_dec_len=args.max_dec_len
        )
        speech_mel = np.array(speech_mel)
        speech_dec_in = np.array(speech_dec_in)
        speech_dec_tgt = np.array(speech_dec_tgt)
        val_speech_mel = np.array(val_speech_mel)
        val_speech_dec_in = np.array(val_speech_dec_in)
        val_speech_dec_tgt = np.array(val_speech_dec_tgt)
        print(f"      {len(speech_mel):,} train / {len(val_speech_mel):,} val speech pairs")

    effective_batch_size = args.batch_size * num_devices

    resume_checkpoint = getattr(args, "checkpoint", None)
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        with open(resume_checkpoint, "rb") as f:
            ckpt_data = pickle.load(f)
        ckpt_params = jax.tree.map(jnp.array, ckpt_data["params"])
        config = TransformerConfig(**ckpt_data["config"])
        print(f"  Config: d={config.d_model}, heads={config.num_heads}, layers={config.num_encoder_layers}/{config.num_decoder_layers}")
    else:
        config = TransformerConfig(
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_kv_heads=getattr(args, "num_kv_heads", None) or args.num_heads,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=getattr(args, "num_dec_layers", args.num_layers),
            d_ff=getattr(args, "d_ff", None) or args.d_model * 4,
            max_seq_len=max(args.max_enc_len, args.max_dec_len),
            dtype=args.dtype,
            activation=getattr(args, "activation", "drelu"),
            num_memory_slots=getattr(args, "num_memory_slots", 64),
            n_mels=n_mels,
        )

    global _GROUP_SIZE, _MAT_FACTORS, _MAT_FF_WIDTHS, _D_FF
    _GROUP_SIZE = getattr(args, "group_size", 32)
    _D_FF = config.d_ff
    mat_factors_raw = getattr(args, "mat_factors", None)
    if mat_factors_raw:
        _MAT_FACTORS = tuple(f for f in mat_factors_raw if f > 1)
        _MAT_FF_WIDTHS = tuple(config.d_ff // f for f in _MAT_FACTORS)
    else:
        _MAT_FACTORS = ()
        _MAT_FF_WIDTHS = ()
    n_widths = 1 + len(_MAT_FF_WIDTHS) if _MAT_FF_WIDTHS else 1
    p_train_step = _make_p_train_step()
    p_train_step_masked = _make_p_train_step_masked()
    p_train_step_speech = _make_p_train_step_speech()
    p_train_step_speech_masked = _make_p_train_step_speech_masked()

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    mat_shared_input = getattr(args, "mat_shared_input", False)
    # With shared input, each unique sample is repeated n_widths times,
    # so we fetch smaller batches but take more steps per epoch.
    unique_batch_size = effective_batch_size // n_widths if (mat_shared_input and n_widths > 1) else effective_batch_size
    text_batches_per_epoch = len(enc_inputs) // unique_batch_size
    if speech_every_override is not None:
        speech_every = speech_every_override
    elif not no_speech and speech_mel is not None:
        speech_batches_available = len(speech_mel) // effective_batch_size
        speech_every = max(1, text_batches_per_epoch // max(1, speech_batches_available))
    else:
        speech_every = 1
    speech_batches_per_epoch = text_batches_per_epoch // speech_every if (not no_speech and speech_mel is not None) else 0
    num_batches = text_batches_per_epoch + speech_batches_per_epoch
    total_steps = num_batches * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    scaled_lr = args.lr * num_devices
    muon_lr = getattr(args, "muon_lr", 0.02) * math.sqrt(num_devices)
    state = create_train_state(init_rng, config, scaled_lr, muon_lr, total_steps, warmup_steps)
    val_loss_fn = _make_val_loss_fn(state.apply_fn)
    speech_vl_fn = _make_speech_val_loss_fn(state.apply_fn) if not no_speech else None

    if resume_checkpoint:
        state = state.replace(params=ckpt_params)
        print(f"  Loaded checkpoint params into train state")

    ema_params = jax.tree.map(jnp.copy, state.params)
    state = jax_utils.replicate(state)
    ema_params = jax_utils.replicate(ema_params)

    param_count = sum(x.size for x in jax.tree.leaves(jax_utils.unreplicate(state).params))
    decay_steps = max(1, int(total_steps * 0.15))
    stable_steps = total_steps - warmup_steps - decay_steps

    print(f"\n  ─────────────────────────────────────")
    print(f"  Parameters    {param_count:>12,}")
    print(f"  d_model       {config.d_model:>12}")
    print(f"  Heads         {config.num_heads:>7} ({config.num_kv_heads} KV)")
    print(f"  Layers        {config.num_encoder_layers:>7} enc / {config.num_decoder_layers} dec")
    print(f"  Memory slots  {config.num_memory_slots:>12}")
    print(f"  Activation    {config.activation:>12}")
    print(f"  Dtype         {config.dtype:>12}")
    if not no_speech:
        print(f"  Speech        every {speech_every} text steps")
        print(f"  n_mels        {n_mels:>12}")
        print(f"  max_mel_len   {max_mel_len:>12}")
    else:
        print(f"  Speech               disabled")
    print(f"  ─────────────────────────────────────")
    print(f"  Devices       {num_devices:>12}")
    print(f"  Batch         {args.batch_size:>7} x {num_devices} = {effective_batch_size}")
    print(f"  Adam LR       {args.lr:>7} x {num_devices} = {scaled_lr}")
    print(f"  Muon LR       {args.muon_lr:>7.4f} -> {muon_lr:.4f}")
    print(f"  Schedule      {warmup_steps}w / {stable_steps}s / {decay_steps}d (WSD)")
    print(f"  Total steps   {total_steps:>12,}")
    print(f"  Epochs        {args.epochs:>12}")
    print(f"  ─────────────────────────────────────\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    global_step = 0
    causal_mask = jnp.broadcast_to(
        make_causal_mask(args.max_dec_len),
        (num_devices, 1, args.max_dec_len, args.max_dec_len),
    )

    # Precompute FFN mask for heterogeneous matryoshka training.
    # Batch is split: 1/N full-width + 1/N per mat factor.
    text_ffn_mask = _make_ffn_mask(args.batch_size, config.d_ff, _MAT_FF_WIDTHS)
    text_ffn_mask = jnp.broadcast_to(
        text_ffn_mask[None, :, :],
        (num_devices, args.batch_size, config.d_ff),
    )
    if n_widths > 1:
        print(f"  Mat factors   {n_widths} (full + {', '.join(str(f)+'x' for f in _MAT_FACTORS)})")
        if mat_shared_input:
            print(f"  Mat mode      shared input ({unique_batch_size // num_devices}/dev x {n_widths} repeats)")
        else:
            print(f"  Mat mode      unique input ({args.batch_size}/dev, random width)")

    adam_schedule = _wsd_schedule(scaled_lr, total_steps, warmup_steps)
    muon_schedule = _wsd_schedule(muon_lr, total_steps, warmup_steps)
    tokens_per_batch = effective_batch_size * (args.max_enc_len + args.max_dec_len)

    enc_inputs = np.array(enc_inputs)
    dec_inputs = np.array(dec_inputs)
    dec_targets = np.array(dec_targets)

    last_val_ppl = None
    sparsity_ratio = getattr(args, "sparsity_ratio", 0.0)
    prune_mask = None
    gradual_sparsify_done = False

    prune_interval = getattr(args, "prune_interval", 100)
    prune_start_frac = getattr(args, "prune_start_frac", 0.33)
    prune_end_frac = getattr(args, "prune_end_frac", 0.67)

    weight_prune_epoch = 0 if sparsity_ratio > 0 else -1

    for epoch in range(args.epochs):
        if epoch == weight_prune_epoch and not gradual_sparsify_done:
            t_start = int(num_batches * prune_start_frac)
            t_end = int(num_batches * prune_end_frac)
            print(f"\nGradual magnitude sparsification: 0% -> {sparsity_ratio*100:.0f}% over epoch {epoch+1} "
                  f"(steps {t_start}-{t_end}/{num_batches}, interval={prune_interval}, group_size={_GROUP_SIZE})")
            epoch_step = 0

        text_losses = []
        speech_losses = []
        text_batches = get_batches(enc_inputs, dec_inputs, dec_targets, unique_batch_size)

        # Speech batches cycle via itertools.cycle
        speech_iter = None
        if not no_speech and speech_mel is not None:
            speech_iter = iter(itertools.cycle(
                get_speech_batches(speech_mel, speech_dec_in, speech_dec_tgt, unique_batch_size)
            ))

        text_step_in_epoch = 0
        pbar = tqdm(text_batches, total=text_batches_per_epoch, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for src, tgt_in, tgt_out in pbar:
            # --- Text step ---
            t0 = time.perf_counter()

            if n_widths > 1 and mat_shared_input:
                # Shared: each unique input sees all widths. Shard unique samples
                # across devices, then tile within each device's portion so the
                # ffn_mask width sections each get the same unique inputs.
                per_width = args.batch_size // n_widths
                def _tile_for_mat(arr):
                    # (unique_batch, ...) -> (devices, per_width, ...) -> tile -> (devices, batch, ...)
                    s = arr.reshape(num_devices, per_width, *arr.shape[1:])
                    s = np.tile(s, (1, n_widths) + (1,) * (arr.ndim - 1))
                    return s
                src_b = _tile_for_mat(src)
                tgt_in_b = _tile_for_mat(tgt_in)
                tgt_out_b = _tile_for_mat(tgt_out)
            else:
                src_b = shard_batch(src, num_devices)
                tgt_in_b = shard_batch(tgt_in, num_devices)
                tgt_out_b = shard_batch(tgt_out, num_devices)

            if prune_mask is not None:
                state, ema_params, loss, grad_norm = p_train_step_masked(
                    state, ema_params, src_b, tgt_in_b, tgt_out_b, causal_mask, prune_mask, text_ffn_mask,
                )
            else:
                state, ema_params, loss, grad_norm = p_train_step(
                    state, ema_params, src_b, tgt_in_b, tgt_out_b, causal_mask, text_ffn_mask,
                )

            text_loss_val = float(loss[0])
            text_losses.append(text_loss_val)
            global_step += 1
            text_step_in_epoch += 1

            # --- Speech step (every speech_every text steps) ---
            speech_loss_val = None
            if speech_iter is not None and text_step_in_epoch % speech_every == 0:
                mel_batch, sp_tgt_in, sp_tgt_out = next(speech_iter)

                if n_widths > 1 and mat_shared_input:
                    per_width = args.batch_size // n_widths
                    def _tile_sp(arr):
                        s = arr.reshape(num_devices, per_width, *arr.shape[1:])
                        return np.tile(s, (1, n_widths) + (1,) * (arr.ndim - 1))
                    mel_b = _tile_sp(mel_batch)
                    sp_tgt_in_b = _tile_sp(sp_tgt_in)
                    sp_tgt_out_b = _tile_sp(sp_tgt_out)
                else:
                    mel_b = shard_batch(mel_batch, num_devices)
                    sp_tgt_in_b = shard_batch(sp_tgt_in, num_devices)
                    sp_tgt_out_b = shard_batch(sp_tgt_out, num_devices)

                speech_ffn_mask = text_ffn_mask
                rng, spec_rng = jax.random.split(rng)
                spec_rngs = jax.random.split(spec_rng, num_devices)

                if prune_mask is not None:
                    state, ema_params, sp_loss, sp_grad_norm = p_train_step_speech_masked(
                        state, ema_params, mel_b, sp_tgt_in_b, sp_tgt_out_b, causal_mask, prune_mask, speech_ffn_mask, spec_rngs,
                    )
                else:
                    state, ema_params, sp_loss, sp_grad_norm = p_train_step_speech(
                        state, ema_params, mel_b, sp_tgt_in_b, sp_tgt_out_b, causal_mask, speech_ffn_mask, spec_rngs,
                    )
                speech_loss_val = float(sp_loss[0])
                speech_losses.append(speech_loss_val)
                global_step += 1

            # --- Gradual magnitude sparsification (epoch 1 only) ---
            if epoch == weight_prune_epoch and not gradual_sparsify_done:
                epoch_step += 1
                current_sparsity = _cubic_sparsity_schedule(epoch_step, t_start, t_end, sparsity_ratio)
                if epoch_step >= t_start and epoch_step % prune_interval == 0 and current_sparsity > 0:
                    ema_unr = jax_utils.unreplicate(ema_params)
                    mask = _make_prune_mask(ema_unr, current_sparsity, _GROUP_SIZE)
                    del ema_unr
                    prune_mask = jax_utils.replicate(mask)
                    del mask

            dt = time.perf_counter() - t0
            eval_every = getattr(args, "eval_every", 100)
            if global_step % eval_every == 0 or global_step == total_steps:
                _eval_params = jax_utils.unreplicate(ema_params)
                val_causal = make_causal_mask(args.max_dec_len)
                total_loss, total_toks = 0.0, 0.0
                for vb in get_batches(val_enc, val_dec_in, val_dec_tgt, args.batch_size, shuffle=False):
                    vl, vt = val_loss_fn(_eval_params, vb[0], vb[1], vb[2], val_causal)
                    total_loss += float(vl)
                    total_toks += float(vt)
                last_val_ppl = float(math.exp(total_loss / max(total_toks, 1)))

                del _eval_params

            # --- tqdm postfix ---
            postfix = {
                "text_loss": f"{text_loss_val:.4f}",
                "text_ppl": f"{last_val_ppl:.2f}" if last_val_ppl is not None else "?",
            }
            if speech_loss_val is not None:
                postfix["speech_loss"] = f"{speech_loss_val:.4f}"
            if epoch == weight_prune_epoch and not gradual_sparsify_done:
                postfix["prune"] = f"{current_sparsity*100:.1f}%"
            pbar.set_postfix(**postfix)

            if use_wandb:
                log_dict = {
                    "train/text_loss": text_loss_val,
                    "train/grad_norm": float(grad_norm[0]),
                    "train/adam_lr": float(adam_schedule(global_step)),
                    "train/muon_lr": float(muon_schedule(global_step)),
                    "train/tokens_per_sec": tokens_per_batch / dt,
                    "train/step": global_step,
                }
                if speech_loss_val is not None:
                    log_dict["train/speech_loss"] = speech_loss_val
                if epoch == weight_prune_epoch and not gradual_sparsify_done:
                    log_dict["train/scheduled_sparsity"] = current_sparsity
                if global_step % eval_every == 0 or global_step == total_steps:
                    log_dict["val/text_ppl"] = last_val_ppl
                wandb.log(log_dict)

        # --- Lock mask after epoch 1 gradual sparsification ---
        if epoch == weight_prune_epoch and not gradual_sparsify_done:
            gradual_sparsify_done = True
            if prune_mask is None:
                # Edge case: epoch had fewer steps than t_start
                ema_unr = jax_utils.unreplicate(ema_params)
                mask = _make_prune_mask(ema_unr, sparsity_ratio, _GROUP_SIZE)
                del ema_unr
                prune_mask = jax_utils.replicate(mask)
                del mask
            # Apply final mask to both params and EMA (EMA lags due to slow decay)
            state = state.replace(
                params=jax.tree.map(lambda w, m: w * m, state.params, prune_mask))
            ema_params = jax.tree.map(lambda w, m: w * m, ema_params, prune_mask)
            final_pruned = jax.tree.map(np.array, jax_utils.unreplicate(ema_params))
            total_p = sum(x.size for x in jax.tree.leaves(final_pruned))
            zero_p = sum(int(np.sum(np.abs(x) < 1e-6)) for x in jax.tree.leaves(final_pruned))
            print(f"\n  Gradual sparsification complete — mask locked.")
            print(f"  Final sparsity: {zero_p/total_p*100:.2f}% ({zero_p:,}/{total_p:,} near-zero)")
            del final_pruned

        epoch_avg_loss = sum(text_losses) / len(text_losses) if text_losses else float("nan")
        final_loss = text_losses[-1] if text_losses else float("nan")
        final_ppl = math.exp(final_loss) if not math.isnan(final_loss) else float("nan")

        eval_params = jax_utils.unreplicate(ema_params)
        val_causal = make_causal_mask(args.max_dec_len)
        total_loss, total_toks = 0.0, 0.0
        for vb in get_batches(val_enc, val_dec_in, val_dec_tgt, args.batch_size, shuffle=False):
            vl, vt = val_loss_fn(eval_params, vb[0], vb[1], vb[2], val_causal)
            total_loss += float(vl)
            total_toks += float(vt)
        last_val_ppl = float(math.exp(total_loss / max(total_toks, 1)))

        speech_val_ppl = None
        if speech_vl_fn is not None and val_speech_mel is not None:
            sp_total_loss, sp_total_toks = 0.0, 0.0
            for sp_batch in get_speech_batches(val_speech_mel, val_speech_dec_in, val_speech_dec_tgt, args.batch_size, shuffle=False):
                vl, vt = speech_vl_fn(eval_params, sp_batch[0], sp_batch[1], sp_batch[2], val_causal)
                sp_total_loss += float(vl)
                sp_total_toks += float(vt)
            speech_val_loss = sp_total_loss / max(sp_total_toks, 1)
            speech_val_ppl = float(math.exp(min(speech_val_loss, 20)))

        q_params = _quantize_params(eval_params, group_size=_GROUP_SIZE)
        q_total_loss, q_total_toks = 0.0, 0.0
        for vb in get_batches(val_enc, val_dec_in, val_dec_tgt, args.batch_size, shuffle=False):
            vl, vt = val_loss_fn(q_params, vb[0], vb[1], vb[2], val_causal)
            q_total_loss += float(vl)
            q_total_toks += float(vt)
        quant_val_ppl = float(math.exp(q_total_loss / max(q_total_toks, 1)))
        del q_params  # free device memory

        # Matryoshka per-factor evaluation (reuse val_loss_fn's apply_fn)
        mat_results = {}
        if _MAT_FACTORS:
            apply_fn = jax_utils.unreplicate(state).apply_fn
            for factor, ff_w in zip(_MAT_FACTORS, _MAT_FF_WIDTHS):
                mat_vl_fn = _make_mat_val_loss_fn(apply_fn, ff_w)
                mat_total_loss, mat_total_toks = 0.0, 0.0
                for vb in get_batches(val_enc, val_dec_in, val_dec_tgt, args.batch_size, shuffle=False):
                    vl, vt = mat_vl_fn(eval_params, vb[0], vb[1], vb[2], val_causal)
                    mat_total_loss += float(vl)
                    mat_total_toks += float(vt)
                avg_loss = mat_total_loss / max(mat_total_toks, 1)
                mat_ppl = float(math.exp(min(avg_loss, 20)))
                mat_params = _estimate_mat_params(config, factor)
                mat_results[factor] = (mat_ppl, mat_params, ff_w)
            del apply_fn

        # Sparsity + checkpoint — move to CPU, reuse single copy
        params_np = jax.tree.map(np.array, eval_params)
        del eval_params  # free device memory
        total_params = sum(x.size for x in jax.tree.leaves(params_np))
        near_zero = sum(int(np.sum(np.abs(x) < 1e-6)) for x in jax.tree.leaves(params_np))
        sparsity = near_zero / total_params * 100

        ckpt_name = f"needle_{args.num_layers}_{args.d_model}_{global_step}.pkl"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)

        with open(ckpt_path, "wb") as f:
            pickle.dump({"params": params_np, "config": config.__dict__}, f)

        from .test import measure_throughput, benchmark_generation_quality, compute_wer
        from .run import generate, transcribe
        from .data import _decode_audio
        eval_params_jnp = jax.tree.map(jnp.array, params_np)
        del params_np  # free CPU memory
        model = EncoderDecoderTransformer(config)
        tp = measure_throughput(model, eval_params_jnp, tokenizer, num_runs=5)
        prompts = ["Once upon a time", "The little dog", "She was very happy because"]
        quality = benchmark_generation_quality(model, eval_params_jnp, tokenizer, prompts, max_gen_len=64, temperature=0.8)

        speech_wer = None
        transcribe_samples = []
        if not no_speech and val_speech_mel is not None:
            speech_val_ds = load_librispeech("validation.clean", max_samples=3)
            hyps, refs = [], []
            for i, example in enumerate(speech_val_ds):
                audio, sr = _decode_audio(example["audio"])
                ref_text = example["text"].strip().lower()
                hyp_text = transcribe(
                    model, eval_params_jnp, tokenizer, audio, sr=sr,
                    max_gen_len=64, temperature=0.0, seed=i, stream=False,
                ).strip().lower()
                hyps.append(hyp_text)
                refs.append(ref_text)
                transcribe_samples.append((ref_text, hyp_text))
            speech_wer = compute_wer(hyps, refs)

        del eval_params_jnp  # free device memory

        print(f"\n  ─────────────────────────────────────")
        print(f"  Epoch {epoch + 1}/{args.epochs}")
        print(f"  ─────────────────────────────────────")
        print(f"  Avg loss       {epoch_avg_loss:>12.4f}")
        print(f"  Final loss     {final_loss:>12.4f}")
        print(f"  Train ppl      {final_ppl:>12.2f}")
        print(f"  Text val ppl   {last_val_ppl:>12.2f}")
        if speech_wer is not None:
            print(f"  Speech WER     {speech_wer:>12.4f}")
            epoch_avg_speech = sum(speech_losses) / len(speech_losses) if speech_losses else float("nan")
            print(f"  Avg speech loss{epoch_avg_speech:>12.4f}")
        if speech_val_ppl is not None:
            print(f"  Speech val ppl {speech_val_ppl:>12.2f}")
        print(f"  Quant val ppl  {quant_val_ppl:>12.2f}  (INT4 g{_GROUP_SIZE})")
        print(f"  Sparsity       {sparsity:>11.2f}%  ({near_zero:,}/{total_params:,})")
        if mat_results:
            print(f"  ─────────────────────────────────────")
            print(f"  Matryoshka exportable sub-models (FFN interior slicing):")
            print(f"  {'factor':>6}  {'d_ff':>6}  {'val ppl':>10}  {'params':>12}")
            print(f"  {'1x':>6}  {config.d_ff:>6}  {last_val_ppl:>10.2f}  {total_params:>12,}  (full)")
            for factor in sorted(mat_results.keys()):
                mat_ppl, mat_params, ff_w = mat_results[factor]
                print(f"  {str(factor)+'x':>6}  {ff_w:>6}  {mat_ppl:>10.2f}  {mat_params:>12,}")
        print(f"  ─────────────────────────────────────")
        print(f"  Throughput     {tp['tokens_per_second']:>10.1f} tok/s")
        print(f"  Latency        {tp['avg_latency_s']:>11.3f}s")
        print(f"  Gen length     {quality['avg_generation_length']:>10.1f} tok")
        print(f"  Repetition     {quality['bigram_repetition_rate']:>12.3f}")
        print(f"  ─────────────────────────────────────")
        for prompt, gen in quality["generations"]:
            gen_clean = gen.replace("<story>", "").replace("<transcribe>", "").strip()
            print(f"  <story>[{prompt}] {gen_clean[:80]}")
        if transcribe_samples:
            print(f"  ─────────────────────────────────────")
            print(f"  Transcribe (WER={speech_wer:.4f}):")
            for ref, hyp in transcribe_samples:
                print(f"  <transcribe> ref: {ref[:80]}")
                print(f"               hyp: {hyp[:80]}")
        print(f"  ─────────────────────────────────────")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  ─────────────────────────────────────\n")

        if use_wandb:
            log_dict = {
                "epoch/avg_text_loss": epoch_avg_loss,
                "epoch/final_text_loss": final_loss,
                "epoch/text_val_ppl": last_val_ppl,
                "epoch/quant_val_ppl": quant_val_ppl,
                "epoch/weight_sparsity": sparsity,
                "epoch": epoch + 1,
            }
            if speech_wer is not None:
                log_dict["epoch/speech_wer"] = speech_wer
            if speech_losses:
                log_dict["epoch/avg_speech_loss"] = sum(speech_losses) / len(speech_losses)
            if speech_val_ppl is not None:
                log_dict["epoch/speech_val_ppl"] = speech_val_ppl
            for factor, (mat_ppl, mat_params, _) in mat_results.items():
                log_dict[f"epoch/mat_ppl_{factor}x"] = mat_ppl
                log_dict[f"epoch/mat_params_{factor}x"] = mat_params
            wandb.log(log_dict)

    if use_wandb:
        wandb.finish()
    print("\nTraining complete.")


