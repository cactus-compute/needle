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
    get_batches, get_tokenizer, get_speech_batches,
    build_audio_augmenter,
    load_prepared_data, load_prepared_mels, load_tool_calls,
    PrefetchIterator, count_batches,
    get_contrastive_batches,
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


def _wsd_schedule(peak_value, total_steps, warmup_steps, decay_ratio=0.40):
    """Warmup-Stable-Decay schedule: linear warmup, hold peak, cosine decay."""
    decay_steps = max(1, int(total_steps * decay_ratio))
    stable_steps = total_steps - warmup_steps - decay_steps
    return optax.join_schedules(
        [
            optax.linear_schedule(0.0, peak_value, warmup_steps),
            optax.constant_schedule(peak_value),
            optax.cosine_decay_schedule(peak_value, decay_steps, alpha=0.05),
        ],
        boundaries=[warmup_steps, warmup_steps + stable_steps],
    )


def create_train_state(rng, config, learning_rate, muon_lr, total_steps, warmup_steps, decay_ratio=0.40):
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

    adam_schedule = _wsd_schedule(learning_rate, total_steps, warmup_steps, decay_ratio)
    muon_schedule = _wsd_schedule(muon_lr, total_steps, warmup_steps, decay_ratio)

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
_MAT_FF_WIDTHS = ()
_D_FF = 2048
_CONTRASTIVE_WEIGHT = 0.1


def _clip_contrastive_loss(q_emb, t_emb, log_temp):
    """CLIP-style symmetric contrastive loss with learnable temperature."""
    temp = jnp.exp(jnp.clip(log_temp, -jnp.log(100.0), jnp.log(100.0)))
    logits = q_emb @ t_emb.T / temp  # (B, B)
    B = logits.shape[0]
    labels = jnp.arange(B)
    loss_q = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss_t = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels)
    return (jnp.mean(loss_q) + jnp.mean(loss_t)) / 2.0


def _text_loss_fn(state, params, src, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask):
    pad_id = 0
    q_params = _quantize_params(params, group_size=_GROUP_SIZE)
    src_mask = make_padding_mask(src, pad_id)
    tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
    logits, slot_div = state.apply_fn(
        {"params": q_params},
        src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask,
        ffn_mask=ffn_mask,
        deterministic=False,
        method="forward_masked",
        rngs={"dropout": rng},
    )
    logits_f32 = logits.astype(jnp.float32)
    mask = loss_mask
    ce_loss = jnp.sum(
        optax.softmax_cross_entropy_with_integer_labels(logits_f32, tgt_out) * mask
    ) / jnp.maximum(jnp.sum(mask), 1.0)
    z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits_f32, axis=-1) ** 2)
    return ce_loss + z_loss


def _speech_loss_fn(state, params, mel, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask):
    pad_id = 0
    q_params = _quantize_params(params, group_size=_GROUP_SIZE)
    src_mask = make_mel_padding_mask(mel)
    tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
    spec_rng, drop_rng = jax.random.split(rng)
    logits, _ = state.apply_fn(
        {"params": q_params},
        mel, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask,
        ffn_mask=ffn_mask,
        deterministic=False,
        method="forward_speech_masked",
        rngs={"specaugment": spec_rng, "dropout": drop_rng},
    )
    logits_f32 = logits.astype(jnp.float32)
    mask = loss_mask
    ce_loss = jnp.sum(
        optax.softmax_cross_entropy_with_integer_labels(logits_f32, tgt_out) * mask
    ) / jnp.maximum(jnp.sum(mask), 1.0)
    z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits_f32, axis=-1) ** 2)
    return ce_loss + z_loss


def _contrastive_loss_fn(state, params, query_tokens, tool_tokens, rng):
    """Compute CLIP contrastive loss on query/tool pairs."""
    q_params = _quantize_params(params, group_size=_GROUP_SIZE)
    q_emb, t_emb, log_temp = state.apply_fn(
        {"params": q_params},
        query_tokens, tool_tokens,
        deterministic=False,
        method="forward_contrastive",
        rngs={"dropout": rng},
    )
    return _clip_contrastive_loss(q_emb, t_emb, log_temp)



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


def _train_step_text(state, ema_params, src, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask):
    ema_decay = 0.999
    loss, grads = jax.value_and_grad(
        lambda p: _text_loss_fn(state, p, src, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask)
    )(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    ema_params = jax.tree.map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema_params, state.params)
    return state, ema_params, loss, grad_norm


def _train_step_text_masked(state, ema_params, src, tgt_in, tgt_out, causal_mask, prune_mask, ffn_mask, rng, loss_mask):
    """Text training step with fused prune mask application."""
    ema_decay = 0.999
    loss, grads = jax.value_and_grad(
        lambda p: _text_loss_fn(state, p, src, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask)
    )(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    masked_params = jax.tree.map(lambda w, m: w * m, state.params, prune_mask)
    state = state.replace(params=masked_params)
    ema_params = jax.tree.map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema_params, masked_params)
    return state, ema_params, loss, grad_norm


def _train_step_speech(state, ema_params, mel, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask):
    ema_decay = 0.999
    loss, grads = jax.value_and_grad(
        lambda p: _speech_loss_fn(state, p, mel, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask)
    )(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    ema_params = jax.tree.map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema_params, state.params)
    return state, ema_params, loss, grad_norm


def _train_step_speech_masked(state, ema_params, mel, tgt_in, tgt_out, causal_mask, prune_mask, ffn_mask, rng, loss_mask):
    """Speech training step with fused prune mask application."""
    ema_decay = 0.999
    loss, grads = jax.value_and_grad(
        lambda p: _speech_loss_fn(state, p, mel, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask)
    )(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    masked_params = jax.tree.map(lambda w, m: w * m, state.params, prune_mask)
    state = state.replace(params=masked_params)
    ema_params = jax.tree.map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema_params, masked_params)
    return state, ema_params, loss, grad_norm


def _train_step_contrastive(state, ema_params, query_tokens, tool_tokens, rng):
    """Separate contrastive training step using CLIP loss."""
    ema_decay = 0.999
    cl_loss, grads = jax.value_and_grad(
        lambda p: _contrastive_loss_fn(state, p, query_tokens, tool_tokens, rng)
    )(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    cl_loss = jax.lax.pmean(cl_loss, axis_name="batch")
    grads = jax.tree.map(lambda g: g * _CONTRASTIVE_WEIGHT, grads)
    state = state.apply_gradients(grads=grads)
    ema_params = jax.tree.map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema_params, state.params)
    return state, ema_params, cl_loss




def _make_p_train_step():
    return jax.pmap(_train_step_text, axis_name="batch", donate_argnums=(0, 1))


def _make_p_train_step_masked():
    return jax.pmap(_train_step_text_masked, axis_name="batch", donate_argnums=(0, 1))


def _make_p_train_step_speech():
    return jax.pmap(_train_step_speech, axis_name="batch", donate_argnums=(0, 1))


def _make_p_train_step_speech_masked():
    return jax.pmap(_train_step_speech_masked, axis_name="batch", donate_argnums=(0, 1))


def _make_p_train_step_contrastive():
    return jax.pmap(_train_step_contrastive, axis_name="batch", donate_argnums=(0, 1))


def _make_val_loss_fn(apply_fn):
    @jax.jit
    def val_loss_batch(params, src, tgt_in, tgt_out, causal_mask, loss_mask):
        pad_id = 0
        src_mask = make_padding_mask(src, pad_id)
        tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
        logits = apply_fn(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tgt_out)
        return jnp.sum(loss * loss_mask), jnp.sum(loss_mask)
    return val_loss_batch


def _make_mat_val_loss_fn(apply_fn, ff_width):
    """Val loss for matryoshka sub-model at given FFN width."""
    @jax.jit
    def val_loss_batch(params, src, tgt_in, tgt_out, causal_mask, loss_mask):
        pad_id = 0
        src_mask = make_padding_mask(src, pad_id)
        tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
        logits, _, mat_logits = apply_fn(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask,
            mat_ff_widths=(ff_width,),
            method="forward_with_aux",
        )
        trunc_logits = mat_logits[0].astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(trunc_logits, tgt_out)
        return jnp.sum(loss * loss_mask), jnp.sum(loss_mask)
    return val_loss_batch


def _make_speech_val_loss_fn(apply_fn):
    @jax.jit
    def val_loss_batch(params, mel, tgt_in, tgt_out, causal_mask, loss_mask):
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
        return jnp.sum(loss * loss_mask), jnp.sum(loss_mask)
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
    d_ff = config.d_ff // matryoshka_factor

    emb = v * d
    attn = d * d + d * kv_dim * 2 + d * d
    ffn = d * d_ff * 3
    # Encoder: self-attn + FFN per block, plus downsample conv params
    downsample = d * 4 + d * d  # depthwise conv (kernel=4) + pointwise Dense
    enc_block = attn + ffn
    dec_block = attn * 2 + ffn  # self-attn + cross-attn + FFN
    total = emb + downsample + n_enc * enc_block + n_dec * dec_block
    return int(total)


def shard_batch(batch, num_devices):
    """Reshape a batch array so leading dim is (num_devices, per_device_batch, ...)."""
    return batch.reshape(num_devices, -1, *batch.shape[1:])


def train(args):
    num_devices = jax.local_device_count()
    no_speech = getattr(args, "no_speech", False)
    n_mels = getattr(args, "n_mels", 80)
    max_mel_len = getattr(args, "max_mel_len", 1024)

    use_wandb = getattr(args, "wandb", False)
    if use_wandb:
        import wandb
        if wandb.run is None:
            wandb.init(project="needle-v1", config=vars(args))

    total_data_steps = 4 if not no_speech else 3
    step_idx = 0

    step_idx += 1
    print(f"\n[{step_idx}/{total_data_steps}] Detecting devices...")
    print(f"      {num_devices} device(s) for data-parallel training")

    step_idx += 1
    print(f"\n[{step_idx}/{total_data_steps}] Loading tokenizer...")
    tokenizer = get_tokenizer(max_samples=args.max_samples)

    step_idx += 1
    print(f"\n[{step_idx}/{total_data_steps}] Loading prepared data from disk (mmap)...")
    train_data = load_prepared_data("train", mmap=True)
    val_data = load_prepared_data("val", mmap=True)
    enc_inputs = train_data["enc_inputs"]
    dec_inputs = train_data["dec_inputs"]
    dec_targets = train_data["dec_targets"]
    train_loss_mask = train_data["loss_mask"]
    tool_counts = train_data["tool_counts"]
    val_enc = val_data["enc_inputs"]
    val_dec_in = val_data["dec_inputs"]
    val_dec_tgt = val_data["dec_targets"]
    val_loss_mask = val_data["loss_mask"]
    print(f"      {len(enc_inputs):,} train / {len(val_enc):,} val tool-call pairs (memory-mapped)")

    train_mels = None
    val_mels = None
    speech_augmenter = None
    if not no_speech:
        step_idx += 1
        print(f"\n[{step_idx}/{total_data_steps}] Loading precomputed mel spectrograms (mmap)...")
        train_mels = load_prepared_mels(train_data["mel_cache_id"], mmap=True)
        val_mels = load_prepared_mels(val_data["mel_cache_id"], mmap=True)
        print(f"      {len(train_mels):,} train / {len(val_mels):,} val mel spectrograms (memory-mapped)")

        speech_augmenter = build_audio_augmenter(
            sr=16000,
            mode=getattr(args, "audio_aug_mode", "white"),
            white_noise_p=getattr(args, "white_noise_p", 0.5),
            white_noise_min_snr_db=getattr(args, "white_noise_min_snr_db", 8.0),
            white_noise_max_snr_db=getattr(args, "white_noise_max_snr_db", 30.0),
            person_noise_n=getattr(args, "person_noise_n", 10),
            person_noise_r1=getattr(args, "person_noise_r1", 3.0),
            person_noise_r2=getattr(args, "person_noise_r2", 10.0),
            person_noise_r_ref=getattr(args, "person_noise_r_ref", 1.0),
            person_noise_min_snr_db=getattr(args, "person_noise_min_snr_db", 15.0),
            person_noise_max_snr_db=getattr(args, "person_noise_max_snr_db", 40.0),
            person_noise_pool=train_mels,
        )
        if speech_augmenter is not None:
            aug_name = getattr(speech_augmenter, "name", getattr(args, "audio_aug_mode", "white"))
            num_transforms = len(getattr(speech_augmenter, "transforms", ()))
            if num_transforms > 0:
                print(f"      Speech augmentation: {aug_name} ({num_transforms} transforms)")
            else:
                print(f"      Speech augmentation: {aug_name}")

    # Contrastive data (optional — graceful if missing)
    cl_query_tokens = train_data.get("query_only")
    cl_tool_tokens = train_data.get("tool_individual")
    cl_tool_ex_idx = train_data.get("tool_ex_idx")
    cl_tool_is_pos = train_data.get("tool_is_pos")
    has_contrastive = all(x is not None for x in [cl_query_tokens, cl_tool_tokens, cl_tool_ex_idx, cl_tool_is_pos])
    if has_contrastive:
        print(f"      Contrastive: {len(cl_query_tokens):,} queries, {len(cl_tool_tokens):,} tools")

    # Contrastive data (optional — graceful if missing)
    cl_query_tokens = train_data.get("query_only")
    cl_tool_tokens = train_data.get("tool_individual")
    cl_tool_ex_idx = train_data.get("tool_ex_idx")
    cl_tool_is_pos = train_data.get("tool_is_pos")
    has_contrastive = all(x is not None for x in [cl_query_tokens, cl_tool_tokens, cl_tool_ex_idx, cl_tool_is_pos])
    if has_contrastive:
        print(f"      Contrastive: {len(cl_query_tokens):,} queries, {len(cl_tool_tokens):,} tools")

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
            dropout_rate=getattr(args, "dropout", 0.1),
            contrastive_dim=getattr(args, "contrastive_dim", 128),
        )

    global _GROUP_SIZE, _MAT_FACTORS, _MAT_FF_WIDTHS, _D_FF, _CONTRASTIVE_WEIGHT
    _GROUP_SIZE = getattr(args, "group_size", 32)
    _CONTRASTIVE_WEIGHT = getattr(args, "contrastive_weight", 0.1)
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
    p_train_step_contrastive = _make_p_train_step_contrastive()

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    unique_batch_size = (effective_batch_size // num_devices) * num_devices
    text_batches_per_epoch = count_batches(len(enc_inputs), unique_batch_size)
    if not no_speech and train_mels is not None:
        speech_batches_per_epoch = text_batches_per_epoch
    else:
        speech_batches_per_epoch = 0
    num_batches = text_batches_per_epoch + speech_batches_per_epoch
    total_steps = num_batches * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    scaled_lr = args.lr * num_devices
    muon_lr = getattr(args, "muon_lr", 0.02) * math.sqrt(num_devices)
    decay_ratio = getattr(args, "decay_ratio", 0.40)
    state = create_train_state(init_rng, config, scaled_lr, muon_lr, total_steps, warmup_steps, decay_ratio)
    val_loss_fn = _make_val_loss_fn(state.apply_fn)
    speech_vl_fn = _make_speech_val_loss_fn(state.apply_fn) if speech_batches_per_epoch > 0 else None

    if resume_checkpoint:
        state = state.replace(params=ckpt_params)
        print(f"  Loaded checkpoint params into train state")

    ema_params = jax.tree.map(jnp.copy, state.params)
    state = jax_utils.replicate(state)
    ema_params = jax_utils.replicate(ema_params)

    param_count = sum(x.size for x in jax.tree.leaves(jax_utils.unreplicate(state).params))
    decay_steps = max(1, int(total_steps * decay_ratio))
    stable_steps = total_steps - warmup_steps - decay_steps
    best_call_f1 = 0.0
    best_ckpt_path = None

    print(f"\n  ─────────────────────────────────────")
    print(f"  Parameters    {param_count:>12,}")
    print(f"  d_model       {config.d_model:>12}")
    print(f"  Heads         {config.num_heads:>7} ({config.num_kv_heads} KV)")
    print(f"  Layers        {config.num_encoder_layers:>7} enc / {config.num_decoder_layers} dec")
    print(f"  Activation    {config.activation:>12}")
    print(f"  Dtype         {config.dtype:>12}")
    print(f"  Dropout       {config.dropout_rate:>12}")
    if speech_batches_per_epoch > 0:
        print(f"  Speech        {speech_batches_per_epoch} batches/epoch")
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

    text_ffn_mask = _make_ffn_mask(args.batch_size, config.d_ff, _MAT_FF_WIDTHS)
    text_ffn_mask = jnp.broadcast_to(
        text_ffn_mask[None, :, :],
        (num_devices, args.batch_size, config.d_ff),
    )
    if n_widths > 1:
        print(f"  Mat factors   {n_widths} (full + {', '.join(str(f)+'x' for f in _MAT_FACTORS)})")
        print(f"  Mat mode      unique input ({args.batch_size}/dev, split by width)")

    adam_schedule = _wsd_schedule(scaled_lr, total_steps, warmup_steps, decay_ratio)
    muon_schedule = _wsd_schedule(muon_lr, total_steps, warmup_steps, decay_ratio)
    tokens_per_batch = effective_batch_size * (args.max_enc_len + args.max_dec_len)

    eval_model = EncoderDecoderTransformer(config)

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
        _curriculum_tc = tool_counts if getattr(args, "curriculum", False) else None
        text_batch_iter = PrefetchIterator(
            lambda: get_batches(enc_inputs, dec_inputs, dec_targets, unique_batch_size,
                                loss_mask=train_loss_mask,
                                tool_counts=_curriculum_tc),
            prefetch=4,
        )

        speech_batch_iter = None
        if speech_batches_per_epoch > 0:
            speech_batch_iter = PrefetchIterator(
                lambda: get_speech_batches(train_mels, dec_inputs, dec_targets, unique_batch_size,
                                           loss_mask=train_loss_mask, augmenter=speech_augmenter),
                prefetch=4,
            )

        cl_batch_iter = None
        if has_contrastive and _CONTRASTIVE_WEIGHT > 0:
            cl_batch_iter = PrefetchIterator(
                lambda: get_contrastive_batches(
                    cl_query_tokens, cl_tool_tokens, cl_tool_ex_idx, cl_tool_is_pos,
                    unique_batch_size),
                prefetch=4,
            )

        steps_this_epoch = text_batches_per_epoch + speech_batches_per_epoch
        text_idx = 0
        speech_idx = 0
        speech_loss_val = None
        pbar = tqdm(range(steps_this_epoch), desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step_i in pbar:
            t0 = time.perf_counter()

            do_speech = (step_i % 2 == 1) and speech_idx < speech_batches_per_epoch
            do_text = not do_speech and text_idx < text_batches_per_epoch
            if not do_speech and not do_text:
                if text_idx < text_batches_per_epoch:
                    do_text = True
                elif speech_idx < speech_batches_per_epoch:
                    do_speech = True
                else:
                    break

            step_grad_norm = None

            if do_text:
                src, tgt_in, tgt_out, lm = next(text_batch_iter)
                text_idx += 1

                cl_q_b = None
                cl_t_b = None
                if cl_batch_iter is not None:
                    try:
                        cl_q, cl_t = next(cl_batch_iter)
                        cl_q_b = shard_batch(cl_q, num_devices)
                        cl_t_b = shard_batch(cl_t, num_devices)
                    except StopIteration:
                        cl_batch_iter = None

                src_b = shard_batch(src, num_devices)
                tgt_in_b = shard_batch(tgt_in, num_devices)
                tgt_out_b = shard_batch(tgt_out, num_devices)
                lm_b = shard_batch(lm, num_devices)

                rng, text_rng = jax.random.split(rng)
                text_rngs = jax.random.split(text_rng, num_devices)

                if prune_mask is not None:
                    state, ema_params, loss, grad_norm = p_train_step_masked(
                        state, ema_params, src_b, tgt_in_b, tgt_out_b, causal_mask, prune_mask, text_ffn_mask, text_rngs, lm_b,
                    )
                else:
                    state, ema_params, loss, grad_norm = p_train_step(
                        state, ema_params, src_b, tgt_in_b, tgt_out_b, causal_mask, text_ffn_mask, text_rngs, lm_b,
                    )

                if cl_q_b is not None and cl_t_b is not None:
                    rng, cl_rng = jax.random.split(rng)
                    cl_rngs = jax.random.split(cl_rng, num_devices)
                    state, ema_params, cl_loss = p_train_step_contrastive(
                        state, ema_params, cl_q_b, cl_t_b, cl_rngs,
                    )

                text_loss_val = float(loss[0])
                text_losses.append(text_loss_val)
                step_grad_norm = float(grad_norm[0])
                global_step += 1
            else:
                mel_batch, sp_tgt_in, sp_tgt_out, sp_lm = next(speech_batch_iter)
                speech_idx += 1

                mel_b = shard_batch(mel_batch, num_devices)
                sp_tgt_in_b = shard_batch(sp_tgt_in, num_devices)
                sp_tgt_out_b = shard_batch(sp_tgt_out, num_devices)
                sp_lm_b = shard_batch(sp_lm, num_devices)

                rng, spec_rng = jax.random.split(rng)
                spec_rngs = jax.random.split(spec_rng, num_devices)

                if prune_mask is not None:
                    state, ema_params, sp_loss, sp_grad_norm = p_train_step_speech_masked(
                        state, ema_params, mel_b, sp_tgt_in_b, sp_tgt_out_b, causal_mask, prune_mask, text_ffn_mask, spec_rngs, sp_lm_b,
                    )
                else:
                    state, ema_params, sp_loss, sp_grad_norm = p_train_step_speech(
                        state, ema_params, mel_b, sp_tgt_in_b, sp_tgt_out_b, causal_mask, text_ffn_mask, spec_rngs, sp_lm_b,
                    )

                speech_loss_val = float(sp_loss[0])
                speech_losses.append(speech_loss_val)
                step_grad_norm = float(sp_grad_norm[0])
                text_loss_val = text_losses[-1] if text_losses else float("nan")
                global_step += 1

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
                for vb in get_batches(val_enc, val_dec_in, val_dec_tgt, args.batch_size, shuffle=False, loss_mask=val_loss_mask):
                    vl, vt = val_loss_fn(_eval_params, vb[0], vb[1], vb[2], val_causal, vb[3])
                    total_loss += float(vl)
                    total_toks += float(vt)
                last_val_ppl = float(math.exp(min(total_loss / max(total_toks, 1), 20)))

                del _eval_params

            postfix = {
                "speech_loss": f"{speech_loss_val:.4f}" if speech_loss_val is not None else "-",
                "text_loss": f"{text_loss_val:.4f}",
                "text_ppl": f"{last_val_ppl:.2f}" if last_val_ppl is not None else "?",
            }
            if sparsity_ratio > 0:
                if epoch == weight_prune_epoch and not gradual_sparsify_done:
                    postfix["sparsification"] = f"{current_sparsity*100:.1f}%"
                else:
                    postfix["sparsification"] = "done"
            pbar.set_postfix(**postfix)

            if use_wandb:
                log_dict = {
                    "train/text_loss": text_loss_val,
                    "train/grad_norm": step_grad_norm,
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

        text_batch_iter.close()
        if speech_batch_iter is not None:
            speech_batch_iter.close()
        if cl_batch_iter is not None:
            cl_batch_iter.close()

        if epoch == weight_prune_epoch and not gradual_sparsify_done:
            gradual_sparsify_done = True
            if prune_mask is None:
                ema_unr = jax_utils.unreplicate(ema_params)
                mask = _make_prune_mask(ema_unr, sparsity_ratio, _GROUP_SIZE)
                del ema_unr
                prune_mask = jax_utils.replicate(mask)
                del mask
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
        final_ppl = math.exp(min(final_loss, 20)) if not math.isnan(final_loss) else float("nan")

        eval_params = jax_utils.unreplicate(ema_params)
        val_causal = make_causal_mask(args.max_dec_len)

        q_params = _quantize_params(eval_params, group_size=_GROUP_SIZE)
        mat_vl_fns = {}
        if _MAT_FACTORS:
            _apply_fn = jax_utils.unreplicate(state).apply_fn
            mat_vl_fns = {f: _make_mat_val_loss_fn(_apply_fn, fw)
                          for f, fw in zip(_MAT_FACTORS, _MAT_FF_WIDTHS)}
            del _apply_fn

        full_loss, full_toks = 0.0, 0.0
        q_loss, q_toks = 0.0, 0.0
        mat_accum = {f: [0.0, 0.0] for f in _MAT_FACTORS}

        for vb in get_batches(val_enc, val_dec_in, val_dec_tgt, args.batch_size,
                              shuffle=False, loss_mask=val_loss_mask):
            src, dec_in, dec_tgt, lm = vb[0], vb[1], vb[2], vb[3]
            vl, vt = val_loss_fn(eval_params, src, dec_in, dec_tgt, val_causal, lm)
            full_loss += float(vl); full_toks += float(vt)
            vl, vt = val_loss_fn(q_params, src, dec_in, dec_tgt, val_causal, lm)
            q_loss += float(vl); q_toks += float(vt)
            for f, fn in mat_vl_fns.items():
                vl, vt = fn(eval_params, src, dec_in, dec_tgt, val_causal, lm)
                mat_accum[f][0] += float(vl)
                mat_accum[f][1] += float(vt)

        last_val_ppl = float(math.exp(min(full_loss / max(full_toks, 1), 20)))
        quant_val_ppl = float(math.exp(min(q_loss / max(q_toks, 1), 20)))
        del q_params

        mat_results = {}
        for f in _MAT_FACTORS:
            avg = mat_accum[f][0] / max(mat_accum[f][1], 1)
            mat_results[f] = (float(math.exp(min(avg, 20))),
                              _estimate_mat_params(config, f), config.d_ff // f)

        speech_val_ppl = None
        if speech_vl_fn is not None and val_mels is not None:
            sp_total_loss, sp_total_toks = 0.0, 0.0
            for sp_batch in get_speech_batches(val_mels, val_dec_in, val_dec_tgt, args.batch_size,
                                               shuffle=False, loss_mask=val_loss_mask):
                vl, vt = speech_vl_fn(eval_params, sp_batch[0], sp_batch[1], sp_batch[2], val_causal, sp_batch[3])
                sp_total_loss += float(vl)
                sp_total_toks += float(vt)
            speech_val_loss = sp_total_loss / max(sp_total_toks, 1)
            speech_val_ppl = float(math.exp(min(speech_val_loss, 20)))

        params_np = jax.tree.map(np.array, eval_params)
        total_params = sum(x.size for x in jax.tree.leaves(params_np))
        near_zero = sum(int(np.sum(np.abs(x) < 1e-6)) for x in jax.tree.leaves(params_np))
        sparsity = near_zero / total_params * 100

        ckpt_name = f"needle_{args.num_layers}_{args.d_model}_{global_step}.pkl"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        with open(ckpt_path, "wb") as f:
            pickle.dump({"params": params_np, "config": config.__dict__}, f)
        del params_np

        from .eval import measure_throughput
        from .run import generate_batch
        tp = measure_throughput(eval_model, eval_params, tokenizer, num_runs=5)

        val_kept = val_data["kept_indices"]
        val_ds = load_tool_calls("val", max_samples=args.max_samples)

        # Pick 5 display samples (shuffled for diversity): 4 with tool calls, 1 without
        sample_rng = np.random.RandomState(epoch + 7)
        sample_pool = sample_rng.permutation(len(val_kept))
        display_with, display_without = [], []
        for k in sample_pool:
            if len(display_with) >= 4 and len(display_without) >= 1:
                break
            local_idx = int(val_kept[k])
            ex = val_ds[local_idx]
            is_empty = ex["answers"].strip() in ("", "[]")
            if not is_empty and len(display_with) < 4:
                display_with.append(ex)
            elif is_empty and len(display_without) < 1:
                display_without.append(ex)
        display_pairs = display_with + display_without

        # Collect all eval examples, then batch-generate everything at once
        import json as _json_mod
        tc_with_n = 85
        tc_without_n = 15
        tc_rng = np.random.RandomState(epoch + 42)
        tc_pool = tc_rng.permutation(len(val_kept))
        tc_with, tc_without = [], []
        for k in tc_pool:
            if len(tc_with) >= tc_with_n and len(tc_without) >= tc_without_n:
                break
            local_idx = int(val_kept[k])
            ex = val_ds[local_idx]
            ref_text = ex["answers"].strip()
            is_empty = ref_text in ("", "[]")
            if not is_empty and len(tc_with) < tc_with_n:
                tc_with.append(ex)
            elif is_empty and len(tc_without) < tc_without_n:
                tc_without.append(ex)
        tc_eval_pairs = tc_with + tc_without

        # Batch-generate: display samples + tc eval samples in one pass
        all_eval_examples = display_pairs + tc_eval_pairs
        eval_gen_len = min(args.max_dec_len, 512)
        all_preds = generate_batch(
            eval_model, eval_params, tokenizer,
            [ex["query"] for ex in all_eval_examples],
            [ex["tools"] for ex in all_eval_examples],
            max_gen_len=eval_gen_len,
            max_enc_len=args.max_enc_len,
        )

        display_preds = all_preds[:len(display_pairs)]
        tc_preds = all_preds[len(display_pairs):]

        unified_samples = []
        for ex, pred in zip(display_pairs, display_preds):
            unified_samples.append({
                "query": ex["query"],
                "tools": ex["tools"],
                "ref": ex["answers"],
                "text": pred.strip(),
            })

        tc_n, tc_exact, tc_name_tp, tc_name_fp, tc_name_fn = 0, 0, 0, 0, 0
        tc_call_tp, tc_call_fp, tc_call_fn, tc_parse_err = 0, 0, 0, 0
        tc_args_correct, tc_args_total = 0, 0
        tc_halluc_params, tc_total_pred_params = 0, 0
        tc_missing_params, tc_total_ref_params = 0, 0
        tc_correct_values, tc_matched_params = 0, 0

        def _call_key(c):
            if not isinstance(c, dict): return None
            return _json_mod.dumps({"name": c.get("name"), "arguments": c.get("arguments")}, sort_keys=True)

        for ex, pred_text in zip(tc_eval_pairs, tc_preds):
            ref_text = ex["answers"].strip()
            pred_text = pred_text.strip()
            try:
                ref_calls = _json_mod.loads(ref_text)
            except (ValueError, TypeError):
                ref_calls = []
            try:
                pred_calls = _json_mod.loads(pred_text)
                if not isinstance(pred_calls, list):
                    pred_calls = [pred_calls] if isinstance(pred_calls, dict) else []
            except (ValueError, TypeError):
                tc_parse_err += 1
                pred_calls = []
            tc_n += 1
            if _json_mod.dumps(pred_calls, sort_keys=True) == _json_mod.dumps(ref_calls, sort_keys=True):
                tc_exact += 1
            ref_names = {c["name"] for c in ref_calls if isinstance(c, dict) and "name" in c}
            pred_names = {c["name"] for c in pred_calls if isinstance(c, dict) and "name" in c}
            tc_name_tp += len(pred_names & ref_names)
            tc_name_fp += len(pred_names - ref_names)
            tc_name_fn += len(ref_names - pred_names)
            rk = {_call_key(c) for c in ref_calls} - {None}
            pk = {_call_key(c) for c in pred_calls} - {None}
            tc_call_tp += len(pk & rk)
            tc_call_fp += len(pk - rk)
            tc_call_fn += len(rk - pk)

            # Argument correctness: for name-matched calls, check if arguments match
            ref_by_name = {}
            for c in ref_calls:
                if isinstance(c, dict) and "name" in c:
                    ref_by_name.setdefault(c["name"], []).append(c.get("arguments", {}))
            for c in pred_calls:
                if isinstance(c, dict) and "name" in c and c["name"] in ref_by_name:
                    tc_args_total += 1
                    pred_args = _json_mod.dumps(c.get("arguments", {}), sort_keys=True)
                    if any(pred_args == _json_mod.dumps(ra, sort_keys=True) for ra in ref_by_name[c["name"]]):
                        tc_args_correct += 1

            # Parameter-level metrics
            try:
                tool_defs = _json_mod.loads(ex["tools"])
                tool_param_map = {t["name"]: set((t.get("parameters") or {}).keys()) for t in tool_defs if isinstance(t, dict) and "name" in t}
            except (ValueError, TypeError):
                tool_param_map = {}
            for c in pred_calls:
                if not isinstance(c, dict) or "name" not in c:
                    continue
                cname = c["name"]
                if cname not in tool_param_map:
                    continue
                schema_keys = tool_param_map[cname]
                pred_keys = set((c.get("arguments") or {}).keys())
                tc_total_pred_params += len(pred_keys)
                tc_halluc_params += len(pred_keys - schema_keys)
                # Find matching reference call for missing/value metrics
                if cname in ref_by_name:
                    ref_args = ref_by_name[cname][0]
                    ref_keys = set((ref_args if isinstance(ref_args, dict) else {}).keys())
                    tc_total_ref_params += len(ref_keys)
                    tc_missing_params += len(ref_keys - pred_keys)
                    matched_keys = pred_keys & ref_keys
                    tc_matched_params += len(matched_keys)
                    for k in matched_keys:
                        if _json_mod.dumps(c.get("arguments", {})[k], sort_keys=True) == _json_mod.dumps(ref_args[k], sort_keys=True):
                            tc_correct_values += 1

        tc_metrics = {}
        if tc_n > 0:
            tc_metrics["parse_rate"] = 1.0 - tc_parse_err / tc_n
            tc_metrics["exact_match"] = tc_exact / tc_n
            np_ = tc_name_tp + tc_name_fp
            nr_ = tc_name_tp + tc_name_fn
            tc_metrics["name_f1"] = 2 * tc_name_tp / max(np_ + nr_, 1)
            cp_ = tc_call_tp + tc_call_fp
            cr_ = tc_call_tp + tc_call_fn
            tc_metrics["call_f1"] = 2 * tc_call_tp / max(cp_ + cr_, 1)
            tc_metrics["args_acc"] = tc_args_correct / max(tc_args_total, 1)
            tc_metrics["param_haluc"] = tc_halluc_params / max(tc_total_pred_params, 1)
            tc_metrics["param_miss"] = tc_missing_params / max(tc_total_ref_params, 1)
            tc_metrics["value_acc"] = tc_correct_values / max(tc_matched_params, 1)

        # Save best checkpoint based on call_f1
        if tc_metrics and tc_metrics["call_f1"] > best_call_f1:
            best_call_f1 = tc_metrics["call_f1"]
            best_ckpt_path = os.path.join(args.checkpoint_dir, f"needle_{args.num_layers}_{args.d_model}_best.pkl")
            params_best = jax.tree.map(np.array, jax_utils.unreplicate(ema_params))
            with open(best_ckpt_path, "wb") as f:
                pickle.dump({"params": params_best, "config": config.__dict__}, f)
            del params_best
            print(f"  ** New best call_f1={best_call_f1:.1%} → {best_ckpt_path}")

        # Contrastive retrieval eval
        retrieval_metrics = None
        if has_contrastive and _CONTRASTIVE_WEIGHT > 0:
            from .eval import benchmark_retrieval
            retrieval_metrics = benchmark_retrieval(
                eval_model, eval_params, tokenizer,
                num_samples=min(500, getattr(args, "max_eval_samples", 500)),
            )

        del eval_params
        final_speech_loss = speech_losses[-1] if speech_losses else None

        print(f"\n  ─────────────────────────────────────")
        print(f"  Epoch {epoch + 1}/{args.epochs}")
        print(f"  ─────────────────────────────────────")
        print(f"  Text loss      {final_loss:>12.4f}")
        print(f"  Text val ppl   {last_val_ppl:>12.2f}")
        if final_speech_loss is not None:
            print(f"  Speech loss    {final_speech_loss:>12.4f}")
        if speech_val_ppl is not None:
            print(f"  Speech val ppl {speech_val_ppl:>12.2f}")
        print(f"  Quant val ppl  {quant_val_ppl:>12.2f}  (INT4 g{_GROUP_SIZE})")
        print(f"  Sparsity       {sparsity:>11.2f}%  ({near_zero:,}/{total_params:,})")
        if mat_results:
            print(f"  ─────────────────────────────────────")
            print(f"  Matryoshka sub-models:")
            print(f"  {'factor':>6}  {'d_ff':>6}  {'val ppl':>10}  {'params':>12}")
            print(f"  {'1x':>6}  {config.d_ff:>6}  {last_val_ppl:>10.2f}  {total_params:>12,}  (full)")
            for factor in sorted(mat_results.keys()):
                mat_ppl, mat_params, ff_w = mat_results[factor]
                print(f"  {str(factor)+'x':>6}  {ff_w:>6}  {mat_ppl:>10.2f}  {mat_params:>12,}")
        if tc_metrics:
            print(f"  ─── Tool-Call Accuracy ({tc_n} samples) ──")
            print(f"  JSON parse     {tc_metrics['parse_rate']:>10.1%}")
            print(f"  Name F1        {tc_metrics['name_f1']:>10.1%}")
            print(f"  Param haluc    {tc_metrics['param_haluc']:>10.1%}")
            print(f"  Param miss     {tc_metrics['param_miss']:>10.1%}")
            print(f"  Value acc      {tc_metrics['value_acc']:>10.1%}")
            print(f"  Args acc       {tc_metrics['args_acc']:>10.1%}")
            print(f"  Call F1        {tc_metrics['call_f1']:>10.1%}")
            print(f"  Exact match    {tc_metrics['exact_match']:>10.1%}")
        if retrieval_metrics and retrieval_metrics["num_queries"] > 0:
            rm = retrieval_metrics
            print(f"  ─── Retrieval ({rm['num_queries']} queries) ─────")
            for k, v in sorted(rm["recall@k"].items()):
                print(f"  Recall@{k:<3}     {v:>10.1%}")
            print(f"  MRR            {rm['mrr']:>10.3f}")
        print(f"  ─────────────────────────────────────")
        print(f"  Throughput     {tp['tokens_per_second']:>10.1f} tok/s")
        print(f"  Latency        {tp['avg_latency_s']:>11.3f}s")
        if unified_samples:
            print(f"  ─── Samples ({len(unified_samples)}) ───────────────────")
            for j, s in enumerate(unified_samples):
                print(f"  [{j+1}] Query: {s['query'][:120]}")
                tools_short = s["tools"][:120]
                if len(s["tools"]) > 120:
                    tools_short += "..."
                print(f"      Tools: {tools_short}")
                print(f"      Ref:   {s['ref'][:200] or '[]'}")
                print(f"      Text:  {s['text'][:200] or '(empty)'}")
                if j < len(unified_samples) - 1:
                    print()
        print(f"  ─────────────────────────────────────")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  ─────────────────────────────────────\n")

        if use_wandb:
            log_dict = {
                "epoch/text_loss": final_loss,
                "epoch/text_val_ppl": last_val_ppl,
                "epoch/quant_val_ppl": quant_val_ppl,
                "epoch/weight_sparsity": sparsity,
                "epoch": epoch + 1,
            }
            if final_speech_loss is not None:
                log_dict["epoch/speech_loss"] = final_speech_loss
            if speech_val_ppl is not None:
                log_dict["epoch/speech_val_ppl"] = speech_val_ppl
            for factor, (mat_ppl, mat_params, _) in mat_results.items():
                log_dict[f"epoch/mat_ppl_{factor}x"] = mat_ppl
                log_dict[f"epoch/mat_params_{factor}x"] = mat_params
            if tc_metrics:
                log_dict["epoch/tc_parse_rate"] = tc_metrics["parse_rate"]
                log_dict["epoch/tc_exact_match"] = tc_metrics["exact_match"]
                log_dict["epoch/tc_name_f1"] = tc_metrics["name_f1"]
                log_dict["epoch/tc_param_haluc"] = tc_metrics["param_haluc"]
                log_dict["epoch/tc_param_miss"] = tc_metrics["param_miss"]
                log_dict["epoch/tc_value_acc"] = tc_metrics["value_acc"]
                log_dict["epoch/tc_args_acc"] = tc_metrics["args_acc"]
                log_dict["epoch/tc_call_f1"] = tc_metrics["call_f1"]
            if retrieval_metrics and retrieval_metrics["num_queries"] > 0:
                for k, v in retrieval_metrics["recall@k"].items():
                    log_dict[f"epoch/retrieval_recall@{k}"] = v
                log_dict["epoch/retrieval_mrr"] = retrieval_metrics["mrr"]
            wandb.log(log_dict)

    if use_wandb:
        wandb.finish()
    if best_ckpt_path:
        print(f"\nBest checkpoint (call_f1={best_call_f1:.1%}): {best_ckpt_path}")
    print("\nTraining complete.")
