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
    load_prepared_data, load_prepared_mels,
    load_example_with_audio,
    PrefetchIterator, count_batches,
)
from .model import (
    EncoderDecoderTransformer,
    TransformerConfig,
    count_params,
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
_MAT_FF_WIDTHS = ()  # precomputed d_ff widths per factor
_D_FF = 2048  # set in train()
_N_BLOCKS = 12  # num_encoder_layers + num_decoder_layers, set in train()
_MAT_SPREAD_LAMBDA = 0.01
_MAT_GUMBEL = False


def topk_mask(logits, k, tau, hard):
    """Differentiable top-k mask. logits: (d_ff,), k: int, tau/hard: JAX scalars.

    Learning phase (hard=False): returns soft sigmoid mask for gradient flow.
    Freeze phase (hard=True): returns stop_gradient hard mask — no STE.
    Ties at threshold may select slightly more than k neurons; this is negligible
    and export uses exact argsort top-k for the final mask.
    """
    if k >= logits.shape[0]:
        return jnp.ones_like(logits)
    topk_vals = jax.lax.top_k(logits, k)[0]
    threshold = topk_vals[-1]
    y_soft = jax.nn.sigmoid((logits - threshold) / tau)
    y_hard = (y_soft >= 0.5).astype(y_soft.dtype)
    frozen = jax.lax.stop_gradient(y_hard)
    return jnp.where(hard, frozen, y_soft)


def _gumbel_sample(rng, shape):
    """Sample from Gumbel(0, 1) distribution."""
    u = jax.random.uniform(rng, shape, minval=1e-20, maxval=1.0)
    return -jnp.log(-jnp.log(u))


def _make_ffn_mask_topk(batch_size, d_ff, mask_logits, mat_ff_widths, tau, hard, step_rng):
    """Build (n_blocks, batch, d_ff) per-layer topk mask.

    mask_logits: (n_mat, n_blocks, d_ff). tau/hard: JAX scalars.
    step_rng: PRNGKey for Gumbel noise (used only if _MAT_GUMBEL is True).
    Returns (n_blocks, batch, d_ff) stacked mask.
    """
    n_blocks = mask_logits.shape[1]
    n_widths = 1 + len(mat_ff_widths)
    per_width = batch_size // n_widths
    remainder = batch_size - per_width * n_widths

    block_masks = []
    for b in range(n_blocks):
        rows = [jnp.ones((per_width, d_ff), dtype=jnp.bfloat16)]
        for i, ff_w in enumerate(mat_ff_widths):
            logits_b = mask_logits[i, b]  # (d_ff,)
            if _MAT_GUMBEL:
                sub_rng = jax.random.fold_in(step_rng, b * 1000 + i)
                noise = _gumbel_sample(sub_rng, (per_width, d_ff))
                # Zero noise in hard/freeze mode so all items converge
                noise = noise * (1.0 - hard.astype(jnp.float32))
                noisy = logits_b[None, :] + noise  # (per_width, d_ff)
                m = jax.vmap(lambda l: topk_mask(l, k=ff_w, tau=tau, hard=hard))(noisy)
            else:
                m = topk_mask(logits_b, k=ff_w, tau=tau, hard=hard)
                m = jnp.broadcast_to(m[None, :], (per_width, d_ff))
            rows.append(m.astype(jnp.bfloat16))
        if remainder > 0:
            rows.append(jnp.ones((remainder, d_ff), dtype=jnp.bfloat16))
        block_masks.append(jnp.concatenate(rows, axis=0))
    return jnp.stack(block_masks)  # (n_blocks, batch, d_ff)


def _compute_ce(logits, tgt_out, slot_div, loss_mask=None):
    """Shared CE + z-loss + slot-div computation."""
    pad_id = 0
    logits_f32 = logits.astype(jnp.float32)
    if loss_mask is not None:
        mask = loss_mask
    else:
        mask = (tgt_out != pad_id).astype(jnp.float32)
    ce_loss = jnp.sum(
        optax.softmax_cross_entropy_with_integer_labels(logits_f32, tgt_out) * mask
    ) / jnp.maximum(jnp.sum(mask), 1.0)
    z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits_f32, axis=-1) ** 2)
    div_loss = 1e-4 * slot_div
    return ce_loss + z_loss + div_loss


def _text_loss_fn(state, params, src, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask):
    pad_id = 0
    src_mask = make_padding_mask(src, pad_id)
    tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
    logits, slot_div = state.apply_fn(
        {"params": _quantize_params(params, group_size=_GROUP_SIZE)},
        src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask,
        ffn_mask=ffn_mask,
        deterministic=False,
        method="forward_masked",
        rngs={"dropout": rng},
    )
    return _compute_ce(logits, tgt_out, slot_div, loss_mask=loss_mask)


def _speech_loss_fn(state, params, mel, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask):
    pad_id = 0
    src_mask = make_mel_padding_mask(mel)
    tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
    spec_rng, drop_rng = jax.random.split(rng)
    logits, slot_div = state.apply_fn(
        {"params": _quantize_params(params, group_size=_GROUP_SIZE)},
        mel, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask,
        ffn_mask=ffn_mask,
        deterministic=False,
        method="forward_speech_masked",
        rngs={"specaugment": spec_rng, "dropout": drop_rng},
    )
    return _compute_ce(logits, tgt_out, slot_div, loss_mask=loss_mask)


def _forward_masked(state, params, src, tgt_in, causal_mask, ffn_mask, is_speech=False, spec_rng=None):
    """Dispatch forward_masked or forward_speech_masked based on is_speech."""
    q_params = _quantize_params(params, group_size=_GROUP_SIZE)
    tgt_mask = causal_mask & make_padding_mask(tgt_in, 0)
    if is_speech:
        return state.apply_fn(
            {"params": q_params}, src, tgt_in,
            src_mask=make_mel_padding_mask(src), tgt_mask=tgt_mask,
            ffn_mask=ffn_mask, deterministic=False,
            method="forward_speech_masked", rngs={"specaugment": spec_rng},
        )
    else:
        return state.apply_fn(
            {"params": q_params}, src, tgt_in,
            src_mask=make_padding_mask(src, 0), tgt_mask=tgt_mask,
            ffn_mask=ffn_mask, method="forward_masked",
        )


def _topk_loss(state, params, mask_logits, src, tgt_in, tgt_out, causal_mask,
               tau, hard, step_rng, is_speech=False, spec_rng=None, loss_mask=None):
    """Topk loss for text or speech. Builds masks inside for gradient flow."""
    ffn_mask = _make_ffn_mask_topk(src.shape[0], _D_FF, mask_logits, _MAT_FF_WIDTHS, tau, hard, step_rng)
    logits, slot_div = _forward_masked(state, params, src, tgt_in, causal_mask, ffn_mask,
                                       is_speech=is_speech, spec_rng=spec_rng)
    loss = _compute_ce(logits, tgt_out, slot_div, loss_mask=loss_mask)
    if _MAT_SPREAD_LAMBDA > 0:
        spread = jnp.mean(jnp.var(mask_logits, axis=-1))
        loss = loss - _MAT_SPREAD_LAMBDA * spread
    return loss


def _warmup_loss(state, params, src, tgt_in, tgt_out, causal_mask, is_speech=False, spec_rng=None, loss_mask=None):
    """Full-model-only loss (no matryoshka) for topk warmup phase."""
    ffn_mask = jnp.ones((src.shape[0], _D_FF), dtype=jnp.bfloat16)
    logits, slot_div = _forward_masked(state, params, src, tgt_in, causal_mask, ffn_mask,
                                       is_speech=is_speech, spec_rng=spec_rng)
    return _compute_ce(logits, tgt_out, slot_div, loss_mask=loss_mask)


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


def _grad_step(state, ema_params, loss_fn, prune_mask=None):
    """Shared body for all non-topk train steps: grad, pmean, apply, ema."""
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    state, ema_params = _apply_and_ema(state, ema_params, grads, prune_mask)
    return state, ema_params, loss, optax.global_norm(grads)


def _train_step_text(state, ema_params, src, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask):
    return _grad_step(state, ema_params,
        lambda p: _text_loss_fn(state, p, src, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask))


def _train_step_text_masked(state, ema_params, src, tgt_in, tgt_out, causal_mask, prune_mask, ffn_mask, rng, loss_mask):
    return _grad_step(state, ema_params,
        lambda p: _text_loss_fn(state, p, src, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask),
        prune_mask=prune_mask)


def _train_step_speech(state, ema_params, mel, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask):
    return _grad_step(state, ema_params,
        lambda p: _speech_loss_fn(state, p, mel, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask))


def _train_step_speech_masked(state, ema_params, mel, tgt_in, tgt_out, causal_mask, prune_mask, ffn_mask, rng, loss_mask):
    return _grad_step(state, ema_params,
        lambda p: _speech_loss_fn(state, p, mel, tgt_in, tgt_out, causal_mask, ffn_mask, rng, loss_mask),
        prune_mask=prune_mask)


def _apply_and_ema(state, ema_params, grads, prune_mask=None):
    """Apply gradients, optional prune mask, and EMA update. Returns (state, ema)."""
    state = state.apply_gradients(grads=grads)
    if prune_mask is not None:
        params = jax.tree.map(lambda w, m: w * m, state.params, prune_mask)
        state = state.replace(params=params)
    else:
        params = state.params
    ema = jax.tree.map(lambda e, p: 0.999 * e + 0.001 * p, ema_params, params)
    return state, ema


def _train_step_text_warmup(state, ema_params, src, tgt_in, tgt_out, causal_mask, loss_mask):
    """Text warmup step (no matryoshka). Returns grads for saliency accumulation."""
    loss, grads = jax.value_and_grad(
        lambda p: _warmup_loss(state, p, src, tgt_in, tgt_out, causal_mask, loss_mask=loss_mask)
    )(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    state, ema_params = _apply_and_ema(state, ema_params, grads)
    return state, ema_params, loss, optax.global_norm(grads), grads


def _train_step_speech_warmup(state, ema_params, mel, tgt_in, tgt_out, causal_mask, spec_rng, loss_mask):
    """Speech warmup step (no matryoshka)."""
    loss, grads = jax.value_and_grad(
        lambda p: _warmup_loss(state, p, mel, tgt_in, tgt_out, causal_mask, is_speech=True, spec_rng=spec_rng, loss_mask=loss_mask)
    )(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    state, ema_params = _apply_and_ema(state, ema_params, grads)
    return state, ema_params, loss, optax.global_norm(grads), grads


def _topk_grad_step(state, ema_params, mask_logits, loss_fn, prune_mask=None):
    """Shared body for all topk train steps: grad, pmean, apply, ema."""
    (loss, (p_grads, ml_grads)) = jax.value_and_grad(loss_fn, argnums=(0, 1))(state.params, mask_logits)
    p_grads = jax.lax.pmean(p_grads, axis_name="batch")
    ml_grads = jax.lax.pmean(ml_grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    state, ema_params = _apply_and_ema(state, ema_params, p_grads, prune_mask)
    return state, ema_params, ml_grads, loss, optax.global_norm(p_grads)


def _train_step_text_topk(state, ema_params, mask_logits, src, tgt_in, tgt_out, causal_mask, tau, hard, step_rng, loss_mask):
    return _topk_grad_step(state, ema_params, mask_logits,
        lambda p, ml: _topk_loss(state, p, ml, src, tgt_in, tgt_out, causal_mask, tau, hard, step_rng, loss_mask=loss_mask))


def _train_step_text_topk_masked(state, ema_params, mask_logits, src, tgt_in, tgt_out, causal_mask, prune_mask, tau, hard, step_rng, loss_mask):
    return _topk_grad_step(state, ema_params, mask_logits,
        lambda p, ml: _topk_loss(state, p, ml, src, tgt_in, tgt_out, causal_mask, tau, hard, step_rng, loss_mask=loss_mask),
        prune_mask=prune_mask)


def _train_step_speech_topk(state, ema_params, mask_logits, mel, tgt_in, tgt_out, causal_mask, tau, hard, step_rng, spec_rng, loss_mask):
    return _topk_grad_step(state, ema_params, mask_logits,
        lambda p, ml: _topk_loss(state, p, ml, mel, tgt_in, tgt_out, causal_mask, tau, hard, step_rng, is_speech=True, spec_rng=spec_rng, loss_mask=loss_mask))


def _train_step_speech_topk_masked(state, ema_params, mask_logits, mel, tgt_in, tgt_out, causal_mask, prune_mask, tau, hard, step_rng, spec_rng, loss_mask):
    return _topk_grad_step(state, ema_params, mask_logits,
        lambda p, ml: _topk_loss(state, p, ml, mel, tgt_in, tgt_out, causal_mask, tau, hard, step_rng, is_speech=True, spec_rng=spec_rng, loss_mask=loss_mask),
        prune_mask=prune_mask)


def _extract_ffn_saliency(grads, d_ff, n_enc, n_dec):
    """Extract per-layer per-FFN-neuron saliency from param gradients.

    Returns (n_blocks, d_ff) array of neuron importance scores per block.
    """
    n_blocks = n_enc + n_dec
    saliency = np.zeros((n_blocks, d_ff), dtype=np.float32)
    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        path_str = "/".join(p.key if hasattr(p, "key") else str(p) for p in path)
        if "down_proj" not in path_str or "kernel" not in path_str or leaf.ndim != 2:
            continue
        g = np.array(leaf)
        if g.shape[0] != d_ff:
            continue
        neuron_sal = np.sum(g ** 2, axis=1)
        for part in path:
            name = part.key if hasattr(part, "key") else str(part)
            if name.startswith("block_"):
                block_idx = int(name.split("_")[1])
                if "encoder" in path_str:
                    saliency[block_idx] += neuron_sal
                elif "decoder" in path_str:
                    saliency[n_enc + block_idx] += neuron_sal
                break
    return saliency


def _update_mask_logits(ml_grads, mask_logits, mask_tx, mask_opt_state):
    """Host-side mask logit optimizer step. Returns (updated mask_logits, opt_state)."""
    ml_grads_np = np.array(jax_utils.unreplicate(ml_grads))
    ml_np = np.array(jax_utils.unreplicate(mask_logits))
    updates, mask_opt_state = mask_tx.update(ml_grads_np, mask_opt_state, ml_np)
    ml_np = optax.apply_updates(ml_np, updates)
    return jax_utils.replicate(jnp.array(ml_np)), mask_opt_state


def _pmap_train_step(step_fn):
    """Create a pmap'd train step from any step function."""
    return jax.pmap(step_fn, axis_name="batch", donate_argnums=(0, 1))


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


def _make_mat_val_loss_fn(apply_fn, ff_width=None, ffn_mask=None):
    """Val loss for matryoshka sub-model at given FFN width or with a topk ffn_mask."""
    @jax.jit
    def val_loss_batch(params, src, tgt_in, tgt_out, causal_mask, loss_mask):
        pad_id = 0
        src_mask = make_padding_mask(src, pad_id)
        tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
        kwargs = {}
        if ffn_mask is not None:
            kwargs["mat_ffn_masks"] = [ffn_mask]
        else:
            kwargs["mat_ff_widths"] = (ff_width,)
        logits, _, mat_logits = apply_fn(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask,
            method="forward_with_aux",
            **kwargs,
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
        mask = loss_mask
        return jnp.sum(loss * mask), jnp.sum(mask)
    return val_loss_batch


def _eval_val_ppl(loss_fn, params, batches, causal_mask):
    """Evaluate validation perplexity over batches. Returns PPL (capped at exp(20))."""
    total_loss, total_toks = 0.0, 0.0
    for vb in batches:
        vl, vt = loss_fn(params, vb[0], vb[1], vb[2], causal_mask, vb[3])
        total_loss += float(vl)
        total_toks += float(vt)
    return float(math.exp(min(total_loss / max(total_toks, 1), 20)))


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


def _tile_for_mat(arr, num_devices, per_width, n_widths):
    """Tile a batch for shared-input matryoshka: repeat each sample across all widths."""
    s = arr.reshape(num_devices, per_width, *arr.shape[1:])
    return np.tile(s, (1, n_widths) + (1,) * (arr.ndim - 1))


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
    val_enc = val_data["enc_inputs"]
    val_dec_in = val_data["dec_inputs"]
    val_dec_tgt = val_data["dec_targets"]
    val_loss_mask = val_data["loss_mask"]
    print(f"      {len(enc_inputs):,} train / {len(val_enc):,} val tool-call pairs (memory-mapped)")

    train_mels = None
    val_mels = None
    if not no_speech:
        step_idx += 1
        print(f"\n[{step_idx}/{total_data_steps}] Loading precomputed mel spectrograms (mmap)...")
        train_mels = load_prepared_mels(train_data["mel_cache_id"], mmap=True)
        val_mels = load_prepared_mels(val_data["mel_cache_id"], mmap=True)
        print(f"      {len(train_mels):,} train / {len(val_mels):,} val mel spectrograms (memory-mapped)")

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
        )

    global _GROUP_SIZE, _MAT_FACTORS, _MAT_FF_WIDTHS, _D_FF, _N_BLOCKS, _MAT_SPREAD_LAMBDA, _MAT_GUMBEL
    _GROUP_SIZE = getattr(args, "group_size", 32)
    _D_FF = config.d_ff
    _N_BLOCKS = config.num_encoder_layers + config.num_decoder_layers
    mat_factors_raw = getattr(args, "mat_factors", None)
    if mat_factors_raw:
        _MAT_FACTORS = tuple(f for f in mat_factors_raw if f > 1)
        _MAT_FF_WIDTHS = tuple(config.d_ff // f for f in _MAT_FACTORS)
    else:
        _MAT_FACTORS = ()
        _MAT_FF_WIDTHS = ()
    n_widths = 1 + len(_MAT_FF_WIDTHS) if _MAT_FF_WIDTHS else 1

    # Fallback defaults here are for programmatic callers; CLI defaults are in cli.py
    mat_method = getattr(args, "mat_method", "static-prefix")
    use_topk = mat_method == "topk" and _MAT_FF_WIDTHS
    mat_tau_start = getattr(args, "mat_tau_start", 0.5)
    mat_tau_end = getattr(args, "mat_tau_end", 0.1)
    mat_warmup_frac = getattr(args, "mat_warmup_frac", 0.15)
    mat_freeze_frac = getattr(args, "mat_freeze_frac", 0.2)
    # These globals are read inside pmap-traced functions; must be set before pmap creation below
    _MAT_SPREAD_LAMBDA = getattr(args, "mat_spread_lambda", 0.01)
    _MAT_GUMBEL = getattr(args, "mat_gumbel", False)

    if use_topk:
        p_train_step_warmup = _pmap_train_step(_train_step_text_warmup)
        p_train_step_warmup_speech = _pmap_train_step(_train_step_speech_warmup)
        p_train_step_topk = _pmap_train_step(_train_step_text_topk)
        p_train_step_topk_masked = _pmap_train_step(_train_step_text_topk_masked)
        p_train_step_speech_topk = _pmap_train_step(_train_step_speech_topk)
        p_train_step_speech_topk_masked = _pmap_train_step(_train_step_speech_topk_masked)
    p_train_step = _pmap_train_step(_train_step_text)
    p_train_step_masked = _pmap_train_step(_train_step_text_masked)
    p_train_step_speech = _pmap_train_step(_train_step_speech)
    p_train_step_speech_masked = _pmap_train_step(_train_step_speech_masked)

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    mat_shared_input = getattr(args, "mat_shared_input", False)
    if use_topk and mat_shared_input:
        raise ValueError("--mat-shared-input is incompatible with --mat-method topk")
    # With shared input, each unique sample is repeated n_widths times,
    # so we fetch smaller batches but take more steps per epoch.
    unique_batch_size = effective_batch_size // n_widths if (mat_shared_input and n_widths > 1) else effective_batch_size
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
    state = create_train_state(init_rng, config, scaled_lr, muon_lr, total_steps, warmup_steps)
    val_loss_fn = _make_val_loss_fn(state.apply_fn)
    speech_vl_fn = _make_speech_val_loss_fn(state.apply_fn) if not no_speech else None

    if resume_checkpoint:
        state = state.replace(params=ckpt_params)
        print(f"  Loaded checkpoint params into train state")

    ema_params = jax.tree.map(jnp.copy, state.params)

    # --- TopK mask logit init ---
    mask_logits = None
    mask_opt_state = None
    mask_tx = None
    use_saliency = False
    saliency_accum = None
    saliency_steps = 0
    if use_topk:
        n_mat = len(_MAT_FF_WIDTHS)
        n_blocks = _N_BLOCKS
        init_mode = getattr(args, "mat_init_mode", "normal")
        init_value = getattr(args, "mat_init_value", 0.5)
        saliency_scale = getattr(args, "mat_saliency_scale", 1.0)
        rng, mask_rng = jax.random.split(rng)
        d_ff = config.d_ff
        use_saliency = init_mode == "saliency"
        if init_mode == "prefix":
            positions = jnp.arange(d_ff, dtype=jnp.float32)
            ramp = init_value * (1.0 - 2.0 * positions / max(1, d_ff - 1))
            mask_logits = jnp.broadcast_to(ramp[None, None, :], (n_mat, n_blocks, d_ff)).copy()
        elif init_mode == "shuffled_prefix":
            positions = jnp.arange(d_ff, dtype=jnp.float32)
            ramp = init_value * (1.0 - 2.0 * positions / max(1, d_ff - 1))
            rows = []
            for i in range(n_mat * n_blocks):
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, d_ff)
                rows.append(ramp[perm])
            mask_logits = jnp.stack(rows).reshape(n_mat, n_blocks, d_ff)
        elif init_mode == "saliency":
            mask_logits = jnp.zeros((n_mat, n_blocks, d_ff))
            saliency_accum = np.zeros((n_blocks, d_ff), dtype=np.float32)
        elif init_mode == "normal":
            mask_logits = jax.random.normal(mask_rng, (n_mat, n_blocks, d_ff)) * init_value
        else:
            mask_logits = jnp.zeros((n_mat, n_blocks, d_ff))
        mask_lr = getattr(args, "mat_mask_lr", 3e-3)
        mask_tx = optax.adam(learning_rate=mask_lr)
        mask_opt_state = mask_tx.init(np.array(mask_logits))
        if resume_checkpoint and "mask_logits" in ckpt_data:
            mask_logits = jnp.array(ckpt_data["mask_logits"])
            use_saliency = False
            print(f"  Loaded mask logits from checkpoint")

    state = jax_utils.replicate(state)
    ema_params = jax_utils.replicate(ema_params)
    if use_topk:
        mask_logits = jax_utils.replicate(mask_logits)

    param_count = count_params(jax_utils.unreplicate(state).params)
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
    print(f"  Dropout       {config.dropout_rate:>12}")
    if not no_speech and train_mels is not None:
        print(f"  Speech        {speech_batches_per_epoch} batches/epoch")
        print(f"  n_mels        {n_mels:>12}")
        print(f"  max_mel_len   {max_mel_len:>12}")
    else:
        print(f"  Speech               disabled")
    if use_topk:
        print(f"  Mat method          topk (learned)")
        print(f"  Mat tau       {mat_tau_start:.2f} -> {mat_tau_end:.2f}")
        print(f"  Mat warmup    {mat_warmup_frac*100:.0f}% / freeze {mat_freeze_frac*100:.0f}%")
        print(f"  Mat spread λ  {_MAT_SPREAD_LAMBDA}")
        print(f"  Mat per-layer  {n_blocks} blocks")
        if _MAT_GUMBEL:
            print(f"  Mat Gumbel           enabled")
    else:
        print(f"  Mat method           static-prefix")
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
        if mat_shared_input:
            print(f"  Mat mode      shared input ({unique_batch_size // num_devices}/dev x {n_widths} repeats)")
        else:
            print(f"  Mat mode      unique input ({args.batch_size}/dev, random width)")

    adam_schedule = _wsd_schedule(scaled_lr, total_steps, warmup_steps)
    muon_schedule = _wsd_schedule(muon_lr, total_steps, warmup_steps)
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
    mat_warmup_end = int(total_steps * mat_warmup_frac) if use_topk else 0
    # freeze_frac=1.0 → freeze_start=0 → no learning phase (saliency-only mode, intentional)
    mat_freeze_start = int(total_steps * (1 - mat_freeze_frac)) if use_topk else 0

    for epoch in range(args.epochs):
        if epoch == weight_prune_epoch and not gradual_sparsify_done:
            t_start = int(num_batches * prune_start_frac)
            t_end = int(num_batches * prune_end_frac)
            print(f"\nGradual magnitude sparsification: 0% -> {sparsity_ratio*100:.0f}% over epoch {epoch+1} "
                  f"(steps {t_start}-{t_end}/{num_batches}, interval={prune_interval}, group_size={_GROUP_SIZE})")
            epoch_step = 0

        text_losses = []
        speech_losses = []
        text_batch_iter = PrefetchIterator(
            lambda: get_batches(enc_inputs, dec_inputs, dec_targets, unique_batch_size,
                                loss_mask=train_loss_mask),
            prefetch=4,
        )

        speech_batch_iter = None
        if not no_speech and train_mels is not None:
            speech_batch_iter = PrefetchIterator(
                lambda: get_speech_batches(train_mels, dec_inputs, dec_targets, unique_batch_size,
                                           loss_mask=train_loss_mask),
                prefetch=4,
            )

        steps_this_epoch = text_batches_per_epoch + speech_batches_per_epoch
        text_idx = 0
        speech_idx = 0
        speech_loss_val = None
        pbar = tqdm(range(steps_this_epoch), desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step_i in pbar:
            # --- TopK phase tracking ---
            topk_active = False
            cur_tau = mat_tau_start
            cur_hard = False
            if use_topk:
                topk_active = global_step >= mat_warmup_end
                cur_hard = global_step >= mat_freeze_start
                if topk_active and not cur_hard:
                    progress = min(1.0, (global_step - mat_warmup_end) / max(1, mat_freeze_start - mat_warmup_end))
                    cur_tau = mat_tau_start * (mat_tau_end / mat_tau_start) ** progress
                elif cur_hard:
                    cur_tau = 0.001

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

                if not use_topk and n_widths > 1 and mat_shared_input:
                    tile = lambda arr: _tile_for_mat(arr, num_devices, args.batch_size // n_widths, n_widths)
                    src_b, tgt_in_b, tgt_out_b, lm_b = tile(src), tile(tgt_in), tile(tgt_out), tile(lm)
                else:
                    src_b = shard_batch(src, num_devices)
                    tgt_in_b = shard_batch(tgt_in, num_devices)
                    tgt_out_b = shard_batch(tgt_out, num_devices)
                    lm_b = shard_batch(lm, num_devices)

                rng, text_rng = jax.random.split(rng)
                text_rngs = jax.random.split(text_rng, num_devices)

                if topk_active:
                    tau_arr = jax_utils.replicate(jnp.float32(cur_tau))
                    hard_arr = jax_utils.replicate(jnp.bool_(cur_hard))
                    rng, step_rng = jax.random.split(rng)
                    step_rngs = jax.random.split(step_rng, num_devices)
                    if prune_mask is not None:
                        state, ema_params, ml_grads, loss, grad_norm = p_train_step_topk_masked(
                            state, ema_params, mask_logits, src_b, tgt_in_b, tgt_out_b, causal_mask, prune_mask, tau_arr, hard_arr, step_rngs, lm_b,
                        )
                    else:
                        state, ema_params, ml_grads, loss, grad_norm = p_train_step_topk(
                            state, ema_params, mask_logits, src_b, tgt_in_b, tgt_out_b, causal_mask, tau_arr, hard_arr, step_rngs, lm_b,
                        )
                    if not cur_hard:
                        mask_logits, mask_opt_state = _update_mask_logits(ml_grads, mask_logits, mask_tx, mask_opt_state)
                elif use_topk and not topk_active:
                    state, ema_params, loss, grad_norm, warmup_grads = p_train_step_warmup(
                        state, ema_params, src_b, tgt_in_b, tgt_out_b, causal_mask, lm_b,
                    )
                    if use_saliency and saliency_accum is not None:
                        grads_unr = jax_utils.unreplicate(warmup_grads)
                        step_saliency = _extract_ffn_saliency(grads_unr, config.d_ff, config.num_encoder_layers, config.num_decoder_layers)
                        saliency_steps += 1
                        beta = 0.99
                        saliency_accum = beta * saliency_accum + (1 - beta) * step_saliency
                        del grads_unr
                else:
                    if prune_mask is not None:
                        state, ema_params, loss, grad_norm = p_train_step_masked(
                            state, ema_params, src_b, tgt_in_b, tgt_out_b, causal_mask, prune_mask, text_ffn_mask, text_rngs, lm_b,
                        )
                    else:
                        state, ema_params, loss, grad_norm = p_train_step(
                            state, ema_params, src_b, tgt_in_b, tgt_out_b, causal_mask, text_ffn_mask, text_rngs, lm_b,
                        )

                text_loss_val = float(loss[0])
                text_losses.append(text_loss_val)
                step_grad_norm = float(grad_norm[0])
                global_step += 1

            else:
                mel_batch, sp_tgt_in, sp_tgt_out, sp_lm = next(speech_batch_iter)
                speech_idx += 1

                if not use_topk and n_widths > 1 and mat_shared_input:
                    tile = lambda arr: _tile_for_mat(arr, num_devices, args.batch_size // n_widths, n_widths)
                    mel_b, sp_tgt_in_b, sp_tgt_out_b, sp_lm_b = tile(mel_batch), tile(sp_tgt_in), tile(sp_tgt_out), tile(sp_lm)
                else:
                    mel_b = shard_batch(mel_batch, num_devices)
                    sp_tgt_in_b = shard_batch(sp_tgt_in, num_devices)
                    sp_tgt_out_b = shard_batch(sp_tgt_out, num_devices)
                    sp_lm_b = shard_batch(sp_lm, num_devices)

                if topk_active:
                    tau_arr = jax_utils.replicate(jnp.float32(cur_tau))
                    hard_arr = jax_utils.replicate(jnp.bool_(cur_hard))
                    rng, step_rng, spec_rng = jax.random.split(rng, 3)
                    step_rngs = jax.random.split(step_rng, num_devices)
                    spec_rngs = jax.random.split(spec_rng, num_devices)
                    if prune_mask is not None:
                        state, ema_params, ml_grads, sp_loss, sp_grad_norm = p_train_step_speech_topk_masked(
                            state, ema_params, mask_logits, mel_b, sp_tgt_in_b, sp_tgt_out_b, causal_mask, prune_mask, tau_arr, hard_arr, step_rngs, spec_rngs, sp_lm_b,
                        )
                    else:
                        state, ema_params, ml_grads, sp_loss, sp_grad_norm = p_train_step_speech_topk(
                            state, ema_params, mask_logits, mel_b, sp_tgt_in_b, sp_tgt_out_b, causal_mask, tau_arr, hard_arr, step_rngs, spec_rngs, sp_lm_b,
                        )
                    if not cur_hard:
                        mask_logits, mask_opt_state = _update_mask_logits(ml_grads, mask_logits, mask_tx, mask_opt_state)
                elif use_topk and not topk_active:
                    # Speech warmup grads not used for saliency, text is the primary task
                    rng, spec_rng = jax.random.split(rng)
                    spec_rngs = jax.random.split(spec_rng, num_devices)
                    state, ema_params, sp_loss, sp_grad_norm, _ = p_train_step_warmup_speech(
                        state, ema_params, mel_b, sp_tgt_in_b, sp_tgt_out_b, causal_mask, spec_rngs, sp_lm_b,
                    )
                else:
                    speech_ffn_mask = text_ffn_mask
                    rng, spec_rng = jax.random.split(rng)
                    spec_rngs = jax.random.split(spec_rng, num_devices)
                    if prune_mask is not None:
                        state, ema_params, sp_loss, sp_grad_norm = p_train_step_speech_masked(
                            state, ema_params, mel_b, sp_tgt_in_b, sp_tgt_out_b, causal_mask, prune_mask, speech_ffn_mask, spec_rngs, sp_lm_b,
                        )
                    else:
                        state, ema_params, sp_loss, sp_grad_norm = p_train_step_speech(
                            state, ema_params, mel_b, sp_tgt_in_b, sp_tgt_out_b, causal_mask, speech_ffn_mask, spec_rngs, sp_lm_b,
                        )
                speech_loss_val = float(sp_loss[0])
                speech_losses.append(speech_loss_val)
                step_grad_norm = float(sp_grad_norm[0])
                text_loss_val = text_losses[-1] if text_losses else float("nan")
                global_step += 1

            # Saliency init: promote accumulated saliency into mask_logits at warmup end
            if use_saliency and saliency_accum is not None and global_step >= mat_warmup_end:
                print(f"\n  Saliency init: {saliency_steps} warmup steps accumulated")
                logits_np = np.zeros_like(saliency_accum)
                for bl in range(n_blocks):
                    ranks = np.argsort(np.argsort(-saliency_accum[bl])).astype(np.float32)
                    logits_np[bl] = saliency_scale * (1.0 - 2.0 * ranks / max(1, config.d_ff - 1))
                mask_logits_np = np.broadcast_to(logits_np[None, :, :], (n_mat, n_blocks, config.d_ff)).copy()
                mask_logits = jax_utils.replicate(jnp.array(mask_logits_np))
                mask_opt_state = mask_tx.init(mask_logits_np)
                saliency_accum = None
                print(f"  Saliency logit range: [{logits_np.min():.3f}, {logits_np.max():.3f}]")

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
                last_val_ppl = _eval_val_ppl(val_loss_fn, _eval_params,
                    get_batches(val_enc, val_dec_in, val_dec_tgt, args.batch_size, shuffle=False, loss_mask=val_loss_mask),
                    val_causal)
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
            if use_topk and topk_active:
                phase = "freeze" if cur_hard else "learn"
                postfix["mat"] = f"{phase} τ={cur_tau:.3f}"
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
                if use_topk:
                    log_dict["train/mat_tau"] = cur_tau
                    log_dict["train/mat_topk_active"] = int(topk_active)
                if global_step % eval_every == 0 or global_step == total_steps:
                    log_dict["val/text_ppl"] = last_val_ppl
                wandb.log(log_dict)

        text_batch_iter.close()
        if speech_batch_iter is not None:
            speech_batch_iter.close()

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
            total_p = count_params(final_pruned)
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
        topk_hard_masks = {}
        mat_vl_fns = {}
        if _MAT_FACTORS:
            _apply_fn = jax_utils.unreplicate(state).apply_fn
            if use_topk:
                ml_unr = jax_utils.unreplicate(mask_logits)
                n_blocks_eval = ml_unr.shape[1]
                for i, ff_w in enumerate(_MAT_FF_WIDTHS):
                    block_masks = [topk_mask(ml_unr[i, b], k=ff_w, tau=jnp.float32(0.001), hard=jnp.bool_(True))
                                   for b in range(n_blocks_eval)]
                    topk_hard_masks[ff_w] = jnp.stack(block_masks)
            for f, fw in zip(_MAT_FACTORS, _MAT_FF_WIDTHS):
                if fw in topk_hard_masks:
                    mat_vl_fns[f] = _make_mat_val_loss_fn(_apply_fn, ffn_mask=topk_hard_masks[fw])
                else:
                    mat_vl_fns[f] = _make_mat_val_loss_fn(_apply_fn, fw)
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
        ckpt_data_out = {"params": params_np, "config": config.__dict__}
        if use_topk:
            ml_np = np.array(jax_utils.unreplicate(mask_logits))
            ckpt_data_out["mask_logits"] = ml_np
            ckpt_data_out["mat_method"] = "topk"
            ckpt_data_out["mat_factors"] = list(_MAT_FACTORS)
        with open(ckpt_path, "wb") as f:
            pickle.dump(ckpt_data_out, f)
        del params_np

        from .test import measure_throughput
        from .run import generate, generate_from_audio
        tp = measure_throughput(eval_model, eval_params, tokenizer, num_runs=5)

        from .data import _load_unified_dataset
        _ds_full = _load_unified_dataset()
        _val_start = int(len(_ds_full) * 0.9)
        val_kept = val_data["kept_indices"]
        n_eval_samples = min(3, len(val_kept))
        step = max(1, len(val_kept) // n_eval_samples)
        sample_indices = [val_kept[i * step] for i in range(n_eval_samples)]
        eval_indices = np.array(sample_indices) + _val_start

        unified_samples = []
        for i, ds_idx in enumerate(eval_indices):
            pair = load_example_with_audio(int(ds_idx))
            query = pair["query"][:80]
            ref = pair["answers"][:120]

            # Text prediction
            text_pred = generate(
                eval_model, eval_params, tokenizer, pair["query"],
                tools=pair["tools"], max_gen_len=args.max_dec_len, seed=i, stream=False,
            ).strip()[:120]

            # Voice prediction
            voice_pred = None
            if not no_speech and pair["audio_array"] is not None:
                voice_pred = generate_from_audio(
                    eval_model, eval_params, tokenizer, pair["audio_array"], sr=pair["sampling_rate"],
                    tools=pair["tools"], max_gen_len=args.max_dec_len, seed=i, stream=False,
                ).strip()[:120]

            unified_samples.append((query, ref, text_pred, voice_pred))

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
        print(f"  ─────────────────────────────────────")
        print(f"  Throughput     {tp['tokens_per_second']:>10.1f} tok/s")
        print(f"  Latency        {tp['avg_latency_s']:>11.3f}s")
        if unified_samples:
            print(f"  ─── Samples ────────────────────────")
            for query, ref, text_pred, voice_pred in unified_samples:
                print(f"  Q: {query}")
                print(f"  R: {ref}")
                print(f"  T: {text_pred or '(empty)'}")
                if voice_pred is not None:
                    print(f"  V: {voice_pred or '(empty)'}")
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
            wandb.log(log_dict)

    if use_wandb:
        wandb.finish()
    print("\nTraining complete.")


