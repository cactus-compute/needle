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
    get_batches, get_tokenizer,
    load_prepared_data, load_tool_calls,
    PrefetchIterator, count_batches,
    get_contrastive_batches,
)
from .model import (
    EncoderDecoderTransformer,
    TransformerConfig,
    make_packing_mask,
    make_causal_packing_mask,
    make_cross_packing_mask,
)

_HF_CHECKPOINT_REPO = "Cactus-Compute/checkpoints"


def _upload_checkpoint(ckpt_path):
    """Upload a checkpoint file to HuggingFace Hub in a background thread."""
    import threading

    def _upload():
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(_HF_CHECKPOINT_REPO, repo_type="model", private=True, exist_ok=True)
            filename = os.path.basename(ckpt_path)
            print(f"[hf] Uploading {filename} to {_HF_CHECKPOINT_REPO} ...")
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=filename,
                repo_id=_HF_CHECKPOINT_REPO,
                repo_type="model",
            )
            print(f"[hf] Checkpoint uploaded: {_HF_CHECKPOINT_REPO}/{filename}")
        except Exception as e:
            print(f"[hf] Warning: checkpoint upload failed: {e}")

    threading.Thread(target=_upload, daemon=True).start()


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
            if g.ndim == 3:
                return jax.vmap(_newton_schulz, in_axes=(0,))(g)
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
        if name == "kernel" and leaf.ndim in (2, 3):
            return "muon"
        return "adam"

    return jax.tree_util.tree_map_with_path(_label, params)


def _wsd_schedule(peak_value, total_steps, warmup_steps, decay_ratio=0.15):
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


def create_train_state(rng, config, learning_rate, muon_lr, total_steps, warmup_steps, decay_ratio=0.15):
    model = EncoderDecoderTransformer(config)

    rng, init_rng = jax.random.split(rng)
    dummy_src = jnp.ones((1, 128), dtype=jnp.int32)
    dummy_tgt = jnp.ones((1, 128), dtype=jnp.int32)
    init_args = [dummy_src, dummy_tgt]
    if config.enable_speech:
        init_args.append(jnp.ones((1, 128, config.n_mels), dtype=jnp.float32))
    variables = model.init(
        {"params": init_rng},
        *init_args,
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
    Handles both 2D (in, out) and 3D (num_layers, in, out) kernels from nn.scan.
    """
    all_scores = []
    for _, leaf in jax.tree_util.tree_leaves_with_path(params):
        if leaf.ndim == 2:
            in_feat, out_feat = leaf.shape
            gs = min(group_size, in_feat)
            pad = (gs - in_feat % gs) % gs
            w = jnp.pad(leaf, ((0, pad), (0, 0))) if pad else leaf
            scores = jnp.sum(jnp.abs(w.reshape(-1, gs, out_feat)), axis=1).ravel()
            all_scores.append(scores)
        elif leaf.ndim == 3:
            num_layers, in_feat, out_feat = leaf.shape
            gs = min(group_size, in_feat)
            pad = (gs - in_feat % gs) % gs
            for li in range(num_layers):
                w = jnp.pad(leaf[li], ((0, pad), (0, 0))) if pad else leaf[li]
                scores = jnp.sum(jnp.abs(w.reshape(-1, gs, out_feat)), axis=1).ravel()
                all_scores.append(scores)

    if not all_scores:
        return jax.tree.map(jnp.ones_like, params)

    threshold = jnp.percentile(jnp.concatenate(all_scores), sparsity * 100)

    def _leaf_mask(path, leaf):
        if leaf.ndim == 3:
            num_layers, in_feat, out_feat = leaf.shape
            gs = min(group_size, in_feat)
            pad = (gs - in_feat % gs) % gs
            masks = []
            for li in range(num_layers):
                w = jnp.pad(leaf[li], ((0, pad), (0, 0))) if pad else leaf[li]
                w_grouped = w.reshape(-1, gs, out_feat)
                block_keep = (jnp.sum(jnp.abs(w_grouped), axis=1, keepdims=True) > threshold)
                m = jnp.broadcast_to(block_keep, w_grouped.shape).reshape(-1, out_feat)[:in_feat]
                masks.append(m)
            return jnp.stack(masks, axis=0).astype(leaf.dtype)
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
        if name == "kernel" and leaf.ndim == 3:
            return jax.vmap(_fake_quantize_int4, in_axes=(0,))(leaf)
        if name == "kernel" and leaf.ndim == 2:
            return _fake_quantize_int4(leaf, group_size=group_size)
        return leaf
    return jax.tree_util.tree_map_with_path(_maybe_quantize, params)


_GROUP_SIZE = 32
_MAT_FACTORS = ()
_MAT_FF_WIDTHS = ()
_D_FF = 2048
_CONTRASTIVE_WEIGHT = 0.1
# Weight map: class 0=base, 1=name, 2=value, 3=key (set from CLI args at train time)
_LOSS_WEIGHT_MAP = jnp.array([1.0, 3.0, 2.0, 1.5], dtype=jnp.float32)


def _clip_contrastive_loss(q_emb, t_emb, log_temp):
    """CLIP-style symmetric contrastive loss with learnable temperature."""
    temp = jnp.exp(jnp.clip(log_temp, -jnp.log(100.0), jnp.log(100.0)))
    logits = q_emb @ t_emb.T / temp  # (B, B)
    B = logits.shape[0]
    labels = jnp.arange(B)
    loss_q = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss_t = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels)
    return (jnp.mean(loss_q) + jnp.mean(loss_t)) / 2.0


def _text_loss_fn(state, params, src, tgt_in, tgt_out, ffn_mask, rng, loss_mask, do_quantize,
                  enc_seg_ids, dec_seg_ids):
    q_params = jax.lax.cond(
        do_quantize,
        lambda p: _quantize_params(p, group_size=_GROUP_SIZE),
        lambda p: p,
        params,
    )
    # Segment-aware masks work for both packed and non-packed batches.
    # Non-packed: seg_id=1 for all non-padding tokens → equivalent to padding mask + causal.
    # Packed: seg_id=1,2,3... per sub-sequence → block-diagonal attention.
    src_mask = make_packing_mask(enc_seg_ids)
    tgt_mask = make_causal_packing_mask(dec_seg_ids)
    cross_mask = make_cross_packing_mask(enc_seg_ids, dec_seg_ids)
    logits, slot_div = state.apply_fn(
        {"params": q_params},
        src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask,
        cross_mask=cross_mask,
        ffn_mask=ffn_mask,
        deterministic=False,
        method="forward_masked",
        rngs={"dropout": rng},
    )
    logits_f32 = logits.astype(jnp.float32)
    # loss_mask contains class labels (0=base,1=name,2=value,3=key) from packed data;
    # non-padding positions have seg_id > 0, padding has class 0 and seg_id 0.
    # Convert to weights via lookup, then zero out padding using dec_seg_ids.
    token_weights = _LOSS_WEIGHT_MAP[loss_mask.astype(jnp.int32)]
    padding_mask = (dec_seg_ids > 0).astype(jnp.float32)
    mask = token_weights * padding_mask
    num_tokens = jnp.maximum(jnp.sum(padding_mask), 1.0)
    ce_loss = jnp.sum(
        optax.softmax_cross_entropy_with_integer_labels(logits_f32, tgt_out) * mask
    ) / num_tokens
    z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits_f32, axis=-1) ** 2)
    return ce_loss + z_loss


def _contrastive_loss_fn(state, params, query_tokens, tool_tokens, rng, do_quantize):
    """Compute CLIP contrastive loss on query/tool pairs."""
    q_params = jax.lax.cond(
        do_quantize,
        lambda p: _quantize_params(p, group_size=_GROUP_SIZE),
        lambda p: p,
        params,
    )
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


def _train_step(state, src, tgt_in, tgt_out, enc_seg_ids, dec_seg_ids, prune_mask, ffn_mask, rng, loss_mask, query_tokens, tool_tokens, cl_rng, do_quantize, do_contrastive, do_prune):
    """Unified train step with boolean flags for contrastive loss and pruning."""

    def combined_loss(p):
        text_loss = _text_loss_fn(state, p, src, tgt_in, tgt_out, ffn_mask, rng, loss_mask, do_quantize,
                                  enc_seg_ids, dec_seg_ids)
        cl_loss = jax.lax.cond(
            do_contrastive,
            lambda: _contrastive_loss_fn(state, p, query_tokens, tool_tokens, cl_rng, do_quantize),
            lambda: 0.0,
        )
        return text_loss + _CONTRASTIVE_WEIGHT * cl_loss, text_loss

    (loss, text_loss), grads = jax.value_and_grad(combined_loss, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    text_loss = jax.lax.pmean(text_loss, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    new_params = jax.lax.cond(
        do_prune,
        lambda: jax.tree.map(lambda w, m: w * m, state.params, prune_mask),
        lambda: state.params,
    )
    state = state.replace(params=new_params)
    return state, text_loss, grad_norm


def _make_p_train_step():
    return jax.pmap(_train_step, axis_name="batch", donate_argnums=(0, 1))


def _make_val_loss_fn(apply_fn):
    @jax.jit
    def val_loss_batch(params, src, tgt_in, tgt_out, loss_mask, enc_seg_ids, dec_seg_ids):
        src_mask = make_packing_mask(enc_seg_ids)
        tgt_mask = make_causal_packing_mask(dec_seg_ids)
        cross_mask = make_cross_packing_mask(enc_seg_ids, dec_seg_ids)
        logits = apply_fn(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tgt_out)
        # Val uses uniform weights for consistent PPL — just mask padding via seg_ids
        padding_mask = (dec_seg_ids > 0).astype(jnp.float32)
        return jnp.sum(loss * padding_mask), jnp.sum(padding_mask)
    return val_loss_batch


def _make_mat_val_loss_fn(apply_fn, ff_width):
    """Val loss for matryoshka sub-model at given FFN width."""
    @jax.jit
    def val_loss_batch(params, src, tgt_in, tgt_out, loss_mask, enc_seg_ids, dec_seg_ids):
        src_mask = make_packing_mask(enc_seg_ids)
        tgt_mask = make_causal_packing_mask(dec_seg_ids)
        cross_mask = make_cross_packing_mask(enc_seg_ids, dec_seg_ids)
        logits, _, mat_logits = apply_fn(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask,
            mat_ff_widths=(ff_width,),
            method="forward_with_aux",
        )
        trunc_logits = mat_logits[0].astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(trunc_logits, tgt_out)
        padding_mask = (dec_seg_ids > 0).astype(jnp.float32)
        return jnp.sum(loss * padding_mask), jnp.sum(padding_mask)
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
    speech_sub = 0
    if getattr(config, "enable_speech", True):
        C = 256
        n_mels = getattr(config, "n_mels", 80)
        freq_out = n_mels // 8
        speech_sub = (1 * C * 9) + (C * 9 + C * C) + (C * 9 + C * C) + (C * freq_out * d)
    enc_block = attn + ffn
    dec_block = attn * 2 + ffn
    total = emb + speech_sub + n_enc * enc_block + n_dec * dec_block
    return int(total)


def shard_batch(batch, num_devices):
    """Reshape a batch array so leading dim is (num_devices, per_device_batch, ...)."""
    return batch.reshape(num_devices, -1, *batch.shape[1:])


def train(args):
    num_devices = jax.local_device_count()

    use_wandb = getattr(args, "wandb", False)
    if use_wandb:
        import wandb
        if wandb.run is None:
            wandb.init(project="needle-v1", config=vars(args))

    print(f"\n[1/3] Detecting devices...")
    print(f"      {num_devices} device(s) for data-parallel training")

    print(f"\n[2/3] Loading tokenizer...")
    tokenizer = get_tokenizer(max_samples=args.max_samples)

    print(f"\n[3/3] Loading prepared data from disk (mmap)...")
    train_data = load_prepared_data("train", mmap=True)
    val_data = load_prepared_data("val", mmap=True)

    enc_inputs = train_data["packed_enc"]
    dec_inputs = train_data["packed_dec_in"]
    dec_targets = train_data["packed_dec_tgt"]
    train_loss_mask = train_data["packed_loss"]
    train_enc_seg = train_data["packed_enc_seg"]
    train_dec_seg = train_data["packed_dec_seg"]

    val_enc = val_data["packed_enc"]
    val_dec_in = val_data["packed_dec_in"]
    val_dec_tgt = val_data["packed_dec_tgt"]
    val_loss_mask = val_data["packed_loss"]
    val_enc_seg = val_data["packed_enc_seg"]
    val_dec_seg = val_data["packed_dec_seg"]
    print(f"      {len(enc_inputs):,} train / {len(val_enc):,} val packed bins (memory-mapped)")

    val_ds = load_tool_calls("val", max_samples=args.max_samples)

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
            contrastive_dim=getattr(args, "contrastive_dim", 128),
            enable_speech=getattr(args, "enable_speech", False),
            no_feedforward=getattr(args, "no_feedforward", True),
        )

    global _GROUP_SIZE, _MAT_FACTORS, _MAT_FF_WIDTHS, _D_FF, _CONTRASTIVE_WEIGHT, _LOSS_WEIGHT_MAP
    _GROUP_SIZE = getattr(args, "group_size", 32)
    _CONTRASTIVE_WEIGHT = getattr(args, "contrastive_weight", 0.1)
    _LOSS_WEIGHT_MAP = jnp.array([
        1.0,                                       # 0: base (boilerplate)
        getattr(args, "w_name", 2.0),              # 1: tool name
        getattr(args, "w_value", 4.0),             # 2: argument value
        getattr(args, "w_key", 1.5),               # 3: argument key
    ], dtype=jnp.float32)
    _D_FF = config.d_ff
    mat_factors_raw = getattr(args, "mat_factors", None)
    if mat_factors_raw and not config.no_feedforward:
        _MAT_FACTORS = tuple(f for f in mat_factors_raw if f > 1)
        _MAT_FF_WIDTHS = tuple(config.d_ff // f for f in _MAT_FACTORS)
    else:
        _MAT_FACTORS = ()
        _MAT_FF_WIDTHS = ()
    n_widths = 1 + len(_MAT_FF_WIDTHS) if _MAT_FF_WIDTHS else 1
    p_train_step = _make_p_train_step()

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    unique_batch_size = (effective_batch_size // num_devices) * num_devices
    text_batches_per_epoch = count_batches(len(enc_inputs), unique_batch_size)
    num_batches = text_batches_per_epoch
    total_steps = num_batches * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    scaled_lr = args.lr * num_devices
    muon_lr = getattr(args, "muon_lr", 0.02) * math.sqrt(num_devices)
    decay_ratio = getattr(args, "decay_ratio", 0.15)
    state = create_train_state(init_rng, config, scaled_lr, muon_lr, total_steps, warmup_steps, decay_ratio)
    val_loss_fn = _make_val_loss_fn(state.apply_fn)

    if resume_checkpoint:
        state = state.replace(params=ckpt_params)
        print(f"  Loaded checkpoint params into train state")

    state = jax_utils.replicate(state)

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

    text_ffn_mask = _make_ffn_mask(args.batch_size, config.d_ff, _MAT_FF_WIDTHS)
    text_ffn_mask = jnp.broadcast_to(
        text_ffn_mask[None, :, :],
        (num_devices, args.batch_size, config.d_ff),
    )
    if n_widths > 1:
        print(f"  Mat factors   {n_widths} (full + {', '.join(str(f)+'x' for f in _MAT_FACTORS)})")
        print(f"  Mat mode      unique input ({args.batch_size}/dev, split by width)")

    adam_schedule = _wsd_schedule(scaled_lr, total_steps, warmup_steps)
    muon_schedule = _wsd_schedule(muon_lr, total_steps, warmup_steps)
    tokens_per_batch = effective_batch_size * (args.max_enc_len + args.max_dec_len)

    eval_model = EncoderDecoderTransformer(config)

    last_val_ppl = None
    sparsity_ratio = getattr(args, "sparsity_ratio", 0.0)
    prune_mask = None
    gradual_sparsify_done = False
    # Dummy all-ones prune mask (replicated) — used when pruning is inactive
    _unrep_params = jax_utils.unreplicate(state).params
    dummy_prune_mask = jax_utils.replicate(jax.tree.map(jnp.ones_like, _unrep_params))
    del _unrep_params
    # Dummy contrastive arrays — used when contrastive is inactive
    dummy_cl_tokens = jnp.zeros((num_devices, args.batch_size, 128), dtype=jnp.int32)
    dummy_cl_rng = jax.random.split(jax.random.PRNGKey(0), num_devices)

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
        text_batch_iter = PrefetchIterator(
            lambda: get_batches(enc_inputs, dec_inputs, dec_targets, unique_batch_size,
                                loss_mask=train_loss_mask,
                                enc_seg_ids=train_enc_seg,
                                dec_seg_ids=train_dec_seg),
            prefetch=4,
        )

        # Contrastive batch iterator (cycles through contrastive data alongside text)
        cl_batch_iter = None
        if has_contrastive and _CONTRASTIVE_WEIGHT > 0:
            cl_batch_iter = PrefetchIterator(
                lambda: get_contrastive_batches(
                    cl_query_tokens, cl_tool_tokens, cl_tool_ex_idx, cl_tool_is_pos,
                    unique_batch_size),
                prefetch=4,
            )

        pbar = tqdm(range(text_batches_per_epoch), desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step_i in pbar:
            t0 = time.perf_counter()

            src, tgt_in, tgt_out, lm, enc_seg, dec_seg = next(text_batch_iter)

            # Upcast from int16/float16 storage to int32/float32 for JAX
            src = np.asarray(src, dtype=np.int32)
            tgt_in = np.asarray(tgt_in, dtype=np.int32)
            tgt_out = np.asarray(tgt_out, dtype=np.int32)
            lm = np.asarray(lm, dtype=np.int32)  # class labels (0=base,1=name,2=value,3=key)
            enc_seg = np.asarray(enc_seg, dtype=np.int32)
            dec_seg = np.asarray(dec_seg, dtype=np.int32)

            src_b = shard_batch(src, num_devices)
            tgt_in_b = shard_batch(tgt_in, num_devices)
            tgt_out_b = shard_batch(tgt_out, num_devices)
            lm_b = shard_batch(lm, num_devices)
            enc_seg_b = shard_batch(enc_seg, num_devices)
            dec_seg_b = shard_batch(dec_seg, num_devices)

            rng, text_rng = jax.random.split(rng)
            text_rngs = jax.random.split(text_rng, num_devices)

            do_qat = jnp.broadcast_to(jnp.array(global_step % 100 == 0), (num_devices,))

            do_contrastive = cl_batch_iter is not None and global_step % 1000 == 0
            cl_q_b, cl_t_b, cl_rngs = dummy_cl_tokens, dummy_cl_tokens, dummy_cl_rng
            if do_contrastive:
                try:
                    cl_q, cl_t = next(cl_batch_iter)
                    cl_q_b = shard_batch(cl_q, num_devices)
                    cl_t_b = shard_batch(cl_t, num_devices)
                    rng, cl_rng = jax.random.split(rng)
                    cl_rngs = jax.random.split(cl_rng, num_devices)
                except StopIteration:
                    cl_batch_iter = None
                    do_contrastive = False

            do_cl = jnp.broadcast_to(jnp.array(do_contrastive), (num_devices,))
            do_prune = jnp.broadcast_to(jnp.array(prune_mask is not None), (num_devices,))
            cur_prune_mask = prune_mask if prune_mask is not None else dummy_prune_mask

            state, loss, grad_norm = p_train_step(
                state, src_b, tgt_in_b, tgt_out_b, enc_seg_b, dec_seg_b,
                cur_prune_mask, text_ffn_mask, text_rngs, lm_b,
                cl_q_b, cl_t_b, cl_rngs, do_qat, do_cl, do_prune,
            )

            text_loss_val = float(loss[0])
            text_losses.append(text_loss_val)
            step_grad_norm = float(grad_norm[0])
            global_step += 1

            if epoch == weight_prune_epoch and not gradual_sparsify_done:
                epoch_step += 1
                current_sparsity = _cubic_sparsity_schedule(epoch_step, t_start, t_end, sparsity_ratio)
                if epoch_step >= t_start and epoch_step % prune_interval == 0 and current_sparsity > 0:
                    ema_unr = jax_utils.unreplicate(state.params)
                    mask = _make_prune_mask(ema_unr, current_sparsity, _GROUP_SIZE)
                    del ema_unr
                    prune_mask = jax_utils.replicate(mask)
                    del mask

            dt = time.perf_counter() - t0
            eval_every = getattr(args, "eval_every", 100)
            if global_step % eval_every == 0 or global_step == total_steps:
                _eval_params = jax_utils.unreplicate(state).params
                total_loss, total_toks = 0.0, 0.0
                _val_n = count_batches(len(val_enc), args.batch_size)
                for vb in tqdm(get_batches(val_enc, val_dec_in, val_dec_tgt, args.batch_size, shuffle=False,
                                           loss_mask=val_loss_mask, enc_seg_ids=val_enc_seg, dec_seg_ids=val_dec_seg),
                               total=_val_n, desc="  val loss", leave=False):
                    src_v, di_v, dt_v, lm_v, es_v, ds_v = vb
                    src_v = jnp.asarray(src_v, dtype=jnp.int32)
                    di_v = jnp.asarray(di_v, dtype=jnp.int32)
                    dt_v = jnp.asarray(dt_v, dtype=jnp.int32)
                    lm_v = jnp.asarray(lm_v, dtype=jnp.float32)
                    es_v = jnp.asarray(es_v, dtype=jnp.int32)
                    ds_v = jnp.asarray(ds_v, dtype=jnp.int32)
                    vl, vt = val_loss_fn(_eval_params, src_v, di_v, dt_v, lm_v, es_v, ds_v)
                    total_loss += float(vl)
                    total_toks += float(vt)
                last_val_ppl = float(math.exp(min(total_loss / max(total_toks, 1), 20)))

                del _eval_params

            postfix = {
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
                if epoch == weight_prune_epoch and not gradual_sparsify_done:
                    log_dict["train/scheduled_sparsity"] = current_sparsity
                if global_step % eval_every == 0 or global_step == total_steps:
                    log_dict["val/text_ppl"] = last_val_ppl
                wandb.log(log_dict)

        text_batch_iter.close()
        if cl_batch_iter is not None:
            cl_batch_iter.close()

        if epoch == weight_prune_epoch and not gradual_sparsify_done:
            gradual_sparsify_done = True
            if prune_mask is None:
                ema_unr = jax_utils.unreplicate(state.params)
                mask = _make_prune_mask(ema_unr, sparsity_ratio, _GROUP_SIZE)
                del ema_unr
                prune_mask = jax_utils.replicate(mask)
                del mask
            state = state.replace(
                params=jax.tree.map(lambda w, m: w * m, state.params, prune_mask))
            final_pruned = jax.tree.map(np.array, jax_utils.unreplicate(state).params)
            total_p = sum(x.size for x in jax.tree.leaves(final_pruned))
            zero_p = sum(int(np.sum(np.abs(x) < 1e-6)) for x in jax.tree.leaves(final_pruned))
            print(f"\n  Gradual sparsification complete — mask locked.")
            print(f"  Final sparsity: {zero_p/total_p*100:.2f}% ({zero_p:,}/{total_p:,} near-zero)")
            del final_pruned

        epoch_avg_loss = sum(text_losses) / len(text_losses) if text_losses else float("nan")
        final_loss = text_losses[-1] if text_losses else float("nan")
        final_ppl = math.exp(min(final_loss, 20)) if not math.isnan(final_loss) else float("nan")

        eval_params = jax_utils.unreplicate(state).params

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

        _val_n = count_batches(len(val_enc), args.batch_size)
        _val_bar = tqdm(get_batches(val_enc, val_dec_in, val_dec_tgt, args.batch_size,
                              shuffle=False, loss_mask=val_loss_mask,
                              enc_seg_ids=val_enc_seg, dec_seg_ids=val_dec_seg),
                        total=_val_n, desc="  eval val loss", leave=False)
        for vb in _val_bar:
            src, dec_in, dec_tgt, lm, es, ds = vb
            src = jnp.asarray(src, dtype=jnp.int32)
            dec_in = jnp.asarray(dec_in, dtype=jnp.int32)
            dec_tgt = jnp.asarray(dec_tgt, dtype=jnp.int32)
            lm = jnp.asarray(lm, dtype=jnp.float32)
            es = jnp.asarray(es, dtype=jnp.int32)
            ds = jnp.asarray(ds, dtype=jnp.int32)
            vl, vt = val_loss_fn(eval_params, src, dec_in, dec_tgt, lm, es, ds)
            full_loss += float(vl); full_toks += float(vt)
            vl, vt = val_loss_fn(q_params, src, dec_in, dec_tgt, lm, es, ds)
            q_loss += float(vl); q_toks += float(vt)
            for f, fn in mat_vl_fns.items():
                vl, vt = fn(eval_params, src, dec_in, dec_tgt, lm, es, ds)
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

        params_np = jax.tree.map(lambda x: np.array(x).astype(np.float16), eval_params)
        total_params = sum(x.size for x in jax.tree.leaves(params_np))
        near_zero = sum(int(np.sum(np.abs(x) < 1e-6)) for x in jax.tree.leaves(params_np))
        sparsity = near_zero / total_params * 100

        ckpt_name = f"needle_{args.num_layers}_{args.d_model}_{global_step}.pkl"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        with open(ckpt_path, "wb") as f:
            pickle.dump({"params": params_np, "config": config.__dict__}, f)
        del params_np

        from .run import generate_batch

        val_kept = val_data["kept_indices"]

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

        import json as _json_mod
        tc_per_bucket = 50
        tc_rng = np.random.RandomState(epoch + 42)

        def _classify_sample(ex):
            """Classify a sample as single or multi call."""
            try:
                answers = _json_mod.loads(ex["answers"])
            except (ValueError, TypeError):
                return "empty"
            if not answers or not isinstance(answers, list):
                return "empty"
            return "single" if len(answers) == 1 else "multi"

        # 2 pools: single/multi, each balanced by tool count
        _pool_names = ["single", "multi"]
        _pool_buckets = {name: {t: [] for t in range(11)} for name in _pool_names}

        for k in range(len(val_kept)):
            ex = val_ds[int(val_kept[k])]
            try:
                nc = min(len(_json_mod.loads(ex["tools"])), 10)
            except (ValueError, TypeError):
                nc = 0
            call_type = _classify_sample(ex)
            if call_type == "empty":
                continue
            _pool_buckets[call_type][nc].append(k)

        def _balanced_sample(buckets, per_bucket, rng):
            pool = []
            for t in range(11):
                b = np.array(buckets[t])
                if len(b) > 0:
                    rng.shuffle(b)
                    pool.extend(b[:per_bucket].tolist())
            rng.shuffle(pool)
            return [val_ds[int(val_kept[k])] for k in pool]

        eval_pools = {}
        for name in _pool_names:
            eval_pools[name] = _balanced_sample(_pool_buckets[name], tc_per_bucket, tc_rng)

        all_eval_examples = display_pairs
        _pool_offsets = {}
        for name in _pool_names:
            _pool_offsets[name] = len(all_eval_examples)
            all_eval_examples = all_eval_examples + eval_pools[name]
        eval_gen_len = min(args.max_dec_len, 512)
        _EVAL_BATCH = 32
        _n_chunks = (len(all_eval_examples) + _EVAL_BATCH - 1) // _EVAL_BATCH

        # Use smallest matryoshka slice for generation eval if available
        if _MAT_FACTORS:
            from .export import slice_params
            _max_factor = max(_MAT_FACTORS)
            _mat_params, _mat_config = slice_params(eval_params, config, _max_factor)
            _mat_params = jax.tree.map(jnp.array, _mat_params)
            _gen_model = EncoderDecoderTransformer(_mat_config)
            _gen_label = f"{_max_factor}x (d_ff={_mat_config.d_ff})"
        else:
            _mat_params = eval_params
            _gen_model = eval_model
            _gen_label = "full"

        _mat_params = _quantize_params(_mat_params, group_size=_GROUP_SIZE)
        _gen_label += " INT4"

        all_preds = []
        _gen_t0 = time.perf_counter()
        for _ei in tqdm(range(0, len(all_eval_examples), _EVAL_BATCH),
                        total=_n_chunks, desc=f"  eval generate ({_gen_label})", leave=False):
            _chunk = all_eval_examples[_ei:_ei + _EVAL_BATCH]
            all_preds.extend(generate_batch(
                _gen_model, _mat_params, tokenizer,
                [ex["query"] for ex in _chunk],
                [ex["tools"] for ex in _chunk],
                max_gen_len=eval_gen_len,
                max_enc_len=args.max_enc_len,
                constrained=False,
            ))
        _gen_elapsed = time.perf_counter() - _gen_t0
        _gen_total_toks = sum(len(tokenizer.encode(p)) for p in all_preds)
        _gen_tok_per_sec = _gen_total_toks / max(_gen_elapsed, 1e-6)
        del _mat_params

        display_preds = all_preds[:len(display_pairs)]
        pool_preds = {}
        for name in _pool_names:
            start = _pool_offsets[name]
            end = start + len(eval_pools[name])
            pool_preds[name] = all_preds[start:end]

        unified_samples = []
        for ex, pred in zip(display_pairs, display_preds):
            try:
                ref = _json_mod.dumps(_json_mod.loads(ex["answers"]), separators=(",", ":"))
            except (ValueError, TypeError):
                ref = ex["answers"].strip() or "[]"
            unified_samples.append({
                "query": ex["query"],
                "tools": ex["tools"],
                "ref": ref,
                "text": pred.strip(),
            })

        def _call_key(c):
            if not isinstance(c, dict): return None
            return _json_mod.dumps({"name": c.get("name"), "arguments": c.get("arguments")}, sort_keys=True)

        _pc_keys = ("n", "exact", "name_tp", "name_fp", "name_fn",
                     "call_tp", "call_fp", "call_fn", "parse_err")

        def _eval_pool(eval_pairs, preds):
            """Compute tool-call metrics for a pool of (example, prediction) pairs."""
            m_n, m_exact, m_name_tp, m_name_fp, m_name_fn = 0, 0, 0, 0, 0
            m_call_tp, m_call_fp, m_call_fn, m_parse_err = 0, 0, 0, 0
            m_args_correct, m_args_total = 0, 0
            m_halluc, m_total_pred_params = 0, 0
            m_missing, m_total_ref_params = 0, 0
            m_correct_values, m_matched_params = 0, 0
            m_per_count = {t: {k: 0 for k in _pc_keys} for t in range(11)}
            m_failures = []

            for ex, pred_text in zip(eval_pairs, preds):
                try:
                    tool_defs = _json_mod.loads(ex["tools"])
                    num_tools = min(len(tool_defs), 10)
                except (ValueError, TypeError):
                    tool_defs = []
                    num_tools = 0
                pc = m_per_count[num_tools]

                ref_text = ex["answers"].strip()
                pred_text = pred_text.strip()
                ref_is_empty = ref_text in ("", "[]")
                pred_is_empty = pred_text in ("[]", "")
                try:
                    ref_calls = _json_mod.loads(ref_text) if not ref_is_empty else []
                except (ValueError, TypeError):
                    ref_calls = []
                try:
                    pred_calls = _json_mod.loads(pred_text) if not pred_is_empty else []
                    if not isinstance(pred_calls, list):
                        pred_calls = [pred_calls] if isinstance(pred_calls, dict) else []
                except (ValueError, TypeError):
                    m_parse_err += 1
                    pc["parse_err"] += 1
                    pred_calls = []
                m_n += 1
                pc["n"] += 1
                if ref_is_empty and pred_is_empty:
                    m_exact += 1
                    pc["exact"] += 1
                elif not ref_is_empty and not pred_is_empty and _json_mod.dumps(pred_calls, sort_keys=True) == _json_mod.dumps(ref_calls, sort_keys=True):
                    m_exact += 1
                    pc["exact"] += 1
                ref_names = {c["name"] for c in ref_calls if isinstance(c, dict) and "name" in c}
                pred_names = {c["name"] for c in pred_calls if isinstance(c, dict) and "name" in c}
                m_name_tp += len(pred_names & ref_names)
                m_name_fp += len(pred_names - ref_names)
                m_name_fn += len(ref_names - pred_names)
                pc["name_tp"] += len(pred_names & ref_names)
                pc["name_fp"] += len(pred_names - ref_names)
                pc["name_fn"] += len(ref_names - pred_names)
                rk = {_call_key(c) for c in ref_calls} - {None}
                pk = {_call_key(c) for c in pred_calls} - {None}
                m_call_tp += len(pk & rk)
                m_call_fp += len(pk - rk)
                m_call_fn += len(rk - pk)
                pc["call_tp"] += len(pk & rk)
                pc["call_fp"] += len(pk - rk)
                pc["call_fn"] += len(rk - pk)

                ref_by_name = {}
                for c in ref_calls:
                    if isinstance(c, dict) and "name" in c:
                        ref_by_name.setdefault(c["name"], []).append(c.get("arguments", {}))
                for c in pred_calls:
                    if isinstance(c, dict) and "name" in c and c["name"] in ref_by_name:
                        m_args_total += 1
                        pred_args = _json_mod.dumps(c.get("arguments", {}), sort_keys=True)
                        if any(pred_args == _json_mod.dumps(ra, sort_keys=True) for ra in ref_by_name[c["name"]]):
                            m_args_correct += 1

                tool_param_map = {}
                for t in tool_defs:
                    if isinstance(t, dict) and "name" in t:
                        tool_param_map[t["name"]] = set((t.get("parameters") or {}).keys())
                for c in pred_calls:
                    if not isinstance(c, dict) or "name" not in c:
                        continue
                    cname = c["name"]
                    if cname not in tool_param_map:
                        continue
                    pred_keys = set((c.get("arguments") or {}).keys())
                    m_total_pred_params += len(pred_keys)
                    m_halluc += len(pred_keys - tool_param_map[cname])
                    if cname in ref_by_name:
                        ref_args = ref_by_name[cname][0]
                        ref_keys = set((ref_args if isinstance(ref_args, dict) else {}).keys())
                        m_total_ref_params += len(ref_keys)
                        m_missing += len(ref_keys - pred_keys)
                        matched_keys = pred_keys & ref_keys
                        m_matched_params += len(matched_keys)
                        for k in matched_keys:
                            if _json_mod.dumps(c.get("arguments", {})[k], sort_keys=True) == _json_mod.dumps(ref_args[k], sort_keys=True):
                                m_correct_values += 1

                # Failure diagnosis
                is_exact = (ref_is_empty and pred_is_empty) or (
                    not ref_is_empty and not pred_is_empty
                    and _json_mod.dumps(pred_calls, sort_keys=True) == _json_mod.dumps(ref_calls, sort_keys=True)
                )
                if not is_exact and len(m_failures) < 30:
                    reasons = []
                    ex_fp = pred_names - ref_names
                    ex_fn = ref_names - pred_names
                    if ex_fp:
                        reasons.append(f"wrong_tools:{','.join(sorted(ex_fp))}")
                    if ex_fn:
                        reasons.append(f"missing_tools:{','.join(sorted(ex_fn))}")
                    if not ex_fp and not ex_fn and pred_names:
                        for c in pred_calls:
                            if not isinstance(c, dict) or "name" not in c or c["name"] not in ref_by_name:
                                continue
                            pa = c.get("arguments", {})
                            ra = ref_by_name[c["name"]][0]
                            for k in set(pa.keys()) | set(ra.keys()):
                                pv, rv = pa.get(k, "<MISSING>"), ra.get(k, "<MISSING>")
                                if _json_mod.dumps(pv, sort_keys=True) != _json_mod.dumps(rv, sort_keys=True):
                                    reasons.append(f"{c['name']}.{k}={_json_mod.dumps(pv)[:50]}!={_json_mod.dumps(rv)[:50]}")
                    if ref_is_empty and not pred_is_empty:
                        reasons.append("false_positive")
                    if not ref_is_empty and pred_is_empty:
                        reasons.append("false_negative")
                    if not reasons:
                        reasons.append("unknown")
                    m_failures.append({
                        "query": ex["query"][:150],
                        "ref": ref_text[:200],
                        "pred": pred_text[:200],
                        "reasons": reasons,
                    })

            metrics = {}
            if m_n > 0:
                metrics["n"] = m_n
                metrics["parse_rate"] = 1.0 - m_parse_err / m_n
                metrics["exact_match"] = m_exact / m_n
                np_ = m_name_tp + m_name_fp
                nr_ = m_name_tp + m_name_fn
                metrics["name_f1"] = 2 * m_name_tp / max(np_ + nr_, 1)
                cp_ = m_call_tp + m_call_fp
                cr_ = m_call_tp + m_call_fn
                metrics["call_f1"] = 2 * m_call_tp / max(cp_ + cr_, 1)
                metrics["args_acc"] = m_args_correct / max(m_args_total, 1)
                metrics["param_haluc"] = m_halluc / max(m_total_pred_params, 1)
                metrics["param_miss"] = m_missing / max(m_total_ref_params, 1)
                metrics["value_acc"] = m_correct_values / max(m_matched_params, 1)
                metrics["failures"] = m_failures
            return metrics, m_per_count

        pool_metrics = {}
        pool_pc = {}
        for name in _pool_names:
            pool_metrics[name], pool_pc[name] = _eval_pool(eval_pools[name], pool_preds[name])

        _best_metric = pool_metrics.get("single", {}).get("call_f1", 0)
        if _best_metric > best_call_f1:
            best_call_f1 = _best_metric
            if config.no_feedforward:
                best_ckpt_path = os.path.join(args.checkpoint_dir, f"needle_{args.num_layers}_{args.d_model}_best.pkl")
            else:
                best_ckpt_path = os.path.join(args.checkpoint_dir, f"needle_{args.num_layers}_{args.d_model}_{config.d_ff}_best.pkl")
            import shutil as _shutil
            _shutil.copy2(ckpt_path, best_ckpt_path)
            print(f"  ** New best single call_f1={best_call_f1:.1%} → {best_ckpt_path}")
            _upload_checkpoint(best_ckpt_path)
            if _MAT_FACTORS:
                from .export import export_submodel
                for factor in _MAT_FACTORS:
                    d_ff_sub = config.d_ff // factor
                    mat_ckpt_path = os.path.join(
                        args.checkpoint_dir,
                        f"needle_{args.num_layers}_{args.d_model}_{d_ff_sub}_best.pkl",
                    )
                    export_submodel(best_ckpt_path, factor, mat_ckpt_path)
                    _upload_checkpoint(mat_ckpt_path)

        retrieval_metrics = None
        if has_contrastive and _CONTRASTIVE_WEIGHT > 0:
            from .eval import benchmark_retrieval
            retrieval_metrics = benchmark_retrieval(
                eval_model, eval_params, tokenizer,
                num_samples=min(500, getattr(args, "max_eval_samples", 500)),
                ds=val_ds,
            )

        del eval_params

        print(f"\n  ─────────────────────────────────────")
        print(f"  Epoch {epoch + 1}/{args.epochs}")
        print(f"  ─────────────────────────────────────")
        print(f"  Text loss      {final_loss:>12.4f}")
        print(f"  Text val ppl   {last_val_ppl:>12.2f}")
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
        def _print_tc_metrics(label, metrics, pc):
            if not metrics:
                return
            n = metrics["n"]
            print(f"  ─── {label} ({n} samples) ──")
            print(f"  JSON parse     {metrics['parse_rate']:>10.1%}")
            print(f"  Name F1        {metrics['name_f1']:>10.1%}")
            print(f"  Param haluc    {metrics['param_haluc']:>10.1%}")
            print(f"  Param miss     {metrics['param_miss']:>10.1%}")
            print(f"  Value acc      {metrics['value_acc']:>10.1%}")
            print(f"  Args acc       {metrics['args_acc']:>10.1%}")
            print(f"  Call F1        {metrics['call_f1']:>10.1%}")
            print(f"  Exact match    {metrics['exact_match']:>10.1%}")
            has_any_pc = any(pc[t]["n"] > 0 for t in range(11))
            if has_any_pc:
                print(f"  {'#tools':>6}  {'n':>4}  {'name_f1':>8}  {'nTP':>4} {'nFP':>4} {'nFN':>4}  {'call_f1':>8}  {'cTP':>4} {'cFP':>4} {'cFN':>4}  {'exact':>6}  {'parse':>6}")
                for t in range(11):
                    d = pc[t]
                    if d["n"] == 0:
                        continue
                    np_ = d["name_tp"] + d["name_fp"]
                    nr_ = d["name_tp"] + d["name_fn"]
                    nf1 = 2 * d["name_tp"] / max(np_ + nr_, 1)
                    cp_ = d["call_tp"] + d["call_fp"]
                    cr_ = d["call_tp"] + d["call_fn"]
                    cf1 = 2 * d["call_tp"] / max(cp_ + cr_, 1)
                    ex_ = d["exact"] / d["n"]
                    pr_ = 1.0 - d["parse_err"] / d["n"]
                    print(f"  {t:>6}  {d['n']:>4}  {nf1:>7.1%}  {d['name_tp']:>4} {d['name_fp']:>4} {d['name_fn']:>4}  {cf1:>7.1%}  {d['call_tp']:>4} {d['call_fp']:>4} {d['call_fn']:>4}  {ex_:>5.1%}  {pr_:>5.1%}")
            if metrics.get("failures"):
                print(f"  ─── Failures ({len(metrics['failures'])} captured) ───")
                for j, fail in enumerate(metrics["failures"][:10]):
                    print(f"  [{j+1}] Q: {fail['query'][:120]}")
                    print(f"      Ref:  {fail['ref'][:200]}")
                    print(f"      Pred: {fail['pred'][:200]}")
                    print(f"      Why:  {', '.join(fail['reasons'])}")
                    print()

        _label_map = {
            "single": "Single-Call",
            "multi": "Multi-Call",
        }
        for name in _pool_names:
            _print_tc_metrics(_label_map[name], pool_metrics[name], pool_pc[name])
        if retrieval_metrics and retrieval_metrics["num_queries"] > 0:
            rm = retrieval_metrics
            print(f"  ─── Retrieval ({rm['num_queries']} queries) ─────")
            for k, v in sorted(rm["recall@k"].items()):
                print(f"  Recall@{k:<3}     {v:>10.1%}")
            print(f"  MRR            {rm['mrr']:>10.3f}")
        print(f"  ─────────────────────────────────────")
        print(f"  Throughput     {_gen_tok_per_sec:>10.1f} tok/s  ({len(all_eval_examples)} samples, {_gen_elapsed:.1f}s, {_gen_label})")
        if unified_samples:
            print(f"  ─── Samples ({len(unified_samples)}) ───────────────────")
            for j, s in enumerate(unified_samples):
                print(f"  [{j+1}] Query: {s['query'][:120]}")
                tools_short = s["tools"][:120]
                if len(s["tools"]) > 120:
                    tools_short += "..."
                print(f"      Tools: {tools_short}")
                print(f"      Ref:   {s['ref'][:200]}")
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
            for factor, (mat_ppl, mat_params, _) in mat_results.items():
                log_dict[f"epoch/mat_ppl_{factor}x"] = mat_ppl
                log_dict[f"epoch/mat_params_{factor}x"] = mat_params
            for name in _pool_names:
                m = pool_metrics.get(name)
                if not m:
                    continue
                for k in ("parse_rate", "exact_match", "name_f1", "call_f1",
                          "args_acc", "param_haluc", "param_miss", "value_acc"):
                    log_dict[f"epoch/{name}_{k}"] = m[k]
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

    # Auto-calibrate confidence head on best checkpoint
    if getattr(args, "calibrate", True) and best_ckpt_path:
        print(f"\n{'='*50}")
        print(f"Running confidence head calibration on {best_ckpt_path}")
        print(f"{'='*50}")
        from .calibrate import main as calibrate_main
        from argparse import Namespace
        cal_args = Namespace(
            checkpoint=best_ckpt_path,
            output=best_ckpt_path,
            batch_size=args.batch_size * 4,
            num_samples=None,  # use full dataset
            epochs=getattr(args, "calibrate_epochs", 1),
            lr=getattr(args, "calibrate_lr", 1e-3),
            k=getattr(args, "calibrate_k", 3.0),
        )
        calibrate_main(cal_args)
        print(f"\nCalibration complete. Checkpoint updated: {best_ckpt_path}")


