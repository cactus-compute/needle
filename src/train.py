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

from .data import get_batches, get_tokenizer, load_tinystories, prepare_encoder_decoder_pairs
from .model import (
    EncoderDecoderTransformer,
    TransformerConfig,
    make_causal_mask,
    make_padding_mask,
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
            if g.ndim >= 2:
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

    def _label(path, _):
        name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        if name == "kernel":
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

    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_src = jnp.ones((1, 128), dtype=jnp.int32)
    dummy_tgt = jnp.ones((1, 128), dtype=jnp.int32)
    variables = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        dummy_src,
        dummy_tgt,
        deterministic=False,
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


def _block_prune_params(params, prune_ratio=0.25, group_size=32):
    """Block magnitude pruning aligned to quantization groups.

    All 2D weight matrices (Dense kernels + embeddings) are divided into
    (group_size, 1) blocks along axis 0. Blocks are scored by L1 norm,
    and the bottom `prune_ratio` fraction (globally) are zeroed out.
    """
    all_scores = []
    for _, leaf in jax.tree_util.tree_leaves_with_path(params):
        if leaf.ndim != 2:
            continue
        w = np.array(leaf)
        in_feat, out_feat = w.shape
        gs = min(group_size, in_feat)
        pad = (gs - in_feat % gs) % gs
        if pad:
            w = np.pad(w, ((0, pad), (0, 0)))
        w_grouped = w.reshape(-1, gs, out_feat)
        scores = np.sum(np.abs(w_grouped), axis=1).ravel()
        all_scores.append(scores)

    if not all_scores:
        return params

    threshold = np.percentile(np.concatenate(all_scores), prune_ratio * 100)

    def _prune_weight(path, leaf):
        if leaf.ndim != 2:
            return leaf
        w = np.array(leaf)
        in_feat, out_feat = w.shape
        gs = min(group_size, in_feat)
        pad = (gs - in_feat % gs) % gs
        if pad:
            w = np.pad(w, ((0, pad), (0, 0)))
        w_grouped = w.reshape(-1, gs, out_feat)
        scores = np.sum(np.abs(w_grouped), axis=1, keepdims=True)
        mask = (scores > threshold).astype(np.float32)
        w_pruned = (w_grouped * mask).reshape(-1, out_feat)[:in_feat]
        return jnp.array(w_pruned)

    return jax.tree_util.tree_map_with_path(_prune_weight, params)


def _layer_prune(params, config, layer_prune_ratio=0.25):
    """Remove the lowest-scoring encoder/decoder blocks by L1 magnitude.

    Returns (new_params, new_config) with surviving blocks renumbered.
    """
    from flax.core import unfreeze
    p = unfreeze(params)

    def _score_blocks(module_params, prefix):
        scores = {}
        for key, val in module_params.items():
            if key.startswith("block_"):
                s = sum(float(np.sum(np.abs(np.array(l)))) for l in jax.tree.leaves(val))
                scores[key] = s
        return scores

    enc_scores = _score_blocks(p["encoder"], "encoder")
    dec_scores = _score_blocks(p["decoder"], "decoder")

    # Pool all block scores, find global threshold
    all_scores = list(enc_scores.values()) + list(dec_scores.values())
    if not all_scores:
        return params, config

    # Keep at least 1 encoder and 1 decoder block
    num_enc = len(enc_scores)
    num_dec = len(dec_scores)
    total_to_remove = max(0, int(len(all_scores) * layer_prune_ratio))

    # Score and rank all blocks together
    tagged = [(s, "encoder", k) for k, s in enc_scores.items()] + \
             [(s, "decoder", k) for k, s in dec_scores.items()]
    tagged.sort(key=lambda x: x[0])

    remove_enc = set()
    remove_dec = set()
    for score, module, key in tagged:
        if len(remove_enc) + len(remove_dec) >= total_to_remove:
            break
        if module == "encoder" and (num_enc - len(remove_enc)) > 1:
            remove_enc.add(key)
        elif module == "decoder" and (num_dec - len(remove_dec)) > 1:
            remove_dec.add(key)

    # Rebuild with surviving blocks renumbered
    def _keep_and_renumber(module_params, remove_set):
        surviving = [(k, v) for k, v in sorted(module_params.items()) if k.startswith("block_") and k not in remove_set]
        non_blocks = {k: v for k, v in module_params.items() if not k.startswith("block_")}
        new = dict(non_blocks)
        for i, (_, v) in enumerate(surviving):
            new[f"block_{i}"] = v
        return new

    p["encoder"] = _keep_and_renumber(p["encoder"], remove_enc)
    p["decoder"] = _keep_and_renumber(p["decoder"], remove_dec)

    new_num_enc = num_enc - len(remove_enc)
    new_num_dec = num_dec - len(remove_dec)
    print(f"  Encoder: {num_enc} -> {new_num_enc} blocks (removed {len(remove_enc)})")
    print(f"  Decoder: {num_dec} -> {new_num_dec} blocks (removed {len(remove_dec)})")

    from dataclasses import replace as dc_replace
    new_config = dc_replace(config,
                            num_encoder_layers=new_num_enc,
                            num_decoder_layers=new_num_dec)

    return p, new_config


def _quantize_params(params, group_size=32):
    """Fake-quantize all Dense kernels in the param tree."""
    def _maybe_quantize(path, leaf):
        name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        if name == "kernel":
            return _fake_quantize_int4(leaf, group_size=group_size)
        return leaf
    return jax.tree_util.tree_map_with_path(_maybe_quantize, params)


# Module-level group_size used by _train_step (set before pmap in train())
_GROUP_SIZE = 32


def _train_step(state, ema_params, src, tgt_in, tgt_out, causal_mask, dropout_rng):
    pad_id = 0
    ema_decay = 0.999

    def loss_fn(params):
        src_mask = make_padding_mask(src, pad_id)
        tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
        cross_mask = src_mask

        logits = state.apply_fn(
            {"params": _quantize_params(params, group_size=_GROUP_SIZE)},
            src,
            tgt_in,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            cross_mask=cross_mask,
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )

        logits_f32 = logits.astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_f32, tgt_out)
        mask = (tgt_out != pad_id).astype(jnp.float32)
        ce_loss = jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits_f32, axis=-1) ** 2)
        return ce_loss + z_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    ema_params = jax.tree.map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema_params, state.params)
    return state, ema_params, loss, grad_norm


def _make_p_train_step():
    return jax.pmap(_train_step, axis_name="batch", donate_argnums=(0, 1))


def _make_val_loss_fn(apply_fn):
    @jax.jit
    def val_loss_batch(params, src, tgt_in, tgt_out, causal_mask):
        pad_id = 0
        src_mask = make_padding_mask(src, pad_id)
        tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
        logits = apply_fn(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=src_mask,
            deterministic=True,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tgt_out)
        mask = (tgt_out != pad_id).astype(jnp.float32)
        return jnp.sum(loss * mask), jnp.sum(mask)
    return val_loss_batch


def shard_batch(batch, num_devices):
    """Reshape a batch array so leading dim is (num_devices, per_device_batch, ...)."""
    return batch.reshape(num_devices, -1, *batch.shape[1:])


def train(args):
    num_devices = jax.local_device_count()
    print(f"Detected {num_devices} device(s) for data-parallel training")

    use_wandb = getattr(args, "wandb", False)
    if use_wandb:
        import wandb
        if wandb.run is None:
            wandb.init(project="needle-v1", config=vars(args))

    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    print("Loading TinyStories dataset...")
    ds = load_tinystories("train", max_samples=args.max_samples)
    val_ds = load_tinystories("validation", max_samples=getattr(args, "max_eval_samples", None))

    print("Preparing encoder-decoder pairs...")
    enc_inputs, dec_inputs, dec_targets = prepare_encoder_decoder_pairs(
        ds, tokenizer, max_enc_len=args.max_enc_len, max_dec_len=args.max_dec_len
    )
    print(f"Prepared {len(enc_inputs)} training pairs")

    val_enc, val_dec_in, val_dec_tgt = prepare_encoder_decoder_pairs(
        val_ds, tokenizer, max_enc_len=args.max_enc_len, max_dec_len=args.max_dec_len
    )
    val_enc = jnp.array(val_enc)
    val_dec_in = jnp.array(val_dec_in)
    val_dec_tgt = jnp.array(val_dec_tgt)
    print(f"Prepared {len(val_enc)} validation pairs")

    # Effective batch size must be divisible by num_devices
    effective_batch_size = args.batch_size * num_devices

    config = TransformerConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_model * 4,
        max_seq_len=max(args.max_enc_len, args.max_dec_len),
        dropout_rate=args.dropout,
        dtype=args.dtype,
        activation=getattr(args, "activation", "drelu"),
    )

    # Set group size for QAT and sparsification (captured by _train_step closure)
    global _GROUP_SIZE
    _GROUP_SIZE = getattr(args, "group_size", 32)
    p_train_step = _make_p_train_step()

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    num_batches = len(enc_inputs) // effective_batch_size
    total_steps = num_batches * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    scaled_lr = args.lr * num_devices
    muon_lr = getattr(args, "muon_lr", 0.02) * math.sqrt(num_devices)
    state = create_train_state(init_rng, config, scaled_lr, muon_lr, total_steps, warmup_steps)
    val_loss_fn = _make_val_loss_fn(state.apply_fn)

    ema_params = jax.tree.map(jnp.copy, state.params)
    state = jax_utils.replicate(state)
    ema_params = jax_utils.replicate(ema_params)

    param_count = sum(x.size for x in jax.tree.leaves(jax_utils.unreplicate(state).params))
    print(f"Model parameters: {param_count:,}")
    print(f"Devices: {num_devices}, per-device batch: {args.batch_size}, effective batch: {effective_batch_size}")
    print(f"Adam LR: {args.lr} x {num_devices} = {scaled_lr}, Muon LR: {args.muon_lr} x sqrt({num_devices}) = {muon_lr:.4f}")
    decay_steps = max(1, int(total_steps * 0.15))
    stable_steps = total_steps - warmup_steps - decay_steps
    print(f"LR schedule: warmup {warmup_steps}, stable {stable_steps}, decay {decay_steps} steps (WSD)")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    global_step = 0
    causal_mask = jnp.broadcast_to(
        make_causal_mask(args.max_dec_len),
        (num_devices, 1, args.max_dec_len, args.max_dec_len),
    )

    adam_schedule = _wsd_schedule(scaled_lr, total_steps, warmup_steps)
    muon_schedule = _wsd_schedule(muon_lr, total_steps, warmup_steps)
    tokens_per_batch = effective_batch_size * (args.max_enc_len + args.max_dec_len)

    enc_inputs = jnp.array(enc_inputs)
    dec_inputs = jnp.array(dec_inputs)
    dec_targets = jnp.array(dec_targets)

    last_val_ppl = None
    sparsity_ratio = getattr(args, "sparsity_ratio", 0.0)
    layer_prune_ratio = getattr(args, "layer_prune_ratio", 0.0)
    prune_mask = None

    # Determine which epoch each pruning type fires on:
    #   both flags:              group sparsify @ epoch 1, layer prune @ epoch 2
    #   only --sparsity-ratio:   group sparsify @ epoch 1
    #   only --layer-prune:      layer prune    @ epoch 1
    weight_prune_epoch = 1 if sparsity_ratio > 0 else -1
    layer_prune_epoch = (2 if sparsity_ratio > 0 else 1) if layer_prune_ratio > 0 else -1

    for epoch in range(args.epochs):
        # --- Weight block pruning ---
        if epoch == weight_prune_epoch and prune_mask is None:
            print(f"\nGroup-wise sparsification: {sparsity_ratio*100:.0f}% of blocks (group_size={_GROUP_SIZE})...")
            pruned_params = _block_prune_params(jax_utils.unreplicate(ema_params), prune_ratio=sparsity_ratio, group_size=_GROUP_SIZE)

            pruned_np = jax.tree.map(np.array, pruned_params)
            total_p = sum(x.size for x in jax.tree.leaves(pruned_np))
            zero_p = sum(int(np.sum(np.abs(x) < 1e-6)) for x in jax.tree.leaves(pruned_np))
            print(f"  Post-prune sparsity: {zero_p/total_p*100:.2f}% ({zero_p:,}/{total_p:,} near-zero)")

            val_causal = make_causal_mask(args.max_dec_len)
            total_loss, total_toks = 0.0, 0.0
            for vb in get_batches(val_enc, val_dec_in, val_dec_tgt, args.batch_size, shuffle=False):
                vl, vt = val_loss_fn(pruned_params, vb[0], vb[1], vb[2], val_causal)
                total_loss += float(vl)
                total_toks += float(vt)
            prune_val_ppl = float(math.exp(total_loss / max(total_toks, 1)))
            print(f"  Post-prune val ppl:  {prune_val_ppl:.2f} (was {last_val_ppl:.2f})" if last_val_ppl is not None else f"  Post-prune val ppl:  {prune_val_ppl:.2f}")

            state = jax_utils.unreplicate(state)
            state = state.replace(params=pruned_params)
            ema_params = jax.tree.map(jnp.copy, pruned_params)
            prune_mask = jax.tree.map(lambda w: (jnp.abs(w) > 1e-8).astype(jnp.float32), pruned_params)
            prune_mask = jax_utils.replicate(prune_mask)
            state = jax_utils.replicate(state)
            ema_params = jax_utils.replicate(ema_params)
            print(f"Continuing training with sparsity mask locked...\n")

        # --- Layer pruning ---
        if epoch == layer_prune_epoch:
            print(f"\nLayer pruning: removing {layer_prune_ratio*100:.0f}% of layers...")
            current_params = jax_utils.unreplicate(ema_params)
            new_params, config = _layer_prune(current_params, config, layer_prune_ratio=layer_prune_ratio)

            # Rebuild train state with new config/model so tree types are consistent
            rng, reinit_rng = jax.random.split(rng)
            state = create_train_state(reinit_rng, config, scaled_lr, muon_lr, total_steps, warmup_steps)
            matched_params = jax.tree.map(lambda init_leaf, prune_leaf: prune_leaf, state.params, new_params)
            state = state.replace(params=matched_params)
            ema_params = jax.tree.map(jnp.copy, matched_params)

            param_count = sum(x.size for x in jax.tree.leaves(matched_params))
            print(f"  Parameters: {param_count:,}")

            # Eval val perplexity after layer pruning
            val_loss_fn = _make_val_loss_fn(state.apply_fn)
            val_causal = make_causal_mask(args.max_dec_len)
            total_loss, total_toks = 0.0, 0.0
            for vb in get_batches(val_enc, val_dec_in, val_dec_tgt, args.batch_size, shuffle=False):
                vl, vt = val_loss_fn(matched_params, vb[0], vb[1], vb[2], val_causal)
                total_loss += float(vl)
                total_toks += float(vt)
            layer_prune_ppl = float(math.exp(total_loss / max(total_toks, 1)))
            print(f"  Post-layer-prune val ppl: {layer_prune_ppl:.2f} (was {last_val_ppl:.2f})" if last_val_ppl is not None else f"  Post-layer-prune val ppl: {layer_prune_ppl:.2f}")

            # Rebuild prune mask if weight pruning was already applied
            if prune_mask is not None:
                prune_mask = jax.tree.map(lambda w: (jnp.abs(w) > 1e-8).astype(jnp.float32), matched_params)
                prune_mask = jax_utils.replicate(prune_mask)

            state = jax_utils.replicate(state)
            ema_params = jax_utils.replicate(ema_params)
            print(f"Continuing training with {config.num_encoder_layers}+{config.num_decoder_layers} layers...\n")

        losses = []
        batches = get_batches(enc_inputs, dec_inputs, dec_targets, effective_batch_size)
        pbar = tqdm(batches, total=num_batches, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for src, tgt_in, tgt_out in pbar:
            rng, dropout_rng = jax.random.split(rng)
            dropout_rngs = jax.random.split(dropout_rng, num_devices)

            t0 = time.perf_counter()

            state, ema_params, loss, grad_norm = p_train_step(
                state,
                ema_params,
                shard_batch(src, num_devices),
                shard_batch(tgt_in, num_devices),
                shard_batch(tgt_out, num_devices),
                causal_mask,
                dropout_rngs,
            )

            # Re-apply prune mask to keep pruned weights at zero
            if prune_mask is not None:
                state = state.replace(params=jax.tree.map(lambda w, m: w * m, state.params, prune_mask))
                ema_params = jax.tree.map(lambda w, m: w * m, ema_params, prune_mask)

            loss_val = float(loss[0])
            losses.append(loss_val)
            global_step += 1
            dt = time.perf_counter() - t0
            eval_every = getattr(args, "eval_every", 100)
            if global_step % eval_every == 0 or global_step == total_steps:
                eval_params = jax_utils.unreplicate(ema_params)
                val_causal = make_causal_mask(args.max_dec_len)
                total_loss, total_toks = 0.0, 0.0
                for vb in get_batches(val_enc, val_dec_in, val_dec_tgt, args.batch_size, shuffle=False):
                    vl, vt = val_loss_fn(eval_params, vb[0], vb[1], vb[2], val_causal)
                    total_loss += float(vl)
                    total_toks += float(vt)
                last_val_ppl = float(math.exp(total_loss / max(total_toks, 1)))
            pbar.set_postfix(loss=f"{loss_val:.4f}", val_ppl=f"{last_val_ppl:.2f}" if last_val_ppl is not None else "?")

            if use_wandb:
                log_dict = {
                    "train/loss": loss_val,
                    "train/grad_norm": float(grad_norm[0]),
                    "train/adam_lr": float(adam_schedule(global_step)),
                    "train/muon_lr": float(muon_schedule(global_step)),
                    "train/tokens_per_sec": tokens_per_batch / dt,
                    "train/step": global_step,
                }
                if global_step % eval_every == 0 or global_step == total_steps:
                    log_dict["val/ppl"] = last_val_ppl
                wandb.log(log_dict)

        epoch_avg_loss = sum(losses) / len(losses) if losses else float("nan")
        final_loss = losses[-1] if losses else float("nan")
        final_ppl = math.exp(final_loss) if not math.isnan(final_loss) else float("nan")

        # Compute val perplexity at end of epoch
        eval_params = jax_utils.unreplicate(ema_params)
        val_causal = make_causal_mask(args.max_dec_len)
        total_loss, total_toks = 0.0, 0.0
        for vb in get_batches(val_enc, val_dec_in, val_dec_tgt, args.batch_size, shuffle=False):
            vl, vt = val_loss_fn(eval_params, vb[0], vb[1], vb[2], val_causal)
            total_loss += float(vl)
            total_toks += float(vt)
        last_val_ppl = float(math.exp(total_loss / max(total_toks, 1)))

        epoch_params = jax.tree.map(np.array, eval_params)
        total_params = sum(x.size for x in jax.tree.leaves(epoch_params))
        near_zero = sum(int(np.sum(np.abs(x) < 1e-6)) for x in jax.tree.leaves(epoch_params))
        sparsity = near_zero / total_params * 100

        print(f"\n  Epoch {epoch + 1}/{args.epochs} report:")
        print(f"    Avg loss:       {epoch_avg_loss:.4f}")
        print(f"    Final loss:     {final_loss:.4f}")
        print(f"    Perplexity:     {final_ppl:.2f}")
        print(f"    Val perplexity: {last_val_ppl:.2f}")
        print(f"    Weight sparsity: {sparsity:.2f}% ({near_zero:,}/{total_params:,} near-zero)")

        if use_wandb:
            wandb.log({
                "epoch/avg_loss": epoch_avg_loss,
                "epoch/final_loss": final_loss,
                "epoch/val_ppl": last_val_ppl,
                "epoch/weight_sparsity": sparsity,
                "epoch": epoch + 1,
            })

        ckpt_name = f"needle_{args.num_layers}_{args.d_model}_{global_step}.pkl"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        params_np = jax.tree.map(np.array, jax_utils.unreplicate(ema_params))

        with open(ckpt_path, "wb") as f:
            pickle.dump({"params": params_np, "config": config.__dict__}, f)
        print(f"Saved checkpoint: {ckpt_path}")

    if use_wandb:
        wandb.finish()
    print("\nTraining complete.")


def sweep(args):
    import wandb
    import yaml

    with open(args.sweep_config) as f:
        sweep_config = yaml.safe_load(f)

    def sweep_train():
        wandb.init(project=args.project)
        for key, value in dict(wandb.config).items():
            if hasattr(args, key):
                setattr(args, key, value)
        args.wandb = True
        train(args)
        wandb.finish()

    sweep_id = wandb.sweep(sweep_config, project=args.project)
    wandb.agent(sweep_id, function=sweep_train, count=args.count)


def parse_args():
    parser = argparse.ArgumentParser(description="Train encoder-decoder transformer on TinyStories")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-enc-len", type=int, default=128)
    parser.add_argument("--max-dec-len", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
