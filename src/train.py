import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from tqdm import tqdm

from .data import (
    PrefetchIterator, count_batches, get_batches, get_tokenizer, load_prepared_data,
    prepare_audio_toolcall_val,
)
from .model import EncoderDecoderTransformer, make_causal_mask, make_mel_padding_mask, make_padding_mask
from .train_utils import (
    count_params,
    create_config_from_args,
    create_train_state,
    load_checkpoint,
    quantize_params,
    save_checkpoint,
    shard_batch,
    wsd_schedule,
    zero_non_decoder_grads,
)


_GROUP_SIZE = 32


def _text_loss_fn(state, params, src, tgt_in, tgt_out, causal_mask, rng, loss_mask):
    pad_id = 0
    src_mask = make_padding_mask(src, pad_id)
    tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
    logits, slot_div = state.apply_fn(
        {"params": quantize_params(params, group_size=_GROUP_SIZE)},
        src,
        tgt_in,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        deterministic=False,
        method="forward_masked",
        rngs={"dropout": rng},
    )
    logits_f32 = logits.astype(jnp.float32)
    token_loss = optax.softmax_cross_entropy_with_integer_labels(logits_f32, tgt_out)
    ce_loss = jnp.sum(token_loss * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)
    z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits_f32, axis=-1) ** 2)
    div_loss = 1e-4 * slot_div
    return ce_loss + z_loss + div_loss


def _train_step(state, ema_params, src, tgt_in, tgt_out, causal_mask, rng, loss_mask):
    ema_decay = 0.999
    loss, grads = jax.value_and_grad(
        lambda p: _text_loss_fn(state, p, src, tgt_in, tgt_out, causal_mask, rng, loss_mask)
    )(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    grads = zero_non_decoder_grads(grads)
    loss = jax.lax.pmean(loss, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    ema_params = jax.tree.map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema_params, state.params)
    return state, ema_params, loss, grad_norm


def _make_p_train_step():
    return jax.pmap(_train_step, axis_name="batch", donate_argnums=(0, 1))


def _speech_replay_loss_fn(state, params, mel, tgt_in, tgt_out, causal_mask, rng, loss_mask):
    """Speech transcription loss for decoder-only replay (encoder frozen via zero_non_decoder_grads)."""
    pad_id = 0
    src_mask = make_mel_padding_mask(mel)
    tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
    logits, _ = state.apply_fn(
        {"params": quantize_params(params, group_size=_GROUP_SIZE)},
        mel,
        tgt_in,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        deterministic=False,
        method="forward_speech_masked",
        rngs={"dropout": rng},
    )
    logits_f32 = logits.astype(jnp.float32)
    token_loss = optax.softmax_cross_entropy_with_integer_labels(logits_f32, tgt_out)
    ce_loss = jnp.sum(token_loss * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)
    z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits_f32, axis=-1) ** 2)
    return ce_loss + z_loss


def _speech_replay_step(state, ema_params, mel, tgt_in, tgt_out, causal_mask, rng, loss_mask):
    ema_decay = 0.999
    loss, grads = jax.value_and_grad(
        lambda p: _speech_replay_loss_fn(state, p, mel, tgt_in, tgt_out, causal_mask, rng, loss_mask)
    )(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    grads = zero_non_decoder_grads(grads)
    loss = jax.lax.pmean(loss, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    ema_params = jax.tree.map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema_params, state.params)
    return state, ema_params, loss, grad_norm


def _make_p_speech_replay_step():
    return jax.pmap(_speech_replay_step, axis_name="batch", donate_argnums=(0, 1))


def _make_val_loss_fn(apply_fn):
    @jax.jit
    def val_loss_batch(params, src, tgt_in, tgt_out, causal_mask, loss_mask):
        pad_id = 0
        src_mask = make_padding_mask(src, pad_id)
        tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
        logits = apply_fn({"params": params}, src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tgt_out)
        return jnp.sum(loss * loss_mask), jnp.sum(loss_mask)

    return val_loss_batch


def _evaluate_val_ppl(val_loss_fn, params, val_enc, val_dec_in, val_dec_tgt, val_loss_mask,
                      batch_size, max_dec_len, max_eval_samples=None):
    val_causal = make_causal_mask(max_dec_len)
    total_loss = 0.0
    total_toks = 0.0
    seen = 0
    for batch in get_batches(val_enc, val_dec_in, val_dec_tgt, batch_size, shuffle=False, loss_mask=val_loss_mask):
        if max_eval_samples is not None and seen >= max_eval_samples:
            break
        src, dec_in, dec_tgt, lm = batch
        vl, vt = val_loss_fn(params, src, dec_in, dec_tgt, val_causal, lm)
        total_loss += float(vl)
        total_toks += float(vt)
        seen += len(src)
    return float(math.exp(min(total_loss / max(total_toks, 1.0), 20.0)))


def _filter_nonempty(val_enc, val_dec_in, val_dec_tgt, val_loss_mask, min_answer_tokens=4):
    """Return val arrays filtered to examples with at least min_answer_tokens answer tokens."""
    answer_len = np.array(val_loss_mask).sum(axis=1)
    mask = answer_len >= min_answer_tokens
    idx = np.where(mask)[0]
    return (val_enc[idx], val_dec_in[idx], val_dec_tgt[idx], val_loss_mask[idx])


def _make_speech_val_loss_fn(apply_fn):
    """Val loss using frozen speech encoder path (forward_speech_masked)."""
    @jax.jit
    def speech_val_loss_batch(params, mel, tgt_in, tgt_out, causal_mask, loss_mask):
        pad_id = 0
        src_mask = make_mel_padding_mask(mel)
        tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
        logits, _ = apply_fn(
            {"params": params},
            mel,
            tgt_in,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            deterministic=True,
            method="forward_speech_masked",
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tgt_out)
        return jnp.sum(loss * loss_mask), jnp.sum(loss_mask)

    return speech_val_loss_batch


def _evaluate_audio_toolcall_ppl(speech_val_loss_fn, params, audio_val_data,
                                   batch_size, max_dec_len, max_eval_samples=None):
    """Evaluate tool-call PPL using audio (mel) encoder input — the correct speech transfer metric.

    Uses speech query audio (from dataset `audio` field) → memory slots → tool call answer.
    Same decoder targets as the text val path, so directly comparable.
    """
    val_causal = make_causal_mask(max_dec_len)
    mels = audio_val_data["mels"]
    dec_in = audio_val_data["dec_inputs"]
    dec_tgt = audio_val_data["dec_targets"]
    loss_mask = audio_val_data["loss_mask"]

    total_loss = 0.0
    total_toks = 0.0
    seen = 0
    n = len(mels)
    for start in range(0, n, batch_size):
        if max_eval_samples is not None and seen >= max_eval_samples:
            break
        end = min(start + batch_size, n)
        mel_b = mels[start:end]
        di_b = dec_in[start:end]
        dt_b = dec_tgt[start:end]
        lm_b = loss_mask[start:end]
        vl, vt = speech_val_loss_fn(params, mel_b, di_b, dt_b, val_causal, lm_b)
        total_loss += float(vl)
        total_toks += float(vt)
        seen += len(mel_b)
    if total_toks == 0:
        return None
    return float(math.exp(min(total_loss / total_toks, 20.0)))


def train(args):
    global _GROUP_SIZE
    _GROUP_SIZE = getattr(args, "group_size", 32)

    num_devices = jax.local_device_count()
    use_wandb = getattr(args, "wandb", False)
    if use_wandb:
        import wandb
        if wandb.run is None:
            wandb.init(project="needle-stage2", config=vars(args))

    print(f"\n[1/3] Loading tokenizer...")
    tokenizer = get_tokenizer(max_samples=args.max_samples)
    del tokenizer

    print(f"\n[2/3] Loading prepared tool-call data...")
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
    nonempty_val_enc, nonempty_val_dec_in, nonempty_val_dec_tgt, nonempty_val_loss_mask = \
        _filter_nonempty(val_enc, val_dec_in, val_dec_tgt, val_loss_mask)
    if args.max_samples is not None:
        keep = min(args.max_samples, len(enc_inputs))
        enc_inputs = enc_inputs[:keep]
        dec_inputs = dec_inputs[:keep]
        dec_targets = dec_targets[:keep]
        train_loss_mask = train_loss_mask[:keep]
    print(f"      {len(enc_inputs):,} train / {len(val_enc):,} val tool-call pairs "
          f"({len(nonempty_val_enc):,} non-empty val)")

    # Load audio+tool-call val data for speech transfer eval
    # (speech query audio → same tool-call targets as text val, measures audio→tool-call PPL)
    n_mels = getattr(args, "n_mels", 80)
    max_mel_len = getattr(args, "max_mel_len", 1024)
    print(f"      Preparing audio+tool-call val data for speech transfer eval...")
    audio_toolcall_val = prepare_audio_toolcall_val(val_data, n_mels=n_mels, max_mel_len=max_mel_len)
    if audio_toolcall_val is not None:
        print(f"      {len(audio_toolcall_val['mels']):,} audio+tool-call val pairs ready")
    else:
        print(f"      Audio+tool-call val data unavailable")

    print(f"\n[3/3] Building model...")
    resume_checkpoint = getattr(args, "checkpoint", None)
    if resume_checkpoint:
        ckpt_params, config, _ = load_checkpoint(resume_checkpoint)
        print(f"      loaded {resume_checkpoint}")
    else:
        config = create_config_from_args(args, n_mels=getattr(args, "n_mels", 80))
        ckpt_params = None

    effective_batch_size = args.batch_size * num_devices
    batches_per_epoch = count_batches(len(enc_inputs), effective_batch_size)
    total_steps = batches_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    scaled_lr = args.lr * num_devices
    muon_lr = getattr(args, "muon_lr", 0.02) * math.sqrt(num_devices)
    state = create_train_state(init_rng, config, scaled_lr, muon_lr, total_steps, warmup_steps)
    if ckpt_params is not None:
        if getattr(args, "reinit_decoder", False):
            # Keep encoder/embedding/mel_proj from checkpoint; reinitialise decoder from scratch.
            fresh_params = state.params
            merged = {k: (ckpt_params[k] if k != "decoder" else fresh_params[k])
                      for k in ckpt_params}
            state = state.replace(params=merged)
            print(f"      decoder reinitialised from scratch (encoder weights kept)")
        else:
            state = state.replace(params=ckpt_params)

    val_loss_fn = _make_val_loss_fn(state.apply_fn)
    speech_val_loss_fn = _make_speech_val_loss_fn(state.apply_fn) if audio_toolcall_val is not None else None
    p_train_step = _make_p_train_step()
    speech_replay_every = getattr(args, "speech_replay_every", 0)
    p_speech_replay_step = _make_p_speech_replay_step() if speech_replay_every > 0 else None

    ema_params = jax.tree.map(jnp.copy, state.params)
    state = jax_utils.replicate(state)
    ema_params = jax_utils.replicate(ema_params)

    param_count = count_params(jax_utils.unreplicate(state).params)
    print(f"\n  Parameters    {param_count:,}")
    print(f"  d_model       {config.d_model}")
    print(f"  Heads         {config.num_heads} ({config.num_kv_heads} KV)")
    print(f"  Layers        {config.num_encoder_layers} enc / {config.num_decoder_layers} dec")
    print(f"  Encoder       frozen during Stage 2")
    print(f"  Batch         {args.batch_size} x {num_devices} = {effective_batch_size}")
    print(f"  Total steps   {total_steps:,}\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    causal_mask = jnp.broadcast_to(
        make_causal_mask(args.max_dec_len),
        (num_devices, 1, args.max_dec_len, args.max_dec_len),
    )
    adam_schedule = wsd_schedule(scaled_lr, total_steps, warmup_steps)
    muon_schedule = wsd_schedule(muon_lr, total_steps, warmup_steps)

    global_step = 0
    last_val_ppl = None
    last_nonempty_ppl = None
    last_speech_ppl = None
    best_nonempty_ppl = float("inf")
    eval_every = getattr(args, "eval_every", 1000)

    # Speech replay (optional, disabled by default)
    speech_replay_iter = None

    for epoch in range(args.epochs):
        losses = []
        batch_iter = PrefetchIterator(
            lambda: get_batches(enc_inputs, dec_inputs, dec_targets, effective_batch_size, loss_mask=train_loss_mask),
            prefetch=4,
        )
        pbar = tqdm(range(batches_per_epoch), desc=f"Stage 2 Epoch {epoch + 1}/{args.epochs}")

        for _ in pbar:
            src, tgt_in, tgt_out, lm = next(batch_iter)
            src_b = shard_batch(src, num_devices)
            tgt_in_b = shard_batch(tgt_in, num_devices)
            tgt_out_b = shard_batch(tgt_out, num_devices)
            lm_b = shard_batch(lm, num_devices)

            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_devices)
            t0 = time.perf_counter()
            state, ema_params, loss, grad_norm = p_train_step(
                state, ema_params, src_b, tgt_in_b, tgt_out_b, causal_mask, step_rngs, lm_b
            )
            dt = time.perf_counter() - t0
            loss_val = float(loss[0])
            grad_norm_val = float(grad_norm[0])
            losses.append(loss_val)
            global_step += 1

            # Speech replay step (decoder-only, prevents catastrophic forgetting)
            if (p_speech_replay_step is not None and speech_replay_every > 0
                    and global_step % speech_replay_every == 0):
                try:
                    speech_batch = next(speech_replay_iter)
                    mel_b, _, sp_tgt_in, sp_tgt_out, sp_lm = speech_batch
                    mel_b = shard_batch(mel_b, num_devices)
                    sp_tgt_in_b = shard_batch(sp_tgt_in, num_devices)
                    sp_tgt_out_b = shard_batch(sp_tgt_out, num_devices)
                    sp_lm_b = shard_batch(sp_lm, num_devices)
                    rng, sp_rng = jax.random.split(rng)
                    sp_rngs = jax.random.split(sp_rng, num_devices)
                    state, ema_params, _, _ = p_speech_replay_step(
                        state, ema_params, mel_b, sp_tgt_in_b, sp_tgt_out_b, causal_mask, sp_rngs, sp_lm_b
                    )
                except StopIteration:
                    pass

            if global_step % eval_every == 0 or global_step == total_steps:
                eval_params = jax_utils.unreplicate(ema_params)
                last_val_ppl = _evaluate_val_ppl(
                    val_loss_fn,
                    eval_params,
                    val_enc,
                    val_dec_in,
                    val_dec_tgt,
                    val_loss_mask,
                    args.batch_size,
                    args.max_dec_len,
                    max_eval_samples=getattr(args, "max_eval_samples", None),
                )
                last_nonempty_ppl = _evaluate_val_ppl(
                    val_loss_fn,
                    eval_params,
                    nonempty_val_enc,
                    nonempty_val_dec_in,
                    nonempty_val_dec_tgt,
                    nonempty_val_loss_mask,
                    args.batch_size,
                    args.max_dec_len,
                    max_eval_samples=getattr(args, "max_eval_samples", None),
                )
                if speech_val_loss_fn is not None:
                    last_speech_ppl = _evaluate_audio_toolcall_ppl(
                        speech_val_loss_fn,
                        eval_params,
                        audio_toolcall_val,
                        args.batch_size,
                        max_dec_len=args.max_dec_len,
                        max_eval_samples=getattr(args, "max_eval_samples", None),
                    )
                if last_nonempty_ppl < best_nonempty_ppl:
                    best_nonempty_ppl = last_nonempty_ppl
                    best_ckpt = os.path.join(args.checkpoint_dir, f"needle_stage2_{config.num_encoder_layers}_{config.d_model}_best.pkl")
                    save_checkpoint(best_ckpt, eval_params, config, extra={"stage": "stage2", "step": global_step, "val_ppl": last_val_ppl, "nonempty_val_ppl": best_nonempty_ppl, "speech_ppl": last_speech_ppl})

            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                ppl=f"{last_val_ppl:.2f}" if last_val_ppl is not None else "?",
                ne_ppl=f"{last_nonempty_ppl:.2f}" if last_nonempty_ppl is not None else "?",
                aud_ppl=f"{last_speech_ppl:.2f}" if last_speech_ppl is not None else "?",
            )

            if use_wandb:
                wandb.log(
                    {
                        "train/loss": loss_val,
                        "train/grad_norm": grad_norm_val,
                        "train/adam_lr": float(adam_schedule(global_step)),
                        "train/muon_lr": float(muon_schedule(global_step)),
                        "train/tokens_per_sec": effective_batch_size * (args.max_enc_len + args.max_dec_len) / max(dt, 1e-6),
                        "train/step": global_step,
                        **({"val/ppl": last_val_ppl, "val/nonempty_ppl": last_nonempty_ppl}
                           if last_val_ppl is not None and (global_step % eval_every == 0 or global_step == total_steps) else {}),
                    }
                )

        batch_iter.close()

        eval_params = jax_utils.unreplicate(ema_params)
        val_ppl = _evaluate_val_ppl(
            val_loss_fn,
            eval_params,
            val_enc,
            val_dec_in,
            val_dec_tgt,
            val_loss_mask,
            args.batch_size,
            args.max_dec_len,
            max_eval_samples=getattr(args, "max_eval_samples", None),
        )
        nonempty_ppl = _evaluate_val_ppl(
            val_loss_fn,
            eval_params,
            nonempty_val_enc,
            nonempty_val_dec_in,
            nonempty_val_dec_tgt,
            nonempty_val_loss_mask,
            args.batch_size,
            args.max_dec_len,
            max_eval_samples=getattr(args, "max_eval_samples", None),
        )
        speech_ppl = None
        if speech_val_loss_fn is not None:
            speech_ppl = _evaluate_audio_toolcall_ppl(
                speech_val_loss_fn,
                eval_params,
                audio_toolcall_val,
                args.batch_size,
                max_dec_len=args.max_dec_len,
                max_eval_samples=getattr(args, "max_eval_samples", None),
            )
        ckpt_name = f"needle_stage2_{config.num_encoder_layers}_{config.d_model}_{global_step}.pkl"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        save_checkpoint(ckpt_path, eval_params, config, extra={"stage": "stage2"})

        print(f"\n  Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train loss     {sum(losses) / max(len(losses), 1):.4f}")
        print(f"  Val ppl        {val_ppl:.2f}  (all)   {nonempty_ppl:.2f}  (non-empty only)")
        if speech_ppl is not None:
            print(f"  Audio val ppl  {speech_ppl:.2f}  (speech query → tool call)")
        print(f"  Checkpoint     {ckpt_path}\n")

    if use_wandb:
        wandb.finish()
    print("Stage 2 training complete.")
