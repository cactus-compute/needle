import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from tqdm import tqdm

from .data import PrefetchIterator, count_batches, get_batches, get_tokenizer, load_prepared_data
from .model import EncoderDecoderTransformer, make_causal_mask, make_padding_mask
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
    if args.max_samples is not None:
        keep = min(args.max_samples, len(enc_inputs))
        enc_inputs = enc_inputs[:keep]
        dec_inputs = dec_inputs[:keep]
        dec_targets = dec_targets[:keep]
        train_loss_mask = train_loss_mask[:keep]
    print(f"      {len(enc_inputs):,} train / {len(val_enc):,} val tool-call pairs")

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
        state = state.replace(params=ckpt_params)

    val_loss_fn = _make_val_loss_fn(state.apply_fn)
    p_train_step = _make_p_train_step()

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
    eval_every = getattr(args, "eval_every", 1000)

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

            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                ppl=f"{last_val_ppl:.2f}" if last_val_ppl is not None else "?",
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
                        **({"val/ppl": last_val_ppl} if last_val_ppl is not None and (global_step % eval_every == 0 or global_step == total_steps) else {}),
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
        ckpt_name = f"needle_stage2_{config.num_encoder_layers}_{config.d_model}_{global_step}.pkl"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        save_checkpoint(ckpt_path, eval_params, config, extra={"stage": "stage2"})

        print(f"\n  Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train loss     {sum(losses) / max(len(losses), 1):.4f}")
        print(f"  Val ppl        {val_ppl:.2f}")
        print(f"  Checkpoint     {ckpt_path}\n")

    if use_wandb:
        wandb.finish()
    print("Stage 2 training complete.")
