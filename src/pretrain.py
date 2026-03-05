import math
import os
import pickle
import time
from itertools import cycle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from flax.training import train_state
from tqdm import tqdm

from .data import (
    get_batches,
    get_speech_batches,
    get_tokenizer,
    load_librilight,
    load_tinystories,
    prepare_speech_masked,
    prepare_text_masked,
)
from .model import Discriminator, PretrainingModel, TransformerConfig
from .train import _param_labels, _wsd_schedule, scale_by_muon, shard_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lambda_adv_schedule(step, total_steps, warmup_frac=0.10, lambda_final=0.10):
    """Ramp adversarial weight from 0 → lambda_final over first warmup_frac of training."""
    warmup_end = jnp.maximum(jnp.int32(int(total_steps * warmup_frac)), 1)
    return jnp.minimum(step.astype(jnp.float32) / warmup_end.astype(jnp.float32),
                       1.0) * lambda_final


def _norm_mel(mel):
    """Per-clip (B,1,1) normalisation of mel. Uses max(std, 0.01) floor for silent clips."""
    mean = mel.mean(axis=(1, 2), keepdims=True)
    std  = jnp.maximum(mel.std(axis=(1, 2), keepdims=True), 0.01)
    return (mel - mean) / std


# ---------------------------------------------------------------------------
# Train state creation
# ---------------------------------------------------------------------------

def create_pretrain_state(rng, config, learning_rate, muon_lr, total_steps, warmup_steps,
                          num_audio_dec_layers=2, clip_frames=1000):
    model = PretrainingModel(config, num_audio_dec_layers=num_audio_dec_layers,
                             clip_frames=clip_frames)
    rng, init_rng = jax.random.split(rng)
    dummy_src = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    dummy_dec = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    dummy_mel = jnp.zeros((1, clip_frames, config.n_mels), dtype=jnp.float32)
    variables = model.init({"params": init_rng}, dummy_src, dummy_dec, dummy_mel)

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
        optax.multi_transform({"muon": muon_opt, "adam": adam_opt}, _param_labels),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )


def create_disc_state(rng, config, disc_lr):
    disc = Discriminator(d_model=config.d_model)
    dummy = jnp.zeros((1, config.num_memory_slots, config.d_model), dtype=jnp.float32)
    variables = disc.init({"params": rng}, dummy)
    tx = optax.adamw(learning_rate=disc_lr, b1=0.9, b2=0.999)
    return train_state.TrainState.create(
        apply_fn=disc.apply,
        params=variables["params"],
        tx=tx,
    )


# ---------------------------------------------------------------------------
# pmap step functions
# ---------------------------------------------------------------------------

_EMA_DECAY = 0.999


def _train_step_text(main_state, ema_params, masked_enc, dec_in, dec_tgt):
    pad_id = 0

    def loss_fn(params):
        logits, slot_div = main_state.apply_fn(
            {"params": params}, masked_enc, dec_in, method="forward_text"
        )
        logits_f32 = logits.astype(jnp.float32)
        mask = (dec_tgt != pad_id).astype(jnp.float32)
        ce_loss = jnp.sum(
            optax.softmax_cross_entropy_with_integer_labels(logits_f32, dec_tgt) * mask
        ) / jnp.maximum(jnp.sum(mask), 1.0)
        z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits_f32, axis=-1) ** 2)
        div_loss = 1e-4 * slot_div
        total = ce_loss + z_loss + div_loss
        return total, (ce_loss, slot_div)

    (total_loss, (ce_loss, slot_div)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        main_state.params
    )
    grads = jax.lax.pmean(grads, axis_name="batch")
    total_loss = jax.lax.pmean(total_loss, axis_name="batch")
    ce_loss = jax.lax.pmean(ce_loss, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    main_state = main_state.apply_gradients(grads=grads)
    ema_params = jax.tree.map(
        lambda e, p: _EMA_DECAY * e + (1 - _EMA_DECAY) * p, ema_params, main_state.params
    )
    return main_state, ema_params, total_loss, ce_loss, grad_norm


def _train_step_audio_disc(main_state, ema_params, disc_state,
                            masked_mel, orig_mel, masked_enc_real,
                            step, total_steps, lambda_final):
    # --- Main network: audio recon + generator adversarial loss ---
    def main_loss_fn(params):
        mel_pred, z_audio, u_audio, slot_div = main_state.apply_fn(
            {"params": params}, masked_mel, method="forward_audio"
        )
        orig_norm = _norm_mel(orig_mel)
        pred_norm = _norm_mel(mel_pred)
        l_audio = jnp.mean((pred_norm - orig_norm) ** 2)

        lam = lambda_adv_schedule(step, total_steps, lambda_final=lambda_final)
        disc_logits = disc_state.apply_fn(
            {"params": jax.lax.stop_gradient(disc_state.params)}, u_audio
        )
        l_adv_gen = -jnp.mean(jnp.log(1.0 - jax.nn.sigmoid(disc_logits) + 1e-8))
        div_loss = 1e-4 * slot_div
        total = l_audio + lam * l_adv_gen + div_loss
        return total, (l_audio, l_adv_gen, slot_div, u_audio, z_audio)

    (total_loss, (l_audio, l_adv_gen, slot_div, u_audio, z_audio)), grads = (
        jax.value_and_grad(main_loss_fn, has_aux=True)(main_state.params)
    )
    grads = jax.lax.pmean(grads, axis_name="batch")
    total_loss = jax.lax.pmean(total_loss, axis_name="batch")
    l_audio = jax.lax.pmean(l_audio, axis_name="batch")
    l_adv_gen = jax.lax.pmean(l_adv_gen, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    main_state = main_state.apply_gradients(grads=grads)
    ema_params = jax.tree.map(
        lambda e, p: _EMA_DECAY * e + (1 - _EMA_DECAY) * p, ema_params, main_state.params
    )

    # --- Discriminator update ---
    z_text_sg = jax.lax.stop_gradient(
        main_state.apply_fn({"params": main_state.params}, masked_enc_real,
                            method="encode_text")
    )
    u_audio_sg = jax.lax.stop_gradient(u_audio)

    def disc_loss_fn(disc_params):
        real_logits = disc_state.apply_fn({"params": disc_params}, z_text_sg)   # label 0
        fake_logits = disc_state.apply_fn({"params": disc_params}, u_audio_sg)  # label 1
        l_real = -jnp.mean(jnp.log(1.0 - jax.nn.sigmoid(real_logits) + 1e-8))
        l_fake = -jnp.mean(jnp.log(jax.nn.sigmoid(fake_logits) + 1e-8))
        l_disc = (l_real + l_fake) / 2.0
        real_correct = (jax.nn.sigmoid(real_logits) < 0.5).astype(jnp.float32)
        fake_correct = (jax.nn.sigmoid(fake_logits) >= 0.5).astype(jnp.float32)
        acc = (jnp.mean(real_correct) + jnp.mean(fake_correct)) / 2.0
        return l_disc, acc

    (l_disc, disc_acc), disc_grads = jax.value_and_grad(disc_loss_fn, has_aux=True)(
        disc_state.params
    )
    disc_grads = jax.lax.pmean(disc_grads, axis_name="batch")
    l_disc   = jax.lax.pmean(l_disc,   axis_name="batch")
    disc_acc = jax.lax.pmean(disc_acc, axis_name="batch")
    disc_state = disc_state.apply_gradients(grads=disc_grads)

    text_latent_norm  = jnp.mean(jnp.linalg.norm(z_text_sg.astype(jnp.float32), axis=-1))
    audio_latent_norm = jnp.mean(jnp.linalg.norm(z_audio.astype(jnp.float32), axis=-1))

    return (main_state, ema_params, disc_state,
            total_loss, l_audio, l_adv_gen, l_disc, disc_acc, grad_norm,
            text_latent_norm, audio_latent_norm)


# ---------------------------------------------------------------------------
# JIT'd validation helpers (defined once, reused every epoch)
# ---------------------------------------------------------------------------

def _make_val_fns(apply_fn):
    @jax.jit
    def text_val_loss(params, enc, din, dtgt):
        logits, _ = apply_fn({"params": params}, enc, din, method="forward_text")
        mask = (dtgt != 0).astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), dtgt
        )
        return jnp.sum(loss * mask), jnp.sum(mask)

    @jax.jit
    def audio_val_mse(params, mmel, omel):
        # Unnormalised MSE (raw mel scale, for interpretability)
        mel_pred, _, _, _ = apply_fn({"params": params}, mmel, method="forward_audio")
        return jnp.mean((mel_pred - omel) ** 2)

    return text_val_loss, audio_val_mse


# ---------------------------------------------------------------------------
# Main pretrain() function
# ---------------------------------------------------------------------------

def pretrain(args):
    num_devices = jax.local_device_count()
    effective_batch = args.batch_size * num_devices
    no_speech = getattr(args, "no_speech", False)
    max_steps = getattr(args, "max_steps", None)
    lambda_final = getattr(args, "lambda_adv", 0.10)

    use_wandb = getattr(args, "wandb", False)
    run_name = getattr(args, "run_name", None)
    if use_wandb:
        import wandb
        if wandb.run is None:
            wandb.init(project="needle-pretrain", name=run_name, config=vars(args))

    print(f"\n[1/5] Detecting devices...")
    print(f"      {num_devices} device(s) — effective batch {effective_batch}")

    print(f"\n[2/5] Loading tokenizer...")
    tokenizer = get_tokenizer(max_samples=getattr(args, "max_samples", None))

    print(f"\n[3/5] Loading text data (TinyStories)...")
    ds     = load_tinystories("train",      max_samples=getattr(args, "max_samples", None))
    val_ds = load_tinystories("validation", max_samples=2000)

    max_enc_len = getattr(args, "max_enc_len", 128)
    max_dec_len = getattr(args, "max_dec_len", 128)
    mask_ratio  = getattr(args, "mask_ratio", 0.50)

    masked_enc, dec_in, dec_tgt = prepare_text_masked(
        ds, tokenizer, max_enc_len=max_enc_len, max_dec_len=max_dec_len,
        mask_ratio=mask_ratio,
    )
    val_menc, val_dec_in, val_dec_tgt = prepare_text_masked(
        val_ds, tokenizer, max_enc_len=max_enc_len, max_dec_len=max_dec_len,
        mask_ratio=mask_ratio,
    )

    clip_frames = 1000
    val_masked_mel = val_orig_mel = None
    if not no_speech:
        print(f"\n[4/5] Loading speech data (Libri Light)...")
        librilight_subset = getattr(args, "librilight_subset", "small")
        speech_ds = load_librilight(
            subset=librilight_subset,
            max_samples=getattr(args, "max_speech_samples", None),
        )
        s_masked_np, s_orig_np, _ = prepare_speech_masked(
            speech_ds, n_mels=args.n_mels, clip_frames=clip_frames,
            mask_ratio=mask_ratio,
        )
        val_speech_ds = load_librilight(subset="small", max_samples=500)
        val_masked_mel, val_orig_mel, _ = prepare_speech_masked(
            val_speech_ds, n_mels=args.n_mels, clip_frames=clip_frames,
            mask_ratio=mask_ratio,
        )
    else:
        print(f"\n[4/5] Speech disabled (--no-speech).")
        s_masked_np = s_orig_np = None

    print(f"\n[5/5] Building model and train states...")
    config = TransformerConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_kv_heads=getattr(args, "num_kv_heads", None) or args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_dec_layers,
        d_ff=getattr(args, "d_ff", None) or args.d_model * 4,
        max_seq_len=max(max_enc_len, max_dec_len),
        activation=getattr(args, "activation", "drelu"),
        num_memory_slots=getattr(args, "num_memory_slots", 64),
        n_mels=args.n_mels,
    )

    num_text_batches = len(masked_enc) // effective_batch
    total_steps = num_text_batches * args.epochs
    if max_steps:
        total_steps = min(total_steps, max_steps)
    warmup_steps = max(1, int(total_steps * getattr(args, "warmup_ratio", 0.05)))

    scaled_lr   = args.lr       * num_devices
    scaled_muon = args.muon_lr  * math.sqrt(num_devices)

    rng = jax.random.PRNGKey(getattr(args, "seed", 42))
    rng, init_rng, disc_rng = jax.random.split(rng, 3)

    main_state = create_pretrain_state(
        init_rng, config, scaled_lr, scaled_muon, total_steps, warmup_steps,
        num_audio_dec_layers=args.num_dec_layers,
        clip_frames=clip_frames,
    )
    disc_state = create_disc_state(disc_rng, config, args.disc_lr)

    ema_params = jax.tree.map(jnp.copy, main_state.params)
    main_state = jax_utils.replicate(main_state)
    ema_params = jax_utils.replicate(ema_params)
    disc_state = jax_utils.replicate(disc_state)

    param_count = sum(x.size for x in jax.tree.leaves(jax_utils.unreplicate(main_state).params))
    disc_param_count = sum(x.size for x in jax.tree.leaves(jax_utils.unreplicate(disc_state).params))

    print(f"\n  ─────────────────────────────────────")
    print(f"  Main params   {param_count:>12,}")
    print(f"  Disc params   {disc_param_count:>12,}")
    print(f"  d_model       {config.d_model:>12}")
    print(f"  Layers        {config.num_encoder_layers} enc / {config.num_decoder_layers} dec")
    print(f"  Memory slots  {config.num_memory_slots:>12}")
    print(f"  Total steps   {total_steps:>12,}")
    print(f"  Warmup steps  {warmup_steps:>12,}")
    print(f"  lambda_adv    {lambda_final:>12.3f}")
    print(f"  Speech        {'disabled' if no_speech else 'enabled':>12}")
    print(f"  Run name      {run_name or '(none)':>12}")
    print(f"  ─────────────────────────────────────\n")

    # Build pmap'd step functions (total_steps and lambda_final captured in closure)
    p_text_step = jax.pmap(_train_step_text, axis_name="batch", donate_argnums=(0, 1))
    _audio_disc_static = jax.pmap(
        lambda ms, ep, ds, mmel, omel, enc, step: _train_step_audio_disc(
            ms, ep, ds, mmel, omel, enc, step, total_steps, lambda_final
        ),
        axis_name="batch",
        donate_argnums=(0, 1, 2),
    )

    # Build JIT'd val functions (defined once outside epoch loop)
    apply_fn = jax_utils.unreplicate(main_state).apply_fn
    text_val_loss_fn, audio_val_mse_fn = _make_val_fns(apply_fn)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    global_step = 0
    consecutive_high_disc = 0
    consecutive_low_disc  = 0

    # Cycle through speech batches — reshuffles each epoch via new generator
    def _make_speech_iter():
        if no_speech or s_masked_np is None:
            return iter([])
        return cycle(get_speech_batches(s_masked_np, s_orig_np, effective_batch))

    speech_iter = _make_speech_iter()
    done = False

    for epoch in range(args.epochs):
        if done:
            break
        losses_text  = []
        losses_audio = []

        text_batches = get_batches(masked_enc, dec_in, dec_tgt, effective_batch)
        pbar = tqdm(text_batches, total=num_text_batches,
                    desc=f"Epoch {epoch + 1}/{args.epochs}")

        for enc_b, din_b, dtgt_b in pbar:
            if max_steps and global_step >= max_steps:
                done = True
                break

            t0 = time.perf_counter()
            enc_s  = shard_batch(enc_b,  num_devices)
            din_s  = shard_batch(din_b,  num_devices)
            dtgt_s = shard_batch(dtgt_b, num_devices)

            main_state, ema_params, total_loss, ce_loss, grad_norm = p_text_step(
                main_state, ema_params, enc_s, din_s, dtgt_s
            )
            ce_val = float(ce_loss[0])
            losses_text.append(ce_val)
            global_step += 1

            disc_acc_val = None

            # Audio step every speech_every text steps
            if not no_speech and global_step % args.speech_every == 0:
                mmel_b, omel_b = next(speech_iter)
                mmel_s = shard_batch(mmel_b, num_devices)
                omel_s = shard_batch(omel_b, num_devices)
                step_arr = jnp.broadcast_to(jnp.int32(global_step), (num_devices,))

                (main_state, ema_params, disc_state,
                 audio_total, l_audio, l_adv_gen, l_disc, disc_acc, audio_gnorm,
                 text_lnorm, audio_lnorm) = _audio_disc_static(
                    main_state, ema_params, disc_state,
                    mmel_s, omel_s, enc_s, step_arr,
                )
                audio_val    = float(l_audio[0])
                disc_acc_val = float(disc_acc[0])
                l_disc_val   = float(l_disc[0])
                losses_audio.append(audio_val)

                # Collapse detection
                if disc_acc_val > 0.85:
                    consecutive_high_disc += 1
                    consecutive_low_disc   = 0
                elif disc_acc_val < 0.15:
                    consecutive_low_disc  += 1
                    consecutive_high_disc  = 0
                else:
                    consecutive_high_disc = consecutive_low_disc = 0

                if consecutive_high_disc >= 100:
                    print(f"\nWARNING: disc_accuracy > 0.85 for 100 audio steps "
                          f"(discriminator winning, encoder not aligning)")
                    consecutive_high_disc = 0
                if consecutive_low_disc >= 100:
                    print(f"\nWARNING: disc_accuracy < 0.15 for 100 audio steps "
                          f"(generator collapsed)")
                    consecutive_low_disc = 0

                if use_wandb:
                    import wandb
                    lam_val = float(lambda_adv_schedule(jnp.int32(global_step), total_steps,
                                                        lambda_final=lambda_final))
                    wandb.log({
                        "train/audio_recon_loss":  audio_val,
                        "train/disc_loss":         l_disc_val,
                        "train/disc_accuracy":     disc_acc_val,
                        "train/adv_loss_gen":      float(l_adv_gen[0]),
                        "train/lambda_adv":        lam_val,
                        "train/text_latent_norm":  float(text_lnorm[0]),
                        "train/audio_latent_norm": float(audio_lnorm[0]),
                        "train/step": global_step,
                    })

            dt = time.perf_counter() - t0
            postfix = {"ce": f"{ce_val:.4f}", "step/s": f"{1/dt:.1f}"}
            if disc_acc_val is not None:
                postfix["disc"] = f"{disc_acc_val:.3f}"
            pbar.set_postfix(**postfix)

            if use_wandb:
                import wandb
                wandb.log({
                    "train/text_recon_loss": ce_val,
                    "train/grad_norm":       float(grad_norm[0]),
                    "train/step":            global_step,
                })

        # ---- Epoch-end validation ----
        eval_params = jax_utils.unreplicate(ema_params)

        total_loss_v, total_toks = 0.0, 0.0
        for vb in get_batches(val_menc, val_dec_in, val_dec_tgt, args.batch_size, shuffle=False):
            vl, vt = text_val_loss_fn(eval_params, vb[0], vb[1], vb[2])
            total_loss_v += float(vl)
            total_toks   += float(vt)
        val_ppl = math.exp(min(total_loss_v / max(total_toks, 1), 20))

        val_audio_mse = None
        if not no_speech and val_masked_mel is not None:
            mse_vals = []
            for vmb, vob in get_speech_batches(val_masked_mel, val_orig_mel,
                                               args.batch_size, shuffle=False):
                mse_vals.append(float(audio_val_mse_fn(eval_params, vmb, vob)))
            val_audio_mse = float(np.mean(mse_vals))

        avg_text  = np.mean(losses_text)  if losses_text  else float("nan")
        avg_audio = np.mean(losses_audio) if losses_audio else float("nan")

        print(f"\n  ─────────────────────────────────────")
        print(f"  Epoch {epoch + 1}/{args.epochs}  (step {global_step}/{total_steps})")
        print(f"  Text CE (avg)  {avg_text:>12.4f}")
        if not math.isnan(avg_audio):
            print(f"  Audio MSE      {avg_audio:>12.4f}")
        print(f"  Val text ppl   {val_ppl:>12.2f}")
        if val_audio_mse is not None:
            print(f"  Val audio MSE  {val_audio_mse:>12.4f}  (unnorm, raw mel scale)")
        print(f"  ─────────────────────────────────────")

        if use_wandb:
            import wandb
            log_dict = {
                "val/text_recon_ppl":  val_ppl,
                "epoch/text_ce":       avg_text,
                "epoch": epoch + 1,
            }
            if not math.isnan(avg_audio):
                log_dict["epoch/audio_recon_loss"] = avg_audio
            if val_audio_mse is not None:
                log_dict["val/audio_recon_mse"] = val_audio_mse
            wandb.log(log_dict)

        # ---- Checkpoint ----
        params_np = jax.tree.map(np.array, eval_params)
        tag = run_name or f"{args.num_layers}L_{args.d_model}d"
        ckpt_name = f"pretrain_{tag}_{global_step}.pkl"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        with open(ckpt_path, "wb") as f:
            pickle.dump({"params": params_np, "config": config.__dict__, "step": global_step}, f)
        print(f"  Checkpoint: {ckpt_path}\n")
        del params_np

        if done:
            break

    if use_wandb:
        import wandb
        wandb.finish()
    print(f"Pretraining complete. ({global_step} steps)")
