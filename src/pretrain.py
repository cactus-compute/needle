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
    EMILIA_SPEECH_GCS_PREFIX,
    PrefetchIterator,
    count_batches,
    get_tokenizer,
    get_transcription_batches,
    load_emilia_speech_metadata,
    load_prepared_transcription_data,
    prepare_transcription_pairs,
)
from .model import make_causal_mask, make_mel_padding_mask, make_padding_mask
from .toucan import cache_toucan_examples, get_toucan_batches, load_toucan_contrastive_data
from .train_utils import (
    count_params,
    create_config_from_args,
    create_train_state,
    load_checkpoint,
    quantize_params,
    save_checkpoint,
    shard_batch,
    wsd_schedule,
)


_GROUP_SIZE = 32
_TOOL_CONTRASTIVE_WEIGHT = 1.0
_AUDIO_TEXT_CONTRASTIVE_WEIGHT = 1.0


def _assert_stage1_batch_shapes(mel, transcript_text, tgt_in, tgt_out, loss_mask,
                                tc_q, tc_tools, tc_labels, tc_tool_mask,
                                effective_batch_size, max_mel_len, n_mels,
                                max_enc_len, max_dec_len):
    expected = {
        "mel": (effective_batch_size, max_mel_len, n_mels),
        "transcript_text": (effective_batch_size, max_enc_len),
        "tgt_in": (effective_batch_size, max_dec_len),
        "tgt_out": (effective_batch_size, max_dec_len),
        "loss_mask": (effective_batch_size, max_dec_len),
        "tc_q": (effective_batch_size, max_enc_len),
    }
    actual = {
        "mel": mel.shape,
        "transcript_text": transcript_text.shape,
        "tgt_in": tgt_in.shape,
        "tgt_out": tgt_out.shape,
        "loss_mask": loss_mask.shape,
        "tc_q": tc_q.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"Unexpected {name} shape {actual[name]}, expected {shape}")

    if tc_tools.shape[0] != effective_batch_size or tc_tools.shape[2] != max_enc_len:
        raise ValueError(
            f"Unexpected tc_tools shape {tc_tools.shape}, expected "
            f"({effective_batch_size}, n_tools, {max_enc_len})"
        )
    if tc_labels.shape != tc_tool_mask.shape or tc_labels.shape[0] != effective_batch_size:
        raise ValueError(
            f"Unexpected Toucan label/mask shapes {tc_labels.shape} and {tc_tool_mask.shape}"
        )
    if tc_labels.shape[1] != tc_tools.shape[1]:
        raise ValueError(
            f"Toucan tool dimension mismatch: tc_tools={tc_tools.shape}, tc_labels={tc_labels.shape}"
        )


def _tool_contrastive_loss(state, params, contrastive_q, contrastive_tools, contrastive_labels, contrastive_tool_mask):
    pad_id = 0
    q_mask = make_padding_mask(contrastive_q, pad_id)
    q_slots = state.apply_fn(
        {"params": quantize_params(params, group_size=_GROUP_SIZE)},
        contrastive_q,
        src_mask=q_mask,
        method="encode",
    ).astype(jnp.float32)

    bsz, n_tools, seq_len = contrastive_tools.shape
    flat_tools = contrastive_tools.reshape(bsz * n_tools, seq_len)
    tool_mask = make_padding_mask(flat_tools, pad_id)
    t_slots = state.apply_fn(
        {"params": quantize_params(params, group_size=_GROUP_SIZE)},
        flat_tools,
        src_mask=tool_mask,
        method="encode",
    ).astype(jnp.float32)
    t_slots = t_slots.reshape(bsz, n_tools, q_slots.shape[1], q_slots.shape[2])

    scores = jnp.sum(q_slots[:, None, :, :] * t_slots, axis=(2, 3)) / q_slots.shape[1]
    loss = optax.sigmoid_binary_cross_entropy(scores, contrastive_labels)
    return jnp.sum(loss * contrastive_tool_mask) / jnp.maximum(jnp.sum(contrastive_tool_mask), 1.0)


def _siglip_pair_loss(scores):
    labels = 2.0 * jnp.eye(scores.shape[0], dtype=scores.dtype) - 1.0
    return jnp.mean(jax.nn.softplus(-labels * scores))


def _audio_text_contrastive_loss(state, params, contrastive_text, contrastive_mel):
    pad_id = 0
    text_mask = make_padding_mask(contrastive_text, pad_id)
    text_slots = state.apply_fn(
        {"params": quantize_params(params, group_size=_GROUP_SIZE)},
        contrastive_text,
        src_mask=text_mask,
        method="encode",
    ).astype(jnp.float32)

    mel_mask = make_mel_padding_mask(contrastive_mel)
    audio_slots = state.apply_fn(
        {"params": quantize_params(params, group_size=_GROUP_SIZE)},
        contrastive_mel,
        src_mask=mel_mask,
        deterministic=True,
        method="encode_speech",
    ).astype(jnp.float32)

    scores = jnp.einsum("bmd,cmd->bc", text_slots, audio_slots) / text_slots.shape[1]
    return _siglip_pair_loss(scores)


def _stage1_loss_fn(state, params, mel, transcript_text, tgt_in, tgt_out, causal_mask, rng, loss_mask,
                    contrastive_q, contrastive_tools, contrastive_labels, contrastive_tool_mask):
    pad_id = 0
    src_mask = make_mel_padding_mask(mel)
    tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
    spec_rng, drop_rng = jax.random.split(rng)
    logits, slot_div = state.apply_fn(
        {"params": quantize_params(params, group_size=_GROUP_SIZE)},
        mel,
        tgt_in,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        deterministic=False,
        method="forward_speech_masked",
        rngs={"specaugment": spec_rng, "dropout": drop_rng},
    )
    logits_f32 = logits.astype(jnp.float32)
    token_loss = optax.softmax_cross_entropy_with_integer_labels(logits_f32, tgt_out)
    ce_loss = jnp.sum(token_loss * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)
    z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits_f32, axis=-1) ** 2)
    div_loss = 1e-4 * slot_div
    at_loss = _audio_text_contrastive_loss(state, params, transcript_text, mel)
    tc_loss = _tool_contrastive_loss(
        state, params, contrastive_q, contrastive_tools, contrastive_labels, contrastive_tool_mask
    )
    return ce_loss + z_loss + div_loss + _AUDIO_TEXT_CONTRASTIVE_WEIGHT * at_loss + _TOOL_CONTRASTIVE_WEIGHT * tc_loss


def _train_step(state, ema_params, mel, transcript_text, tgt_in, tgt_out, causal_mask, rng, loss_mask,
                contrastive_q, contrastive_tools, contrastive_labels, contrastive_tool_mask):
    ema_decay = 0.999
    loss, grads = jax.value_and_grad(
        lambda p: _stage1_loss_fn(
            state,
            p,
            mel,
            transcript_text,
            tgt_in,
            tgt_out,
            causal_mask,
            rng,
            loss_mask,
            contrastive_q,
            contrastive_tools,
            contrastive_labels,
            contrastive_tool_mask,
        )
    )(state.params)
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
    def val_loss_batch(params, mel, tgt_in, tgt_out, causal_mask, loss_mask):
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

    return val_loss_batch


def _evaluate_val_ppl(val_loss_fn, params, prepared, batch_size, max_mel_len, n_mels,
                      gcs_prefix, max_dec_len, max_eval_samples=None):
    val_causal = make_causal_mask(max_dec_len)
    total_loss = 0.0
    total_toks = 0.0
    seen = 0
    for batch in get_transcription_batches(
        prepared,
        batch_size,
        max_mel_len=max_mel_len,
        n_mels=n_mels,
        gcs_prefix=gcs_prefix,
        shuffle=False,
        require_local_mels=True,
    ):
        if max_eval_samples is not None and seen >= max_eval_samples:
            break
        mel, _, tgt_in, tgt_out, lm = batch
        vl, vt = val_loss_fn(params, mel, tgt_in, tgt_out, val_causal, lm)
        total_loss += float(vl)
        total_toks += float(vt)
        seen += len(mel)
    return float(math.exp(min(total_loss / max(total_toks, 1.0), 20.0)))


def pretrain(args):
    global _GROUP_SIZE, _TOOL_CONTRASTIVE_WEIGHT, _AUDIO_TEXT_CONTRASTIVE_WEIGHT
    _GROUP_SIZE = getattr(args, "group_size", 32)
    _TOOL_CONTRASTIVE_WEIGHT = getattr(args, "tool_contrastive_weight", 1.0)
    _AUDIO_TEXT_CONTRASTIVE_WEIGHT = getattr(args, "audio_text_contrastive_weight", 1.0)

    num_devices = jax.local_device_count()
    use_wandb = getattr(args, "wandb", False)
    if use_wandb:
        import wandb
        if wandb.run is None:
            wandb.init(project="needle-stage1", config=vars(args))

    speech_prefix = getattr(args, "speech_gcs_prefix", EMILIA_SPEECH_GCS_PREFIX)
    speech_val_ratio = getattr(args, "speech_val_ratio", 0.01)
    speech_max_samples = getattr(args, "max_speech_samples", None) or getattr(args, "max_samples", None)

    print(f"\n[1/4] Loading tokenizer...")
    tokenizer = get_tokenizer(max_samples=args.max_samples)

    print(f"\n[2/4] Loading Emilia metadata...")
    train_prepared = load_prepared_transcription_data(
        "train",
        args.max_enc_len,
        args.max_dec_len,
        gcs_prefix=speech_prefix,
        val_ratio=speech_val_ratio,
        max_samples=speech_max_samples,
        mmap=True,
    )
    val_prepared = load_prepared_transcription_data(
        "val",
        args.max_enc_len,
        args.max_dec_len,
        gcs_prefix=speech_prefix,
        val_ratio=speech_val_ratio,
        max_samples=getattr(args, "max_eval_samples", None),
        mmap=True,
    )
    if train_prepared is None or val_prepared is None:
        train_rows = load_emilia_speech_metadata(
            "train",
            gcs_prefix=speech_prefix,
            max_samples=speech_max_samples,
            val_ratio=speech_val_ratio,
            seed=args.seed,
        )
        val_rows = load_emilia_speech_metadata(
            "val",
            gcs_prefix=speech_prefix,
            max_samples=getattr(args, "max_eval_samples", None),
            val_ratio=speech_val_ratio,
            seed=args.seed,
        )
        train_prepared = prepare_transcription_pairs(train_rows, tokenizer, args.max_enc_len, args.max_dec_len)
        val_prepared = prepare_transcription_pairs(val_rows, tokenizer, args.max_enc_len, args.max_dec_len)
        print(f"      {len(train_rows):,} train / {len(val_rows):,} val speech examples")
    else:
        print(
            f"      loaded cached Stage 1 speech data: "
            f"{len(train_prepared['text_inputs']):,} train / {len(val_prepared['text_inputs']):,} val"
        )

    print(f"\n[3/4] Loading Toucan contrastive data...")
    toucan_config = getattr(args, "toucan_config", None) or "Kimi-K2"
    toucan_path = cache_toucan_examples(
        config=toucan_config,
        split="train",
        max_samples=getattr(args, "toucan_max_samples", None),
        tokenizer=tokenizer,
        max_text_len=args.max_enc_len,
    )
    toucan_batch = load_toucan_contrastive_data(toucan_path, max_text_len=args.max_enc_len)

    print(f"\n[4/4] Building model...")
    resume_checkpoint = getattr(args, "checkpoint", None)
    if resume_checkpoint:
        ckpt_params, config, _ = load_checkpoint(resume_checkpoint)
        print(f"      loaded {resume_checkpoint}")
    else:
        config = create_config_from_args(args, n_mels=getattr(args, "n_mels", 80))
        ckpt_params = None

    effective_batch_size = args.batch_size * num_devices
    if effective_batch_size % num_devices != 0:
        raise ValueError(
            f"Effective batch size {effective_batch_size} must be divisible by num_devices {num_devices}"
        )
    batches_per_epoch = count_batches(len(train_prepared["text_inputs"]), effective_batch_size)
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
    print(f"  Speech data   {speech_prefix}")
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
    checked_batch_shapes = False

    for epoch in range(args.epochs):
        losses = []
        speech_iter = PrefetchIterator(
            lambda: get_transcription_batches(
                train_prepared,
                effective_batch_size,
                max_mel_len=args.max_mel_len,
                n_mels=args.n_mels,
                gcs_prefix=speech_prefix,
                require_local_mels=True,
            ),
            prefetch=2,
        )
        contrastive_iter = PrefetchIterator(
            lambda: get_toucan_batches(*toucan_batch, effective_batch_size),
            prefetch=2,
        )
        pbar = tqdm(range(batches_per_epoch), desc=f"Stage 1 Epoch {epoch + 1}/{args.epochs}")

        for _ in pbar:
            mel, text_enc, tgt_in, tgt_out, lm = next(speech_iter)
            tc_q, tc_tools, tc_labels, tc_tool_mask = next(contrastive_iter)

            if not checked_batch_shapes:
                _assert_stage1_batch_shapes(
                    mel,
                    text_enc,
                    tgt_in,
                    tgt_out,
                    lm,
                    tc_q,
                    tc_tools,
                    tc_labels,
                    tc_tool_mask,
                    effective_batch_size,
                    args.max_mel_len,
                    args.n_mels,
                    args.max_enc_len,
                    args.max_dec_len,
                )
                print(
                    "  Stage 1 batch shapes "
                    f"mel={mel.shape} text={text_enc.shape} tgt={tgt_in.shape} "
                    f"tc_q={tc_q.shape} tc_tools={tc_tools.shape}"
                )
                checked_batch_shapes = True

            mel_b = shard_batch(mel, num_devices)
            text_enc_b = shard_batch(text_enc, num_devices)
            tgt_in_b = shard_batch(tgt_in, num_devices)
            tgt_out_b = shard_batch(tgt_out, num_devices)
            lm_b = shard_batch(lm, num_devices)
            tc_q_b = shard_batch(tc_q, num_devices)
            tc_tools_b = shard_batch(tc_tools, num_devices)
            tc_labels_b = shard_batch(tc_labels, num_devices)
            tc_tool_mask_b = shard_batch(tc_tool_mask, num_devices)

            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_devices)
            t0 = time.perf_counter()
            state, ema_params, loss, grad_norm = p_train_step(
                state,
                ema_params,
                mel_b,
                text_enc_b,
                tgt_in_b,
                tgt_out_b,
                causal_mask,
                step_rngs,
                lm_b,
                tc_q_b,
                tc_tools_b,
                tc_labels_b,
                tc_tool_mask_b,
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
                    val_prepared,
                    args.batch_size,
                    args.max_mel_len,
                    args.n_mels,
                    speech_prefix,
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
                        "train/tokens_per_sec": effective_batch_size * args.max_dec_len / max(dt, 1e-6),
                        "train/step": global_step,
                        **({"val/ppl": last_val_ppl} if last_val_ppl is not None and (global_step % eval_every == 0 or global_step == total_steps) else {}),
                    }
                )

        speech_iter.close()
        contrastive_iter.close()

        eval_params = jax_utils.unreplicate(ema_params)
        val_ppl = _evaluate_val_ppl(
            val_loss_fn,
            eval_params,
            val_prepared,
            args.batch_size,
            args.max_mel_len,
            args.n_mels,
            speech_prefix,
            args.max_dec_len,
            max_eval_samples=getattr(args, "max_eval_samples", None),
        )
        ckpt_path = "(skipped)"
        if not getattr(args, "no_checkpoints", False):
            ckpt_name = f"needle_stage1_{config.num_encoder_layers}_{config.d_model}_{global_step}.pkl"
            ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
            save_checkpoint(ckpt_path, eval_params, config, extra={"stage": "stage1"})

        print(f"\n  Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train loss     {sum(losses) / max(len(losses), 1):.4f}")
        print(f"  Val ppl        {val_ppl:.2f}")
        print(f"  Checkpoint     {ckpt_path}\n")

    if use_wandb:
        wandb.finish()
    print("Stage 1 pretraining complete.")
