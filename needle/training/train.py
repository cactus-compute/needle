import math
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from ..dataset.dataset import (
    get_batches, get_tokenizer,
    load_prepared_data, load_tool_calls,
    PrefetchIterator, count_batches,
    get_contrastive_batches,
)
from ..model.architecture import (
    SimpleAttentionNetwork,
    TransformerConfig,
    make_packing_mask,
    make_causal_packing_mask,
    make_cross_packing_mask,
)
from .optim import create_train_state, _wsd_schedule
from ..utils.distributed import (
    _replicate, _unreplicate, shard_batch, _upload_checkpoint,
    load_pretrained_params, partial_load_params,
)
from ..model.quantize import _quantize_params


_PRECISION = "int4"
_CONTRASTIVE_WEIGHT = 0.1
# Weight map: class 0=base, 1=name, 2=value, 3=key (set from CLI args at train time)
# Use numpy at module level to avoid initializing JAX backend before jax.distributed.initialize()
_LOSS_WEIGHT_MAP = np.array([1.0, 3.0, 2.0, 1.5], dtype=np.float32)


def _clip_contrastive_loss(q_emb, t_emb, log_temp):
    """CLIP-style symmetric contrastive loss with learnable temperature."""
    temp = jnp.exp(jnp.clip(log_temp, -jnp.log(100.0), jnp.log(100.0)))
    logits = q_emb @ t_emb.T / temp  # (B, B)
    B = logits.shape[0]
    labels = jnp.arange(B)
    loss_q = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss_t = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels)
    return (jnp.mean(loss_q) + jnp.mean(loss_t)) / 2.0


def _text_loss_fn(state, params, src, tgt_in, tgt_out, rng, loss_mask, do_quantize,
                  enc_seg_ids, dec_seg_ids):
    q_params = jax.lax.cond(
        do_quantize,
        lambda p: _quantize_params(p, precision=_PRECISION),
        lambda p: p,
        params,
    )
    src_mask = make_packing_mask(enc_seg_ids)
    tgt_mask = make_causal_packing_mask(dec_seg_ids)
    cross_mask = make_cross_packing_mask(enc_seg_ids, dec_seg_ids)
    logits, slot_div = state.apply_fn(
        {"params": q_params},
        src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask,
        cross_mask=cross_mask,
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
        lambda p: _quantize_params(p, precision=_PRECISION),
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



def _train_step(state, src, tgt_in, tgt_out, enc_seg_ids, dec_seg_ids, rng, loss_mask, query_tokens, tool_tokens, cl_rng, do_quantize, do_contrastive):
    """Unified train step with boolean flag for contrastive loss."""

    def combined_loss(p):
        text_loss = _text_loss_fn(state, p, src, tgt_in, tgt_out, rng, loss_mask, do_quantize,
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
    return state, text_loss, grad_norm


def _make_p_train_step():
    return jax.pmap(_train_step, axis_name="batch", donate_argnums=(0,))


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


def train(args):
    num_devices = jax.local_device_count()

    experiment_name = getattr(args, "name", "baseline")
    use_wandb = getattr(args, "wandb", False)
    if use_wandb and jax.process_index() == 0:
        import wandb
        if wandb.run is None:
            wandb.init(project="needle-v1", name=experiment_name, config=vars(args))

    print(f"\n[1/3] Detecting devices...")
    print(f"      {num_devices} local device(s), {jax.device_count()} total across {jax.process_count()} hosts")

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

    val_ds = getattr(args, "val_ds", None)
    if val_ds is None:
        try:
            val_ds = load_tool_calls("val", max_samples=args.max_samples)
        except (FileNotFoundError, Exception) as e:
            print(f"      WARNING: could not load val dataset for eval samples: {e}")
            val_ds = None

    cl_query_tokens = train_data.get("query_only")
    cl_tool_tokens = train_data.get("tool_individual")
    cl_tool_ex_idx = train_data.get("tool_ex_idx")
    cl_tool_is_pos = train_data.get("tool_is_pos")
    has_contrastive = all(x is not None for x in [cl_query_tokens, cl_tool_tokens, cl_tool_ex_idx, cl_tool_is_pos])
    if has_contrastive:
        print(f"      Contrastive: {len(cl_query_tokens):,} queries, {len(cl_tool_tokens):,} tools")

    num_hosts = jax.process_count()
    host_id = jax.process_index()
    total_devices = jax.device_count()
    effective_batch_size = args.batch_size * num_devices  # per-host batch for pmap

    resume_checkpoint = getattr(args, "checkpoint", None)
    if resume_checkpoint:
        if not os.path.exists(resume_checkpoint):
            print(f"  Checkpoint not found locally, downloading from HF...", flush=True)
            from huggingface_hub import hf_hub_download
            local_dir = os.path.dirname(resume_checkpoint) or "checkpoints"
            os.makedirs(local_dir, exist_ok=True)
            resume_checkpoint = hf_hub_download(
                repo_id="Cactus-Compute/checkpoints",
                filename=os.path.basename(resume_checkpoint),
                repo_type="model",
                local_dir=local_dir,
            )
            print(f"  Downloaded to {resume_checkpoint}", flush=True)
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        with open(resume_checkpoint, "rb") as f:
            ckpt_data = pickle.load(f)
        config = TransformerConfig(**ckpt_data["config"])
        if getattr(args, "lora_rank", 0) > 0:
            config.lora_rank = args.lora_rank
            config.lora_alpha = args.lora_alpha
            if getattr(args, "init_from", None) is None and config.lora_rank > 0 and ckpt_data["config"].get("lora_rank", 0) == 0:
                print(f"  Switching from resume checkpoint to init_from for LoRA training (rank={config.lora_rank})")
                args.init_from = resume_checkpoint
                resume_checkpoint = None
        ckpt_params = jax.tree.map(jnp.array, ckpt_data["params"])
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
            num_memory_slots=getattr(args, "num_memory_slots", 64),
            contrastive_dim=getattr(args, "contrastive_dim", 128),
            lora_rank=getattr(args, "lora_rank", 0),
            lora_alpha=getattr(args, "lora_alpha", 16),
        )

    global _PRECISION, _CONTRASTIVE_WEIGHT, _LOSS_WEIGHT_MAP
    _PRECISION = getattr(args, "precision", "int4")
    _CONTRASTIVE_WEIGHT = getattr(args, "contrastive_weight", 0.1)
    _LOSS_WEIGHT_MAP = jnp.array([
        1.0,                                       # 0: base (boilerplate)
        getattr(args, "w_name", 2.0),              # 1: tool name
        getattr(args, "w_value", 4.0),             # 2: argument value
        getattr(args, "w_key", 1.5),               # 3: argument key
    ], dtype=jnp.float32)
    p_train_step = _make_p_train_step()

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    unique_batch_size = (effective_batch_size // num_devices) * num_devices
    # Each host processes unique_batch_size samples; total across hosts = unique_batch_size * num_hosts
    global_batch_size = unique_batch_size * num_hosts
    text_batches_per_epoch = count_batches(len(enc_inputs), global_batch_size)
    num_batches = text_batches_per_epoch
    total_steps = num_batches * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    scaled_lr = args.lr * total_devices
    muon_lr = getattr(args, "muon_lr", 0.02) * math.sqrt(total_devices)
    decay_ratio = getattr(args, "decay_ratio", 0.15)
    state = create_train_state(init_rng, config, scaled_lr, muon_lr, total_steps, warmup_steps, decay_ratio)
    val_loss_fn = _make_val_loss_fn(state.apply_fn)

    if resume_checkpoint:
        # Cast each loaded leaf to the dtype of the freshly-initialized param
        # (most weights -> bf16, scalars like log_temp stay fp32)
        ckpt_params = jax.tree.map(
            lambda ref, loaded: jnp.asarray(loaded, dtype=ref.dtype),
            state.params, ckpt_params,
        )
        state = state.replace(params=ckpt_params)
        print(f"  Loaded checkpoint params into train state")

    init_from = getattr(args, "init_from", None)
    if init_from and not resume_checkpoint:
        print(f"\n  Initializing from pretrained base: {init_from}")
        loaded_params, _loaded_cfg, local_path = load_pretrained_params(init_from)
        merged, stats = partial_load_params(state.params, loaded_params)
        state = state.replace(params=merged)
        total = stats["loaded"] + stats["random_init"]
        print(f"  [init-from] {stats['loaded']}/{total} leaves loaded from {os.path.basename(local_path)}, "
              f"{stats['random_init']} kept at random init")
        if stats["shape_mismatches"]:
            print(f"  [init-from] {len(stats['shape_mismatches'])} shape mismatch(es):")
            for path, loaded_shape, init_shape in stats["shape_mismatches"][:8]:
                print(f"    {'/'.join(path)}: ckpt={loaded_shape} vs init={init_shape}")
            if len(stats["shape_mismatches"]) > 8:
                print(f"    ... ({len(stats['shape_mismatches']) - 8} more)")
        if stats["extra_in_ckpt"]:
            print(f"  [init-from] {len(stats['extra_in_ckpt'])} extra leaf(s) in ckpt (ignored)")
        del loaded_params, merged

    state = _replicate(state)

    param_count = sum(x.size for x in jax.tree.leaves(_unreplicate(state).params))
    decay_steps = max(1, int(total_steps * decay_ratio))
    stable_steps = total_steps - warmup_steps - decay_steps
    best_call_f1 = 0.0
    best_ckpt_path = None

    is_main = host_id == 0
    if is_main:
        print(f"\n  ─────────────────────────────────────")
        print(f"  Parameters    {param_count:>12,}")
        print(f"  d_model       {config.d_model:>12}")
        print(f"  Heads         {config.num_heads:>7} ({config.num_kv_heads} KV)")
        print(f"  Layers        {config.num_encoder_layers:>7} enc / {config.num_decoder_layers} dec")
        print(f"  Dtype         {config.dtype:>12}")
        print(f"  ─────────────────────────────────────")
        print(f"  Hosts         {num_hosts:>12}")
        print(f"  Devices       {num_devices:>5}/host, {total_devices} total")
        print(f"  Batch         {args.batch_size:>7} x {total_devices} = {global_batch_size}")
        print(f"  Adam LR       {args.lr:>7} x {total_devices} = {scaled_lr}")
        print(f"  Muon LR       {args.muon_lr:>7.4f} -> {muon_lr:.4f}")
        print(f"  Schedule      {warmup_steps}w / {stable_steps}s / {decay_steps}d (WSD)")
        print(f"  Total steps   {total_steps:>12,}")
        print(f"  Epochs        {args.epochs:>12}")
        print(f"  ─────────────────────────────────────\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    global_step = 0

    adam_schedule = _wsd_schedule(scaled_lr, total_steps, warmup_steps)
    muon_schedule = _wsd_schedule(muon_lr, total_steps, warmup_steps)
    tokens_per_batch = global_batch_size * (args.max_enc_len + args.max_dec_len)

    eval_model = SimpleAttentionNetwork(config)

    last_val_ppl = None
    # Dummy contrastive arrays — used when contrastive is inactive
    dummy_cl_tokens = jnp.zeros((num_devices, args.batch_size, 128), dtype=jnp.int32)
    dummy_cl_rng = jax.random.split(jax.random.PRNGKey(0), num_devices)

    for epoch in range(args.epochs):
        text_losses = []
        text_batch_iter = PrefetchIterator(
            lambda: get_batches(enc_inputs, dec_inputs, dec_targets, global_batch_size,
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
                    global_batch_size),
                prefetch=4,
            )

        pbar = tqdm(range(text_batches_per_epoch), desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step_i in pbar:
            t0 = time.perf_counter()

            batch = next(text_batch_iter)

            # Slice this host's portion from the global batch
            host_slice = slice(host_id * unique_batch_size, (host_id + 1) * unique_batch_size)
            src, tgt_in, tgt_out, lm, enc_seg, dec_seg = [b[host_slice] for b in batch]

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

            do_qat = jnp.broadcast_to(jnp.array(global_step > 0 and global_step % 100 == 0), (num_devices,))

            do_contrastive = cl_batch_iter is not None and global_step > 0 and global_step % 1000 == 0
            cl_q_b, cl_t_b, cl_rngs = dummy_cl_tokens, dummy_cl_tokens, dummy_cl_rng
            if do_contrastive:
                try:
                    cl_q, cl_t = next(cl_batch_iter)
                    cl_q = cl_q[host_slice]
                    cl_t = cl_t[host_slice]
                    cl_q_b = shard_batch(cl_q, num_devices)
                    cl_t_b = shard_batch(cl_t, num_devices)
                    rng, cl_rng = jax.random.split(rng)
                    cl_rngs = jax.random.split(cl_rng, num_devices)
                except StopIteration:
                    cl_batch_iter = None
                    do_contrastive = False

            do_cl = jnp.broadcast_to(jnp.array(do_contrastive), (num_devices,))

            state, loss, grad_norm = p_train_step(
                state, src_b, tgt_in_b, tgt_out_b, enc_seg_b, dec_seg_b,
                text_rngs, lm_b,
                cl_q_b, cl_t_b, cl_rngs, do_qat, do_cl,
            )

            text_loss_val = float(loss.addressable_shards[0].data[0])
            text_losses.append(text_loss_val)
            step_grad_norm = float(grad_norm.addressable_shards[0].data[0])
            global_step += 1

            dt = time.perf_counter() - t0
            eval_every = getattr(args, "eval_every", 100)
            if is_main and (global_step % eval_every == 0 or global_step == total_steps):
                _eval_params = _unreplicate(state).params
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
            pbar.set_postfix(**postfix)

            if use_wandb and is_main:
                log_dict = {
                    "train/text_loss": text_loss_val,
                    "train/grad_norm": step_grad_norm,
                    "train/adam_lr": float(adam_schedule(global_step)),
                    "train/muon_lr": float(muon_schedule(global_step)),
                    "train/tokens_per_sec": tokens_per_batch / dt,
                    "train/step": global_step,
                }
                if global_step % eval_every == 0 or global_step == total_steps:
                    log_dict["val/text_ppl"] = last_val_ppl
                wandb.log(log_dict)

        text_batch_iter.close()
        if cl_batch_iter is not None:
            cl_batch_iter.close()

        epoch_avg_loss = sum(text_losses) / len(text_losses) if text_losses else float("nan")
        final_loss = text_losses[-1] if text_losses else float("nan")
        final_ppl = math.exp(min(final_loss, 20)) if not math.isnan(final_loss) else float("nan")

        # ── All hosts: unreplicate params for eval ──
        eval_params = _unreplicate(state).params

        # ── Host 0 only: val PPL + checkpoint save (fast) ──
        quant_val_ppl = 0.0
        total_params = 0
        ckpt_path = ""

        if is_main:
            q_params = _quantize_params(eval_params, precision=_PRECISION)

            full_loss, full_toks = 0.0, 0.0
            q_loss, q_toks = 0.0, 0.0

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
            last_val_ppl = float(math.exp(min(full_loss / max(full_toks, 1), 20)))
            quant_val_ppl = float(math.exp(min(q_loss / max(q_toks, 1), 20)))
            del q_params

            params_np = jax.tree.map(lambda x: np.array(x).astype(np.float16), eval_params)
            total_params = sum(x.size for x in jax.tree.leaves(params_np))

            ckpt_name = f"{experiment_name}_{args.num_layers}_{args.d_model}_{global_step}.pkl"
            ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
            with open(ckpt_path, "wb") as f:
                pickle.dump({"params": params_np, "config": config.__dict__}, f)
            del params_np

        # ── All hosts: distributed generation eval ──
        from ..model.run import generate_batch
        import json as _json_mod

        _pool_names = ["single", "multi"]
        display_pairs = []
        eval_pools = {name: [] for name in _pool_names}

        if val_ds is not None:
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

            tc_per_bucket = 50
            tc_rng = np.random.RandomState(epoch + 42)

            def _classify_sample(ex):
                try:
                    answers = _json_mod.loads(ex["answers"])
                except (ValueError, TypeError):
                    return "empty"
                if not answers or not isinstance(answers, list):
                    return "empty"
                return "single" if len(answers) == 1 else "multi"

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

            for name in _pool_names:
                eval_pools[name] = _balanced_sample(_pool_buckets[name], tc_per_bucket, tc_rng)

        all_eval_examples = display_pairs
        _pool_offsets = {}
        for name in _pool_names:
            _pool_offsets[name] = len(all_eval_examples)
            all_eval_examples = all_eval_examples + eval_pools[name]
        eval_gen_len = min(args.max_dec_len, 512)
        _EVAL_BATCH = 32

        # Prepare generation model (all hosts)
        _gen_params = _quantize_params(eval_params, precision=_PRECISION)
        _gen_model = eval_model
        _gen_label = f"full {_PRECISION.upper()}"

        # Split eval examples across hosts — each host generates its slice
        total_eval = len(all_eval_examples)
        per_host_eval = (total_eval + num_hosts - 1) // num_hosts
        my_eval_start = host_id * per_host_eval
        my_eval_end = min(my_eval_start + per_host_eval, total_eval)
        my_eval_examples = all_eval_examples[my_eval_start:my_eval_end]
        _my_n_chunks = (len(my_eval_examples) + _EVAL_BATCH - 1) // _EVAL_BATCH

        my_preds = []
        _gen_t0 = time.perf_counter()
        for _ei in tqdm(range(0, len(my_eval_examples), _EVAL_BATCH),
                        total=_my_n_chunks,
                        desc=f"  eval generate h{host_id} ({_gen_label})",
                        leave=False, disable=not is_main):
            _chunk = my_eval_examples[_ei:_ei + _EVAL_BATCH]
            my_preds.extend(generate_batch(
                _gen_model, _gen_params, tokenizer,
                [ex["query"] for ex in _chunk],
                [ex["tools"] for ex in _chunk],
                max_gen_len=eval_gen_len,
                max_enc_len=args.max_enc_len,
                constrained=False,
            ))
        _gen_elapsed = time.perf_counter() - _gen_t0
        del _gen_params

        # All-gather predictions across hosts via padded token ID arrays
        _MAX_PRED_TOKENS = eval_gen_len
        pred_ids = np.zeros((per_host_eval, _MAX_PRED_TOKENS), dtype=np.int32)
        for i, p in enumerate(my_preds):
            toks = tokenizer.encode(p)[:_MAX_PRED_TOKENS]
            pred_ids[i, :len(toks)] = toks
        all_pred_ids = jax.experimental.multihost_utils.process_allgather(
            jnp.array(pred_ids), tiled=True)
        all_pred_ids = np.array(all_pred_ids)[:total_eval]

        # Decode predictions on all hosts (cheap), metrics computed on host 0
        all_preds = []
        for row in all_pred_ids:
            nonzero = np.nonzero(row)[0]
            if len(nonzero) > 0:
                all_preds.append(tokenizer.decode(row[:nonzero[-1] + 1].tolist()))
            else:
                all_preds.append("")
        _gen_total_toks = sum(len(tokenizer.encode(p)) for p in all_preds)
        _gen_tok_per_sec = _gen_total_toks / max(_gen_elapsed, 1e-6)

        # ── Non-main hosts: done with this epoch ──
        if not is_main:
            jax.experimental.multihost_utils.sync_global_devices("epoch_eval")
            continue

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
                elif not ref_is_empty and not pred_is_empty:
                    ref_sorted = sorted([_call_key(c) for c in ref_calls if _call_key(c)])
                    pred_sorted = sorted([_call_key(c) for c in pred_calls if _call_key(c)])
                    if ref_sorted == pred_sorted and len(ref_sorted) == len(ref_calls) and len(pred_sorted) == len(pred_calls):
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
                    and sorted([_call_key(c) for c in ref_calls if _call_key(c)])
                    == sorted([_call_key(c) for c in pred_calls if _call_key(c)])
                    and len(ref_calls) == len(pred_calls)
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
            best_ckpt_path = os.path.join(args.checkpoint_dir, f"{experiment_name}_{args.num_layers}_{args.d_model}_best.pkl")
            import shutil as _shutil
            _shutil.copy2(ckpt_path, best_ckpt_path)
            print(f"  ** New best single call_f1={best_call_f1:.1%} → {best_ckpt_path}")
            _upload_checkpoint(best_ckpt_path)

        retrieval_metrics = None
        if has_contrastive and _CONTRASTIVE_WEIGHT > 0 and val_ds is not None:
            from ..training.eval import benchmark_retrieval
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
        print(f"  Quant val ppl  {quant_val_ppl:>12.2f}  ({_PRECISION.upper()})")
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
                "epoch": epoch + 1,
            }
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

        # Sync with non-main hosts that are waiting at the barrier
        jax.experimental.multihost_utils.sync_global_devices("epoch_eval")

    if use_wandb and is_main:
        wandb.finish()
    if is_main and best_ckpt_path:
        print(f"\nBest checkpoint (call_f1={best_call_f1:.1%}): {best_ckpt_path}")
    if is_main:
        print("\nTraining complete.")


