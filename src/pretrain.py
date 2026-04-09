"""Streaming pretraining on PleIAs/SYNTH dataset.

Streams 80M examples from HuggingFace, tokenizes on-the-fly, and trains
with simple CE loss (uniform weights). Saves needle_base.pkl every N steps.

Usage:
    needle pretrain --wandb
    needle pretrain --wandb --max-steps 10000
    needle tpu train large --name pretrain --wandb
"""

import math
import os
import pickle
import time
import queue
import threading

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from .tokenizer import get_tokenizer, EOS_ID, TOOLS_ID, PAD_ID
from .model import (
    EncoderDecoderTransformer,
    TransformerConfig,
    make_padding_mask,
    make_causal_mask,
)
from .optim import create_train_state, _wsd_schedule
from .distributed import _replicate, _unreplicate, shard_batch, _upload_checkpoint

_HF_PRETRAIN_REPO = "PleIAs/SYNTH"


def _stream_batches(tokenizer, batch_size, max_enc_len, max_dec_len, seed=42):
    """Stream batches from PleIAs/SYNTH, tokenizing on-the-fly.

    Encoder: [query_tokens, <tools>, seed_text_tokens] truncated to max_enc_len
    Decoder input:  [EOS, answer_tokens] truncated to max_dec_len
    Decoder target: [answer_tokens, EOS] truncated to max_dec_len

    Yields (enc, dec_in, dec_tgt) as numpy int32 arrays of shape (batch_size, seq_len).
    """
    from datasets import load_dataset

    ds = load_dataset(_HF_PRETRAIN_REPO, split="train", streaming=True)
    # Small shuffle buffer so first batch yields quickly on resume.
    # Dataset is already split across 500 shards so it's already partially shuffled.
    ds = ds.shuffle(seed=seed, buffer_size=1000)

    tools_sep_id = TOOLS_ID
    eos_id = EOS_ID
    pad_id = PAD_ID

    enc_batch = np.full((batch_size, max_enc_len), pad_id, dtype=np.int32)
    dec_in_batch = np.full((batch_size, max_dec_len), pad_id, dtype=np.int32)
    dec_tgt_batch = np.full((batch_size, max_dec_len), pad_id, dtype=np.int32)
    idx = 0

    for example in ds:
        query = example.get("query") or ""
        seed_text = example.get("query_seed_text") or ""
        answer = example.get("synthetic_answer") or ""

        if not query.strip() or not answer.strip():
            continue

        # Tokenize
        q_toks = tokenizer.encode(query)
        s_toks = tokenizer.encode(seed_text) if seed_text.strip() else []
        a_toks = tokenizer.encode(answer)

        # Build encoder input: [query, <tools>, seed_text]
        max_query = max_enc_len - 2
        if len(q_toks) > max_query:
            q_toks = q_toks[:max_query]
        remaining = max_enc_len - len(q_toks) - 1
        s_toks = s_toks[:remaining]
        enc_tokens = q_toks + [tools_sep_id] + s_toks

        # Build decoder: input=[EOS, answer], target=[answer, EOS]
        max_ans = max_dec_len - 1
        a_toks = a_toks[:max_ans]
        if len(a_toks) == 0:
            continue
        dec_in_tokens = [eos_id] + a_toks
        dec_tgt_tokens = a_toks + [eos_id]

        # Fill batch
        enc_batch[idx, :len(enc_tokens)] = enc_tokens
        dec_in_batch[idx, :len(dec_in_tokens)] = dec_in_tokens
        dec_tgt_batch[idx, :len(dec_tgt_tokens)] = dec_tgt_tokens
        idx += 1

        if idx == batch_size:
            yield enc_batch.copy(), dec_in_batch.copy(), dec_tgt_batch.copy()
            enc_batch[:] = pad_id
            dec_in_batch[:] = pad_id
            dec_tgt_batch[:] = pad_id
            idx = 0


class _PrefetchStream:
    """Prefetch streaming batches in a background thread."""

    def __init__(self, generator_fn, prefetch=4):
        self._queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, args=(generator_fn,), daemon=True)
        self._thread.start()

    def _produce(self, gen_fn):
        try:
            for batch in gen_fn():
                if self._stop.is_set():
                    return
                self._queue.put(batch)
            self._queue.put(None)
        except Exception as e:
            self._queue.put(e)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self._stop.set()


def _pretrain_step(state, src, tgt_in, tgt_out, rng):
    """Simple CE loss train step for pretraining. No QAT, no contrastive, no pruning."""

    def loss_fn(params):
        src_mask = make_padding_mask(src, PAD_ID)
        tgt_mask = make_causal_mask(tgt_in.shape[1]) & make_padding_mask(tgt_in, PAD_ID)
        cross_mask = src_mask

        logits = state.apply_fn(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask,
        )
        logits_f32 = logits.astype(jnp.float32)

        padding_mask = (tgt_out != PAD_ID).astype(jnp.float32)
        num_tokens = jnp.maximum(jnp.sum(padding_mask), 1.0)
        ce = jnp.sum(
            optax.softmax_cross_entropy_with_integer_labels(logits_f32, tgt_out) * padding_mask
        ) / num_tokens
        z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits_f32, axis=-1) ** 2)
        return ce + z_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    state = state.apply_gradients(grads=grads)
    return state, loss


_p_pretrain_step = None


def _make_p_pretrain_step():
    return jax.pmap(_pretrain_step, axis_name="batch", donate_argnums=(0,))


def pretrain(args):
    num_devices = jax.local_device_count()
    num_hosts = jax.process_count()
    host_id = jax.process_index()
    total_devices = jax.device_count()
    is_main = host_id == 0

    experiment_name = getattr(args, "name", "pretrain")
    use_wandb = getattr(args, "wandb", False)
    if use_wandb and is_main:
        import wandb
        if wandb.run is None:
            wandb.init(project="needle-v1", name=experiment_name, config=vars(args))

    print(f"\n[1/3] Detecting devices...")
    print(f"      {num_devices} local, {total_devices} total across {num_hosts} hosts")

    print(f"\n[2/3] Loading tokenizer...")
    tokenizer = get_tokenizer()

    # If resuming, download checkpoint from HF if not local, then load config
    resume_checkpoint_arg = getattr(args, "checkpoint", None)
    _preloaded_ckpt = None
    if resume_checkpoint_arg:
        if not os.path.exists(resume_checkpoint_arg):
            print(f"  Checkpoint not found locally, downloading from HF...", flush=True)
            try:
                from huggingface_hub import hf_hub_download
                local_dir = os.path.dirname(resume_checkpoint_arg) or "checkpoints"
                os.makedirs(local_dir, exist_ok=True)
                local_path = hf_hub_download(
                    repo_id="Cactus-Compute/checkpoints",
                    filename=os.path.basename(resume_checkpoint_arg),
                    repo_type="model",
                    local_dir=local_dir,
                )
                resume_checkpoint_arg = local_path
                print(f"  Downloaded to {local_path}", flush=True)
            except Exception as e:
                print(f"  ERROR downloading checkpoint: {e}", flush=True)
                return

        print(f"  Pre-loading config from {resume_checkpoint_arg}", flush=True)
        with open(resume_checkpoint_arg, "rb") as f:
            _preloaded_ckpt = pickle.load(f)
        config = TransformerConfig(**_preloaded_ckpt["config"])
        print(f"  Config from ckpt: d={config.d_model}, "
              f"heads={config.num_heads}, layers={config.num_encoder_layers}/{config.num_decoder_layers}",
              flush=True)
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
            activation=getattr(args, "activation", "swiglu"),
            no_feedforward=getattr(args, "no_feedforward", True),
            contrastive_dim=getattr(args, "contrastive_dim", 128),
            enable_speech=False,
        )

    effective_batch_size = args.batch_size * num_devices
    global_batch_size = effective_batch_size * num_hosts

    # Estimate total steps: 80M rows / global_batch_size
    estimated_rows = 80_000_000
    max_steps = getattr(args, "max_steps", None)
    estimated_steps = estimated_rows // global_batch_size
    if max_steps:
        estimated_steps = min(estimated_steps, max_steps)
    total_steps = estimated_steps * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    scaled_lr = args.lr * total_devices
    muon_lr = getattr(args, "muon_lr", 0.02) * math.sqrt(total_devices)
    decay_ratio = getattr(args, "decay_ratio", 0.05)

    resume_step = 0
    resume_checkpoint = resume_checkpoint_arg
    ckpt_data = _preloaded_ckpt

    print(f"\n[3/3] Initializing model...")
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config, scaled_lr, muon_lr, total_steps, warmup_steps, decay_ratio)

    if resume_checkpoint and ckpt_data is not None:
        print(f"  Resuming from {resume_checkpoint}", flush=True)
        ckpt_params = jax.tree.map(jnp.array, ckpt_data["params"])
        state = state.replace(params=ckpt_params)
        del ckpt_params
        print(f"  Loaded checkpoint params", flush=True)
    # Use --resume-step override, else read from checkpoint, else 0
    if ckpt_data is not None:
        manual_step = getattr(args, "resume_step", None)
        if manual_step is not None:
            resume_step = manual_step
        else:
            resume_step = ckpt_data.get("pretrain_step", 0)
        if resume_step > 0:
            print(f"  Resuming from step {resume_step}", flush=True)

    print(f"  Replicating state across {num_devices} local devices...", flush=True)
    state = _replicate(state)
    print(f"  Replicated.", flush=True)

    param_count = sum(x.size for x in jax.tree.leaves(_unreplicate(state).params))
    print(f"  Params: {param_count:,}", flush=True)

    global _p_pretrain_step
    _p_pretrain_step = _make_p_pretrain_step()
    print(f"  pmap ready.", flush=True)

    save_every = getattr(args, "save_every", 1000)
    checkpoint_dir = getattr(args, "checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if is_main:
        decay_steps = max(1, int(total_steps * decay_ratio))
        stable_steps = total_steps - warmup_steps - decay_steps
        print(f"\n  ─────────────────────────────────────")
        print(f"  Pretraining on PleIAs/SYNTH")
        print(f"  ─────────────────────────────────────")
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
        print(f"  Muon LR       {getattr(args, 'muon_lr', 0.02):>7.4f} -> {muon_lr:.4f}")
        print(f"  Schedule      {warmup_steps}w / {stable_steps}s / {decay_steps}d (WSD)")
        print(f"  Est. steps    {estimated_steps:>12,}")
        print(f"  Save every    {save_every:>12,}")
        print(f"  ─────────────────────────────────────\n")

    global_step = resume_step

    for epoch in range(args.epochs):
        # Use a different seed on resume to get fresh data ordering
        # (avoids re-seeing exactly the same examples we saw before the crash)
        stream_seed = args.seed + epoch + resume_step
        batch_stream = _PrefetchStream(
            lambda: _stream_batches(tokenizer, global_batch_size,
                                    args.max_enc_len, args.max_dec_len,
                                    seed=stream_seed),
            prefetch=8,
        )

        pbar = tqdm(desc=f"Pretrain epoch {epoch + 1}/{args.epochs}",
                     total=estimated_steps, initial=resume_step,
                     disable=not is_main)

        for batch in batch_stream:
            if max_steps and global_step >= max_steps:
                break

            src, tgt_in, tgt_out = batch
            t0 = time.perf_counter()

            # Slice this host's portion
            host_slice = slice(host_id * effective_batch_size,
                               (host_id + 1) * effective_batch_size)
            src = src[host_slice]
            tgt_in = tgt_in[host_slice]
            tgt_out = tgt_out[host_slice]

            # Shard for pmap
            src_b = shard_batch(src, num_devices)
            tgt_in_b = shard_batch(tgt_in, num_devices)
            tgt_out_b = shard_batch(tgt_out, num_devices)

            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_devices)

            state, loss = _p_pretrain_step(state, src_b, tgt_in_b, tgt_out_b, step_rngs)

            loss_val = float(loss.addressable_shards[0].data[0])
            ppl = math.exp(min(loss_val, 20))
            dt = time.perf_counter() - t0
            global_step += 1

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss_val:.4f}", ppl=f"{ppl:.2f}",
                             tok_s=f"{global_batch_size * (args.max_enc_len + args.max_dec_len) / dt:.0f}")

            if use_wandb and is_main:
                import wandb
                wandb.log({
                    "pretrain/loss": loss_val,
                    "pretrain/ppl": ppl,
                    "pretrain/step": global_step,
                    "pretrain/tokens_per_sec": global_batch_size * (args.max_enc_len + args.max_dec_len) / dt,
                })

            # Save checkpoint every N steps
            if is_main and global_step % save_every == 0:
                _save_pretrain_checkpoint(state, config, checkpoint_dir, global_step)

        batch_stream.close()
        pbar.close()

    # Final save
    if is_main:
        ckpt_path = _save_pretrain_checkpoint(state, config, checkpoint_dir, global_step)
        print(f"\nPretraining complete. {global_step} steps.")
        print(f"Base checkpoint: {ckpt_path}")

    if use_wandb and is_main:
        import wandb
        wandb.finish()

    # Sync all hosts before exit
    jax.experimental.multihost_utils.sync_global_devices("pretrain_done")


def _save_pretrain_checkpoint(state, config, checkpoint_dir, global_step):
    """Save and upload needle_base.pkl."""
    params = _unreplicate(state).params
    params_np = jax.tree.map(lambda x: np.array(x).astype(np.float16), params)

    ckpt_path = os.path.join(checkpoint_dir, "needle_base.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump({"params": params_np, "config": config.__dict__,
                      "pretrain_step": global_step}, f)

    param_count = sum(x.size for x in jax.tree.leaves(params_np))
    size_mb = sum(x.nbytes for x in jax.tree.leaves(params_np)) / 1e6
    print(f"\n  [step {global_step}] Saved {ckpt_path} ({param_count:,} params, {size_mb:.1f} MB)")

    _upload_checkpoint(ckpt_path)
    return ckpt_path
