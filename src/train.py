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


def _train_step(state, ema_params, src, tgt_in, tgt_out, causal_mask, dropout_rng):
    pad_id = 0
    ema_decay = 0.999

    def loss_fn(params):
        src_mask = make_padding_mask(src, pad_id)
        tgt_mask = causal_mask & make_padding_mask(tgt_in, pad_id)
        cross_mask = src_mask

        logits = state.apply_fn(
            {"params": params},
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


p_train_step = jax.pmap(_train_step, axis_name="batch", donate_argnums=(0, 1))


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
    )

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    num_batches = len(enc_inputs) // effective_batch_size
    total_steps = num_batches * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    # Scale LRs for data parallelism: linear for Adam, sqrt for Muon
    # (Muon's Newton-Schulz orthogonalization is sensitive to large LRs).
    scaled_lr = args.lr * num_devices
    muon_lr = getattr(args, "muon_lr", 0.02) * math.sqrt(num_devices)
    state = create_train_state(init_rng, config, scaled_lr, muon_lr, total_steps, warmup_steps)
    val_loss_fn = _make_val_loss_fn(state.apply_fn)

    # Replicate state and EMA params across all devices
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

    for epoch in range(args.epochs):
        losses = []
        batches = get_batches(enc_inputs, dec_inputs, dec_targets, effective_batch_size)
        pbar = tqdm(batches, total=num_batches, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for src, tgt_in, tgt_out in pbar:
            rng, dropout_rng = jax.random.split(rng)
            # Give each device a different dropout rng
            dropout_rngs = jax.random.split(dropout_rng, num_devices)

            t0 = time.perf_counter()
            # Shard batch across devices: (num_devices, per_device_batch, seq_len)
            state, ema_params, loss, grad_norm = p_train_step(
                state,
                ema_params,
                shard_batch(src, num_devices),
                shard_batch(tgt_in, num_devices),
                shard_batch(tgt_out, num_devices),
                causal_mask,
                dropout_rngs,
            )
            # loss/grad_norm are replicated across devices; take first
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
                val_ppl = float(math.exp(total_loss / max(total_toks, 1)))
                pbar.set_postfix(loss=f"{loss_val:.4f}", val_ppl=f"{val_ppl:.2f}")
            else:
                pbar.set_postfix(loss=f"{loss_val:.4f}")

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
                    log_dict["val/ppl"] = val_ppl
                wandb.log(log_dict)

        epoch_avg_loss = sum(losses) / len(losses) if losses else float("nan")
        final_loss = losses[-1] if losses else float("nan")
        print(f"Epoch {epoch + 1}/{args.epochs} — avg loss: {epoch_avg_loss:.4f}, final loss: {final_loss:.4f}")

        if use_wandb:
            wandb.log({
                "epoch/avg_loss": epoch_avg_loss,
                "epoch/final_loss": final_loss,
                "epoch": epoch + 1,
            })

        # Checkpoint EMA params (smoother, better for eval)
        ckpt_name = f"needle_{args.num_layers}_{args.d_model}_{global_step}.pkl"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        params_np = jax.tree.map(np.array, jax_utils.unreplicate(ema_params))
        with open(ckpt_path, "wb") as f:
            pickle.dump({"params": params_np, "config": config.__dict__}, f)
        print(f"Saved checkpoint: {ckpt_path}")

    if use_wandb:
        wandb.finish()
    print("Training complete.")


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
