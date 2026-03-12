import math
import pickle
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from .model import EncoderDecoderTransformer, TransformerConfig


def _newton_schulz(G, steps=5):
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
        new_updates = jax.tree.map(lambda g, m: g + momentum * m, ortho_g, new_mu)
        return new_updates, MuonState(mu=new_mu)

    return optax.GradientTransformation(init_fn, update_fn)


def _param_labels(params):
    def _label(path, leaf):
        name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        if name == "kernel" and leaf.ndim == 2:
            return "muon"
        return "adam"

    return jax.tree_util.tree_map_with_path(_label, params)


def wsd_schedule(peak_value, total_steps, warmup_steps, decay_ratio=0.15):
    decay_steps = max(1, int(total_steps * decay_ratio))
    stable_steps = max(0, total_steps - warmup_steps - decay_steps)
    return optax.join_schedules(
        [
            optax.linear_schedule(0.0, peak_value, warmup_steps),
            optax.constant_schedule(peak_value),
            optax.linear_schedule(peak_value, peak_value * 0.1, decay_steps),
        ],
        boundaries=[warmup_steps, warmup_steps + stable_steps],
    )


def create_config_from_args(args, n_mels=80):
    return TransformerConfig(
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
        dropout_rate=getattr(args, "dropout", 0.0),
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

    adam_schedule = wsd_schedule(learning_rate, total_steps, warmup_steps)
    muon_schedule = wsd_schedule(muon_lr, total_steps, warmup_steps)

    muon_opt = optax.chain(
        scale_by_muon(momentum=0.95, ns_steps=5),
        optax.add_decayed_weights(weight_decay=0.01),
        optax.scale_by_schedule(muon_schedule),
        optax.scale(-1.0),
    )
    adam_opt = optax.chain(optax.adamw(adam_schedule, b2=0.95, weight_decay=0.0))

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


def fake_quantize_int4(w, group_size=32):
    in_feat, out_feat = w.shape
    gs = min(group_size, in_feat)
    pad = (gs - in_feat % gs) % gs
    w_padded = jnp.pad(w, ((0, pad), (0, 0))) if pad else w
    grouped = w_padded.reshape(-1, gs, out_feat)
    scale = jnp.max(jnp.abs(grouped), axis=1, keepdims=True) / 7.0
    scale = jnp.maximum(scale, 1e-8)
    w_q = jnp.clip(jnp.round(grouped / scale), -8, 7) * scale
    w_q = w_q.reshape(-1, out_feat)[:in_feat]
    return w + jax.lax.stop_gradient(w_q - w)


def quantize_params(params, group_size=32):
    def _maybe_quantize(path, leaf):
        name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        if name == "kernel" and leaf.ndim == 2:
            return fake_quantize_int4(leaf, group_size=group_size)
        return leaf

    return jax.tree_util.tree_map_with_path(_maybe_quantize, params)


def shard_batch(batch, num_devices):
    return batch.reshape(num_devices, -1, *batch.shape[1:])


def count_params(params):
    return sum(x.size for x in jax.tree.leaves(params))


def save_checkpoint(path, params, config, extra=None):
    payload = {"params": jax.tree.map(np.array, params), "config": config.__dict__}
    if extra:
        payload.update(extra)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    params = jax.tree.map(jnp.array, data["params"])
    config = TransformerConfig(**data["config"])
    return params, config, data


def zero_non_decoder_grads(grads):
    def _mask(path, leaf):
        top = path[0].key if hasattr(path[0], "key") else str(path[0])
        if top == "decoder":
            return leaf
        return jnp.zeros_like(leaf)

    return jax.tree_util.tree_map_with_path(_mask, grads)
