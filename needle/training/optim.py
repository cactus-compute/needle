import math
from typing import NamedTuple
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from ..model.architecture import SimpleAttentionNetwork


def _newton_schulz(G, steps=5):
    """Approximate polar decomposition via Newton-Schulz, with aspect-ratio scaling.

    Flax kernels are (fan_in, fan_out); the max(1, fan_out/fan_in)**0.5 factor
    keeps the update RMS consistent across non-square weights.
    """
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
    scale = jnp.sqrt(jnp.maximum(1.0, G.shape[1] / G.shape[0]))
    return (X * scale).astype(orig_dtype)


class MuonState(NamedTuple):
    mu: optax.Updates


def scale_by_muon(momentum=0.95, ns_steps=5):
    """Muon gradient transform: Nesterov momentum on the raw grad, then orthogonalize.

    Orthogonalizing before momentum accumulates a sum of orthogonal matrices,
    which is not itself orthogonal — so the buffer holds raw grads and only the
    Nesterov-blended update is passed through Newton-Schulz.
    """

    def init_fn(params):
        return MuonState(mu=jax.tree.map(jnp.zeros_like, params))

    def ortho(g):
        if g.ndim == 3:
            return jax.vmap(lambda m: _newton_schulz(m, steps=ns_steps), in_axes=(0,))(g)
        if g.ndim == 2:
            return _newton_schulz(g, steps=ns_steps)
        return g

    def update_fn(updates, state, params=None):
        del params
        new_mu = jax.tree.map(lambda m, g: momentum * m + g, state.mu, updates)
        blended = jax.tree.map(lambda g, m: g + momentum * m, updates, new_mu)
        new_updates = jax.tree.map(ortho, blended)
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
    model = SimpleAttentionNetwork(config)

    rng, init_rng = jax.random.split(rng)
    dummy_src = jnp.ones((1, 128), dtype=jnp.int32)
    dummy_tgt = jnp.ones((1, 128), dtype=jnp.int32)
    variables = model.init(
        {"params": init_rng},
        dummy_src, dummy_tgt,
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
