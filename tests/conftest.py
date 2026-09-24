import os

import pytest


def _engine_available(generation=3):
    """Whether ``needle._library_path`` would find an engine without downloading.

    Every location this looks in has to be one that function looks in, or the
    gate skips tests an installed engine could have run.
    """
    try:
        import needle
        from needle.agent import fetch

        override = os.environ.get(f"NEEDLE{generation}_LIB_PATH")
        if generation == 2 and not override:
            override = os.environ.get("NEEDLE_LIB_PATH")
        if override:
            return os.path.exists(override)

        here = os.path.dirname(needle.__file__)
        name = fetch._lib_name()
        stem, suffix = os.path.splitext(name)
        local_names = [f"{stem}{generation}{suffix}"]
        if generation == 2:
            local_names.append(name)
        if any(os.path.exists(os.path.join(here, local)) for local in local_names):
            return True

        cache = os.path.join(os.path.expanduser("~"), ".cache", "cactus-needle",
                             f"v{generation}", fetch.engine_version(generation), name)
        return os.path.exists(cache)
    except Exception:
        return False


requires_engine = pytest.mark.skipif(
    not _engine_available(),
    reason="needle C++ engine not installed (auto-fetched from HF on first real use)")


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig
    from needle.model.checkpoints import write_checkpoint

    config = TransformerConfig(
        vocab_size=8192, out_vocab=8192, d_model=64, num_heads=4, num_kv_heads=2,
        num_layers=4, qk_head_dim=16, v_head_dim=16, max_seq_len=128,
        engram_layers=(1, 3), engram_slots=64, global_layers=(3,), sliding_window=32,
        mhc_lanes=2, qkv_conv_taps=3, flash=False,
    )
    model = SimpleAttentionNetwork(config)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 8), jnp.int32))["params"]
    params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)

    path = tmp_path_factory.mktemp("ckpt") / "tiny.safetensors"
    write_checkpoint(path, {"format_version": 2, "params": params, "config": dict(vars(config))})
    return str(path)


@pytest.fixture(scope="session")
def tiny_checkpoint_safetensors(tiny_checkpoint):
    return tiny_checkpoint


@pytest.fixture(scope="session")
def tiny_base_archive(tiny_checkpoint, tmp_path_factory):
    from needle.model.export import write_export
    from needle.model.run import load_checkpoint
    from needle.model.tokenizer import get_tokenizer

    params, config = load_checkpoint(tiny_checkpoint)
    path = tmp_path_factory.mktemp("base") / "needle3.cact"
    write_export(params, config, str(path), bits=4, tokenizer=get_tokenizer(config.vocab_size))
    return str(path)


@pytest.fixture
def published_base(tiny_base_archive, monkeypatch):
    from needle.agent import fetch

    calls = []

    def fake_fetch(generation=2, dest_dir=None, force=False):
        calls.append((generation, force))
        return tiny_base_archive

    monkeypatch.setattr(fetch, "fetch_weights", fake_fetch)
    return calls


@pytest.fixture
def fake_cact():
    return _fake_cact


def _fake_cact(path, heads):
    import struct

    header_fmt, record_fmt = "<48If", "<BBHIIIIQQII"
    body = bytearray()
    records = []

    def add(dtype, shape, blob):
        records.append((dtype, len(shape), 0, *(list(shape) + [0] * (4 - len(shape))), len(body), len(blob), 0, 0))
        body.extend(blob)

    add(1, (4,), struct.pack("<4e", 0, 0, 0, 0))
    if heads:
        add(1, (len(heads),), struct.pack(f"<{len(heads)}e", *heads))
        for _ in heads:
            for _ in range(6):
                add(1, (2,), struct.pack("<2e", 0, 0))
    add(4, (3,), b"tok")
    head = struct.calcsize(header_fmt) + struct.calcsize(record_fmt) * len(records)
    fields = [0x05E12A84, len(records), 0] + [0] * 45
    out = bytearray(struct.pack(header_fmt, *fields, 0.0))
    for rec in records:
        rec = list(rec)
        rec[7] += head
        out += struct.pack(record_fmt, *rec)
    out += body
    path.write_bytes(bytes(out))


@pytest.fixture(scope="session")
def needle3_checkpoint():
    path = os.environ.get("NEEDLE_MLX_PARITY_CHECKPOINT", "checkpoints/needle3.safetensors")
    if not os.path.exists(path):
        pytest.skip(f"no Needle 3 checkpoint at {path} (set NEEDLE_MLX_PARITY_CHECKPOINT)")
    return path


@pytest.fixture
def perturbed_model():
    return _perturbed_model


def _perturbed_model(**overrides):
    """A tiny model with every parameter moved off its init, so identity-initialised
    paths (conv taps, cond_u, b2, Sinkhorn bias) are exercised."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig
    from needle.model.checkpoints import flatten, unflatten

    fields = dict(
        vocab_size=512, out_vocab=500, d_model=48, num_heads=4, num_kv_heads=2,
        num_layers=5, qk_head_dim=8, v_head_dim=12, max_seq_len=64, engram_layers=(1, 4),
        engram_slots=64, global_layers=(2, 4), sliding_window=8, mhc_lanes=4,
        qkv_conv_taps=3, dtype="float32", flash=False)
    fields.update(overrides)
    config = TransformerConfig(**fields)
    model = SimpleAttentionNetwork(config)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 8), jnp.int32))["params"]
    rng = np.random.default_rng(1)
    params = unflatten({k: (v + 0.05 * rng.standard_normal(v.shape)).astype(np.float32)
                        for k, v in flatten(jax.device_get(params)).items()})
    tokens = rng.integers(1, config.out_vocab, size=(2, 24)).astype(np.int32)
    return model, config, params, tokens
