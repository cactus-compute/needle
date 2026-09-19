import os
import pickle

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

    config = TransformerConfig(
        vocab_size=8192, out_vocab=8192, d_model=64, num_heads=4, num_kv_heads=2,
        num_layers=4, qk_head_dim=16, v_head_dim=16, max_seq_len=128,
        engram_layers=(1, 3), engram_slots=64, global_layers=(3,), sliding_window=32,
        mhc_lanes=2, qkv_conv_taps=3, flash=False,
    )
    model = SimpleAttentionNetwork(config)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 8), jnp.int32))["params"]
    params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)

    path = tmp_path_factory.mktemp("ckpt") / "tiny.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"format_version": 2, "params": params,
                     "config": dict(vars(config))}, handle)
    return str(path)


@pytest.fixture(scope="session")
def tiny_checkpoint_safetensors(tiny_checkpoint, tmp_path_factory):
    from needle.model.checkpoints import read_checkpoint, write_checkpoint

    path = tmp_path_factory.mktemp("ckpt") / "tiny.safetensors"
    write_checkpoint(path, read_checkpoint(tiny_checkpoint))
    return str(path)


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
