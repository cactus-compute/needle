import os
import pickle

import pytest


def _engine_available():
    try:
        import needle
        from needle.agent import fetch
        here = os.path.dirname(needle.__file__)
        name = fetch._lib_name()
        cache = os.path.join(os.path.expanduser("~"), ".cache",
                             "cactus-needle", fetch.ENGINE_VERSION, name)
        return os.path.exists(os.path.join(here, name)) or os.path.exists(cache)
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
        vocab_size=8192, d_model=64, num_heads=4, num_kv_heads=2, num_layers=2,
        max_seq_len=128, engram_layers=(1,), engram_slots=64, mhc_lanes=2,
        flash=False,
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
def engine_checkpoint(tmp_path_factory):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig

    config = TransformerConfig(
        vocab_size=8192, d_model=512, num_heads=8, num_kv_heads=4, num_layers=4,
        max_seq_len=2048, engram_layers=(1, 2), engram_slots=8192, mhc_lanes=4,
        flash=False,
    )
    model = SimpleAttentionNetwork(config)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 8), jnp.int32))["params"]
    params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)

    path = tmp_path_factory.mktemp("ckpt") / "engine_ckpt.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"format_version": 2, "params": params,
                     "config": dict(vars(config))}, handle)
    return str(path)
