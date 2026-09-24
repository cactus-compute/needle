import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
jax = pytest.importorskip("jax")

pytestmark = pytest.mark.slow


def _jax_logits(model, params, tokens, **kw):
    return model.apply({"params": params}, jax.numpy.asarray(tokens), **kw)


def _mlx_logits(config, params, tokens, **kw):
    from needle.model.architecture_mlx import forward, to_mlx

    return forward(to_mlx(params), config, mx.array(tokens), **kw)


@pytest.mark.parametrize("flash", [False, True])
def test_forward_matches_jax(flash, perturbed_model):
    model, config, params, tokens = perturbed_model(flash=flash)
    ref = np.asarray(_jax_logits(model, params, tokens))
    got = np.asarray(_mlx_logits(config, params, tokens))
    assert got.shape == ref.shape == (2, 24, 500)
    np.testing.assert_allclose(got, ref, atol=5e-5)


def test_bfloat16_forward_is_as_close_to_float32_as_jax(perturbed_model):
    model, config, params, tokens = perturbed_model()
    ref32 = np.asarray(_jax_logits(model, params, tokens))
    bf16_model, bf16_config, _, _ = perturbed_model(dtype="bfloat16")
    jax_err = np.abs(np.asarray(_jax_logits(bf16_model, params, tokens)) - ref32).mean()
    mlx_err = np.abs(np.asarray(_mlx_logits(bf16_config, params, tokens)) - ref32).mean()
    assert 0 < mlx_err < 2 * jax_err


def test_split_hadamard_perms_match_jax(perturbed_model):
    model, config, params, tokens = perturbed_model(ladder_widths=(24,))
    np.testing.assert_allclose(np.asarray(_mlx_logits(config, params, tokens)),
                               np.asarray(_jax_logits(model, params, tokens)), atol=5e-5)


def test_exit_depth_matches_jax(perturbed_model):
    model, config, params, tokens = perturbed_model()
    ref_full, ref_exit = _jax_logits(model, params, tokens, exit_depth=3)
    got_full, got_exit = _mlx_logits(config, params, tokens, exit_depth=3)
    np.testing.assert_allclose(np.asarray(got_full), np.asarray(ref_full), atol=5e-5)
    np.testing.assert_allclose(np.asarray(got_exit), np.asarray(ref_exit), atol=5e-5)

    ref_sub = _jax_logits(model, params, tokens, exit_depth=3, subnetwork_only=True)
    got_sub = _mlx_logits(config, params, tokens, exit_depth=3, subnetwork_only=True)
    np.testing.assert_allclose(np.asarray(got_sub), np.asarray(ref_sub), atol=5e-5)


def test_padding_mask_matches_jax(perturbed_model):
    from needle.model.architecture import make_causal_mask, make_padding_mask

    model, config, params, tokens = perturbed_model()
    tokens[:, -5:] = config.pad_token_id
    mask = make_causal_mask(tokens.shape[1]) & make_padding_mask(tokens, config.pad_token_id)
    ref = _jax_logits(model, params, tokens, mask=mask)
    got = _mlx_logits(config, params, tokens, mask=mx.array(np.asarray(mask)))
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=5e-5)


def test_published_checkpoint_matches_jax(needle3_checkpoint):
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.run import load_checkpoint

    params, config = load_checkpoint(needle3_checkpoint)
    config.dtype = "float32"
    params = jax.tree.map(lambda a: np.asarray(a, np.float32), params)
    tokens = np.random.default_rng(0).integers(14, config.vocab_size, (2, 96)).astype(np.int32)

    ref = np.asarray(_jax_logits(SimpleAttentionNetwork(config), params, tokens))
    got = np.asarray(_mlx_logits(config, params, tokens))
    np.testing.assert_allclose(got, ref, atol=5e-4)
    assert (got.argmax(-1) == ref.argmax(-1)).all()
