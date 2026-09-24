import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
jax = pytest.importorskip("jax")

pytestmark = pytest.mark.slow


def _jax_logits(model, params, tokens):
    from needle.model.quantize import WEIGHT_BITS, cq_ste_params

    return model.apply({"params": cq_ste_params(params, WEIGHT_BITS)},
                       jax.numpy.asarray(tokens), quant=True)


def _mlx_logits(config, params, tokens):
    from needle.model.architecture_mlx import forward, to_mlx
    from needle.model.quantize import WEIGHT_BITS
    from needle.model.quantize_mlx import cq_ste_params

    return forward(cq_ste_params(to_mlx(params), WEIGHT_BITS), config, mx.array(tokens),
                   quant=True)


@pytest.mark.parametrize("bits", [2, 4])
def test_cq_quantize_matches_jax(bits):
    from needle.model import quantize, quantize_mlx

    w = np.random.default_rng(bits).standard_normal((3, 200, 96)).astype(np.float32)
    ref = np.asarray(quantize.cq_quantize(jax.numpy.asarray(w), bits))
    got = np.asarray(quantize_mlx.cq_quantize(mx.array(w), bits))
    assert np.mean(np.abs(got - ref) > 1e-5) < 1e-3
    np.testing.assert_allclose(np.median(np.abs(got - ref)), 0, atol=1e-6)


@pytest.mark.parametrize("flash", [False, True])
@pytest.mark.parametrize("kv_bits", [8, 4])
def test_quantized_forward_matches_jax(flash, kv_bits, deploy_numerics, perturbed_model):
    """An ulp-level difference can move a value across a rounding boundary; that shows
    up as a few whole positions off, never as a drift across all of them."""
    model, config, params, tokens = perturbed_model(flash=flash, kv_bits=kv_bits)
    deploy_numerics(config)
    ref = np.asarray(_jax_logits(model, params, tokens))
    got = np.asarray(_mlx_logits(config, params, tokens))
    exact = np.all(np.abs(got - ref) <= 5e-5, axis=-1)
    assert exact.mean() >= 0.9
    assert np.abs(got - ref).max() < 1e-2


def test_published_checkpoint_quantized_loss_matches_jax(needle3_checkpoint, deploy_numerics):
    """Under QAT the published model flips on ulp-level differences (JAX against itself
    with 1e-7 weight noise agrees on ~96% of argmaxes), so it is compared by loss."""
    import optax
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.run import load_checkpoint

    params, config = load_checkpoint(needle3_checkpoint)
    config.dtype = "float32"
    params = jax.tree.map(lambda a: np.asarray(a, np.float32), params)
    deploy_numerics(config)
    tokens = np.random.default_rng(0).integers(14, config.vocab_size, (2, 96)).astype(np.int32)

    def loss(logits):
        return float(optax.softmax_cross_entropy_with_integer_labels(
            np.asarray(logits)[:, :-1], tokens[:, 1:]).mean())

    ref = _jax_logits(SimpleAttentionNetwork(config), params, tokens)
    got = _mlx_logits(config, params, tokens)
    assert abs(loss(got) - loss(ref)) < 1e-2 * loss(ref)
    assert (np.asarray(got).argmax(-1) == np.asarray(ref).argmax(-1)).mean() > 0.9


def test_ab_scales_checkpoints_are_rejected():
    from needle.model.quantize_mlx import cq_ste_params

    with pytest.raises(NotImplementedError):
        cq_ste_params({"ab_scales/stack/a": mx.zeros((2, 2))}, 4)
