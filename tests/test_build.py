import os
import types

import pytest

pytestmark = pytest.mark.slow


def _build_args(checkpoint, out, lora=None, layers=None, platform=None):
    return types.SimpleNamespace(checkpoint=checkpoint, lora=lora, out=out, upload=False,
                                 layers=layers, platform=platform)


def test_build_exports_loadable_cact(tiny_checkpoint, tmp_path, published_base):
    from needle.model.finetune import build_main
    from needle.model.export import read_export

    out = str(tmp_path / "tiny.cact")
    build_main(_build_args(tiny_checkpoint, out))

    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
    header, tensors = read_export(out)
    assert header["num_tensors"] > 0
    assert len(tensors) == header["num_tensors"]
    assert any(isinstance(t, (bytes, bytearray)) for t in tensors)
    assert published_base == [(3, True)]


def test_load_checkpoint_drops_the_mtp_block(tiny_checkpoint, tmp_path):
    import pickle
    import numpy as np
    from needle.model.run import load_checkpoint

    with open(tiny_checkpoint, "rb") as handle:
        checkpoint = pickle.load(handle)
    checkpoint["params"]["mtp_combine"] = {"kernel": np.ones((4, 4), np.float32)}
    path = tmp_path / "with_mtp.pkl"
    with open(path, "wb") as handle:
        pickle.dump(checkpoint, handle)
    params, _ = load_checkpoint(str(path))
    assert "mtp_combine" not in params


def test_export_round_trips_a_projection(tiny_checkpoint, tmp_path):
    import pickle
    import numpy as np
    from needle.model.export import write_export, read_export
    from needle.model.architecture import TransformerConfig, effective_kv_window
    from needle.model.tokenizer import get_tokenizer

    with open(tiny_checkpoint, "rb") as handle:
        ckpt = pickle.load(handle)
    params, config = ckpt["params"], TransformerConfig(**ckpt["config"])

    out = str(tmp_path / "rt.cact")
    write_export(params, config, out, bits=4,
                 tokenizer=get_tokenizer(config.vocab_size),
                 kv_window=effective_kv_window(config))
    header, tensors = read_export(out)

    original = np.asarray(params["stack"]["layers"]["block"]["self_attn"]["q_proj"]["kernel"][0]).T
    dequant = tensors[2]
    assert dequant.shape == original.shape
    assert np.corrcoef(dequant.ravel(), original.ravel())[0, 1] > 0.9


def test_build_from_safetensors_checkpoint(tiny_checkpoint, tiny_checkpoint_safetensors, tmp_path, published_base):
    from needle.model.finetune import build_main
    from needle.model.export import read_export
    from needle.model.run import load_checkpoint
    import numpy as np
    import jax

    a, cfg_a = load_checkpoint(tiny_checkpoint)
    b, cfg_b = load_checkpoint(tiny_checkpoint_safetensors)
    assert vars(cfg_a) == vars(cfg_b)
    assert all(np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))
    out = str(tmp_path / "from_safetensors.cact")
    build_main(_build_args(tiny_checkpoint_safetensors, out))
    header, _ = read_export(out)
    assert header["num_tensors"] > 0


def test_build_layers_exports_the_rung(tiny_checkpoint, tmp_path, published_base):
    from needle.model.finetune import build_main
    from needle.model.export import read_export

    out = str(tmp_path / "rung.cact")
    args = _build_args(tiny_checkpoint, out)
    args.layers = 3
    build_main(args)
    header, _ = read_export(out)
    assert header["num_layers"] == 3


def test_build_platform_places_the_engine_and_the_archive_together(tiny_checkpoint, tmp_path, published_base, monkeypatch):
    from needle.agent import fetch
    from needle.model.finetune import build_main
    from needle.model.export import read_layers

    def fake_platform(name, out_dir, generation=3, dest=None):
        os.makedirs(dest, exist_ok=True)
        runner = os.path.join(dest, "needle")
        with open(runner, "wb") as handle:
            handle.write(b"engine")
        return [runner]

    monkeypatch.setattr(fetch, "download_platform", fake_platform)
    folder = str(tmp_path / "pi")
    build_main(_build_args(None, folder, platform="linux-arm64"))
    assert os.path.exists(os.path.join(folder, "needle"))
    archive = os.path.join(folder, "needle3.cact")
    assert read_layers(archive) == read_layers(fetch.fetch_weights(3))
    build_main(_build_args(tiny_checkpoint, folder, platform="linux-arm64", layers=3))
    assert read_layers(archive) == 3
