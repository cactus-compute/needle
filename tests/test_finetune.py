import json
import os
import pickle
import sys
import types

import numpy as np
import pytest
from safetensors import safe_open

pytestmark = pytest.mark.slow

TOOLS = [{"name": "send_email", "parameters": {"type": "object", "properties": {
    "to": {"type": "string"}, "subject": {"type": "string"}}, "required": ["to"]}}]


def _write_data(path):
    rows = [
        {"tools": TOOLS, "query": "email a@b.com about lunch",
         "reasoning": "to from query", "answers": [
             {"name": "send_email", "arguments": {"to": "a@b.com", "subject": "lunch"}}]},
        {"tools": TOOLS, "query": "nothing actionable here",
         "reasoning": "off-topic", "answers": []},
    ]
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _finetune_args(data, checkpoint, out, ckpt_dir, qat_bits="auto"):
    return types.SimpleNamespace(
        jsonl_path=str(data), checkpoint=checkpoint, epochs=1, batch_size=2,
        lr=1e-3, lora_rank=4, lora_alpha=8.0, max_len=64, generate=0,
        model=None, checkpoint_dir=str(ckpt_dir), out=str(out), qat_bits=qat_bits)


def test_finetune_writes_adapter(tiny_checkpoint, tmp_path):
    from needle.model.finetune import finetune_local

    data = tmp_path / "data.jsonl"
    _write_data(data)
    out = tmp_path / "adapter.safetensors"
    progress = []
    finetune_local(_finetune_args(data, tiny_checkpoint, out, tmp_path / "ck", qat_bits=4),
                   progress=progress.append)

    assert any("loss" in m for m in progress)
    assert any("CQ W4 STE + A8" in m for m in progress)
    assert out.exists()
    with safe_open(out, framework="np") as handle:
        metadata = handle.metadata()
    assert metadata["rank"] == "4"
    assert abs(float(metadata["scale"]) - 2.0) < 1e-6
    assert metadata["base"] == tiny_checkpoint
    assert json.loads(metadata["qat_bits"]) == 4
    from needle.model.finetune import _load_lora_adapter
    lora, _, _ = _load_lora_adapter(str(out))
    assert lora
    for value in lora.values():
        assert "A" in value and "B" in value

    from needle.model.finetune import build_main
    with pytest.raises(ValueError, match="trained for CQ W4"):
        build_main(types.SimpleNamespace(checkpoint=tiny_checkpoint, lora=str(out),
                                         out=str(tmp_path / "wrong.cact"),
                                         upload=False, bits="2"))


def test_finetune_then_build_merges(tiny_checkpoint, tmp_path):
    from needle.model.finetune import finetune_local, build_main
    from needle.model.export import read_export

    data = tmp_path / "data.jsonl"
    _write_data(data)
    adapter = tmp_path / "adapter.safetensors"
    finetune_local(_finetune_args(data, tiny_checkpoint, adapter, tmp_path / "ck"))

    out = str(tmp_path / "merged.cact")
    build_main(types.SimpleNamespace(checkpoint=tiny_checkpoint, lora=str(adapter),
                                     out=out, upload=False, bits="4"))
    assert os.path.exists(out)
    header, _ = read_export(out)
    assert header["num_tensors"] > 0


def test_auto_qat_preserves_checkpoint_mixed_bit_map(tiny_checkpoint, tmp_path):
    from needle.model.finetune import finetune_local, build_main
    from needle.model.export import read_export

    with open(tiny_checkpoint, "rb") as handle:
        checkpoint = pickle.load(handle)
    bit_map = "embedding=4,mhc=4,default=2"
    checkpoint["config"]["weight_bits"] = bit_map
    mixed_checkpoint = tmp_path / "mixed.pkl"
    with open(mixed_checkpoint, "wb") as handle:
        pickle.dump(checkpoint, handle)

    data = tmp_path / "data.jsonl"
    _write_data(data)
    adapter_path = tmp_path / "mixed-adapter.safetensors"
    progress = []
    finetune_local(_finetune_args(data, str(mixed_checkpoint), adapter_path,
                                  tmp_path / "ck"), progress=progress.append)
    assert any(f"mixed[{bit_map}]" in message for message in progress)
    from needle.model.finetune import _load_lora_adapter
    _, _, adapter_meta = _load_lora_adapter(str(adapter_path))
    assert adapter_meta["qat_bits"] is None
    assert adapter_meta["qat_bits_map"] == bit_map

    out = tmp_path / "mixed.cact"
    build_main(types.SimpleNamespace(checkpoint=str(mixed_checkpoint),
                                     lora=str(adapter_path), out=str(out),
                                     upload=False, bits=None))
    header, tensors = read_export(out)
    assert header["num_tensors"] == len(tensors)

    with pytest.raises(ValueError, match="mixed CQ bit map"):
        build_main(types.SimpleNamespace(checkpoint=str(mixed_checkpoint),
                                         lora=str(adapter_path),
                                         out=str(tmp_path / "wrong.cact"),
                                         upload=False, bits="4"))


def test_finetune_rng_is_controlled_by_seed():
    from needle.model.finetune import _training_rng

    a = _training_rng(17)
    b = _training_rng(17)
    c = _training_rng(18)
    a_orders = [a.permutation(12).tolist() for _ in range(3)]
    b_orders = [b.permutation(12).tolist() for _ in range(3)]
    c_orders = [c.permutation(12).tolist() for _ in range(3)]
    assert a_orders == b_orders
    assert a_orders != c_orders


def test_finetune_adapter_records_realized_seed(tiny_checkpoint, tmp_path):
    from needle.model.finetune import finetune_local

    data = tmp_path / "data.jsonl"
    rows = []
    for i in range(4):
        rows.append({
            "tools": TOOLS,
            "query": f"email user{i}@example.com about item {i}",
            "answers": [{"name": "send_email", "arguments": {"to": f"user{i}@example.com", "subject": f"item {i}"}}],
        })
    with data.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    out = tmp_path / "seeded-adapter.safetensors"
    args = _finetune_args(data, tiny_checkpoint, out, tmp_path / "ck", qat_bits="none")
    args.seed = 17
    args.val_split = 0.0
    finetune_local(args)

    from needle.model.finetune import _load_lora_adapter
    _, _, adapter_meta = _load_lora_adapter(str(out))
    assert adapter_meta["seed"] == 17


def _mark_pickle_loaded(marker):
    with open(marker, "w") as handle:
        handle.write("loaded")
    return object()


class _PicklePayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return _mark_pickle_loaded, (self.marker,)


def test_adapter_paths_and_arrays_round_trip_without_collisions(tmp_path):
    from needle.model.finetune import _load_lora_adapter, _save_lora_adapter

    lora = {
        ("stack", "layers", "a/b", "kernel"): {
            "A": np.arange(6, dtype=np.float32).reshape(2, 3),
            "B": np.arange(6, dtype=np.float16).reshape(3, 2),
        },
        ("stack", "layers/a", "b", "kernel"): {
            "A": np.arange(4, dtype=np.float16).reshape(2, 2),
            "B": np.arange(4, dtype=np.float32).reshape(2, 2),
        },
    }
    out = tmp_path / "adapter.custom-extension"
    _save_lora_adapter(str(out), lora, 2.5, "checkpoints/base.pkl", 3)

    loaded, scale, meta = _load_lora_adapter(str(out))

    assert scale == 2.5
    assert meta == {"qat_bits": None, "qat_bits_map": None, "seed": None}
    assert loaded.keys() == lora.keys()
    for path, matrices in lora.items():
        for name, value in matrices.items():
            np.testing.assert_array_equal(loaded[path][name], value)
            assert loaded[path][name].dtype == value.dtype


def test_build_rejects_pickle_adapter_without_deserializing(monkeypatch, tmp_path):
    from safetensors import SafetensorError
    from needle.model.finetune import build_main

    jax = types.ModuleType("jax")
    jax_numpy = types.ModuleType("jax.numpy")
    jax.numpy = jax_numpy
    run = types.ModuleType("needle.model.run")
    run.load_checkpoint = lambda *args, **kwargs: ({}, object(), {})
    architecture = types.ModuleType("needle.model.architecture")
    architecture.effective_kv_window = lambda config: 0
    export = types.ModuleType("needle.model.export")
    export.write_export = lambda *args, **kwargs: pytest.fail("export should not run")
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", jax_numpy)
    monkeypatch.setitem(sys.modules, "needle.model.run", run)
    monkeypatch.setitem(sys.modules, "needle.model.architecture", architecture)
    monkeypatch.setitem(sys.modules, "needle.model.export", export)

    marker = tmp_path / "pickle-loaded"
    adapter = tmp_path / "legacy.pkl"
    with open(adapter, "wb") as handle:
        pickle.dump(_PicklePayload(str(marker)), handle)
    out = tmp_path / "should-not-exist.cact"

    with pytest.raises(SafetensorError):
        build_main(types.SimpleNamespace(
            checkpoint="base.pkl", lora=str(adapter), out=str(out),
            upload=False, bits="4"))

    assert not marker.exists()
    assert not out.exists()


def test_playground_uses_safetensors_adapter_for_build(monkeypatch, tmp_path):
    from needle.model import finetune
    from needle.playground import server

    calls = {}
    monkeypatch.setattr(server, "_DOWNLOADS", tmp_path)
    monkeypatch.setattr(finetune, "generate_dataset", lambda *args, **kwargs: [])

    def fake_finetune(args, progress=None):
        calls["finetune"] = args.out

    def fake_build(args):
        calls["build"] = args.lora
        with open(args.out, "wb") as handle:
            handle.write(b"cact")

    monkeypatch.setattr(finetune, "finetune_local", fake_finetune)
    monkeypatch.setattr(finetune, "build_main", fake_build)

    engine = types.SimpleNamespace(load_weights=lambda path: calls.setdefault("weights", path))
    server._finetune_worker("[]", "key", 1, engine)

    expected = str(tmp_path / "needle_playground_lora.safetensors")
    assert calls["finetune"] == expected
    assert calls["build"] == expected
    assert calls["weights"] == str(tmp_path / "needle_tuned.cact")
