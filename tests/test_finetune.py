import json
import os
import pickle
import types

import pytest

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
    out = tmp_path / "adapter.pkl"
    progress = []
    finetune_local(_finetune_args(data, tiny_checkpoint, out, tmp_path / "ck", qat_bits=4),
                   progress=progress.append)

    assert any("loss" in m for m in progress)
    assert any("CQ W4 STE + A8" in m for m in progress)
    assert out.exists()
    with open(out, "rb") as handle:
        adapter = pickle.load(handle)
    assert adapter["rank"] == 4
    assert abs(adapter["scale"] - 2.0) < 1e-6
    assert adapter["base"] == tiny_checkpoint
    assert adapter["qat_bits"] == 4
    assert adapter["lora"]
    for value in adapter["lora"].values():
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
    adapter = tmp_path / "adapter.pkl"
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
    adapter_path = tmp_path / "mixed-adapter.pkl"
    progress = []
    finetune_local(_finetune_args(data, str(mixed_checkpoint), adapter_path,
                                  tmp_path / "ck"), progress=progress.append)
    assert any(f"mixed[{bit_map}]" in message for message in progress)
    with open(adapter_path, "rb") as handle:
        adapter = pickle.load(handle)
    assert adapter["qat_bits"] is None
    assert adapter["qat_bits_map"] == bit_map

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

    out = tmp_path / "seeded-adapter.pkl"
    args = _finetune_args(data, tiny_checkpoint, out, tmp_path / "ck", qat_bits="none")
    args.seed = 17
    args.val_split = 0.0
    finetune_local(args)

    with out.open("rb") as handle:
        adapter = pickle.load(handle)
    assert adapter["seed"] == 17
