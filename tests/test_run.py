import pytest
import argparse
import json


def test_build_prompt_passthrough_without_tools():
    from needle.model.run import build_prompt

    assert build_prompt("hello world") == "hello world"
    assert build_prompt("hello world", tools=[]) == "hello world"
    assert build_prompt("hello world", tools=None) == "hello world"


def test_build_prompt_matches_training_template():
    from needle.model.finetune import render_example
    from needle.model.run import build_prompt
    from needle.model.tokenizer import IM_START, TOOLS_START

    tools = [{"name": "f", "parameters": {"type": "object", "properties": {}}}]
    expected, _ = render_example({"query": "do the thing", "tools": tools})
    out = build_prompt("do the thing", tools=tools)

    assert out == expected
    assert IM_START in out
    assert TOOLS_START in out
    assert "do the thing" in out


def test_needle2_checkpoint_is_rejected_with_a_version_hint(tmp_path):
    import pickle
    import pytest
    from needle.model.run import load_checkpoint

    path = tmp_path / "needle2.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"format_version": 2, "params": {}, "config": {
            "d_model": 512, "attn_dim": 512, "num_heads": 8, "num_layers": 27}}, handle)
    with pytest.raises(ValueError, match="Needle 2 checkpoint.*cactus-needle<3"):
        load_checkpoint(str(path))


def test_main_loads_tools_file_and_runs(tiny_checkpoint, tmp_path, capsys):
    from needle.model.run import main

    tools = [{"name": "f", "parameters": {"type": "object", "properties": {}}}]
    path = tmp_path / "tools.json"
    path.write_text(json.dumps(tools))

    main(argparse.Namespace(
        checkpoint=tiny_checkpoint, query="use f", tools=str(path),
        max_len=4, seed=0, temperature=0.0))
    out = capsys.readouterr().out

    assert "<tools>" in out
    assert '"name":"f"' in out


def test_missing_checkpoint_is_looked_up_under_checkpoints_then_at_the_repo_root(monkeypatch, tmp_path):
    from huggingface_hub.errors import EntryNotFoundError
    import huggingface_hub
    from needle.agent import fetch
    from needle.model import run

    attempted = []
    registered = []

    def fake_download(**kwargs):
        attempted.append(kwargs["filename"])
        raise EntryNotFoundError("missing")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(fetch, "_register_download", lambda generation: registered.append(generation))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        run.load_checkpoint("needle3.safetensors")
    assert attempted == ["checkpoints/needle3.safetensors", "needle3.safetensors"]
    assert registered == [3]
