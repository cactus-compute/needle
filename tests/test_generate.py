import json
import importlib

import pytest


def test_parse_array_variants():
    from needle.model.finetune import _parse_array

    assert len(_parse_array('[{"query":"q","answers":[]}]')) == 1
    assert len(_parse_array('Sure!\n[{"query":"q","answers":[]}]\nDone.')) == 1
    assert _parse_array("no array at all") == []
    assert _parse_array("[not valid json]") == []

    mixed = '[{"query":"q","answers":[]},{"nope":1},{"query":"only"}]'
    assert len(_parse_array(mixed)) == 1


def test_collect_tools_dedups_by_name():
    from needle.model.finetune import _collect_tools

    examples = [{"tools": [{"name": "a"}, {"name": "b"}]},
                {"tools": [{"name": "a"}, {"name": "c"}]}]
    assert [t["name"] for t in _collect_tools(examples)] == ["a", "b", "c"]


def test_generate_examples_attaches_tools(monkeypatch):
    from needle.model import finetune

    tools = [{"name": "f", "parameters": {"type": "object", "properties": {}}}]
    monkeypatch.setattr(finetune, "_openrouter",
                        lambda *a, **k: '[{"query":"q","reasoning":"r",'
                                        '"answers":[{"name":"f","arguments":{}}]}]')
    rows = finetune.generate_examples(tools, n=1, api_key="test-key")
    assert len(rows) == 1
    assert rows[0]["tools"] == tools


def test_generate_examples_requires_api_key(monkeypatch):
    from needle.model import finetune

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        finetune.generate_examples([{"name": "f"}], n=1)


def test_openrouter_url_can_be_overridden_from_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_URL", "https://gateway.example.test/v1/chat")
    from needle.model import finetune
    importlib.reload(finetune)

    try:
        assert finetune.OPENROUTER_URL == "https://gateway.example.test/v1/chat"
    finally:
        monkeypatch.delenv("OPENROUTER_URL", raising=False)
        importlib.reload(finetune)


def _unique_generator():
    counter = {"i": 0}

    def fake(tools, n=1, **k):
        rows = []
        for _ in range(n):
            counter["i"] += 1
            rows.append({"tools": tools, "query": "gen %d" % counter["i"], "answers": []})
        return rows
    return fake


def test_generate_dataset_hits_target_concurrently(monkeypatch):
    from needle.model import finetune

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(finetune, "generate_examples", _unique_generator())
    rows = finetune.generate_dataset([{"name": "f"}], 10, batch_size=3, workers=4)
    assert len(rows) == 10
    assert len({finetune._dedup_key(r) for r in rows}) == 10


def test_generate_dataset_dedups_and_terminates(monkeypatch):
    from needle.model import finetune

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(finetune, "generate_examples",
                        lambda tools, n=1, **k: [{"tools": tools, "query": "same", "answers": []}])
    rows = finetune.generate_dataset([{"name": "f"}], 5, batch_size=1, workers=2)
    assert len(rows) == 1


def test_augment_jsonl_appends_generated(monkeypatch, tmp_path):
    from needle.model import finetune

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    tools = [{"name": "f", "parameters": {"type": "object", "properties": {}}}]
    src = tmp_path / "seed.jsonl"
    with open(src, "w") as handle:
        handle.write(json.dumps({"tools": tools, "query": "seed", "answers": []}) + "\n")

    monkeypatch.setattr(finetune, "generate_examples", _unique_generator())
    out = finetune.augment_jsonl(str(src), num_samples=3, out_path=str(tmp_path / "out.jsonl"))
    lines = [line for line in open(out) if line.strip()]
    assert len(lines) == 1 + 3
