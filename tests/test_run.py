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
