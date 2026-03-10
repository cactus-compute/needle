import hashlib
import json
import os

from datasets import load_dataset
import numpy as np

from .data import CACHE_DIR

TOUCAN_DATASET = "Agent-Ark/Toucan-1.5M"


def parse_target_tools(text):
    return [name.strip() for name in text.split(",") if name.strip()]


def _tool_type_text(spec):
    if isinstance(spec, dict) and "type" in spec and isinstance(spec["type"], str):
        return spec["type"]
    return "any"


def _tool_desc_text(spec):
    if isinstance(spec, dict) and "description" in spec and isinstance(spec["description"], str):
        return spec["description"].strip()
    return ""


def _tool_properties(tool):
    params = tool["function"]["parameters"]
    if isinstance(params, dict) and "properties" in params and isinstance(params["properties"], dict):
        return params["properties"]
    return {}


def _tool_required(tool):
    params = tool["function"]["parameters"]
    if isinstance(params, dict) and "required" in params and isinstance(params["required"], list):
        return set(params["required"])
    return set()


def _tool_matches_target(tool_name, target_name):
    return (
        tool_name == target_name
        or tool_name.endswith(f"-{target_name}")
        or tool_name.endswith(f"::{target_name}")
    )


def compact_toucan_tool(tool):
    properties = _tool_properties(tool)
    return {
        "name": tool["function"]["name"],
        "parameters": {name: _tool_type_text(spec) for name, spec in properties.items()},
    }


def format_toucan_tool(tool):
    function = tool["function"]
    properties = _tool_properties(tool)
    required = _tool_required(tool)
    lines = [
        f"name: {function['name']}",
        f"description: {_tool_desc_text(function)}",
        "parameters:",
    ]
    for name, spec in properties.items():
        req = " required" if name in required else ""
        desc = _tool_desc_text(spec)
        line = f"- {name}: {_tool_type_text(spec)}{req}"
        if desc:
            line = f"{line} | {desc}"
        lines.append(line)
    return "\n".join(lines)


def prepare_toucan_example(row):
    tools = json.loads(row["available_tools"])
    compact_tools = [compact_toucan_tool(tool) for tool in tools]
    tool_texts = [format_toucan_tool(tool) for tool in tools]
    target_tools = parse_target_tools(row["target_tools"])
    positive_indices = [
        i for i, tool in enumerate(compact_tools)
        if any(_tool_matches_target(tool["name"], target_name) for target_name in target_tools)
    ]
    return {
        "subset_name": row["subset_name"],
        "question": row["question"],
        "target_tools": target_tools,
        "tool_names": [tool["name"] for tool in compact_tools],
        "positive_indices": positive_indices,
        "tools_json": json.dumps(compact_tools, ensure_ascii=True, separators=(",", ":")),
        "tool_texts": tool_texts,
    }


def _toucan_cache_path(config, split, max_samples):
    cache_id = hashlib.md5(f"toucan_{config}_{split}_{max_samples}".encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{cache_id}_toucan.jsonl")


def cache_toucan_examples(config="Kimi-K2", split="train", max_samples=None, tokenizer=None, max_text_len=256):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _toucan_cache_path(config, split, max_samples)
    if os.path.exists(path):
        return path

    ds = load_dataset(TOUCAN_DATASET, config, split=split, streaming=True)
    with open(path, "w") as f:
        for i, row in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            ex = prepare_toucan_example(row)
            if tokenizer is not None:
                ex["question_ids"] = tokenizer.encode(ex["question"])[:max_text_len]
                ex["tool_ids"] = [tokenizer.encode(text)[:max_text_len] for text in ex["tool_texts"]]
            f.write(json.dumps(ex, ensure_ascii=True) + "\n")
    return path


def load_toucan_contrastive_data(path, max_text_len=256):
    rows = [json.loads(line) for line in open(path)]
    rows = [row for row in rows if row["positive_indices"] and len(row["positive_indices"]) < len(row["tool_names"])]
    max_tools = max(len(row["tool_ids"]) for row in rows)
    questions = np.zeros((len(rows), max_text_len), dtype=np.int32)
    tools = np.zeros((len(rows), max_tools, max_text_len), dtype=np.int32)
    labels = np.zeros((len(rows), max_tools), dtype=np.float32)
    tool_mask = np.zeros((len(rows), max_tools), dtype=np.float32)

    for i, row in enumerate(rows):
        q_ids = row["question_ids"][:max_text_len]
        questions[i, :len(q_ids)] = q_ids
        positive = set(row["positive_indices"])
        for j, tool_ids in enumerate(row["tool_ids"][:max_tools]):
            tool_ids = tool_ids[:max_text_len]
            tools[i, j, :len(tool_ids)] = tool_ids
            labels[i, j] = 1.0 if j in positive else 0.0
            tool_mask[i, j] = 1.0
    return questions, tools, labels, tool_mask


def get_toucan_batches(questions, tools, labels, tool_mask, batch_size, shuffle=True):
    n = len(questions)
    while True:
        indices = np.random.permutation(n) if shuffle else np.arange(n)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = indices[i:i + batch_size]
            yield questions[idx], tools[idx], labels[idx], tool_mask[idx]
