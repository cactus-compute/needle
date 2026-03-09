"""
Unified tool-calling dataset builder.

Loads four HuggingFace tool-calling datasets and converts them all into the
xlam-function-calling-60k schema:

    query   : str   – the user's natural-language request
    answers : str   – JSON list of {"name": ..., "arguments": {...}} tool calls
    tools   : str   – JSON list of available tool definitions
    source  : str   – which dataset the example came from

Datasets and their native formats:

1. Salesforce/xlam-function-calling-60k
   Already in the target schema (query, answers, tools).

2. Salesforce/APIGen-MT-5k
   Multi-turn conversations with human/gpt/function_call/observation roles.
   We extract the first user message as the query and collect all function_call
   messages (excluding internal "think" calls) as answers.

3. glaiveai/glaive-function-calling-v2
   Tool definitions embedded in the system prompt; chat is a flat string with
   USER:/A: role markers and <functioncall> tags. Examples with empty answers
   (answers=[]) are kept so the model learns when NOT to call functions.

4. Team-ACE/ToolACE
   Tool definitions as a JSON array in the system prompt; function calls use a
   bracket syntax [FuncName(key=val, ...)] where names may contain spaces. Only
   examples with at least one parsed function call are kept.
"""

import json
import re
from collections import Counter

from datasets import load_dataset, Dataset, concatenate_datasets


DATASETS = {
    "xlam-function-calling-60k": "Salesforce/xlam-function-calling-60k",
    "APIGen-MT-5k": "Salesforce/APIGen-MT-5k",
    "glaive-function-calling": "glaiveai/glaive-function-calling-v2",
    "ToolACE": "Team-ACE/ToolACE",
}


def convert_xlam(dataset):
    return dataset.select_columns(["query", "answers", "tools"])


def convert_apigen_mt(dataset):
    rows = []
    for ex in dataset:
        convs = ex["conversations"]
        tools = ex["tools"]

        query = None
        answers = []
        for msg in convs:
            if msg["from"] == "human" and query is None:
                query = msg["value"]
            elif msg["from"] == "function_call":
                try:
                    fc = json.loads(msg["value"])
                    if fc.get("name") != "think":
                        answers.append(fc)
                except json.JSONDecodeError:
                    pass

        if query:
            rows.append({
                "query": query,
                "answers": json.dumps(answers),
                "tools": tools,
            })
    return Dataset.from_list(rows)


def convert_glaive(dataset):
    rows = []
    for ex in dataset:
        tools = _extract_json_objects(ex["system"])
        query, answers = _parse_glaive_chat(ex["chat"])
        if query:
            rows.append({
                "query": query,
                "answers": json.dumps(answers),
                "tools": json.dumps(tools),
            })
    return Dataset.from_list(rows)


def _extract_json_objects(text):
    """Extract top-level JSON objects with a 'name' field from text."""
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                    if "name" in obj:
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return objects


def _parse_glaive_chat(chat):
    """Extract first user query and all function calls from glaive chat."""
    query = None
    answers = []

    user_match = re.search(r'USER:\s*(.*?)(?=\n\s*(?:A:|ASSISTANT:)|$)', chat, re.DOTALL)
    if user_match:
        query = user_match.group(1).strip()

    for m in re.finditer(r'<functioncall>\s*(\{.+?\})\s*(?:<\|endoftext\|>|$)', chat, re.DOTALL):
        raw = m.group(1)
        raw = re.sub(r"'(\{.*?\})'", r'"\1"', raw)
        try:
            fc = json.loads(raw)
        except json.JSONDecodeError:
            raw2 = re.sub(
                r"'(\{.*?\})'",
                lambda x: json.dumps(x.group(1)),
                m.group(1),
            )
            try:
                fc = json.loads(raw2)
            except json.JSONDecodeError:
                continue
        args = fc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        answers.append({"name": fc["name"], "arguments": args})

    return query, answers


def convert_toolace(dataset):
    rows = []
    for ex in dataset:
        tools = _parse_toolace_tools(ex["system"])
        convs = ex["conversations"]

        query = None
        answers = []
        for msg in convs:
            if msg["from"] == "user" and query is None:
                query = msg["value"]
            elif msg["from"] == "assistant" and query is not None:
                calls = _parse_toolace_calls(msg["value"])
                if calls:
                    answers.extend(calls)
                    break

        if query and answers:
            rows.append({
                "query": query,
                "answers": json.dumps(answers),
                "tools": json.dumps(tools),
            })
    return Dataset.from_list(rows)


def _parse_toolace_tools(system_text):
    """Extract tool list from ToolACE system prompt (JSON array after 'invoke:')."""
    match = re.search(r'invoke:\s*(\[)', system_text)
    if not match:
        return []
    start = match.start(1)
    depth = 0
    for i in range(start, len(system_text)):
        if system_text[i] == '[':
            depth += 1
        elif system_text[i] == ']':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(system_text[start:i + 1])
                except json.JSONDecodeError:
                    return []
    return []


def _parse_toolace_calls(text):
    """Parse [FuncName(key=val, ...)] calls. Function names may contain spaces."""
    answers = []
    bracket_match = re.search(r'\[(.+)\]', text, re.DOTALL)
    if not bracket_match:
        return answers

    inner = bracket_match.group(1)
    for m in re.finditer(r'([\w][\w ]*?\w|\w)\(([^)]*)\)', inner):
        name = m.group(1).strip()
        args_str = m.group(2)
        arguments = {}
        for arg_m in re.finditer(
            r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'|(\[[^\]]*\])|([^,)]+))',
            args_str,
        ):
            key = arg_m.group(1)
            val = next(
                (g for g in (arg_m.group(2), arg_m.group(3), arg_m.group(4), arg_m.group(5)) if g is not None),
                None,
            )
            if val is not None:
                val = val.strip()
                try:
                    val = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
            arguments[key] = val
        answers.append({"name": name, "arguments": arguments})
    return answers


def compact_tools(tools_json):
    """Strip verbose schema descriptions, keep only function name + param name/type."""
    try:
        tools = json.loads(tools_json)
    except (json.JSONDecodeError, TypeError):
        return tools_json
    compact = []
    for t in tools:
        params = {}
        raw_params = t.get("parameters", {})
        props = raw_params.get("properties", raw_params) if isinstance(raw_params, dict) else {}
        for pname, pdef in props.items():
            if isinstance(pdef, dict):
                params[pname] = pdef.get("type", "string")
            else:
                params[pname] = str(pdef)
        compact.append({"name": t["name"], "parameters": params})
    return json.dumps(compact)


CONVERTERS = {
    "xlam-function-calling-60k": convert_xlam,
    "APIGen-MT-5k": convert_apigen_mt,
    "glaive-function-calling": convert_glaive,
    "ToolACE": convert_toolace,
}

_PLACEHOLDER_VALUES = {
    "john_doe", "jane_doe", "john.doe", "jane.doe",
    "john_doe_456", "jane_doe_123",
    "john.doe@example.com", "jane.doe@example.com",
    "user@example.com", "test@example.com",
    "example@example.com", "boss@company.com",
    "1234567890", "0987654321",
    "123 main st", "123 main street",
}


def _has_placeholder_args(answers_json):
    """Check if any argument values are known placeholders."""
    try:
        answers = json.loads(answers_json)
    except (json.JSONDecodeError, TypeError):
        return False
    if not answers:
        return False
    for call in answers:
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(args, dict):
            continue
        for val in args.values():
            if isinstance(val, str) and val.lower().strip() in _PLACEHOLDER_VALUES:
                return True
    return False


def _answer_grounded(query, answers_json):
    """Check that at least one non-trivial argument value appears in the query.

    For examples with function calls, this catches cases where the answer
    contains hallucinated arguments that have nothing to do with the query.
    Skips examples with no answers (no-call examples) — those are always kept.
    Also skips examples where all arguments are booleans/numbers/short enums,
    since those don't need to be grounded in the query text.
    """
    try:
        answers = json.loads(answers_json)
    except (json.JSONDecodeError, TypeError):
        return True
    if not answers:
        return True

    query_lower = query.lower()
    has_string_arg = False

    for call in answers:
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(args, dict):
            continue
        for val in args.values():
            if isinstance(val, str) and len(val) >= 3:
                has_string_arg = True
                val_lower = val.lower().strip()
                if val_lower in query_lower:
                    return True
                val_words = val_lower.split()
                if len(val_words) > 1:
                    matches = sum(1 for w in val_words if len(w) >= 3 and w in query_lower)
                    if matches >= len(val_words) * 0.5:
                        return True

    if not has_string_arg:
        return True

    return False


def load_and_combine():
    parts = []
    for name, hf_id in DATASETS.items():
        print(f"\nLoading {name} ({hf_id})...")
        try:
            ds = load_dataset(hf_id)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Converting to xlam format...")
        converted = CONVERTERS[name](ds["train"])
        converted = converted.add_column("source", [name] * len(converted))

        before = len(converted)
        converted = converted.filter(
            lambda ex: not _has_placeholder_args(ex["answers"]) and _answer_grounded(ex["query"], ex["answers"]),
            desc=f"Filtering {name}",
        )
        dropped = before - len(converted)
        if dropped:
            print(f"  Filtered out {dropped} low-quality examples ({dropped*100/before:.1f}%)")

        print(f"  -> {len(converted)} examples")
        parts.append(converted)

    combined = concatenate_datasets(parts)

    print(f"\nCompacting tool schemas...")
    combined = combined.map(
        lambda ex: {"tools": compact_tools(ex["tools"])},
        desc="Compacting tools",
    )

    empty_count = sum(1 for ex in combined if json.loads(ex["answers"]) == [])
    print(f"\n{'='*60}")
    print(f"Combined dataset: {len(combined)} examples ({empty_count} with empty answers = no-call examples)")
    print(f"Columns: {combined.column_names}")

    counts = Counter(combined["source"])
    print(f"\nBreakdown:")
    for src, cnt in counts.items():
        print(f"  {src}: {cnt}")

    seen = set()
    for i in range(len(combined)):
        src = combined[i]["source"]
        if src in seen:
            continue
        seen.add(src)
        print(f"\n--- Sample from {src} ---")
        for k in ["query", "answers", "tools"]:
            v = str(combined[i][k])[:200]
            print(f"  {k}: {v}")

    return combined


if __name__ == "__main__":
    combined = load_and_combine()
    out_path = "data/tool_calls"
    combined.save_to_disk(out_path)
    print(f"\nSaved to {out_path}/")

    from datasets import load_from_disk
    reloaded = load_from_disk(out_path)
    assert len(reloaded) == len(combined), f"Row count mismatch: {len(reloaded)} != {len(combined)}"
    assert reloaded.column_names == combined.column_names, "Column mismatch"
    assert reloaded[0] == combined[0], "First row mismatch"
    print(f"Verified: {len(reloaded)} rows, columns {reloaded.column_names}")
