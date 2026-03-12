"""
Unified tool-calling dataset builder.

Loads HuggingFace tool-calling datasets and converts them into a common schema:

    query   : str   – the user's natural-language request
    answers : str   – JSON list of {"name": ..., "arguments": {...}} tool calls
    tools   : str   – JSON list of available tool definitions
    source  : str   – which dataset the example came from

Datasets:

1. Salesforce/xlam-function-calling-60k
   Already in the target schema (query, answers, tools).

2. glaiveai/glaive-function-calling-v2
   Tool definitions embedded in the system prompt; chat is a flat string with
   USER:/A: role markers and <functioncall> tags. Examples with empty answers
   (answers=[]) are kept so the model learns when NOT to call functions.

3. Agent-Ark/Toucan-1.5M (OSS, Kimi-K2, Qwen3 subsets)
   Large-scale tool-calling dataset. Tool calls extracted from 'messages' column
   (assistant messages with function_call objects), filtered by 'target_tools'.
   Tool schemas converted from OpenAI format to flat format.

4. Team-ACE/ToolACE
   Synthetic tool-calling dataset (11.3k) with dual-layer verification.
   Tool definitions in system prompt JSON; calls use bracket notation
   [func(key=val)]. Grounding filter skipped (pre-verified).
"""

import ast
import json
import random
import re
from collections import Counter

from datasets import load_dataset, Dataset, concatenate_datasets


DATASETS = {
    "xlam-function-calling-60k": "Salesforce/xlam-function-calling-60k",
    "glaive-function-calling": "glaiveai/glaive-function-calling-v2",
    "toucan-oss": ("Agent-Ark/Toucan-1.5M", "OSS"),
    "toucan-kimi": ("Agent-Ark/Toucan-1.5M", "Kimi-K2"),
    "toucan-qwen": ("Agent-Ark/Toucan-1.5M", "Qwen3"),
    "toucan-sft": ("Agent-Ark/Toucan-1.5M", "SFT"),
    "toolace": "Team-ACE/ToolACE",
    "nemotron-tc": ("nvidia/Nemotron-Agentic-v1", None, "tool_calling"),
}


def convert_xlam(dataset):
    return dataset.select_columns(["query", "answers", "tools"])


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


def _convert_openai_tools(tools_json):
    """Convert OpenAI-format tools to flat format.

    OpenAI: [{"type": "function", "function": {"name": ..., "parameters": {"type": "object", "properties": {...}}}}]
    Flat:   [{"name": ..., "description": ..., "parameters": {"param_name": {"type": ..., "description": ...}}}]
    """
    try:
        tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
    except (json.JSONDecodeError, TypeError):
        return []
    flat = []
    for t in tools:
        if isinstance(t, dict) and "function" in t:
            fn = t["function"]
        elif isinstance(t, dict) and "name" in t:
            fn = t
        else:
            continue
        params = fn.get("parameters") or {}
        # Unwrap JSON Schema object wrapper → flat param dict
        if isinstance(params, dict) and "properties" in params:
            params = params["properties"]
        flat.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": params,
        })
    return flat


def _extract_toucan_calls(messages_json, tool_names=None):
    """Extract tool calls from Toucan messages.

    Collects all function_call entries from assistant messages.
    If tool_names is provided, only keeps calls whose name matches an available tool.
    """
    try:
        messages = json.loads(messages_json) if isinstance(messages_json, str) else messages_json
    except (json.JSONDecodeError, TypeError):
        return []
    calls = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        fc = msg.get("function_call")
        if not fc or not isinstance(fc, dict):
            continue
        name = fc.get("name", "")
        if not name:
            continue
        if tool_names is not None and name not in tool_names:
            continue
        args = fc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError, ValueError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": name, "arguments": args})
    return calls


def convert_toucan(dataset):
    rows = []
    for ex in dataset:
        query = (ex.get("question") or "").strip()
        if not query:
            continue
        tools = _convert_openai_tools(ex.get("available_tools") or ex.get("tools") or "[]")
        if not tools:
            continue
        tool_names = {t["name"] for t in tools}
        calls = _extract_toucan_calls(ex.get("messages", "[]"), tool_names)
        if not calls:
            continue
        rows.append({
            "query": query,
            "answers": json.dumps(calls),
            "tools": json.dumps(tools),
        })
    return Dataset.from_list(rows)


def _extract_toolace_tools(system_text):
    """Extract JSON tool definitions array from ToolACE system prompt."""
    start = system_text.find('[{')
    if start == -1:
        return []
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


def _split_bracket_args(s, sep=','):
    """Split on separator respecting nested brackets and quoted strings."""
    parts = []
    depth = 0
    in_str = False
    str_char = None
    current = []
    for i, ch in enumerate(s):
        if in_str:
            current.append(ch)
            if ch == str_char and (i == 0 or s[i - 1] != '\\'):
                in_str = False
        elif ch in ('"', "'"):
            in_str = True
            str_char = ch
            current.append(ch)
        elif ch in ('(', '[', '{'):
            depth += 1
            current.append(ch)
        elif ch in (')', ']', '}'):
            depth -= 1
            current.append(ch)
        elif ch == sep and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts


def _parse_bracket_value(s):
    """Parse a value string from bracket notation arguments."""
    s = s.strip()
    if not s:
        return s
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
    if s == 'True':
        return True
    if s == 'False':
        return False
    if s in ('None', 'null'):
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    return s


def _parse_toolace_calls(text, tool_names):
    """Parse ToolACE bracket notation into tool calls using known names."""
    text = text.strip()
    if not text.startswith('[') or not text.endswith(']'):
        return []
    inner = text[1:-1].strip()
    if not inner:
        return []

    sorted_names = sorted(tool_names, key=len, reverse=True)
    calls = []
    pos = 0

    while pos < len(inner):
        while pos < len(inner) and inner[pos] in (' ', ',', '\n', '\t'):
            pos += 1
        if pos >= len(inner):
            break

        matched = None
        for name in sorted_names:
            if inner[pos:pos + len(name)] == name:
                rest = inner[pos + len(name):].lstrip()
                if rest and rest[0] == '(':
                    matched = name
                    break

        if not matched:
            break

        pos += len(matched)
        while pos < len(inner) and inner[pos] == ' ':
            pos += 1
        if pos >= len(inner) or inner[pos] != '(':
            break

        depth = 0
        in_str = False
        str_char = None
        end = pos
        while end < len(inner):
            ch = inner[end]
            if in_str:
                if ch == str_char and (end == 0 or inner[end - 1] != '\\'):
                    in_str = False
            elif ch in ('"', "'"):
                in_str = True
                str_char = ch
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    break
            end += 1

        args_str = inner[pos + 1:end].strip()
        arguments = {}
        if args_str:
            for pair in _split_bracket_args(args_str, ','):
                pair = pair.strip()
                eq = pair.find('=')
                if eq < 0:
                    continue
                key = pair[:eq].strip()
                val = _parse_bracket_value(pair[eq + 1:])
                arguments[key] = val

        calls.append({"name": matched, "arguments": arguments})
        pos = end + 1

    return calls


def convert_toolace(dataset):
    rows = []
    for ex in dataset:
        system = ex.get("system", "")
        convs = ex.get("conversations", [])

        raw_tools = _extract_toolace_tools(system)
        tools = _convert_openai_tools(raw_tools)
        if not tools:
            continue

        tool_names = {t["name"] for t in tools}

        query = None
        assistant_text = None
        for msg in convs:
            role = msg.get("from", "")
            val = (msg.get("value") or "").strip()
            if role == "user" and query is None:
                query = val
            elif role == "assistant" and assistant_text is None:
                assistant_text = val

        if not query:
            continue

        calls = []
        if assistant_text:
            calls = _parse_toolace_calls(assistant_text, tool_names)

        rows.append({
            "query": query,
            "answers": json.dumps(calls),
            "tools": json.dumps(tools),
        })
    return Dataset.from_list(rows)


def _extract_nemotron_calls(messages, tool_names=None):
    """Extract tool calls from Nemotron messages (OpenAI tool_calls format)."""
    calls = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue

        # OpenAI tool_calls format (newer)
        for tc in (msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", tc)
            name = fn.get("name", "")
            if not name:
                continue
            if tool_names and name not in tool_names:
                continue
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append({"name": name, "arguments": args})

        # Fallback: function_call format (older OpenAI)
        fc = msg.get("function_call")
        if fc and isinstance(fc, dict):
            name = fc.get("name", "")
            if name and (not tool_names or name in tool_names):
                args = fc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                calls.append({"name": name, "arguments": args})

    return calls


def convert_nemotron(dataset):
    rows = []
    for ex in dataset:
        tools_raw = ex.get("tools") or []
        tools = _convert_openai_tools(tools_raw)
        if not tools:
            continue

        tool_names = {t["name"] for t in tools}
        messages = ex.get("messages") or []

        query = None
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                query = (msg.get("content") or "").strip()
                break

        if not query:
            continue

        calls = _extract_nemotron_calls(messages, tool_names)
        if not calls:
            continue

        rows.append({
            "query": query,
            "answers": json.dumps(calls),
            "tools": json.dumps(tools),
        })
    return Dataset.from_list(rows)


CONVERTERS = {
    "xlam-function-calling-60k": convert_xlam,
    "glaive-function-calling": convert_glaive,
    "toucan-oss": convert_toucan,
    "toucan-kimi": convert_toucan,
    "toucan-qwen": convert_toucan,
    "toucan-sft": convert_toucan,
    "toolace": convert_toolace,
    "nemotron-tc": convert_nemotron,
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



def _answer_calls_valid_tools(ex):
    """Check that every tool called in the answer exists in the tools list."""
    try:
        answers = json.loads(ex["answers"])
        tools = json.loads(ex["tools"])
    except (json.JSONDecodeError, TypeError):
        return True
    if not answers:
        return True
    tool_names = {t["name"] for t in tools}
    return all(a["name"] in tool_names for a in answers)


def _answer_uses_valid_params(ex):
    """Check that arguments in each call are a subset of the tool's schema params."""
    try:
        answers = json.loads(ex["answers"])
        tools = json.loads(ex["tools"])
    except (json.JSONDecodeError, TypeError):
        return True
    if not answers:
        return True
    tool_map = {t["name"]: set((t.get("parameters") or {}).keys()) for t in tools}
    for a in answers:
        schema_params = tool_map.get(a["name"])
        if schema_params is None:
            continue  # caught by _answer_calls_valid_tools
        args = a.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(args, dict):
            continue
        if args and not set(args.keys()).issubset(schema_params):
            return False
    return True



def _deduplicate(dataset):
    """Remove duplicates by (query, tools) pair, keeping the first occurrence.

    Same query with different tool sets is kept — those are distinct examples.
    Same query with same tools but different answers (e.g. from different LLMs)
    is deduplicated since the model only needs one answer per (query, tools) pair.
    """
    seen = set()
    keep = []
    for i in range(len(dataset)):
        key = (dataset[i]["query"], dataset[i]["tools"])
        if key not in seen:
            seen.add(key)
            keep.append(i)
    if len(keep) == len(dataset):
        return dataset
    return dataset.select(keep)


def _augment_irrelevance(dataset, ratio=0.1, seed=42):
    """Create irrelevance samples by swapping tools with unrelated queries.

    For a fraction of examples with tool calls, replace the available tools
    with tools from a different example and set the answer to [] (no calls).
    Ensures no overlap between the query's needed tools and the donor tools.
    """
    rng = random.Random(seed)

    all_answers = dataset["answers"]
    all_tools = dataset["tools"]
    all_queries = dataset["query"]

    with_calls = []
    call_names = {}
    tool_names_by_idx = {}

    for i, ans_str in enumerate(all_answers):
        try:
            answers = json.loads(ans_str)
        except (json.JSONDecodeError, TypeError):
            continue
        if not answers:
            continue
        with_calls.append(i)
        call_names[i] = {c["name"] for c in answers if isinstance(c, dict)}
        try:
            tools = json.loads(all_tools[i])
            tool_names_by_idx[i] = {t["name"] for t in tools if isinstance(t, dict)}
        except (json.JSONDecodeError, TypeError):
            tool_names_by_idx[i] = set()

    n_aug = int(len(with_calls) * ratio)
    selected = rng.sample(with_calls, min(n_aug, len(with_calls)))

    rows = []
    for idx in selected:
        orig_call_names = call_names[idx]
        for _ in range(5):
            donor = rng.choice(with_calls)
            if donor == idx:
                continue
            # Ensure none of the query's needed tools appear in donor's tool set
            if not (orig_call_names & tool_names_by_idx.get(donor, set())):
                rows.append({
                    "query": all_queries[idx],
                    "answers": "[]",
                    "tools": all_tools[donor],
                    "source": "irrelevance-aug",
                })
                break

    if rows:
        print(f"\nIrrelevance augmentation: {len(rows)} samples "
              f"({len(rows)/len(dataset)*100:.1f}% of dataset)")
        return concatenate_datasets([dataset, Dataset.from_list(rows)])
    return dataset


def load_and_combine():
    parts = []
    for name, hf_id in DATASETS.items():
        split_override = None
        if isinstance(hf_id, tuple):
            if len(hf_id) == 3:
                repo, subset, split_override = hf_id
            else:
                repo, subset = hf_id
            label = f"{repo}" + (f" [{subset}]" if subset else "")
        else:
            repo, subset = hf_id, None
            label = repo
        print(f"\nLoading {name} ({label})...")
        try:
            if split_override:
                raw = load_dataset(repo, subset, split=split_override) if subset else load_dataset(repo, split=split_override)
            else:
                ds = load_dataset(repo, subset) if subset else load_dataset(repo)
                split = "train" if "train" in ds else list(ds.keys())[0]
                raw = ds[split]
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Converting to xlam format ({len(raw):,} rows)...")
        converted = CONVERTERS[name](raw)
        converted = converted.add_column("source", [name] * len(converted))

        before = len(converted)
        converted = converted.filter(
            lambda ex: (
                not _has_placeholder_args(ex["answers"])
                and _answer_calls_valid_tools(ex)
                and _answer_uses_valid_params(ex)
            ),
            desc=f"Filtering {name}",
        )
        dropped = before - len(converted)
        if dropped:
            print(f"  Filtered out {dropped} low-quality examples ({dropped*100/before:.1f}%)")

        print(f"  -> {len(converted)} examples")
        parts.append(converted)

    combined = concatenate_datasets(parts)

    # Drop no-call examples with empty tools (trivial negatives)
    before_empty = len(combined)
    combined = combined.filter(
        lambda ex: json.loads(ex["answers"]) != [] or json.loads(ex["tools"]) != [],
        desc="Removing empty-tools no-call examples",
    )
    empty_dropped = before_empty - len(combined)
    if empty_dropped:
        print(f"\nRemoved {empty_dropped} no-call examples with empty tools")

    before_dedup = len(combined)
    combined = _deduplicate(combined)
    dedup_dropped = before_dedup - len(combined)
    if dedup_dropped:
        print(f"\nDeduplicated: removed {dedup_dropped} duplicate queries")

    combined = _augment_irrelevance(combined, ratio=0.1, seed=42)

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
