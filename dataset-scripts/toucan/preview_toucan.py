import ast
import json
import uuid
from datasets import load_dataset
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

NUM_EXAMPLES = 1000
OUT_FILE = "toucan_preview.jsonl"


def strip_titles(obj):
    if isinstance(obj, dict):
        return {k: strip_titles(v) for k, v in obj.items() if k != "title"}
    if isinstance(obj, list):
        return [strip_titles(v) for v in obj]
    return obj


def build_tool_subset(available_tools, target_names):
    by_name = {t["function"]["name"]: t for t in available_tools if t.get("function", {}).get("name")}
    subset = [by_name[n] for n in target_names if n in by_name]
    for tool in available_tools:
        if len(subset) >= 5:
            break
        if tool.get("function", {}).get("name") not in target_names:
            subset.append(tool)
    return [strip_titles(t) for t in subset[:5]]


def parse_tool_call(m):
    try:
        fc = ast.literal_eval(m["content"])
    except Exception:
        return None
    try:
        args = json.loads(fc.get("arguments", "{}"))
    except Exception:
        args = fc.get("arguments", {})
    return {
        "id": f"call_1",
        "type": "function",
        "function": {
            "name": fc.get("name", ""),
            "arguments": args,
        },
    }


def convert(row):
    target_names = [n.strip() for n in row["target_tools"].split(",")]
    if len(target_names) > 3:
        return []

    try:
        if detect(row["question"]) != "en":
            return []
    except Exception:
        return []

    available_tools = json.loads(row["tools"])
    tool_subset = build_tool_subset(available_tools, target_names)

    messages = json.loads(row["messages"])
    query = next((m["content"] for m in messages if m.get("role") == "user"), None)
    if query is None:
        return []

    records = []
    current_calls = []
    for m in messages:
        role = m.get("role")
        if role == "tool_call":
            tc = parse_tool_call(m)
            if tc:
                tc["id"] = f"call_{len(current_calls) + 1}"
                current_calls.append(tc)
        elif role in ("tool_response", "assistant") and current_calls:
            records.append({
                "id": str(uuid.uuid4()),
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "tool_calls": current_calls},
                ],
                "tools": tool_subset,
            })
            current_calls = []

    return records


ds = load_dataset("Agent-Ark/Toucan-1.5M", "SFT", split="train", streaming=True)

count = 0
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for row in ds:
        if count >= NUM_EXAMPLES:
            break
        for record in convert(row):
            if count >= NUM_EXAMPLES:
                break
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"--- Example {count} ---")
            print(json.dumps(record, indent=2))
            print()
            count += 1

print(f"Saved {count} examples to {OUT_FILE}")
