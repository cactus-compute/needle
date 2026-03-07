import json
from datasets import load_dataset

NUM_EXAMPLES = 1000
OUT_FILE = "xlam_preview.jsonl"

TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def normalize_type(raw):
    base = raw.split(",")[0].strip().lower()
    if base.startswith(("list[", "tuple[", "set[", "list", "tuple", "set")):
        return "array"
    if base.startswith("callable"):
        return "string"
    if base.startswith(("dict[", "union[")):
        return "object"
    return TYPE_MAP.get(base, base)


def to_openai_tool(tool):
    props = {}
    required = []
    for pname, pinfo in tool.get("parameters", {}).items():
        props[pname] = {
            "type": normalize_type(pinfo.get("type", "string")),
            "description": pinfo.get("description", ""),
        }
        if pinfo.get("required", False):
            required.append(pname)

    schema = {"type": "object", "properties": props}
    if required:
        schema["required"] = required

    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": schema,
        },
    }


def to_openai_tool_call(ans, idx):
    return {
        "id": f"call_{idx}",
        "type": "function",
        "function": {
            "name": ans.get("name", ""),
            "arguments": ans.get("arguments", {}),
        },
    }


def maybe_parse(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return val
    return val


ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for i in range(NUM_EXAMPLES):
        ex = ds[i]
        query = maybe_parse(ex.get("query", ""))
        tools = maybe_parse(ex.get("tools", []))
        answers = maybe_parse(ex.get("answers", []))

        record = {
            "id": ex.get("id", f"ex_{i}"),
            "messages": [
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "tool_calls": [
                        to_openai_tool_call(a, j + 1)
                        for j, a in enumerate(answers)
                    ],
                },
            ],
            "tools": [to_openai_tool(t) for t in tools],
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"--- Example {i} ---")
        print(json.dumps(record, indent=2))
        print()

print(f"Saved {NUM_EXAMPLES} examples to {OUT_FILE}")
