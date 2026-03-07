import os
import math
import json
import subprocess
from datasets import load_dataset

BUCKET_PATH = "gs://needle-datasets-bucket/datasets/xlam/openai_jsonl_v1"
OUT_DIR = "xlam_openai_jsonl/train"
SHARD_SIZE = 2000


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


def convert_and_shard(ds):
    os.makedirs(OUT_DIR, exist_ok=True)
    num_shards = math.ceil(len(ds) / SHARD_SIZE)

    for s in range(num_shards):
        start = s * SHARD_SIZE
        end = min((s + 1) * SHARD_SIZE, len(ds))
        path = os.path.join(OUT_DIR, f"part-{s:05d}.jsonl")

        with open(path, "w", encoding="utf-8") as f:
            for i in range(start, end):
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

        print(f"  wrote shard {s + 1}/{num_shards}: {path}")

    return num_shards


def write_manifest(num_examples, num_shards):
    import datetime
    manifest = {
        "source": "Salesforce/xlam-function-calling-60k",
        "format": "openai_tool_calls",
        "created": datetime.datetime.utcnow().isoformat() + "Z",
        "shard_size": SHARD_SIZE,
        "num_shards": num_shards,
        "num_examples": num_examples,
        "bucket_path": BUCKET_PATH,
        "tool_registry": "inline_per_record",
    }
    path = "xlam_openai_jsonl/MANIFEST.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {path}")


def upload():
    print("Uploading to GCS...")
    subprocess.run(
        ["gsutil.cmd", "-m", "cp", "-r", "xlam_openai_jsonl", BUCKET_PATH],
        check=True,
    )
    print(f"Done. Files at {BUCKET_PATH}/xlam_openai_jsonl/")


if __name__ == "__main__":
    print("Downloading dataset...")
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    print(f"  {len(ds)} examples")

    print("Converting and sharding...")
    num_shards = convert_and_shard(ds)

    print("Writing manifest...")
    write_manifest(len(ds), num_shards)

    upload()
