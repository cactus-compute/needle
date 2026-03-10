import ast
import datetime
import json
import math
import os
import subprocess
import uuid
from datasets import load_dataset
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

BUCKET_PATH = "gs://needle-datasets-bucket/datasets/toucan/openai_jsonl_v1"
OUT_DIR = "toucan_openai_jsonl/train"
SHARD_SIZE = 2000


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
        "id": "call_1",
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


def convert_and_shard(ds):
    os.makedirs(OUT_DIR, exist_ok=True)
    records = []
    skipped = 0
    for row in ds:
        converted = convert(row)
        if not converted:
            skipped += 1
            continue
        records.extend(converted)

    print(f"  {len(records)} records from {len(ds) - skipped} source rows, {skipped} skipped")

    num_shards = math.ceil(len(records) / SHARD_SIZE)
    for s in range(num_shards):
        start = s * SHARD_SIZE
        end = min((s + 1) * SHARD_SIZE, len(records))
        path = os.path.join(OUT_DIR, f"part-{s:05d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for record in records[start:end]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  wrote shard {s + 1}/{num_shards}: {path}")

    return len(records), num_shards


def write_manifest(num_examples, num_shards):
    manifest = {
        "source": "Agent-Ark/Toucan-1.5M",
        "format": "openai_tool_calls",
        "created": datetime.datetime.utcnow().isoformat() + "Z",
        "shard_size": SHARD_SIZE,
        "num_shards": num_shards,
        "num_examples": num_examples,
        "bucket_path": BUCKET_PATH,
        "tool_registry": "inline_per_record",
    }
    path = "toucan_openai_jsonl/MANIFEST.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {path}")


def upload():
    print("Uploading to GCS...")
    subprocess.run(
        ["gsutil.cmd", "-m", "cp", "-r", "toucan_openai_jsonl", BUCKET_PATH],
        check=True,
    )
    print(f"Done. Files at {BUCKET_PATH}/toucan_openai_jsonl/")


if __name__ == "__main__":
    print("Downloading dataset...")
    ds = load_dataset("Agent-Ark/Toucan-1.5M", "SFT", split="train")
    print(f"  {len(ds)} examples")

    print("Converting and sharding...")
    num_examples, num_shards = convert_and_shard(ds)

    print("Writing manifest...")
    write_manifest(num_examples, num_shards)

    upload()
