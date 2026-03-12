#!/usr/bin/env python3
"""Synthesize tool-calling data from PleIAs/SYNTH queries via OpenRouter.

Downloads the first parquet shard of PleIAs/SYNTH (~1M diverse multilingual
queries), then prompts an LLM to generate (tools, answer) pairs for each query.

Usage:
    python scripts/synthesize_tools_data.py
    python scripts/synthesize_tools_data.py --max-samples 10000 --model google/gemini-2.5-flash
    python scripts/synthesize_tools_data.py --resume  # continue from last checkpoint
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "synth_tool_calls"
_CHECKPOINT_PATH = _OUTPUT_DIR / "checkpoint.jsonl"
_PARQUET_CACHE = _PROJECT_ROOT / ".data_cache" / "synth_shard0.parquet"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"

SYSTEM_PROMPT = """\
You are a synthetic data generator for training a tool-calling AI model.

Given a user query, you must invent a realistic set of tools (APIs/functions) that \
could help answer it, then produce the correct tool call(s).

Rules:
1. Invent 3-8 tools as JSON. Each tool has: name, description, parameters (dict of param_name -> {type, description}). \
Only ONE or TWO tools should be relevant; the rest are distractors.
2. Tool names should be snake_case, realistic API-style (e.g. search_web, get_weather, translate_text).
3. Produce the answer as a JSON list of tool calls: [{"name": "...", "arguments": {...}}].
4. If the query genuinely needs no tool, return an empty list [].
5. Argument values MUST be grounded in the query — do not hallucinate values.
6. Respond in the SAME LANGUAGE as the query.
7. Return ONLY valid JSON, no markdown fences, no commentary.

Output format (strict JSON object):
{"tools": [...], "answer": [...]}"""

USER_TEMPLATE = "Query: {query}"


def download_first_shard(token):
    """Download the first parquet shard of PleIAs/SYNTH."""
    if _PARQUET_CACHE.exists():
        logger.info(f"Using cached shard: {_PARQUET_CACHE}")
        return _PARQUET_CACHE

    _PARQUET_CACHE.parent.mkdir(parents=True, exist_ok=True)

    url = (
        "https://huggingface.co/datasets/PleIAs/SYNTH"
        "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    )
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    logger.info(f"Downloading first shard from PleIAs/SYNTH...")
    resp = requests.get(url, headers=headers, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(_PARQUET_CACHE, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%)", end="", flush=True)
    print()
    logger.info(f"Saved to {_PARQUET_CACHE}")
    return _PARQUET_CACHE


def load_queries(path, max_samples=None):
    """Load queries from parquet, return list of (synth_id, language, query)."""
    import pyarrow.parquet as pq

    logger.info(f"Reading parquet: {path}")
    table = pq.read_table(path, columns=["synth_id", "language", "query"])
    n = table.num_rows
    logger.info(f"  {n:,} rows in shard")

    ids = table.column("synth_id").to_pylist()
    langs = table.column("language").to_pylist()
    queries = table.column("query").to_pylist()

    samples = list(zip(ids, langs, queries))

    # Shuffle for diversity (don't process in order)
    rng = random.Random(42)
    rng.shuffle(samples)

    if max_samples:
        samples = samples[:max_samples]

    logger.info(f"  Selected {len(samples):,} samples")
    return samples


def load_checkpoint():
    """Load completed synth_ids from checkpoint file."""
    done = set()
    if _CHECKPOINT_PATH.exists():
        with open(_CHECKPOINT_PATH) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done.add(row["synth_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(f"Resuming: {len(done):,} already completed")
    return done


def call_openrouter(query, api_key, model=DEFAULT_MODEL, max_retries=3):
    """Call OpenRouter API to generate tools + answer for a query."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(query=query)},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=60
            )
            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, 30)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return _parse_response(content)
        except (requests.RequestException, KeyError, IndexError) as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def _parse_response(content):
    """Parse LLM response into (tools, answer) or None."""
    content = content.strip()
    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None

    tools = obj.get("tools")
    answer = obj.get("answer")

    if not isinstance(tools, list):
        return None
    if not isinstance(answer, list):
        return None

    # Validate tool structure
    for t in tools:
        if not isinstance(t, dict) or "name" not in t:
            return None

    # Validate answer structure
    for a in answer:
        if not isinstance(a, dict) or "name" not in a:
            return None
        if "arguments" not in a:
            a["arguments"] = {}

    return tools, answer


def _validate_and_format(synth_id, language, query, tools, answer):
    """Validate the generated data and return a formatted row or None."""
    # Check that every called tool exists in the tool list
    tool_names = {t["name"] for t in tools}
    for a in answer:
        if a["name"] not in tool_names:
            return None

    # Check argument keys are valid
    tool_params = {}
    for t in tools:
        params = t.get("parameters") or {}
        tool_params[t["name"]] = set(params.keys())

    for a in answer:
        args = a.get("arguments", {})
        if not isinstance(args, dict):
            return None
        schema_params = tool_params.get(a["name"], set())
        if args and not set(args.keys()).issubset(schema_params):
            return None

    return {
        "synth_id": synth_id,
        "language": language,
        "query": query,
        "answers": json.dumps(answer),
        "tools": json.dumps(tools),
        "source": "synth-pleias",
    }


def process_sample(sample, api_key, model):
    """Process a single sample: call LLM, validate, return row or None."""
    synth_id, language, query = sample
    result = call_openrouter(query, api_key, model=model)
    if result is None:
        return None
    tools, answer = result
    return _validate_and_format(synth_id, language, query, tools, answer)


def main():
    parser = argparse.ArgumentParser(description="Synthesize tool-calling data from PleIAs/SYNTH")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max queries to process (default: all in shard)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"OpenRouter model (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=32,
                        help="Concurrent API requests (default: 32)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Samples between progress logs (default: 500)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    parquet_path = download_first_shard(hf_token)
    samples = load_queries(parquet_path, max_samples=args.max_samples)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    done_ids = load_checkpoint() if args.resume else set()
    if not args.resume and _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()

    pending = [s for s in samples if s[0] not in done_ids]
    logger.info(f"Processing {len(pending):,} samples ({len(done_ids):,} already done)")

    generated = 0
    failed = 0
    out_file = open(_CHECKPOINT_PATH, "a")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for sample in pending:
                fut = executor.submit(process_sample, sample, api_key, args.model)
                futures[fut] = sample[0]  # synth_id

            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    row = fut.result()
                except Exception:
                    failed += 1
                    row = None

                if row is not None:
                    out_file.write(json.dumps(row) + "\n")
                    generated += 1
                else:
                    failed += 1

                if i % args.batch_size == 0:
                    out_file.flush()
                    total = generated + failed
                    rate = generated / max(total, 1) * 100
                    logger.info(
                        f"  [{total:,}/{len(pending):,}] "
                        f"generated={generated:,} failed={failed:,} "
                        f"({rate:.0f}% success)"
                    )
    except KeyboardInterrupt:
        logger.info("\nInterrupted — progress saved to checkpoint")
    finally:
        out_file.flush()
        out_file.close()

    logger.info(f"\nDone: {generated:,} generated, {failed:,} failed")
    logger.info(f"Output: {_CHECKPOINT_PATH}")

    # Convert checkpoint to HF dataset format
    if generated > 0:
        _export_dataset()


def _export_dataset():
    """Convert checkpoint JSONL to the unified format expected by build_dataset.py."""
    from datasets import Dataset

    rows = []
    with open(_CHECKPOINT_PATH) as f:
        for line in f:
            try:
                row = json.loads(line)
                rows.append({
                    "query": row["query"],
                    "answers": row["answers"],
                    "tools": row["tools"],
                    "source": row.get("source", "synth-pleias"),
                })
            except (json.JSONDecodeError, KeyError):
                continue

    if not rows:
        return

    ds = Dataset.from_list(rows)
    out_path = _OUTPUT_DIR / "dataset"
    ds.save_to_disk(str(out_path))
    logger.info(f"Exported {len(rows):,} examples to {out_path}/")
    logger.info(
        "To include in training, add to tools_data.py or merge with unified dataset."
    )


if __name__ == "__main__":
    main()
