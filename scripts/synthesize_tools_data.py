#!/usr/bin/env python3
"""Synthesize tool-calling data from PleIAs/SYNTH via OpenRouter.

Each query is sent to a randomly-selected frontier model to generate
(tools, answer) pairs, then validated and checkpointed as JSONL.

Usage:
    python scripts/synthesize_tools_data.py
    python scripts/synthesize_tools_data.py --model google/gemini-3.1-flash-lite-preview
    python scripts/synthesize_tools_data.py --max-shards 1 --max-samples 100
    python scripts/synthesize_tools_data.py --resume               # continue from checkpoint
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
_PARQUET_CACHE_DIR = _PROJECT_ROOT / ".data_cache"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_POOL = [
    "google/gemini-3.1-flash-lite-preview",
    # "anthropic/claude-opus-4.6",
    # "x-ai/grok-4.1-fast",
    # "openai/gpt-5.3-chat",
]
DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"

SYSTEM_PROMPT = """\
You are a synthetic data generator for training a tool-calling AI model.

Given a user query, you must invent a realistic set of tools (APIs/functions) that \
could help answer it, then produce the correct tool call(s).

Vary the domain widely — cover areas like software development, data analysis, \
e-commerce, finance, healthcare, travel, IoT, social media, education, DevOps, legal, \
HR, and more. Do NOT default to simple personal-assistant tasks.

Rules:
1. Invent 1-5 tools as JSON. Each tool MUST have: name, description, parameters \
(dict of param_name -> {type, description, required}). Use realistic parameter types \
including string, number, boolean, array, object. Include nested objects or arrays \
where appropriate.
2. Only ONE or TWO tools should be relevant to the query; the rest are plausible \
distractors from the SAME domain.
3. Tool names should be snake_case, realistic API-style, and DIVERSE — avoid repeating \
common names like search_web or get_weather across samples. Think of specific, \
domain-appropriate APIs (e.g. query_elasticsearch, run_sql_migration, price_option_chain, \
schedule_k8s_cronjob, fetch_patient_labs).
4. Produce the answer as a JSON list of tool calls: [{"name": "...", "arguments": {...}}].
5. For complex queries, use MULTIPLE tool calls (chained or parallel) where realistic.
6. If the query genuinely needs no tool, return an empty answer list [].
7. Argument values MUST be grounded in the query — do not hallucinate values.
8. Respond in the SAME LANGUAGE as the query.
9. Return ONLY valid JSON, no markdown fences, no commentary.

Output format (strict JSON object):
{"tools": [...], "answer": [...]}"""

USER_TEMPLATE = "Query: {query}"


def _discover_shard_urls(token, max_shards=None):
    """Discover available parquet shard URLs from the HF API."""
    api_url = "https://huggingface.co/api/datasets/PleIAs/SYNTH/parquet"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(api_url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()  # {"default": {"train": [url, ...]}}
    urls = data["default"]["train"]
    if max_shards:
        urls = urls[:max_shards]
    logger.info(f"Discovered {len(urls)} parquet shards")
    return urls


def download_shard(url, shard_idx, token):
    """Download a single parquet shard, returning the local path."""
    cache_path = _PARQUET_CACHE_DIR / f"synth_shard{shard_idx}.parquet"
    if cache_path.exists():
        logger.info(f"Using cached shard {shard_idx}: {cache_path}")
        return cache_path

    _PARQUET_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    logger.info(f"Downloading shard {shard_idx}...")
    resp = requests.get(url, headers=headers, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(cache_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%)", end="", flush=True)
    print()
    logger.info(f"Saved shard {shard_idx} to {cache_path}")
    return cache_path


def load_queries_from_parquet(path, max_samples=None):
    """Load queries from parquet, return list of (sample_id, language, query)."""
    import pyarrow.parquet as pq

    logger.info(f"Reading parquet: {path}")
    table = pq.read_table(path, columns=["synth_id", "language", "query"])
    n = table.num_rows
    logger.info(f"  {n:,} rows in shard")

    ids = table.column("synth_id").to_pylist()
    langs = table.column("language").to_pylist()
    queries = table.column("query").to_pylist()

    samples = list(zip(ids, langs, queries))

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
        "max_tokens": 4096,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=60
            )
            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, 30)
                logger.warning(f"[{model}] 429 rate-limited, retrying in {wait}s")
                time.sleep(wait)
                continue
            if not resp.ok:
                logger.warning(f"[{model}] HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
                continue
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = _parse_response(content)
            if parsed is None:
                logger.warning(f"[{model}] Failed to parse response: {content[:200]}")
            return parsed
        except (requests.RequestException, KeyError, IndexError) as e:
            logger.warning(f"[{model}] Request error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def _parse_response(content):
    """Parse LLM response into (tools, answer) or None."""
    content = content.strip()
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

    for t in tools:
        if not isinstance(t, dict) or "name" not in t:
            return None

    for a in answer:
        if not isinstance(a, dict) or "name" not in a:
            return None
        if "arguments" not in a:
            a["arguments"] = {}

    return tools, answer


def _validate_and_format(synth_id, language, query, tools, answer, model, source="synth-pleias"):
    """Validate the generated data and return a formatted row or None."""
    tool_names = {t["name"] for t in tools}
    for a in answer:
        if a["name"] not in tool_names:
            return None

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
        "source": source,
        "model": model,
    }


def _pick_model(models, rng):
    """Randomly select a model from the pool."""
    return rng.choice(models)


def process_sample(sample, api_key, models, rng, source_tag="synth-pleias"):
    """Process a single sample: call LLM, validate, return row or None."""
    synth_id, language, query = sample
    model = _pick_model(models, rng)
    result = call_openrouter(query, api_key, model=model)
    if result is None:
        return None
    tools, answer = result
    row = _validate_and_format(synth_id, language, query, tools, answer, model, source_tag)
    if row is None:
        logger.warning(f"[{model}] Validation failed for synth_id={synth_id}")
    return row


def _process_batch(samples, label, batch_idx, api_key, models, args, tqdm, source_tag,
                    since_last_upload):
    """Process a batch of samples, append to checkpoint. Returns (generated, failed)."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Batch: {label}")
    logger.info(f"{'=' * 60}")

    done_ids = load_checkpoint()
    pending = [s for s in samples if s[0] not in done_ids]
    if not pending:
        logger.info("All samples already done, skipping")
        return 0, 0
    logger.info(f"Processing {len(pending):,} samples ({len(done_ids):,} already done)")

    generated = 0
    failed = 0
    out_file = open(_CHECKPOINT_PATH, "a")
    rng = random.Random(42 + batch_idx)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for sample in pending:
                fut = executor.submit(
                    process_sample, sample, api_key, models, rng, source_tag,
                )
                futures[fut] = sample[0]

            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=label,
                unit="sample",
                dynamic_ncols=True,
            ) if tqdm else as_completed(futures)

            for i, fut in enumerate(pbar if tqdm else as_completed(futures), 1):
                try:
                    row = fut.result()
                except Exception:
                    failed += 1
                    row = None

                if row is not None:
                    out_file.write(json.dumps(row) + "\n")
                    generated += 1
                    since_last_upload[0] += 1
                else:
                    failed += 1

                if tqdm:
                    rate = generated / max(generated + failed, 1) * 100
                    pbar.set_postfix(ok=generated, fail=failed, rate=f"{rate:.0f}%")
                    if i % args.batch_size == 0:
                        out_file.flush()
                else:
                    if i % args.batch_size == 0:
                        out_file.flush()
                        total = generated + failed
                        rate = generated / max(total, 1) * 100
                        logger.info(
                            f"  [{total:,}/{len(pending):,}] "
                            f"generated={generated:,} failed={failed:,} "
                            f"({rate:.0f}% success)"
                        )

                if since_last_upload[0] >= args.upload_every:
                    out_file.flush()
                    _export_dataset()
                    since_last_upload[0] = 0

            if tqdm:
                pbar.close()
    except KeyboardInterrupt:
        logger.info("\nInterrupted — progress saved to checkpoint")
    finally:
        out_file.flush()
        out_file.close()

    logger.info(f"{label}: {generated:,} generated, {failed:,} failed")
    return generated, failed


def main():
    parser = argparse.ArgumentParser(description="Synthesize tool-calling data from PleIAs/SYNTH")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max queries to process (default: all in shard)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help=f"OpenRouter models to rotate (default: built-in pool)")
    parser.add_argument("--model", type=str, default=None,
                        help="Use a single model instead of the pool")
    parser.add_argument("--workers", type=int, default=32,
                        help="Concurrent API requests (default: 32)")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Max parquet shards to process (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Samples between progress logs (default: 500)")
    parser.add_argument("--upload-every", type=int, default=1000,
                        help="Upload to GCS every N successful generations (default: 1000)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    if args.model:
        models = [args.model]
    elif args.models:
        models = args.models
    else:
        models = list(MODEL_POOL)
    logger.info(f"Model pool ({len(models)}): {', '.join(models)}")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.resume and _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()
        logger.info("Cleared previous checkpoint (use --resume to keep)")

    try:
        from tqdm import tqdm
    except ImportError:
        logger.warning("tqdm not installed, falling back to basic progress logging")
        tqdm = None

    total_generated = 0
    total_failed = 0
    batch_idx = 0
    since_last_upload = [0]

    hf_token = os.environ.get("HF_TOKEN")
    shard_urls = _discover_shard_urls(hf_token, max_shards=args.max_shards)

    for shard_idx, shard_url in enumerate(shard_urls):
        parquet_path = download_shard(shard_url, shard_idx, hf_token)
        samples = load_queries_from_parquet(parquet_path, max_samples=args.max_samples)
        gen, fail = _process_batch(
            samples, f"PleIAs shard {shard_idx}", batch_idx, api_key, models,
            args, tqdm, source_tag="synth-pleias",
            since_last_upload=since_last_upload,
        )
        total_generated += gen
        total_failed += fail
        batch_idx += 1

    if since_last_upload[0] > 0:
        _export_dataset()

    logger.info(f"\nAll done: {total_generated:,} generated, {total_failed:,} failed")


def _export_dataset():
    """Convert checkpoint JSONL to the unified format and push to GCS."""
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
                    "model": row.get("model", "unknown"),
                })
            except (json.JSONDecodeError, KeyError):
                continue

    if not rows:
        return

    ds = Dataset.from_list(rows)
    out_path = _OUTPUT_DIR / "dataset"
    ds.save_to_disk(str(out_path))
    logger.info(f"Exported {len(rows):,} examples to {out_path}/")

    try:
        sys.path.insert(0, str(_PROJECT_ROOT))
        from src.gcs import upload_directory
        gcs_prefix = "synth_tool_calls"
        upload_directory(str(out_path), gcs_prefix)
    except Exception as e:
        logger.warning(f"GCS upload failed: {e}")

    for i, row in enumerate(rows[:10]):
        try:
            tools = json.loads(row["tools"])
            answers = json.loads(row["answers"])
        except (json.JSONDecodeError, TypeError):
            tools, answers = [], []
        called_names = [a["name"] for a in answers if isinstance(a, dict)]
        print(f"\n{'─' * 60}")
        print(f"[{i+1}] Query:   {row['query'][:200]}")
        print(f"    Available tools ({len(tools)}): {', '.join(t.get('name','?') for t in tools)}")
        print(f"    Called tools ({len(called_names)}):   {', '.join(called_names) or '(none)'}")
        print(f"    Answers: {row['answers'][:200]}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
