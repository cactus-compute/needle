#!/usr/bin/env python3
"""Translate xlam-sourced examples from the unified dataset into 24 non-English
languages using Gemini, then merge back into the dataset.

Translates both the query AND free-text argument values (message text, notes,
search queries, labels, destinations, etc.) so the data matches real-world usage
where a French speaker's message content is also in French. Structured values
(tool names, param keys, contact_ids, enum choices, booleans, numbers) stay
unchanged.

Examples are batched (default 10 per Gemini call) and grouped by language for
efficiency — ~60k xlam × 24 languages = ~1.4M rows in ~144k API calls instead
of 1.4M.

Usage:
    python -m src.translate_xlam                           # translate + upload
    python -m src.translate_xlam --dry-run                 # translate only
    python -m src.translate_xlam --max-samples 100         # small test run
    python -m src.translate_xlam --workers 16              # more parallelism
"""

import argparse
import concurrent.futures
import json
import os
import random
import sys
import threading

from tqdm import tqdm

try:
    from google import genai
except ImportError:
    print("Error: google-genai not installed. Run: pip install google-genai", file=sys.stderr)
    sys.exit(1)


# Non-English target languages (English already covered by original xlam data)
LANGUAGES = [
    "Bulgarian", "Croatian", "Czech", "Danish", "Dutch",
    "Estonian", "Finnish", "French", "German", "Greek", "Hungarian",
    "Italian", "Latvian", "Lithuanian", "Maltese", "Polish",
    "Portuguese", "Romanian", "Slovak", "Slovenian", "Spanish",
    "Swedish", "Russian", "Ukrainian",
]

MODEL = "gemini-3.1-flash-lite-preview"

# Param names/patterns where values should NOT be translated
_KEEP_ENGLISH = frozenset({
    "contact_id", "password", "url", "source", "confirmation_code",
    "order_id", "post_id", "file_name", "archive_name", "folder_name",
    "destination_folder", "app_name", "device_name", "ssid",
    "site", "username", "email", "to", "cc", "bcc",
    "phone", "phone_number", "js",
})

# Enum-like params where the value is from a fixed set — keep as-is
_ENUM_HINTS = frozenset({
    "action", "mode", "direction", "position", "camera", "cycle",
    "ride_type", "service_type", "trigger", "data_type", "region",
    "restriction_level", "level", "unit", "connection_type",
    "setting", "goal_type", "input", "connector_type",
})


LOCAL_UNIFIED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tool_calls_unified")
HF_DATASET_REPO = "Cactus-Compute/tool-calls"


class ClientPool:
    """Round-robin pool of Gemini clients."""

    def __init__(self, clients):
        self._clients = clients
        self._idx = 0
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            client = self._clients[self._idx % len(self._clients)]
            self._idx += 1
            return client


def make_clients():
    raw = os.environ.get("GEMINI_API_KEY", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        print("Error: GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    clients = [genai.Client(api_key=k) for k in keys]
    print(f"Using {len(clients)} API key(s)")
    return clients


def _should_translate_param(pname, pval):
    """Decide if a parameter value should be translated."""
    if pname in _KEEP_ENGLISH:
        return False
    if pname in _ENUM_HINTS:
        return False
    if not isinstance(pval, str):
        return False
    # Short values that look like identifiers/codes — skip
    if len(pval) < 3:
        return False
    # Values that look like file paths, URLs, or IDs
    if "/" in pval or "@" in pval or pval.startswith(("http", "www")):
        return False
    # Pure numbers or booleans encoded as strings
    try:
        float(pval)
        return False
    except ValueError:
        pass
    if pval.lower() in ("true", "false", "null", "none"):
        return False
    return True


def _extract_translatable(example):
    """Extract query + translatable arg values from an example.

    Returns (query, translatable_args) where translatable_args is a list of
    (call_idx, param_name, param_value) tuples.
    """
    answers = json.loads(example["answers"])
    translatable_args = []
    for call_idx, call in enumerate(answers):
        args = call.get("arguments", {})
        for pname, pval in args.items():
            if _should_translate_param(pname, pval):
                translatable_args.append((call_idx, pname, pval))
    return example["query"], translatable_args


def _apply_translation(example, translated_item, language):
    """Apply a translated payload back onto an example."""
    answers = json.loads(example["answers"])
    query = example["query"]

    new_query = translated_item.get("query", query)

    # Reconstruct translatable_args to map arg_N keys back
    _, translatable_args = _extract_translatable(example)
    new_answers = json.loads(json.dumps(answers))  # deep copy
    for i, (call_idx, pname, _) in enumerate(translatable_args):
        key = f"arg_{i}"
        if key in translated_item:
            new_answers[call_idx]["arguments"][pname] = translated_item[key]

    return {
        "query": new_query,
        "tools": example["tools"],
        "answers": json.dumps(new_answers, separators=(",", ":"), ensure_ascii=False),
        "source": "xlam-translated",
        "model": example.get("_model_tag", ""),
        "language": language,
    }


def translate_batch(client_pool, examples, language, model):
    """Translate a batch of examples into a single language in one Gemini call.

    Packs all examples into one JSON array prompt, gets back a translated array.
    Returns list of translated example dicts (may be shorter than input on errors).
    """
    # Build batch payload: list of {query, arg_0, arg_1, ...} per example
    batch_payload = []
    for ex in examples:
        query, translatable_args = _extract_translatable(ex)
        item = {"query": query}
        for i, (_, _, pval) in enumerate(translatable_args):
            item[f"arg_{i}"] = pval
        batch_payload.append(item)

    prompt = f"""Translate each JSON object's values from English to {language}.

RULES:
- Translate ONLY the string values, not the keys
- Use natural, native-sounding {language} — not literal word-for-word translation
- Use culturally appropriate names, places, and expressions where relevant
- Keep proper nouns (brand names, app names) unchanged
- Keep numbers, dates, and times in their original format
- If a value is already a single common word that works in {language}, keep it
- Return a JSON array with the SAME number of objects in the SAME order
- Each object must have the SAME keys as the input — only values change

INPUT ({len(batch_payload)} items):
{json.dumps(batch_payload, ensure_ascii=False)}

OUTPUT (valid JSON array, no markdown, {len(batch_payload)} items):"""

    client = client_pool.get()
    response = client.models.generate_content(
        model=model, contents=prompt,
        config={"temperature": 0.3, "max_output_tokens": 16384},
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]

    translated_batch = json.loads(text.strip())
    if not isinstance(translated_batch, list):
        return []

    # Apply translations back to examples
    results = []
    model_tag = f"{model}-translate"
    for ex, translated_item in zip(examples, translated_batch):
        if not isinstance(translated_item, dict):
            continue
        ex_with_tag = {**ex, "_model_tag": model_tag}
        try:
            result = _apply_translation(ex_with_tag, translated_item, language)
            results.append(result)
        except Exception:
            continue

    return results


def main(args):
    from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

    client_pool = ClientPool(make_clients())

    # Load existing dataset
    local = os.path.abspath(LOCAL_UNIFIED_DIR)
    if os.path.exists(local) and any(f.endswith(".arrow") for f in os.listdir(local)):
        ds = load_from_disk(local)
        print(f"Loaded existing dataset: {len(ds)} rows")
    else:
        from .data import download_hf_split
        print(f"Downloading from {HF_DATASET_REPO}...")
        ds = download_hf_split("train", HF_DATASET_REPO)
        os.makedirs(local, exist_ok=True)
        ds.save_to_disk(local)
        print(f"Downloaded: {len(ds)} rows")

    # Filter to xlam-sourced examples only
    xlam_indices = [i for i, src in enumerate(ds["source"]) if "xlam" in src.lower()]
    xlam_ds = ds.select(xlam_indices)
    print(f"Found {len(xlam_ds)} xlam-sourced examples to translate")

    if args.max_samples and args.max_samples < len(xlam_ds):
        xlam_ds = xlam_ds.shuffle(seed=42).select(range(args.max_samples))
        print(f"Sampled down to {len(xlam_ds)} examples")

    # Prepare examples as dicts
    examples = []
    for i in range(len(xlam_ds)):
        examples.append({
            "query": xlam_ds[i]["query"],
            "tools": xlam_ds[i]["tools"],
            "answers": xlam_ds[i]["answers"],
        })

    # Assign languages round-robin and shuffle to decorrelate from xlam order
    rng = random.Random(42)
    lang_assignments = [LANGUAGES[i % len(LANGUAGES)] for i in range(len(examples))]
    combined = list(zip(examples, lang_assignments))
    rng.shuffle(combined)
    examples, lang_assignments = zip(*combined)
    examples, lang_assignments = list(examples), list(lang_assignments)

    batches = []  
    lang_buckets = {}
    for ex, lang in zip(examples, lang_assignments):
        if lang not in lang_buckets:
            lang_buckets[lang] = []
        lang_buckets[lang].append(ex)
        if len(lang_buckets[lang]) >= args.batch_size:
            batches.append((lang, lang_buckets[lang]))
            lang_buckets[lang] = []
    
    for lang, remaining in lang_buckets.items():
        if remaining:
            batches.append((lang, remaining))

    rng.shuffle(batches)  
    total_examples = sum(len(b) for _, b in batches)
    print(f"Prepared {len(batches)} batches ({total_examples} examples, batch_size={args.batch_size})")

    # Translate in parallel — one future per batch
    translated = []
    failed_batches = 0
    failed_examples = 0
    pbar = tqdm(total=len(batches), desc="Translating", unit="batch")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for i, (lang, batch_examples) in enumerate(batches):
            f = pool.submit(translate_batch, client_pool, batch_examples, lang, args.model)
            futures[f] = (i, len(batch_examples))

        for f in concurrent.futures.as_completed(futures):
            batch_idx, batch_len = futures[f]
            try:
                results = f.result()
                translated.extend(results)
                if len(results) < batch_len:
                    failed_examples += batch_len - len(results)
            except Exception:
                failed_batches += 1
                failed_examples += batch_len
            pbar.update(1)
            pbar.set_postfix(
                translated=len(translated),
                failed_batches=failed_batches,
                failed_ex=failed_examples,
            )

    pbar.close()

    # Stats
    from collections import Counter
    lang_counts = Counter(ex["language"] for ex in translated)
    print(f"\nTranslated {len(translated)} examples "
          f"({failed_batches} failed batches, {failed_examples} failed examples)")
    print(f"Language distribution: {dict(lang_counts.most_common())}")

    # Show samples
    rng.shuffle(translated)
    print(f"\nSample translations:")
    for ex in translated[:20]:
        print(f"  [{ex['language']}] Q: {ex['query'][:120]}")
        print(f"         A: {ex['answers'][:150]}")
        print()

    if args.dry_run:
        print("--dry-run: skipping save and upload")
        return

    # Merge into existing dataset
    new_ds = Dataset.from_dict({
        "query": [ex["query"] for ex in translated],
        "tools": [ex["tools"] for ex in translated],
        "answers": [ex["answers"] for ex in translated],
        "source": [ex["source"] for ex in translated],
        "model": [ex["model"] for ex in translated],
        "language": [ex["language"] for ex in translated],
    })

    # Handle schema compatibility
    for col in new_ds.column_names:
        if col not in ds.column_names:
            ds = ds.add_column(col, ["English"] * len(ds) if col == "language" else [""] * len(ds))
    new_ds = new_ds.select_columns(ds.column_names)

    merged = concatenate_datasets([ds, new_ds])
    print(f"\nMerged: {len(merged)} rows (+{len(new_ds)} translated)")

    for src, cnt in Counter(merged["source"]).most_common():
        print(f"  {src}: {cnt}")

    # Save locally
    import shutil
    tmp_dir = local + "_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    merged.save_to_disk(tmp_dir)
    if os.path.exists(local):
        shutil.rmtree(local)
    os.rename(tmp_dir, local)
    print("Saved locally.")

    # Upload
    import tempfile
    from huggingface_hub import HfApi, CommitOperationDelete

    api = HfApi()
    api.create_repo(HF_DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)

    parquet_dir = tempfile.mkdtemp(prefix="needle_xlam_translate_")
    print("Exporting to parquet...")
    merged.to_parquet(os.path.join(parquet_dir, "train.parquet"))

    files = api.list_repo_files(HF_DATASET_REPO, repo_type="dataset", token=True)
    old_train = [f for f in files if f.startswith("data/train-")]
    if old_train:
        print(f"Deleting {len(old_train)} old train shards...")
        ops = [CommitOperationDelete(path_in_repo=f) for f in old_train]
        api.create_commit(
            repo_id=HF_DATASET_REPO, repo_type="dataset", operations=ops,
            commit_message="Remove old train shards before re-upload", token=True,
        )

    print(f"Uploading to {HF_DATASET_REPO}...")
    api.upload_file(
        path_or_fileobj=os.path.join(parquet_dir, "train.parquet"),
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=HF_DATASET_REPO, repo_type="dataset", token=True,
        commit_message=f"Upload train data with xlam translations ({len(merged)} rows)",
    )

    shutil.rmtree(parquet_dir)
    print(f"Upload complete: {HF_DATASET_REPO}")
    print("NOTE: Run 'needle split-dataset' to create the validation split.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Translate xlam examples to 24 languages")
    p.add_argument("--max-samples", type=int, default=None, help="Limit examples to translate")
    p.add_argument("--workers", type=int, default=8, help="Parallel Gemini calls")
    p.add_argument("--model", type=str, default=MODEL, help="Gemini model for translation")
    p.add_argument("--batch-size", type=int, default=10, help="Examples per Gemini call")
    p.add_argument("--dry-run", action="store_true", help="Translate only, skip save/upload")
    main(p.parse_args())
