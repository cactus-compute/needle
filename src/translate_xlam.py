#!/usr/bin/env python3
"""Translate xlam-sourced examples from the unified dataset into 24 non-English
languages using Gemini, then merge back into the dataset.

Translates both the query AND free-text argument values (message text, notes,
search queries, labels, destinations, etc.) so the data matches real-world usage
where a French speaker's message content is also in French. Structured values
(tool names, param keys, contact_ids, enum choices, booleans, numbers) stay
unchanged.

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

# Param types that should never be translated
_KEEP_TYPES = frozenset({"number", "boolean"})

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


def _build_translation_prompt(query, translatable_args, language):
    """Build a prompt to translate query + arg values in one shot."""
    payload = {"query": query}
    for i, (call_idx, pname, pval) in enumerate(translatable_args):
        payload[f"arg_{i}"] = pval

    return f"""Translate the following JSON values from English to {language}.

RULES:
- Translate ONLY the values, not the keys
- Use natural, native-sounding {language} — not literal word-for-word translation
- Use culturally appropriate names, places, and expressions where relevant
- Keep proper nouns (brand names, app names) unchanged
- Keep numbers, dates, and times in their original format
- If a value is already a single common word that works in {language}, keep it
- Return ONLY valid JSON with the same keys

INPUT:
{json.dumps(payload, ensure_ascii=False)}

OUTPUT (valid JSON only, no markdown):"""


def translate_example(client_pool, example, language, model):
    """Translate a single example's query and free-text argument values."""
    query = example["query"]
    answers = json.loads(example["answers"])
    tools = example["tools"]  # pass through unchanged

    # Collect translatable argument values
    translatable_args = []  # (call_idx, param_name, param_value)
    for call_idx, call in enumerate(answers):
        args = call.get("arguments", {})
        for pname, pval in args.items():
            if _should_translate_param(pname, pval):
                translatable_args.append((call_idx, pname, pval))

    # Build and send translation prompt
    prompt = _build_translation_prompt(query, translatable_args, language)

    client = client_pool.get()
    response = client.models.generate_content(
        model=model, contents=prompt,
        config={"temperature": 0.3, "max_output_tokens": 4096},
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]

    translated = json.loads(text.strip())

    # Apply translations
    new_query = translated.get("query", query)

    new_answers = json.loads(json.dumps(answers))  # deep copy
    for i, (call_idx, pname, _) in enumerate(translatable_args):
        key = f"arg_{i}"
        if key in translated:
            new_answers[call_idx]["arguments"][pname] = translated[key]

    return {
        "query": new_query,
        "tools": tools,
        "answers": json.dumps(new_answers, separators=(",", ":"), ensure_ascii=False),
        "source": "xlam-translated",
        "model": f"{model}-translate",
        "language": language,
    }


def translate_batch(client_pool, examples, languages, model):
    """Translate a batch of examples, round-robin across languages."""
    results = []
    for i, ex in enumerate(examples):
        lang = languages[i % len(languages)]
        try:
            translated = translate_example(client_pool, ex, lang, model)
            results.append(translated)
        except Exception:
            pass  # skip failures silently
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
        print(f"Downloading from {HF_DATASET_REPO}...")
        ds = load_dataset(HF_DATASET_REPO, split="train", token=True)
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

    # Shuffle languages for each pass to get even distribution
    rng = random.Random(42)
    lang_assignments = []
    for i in range(len(examples)):
        lang_assignments.append(LANGUAGES[i % len(LANGUAGES)])
    # Shuffle to avoid correlating language with xlam ordering
    combined = list(zip(examples, lang_assignments))
    rng.shuffle(combined)
    examples, lang_assignments = zip(*combined)
    examples, lang_assignments = list(examples), list(lang_assignments)

    # Translate in parallel
    translated = []
    failed = 0
    pbar = tqdm(total=len(examples), desc="Translating", unit="ex")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for i, (ex, lang) in enumerate(zip(examples, lang_assignments)):
            f = pool.submit(translate_example, client_pool, ex, lang, args.model)
            futures[f] = i

        for f in concurrent.futures.as_completed(futures):
            try:
                result = f.result()
                translated.append(result)
            except Exception as e:
                failed += 1
            pbar.update(1)
            pbar.set_postfix(translated=len(translated), failed=failed)

    pbar.close()

    # Stats
    from collections import Counter
    lang_counts = Counter(ex["language"] for ex in translated)
    print(f"\nTranslated {len(translated)} examples ({failed} failed)")
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
    p.add_argument("--batch-size", type=int, default=10, help="Examples per batch")
    p.add_argument("--dry-run", action="store_true", help="Translate only, skip save/upload")
    main(p.parse_args())
