#!/usr/bin/env python3
"""Build text-only tool-call dataset and save locally.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --max-samples 1000
    python scripts/build_dataset.py --output data/tool_calls_unified
"""

import argparse
import json
import logging
import os
import sys

_shm = "/dev/shm"
if os.path.isdir(_shm):
    try:
        _st = os.statvfs(_shm)
        if _st.f_bavail * _st.f_frsize >= 200 * 1024**3:
            _hf_cache = os.path.join(_shm, "needle_hf_cache")
            os.makedirs(_hf_cache, exist_ok=True)
            os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_hf_cache, "datasets"))
    except OSError:
        pass

_token_path = os.path.expanduser("~/.cache/huggingface/token")
if os.path.isfile(_token_path) and "HF_TOKEN" not in os.environ:
    with open(_token_path) as _f:
        _tok = _f.read().strip()
    if _tok:
        os.environ["HF_TOKEN"] = _tok

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tools_data import load_and_combine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _default_output():
    """Use /dev/shm for output when RAM is plentiful."""
    shm = "/dev/shm"
    if os.path.isdir(shm):
        try:
            st = os.statvfs(shm)
            if st.f_bavail * st.f_frsize >= 200 * 1024**3:
                return os.path.join(shm, "needle_data", "tool_calls_unified")
        except OSError:
            pass
    return "data/tool_calls_unified"


def main():
    parser = argparse.ArgumentParser(description="Build text-only tool-call dataset")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to include")
    parser.add_argument("--output", type=str, default=_default_output(), help="Output directory")
    args = parser.parse_args()

    logger.info("Loading and combining text datasets...")
    combined = load_and_combine()

    if args.max_samples:
        combined = combined.select(range(min(args.max_samples, len(combined))))

    logger.info(f"Dataset: {len(combined)} rows")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    combined.save_to_disk(args.output)
    logger.info(f"Saved {len(combined)} rows to {args.output}/")

    # Persist to GCS
    from src.gcs import upload_raw_data
    upload_raw_data(args.output)

    print_summary(combined)


def print_summary(ds, n_samples=5):
    """Print dataset statistics and sample examples."""
    n = len(ds)

    # Parse tools/answers counts
    n_tools, n_calls = [], []
    n_with_calls, n_no_calls = 0, 0
    query_lens, tools_lens, answers_lens = [], [], []
    for ex in ds:
        try:
            tools = json.loads(ex["tools"])
        except (json.JSONDecodeError, TypeError):
            tools = []
        try:
            answers = json.loads(ex["answers"])
        except (json.JSONDecodeError, TypeError):
            answers = []
        n_tools.append(len(tools))
        n_calls.append(len(answers))
        if answers:
            n_with_calls += 1
        else:
            n_no_calls += 1
        query_lens.append(len(ex["query"]))
        tools_lens.append(len(ex["tools"]))
        answers_lens.append(len(ex["answers"]))

    n_tools = np.array(n_tools)
    n_calls = np.array(n_calls)
    query_lens = np.array(query_lens)
    tools_lens = np.array(tools_lens)
    answers_lens = np.array(answers_lens)

    def _stats(arr):
        return f"min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}, median={np.median(arr):.0f}"

    print(f"\n{'=' * 60}")
    print(f"DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total examples:       {n:,}")
    print(f"  With tool calls:      {n_with_calls:,} ({n_with_calls/n*100:.1f}%)")
    print(f"  No-call (empty):      {n_no_calls:,} ({n_no_calls/n*100:.1f}%)")
    print()
    print(f"  Available tools/sample:  {_stats(n_tools)}")
    print(f"  Tool calls/sample:       {_stats(n_calls)}")
    print()
    print(f"  Query length (chars):    {_stats(query_lens)}")
    print(f"  Tools length (chars):    {_stats(tools_lens)}")
    print(f"  Answers length (chars):  {_stats(answers_lens)}")

    # Samples with tool calls
    with_calls_idx = [i for i in range(n) if n_calls[i] > 0]
    no_calls_idx = [i for i in range(n) if n_calls[i] == 0]

    rng = np.random.RandomState(42)

    if with_calls_idx:
        sample_idx = rng.choice(with_calls_idx, size=min(n_samples, len(with_calls_idx)), replace=False)
        print(f"\n{'─' * 60}")
        print(f"SAMPLES WITH TOOL CALLS ({len(sample_idx)})")
        print(f"{'─' * 60}")
        for i, idx in enumerate(sample_idx):
            ex = ds[int(idx)]
            print(f"\n  [{i+1}] Query:   {ex['query']}")
            print(f"      Tools:   {ex['tools']}")
            print(f"      Answers: {ex['answers']}")

    if no_calls_idx:
        sample_idx = rng.choice(no_calls_idx, size=min(n_samples, len(no_calls_idx)), replace=False)
        print(f"\n{'─' * 60}")
        print(f"SAMPLES WITHOUT TOOL CALLS ({len(sample_idx)})")
        print(f"{'─' * 60}")
        for i, idx in enumerate(sample_idx):
            ex = ds[int(idx)]
            print(f"\n  [{i+1}] Query:   {ex['query']}")
            print(f"      Tools:   {ex['tools']}")
            print(f"      Answers: {ex['answers']}")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
