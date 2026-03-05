#!/usr/bin/env python3
"""
Ablation runner — reads schedule.tsv and runs experiments one by one.

Usage:
    python ablations/runner.py                  # run all pending
    python ablations/runner.py --dry-run        # print what would run
    python ablations/runner.py --list           # show schedule status
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
SCHEDULE = Path(__file__).parent / "schedule.tsv"
PYTHON = str(REPO / ".venv" / "bin" / "python")

# Base args shared by every ablation
BASE_ARGS = [
    "pretrain",
    "--full",                          # d_model=512, 8+4 layers
    "--wandb",
    "--epochs",      "999",            # epochs is a ceiling; max-steps controls runtime
    "--max-steps",   "315",            # ~30 min at ~0.18 steps/sec on v6e-8 w/ Muon
    "--batch-size",  "32",
    "--lr",          "3e-4",
    "--muon-lr",     "0.02",
    "--max-enc-len", "256",
    "--max-dec-len", "256",
    "--n-mels",      "80",
    "--speech-every","3",
    "--librilight-subset", "small",
    "--warmup-ratio","0.05",
    "--seed",        "42",
]


def read_schedule():
    lines = SCHEDULE.read_text().splitlines()
    entries = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0].strip()
        name   = parts[1].strip()
        extra  = parts[2].strip() if len(parts) > 2 else ""
        entries.append((status, name, extra, line))
    return entries


def update_status(name, new_status):
    text = SCHEDULE.read_text()
    new_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            new_lines.append(line)
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip() == name:
            parts[0] = new_status
            new_lines.append("\t".join(parts))
        else:
            new_lines.append(line)
    SCHEDULE.write_text("\n".join(new_lines) + "\n")


LOG_DIR = Path(__file__).parent.parent / "logs"


def run_experiment(name, extra_args, dry_run=False):
    ckpt_dir = str(REPO / "checkpoints" / "pretrain" / name)
    cmd = (
        [PYTHON, "-m", "src.cli"]
        + BASE_ARGS
        + ["--checkpoint-dir", ckpt_dir]
        + shlex.split(extra_args)
    )
    print(f"\n{'='*64}")
    print(f"  STARTING: {name}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*64}\n")

    if dry_run:
        return True

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"

    env = os.environ.copy()
    start = time.time()
    with open(log_path, "w") as log_file:
        log_file.write(f"CMD: {' '.join(cmd)}\n\n")
        log_file.flush()
        result = subprocess.run(cmd, cwd=str(REPO), env=env,
                                stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    status = "FINISHED" if result.returncode == 0 else "FAILED"
    print(f"\n{'='*64}")
    print(f"  {status}: {name}")
    print(f"  Elapsed: {elapsed/60:.1f} min  |  exit code: {result.returncode}")
    print(f"  Log: {log_path}")
    print(f"{'='*64}\n")
    return result.returncode == 0


def list_schedule():
    entries = read_schedule()
    print(f"\n{'Status':<12} {'Name':<25} Extra args")
    print("-" * 80)
    for status, name, extra, _ in entries:
        marker = {"pending": "⏳", "running": "🔄", "finished": "✅", "failed": "❌"}.get(status, "?")
        print(f"  {marker} {status:<10} {name:<25} {extra[:40]}")
    counts = {}
    for status, *_ in entries:
        counts[status] = counts.get(status, 0) + 1
    print(f"\n  Total: {len(entries)}  |  " + "  ".join(f"{s}={n}" for s, n in counts.items()))
    print()


def main():
    parser = argparse.ArgumentParser(description="Ablation runner")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--list",    action="store_true", help="Show schedule status")
    parser.add_argument("--name",    type=str, default=None,
                        help="Run a specific experiment by name (skips others)")
    args = parser.parse_args()

    if args.list:
        list_schedule()
        return

    print(f"Ablation runner starting. Schedule: {SCHEDULE}")
    if args.dry_run:
        print("DRY RUN — no experiments will execute.\n")

    ran = 0
    shown = set()  # for dry-run: track which we've already printed

    while True:
        entries = read_schedule()
        next_run = None
        for status, name, extra, _ in entries:
            if args.name and name != args.name:
                continue
            if status == "pending" and name not in shown:
                next_run = (name, extra)
                break

        if next_run is None:
            print("No pending experiments." if ran == 0 else f"All done. Ran {ran} experiment(s).")
            break

        name, extra = next_run
        shown.add(name)

        if not args.dry_run:
            update_status(name, "running")

        ok = run_experiment(name, extra, dry_run=args.dry_run)

        if not args.dry_run:
            update_status(name, "finished" if ok else "failed")

        ran += 1
        if not args.dry_run and not ok:
            print(f"Experiment '{name}' failed. Continuing to next...\n")
            time.sleep(2)

    list_schedule()


if __name__ == "__main__":
    main()
