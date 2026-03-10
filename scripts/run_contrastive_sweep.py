import argparse
import os
import subprocess
import sys
from datetime import datetime


WEIGHTS = [0.0, 0.1, 0.5, 1.0, 2.0]


def _fmt_weight(value):
    text = str(value)
    return text.replace(".", "p")


def _needle_path(repo_root):
    local = os.path.join(repo_root, ".venv", "bin", "needle")
    return local if os.path.exists(local) else "needle"


def _run_one(repo_root, log_root, ckpt_root, name, audio_text_weight, tool_weight, args):
    run_log = os.path.join(log_root, f"{name}.log")
    run_ckpt = os.path.join(ckpt_root, name)
    if not args.no_checkpoints:
        os.makedirs(run_ckpt, exist_ok=True)

    cmd = [
        _needle_path(repo_root),
        "train",
        "--epochs", str(args.epochs),
        "--eval-every", str(args.eval_every),
        "--audio-text-contrastive-weight", str(audio_text_weight),
        "--tool-contrastive-weight", str(tool_weight),
    ]
    if args.no_checkpoints:
        cmd.append("--no-checkpoints")
    else:
        cmd.extend(["--checkpoint-dir", run_ckpt])
    if args.skip_epoch_extras:
        cmd.append("--skip-epoch-extras")
    if args.cfg_inference:
        cmd.append("--cfg-inference")

    with open(run_log, "w") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=repo_root, stdout=f, stderr=subprocess.STDOUT, check=False)
        return proc.returncode, run_log


def main():
    parser = argparse.ArgumentParser(description="Run contrastive-weight sweeps with separate log files")
    parser.add_argument("--mode", choices=["audio5", "grid25"], default="audio5")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1000000)
    parser.add_argument("--skip-epoch-extras", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--cfg-inference", action="store_true")
    parser.add_argument("--log-root", type=str, default="logs/contrastive_sweeps")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints/contrastive_sweeps")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(__file__))
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(repo_root, args.log_root, f"{args.mode}_{stamp}")
    os.makedirs(run_root, exist_ok=True)
    ckpt_root = os.path.join(repo_root, args.checkpoint_root, f"{args.mode}_{stamp}")
    if not args.no_checkpoints:
        os.makedirs(ckpt_root, exist_ok=True)

    combos = []
    if args.mode == "audio5":
        for audio_text_weight in WEIGHTS:
            combos.append((audio_text_weight, 0.0))
    else:
        for audio_text_weight in WEIGHTS:
            for tool_weight in WEIGHTS:
                combos.append((audio_text_weight, tool_weight))

    summary_path = os.path.join(run_root, "summary.txt")
    with open(summary_path, "w") as summary:
        summary.write(f"mode={args.mode}\n")
        summary.write(f"epochs={args.epochs}\n")
        summary.write(f"eval_every={args.eval_every}\n")
        summary.write(f"skip_epoch_extras={args.skip_epoch_extras}\n")
        summary.write(f"no_checkpoints={args.no_checkpoints}\n")
        summary.write(f"cfg_inference={args.cfg_inference}\n\n")
        for audio_text_weight, tool_weight in combos:
            name = f"at_{_fmt_weight(audio_text_weight)}__tool_{_fmt_weight(tool_weight)}"
            code, run_log = _run_one(repo_root, run_root, ckpt_root, name, audio_text_weight, tool_weight, args)
            summary.write(f"{name}\treturncode={code}\tlog={run_log}\n")
            summary.flush()
            print(f"{name}: returncode={code} log={run_log}")


if __name__ == "__main__":
    main()
