import argparse
import os
import re
import sys
import threading

HELP = """usage: needle <command> [options]

  run            run a checkpoint on a query
  finetune       train a LoRA adapter on JSONL data
  generate-data  synthesise training data via OpenRouter
  build          export a checkpoint to a .cact archive
  download       download weights or an engine build
  fetch          fetch the engine for this platform
  playground     serve the browser playground

needle <command> --help for the options of one command.
Check the readme for the rest."""


def _weights_spec(spec):
    parts = [p for p in spec.split("/") if p]
    if len(parts) < 2:
        raise SystemExit("pass <org>/<repo>/<file>.cact or <org>/<repo>")
    return "/".join(parts[:2]), "/".join(parts[2:]) or None


_ABSL_LOG_START = re.compile(rb"^[EIWF]\d{4} \d\d:\d\d:\d\d")
_NOISY_LOG_HEADER = re.compile(
    rb"\] (?:Fusion: .*gemm_fusion|Computation: .*_computation|Delay kernel timed out)"
)

_log_filter_installed = False


def _install_xla_log_filter():
    """Drop XLA Triton autotuner noise from stderr.

    XLA's Triton GEMM autotuner logs failed candidate fusions via LOG(ERROR)
    in xtile_compiler.cc and cuda_timer.cc. These are unconditional and do
    not respect TF_CPP_MIN_LOG_LEVEL, so we filter them at the file
    descriptor level.

    Strategy:
      - Rebind Python's sys.stderr to a fresh file object over the real
        terminal fd, so tqdm and print() writes go straight to the terminal
        and never enter our pipe. This keeps progress bars (which use \\r
        without trailing \\n) from stalling the filter's line parser.
      - Replace fd 2 with a pipe. Only C-level writes (absl / XLA LOG(...))
        now flow through the pipe, and they are always \\n-terminated and
        well-formed, so a simple line-based filter is reliable.
    """
    global _log_filter_installed
    if _log_filter_installed:
        return
    _log_filter_installed = True

    py_stderr_fd = os.dup(2)
    try:
        sys.stderr.flush()
    except Exception:
        pass
    sys.stderr = os.fdopen(py_stderr_fd, "w", encoding="utf-8",
                           errors="replace", buffering=1)

    out_fd = os.dup(2) 

    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)

    def pump():
        reader = os.fdopen(r_fd, "rb", buffering=0)
        out = os.fdopen(out_fd, "wb", buffering=0)
        buf = b""
        skipping = False
        try:
            while True:
                chunk = reader.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    idx = buf.find(b"\n")
                    if idx == -1:
                        break
                    line = bytes(buf[:idx])
                    buf = buf[idx + 1:]
                    is_log_start = bool(_ABSL_LOG_START.match(line))
                    if skipping:
                        if is_log_start:
                            if _NOISY_LOG_HEADER.search(line):
                                continue
                            skipping = False
                            out.write(line + b"\n")
                        # else: continuation body of a skipped log block — drop
                    else:
                        if is_log_start and _NOISY_LOG_HEADER.search(line):
                            skipping = True
                            continue
                        out.write(line + b"\n")
        except Exception:
            pass

    t = threading.Thread(target=pump, daemon=True, name="xla-log-filter")
    t.start()


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
_install_xla_log_filter()



def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        sys.exit(0)

    parser = argparse.ArgumentParser(prog="needle", add_help=False)
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("run")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--query", type=str, default=None, help="Query text for tool-call generation")
    p.add_argument("--tools", type=str, default=None, help="Tools JSON for tool-call generation")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy)")

    p = sub.add_parser("finetune")
    p.add_argument("jsonl_path", type=str, help="Path to JSONL training data")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Base model checkpoint (auto-downloads from HuggingFace if omitted)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16, help="LoRA adapter rank (default: 16)")
    p.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA scaling alpha (default: 32)")
    p.add_argument("--max-len", type=int, default=1024, help="Max training sequence length")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of examples held out for validation (0 disables)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for LoRA init, validation split, and epoch shuffling")
    p.add_argument("--generate", type=int, default=0,
                   help="Generate N extra examples via OpenRouter before training (0 = off)")
    p.add_argument("--model", type=str, default="deepseek/deepseek-v4-flash",
                   help="OpenRouter model for --generate")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent OpenRouter requests when generating (default: 8)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--out", type=str, default=None, help="Output adapter path (.pkl)")
    p.add_argument(
        "--qat-bits", choices=["auto", "none", "2", "4"], default="auto",
        help="LoRA training numerics: auto matches the checkpoint export scheme "
             "(default), or force none/2/4")

    p = sub.add_parser("generate-data")
    p.add_argument("--tools", type=str, default=None, help="Tool schemas JSON to seed generation")
    p.add_argument("--augment", type=str, default=None, help="Existing JSONL to expand")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--workers", type=int, default=16,
                   help="Concurrent OpenRouter requests (default: 16)")
    p.add_argument("--model", type=str, default="deepseek/deepseek-v4-flash")
    p.add_argument("--output", type=str, default=None)

    p = sub.add_parser("build")
    p.add_argument("checkpoint", type=str, help="Base checkpoint (.pkl) to export")
    p.add_argument("--lora", type=str, default=None, help="LoRA adapter to merge before export")
    p.add_argument("--out", type=str, default=None, help="Output .cact path")
    p.add_argument("--upload", action="store_true", help="Push the .cact to $NEEDLE_HF_REPO")
    p.add_argument("--bits", type=str, default=None, choices=["2", "4"])

    p = sub.add_parser("download")
    p.add_argument("spec", type=str,
                   help="Platform folder (e.g. macos-arm64), or Hugging Face spec: "
                        "<org>/<repo>/<file>.cact, or <org>/<repo> if it holds one archive")
    p.add_argument("--out", type=str, default=".", help="Directory to place the files")
    p.add_argument("--generation", type=int, choices=[2, 3], default=2,
                   help="Engine generation when downloading a platform build (default: 2)")

    p = sub.add_parser("fetch")
    p.add_argument("--out", type=str, default=None,
                   help="Directory to place the engine (default: the cache)")
    p.add_argument("--platform-tag", type=str, default=None,
                   help="Fetch the build for another device, e.g. manylinux2014_aarch64")
    p.add_argument("--generation", type=int, choices=[2, 3], default=2,
                   help="Needle engine generation to fetch (default: 2)")

    p = sub.add_parser("playground")
    p.add_argument("--weights", type=str, default=None,
                   help="Tuned .cact to serve (defaults to the base model from HuggingFace)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="127.0.0.1")

    args = parser.parse_args()

    if not args.command:
        print(HELP)
        sys.exit(0)

    from ._telemetry import track
    track("cli:" + args.command,
          {"generation": getattr(args, "generation", 2)})

    if args.command == "run":
        from .model.run import main as run_main
        run_main(args)
    elif args.command == "finetune":
        from .model.finetune import finetune_local
        finetune_local(args)
    elif args.command == "generate-data":
        from .model.finetune import generate_main
        generate_main(args)
    elif args.command == "build":
        from .model.finetune import build_main
        build_main(args)
    elif args.command == "download":
        import shutil
        from huggingface_hub import hf_hub_download, list_repo_files
        from .agent import fetch
        if "/" not in args.spec:
            if args.spec not in fetch.PLATFORMS:
                raise SystemExit("unknown platform, pick one of: "
                                 + ", ".join(fetch.PLATFORMS))
            paths = fetch.download_platform(args.spec, args.out,
                                            generation=args.generation)
            for path in paths:
                print(f"  {'file':<9} {path}  {os.path.getsize(path) / 1e6:.2f} MB")
            runner = next((p for p in paths
                           if os.path.basename(p) in ("needle", "needle.exe")), None)
            if runner:
                print(f"  {'next':<9} {runner} --tools tools.json --serve")
        else:
            fetch._register_download(args.generation)
            repo, filename = _weights_spec(args.spec)
            if not filename:
                cacts = [f for f in list_repo_files(repo) if f.endswith(".cact")]
                if len(cacts) != 1:
                    raise SystemExit(f"{repo} holds {len(cacts)} .cact files, name one: "
                                     + ", ".join(cacts[:5]))
                filename = cacts[0]
            cached = hf_hub_download(repo_id=repo, filename=filename, repo_type="model")
            os.makedirs(args.out, exist_ok=True)
            dest = os.path.join(args.out, os.path.basename(filename))
            shutil.copyfile(cached, dest)
            print(f"  {'weights':<9} {dest}  {os.path.getsize(dest) / 1e6:.2f} MB")
            print(f"  {'next':<9} needle.Needle(weights={dest!r}, tools=[...])")
    elif args.command == "fetch":
        from .agent import fetch
        generation = args.generation
        version = fetch.engine_version(generation)
        dest = args.out or os.path.join(os.path.expanduser("~"), ".cache",
                                        "cactus-needle", f"v{generation}", version)
        os.makedirs(dest, exist_ok=True)
        path = fetch.fetch_library(version, dest, tag=args.platform_tag,
                                   generation=generation)
        print(f"  {'engine':<9} {path}")
        print(f"  {'deploy':<9} copy to ~/.cache/cactus-needle/v{generation}/{version}/ "
              f"on the device, or point NEEDLE{generation}_LIB_PATH at the file")
    elif args.command == "playground":
        from .playground.server import main as playground_main
        playground_main(args)
