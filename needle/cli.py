import argparse
import json
import os
import re
import sys
import threading

HELP = """usage: needle <command> [options]

  run            run a checkpoint on a query (JAX, Needle 3)
  finetune       train a LoRA adapter on JSONL data (--layers N for a rung)
  generate-data  synthesise training data via a gateway (OpenRouter or OrcaRouter)
  connect        connect an OrcaRouter account (OAuth 2.0 + PKCE) and store the key
  models         list the models a provider offers, filtered by capability
  build          export a checkpoint (+ adapter) to a .cact archive
  download       needle3 | needle3.safetensors | <platform> | <org>/<repo>[/<file>.cact]
  fetch          fetch the engine library for this platform
  playground     serve the browser playground

needle <command> --help for the options of one command.
Check the readme for the rest."""


def _weights_spec(spec):
    parts = [p for p in spec.split("/") if p]
    if len(parts) < 2:
        raise SystemExit("pass <org>/<repo>/<file>.cact or <org>/<repo>")
    return "/".join(parts[:2]), "/".join(parts[2:]) or None


def _download_target(spec):
    """Classify a `needle download` spec: engine build, base weights, checkpoint or hub archive."""
    from .agent import fetch

    if "/" in spec:
        return "hub", spec
    if spec in fetch.PLATFORMS:
        return "platform", spec
    name = spec[:-5] if spec.endswith(".cact") else spec
    if name in ("needle2", "needle3"):
        return "base", int(name[-1])
    if spec.endswith((".safetensors", ".pkl")):
        return "checkpoint", spec
    raise SystemExit(
        f"unknown download {spec!r}: pass needle3 (base weights), needle3.safetensors "
        f"(the checkpoint to fine-tune), a platform ("
        + ", ".join(fetch.PLATFORMS) + "), or <org>/<repo>[/<file>.cact]")


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

    from .agent import fetch

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
                   help="Generate N extra examples via the gateway before training (0 = off)")
    p.add_argument("--provider", type=str, default=None,
                   help="Provider for generated data: orcarouter or openrouter "
                        "(default: openrouter, or $ORCAROUTER_API_KEY when set)")
    p.add_argument("--model", type=str, default=None,
                   help="Model for --generate (defaults to the provider's)")
    p.add_argument("--input-modality", type=str, default=None,
                   choices=["text", "image"],
                   help="Modality the generated prompts carry (with --generate)")
    p.add_argument("--image-url", type=str, default=None,
                   help="Image sent with every generated prompt when the modality is image")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent gateway requests when generating (default: 8)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--out", type=str, default=None,
                   help="Output adapter path (.safetensors, or .pkl)")

    p = sub.add_parser("generate-data")
    p.add_argument("--tools", type=str, default=None, help="Tool schemas JSON to seed generation")
    p.add_argument("--augment", type=str, default=None, help="Existing JSONL to expand")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--workers", type=int, default=16,
                   help="Concurrent gateway requests (default: 16)")
    p.add_argument("--provider", type=str, default=None,
                   help="orcarouter or openrouter (default: openrouter, or "
                        "$ORCAROUTER_API_KEY when that is the only key set)")
    p.add_argument("--model", type=str, default=None,
                   help="Model id (defaults to the provider's own default)")
    p.add_argument("--input-modality", type=str, default=None,
                   choices=["text", "image"],
                   help="Modality the prompts carry; image requires a model that "
                        "declares an image input, and is rejected otherwise")
    p.add_argument("--image-url", type=str, default=None,
                   help="Image sent with every prompt when --input-modality image")
    p.add_argument("--output", type=str, default=None)

    p = sub.add_parser("connect")
    p.add_argument("--oob", action="store_true",
                   help="Use the out-of-band code flow instead of a loopback redirect")
    p.add_argument("--app-name", type=str, default=None,
                   help="Name shown on the consent screen (default: Needle)")
    p.add_argument("--scope", type=str, default="api", choices=["api", "connector"],
                   help="Requested scope (default: api)")
    p.add_argument("--no-browser", action="store_true",
                   help="Print the authorization URL instead of opening a browser")
    p.add_argument("--login-hint", type=str, default=None,
                   help="Pre-fill the account email on the consent screen")
    p.add_argument("--workspace-hint", type=str, default=None,
                   help="Pre-select a workspace on the consent screen")
    p.add_argument("--timeout", type=int, default=600,
                   help="Seconds to wait for the loopback callback (default: 600)")

    p = sub.add_parser("models")
    p.add_argument("--provider", type=str, default=None,
                   help="orcarouter or openrouter (default: orcarouter)")
    p.add_argument("--capability", type=str, default="chat",
                   choices=["chat", "embedding", "image", "video", "rerank"],
                   help="Only list models that can serve this (default: chat)")
    p.add_argument("--input-modality", type=str, default=None,
                   help="Require a declared input modality, e.g. image")
    p.add_argument("--json", action="store_true", help="Print the full records as JSON")

    p = sub.add_parser("build")
    p.add_argument("checkpoint", type=str, nargs="?", default=None,
                   help="Base checkpoint (.safetensors or .pkl); defaults to the adapter's "
                        "base, else the Needle 3 base (auto-downloads)")
    p.add_argument("--lora", type=str, default=None, help="LoRA adapter to merge before export")
    p.add_argument("--out", type=str, default=None, help="Output .cact path")
    p.add_argument("--upload", action="store_true", help="Push the .cact to $NEEDLE_HF_REPO")
    p.add_argument("--layers", type=int, default=None,
                   help="Export the N-layer rung of the base (2..20); default the full 20")
    p.add_argument("--platform", type=str, default=None, choices=fetch.PLATFORMS,
                   help="Also download that platform's engine and header, and place the "
                        "archive beside them as needle3.cact (--out is then a directory)")

    p = sub.add_parser("download")
    p.add_argument("spec", type=str,
                   help="needle3 (base weights), needle3.safetensors (the checkpoint to "
                        "fine-tune), a platform "
                        "folder (e.g. macos-arm64), or a Hugging Face spec: "
                        "<org>/<repo>/<file>.cact, or <org>/<repo> if it holds one archive")
    p.add_argument("--out", type=str, default=".", help="Directory to place the files")
    p.add_argument("--generation", type=int, choices=[2, 3], default=3,
                   help="Engine generation when downloading a platform build (default: 3)")

    p = sub.add_parser("fetch")
    p.add_argument("--out", type=str, default=None,
                   help="Directory to place the engine (default: the cache)")
    p.add_argument("--platform-tag", type=str, default=None,
                   help="Fetch the build for another device, e.g. manylinux2014_aarch64")
    p.add_argument("--generation", type=int, choices=[2, 3], default=3,
                   help="Needle engine generation to fetch (default: 3)")

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
        kind, target = _download_target(args.spec)
        if kind == "platform":
            paths = fetch.download_platform(target, args.out,
                                            generation=args.generation)
            if args.generation >= 3:
                paths.append(fetch.fetch_weights(args.generation, os.path.join(args.out, target)))
            for path in paths:
                print(f"  {'file':<9} {path}  {os.path.getsize(path) / 1e6:.2f} MB")
            runner = next((p for p in paths
                           if os.path.basename(p) in ("needle", "needle.exe")), None)
            if runner:
                weights = f" --model {fetch.base_weights(args.generation)}" if args.generation >= 3 else ""
                print(f"  {'next':<9} {runner}{weights} --tools tools.json --serve")
        elif kind == "base":
            os.makedirs(args.out, exist_ok=True)
            path = fetch.fetch_weights(target, args.out)
            print(f"  {'weights':<9} {path}  {os.path.getsize(path) / 1e6:.2f} MB")
            print(f"  {'next':<9} needle.Needle(weights={path!r}, tools=[...])")
        elif kind == "checkpoint":
            generation = 2 if target.startswith("needle2") else 3
            path = fetch.fetch_checkpoint(target, os.path.join(args.out, fetch.CHECKPOINT_PREFIX),
                                          generation=generation)
            print(f"  {'file':<9} {path}  {os.path.getsize(path) / 1e6:.2f} MB")
            print(f"  {'next':<9} needle finetune data.jsonl --checkpoint {path} [--layers N]")
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
    elif args.command == "connect":
        _connect_main(args)
    elif args.command == "models":
        _models_main(args)


def _connect_main(args):
    """`needle connect` — obtain an OrcaRouter key through OAuth 2.0 + PKCE."""
    import webbrowser

    from .model.credentials import (CredentialError, CredentialStore, begin_connect,
                                    finish_connect, mask_secret)
    from .model.providers import resolve_orcarouter_origins

    auth_base, _ = resolve_orcarouter_origins()
    flow = "oob" if args.oob else "loopback"
    try:
        session, public = begin_connect(
            flow=flow, app_name=args.app_name or "Needle", scope=args.scope,
            login_hint=args.login_hint, workspace_hint=args.workspace_hint)
    except CredentialError as exc:
        raise SystemExit("  %-9s %s" % ("error", exc)) from None

    print(f"  {'auth':<9} {auth_base}")
    print(f"  {'flow':<9} {'B (out-of-band code)' if flow == 'oob' else 'A (loopback redirect)'}")
    print(f"  {'scope':<9} {public['scope']}")
    if flow == "oob":
        print("  Authorize in your browser, then paste the code back here:")
    elif not args.no_browser:
        print("  Opening your browser to authorize...")
    print("  " + public["url"])
    if flow == "loopback" and not args.no_browser:
        try:
            webbrowser.open(public["url"])
        except Exception:
            pass  # the URL is already printed

    code = None
    if flow == "oob":
        try:
            code = input("  Code: ").strip()
        except (EOFError, KeyboardInterrupt):
            session.close()
            raise SystemExit("  %-9s cancelled" % "cancelled") from None
        if not code:
            session.close()
            raise SystemExit("  %-9s no code entered" % "cancelled")

    try:
        result = finish_connect(session, code=code,
                                store=CredentialStore(), timeout=args.timeout)
    except CredentialError as exc:
        raise SystemExit("  %-9s %s" % ("failed", exc)) from None

    print(f"  {'stored':<9} {mask_secret(result.api_key)}  "
          f"(key {result.generation}, account {result.account})")
    if result.scope_downgraded:
        print(f"  {'note':<9} granted scope is {result.scope!r}, not "
              f"{args.scope!r} — re-run with a workspace that allows it")
    print(f"  {'next':<9} needle generate-data --tools tools.json --provider orcarouter")


def _models_main(args):
    """`needle models` — the provider's real catalog, filtered by capability."""
    from .model.credentials import CredentialError, CredentialStore, mask_secret
    from .model.providers import (discover_models, filter_models, get_provider)

    provider = get_provider(args.provider or "orcarouter")
    api_key = None
    try:
        from .model.credentials import credential_for
        api_key = credential_for(provider.id, store=CredentialStore()).api_key
    except CredentialError as exc:
        print(f"  {'key':<9} {exc}")
        print(f"  {'note':<9} listing the unauthenticated catalog; "
              f"run `needle connect` for your workspace's models")

    catalog = discover_models(provider, api_key=api_key)
    modalities = [args.input_modality] if args.input_modality else None
    models = filter_models(catalog.models, capability=args.capability,
                           input_modalities=modalities)
    if args.json:
        print(json.dumps(models, indent=2))
        return
    if catalog.degraded:
        print(f"  {'catalog':<9} {catalog.source} (degraded: {catalog.error})")
    else:
        print(f"  {'catalog':<9} live  {len(catalog.models)} models from {provider.models_url}")
    if not models:
        print(f"  {'models':<9} none match capability {args.capability!r}")
        return
    for model in models:
        context = model.get("context_length")
        suffix = f"  ctx {context}" if context else ""
        architecture = model.get("architecture") or {}
        inputs = ",".join(architecture.get("input_modalities") or [])
        print(f"  {model['id']:<44} {inputs or 'undeclared':<12}{suffix}")
