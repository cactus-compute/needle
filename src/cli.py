import argparse
import sys

HELP = """Check the readme"""

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        sys.exit(0)

    parser = argparse.ArgumentParser(prog="needle", add_help=False)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("train", add_help=False)
    p.add_argument("--full", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-dec-layers", type=int, default=4)
    p.add_argument("--max-enc-len", type=int, default=256)
    p.add_argument("--max-dec-len", type=int, default=1024)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--sparsity-ratio", type=float, default=0.0)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--prune-interval", type=int, default=100,
                   help="Steps between mask updates during gradual pruning (default: 100)")
    p.add_argument("--prune-start-frac", type=float, default=0.33,
                   help="Fraction of epoch to train before starting gradual pruning (default: 0.33)")
    p.add_argument("--prune-end-frac", type=float, default=0.67,
                   help="Fraction of epoch at which pruning finishes and mask locks (default: 0.67)")
    p.add_argument("--activation", type=str, default="drelu", choices=["drelu", "swiglu", "geglu"])
    p.add_argument("--num-memory-slots", type=int, default=64)
    p.add_argument("--mat-factors", type=int, nargs="*", default=[2, 4, 8],
                   help="Matryoshka FFN shrink factors, e.g. 2=half width (default: 2 4 8)")
    p.add_argument("--mat-shared-input", action="store_true",
                   help="Each unique input is repeated across all mat widths (default: unique input per width)")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout rate for residual connections (default: 0.1)")
    p.add_argument("--no-speech", action="store_true", help="Disable speech training (text-only)")
    p.add_argument("--max-mel-len", type=int, default=1024,
                   help="Max mel spectrogram frames (default: 1024)")
    p.add_argument("--n-mels", type=int, default=80,
                   help="Number of mel frequency bins (default: 80)")
    p.add_argument("--max-speech-samples", type=int, default=None,
                   help="Max voice-tool-call training samples (default: all)")
    p.add_argument("--audio-aug-mode", type=str, default="white", choices=["none", "white", "person", "full"],
                   help="Speech augmentation mode for precomputed mels: none, white, person, or full (default: white)")
    p.add_argument("--white-noise-p", type=float, default=0.5,
                   help="Probability of applying mel-white-noise per sample (default: 0.5)")
    p.add_argument("--white-noise-min-snr-db", type=float, default=8.0,
                   help="Minimum white-noise SNR in dB (default: 8.0)")
    p.add_argument("--white-noise-max-snr-db", type=float, default=30.0,
                   help="Maximum white-noise SNR in dB (default: 30.0)")
    p.add_argument("--person-noise-n", type=int, default=10,
                   help="Number of background speaker clips to mix per sample (default: 10)")
    p.add_argument("--person-noise-r1", type=float, default=3.0,
                   help="Minimum distance for person noise sampling (default: 3.0)")
    p.add_argument("--person-noise-r2", type=float, default=10.0,
                   help="Maximum distance for person noise sampling (default: 10.0)")
    p.add_argument("--person-noise-r-ref", type=float, default=1.0,
                   help="Reference distance used in distance gain computation (default: 1.0)")
    p.add_argument("--person-noise-min-snr-db", type=float, default=15.0,
                   help="Minimum target SNR for person noise mixing (default: 15.0)")
    p.add_argument("--person-noise-max-snr-db", type=float, default=40.0,
                   help="Maximum target SNR for person noise mixing (default: 40.0)")
    p = sub.add_parser("tokenize", add_help=False)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit samples per split (for dev/test)")
    p.add_argument("--cleanup", action="store_true",
                   help="Delete local .data_cache/ after GCS upload")
    p.add_argument("--n-mels", type=int, default=80,
                   help="Number of mel frequency bins (default: 80)")
    p.add_argument("--max-mel-len", type=int, default=1024,
                   help="Max mel spectrogram frames (default: 1024)")
    p.add_argument("--max-enc-len", type=int, default=256,
                   help="Max encoder sequence length (default: 256)")
    p.add_argument("--max-dec-len", type=int, default=1024,
                   help="Max decoder sequence length (default: 1024)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Process in batches of this size, uploading shards to GCS incrementally")

    p = sub.add_parser("run", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--query", type=str, default=None, help="Query text for tool-call generation")
    p.add_argument("--tools", type=str, default=None, help="Tools JSON for tool-call generation")
    p.add_argument("--audio", type=str, nargs="*", help="Audio file paths for voice-to-tool-call")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("test", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-eval-samples", type=int, default=1000)
    p.add_argument("--max-enc-len", type=int, default=256)
    p.add_argument("--max-dec-len", type=int, default=1024)
    p.add_argument("--max-gen-len", type=int, default=512)
    p.add_argument("--tool-call-samples", type=int, default=200,
                   help="Samples for tool-call accuracy eval (default: 200)")
    p.add_argument("--voice-tc-samples", type=int, default=50,
                   help="Samples for voice-to-tool-call eval (default: 50)")
    p.add_argument("--throughput-runs", type=int, default=10)

    p = sub.add_parser("evaluate", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--benchmarks", type=str, nargs="*",
                   choices=["wikitext2", "lambada", "hellaswag", "arc_easy"])
    p.add_argument("--max-samples", type=int, default=500)

    p = sub.add_parser("tpu", add_help=False)
    tpu_sub = p.add_subparsers(dest="tpu_action")

    tp = tpu_sub.add_parser("create", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--type", dest="accel_type", type=str, default="v6e-8")
    tp.add_argument("--version", type=str, default=None,
                    help="Software version (auto-detected from --type if omitted)")

    tp = tpu_sub.add_parser("connect", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("claude", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("stop", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("start", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("delete", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tpu_sub.add_parser("list", add_help=False)

    args = parser.parse_args()

    if not args.command:
        print(HELP)
        sys.exit(0)

    if args.command == "tokenize":
        from .tokenize_data import tokenize
        tokenize(args)
    elif args.command == "train":
        if getattr(args, "full", False):
            args.d_model = 1536
            args.num_heads = 24
            args.num_kv_heads = 8
            args.num_layers = 12
            args.num_dec_layers = 4
            args.num_memory_slots = 128
            args.mat_factors = [2, 3, 4, 8, 16]
        from .train import train
        train(args)
    elif args.command == "run":
        from .run import main as run_main
        run_main(args)
    elif args.command == "test":
        from .test import main as test_main
        test_main(args)
    elif args.command == "evaluate":
        from .evaluate import main as eval_main
        eval_main(args)
    elif args.command == "tpu":
        from .tpu import tpu_dispatch
        tpu_dispatch(args)
