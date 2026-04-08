import argparse
import os
import sys

from .data import DEFAULT_MAX_ENC_LEN, DEFAULT_MAX_DEC_LEN, DEFAULT_MAX_GEN_LEN

HELP = """Check the readme"""

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        sys.exit(0)

    parser = argparse.ArgumentParser(prog="needle", add_help=False)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("train", add_help=False)
    p.add_argument("--name", type=str, default="baseline",
                   help="Experiment name for checkpoints and wandb (default: baseline)")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-dec-layers", type=int, default=8)
    p.add_argument("--max-enc-len", type=int, default=DEFAULT_MAX_ENC_LEN)
    p.add_argument("--max-dec-len", type=int, default=DEFAULT_MAX_DEC_LEN)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--decay-ratio", type=float, default=0.05)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--max-eval-samples", type=int, default=5000)
    p.add_argument("--sparsity-ratio", type=float, default=0.0)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--precision", type=str, default="int4", choices=["int4", "int8"],
                   help="QAT precision: int4 (4-bit) or int8 (8-bit) fake quantization (default: int4)")
    p.add_argument("--prune-interval", type=int, default=100,
                   help="Steps between mask updates during gradual pruning (default: 100)")
    p.add_argument("--prune-start-frac", type=float, default=0.33,
                   help="Fraction of epoch to train before starting gradual pruning (default: 0.33)")
    p.add_argument("--prune-end-frac", type=float, default=0.67,
                   help="Fraction of epoch at which pruning finishes and mask locks (default: 0.67)")
    p.add_argument("--activation", type=str, default="swiglu", choices=["drelu", "swiglu", "geglu"])
    p.add_argument("--mat-factors", type=int, nargs="*", default=[2, 4],
                   help="Matryoshka FFN shrink factors, e.g. 2=half width (default: 2 4)")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout rate for residual connections (default: 0.0)")
    p.add_argument("--contrastive-weight", type=float, default=0.1,
                   help="Weight for CLIP-style contrastive loss (default: 0.1)")
    p.add_argument("--contrastive-dim", type=int, default=128,
                   help="Dimension of contrastive projection head (default: 128)")
    p.add_argument("--w-name", type=float, default=2.0,
                   help="Loss weight for tool name tokens (default: 2.0)")
    p.add_argument("--w-value", type=float, default=4.0,
                   help="Loss weight for argument value tokens (default: 4.0)")
    p.add_argument("--w-key", type=float, default=1.5,
                   help="Loss weight for argument key tokens (default: 1.5)")
    p.add_argument("--no-feedforward", action=argparse.BooleanOptionalAction, default=True,
                   help="Remove feedforward layers entirely (default: True)")
    p.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=False,
                   help="Run confidence head calibration after training (default: False)")
    p.add_argument("--calibrate-epochs", type=int, default=1,
                   help="Epochs for confidence head calibration (default: 1)")
    p.add_argument("--calibrate-lr", type=float, default=1e-3,
                   help="Learning rate for confidence head calibration (default: 1e-3)")
    p.add_argument("--calibrate-k", type=float, default=5.0,
                   help="Sigmoid steepness for PPL→confidence mapping (default: 5.0)")

    p = sub.add_parser("pretrain", add_help=False)
    p.add_argument("--name", type=str, default="pretrain",
                   help="Experiment name for wandb (default: pretrain)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-dec-layers", type=int, default=8)
    p.add_argument("--max-enc-len", type=int, default=DEFAULT_MAX_ENC_LEN)
    p.add_argument("--max-dec-len", type=int, default=DEFAULT_MAX_DEC_LEN)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--decay-ratio", type=float, default=0.05)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Stop after N steps (default: full epoch)")
    p.add_argument("--save-every", type=int, default=1000,
                   help="Save and upload checkpoint every N steps (default: 1000)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--activation", type=str, default="swiglu", choices=["drelu", "swiglu", "geglu"])
    p.add_argument("--no-feedforward", action=argparse.BooleanOptionalAction, default=True)

    p = sub.add_parser("tokenize", add_help=False)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit samples per split (for dev/test)")
    p.add_argument("--max-enc-len", type=int, default=DEFAULT_MAX_ENC_LEN,
                   help=f"Max encoder sequence length (default: {DEFAULT_MAX_ENC_LEN})")
    p.add_argument("--max-dec-len", type=int, default=DEFAULT_MAX_DEC_LEN,
                   help=f"Max decoder sequence length (default: {DEFAULT_MAX_DEC_LEN})")
    p.add_argument("--shuffle-tools", action=argparse.BooleanOptionalAction, default=True,
                   help="Shuffle tool order in encoder input (default: True)")
    p.add_argument("--max-tool-len", type=int, default=256,
                   help="Max token length for individual tool descriptions (default: 256)")

    p = sub.add_parser("run", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--query", type=str, default=None, help="Query text for tool-call generation")
    p.add_argument("--tools", type=str, default=None, help="Tools JSON for tool-call generation")
    p.add_argument("--audio", type=str, nargs="*", help="Audio file paths for voice-to-tool-call")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-constrained", action="store_true",
                   help="Disable grammar-constrained decoding for tool names/arg keys")

    p = sub.add_parser("eval", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-eval-samples", type=int, default=5000)
    p.add_argument("--max-enc-len", type=int, default=DEFAULT_MAX_ENC_LEN)
    p.add_argument("--max-dec-len", type=int, default=DEFAULT_MAX_DEC_LEN)
    p.add_argument("--max-gen-len", type=int, default=DEFAULT_MAX_GEN_LEN)
    p.add_argument("--tool-call-samples", type=int, default=200,
                   help="Samples for tool-call accuracy eval (default: 200)")
    p.add_argument("--throughput-runs", type=int, default=10)
    p.add_argument("--no-constrained", action="store_true",
                   help="Disable grammar-constrained decoding for tool names/arg keys")

    p = sub.add_parser("calibrate", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default=None, help="Output checkpoint path (default: overwrite input)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-samples", type=int, default=None, help="Limit training samples for PPL computation")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k", type=float, default=5.0, help="Sigmoid steepness for PPL→confidence mapping")

    p = sub.add_parser("generate-data", add_help=False)
    p.add_argument("--num-samples", type=int, default=500, help="Number of samples to generate")
    p.add_argument("--batch-size", type=int, default=25, help="Examples per Gemini call")
    p.add_argument("--workers", type=int, default=8, help="Parallel Gemini calls")
    p.add_argument("--model", type=str, default=None, help="Gemini model override")
    p.add_argument("--dry-run", action="store_true", help="Generate only, skip save and upload")
    p.add_argument("--output-jsonl", type=str, default=None, help="Also save raw generations to JSONL")
    p.add_argument("--upload-every", type=int, default=None, help="Merge+upload every N samples")

    p = sub.add_parser("merge-xlam", add_help=False)
    p.add_argument("--dry-run", action="store_true", help="Skip upload")
    p.add_argument("--max-samples", type=int, default=None, help="Limit xlam samples")

    p = sub.add_parser("translate-xlam", add_help=False)
    p.add_argument("--max-samples", type=int, default=None, help="Limit examples to translate")
    p.add_argument("--workers", type=int, default=8, help="Parallel Gemini calls")
    p.add_argument("--model", type=str, default=None, help="Gemini model for translation")
    p.add_argument("--batch-size", type=int, default=10, help="Examples per Gemini call")
    p.add_argument("--dry-run", action="store_true", help="Translate only, skip save/upload")

    p = sub.add_parser("rebalance-tools", add_help=False)
    p.add_argument("--dry-run", action="store_true", help="Preview without modifying")

    p = sub.add_parser("split-dataset", add_help=False)
    p.add_argument("--val-per-source", type=int, default=None,
                   help="Validation samples per source (default: 2500)")

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
    tp.add_argument("--preemptible", action="store_true", default=False,
                    help="Create a preemptible (spot) TPU VM")

    tp = tpu_sub.add_parser("connect", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("setup", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("sync", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("train", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)
    tp.add_argument("train_args", nargs=argparse.REMAINDER,
                    help="Extra args passed to needle train")

    tp = tpu_sub.add_parser("pretrain", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)
    tp.add_argument("train_args", nargs=argparse.REMAINDER,
                    help="Extra args passed to needle pretrain")

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
    elif args.command == "pretrain":
        import jax
        if os.path.exists("/dev/accel0"):
            jax.distributed.initialize()
        from .pretrain import pretrain
        pretrain(args)
    elif args.command == "train":
        import jax
        if os.path.exists("/dev/accel0"):
            jax.distributed.initialize()
        from .train import train
        train(args)
    elif args.command == "run":
        from .run import main as run_main
        run_main(args)
    elif args.command == "eval":
        from .eval import main as eval_main_fn
        eval_main_fn(args)
    elif args.command == "calibrate":
        from .calibrate import main as calibrate_main
        calibrate_main(args)
    elif args.command == "generate-data":
        from .generate_data import main as gendata_main, MODEL as _MODEL, UPLOAD_EVERY as _UE
        if args.model is None:
            args.model = _MODEL
        if args.upload_every is None:
            args.upload_every = _UE
        gendata_main(args)
    elif args.command == "merge-xlam":
        from .merge_xlam import main as merge_main
        merge_main(args)
    elif args.command == "translate-xlam":
        from .translate_xlam import main as translate_main, MODEL as _TMODEL
        if args.model is None:
            args.model = _TMODEL
        translate_main(args)
    elif args.command == "rebalance-tools":
        from .rebalance_tools import main as rebalance_main
        rebalance_main(args)
    elif args.command == "split-dataset":
        from .split_dataset import main as split_main
        split_main(args)
    elif args.command == "evaluate":
        from .evaluate import main as eval_main
        eval_main(args)
    elif args.command == "tpu":
        from .tpu import tpu_dispatch
        tpu_dispatch(args)
