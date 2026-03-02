import argparse
import sys

HELP = """
  ┌───────────────────────────────────────────────────────────────────┐
  │                                                                   │
  │      ┌─┐┌─┐┌─┐┌┬┐┬ ┬┌─┐  ┌┐┌┌─┐┌─┐┌┬┐┬  ┌─┐                       │
  │      │  ├─┤│   │ │ │└─┐  │││├┤ ├┤  │││  ├┤                        │
  │      └─┘┴ ┴└─┘ ┴ └─┘└─┘  ┘└┘└─┘└─┘─┴┘┴─┘└─┘                       │
  │      ...the tiny model to rule them all...                        │
  │                                                                   │
  │   train                                                           │
  │     --toy                   Use toy config for quick iteration    │
  │     --epochs INT            Training epochs (default: 1)          │
  │     --batch-size INT        Batch size (default: 32)              │
  │     --lr FLOAT              AdamW learning rate (default: 3e-4)   │
  │     --muon-lr FLOAT         Muon learning rate (default: 0.02)    │
  │     --d-model INT           Model dim (default: max of mrl-dims)  │
  │     --num-heads INT         Attention heads (default: 4)          │
  │     --num-layers INT        Encoder layers (default: 12)          │
  │     --num-dec-layers INT    Decoder layers (default: 4)           │
  │     --max-enc-len INT       Max encoder seq length (default: 256) │
  │     --max-dec-len INT       Max decoder seq length (default: 256) │
  │     --max-samples INT       Training samples (default: 1000000)   │
  │     --mrl-dims INT [...]    MRL dim targets (default: 512 256 ..) │
  │     --sparsity-ratio FLOAT  Block prune ratio (default: 0.33)     │
  │     --layer-prune-ratio FL  Layer prune ratio (default: 0.0)      │
  │     --group-size INT        Quant/prune group size (default: 32)  │
  │     --prune-interval INT    Steps between mask updates (def: 100) │
  │     --prune-start-frac FL   Start pruning at this frac (def: 0.33)│
  │     --prune-end-frac FL     Lock mask at this frac (def: 0.67)    │
  │     --activation STR        drelu|swiglu|geglu (default: drelu)   │
  │     --warmup-ratio FLOAT    LR warmup ratio (default: 0.05)       │
  │     --eval-every INT        Val eval interval (default: 1000)     │
  │     --wandb                 Enable W&B logging                    │
  │     --checkpoint PATH       Resume from checkpoint                │
  │     --checkpoint-dir DIR    Checkpoint directory                  │
  │     --seed INT              Random seed (default: 42)             │
  │                                                                   │
  │   run                                                             │
  │     --checkpoint PATH       Path to model checkpoint (required)   │
  │     --prompts STR [...]     One or more prompts to continue       │
  │     --max-len INT           Max tokens to generate (default: 128) │
  │     --temperature FLOAT     Sampling temperature (default: 0.8)   │
  │     --seed INT              Random seed (default: 0)              │
  │                                                                   │
  │   test                                                            │
  │     --checkpoint PATH       Path to model checkpoint (required)   │
  │     --batch-size INT        Batch size (default: 32)              │
  │     --max-eval-samples INT  Evaluation samples (default: 1000)    │
  │     --max-gen-len INT       Max generation length (default: 128)  │
  │     --temperature FLOAT     Sampling temperature (default: 0.8)   │
  │     --throughput-runs INT   Throughput runs (default: 10)         │
  │                                                                   │
  │   evaluate                                                        │
  │     --checkpoint PATH       Path to model checkpoint (required)   │
  │     --benchmarks [...]      wikitext2 lambada hellaswag arc_easy  │
  │     --max-samples INT       Samples per benchmark (default: 500)  │
  │                                                                   │
  │   tpu                                                             │
  │     create NAME             Create TPU (auto-finds zone)          │
  │       --type STR            Accelerator (default: v6e-8)          │
  │       --version STR         TPU OS (auto-detected from --type)    │
  │     connect NAME            SSH config + connect (auto-zone)      │
  │     claude NAME             Install Claude Code on instance       │
  │     stop NAME               Stop instance (auto-zone)             │
  │     start NAME              Start stopped instance (auto-zone)    │
  │     delete NAME             Delete instance (auto-zone)           │
  │     list                    List all TPU instances                │
  │       --zone ZONE           Override auto-detected zone           │
  │                                                                   │
  └───────────────────────────────────────────────────────────────────┘
"""

TOY_CONFIG = {
    "num_heads": 4,
    "num_layers": 2,
    "num_dec_layers": 2,
    "max_enc_len": 128,
    "max_dec_len": 128,
    "max_samples": 10000,
}

MAIN_CONFIG = {
    "num_heads": 4,
    "num_layers": 4,
    "num_dec_layers": 2,
    "max_enc_len": 256,
    "max_dec_len": 256,
    "max_samples": None,
}


def _apply_train_defaults(args):
    config = TOY_CONFIG if args.toy else MAIN_CONFIG
    for key, value in config.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if getattr(args, "d_model", None) is None:
        args.d_model = max(args.mrl_dims)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        sys.exit(0)

    parser = argparse.ArgumentParser(prog="needle", add_help=False)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("train", add_help=False)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-dec-layers", type=int, default=None)
    p.add_argument("--max-enc-len", type=int, default=None)
    p.add_argument("--max-dec-len", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--sparsity-ratio", type=float, default=0.5)
    p.add_argument("--layer-prune-ratio", type=float, default=0.0)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--prune-interval", type=int, default=100,
                   help="Steps between mask updates during gradual pruning (default: 100)")
    p.add_argument("--prune-start-frac", type=float, default=0.33,
                   help="Fraction of epoch to train before starting gradual pruning (default: 0.33)")
    p.add_argument("--prune-end-frac", type=float, default=0.67,
                   help="Fraction of epoch at which pruning finishes and mask locks (default: 0.67)")
    p.add_argument("--activation", type=str, default="drelu", choices=["drelu", "swiglu", "geglu"])
    p.add_argument("--num-memory-slots", type=int, default=64)
    p.add_argument("--mrl-dims", type=int, nargs="*", default=[1024, 512, 256, 128, 64],
                   help="MRL dimension pruning targets (default: 1024 512 256 128 64)")

    p = sub.add_parser("run", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompts", type=str, nargs="*")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("test", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-eval-samples", type=int, default=1000)
    p.add_argument("--max-enc-len", type=int, default=128)
    p.add_argument("--max-dec-len", type=int, default=128)
    p.add_argument("--max-gen-len", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
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

    if args.command == "train":
        _apply_train_defaults(args)
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
