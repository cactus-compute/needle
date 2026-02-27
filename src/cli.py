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
  │     --epochs INT            Training epochs (default: 1)          │
  │     --batch-size INT        Batch size (default: 32)              │
  │     --lr FLOAT              AdamW learning rate (default: 3e-4)   │
  │     --muon-lr FLOAT         Muon learning rate (default: 0.02)    │
  │     --d-model INT           Model dimension (default: 128)        │
  │     --num-heads INT         Attention heads (default: 4)          │
  │     --num-layers INT        Encoder/decoder layers (default: 2)   │
  │     --dropout FLOAT         Dropout rate (default: 0.1)           │
  │     --max-enc-len INT       Max encoder seq length (default: 128) │
  │     --max-dec-len INT       Max decoder seq length (default: 128) │
  │     --max-samples INT       Training samples (default: 20000)     │
  │     --checkpoint-dir DIR    Checkpoint directory                  │
  │     --warmup-ratio FLOAT     LR warmup ratio (default: 0.05)      │
  │     --seed INT              Random seed (default: 42)             │
  │                                                                   │
  │                                                                   │
  │   sweep                                                          │
  │     --sweep-config PATH     Sweep YAML config (default: sweep.yaml)│
  │     --project STR           W&B project name (default: needle-v1)│
  │     --count INT             Number of trials (default: 20)       │
  │     (also accepts all train flags as defaults for non-swept params)│
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
  │       --type STR            Accelerator type (default: v6e-4)     │
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


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        sys.exit(0)

    parser = argparse.ArgumentParser(prog="needle", add_help=False)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("train", add_help=False)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-enc-len", type=int, default=128)
    p.add_argument("--max-dec-len", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)

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

    p = sub.add_parser("sweep", add_help=False)
    p.add_argument("--sweep-config", type=str, default="sweep.yaml")
    p.add_argument("--project", type=str, default="needle-v1")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-enc-len", type=int, default=128)
    p.add_argument("--max-dec-len", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("evaluate", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--benchmarks", type=str, nargs="*",
                   choices=["wikitext2", "lambada", "hellaswag", "arc_easy"])
    p.add_argument("--max-samples", type=int, default=500)

    p = sub.add_parser("tpu", add_help=False)
    tpu_sub = p.add_subparsers(dest="tpu_action")

    tp = tpu_sub.add_parser("create", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--type", dest="accel_type", type=str, default="v6e-4")
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
        from .train import train
        train(args)
    elif args.command == "sweep":
        from .train import sweep
        sweep(args)
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
