import argparse
import json

from datasets import load_dataset

from src.toucan import prepare_toucan_example


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Kimi-K2")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--samples", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    ds = load_dataset("Agent-Ark/Toucan-1.5M", args.config, split=args.split, streaming=True)
    for i, row in enumerate(ds):
        if i >= args.samples:
            break
        ex = prepare_toucan_example(row)
        print(f"ROW {i}")
        print(f"subset_name: {ex['subset_name']}")
        print(f"question: {ex['question'][:160]}")
        print(f"target_tools: {ex['target_tools']}")
        print(f"positive_indices: {ex['positive_indices']}")
        print(f"tool_names: {ex['tool_names'][:5]}")
        print(f"tools_json: {ex['tools_json'][:200]}")
        print(f"tool_text[0]: {ex['tool_texts'][0][:300]}")
        print()


if __name__ == "__main__":
    main()
