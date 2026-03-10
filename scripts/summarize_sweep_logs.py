import argparse
import csv
import re
from pathlib import Path


PATTERNS = {
    "text_loss": re.compile(r"Text loss\s+([0-9.]+)"),
    "text_val_ppl": re.compile(r"Text val ppl\s+([0-9.]+)"),
    "speech_loss": re.compile(r"Speech loss\s+([0-9.]+)"),
    "speech_val_ppl": re.compile(r"Speech val ppl\s+([0-9.]+)"),
}


def _extract_last(text, pattern):
    matches = pattern.findall(text)
    return matches[-1] if matches else ""


def _rows(log_dir):
    for log_path in sorted(Path(log_dir).glob("*.log")):
        text = log_path.read_text(errors="ignore")
        row = {"run": log_path.stem}
        for key, pattern in PATTERNS.items():
            row[key] = _extract_last(text, pattern)
        yield row


def main():
    parser = argparse.ArgumentParser(description="Extract final epoch metrics from sweep logs")
    parser.add_argument("log_dir", type=str, help="Directory containing per-run .log files")
    parser.add_argument("--csv", action="store_true", help="Print CSV instead of TSV")
    args = parser.parse_args()

    fieldnames = ["run", "text_loss", "text_val_ppl", "speech_loss", "speech_val_ppl"]
    rows = sorted(list(_rows(args.log_dir)), key=lambda x: float(x['speech_val_ppl']))

    if args.csv:
        writer = csv.DictWriter(
            open("/dev/stdout", "w", newline=""),
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)
        return

    print("\t".join(fieldnames))
    for row in rows:
        print("\t".join(row[name] for name in fieldnames))


if __name__ == "__main__":
    main()
