#!/usr/bin/env python3
"""Summarize fixed-EMA protocol logs without selecting on test labels."""

import argparse
import csv
import re
from pathlib import Path


PATTERN = re.compile(
    r"\[protocol\] FINAL \((?P<protocol>[^)]+)\): "
    r"epoch=(?P<epoch>\d+) mcc=(?P<mcc>[-+0-9.eE]+)"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datr_summary.csv"))
    args = parser.parse_args()
    rows = []
    for path in sorted(args.log_dir.glob("*.log")):
        matches = list(PATTERN.finditer(path.read_text(errors="replace")))
        if not matches:
            continue
        match = matches[-1]
        stem = path.stem
        method_match = re.search(r"_(baseline|datr)_s(\d+)$", stem)
        rows.append({
            "method": method_match.group(1) if method_match else "unknown",
            "seed": int(method_match.group(2)) if method_match else -1,
            "protocol": match.group("protocol"),
            "epoch": int(match.group("epoch")),
            "mcc": float(match.group("mcc")),
            "log": str(path),
        })
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "method", "seed", "protocol", "epoch", "mcc", "log"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
