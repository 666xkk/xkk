"""Evaluate already-frozen semantic-chain window scores against attack times."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from .semantic_chain import evaluate_frozen_windows, parse_interval


def read_frozen_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            converted: Dict[str, object] = {
                **row,
                "window_start_ns": int(row["window_start_ns"]),
                "window_end_ns": int(row["window_end_ns"]),
            }
            for key in (
                "selected_for_review",
                "selected_by_event_count",
                "selected_by_actor_count",
            ):
                if key in row:
                    converted[key] = row[key].lower() == "true"
            rows.append(converted)
    if not rows:
        raise ValueError(f"no window rows in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-scores", required=True)
    parser.add_argument("--attack-interval", action="append", required=True)
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_frozen_rows(Path(args.window_scores))
    intervals = [parse_interval(value, args.timezone) for value in args.attack_interval]
    result: Dict[str, object] = {
        "window_scores": args.window_scores,
        "fixed_selected_windows": sum(bool(row["selected_for_review"]) for row in rows),
        "semantic_chain": evaluate_frozen_windows(rows, intervals),
    }
    if "selected_by_event_count" in rows[0]:
        result["event_count_baseline"] = evaluate_frozen_windows(
            rows, intervals, "selected_by_event_count"
        )
    if "selected_by_actor_count" in rows[0]:
        result["actor_count_baseline"] = evaluate_frozen_windows(
            rows, intervals, "selected_by_actor_count"
        )
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
