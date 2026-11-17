#!/usr/bin/env python3
"""Merge frozen collective windows into ranked, label-free causal incidents."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd


_SEED_RE = re.compile(r"window_scores_seed(\d+)\.csv")


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def cluster_records(records: list[dict], max_gap: int = 2) -> list[list[dict]]:
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    ordered = sorted(records, key=lambda row: int(row["window"]))
    clusters: list[list[dict]] = []
    current: list[dict] = []
    for row in ordered:
        if (
            not current
            or int(row["window"]) - int(current[-1]["window"]) <= max_gap
        ):
            current.append(row)
        else:
            clusters.append(current)
            current = [row]
    if current:
        clusters.append(current)
    return clusters


def score_clusters(
    records: list[dict],
    max_gap: int = 2,
    top_score_count: int = 3,
) -> list[dict]:
    if top_score_count < 1:
        raise ValueError("top_score_count must be positive")
    selected = [row for row in records if as_bool(row["selected_for_review"])]
    output = []
    for incident_id, cluster in enumerate(cluster_records(selected, max_gap)):
        scores = sorted(
            (float(row["causal_collective_score"]) for row in cluster),
            reverse=True,
        )
        windows = [int(row["window"]) for row in cluster]
        output.append({
            "incident_id": incident_id,
            "first_window": min(windows),
            "last_window": max(windows),
            "selected_window_count": len(windows),
            "span_windows": max(windows) - min(windows) + 1,
            "incident_score": sum(scores[:top_score_count])
            / min(top_score_count, len(scores)),
            "max_window_score": max(scores),
            "mean_window_score": sum(scores) / len(scores),
            "start_ns": min(int(row["start_ns"]) for row in cluster),
            "end_ns": max(int(row["end_ns"]) for row in cluster),
            "first_file": cluster[0]["file"],
            "last_file": cluster[-1]["file"],
            "window_ids": ";".join(map(str, windows)),
            "window_files": ";".join(str(row["file"]) for row in cluster),
        })
    output.sort(key=lambda row: (-row["incident_score"], row["first_window"]))
    for rank, row in enumerate(output, 1):
        row["incident_rank"] = rank
    return output


def evaluate_incidents(records: list[dict], incidents: list[dict], max_gap: int):
    if not records or "is_attack_window" not in records[0]:
        return None
    attack_rows = [row for row in records if as_bool(row["is_attack_window"])]
    attack_episodes = cluster_records(attack_rows, max_gap)
    attack_sets = [
        {int(row["window"]) for row in cluster} for cluster in attack_episodes
    ]
    covered = set()
    ranks = []
    for incident in incidents:
        incident_windows = {
            int(value) for value in str(incident["window_ids"]).split(";")
        }
        matches = {
            index
            for index, attack_windows in enumerate(attack_sets)
            if incident_windows & attack_windows
        }
        if matches:
            covered.update(matches)
            ranks.append(int(incident["incident_rank"]))
    return {
        "total_attack_episodes": len(attack_sets),
        "covered_attack_episodes": len(covered),
        "attack_episode_recall": (
            len(covered) / len(attack_sets) if attack_sets else 0.0
        ),
        "attack_incident_ranks": ";".join(map(str, ranks)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--top-score-count", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    evaluation_rows = []
    paths = sorted(args.window_root.glob("window_scores_seed*.csv"))
    if not paths:
        raise FileNotFoundError(f"no frozen window scores in {args.window_root}")
    for path in paths:
        match = _SEED_RE.fullmatch(path.name)
        if match is None:
            continue
        seed = int(match.group(1))
        records = pd.read_csv(path).to_dict("records")
        incidents = score_clusters(
            records,
            max_gap=args.max_gap,
            top_score_count=args.top_score_count,
        )
        for row in incidents:
            row["seed"] = seed
        write_csv(args.output_dir / f"incident_scores_seed{seed}.csv", incidents)
        selected_count = sum(as_bool(row["selected_for_review"]) for row in records)
        summary_rows.append({
            "seed": seed,
            "selected_windows": selected_count,
            "incidents": len(incidents),
            "window_reduction_ratio": (
                1 - len(incidents) / selected_count if selected_count else 0.0
            ),
        })
        evaluation = evaluate_incidents(records, incidents, args.max_gap)
        if evaluation is not None:
            evaluation_rows.append({"seed": seed, **evaluation})
        print(
            f"seed={seed} selected={selected_count} incidents={len(incidents)}"
        )
    write_csv(args.output_dir / "summary.csv", summary_rows)
    if evaluation_rows:
        write_csv(args.output_dir / "frozen_label_evaluation.csv", evaluation_rows)
    with (args.output_dir / "run_config.json").open("w") as handle:
        json.dump(
            {
                "window_root": str(args.window_root),
                "max_gap": args.max_gap,
                "top_score_count": args.top_score_count,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"wrote: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
