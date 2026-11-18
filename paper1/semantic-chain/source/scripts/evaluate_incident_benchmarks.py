#!/usr/bin/env python3
"""Evaluate frozen causal-window incidents against an exact random budget.

Attack labels are consumed only for post-hoc evaluation.  The selected windows
must already have been written by the label-free causal collective scorer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


SEED_RE = re.compile(r"window_scores_seed(\d+)\.csv")


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def cluster_indices(indices: list[int], max_gap: int = 2) -> list[list[int]]:
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    clusters: list[list[int]] = []
    for index in sorted(indices):
        if not clusters or index - clusters[-1][-1] > max_gap:
            clusters.append([index])
        else:
            clusters[-1].append(index)
    return clusters


def exact_random_episode_distribution(
    windows: int,
    budget: int,
    episode_sizes: list[int],
) -> dict[int, float]:
    """Distribution of episodes hit by a uniform fixed-budget window sample."""
    if windows < 1 or not 0 <= budget <= windows:
        raise ValueError("budget must be between zero and windows")
    if any(size < 1 for size in episode_sizes):
        raise ValueError("episode sizes must be positive")
    attack_windows = sum(episode_sizes)
    if attack_windows > windows:
        raise ValueError("episode sizes exceed total windows")

    # (drawn windows, hit episodes) -> number of equally likely selections.
    states: dict[tuple[int, int], int] = {(0, 0): 1}
    for size in episode_sizes:
        updated: dict[tuple[int, int], int] = defaultdict(int)
        for (drawn, hits), ways in states.items():
            for selected in range(min(size, budget - drawn) + 1):
                updated[(drawn + selected, hits + int(selected > 0))] += (
                    ways * math.comb(size, selected)
                )
        states = updated

    benign = windows - attack_windows
    counts: dict[int, int] = defaultdict(int)
    for (drawn, hits), ways in states.items():
        remaining = budget - drawn
        if 0 <= remaining <= benign:
            counts[hits] += ways * math.comb(benign, remaining)

    denominator = math.comb(windows, budget)
    return {hits: ways / denominator for hits, ways in sorted(counts.items())}


def rank_incidents(rows: list[dict], max_gap: int, top_score_count: int):
    selected = [
        int(row["window"])
        for row in rows
        if as_bool(row["selected_for_review"])
    ]
    by_window = {int(row["window"]): row for row in rows}
    incidents = []
    for cluster in cluster_indices(selected, max_gap):
        scores = sorted(
            (float(by_window[index]["causal_collective_score"]) for index in cluster),
            reverse=True,
        )
        incidents.append(
            {
                "windows": set(cluster),
                "score": mean(scores[:top_score_count]),
            }
        )
    incidents.sort(key=lambda item: -item["score"])
    return incidents


def evaluate_seed(
    dataset: str,
    seed: int,
    path: Path,
    max_gap: int,
    top_score_count: int,
) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "is_attack_window" not in rows[0]:
        raise ValueError(f"frozen attack labels missing from {path}")

    selected = {
        int(row["window"])
        for row in rows
        if as_bool(row["selected_for_review"])
    }
    attack = {
        int(row["window"])
        for row in rows
        if as_bool(row["is_attack_window"])
    }
    episodes = [set(group) for group in cluster_indices(list(attack), max_gap)]
    covered = [episode for episode in episodes if episode & selected]
    incidents = rank_incidents(rows, max_gap, top_score_count)
    attack_incident_ranks = [
        rank
        for rank, incident in enumerate(incidents, 1)
        if any(incident["windows"] & episode for episode in episodes)
    ]

    distribution = exact_random_episode_distribution(
        len(rows), len(selected), [len(episode) for episode in episodes]
    )
    observed_covered = len(covered)
    expected_covered = sum(hits * probability for hits, probability in distribution.items())
    tail_probability = sum(
        probability
        for hits, probability in distribution.items()
        if hits >= observed_covered
    )
    episode_count = len(episodes)
    return {
        "dataset": dataset,
        "seed": seed,
        "windows": len(rows),
        "review_budget": len(selected),
        "review_rate": len(selected) / len(rows),
        "attack_windows": len(attack),
        "covered_attack_windows": len(attack & selected),
        "attack_window_recall": len(attack & selected) / len(attack),
        "attack_episodes": episode_count,
        "covered_attack_episodes": observed_covered,
        "attack_episode_recall": observed_covered / episode_count,
        "incidents": len(incidents),
        "window_reduction_ratio": 1 - len(incidents) / len(selected),
        "attack_incident_ranks": ";".join(map(str, attack_incident_ranks)),
        "random_expected_attack_window_recall": len(selected) / len(rows),
        "random_expected_covered_episodes": expected_covered,
        "random_expected_episode_recall": expected_covered / episode_count,
        "random_p_episode_coverage_ge_observed": tail_probability,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    metrics = [
        "review_rate",
        "attack_window_recall",
        "attack_episode_recall",
        "incidents",
        "window_reduction_ratio",
        "random_expected_episode_recall",
        "random_p_episode_coverage_ge_observed",
    ]
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset]
        summary = {"dataset": dataset, "seeds": len(group)}
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry",
        action="append",
        nargs=2,
        metavar=("DATASET", "WINDOW_ROOT"),
        required=True,
        help="Repeat once per dataset; root contains window_scores_seedN.csv.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--top-score-count", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = []
    for dataset, root_text in args.entry:
        root = Path(root_text)
        paths = sorted(root.glob("window_scores_seed*.csv"))
        if not paths:
            raise FileNotFoundError(f"no window scores in {root}")
        for path in paths:
            match = SEED_RE.fullmatch(path.name)
            if match:
                row = evaluate_seed(
                    dataset,
                    int(match.group(1)),
                    path,
                    args.max_gap,
                    args.top_score_count,
                )
                rows.append(row)
                print(
                    f"{dataset} seed={row['seed']} "
                    f"episodes={row['covered_attack_episodes']}/"
                    f"{row['attack_episodes']} incidents={row['incidents']} "
                    f"random_p={row['random_p_episode_coverage_ge_observed']:.6g}"
                )

    summaries = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed.csv", rows)
    write_csv(args.output_dir / "dataset_summary.csv", summaries)
    with (args.output_dir / "run_config.json").open("w") as handle:
        json.dump(
            {
                "entries": args.entry,
                "max_gap": args.max_gap,
                "top_score_count": args.top_score_count,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"wrote: {args.output_dir / 'per_seed.csv'}")
    print(f"wrote: {args.output_dir / 'dataset_summary.csv'}")


if __name__ == "__main__":
    main()
