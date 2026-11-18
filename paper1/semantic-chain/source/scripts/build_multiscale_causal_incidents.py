#!/usr/bin/env python3
"""Build label-free causal incidents from a frozen union of temporal scales."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean


SEED_RE = re.compile(r"window_scores_seed(\d+)\.csv")


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank_percentiles(rows: list[dict]) -> dict[int, tuple[int, float]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["causal_collective_score"]),
            int(row["window"]),
        ),
    )
    denominator = max(len(ordered) - 1, 1)
    return {
        int(row["window"]): (rank, 1 - (rank - 1) / denominator)
        for rank, row in enumerate(ordered, 1)
    }


def cluster_windows(rows: list[dict], max_gap: int) -> list[list[dict]]:
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    clusters: list[list[dict]] = []
    for row in sorted(rows, key=lambda item: int(item["window"])):
        if (
            not clusters
            or int(row["window"]) - int(clusters[-1][-1]["window"]) > max_gap
        ):
            clusters.append([row])
        else:
            clusters[-1].append(row)
    return clusters


def validate_alignment(scale_rows: dict[str, list[dict]]) -> None:
    names = list(scale_rows)
    reference = scale_rows[names[0]]
    reference_keys = [
        (int(row["window"]), row["file"], row["start_ns"], row["end_ns"])
        for row in reference
    ]
    for name in names[1:]:
        candidate_keys = [
            (int(row["window"]), row["file"], row["start_ns"], row["end_ns"])
            for row in scale_rows[name]
        ]
        if candidate_keys != reference_keys:
            raise ValueError(f"window alignment differs for scale {name}")


def build_seed(
    seed: int,
    scale_rows: dict[str, list[dict]],
    max_gap: int,
    top_score_count: int,
    expected_selected_per_scale: int,
) -> tuple[list[dict], list[dict], dict, dict | None]:
    if len(scale_rows) < 2:
        raise ValueError("coverage mode requires at least two temporal scales")
    if top_score_count < 1:
        raise ValueError("top_score_count must be positive")
    validate_alignment(scale_rows)

    selected_by_scale: dict[str, set[int]] = {}
    ranks_by_scale = {}
    rows_by_scale = {}
    for name, rows in scale_rows.items():
        rows_by_scale[name] = {int(row["window"]): row for row in rows}
        ranks_by_scale[name] = rank_percentiles(rows)
        selected = {
            int(row["window"])
            for row in rows
            if as_bool(row["selected_for_review"])
        }
        if expected_selected_per_scale >= 0 and len(selected) != expected_selected_per_scale:
            raise ValueError(
                f"seed {seed} scale {name} selected {len(selected)} windows; "
                f"expected {expected_selected_per_scale}"
            )
        selected_by_scale[name] = selected

    union = set().union(*selected_by_scale.values())
    reference_name = next(iter(scale_rows))
    reference = rows_by_scale[reference_name]
    candidate_rows = []
    for window in sorted(union):
        selected_scales = [
            name for name, selected in selected_by_scale.items() if window in selected
        ]
        percentiles = {
            name: ranks_by_scale[name][window][1] for name in scale_rows
        }
        ranks = {name: ranks_by_scale[name][window][0] for name in scale_rows}
        base = reference[window]
        candidate = {
            "seed": seed,
            "window": window,
            "file": base["file"],
            "start_ns": int(base["start_ns"]),
            "end_ns": int(base["end_ns"]),
            "selected_scales": ";".join(selected_scales),
            "scale_support": len(selected_scales),
            "multiscale_score": max(percentiles.values()),
            "mean_rank_percentile": mean(percentiles.values()),
        }
        for name in scale_rows:
            candidate[f"{name}_rank"] = ranks[name]
            candidate[f"{name}_rank_percentile"] = percentiles[name]
            candidate[f"{name}_selected"] = window in selected_by_scale[name]
        if "is_attack_window" in base:
            candidate["is_attack_window"] = as_bool(base["is_attack_window"])
        candidate_rows.append(candidate)

    incidents = []
    for incident_id, cluster in enumerate(cluster_windows(candidate_rows, max_gap)):
        scores = sorted(
            (float(row["multiscale_score"]) for row in cluster), reverse=True
        )
        windows = [int(row["window"]) for row in cluster]
        supported_scales = sorted({
            scale
            for row in cluster
            for scale in str(row["selected_scales"]).split(";")
            if scale
        })
        incidents.append({
            "seed": seed,
            "incident_id": incident_id,
            "first_window": min(windows),
            "last_window": max(windows),
            "selected_window_count": len(windows),
            "span_windows": max(windows) - min(windows) + 1,
            "incident_score": mean(scores[:top_score_count]),
            "max_window_score": max(scores),
            "max_scale_support": max(int(row["scale_support"]) for row in cluster),
            "supported_scales": ";".join(supported_scales),
            "start_ns": min(int(row["start_ns"]) for row in cluster),
            "end_ns": max(int(row["end_ns"]) for row in cluster),
            "window_ids": ";".join(map(str, windows)),
            "window_files": ";".join(str(row["file"]) for row in cluster),
        })
    incidents.sort(
        key=lambda row: (
            -float(row["incident_score"]),
            -int(row["max_scale_support"]),
            int(row["first_window"]),
        )
    )
    for rank, incident in enumerate(incidents, 1):
        incident["incident_rank"] = rank

    all_selected = list(selected_by_scale.values())
    intersection = set.intersection(*all_selected)
    summary = {
        "seed": seed,
        "scales": ";".join(scale_rows),
        "selected_per_scale": ";".join(
            f"{name}:{len(selected_by_scale[name])}" for name in scale_rows
        ),
        "intersection_windows": len(intersection),
        "union_windows": len(union),
        "incidents": len(incidents),
        "window_reduction_ratio": 1 - len(incidents) / len(union),
    }

    evaluation = None
    if candidate_rows and "is_attack_window" in candidate_rows[0]:
        reference_rows = scale_rows[reference_name]
        attack_windows = {
            int(row["window"])
            for row in reference_rows
            if as_bool(row["is_attack_window"])
        }
        attack_groups = cluster_windows(
            [{"window": window} for window in attack_windows], max_gap
        )
        attack_episodes = [
            {int(row["window"]) for row in group} for group in attack_groups
        ]
        covered = [episode for episode in attack_episodes if episode & union]
        attack_ranks = []
        for incident in incidents:
            incident_windows = {
                int(value) for value in str(incident["window_ids"]).split(";")
            }
            if any(incident_windows & episode for episode in attack_episodes):
                attack_ranks.append(int(incident["incident_rank"]))
        evaluation = {
            "seed": seed,
            "attack_windows": len(attack_windows),
            "covered_attack_windows": len(attack_windows & union),
            "attack_window_recall": len(attack_windows & union) / len(attack_windows),
            "attack_episodes": len(attack_episodes),
            "covered_attack_episodes": len(covered),
            "attack_episode_recall": len(covered) / len(attack_episodes),
            "attack_incident_ranks": ";".join(map(str, attack_ranks)),
        }
    return candidate_rows, incidents, summary, evaluation


def discover_seeds(root: Path) -> set[int]:
    seeds = set()
    for path in root.glob("window_scores_seed*.csv"):
        match = SEED_RE.fullmatch(path.name)
        if match:
            seeds.add(int(match.group(1)))
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale-root",
        action="append",
        nargs=2,
        metavar=("NAME", "WINDOW_ROOT"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--top-score-count", type=int, default=3)
    parser.add_argument("--expected-selected-per-scale", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    roots = {name: Path(path) for name, path in args.scale_root}
    if len(roots) != len(args.scale_root):
        raise ValueError("scale names must be unique")
    seed_sets = {name: discover_seeds(root) for name, root in roots.items()}
    if not seed_sets or any(not seeds for seeds in seed_sets.values()):
        raise FileNotFoundError(f"missing scale window scores: {seed_sets}")
    if len({frozenset(seeds) for seeds in seed_sets.values()}) != 1:
        raise ValueError(f"seed sets differ across scales: {seed_sets}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    evaluations = []
    for seed in sorted(next(iter(seed_sets.values()))):
        scale_rows = {
            name: read_csv(root / f"window_scores_seed{seed}.csv")
            for name, root in roots.items()
        }
        candidates, incidents, summary, evaluation = build_seed(
            seed,
            scale_rows,
            args.max_gap,
            args.top_score_count,
            args.expected_selected_per_scale,
        )
        write_csv(args.output_dir / f"multiscale_windows_seed{seed}.csv", candidates)
        write_csv(args.output_dir / f"incident_scores_seed{seed}.csv", incidents)
        summaries.append(summary)
        if evaluation is not None:
            evaluations.append(evaluation)
        print(
            f"seed={seed} union={summary['union_windows']} "
            f"incidents={summary['incidents']} "
            f"episode_recall={evaluation['attack_episode_recall'] if evaluation else 'NA'}"
        )
    write_csv(args.output_dir / "summary.csv", summaries)
    if evaluations:
        write_csv(args.output_dir / "frozen_label_evaluation.csv", evaluations)
    with (args.output_dir / "run_config.json").open("w") as handle:
        json.dump(
            {
                "scale_roots": {name: str(root) for name, root in roots.items()},
                "max_gap": args.max_gap,
                "top_score_count": args.top_score_count,
                "expected_selected_per_scale": args.expected_selected_per_scale,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"wrote: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
