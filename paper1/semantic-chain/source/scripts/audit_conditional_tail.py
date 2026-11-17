#!/usr/bin/env python3
"""Audit label-free edge-type conformal node scores on frozen ORTHRUS losses."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


class MondrianTailCalibrator:
    """Empirical upper-tail p-values conditioned on the event type.

    The add-one correction makes the score finite-sample conservative. Rare
    event types therefore have coarser (not spuriously more extreme) evidence.
    """

    def __init__(self) -> None:
        self.by_type: dict[str, np.ndarray] = {}

    def fit(self, losses: dict[str, list[float]]) -> "MondrianTailCalibrator":
        if not losses:
            raise ValueError("calibration losses are empty")
        for event_type, values in losses.items():
            array = np.asarray(values, dtype=float)
            if array.size == 0 or not np.isfinite(array).all():
                raise ValueError(f"invalid calibration losses for {event_type}")
            self.by_type[str(event_type)] = np.sort(array)
        return self

    def p_value(self, loss: float, event_type: str) -> float:
        values = self.by_type.get(str(event_type))
        if values is None:
            # An unseen normal event family cannot safely be declared anomalous.
            return 1.0
        tail = values.size - np.searchsorted(values, float(loss), side="left")
        return float((tail + 1) / (values.size + 1))


def simes_p_value(p_values: list[float]) -> float:
    """Exposure-adjusted node p-value combining multiple incident events."""
    if not p_values:
        return 1.0
    ordered = np.sort(np.asarray(p_values, dtype=float))
    ranks = np.arange(1, ordered.size + 1, dtype=float)
    return float(min(1.0, np.min(ordered * ordered.size / ranks)))


def load_calibration(directory: Path) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no calibration CSVs in {directory}")
    for path in files:
        frame = pd.read_csv(path, usecols=["loss", "edge_type"])
        for event_type, group in frame.groupby("edge_type"):
            grouped[str(event_type)].extend(group["loss"].astype(float).tolist())
    return grouped


def load_truth(result_path: Path) -> dict[int, int]:
    result = torch.load(result_path, map_location="cpu")
    return {int(node): int(values["y_true"]) for node, values in result.items()}


def score_nodes(test_dir: Path, calibrator: MondrianTailCalibrator):
    node_p_values: dict[int, list[float]] = defaultdict(list)
    node_raw_max: dict[int, float] = defaultdict(lambda: -math.inf)
    files = sorted(test_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no test CSVs in {test_dir}")
    for path in files:
        frame = pd.read_csv(
            path,
            usecols=["loss", "srcnode", "dstnode", "edge_type"],
        )
        for row in frame.itertuples(index=False):
            loss = float(row.loss)
            p_value = calibrator.p_value(loss, str(row.edge_type))
            for node in (int(row.srcnode), int(row.dstnode)):
                node_p_values[node].append(p_value)
                node_raw_max[node] = max(node_raw_max[node], loss)
    scores = {}
    for node, p_values in node_p_values.items():
        node_p = simes_p_value(p_values)
        scores[node] = {
            "conditional_tail_score": -math.log10(max(node_p, 1e-300)),
            "node_p_value": node_p,
            "incident_events": len(p_values),
            "raw_max_loss": node_raw_max[node],
        }
    return scores


def ranking_metrics(scores: dict[int, dict], truth: dict[int, int]) -> dict:
    nodes = sorted(set(scores) & set(truth))
    labels = np.asarray([truth[node] for node in nodes], dtype=int)
    values = np.asarray(
        [scores[node]["conditional_tail_score"] for node in nodes], dtype=float
    )
    if labels.sum() == 0 or labels.sum() == labels.size:
        raise ValueError("ranking metrics require both positive and negative nodes")
    result = {
        "nodes": len(nodes),
        "positives": int(labels.sum()),
        "ap": float(average_precision_score(labels, values)),
        "auc": float(roc_auc_score(labels, values)),
    }
    ranked = sorted(
        nodes,
        key=lambda node: (
            scores[node]["conditional_tail_score"],
            -scores[node]["node_p_value"],
            -node,
        ),
        reverse=True,
    )
    for budget in (25, 50, 100, 200, 500, 1000):
        chosen = ranked[:budget]
        hits = sum(truth[node] for node in chosen)
        result[f"hits_at_{budget}"] = int(hits)
        result[f"recall_at_{budget}"] = float(hits / labels.sum())
    return result


def find_single(pattern: str, description: str) -> Path:
    matches = sorted(Path().glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {description}, found {len(matches)} for {pattern}"
        )
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in [int(value) for value in args.seeds.split(",")]:
        test_dir = Path(
            (args.bundle_root / "metadata" / f"seed{seed}_test_dir.txt")
            .read_text()
            .strip()
        )
        result_files = sorted(
            (args.bundle_root / "raw_results" / f"seed{seed}").glob(
                "result_model_epoch_*.pth"
            )
        )
        if len(result_files) != 1:
            raise RuntimeError(
                f"seed {seed}: expected one frozen result, found {len(result_files)}"
            )
        calibration_dirs = sorted(
            (args.calibration_root / f"seed{seed}" / "calibration_day6").glob(
                "model_epoch_*"
            )
        )
        if len(calibration_dirs) != 1:
            raise RuntimeError(
                f"seed {seed}: expected one calibration directory, "
                f"found {len(calibration_dirs)}"
            )
        calibrator = MondrianTailCalibrator().fit(
            load_calibration(calibration_dirs[0])
        )
        scores = score_nodes(test_dir, calibrator)
        truth = load_truth(result_files[0])
        metrics = ranking_metrics(scores, truth)
        metrics.update({"seed": seed, "test_dir": str(test_dir)})
        rows.append(metrics)

        node_path = args.output_dir / f"node_scores_seed{seed}.csv"
        with node_path.open("w", newline="") as handle:
            fieldnames = [
                "node_id", "y_true", "conditional_tail_score", "node_p_value",
                "incident_events", "raw_max_loss",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for node in sorted(scores):
                writer.writerow({"node_id": node, "y_true": truth.get(node, 0), **scores[node]})
        print(
            f"seed={seed} AP={metrics['ap']:.6f} AUC={metrics['auc']:.6f} "
            f"R@100={metrics['recall_at_100']:.6f}"
        )

    summary = pd.DataFrame(rows).sort_values("seed")
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(f"wrote: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
