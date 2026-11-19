#!/usr/bin/env python3
"""Run frozen, label-free module ablations used by the TDSC figure set.

The expensive edge-loss CSVs are scanned once per dataset/seed.  All method
scores and fixed-budget selections are computed before the frozen attack-window
flags are consulted for post-hoc evaluation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_ROOTS = {
    "CLEARSCOPE_E3": "paper1/results/CLEARSCOPE_E3/VALIDATION_SELECTED_BASELINE",
    "THEIA_E3": "paper1/results/THEIA_E3/VALIDATION_SELECTED_BASELINE",
    "CADETS_E3": "paper1/results/CADETS_E3/FROZEN_SELECTED_REPLAY",
    "OPTC_H201": "paper1/results/OPTC_H201/VALIDATION_SELECTED_BASELINE",
}
SCALE_NAMES = {
    "CLEARSCOPE_E3": ("causal_collective_windows_scale1_frozen_eval", "causal_collective_windows_scale4"),
    "THEIA_E3": ("causal_collective_windows_scale1_frozen_eval", "causal_collective_windows_scale4"),
    "CADETS_E3": ("causal_collective_windows_scale1_frozen_eval", "causal_collective_windows_scale4_frozen_eval"),
    "OPTC_H201": ("causal_collective_windows_scale1", "causal_collective_windows_scale4"),
}


def truth(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def rolling(values, scale):
    history, result, total = deque(maxlen=scale), [], 0.0
    for value in values:
        if len(history) == scale:
            total -= history[0]
        history.append(float(value))
        total += float(value)
        result.append(total / len(history))
    return np.asarray(result)


def top_indices(values, budget):
    return set(sorted(range(len(values)), key=lambda i: (-float(values[i]), i))[:budget])


def clusters(indices, max_gap=2):
    output = []
    for index in sorted(indices):
        if not output or index - output[-1][-1] > max_gap:
            output.append([index])
        else:
            output[-1].append(index)
    return output


def episode_metrics(selected, attack, incident_groups=None, canonical_gap=2, scores=None, top_m=3):
    attack_groups = [set(group) for group in clusters(attack, canonical_gap)]
    covered = sum(bool(group & selected) for group in attack_groups)
    result = {
        "attack_windows": len(attack),
        "covered_attack_windows": len(selected & attack),
        "attack_window_recall": len(selected & attack) / len(attack) if attack else 0.0,
        "attack_episodes": len(attack_groups),
        "covered_attack_episodes": covered,
        "attack_episode_recall": covered / len(attack_groups) if attack_groups else 0.0,
    }
    if incident_groups is None:
        incident_groups = clusters(selected, 2)
    result["incidents"] = len(incident_groups)
    result["window_reduction_ratio"] = 1 - len(incident_groups) / len(selected) if selected else 0.0
    if scores is not None and incident_groups:
        ranked = sorted(
            incident_groups,
            key=lambda group: (-np.mean(sorted((scores[i] for i in group), reverse=True)[:top_m]), group[0]),
        )
        ranks = [rank for rank, group in enumerate(ranked, 1) if set(group) & attack]
        result["best_attack_incident_rank"] = min(ranks) if ranks else np.nan
        result["normalized_best_attack_rank"] = min(ranks) / len(ranked) if ranks else np.nan
    return result


def rank_percentiles(values):
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))
    denominator = max(len(order) - 1, 1)
    result = np.zeros(len(values))
    for rank, index in enumerate(order, 1):
        result[index] = 1 - (rank - 1) / denominator
    return result


def load_scales(root, dataset, seed):
    base = root / DATASET_ROOTS[dataset]
    names = SCALE_NAMES[dataset]
    frames = [pd.read_csv(base / name / f"window_scores_seed{seed}.csv") for name in names]
    key = ["window", "file", "start_ns", "end_ns"]
    if not frames[0][key].equals(frames[1][key]):
        raise ValueError(f"unaligned scales: {dataset} seed={seed}")
    return frames


def infer_test_dirs(scale4_root):
    config_path = scale4_root / "run_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        metadata = Path(config.get("metadata_root", ""))
        if metadata.is_dir():
            found = {}
            for path in metadata.glob("seed*_test_dir.txt"):
                match = re.fullmatch(r"seed(\d+)_test_dir\.txt", path.name)
                if match:
                    found[int(match.group(1))] = Path(path.read_text().strip())
            if found:
                return found
    summary = pd.read_csv(scale4_root / "summary.csv")
    return {int(row.seed): Path(row.test_dir) for row in summary.itertuples()}


def node_values(path, role="file"):
    maxima = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            loss = float(row["loss"])
            for side in ("src", "dst"):
                if f"'{role}':" not in str(row[f"{side}msg"]) and f'"{role}":' not in str(row[f"{side}msg"]):
                    continue
                node = int(row[f"{side}node"])
                if loss > maxima.get(node, -np.inf):
                    maxima[node] = loss
    return sorted(maxima.values(), reverse=True)


def collective_ablation(root, dataset, output):
    base = root / DATASET_ROOTS[dataset]
    scale4_name = SCALE_NAMES[dataset][1]
    scale4_root = base / scale4_name
    test_dirs = infer_test_dirs(scale4_root)
    rows = []
    cache_dir = output / "raw_collective_cache" / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(5):
        reference = pd.read_csv(scale4_root / f"window_scores_seed{seed}.csv")
        attack = set(reference.loc[reference["is_attack_window"].map(truth), "window"].astype(int))
        cache_path = cache_dir / f"seed{seed}.csv"
        if cache_path.is_file():
            raw = pd.read_csv(cache_path)
        else:
            records = []
            paths = sorted(test_dirs[seed].glob("*.csv"))
            if len(paths) != len(reference):
                raise ValueError(f"window count mismatch: {dataset} seed={seed}")
            for window, path in enumerate(paths):
                values = node_values(path)
                records.append({
                    "window": window,
                    "top1": np.mean(values[:1]) if values else 0.0,
                    "top5": np.mean(values[:5]) if values else 0.0,
                    "top10": np.mean(values[:10]) if values else 0.0,
                    "all_mean": np.mean(values) if values else 0.0,
                })
            raw = pd.DataFrame(records)
            raw.to_csv(cache_path, index=False)
        for method in ("top1", "top5", "top10", "all_mean"):
            score = rolling(raw[method], 4)
            selected = top_indices(score, 20)
            metric = episode_metrics(selected, attack, scores=score)
            rows.append({"dataset": dataset, "seed": seed, "method": method, **metric})
        print(f"collective {dataset} seed={seed} done")
    return rows


def multiscale_ablation(root, dataset):
    rows = []
    for seed in range(5):
        scale1, scale4 = load_scales(root, dataset, seed)
        attack = set(scale1.loc[scale1["is_attack_window"].map(truth), "window"].astype(int))
        scores1 = scale1["causal_collective_score"].to_numpy(float)
        scores4 = scale4["causal_collective_score"].to_numpy(float)
        p1, p4 = rank_percentiles(scores1), rank_percentiles(scores4)
        strategies = {
            "scale1": (top_indices(scores1, 20), scores1),
            "scale4": (top_indices(scores4, 20), scores4),
            "union": (top_indices(scores1, 20) | top_indices(scores4, 20), np.maximum(p1, p4)),
            "max_rank_top20": (top_indices(np.maximum(p1, p4), 20), np.maximum(p1, p4)),
            "mean_rank_top20": (top_indices((p1 + p4) / 2, 20), (p1 + p4) / 2),
        }
        for method, (selected, score) in strategies.items():
            metric = episode_metrics(selected, attack, scores=score)
            rows.append({"dataset": dataset, "seed": seed, "method": method, **metric})
    return rows


def incident_sensitivity(root, dataset):
    rows = []
    for seed in range(5):
        scale1, scale4 = load_scales(root, dataset, seed)
        attack = set(scale1.loc[scale1["is_attack_window"].map(truth), "window"].astype(int))
        score1 = scale1["causal_collective_score"].to_numpy(float)
        score4 = scale4["causal_collective_score"].to_numpy(float)
        fusion = np.maximum(rank_percentiles(score1), rank_percentiles(score4))
        for budget in (10, 20, 30):
            selected = top_indices(score1, budget) | top_indices(score4, budget)
            metric = episode_metrics(selected, attack, scores=fusion)
            rows.append({"dataset": dataset, "seed": seed, "parameter": "budget", "value": budget, **metric})
        selected = top_indices(score1, 20) | top_indices(score4, 20)
        for gap in (1, 2, 3, 4):
            groups = clusters(selected, gap)
            metric = episode_metrics(selected, attack, incident_groups=groups, scores=fusion)
            rows.append({"dataset": dataset, "seed": seed, "parameter": "max_gap", "value": gap, **metric})
        groups = clusters(selected, 2)
        for top_m in (1, 3, 5):
            metric = episode_metrics(selected, attack, incident_groups=groups, scores=fusion, top_m=top_m)
            rows.append({"dataset": dataset, "seed": seed, "parameter": "top_m", "value": top_m, **metric})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path,
                        default=Path("paper1/results/TDSC_MODULE_ABLATIONS"))
    parser.add_argument("--skip-collective", action="store_true",
                        help="Skip the expensive raw edge-loss scan.")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    collective, multiscale, sensitivity = [], [], []
    for dataset in DATASET_ROOTS:
        if not args.skip_collective:
            collective.extend(collective_ablation(root, dataset, output))
        multiscale.extend(multiscale_ablation(root, dataset))
        sensitivity.extend(incident_sensitivity(root, dataset))
    if collective:
        pd.DataFrame(collective).to_csv(output / "collective_evidence_ablation.csv", index=False)
    pd.DataFrame(multiscale).to_csv(output / "multiscale_fusion_ablation.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output / "incident_sensitivity.csv", index=False)
    print("wrote:", output)


if __name__ == "__main__":
    main()
