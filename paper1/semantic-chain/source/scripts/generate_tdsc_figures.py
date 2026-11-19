#!/usr/bin/env python3
"""Generate the complete IEEE TDSC figure set for the four-dataset study.

Run from the Orthrus repository root after all final tables have been built:

    python paper1/semantic-chain/source/scripts/generate_tdsc_figures.py

The script never changes experiment outputs.  It reads frozen CSV tables and
writes PDF/SVG/600-dpi PNG figures to a separate TDSC directory.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


DATASETS = ["CADETS_E3", "CLEARSCOPE_E3", "THEIA_E3", "OPTC_H201"]
DISPLAY = {
    "CADETS_E3": "CADETS",
    "CLEARSCOPE_E3": "ClearScope",
    "THEIA_E3": "THEIA",
    "OPTC_H201": "OPTC-H201",
}
METHODS = ["scale4", "multiscale_union"]
METHOD_LABEL = {"scale4": "Scale 4", "multiscale_union": "Multiscale union"}
COLORS = {"scale4": "#3B6FB6", "multiscale_union": "#D55E00"}
HATCHES = {"scale4": "///", "multiscale_union": "..."}
GRAY = "#555555"
LIGHT_GRAY = "#D0D0D0"


def configure_style() -> None:
    """IEEE-compatible typography with embedded TrueType fonts."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.2,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
    })


def require(path: Path, columns: tuple[str, ...] = ()) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen table not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame


def save(fig: plt.Figure, output: Path, stem: str) -> list[Path]:
    paths = []
    for extension in ("pdf", "svg"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path)
        paths.append(path)
    path = output / f"{stem}.png"
    fig.savefig(path, dpi=600)
    paths.append(path)
    plt.close(fig)
    return paths


def panel_label(ax, text: str) -> None:
    ax.text(-0.11, 1.04, text, transform=ax.transAxes, fontweight="bold",
            fontsize=9, ha="left", va="bottom")


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["dataset"] = pd.Categorical(result["dataset"], DATASETS, ordered=True)
    return result.sort_values(["dataset", "method"])


def grouped_bars(ax, frame, metric, ylabel, ylim=(0, 1.05), value_labels=True):
    x = np.arange(len(DATASETS))
    width = 0.36
    for offset, method in zip((-width / 2, width / 2), METHODS):
        part = frame[frame["method"] == method].set_index("dataset")
        values = np.asarray([part.loc[name, metric] for name in DATASETS], dtype=float)
        bars = ax.bar(x + offset, values, width, label=METHOD_LABEL[method],
                      color=COLORS[method], edgecolor="black", linewidth=0.45,
                      hatch=HATCHES[method])
        if value_labels:
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, value + 0.018,
                        f"{100*value:.0f}", ha="center", va="bottom", fontsize=6.6)
    ax.set_xticks(x, [DISPLAY[name] for name in DATASETS])
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y")


def load_comparison(root: Path) -> pd.DataFrame:
    path = root / "paper1/results/FINAL_4DATASET_TABLES/scale4_vs_multiscale.csv"
    columns = (
        "dataset", "method", "attack_window_recall", "attack_episode_recall",
        "selected_windows", "incidents", "window_reduction_ratio",
        "random_expected_episode_recall", "random_p",
    )
    return ordered(require(path, columns))


def figure_dataset_recall(root: Path, output: Path) -> list[Path]:
    data = load_comparison(root)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), constrained_layout=True)
    grouped_bars(axes[0], data, "attack_window_recall", "Attack-window recall")
    grouped_bars(axes[1], data, "attack_episode_recall", "Attack-episode recall")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    return save(fig, output, "fig1_dataset_recall")


def figure_observed_random(root: Path, output: Path) -> list[Path]:
    data = load_comparison(root)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), constrained_layout=True)
    x = np.arange(len(DATASETS))
    for ax, method, tag in zip(axes, METHODS, ("(a)", "(b)")):
        part = data[data["method"] == method].set_index("dataset")
        observed = np.asarray([part.loc[d, "attack_episode_recall"] for d in DATASETS])
        random = np.asarray([part.loc[d, "random_expected_episode_recall"] for d in DATASETS])
        ax.plot(x, observed, color=COLORS[method], marker="o", label="Observed")
        ax.plot(x, random, color=GRAY, marker="s", linestyle="--", label="Random expectation")
        ax.fill_between(x, random, observed, color=COLORS[method], alpha=0.10)
        ax.set_xticks(x, [DISPLAY[d] for d in DATASETS], rotation=15, ha="right")
        ax.set_ylim(0, 1.07)
        ax.set_ylabel("Attack-episode recall")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y")
        ax.set_title(METHOD_LABEL[method], pad=3)
        panel_label(ax, tag)
    axes[0].legend(frameon=False, loc="lower left")
    return save(fig, output, "fig2_observed_vs_random")


def figure_budget_compression(root: Path, output: Path) -> list[Path]:
    data = load_comparison(root)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), constrained_layout=True)
    x = np.arange(len(DATASETS))
    width = 0.36
    for offset, method in zip((-width/2, width/2), METHODS):
        part = data[data["method"] == method].set_index("dataset")
        selected = np.asarray([part.loc[d, "selected_windows"] for d in DATASETS])
        incidents = np.asarray([part.loc[d, "incidents"] for d in DATASETS])
        axes[0].bar(x + offset, selected, width, color="white", edgecolor=COLORS[method],
                    linewidth=1.0, hatch=HATCHES[method])
        axes[0].bar(x + offset, incidents, width, color=COLORS[method], edgecolor="black",
                    linewidth=0.4, label=METHOD_LABEL[method])
        reduction = np.asarray([part.loc[d, "window_reduction_ratio"] for d in DATASETS])
        axes[1].bar(x + offset, reduction, width, color=COLORS[method], edgecolor="black",
                    linewidth=0.45, hatch=HATCHES[method], label=METHOD_LABEL[method])
    axes[0].set_ylabel("Mean count across five seeds")
    axes[0].set_xticks(x, [DISPLAY[d] for d in DATASETS])
    axes[0].grid(axis="y")
    axes[0].text(0.98, 0.96, "outline: selected windows\nfill: resulting incidents",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=6.8, color=GRAY)
    axes[1].set_ylabel("Window-to-incident reduction")
    axes[1].set_xticks(x, [DISPLAY[d] for d in DATASETS])
    axes[1].set_ylim(0, 0.8)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[1].grid(axis="y")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    return save(fig, output, "fig3_budget_and_compression")


def load_cross_seed(root: Path) -> pd.DataFrame:
    sources = {
        "scale4": root / "paper1/results/CAUSAL_INCIDENT_BENCHMARK_4DATASETS_SCALE4/cross_dataset_summary.csv",
        "multiscale_union": root / "paper1/results/MULTISCALE_COVERAGE_FROZEN_4DATASETS/multiscale_random_cross_dataset.csv",
    }
    frames = []
    for method, path in sources.items():
        frame = require(path, ("seed", "attack_episode_recall", "covered_attack_windows",
                               "attack_windows", "incidents", "window_reduction_ratio"))
        frame = frame.copy()
        frame["method"] = method
        frame["attack_window_recall"] = frame["covered_attack_windows"] / frame["attack_windows"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def figure_seed_stability(root: Path, output: Path) -> list[Path]:
    data = load_cross_seed(root)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45), constrained_layout=True)
    for method in METHODS:
        part = data[data["method"] == method].sort_values("seed")
        axes[0].plot(part["seed"], part["attack_window_recall"], marker="o",
                     color=COLORS[method], label=METHOD_LABEL[method])
        axes[1].plot(part["seed"], part["incidents"], marker="o",
                     color=COLORS[method], label=METHOD_LABEL[method])
    axes[0].set_ylabel("Cross-dataset window recall")
    axes[0].set_ylim(0, 0.82)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[1].set_ylabel("Cross-dataset incidents")
    for ax in axes:
        ax.set_xlabel("Random seed")
        ax.set_xticks(range(5))
        ax.grid(axis="y")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    axes[0].legend(frameon=False, ncol=2, loc="lower left")
    return save(fig, output, "fig4_seed_stability")


def load_optc_scales(root: Path) -> pd.DataFrame:
    base = root / "paper1/results/OPTC_H201/VALIDATION_SELECTED_BASELINE"
    frames = []
    for scale in (1, 4):
        path = base / f"causal_collective_windows_scale{scale}/summary.csv"
        frame = require(path, ("seed", "attack_windows", "covered_attack_windows"))
        frame = frame.copy()
        frame["scale"] = f"Scale {scale}"
        frame["recall"] = frame["covered_attack_windows"] / frame["attack_windows"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def figure_optc_scale_gain(root: Path, output: Path) -> list[Path]:
    data = load_optc_scales(root)
    wide = data.pivot(index="seed", columns="scale", values="recall").sort_index()
    fig, ax = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)
    for seed, row in wide.iterrows():
        ax.plot([0, 1], [row["Scale 1"], row["Scale 4"]], color=LIGHT_GRAY, linewidth=1.0)
        ax.scatter([0], [row["Scale 1"]], color=COLORS["scale4"], zorder=3)
        ax.scatter([1], [row["Scale 4"]], color=COLORS["multiscale_union"], zorder=3)
        ax.text(1.04, row["Scale 4"], f"s{int(seed)}", fontsize=6.5, va="center")
    ax.plot([0, 1], [wide["Scale 1"].mean(), wide["Scale 4"].mean()],
            color="black", marker="D", linewidth=1.6, label="Five-seed mean")
    ax.set_xlim(-0.25, 1.28)
    ax.set_xticks([0, 1], ["Scale 1", "Scale 4"])
    ax.set_ylabel("OPTC attack-window recall")
    ax.set_ylim(0, max(0.42, wide.to_numpy().max() + 0.08))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    return save(fig, output, "fig5_optc_scale_gain")


def figure_optc_node_signal(root: Path, output: Path) -> list[Path]:
    path = root / "paper1/results/OPTC_H201/VALIDATION_SELECTED_BASELINE/tables/five_seed_ranking.csv"
    data = require(path, ("seed", "ap", "auc", "mcc", "recall_at_100"))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45), constrained_layout=True)
    axes[0].errorbar(["AP", "AUC", "MCC"],
                     [data["ap"].mean(), data["auc"].mean(), data["mcc"].mean()],
                     yerr=[data["ap"].std(ddof=1), data["auc"].std(ddof=1), data["mcc"].std(ddof=1)],
                     fmt="o", color=COLORS["scale4"], capsize=3, linewidth=1.0)
    axes[0].set_ylabel(r"Metric value (mean $\pm$ SD)")
    axes[0].set_ylim(0, 0.82)
    axes[0].grid(axis="y")
    x = np.arange(len(data))
    axes[1].bar(x, 100 * data["recall_at_100"], color=COLORS["multiscale_union"],
                edgecolor="black", linewidth=0.45, hatch="...")
    axes[1].set_xticks(x, [f"s{int(seed)}" for seed in data["seed"]])
    axes[1].set_xlabel("Random seed")
    axes[1].set_ylabel("Node recall@100 (%)")
    axes[1].grid(axis="y")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    return save(fig, output, "fig6_optc_node_signal")


def figure_collective_ablation(root: Path, output: Path) -> list[Path]:
    path = root / "paper1/results/TDSC_MODULE_ABLATIONS/collective_evidence_ablation.csv"
    data = require(path, ("dataset", "seed", "method", "attack_window_recall", "attack_episode_recall"))
    methods = ["top1", "top5", "top10", "all_mean"]
    labels = ["Top-1", "Top-5 mean", "Top-10 mean (ours)", "All-node mean"]
    colors = ["#8DA0CB", "#66C2A5", "#D55E00", "#B3B3B3"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6), constrained_layout=True)
    x = np.arange(len(DATASETS))
    width = 0.19
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        part = data[data["method"] == method].groupby("dataset", observed=True)
        for ax, metric in zip(axes, ("attack_window_recall", "attack_episode_recall")):
            means = np.asarray([part[metric].mean().get(dataset, np.nan) for dataset in DATASETS])
            errors = np.asarray([part[metric].std(ddof=1).get(dataset, np.nan) for dataset in DATASETS])
            ax.bar(x + (index - 1.5) * width, means, width, yerr=errors,
                   capsize=2, color=color, edgecolor="black", linewidth=0.4, label=label)
    for ax, ylabel, tag in zip(axes, ("Attack-window recall", "Attack-episode recall"), ("(a)", "(b)")):
        ax.set_xticks(x, [DISPLAY[d] for d in DATASETS], rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y")
        panel_label(ax, tag)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    return save(fig, output, "fig7_collective_evidence_ablation")


def figure_multiscale_ablation(root: Path, output: Path) -> list[Path]:
    path = root / "paper1/results/TDSC_MODULE_ABLATIONS/multiscale_fusion_ablation.csv"
    data = require(path, ("dataset", "seed", "method", "attack_window_recall",
                          "attack_episode_recall", "incidents"))
    methods = ["scale1", "scale4", "max_rank_top20", "mean_rank_top20", "union"]
    labels = ["Scale 1", "Scale 4", "Max-rank Top-20", "Mean-rank Top-20", "Union (ours)"]
    colors = ["#8DA0CB", "#66C2A5", "#E6AB02", "#A6761D", "#D55E00"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), constrained_layout=True)
    x = np.arange(len(methods))
    for dataset, marker in zip(DATASETS, ("o", "s", "^", "D")):
        part = data[data["dataset"] == dataset].groupby("method", observed=True)
        for ax, metric in zip(axes, ("attack_window_recall", "attack_episode_recall", "incidents")):
            values = [part[metric].mean().get(method, np.nan) for method in methods]
            ax.plot(x, values, marker=marker, label=DISPLAY[dataset], linewidth=1.0)
    for ax in axes[:2]:
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("Attack-window recall")
    axes[1].set_ylabel("Attack-episode recall")
    axes[2].set_ylabel("Mean incident count")
    for ax, tag in zip(axes, ("(a)", "(b)", "(c)")):
        ax.set_xticks(x, ["S1", "S4", "Max", "Mean", "Union"], rotation=20)
        ax.grid(axis="y")
        panel_label(ax, tag)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    return save(fig, output, "fig8_multiscale_fusion_ablation")


def figure_incident_sensitivity(root: Path, output: Path) -> list[Path]:
    path = root / "paper1/results/TDSC_MODULE_ABLATIONS/incident_sensitivity.csv"
    data = require(path, ("dataset", "seed", "parameter", "value", "attack_episode_recall",
                          "incidents", "normalized_best_attack_rank"))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), constrained_layout=True)
    specifications = [
        ("budget", "Attack-episode recall", "attack_episode_recall", "Per-scale window budget"),
        ("max_gap", "Mean incident count", "incidents", "Maximum causal gap"),
        ("top_m", "Normalized best attack rank", "normalized_best_attack_rank", "Top-$M$ windows per incident"),
    ]
    for ax, (parameter, ylabel, metric, xlabel), tag in zip(axes, specifications, ("(a)", "(b)", "(c)")):
        subset = data[data["parameter"] == parameter]
        for dataset, marker in zip(DATASETS, ("o", "s", "^", "D")):
            part = subset[subset["dataset"] == dataset].groupby("value", observed=True)[metric]
            values = part.mean().sort_index()
            errors = part.std(ddof=1).reindex(values.index)
            ax.errorbar(values.index, values.values, yerr=errors.values,
                        marker=marker, capsize=2, linewidth=1.0, label=DISPLAY[dataset])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        panel_label(ax, tag)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].set_ylim(0, 1.08)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].legend(frameon=False, fontsize=6.3, loc="lower right")
    return save(fig, output, "fig9_incident_reconstruction_sensitivity")


def generate_overview(script_dir: Path, output: Path) -> list[Path]:
    module_path = script_dir / "generate_method_overview.py"
    spec = importlib.util.spec_from_file_location("generate_method_overview", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(output)
    return [output / f"fig0_method_overview.{ext}" for ext in ("pdf", "svg", "png")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path,
                        default=Path("paper1/figures/final_4dataset_TDSC"))
    parser.add_argument("--skip-overview", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output = args.output_dir
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    configure_style()

    generated: list[Path] = []
    if not args.skip_overview:
        generated += generate_overview(Path(__file__).resolve().parent, output)
    generated += figure_dataset_recall(root, output)
    generated += figure_observed_random(root, output)
    generated += figure_budget_compression(root, output)
    generated += figure_seed_stability(root, output)
    generated += figure_optc_scale_gain(root, output)
    generated += figure_optc_node_signal(root, output)
    generated += figure_collective_ablation(root, output)
    generated += figure_multiscale_ablation(root, output)
    generated += figure_incident_sensitivity(root, output)

    manifest = output / "FIGURE_MANIFEST.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figure", "format", "path"])
        for path in generated:
            writer.writerow([path.stem, path.suffix.lstrip("."), path.relative_to(root)])

    print("generated:")
    for path in generated:
        print(" ", path)
    print(" ", manifest)


if __name__ == "__main__":
    main()
