#!/usr/bin/env python3
"""Run all TDSC module ablations and regenerate the complete 10-figure set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--reuse-ablation-cache", action="store_true",
                        help="Reuse collective_evidence_ablation.csv when it already exists.")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    scripts = Path(__file__).resolve().parent
    ablation_dir = root / "paper1/results/TDSC_MODULE_ABLATIONS"
    collective_table = ablation_dir / "collective_evidence_ablation.csv"

    ablation_command = [
        sys.executable, "-u", str(scripts / "run_tdsc_module_ablations.py"),
        "--repo-root", str(root), "--output-dir", str(ablation_dir),
    ]
    if args.reuse_ablation_cache and collective_table.is_file():
        ablation_command.append("--skip-collective")
    subprocess.run(ablation_command, cwd=root, check=True)

    figure_command = [
        sys.executable, "-u", str(scripts / "generate_tdsc_figures.py"),
        "--repo-root", str(root),
        "--output-dir", str(root / "paper1/figures/final_4dataset_TDSC"),
    ]
    subprocess.run(figure_command, cwd=root, check=True)


if __name__ == "__main__":
    main()
