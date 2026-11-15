#!/usr/bin/env python3
"""Audit whether TRACE uncertainty detects attacks or merely benign cold start."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def csv_files(directory):
    return sorted(Path(directory).glob("*.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--result-pth", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    validation_nodes = set()
    for path in csv_files(args.val_dir):
        frame = pd.read_csv(path, usecols=["srcnode", "dstnode"])
        validation_nodes.update(frame["srcnode"].astype(int))
        validation_nodes.update(frame["dstnode"].astype(int))

    results = torch.load(args.result_pth, map_location="cpu")
    rows = []
    for raw_node, record in results.items():
        node = int(raw_node)
        if record.get("y_true", 0):
            stratum = "malicious"
        elif node in validation_nodes:
            stratum = "benign_recurring"
        else:
            stratum = "benign_cold_start"
        rows.append({
            "node": node,
            "stratum": stratum,
            "score": float(record["score"]),
            "uncertainty": float(record.get("epistemic_evidence", np.nan)),
            "route": record.get("trace_route", "unavailable"),
            "y_hat": int(record.get("y_hat", 0)),
        })

    frame = pd.DataFrame(rows)
    summary = []
    for stratum, group in frame.groupby("stratum", sort=True):
        summary.append({
            "stratum": stratum,
            "n": int(len(group)),
            "score_median": float(group["score"].median()),
            "score_q95": float(group["score"].quantile(0.95)),
            "uncertainty_median": float(group["uncertainty"].median()),
            "uncertainty_q95": float(group["uncertainty"].quantile(0.95)),
            "review_rate": float((group["route"] == "review").mean()),
            "alert_rate": float(group["y_hat"].mean()),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output.with_suffix(".nodes.csv"), index=False)
    pd.DataFrame(summary).to_csv(output.with_suffix(".summary.csv"), index=False)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
