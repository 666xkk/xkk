import os
import wandb
import numpy as np
import csv
import pandas as pd
from collections import defaultdict
from pprint import pprint

from . import node_evaluation
from data_utils import *
from provnet_utils import log
from .evaluation_utils import *


def standard_evaluation(cfg, evaluation_fn):
    test_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "test")
    val_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "val")
    
    tw_to_malicious_nodes = compute_tw_labels(cfg)
    
    best_mcc, best_stats = -1e6, {}
    all_stats = []
    for model_epoch_dir in listdir_sorted(test_losses_dir):
        log(f"\nEvaluation of model {model_epoch_dir}...")

        test_tw_path = os.path.join(test_losses_dir, model_epoch_dir)
        val_tw_path = os.path.join(val_losses_dir, model_epoch_dir)

        stats = evaluation_fn(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes=tw_to_malicious_nodes)
            
        out_dir = cfg.detection.evaluation.node_evaluation._precision_recall_dir
        stats["epoch"] = int(model_epoch_dir.split("_")[-1])
        stats["simple_scores_img"] = wandb.Image(os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png"))
        
        scores = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
        if os.path.exists(scores):
            stats["scores_img"] = wandb.Image(scores)
        
        dor = os.path.join(out_dir, f"dor_{model_epoch_dir}.png")
        if os.path.exists(dor):
            stats["dor_img"] = wandb.Image(dor)
        
        pr = os.path.join(out_dir, f"pr_curve_{model_epoch_dir}.png")
        if os.path.exists(pr):
            stats["precision_recall_img"] = wandb.Image(pr)
        
        wandb.log(stats)
        all_stats.append(stats)
        
        if stats["mcc"] > best_mcc:
            best_mcc = stats["mcc"]
            best_stats = stats
        
    selection = cfg.detection.evaluation.checkpoint_selection
    if selection == "test_mcc_oracle":
        selected = best_stats
        label = "test_mcc_oracle"
    elif selection == "validation_loss":
        manifest = os.path.join(
            cfg.detection.gnn_testing._edge_losses_dir,
            "validation_selection.csv",
        )
        frame = pd.read_csv(manifest)
        selected_mask = frame["selected"].map(
            lambda value: str(value).strip().lower() in ("true", "1")
        )
        chosen = frame.loc[selected_mask]
        if len(chosen) != 1:
            raise RuntimeError(
                f"validation selection manifest must contain one selected row; found {len(chosen)}"
            )
        selected_epoch = int(chosen.iloc[0]["epoch"])
        matches = [s for s in all_stats if s["epoch"] == selected_epoch]
        if len(matches) != 1:
            raise RuntimeError(
                f"validation-selected epoch {selected_epoch} has {len(matches)} test results"
            )
        selected = matches[0]
        label = f"validation_loss:model_epoch_{selected_epoch}"
        selected["validation_mean_event_loss"] = float(
            chosen.iloc[0]["validation_mean_event_loss"]
        )
    elif selection in ("fixed_ema", "fixed_epoch"):
        fixed_epoch = (
            cfg.detection.gnn_training.num_epochs + 1
            if selection == "fixed_ema"
            else cfg.detection.evaluation.fixed_epoch
        )
        matches = [s for s in all_stats if s["epoch"] == fixed_epoch]
        if len(matches) != 1:
            raise RuntimeError(
                f"selection={selection} requires exactly one epoch {fixed_epoch}; "
                f"found {len(matches)}"
            )
        selected = matches[0]
        label = f"{selection}:model_epoch_{fixed_epoch}"
    else:
        raise ValueError(f"invalid checkpoint_selection: {selection}")

    scalar_stats = {
        key: value for key, value in selected.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    scalar_stats["selection_protocol"] = label
    out_dir = cfg.detection.evaluation._evaluation_results_dir
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "selected_checkpoint.csv")
    with open(out_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_stats.keys()))
        writer.writeheader()
        writer.writerow(scalar_stats)
    log(
        f"[protocol] FINAL ({label}): epoch={selected['epoch']} "
        f"mcc={selected.get('mcc', float('nan')):.4f}"
    )
    wandb.log({f"selected/{k}": v for k, v in scalar_stats.items()})


def main(cfg):
    method = cfg.detection.evaluation.used_method.strip()
    if method == "node_evaluation":
        standard_evaluation(cfg, evaluation_fn=node_evaluation.main)
    else:
        raise ValueError(f"Invalid evaluation method {cfg.detection.evaluation.used_method}")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
