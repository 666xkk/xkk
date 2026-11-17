#!/usr/bin/env python3
"""Run a frozen ORTHRUS checkpoint on an isolated calibration split only."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def validate_neighbor_offset(cur_e_id: int, training_events: int) -> None:
    if int(cur_e_id) != int(training_events):
        raise RuntimeError(
            "neighbor-loader offset mismatch: "
            f"cur_e_id={cur_e_id}, training_events={training_events}"
        )


def checkpoint_loader_path(checkpoint_dir: Path) -> Path:
    local = checkpoint_dir / "neighbor_loader.pkl"
    shared = checkpoint_dir.parent / "shared_neighbor_loader.pkl"
    path = local if local.is_file() else shared
    if not path.is_file():
        raise FileNotFoundError(
            f"neighbor loader missing; checked {local} and {shared}"
        )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CLEARSCOPE_E3")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split-name", default="calibration_day6")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output-root", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "src"))

    import torch
    from torch_geometric.data import Data

    from config import get_runtime_required_args, get_yml_cfg
    from data_utils import load_data_set, load_model
    from detection.orthrus_gnn_testing import _run_split
    from factory import build_model
    from provnet_utils import gen_nodeid2msg, get_device, init_database_connection

    runtime_args = [
        args.dataset,
        f"--seed={args.seed}",
        f"--detection.gnn_training.encoder.batch_size={args.batch_size}",
        "--detection.gnn_training.alignment.enabled=false",
        "--detection.gnn_training.alignment.uncertainty.enabled=false",
        "--detection.gnn_training.alignment.trace.enabled=false",
        "--detection.gnn_training.alignment.trace_open.enabled=false",
    ]
    runtime = get_runtime_required_args(args=runtime_args)
    cfg = get_yml_cfg(runtime)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "state_dict.pkl").is_file():
        raise FileNotFoundError(f"checkpoint state_dict is missing: {checkpoint_dir}")
    loader_path = checkpoint_loader_path(checkpoint_dir)

    edge_root = Path(cfg.edge_featurization.embed_edges._edge_embeds_dir)
    calibration_dir = edge_root / args.split_name
    if not calibration_dir.is_dir():
        raise FileNotFoundError(f"calibration embeddings are missing: {calibration_dir}")
    train_data = load_data_set(cfg, path=str(edge_root), split="train")
    calibration_data = load_data_set(cfg, path=str(edge_root), split=args.split_name)
    if not calibration_data:
        raise RuntimeError("calibration split is empty")

    training_events = sum(int(graph.src.numel()) for graph in train_data)
    saved_loader = torch.load(loader_path)
    validate_neighbor_offset(saved_loader.cur_e_id, training_events)
    max_node_num = int(saved_loader.neighbors.shape[0])

    ordered = [*train_data, *calibration_data]
    full_data = Data(
        msg=torch.cat([graph.msg for graph in ordered]),
        t=torch.cat([graph.t for graph in ordered]),
        edge_type=torch.cat([graph.edge_type for graph in ordered]),
    )
    device = get_device(cfg)
    model = build_model(
        data_sample=calibration_data[0],
        device=device,
        cfg=cfg,
        max_node_num=max_node_num,
    )
    model = load_model(model, str(checkpoint_dir))
    # load_model must restore exactly the loader whose offset was validated.
    validate_neighbor_offset(model.encoder.neighbor_loader.cur_e_id, training_events)

    cursor, connection = init_database_connection(cfg)
    try:
        nodeid2msg = {
            key: str(value) for key, value in gen_nodeid2msg(cursor).items()
        }
    finally:
        cursor.close()
        connection.close()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cfg.detection.gnn_testing._edge_losses_dir = str(output_root)
    checkpoint_name = checkpoint_dir.name
    mean_loss = _run_split(
        calibration_data,
        args.split_name,
        checkpoint_name,
        model,
        full_data,
        nodeid2msg,
        cfg,
        device,
    )
    output_dir = output_root / args.split_name / checkpoint_name
    files = sorted(output_dir.glob("*.csv"))
    if len(files) != len(calibration_data):
        raise RuntimeError(
            f"calibration loss count mismatch: graphs={len(calibration_data)} csv={len(files)}"
        )
    print(f"training events validated: {training_events}")
    print(f"calibration windows: {len(calibration_data)}")
    print(f"mean calibration event loss: {mean_loss:.8f}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
