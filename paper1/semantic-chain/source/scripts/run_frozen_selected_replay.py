#!/usr/bin/env python3
"""Rebuild ORTHRUS temporal state and replay val/test for a frozen weight."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


_EPOCH_RE = re.compile(r"model_epoch_(\d+)(?:_state_dict)?\.pkl$")


def checkpoint_name_from_weight(path: Path) -> str:
    match = _EPOCH_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(
            "weight filename must be model_epoch_N_state_dict.pkl or "
            f"model_epoch_N.pkl, found {path.name}"
        )
    return f"model_epoch_{int(match.group(1))}"


def validate_replayed_events(cur_e_id: int, expected: int) -> None:
    if int(cur_e_id) != int(expected):
        raise RuntimeError(
            f"replayed neighbor state mismatch: cur_e_id={cur_e_id}, "
            f"expected_training_events={expected}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CADETS_E3")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--weight", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "src"))

    import torch

    from config import get_runtime_required_args, get_yml_cfg
    from data_utils import load_all_datasets
    from detection.orthrus_gnn_testing import _run_split
    from factory import batch_loader_factory, build_model
    from provnet_utils import gen_nodeid2msg, get_device, init_database_connection

    if not args.weight.is_file():
        raise FileNotFoundError(f"selected weight is missing: {args.weight}")
    checkpoint_name = checkpoint_name_from_weight(args.weight)
    runtime_args = [
        args.dataset,
        f"--seed={args.seed}",
        f"--detection.gnn_training.encoder.batch_size={args.batch_size}",
        "--detection.gnn_training.alignment.enabled=false",
        "--detection.gnn_training.alignment.uncertainty.enabled=false",
        "--detection.gnn_training.alignment.trace.enabled=false",
        "--detection.gnn_training.alignment.trace_open.enabled=false",
        "--detection.gnn_training.alignment.ema_enabled=false",
    ]
    cfg = get_yml_cfg(get_runtime_required_args(args=runtime_args))
    device = get_device(cfg)
    train_data, val_data, test_data, full_data, max_node_num = load_all_datasets(cfg)
    model = build_model(
        data_sample=train_data[0],
        device=device,
        cfg=cfg,
        max_node_num=max_node_num,
    )
    state = torch.load(args.weight, map_location=device)
    model.load_state_dict(state)

    loader = model.encoder.neighbor_loader
    loader.reset_state()
    expected_training_events = 0
    for graph in train_data:
        graph.to(device=device)
        for batch in batch_loader_factory(cfg, graph, model.graph_reindexer):
            loader.insert(batch.src, batch.dst)
            expected_training_events += int(batch.src.numel())
        graph.to("cpu")
    validate_replayed_events(loader.cur_e_id, expected_training_events)

    cursor, connection = init_database_connection(cfg)
    try:
        nodeid2msg = {
            key: str(value) for key, value in gen_nodeid2msg(cursor).items()
        }
    finally:
        cursor.close()
        connection.close()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cfg.detection.gnn_testing._edge_losses_dir = str(output_root)
    # Validation must be replayed before test because ORTHRUS is stateful.
    validation_mean = _run_split(
        val_data,
        "val",
        checkpoint_name,
        model,
        full_data,
        nodeid2msg,
        cfg,
        device,
    )
    test_mean = _run_split(
        test_data,
        "test",
        checkpoint_name,
        model,
        full_data,
        nodeid2msg,
        cfg,
        device,
    )
    val_files = sorted((output_root / "val" / checkpoint_name).glob("*.csv"))
    test_files = sorted((output_root / "test" / checkpoint_name).glob("*.csv"))
    if len(val_files) != len(val_data) or len(test_files) != len(test_data):
        raise RuntimeError(
            "replay output count mismatch: "
            f"val={len(val_files)}/{len(val_data)}, "
            f"test={len(test_files)}/{len(test_data)}"
        )
    print(f"checkpoint: {checkpoint_name}")
    print(f"training events replayed: {expected_training_events}")
    print(f"validation windows: {len(val_files)} mean_loss={validation_mean:.8f}")
    print(f"test windows: {len(test_files)} mean_loss={test_mean:.8f}")
    print(f"test output: {output_root / 'test' / checkpoint_name}")


if __name__ == "__main__":
    main()
