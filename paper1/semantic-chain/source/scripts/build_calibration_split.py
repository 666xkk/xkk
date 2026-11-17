#!/usr/bin/env python3
"""Build an isolated edge-embedding split from already constructed graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def ensure_isolated_output(output_dir: Path, standard_split_dirs: list[Path]) -> None:
    resolved = output_dir.resolve()
    if any(resolved == path.resolve() for path in standard_split_dirs):
        raise ValueError("calibration output must not overwrite train/val/test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CLEARSCOPE_E3")
    parser.add_argument("--graph-name", default="graph_6")
    parser.add_argument("--split-name", default="calibration_day6")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "src"))

    from config import get_runtime_required_args, get_yml_cfg
    from edge_featurization.embed_edges_feature_word2vec import (
        gen_relation_onehot,
        gen_vectorized_graphs,
        get_indexid2vec,
    )
    from provnet_utils import (
        gen_nodeid2msg,
        get_logger,
        init_database_connection,
        ntype2id,
        rel2id,
    )

    runtime = get_runtime_required_args(args=[args.dataset, f"--seed={args.seed}"])
    cfg = get_yml_cfg(runtime)
    edge_root = Path(cfg.edge_featurization.embed_edges._edge_embeds_dir)
    output_dir = edge_root / args.split_name
    ensure_isolated_output(
        output_dir,
        [edge_root / name for name in ("train", "val", "test")],
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"non-empty calibration output already exists: {output_dir}; "
            "pass --overwrite only after checking it"
        )

    graph_dir = Path(cfg.graph_construction.build_graphs._graphs_dir) / args.graph_name
    if not graph_dir.is_dir():
        raise FileNotFoundError(f"constructed calibration graph is missing: {graph_dir}")
    graph_files = sorted(path for path in graph_dir.iterdir() if path.is_file())
    if not graph_files:
        raise FileNotFoundError(f"no graph windows in {graph_dir}")

    logger = get_logger(
        name="build_isolated_calibration_embeddings",
        filename=str(Path(cfg.edge_featurization.embed_edges._logs_dir) / "calibration_embed.log"),
    )
    use_types = cfg.edge_featurization.embed_nodes.feature_word2vec.use_node_types
    use_cmd = cfg.edge_featurization.embed_nodes.feature_word2vec.use_cmd
    use_port = cfg.edge_featurization.embed_nodes.feature_word2vec.use_port
    decline = cfg.edge_featurization.embed_nodes.feature_word2vec.decline_rate

    cursor, connection = init_database_connection(cfg)
    try:
        indexid2msg = gen_nodeid2msg(cursor, use_cmd=use_cmd, use_port=use_port)
    finally:
        cursor.close()
        connection.close()
    indexid2msg = dict(sorted(indexid2msg.items(), key=lambda item: int(item[0])))
    model_path = (
        Path(cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir)
        / "feature_word2vec.model"
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"existing node Word2Vec model is missing: {model_path}")
    indexid2vec = get_indexid2vec(
        indexid2msg=indexid2msg,
        model_path=str(model_path),
        use_node_types=use_types,
        decline_percentage=decline,
    )
    gen_vectorized_graphs(
        indexid2vec=indexid2vec,
        etype2oh=gen_relation_onehot(rel2id),
        ntype2oh=gen_relation_onehot(ntype2id),
        split_files=[args.graph_name],
        out_dir=str(output_dir),
        logger=logger,
        cfg=cfg,
    )
    generated = sorted(output_dir.glob("*.TemporalData.simple"))
    if len(generated) != len(graph_files):
        raise RuntimeError(
            f"calibration embedding count mismatch: graphs={len(graph_files)} "
            f"embeddings={len(generated)}"
        )
    print(f"calibration embeddings: {len(generated)}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
