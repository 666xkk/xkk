from tqdm import tqdm

from encoders import OrthrusEncoder
from provnet_utils import *
from data_utils import *
from config import *
from model import *
from factory import *
import torch
import csv
import shutil


@torch.no_grad()
def test(
        data,
        full_data,
        model,
        nodeid2msg,
        split,
        model_epoch_file,
        cfg,
        device,
):
    model.eval()

    time_with_loss = {}  # key: time，  value： the losses
    edge_list = []
    unique_nodes = torch.tensor([]).to(device=device)
    start_time = data.t[0]
    event_count = 0
    tot_loss = 0
    start = time.perf_counter()

    # NOTE: warning, this may reindexes the graph
    batch_loader = batch_loader_factory(cfg, data, model.graph_reindexer)

    for batch in batch_loader:
        unique_nodes = torch.cat([unique_nodes, batch.edge_index.flatten()]).unique()

        each_edge_loss = model(batch, full_data, inference=True)
        trace_components = (
            model.last_trace_components
            if cfg.detection.gnn_training.alignment.trace.enabled
            else None
        )
        tot_loss += each_edge_loss.sum().item()

        # If the graph has been reindexed in the loader, we retrieve original node IDs
        # to later find the labels
        if hasattr(batch, "original_edge_index"):
            edge_index = batch.original_edge_index
        else:
            edge_index = batch.edge_index
        
        num_events = each_edge_loss.shape[0]
        edge_types = torch.argmax(batch.edge_type, dim=1) + 1
        for i in range(num_events):
            srcnode = int(edge_index[0, i])
            dstnode = int(edge_index[1, i])

            srcmsg = nodeid2msg[srcnode]
            dstmsg = nodeid2msg[dstnode]
            t_var = int(batch.t[i])
            edge_type_idx = edge_types[i].item()
            edge_type = rel2id[edge_type_idx]
            loss = each_edge_loss[i]

            temp_dic = {
                'loss': float(loss),
                'srcnode': srcnode,
                'dstnode': dstnode,
                'srcmsg': srcmsg,
                'dstmsg': dstmsg,
                'edge_type': edge_type,
                'time': t_var,
            }
            if trace_components is not None:
                for key in (
                    "interaction_loss",
                    "entity_loss",
                    "temporal_loss",
                    "interaction_evidence",
                    "entity_evidence",
                    "temporal_evidence",
                    "epistemic_uncertainty",
                    "epistemic_evidence",
                    "consensus_score",
                    "prototype_divergence",
                    "prototype_evidence",
                    "virtual_unknown_probability",
                    "virtual_unknown_evidence",
                    "trace_open_score",
                ):
                    temp_dic[key] = float(trace_components[key][i])
                temp_dic["temporal_valid"] = bool(
                    trace_components["temporal_valid"][i]
                )
            edge_list.append(temp_dic)

        event_count += num_events
    raw_tot_loss = tot_loss
    tot_loss /= event_count

    # Here is a checkpoint, which records all edge losses in the current time window
    time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(edge_list[-1]["time"])

    end = time.perf_counter()
    logs_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    csv_file = os.path.join(logs_dir, time_interval + ".csv")

    df = pd.DataFrame(edge_list)
    df.to_csv(csv_file, sep=',', header=True, index=False, encoding='utf-8')

    return raw_tot_loss, event_count

    # log(
    #     f'Time: {time_interval}, Loss: {tot_loss:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s')


def _epoch_number(name):
    return int(name.rsplit("_", 1)[-1])


def _trained_model_directories(gnn_models_dir):
    return sorted(
        [
            name for name in os.listdir(gnn_models_dir)
            if name.startswith("model_epoch_")
            and os.path.isdir(os.path.join(gnn_models_dir, name))
        ],
        key=_epoch_number,
    )


def _load_checkpoint(gnn_models_dir, trained_model, data_sample, device, cfg, max_node_num):
    model = build_model(
        data_sample=data_sample,
        device=device,
        cfg=cfg,
        max_node_num=max_node_num,
    )
    return load_model(model, os.path.join(gnn_models_dir, trained_model))


def _run_split(graphs, split, trained_model, model, full_data, nodeid2msg, cfg, device):
    total_loss = 0.0
    total_events = 0
    log(f"    Testing {split} set...")
    for graph in tqdm(graphs, desc=f"{split} set with {trained_model}"):
        graph.to(device=device)
        graph_loss, graph_events = test(
            data=graph,
            full_data=full_data,
            model=model,
            nodeid2msg=nodeid2msg,
            split=split,
            model_epoch_file=trained_model,
            cfg=cfg,
            device=device,
        )
        total_loss += graph_loss
        total_events += graph_events
        graph.to("cpu")
    return total_loss / max(total_events, 1)


def _write_validation_manifest(cfg, rows, selected_model):
    path = os.path.join(
        cfg.detection.gnn_testing._edge_losses_dir,
        "validation_selection.csv",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["checkpoint", "epoch", "validation_mean_event_loss", "selected"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "selected": row["checkpoint"] == selected_model})
    return path


def select_validation_checkpoint(rows):
    """Select minimum validation loss with an earlier-epoch deterministic tie break."""
    if not rows:
        raise ValueError("validation checkpoint candidates are empty")
    return min(
        rows,
        key=lambda row: (row["validation_mean_event_loss"], row["epoch"]),
    )["checkpoint"]


def _validation_select_and_test(
        cfg, candidates, gnn_models_dir, val_data, test_data, full_data,
        max_node_num, nodeid2msg, device):
    rows = []
    for trained_model in candidates:
        log(f"Validation selection with model {trained_model}...")
        torch.cuda.empty_cache()
        model = _load_checkpoint(
            gnn_models_dir, trained_model, val_data[0], device, cfg, max_node_num
        )
        mean_loss = _run_split(
            val_data, "val", trained_model, model, full_data, nodeid2msg, cfg, device
        )
        rows.append({
            "checkpoint": trained_model,
            "epoch": _epoch_number(trained_model),
            "validation_mean_event_loss": mean_loss,
        })
        log(f"[validation-selection] {trained_model} mean_event_loss={mean_loss:.8f}")
        del model

    selected = select_validation_checkpoint(rows)
    manifest = _write_validation_manifest(cfg, rows, selected)
    log(f"[validation-selection] selected {selected}; manifest={manifest}")

    if not cfg.detection.gnn_testing.retain_validation_candidates:
        val_root = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "val")
        for row in rows:
            if row["checkpoint"] != selected:
                shutil.rmtree(os.path.join(val_root, row["checkpoint"]), ignore_errors=True)
        log("[validation-selection] removed non-selected raw validation CSVs")

    # Replay validation on a fresh selected model before testing. ORTHRUS is
    # stateful, so test must begin with the same causal history as the legacy
    # val-then-test pipeline. The selected validation CSV is simply overwritten.
    model = _load_checkpoint(
        gnn_models_dir, selected, val_data[0], device, cfg, max_node_num
    )
    _run_split(val_data, "val", selected, model, full_data, nodeid2msg, cfg, device)
    _run_split(test_data, "test", selected, model, full_data, nodeid2msg, cfg, device)
    del model


def main(cfg):
    # load the map between nodeID and node labels
    cur, _ = init_database_connection(cfg)
    nodeid2msg = gen_nodeid2msg(cur=cur)
    nodeid2msg = {k: str(v) for k, v in nodeid2msg.items()}  # pre-compute because it's too slow in main loop

    _, val_data, test_data, full_data, max_node_num = load_all_datasets(cfg)

    # For each model trained at a given epoch, we test
    gnn_models_dir = cfg.detection.gnn_training._trained_models_dir
    all_trained_models = ["model_epoch_1"] if cfg._from_weights else _trained_model_directories(gnn_models_dir)
    if not cfg._from_weights:
        policy = cfg.detection.gnn_testing.checkpoint_policy
        if policy == "ema_only":
            expected = f"model_epoch_{cfg.detection.gnn_training.num_epochs + 1}"
            if expected not in all_trained_models:
                raise FileNotFoundError(f"fixed EMA checkpoint is missing: {expected}")
            all_trained_models = [expected]
        elif policy == "last_k":
            k = cfg.detection.gnn_testing.checkpoint_last_k
            all_trained_models = all_trained_models[-k:]
        elif policy == "validation_loss":
            expected = cfg.detection.gnn_training.num_epochs
            if len(all_trained_models) != expected:
                raise RuntimeError(
                    f"validation_loss requires {expected} epoch checkpoints; "
                    f"found {len(all_trained_models)}"
                )
        elif policy != "all":
            raise ValueError(f"invalid checkpoint_policy: {policy}")

    device = get_device(cfg)

    if not cfg._from_weights and cfg.detection.gnn_testing.checkpoint_policy == "validation_loss":
        _validation_select_and_test(
            cfg=cfg,
            candidates=all_trained_models,
            gnn_models_dir=gnn_models_dir,
            val_data=val_data,
            test_data=test_data,
            full_data=full_data,
            max_node_num=max_node_num,
            nodeid2msg=nodeid2msg,
            device=device,
        )
        return

    for trained_model in all_trained_models:
        log(f"Evaluation with model {trained_model}...")
        torch.cuda.empty_cache()
        model = _load_checkpoint(
            gnn_models_dir, trained_model, test_data[0], device, cfg, max_node_num
        )
        
        if cfg._from_weights:
            model.load_state_dict(torch.load(os.path.join(cfg._from_weights_path, f"{cfg.dataset.name}.pkl")))

        # TODO: we may want to move the validation set into the training for early stopping
        for graphs, split in [
            (val_data, "val"),
            (test_data, "test"),
        ]:
            _run_split(
                graphs, split, trained_model, model, full_data, nodeid2msg, cfg, device
            )

        del model


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
