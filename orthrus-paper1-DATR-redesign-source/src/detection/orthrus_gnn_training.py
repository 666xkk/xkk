import logging
from time import perf_counter as timer

import torch.nn as nn
import wandb

from encoders import OrthrusEncoder
from config import *
from data_utils import *
from factory import *
from objectives import ModelEMA, effective_number_weights


def train(data,
          full_data,
          model,
          optimizer,
          cfg,
          ema=None,
          update_ema=False,
          ):
    model.train()

    losses = []
    batch_loader = batch_loader_factory(cfg, data, model.graph_reindexer)

    for batch in batch_loader:
        optimizer.zero_grad()

        loss = model(batch, full_data)

        loss.backward()
        optimizer.step()
        if ema is not None and update_ema:
            ema.update(model)
        losses.append(loss.item())
    return np.mean(losses)


def main(cfg):
    gnn_models_dir = cfg.detection.gnn_training._trained_models_dir
    os.makedirs(gnn_models_dir, exist_ok=True)
    device = get_device(cfg)
    
    # Reset the peak memory usage counter
    if device == torch.device("cuda"):
        torch.cuda.reset_peak_memory_stats(device=device)

    train_data, _, _, full_data, max_node_num = load_all_datasets(cfg)

    model = build_model(data_sample=train_data[0], device=device, cfg=cfg, max_node_num=max_node_num)

    alignment = cfg.detection.gnn_training.alignment
    enabled_methods = sum([
        bool(alignment.enabled),
        bool(alignment.uncertainty.enabled),
        bool(alignment.trace.enabled),
    ])
    if enabled_methods > 1:
        raise ValueError(
            "DATR, ECHO, and TRACE are separate hypotheses; enable only one "
            "so the experiment remains identifiable"
        )
    if alignment.uncertainty.enabled:
        log(
            "[echo-config] "
            f"heads={alignment.uncertainty.num_heads}, "
            f"bootstrap_keep={alignment.uncertainty.bootstrap_keep_probability}, "
            f"disagreement_weight={alignment.uncertainty.disagreement_weight}"
        )
    if alignment.trace.enabled:
        log(
            "[trace-config] "
            f"heads={alignment.trace.num_heads}, "
            f"bootstrap_keep={alignment.trace.bootstrap_keep_probability}, "
            f"entity_weight={alignment.trace.entity_weight}, "
            f"temporal_weight={alignment.trace.temporal_weight}, "
            f"stats_start_epoch={alignment.trace.stats_start_epoch}"
        )
    if alignment.enabled:
        counts = torch.zeros(cfg.dataset.num_edge_types, dtype=torch.long)
        for graph in train_data:
            labels = graph.edge_type.argmax(dim=1).cpu()
            counts += torch.bincount(labels, minlength=counts.numel())
        class_weights = effective_number_weights(
            counts,
            beta=alignment.class_balance_beta,
            max_weight=alignment.max_class_weight,
        )
        model.training_objective.set_class_weights(class_weights.to(device))
        log(f"[aligned-objective] class counts: {counts.tolist()}")
        log(f"[aligned-objective] class weights: {class_weights.tolist()}")

    optimizer = optimizer_factory(cfg, parameters=set(model.parameters()))
    ema = ModelEMA(model, decay=alignment.ema_decay) if alignment.ema_enabled else None

    num_epochs = 1 if cfg._from_weights else cfg.detection.gnn_training.num_epochs
    tot_loss = 0.0
    epoch_times = []
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        start = timer()

        # Before each epoch, we reset the memory
        if isinstance(model.encoder, OrthrusEncoder):
            model.encoder.reset_state()
        if alignment.trace.enabled:
            model.reset_trace_temporal_state()
            if epoch == alignment.trace.stats_start_epoch:
                model.reset_trace_statistics()
            model.trace_update_statistics = epoch >= alignment.trace.stats_start_epoch

        tot_loss = 0
        for g in train_data:
            g.to(device=device)
            loss = train(
                data=g,  # avoids alteration of the graph across epochs
                full_data=full_data,  # full list of edge messages (do not store on CPU)
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                ema=ema,
                update_ema=epoch >= alignment.ema_start_epoch,
            )
            tot_loss += loss
            # log(f"Loss {loss:4f}")
            g.to("cpu")

        tot_loss /= len(train_data)
        log(f'GNN training loss Epoch: {epoch:02d}, Loss: {tot_loss:.4f}')
        if alignment.enabled:
            stats = model.training_objective.last_stats
            log(
                f"[aligned-objective] Epoch {epoch:02d}, "
                f"mean={stats.get('mean', float('nan')):.4f}, "
                f"tail={stats.get('tail', float('nan')):.4f}"
            )
        if alignment.uncertainty.enabled:
            stats = model.last_uncertainty_stats
            log(
                f"[echo-uncertainty] Epoch {epoch:02d}, "
                f"mean_loss={stats.get('mean_loss', float('nan')):.4f}, "
                f"mean_disagreement={stats.get('mean_disagreement', float('nan')):.4f}, "
                f"max_disagreement={stats.get('max_disagreement', float('nan')):.4f}"
            )
        if alignment.trace.enabled:
            stats = model.last_trace_stats
            log(
                f"[trace-evidence] Epoch {epoch:02d}, "
                f"interaction={stats.get('interaction', float('nan')):.4f}, "
                f"entity={stats.get('entity', float('nan')):.4f}, "
                f"temporal={stats.get('temporal', float('nan')):.4f}, "
                f"uncertainty={stats.get('uncertainty', float('nan')):.4f}, "
                f"temporal_coverage={stats.get('valid_temporal_fraction', float('nan')):.4f}"
            )
        
        epoch_times.append(timer() - start)
        
        # Log peak CUDA memory usage
        peak_memory = (
            torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
            if device.type == "cuda" else 0.0
        )
        log(f'Peak CUDA memory usage Epoch {epoch}: {peak_memory:.2f} GB')
        
        wandb.log({
            "train_epoch": epoch,
            "train_loss": round(tot_loss, 4),
            "peak_cuda_memory_GB": round(peak_memory, 2),
        })

        # Checkpoint storage is an experimental choice, not an implicit side
        # effect. The primary protocol saves only the final EMA model and its
        # single neighbor-loader snapshot.
        save_policy = alignment.checkpoint_save_policy
        save_regular = (
            save_policy == "all"
            or (
                save_policy == "last_k"
                and epoch > num_epochs - alignment.checkpoint_save_last_k
            )
        )
        if save_policy not in ("all", "last_k", "ema_only"):
            raise ValueError(f"invalid checkpoint_save_policy: {save_policy}")
        if cfg._test_mode or save_regular:
            model_path = os.path.join(gnn_models_dir, f"model_epoch_{epoch}")
            save_model(model, model_path)

    # A pre-specified, label-free final checkpoint. It is deliberately stored
    # as epoch N+1 so the existing testing/evaluation pipeline can consume it.
    if ema is not None and ema.num_updates > 0 and alignment.save_ema_checkpoint:
        ema.apply(model)
        ema_path = os.path.join(gnn_models_dir, f"model_epoch_{num_epochs + 1}")
        save_model(model, ema_path)
        ema.restore(model)
        log(
            f"[checkpoint] saved fixed EMA checkpoint model_epoch_{num_epochs + 1} "
            f"after {ema.num_updates} updates"
        )
            
    wandb.log({
        "train_epoch_time": round(np.mean(epoch_times), 2),
    })


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
