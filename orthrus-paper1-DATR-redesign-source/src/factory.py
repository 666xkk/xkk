import torch
import torch.nn as nn

from provnet_utils import *
from config import *
from model import *
from encoders import *
from decoders import *
from data_utils import *
from temporal import LastAggregator, LastNeighborLoader
from objectives import DetectionAlignedObjective
from trace_views import EntityCrossViewDecoder, TemporalGapDecoder
from open_set import OpenSetExposureHead, RelationalPrototypeBank


def build_model(data_sample, device, cfg, max_node_num):
    """
    Builds and loads the initial model into memory.
    The `data_sample` is required to infer the shape of the layers.
    """
    msg_dim, edge_dim, in_dim = get_dimensions_from_data_sample(data_sample)

    graph_reindexer = GraphReindexer(
        num_nodes=max_node_num,
        device=device,
    )
    
    encoder = encoder_factory(cfg, msg_dim=msg_dim, in_dim=in_dim, edge_dim=edge_dim, graph_reindexer=graph_reindexer, device=device, max_node_num=max_node_num)
    decoder = decoder_factory(cfg, in_dim=in_dim, device=device, max_node_num=max_node_num)
    model = model_factory(encoder, decoder, cfg, in_dim=in_dim, graph_reindexer=graph_reindexer, device=device, max_node_num=max_node_num)
    
    return model

def model_factory(encoder, decoders, cfg, in_dim, graph_reindexer, device, max_node_num):
    aligned = cfg.detection.gnn_training.alignment
    trace = aligned.trace
    trace_open = aligned.trace_open
    return Orthrus(
        encoder=encoder,
        decoders=decoders,
        num_nodes=max_node_num,
        device=device,
        in_dim=in_dim,
        out_dim=cfg.detection.gnn_training.node_out_dim,
        use_contrastive_learning="predict_edge_contrastive" in cfg.detection.gnn_training.decoder.used_methods,
        graph_reindexer=graph_reindexer,
        training_objective=DetectionAlignedObjective(
            enabled=aligned.enabled,
            tail_fraction=aligned.tail_fraction,
            tail_weight=aligned.tail_weight,
        ),
        uncertainty_enabled=aligned.uncertainty.enabled,
        bootstrap_keep_probability=(
            trace.bootstrap_keep_probability
            if trace.enabled
            else aligned.uncertainty.bootstrap_keep_probability
        ),
        disagreement_weight=aligned.uncertainty.disagreement_weight,
        trace_enabled=trace.enabled,
        trace_entity_decoder=(
            EntityCrossViewDecoder(
                hidden_dim=cfg.detection.gnn_training.node_out_dim,
                feature_dim=in_dim,
            ) if trace.enabled else None
        ),
        trace_temporal_decoder=(
            TemporalGapDecoder(cfg.detection.gnn_training.node_out_dim)
            if trace.enabled else None
        ),
        trace_entity_weight=trace.entity_weight,
        trace_temporal_weight=trace.temporal_weight,
        trace_min_scale=trace.min_scale,
        trace_open_enabled=trace_open.enabled,
        trace_open_head=(
            OpenSetExposureHead(
                cfg.detection.gnn_training.node_out_dim,
                cfg.dataset.num_edge_types,
            ) if trace_open.enabled else None
        ),
        trace_open_prototypes=(
            RelationalPrototypeBank(
                cfg.dataset.num_edge_types,
                momentum=trace_open.prototype_momentum,
            ) if trace_open.enabled else None
        ),
        trace_open_exposure_weight=trace_open.exposure_weight,
        trace_open_num_node_types=cfg.dataset.num_node_types,
    ).to(device)

def encoder_factory(cfg, msg_dim, in_dim, edge_dim, graph_reindexer, device, max_node_num):
    node_hid_dim = cfg.detection.gnn_training.node_hid_dim
    node_out_dim = cfg.detection.gnn_training.node_out_dim
    temporal_dim = cfg.detection.gnn_training.encoder.temporal_dim
    
    # If edge features are used, we set them here
    edge_dim = 0
    edge_features = list(map(lambda x: x.strip(), cfg.detection.gnn_training.encoder.edge_features.split(",")))
    for edge_feat in edge_features:
        if edge_feat == "edge_type":
            edge_dim += cfg.dataset.num_edge_types
        elif edge_feat == "msg":
            edge_dim += msg_dim
        elif edge_feat == "none":
            pass
        else:
            raise ValueError(f"Invalid edge feature {edge_feat}")

    original_in_dim = in_dim
    in_dim = temporal_dim
    
    encoder = GraphTransformer(
        in_dim=in_dim,
        hid_dim=node_hid_dim,
        out_dim=node_out_dim,
        edge_dim=edge_dim or None,
        activation=activation_fn_factory(cfg.detection.gnn_training.encoder.graph_attention.activation),
        dropout=cfg.detection.gnn_training.encoder.graph_attention.dropout,
        num_heads=cfg.detection.gnn_training.encoder.graph_attention.num_heads,
    )
    neighbor_size = cfg.detection.gnn_training.encoder.neighbor_size
    use_node_feats_in_gnn = cfg.detection.gnn_training.encoder.use_node_feats_in_gnn

    neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)

    encoder = OrthrusEncoder(
        encoder=encoder,
        neighbor_loader=neighbor_loader,
        in_dim=original_in_dim,
        temporal_dim=temporal_dim,
        use_node_feats_in_gnn=use_node_feats_in_gnn,
        graph_reindexer=graph_reindexer,
        edge_features=edge_features,
        device=device,
        num_nodes=max_node_num,
        edge_dim=edge_dim,
    )

    return encoder

def decoder_factory(cfg, in_dim, device, max_node_num):
    node_out_dim = cfg.detection.gnn_training.node_out_dim

    decoders = []
    for method in map(lambda x: x.strip(), cfg.detection.gnn_training.decoder.used_methods.split(",")):
        if method == "predict_edge_type":
            def cross_entropy(x, y, inference=False, **kwargs):
                # Always retain per-event losses. The model performs the final
                # reduction so baseline mean risk and aligned tail risk share
                # exactly the same decoder and anomaly score.
                return F.cross_entropy(x, y, reduction="none")

            loss_fn = cross_entropy
            activation = activation_fn_factory(cfg.detection.gnn_training.decoder.predict_edge_type.custom.activation)
            
            decoder = EdgeTypeDecoder(
                in_dim=node_out_dim,
                num_edge_types=cfg.dataset.num_edge_types,
                loss_fn=loss_fn,
                dropout=cfg.detection.gnn_training.decoder.predict_edge_type.custom.dropout,
                num_layers=cfg.detection.gnn_training.decoder.predict_edge_type.custom.num_layers,
                activation=activation,
            )
            alignment = cfg.detection.gnn_training.alignment
            ensemble_enabled = alignment.uncertainty.enabled or alignment.trace.enabled
            num_heads = (
                alignment.trace.num_heads
                if alignment.trace.enabled
                else alignment.uncertainty.num_heads
                if alignment.uncertainty.enabled
                else 1
            )
            if num_heads < 2 and ensemble_enabled:
                raise ValueError("uncertainty/TRACE num_heads must be at least 2")
            uncertainty = cfg.detection.gnn_training.alignment.uncertainty
            if uncertainty.enabled and not 0.0 < uncertainty.bootstrap_keep_probability <= 1.0:
                raise ValueError("uncertainty.bootstrap_keep_probability must be in (0, 1]")
            if uncertainty.enabled and uncertainty.disagreement_weight < 0.0:
                raise ValueError("uncertainty.disagreement_weight must be non-negative")
            trace = alignment.trace
            if trace.enabled and not 0.0 < trace.bootstrap_keep_probability <= 1.0:
                raise ValueError("TRACE bootstrap_keep_probability must be in (0, 1]")
            if trace.enabled and (trace.entity_weight < 0.0 or trace.temporal_weight < 0.0):
                raise ValueError("TRACE view weights must be non-negative")
            trace_open = alignment.trace_open
            if trace_open.enabled and trace_open.exposure_weight < 0.0:
                raise ValueError("TRACE-Open exposure_weight must be non-negative")
            if trace_open.enabled and not 0.0 <= trace_open.prototype_momentum < 1.0:
                raise ValueError("TRACE-Open prototype_momentum must be in [0, 1)")
            decoders.extend([decoder])
            for _ in range(1, num_heads):
                decoders.append(
                    EdgeTypeDecoder(
                        in_dim=node_out_dim,
                        num_edge_types=cfg.dataset.num_edge_types,
                        loss_fn=loss_fn,
                        dropout=cfg.detection.gnn_training.decoder.predict_edge_type.custom.dropout,
                        num_layers=cfg.detection.gnn_training.decoder.predict_edge_type.custom.num_layers,
                        activation=activation_fn_factory(
                            cfg.detection.gnn_training.decoder.predict_edge_type.custom.activation
                        ),
                    )
                )
        
        else:
            raise ValueError(f"Invalid decoder {method}")
        
    return decoders

def batch_loader_factory(cfg, data, graph_reindexer):
    return custom_temporal_data_loader(data, batch_size=cfg.detection.gnn_training.encoder.batch_size)
    
    data = graph_reindexer.reindex_graph(data)
    return [data]

def activation_fn_factory(activation: str):
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "relu":
        return nn.ReLU()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Invalid activation function {activation}")

def optimizer_factory(cfg, parameters):
    lr = cfg.detection.gnn_training.lr
    weight_decay = cfg.detection.gnn_training.weight_decay

    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay) # TODO: parametrize

def get_dimensions_from_data_sample(data):
    msg_dim = data.msg.shape[1]
    edge_dim = data.edge_feats.shape[1] if hasattr(data, "edge_feats") else None
    in_dim = data.x_src.shape[1]
    
    return msg_dim, edge_dim, in_dim
