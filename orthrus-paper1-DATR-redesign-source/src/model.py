from provnet_utils import *
from config import *
import torch
import torch.nn as nn


class Orthrus(nn.Module):
    def __init__(self,
            encoder: nn.Module,
            decoders: list[nn.Module],
            num_nodes: int,
            in_dim: int,
            out_dim: int,
            use_contrastive_learning: bool,
            device,
            graph_reindexer,
            training_objective,
            uncertainty_enabled=False,
            bootstrap_keep_probability=0.8,
            disagreement_weight=0.5,
        ):
        super(Orthrus, self).__init__()

        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)
        self.use_contrastive_learning = use_contrastive_learning
        self.graph_reindexer = graph_reindexer
        self.training_objective = training_objective
        self.uncertainty_enabled = bool(uncertainty_enabled)
        self.bootstrap_keep_probability = float(bootstrap_keep_probability)
        self.disagreement_weight = float(disagreement_weight)
        self.last_uncertainty_stats = {}
        
        self.last_h_storage, self.last_h_non_empty_nodes = None, None
        if self.use_contrastive_learning:
            self.last_h_storage = torch.empty((num_nodes, out_dim), device=device)
            self.last_h_non_empty_nodes = torch.tensor([], dtype=torch.long, device=device)
        
    def forward(self, batch, full_data, inference=False):
        train_mode = not inference
        x = (batch.x_src, batch.x_dst)
        edge_index = batch.edge_index

        with torch.set_grad_enabled(train_mode):
            h = self.encoder(
                edge_index=edge_index,
                t=batch.t,
                x=x,
                msg=batch.msg,
                edge_feats=batch.edge_feats if hasattr(batch, "edge_feats") else None,
                full_data=full_data,
                inference=inference,

                edge_types= batch.edge_type
            )

            h_src, h_dst = (h[edge_index[0]], h[edge_index[1]]) \
                if isinstance(h, torch.Tensor) \
                else h
        
            if x[0].shape[0] != edge_index.shape[1]:
                x = (batch.x_src[edge_index[0]], batch.x_dst[edge_index[1]])
            
            if self.use_contrastive_learning:
                involved_nodes = edge_index.flatten()
                self.last_h_storage[involved_nodes] = torch.cat([h_src, h_dst]).detach()
                self.last_h_non_empty_nodes = torch.cat([involved_nodes, self.last_h_non_empty_nodes]).unique()
            
            # Train mode: loss | Inference mode: edge scores
            per_head = []
            for decoder in self.decoders:
                loss = decoder(
                    h_src=h_src,
                    h_dst=h_dst,
                    x=x,
                    edge_index=edge_index,
                    edge_type=batch.edge_type,
                    inference=inference,
                    last_h_storage=self.last_h_storage,
                    last_h_non_empty_nodes=self.last_h_non_empty_nodes,
                )
                if loss.numel() != edge_index.shape[1]:
                    raise TypeError(
                        f"Shapes of loss/score do not match "
                        f"({loss.numel()} vs {edge_index.shape[1]})"
                    )
                per_head.append(loss)

            head_losses = torch.stack(per_head, dim=0)

            if self.uncertainty_enabled:
                from objectives import bootstrap_ensemble_risk, uncertainty_fused_score

                disagreement = head_losses.std(dim=0, unbiased=False)
                self.last_uncertainty_stats = {
                    "mean_loss": float(head_losses.mean().detach()),
                    "mean_disagreement": float(disagreement.mean().detach()),
                    "max_disagreement": float(disagreement.max().detach()),
                }
                if train_mode:
                    return bootstrap_ensemble_risk(
                        head_losses,
                        keep_probability=self.bootstrap_keep_probability,
                    )
                return uncertainty_fused_score(
                    head_losses,
                    disagreement_weight=self.disagreement_weight,
                )

            loss_or_scores = head_losses.sum(dim=0)

            if train_mode:
                return self.training_objective(loss_or_scores, batch.edge_type)
            return loss_or_scores
