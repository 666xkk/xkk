from provnet_utils import *
from config import *
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
        ):
        super(Orthrus, self).__init__()

        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)
        self.use_contrastive_learning = use_contrastive_learning
        self.graph_reindexer = graph_reindexer
        self.training_objective = training_objective
        
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
            loss_or_scores = torch.zeros(
                edge_index.shape[1], dtype=torch.float, device=h_src.device
            )
            
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
                if loss.numel() != loss_or_scores.numel():
                    raise TypeError(f"Shapes of loss/score do not match ({loss.numel()} vs {loss_or_scores.numel()})")
                loss_or_scores = loss_or_scores + loss

            if train_mode:
                return self.training_objective(loss_or_scores, batch.edge_type)
            return loss_or_scores
