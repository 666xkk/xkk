from provnet_utils import *
from config import *
import torch
import torch.nn as nn

from objectives import bootstrap_ensemble_risk, uncertainty_fused_score
from trace_views import RunningMoments, two_of_three_consensus
from open_set import type_preserving_roll


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
            trace_enabled=False,
            trace_entity_decoder=None,
            trace_temporal_decoder=None,
            trace_entity_weight=0.25,
            trace_temporal_weight=0.10,
            trace_min_scale=1e-3,
            trace_open_enabled=False,
            trace_open_head=None,
            trace_open_prototypes=None,
            trace_open_exposure_weight=0.25,
            trace_open_num_node_types=0,
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
        self.trace_enabled = bool(trace_enabled)
        self.trace_entity_decoder = trace_entity_decoder
        self.trace_temporal_decoder = trace_temporal_decoder
        self.trace_entity_weight = float(trace_entity_weight)
        self.trace_temporal_weight = float(trace_temporal_weight)
        self.trace_update_statistics = False
        self.trace_calibration_phase = None
        self.last_trace_stats = {}
        self.last_trace_components = {}
        self.trace_open_enabled = bool(trace_open_enabled)
        self.trace_open_head = trace_open_head
        self.trace_open_prototypes = trace_open_prototypes
        self.trace_open_exposure_weight = float(trace_open_exposure_weight)
        self.trace_open_num_node_types = int(trace_open_num_node_types)
        if self.trace_enabled:
            self.trace_interaction_moments = RunningMoments(trace_min_scale)
            self.trace_entity_moments = RunningMoments(trace_min_scale)
            self.trace_temporal_moments = RunningMoments(trace_min_scale)
            self.trace_uncertainty_moments = RunningMoments(trace_min_scale)
            self.trace_open_prototype_moments = (
                RunningMoments(trace_min_scale) if self.trace_open_enabled else None
            )
            self.trace_open_unknown_moments = (
                RunningMoments(trace_min_scale) if self.trace_open_enabled else None
            )
            self.register_buffer(
                "trace_last_time",
                torch.full((num_nodes,), -1, dtype=torch.long, device=device),
                persistent=True,
            )
        else:
            # Keep the disabled model state_dict byte-compatible with the
            # baseline/ECHO checkpoints created before TRACE existed.
            self.trace_interaction_moments = None
            self.trace_entity_moments = None
            self.trace_temporal_moments = None
            self.trace_uncertainty_moments = None
            self.trace_open_prototype_moments = None
            self.trace_open_unknown_moments = None
            self.register_buffer(
                "trace_last_time", torch.empty(0, dtype=torch.long), persistent=False
            )
        
        self.last_h_storage, self.last_h_non_empty_nodes = None, None
        if self.use_contrastive_learning:
            self.last_h_storage = torch.empty((num_nodes, out_dim), device=device)
            self.last_h_non_empty_nodes = torch.tensor([], dtype=torch.long, device=device)

    def reset_trace_temporal_state(self):
        self.trace_last_time.fill_(-1)

    def reset_trace_statistics(self):
        for moments in (
            self.trace_interaction_moments,
            self.trace_entity_moments,
            self.trace_temporal_moments,
            self.trace_uncertainty_moments,
            self.trace_open_prototype_moments,
            self.trace_open_unknown_moments,
        ):
            if moments is not None:
                moments.reset()

    @torch.no_grad()
    def _trace_temporal_targets(self, edge_index, timestamps):
        """Compute causal endpoint gaps without looking at future events."""
        src_cpu = edge_index[0].detach().cpu().tolist()
        dst_cpu = edge_index[1].detach().cpu().tolist()
        time_cpu = timestamps.detach().long().cpu().tolist()
        unique_nodes = torch.unique(edge_index).long()
        prior_values = self.trace_last_time[unique_nodes].detach().cpu().tolist()
        state = dict(zip(unique_nodes.detach().cpu().tolist(), prior_values))
        gaps, valid = [], []
        for src, dst, now in zip(src_cpu, dst_cpu, time_cpu):
            previous = max(int(state[src]), int(state[dst]))
            is_valid = previous >= 0 and now >= previous
            gap_seconds = (now - previous) / 1_000_000_000.0 if is_valid else 0.0
            gaps.append(gap_seconds)
            valid.append(is_valid)
            state[src] = now
            state[dst] = now
        latest = torch.tensor(
            [state[node] for node in unique_nodes.detach().cpu().tolist()],
            dtype=torch.long,
            device=edge_index.device,
        )
        self.trace_last_time[unique_nodes] = latest
        gap_tensor = torch.tensor(gaps, dtype=torch.float, device=edge_index.device)
        valid_tensor = torch.tensor(valid, dtype=torch.bool, device=edge_index.device)
        return torch.log1p(gap_tensor), valid_tensor
        
    def forward(self, batch, full_data, inference=False):
        train_mode = not inference
        x = (batch.x_src, batch.x_dst)
        edge_index = batch.edge_index
        if self.trace_enabled:
            log_gap, valid_gap = self._trace_temporal_targets(edge_index, batch.t)

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

            if self.trace_enabled:
                interaction = head_losses.mean(dim=0)
                uncertainty = head_losses.std(dim=0, unbiased=False)
                entity = self.trace_entity_decoder(
                    h_src=h_src,
                    h_dst=h_dst,
                    x_src=x[0],
                    x_dst=x[1],
                )
                temporal = self.trace_temporal_decoder(
                    h_src=h_src,
                    h_dst=h_dst,
                    log_gap=log_gap,
                    valid=valid_gap,
                )

                open_loss = interaction.sum() * 0.0
                prototype_divergence = interaction.new_zeros(interaction.shape)
                unknown_probability = interaction.new_zeros(interaction.shape)
                if self.trace_open_enabled:
                    labels = batch.edge_type.argmax(dim=1)
                    real_logits = self.trace_open_head(h_src, h_dst)
                    real_probabilities = real_logits.softmax(dim=1)
                    known_probabilities = real_probabilities[:, :-1]
                    known_probabilities = known_probabilities / known_probabilities.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-6)
                    prototype_divergence = (
                        self.trace_open_prototypes.minimum_symmetric_kl(
                            known_probabilities
                        )
                    )
                    unknown_probability = real_probabilities[:, -1]
                    if train_mode:
                        virtual_dst, virtual_valid = type_preserving_roll(
                            h_dst,
                            x[1],
                            self.trace_open_num_node_types,
                        )
                        known_loss = torch.nn.functional.cross_entropy(
                            real_logits, labels
                        )
                        if virtual_valid.any():
                            virtual_logits = self.trace_open_head(
                                h_src[virtual_valid], virtual_dst[virtual_valid]
                            )
                            virtual_labels = torch.full(
                                (virtual_logits.shape[0],),
                                real_logits.shape[1] - 1,
                                dtype=torch.long,
                                device=real_logits.device,
                            )
                            virtual_loss = torch.nn.functional.cross_entropy(
                                virtual_logits, virtual_labels
                            )
                        else:
                            virtual_loss = real_logits.sum() * 0.0
                        open_loss = known_loss + virtual_loss
                    if (
                        self.trace_update_statistics
                        and self.trace_calibration_phase in (None, "prototypes")
                    ):
                        self.trace_open_prototypes.update(
                            known_probabilities, labels
                        )

                update_moments = (
                    self.trace_update_statistics
                    and self.trace_calibration_phase in (None, "moments")
                )
                if update_moments:
                    self.trace_interaction_moments.update(interaction)
                    self.trace_entity_moments.update(entity)
                    self.trace_temporal_moments.update(temporal[valid_gap])
                    self.trace_uncertainty_moments.update(uncertainty)
                    if self.trace_open_enabled:
                        self.trace_open_prototype_moments.update(
                            prototype_divergence
                        )
                        self.trace_open_unknown_moments.update(
                            unknown_probability
                        )

                if train_mode:
                    interaction_risk = bootstrap_ensemble_risk(
                        head_losses,
                        keep_probability=self.bootstrap_keep_probability,
                    )
                    entity_risk = entity.mean()
                    temporal_risk = (
                        temporal[valid_gap].mean()
                        if valid_gap.any()
                        else temporal.sum() * 0.0
                    )
                    self.last_trace_stats = {
                        "interaction": float(interaction_risk.detach()),
                        "entity": float(entity_risk.detach()),
                        "temporal": float(temporal_risk.detach()),
                        "uncertainty": float(uncertainty.mean().detach()),
                        "valid_temporal_fraction": float(valid_gap.float().mean().detach()),
                        "open_exposure": float(open_loss.detach()),
                        "prototype_divergence": float(prototype_divergence.mean().detach()),
                        "unknown_probability": float(unknown_probability.mean().detach()),
                    }
                    return (
                        interaction_risk
                        + self.trace_entity_weight * entity_risk
                        + self.trace_temporal_weight * temporal_risk
                        + self.trace_open_exposure_weight * open_loss
                    )

                interaction_z = self.trace_interaction_moments.positive_z(interaction)
                entity_z = self.trace_entity_moments.positive_z(entity)
                temporal_z = self.trace_temporal_moments.positive_z(temporal)
                temporal_z = torch.where(valid_gap, temporal_z, torch.zeros_like(temporal_z))
                uncertainty_z = self.trace_uncertainty_moments.positive_z(uncertainty)
                consensus = two_of_three_consensus(
                    interaction_z, entity_z, temporal_z
                )
                open_consensus = consensus
                prototype_evidence = torch.zeros_like(consensus)
                unknown_evidence = torch.zeros_like(consensus)
                if self.trace_open_enabled:
                    prototype_evidence = self.trace_open_prototype_moments.positive_z(
                        prototype_divergence
                    )
                    unknown_evidence = self.trace_open_unknown_moments.positive_z(
                        unknown_probability
                    )
                    # Two independent mechanisms must agree: multi-view TRACE,
                    # relational-prototype divergence, or the learned virtual
                    # unknown class.  Epistemic uncertainty remains routing-only.
                    open_consensus = torch.topk(
                        torch.stack(
                            [consensus, prototype_evidence, unknown_evidence], dim=0
                        ),
                        k=2,
                        dim=0,
                    ).values[1]
                self.last_trace_components = {
                    "interaction_loss": interaction.detach(),
                    "entity_loss": entity.detach(),
                    "temporal_loss": temporal.detach(),
                    "interaction_evidence": interaction_z.detach(),
                    "entity_evidence": entity_z.detach(),
                    "temporal_evidence": temporal_z.detach(),
                    "epistemic_uncertainty": uncertainty.detach(),
                    "epistemic_evidence": uncertainty_z.detach(),
                    "temporal_valid": valid_gap.detach(),
                    "consensus_score": consensus.detach(),
                    "prototype_divergence": prototype_divergence.detach(),
                    "prototype_evidence": prototype_evidence.detach(),
                    "virtual_unknown_probability": unknown_probability.detach(),
                    "virtual_unknown_evidence": unknown_evidence.detach(),
                    "trace_open_score": open_consensus.detach(),
                }
                return open_consensus

            if self.uncertainty_enabled:
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
