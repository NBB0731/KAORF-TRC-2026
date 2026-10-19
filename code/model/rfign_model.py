"""Development RFIGN model with differentiable fuzzy-risk feedback.

Ablation switches added 2026-09-23 for the F5 mechanism study (E stage).
All new switches default to the pre-ablation behaviour, so any run that does
not pass them is bit-for-bit identical to the locked blind-test pipeline.
Guarded by _cloud_support/kaorf_ablation/smoke_guard.py.
"""

from __future__ import annotations

import torch
from torch import nn


class RFIGN(nn.Module):
    def __init__(
        self,
        feature_dim: int = 8,
        history_steps: int = 20,
        future_steps: int = 6,
        hidden_dim: int = 96,
        heads: int = 4,
        layers: int = 2,
        feedback: bool = True,
        detach_risk_feedback: bool = False,
        isolated_risk_head: bool = False,
        constrained_risk_feedback: bool = False,
        residual_bound: float = 1.0,
        anchor_from_fuzzy: bool = False,
        unbounded_residual: bool = False,
        free_ordinal: bool = False,
        damping: float = 0.5,
        position_scale_m: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.feedback = feedback
        self.detach_risk_feedback = detach_risk_feedback
        self.isolated_risk_head = isolated_risk_head
        self.constrained_risk_feedback = constrained_risk_feedback
        self.residual_bound = residual_bound
        # --- ablation switches (F5 / E stage); defaults preserve locked behaviour
        self.anchor_from_fuzzy = anchor_from_fuzzy
        self.unbounded_residual = unbounded_residual
        self.free_ordinal = free_ordinal
        # ---
        self.damping = damping
        self.position_scale_m = position_scale_m
        self.register_buffer(
            "feature_scale",
            torch.tensor([position_scale_m, position_scale_m, 20.0, 20.0, 1.0, 1.0, 20.0, 10.0]),
        )
        self.token = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.time_embedding = nn.Parameter(torch.zeros(1, history_steps, hidden_dim))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=layers)
        self.static = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.node_attention = nn.MultiheadAttention(hidden_dim, heads, dropout=0.1, batch_first=True)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, future_steps * 2),
        )
        self.risk_message = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_risk = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.kinematic_edge_risk = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.kinematic_moderate_delta = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def decode(self, z: torch.Tensor, current_xy: torch.Tensor) -> torch.Tensor:
        # Three normalized position units permit roughly 30 km displacement.
        delta = 3.0 * torch.tanh(self.decoder(z).view(*z.shape[:2], self.future_steps, 2))
        return current_xy.unsqueeze(2) + delta

    def fuzzy_risk(
        self,
        trajectory_normalized: torch.Tensor,
        static: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        trajectory_m = trajectory_normalized * self.position_scale_m
        difference = trajectory_m[:, :, None] - trajectory_m[:, None, :]
        distance = torch.sqrt(torch.sum(difference.square(), dim=-1) + 1e-6)
        length = static[..., 0].clamp(min=20.0)
        width = static[..., 1].clamp(min=5.0)
        # Development fuzzy domain: size-aware pair radius. The paper version
        # will replace this isotropic form with a relative-bearing domain.
        safe_radius = 250.0 + 1.5 * (length[:, :, None] + length[:, None, :]) + 2.0 * (
            width[:, :, None] + width[:, None, :]
        )
        membership = torch.sigmoid((safe_radius.unsqueeze(-1) - distance) / 120.0)
        risk = membership.amax(dim=-1)
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        eye = torch.eye(risk.shape[1], dtype=torch.bool, device=risk.device).unsqueeze(0)
        return torch.where(valid & ~eye, risk, torch.zeros_like(risk))

    def forward(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        node_mask: torch.Tensor,
        static: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, nodes, steps, features = history.shape
        scaled = history / self.feature_scale
        tokens = self.token(scaled.view(batch * nodes, steps, features)) + self.time_embedding
        flat_mask = history_mask.view(batch * nodes, steps)
        temporal_padding = ~flat_mask
        # Transformer attention is undefined when every token of a padded node
        # is masked. Keep one zero-token visible, then explicitly zero the node.
        empty_node = ~flat_mask.any(dim=-1)
        temporal_padding[empty_node, -1] = False
        temporal = self.temporal(tokens, src_key_padding_mask=temporal_padding)
        z = temporal[:, -1].view(batch, nodes, -1)
        static_scaled = static / static.new_tensor([300.0, 50.0])
        z = z + self.static(static_scaled)
        attended, _ = self.node_attention(z, z, z, key_padding_mask=~node_mask, need_weights=False)
        z = self.node_norm(z + attended)
        z = torch.where(node_mask.unsqueeze(-1), z, torch.zeros_like(z))
        current_xy = history[:, :, -1, :2] / self.position_scale_m
        preliminary = self.decode(z, current_xy)
        preliminary_risk = self.fuzzy_risk(preliminary, static, node_mask)

        # Supervised future-risk edge state. The constant-velocity risk is the
        # observed-system prior, while preliminary_risk is the learned forecast.
        offsets_seconds = history.new_tensor([300, 600, 900, 1200, 1500, 1800])
        current_velocity = history[:, :, -1, 2:4]
        cv_trajectory = current_xy.unsqueeze(2) + (
            current_velocity.unsqueeze(2) * offsets_seconds.view(1, 1, -1, 1) / self.position_scale_m
        )
        observed_risk = self.fuzzy_risk(cv_trajectory, static, node_mask)
        anchor_logit = None
        moderate_anchor_logit = None
        if self.isolated_risk_head or self.constrained_risk_feedback:
            # A safety-specific feature stream prevents the trajectory loss
            # from changing risk representations through the shared encoder.
            current_position = history[:, :, -1, :2]
            relative_position = current_position[:, :, None] - current_position[:, None, :]
            relative_velocity = current_velocity[:, :, None] - current_velocity[:, None, :]
            distance = torch.linalg.vector_norm(relative_position, dim=-1, keepdim=True)
            velocity_square = relative_velocity.square().sum(dim=-1, keepdim=True).clamp(min=1e-4)
            tcpa_seconds = -(
                relative_position * relative_velocity
            ).sum(dim=-1, keepdim=True) / velocity_square
            tcpa_seconds = tcpa_seconds.clamp(min=-1800.0, max=1800.0)
            dcpa_position = relative_position + relative_velocity * tcpa_seconds.clamp(min=0.0)
            dcpa = torch.linalg.vector_norm(dcpa_position, dim=-1, keepdim=True)
            length = static[..., 0]
            width = static[..., 1]
            size_features = torch.stack(
                (
                    (length[:, :, None] + length[:, None, :]) / 600.0,
                    (width[:, :, None] + width[:, None, :]) / 100.0,
                    torch.abs(length[:, :, None] - length[:, None, :]) / 300.0,
                    torch.abs(width[:, :, None] - width[:, None, :]) / 50.0,
                ),
                dim=-1,
            )
            risk_features = torch.cat(
                (
                    relative_position / self.position_scale_m,
                    relative_velocity / 20.0,
                    distance / self.position_scale_m,
                    tcpa_seconds / 1800.0,
                    dcpa / self.position_scale_m,
                    size_features,
                    observed_risk.unsqueeze(-1),
                ),
                dim=-1,
            )
            if self.anchor_from_fuzzy:
                # Ablation: the learned kinematic anchor head is replaced by the
                # non-learned geometric fuzzy prior. Everything else (anchor BCE,
                # calibration KL, bounded residual, ordinal head) is untouched, so
                # this isolates the value of learning the anchor.
                anchor_logit = torch.logit(observed_risk.clamp(1e-4, 1.0 - 1e-4))
            else:
                anchor_logit = self.kinematic_edge_risk(risk_features).squeeze(-1)
            # The 0.5-nm event contains the 0.25-nm event. A non-negative
            # log-odds increment enforces this ordinal relationship.
            ordinal_delta = self.kinematic_moderate_delta(risk_features).squeeze(-1)
            if self.free_ordinal:
                # Ablation: drop the non-negative increment, so the moderate head
                # is no longer constrained to dominate the severe head.
                moderate_anchor_logit = anchor_logit + ordinal_delta
            else:
                moderate_anchor_logit = anchor_logit + nn.functional.softplus(ordinal_delta)

        if self.constrained_risk_feedback:
            # The kinematic stream is the calibrated safety anchor. The shared
            # trajectory representation may only make a bounded log-odds
            # correction, preventing the trajectory objective from freely
            # distorting the safety probability.
            zi = z[:, :, None].expand(-1, -1, nodes, -1)
            zj = z[:, None, :].expand(-1, nodes, -1, -1)
            pair_features = torch.cat(
                (
                    zi + zj,
                    torch.abs(zi - zj),
                    observed_risk.unsqueeze(-1),
                    preliminary_risk.unsqueeze(-1),
                ),
                dim=-1,
            )
            raw_residual = self.edge_risk(pair_features).squeeze(-1)
            if self.unbounded_residual:
                # Ablation: remove the tanh saturation and the residual bound, so
                # the learned correction may move the log-odds arbitrarily far.
                residual_logit = raw_residual
            else:
                residual_logit = self.residual_bound * torch.tanh(raw_residual)
            edge_risk = torch.sigmoid(anchor_logit + residual_logit)
            moderate_risk = torch.sigmoid(moderate_anchor_logit + residual_logit)
            anchor_risk = torch.sigmoid(anchor_logit)
            moderate_anchor_risk = torch.sigmoid(moderate_anchor_logit)
        elif self.isolated_risk_head:
            residual_logit = torch.zeros_like(anchor_logit)
            edge_risk = torch.sigmoid(anchor_logit)
            moderate_risk = torch.sigmoid(moderate_anchor_logit)
            anchor_risk = edge_risk
            moderate_anchor_risk = moderate_risk
        else:
            zi = z[:, :, None].expand(-1, -1, nodes, -1)
            zj = z[:, None, :].expand(-1, nodes, -1, -1)
            pair_features = torch.cat(
                (
                    zi + zj,
                    torch.abs(zi - zj),
                    observed_risk.unsqueeze(-1),
                    preliminary_risk.unsqueeze(-1),
                ),
                dim=-1,
            )
            edge_risk = torch.sigmoid(self.edge_risk(pair_features).squeeze(-1))
            anchor_risk = observed_risk
            moderate_risk = observed_risk
            moderate_anchor_risk = observed_risk
            residual_logit = torch.zeros_like(edge_risk)
        edge_valid = node_mask[:, :, None] & node_mask[:, None, :]
        edge_eye = torch.eye(nodes, dtype=torch.bool, device=edge_risk.device).unsqueeze(0)
        edge_risk = torch.where(edge_valid & ~edge_eye, edge_risk, torch.zeros_like(edge_risk))
        anchor_risk = torch.where(
            edge_valid & ~edge_eye, anchor_risk, torch.zeros_like(anchor_risk)
        )
        moderate_risk = torch.where(
            edge_valid & ~edge_eye, moderate_risk, torch.zeros_like(moderate_risk)
        )
        moderate_anchor_risk = torch.where(
            edge_valid & ~edge_eye,
            moderate_anchor_risk,
            torch.zeros_like(moderate_anchor_risk),
        )
        residual_logit = torch.where(
            edge_valid & ~edge_eye, residual_logit, torch.zeros_like(residual_logit)
        )

        if self.feedback:
            # In the detached variant, the risk head is optimized only by its
            # proper scoring loss. This prevents the trajectory objective from
            # distorting calibrated risk probabilities while retaining risk-
            # weighted interaction messages in the trajectory network.
            message_risk = edge_risk.detach() if self.detach_risk_feedback else edge_risk
            denominator = message_risk.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            message = torch.bmm(message_risk, z) / denominator
            update = self.risk_message(torch.cat((z, message), dim=-1))
            z = (1.0 - self.damping) * z + self.damping * update
            z = torch.where(node_mask.unsqueeze(-1), z, torch.zeros_like(z))
        prediction = self.decode(z, current_xy)
        geometric_risk = self.fuzzy_risk(prediction, static, node_mask)
        return {
            "prediction": prediction,
            "risk": edge_risk,
            "anchor_risk": anchor_risk,
            "moderate_risk": moderate_risk,
            "moderate_anchor_risk": moderate_anchor_risk,
            "risk_residual_logit": residual_logit,
            "geometric_risk": geometric_risk,
            "observed_risk": observed_risk,
            "preliminary": preliminary,
        }
