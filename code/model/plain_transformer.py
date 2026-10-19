# -*- coding: utf-8 -*-
"""F2-C1: plain temporal-Transformer baseline for the KAORF comparison.

Why this file exists
--------------------
Section VII-C of the manuscript now compares KAORF against two **non-learned**
physical predictors (constant velocity, acceleration-clamped IMM) on the locked
blind partition.  A reviewer's first question is the *learned* one: is the
proposed architecture better than a straightforward learned sequence model with
the same inputs, the same split and the same selection rule?

``rfign/model.py`` cannot answer that on its own, because every mode of RFIGN
retains at least one KAORF-specific ingredient:

  * ``constrained_feedback``  -> kinematic anchor + bounded residual + feedback
  * ``--no-feedback``         -> kinematic anchor + preliminary-geometry risk head
  * ``--isolated-risk-head``  -> kinematic anchor, no feedback

So the honest baseline is a **separate**, deliberately simple model built from
the same primitives (token embedding -> temporal TransformerEncoder -> node
self-attention -> MLP decoder) with the KAORF additions removed:

  * no learned kinematic anchor and no fuzzy-domain prior;
  * no bounded residual and no ordinal nesting;
  * no risk message fed back into the trajectory representation.

Risk supervision is retained, because the baseline must be able to emit a risk
probability to be comparable on the risk axes at all.  Its risk head reads only
the shared scene representation and the two predicted trajectories -- i.e. the
same information a plain interaction-aware Transformer has.  This keeps the
comparison **architecture-level** (does the KAORF inference structure help?)
rather than confounding it with "the baseline had no risk objective".

The class mirrors RFIGN's ``forward`` signature and return dict exactly, so
``SceneDataset`` and ``run_epoch`` in ``train_rfign_dev.py`` consume it without
modification.  This file adds a new module only; it does not edit any locked
artefact, so the completed blind test / ablation / external-port results are
unaffected.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PlainTransformer(nn.Module):
    """Interaction-aware Transformer with no kinematic anchor and no feedback.

    Parameters are chosen to match RFIGN's *shared* trunk (hidden width, head
    count, depth, dropout, feature scaling) so that the only differences are the
    KAORF-specific mechanisms, not capacity.
    """

    def __init__(
        self,
        feature_dim: int = 8,
        history_steps: int = 20,
        future_steps: int = 6,
        hidden_dim: int = 96,
        heads: int = 4,
        layers: int = 2,
        damping: float = 0.5,
        position_scale_m: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.position_scale_m = position_scale_m
        # read by run_epoch in train_rfign_dev.py; the baseline deliberately has
        # no anchor / residual / ordinal terms, so this stays False and the
        # trainer's constrained-only loss block is skipped.
        self.constrained_risk_feedback = False
        self.register_buffer(
            "feature_scale",
            torch.tensor([position_scale_m, position_scale_m, 20.0, 20.0, 1.0, 1.0, 20.0, 10.0]),
        )

        self.token = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
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
        self.static = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_attention = nn.MultiheadAttention(hidden_dim, heads, dropout=0.1, batch_first=True)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, future_steps * 2),
        )

        # A single risk head reading the shared representation of the pair plus
        # the model's own predicted geometry.  No anchor, no residual, no
        # ordinal coupling -- one free logit per edge.
        # Input width: [z_i + z_j, |z_i - z_j|] = 2*hidden_dim, plus the two
        # scalar edge features (min predicted separation, relative speed).
        self.edge_risk = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------ #
    def decode(self, z: torch.Tensor, current_xy: torch.Tensor) -> torch.Tensor:
        residual = self.decoder(z).view(*z.shape[:-1], self.future_steps, 2)
        return residual + current_xy.unsqueeze(-2)

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
        # A fully padded node would make attention undefined; keep one token
        # visible and zero the node afterwards (same guard as RFIGN).
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
        prediction = self.decode(z, current_xy)

        # Predicted pairwise geometry, in metres, as explicit edge features.
        pred_m = prediction * self.position_scale_m
        rel_pos = pred_m[:, :, None] - pred_m[:, None, :]              # B,N,N,T,2
        rel_dist = torch.linalg.vector_norm(rel_pos, dim=-1)           # B,N,N,T
        min_dist = rel_dist.min(dim=-1).values                         # B,N,N
        current_vel = history[:, :, -1, 2:4]
        rel_vel = current_vel[:, :, None] - current_vel[:, None, :]
        rel_speed = torch.linalg.vector_norm(rel_vel, dim=-1)          # B,N,N

        zi = z[:, :, None].expand(-1, -1, nodes, -1)
        zj = z[:, None, :].expand(-1, nodes, -1, -1)
        pair_features = torch.cat(
            (
                zi + zj,
                torch.abs(zi - zj),
                (min_dist / self.position_scale_m).unsqueeze(-1),
                (rel_speed / 20.0).unsqueeze(-1),
            ),
            dim=-1,
        )
        edge_risk = torch.sigmoid(self.edge_risk(pair_features).squeeze(-1))

        edge_valid = node_mask[:, :, None] & node_mask[:, None, :]
        edge_eye = torch.eye(nodes, dtype=torch.bool, device=edge_risk.device).unsqueeze(0)
        keep = edge_valid & ~edge_eye
        edge_risk = torch.where(keep, edge_risk, torch.zeros_like(edge_risk))

        # Return the same keys RFIGN returns so run_epoch works unchanged.
        # `anchor_risk` / `moderate_*` are reported as the model's own risk: the
        # baseline has no anchor and no separate moderate head, and labelling it
        # otherwise would overstate what this architecture provides.
        zero = torch.zeros_like(edge_risk)
        return {
            "prediction": prediction,
            "risk": edge_risk,
            "anchor_risk": edge_risk,
            "moderate_risk": edge_risk,
            "moderate_anchor_risk": edge_risk,
            "residual_logit": zero,
            "geometric_risk": edge_risk,
        }
