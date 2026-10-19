# -*- coding: utf-8 -*-
"""F2-C1-graph: graph-interaction baseline for the KAORF comparison.

Why this file exists
--------------------
The manuscript's outstanding-baselines item asks for "a graph-based learned
model" under the identical locked protocol, complementing the plain temporal
Transformer (C1, ``plain_transformer.py``).  C1 keeps RFIGN's cross-vessel
**self-attention**; this baseline replaces exactly that interaction module
with a **graph convolution**, so the two learned baselines differ from each
other only in how vessel interactions are encoded:

  C1   : trunk + MultiheadAttention over nodes      (attention interaction)
  C-gr : trunk + distance-thresholded GCN           (graph interaction)

Everything else mirrors C1 deliberately:

  * identical trunk (token embedding -> 2-layer temporal TransformerEncoder
    -> static-dimension embedding) and identical MLP trajectory decoder;
  * no kinematic anchor, no bounded residual, no ordinal nesting, no risk
    feedback into the trajectory representation;
  * one free risk head reading the shared pair representation and the model's
    own predicted geometry (same information as C1's risk head).

Graph definition (canonical Kipf & Welling 2017)
------------------------------------------------
* nodes: the (up to eight) vessels of the scene at the anchor instant;
* edges: unordered pairs whose anchor-time separation is within the scene
  constructor's own 5 nm candidate radius (9,260 m), plus a self-loop for
  every valid node, so an isolated vessel still keeps its own state and the
  aggregation is never undefined;
* weights: symmetric degree normalization (D^{-1/2} A D^{-1/2});
* two graph-convolution layers, each a residual update
  z <- LayerNorm(z + dropout(W (A_norm z))), mirroring C1's residual
  attention block structurally.

The class mirrors RFIGN's ``forward`` signature and return-dict contract, so
``SceneDataset`` and ``run_epoch`` in ``train_rfign_dev.py`` consume it without
modification.  This file adds a new module only; it does not edit any locked
artefact.
"""
from __future__ import annotations

import torch
import torch.nn as nn

EDGE_RADIUS_M = 5.0 * 1852.0  # scene constructor's 5 nm candidate radius


class GraphBaseline(nn.Module):
    """Graph-interaction forecaster: trunk + GCN, no kinematic mechanism."""

    def __init__(
        self,
        feature_dim: int = 8,
        history_steps: int = 20,
        future_steps: int = 6,
        hidden_dim: int = 96,
        layers: int = 2,
        gcn_dropout: float = 0.1,
        position_scale_m: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.position_scale_m = position_scale_m
        # Read by run_epoch in train_rfign_dev.py; the baseline has no
        # anchor / residual / ordinal terms, so the trainer's constrained-only
        # loss block is skipped (same contract as PlainTransformer).
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
            nhead=4,
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

        # Graph interaction: two Kipf-Welling graph-convolution layers with
        # residual + LayerNorm, replacing C1's node MultiheadAttention block.
        self.gcn_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(2)]
        )
        self.gcn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(2)]
        )
        self.gcn_dropout = nn.Dropout(gcn_dropout)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, future_steps * 2),
        )

        # Single free risk head, identical in shape and inputs to C1's.
        self.edge_risk = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------ #
    def normalized_adjacency(
        self, history: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        """Distance-thresholded, symmetrically degree-normalized adjacency.

        history positions at the anchor instant are metres (raw feature scale).
        Returns (B, N, N) with self-loops; invalid nodes take part in no edge.
        """
        position_m = history[:, :, -1, :2]
        difference = position_m[:, :, None] - position_m[:, None, :]
        distance = torch.linalg.vector_norm(difference, dim=-1)
        eye = torch.eye(
            distance.shape[-1], dtype=history.dtype, device=history.device
        )
        adjacency = (distance <= EDGE_RADIUS_M).to(history.dtype) * (1.0 - eye)
        adjacency = adjacency + eye
        valid = (
            node_mask.to(history.dtype)[:, :, None]
            * node_mask.to(history.dtype)[:, None, :]
        )
        adjacency = adjacency * valid
        degree = adjacency.sum(dim=-1).clamp(min=1.0)
        degree_inv_sqrt = degree.pow(-0.5)
        return adjacency * degree_inv_sqrt.unsqueeze(-1) * degree_inv_sqrt.unsqueeze(-2)

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
        # visible and zero the node afterwards (same guard as RFIGN / C1).
        empty_node = ~flat_mask.any(dim=-1)
        temporal_padding[empty_node, -1] = False
        temporal = self.temporal(tokens, src_key_padding_mask=temporal_padding)
        z = temporal[:, -1].view(batch, nodes, -1)

        static_scaled = static / static.new_tensor([300.0, 50.0])
        z = z + self.static(static_scaled)

        adjacency_norm = self.normalized_adjacency(history, node_mask)
        keep = node_mask.unsqueeze(-1)
        for linear, norm in zip(self.gcn_linears, self.gcn_norms):
            aggregated = torch.bmm(adjacency_norm, z)
            z = norm(z + self.gcn_dropout(linear(aggregated)))
            z = torch.where(keep, z, torch.zeros_like(z))

        current_xy = history[:, :, -1, :2] / self.position_scale_m
        prediction = self.decode(z, current_xy)

        # Predicted pairwise geometry as explicit edge features (same inputs
        # as C1's risk head; no anchor, no residual, no ordinal coupling).
        pred_m = prediction * self.position_scale_m
        rel_pos = pred_m[:, :, None] - pred_m[:, None, :]
        rel_dist = torch.linalg.vector_norm(rel_pos, dim=-1)
        min_dist = rel_dist.min(dim=-1).values
        current_vel = history[:, :, -1, 2:4]
        rel_vel = current_vel[:, :, None] - current_vel[:, None, :]
        rel_speed = torch.linalg.vector_norm(rel_vel, dim=-1)

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
        keep_edges = edge_valid & ~edge_eye
        edge_risk = torch.where(keep_edges, edge_risk, torch.zeros_like(edge_risk))

        # Same return keys RFIGN / PlainTransformer expose so run_epoch works
        # unchanged; the baseline has no anchor and no separate moderate head.
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
