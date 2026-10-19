# -*- coding: utf-8 -*-
"""F2-C2: recurrent sequence baseline for the KAORF comparison.

Why this file exists
--------------------
The outstanding-baselines item asks for strong learned comparators under the
identical locked protocol.  Two have been run:

  C1   : trunk + cross-vessel MultiheadAttention   (attention interaction)
  C-gr : trunk + distance-thresholded GCN          (graph interaction)

Both keep the *temporal Transformer encoder* of the KAORF trunk.  This third
arm replaces the temporal encoder itself, so that the comparison sweeps three
different architecture families rather than three variants of one:

  C2   : LSTM encoder + direct multi-output MLP decoder, NO cross-vessel
         interaction module at all

C2 therefore isolates two things at once: a recurrent rather than attentional
temporal model, and the complete absence of an interaction module.  It is the
"per-vessel sequence model" family that maritime AIS forecasting has used
since the early seq2seq literature, and it is the strongest form of the
question "does any of the proposed interaction/risk machinery matter, or would
a plain sequence model do?"

Design rules (all chosen to keep the comparison fair)
-----------------------------------------------------
* identical front end: token embedding from the same 8 features, the same
  learned time embedding, the same static-dimension embedding;
* identical supervision and loss: ``run_epoch`` from ``train_rfign_dev.py``
  is imported verbatim, with ``--close-event-target`` labels at 0.25/0.5 nm;
* identical free risk head shape and inputs as C1 and the graph baseline: the
  pair representation (sum and absolute difference of the two node states)
  plus the model's own predicted minimum separation and relative speed;
* one free risk head only -- no kinematic anchor, no bounded residual, no
  ordinal nesting, no stop-gradient feedback.

Hidden width is 128 rather than 96 because a two-layer LSTM of width 96 would
leave the model at ~219k parameters against KAORF's 353k, i.e. 38% smaller --
a size difference that would confound the comparison.  At width 128 the total
is ~354k, within 1% of KAORF, so the arm is comparable in capacity while
remaining a genuinely different family.

Padding is handled with ``pack_padded_sequence`` over the true per-node history
length, so a vessel with a short observed history is encoded from the steps it
actually has; fully empty nodes are zeroed afterwards, matching the guard used
by RFIGN, C1 and the graph baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMBaseline(nn.Module):
    """Recurrent per-vessel forecaster: LSTM encoder + MLP decoder, no interaction."""

    def __init__(
        self,
        feature_dim: int = 8,
        history_steps: int = 20,
        future_steps: int = 6,
        hidden_dim: int = 128,
        encoder_layers: int = 2,
        encoder_dropout: float = 0.1,
        position_scale_m: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.position_scale_m = position_scale_m
        # Read by run_epoch in train_rfign_dev.py; the baseline has no
        # anchor / residual / ordinal terms, so the trainer's constrained-only
        # loss block is skipped (same contract as PlainTransformer / GraphBaseline).
        self.constrained_risk_feedback = False
        self.register_buffer(
            "feature_scale",
            torch.tensor([position_scale_m, position_scale_m, 20.0, 20.0, 1.0, 1.0, 20.0, 10.0]),
        )

        self.token = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.time_embedding = nn.Parameter(torch.zeros(1, history_steps, hidden_dim))
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=encoder_layers,
            dropout=encoder_dropout if encoder_layers > 1 else 0.0,
            batch_first=True,
        )
        self.static = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # Direct multi-output decoder: one shot from the final encoder state to
        # all six future offsets (no autoregression, no cross-vessel mixing).
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
        lengths = flat_mask.sum(dim=-1).clamp(min=1)
        packed = pack_padded_sequence(
            tokens, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.encoder(packed)
        z = hidden[-1]  # (batch*nodes, hidden) -- final layer state per sequence

        static_scaled = static / static.new_tensor([300.0, 50.0])
        z = z + self.static(static_scaled.view(batch * nodes, -1))
        keep = node_mask.view(batch * nodes, 1)
        z = torch.where(keep, z, torch.zeros_like(z))
        z = z.view(batch, nodes, -1)

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

        # Same return keys RFIGN / PlainTransformer / GraphBaseline expose so
        # run_epoch works unchanged; no anchor and no separate moderate head.
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
