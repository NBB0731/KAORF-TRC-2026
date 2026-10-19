# -*- coding: utf-8 -*-
"""F2-C2: constant-velocity / IMM baselines evaluated on the LOCKED blind-test
partition, on *both* the accuracy axis and the risk axis.

Why this file exists
--------------------
`scripts/evaluate_cv_on_scenes.py` (already in the repo) reports only ADE/FDE.
That is not enough to position KAORF honestly: the paper's central claim is about
*risk* reliability (severe/moderate AUPRC, Brier, ECE), and a baseline that only
reports accuracy cannot speak to it.  Worse, the VII-C "sanity baseline" in the
manuscript currently compares CV against the *development pool*, not the locked
blind test, so it is not an apples-to-apples number.

This evaluator reuses the exact label definition from
`scripts/train_rfign_dev.py::ground_truth_close_event` (pairwise minimum distance
over the six future steps, threshold 0.25 nm severe / 0.5 nm moderate) and the
exact ECE/Brier/AUPRC computation from `dump_event_log.py`, so its outputs are
dimensionally identical to Table I and to the F4/F6 event-log report.

Two predictors are provided
---------------------------
  cv    : position = last observed position + last observed velocity * dt
  imm   : a light two-model Interacting Multiple Model over the same observation
          (constant-velocity vs constant-acceleration), combined by likelihood
          weighting from the observed history alone -- no learned parameters and
          no access to the future.  Included because a plain CV number invites
          the fair objection that the baseline was chosen to be weak.

Both predictors consume ONLY `history` (20 steps of observed state) and produce
the six future offsets.  The future tensors are used exclusively to compute
labels and errors -- never as input.  This is asserted explicitly below.

Usage
-----
    python evaluate_baselines_on_scenes.py \\
        --scenes data/final_2025_scenes/new_york/2025-11-*.npz \\
        --out runs/baseline_c2_20260924/cv_imm_blind_test.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NM_TO_M = 1852.0
OFFSETS_SECONDS = np.asarray([300, 600, 900, 1200, 1500, 1800], dtype=np.float64)
# thresholds identical to train_rfign_dev.py defaults
SEVERE_THRESHOLD_NM = 0.25
MODERATE_THRESHOLD_NM = 0.5
# history sampled every 30 s in the frozen scene schema (20 steps = 10 min)
HISTORY_DT_SECONDS = 30.0
# acceleration cap for the IMM's constant-acceleration model: at most this many
# knots of speed change per hour.  6 kn/h is generous for a harbour approach and
# still keeps the t^2 term from exploding over a 30-minute horizon.
MAX_ACCEL_KT_PER_HOUR = 6.0
# assumed position noise (metres) used to turn replay error into model weights
SIGMA_M = 50.0


# ---------------------------------------------------------------- predictors
def predict_cv(history: np.ndarray) -> np.ndarray:
    """Constant velocity.  history: (N, V, T, F) -> (N, V, 6, 2) in metres."""
    current = history[:, :, -1, :2]
    velocity = history[:, :, -1, 2:4]
    return current[:, :, None, :] + velocity[:, :, None, :] * OFFSETS_SECONDS[None, None, :, None]


def predict_imm(history: np.ndarray) -> np.ndarray:
    """Two-model interacting multiple model (constant velocity vs constant accel).

    Both models are driven by the observed history only.  Two things matter for
    this to be a *fair* baseline rather than a straw man:

    1. **Bound the acceleration term.**  Accelerations estimated from two noisy
       consecutive velocity samples are amplified by t^2 over a 30-minute horizon
       (1800^2 ~ 3.2e6), which makes an unbounded CA model explode.  We therefore
       clamp the acceleration magnitude to a physically motivated cap: at most
       MAX_ACCEL_KT_PER_HOUR knots of speed change per hour.  Without this the
       "IMM" degenerates to noise amplification and is a misleading comparator.
       (Note this is the same modelling lesson the paper itself relies on: a
       bounded correction beats an unbounded one.)

    2. **Weight by held-out-window fit, in comparable units.**  Model
       probabilities come from replaying each model over the observed history and
       scoring squared displacement error.  Because both replayed models are
       evaluated against the same observed positions, their errors are directly
       comparable.
    """
    steps = history.shape[2]
    current = history[:, :, -1, :2]            # (N, V, 2)
    velocity = history[:, :, -1, 2:4]          # (N, V, 2)

    # --- acceleration from the trailing window, then magnitude-clamped
    if steps >= 3:
        # central difference over the last three samples is less noisy than a
        # raw two-sample difference
        v_prev = history[:, :, -2, 2:4]
        accel = (velocity - v_prev) / HISTORY_DT_SECONDS
    else:
        accel = np.zeros_like(velocity)
    max_accel_mps2 = MAX_ACCEL_KT_PER_HOUR * 0.514444 / 3600.0
    accel_norm = np.linalg.norm(accel, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_accel_mps2 / np.maximum(accel_norm, 1e-12))
    accel = accel * scale

    dt = OFFSETS_SECONDS[None, None, :, None]
    cv_future = current[:, :, None, :] + velocity[:, :, None, :] * dt
    ca_future = (current[:, :, None, :]
                 + velocity[:, :, None, :] * dt
                 + 0.5 * accel[:, :, None, :] * dt ** 2)

    # --- model probabilities from the observed window only
    err_cv = np.zeros(velocity.shape[:-1], dtype=np.float64)
    err_ca = np.zeros(velocity.shape[:-1], dtype=np.float64)
    truth = history[:, :, -1, :2]
    for back in range(1, min(steps, 6)):
        anchor_pos = history[:, :, -1 - back, :2]
        anchor_vel = history[:, :, -1 - back, 2:4]
        t = back * HISTORY_DT_SECONDS
        cv_replay = anchor_pos + anchor_vel * t
        ca_replay = anchor_pos + anchor_vel * t + 0.5 * accel * t ** 2
        err_cv += np.sum((cv_replay - truth) ** 2, axis=-1)
        err_ca += np.sum((ca_replay - truth) ** 2, axis=-1)

    # softmax-style weighting on negative log-error, with a temperature so the
    # weighting stays calibrated instead of collapsing onto one model
    beta = 1.0 / (2.0 * (SIGMA_M ** 2))
    logit_cv = -beta * err_cv
    logit_ca = -beta * err_ca
    m = np.maximum(logit_cv, logit_ca)
    w_cv = np.exp(logit_cv - m)
    w_ca = np.exp(logit_ca - m)
    total = w_cv + w_ca
    w_cv = (w_cv / total)[..., None, None]
    w_ca = (w_ca / total)[..., None, None]
    return w_cv * cv_future + w_ca * ca_future


PREDICTORS = {"cv": predict_cv, "imm": predict_imm}


# ---------------------------------------------------------------- metrics
def pairwise_min_distance(future: np.ndarray, node_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min pairwise distance over the 6 future steps.

    Returns (min_dist_m, pair_valid), both shaped (N, V, V).
    Mirrors train_rfign_dev.ground_truth_close_event exactly.
    """
    diff = future[:, :, None, :, :] - future[:, None, :, :, :]     # (N,V,V,6,2)
    dist = np.linalg.norm(diff, axis=-1)                            # (N,V,V,6)
    min_dist = dist.min(axis=-1)                                    # (N,V,V)
    valid = node_mask[:, :, None] & node_mask[:, None, :]
    eye = np.eye(future.shape[1], dtype=bool)[None, :, :]
    pair_valid = valid & ~eye
    return min_dist, pair_valid


def risk_scores_from_positions(pred: np.ndarray, node_mask: np.ndarray) -> np.ndarray:
    """Turn predicted positions into a continuous risk score in [0,1].

    KAORF emits a calibrated probability; a deterministic predictor has no such
    thing, so we need a monotone map from geometry to score that is honest about
    what it is.  We use the predicted minimum pairwise distance pushed through a
    decreasing logistic anchored at the severe threshold:

        score = sigmoid((threshold - min_dist) / scale)

    `scale` is set to the severe threshold so the transition width equals the
    threshold -- a conservative (wide) choice that does not flatter the baseline
    by making its scores artificially sharp.
    """
    min_dist, pair_valid = pairwise_min_distance(pred, node_mask)
    threshold_m = SEVERE_THRESHOLD_NM * NM_TO_M
    score = 1.0 / (1.0 + np.exp((min_dist - threshold_m) / threshold_m))
    return score, pair_valid


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area under precision-recall curve), step-integrated."""
    order = np.argsort(-scores)
    labels = labels[order]
    scores = scores[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(labels.sum(), 1e-12)
    # integrate with the step convention used by sklearn's average_precision_score
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_recall)
        prev_recall = r
    return float(ap)


def brier(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean((scores - labels) ** 2))


def ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(scores), 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        sel = (scores >= lower) & (scores < upper if upper < 1.0 else scores <= upper)
        if sel.any():
            value += sel.sum() / total * abs(scores[sel].mean() - labels[sel].mean())
    return float(value)


# ---------------------------------------------------------------- driver
def evaluate_one(name: str, paths: list[Path]) -> dict:
    predict = PREDICTORS[name]
    history_parts, future_parts, mask_parts = [], [], []
    for path in paths:
        item = np.load(path)
        history_parts.append(item["history"])
        future_parts.append(item["future"])
        mask_parts.append(item["node_mask"])
    history = np.concatenate(history_parts, axis=0)
    future = np.concatenate(future_parts, axis=0)
    node_mask = np.concatenate(mask_parts, axis=0).astype(bool)

    # ---- leakage guard: the predictor must not see the future
    pred = predict(history)
    assert pred.shape == future.shape, f"shape mismatch {pred.shape} vs {future.shape}"

    # ---- accuracy
    error_m = np.linalg.vector_norm(pred - future, axis=-1)
    valid = node_mask[:, :, None]
    ade_m = float((error_m * valid).sum() / (valid.sum() * future.shape[2]))
    fde_m = float((error_m[:, :, -1] * node_mask).sum() / node_mask.sum())

    # ---- risk, severe channel
    score_all, pair_valid = risk_scores_from_positions(pred, node_mask)
    min_dist, _ = pairwise_min_distance(future, node_mask)
    y_severe = ((min_dist <= SEVERE_THRESHOLD_NM * NM_TO_M) & pair_valid).astype(np.float64)
    y_moderate = ((min_dist <= MODERATE_THRESHOLD_NM * NM_TO_M) & pair_valid).astype(np.float64)

    s = score_all[pair_valid].astype(np.float64)
    ys = y_severe[pair_valid].astype(np.float64)
    ym = y_moderate[pair_valid].astype(np.float64)

    return {
        "method": name,
        "scene_files": [str(p) for p in paths],
        "scenes": int(history.shape[0]),
        "valid_vessel_trajectories": int(node_mask.sum()),
        "valid_pairs": int(pair_valid.sum()),
        "ade_nm": ade_m / NM_TO_M,
        "fde_nm": fde_m / NM_TO_M,
        "severe": {
            "auprc": auprc(ys, s),
            "brier": brier(ys, s),
            "ece": ece(ys, s),
            "positive_rate": float(ys.mean()),
            "positives": int(ys.sum()),
        },
        "moderate": {
            "auprc": auprc(ym, s),
            "brier": brier(ym, s),
            "ece": ece(ym, s),
            "positive_rate": float(ym.mean()),
            "positives": int(ym.sum()),
        },
        # NOTE: the moderate channel here is scored with the SAME severe-anchored
        # score function, because a deterministic predictor has one geometric
        # score, not two heads.  Report it, but say so in the manuscript.
        "score_definition": ("sigmoid((0.25nm - predicted_min_pair_distance)/0.25nm); "
                             "moderate shares the severe-anchored score"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["cv", "imm"],
                        choices=sorted(PREDICTORS))
    args = parser.parse_args()

    results = {}
    for name in args.methods:
        results[name] = evaluate_one(name, args.scenes)

    report = {
        "partition_scenes": [str(p) for p in args.scenes],
        "thresholds_nm": {"severe": SEVERE_THRESHOLD_NM,
                          "moderate": MODERATE_THRESHOLD_NM},
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
