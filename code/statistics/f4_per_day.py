# -*- coding: utf-8 -*-
"""Per-day breakdown of the F4 severe/moderate AUPRC deltas.

The headline F4 result is that day-clustered inference flips severe AUPRC from
"statistically tied" (seed level: +0.000024) to "significantly worse" (day level:
-0.006931, 95% CI [-0.0108, -0.0034]).  A reviewer will ask the obvious follow-up:
is that driven by one anomalous day, or is it consistent across the eight?

This script answers that directly: it recomputes each metric per calendar day for
both variants, averages across the 10 paired seeds, and reports the sign pattern.

Outputs JSON + a human-readable table so the manuscript can state the pattern.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LOG_DIR = Path("/root/kaorf/runs/event_log_20260924/blind")
VARIANTS = ("constrained_feedback", "constrained_no_feedback")
SEEDS = (17, 29, 42, 71, 101, 131, 211, 307, 509, 887)


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(labels.sum(), 1e-12)
    ap = 0.0
    prev = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev)
        prev = r
    return float(ap)


def brier(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean((scores - labels) ** 2))


def ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(scores), 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if sel.any():
            value += sel.sum() / total * abs(scores[sel].mean() - labels[sel].mean())
    return float(value)


SCENES_DIR = Path("/root/kaorf/data/final_2025_scenes/new_york")


def day_slices(meta_path: Path, npz: dict) -> list[tuple[str, np.ndarray]]:
    """Split the pooled arrays into per-day boolean masks.

    The dump script's `guard.per_day_scene_counts` is a per-BATCH list, not a
    per-day list: with batch_size=32 and 12103 scenes it has 379 entries of 32.
    Using it as a day mapping would silently produce garbage, so we do NOT.

    Instead we derive boundaries from the source of truth.  `SceneDataset` is
    constructed from `[scenes/{date}.npz for date in blind_dates]` and the loader
    runs with shuffle=False, so scene rows appear in date order.  Each per-date
    npz reports its own scene count, so cumulative sums of those counts are the
    exact day boundaries.  We assert the three totals agree before proceeding.
    """
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dates = list(meta.get("blind_dates") or [])
    if not dates:
        raise SystemExit("blind_dates missing from %s" % meta_path)

    counts = []
    for date in dates:
        item = np.load(SCENES_DIR / ("%s.npz" % date))
        counts.append(int(item["history"].shape[0]))

    n_scenes_log = int(npz["scene_index"].max()) + 1
    n_scenes_meta = int(meta.get("guard", {}).get("n_scenes", -1))
    if not (sum(counts) == n_scenes_log == n_scenes_meta):
        raise SystemExit(
            "day-boundary derivation failed in %s: per-file sum=%d, "
            "log max scene=%d, meta n_scenes=%d"
            % (meta_path, sum(counts), n_scenes_log, n_scenes_meta))

    scene_index = npz["scene_index"]
    edges = [0] + list(np.cumsum(counts))
    out = []
    for i, date in enumerate(dates):
        mask = (scene_index >= edges[i]) & (scene_index < edges[i + 1])
        out.append((date, mask))
    return out


def collect(variant: str):
    """Return {date: {metric: [per-seed values]}}."""
    per_day: dict[str, dict[str, list[float]]] = {}
    for seed in SEEDS:
        d = LOG_DIR / ("seed_%d" % seed) / variant
        npz_path = d / "events.npz"
        meta_path = d / "event_log_meta.json"
        if not npz_path.exists() or not meta_path.exists():
            continue
        npz = np.load(npz_path)
        for date, mask in day_slices(meta_path, npz):
            ys = npz["y_severe"][mask].astype(np.float64)
            ps = npz["p_severe"][mask].astype(np.float64)
            ym = npz["y_moderate"][mask].astype(np.float64)
            pm = npz["p_moderate"][mask].astype(np.float64)
            slot = per_day.setdefault(date, {k: [] for k in
                                             ("severe_auprc", "severe_brier", "severe_ece",
                                              "moderate_auprc", "moderate_brier",
                                              "moderate_ece", "severe_pos_rate")})
            slot["severe_auprc"].append(auprc(ys, ps))
            slot["severe_brier"].append(brier(ys, ps))
            slot["severe_ece"].append(ece(ys, ps))
            slot["moderate_auprc"].append(auprc(ym, pm))
            slot["moderate_brier"].append(brier(ym, pm))
            slot["moderate_ece"].append(ece(ym, pm))
            slot["severe_pos_rate"].append(float(ys.mean()))
    return per_day


def main() -> None:
    a = collect(VARIANTS[0])
    b = collect(VARIANTS[1])
    dates = sorted(set(a) & set(b))
    metrics = ("severe_auprc", "severe_brier", "severe_ece",
               "moderate_auprc", "moderate_brier", "moderate_ece")

    report = {"dates": dates, "per_day": {}, "sign_summary": {}}
    print("Per-day mean across 10 paired seeds (A = constrained_feedback, B = no_feedback)")
    print("Δ convention: A - B; for AUPRC higher is better, for Brier/ECE lower is better\n")
    header = "%-12s %10s %10s %10s | %10s %10s" % (
        "date", "sevAUPRC", "modAUPRC", "sevECE", "d_sevAU", "d_modAU")
    print(header)
    print("-" * len(header))

    for date in dates:
        row = {"severe_pos_rate": float(np.mean(a[date]["severe_pos_rate"]))}
        for m in metrics:
            va = float(np.mean(a[date][m]))
            vb = float(np.mean(b[date][m]))
            row[m] = {"A": va, "B": vb, "delta": va - vb}
        report["per_day"][date] = row
        print("%-12s %10.4f %10.4f %10.4f | %+10.4f %+10.4f"
              % (date, row["severe_auprc"]["A"], row["moderate_auprc"]["A"],
                 row["severe_ece"]["A"], row["severe_auprc"]["delta"],
                 row["moderate_auprc"]["delta"]))

    print("\nSign of Δ (A - B) per day")
    for m in metrics:
        deltas = [report["per_day"][d][m]["delta"] for d in dates]
        neg = sum(1 for x in deltas if x < 0)
        pos = sum(1 for x in deltas if x > 0)
        better = "higher" if "auprc" in m else "lower"
        # for AUPRC, A better means delta > 0; for error metrics, A better means delta < 0
        a_better = pos if better == "higher" else neg
        report["sign_summary"][m] = {
            "delta_negative_days": neg, "delta_positive_days": pos,
            "A_better_days": a_better, "n_days": len(dates),
        }
        print("  %-16s A better on %d/%d days (Δ<0: %d, Δ>0: %d)"
              % (m, a_better, len(dates), neg, pos))

    out = Path("/root/kaorf/runs/event_log_20260924/analysis/F4_PER_DAY.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
