# -*- coding: utf-8 -*-
"""Three-way paired comparison of *fitted* blind ECE: graph (C3) vs KAORF vs C1.

Inputs (all already on disk, no new forward pass):
  kaorf_f6_20260924/runs/event_log_20260924/analysis/f6_fitted_calibration.json
      per_seed[seed]["constrained_feedback/severe"|".../moderate"]
  kaorf_f6_20260924/runs/event_log_20260924_c1/analysis/f6_fitted_calibration.json
      per_seed[seed]["plain_transformer/severe"|".../moderate"]
  kaorf_c1graph_20260925/fitted/F6_GRAPH_FITTED.json
      per_seed[seed]["severe"|"moderate"]

Writes PAIRED_FITTED_3WAY.md / .json next to this file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2] / "_cloud_results"
KAORF = ROOT / "kaorf_f6_20260924/runs/event_log_20260924/analysis/f6_fitted_calibration.json"
C1 = ROOT / "kaorf_f6_20260924/runs/event_log_20260924_c1/analysis/f6_fitted_calibration.json"
GRAPH = ROOT / "kaorf_c1graph_20260925/fitted/F6_GRAPH_FITTED.json"
OUT = Path(__file__).resolve().parent

SEEDS = [17, 29, 42, 71, 101, 131, 211, 307, 509, 887]


def load_kaorf_family(path: Path, prefix: str) -> dict[tuple[int, str, str], float]:
    """Return {(seed, channel, regime): ece} with regime in raw/platt/isotonic."""
    d = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[int, str, str], float] = {}
    for seed in SEEDS:
        key = f"seed_{seed}"
        for channel in ("severe", "moderate"):
            rec = d["per_seed"][key][f"{prefix}/{channel}"]
            out[(seed, channel, "raw")] = rec["blind_raw"]["ece"]
            for method in ("temperature", "platt", "isotonic"):
                out[(seed, channel, method)] = rec["methods"][method]["blind_calibrated"]["ece"]
    return out


def load_graph(path: Path) -> dict[tuple[int, str, str], float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[int, str, str], float] = {}
    for seed in SEEDS:
        rec = d["per_seed"][str(seed)]["channels"]
        for channel in ("severe", "moderate"):
            out[(seed, channel, "raw")] = rec[channel]["raw"]["ece"]
            for method in ("temperature", "platt", "isotonic"):
                out[(seed, channel, method)] = rec[channel]["fitted"][method]["ece"]
    return out


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    sd = diff.std(ddof=1)
    t, p_t = stats.ttest_rel(a, b)
    try:
        w, p_w = stats.wilcoxon(a, b)
    except ValueError:
        w, p_w = float("nan"), float("nan")
    return {
        "a_mean": float(a.mean()),
        "b_mean": float(b.mean()),
        "a_sd": float(a.std(ddof=1)),
        "b_sd": float(b.std(ddof=1)),
        "delta": float(diff.mean()),
        "a_wins": int((diff < 0).sum()),  # lower is better for ECE
        "t": float(t),
        "p_t": float(p_t),
        "wilcoxon_W": float(w),
        "p_w": float(p_w),
        "dz": float(diff.mean() / sd) if sd > 0 else float("nan"),
    }


def vec(table, channel, regime):
    return np.array([table[(s, channel, regime)] for s in SEEDS], dtype=float)


def main() -> None:
    kaorf = load_kaorf_family(KAORF, "constrained_feedback")
    c1 = load_kaorf_family(C1, "c1")
    graph = load_graph(GRAPH)

    rows = []
    for channel in ("severe", "moderate"):
        for regime in ("raw", "temperature", "platt", "isotonic"):
            g = vec(graph, channel, regime)
            k = vec(kaorf, channel, regime)
            c = vec(c1, channel, regime)
            rows.append((channel, regime, "graph vs KAORF", paired(g, k)))
            rows.append((channel, regime, "graph vs C1", paired(g, c)))
            rows.append((channel, regime, "C1 vs KAORF (recheck)", paired(c, k)))

    lines = ["# Paired fitted-ECE comparison: graph (C3) vs KAORF vs C1",
             "",
             "Blind partition, ten locked seeds, paired by seed. ECE lower is better;",
             "`a_wins` counts seeds where the first model is lower.",
             "",
             "| channel | regime | comparison | a mean±sd | b mean±sd | Δ(a−b) | a wins | p(t) | p(W) | d_z |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for channel, regime, name, r in rows:
        w_text = "-" if math.isnan(r["wilcoxon_W"]) else f"{r['wilcoxon_W']:.1f}"
        lines.append(
            f"| {channel} | {regime} | {name} | {r['a_mean']:.4f}±{r['a_sd']:.4f} | "
            f"{r['b_mean']:.4f}±{r['b_sd']:.4f} | {r['delta']:+.4f} | {r['a_wins']}/10 | "
            f"{r['p_t']:.2e} | {r['p_w']:.2e} | {r['dz']:+.2f} |"
        )
    (OUT / "PAIRED_FITTED_3WAY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "PAIRED_FITTED_3WAY.json").write_text(
        json.dumps(
            [{"channel": c, "regime": g, "comparison": n, **r} for c, g, n, r in rows],
            indent=1,
        ),
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
