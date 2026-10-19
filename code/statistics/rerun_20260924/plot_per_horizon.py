#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_per_horizon.py -- per-horizon displacement error from the locked
blind-test trajectory logs (dump_event_log_traj.py output).

Per arm: per-horizon ADE (nm) computed per seed, plotted as seed mean with
min-max band.  Also reports the paired feedback vs no_feedback difference per
horizon (seed-level mean, Cohen's dz, Wilcoxon signed-rank p) in the JSON.

Usage
-----
    python plot_per_horizon.py --log <out>/blind --out <out>/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_SEEDS = [17, 29, 42, 71, 101, 131, 211, 307, 509, 887]
VARIANTS = ("constrained_feedback", "constrained_no_feedback")
LABELS = {
    "constrained_feedback": "KAORF (kinematic-anchor feedback)",
    "constrained_no_feedback": "No feedback (same trunk)",
}
COLORS = {"constrained_feedback": "#1f77b4", "constrained_no_feedback": "#d95f02"}
STYLES = {"constrained_feedback": "-", "constrained_no_feedback": "--"}
HORIZONS_MIN = np.array([5, 10, 15, 20, 25, 30])


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per_seed = {v: np.stack([
        np.load(args.log / f"seed_{s}" / v / "traj_errors.npz")["err_nm"]
        .mean(axis=0) / 1852.0
        for s in args.seeds
    ]) for v in VARIANTS}  # (10, 6)

    payload = {"horizons_min": HORIZONS_MIN.tolist(), "arms": {}}
    for v in VARIANTS:
        m = per_seed[v].mean(axis=0)
        sd = per_seed[v].std(axis=0, ddof=1)
        payload["arms"][v] = {
            "per_seed": per_seed[v].tolist(),
            "mean": m.tolist(),
            "sd": sd.tolist(),
        }
    diff = per_seed["constrained_feedback"] - per_seed["constrained_no_feedback"]
    dz = diff.mean(axis=0) / diff.std(axis=0, ddof=1)
    pvals = [float(wilcoxon(diff[:, h]).pvalue) for h in range(6)]
    payload["paired_feedback_minus_no_feedback"] = {
        "mean": diff.mean(axis=0).tolist(),
        "dz": dz.tolist(),
        "wilcoxon_p": pvals,
        "wins": (diff < 0).sum(axis=0).tolist(),
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for v in VARIANTS:
        m = per_seed[v].mean(axis=0)
        lo = per_seed[v].min(axis=0)
        hi = per_seed[v].max(axis=0)
        ax.fill_between(HORIZONS_MIN, lo, hi, color=COLORS[v], alpha=0.15, linewidth=0)
        ax.plot(HORIZONS_MIN, m, STYLES[v], marker="o", ms=2.8, color=COLORS[v],
                label=LABELS[v])
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("Displacement error (nm)")
    ax.set_xticks(HORIZONS_MIN)
    ax.set_xlim(4, 31)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "fig_per_horizon_ade.pdf", bbox_inches="tight")
    fig.savefig(args.out / "fig_per_horizon_ade.png", dpi=300, bbox_inches="tight")
    (args.out / "fig_per_horizon_ade_numbers.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out / 'fig_per_horizon_ade.pdf'}")

    for v in VARIANTS:
        m = per_seed[v].mean(axis=0)
        print(f"{v}: " + " ".join(f"{h}min={x:.4f}" for h, x in zip(HORIZONS_MIN, m)))
    print("paired (feedback - no_feedback):")
    for h, d, z, p, w in zip(HORIZONS_MIN, diff.mean(axis=0), dz, pvals,
                             (diff < 0).sum(axis=0)):
        print(f"  {h:>2} min: d={d:+.5f} nm  dz={z:+.2f}  p={p:.4f}  fb wins {w}/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
