#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_calibration_figures.py -- reliability diagrams + PR curves from the
per-event blind-test logs written by dump_event_log.py (local re-evaluation
2026-09-24, guards passed).

Pure post-processing: reads events.npz per (seed, variant), never touches a
model or a checkpoint.

What is plotted (per severity channel)
--------------------------------------
* Reliability diagram: 10-bin calibration curve, identical binning to the
  locked pipeline (edges = linspace(0,1,11), np.digitize).  Per-seed curves are
  computed first; the plotted line is the seed mean, the band the seed min-max
  range.  A small sub-panel under each reliability diagram shows the mean bin
  occupancy (fraction of events per prediction bin, log scale).
* Precision-recall curve: per-seed precision_recall_curve, interpolated onto a
  common recall grid (201 points); line = seed mean, band = seed min-max.
  Average precision per seed is recorded in the JSON sidecar.

Outputs (in --out)
------------------
fig_calibration_pr.pdf   manuscript vector version (IEEE double column)
fig_calibration_pr.png   300 dpi preview
fig_calibration_pr_numbers.json   all plotted numbers + per-seed AP/ECE

Usage
-----
    python plot_calibration_figures.py --log <out>/blind --out <out>/figures
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
CHANNELS = ("severe", "moderate")
CHANNEL_TITLES = {
    "severe": "Severe (future min. separation $\\leq$ 0.25 nm)",
    "moderate": "Moderate (future min. separation $\\leq$ 0.50 nm)",
}

RECALL_GRID = np.linspace(0.0, 1.0, 201)
BIN_CENTERS = np.linspace(0.05, 0.95, 10)


def reliability_curve(y: np.ndarray, p: np.ndarray, bins: int = 10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    conf = np.full(bins, np.nan)
    obs = np.full(bins, np.nan)
    frac = np.zeros(bins)
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        conf[b] = p[mask].mean()
        obs[b] = y[mask].mean()
        frac[b] = mask.mean()
    return conf, obs, frac


def pr_curve(y: np.ndarray, p: np.ndarray):
    from sklearn.metrics import average_precision_score, precision_recall_curve

    prec, rec, _ = precision_recall_curve(y, p)
    ap = float(average_precision_score(y, p))
    order = np.argsort(rec)
    rec_s, prec_s = rec[order], prec[order]
    interp = np.interp(RECALL_GRID, rec_s, prec_s, left=np.nan,
                       right=prec_s[-1] if rec_s.size else np.nan)
    return ap, interp


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True,
                        help="directory containing seed_<s>/<variant>/events.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- collect all per-seed statistics first -----------------------------
    data: dict = {}
    payload: dict = {"recall_grid": RECALL_GRID.tolist(), "arms": {}}
    for variant in VARIANTS:
        arm: dict = {"seeds": args.seeds}
        for channel in CHANNELS:
            rel_conf, rel_obs, rel_frac, pr_aps, pr_curves = [], [], [], [], []
            for seed in args.seeds:
                npz = np.load(args.log / f"seed_{seed}" / variant / "events.npz")
                y = npz[f"y_{channel}"].astype(np.float64)
                p = npz[f"p_{channel}"].astype(np.float64)
                conf, obs, frac = reliability_curve(y, p)
                rel_conf.append(conf)
                rel_obs.append(obs)
                rel_frac.append(frac)
                ap, curve = pr_curve(y, p)
                pr_aps.append(ap)
                pr_curves.append(curve)
                arm.setdefault(f"{channel}_ece_per_seed", []).append(
                    float(np.sum([
                        frac[b] * abs(conf[b] - obs[b])
                        for b in range(10) if not np.isnan(obs[b])
                    ])))
            rel_conf = np.stack(rel_conf)
            rel_obs = np.stack(rel_obs)
            rel_frac = np.stack(rel_frac)
            pr_curves = np.stack(pr_curves)
            with np.errstate(invalid="ignore"):
                arm[f"{channel}_rel"] = {
                    "conf_mean": np.nanmean(rel_conf, axis=0),
                    "obs_mean": np.nanmean(rel_obs, axis=0),
                    "obs_lo": np.nanmin(rel_obs, axis=0),
                    "obs_hi": np.nanmax(rel_obs, axis=0),
                    "frac_mean": np.nanmean(rel_frac, axis=0),
                }
                arm[f"{channel}_pr"] = {
                    "mean": np.nanmean(pr_curves, axis=0),
                    "lo": np.nanmin(pr_curves, axis=0),
                    "hi": np.nanmax(pr_curves, axis=0),
                }
            arm[f"{channel}_ap_per_seed"] = pr_aps
            arm[f"{channel}_ap_mean"] = float(np.mean(pr_aps))
            arm[f"{channel}_ap_sd"] = float(np.std(pr_aps, ddof=1))
            arm[f"{channel}_bin_conf_mean"] = [
                None if np.isnan(v) else float(v)
                for v in arm[f"{channel}_rel"]["conf_mean"]]
            arm[f"{channel}_bin_obs_mean"] = [
                None if np.isnan(v) else float(v)
                for v in arm[f"{channel}_rel"]["obs_mean"]]
            arm[f"{channel}_bin_occupancy_mean"] = [
                float(v) for v in arm[f"{channel}_rel"]["frac_mean"]]
        payload["arms"][variant] = {
            k: v for k, v in arm.items() if not k.endswith(("_rel", "_pr"))
        }
        data[variant] = arm

    # ---- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(7.16, 5.6))
    outer = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.26)

    for row, channel in enumerate(CHANNELS):
        # Left column: reliability diagram + occupancy sub-panel.
        left = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, 0],
                                       height_ratios=[3.6, 1.0], hspace=0.06)
        ax_rel = fig.add_subplot(left[0])
        ax_occ = fig.add_subplot(left[1], sharex=ax_rel)
        ax_pr = fig.add_subplot(outer[row, 1])

        for variant in VARIANTS:
            rel = data[variant][f"{channel}_rel"]
            valid = ~np.isnan(rel["obs_mean"])
            color, style = COLORS[variant], STYLES[variant]
            ax_rel.fill_between(rel["conf_mean"][valid], rel["obs_lo"][valid],
                                rel["obs_hi"][valid], color=color, alpha=0.15,
                                linewidth=0)
            ax_rel.plot(rel["conf_mean"][valid], rel["obs_mean"][valid],
                        style, marker="o", ms=2.6, color=color,
                        label=LABELS[variant])
            occ = rel["frac_mean"]
            ax_occ.bar(BIN_CENTERS, np.where(occ > 0, occ, np.nan),
                       width=0.09, color="0.78", linewidth=0)

        ax_rel.plot([0, 1], [0, 1], ls=":", lw=0.8, color="0.35", zorder=0)
        ax_rel.set_xlim(0, 1)
        ax_rel.set_ylim(0, 1)
        ax_rel.set_ylabel("Observed frequency")
        ax_rel.set_title(CHANNEL_TITLES[channel], fontsize=8, pad=4)
        ax_rel.tick_params(labelbottom=False)
        ax_rel.legend(loc="upper left", frameon=False)
        ax_occ.set_yscale("log")
        ax_occ.set_ylim(1e-4, 1.0)
        ax_occ.set_yticks([1e-4, 1e-2, 1])
        ax_occ.set_yticklabels(["$10^{-4}$", "$10^{-2}$", "1"])
        ax_occ.set_xlim(0, 1)
        ax_occ.set_xlabel("Mean predicted probability")
        ax_occ.set_ylabel("Events / bin")

        for variant in VARIANTS:
            pr = data[variant][f"{channel}_pr"]
            color, style = COLORS[variant], STYLES[variant]
            ax_pr.fill_between(RECALL_GRID, pr["lo"], pr["hi"], color=color,
                               alpha=0.15, linewidth=0)
            ax_pr.plot(RECALL_GRID, pr["mean"], style, color=color,
                       label=LABELS[variant])
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1.02)
        ax_pr.legend(loc="lower left", frameon=False)

    pdf_path = args.out / "fig_calibration_pr.pdf"
    png_path = args.out / "fig_calibration_pr.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    (args.out / "fig_calibration_pr_numbers.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {args.out / 'fig_calibration_pr_numbers.json'}")

    for variant in VARIANTS:
        s = payload["arms"][variant]
        print(f"{variant}: severe AP {s['severe_ap_mean']:.4f}±{s['severe_ap_sd']:.4f}, "
              f"severe ECE {np.mean(s['severe_ece_per_seed']):.4f}; "
              f"moderate AP {s['moderate_ap_mean']:.4f}±{s['moderate_ap_sd']:.4f}, "
              f"moderate ECE {np.mean(s['moderate_ece_per_seed']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
