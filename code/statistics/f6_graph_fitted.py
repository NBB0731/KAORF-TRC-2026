#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fitted post-hoc calibration for the F2-C3 graph baseline.

Same question the F6 stage asked of the plain Transformer: the graph baseline
is badly calibrated as deployed (blind severe ECE 0.0228 vs KAORF 0.0072), so
does simple post-hoc fitting close the gap?

Design (identical to the F6 protocol, applied to one architecture)
-----------------------------------------------------------------
* fitting uses the **validation** log only, per (seed, channel);
* the frozen map is applied to the **already-exported blind** log — no new
  forward pass is performed here;
* the three maps (temperature by golden-section search, Platt, isotonic) are
  imported verbatim from ``f6_fitted_calibration.py``, and ECE uses the same
  10-bin estimator as every other table in the paper.

Scope note that must be carried into the manuscript
---------------------------------------------------
KAORF's *validation-side* per-event logs were produced on the retired compute
instance and are not recoverable offline, so the KAORF-vs-graph comparison
under fitting is **descriptive** (means quoted from Table VIII), not a new
paired seed-level test. What *is* paired here is raw → fitted within the graph
baseline, across the same ten seeds.

Usage
-----
    python scripts/f6_graph_fitted.py \
        --log-root runs/event_log_20260925_graph \
        --out runs/event_log_20260925_graph/fitted
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
WORK = SCRIPT_DIR.parent
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
SCRIPTS = WORK / "training"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import f6_fitted_calibration as f6  # noqa: E402
import dump_event_log as base  # noqa: E402

SEEDS = (17, 29, 42, 71, 101, 131, 211, 307, 509, 887)
CHANNELS = ("severe", "moderate")
METHODS = ("temperature", "platt", "isotonic")
VARIANT = "graph"


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def fmt_p(value: float) -> str:
    return f"{value:.1e}" if value < 1e-4 else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--val-tag", default="val")
    parser.add_argument("--blind-tag", default="blind")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    val_root = args.log_root / args.val_tag
    blind_root = args.log_root / args.blind_tag
    args.out.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    per_seed: dict = {}
    for seed in args.seeds:
        val_path = val_root / f"seed_{seed}" / VARIANT / "events.npz"
        blind_path = blind_root / f"seed_{seed}" / VARIANT / "events.npz"
        if not val_path.is_file() or not blind_path.is_file():
            failures.append(f"missing log seed {seed}")
            continue
        val = np.load(val_path)
        blind = np.load(blind_path)
        for key in ("y_severe", "y_moderate"):
            if not blind[key].any():
                failures.append(f"no positives seed {seed} {key}")

        record: dict = {"seed": seed, "channels": {}}
        for channel in CHANNELS:
            p_val = val[f"p_{channel}"].astype(np.float64)
            y_val = val[f"y_{channel}"].astype(bool)
            p_blind = blind[f"p_{channel}"].astype(np.float64)
            y_blind = blind[f"y_{channel}"].astype(bool)
            z_val = f6.logit(p_val)

            fitted = {
                "temperature": f6.fit_temperature(z_val, y_val),
                "platt": f6.fit_platt(z_val, y_val),
                "isotonic": f6.fit_isotonic(p_val, y_val),
            }
            raw = base.aggregate(y_blind, p_blind)
            entry = {
                "raw": raw,
                "val_positive_rate": float(y_val.mean()),
                "blind_positive_rate": float(y_blind.mean()),
                "n_events_val": int(y_val.size),
                "n_events_blind": int(y_blind.size),
                "fitted": {},
            }
            for method in METHODS:
                p_fit = f6.apply_method(method, fitted[method], p_blind)
                if not np.isfinite(p_fit).all() or p_fit.min() < 0 or p_fit.max() > 1:
                    failures.append(f"bad fitted prob seed {seed} {channel} {method}")
                entry["fitted"][method] = base.aggregate(y_blind, p_fit)
            record["channels"][channel] = entry
        per_seed[seed] = record
        say(f"seed {seed}: raw ECE "
            f"{record['channels']['severe']['raw']['ece']:.6f} -> platt "
            f"{record['channels']['severe']['fitted']['platt']['ece']:.6f}")

    # Paired raw -> fitted, across seeds, per channel and method.
    comparisons: dict = {}
    for channel in CHANNELS:
        comparisons[channel] = {}
        raw_ece = [per_seed[s]["channels"][channel]["raw"]["ece"] for s in args.seeds
                   if s in per_seed]
        raw_brier = [per_seed[s]["channels"][channel]["raw"]["brier"] for s in args.seeds
                     if s in per_seed]
        for method in METHODS:
            fit_ece = [per_seed[s]["channels"][channel]["fitted"][method]["ece"]
                       for s in args.seeds if s in per_seed]
            fit_brier = [per_seed[s]["channels"][channel]["fitted"][method]["brier"]
                         for s in args.seeds if s in per_seed]
            diff = np.asarray(fit_ece) - np.asarray(raw_ece)
            md = float(diff.mean())
            sd = float(diff.std(ddof=1))
            t_stat, p_t = stats.ttest_rel(fit_ece, raw_ece)
            try:
                w_stat, p_w = stats.wilcoxon(fit_ece, raw_ece)
            except ValueError:
                w_stat, p_w = float("nan"), float("nan")
            comparisons[channel][method] = {
                "raw_ece_mean": float(np.mean(raw_ece)),
                "fitted_ece_mean": float(np.mean(fit_ece)),
                "mean_diff": md,
                "sd_diff": sd,
                "cohens_dz": md / sd if sd > 0 else float("inf"),
                "t": float(t_stat), "p_t": float(p_t),
                "wilcoxon_W": float(w_stat), "p_w": float(p_w),
                "fitted_better_seeds": int((diff < 0).sum()),
                "n": len(diff),
                "raw_brier_mean": float(np.mean(raw_brier)),
                "fitted_brier_mean": float(np.mean(fit_brier)),
                "raw_ece_per_seed": [float(v) for v in raw_ece],
                "fitted_ece_per_seed": [float(v) for v in fit_ece],
            }

    payload = {
        "analysis": "F6 extension: fitted calibration for the F2-C3 graph baseline",
        "variant": VARIANT,
        "seeds": list(args.seeds),
        "channels": list(CHANNELS),
        "methods": list(METHODS),
        "fit_partition": args.val_tag,
        "eval_partition": args.blind_tag,
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "comparisons": comparisons,
        "failures": failures,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out / "F6_GRAPH_FITTED.json").write_text(json.dumps(payload, indent=2),
                                                   encoding="utf-8")

    lines = ["# Fitted calibration, F2-C3 graph baseline", "",
             "Fit on the validation log, applied frozen to the blind log; the same "
             "three maps and the same 10-bin ECE estimator as Table VII/VIII.", ""]
    for channel in CHANNELS:
        lines.append(f"## {channel} channel")
        lines.append("")
        lines.append("| regime | ECE mean | Brier mean | Δ ECE vs raw | seeds improved | t | p(t) | W | p(W) | dz |")
        lines.append("|" + "---|" * 10)
        raw = comparisons[channel]["platt"]["raw_ece_mean"]
        lines.append(f"| raw (as deployed) | {raw:.4f} | "
                     f"{comparisons[channel]['platt']['raw_brier_mean']:.4f} | — | — | — | — | — | — | — |")
        for method in METHODS:
            c = comparisons[channel][method]
            w_text = "-" if math.isnan(c["wilcoxon_W"]) else f"{c['wilcoxon_W']:.1f}"
            lines.append(
                f"| {method} | {c['fitted_ece_mean']:.4f} | {c['fitted_brier_mean']:.4f} | "
                f"{c['mean_diff']:+.4f} | {c['fitted_better_seeds']}/{c['n']} | "
                f"{c['t']:.3f} | {fmt_p(c['p_t'])} | {w_text} | {fmt_p(c['p_w'])} | "
                f"{c['cohens_dz']:+.2f} |")
        lines.append("")
    (args.out / "F6_GRAPH_FITTED.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    if failures:
        say(f"FAILED with {len(failures)} problem(s)")
        for problem in failures:
            say(f"  - {problem}")
        return 2
    say(f"done, wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
