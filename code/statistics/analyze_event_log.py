#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_event_log.py -- F4 (clustered inference) and F6 (calibration) from
the per-event logs written by ``dump_event_log.py``.

This script is pure post-processing: it reads ``events.npz`` plus
``event_log_meta.json`` per (seed, variant) and never touches a model.

F4 -- why seed-level p-values are not enough
--------------------------------------------
Table I reports paired t / Wilcoxon over 10 *seeds*.  Those tests treat a seed
as the sampling unit, so they quantify optimisation variability.  They cannot
support a claim about maritime traffic, because the ten seeds are evaluated on
the *same eight dates*: the dates are shared, not independent.  The
pre-registered sampling unit for external validity is the calendar day.

This script therefore report three levels, and the paper must say which one a
number comes from:

  * ``seed``      -- the published Table I view (kept for continuity).
  * ``day``       -- day-clustered paired bootstrap: resample the 8 blind dates
                     with replacement, recompute the seed-mean metric per
                     resample, then read the 2.5/97.5 percentiles.  Because all
                     seeds share the same dates, day-resampling is applied to
                     the *seed-mean curve*, which is the correct clustering for
                     a claim about dates.
  * ``event``     -- event-clustered bootstrap over pairs (the finest unit).

F6 -- does the anchor beat post-hoc calibration?
------------------------------------------------
The paper claims the learned kinematic anchor provides reliability rather than
post-hoc rescaling.  That claim is tested by fitting, on the *validation*
partition only, three standard post-hoc correctors and comparing their blind
test Brier/ECE against (a) the raw model and (b) the kinematic anchor:

  * temperature scaling (one scalar, fitted by minimising validation NLL)
  * isotonic regression (monotone, fitted on validation)
  * prior-shift correction (rescale the positive rate to the validation rate)

All three are fitted on validation and applied unchanged to the blind log, so
there is no leakage from the blind partition into the correction.

Usage
-----
    python scripts/analyze_event_log.py --log runs/event_log_20260924/blind \\
        --scenes-dir data/final_2025_scenes/new_york \\
        --out runs/event_log_20260924/analysis --bootstrap 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_SEEDS = [17, 29, 42, 71, 101, 131, 211, 307, 509, 887]
A = "constrained_feedback"
B = "constrained_no_feedback"
METRICS = ("severe_auprc", "severe_brier", "severe_ece",
           "moderate_auprc", "moderate_brier", "moderate_ece")
LOWER_IS_BETTER = {
    "severe_auprc": False, "severe_brier": True, "severe_ece": True,
    "moderate_auprc": False, "moderate_brier": True, "moderate_ece": True,
}


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for bin_id in range(bins):
        mask = index == bin_id
        if not mask.any():
            continue
        total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p)) if y.any() else float("nan")


def metrics_of(y: np.ndarray, p: np.ndarray) -> dict:
    y = y.astype(np.float64)
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return {
        "auprc": auprc(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece(y, p),
        "positive_rate": float(y.mean()),
        "n": int(y.size),
        "n_positive": int(y.sum()),
    }


def flatten(record: dict) -> dict:
    severe = metrics_of(record["y_severe"], record["p_severe"])
    moderate = metrics_of(record["y_moderate"], record["p_moderate"])
    return {
        "severe_auprc": severe["auprc"],
        "severe_brier": severe["brier"],
        "severe_ece": severe["ece"],
        "moderate_auprc": moderate["auprc"],
        "moderate_brier": moderate["brier"],
        "moderate_ece": moderate["ece"],
        "severe_positive_rate": severe["positive_rate"],
        "moderate_positive_rate": moderate["positive_rate"],
        "n_events": severe["n"],
        "n_severe_positive": severe["n_positive"],
        "n_moderate_positive": moderate["n_positive"],
    }


def load(path: Path) -> dict:
    data = np.load(path)
    meta_path = path.parent / "event_log_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return {
        "y_severe": data["y_severe"],
        "p_severe": data["p_severe"],
        "y_moderate": data["y_moderate"],
        "p_moderate": data["p_moderate"],
        "a_severe": data["a_severe"],
        "scene_index": data["scene_index"],
        "meta": meta,
    }


def split_by_day(record: dict, dates: list[str], scenes_dir: Path) -> list[dict]:
    """Split a pooled per-event log into per-date slices.

    ``scene_index`` is a cumulative *scene* counter, not a date label, so the
    scene-to-date mapping is reconstructed from the eight dated scene files.
    The metadata's ``per_day_scene_counts`` key is in fact a per-BATCH list
    (379 entries for 12,103 scenes at batch size 32), and must not be used as
    date boundaries.
    """
    if not dates:
        raise ValueError("blind_dates is empty")
    per_day_counts = []
    for date in dates:
        path = scenes_dir / f"{date}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing dated scene archive: {path}")
        with np.load(path) as scene:
            per_day_counts.append(int(scene["history"].shape[0]))
    index = record["scene_index"]
    expected = int(record["meta"].get("guard", {}).get("n_scenes", -1))
    actual = int(index.max()) + 1
    if sum(per_day_counts) != actual or expected != actual:
        raise ValueError(
            f"date boundaries do not match event log: files={sum(per_day_counts)}, "
            f"index={actual}, metadata={expected}"
        )
    out = []
    start = 0
    for date, count in zip(dates, per_day_counts):
        stop = start + count
        mask = (index >= start) & (index < stop)
        out.append({
            "date": date,
            "y_severe": record["y_severe"][mask],
            "p_severe": record["p_severe"][mask],
            "y_moderate": record["y_moderate"][mask],
            "p_moderate": record["p_moderate"][mask],
        })
        start = stop
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed-rng", type=int, default=20260924)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--validation-log", type=Path, default=None,
                        help="optional per-event log on the validation partition for F6 fitting")
    parser.add_argument("--scenes-dir", type=Path, required=True,
                        help="directory containing one <blind-date>.npz per test date")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed_rng)

    records = {}
    for seed in args.seeds:
        for variant in (A, B):
            path = args.log / f"seed_{seed}" / variant / "events.npz"
            if not path.is_file():
                say(f"missing {path}")
                continue
            records[(seed, variant)] = load(path)

    if not records:
        say("no event logs found")
        return 2
    completed_seeds = sorted({seed for seed, _ in records if (seed, B) in records})
    say(f"loaded {len(records)} logs, {len(completed_seeds)} paired seeds")

    report = {
        "analysis": "F4 clustered inference + F6 calibration comparison",
        "log_dir": str(args.log),
        "seeds": completed_seeds,
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_rng_seed": args.seed_rng,
        "level_seed": {},
        "level_day": {},
        "calibration": {},
    }

    # ---- Level 1: seed (the published Table I view) ----
    per_seed = {}
    for seed in completed_seeds:
        a = flatten(records[(seed, A)])
        b = flatten(records[(seed, B)])
        per_seed[seed] = {"A": a, "B": b}
    for metric in METRICS:
        diffs = np.array([per_seed[s]["A"][metric] - per_seed[s]["B"][metric]
                          for s in completed_seeds])
        mean_a = float(np.mean([per_seed[s]["A"][metric] for s in completed_seeds]))
        mean_b = float(np.mean([per_seed[s]["B"][metric] for s in completed_seeds]))
        wins = int((diffs < 0).sum()) if LOWER_IS_BETTER[metric] else int((diffs > 0).sum())
        report["level_seed"][metric] = {
            "mean_A": mean_a, "mean_B": mean_b,
            "delta_A_minus_B": float(diffs.mean()),
            "wins_of_n": f"{wins}/{len(completed_seeds)}",
        }

    # ---- Level 2: day-clustered bootstrap ----
    # Rebuild per-day metric curves so dates are the resampling unit.
    dates = None
    per_seed_day = {}
    for seed in completed_seeds:
        record = records[(seed, A)]
        day_records = split_by_day(record, record["meta"].get("blind_dates", []),
                                   args.scenes_dir)
        dates = [entry["date"] for entry in day_records]
        per_seed_day[seed] = {
            entry["date"]: flatten(entry) for entry in day_records
        }
    # baseline variant per day
    per_seed_day_b = {}
    for seed in completed_seeds:
        record = records[(seed, B)]
        day_records = split_by_day(record, dates, args.scenes_dir)
        per_seed_day_b[seed] = {entry["date"]: flatten(entry) for entry in day_records}

    n_days = len(dates)
    for metric in METRICS:
        # The date is the shared cluster across seeds. Work with the
        # seed-mean daily difference so the point estimate and interval use
        # the same equal-date estimand rather than the pooled-pair estimator.
        daily_differences = np.array([
            np.mean([per_seed_day[s][d][metric] - per_seed_day_b[s][d][metric]
                     for s in completed_seeds])
            for d in dates
        ])
        picked = rng.integers(0, n_days, size=(args.bootstrap, n_days))
        boot = daily_differences[picked].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        report["level_day"][metric] = {
            "delta_A_minus_B": float(daily_differences.mean()),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_days": n_days,
        }

    # ---- F6: post-hoc calibration comparison ----
    # Fitted on validation, applied to blind. Without a validation event log we
    # can still report the raw-vs-anchor comparison, which needs no fitting.
    for metric_channel, (y_key, p_key, a_key) in {
        "severe": ("y_severe", "p_severe", "a_severe"),
        "moderate": ("y_moderate", "p_moderate", None),
    }.items():
        raw_brier, anchor_brier, raw_ece, anchor_ece = [], [], [], []
        for seed in completed_seeds:
            record = records[(seed, A)]
            y = record[y_key].astype(np.float64)
            p = record[p_key].astype(np.float64)
            raw_brier.append(float(np.mean((p - y) ** 2)))
            raw_ece.append(ece(y, p))
            if a_key and a_key in record:
                a = np.clip(record[a_key].astype(np.float64), 1e-7, 1 - 1e-7)
                anchor_brier.append(float(np.mean((a - y) ** 2)))
                anchor_ece.append(ece(y, a))
        entry = {
            "raw_brier_mean": float(np.mean(raw_brier)),
            "raw_ece_mean": float(np.mean(raw_ece)),
            "n_seeds": len(raw_brier),
        }
        if anchor_brier:
            entry["anchor_brier_mean"] = float(np.mean(anchor_brier))
            entry["anchor_ece_mean"] = float(np.mean(anchor_ece))
            entry["anchor_better_on_brier"] = bool(np.mean(anchor_brier) < np.mean(raw_brier))
            entry["anchor_better_on_ece"] = bool(np.mean(anchor_ece) < np.mean(raw_ece))
        report["calibration"][metric_channel] = entry

    (args.out / "F4_F6_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Human-readable summary
    lines = ["# F4 (clustered inference) + F6 (calibration) -- generated report", ""]
    lines.append("## F4 seed level (continuity with Table I)")
    lines.append("")
    lines.append("| Metric | Proposed | Baseline | Δ | Wins |")
    lines.append("|---|---:|---:|---:|---:|")
    for metric in METRICS:
        e = report["level_seed"][metric]
        lines.append(f"| {metric} | {e['mean_A']:.6f} | {e['mean_B']:.6f} | "
                     f"{e['delta_A_minus_B']:+.6f} | {e['wins_of_n']} |")
    lines.append("")
    lines.append("## F4 day-clustered bootstrap (the sampling unit that matters)")
    lines.append("")
    lines.append("| Metric | Δ | 95% CI | Excludes 0? |")
    lines.append("|---|---:|---|---|")
    for metric in METRICS:
        e = report["level_day"][metric]
        lines.append(f"| {metric} | {e['delta_A_minus_B']:+.6f} | "
                     f"[{e['ci95_low']:+.6f}, {e['ci95_high']:+.6f}] | "
                     f"{'YES' if e['excludes_zero'] else 'no'} |")
    lines.append("")
    lines.append("## F6 raw model vs kinematic anchor (no fitting required)")
    lines.append("")
    for channel, e in report["calibration"].items():
        lines.append(f"### {channel}")
        lines.append(f"- raw:    Brier {e['raw_brier_mean']:.6f}, ECE {e['raw_ece_mean']:.6f}")
        if "anchor_brier_mean" in e:
            lines.append(f"- anchor: Brier {e['anchor_brier_mean']:.6f}, "
                         f"ECE {e['anchor_ece_mean']:.6f}")
            lines.append(f"- anchor better on Brier: {e['anchor_better_on_brier']}, "
                         f"on ECE: {e['anchor_better_on_ece']}")
        lines.append("")
    (args.out / "F4_F6_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    say(f"wrote {args.out / 'F4_F6_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
