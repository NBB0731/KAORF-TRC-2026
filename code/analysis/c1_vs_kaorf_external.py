# -*- coding: utf-8 -*-
"""External-port statistics for the retrained plain Transformer (C1).

C1 is the archived New York baseline of the manuscript's Tables 8-10. For the
external ports it was retrained from the same code, partition, selection rule
and seed set, because the archived checkpoints were serialised on the original
GPU and were not reused across ports. This script compares that retrained C1
with the two KAORF arms on the twelve shared dates of each port.

Three contrasts are produced per port and per metric, all paired by seed and
resampled by date (200,000 draws, RNG seed 20260924):
  * KAORF feedback     - C1
  * KAORF no-feedback  - C1
  * KAORF feedback     - KAORF no-feedback

Inputs
------
results/external_d/sub12/<port>/seed_<s>/<arm>/metrics.json
results/external_d/sub12/<port>/seed_<s>/plain_transformer/metrics.json

Output
------
results/trc_20261018/c1_external_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CODE = next(p for p in Path(__file__).resolve().parents if p.name == "code")
PKG = CODE.parent

SEEDS = (17, 29, 42, 71, 101, 131, 211, 307, 509, 887)
PORTS = ("houston_galveston", "chesapeake")
METRICS = ("ade_nm", "fde_nm")
N_BOOT = 200_000
BOOT_SEED = 20260924


def load_arm(root: Path, port: str, arm: str) -> dict:
    """{seed: {date: {"ade_nm": x, "fde_nm": y, "node_samples": n}}}"""
    out = {}
    for seed in SEEDS:
        path = root / port / f"seed_{seed}" / arm / "metrics.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        blob = json.loads(path.read_text(encoding="utf-8"))
        per_day = {}
        for row in blob["per_day"]:
            per_day[row["date"]] = {
                "ade_nm": float(row["metrics"]["ade_nm"]),
                "fde_nm": float(row["metrics"]["fde_nm"]),
                "node_samples": int(row["node_samples"]),
            }
        out[seed] = per_day
    return out


def node_weighted(per_day: dict, metric: str) -> float:
    num = 0.0
    den = 0.0
    for day in per_day.values():
        num += day[metric] * day["node_samples"]
        den += day["node_samples"]
    return num / den


def paired_stats(a: dict, b: dict, metric: str) -> dict:
    dates = sorted(a[SEEDS[0]].keys())
    for seed in SEEDS:
        if sorted(a[seed].keys()) != dates or sorted(b[seed].keys()) != dates:
            raise SystemExit(f"date mismatch on seed {seed}")

    daily = np.array([np.mean([a[s][d][metric] - b[s][d][metric] for s in SEEDS]) for d in dates])
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(dates), size=(N_BOOT, len(dates)))
    boot = daily[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    per_seed = np.array([node_weighted(a[s], metric) - node_weighted(b[s], metric) for s in SEEDS])
    return {
        "equal_day_mean_delta_nm": float(daily.mean()),
        "equal_day_ci95_nm": [float(lo), float(hi)],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "pooled_seed_mean_delta_nm": float(per_seed.mean()),
        "seed_level_wins": int((per_seed < 0).sum()),
        "negative_days": int((daily < 0).sum()),
        "positive_days": int((daily > 0).sum()),
        "n_days": len(dates),
        "daily_deltas_nm": {d: float(v) for d, v in zip(dates, daily)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ext-root", type=Path, default=PKG / "results/external_d/sub12")
    ap.add_argument("--out", type=Path, default=PKG / "results/trc_20261018/c1_external_stats.json")
    args = ap.parse_args()

    report = {"seeds": list(SEEDS), "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
              "sign_convention": "delta = first arm minus second arm; negative favours the first",
              "ports": {}}

    for port in PORTS:
        c1 = load_arm(args.ext_root, port, "plain_transformer")
        fb = load_arm(args.ext_root, port, "constrained_feedback")
        nf = load_arm(args.ext_root, port, "constrained_no_feedback")
        block = {
            "c1_headline": {m: float(np.mean([node_weighted(c1[s], m) for s in SEEDS])) for m in METRICS},
            "kaorf_feedback_headline": {m: float(np.mean([node_weighted(fb[s], m) for s in SEEDS])) for m in METRICS},
            "kaorf_no_feedback_headline": {m: float(np.mean([node_weighted(nf[s], m) for s in SEEDS])) for m in METRICS},
            "metrics": {},
        }
        for m in METRICS:
            block["metrics"][m] = {
                "feedback_minus_c1": paired_stats(fb, c1, m),
                "no_feedback_minus_c1": paired_stats(nf, c1, m),
                "feedback_minus_no_feedback": paired_stats(fb, nf, m),
            }
        report["ports"][port] = block
        print(f"done {port}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"written {args.out}", flush=True)


if __name__ == "__main__":
    main()
