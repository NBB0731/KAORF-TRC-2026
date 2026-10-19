# -*- coding: utf-8 -*-
"""Day-clustered bootstrap for ADE/FDE on the locked New York blind test.

The seed-level paired test treats a seed as the sampling unit. Traffic varies
far more between calendar days than between optimisation runs, so this script
reports an interval that resamples the eight shared blind dates instead: the
same eight dates, the same ten paired seeds, 200,000 draws, RNG seed 20260924.

Estimand: mean over the ten paired seeds of the per-date
(KAORF - no-feedback) difference, computed with each date weighted equally.

Inputs
------
results/events_blind_20260924/seed_<s>/<arm>/traj_errors.npz
    Per-scene trajectory errors, archived from the locked checkpoints.
    The field is named ``err_nm`` for historical reasons but stores metres;
    the conversion to nautical miles is explicit below.
$KAORF_DATA_DIR/new_york/<date>.npz   (default: <package>/data/new_york)
    Scene archives, used only to recover the per-date scene boundaries.

Output
------
results/trc_20261018/day_clustered_ade_fde_ny.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

CODE = next(p for p in Path(__file__).resolve().parents if p.name == "code")
PKG = CODE.parent

SEEDS = (17, 29, 42, 71, 101, 131, 211, 307, 509, 887)
ARMS = ("constrained_feedback", "constrained_no_feedback")
METRICS = ("ade_nm", "fde_nm")
N_BOOT = 200_000
BOOT_SEED = 20260924


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, default=PKG / "results/events_blind_20260924")
    ap.add_argument("--scenes-dir", type=Path,
                    default=Path(os.environ.get("KAORF_DATA_DIR", PKG / "data")) / "new_york")
    ap.add_argument("--out", type=Path, default=PKG / "results/trc_20261018/day_clustered_ade_fde_ny.json")
    args = ap.parse_args()

    first_meta = json.loads((args.logs / f"seed_{SEEDS[0]}" / ARMS[0] /
                             "traj_meta.json").read_text(encoding="utf-8"))
    dates = first_meta["blind_dates"]
    if len(dates) != 8:
        raise SystemExit(f"expected the eight locked blind dates, found {len(dates)}")

    counts = [len(np.load(args.scenes_dir / f"{d}.npz")["history"]) for d in dates]
    edges = np.r_[0, np.cumsum(counts)]
    if int(edges[-1]) != 12103:
        raise SystemExit(f"scene total {int(edges[-1])} does not match the locked 12,103")

    values: dict[str, dict[str, list[list[float]]]] = {}
    pooled: dict[str, dict[str, list[float]]] = {}
    scene_counts = None
    for arm in ARMS:
        values[arm] = {m: [] for m in METRICS}
        pooled[arm] = {m: [] for m in METRICS}
        for seed in SEEDS:
            directory = args.logs / f"seed_{seed}" / arm
            meta = json.loads((directory / "traj_meta.json").read_text(encoding="utf-8"))
            if meta["blind_dates"] != dates:
                raise SystemExit(f"date order mismatch for seed {seed} {arm}")
            with np.load(directory / "traj_errors.npz") as z:
                index = z["scene_index"]
                err_nm = z["err_nm"].astype(np.float64) / 1852.0
                if err_nm.shape[0] != index.shape[0] or err_nm.shape[1] != 6:
                    raise SystemExit(f"unexpected error array shape for seed {seed} {arm}")
                if int(index.max()) + 1 != int(edges[-1]):
                    raise SystemExit(f"scene index out of range for seed {seed} {arm}")
                daily_n = [int(np.count_nonzero((index >= edges[i]) & (index < edges[i + 1])))
                           for i in range(len(dates))]
                if scene_counts is None:
                    scene_counts = daily_n
                elif scene_counts != daily_n:
                    raise SystemExit(f"per-date scene counts differ for seed {seed} {arm}")
                for j, metric in enumerate(METRICS):
                    col = err_nm if j == 0 else err_nm[:, 5]
                    pooled[arm][metric].append(float(col.mean()))
                    daily = [float(col[(index >= edges[i]) & (index < edges[i + 1])].mean())
                             for i in range(len(dates))]
                    values[arm][metric].append(daily)
            print(f"loaded seed={seed} arm={arm}", flush=True)

    rng = np.random.default_rng(BOOT_SEED)
    result = {
        "dates": dates,
        "scenes_per_day": counts,
        "seeds": SEEDS,
        "bootstrap_resamples": N_BOOT,
        "bootstrap_rng_seed": BOOT_SEED,
        "estimand": "equal-date mean of the paired (feedback - no-feedback) difference",
        "unit": "nautical miles, node-weighted within a date",
        "metrics": {},
    }
    for metric in METRICS:
        a = np.asarray(values[ARMS[0]][metric], dtype=np.float64)
        b = np.asarray(values[ARMS[1]][metric], dtype=np.float64)
        daily = (a - b).mean(axis=0)
        indices = rng.integers(0, len(dates), size=(N_BOOT, len(dates)))
        boot = daily[indices].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        seed_delta = np.asarray(pooled[ARMS[0]][metric]) - np.asarray(pooled[ARMS[1]][metric])
        result["metrics"][metric] = {
            "feedback_mean_nm": float(np.mean(pooled[ARMS[0]][metric])),
            "no_feedback_mean_nm": float(np.mean(pooled[ARMS[1]][metric])),
            "pooled_seed_mean_delta_nm": float(seed_delta.mean()),
            "equal_day_mean_delta_nm": float(daily.mean()),
            "equal_day_ci95_nm": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "daily_deltas_nm": dict(zip(dates, map(float, daily))),
            "negative_days": int((daily < 0).sum()),
            "positive_days": int((daily > 0).sum()),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2), flush=True)
    print(f"written {args.out}", flush=True)


if __name__ == "__main__":
    main()
