# -*- coding: utf-8 -*-
"""Day-clustered bootstrap for the paired comparison at the two external ports.

Twelve calendar dates are shared by both ports and by all ten paired seeds; the
dates, not the seeds, are resampled. Design mirrors
``day_clustered_ade_fde.py``: 200,000 draws, RNG seed 20260924, equal-date
weighting, with the pooled-seed estimate reported alongside.

Inputs
------
results/external_d/sub12/<port>/seed_<s>/<arm>/metrics.json
    Written by ``evaluation/eval_external_zeroshot.py``; the ``per_day`` block
    is what this script reads.

Output
------
results/trc_20261018/day_clustered_external.json
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
ARMS = ("constrained_feedback", "constrained_no_feedback")
KEYS = (("ade_nm", "ADE"), ("fde_nm", "FDE"))
N_BOOT = 200_000
BOOT_SEED = 20260924


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ext-root", type=Path, default=PKG / "results/external_d/sub12")
    ap.add_argument("--out", type=Path, default=PKG / "results/trc_20261018/day_clustered_external.json")
    args = ap.parse_args()

    result = {"ports": {}, "seeds": SEEDS, "bootstrap_resamples": N_BOOT,
              "bootstrap_rng_seed": BOOT_SEED,
              "unit": "nautical miles, node-weighted per date",
              "cluster": "12 calendar dates shared across the ten paired seeds"}

    for port in PORTS:
        dates = None
        per_arm: dict[str, dict[str, list[list[float]]]] = {}
        for arm in ARMS:
            per_arm[arm] = {k: [] for k, _ in KEYS}
            for seed in SEEDS:
                path = args.ext_root / port / f"seed_{seed}" / arm / "metrics.json"
                if not path.is_file():
                    raise SystemExit(f"missing {path}")
                record = json.loads(path.read_text(encoding="utf-8"))
                days = record["per_day"]
                day_dates = [d["date"] for d in days]
                if dates is None:
                    dates = day_dates
                elif dates != day_dates:
                    raise SystemExit(f"date order mismatch: {port} seed {seed} {arm}")
                for key, _ in KEYS:
                    per_arm[arm][key].append([d["metrics"][key] for d in days])
            print(f"loaded {port} {arm}", flush=True)

        n_days = len(dates)
        if n_days != 12:
            raise SystemExit(f"expected 12 dates for {port}, found {n_days}")

        result["ports"][port] = {"dates": dates, "metrics": {}}
        for key, label in KEYS:
            a = np.asarray(per_arm[ARMS[0]][key], dtype=np.float64)
            b = np.asarray(per_arm[ARMS[1]][key], dtype=np.float64)
            if a.shape != (len(SEEDS), n_days):
                raise SystemExit(f"unexpected shape for {port} {key}: {a.shape}")
            daily = (a - b).mean(axis=0)
            rng = np.random.default_rng(BOOT_SEED)
            idx = rng.integers(0, n_days, size=(N_BOOT, n_days))
            boot = daily[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            seed_delta = a.mean(axis=1) - b.mean(axis=1)
            result["ports"][port]["metrics"][key] = {
                "label": label,
                "feedback_mean_nm": float(a.mean()),
                "no_feedback_mean_nm": float(b.mean()),
                "equal_day_mean_delta_nm": float(daily.mean()),
                "equal_day_ci95_nm": [float(lo), float(hi)],
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "pooled_seed_mean_delta_nm": float(seed_delta.mean()),
                "seed_level_wins": int((seed_delta < 0).sum()),
                "daily_deltas_nm": dict(zip(dates, map(float, daily))),
                "negative_days": int((daily < 0).sum()),
                "positive_days": int((daily > 0).sum()),
                "n_days": n_days,
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({p: {k: {kk: vv for kk, vv in v.items()
                              if kk in ("equal_day_mean_delta_nm", "equal_day_ci95_nm",
                                        "ci_excludes_zero", "negative_days",
                                        "positive_days", "seed_level_wins")}
                          for k, v in result["ports"][p]["metrics"].items()}
                      for p in PORTS}, indent=2), flush=True)
    print(f"written {args.out}", flush=True)


if __name__ == "__main__":
    main()
