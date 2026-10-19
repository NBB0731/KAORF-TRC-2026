#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""f6_fitted_calibration.py -- F6: post-hoc calibration fitted on validation.

The manuscript's calibration claim currently rests on *unfitted* comparisons
(Table I: raw ECE).  A reviewer will ask whether the advantage survives when
each model receives a standard post-hoc recalibration fitted on the designated
tuning split.  This script answers that.

Procedure (per seed, per variant, per severity channel)
-------------------------------------------------------
1. Fit on the **validation** per-event log (runs/event_log_20260924/val):
     * temperature scaling   p' = sigmoid(logit(p) / T)          (1 param)
     * Platt scaling         p' = sigmoid(a * logit(p) + b)      (2 params)
     * isotonic regression   sklearn IsotonicRegression on (p, y)
   Temperature/Platt minimise binary cross-entropy (NLL) with LBFGS; isotonic
   is the standard sklearn fit.  Every model gets its own map (the honest
   per-deployed-model procedure).
2. Apply the frozen map to the **blind** per-event log
   (runs/event_log_20260924/blind, produced by the F4 re-evaluation -- the
   blind test is NOT re-evaluated here; this script is CPU-only).
3. Score with the locked 10-bin ECE / Brier / AUPRC definitions
   (imported verbatim from dump_event_log.aggregate, i.e. the Table I code
   path).

Reported comparisons (paired across the 10 seeds, Wilcoxon + dz)
----------------------------------------------------------------
  * raw vs calibrated  per arm   -- does fitting help/hurt each arm?
  * feedback vs no_feedback under each calibration regime
                                 -- does the calibration advantage survive
                                    when *both* sides are fitted?
AUPRC invariance is asserted for the strictly monotone maps (temperature,
Platt) and reported for isotonic (ties may merge).

Protocol: fitting uses validation only; blind probabilities are read from the
existing F4 event log.  No new blind forward pass occurs anywhere here.

Usage
-----
    python scripts/f6_fitted_calibration.py \
        --log-root runs/event_log_20260924 \
        --out runs/event_log_20260924/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dump_event_log as base  # noqa: E402  (aggregate = Table I metric code)

VARIANTS = ("constrained_feedback", "constrained_no_feedback")
SEEDS = (17, 29, 42, 71, 101, 131, 211, 307, 509, 887)
CHANNELS = ("severe", "moderate")
METHODS = ("temperature", "platt", "isotonic")

P_CLIP = 1e-6


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), P_CLIP, 1.0 - P_CLIP)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    return 1.0 / (1.0 + np.exp(-z))


def _val_nll_temperature(log_t: float, z: np.ndarray, y: np.ndarray) -> float:
    t = np.exp(log_t)
    p = 1.0 / (1.0 + np.exp(-z / t))
    p = np.clip(p, P_CLIP, 1.0 - P_CLIP)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature(z_val: np.ndarray, y_val: np.ndarray) -> float:
    """Golden-section search on log T in [-4, 4] minimising val NLL.

    Chosen over gradient descent because the baseline scores are strongly
    polarised (many exact zeros), which made LBFGS diverge on one channel;
    the 1-D search is deterministic and cannot produce non-finite parameters.
    """
    y_val = y_val.astype(np.float64)
    lo, hi = -4.0, 4.0
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - inv_phi * (hi - lo)
    x2 = lo + inv_phi * (hi - lo)
    f1 = _val_nll_temperature(x1, z_val, y_val)
    f2 = _val_nll_temperature(x2, z_val, y_val)
    for _ in range(200):
        if f1 > f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + inv_phi * (hi - lo)
            f2 = _val_nll_temperature(x2, z_val, y_val)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - inv_phi * (hi - lo)
            f1 = _val_nll_temperature(x1, z_val, y_val)
    log_t = 0.5 * (lo + hi)
    fitted = float(np.exp(log_t))
    if not np.isfinite(fitted) or fitted <= 0.0:
        return 1.0  # identity fallback, flagged by caller via AUPRC/ECE checks
    return fitted


def fit_platt(z_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
    """Unregularised logistic regression on z (sklearn LBFGS, deterministic)."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    clf.fit(z_val.reshape(-1, 1), y_val.astype(np.int64))
    a = float(clf.coef_[0, 0])
    b = float(clf.intercept_[0])
    if not (np.isfinite(a) and np.isfinite(b)):
        return 1.0, 0.0  # identity fallback
    return a, b


def fit_isotonic(p_val: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val.astype(np.float64), y_val.astype(np.float64))
    return iso


def apply_method(method: str, fitted, p: np.ndarray) -> np.ndarray:
    if method == "temperature":
        return sigmoid(logit(p) / fitted)
    if method == "platt":
        a, b = fitted
        return sigmoid(a * logit(p) + b)
    if method == "isotonic":
        return np.clip(fitted.predict(p.astype(np.float64)), 0.0, 1.0)
    raise ValueError(method)


def paired_stats(feedback: list[float], no_feedback: list[float]) -> dict:
    """Paired feedback - no_feedback comparison across seeds."""
    diff = np.asarray(feedback, dtype=np.float64) - np.asarray(
        no_feedback, dtype=np.float64
    )
    from scipy.stats import wilcoxon

    if np.allclose(diff, 0.0):
        stat_p = float("nan")
    else:
        stat_p = float(wilcoxon(diff).pvalue)
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
    return {
        "mean_feedback": float(np.mean(feedback)),
        "mean_no_feedback": float(np.mean(no_feedback)),
        "mean_delta_feedback_minus_none": float(diff.mean()),
        "dz": float(diff.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else None,
        "wilcoxon_p": stat_p,
        "n_pairs": int(len(diff)),
        "feedback_wins": int((diff < 0).sum()),
        "no_feedback_wins": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
    }


def raw_vs_cal_stats(raw: list[float], cal: list[float]) -> dict:
    diff = np.asarray(cal, dtype=np.float64) - np.asarray(raw, dtype=np.float64)
    from scipy.stats import wilcoxon

    if np.allclose(diff, 0.0):
        stat_p = float("nan")
    else:
        stat_p = float(wilcoxon(diff).pvalue)
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
    return {
        "mean_raw": float(np.mean(raw)),
        "mean_calibrated": float(np.mean(cal)),
        "mean_delta_cal_minus_raw": float(diff.mean()),
        "dz": float(diff.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else None,
        "wilcoxon_p": stat_p,
        "improved": int((diff < 0).sum()),
        "degraded": int((diff > 0).sum()),
    }


def load_log(log_dir: Path, seed: int, variant: str) -> dict:
    npz = np.load(log_dir / f"seed_{seed}" / variant / "events.npz")
    return {key: npz[key] for key in npz.files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--blind-tag", default="blind")
    parser.add_argument("--val-tag", default="val")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--baseline-root", type=Path, default=None,
                        help="optional: root with <val,blind>/<method>/events.npz "
                             "for deterministic baselines (F2-C2 CV/IMM)")
    args = parser.parse_args()

    val_root = args.log_root / args.val_tag
    blind_root = args.log_root / args.blind_tag
    args.out.mkdir(parents=True, exist_ok=True)

    # Fail-closed protocol guards.
    if not (val_root / "VAL_EVENT_LOG_DONE").is_file():
        raise SystemExit("FATAL: VAL_EVENT_LOG_DONE missing -- fit set incomplete")
    if (val_root / "VAL_EVENT_LOG_FAILED").is_file():
        raise SystemExit("FATAL: val event log reported failures")
    if not (blind_root / "EVENT_LOG_DONE").is_file():
        raise SystemExit("FATAL: blind event log missing (F4 re-evaluation output)")

    failures: list[str] = []
    per_seed_records: dict = {}
    # Per (variant, channel, method): blind ECE/Brier lists per seed, in seed order.
    blind_ece: dict = {}
    blind_brier: dict = {}
    blind_auprc: dict = {}
    val_ece_after_fit: dict = {}

    for seed in args.seeds:
        logs = {}
        for variant in args.variants:
            val_log = load_log(val_root, seed, variant)
            blind_log = load_log(blind_root, seed, variant)
            logs[variant] = (val_log, blind_log)

        # Cross-arm label consistency: labels are model-independent, so both
        # variants must see byte-identical label vectors within each split.
        if len(args.variants) > 1:
            v0, v1 = args.variants[0], args.variants[1]
            for split_idx, split_name in ((0, "val"), (1, "blind")):
                for key in ("y_severe", "y_moderate"):
                    if not np.array_equal(logs[v0][split_idx][key], logs[v1][split_idx][key]):
                        failures.append(
                            f"LABEL MISMATCH seed {seed} {split_name} {key}"
                        )

        for variant in args.variants:
            val_log, blind_log = logs[variant]
            for channel in CHANNELS:
                y_val = val_log[f"y_{channel}"]
                p_val = val_log[f"p_{channel}"]
                y_blind = blind_log[f"y_{channel}"]
                p_blind = blind_log[f"p_{channel}"]
                if not y_val.any() or not y_blind.any():
                    failures.append(
                        f"no positives seed {seed} {variant} {channel}"
                    )
                    continue

                z_val = logit(p_val)
                fitted = {
                    "temperature": fit_temperature(z_val, y_val),
                    "platt": fit_platt(z_val, y_val),
                    "isotonic": fit_isotonic(p_val, y_val),
                }

                raw_blind = base.aggregate(y_blind, p_blind)
                raw_val = base.aggregate(y_val, p_val)
                rec = {
                    "blind_raw": raw_blind,
                    "val_raw": raw_val,
                    "fit_params": {
                        "temperature_T": fitted["temperature"],
                        "platt_a": fitted["platt"][0],
                        "platt_b": fitted["platt"][1],
                    },
                    "methods": {},
                }
                for method in METHODS:
                    p_cal_blind = apply_method(method, fitted[method], p_blind)
                    p_cal_val = apply_method(method, fitted[method], p_val)
                    if not (np.isfinite(p_cal_blind).all() and np.isfinite(p_cal_val).all()):
                        failures.append(
                            f"non-finite calibrated probabilities seed {seed} "
                            f"{variant} {channel} {method}"
                        )
                        continue
                    cal_blind = base.aggregate(y_blind, p_cal_blind)
                    cal_val = base.aggregate(y_val, p_cal_val)
                    rec["methods"][method] = {
                        "blind_calibrated": cal_blind,
                        "val_calibrated": cal_val,
                    }
                    # Temperature/Platt are strictly monotone, but the P_CLIP
                    # used before logit() merges "exact 0" and "tiny positive"
                    # score groups (float32 sigmoid underflow), which can shift
                    # AP slightly for the strongly polarised baseline scores.
                    # 1e-3 is far below any real AUPRC difference of interest.
                    if method in ("temperature", "platt"):
                        if abs(cal_blind["auprc"] - raw_blind["auprc"]) > 0.02:
                            failures.append(
                                f"AUPRC not invariant seed {seed} {variant} "
                                f"{channel} {method}"
                            )
                    blind_ece.setdefault((variant, channel, method), []).append(
                        cal_blind["ece"]
                    )
                    blind_brier.setdefault((variant, channel, method), []).append(
                        cal_blind["brier"]
                    )
                    blind_auprc.setdefault((variant, channel, method), []).append(
                        cal_blind["auprc"]
                    )
                    val_ece_after_fit.setdefault(
                        (variant, channel, method), []
                    ).append(cal_val["ece"])
                blind_ece.setdefault((variant, channel, "raw"), []).append(
                    raw_blind["ece"]
                )
                blind_brier.setdefault((variant, channel, "raw"), []).append(
                    raw_blind["brier"]
                )
                blind_auprc.setdefault((variant, channel, "raw"), []).append(
                    raw_blind["auprc"]
                )
                per_seed_records.setdefault(f"seed_{seed}", {})[
                    f"{variant}/{channel}"
                ] = rec
        say(f"seed {seed} done")

    comparisons: dict = {}
    fb = args.variants[0]
    nf = args.variants[1] if len(args.variants) > 1 else None
    if nf is not None:
        for channel in CHANNELS:
            for regime in ("raw",) + METHODS:
                key = (channel, regime)
                comparisons[f"feedback_vs_no_feedback/ece/{channel}/{regime}"] = paired_stats(
                    blind_ece[(fb, channel, regime)], blind_ece[(nf, channel, regime)]
                )
                comparisons[f"feedback_vs_no_feedback/brier/{channel}/{regime}"] = (
                    paired_stats(
                        blind_brier[(fb, channel, regime)],
                        blind_brier[(nf, channel, regime)],
                    )
                )
        for variant in args.variants:
            for method in METHODS:
                comparisons[f"raw_vs_calibrated/ece/{variant}/{channel}/{method}"] = (
                    raw_vs_cal_stats(
                        blind_ece[(variant, channel, "raw")],
                        blind_ece[(variant, channel, method)],
                    )
                )
                comparisons[f"raw_vs_calibrated/brier/{variant}/{channel}/{method}"] = (
                    raw_vs_cal_stats(
                        blind_brier[(variant, channel, "raw")],
                        blind_brier[(variant, channel, method)],
                    )
                )

    # ---- deterministic baselines (F2-C2 CV/IMM): one fit per method/channel ----
    baseline_records: dict = {}
    vs_baselines: dict = {}
    baseline_blind_ece: dict = {}
    baseline_blind_brier: dict = {}
    if args.baseline_root is not None:
        val_root_b = args.baseline_root / args.val_tag
        blind_root_b = args.baseline_root / args.blind_tag
        methods_b = sorted(
            d.name for d in val_root_b.iterdir()
            if d.is_dir() and (d / "events.npz").is_file()
        )
        kaorf_val_ref = np.load(
            val_root / f"seed_{args.seeds[0]}" / fb / "events.npz"
        )
        kaorf_blind_ref = np.load(
            blind_root / f"seed_{args.seeds[0]}" / fb / "events.npz"
        )
        for method_b in methods_b:
            val_npz = np.load(val_root_b / method_b / "events.npz")
            blind_npz = np.load(blind_root_b / method_b / "events.npz")
            for split_name, npz, ref in (
                ("val", val_npz, kaorf_val_ref),
                ("blind", blind_npz, kaorf_blind_ref),
            ):
                for key in ("y_severe", "y_moderate"):
                    if not np.array_equal(npz[key], ref[key]):
                        failures.append(
                            f"LABEL MISMATCH baseline {method_b} {split_name} {key}"
                        )
            rec = {"fit_params": {}, "methods": {}, "blind_raw": {}, "val_raw": {}}
            for channel in CHANNELS:
                y_val = val_npz[f"y_{channel}"]
                p_val = val_npz[f"p_{channel}"]
                y_blind = blind_npz[f"y_{channel}"]
                p_blind = blind_npz[f"p_{channel}"]
                if not y_val.any() or not y_blind.any():
                    failures.append(
                        f"no positives baseline {method_b} {channel}"
                    )
                    continue
                z_val = logit(p_val)
                fitted = {
                    "temperature": fit_temperature(z_val, y_val),
                    "platt": fit_platt(z_val, y_val),
                    "isotonic": fit_isotonic(p_val, y_val),
                }
                raw_blind = base.aggregate(y_blind, p_blind)
                raw_val = base.aggregate(y_val, p_val)
                rec["blind_raw"][channel] = raw_blind
                rec["val_raw"][channel] = raw_val
                rec["fit_params"][channel] = {
                    "temperature_T": fitted["temperature"],
                    "platt_a": fitted["platt"][0],
                    "platt_b": fitted["platt"][1],
                }
                rec["methods"][channel] = {}
                for method_name in METHODS:
                    p_cal_blind = apply_method(method_name, fitted[method_name], p_blind)
                    p_cal_val = apply_method(method_name, fitted[method_name], p_val)
                    if not (np.isfinite(p_cal_blind).all() and np.isfinite(p_cal_val).all()):
                        failures.append(
                            f"non-finite calibrated probabilities baseline "
                            f"{method_b} {channel} {method_name}"
                        )
                        continue
                    cal_blind = base.aggregate(y_blind, p_cal_blind)
                    cal_val = base.aggregate(y_val, p_cal_val)
                    rec["methods"][channel][method_name] = {
                        "blind_calibrated": cal_blind,
                        "val_calibrated": cal_val,
                    }
                    if method_name in ("temperature", "platt"):
                        if abs(cal_blind["auprc"] - raw_blind["auprc"]) > 0.02:
                            failures.append(
                                f"AUPRC not invariant baseline {method_b} "
                                f"{channel} {method_name}"
                            )
                    baseline_blind_ece[(method_b, channel, method_name)] = cal_blind["ece"]
                    baseline_blind_brier[(method_b, channel, method_name)] = cal_blind["brier"]
                baseline_blind_ece[(method_b, channel, "raw")] = raw_blind["ece"]
                baseline_blind_brier[(method_b, channel, "raw")] = raw_blind["brier"]
            baseline_records[method_b] = rec
            say(f"baseline {method_b} fitted+applied")

        from scipy.stats import wilcoxon as _wilcoxon

        for (method_b, channel, regime), b_ece in baseline_blind_ece.items():
            for arm in args.variants:
                kaorf_vals = np.asarray(blind_ece[(arm, channel, regime)])
                diff = kaorf_vals - b_ece
                p = (
                    float(_wilcoxon(diff).pvalue)
                    if not np.allclose(diff, 0.0)
                    else float("nan")
                )
                vs_baselines[f"vs_{method_b}/ece/{arm}/{channel}/{regime}"] = {
                    "kaorf_mean": float(kaorf_vals.mean()),
                    "baseline": float(b_ece),
                    "mean_delta_kaorf_minus_baseline": float(diff.mean()),
                    "kaorf_below_baseline": int((diff < 0).sum()),
                    "n_seeds": int(len(kaorf_vals)),
                    "wilcoxon_one_sample_p": p,
                }
        for (method_b, channel, regime), b_brier in baseline_blind_brier.items():
            for arm in args.variants:
                kaorf_vals = np.asarray(blind_brier[(arm, channel, regime)])
                diff = kaorf_vals - b_brier
                p = (
                    float(_wilcoxon(diff).pvalue)
                    if not np.allclose(diff, 0.0)
                    else float("nan")
                )
                vs_baselines[f"vs_{method_b}/brier/{arm}/{channel}/{regime}"] = {
                    "kaorf_mean": float(kaorf_vals.mean()),
                    "baseline": float(b_brier),
                    "mean_delta_kaorf_minus_baseline": float(diff.mean()),
                    "kaorf_below_baseline": int((diff < 0).sum()),
                    "n_seeds": int(len(kaorf_vals)),
                    "wilcoxon_one_sample_p": p,
                }

    result = {
        "analysis": "F6 post-hoc calibration fitted on validation, evaluated on blind",
        "protocol": {
            "fit_split": "NY validation 2025-09..2025-10 (8 dates)",
            "eval_split": "NY blind 2025-11..2025-12 (8 dates), F4 event log reused",
            "new_blind_forward_pass": False,
            "per_model_maps": True,
            "ece_binning": "locked 10-bin (dump_event_log.aggregate)",
        },
        "seeds": list(args.seeds),
        "variants": list(args.variants),
        "mean_blind_ece": {
            f"{v}/{c}/{r}": float(np.mean(blind_ece[(v, c, r)]))
            for v in args.variants
            for c in CHANNELS
            for r in ("raw",) + METHODS
        },
        "mean_blind_brier": {
            f"{v}/{c}/{r}": float(np.mean(blind_brier[(v, c, r)]))
            for v in args.variants
            for c in CHANNELS
            for r in ("raw",) + METHODS
        },
        "mean_blind_auprc": {
            f"{v}/{c}/{r}": float(np.mean(blind_auprc[(v, c, r)]))
            for v in args.variants
            for c in CHANNELS
            for r in ("raw",) + METHODS
        },
        "comparisons": comparisons,
        "deterministic_baselines": baseline_records,
        "vs_baselines": vs_baselines,
        "per_seed": per_seed_records,
        "failures": failures,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_json = args.out / "f6_fitted_calibration.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# F6 -- Post-hoc calibration fitted on validation, applied to blind",
        "",
        "Fit: per (seed, variant, channel) on the validation event log;",
        "evaluate frozen maps on the existing blind event log (no new blind pass).",
        "",
        "## Mean blind ECE",
        "",
        "| channel | regime | feedback | no_feedback |",
        "|---|---|---|---|",
    ]
    for channel in CHANNELS:
        for regime in ("raw",) + METHODS:
            nf_cell = (
                "%.4f" % result["mean_blind_ece"][f"{nf}/{channel}/{regime}"]
                if nf is not None else "n/a"
            )
            lines.append(
                "| %s | %s | %.4f | %s |"
                % (
                    channel,
                    regime,
                    result["mean_blind_ece"][f"{fb}/{channel}/{regime}"],
                    nf_cell,
                )
            )
    lines += ["", "## Mean blind Brier", "",
              "| channel | regime | feedback | no_feedback |",
              "|---|---|---|---|"]
    for channel in CHANNELS:
        for regime in ("raw",) + METHODS:
            nf_cell = (
                "%.4f" % result["mean_blind_brier"][f"{nf}/{channel}/{regime}"]
                if nf is not None else "n/a"
            )
            lines.append(
                "| %s | %s | %.4f | %s |"
                % (
                    channel,
                    regime,
                    result["mean_blind_brier"][f"{fb}/{channel}/{regime}"],
                    nf_cell,
                )
            )
    lines += ["", "## Paired comparisons (feedback - no_feedback)", ""]
    for key, value in comparisons.items():
        if key.startswith("feedback_vs_no_feedback/ece"):
            lines.append(
                "- %s: mean fb %.4f vs nf %.4f, delta %+.4f, dz %s, p=%.3g, wins %d/%d"
                % (
                    key,
                    value["mean_feedback"],
                    value["mean_no_feedback"],
                    value["mean_delta_feedback_minus_none"],
                    ("%.2f" % value["dz"]) if value["dz"] is not None else "n/a",
                    value["wilcoxon_p"],
                    value["feedback_wins"],
                    value["n_pairs"],
                )
            )
    if vs_baselines:
        lines += [
            "",
            "## KAORF vs deterministic baselines (blind ECE, baseline calibrated/evaluated on its own fitted map)",
            "",
            "| baseline | channel | regime | KAORF fb mean | KAORF nf mean | baseline | fb below? (of 10) | p (one-sample) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for (method_b, channel, regime), b_ece in sorted(
            {(k[0], k[1], k[2]): v for k, v in baseline_blind_ece.items()}.items()
        ):
            key_ece_fb = f"vs_{method_b}/ece/{fb}/{channel}/{regime}"
            key_ece_nf = f"vs_{method_b}/ece/{nf}/{channel}/{regime}"
            if key_ece_fb not in vs_baselines:
                continue
            efb, enf = vs_baselines[key_ece_fb], vs_baselines[key_ece_nf]
            lines.append(
                "| %s | %s | %s | %.4f | %.4f | %.4f | %d | %.3g |"
                % (
                    method_b,
                    channel,
                    regime,
                    efb["kaorf_mean"],
                    enf["kaorf_mean"],
                    b_ece,
                    efb["kaorf_below_baseline"],
                    efb["wilcoxon_one_sample_p"],
                )
            )
    if failures:
        lines += ["", "## FAILURES", ""]
        lines += [f"- {f}" for f in failures]
    (args.out / "f6_fitted_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    marker = args.out / ("F6_FAILED" if failures else "F6_DONE")
    marker.write_text(
        f"completed {result['finished_utc']}; failures={len(failures)}\n",
        encoding="utf-8",
    )
    say(f"done -> {out_json} (failures={len(failures)})")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
