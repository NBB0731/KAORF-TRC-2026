#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dump_baseline_event_log.py -- per-event risk scores for the F2-C2 baselines.

Why this file exists
--------------------
The manuscript's calibration claim ("severe ECE lower by an order of magnitude
than CV/IMM") compares KAORF against *unfitted* baseline scores.  F6 asks the
obvious reviewer question: does that advantage survive when the baseline also
receives a standard post-hoc recalibration fitted on validation?  Answering it
needs per-event baseline scores on the same two splits used for the KAORF
event logs (validation = fitting side, blind = evaluation side).

This script re-plays the F2-C2 evaluators (``predict_cv`` / ``predict_imm`` /
``risk_scores_from_positions``, imported verbatim from
``scripts/evaluate_baselines_on_scenes.py``) over the NY validation and blind
scene partitions and stores per-pair scores in the *same* npz schema as the
KAORF event logs, with the same upper-triangular pair convention so the label
arrays can be cross-checked byte-for-byte against the KAORF logs.

Protocol position
-----------------
* The baselines are deterministic functions of ``history`` -- no learned
  parameters, no RNG.  Re-running them on the blind partition changes nothing
  about model selection; it is the same "re-evaluation, not new experiment"
  status as the F4 event log.
* Guard (a): recomputed blind aggregates (using the C2 evaluator's own metric
  functions, since the archived JSON was produced with those) must match the
  archived ``cv_imm_blind_test.json`` to tight tolerance.
* Guard (b): the label arrays must be byte-identical to the KAORF event logs
  (labels come from two independent code paths: torch
  ``ground_truth_close_event`` vs numpy ``pairwise_min_distance``).
* The validation side has no archived reference; structural guards only.

Usage
-----
    python scripts/dump_baseline_event_log.py \
        --manifest outputs/ais_2025_preregistered_manifest.csv \
        --scenes data/final_2025_scenes/new_york \
        --archived runs/baseline_c2_20260924/cv_imm_blind_test.json \
        --out runs/event_log_20260924/baselines
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
WORK = SCRIPT_DIR.parent
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

# Resolve the package layout (code/, code/model, code/training, code/evaluation)
# so the same import lines work from any working directory.
_CODE_ROOT = next(_p for _p in Path(__file__).resolve().parents if _p.name == "code")
for _extra in ("", "model", "training", "evaluation"):
    _cand = str(_CODE_ROOT / _extra) if _extra else str(_CODE_ROOT)
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from evaluate_baselines_on_scenes import (  # noqa: E402

    MODERATE_THRESHOLD_NM,
    SEVERE_THRESHOLD_NM,
    PREDICTORS,
    pairwise_min_distance,
    risk_scores_from_positions,
    auprc as c2_auprc,
    brier as c2_brier,
    ece as c2_ece,
)

GUARD_ABS_TOL = 2e-6
GUARD_REL_TOL = 2e-5


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def split_rows(manifest: Path, ny_split: str) -> list[str]:
    with open(manifest, encoding="utf-8-sig") as handle:
        rows = [
            row["date"]
            for row in csv.DictReader(handle)
            if row["port"] == "new_york" and row["ny_split"] == ny_split
        ]
    if not rows:
        raise SystemExit(f"FATAL: no {ny_split} rows in the manifest")
    return rows


def evaluate_method(method: str, paths: list[Path]) -> dict:
    """Replay one baseline over scenes -> per-pair arrays (upper triangular)."""
    predict = PREDICTORS[method]
    history_parts, future_parts, mask_parts = [], [], []
    for path in paths:
        item = np.load(path)
        history_parts.append(item["history"])
        future_parts.append(item["future"])
        mask_parts.append(item["node_mask"])
    history = np.concatenate(history_parts, axis=0)
    future = np.concatenate(future_parts, axis=0)
    node_mask = np.concatenate(mask_parts, axis=0).astype(bool)

    pred = predict(history)
    assert pred.shape == future.shape, f"shape mismatch {pred.shape} vs {future.shape}"

    score_all, pair_valid = risk_scores_from_positions(pred, node_mask)
    min_dist, _ = pairwise_min_distance(future, node_mask)
    y_severe = (
        (min_dist <= SEVERE_THRESHOLD_NM * 1852.0) & pair_valid
    )
    y_moderate = (
        (min_dist <= MODERATE_THRESHOLD_NM * 1852.0) & pair_valid
    )

    # Pooled aggregates for guard (a) must be computed over the *full*
    # symmetric pair matrix -- exactly the convention of the archived C2 JSON
    # -- so the float reduction order matches and the guard is tight.
    s_full = score_all[pair_valid].astype(np.float64)
    ys_full = y_severe[pair_valid].astype(np.float64)
    ym_full = y_moderate[pair_valid].astype(np.float64)
    severe_full = pooled_metrics(ys_full, s_full)
    moderate_full = pooled_metrics(ym_full, s_full)

    # Same upper-triangular pair convention as the KAORF event logs (for the
    # stored events and the byte-level label cross-check).
    eye = np.eye(future.shape[1], dtype=bool)[None, :, :]
    upper = np.triu(pair_valid & ~eye, k=1)

    n_scenes = history.shape[0]
    chunks = {"y_severe": [], "y_moderate": [], "score": [], "scene_nodes": []}
    scene_cursor = 0
    for scene in range(n_scenes):
        mask = upper[scene]
        scene_cursor += 1
        if not mask.any():
            continue
        chunks["y_severe"].append(y_severe[scene][mask])
        chunks["y_moderate"].append(y_moderate[scene][mask])
        chunks["score"].append(score_all[scene][mask].astype(np.float64))
        chunks["scene_nodes"].append(
            np.full(int(mask.sum()), scene_cursor - 1, dtype=np.int64)
        )
    collected = {
        name: (
            np.concatenate(values) if name != "scene_nodes"
            else np.concatenate(values)
        )
        for name, values in chunks.items()
    }
    return collected, n_scenes, severe_full, moderate_full


def pooled_metrics(y: np.ndarray, s: np.ndarray) -> dict:
    return {
        "auprc": c2_auprc(y.astype(np.float64), s.astype(np.float64)),
        "brier": c2_brier(y.astype(np.float64), s.astype(np.float64)),
        "ece": c2_ece(y.astype(np.float64), s.astype(np.float64)),
        "positive_rate": float(y.mean()),
        "positives": int(y.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True,
                        help="archived cv_imm_blind_test.json for guard (a)")
    parser.add_argument("--kaorf-log-root", type=Path,
                        default=Path("runs/event_log_20260924"),
                        help="for guard (b) label cross-checks")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["cv", "imm"],
                        choices=sorted(PREDICTORS))
    args = parser.parse_args()

    archived = json.loads(args.archived.read_text(encoding="utf-8"))
    splits = {
        "val": split_rows(args.manifest, "validation"),
        "blind": split_rows(args.manifest, "blind_test"),
    }
    paths_by_split = {
        name: [args.scenes / f"{day}.npz" for day in days]
        for name, days in splits.items()
    }
    for name, paths in paths_by_split.items():
        for path in paths:
            if not path.is_file():
                raise SystemExit(f"FATAL: missing scene file {path}")
    if len(splits["val"]) != 8 or len(splits["blind"]) != 8:
        raise SystemExit("FATAL: expected 8 validation and 8 blind dates")

    info: dict = {
        "analysis": "F2-C2 baseline per-event risk scores (F6 extension)",
        "methods": args.methods,
        "splits": {k: v for k, v in splits.items()},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": {},
        "failures": [],
    }
    failures: list[str] = []

    for method in args.methods:
        for split_name, paths in paths_by_split.items():
            out_dir = args.out / split_name / method
            event_path = out_dir / "events.npz"
            if event_path.exists():
                say(f"skip existing {event_path}")
                continue
            started = time.time()
            collected, n_scenes, severe_full, moderate_full = evaluate_method(
                method, paths
            )
            # Upper-triangular metrics (the stored events' own aggregates) and
            # full-matrix metrics (archived convention, used for guard (a)).
            severe = pooled_metrics(collected["y_severe"], collected["score"])
            moderate = pooled_metrics(collected["y_moderate"], collected["score"])

            # Guard (a): blind aggregates must reproduce the archived JSON
            # (same evaluator code path, deterministic predictor, same
            # full-symmetric-matrix reduction).
            if split_name == "blind":
                ref = archived["results"][method]
                for channel, block in (
                    ("severe", severe_full), ("moderate", moderate_full)
                ):
                    for key in ("auprc", "brier", "ece", "positive_rate"):
                        reference = float(ref[channel][key])
                        value = float(block[key])
                        delta = abs(value - reference)
                        rel = delta / max(abs(reference), 1e-12)
                        if delta > GUARD_ABS_TOL and rel > GUARD_REL_TOL:
                            failures.append(
                                f"GUARD FAIL blind {method} {channel} {key}: "
                                f"recomputed {value:.10g} vs archived {reference:.10g}"
                            )

            # Guard (b): labels must be byte-identical to the KAORF event logs
            # (independent label code paths must agree exactly).
            kaorf_dir = args.kaorf_log_root / split_name / "seed_17" / "constrained_feedback"
            kaorf = np.load(kaorf_dir / "events.npz")
            for key in ("y_severe", "y_moderate"):
                if not np.array_equal(collected[key], kaorf[key]):
                    failures.append(
                        f"LABEL MISMATCH {split_name} {method} {key} vs KAORF log"
                    )
            if len(collected["score"]) != len(kaorf["p_severe"]):
                failures.append(
                    f"EVENT COUNT MISMATCH {split_name} {method}: "
                    f"{len(collected['score'])} vs KAORF {len(kaorf['p_severe'])}"
                )

            out_dir.mkdir(parents=True, exist_ok=True)
            # p_severe == p_moderate by construction (one geometric score);
            # store the same array under both keys so downstream F6 code can
            # treat baselines and KAORF arms uniformly.
            np.savez_compressed(
                event_path,
                y_severe=collected["y_severe"],
                y_moderate=collected["y_moderate"],
                p_severe=collected["score"].astype(np.float32),
                p_moderate=collected["score"].astype(np.float32),
                scene_index=collected["scene_nodes"],
            )
            meta = {
                "method": method,
                "split": split_name,
                "dates": splits[split_name],
                "n_scenes": n_scenes,
                "n_events": int(len(collected["score"])),
                "recomputed_severe_upper_tri": severe,
                "recomputed_moderate_upper_tri": moderate,
                "recomputed_severe_full_matrix": severe_full,
                "recomputed_moderate_full_matrix": moderate_full,
                "score_definition": (
                    "sigmoid((0.25nm - predicted_min_pair_distance)/0.25nm); "
                    "moderate shares the severe-anchored score"
                ),
                "seconds": round(time.time() - started, 1),
            }
            (out_dir / "event_log_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            say(
                f"{split_name}/{method}: scenes={n_scenes:,} "
                f"events={meta['n_events']:,} severe_ece={severe_full['ece']:.6f} "
                f"({meta['seconds']}s)"
            )
            info["records"][f"{split_name}/{method}"] = meta

    info["failures"] = failures
    info["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (args.out / "BASELINE_EVENT_LOG_INFO.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    if failures:
        say(f"FAILED with {len(failures)} problem(s)")
        for problem in failures:
            say(f"  - {problem}")
        (args.out / "BASELINE_EVENT_LOG_FAILED").write_text(
            "\n".join(failures) + "\n", encoding="utf-8"
        )
        return 2
    (args.out / "BASELINE_EVENT_LOG_DONE").write_text(
        f"completed {info['finished_utc']}\n", encoding="utf-8"
    )
    say(f"done, wrote {args.out / 'BASELINE_EVENT_LOG_DONE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
