#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dump_event_log_lstm.py -- per-event log for the F2-C2 recurrent sequence baseline.

Why this file exists
--------------------
The recurrent sequence baseline (C2) replaces the trunk's temporal Transformer
encoder with a two-layer LSTM and removes the cross-vessel interaction module
entirely, keeping the same front end, the same risk supervision, the same
partitions and the same selection rule as KAORF, the C1 baseline and the graph
baseline.
F6 must therefore ask the same question for this architecture: is KAORF's
calibration advantage robust to *fitted* post-hoc recalibration of the
competing architecture itself?

This script produces the fitting (validation) and evaluation (blind) per-event
logs for the 10 locked C2 LSTM checkpoints, in the same npz schema as the KAORF
and C1 event logs so f6_fitted_calibration.py can consume them unchanged.

Protocol position
-----------------
* Re-evaluation only: the C2 LSTM checkpoints are loaded strict=True and the
  forward path (forward_scene / collect) is imported verbatim from
  scripts/dump_event_log.py -- byte-identical code for all model families.
* Guard (a): per-date scene counts (source of truth) must tile the pass.
* Guard (b): recomputed severe aggregates must match the trainer's archived
  metrics (metrics.json -> test_metrics for blind,
  validation_metrics_at_selected_epoch for val) at the same tight
  tolerances used for the KAORF and C1 blind guards -- same eval code path.
* Guard (c): probabilities finite in [0, 1]; severe positives present.
* The C2 LSTM baseline has a single free risk head: p_moderate duplicates
  p_severe (the architecture provides no separate moderate channel).
  Manuscript quotes must use the severe channel.

Usage
-----
    python scripts/dump_event_log_lstm.py         --runs runs/baseline_c2lstm_20260925         --scenes data/final_2025_scenes/new_york         --manifest outputs/ais_2025_preregistered_manifest.csv         --out runs/event_log_20260925_lstm
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
WORK = SCRIPT_DIR.parent
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

import dump_event_log as base  # noqa: E402  (verbatim forward path)
# Resolve the package layout (code/, code/model, code/training, code/evaluation)
# so the same import lines work from any working directory.
_CODE_ROOT = next(_p for _p in Path(__file__).resolve().parents if _p.name == "code")
for _extra in ("", "model", "training", "evaluation"):
    _cand = str(_CODE_ROOT / _extra) if _extra else str(_CODE_ROOT)
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from lstm_baseline import LSTMBaseline  # noqa: E402


CLOSE_NM = 0.25
MODERATE_NM = 0.5
# Same tolerances as the locked KAORF blind guard: the C1 metrics were produced
# by the same run_epoch evaluation path, so float-order drift is the only
# expected difference.
AGG_ABS_TOL = 2e-6
AGG_REL_TOL = 2e-5


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def split_dates(manifest: Path, ny_split: str) -> list[str]:
    with open(manifest, encoding="utf-8-sig") as handle:
        dates = [
            row["date"]
            for row in csv.DictReader(handle)
            if row["port"] == "new_york" and row["ny_split"] == ny_split
        ]
    if not dates:
        raise SystemExit(f"FATAL: no {ny_split} rows in the manifest")
    return dates


def per_date_counts(paths: list[Path]) -> list[int]:
    return [int(np.load(p)["history"].shape[0]) for p in paths]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=base.DEFAULT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    splits = {
        "val": split_dates(args.manifest, "validation"),
        "blind": split_dates(args.manifest, "blind_test"),
    }
    paths_by_split = {
        name: [args.scenes / f"{d}.npz" for d in dates]
        for name, dates in splits.items()
    }
    for paths in paths_by_split.values():
        for path in paths:
            if not path.is_file():
                raise SystemExit(f"FATAL: missing scene file {path}")

    info: dict = {
        "analysis": "F2-C3 recurrent sequence baseline (C2) per-event logs (F6 extension)",
        "seeds": args.seeds,
        "splits": splits,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": {},
        "failures": [],
    }
    failures: list[str] = []

    for split_name, dates in splits.items():
        paths = paths_by_split[split_name]
        counts_truth = per_date_counts(paths)
        if any(c <= 0 for c in counts_truth):
            raise SystemExit(f"FATAL: zero-scene date in {split_name}: {counts_truth}")
        dataset = base.SceneDataset(paths)
        loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.workers)

        for seed in args.seeds:
            out_dir = args.out / split_name / f"seed_{seed}" / "lstm"
            event_path = out_dir / "events.npz"
            if event_path.exists() and not args.force:
                say(f"skip existing {event_path}")
                continue

            checkpoint_path = args.runs / f"seed_{seed}" / "lstm_baseline" / "best.pt"
            if not checkpoint_path.is_file():
                failures.append(f"missing checkpoint {checkpoint_path}")
                continue
            archived_path = args.runs / f"seed_{seed}" / "lstm_baseline" / "metrics.json"
            archived_all = json.loads(archived_path.read_text(encoding="utf-8"))
            if split_name == "blind":
                archived = archived_all["test_metrics"]
            else:
                archived = archived_all["validation_metrics_at_selected_epoch"]

            started = time.time()
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            model = LSTMBaseline().to(device)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()

            collected, n_scenes, _per_batch = base.collect(
                model, loader, device, CLOSE_NM, MODERATE_NM
            )
            if collected is None:
                failures.append(f"no events collected for seed {seed} {split_name}")
                continue

            guard: dict = {
                "n_scenes": n_scenes,
                "n_events": int(collected["p_severe"].size),
                "per_day_scene_counts": counts_truth,
                "select_epoch": checkpoint.get("select_epoch"),
            }
            if sum(counts_truth) != n_scenes:
                failures.append(
                    f"GUARD FAIL seed {seed} {split_name}: day counts "
                    f"{sum(counts_truth)} != n_scenes {n_scenes}"
                )
            for key in ("p_severe", "p_moderate", "a_severe"):
                values = collected[key]
                if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
                    failures.append(f"GUARD FAIL seed {seed} {split_name} {key}")
            if not collected["y_severe"].any():
                failures.append(f"GUARD FAIL seed {seed} {split_name}: no severe positives")

            severe = base.aggregate(collected["y_severe"], collected["p_severe"])
            moderate = base.aggregate(collected["y_moderate"], collected["p_moderate"])
            checks = {
                "risk_auprc": severe["auprc"],
                "risk_brier": severe["brier"],
                "risk_ece": severe["ece"],
                "risk_positive_rate": severe["positive_rate"],
            }
            deltas = {}
            for key, value in checks.items():
                if key not in archived:
                    continue
                reference = float(archived[key])
                delta = abs(value - reference)
                rel = delta / max(abs(reference), 1e-12)
                deltas[key] = {"recomputed": value, "archived": reference,
                               "abs": delta, "rel": rel}
                if delta > AGG_ABS_TOL and rel > AGG_REL_TOL:
                    failures.append(
                        f"GUARD FAIL seed {seed} {split_name} {key}: recomputed "
                        f"{value:.10g} vs archived {reference:.10g}"
                    )
            guard["archived_deltas"] = deltas

            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                event_path,
                y_severe=collected["y_severe"],
                p_severe=collected["p_severe"],
                y_moderate=collected["y_moderate"],
                p_moderate=collected["p_moderate"],
                a_severe=collected["a_severe"],
                scene_index=collected["scene_nodes"],
            )
            meta = {
                "seed": seed,
                "variant": "lstm",
                "split": split_name,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": base.sha256_of(checkpoint_path),
                "dates": dates,
                "close_threshold_nm": CLOSE_NM,
                "moderate_threshold_nm": MODERATE_NM,
                "selection_metric": checkpoint.get("select_metric"),
                "selected_epoch": checkpoint.get("select_epoch"),
                "recomputed_severe": severe,
                "recomputed_moderate": moderate,
                "guard": guard,
                "seconds": round(time.time() - started, 1),
            }
            (out_dir / "event_log_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            say(
                f"{split_name} seed {seed}: events={guard['n_events']:,} "
                f"auprc={severe['auprc']:.6f} ({meta['seconds']}s)"
            )
            info["records"][f"{split_name}/seed_{seed}"] = {
                "n_scenes": n_scenes,
                "n_events": guard["n_events"],
                "seconds": meta["seconds"],
            }

    info["failures"] = failures
    info["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if failures:
        for split_name in splits:
            (args.out / split_name / "LSTM_EVENT_LOG_FAILED").parent.mkdir(
                parents=True, exist_ok=True
            )
            (args.out / split_name / "LSTM_EVENT_LOG_FAILED").write_text(
                "\n".join(failures) + "\n", encoding="utf-8"
            )
        (args.out / "LSTM_EVENT_LOG_INFO.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8"
        )
        say(f"FAILED with {len(failures)} problem(s)")
        for problem in failures:
            say(f"  - {problem}")
        return 2
    (args.out / "val" / "VAL_EVENT_LOG_DONE").write_text(
        f"completed {info['finished_utc']}\n", encoding="utf-8"
    )
    (args.out / "blind" / "EVENT_LOG_DONE").write_text(
        f"completed {info['finished_utc']}\n", encoding="utf-8"
    )
    (args.out / "LSTM_EVENT_LOG_INFO.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    say(f"done, wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
