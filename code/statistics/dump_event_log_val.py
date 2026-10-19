#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dump_event_log_val.py -- F6 fitting set: per-event log on NY *validation*.

Why this file exists
--------------------
F6 (post-hoc calibration comparison) requires fitting temperature / Platt /
isotonic maps on a designated fitting split and applying them to the blind
test.  The raw *blind* probabilities already exist from the F4 re-evaluation
(``runs/event_log_20260924/blind/``).  What is missing is the matching
per-event log on the pre-registered **validation** split (2025-09..2025-10,
8 dates) -- the fitting side.  This script produces exactly that.

Protocol position
-----------------
* Repeated evaluation of the validation split is protocol-legal: it is the
  designated tuning/fitting split.  The blind partition is never touched here.
* No training, no selection change: the same locked checkpoints (New York
  validation-selected, selection_metric=joint) are loaded ``strict=True``.
* The forward path is *imported verbatim* from ``scripts/dump_event_log.py``
  (``forward_scene`` / ``collect`` / ``aggregate`` / ``build_model``), so the
  blind and validation logs are produced by byte-identical forward code.

Guards (fail-closed)
--------------------
(a) Exactly 8 validation dates from the manifest; per-date scene counts are
    derived from the per-date scene files (source of truth), each > 0, and
    their sum must equal the pooled scene count of the pass.
(b) Recomputed validation aggregates are compared against the *trainer's own*
    archived per-epoch validation metrics (``metrics.json``) for the selected
    epoch.  Tolerances are wide enough for aggregation-order drift and
    cross-epoch noise, but far below any wrong-checkpoint signature.  The
    label-derived ``positive_rate`` is model-independent and therefore guarded
    tightly.
(c) All stored probabilities finite and within [0, 1]; both severity channels
    contain at least one positive event.

Usage
-----
    python scripts/dump_event_log_val.py \
        --runs runs/final_ny_temporal_10seed_20260921 \
        --scenes data/final_2025_scenes/new_york \
        --manifest outputs/ais_2025_preregistered_manifest.csv \
        --out runs/event_log_20260924 --tag val
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

import dump_event_log as base  # noqa: E402  (verbatim forward path)

# Per-key guard tolerances vs the trainer's archived per-epoch val metrics.
# Rationale: adjacent epochs differ by ~0.5-2% rel in AUPRC and up to ~50% in
# ECE, so these thresholds detect wrong-checkpoint / wrong-scene-set failures
# while ignoring float-order drift (~1e-5).  positive_rate is a pure function
# of scenes + thresholds (model-free) and is guarded tightly.
VAL_GUARD_TOL = {
    "risk_auprc": (2e-3, 0.02),
    "risk_brier": (2e-3, 0.02),
    "risk_ece": (5e-3, 0.50),
    "risk_positive_rate": (1e-6, 1e-4),
    "moderate_risk_auprc": (2e-3, 0.02),
    "moderate_risk_brier": (2e-3, 0.02),
    "moderate_risk_ece": (5e-3, 0.50),
    "moderate_risk_positive_rate": (1e-6, 1e-4),
}


def val_rows(manifest: Path) -> list[dict]:
    """The 8 pre-registered validation dates, in manifest order."""
    with open(manifest, encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["port"] == "new_york" and row["ny_split"] == "validation"
        ]
    if not rows:
        raise SystemExit("FATAL: no validation rows in the manifest")
    if len(rows) != 8:
        raise SystemExit(f"FATAL: expected 8 validation dates, manifest has {len(rows)}")
    return rows


def per_date_scene_counts(paths: list[Path]) -> list[int]:
    """Scene counts straight from the per-date scene files (source of truth)."""
    counts = []
    for path in paths:
        item = np.load(path)
        counts.append(int(item["history"].shape[0]))
    return counts


def check_probabilities(collected: dict) -> list[str]:
    """Guard (c): probabilities finite and within [0, 1]."""
    problems = []
    for key in ("p_severe", "p_moderate", "a_severe"):
        values = collected[key]
        if not np.isfinite(values).all():
            problems.append(f"{key} contains non-finite values")
        if values.min() < 0.0 or values.max() > 1.0:
            problems.append(
                f"{key} outside [0,1]: min={values.min():.6g} max={values.max():.6g}"
            )
    return problems


def trainer_val_entry(runs: Path, seed: int, variant: str, epoch) -> dict | None:
    """The archived per-epoch val metrics matching the checkpoint's epoch."""
    metrics_path = runs / f"seed_{seed}" / variant / "metrics.json"
    if not metrics_path.is_file():
        return None
    entries = json.loads(metrics_path.read_text(encoding="utf-8"))
    for entry in entries:
        if entry.get("epoch") == epoch:
            return entry.get("val", {})
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True,
                        help="New York scene directory (contains <date>.npz)")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", default="val")
    parser.add_argument("--seeds", type=int, nargs="+", default=base.DEFAULT_SEEDS)
    parser.add_argument("--variants", nargs="+", default=list(base.VARIANTS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    out_root = args.out / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    rows = val_rows(args.manifest)
    dates = [row["date"] for row in rows]
    paths = [args.scenes / f"{day}.npz" for day in dates]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"FATAL: missing validation scene file {path}")

    # Source-of-truth day boundaries, asserted before any model runs.
    day_counts_truth = per_date_scene_counts(paths)
    if any(c <= 0 for c in day_counts_truth):
        raise SystemExit(f"FATAL: a validation date has zero scenes: {day_counts_truth}")

    dataset = base.SceneDataset(paths)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.workers)

    info = {
        "tag": args.tag,
        "analysis": "F6 fitting set (post-hoc calibration, validation side)",
        "val_dates": dates,
        "seeds": args.seeds,
        "variants": args.variants,
        "device": str(device),
        "protocol": {
            "re_evaluation_only": True,
            "retrained": False,
            "checkpoints": "New York validation-selected, selection_metric=joint",
            "blind_partition_touched": False,
            "forward_path": "imported verbatim from scripts/dump_event_log.py",
        },
        "code_sha256": {
            "scripts/dump_event_log.py": base.sha256_of(
                Path(base.__file__).resolve()
            ),
            "scripts/dump_event_log_val.py": base.sha256_of(Path(__file__).resolve()),
        },
        "per_date_scene_counts": day_counts_truth,
        "guard_tolerance": {k: {"abs": v[0], "rel": v[1]} for k, v in VAL_GUARD_TOL.items()},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": {},
    }

    failures: list[str] = []
    for seed in args.seeds:
        for variant in args.variants:
            out_dir = out_root / f"seed_{seed}" / variant
            event_path = out_dir / "events.npz"
            if event_path.exists() and not args.force:
                base.say(f"skip existing {event_path.relative_to(args.out)}")
                continue

            checkpoint_path = args.runs / f"seed_{seed}" / variant / "best.pt"
            if not checkpoint_path.is_file():
                failures.append(f"missing checkpoint {checkpoint_path}")
                base.say(f"  WARN missing checkpoint {checkpoint_path}")
                continue

            started = time.time()
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            ckpt_args = checkpoint["args"]
            if not ckpt_args.get("close_event_target", False):
                raise SystemExit(
                    f"FATAL: {checkpoint_path} was not trained with --close-event-target"
                )
            close_nm = ckpt_args.get("close_threshold_nm", 0.25)
            moderate_nm = ckpt_args.get("moderate_threshold_nm", 0.5)
            model, _ = base.build_model(checkpoint, device)

            collected, n_scenes, _day_counts_per_batch = base.collect(
                model, loader, device, close_nm, moderate_nm
            )
            if collected is None:
                failures.append(f"no events collected for seed {seed} {variant}")
                continue

            guard: dict = {
                "n_scenes": n_scenes,
                "n_events": int(collected["p_severe"].size),
                "per_day_scene_counts": day_counts_truth,
            }

            # Guard (a): source-of-truth day counts must tile the pooled pass.
            if sum(day_counts_truth) != n_scenes:
                failures.append(
                    f"GUARD FAIL seed {seed} {variant}: per-date scene sum "
                    f"{sum(day_counts_truth)} != pooled n_scenes {n_scenes}"
                )

            # Guard (c): probability sanity + label availability.
            problems = check_probabilities(collected)
            if not collected["y_severe"].any():
                problems.append("severe channel has zero positive events")
            if not collected["y_moderate"].any():
                problems.append("moderate channel has zero positive events")
            failures.extend(f"GUARD FAIL seed {seed} {variant}: {p}" for p in problems)

            # Recomputed aggregates + Guard (b) vs the trainer's val metrics.
            severe = base.aggregate(collected["y_severe"], collected["p_severe"])
            moderate = base.aggregate(collected["y_moderate"], collected["p_moderate"])
            epoch = checkpoint.get("epoch")
            archived_val = trainer_val_entry(args.runs, seed, variant, epoch)
            if archived_val is None:
                failures.append(
                    f"GUARD FAIL seed {seed} {variant}: no metrics.json val entry "
                    f"for epoch {epoch}"
                )
            else:
                checks = {
                    "risk_auprc": severe["auprc"],
                    "risk_brier": severe["brier"],
                    "risk_ece": severe["ece"],
                    "risk_positive_rate": severe["positive_rate"],
                    "moderate_risk_auprc": moderate["auprc"],
                    "moderate_risk_brier": moderate["brier"],
                    "moderate_risk_ece": moderate["ece"],
                    "moderate_risk_positive_rate": moderate["positive_rate"],
                }
                deltas = {}
                for key, value in checks.items():
                    if key not in archived_val:
                        continue
                    reference = float(archived_val[key])
                    delta = abs(value - reference)
                    rel = delta / max(abs(reference), 1e-12)
                    deltas[key] = {
                        "recomputed": value,
                        "archived": reference,
                        "abs": delta,
                        "rel": rel,
                    }
                    abs_tol, rel_tol = VAL_GUARD_TOL.get(
                        key, (2e-3, 0.02)
                    )
                    if delta > abs_tol and rel > rel_tol:
                        failures.append(
                            f"GUARD FAIL seed {seed} {variant} {key}: recomputed "
                            f"{value:.8g} vs archived {reference:.8g} "
                            f"(abs {delta:.3g} > {abs_tol:g}, rel {rel:.3g} > {rel_tol:g})"
                        )
                guard["archived_val_deltas"] = deltas
                guard["selected_epoch"] = epoch

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
                "variant": variant,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": base.sha256_of(checkpoint_path),
                "val_dates": dates,
                "close_threshold_nm": close_nm,
                "moderate_threshold_nm": moderate_nm,
                "selection_metric": ckpt_args.get("selection_metric"),
                "selected_epoch": epoch,
                "recomputed_severe": severe,
                "recomputed_moderate": moderate,
                "guard": guard,
                "seconds": round(time.time() - started, 1),
            }
            (out_dir / "event_log_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            base.say(
                f"seed {seed} {variant}: events={guard['n_events']:,} "
                f"val_auprc={severe['auprc']:.6f} ({meta['seconds']}s)"
            )
            info["records"][f"seed_{seed}/{variant}"] = {
                "n_scenes": n_scenes,
                "n_events": guard["n_events"],
                "seconds": meta["seconds"],
            }

    info["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    info["failures"] = failures
    (out_root / "VAL_EVENT_LOG_INFO.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )

    if failures:
        base.say(f"FAILED with {len(failures)} problem(s)")
        for problem in failures:
            base.say(f"  - {problem}")
        (out_root / "VAL_EVENT_LOG_FAILED").write_text(
            "\n".join(failures) + "\n", encoding="utf-8"
        )
        return 2
    (out_root / "VAL_EVENT_LOG_DONE").write_text(
        f"completed {info['finished_utc']}\n", encoding="utf-8"
    )
    base.say(f"done, wrote {out_root / 'VAL_EVENT_LOG_DONE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
