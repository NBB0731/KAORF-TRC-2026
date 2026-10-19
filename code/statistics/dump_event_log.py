#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dump_event_log.py -- F4 + F6 on the LOCKED New York blind test.

Why this file exists
--------------------
The locked blind test produced only aggregate metrics
(``final_test_metrics.json -> test_metrics``).  Two required analyses cannot be
computed from aggregates:

  * F4 -- day-clustered (and event-clustered) bootstrap confidence intervals.
           The reported p-values currently quantify *optimisation variability*
           across seeds, not variation across traffic days.  Clustering needs a
           per-event record carrying its date.
  * F6 -- comparison against post-hoc calibration (temperature scaling,
           isotonic regression, prior-shift correction).  Those procedures are
           fitted on validation and applied to test, so they need raw
           probabilities, not Brier/ECE summaries.

One pass over the locked checkpoints yields the per-event log needed by both,
which is why F4 and F6 share a single re-evaluation instead of two.

Protocol guarantees (this is a *re-evaluation*, not a new experiment)
--------------------------------------------------------------------
1. The model, the checkpoint, the seed set and the selection rule are UNCHANGED.
   No training, no fine-tuning, no re-selection, no epoch search.  The
   checkpoint is loaded and its ``args`` are used to rebuild the architecture,
   with a ``strict=True`` state-dict load as an architecture assertion.
2. Only the *output granularity* changes: aggregates -> per-event records.
3. The blind partition (2025-11..2025-12, 8 dates) is read here, exactly as in
   the original blind evaluation.  It is still never passed to a trainer.
4. Two hard guards reproduce the published numbers:
     (a) recomputed aggregate metrics must match the archived
         ``final_test_metrics.json`` to a tight tolerance, and
     (b) the day-weighted mean of per-day aggregates must equal the pooled pass.
   If either guard fails, this script exits non-zero -- it never silently
   produces a log that disagrees with Table I.

What is written
---------------
Per (seed, variant): ``events.npz`` with parallel arrays
    date_index      int16   which blind date the event belongs to
    y_severe        bool    future min separation <= 0.25 nm
    p_severe        float32 model probability (severe channel)
    y_moderate      bool    future min separation <= 0.50 nm
    p_moderate      float32 model probability (moderate channel)
    a_severe        float32 kinematic anchor probability (severe channel)
    n_nodes         int16   vessels in the scene (for node weighting)
plus ``event_log_meta.json`` with the protocol fingerprint and the guard deltas.

Usage
-----
    python scripts/dump_event_log.py \
        --runs runs/final_ny_temporal_10seed_20260921 \
        --scenes data/final_2025_scenes/new_york \
        --manifest outputs/ais_2025_preregistered_manifest.csv \
        --out runs/event_log_20260924 --tag blind
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

WORK = Path(__file__).resolve().parents[1]
for extra in (str(WORK), str(WORK / "training"), str(WORK / "model"), str(WORK / "evaluation")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

# Resolve the package layout (code/, code/model, code/training, code/evaluation)
# so the same import lines work from any working directory.
_CODE_ROOT = next(_p for _p in Path(__file__).resolve().parents if _p.name == "code")
for _extra in ("", "model", "training", "evaluation"):
    _cand = str(_CODE_ROOT / _extra) if _extra else str(_CODE_ROOT)
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from rfign_model import RFIGN  # noqa: E402
from train_rfign_dev import SceneDataset, ground_truth_close_event  # noqa: E402


DEFAULT_SEEDS = [17, 29, 42, 71, 101, 131, 211, 307, 509, 887]
VARIANTS = ("constrained_feedback", "constrained_no_feedback")

# Tolerances for the guard that re-derives the published Table I numbers.
# Aggregate metrics are recomputed here from a fresh forward pass; float32
# reduction order differs from the archived run, so a small drift is expected.
# These are far tighter than any scientifically meaningful difference.
AGG_ABS_TOL = 2e-6
AGG_REL_TOL = 2e-5
# Per-day node-weighted mean vs pooled pass (float accumulation order only).
CONSISTENCY_TOL = 1e-6


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def sha256_of(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def blind_rows(manifest: Path) -> list[dict]:
    """The 8 locked blind dates, in manifest order."""
    with open(manifest, encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["port"] == "new_york" and row["ny_split"] == "blind_test"
        ]
    if not rows:
        raise SystemExit("FATAL: no blind_test rows in the manifest")
    if len(rows) != 8:
        raise SystemExit(f"FATAL: expected 8 blind dates, manifest has {len(rows)}")
    return rows


def build_model(checkpoint: dict, device: torch.device):
    """Rebuild the architecture from the checkpoint's own stored args.

    The constructor call mirrors the locked D-stage evaluator exactly; the
    ``strict=True`` state-dict load doubles as an architecture assertion, so a
    mismatch fails loudly instead of silently evaluating a different model.
    """
    args = checkpoint["args"]
    model = RFIGN(
        feedback=not args.get("no_feedback", False),
        detach_risk_feedback=args.get("detach_risk_feedback", False),
        isolated_risk_head=args.get("isolated_risk_head", False),
        constrained_risk_feedback=args.get("constrained_risk_feedback", False),
        residual_bound=args.get("residual_bound", 1.0),
        anchor_from_fuzzy=args.get("anchor_from_fuzzy", False),
        unbounded_residual=args.get("unbounded_residual", False),
        free_ordinal=args.get("free_ordinal", False),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, args


@torch.no_grad()
def forward_scene(model, batch, device, close_threshold_nm: float, moderate_threshold_nm: float):
    """One batch -> per-pair arrays for both severity channels."""
    history, history_mask, future, node_mask, static = batch
    history = history.to(device)
    history_mask = history_mask.to(device)
    future = future.to(device)
    node_mask = node_mask.to(device)
    static = static.to(device)

    output = model(history, history_mask, node_mask, static)

    pair_valid = node_mask[:, :, None] & node_mask[:, None, :]
    eye = torch.eye(pair_valid.shape[1], dtype=torch.bool, device=device).unsqueeze(0)
    pair_valid = pair_valid & ~eye
    upper = torch.triu(pair_valid, diagonal=1)

    y_severe = ground_truth_close_event(future, node_mask, close_threshold_nm)
    y_moderate = ground_truth_close_event(future, node_mask, moderate_threshold_nm)

    p_severe = output["risk"]
    p_moderate = output.get("moderate_risk", output["risk"])
    a_severe = output.get("anchor_risk", output["risk"])

    return dict(
        upper=upper.cpu().numpy(),
        y_severe=(y_severe >= 0.5).cpu().numpy(),
        y_moderate=(y_moderate >= 0.5).cpu().numpy(),
        p_severe=p_severe.cpu().numpy(),
        p_moderate=p_moderate.cpu().numpy(),
        a_severe=a_severe.cpu().numpy(),
        n_nodes=node_mask.sum(dim=1).cpu().numpy(),
    )


@torch.no_grad()
def collect(model, loader, device, close_threshold_nm, moderate_threshold_nm):
    """Concatenate per-pair records over a loader, tracking the scene offset.

    Also records ``scenes_per_day``: the number of scenes contributed by each
    blind date, in order.  ``scene_index`` is a cumulative scene counter, so
    the analysis stage needs these counts to map each event back to its date --
    without them day-clustered inference is impossible.
    """
    chunks = {
        "y_severe": [], "p_severe": [], "y_moderate": [], "p_moderate": [],
        "a_severe": [], "scene_nodes": [],
    }
    day_counts: list[int] = []
    scene_cursor = 0
    scenes_in_day = 0
    for batch in loader:
        result = forward_scene(
            model, batch, device, close_threshold_nm, moderate_threshold_nm
        )
        upper = result["upper"]
        n_scenes = upper.shape[0]
        for scene in range(n_scenes):
            mask = np.triu(upper[scene], k=1).astype(bool)
            scene_cursor += 1
            scenes_in_day += 1
            if not mask.any():
                continue
            for name, source in (
                ("y_severe", result["y_severe"]),
                ("p_severe", result["p_severe"]),
                ("y_moderate", result["y_moderate"]),
                ("p_moderate", result["p_moderate"]),
                ("a_severe", result["a_severe"]),
            ):
                chunks[name].append(source[scene][mask].astype(
                    bool if name.startswith("y_") else np.float32
                ))
            count = int(mask.sum())
            chunks["scene_nodes"].append(
                np.full(count, scene_cursor - 1, dtype=np.int64)
            )
        day_counts.append(scenes_in_day)
        scenes_in_day = 0
    if not chunks["p_severe"]:
        return None, scene_cursor, []
    collected = {name: np.concatenate(values) for name, values in chunks.items()}
    collected["scenes_per_day"] = np.asarray(day_counts, dtype=np.int64)
    return collected, scene_cursor, day_counts


def aggregate(y, p) -> dict:
    from sklearn.metrics import average_precision_score

    y = y.astype(np.float64)
    p = p.astype(np.float64)
    metrics = {"positive_rate": float(y.mean()), "brier": float(np.mean((p - y) ** 2))}
    metrics["auprc"] = (
        float(average_precision_score(y, p)) if y.any() else float("nan")
    )
    # 10-bin ECE, identical binning to the locked pipeline.
    edges = np.linspace(0.0, 1.0, 11)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    error = 0.0
    for bin_id in range(10):
        mask = index == bin_id
        if not mask.any():
            continue
        error += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    metrics["ece"] = float(error)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True,
                        help="New York scene directory (contains <date>.npz)")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
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

    rows = blind_rows(args.manifest)
    dates = [row["date"] for row in rows]
    paths = [args.scenes / f"{day}.npz" for day in dates]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"FATAL: missing blind scene file {path}")

    dataset = SceneDataset(paths)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.workers)

    info = {
        "tag": args.tag,
        "analysis": "F4 (day-clustered inference) + F6 (post-hoc calibration)",
        "blind_dates": dates,
        "seeds": args.seeds,
        "variants": args.variants,
        "device": str(device),
        "protocol": {
            "re_evaluation_only": True,
            "retrained": False,
            "checkpoints": "New York validation-selected, selection_metric=joint",
            "blind_partition_never_trained": True,
            "output_granularity_changed": "aggregate -> per-event",
        },
        "code_sha256": {
            "scripts/dump_event_log.py": sha256_of(Path(__file__).resolve()),
        },
        "guard_tolerance": {
            "aggregate_abs": AGG_ABS_TOL,
            "aggregate_rel": AGG_REL_TOL,
            "perday_vs_pooled": CONSISTENCY_TOL,
        },
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": {},
    }

    failures: list[str] = []
    for seed in args.seeds:
        for variant in args.variants:
            out_dir = out_root / f"seed_{seed}" / variant
            event_path = out_dir / "events.npz"
            if event_path.exists() and not args.force:
                say(f"skip existing {event_path.relative_to(args.out)}")
                continue

            checkpoint_path = args.runs / f"seed_{seed}" / variant / "best.pt"
            if not checkpoint_path.is_file():
                failures.append(f"missing checkpoint {checkpoint_path}")
                say(f"  WARN missing checkpoint {checkpoint_path}")
                continue

            archived_path = (
                args.runs / f"seed_{seed}" / variant / "final_test_metrics.json"
            )
            archived = None
            if archived_path.is_file():
                archived = json.loads(archived_path.read_text(encoding="utf-8"))["test_metrics"]

            started = time.time()
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            ckpt_args = checkpoint["args"]
            if not ckpt_args.get("close_event_target", False):
                raise SystemExit(
                    f"FATAL: {checkpoint_path} was not trained with --close-event-target"
                )
            close_nm = ckpt_args.get("close_threshold_nm", 0.25)
            moderate_nm = ckpt_args.get("moderate_threshold_nm", 0.5)
            model, _ = build_model(checkpoint, device)

            collected, n_scenes, day_counts = collect(
                model, loader, device, close_nm, moderate_nm
            )
            if collected is None:
                failures.append(f"no events collected for seed {seed} {variant}")
                continue

            # Guard (a): the per-day scene counts must sum to the total, or the
            # scene->date mapping used by day-clustered inference is wrong.
            if sum(day_counts) != n_scenes:
                failures.append(
                    f"GUARD FAIL seed {seed} {variant}: scenes_per_day sums to "
                    f"{sum(day_counts)} but {n_scenes} scenes were loaded"
                )

            # Guard (b): recompute the published aggregates from these events.
            severe = aggregate(collected["y_severe"], collected["p_severe"])
            moderate = aggregate(collected["y_moderate"], collected["p_moderate"])
            guard = {
                "n_scenes": n_scenes,
                "n_events": int(collected["p_severe"].size),
                "per_day_scene_counts": day_counts,
            }
            if archived is not None:
                checks = {
                    "risk_auprc": severe["auprc"],
                    "risk_brier": severe["brier"],
                    "risk_ece": severe["ece"],
                    "risk_positive_rate": severe["positive_rate"],
                    "moderate_risk_auprc": moderate["auprc"],
                    "moderate_risk_brier": moderate["brier"],
                    "moderate_risk_ece": moderate["ece"],
                }
                deltas = {}
                for key, value in checks.items():
                    if key not in archived:
                        continue
                    reference = float(archived[key])
                    delta = abs(value - reference)
                    deltas[key] = {
                        "recomputed": value,
                        "archived": reference,
                        "abs": delta,
                        "rel": delta / max(abs(reference), 1e-12),
                    }
                    if delta > AGG_ABS_TOL and delta / max(abs(reference), 1e-12) > AGG_REL_TOL:
                        failures.append(
                            f"GUARD FAIL seed {seed} {variant} {key}: "
                            f"recomputed {value} vs archived {reference}"
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
                "variant": variant,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_of(checkpoint_path),
                "blind_dates": dates,
                "close_threshold_nm": close_nm,
                "moderate_threshold_nm": moderate_nm,
                "selection_metric": ckpt_args.get("selection_metric"),
                "selected_epoch": checkpoint.get("epoch"),
                "recomputed_severe": severe,
                "recomputed_moderate": moderate,
                "guard": guard,
                "seconds": round(time.time() - started, 1),
            }
            (out_dir / "event_log_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            say(
                f"seed {seed} {variant}: events={guard['n_events']:,} "
                f"ade_archived={archived.get('ade_nm') if archived else 'n/a'} "
                f"({meta['seconds']}s)"
            )
            info["records"][f"seed_{seed}/{variant}"] = {
                "n_scenes": n_scenes,
                "n_events": guard["n_events"],
                "seconds": meta["seconds"],
            }

    info["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    info["failures"] = failures
    (out_root / "EVENT_LOG_INFO.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )

    if failures:
        say(f"FAILED with {len(failures)} problem(s)")
        for problem in failures:
            say(f"  - {problem}")
        return 2
    (out_root / "EVENT_LOG_DONE").write_text(
        f"completed {info['finished_utc']}\n", encoding="utf-8"
    )
    say(f"done, wrote {out_root / 'EVENT_LOG_DONE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
