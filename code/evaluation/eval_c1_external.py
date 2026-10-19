# -*- coding: utf-8 -*-
"""F6-C1 extension: evaluate the plain-Transformer baseline on the external ports.

Protocol parity with ``eval_external_zeroshot.py``
-------------------------------------------------
* same external-only date rows from the pre-registered manifest, same
  stratified 12-date subset, same ports;
* identical ``SceneDataset`` / ``run_epoch`` code path (imported, not
  re-implemented), so metric code, padding and risk-label definition are shared;
* identical zero-shot rule: the NY validation-selected checkpoint is applied
  with no fine-tuning, no threshold adjustment and no external-date selection;
* identical per-day vs pooled consistency guard on ADE/FDE.

The only difference from the KAORF path is the model: ``PlainTransformer``
instead of ``RFIGN``. Its checkpoint stores ``{"model", "args", "select_epoch"}``
and carries no kinematic anchor, so the constrained-only loss block in
``run_epoch`` is skipped by construction (``constrained_risk_feedback=False``).

Usage
-----
    python eval_c1_external.py --runs runs/baseline_c1_20260924 \
        --scenes data/final_2025_scenes --manifest <manifest.csv> \
        --out runs/c1_external_20261018 --tag sub12 --days subset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve()
WORK = HERE.parents[2]
for extra in (str(WORK), str(WORK / "training"), str(WORK / "model"), str(WORK / "evaluation"), str(WORK / "code" / "model"),
              str(WORK / "code" / "training"), str(HERE.parent)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

# Resolve the package layout (code/, code/model, code/training, code/evaluation)
# so the same import lines work from any working directory.
_CODE_ROOT = next(_p for _p in Path(__file__).resolve().parents if _p.name == "code")
for _extra in ("", "model", "training", "evaluation"):
    _cand = str(_CODE_ROOT / _extra) if _extra else str(_CODE_ROOT)
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from plain_transformer import PlainTransformer  # noqa: E402
from train_rfign_dev import SceneDataset, run_epoch  # noqa: E402
# The date selection must be byte-identical to the archived external run, so
# the original helpers are imported rather than re-implemented.
from eval_external_zeroshot import (  # noqa: E402

    CONSISTENCY_TOL,
    FULL_DAY_COUNT,
    STRATA,
    node_weighted,
    read_external_rows,
    stratified_subset,
)

DEFAULT_SEEDS = (17, 29, 42, 71, 101, 131, 211, 307, 509, 887)
SUBSET_DAY_COUNT = 12


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def build_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model = PlainTransformer().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def evaluate_day(model, path: Path, row: dict, options: dict, batch_size: int,
                 workers: int, device: torch.device) -> dict:
    dataset = SceneDataset([path])
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=workers)
    metrics = run_epoch(model, loader, None, device, train=False, **options)
    return {
        "date": row["date"],
        "month": int(row["date"][5:7]),
        "stratum": row["calendar_stratum"],
        "n_scenes": len(dataset),
        "node_samples": int(dataset.node_mask.sum().item()),
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True,
                        help="C1 run root holding seed_*/plain_transformer/best.pt")
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--days", choices=["all", "subset"], default="subset")
    parser.add_argument("--ports", nargs="+", default=["houston_galveston", "chesapeake"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--variant", default="plain_transformer")
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

    options = dict(close_event_target=True, close_threshold_nm=0.25,
                   moderate_threshold_nm=0.5, anchor_loss_weight=0.2,
                   moderate_loss_weight=0.1, calibration_weight=0.1)

    info = {
        "study": "F6-C1 plain Transformer, external zero-shot transfer",
        "tag": args.tag, "days_mode": args.days, "ports": args.ports,
        "seeds": args.seeds, "model": "PlainTransformer", "device": str(device),
        "protocol": {
            "fine_tuning": False,
            "external_used_for_model_selection": False,
            "checkpoints": "New York validation-selected (joint criterion)",
            "external_marked": "external_only / allowed_for_tuning=no",
            "blind_test_untouched": True,
            "trainer_imported_verbatim": "train_rfign_dev.py::SceneDataset, run_epoch",
        },
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ports_detail": {},
    }

    completed = 0
    for port in args.ports:
        rows = read_external_rows(args.manifest, port)
        if args.days == "subset":
            rows = stratified_subset(rows)
        paths = [args.scenes / port / f"{row['date']}.npz" for row in rows]
        for path in paths:
            if "new_york" in str(path):
                raise SystemExit(f"FATAL: New York scene in an external evaluation: {path}")
            if not path.is_file():
                raise SystemExit(f"FATAL: missing scene file {path}")
        info["ports_detail"][port] = {
            "n_days": len(rows), "dates": [r["date"] for r in rows],
            "stratum_counts": {s: sum(1 for r in rows if r["calendar_stratum"] == s)
                               for s in STRATA},
        }
        say(f"[{port}] days={len(rows)}  {rows[0]['date']} .. {rows[-1]['date']}")

        pooled_loader = DataLoader(SceneDataset(paths), args.batch_size,
                                   shuffle=False, num_workers=args.workers)

        for seed in args.seeds:
            out_dir = out_root / port / f"seed_{seed}" / args.variant
            metrics_path = out_dir / "metrics.json"
            if metrics_path.exists() and not args.force:
                say(f"  skip existing {metrics_path}")
                completed += 1
                continue
            checkpoint_path = args.runs / f"seed_{seed}" / args.variant / "best.pt"
            if not checkpoint_path.is_file():
                say(f"  WARN missing checkpoint {checkpoint_path}")
                continue

            started = time.time()
            model, checkpoint = build_model(checkpoint_path, device)
            per_day = [evaluate_day(model, path, row, options, args.batch_size,
                                    args.workers, device)
                       for path, row in zip(paths, rows)]
            pooled = run_epoch(model, pooled_loader, None, device, train=False, **options)

            weighted = {k: node_weighted(per_day, k) for k in ("ade_nm", "fde_nm")}
            consistency = {k: abs(weighted[k] - pooled[k]) / max(abs(pooled[k]), 1e-12)
                           for k in weighted}
            bad = {k: v for k, v in consistency.items() if v > CONSISTENCY_TOL}
            if bad:
                raise SystemExit(f"FATAL: per-day vs pooled mismatch {port} seed {seed}: {bad}")

            record = {
                "port": port, "tag": args.tag, "days_mode": args.days, "seed": seed,
                "variant": args.variant, "model": "PlainTransformer",
                "checkpoint": str(checkpoint_path),
                "checkpoint_mtime_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(checkpoint_path.stat().st_mtime)),
                "selected_epoch": int(checkpoint.get("select_epoch", -1)),
                "select_metric": checkpoint.get("select_metric", "joint"),
                "n_days": len(rows),
                "n_scenes": int(sum(d["n_scenes"] for d in per_day)),
                "n_node_samples": int(sum(d["node_samples"] for d in per_day)),
                "dates": [r["date"] for r in rows],
                "pooled_metrics": pooled,
                "node_weighted_from_days": weighted,
                "consistency_rel_diff": consistency,
                "consistency_tolerance": CONSISTENCY_TOL,
                "per_day": per_day,
                "eval_seconds": round(time.time() - started, 1),
            }
            out_dir.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            completed += 1
            say(f"  done {port} seed={seed}: ADE={pooled['ade_nm']:.4f} "
                f"FDE={pooled['fde_nm']:.4f} nm  severe_AUPRC={pooled.get('risk_auprc', float('nan')):.4f}"
                f"  nodes={record['n_node_samples']}  {record['eval_seconds']}s")

    info["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    info["records_completed"] = completed
    (out_root / "RUN_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    say(f"tag={args.tag} finished, records={completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
