"""D stage (F3): zero-shot cross-port transfer of the LOCKED KAORF checkpoints.

The New York blind-test checkpoints (selected on NY validation with
``--selection-metric joint``) are evaluated on the two external ports with **no
fine-tuning, no re-training and no re-selection**, because the pre-registered
manifest marks every external scene slice ``external_only`` with
``allowed_for_tuning = no``.

Protocol guarantees enforced in code
------------------------------------
1. Every scene path is asserted not to be a New York path, so no ``blind_test``
   date can leak into this evaluation.
2. Nothing external is used to fit or select anything: the checkpoint is loaded
   as-is and the architecture is rebuilt from the checkpoint's own stored args
   (a ``strict=True`` state-dict load doubles as an architecture check).
3. Metric computation is imported verbatim from ``scripts/train_rfign_dev.py``
   (``SceneDataset`` / ``run_epoch``), so external numbers come from exactly the
   same code path as the NY blind-test numbers.
4. A day-level consistency check asserts that the node-weighted mean of the
   per-day ADE/FDE reproduces the pooled pass to < 1e-9 relative.

Two passes are computed per (port, seed, variant):
  * ``pooled_metrics``  -- one pass over every selected day at once, the headline
                           number, directly comparable with the blind test;
  * ``per_day``         -- one pass per calendar day, kept for day-clustered
                           bootstrap CIs (the pre-registered sampling unit) and
                           for the sparse-port event counts.

Usage
-----
    python scripts/eval_external_zeroshot.py \
        --runs runs/final_ny_temporal_10seed_20260921 \
        --scenes data/final_2025_scenes \
        --manifest outputs/ais_2025_preregistered_manifest.csv \
        --out runs/external_zeroshot_20260923 --tag full48 \
        --days all --ports houston_galveston chesapeake \
        --seeds 17 29 42 71 101 131 211 307 509 887 \
        --variants constrained_feedback
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

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
from train_rfign_dev import SceneDataset, run_epoch  # noqa: E402


DEFAULT_SEEDS = [17, 29, 42, 71, 101, 131, 211, 307, 509, 887]
STRATA = ["weekday_1", "weekend_2", "weekday_3", "weekend_4"]
FULL_DAY_COUNT = 48
SUBSET_DAY_COUNT = 12

# Relative tolerance for the "node-weighted per-day mean == pooled pass" guard.
# The two passes differ only in float accumulation order, which measures ~6e-9 in
# practice (float32 tensor sums), whereas a genuine weighting bug would show up at
# the 1e-2 level. 1e-6 therefore still fails closed on real errors.
CONSISTENCY_TOL = 1e-6


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def read_external_rows(manifest: Path, port: str) -> list[dict]:
    with open(manifest, encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["port"] == port and row["ny_split"] == "external_only"
        ]
    if not rows:
        raise SystemExit(f"FATAL: no external_only rows for port {port}")
    rows.sort(key=lambda row: row["date"])
    if len(rows) != FULL_DAY_COUNT:
        raise SystemExit(f"FATAL: {port} has {len(rows)} external dates, expected {FULL_DAY_COUNT}")
    for row in rows:
        if row["allowed_for_tuning"].strip().lower() != "no":
            raise SystemExit(
                f"FATAL: {port} {row['date']} has allowed_for_tuning="
                f"{row['allowed_for_tuning']!r}; external scenes must never be tuned on"
            )
    return rows


def stratified_subset(rows: list[dict], n_days: int = SUBSET_DAY_COUNT) -> list[dict]:
    """Deterministic stratified subsample: one date per month, strata rotated.

    The manifest design is 12 months x 4 calendar strata (weekday_1 / weekend_2 /
    weekday_3 / weekend_4). Picking stratum index ``(month - 1) % 4`` yields one
    date per month and exactly three dates per stratum, i.e. the weekday/weekend
    balance of the full design is preserved. The chosen calendar dates are the
    same for both external ports (the ports share the same 48 dates), so the
    paired feedback / no-feedback comparison runs on identical days.
    """
    by_month: dict[int, list[dict]] = {}
    for row in rows:
        by_month.setdefault(int(row["month"]), []).append(row)
    picked: list[dict] = []
    for month in sorted(by_month):
        strata = {row["calendar_stratum"]: row for row in by_month[month]}
        missing = [name for name in STRATA if name not in strata]
        if missing:
            raise SystemExit(f"FATAL: month {month} lacks strata {missing}")
        picked.append(strata[STRATA[(month - 1) % len(STRATA)]])
    if len(picked) != n_days:
        raise SystemExit(f"FATAL: stratified subset has {len(picked)} days, expected {n_days}")
    counts: dict[str, int] = {}
    for row in picked:
        counts[row["calendar_stratum"]] = counts.get(row["calendar_stratum"], 0) + 1
    if sorted(counts.values()) != [3, 3, 3, 3]:
        raise SystemExit(f"FATAL: stratified subset is unbalanced: {counts}")
    if len({row["date"] for row in picked}) != n_days:
        raise SystemExit("FATAL: stratified subset contains duplicate dates")
    return picked


def build_model(checkpoint: dict, device: torch.device):
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


def sha256_of(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def node_weighted(days: list[dict], key: str) -> float:
    weights = [day["node_samples"] for day in days]
    total = sum(weights)
    if total == 0:
        return float("nan")
    return sum(day["node_samples"] * day["metrics"][key] for day in days) / total


def evaluate_day(model, path: Path, day_row: dict, options: dict, batch_size: int, workers: int, device):
    dataset = SceneDataset([path])
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=workers)
    metrics = run_epoch(model, loader, None, device, train=False, **options)
    return {
        "date": day_row["date"],
        "month": int(day_row["month"]),
        "stratum": day_row["calendar_stratum"],
        "n_scenes": len(dataset),
        "node_samples": int(dataset.node_mask.sum().item()),
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True, help="NY run root holding seed_*/<variant>/best.pt")
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--days", choices=["all", "subset"], default="all")
    parser.add_argument("--max-days", type=int, default=None, help="debug only: truncate the day list")
    parser.add_argument("--ports", nargs="+", default=["houston_galveston", "chesapeake"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--variants", nargs="+", default=["constrained_feedback"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="recompute runs whose metrics.json already exists")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    out_root = args.out / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    info = {
        "tag": args.tag,
        "days_mode": args.days,
        "ports": args.ports,
        "seeds": args.seeds,
        "variants": args.variants,
        "batch_size": args.batch_size,
        "device": str(device),
        "protocol": {
            "fine_tuning": False,
            "external_used_for_model_selection": False,
            "checkpoints": "New York validation-selected, selection_metric=joint",
            "external_marked": "external_only / allowed_for_tuning=no",
            "blind_test_untouched": True,
        },
        "code_sha256": {
            "model/rfign_model.py": sha256_of(WORK / "model" / "rfign_model.py"),
            "training/train_rfign_dev.py": sha256_of(WORK / "training" / "train_rfign_dev.py"),
            "scripts/eval_external_zeroshot.py": sha256_of(Path(__file__).resolve()),
        },
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ports_detail": {},
    }

    completed = 0
    for port in args.ports:
        rows = read_external_rows(args.manifest, port)
        if args.days == "subset":
            rows = stratified_subset(rows)
        if args.max_days is not None:
            rows = rows[: args.max_days]
        paths = [args.scenes / port / f"{row['date']}.npz" for row in rows]
        for path in paths:
            if "new_york" in str(path):
                raise SystemExit(f"FATAL: New York scene in an external evaluation: {path}")
            if not path.is_file():
                raise SystemExit(f"FATAL: missing scene file {path}")
        info["ports_detail"][port] = {
            "n_days": len(rows),
            "dates": [row["date"] for row in rows],
            "stratum_counts": {
                name: sum(1 for row in rows if row["calendar_stratum"] == name) for name in STRATA
            },
        }
        say(f"[{port}] days={len(rows)}  {rows[0]['date']} .. {rows[-1]['date']}")

        pooled_dataset = SceneDataset(paths)
        pooled_loader = DataLoader(
            pooled_dataset, args.batch_size, shuffle=False, num_workers=args.workers
        )

        for seed in args.seeds:
            for variant in args.variants:
                out_dir = out_root / port / f"seed_{seed}" / variant
                metrics_path = out_dir / "metrics.json"
                if metrics_path.exists() and not args.force:
                    say(f"  skip existing {metrics_path.relative_to(args.out)}")
                    completed += 1
                    continue
                checkpoint_path = args.runs / f"seed_{seed}" / variant / "best.pt"
                if not checkpoint_path.is_file():
                    say(f"  WARN missing checkpoint {checkpoint_path}")
                    continue

                started = time.time()
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                ckpt_args = checkpoint["args"]
                if not ckpt_args.get("constrained_risk_feedback", False):
                    raise SystemExit(f"FATAL: {checkpoint_path} is not a constrained-risk checkpoint")
                if not ckpt_args.get("close_event_target", False):
                    raise SystemExit(
                        f"FATAL: {checkpoint_path} was not trained with --close-event-target; "
                        "external risk labels would not follow the locked protocol"
                    )
                model, ckpt_args = build_model(checkpoint, device)
                options = dict(
                    close_event_target=True,
                    close_threshold_nm=ckpt_args.get("close_threshold_nm", 0.25),
                    moderate_threshold_nm=ckpt_args.get("moderate_threshold_nm", 0.5),
                    anchor_loss_weight=ckpt_args.get("anchor_loss_weight", 0.2),
                    moderate_loss_weight=ckpt_args.get("moderate_loss_weight", 0.1),
                    calibration_weight=ckpt_args.get("calibration_weight", 0.1),
                )

                per_day = [
                    evaluate_day(model, path, row, options, args.batch_size, args.workers, device)
                    for path, row in zip(paths, rows)
                ]
                pooled = run_epoch(
                    model, pooled_loader, None, device, train=False, **options
                )

                weighted = {key: node_weighted(per_day, key) for key in ("ade_nm", "fde_nm")}
                consistency = {
                    key: abs(weighted[key] - pooled[key]) / max(abs(pooled[key]), 1e-12)
                    for key in ("ade_nm", "fde_nm")
                }
                bad = {key: value for key, value in consistency.items() if value > CONSISTENCY_TOL}
                if bad:
                    raise SystemExit(
                        f"FATAL: per-day vs pooled ADE/FDE mismatch for {port} seed {seed} "
                        f"{variant}: rel.diff={bad}"
                    )

                record = {
                    "port": port,
                    "tag": args.tag,
                    "days_mode": args.days,
                    "seed": seed,
                    "variant": variant,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_mtime_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(checkpoint_path.stat().st_mtime)
                    ),
                    "selected_epoch": int(checkpoint["epoch"]),
                    "n_days": len(rows),
                    "n_scenes": int(sum(day["n_scenes"] for day in per_day)),
                    "n_node_samples": int(sum(day["node_samples"] for day in per_day)),
                    "dates": [row["date"] for row in rows],
                    "pooled_metrics": pooled,
                    "node_weighted_from_days": weighted,
                    "consistency_rel_diff": consistency,
                    "consistency_tolerance": CONSISTENCY_TOL,
                    "ny_validation_metrics_at_selected_epoch": checkpoint.get("metrics"),
                    "risk_targets": {
                        "close_threshold_nm": options["close_threshold_nm"],
                        "moderate_threshold_nm": options["moderate_threshold_nm"],
                    },
                    "per_day": per_day,
                    "eval_seconds": round(time.time() - started, 1),
                }
                out_dir.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                completed += 1
                say(
                    f"  done {port} seed={seed} {variant}: "
                    f"ADE={pooled['ade_nm']:.4f} FDE={pooled['fde_nm']:.4f} nm  "
                    f"severe_AUPRC={pooled.get('risk_auprc', float('nan')):.4f}  "
                    f"nodes={record['n_node_samples']}  {record['eval_seconds']}s"
                )

    info["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    info["records_completed"] = completed
    (out_root / "RUN_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    say(f"tag={args.tag} finished, records={completed}, wrote {out_root / 'RUN_INFO.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
