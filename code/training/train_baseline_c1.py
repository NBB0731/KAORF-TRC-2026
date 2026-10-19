# -*- coding: utf-8 -*-
"""F2-C1 runner: train and blind-test the plain temporal-Transformer baseline.

Protocol parity with the locked KAORF run
-----------------------------------------
* identical ``SceneDataset`` / ``run_epoch`` code path (imported verbatim from
  ``scripts/train_rfign_dev.py``), so inputs, padding, label definitions and
  metric code are shared, not re-implemented;
* identical partitions: train = New York 2025-01..08 (32 dates), validation =
  2025-09..10 (8 dates), blind test = 2025-11..12 (8 dates, never seen in
  training or selection);
* identical locked seed set {17, 29, 42, 71, 101, 131, 211, 307, 509, 887};
* identical selection metric (``joint``: validation ADE/FDE plus risk terms
  where available) and identical epoch budget;
* identical optimiser settings and the same ``--close-event-target`` label
  definition, so the risk axes are directly comparable with Table I.

What is deliberately different
------------------------------
Only the model. The baseline has no kinematic anchor, no bounded residual, no
ordinal nesting and no risk feedback into the trajectory representation; its
single free risk head reads the shared scene representation and its own
predicted geometry.

The selected checkpoint is the one whose validation epoch minimises the same
joint criterion used for KAORF. Evaluation happens once, on the blind partition,
and the resulting metrics are written per seed.

Usage
-----
    python scripts/train_baseline_c1.py \
        --scenes data/final_2025_scenes/new_york \
        --out runs/baseline_c1_20260924 \
        --seeds 17 29 42 71 101 131 211 307 509 887 \
        --epochs 12 --batch-size 32 --workers 2
"""
from __future__ import annotations

import argparse
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

from plain_transformer import PlainTransformer  # noqa: E402
from train_rfign_dev import SceneDataset, run_epoch  # noqa: E402


OPS = dict(
    close_event_target=True,
    close_threshold_nm=0.25,
    moderate_threshold_nm=0.5,
    anchor_loss_weight=0.2,
    moderate_loss_weight=0.1,
    calibration_weight=0.1,
)


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def joint_score(metrics: dict) -> float:
    """Same joint criterion the locked pipeline selects on.

    Lower is better. Risk terms are included when present so that the baseline
    is not selected purely on displacement.
    """
    ade = metrics.get("ade_nm", float("inf"))
    fde = metrics.get("fde_nm", float("inf"))
    risk = metrics.get("risk_auprc")
    # AUPRC is higher-is-better; convert to a loss-like term with a fixed scale.
    risk_term = 0.0 if risk is None or not np.isfinite(risk) else (1.0 - risk)
    return ade / 1852.0 + fde / 1852.0 + 0.1 * risk_term


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    args.out.mkdir(parents=True, exist_ok=True)

    train_dates = json.loads((args.scenes / "split_train.json").read_text()) \
        if (args.scenes / "split_train.json").is_file() else None
    # Partitions are read from the pre-registered manifest the same way the
    # locked launcher did, so no date can drift between the two studies.
    manifest = Path("/root/kaorf/outputs/ais_2025_preregistered_manifest.csv")
    import csv

    with open(manifest, encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["port"] == "new_york"]
    # The manifest labels the three New York partitions train / validation /
    # blind_test (the external ports use external_only).
    train = sorted(r["date"] for r in rows if r["ny_split"] == "train")
    val = sorted(r["date"] for r in rows if r["ny_split"] == "validation")
    blind = sorted(r["date"] for r in rows if r["ny_split"] == "blind_test")
    assert (len(train), len(val), len(blind)) == (32, 8, 8), (len(train), len(val), len(blind))
    assert not (set(train) & set(val)) and not (set(train) & set(blind)) \
        and not (set(val) & set(blind)), "partition overlap"

    # `--scenes` may be given either as the scene root or as the port directory.
    scene_dir = args.scenes / "new_york" if (args.scenes / "new_york").is_dir() else args.scenes
    train_paths = [scene_dir / f"{d}.npz" for d in train]
    val_paths = [scene_dir / f"{d}.npz" for d in val]
    blind_paths = [scene_dir / f"{d}.npz" for d in blind]
    for path in train_paths + val_paths + blind_paths:
        if not path.is_file():
            raise SystemExit(f"FATAL: missing scene file {path}")

    info = {
        "study": "F2-C1 plain Transformer baseline",
        "model": "PlainTransformer",
        "train_dates": train, "val_dates": val, "blind_dates": blind,
        "seeds": args.seeds, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "device": str(device),
        "protocol": {
            "trainer_imported_verbatim": "scripts/train_rfign_dev.py::SceneDataset, run_epoch",
            "selection": "joint validation criterion",
            "blind_test_reused_for_training": False,
            "external_ports_used_for_tuning": False,
        },
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out / "RUN_INFO.json").write_text(json.dumps(info, indent=2))

    train_ds = SceneDataset(train_paths)
    val_ds = SceneDataset(val_paths)
    blind_ds = SceneDataset(blind_paths)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers)
    blind_loader = DataLoader(blind_ds, args.batch_size, shuffle=False, num_workers=args.workers)

    for seed in args.seeds:
        seed_dir = args.out / f"seed_{seed}" / "plain_transformer"
        metrics_path = seed_dir / "metrics.json"
        if metrics_path.exists():
            say(f"skip existing seed {seed}")
            continue
        seed_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = PlainTransformer().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        best_score, best_epoch, best_state = float("inf"), 0, None
        history = []
        for epoch in range(1, args.epochs + 1):
            tr = run_epoch(model, train_loader, optimizer, device, train=True, **OPS)
            va = run_epoch(model, val_loader, None, device, train=False, **OPS)
            score = joint_score(va)
            history.append({"epoch": epoch, "train": tr, "val": va, "joint": score})
            say(f"seed {seed} epoch {epoch:02d}  train_loss={tr['loss']:.4f}  "
                f"val_ade={va['ade_nm']:.4f}  val_fde={va['fde_nm']:.4f}  joint={score:.6e}")
            if score < best_score:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is None:
            raise SystemExit(f"FATAL: seed {seed} produced no checkpoint")
        model.load_state_dict(best_state)
        torch.save({"model": best_state, "args": vars(args), "select_metric": "joint",
                    "select_epoch": best_epoch, "select_score": best_score}, seed_dir / "best.pt")

        test = run_epoch(model, blind_loader, None, device, train=False, **OPS)
        val_at_best = history[best_epoch - 1]["val"]
        record = {
            "seed": seed, "select_epoch": best_epoch, "select_metric": "joint",
            "test_metrics": test, "validation_metrics_at_selected_epoch": val_at_best,
            "history": history,
        }
        metrics_path.write_text(json.dumps(record, indent=2))
        say(f"seed {seed} DONE  test_ade={test['ade_nm']:.4f}  test_fde={test['fde_nm']:.4f}  "
            f"risk_auprc={test.get('risk_auprc', float('nan')):.4f}")

    say("ALL SEEDS DONE")
    (args.out / "BASELINE_C1_DONE").write_text(
        "plain transformer baseline complete\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
