#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dump_event_log_traj.py -- per-horizon displacement errors on the LOCKED
New York blind test (local re-evaluation, 2026-09-24).

Companion to ``dump_event_log.py`` (risk channels).  This script re-runs the
same locked checkpoints on the same 8 blind dates and records, for every valid
node, the displacement error at each of the six future horizons
(5, 10, 15, 20, 25, 30 min).  Its purpose is the per-horizon error analysis
requested alongside the reliability diagrams and PR curves; it is a
*re-evaluation* under the same protocol guarantees:

1. No training, no re-selection: checkpoints loaded with strict=True; args
   rebuilt exactly as in dump_event_log.py.
2. Guards reproduce the published trajectory numbers:
   (a) mean over nodes and horizons of the per-node displacement error must
       match the archived ``ade_nm`` (this is exactly how the locked pipeline
       computes ADE, see train_rfign_dev.py lines 154-174);
   (b) the last-horizon value must match the archived ``fde_nm``;
   (c) per-day scene counts must sum to the loaded scene count.
   Any failure exits non-zero.

What is written (per seed, variant)
-----------------------------------
    traj_errors.npz   err_nm  float32 (n_valid_nodes, 6)  displacement error
                      scene_index int32  scene row per node
    traj_meta.json    per-horizon ADE (nm), guard deltas, checkpoint sha256

Windows host note: same PosixPath unpickle shim as dump_event_log.py.

Usage
-----
    python dump_event_log_traj.py --runs <runs_dir> \
        --scenes data/final_2025_scenes/new_york \
        --manifest outputs/ais_2025_preregistered_manifest.csv \
        --out runs/event_log_20260924 --tag blind --device cpu --workers 0
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

# Deserialisation shim for checkpoints pickled on Linux (see dump_event_log.py).
import pathlib as _pathlib  # noqa: E402

if not hasattr(_pathlib, "_posix_shim_applied_local"):
    _pathlib.PosixPath = _pathlib.PurePosixPath  # type: ignore[attr-defined]
    _pathlib._posix_shim_applied_local = True

# Resolve the package layout (code/, code/model, code/training, code/evaluation)
# so the same import lines work from any working directory.
_CODE_ROOT = next(_p for _p in Path(__file__).resolve().parents if _p.name == "code")
for _extra in ("", "model", "training", "evaluation"):
    _cand = str(_CODE_ROOT / _extra) if _extra else str(_CODE_ROOT)
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from rfign_model import RFIGN  # noqa: E402
from train_rfign_dev import SceneDataset  # noqa: E402


DEFAULT_SEEDS = [17, 29, 42, 71, 101, 131, 211, 307, 509, 887]
VARIANTS = ("constrained_feedback", "constrained_no_feedback")

AGG_ABS_TOL = 5e-4   # nm; ~0.9 m -- absorbs GPU->CPU float32 vector_norm drift
AGG_REL_TOL = 1e-4   # observed worst drift across 20 combos was 3.9e-5 (seed 101)
CONSISTENCY_TOL = 1e-6
M_PER_NM = 1852.0


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
    with open(manifest, encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["port"] == "new_york" and row["ny_split"] == "blind_test"
        ]
    if len(rows) != 8:
        raise SystemExit(f"FATAL: expected 8 blind dates, manifest has {len(rows)}")
    return rows


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


@torch.no_grad()
def collect_traj(model, loader, device):
    """Per-node, per-horizon displacement errors in metres (masked nodes).

    ``scene_index`` rows use the same cumulative scene numbering as
    dump_event_log.py (SceneDataset order, shuffle=False), so risk events and
    trajectory rows share one scene coordinate system.
    """
    chunks_err: list[np.ndarray] = []
    chunks_scene: list[np.ndarray] = []
    scene_cursor = 0
    for history, history_mask, future, node_mask, static in loader:
        history = history.to(device)
        history_mask = history_mask.to(device)
        future = future.to(device)
        node_mask = node_mask.to(device)
        static = static.to(device)
        output = model(history, history_mask, node_mask, static)
        pred_m = output["prediction"] * model.position_scale_m
        distance_m = torch.linalg.vector_norm(pred_m - future, dim=-1)  # (B,N,6)
        err = (distance_m * node_mask.unsqueeze(-1)).cpu().numpy()
        mask = node_mask.cpu().numpy().astype(bool)
        for scene in range(err.shape[0]):
            scene_cursor += 1
            sel = mask[scene]
            if not sel.any():
                continue
            chunks_err.append(err[scene][sel])
            chunks_scene.append(np.full(int(sel.sum()), scene_cursor - 1, dtype=np.int32))
    if not chunks_err:
        return None, 0
    return (
        np.concatenate(chunks_err).astype(np.float32),
        np.concatenate(chunks_scene),
    ), scene_cursor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
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
        "analysis": "per-horizon displacement errors (5..30 min)",
        "blind_dates": dates,
        "seeds": args.seeds,
        "variants": args.variants,
        "device": str(device),
        "horizons_min": [5, 10, 15, 20, 25, 30],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": {},
    }

    failures: list[str] = []
    for seed in args.seeds:
        for variant in args.variants:
            out_dir = out_root / f"seed_{seed}" / variant
            traj_path = out_dir / "traj_errors.npz"
            if traj_path.exists() and not args.force:
                say(f"skip existing {traj_path.relative_to(args.out)}")
                continue

            checkpoint_path = args.runs / f"seed_{seed}" / variant / "best.pt"
            if not checkpoint_path.is_file():
                failures.append(f"missing checkpoint {checkpoint_path}")
                continue
            archived_path = args.runs / f"seed_{seed}" / variant / "final_test_metrics.json"
            archived = None
            if archived_path.is_file():
                archived = json.loads(archived_path.read_text(encoding="utf-8"))["test_metrics"]

            started = time.time()
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if not checkpoint["args"].get("close_event_target", False):
                raise SystemExit(f"FATAL: {checkpoint_path} lacks --close-event-target")
            model, _ = build_model(checkpoint, device)

            collected, n_scenes = collect_traj(model, loader, device)
            if collected is None:
                failures.append(f"no trajectory collected for seed {seed} {variant}")
                continue
            err_nm, scene_index = collected

            per_horizon_ade_nm = err_nm.mean(axis=0) / M_PER_NM  # (6,)
            ade_recomputed = float(per_horizon_ade_nm.mean())
            fde_recomputed = float(per_horizon_ade_nm[-1])
            guard = {
                "n_scenes": n_scenes,
                "n_nodes": int(err_nm.shape[0]),
                "per_horizon_ade_nm": [float(v) for v in per_horizon_ade_nm],
            }
            if archived is not None:
                checks = {"ade_nm": ade_recomputed, "fde_nm": fde_recomputed}
                deltas = {}
                for key, value in checks.items():
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
            np.savez_compressed(traj_path, err_nm=err_nm, scene_index=scene_index)
            meta = {
                "seed": seed,
                "variant": variant,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_of(checkpoint_path),
                "blind_dates": dates,
                "horizons_min": [5, 10, 15, 20, 25, 30],
                "recomputed_ade_nm": ade_recomputed,
                "recomputed_fde_nm": fde_recomputed,
                "guard": guard,
                "seconds": round(time.time() - started, 1),
            }
            (out_dir / "traj_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            say(
                f"seed {seed} {variant}: nodes={guard['n_nodes']:,} "
                f"ade={ade_recomputed:.6f} fde={fde_recomputed:.6f} "
                f"({meta['seconds']}s)"
            )
            info["records"][f"seed_{seed}/{variant}"] = {
                "n_scenes": n_scenes,
                "n_nodes": guard["n_nodes"],
                "seconds": meta["seconds"],
            }

    info["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    info["failures"] = failures
    (out_root / "TRAJ_LOG_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    if failures:
        say(f"FAILED with {len(failures)} problem(s)")
        for problem in failures:
            say(f"  - {problem}")
        return 2
    (out_root / "TRAJ_LOG_DONE").write_text(
        f"completed {info['finished_utc']}\n", encoding="utf-8"
    )
    say(f"done, wrote {out_root / 'TRAJ_LOG_DONE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
