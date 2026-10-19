# -*- coding: utf-8 -*-
"""Pair-orientation sensitivity of the risk head (post-hoc, no retraining).

The manuscript stores one probability per *unordered* vessel pair, taken from
the upper triangle of the scene matrix. Section 7.5.1 asks how much of the
reported risk estimate depends on which vessel is written first. Two variants
are measured, both from the same locked checkpoints:

``same_forward_reverse_direction``
    compare p[i, j] with p[j, i] produced by the *same* forward pass;
``full_scene_order_reversal``
    reverse the order of every valid vessel in the scene, run the checkpoint
    again, and align p'[j, i] back onto each original unordered pair.

Nothing is trained, tuned or reselected here. Labels and the archived
probabilities are checked against ``results/events_blind_20260924`` before any
statistic is computed, and the script fails rather than continue on a
misalignment.

Inputs
------
checkpoints/seed_<s>/constrained_feedback/best.pt
$KAORF_DATA_DIR/new_york/<date>.npz          (default: <package>/data/new_york)
results/events_blind_20260924/                (archived labels and probabilities)

Output
------
results/trc_20261018/order_sensitivity/order_sensitivity_summary.json
results/trc_20261018/order_sensitivity/seed_<s>_pairwise_order.npz   (optional)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

CODE = next(p for p in Path(__file__).resolve().parents if p.name == "code")
PKG = CODE.parent
for _extra in ("", "model", "training", "evaluation"):
    _cand = str(CODE / _extra) if _extra else str(CODE)
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from rfign_model import RFIGN  # noqa: E402
from train_rfign_dev import SceneDataset, ground_truth_close_event  # noqa: E402

# Checkpoints were serialised on Linux and store pathlib.PosixPath in args.
# Those paths are not consumed here; remap only for Windows-side deserialisation.
pathlib.PosixPath = pathlib.PurePosixPath

SEEDS = [17, 29, 42, 71, 101, 131, 211, 307, 509, 887]
BATCH_SIZE = 32
N_BOOT = 200_000
BOOT_SEED = 20261018


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = y.astype(np.uint8, copy=False)
    p = p.astype(np.float64, copy=False)
    edges = np.linspace(0.0, 1.0, 11)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            ece += float(m.mean() * abs(p[m].mean() - y[m].mean()))
    return {
        "auprc": float(average_precision_score(y, p)) if y.any() else float("nan"),
        "brier": float(np.mean((p - y) ** 2)),
        "ece_10bin": ece,
    }


def get_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    if not args.get("constrained_risk_feedback", False) or args.get("no_feedback", False):
        raise RuntimeError(f"unexpected checkpoint configuration: {checkpoint_path}")
    if not args.get("close_event_target", False):
        raise RuntimeError(f"checkpoint not trained on observed close-event targets: {checkpoint_path}")
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
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, args, ckpt


def reverse_valid_nodes(batch):
    """Reverse the valid vessel order within each scene, keeping padding last."""
    history, history_mask, future, node_mask, static = batch
    perms, inverses = [], []
    reversed_fields = [[] for _ in range(5)]
    fields = (history, history_mask, future, node_mask, static)
    for b in range(node_mask.shape[0]):
        valid = torch.nonzero(node_mask[b], as_tuple=False).flatten()
        invalid = torch.nonzero(~node_mask[b], as_tuple=False).flatten()
        perm = torch.cat((valid.flip(0), invalid))
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(perm.numel())
        perms.append(perm)
        inverses.append(inverse)
        for out, field in zip(reversed_fields, fields):
            out.append(field[b].index_select(0, perm))
    return tuple(torch.stack(x) for x in reversed_fields), perms, inverses


def collect_date(model, loader, device, close_nm, moderate_nm):
    keys = ("y_s", "y_m", "base_s", "samepass_rev_s", "permuted_rev_s",
            "base_m", "samepass_rev_m", "permuted_rev_m")
    result = {name: [] for name in keys}
    scene_count = pair_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = tuple(x.to(device) for x in batch)
            history, history_mask, future, node_mask, static = batch
            out = model(history, history_mask, node_mask, static)
            reversed_batch, _, inverse_perms = reverse_valid_nodes(batch)
            rh, rhm, _, rnm, rst = reversed_batch
            rout = model(rh, rhm, rnm, rst)
            ys = ground_truth_close_event(future, node_mask, close_nm) >= 0.5
            ym = ground_truth_close_event(future, node_mask, moderate_nm) >= 0.5
            for b in range(node_mask.shape[0]):
                ii, jj = torch.triu_indices(node_mask.shape[1], node_mask.shape[1], offset=1, device=device)
                valid = node_mask[b, ii] & node_mask[b, jj]
                i, j = ii[valid], jj[valid]
                if i.numel() == 0:
                    scene_count += 1
                    continue
                inv = inverse_perms[b]
                rev_same_s = out["risk"][b, j, i]
                rev_same_m = out["moderate_risk"][b, j, i]
                rev_input_s = rout["risk"][b, inv[j], inv[i]]
                rev_input_m = rout["moderate_risk"][b, inv[j], inv[i]]
                for name, value in (
                    ("y_s", ys[b, i, j]), ("y_m", ym[b, i, j]),
                    ("base_s", out["risk"][b, i, j]), ("samepass_rev_s", rev_same_s),
                    ("permuted_rev_s", rev_input_s), ("base_m", out["moderate_risk"][b, i, j]),
                    ("samepass_rev_m", rev_same_m), ("permuted_rev_m", rev_input_m),
                ):
                    result[name].append(value.detach().cpu().numpy())
                pair_count += int(i.numel())
                scene_count += 1
    return {k: np.concatenate(v) if v else np.empty(0) for k, v in result.items()}, scene_count, pair_count


def compare_one(y, base, alt):
    corr = spearmanr(base, alt).statistic if np.std(base) and np.std(alt) else float("nan")
    mb, ma = metrics(y, base), metrics(y, alt)
    return {
        "original_upper_triangle": mb,
        "alternative_direction": ma,
        "delta_alternative_minus_original": {k: ma[k] - mb[k] for k in mb},
        "mean_absolute_probability_change": float(np.mean(np.abs(alt - base))),
        "median_absolute_probability_change": float(np.median(np.abs(alt - base))),
        "p95_absolute_probability_change": float(np.quantile(np.abs(alt - base), 0.95)),
        "spearman_rank_correlation": float(corr),
    }


def bootstrap_day_ci(deltas_by_seed_day: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOT_SEED)
    n_seed, n_day = deltas_by_seed_day.shape
    draw = rng.integers(0, n_day, size=(N_BOOT, n_day))
    vals = np.empty(N_BOOT, dtype=np.float64)
    for start in range(0, N_BOOT, 5000):
        d = draw[start:start + 5000]
        vals[start:start + len(d)] = np.mean(deltas_by_seed_day[:, d], axis=(0, 2))
    return [float(x) for x in np.quantile(vals, [0.025, 0.975])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--checkpoints", type=Path, default=PKG / "checkpoints")
    ap.add_argument("--scenes-dir", type=Path,
                    default=Path(os.environ.get("KAORF_DATA_DIR", PKG / "data")) / "new_york")
    ap.add_argument("--event-root", type=Path, default=PKG / "results/events_blind_20260924")
    ap.add_argument("--out", type=Path, default=PKG / "results/trc_20261018/order_sensitivity")
    ap.add_argument("--save-arrays", action="store_true",
                    help="also write the per-seed pairwise probability arrays (about 6 MB each)")
    cli = ap.parse_args()

    out_dir = cli.out
    out_dir.mkdir(parents=True, exist_ok=True)

    info_path = cli.event_root / "EVENT_LOG_INFO.json"
    if info_path.is_file():
        dates = json.loads(info_path.read_text(encoding="utf-8"))["blind_dates"]
    else:
        dates = json.loads((cli.event_root / f"seed_{SEEDS[0]}" / "constrained_feedback" /
                            "event_log_meta.json").read_text(encoding="utf-8"))["blind_dates"]
    if len(dates) != 8:
        raise RuntimeError(f"expected the eight locked blind dates, found {len(dates)}")

    device = torch.device("cpu")
    records, day_metrics, audit = {}, {}, {}
    started = time.time()
    for seed in cli.seeds:
        checkpoint_path = cli.checkpoints / f"seed_{seed}" / "constrained_feedback" / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        model, args, ckpt = get_model(checkpoint_path, device)
        per_date = []
        combined = {k: [] for k in ("y_s", "y_m", "base_s", "samepass_rev_s", "permuted_rev_s",
                                    "base_m", "samepass_rev_m", "permuted_rev_m")}
        scenes_total = pairs_total = 0
        for date in dates:
            ds = SceneDataset([cli.scenes_dir / f"{date}.npz"])
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            arrays, n_scenes, n_pairs = collect_date(
                model, loader, device,
                float(args.get("close_threshold_nm", 0.25)),
                float(args.get("moderate_threshold_nm", 0.5)),
            )
            if not all(len(v) == n_pairs for v in arrays.values()):
                raise RuntimeError(f"pair alignment failed for {date}, seed {seed}")
            per_date.append(arrays)
            scenes_total += n_scenes
            pairs_total += n_pairs
            for k, v in arrays.items():
                combined[k].append(v)
        merged = {k: np.concatenate(v) for k, v in combined.items()}

        archived = np.load(cli.event_root / f"seed_{seed}" / "constrained_feedback" / "events.npz")
        for key, current, saved in (
            ("severe_labels", merged["y_s"], archived["y_severe"]),
            ("moderate_labels", merged["y_m"], archived["y_moderate"]),
            ("severe_probability", merged["base_s"], archived["p_severe"]),
            ("moderate_probability", merged["base_m"], archived["p_moderate"]),
        ):
            if current.shape != saved.shape:
                raise RuntimeError(f"archive alignment failed for seed {seed} {key}: "
                                   f"{current.shape} vs {saved.shape}")
            max_delta = float(np.max(np.abs(current.astype(float) - saved.astype(float)))) if current.size else 0.0
            if key.endswith("labels") and max_delta != 0:
                raise RuntimeError(f"label guard failed for seed {seed} {key}")
            if key.endswith("probability") and max_delta > 2e-5:
                raise RuntimeError(f"prediction guard failed for seed {seed} {key}: {max_delta}")
            audit[f"seed_{seed}/{key}"] = {"n": int(current.size), "max_abs_delta_vs_archived": max_delta}

        if cli.save_arrays:
            np.savez_compressed(out_dir / f"seed_{seed}_pairwise_order.npz", **merged)

        seed_result = {}
        for channel, ykey, basekey in (("severe", "y_s", "base_s"), ("moderate", "y_m", "base_m")):
            y, base = merged[ykey], merged[basekey]
            for mode, altkey in (
                ("same_forward_reverse_direction",
                 f"samepass_rev_{'s' if channel == 'severe' else 'm'}"),
                ("full_scene_order_reversal",
                 f"permuted_rev_{'s' if channel == 'severe' else 'm'}"),
            ):
                seed_result[f"{channel}/{mode}"] = compare_one(y, base, merged[altkey])
                store = day_metrics.setdefault(f"{channel}/{mode}",
                                               {"auprc": [], "brier": [], "ece_10bin": []})
                day_alt = [(metrics(day[ykey], day[altkey]), metrics(day[ykey], day[basekey]))
                           for day in per_date]
                for metric in ("auprc", "brier", "ece_10bin"):
                    store[metric].append([alt[metric] - orig[metric] for alt, orig in day_alt])

        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        records[str(seed)] = {
            "n_scenes": scenes_total,
            "n_pairs": pairs_total,
            "checkpoint": str(checkpoint_path.relative_to(PKG)),
            "checkpoint_sha256": checkpoint_hash,
            "selected_epoch": int(ckpt.get("epoch", -1)),
            "metrics": seed_result,
        }
        print(f"seed={seed} scenes={scenes_total} unordered_pairs={pairs_total}", flush=True)
        del model

    summary = {}
    for key, metric_arrays in day_metrics.items():
        summary[key] = {}
        for metric, per_seed in metric_arrays.items():
            arr = np.asarray(per_seed, dtype=np.float64)
            summary[key][metric] = {
                "equal_day_delta_mean": float(arr.mean()),
                "date_cluster_bootstrap_95ci": bootstrap_day_ci(arr),
                "seed_day_delta_matrix_shape": list(arr.shape),
                "n_dates": len(dates),
                "n_seeds": len(cli.seeds),
            }

    result = {
        "purpose": "post-hoc input-order sensitivity analysis; no training or checkpoint reselection",
        "device": str(device),
        "dates": dates,
        "seeds": cli.seeds,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pairs_are_physical_unordered_pairs": True,
        "comparison_modes": {
            "same_forward_reverse_direction": "compare p[i,j] with p[j,i] from the same forward pass",
            "full_scene_order_reversal": "reverse every valid vessel order, rerun the checkpoint, "
                                         "and align p'[j,i] to each original unordered pair",
        },
        "bootstrap": {"unit": "calendar date", "resamples": N_BOOT, "rng_seed": BOOT_SEED,
                      "interpretation": "descriptive sensitivity intervals; eight dates limit precision"},
        "archive_guards": audit,
        "per_seed": records,
        "equal_day_summaries": summary,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out_dir / "order_sensitivity_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "elapsed_seconds": result["elapsed_seconds"],
                      "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
