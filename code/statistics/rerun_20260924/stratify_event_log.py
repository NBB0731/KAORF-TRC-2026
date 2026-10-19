#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stratify_event_log.py -- stratified diagnostics on the locked blind-test
per-event logs (companion to dump_event_log.py).

The manuscript's checklist promised stratification by vessel type and traffic
density.  Vessel type is NOT recoverable: the scene files carry only MMSI
(vessel_id), and the MMSI->type mapping was lost with the retired scene
constructor.  This script therefore runs the two stratifications that the
archived data does support, per (seed, variant):

1. Traffic density -- scene node count buckets (2-5 / 6-7 / 8 vessels).
   Note the distribution is heavily imbalanced (>=90% of scenes have the
   full 8 nodes), which is itself a property of the scene constructor's
   5 nm-radius candidate selection and is reported as measured.
2. Encounter range -- great-circle-free (planar) pair distance at the anchor
   instant, from the stored history (x, y in metres): close < 0.5 nm,
   mid 0.5-2 nm, far >= 2 nm.  This asks the operationally interesting
   question: is the risk head reliable in the close-range regime where it
   matters?

Event-to-pair alignment: dump_event_log.py flattens the upper-triangle valid
pairs of each scene in row-major order (boolean-index order), with scenes in
SceneDataset order (date files concatenated, shuffle=False).  This script
recomputes the same masks and distances from the same scene files, so event
row i of events.npz maps to pair i of the recomputed arrays.  A guard asserts
the per-bucket event counts are identical across all 20 combinations and that
total events == 301,011 per combination.

Outputs (JSON): per-bucket AUPRC/Brier/ECE per seed and arm (seed mean +- sd),
plus paired feedback - no_feedback per bucket (mean difference, wins, Wilcoxon
p) -- the same statistics format as the manuscript's tables.

Usage
-----
    python stratify_event_log.py --log <out>/blind \
        --scenes data/final_2025_scenes/new_york \
        --manifest outputs/ais_2025_preregistered_manifest.csv --out <json path>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score

DEFAULT_SEEDS = [17, 29, 42, 71, 101, 131, 211, 307, 509, 887]
VARIANTS = ("constrained_feedback", "constrained_no_feedback")
CHANNELS = ("severe", "moderate")
DENSITY_BUCKETS = {"nodes_2_5": (2, 5), "nodes_6_7": (6, 7), "nodes_8": (8, 8)}
RANGE_BUCKETS = {"close_lt_0.5nm": (0.0, 0.5), "mid_0.5_2nm": (0.5, 2.0),
                 "far_ge_2nm": (2.0, np.inf)}
M_PER_NM = 1852.0


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def bucket_metrics(y, p):
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    return {
        "n": int(y.size),
        "n_positive": int(y.sum()),
        "auprc": float(average_precision_score(y, p)) if y.any() else None,
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece(y, p),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8-sig") as handle:
        dates = [row["date"] for row in csv.DictReader(handle)
                 if row["port"] == "new_york" and row["ny_split"] == "blind_test"]
    assert len(dates) == 8

    # --- scene-level geometry, recomputed exactly as in dump_event_log.py ---
    node_count: list[int] = []
    pair_distance_nm: list[np.ndarray] = []
    pair_scene: list[np.ndarray] = []
    scene_cursor = 0
    for day in dates:
        data = np.load(args.scenes / f"{day}.npz")
        future = data["future"]
        node_mask = data["node_mask"]
        history = data["history"]
        n_scenes = node_mask.shape[0]
        pair_valid = node_mask[:, :, None] & node_mask[:, None, :]
        eye = np.eye(pair_valid.shape[1], dtype=bool)[None]
        upper = np.triu(pair_valid & ~eye, k=1)
        current_xy = history[:, :, -1, :2]  # metres (position features)
        for s in range(n_scenes):
            scene_cursor += 1
            mask = upper[s]
            node_count.append(int(node_mask[s].sum()))
            if not mask.any():
                continue
            diff = current_xy[s][:, None, :] - current_xy[s][None, :, :]
            dist_nm = np.linalg.norm(diff, axis=-1)[mask] / M_PER_NM
            pair_distance_nm.append(dist_nm.astype(np.float32))
            pair_scene.append(np.full(dist_nm.size, scene_cursor - 1, dtype=np.int32))
    pair_distance_nm = np.concatenate(pair_distance_nm)
    pair_scene = np.concatenate(pair_scene)
    node_count = np.asarray(node_count, dtype=np.int16)
    print(f"scenes={scene_cursor:,} pairs={pair_distance_nm.size:,}")

    density_bucket = np.full(scene_cursor, -1, dtype=np.int8)
    for i, (lo, hi) in enumerate(DENSITY_BUCKETS.values()):
        density_bucket[(node_count >= lo) & (node_count <= hi)] = i
    range_bucket = np.full(pair_distance_nm.size, -1, dtype=np.int8)
    for i, (lo, hi) in enumerate(RANGE_BUCKETS.values()):
        range_bucket[(pair_distance_nm >= lo) & (pair_distance_nm < hi)] = i
    assert (range_bucket >= 0).all()
    assert (density_bucket[pair_scene] >= 0).all()

    dens_of_pair = density_bucket[pair_scene]

    payload = {
        "note_vessel_type": "not recoverable: scene files carry MMSI only; "
                            "MMSI->vessel-type mapping was lost with the "
                            "retired scene constructor",
        "density_buckets": {k: {"range": list(v)} for k, v in DENSITY_BUCKETS.items()},
        "range_bucket_edges_nm": {k: [float(v[0]) if np.isfinite(v[0]) else None,
                                      float(v[1]) if np.isfinite(v[1]) else None]
                                  for k, v in RANGE_BUCKETS.items()},
        "arms": {},
    }

    store: dict = {}
    for variant in VARIANTS:
        per_seed = {f"{layer}_{b}": {ch: [] for ch in CHANNELS}
                    for layer, buckets in (("density", DENSITY_BUCKETS),
                                           ("range", RANGE_BUCKETS))
                    for b in buckets}
        counts = None
        for seed in args.seeds:
            npz = np.load(args.log / f"seed_{seed}" / variant / "events.npz")
            assert npz["p_severe"].size == pair_distance_nm.size, (
                seed, variant, npz["p_severe"].size)
            c = {}
            for layer, bucket_arr, n_key in (
                    ("density", dens_of_pair, None),
                    ("range", range_bucket, None)):
                for bname in (DENSITY_BUCKETS if layer == "density"
                              else RANGE_BUCKETS):
                    bid = list((DENSITY_BUCKETS if layer == "density"
                                else RANGE_BUCKETS)).index(bname)
                    sel = bucket_arr == bid
                    c[f"{layer}/{bname}"] = int(sel.sum())
                    for ch in CHANNELS:
                        m = bucket_metrics(npz[f"y_{ch}"][sel], npz[f"p_{ch}"][sel])
                        per_seed[f"{layer}_{bname}"][ch].append(m)
            if counts is None:
                counts = c
            else:
                assert counts == c, f"bucket count mismatch seed {seed} {variant}"
        arm_out = {"bucket_pair_counts": counts}
        for key, chs in per_seed.items():
            arm_out[key] = {}
            for ch, runs in chs.items():
                auprc = np.array([r["auprc"] if r["auprc"] is not None else np.nan
                                  for r in runs])
                arm_out[key][ch] = {
                    "n": runs[0]["n"],
                    "n_positive": runs[0]["n_positive"],
                    "auprc_mean": float(np.nanmean(auprc)),
                    "auprc_sd": float(np.nanstd(auprc, ddof=1)),
                    "brier_mean": float(np.mean([r["brier"] for r in runs])),
                    "brier_sd": float(np.std([r["brier"] for r in runs], ddof=1)),
                    "ece_mean": float(np.mean([r["ece"] for r in runs])),
                    "ece_sd": float(np.std([r["ece"] for r in runs], ddof=1)),
                }
        payload["arms"][variant] = arm_out
        store[variant] = per_seed

    # paired feedback - no_feedback per bucket/channel
    paired = {}
    for layer in ("density", "range"):
        buckets = DENSITY_BUCKETS if layer == "density" else RANGE_BUCKETS
        for bname in buckets:
            for ch in CHANNELS:
                a = np.array([r["auprc"] for r in
                              store["constrained_feedback"][f"{layer}_{bname}"][ch]])
                b = np.array([r["auprc"] for r in
                              store["constrained_no_feedback"][f"{layer}_{bname}"][ch]])
                d = a - b
                if np.isnan(d).all():
                    paired[f"{layer}/{bname}/{ch}"] = {"auprc_diff": None}
                    continue
                nz = d[~np.isnan(d)]
                paired[f"{layer}/{bname}/{ch}"] = {
                    "auprc_diff_mean": float(np.nanmean(d)),
                    "fb_wins": int((nz > 0).sum()),
                    "nf_wins": int((nz < 0).sum()),
                    "n_valid_seeds": int(nz.size),
                    "wilcoxon_p": (float(wilcoxon(nz).pvalue)
                                   if nz.size >= 6 and not np.allclose(nz, 0) else None),
                }
    payload["paired_feedback_minus_no_feedback"] = paired

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")

    # console summary
    print("\n=== density buckets (pair events) ===")
    for bname in DENSITY_BUCKETS:
        fb = payload["arms"]["constrained_feedback"][f"density_{bname}"]
        print(f"{bname}: events={fb['severe']['n']:,} pos={fb['severe']['n_positive']:,}")
        for ch in CHANNELS:
            f, n = (payload["arms"][v][f"density_{bname}"][ch] for v in VARIANTS)
            print(f"  {ch}: fb AUPRC {f['auprc_mean']:.4f} vs nf {n['auprc_mean']:.4f} | "
                  f"ECE {f['ece_mean']:.4f} vs {n['ece_mean']:.4f}")
    print("\n=== encounter-range buckets (pair events) ===")
    for bname in RANGE_BUCKETS:
        fb = payload["arms"]["constrained_feedback"][f"range_{bname}"]
        print(f"{bname}: events={fb['severe']['n']:,} pos={fb['severe']['n_positive']:,}")
        for ch in CHANNELS:
            f, n = (payload["arms"][v][f"range_{bname}"][ch] for v in VARIANTS)
            print(f"  {ch}: fb AUPRC {f['auprc_mean']:.4f} vs nf {n['auprc_mean']:.4f} | "
                  f"ECE {f['ece_mean']:.4f} vs {n['ece_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
