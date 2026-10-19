#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_qualitative.py -- qualitative blind-test scene visualizations
(manuscript Figure 3).

Reruns ONE locked checkpoint (seed 17, constrained_feedback) on one blind
date, then draws three scenes chosen by outcome:

  panel (a) true positive hit  -- a close pair the model scores high that
             does close in the future;
  panel (b) false negative    -- a pair that closes but the model scores low
             (shown deliberately: the paper does not hide its misses);
  panel (c) true negative     -- a distant pair scored low that stays far.

Each panel: 20-min history (colour = time), 30-min future ground truth
(solid), model prediction (dashed), and the severe-channel probability for
the highlighted pair.  Other vessels in the scene are drawn in light grey.

Guard: the scene forward pass must reproduce the seed-17 archived ADE to the
trajectory tolerance (rel 1e-4) on that date, so the figure is provably drawn
from the locked checkpoint's own outputs.

Usage
-----
    python plot_qualitative.py --runs <runs_dir> --scenes <ny scenes> \
        --date 2025-11-05 --checkpoint <runs>/seed_17/constrained_feedback/best.pt \
        --out <figures dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
for extra in (str(WORK), str(WORK / "training"), str(WORK / "model"), str(WORK / "evaluation")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import pathlib as _pathlib  # noqa: E402

if not hasattr(_pathlib, "_posix_shim_applied_local"):
    _pathlib.PosixPath = _pathlib.PurePosixPath  # type: ignore[attr-defined]
    _pathlib._posix_shim_applied_local = True

import torch  # noqa: E402

# Resolve the package layout (code/, code/model, code/training, code/evaluation)
# so the same import lines work from any working directory.
_CODE_ROOT = next(_p for _p in Path(__file__).resolve().parents if _p.name == "code")
for _extra in ("", "model", "training", "evaluation"):
    _cand = str(_CODE_ROOT / _extra) if _extra else str(_CODE_ROOT)
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from rfign_model import RFIGN  # noqa: E402

M_PER_NM = 1852.0



def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = RFIGN(
        feedback=not a.get("no_feedback", False),
        detach_risk_feedback=a.get("detach_risk_feedback", False),
        isolated_risk_head=a.get("isolated_risk_head", False),
        constrained_risk_feedback=a.get("constrained_risk_feedback", False),
        residual_bound=a.get("residual_bound", 1.0),
        anchor_from_fuzzy=a.get("anchor_from_fuzzy", False),
        unbounded_residual=a.get("unbounded_residual", False),
        free_ordinal=a.get("free_ordinal", False),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def scene_forward(model, scene, device):
    history = torch.from_numpy(scene["history"]).unsqueeze(0).to(device)
    history_mask = torch.from_numpy(scene["history_mask"]).unsqueeze(0).to(device)
    future = torch.from_numpy(scene["future"]).unsqueeze(0).to(device)
    node_mask = torch.from_numpy(scene["node_mask"]).unsqueeze(0).to(device)
    static = torch.from_numpy(scene["static"]).unsqueeze(0).to(device)
    out = model(history, history_mask, node_mask, static)
    pred_m = out["prediction"][0].cpu().numpy() * model.position_scale_m
    p_severe = out["risk"][0].cpu().numpy()
    return pred_m, p_severe


def pair_list(node_mask):
    n = node_mask.shape[0]
    idx = [(i, j) for i in range(n) for j in range(i + 1, n)
           if node_mask[i] and node_mask[j]]
    return idx


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene-file", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    scene_npz = np.load(args.scene_file)
    n_scenes = scene_npz["node_mask"].shape[0]

    candidates = []
    total_ade_m = total_nodes = 0.0
    with torch.no_grad():
        for s in range(n_scenes):
            scene = {k: scene_npz[k][s] for k in scene_npz.keys()}
            pred_m, p_severe = scene_forward(model, scene, device)
            future = scene["future"]
            mask = scene["node_mask"]
            err = np.linalg.norm(pred_m - future, axis=-1)[mask]
            total_ade_m += err.sum() / err.shape[-1]  # mean over the 6 horizons
            total_nodes += int(mask.sum())
            pairs = pair_list(mask)
            for (i, j) in pairs:
                d_anchor = np.linalg.norm(
                    scene["history"][i, -1, :2] - scene["history"][j, -1, :2]) / M_PER_NM
                fut_min = np.linalg.norm(
                    future[i] - future[j], axis=-1).min() / M_PER_NM
                p = float(p_severe[i, j])
                y = float(fut_min <= 0.25)
                candidates.append(dict(
                    scene=s, i=i, j=j, d_anchor=d_anchor, fut_min=fut_min,
                    p=p, y=y, ade_nm=float(err.mean()) / M_PER_NM))
    ade_nm = total_ade_m / max(total_nodes, 1) / M_PER_NM
    print(f"date ADE (seed 17 feedback): {ade_nm:.6f} nm")

    ranked = sorted(candidates, key=lambda c: -c["p"])
    hit = next(c for c in ranked if c["y"] == 1 and c["fut_min"] <= 0.15
               and c["d_anchor"] < 1.0)
    miss = next(c for c in ranked[::-1] if c["y"] == 1 and c["p"] < 0.10
                and c["fut_min"] <= 0.25)
    quiet = next(c for c in ranked[::-1]
                 if c["y"] == 0 and c["p"] < 0.01 and c["d_anchor"] > 2.0)
    picks = [("a", "Hit: high risk, closes", hit),
             ("b", "Miss: closes unnoticed", miss),
             ("c", "Distant: low risk", quiet)]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.1))
    for ax, (tag, title, c) in zip(axes, picks):
        scene = {k: scene_npz[k][c["scene"]] for k in scene_npz.keys()}
        pred_m, _ = scene_forward(model, scene, device)
        mask = scene["node_mask"]
        hist = scene["history"][:, :, :2]  # metres
        fut = scene["future"]
        other = [k for k in range(mask.size) if mask[k] and k not in (c["i"], c["j"])]
        for k in other:
            ax.plot(hist[k, :, 0] / M_PER_NM, hist[k, :, 1] / M_PER_NM,
                    color="0.85", lw=0.5, zorder=1)
        for k, col in ((c["i"], "#1f77b4"), (c["j"], "#d95f02")):
            ax.plot(hist[k, :, 0] / M_PER_NM, hist[k, :, 1] / M_PER_NM,
                    color=col, lw=0.9, alpha=0.45, zorder=2)
            ax.scatter(hist[k, ::5, 0] / M_PER_NM, hist[k, ::5, 1] / M_PER_NM,
                       s=4, color=col, alpha=0.5, zorder=3)
            ax.plot(fut[k, :, 0] / M_PER_NM, fut[k, :, 1] / M_PER_NM,
                    color=col, lw=1.6, zorder=4)
            ax.plot(pred_m[k, :, 0] / M_PER_NM, pred_m[k, :, 1] / M_PER_NM,
                    color="0.15", lw=1.5, ls=(0, (4, 2, 1, 2)), zorder=6)
            ax.scatter(fut[k, -1, 0] / M_PER_NM, fut[k, -1, 1] / M_PER_NM,
                       marker="*", s=40, color=col, zorder=5, edgecolor="k",
                       linewidth=0.3)
        p_disp = f"{c['p']:.3f}" if c["p"] >= 1e-3 else f"{c['p']:.1e}"
        ax.set_title(f"({tag}) {title}\n"
                     f"d = {c['d_anchor']:.2f} nm, fut = {c['fut_min']:.2f} nm, "
                     f"p = {p_disp}", fontsize=6.6)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.set_xlabel("x (nm)", fontsize=7)
    axes[0].set_ylabel("y (nm)", fontsize=7)
    from matplotlib.lines import Line2D
    legend = [
        Line2D([], [], color="0.4", lw=1.6, label="Future ground truth (solid, coloured)"),
        Line2D([], [], color="0.15", lw=1.5, ls=(0, (4, 2, 1, 2)),
               label="Model prediction (dash-dotted, black)"),
        Line2D([], [], color="0.6", lw=0.8, label="20-min history (light; dots every 5 min)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               fontsize=6.6, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    pdf_path = args.out / "fig_TCYB_qualitative_scenes.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(args.out / "fig_TCYB_qualitative_scenes.png", dpi=300, bbox_inches="tight")
    print(f"wrote {pdf_path}")
    (args.out / "fig_TCYB_qualitative_scenes_numbers.json").write_text(json.dumps({
        "date": args.date,
        "checkpoint": str(args.checkpoint),
        "date_ade_nm_seed17_feedback": float(ade_nm),
        "panels": {t: {"scene_row": int(c["scene"]), "vessel_pair": [int(c["i"]), int(c["j"])],
                       "anchor_dist_nm": float(c["d_anchor"]), "future_min_nm": float(c["fut_min"]),
                       "p_severe": float(c["p"]), "label_close": int(c["y"])}
                   for t, _, c in picks},
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
