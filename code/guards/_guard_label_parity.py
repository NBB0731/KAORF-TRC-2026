# -*- coding: utf-8 -*-
"""Guard: does our numpy label path reproduce the torch reference exactly?

Compares `pairwise_min_distance` in evaluate_baselines_on_scenes.py against a
literal transcription of train_rfign_dev.ground_truth_close_event, on synthetic
data with a known answer plus random data.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Resolve the package layout (code/, code/model, code/training, code/evaluation)
# so the same import lines work from any working directory.
_CODE_ROOT = next(_p for _p in Path(__file__).resolve().parents if _p.name == "code")
for _extra in ("", "model", "training", "evaluation"):
    _cand = str(_CODE_ROOT / _extra) if _extra else str(_CODE_ROOT)
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from evaluate_baselines_on_scenes import (  # noqa: E402

    MODERATE_THRESHOLD_NM, NM_TO_M, SEVERE_THRESHOLD_NM, pairwise_min_distance,
)

try:
    import torch
except ImportError:  # pragma: no cover
    print("torch unavailable -- running numpy-only self test")
    torch = None


def reference_torch(future_m, node_mask, threshold_nm):
    """Literal copy of train_rfign_dev.ground_truth_close_event."""
    difference = future_m[:, :, None] - future_m[:, None, :]
    minimum_distance_m = torch.linalg.vector_norm(difference, dim=-1).amin(dim=-1)
    valid = node_mask[:, :, None] & node_mask[:, None, :]
    eye = torch.eye(valid.shape[1], dtype=torch.bool, device=future_m.device).unsqueeze(0)
    return ((minimum_distance_m <= threshold_nm * 1852.0) & valid & ~eye).float()


rng = np.random.default_rng(20260924)

# ---------- case 1: hand-built, two vessels converging to a known minimum
future = np.zeros((1, 2, 6, 2), dtype=np.float32)
# vessel 0 sits at origin; vessel 1 walks toward it, closest at step 2 (distance 100 m)
future[0, 0, :, :] = 0.0
future[0, 1, :, 0] = [500.0, 200.0, 100.0, 200.0, 400.0, 700.0]
future[0, 1, :, 1] = 0.0
mask = np.ones((1, 2), dtype=bool)
min_dist, pair_valid = pairwise_min_distance(future, mask)
print("case1 min_dist =", min_dist[0, 0, 1], "(expect 100.0)")
assert abs(min_dist[0, 0, 1] - 100.0) < 1e-4, min_dist
assert pair_valid[0, 0, 1] and pair_valid[0, 1, 0]
assert not pair_valid[0, 0, 0], "self-pairs must be masked out"
print("case1 OK: 100 m <= 0.25nm(463m) -> severe positive; > 0.25nm -> as expected")

if torch is not None:
    # ---------- case 2: random data, both thresholds, compare elementwise
    for trial in range(5):
        n, v = 3, 6
        fut = torch.tensor(rng.normal(scale=800.0, size=(n, v, 6, 2)), dtype=torch.float32)
        nm_t = torch.ones((n, v), dtype=torch.bool)
        # randomly drop a few vessels so masking is exercised
        nm_t[0, 3] = False
        nm_t[1, 5] = False
        nm_np = nm_t.numpy()

        for thr in (SEVERE_THRESHOLD_NM, MODERATE_THRESHOLD_NM):
            ref = reference_torch(fut, nm_t, thr).numpy()
            md, pv = pairwise_min_distance(fut.numpy(), nm_np)
            mine = ((md <= thr * NM_TO_M) & pv).astype(np.float32)
            if not np.array_equal(ref, mine):
                bad = np.argwhere(ref != mine)
                print("MISMATCH trial", trial, "thr", thr, "at", bad[:5])
                print(" ref ", ref[tuple(bad[0])], " mine", mine[tuple(bad[0])])
                raise SystemExit("LABEL GUARD FAILED")
        md, pv = pairwise_min_distance(fut.numpy(), nm_np)
        ref_md = torch.linalg.vector_norm(
            fut[:, :, None] - fut[:, None, :], dim=-1).amin(dim=-1).numpy()
        if not np.allclose(md, ref_md, atol=1e-3):
            print("min-distance mismatch, max abs diff",
                  np.abs(md - ref_md).max())
            raise SystemExit("MIN-DIST GUARD FAILED")
    print("case2 OK: labels and min-distances match the torch reference on 5 random trials")
print("GUARD PASSED")
