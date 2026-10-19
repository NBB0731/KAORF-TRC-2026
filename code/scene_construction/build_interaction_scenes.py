"""Build fixed-shape multi-vessel interaction scenes from regional AIS Parquet.

This development builder preserves missing one-minute bins and the actual time
gap feature instead of fabricating interpolated observations. It is intended for
the first RFIGN prototype; the paper dataset builder will add multi-day boundary
handling and richer fuzzy-risk labels after the representation is validated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


FEATURE_NAMES = ["x_m", "y_m", "vx_mps", "vy_mps", "sin_cog", "cos_cog", "sog_kn", "gap_min"]
KNOT_TO_MPS = 0.514444


def neighborhoods(xy: np.ndarray, radius_m: float, max_nodes: int) -> list[np.ndarray]:
    pairs = cKDTree(xy).query_pairs(radius_m, output_type="ndarray")
    if not len(pairs):
        return []
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(xy), len(xy)))
    _, labels = connected_components(graph, directed=False)
    results: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for label in np.unique(labels):
        component = np.flatnonzero(labels == label)
        if len(component) < 2:
            continue
        if len(component) <= max_nodes:
            key = tuple(component.tolist())
            seen.add(key)
            results.append(component)
            continue
        tree = cKDTree(xy[component])
        for local_ego in range(len(component)):
            _, local_idx = tree.query(xy[component[local_ego]], k=max_nodes)
            selected = np.sort(component[np.atleast_1d(local_idx)])
            key = tuple(selected.tolist())
            if key not in seen:
                seen.add(key)
                results.append(selected)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--history-minutes", type=int, default=20)
    parser.add_argument("--future-minutes", type=int, default=30)
    parser.add_argument("--future-step", type=int, default=5)
    parser.add_argument("--anchor-step", type=int, default=5)
    parser.add_argument("--radius-nm", type=float, default=5.0)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--min-history", type=int, default=10)
    args = parser.parse_args()

    if args.input.suffix.lower() == ".csv":
        records = []
        with args.input.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    timestamp = datetime.fromisoformat(row["base_date_time"].replace("Z", "+00:00"))
                    records.append(
                        {
                            "mmsi": row["mmsi"], "base_date_time": timestamp,
                            "longitude": float(row["longitude"]), "latitude": float(row["latitude"]),
                            "sog": float(row["sog"]), "cog": float(row["cog"]),
                            "length": float(row.get("length") or 0.0),
                            "width": float(row.get("width") or 0.0),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        records.sort(key=lambda row: (row["base_date_time"], row["mmsi"]))
        lon0 = float(np.median([row["longitude"] for row in records]))
        lat0 = float(np.median([row["latitude"] for row in records]))
    else:
        import polars as pl

        frame = pl.read_parquet(args.input).sort(["base_date_time", "mmsi"])
        records = list(frame.iter_rows(named=True))
        lon0 = float(frame["longitude"].median())
        lat0 = float(frame["latitude"].median())
    cos_lat0 = math.cos(math.radians(lat0))

    # minute -> latest observation is a deterministic anti-duplication rule.
    tracks: dict[str, dict[int, tuple[float, float, float, float, float, float]]] = defaultdict(dict)
    for row in records:
        timestamp = row["base_date_time"]
        sog, cog = row["sog"], row["cog"]
        if timestamp is None or sog is None or cog is None:
            continue
        if not (0 <= float(sog) < 60 and 0 <= float(cog) < 360):
            continue
        minute = int(timestamp.timestamp()) // 60
        x = (float(row["longitude"]) - lon0) * 111_320.0 * cos_lat0
        y = (float(row["latitude"]) - lat0) * 110_540.0
        tracks[str(row["mmsi"])][minute] = (
            x,
            y,
            float(sog),
            float(cog),
            float(row["length"] or 0.0),
            float(row["width"] or 0.0),
        )

    all_minutes = [minute for track in tracks.values() for minute in track]
    start = min(all_minutes) + args.history_minutes - 1
    end = max(all_minutes) - args.future_minutes
    start += (-start) % args.anchor_step
    future_offsets = np.arange(args.future_step, args.future_minutes + 1, args.future_step)
    h = args.history_minutes
    f = len(future_offsets)
    n = args.max_nodes

    histories: list[np.ndarray] = []
    history_masks: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    node_masks: list[np.ndarray] = []
    statics: list[np.ndarray] = []
    vessel_ids: list[np.ndarray] = []
    anchors: list[int] = []

    for anchor in range(start, end + 1, args.anchor_step):
        candidates = []
        for mmsi, track in tracks.items():
            current = track.get(anchor)
            if current is None or current[2] <= 0.5:
                continue
            hist_count = sum((anchor - offset) in track for offset in range(h))
            if hist_count < args.min_history:
                continue
            if not all((anchor + int(offset)) in track for offset in future_offsets):
                continue
            candidates.append((mmsi, track))
        if len(candidates) < 2:
            continue

        current_xy = np.asarray([track[anchor][:2] for _, track in candidates], dtype=np.float32)
        for selected in neighborhoods(current_xy, args.radius_nm * 1852.0, n):
            count = len(selected)
            scene_center = current_xy[selected].mean(axis=0)
            hist = np.zeros((n, h, len(FEATURE_NAMES)), dtype=np.float32)
            hist_mask = np.zeros((n, h), dtype=bool)
            future = np.zeros((n, f, 2), dtype=np.float32)
            static = np.zeros((n, 2), dtype=np.float32)
            ids = np.zeros(n, dtype=np.int64)
            node_mask = np.zeros(n, dtype=bool)

            for slot, candidate_index in enumerate(selected):
                mmsi, track = candidates[int(candidate_index)]
                node_mask[slot] = True
                ids[slot] = int(mmsi) if mmsi.isdigit() else 0
                current = track[anchor]
                static[slot] = current[4:6]
                previous_minute = None
                for hist_index, minute in enumerate(range(anchor - h + 1, anchor + 1)):
                    state = track.get(minute)
                    if state is None:
                        continue
                    x, y, sog, cog, _, _ = state
                    angle = math.radians(cog)
                    gap = 0.0 if previous_minute is None else float(minute - previous_minute)
                    hist[slot, hist_index] = (
                        x - scene_center[0],
                        y - scene_center[1],
                        sog * KNOT_TO_MPS * math.sin(angle),
                        sog * KNOT_TO_MPS * math.cos(angle),
                        math.sin(angle),
                        math.cos(angle),
                        sog,
                        gap,
                    )
                    hist_mask[slot, hist_index] = True
                    previous_minute = minute
                for future_index, offset in enumerate(future_offsets):
                    x, y, *_ = track[anchor + int(offset)]
                    future[slot, future_index] = (x - scene_center[0], y - scene_center[1])

            histories.append(hist)
            history_masks.append(hist_mask)
            futures.append(future)
            node_masks.append(node_mask)
            statics.append(static)
            vessel_ids.append(ids)
            anchors.append(anchor)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        history=np.asarray(histories),
        history_mask=np.asarray(history_masks),
        future=np.asarray(futures),
        node_mask=np.asarray(node_masks),
        static=np.asarray(statics),
        vessel_id=np.asarray(vessel_ids),
        anchor_minute=np.asarray(anchors, dtype=np.int64),
    )
    metadata = {
        "source": str(args.input),
        "scenes": len(histories),
        "feature_names": FEATURE_NAMES,
        "reference_lon_lat": [lon0, lat0],
        "history_minutes": h,
        "future_offsets_minutes": future_offsets.tolist(),
        "radius_nm": args.radius_nm,
        "max_nodes": n,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
