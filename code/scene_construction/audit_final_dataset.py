"""Create a machine-readable audit manifest for regional AIS inputs and scenes.

The manifest records SHA-256 hashes and basic row/scene counts.  It is intended
for the final, locked experiment only; it does not inspect model outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def selected_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--regional-root", type=Path, required=True)
    parser.add_argument("--scenes-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    entries: list[dict[str, object]] = []
    for row in selected_rows(args.manifest):
        port, day = row["port"], row["date"]
        regional = args.regional_root / port / f"{day}.parquet"
        item: dict[str, object] = {
            "port": port,
            "date": day,
            "calendar_stratum": row["calendar_stratum"],
            "ny_split": row["ny_split"],
            "allowed_for_tuning": row["allowed_for_tuning"],
            "regional_path": str(regional),
            "regional_exists": regional.exists(),
        }
        if regional.exists():
            frame = pl.scan_parquet(regional)
            item.update(
                regional_bytes=regional.stat().st_size,
                regional_sha256=sha256(regional),
                regional_rows=int(frame.select(pl.len()).collect().item()),
                unique_mmsi=int(frame.select(pl.col("mmsi").n_unique()).collect().item()),
            )
        if args.scenes_root is not None:
            scene = args.scenes_root / port / f"{day}.npz"
            item.update(scene_path=str(scene), scene_exists=scene.exists())
            if scene.exists():
                with np.load(scene, allow_pickle=False) as archive:
                    item["scenes"] = int(archive["history"].shape[0])
                item["scene_bytes"] = scene.stat().st_size
                item["scene_sha256"] = sha256(scene)
        entries.append(item)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "polars": pl.__version__,
        "manifest_sha256": sha256(args.manifest),
        "entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    complete = sum(bool(row["regional_exists"]) for row in entries)
    print(f"wrote {args.out}: {complete}/{len(entries)} regional files present")


if __name__ == "__main__":
    main()
