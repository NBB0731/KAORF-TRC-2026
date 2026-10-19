"""Build one fixed-shape scene archive per pre-registered port-day.

This orchestrator intentionally delegates all scene semantics to
``build_interaction_scenes.py`` and only fixes the port/date inventory and
resume behavior.  It never mixes dates while constructing a scene archive.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--regional-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ports", nargs="+", default=["new_york", "houston_galveston", "chesapeake"])
    parser.add_argument("--history-minutes", type=int, default=20)
    parser.add_argument("--future-minutes", type=int, default=30)
    parser.add_argument("--future-step", type=int, default=5)
    parser.add_argument("--anchor-step", type=int, default=5)
    parser.add_argument("--radius-nm", type=float, default=5.0)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--min-history", type=int, default=10)
    args = parser.parse_args()

    builder = Path(__file__).with_name("build_interaction_scenes.py")
    chosen = [row for row in rows(args.manifest) if row["port"] in set(args.ports)]
    missing = []
    for row in chosen:
        port, day = row["port"], row["date"]
        source = args.regional_root / port / f"{day}.parquet"
        output = args.out / port / f"{day}.npz"
        if output.exists() and output.stat().st_size > 0:
            print(f"skip complete {port}/{day}", flush=True)
            continue
        if not source.exists():
            missing.append(str(source))
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(builder), str(source), str(output),
            "--history-minutes", str(args.history_minutes),
            "--future-minutes", str(args.future_minutes),
            "--future-step", str(args.future_step),
            "--anchor-step", str(args.anchor_step),
            "--radius-nm", str(args.radius_nm),
            "--max-nodes", str(args.max_nodes),
            "--min-history", str(args.min_history),
        ]
        print(f"build {port}/{day}", flush=True)
        subprocess.run(command, check=True)
    if missing:
        raise SystemExit("missing regional inputs:\n" + "\n".join(missing))


if __name__ == "__main__":
    main()
