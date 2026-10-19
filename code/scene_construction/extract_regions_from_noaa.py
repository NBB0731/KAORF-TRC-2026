"""Download daily NOAA AIS data, extract study regions, and write Parquet.

The national compressed file is retained only as a temporary cache unless
``--keep-raw`` is specified. A day is considered complete only when every
requested regional Parquet exists, so interrupted runs are safely resumable.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from profile_ais_day import REGIONS


# NOAA moved the 2025 daily files from the legacy coast.noaa.gov directory
# to Azure Blob Storage.  The 2025 files retain the ISO-date filename.
BASE_URL = "https://noaaocm.blob.core.windows.net/ais/csv2/csv{year}/ais-{day}.csv.zst"

SCHEMA = {
    "mmsi": pl.String,
    "base_date_time": pl.Datetime("us"),
    "longitude": pl.Float64,
    "latitude": pl.Float64,
    "sog": pl.Float32,
    "cog": pl.Float32,
    "heading": pl.Float32,
    "vessel_name": pl.String,
    "imo": pl.String,
    "call_sign": pl.String,
    "vessel_type": pl.Int16,
    "status": pl.Int16,
    "length": pl.Float32,
    "width": pl.Float32,
    "draft": pl.Float32,
    "cargo": pl.String,
    "transceiver": pl.String,
}


def day_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def download(url: str, target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    head = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head, timeout=120) as response:
        expected = int(response.headers["Content-Length"])
    if target.exists() and target.stat().st_size == expected:
        return
    if target.exists():
        target.replace(partial)
    if partial.exists() and partial.stat().st_size > expected:
        partial.unlink()
    # The public endpoint occasionally stalls on a full 150--300 MiB response.
    # Fetch verified 8 MiB byte ranges instead, so an interrupted connection
    # loses at most one small chunk and can resume without corrupting the file.
    chunk_bytes = 8 * 1024 * 1024
    failures = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected:
            partial.replace(target)
            return
        stop = min(offset + chunk_bytes - 1, expected - 1)
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{stop}"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206:
                    raise RuntimeError(f"server did not honor range request (status={response.status})")
                chunk = response.read(stop - offset + 1)
            if len(chunk) != stop - offset + 1:
                raise OSError(f"short chunk ({len(chunk)} bytes for requested {stop - offset + 1})")
            with partial.open("ab") as handle:
                handle.write(chunk)
            failures = 0
        except (urllib.error.URLError, OSError) as error:
            failures += 1
            if failures >= 20:
                raise RuntimeError(
                    f"could not complete {url}: 20 consecutive chunk failures at byte {offset}/{expected}"
                ) from error
            print(f"chunk retry {failures}/20 at {offset}/{expected}: {error}", flush=True)
            time.sleep(min(5 * failures, 30))
    partial.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--start", type=date.fromisoformat)
    date_group.add_argument(
        "--manifest",
        type=Path,
        help="CSV with a UTF-8 date column; processes each unique listed date once.",
    )
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=REGIONS,
        default=["new_york", "houston_galveston", "chesapeake"],
    )
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    if args.start is not None:
        if args.end is None:
            parser.error("--end is required with --start")
        selected_days = list(day_range(args.start, args.end))
    else:
        if args.end is not None:
            parser.error("--end cannot be used with --manifest")
        with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "date" not in reader.fieldnames:
                parser.error("--manifest must contain a date column")
            selected_days = sorted({date.fromisoformat(row["date"]) for row in reader})
        if not selected_days:
            parser.error("--manifest contains no dates")

    raw_dir = args.out / "raw_cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for region in args.regions:
        (args.out / region).mkdir(parents=True, exist_ok=True)

    print(f"selected {len(selected_days)} unique daily files", flush=True)
    for day in selected_days:
        outputs = {region: args.out / region / f"{day.isoformat()}.parquet" for region in args.regions}
        if all(path.exists() and path.stat().st_size > 0 for path in outputs.values()):
            print(f"skip complete day {day}", flush=True)
            continue

        raw = raw_dir / f"ais-{day.isoformat()}.csv.zst"
        url = BASE_URL.format(year=day.year, day=day.isoformat())
        # ``download`` verifies an existing cache file against the remote
        # Content-Length before accepting it.  This prevents an interrupted
        # file that was previously named without ``.part`` from being parsed.
        print(f"verify/download {day}: {url}", flush=True)
        download(url, raw)

        print(f"read {day}: {raw.stat().st_size / 2**20:.1f} MiB compressed", flush=True)
        frame = pl.read_csv(
            raw,
            schema_overrides=SCHEMA,
            try_parse_dates=True,
            null_values=["", "NA", "None"],
            low_memory=True,
        )
        frame = frame.filter(
            pl.col("longitude").is_between(-180, 180)
            & pl.col("latitude").is_between(-90, 90)
            & pl.col("mmsi").is_not_null()
            & pl.col("base_date_time").is_not_null()
        )

        for region, output in outputs.items():
            if output.exists() and output.stat().st_size > 0:
                continue
            xmin, xmax, ymin, ymax = REGIONS[region]
            subset = frame.filter(
                pl.col("longitude").is_between(xmin, xmax)
                & pl.col("latitude").is_between(ymin, ymax)
            ).sort(["base_date_time", "mmsi"])
            temporary = output.with_suffix(".parquet.part")
            subset.write_parquet(temporary, compression="zstd", statistics=True)
            temporary.replace(output)
            print(
                f"write {region}/{day}: {subset.height:,} rows, "
                f"{output.stat().st_size / 2**20:.2f} MiB",
                flush=True,
            )

        del frame
        if not args.keep_raw:
            raw.unlink()


if __name__ == "__main__":
    main()
