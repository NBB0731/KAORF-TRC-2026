"""Download daily NOAA MarineCadastre AIS files with resume support."""

from __future__ import annotations

import argparse
import shutil
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path


# NOAA moved the 2025 daily files from the legacy coast.noaa.gov directory
# to Azure Blob Storage.  The 2025 files retain the ISO-date filename.
BASE_URL = "https://noaaocm.blob.core.windows.net/ais/csv2/csv{year}/ais-{day}.csv.zst"


def content_length(url: str) -> int:
    for attempt in range(1, 11):
        try:
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request) as response:
                return int(response.headers["Content-Length"])
        except (urllib.error.URLError, OSError):
            if attempt == 10:
                raise
            time.sleep(min(5 * attempt, 30))
    raise RuntimeError("unreachable")


def dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for day in dates(args.start, args.end):
        name = f"ais-{day.isoformat()}.csv.zst"
        target = args.out / name
        partial = target.with_suffix(target.suffix + ".part")
        url = BASE_URL.format(year=day.year, day=day.isoformat())
        expected = content_length(url)
        if target.exists():
            if target.stat().st_size == expected:
                print(f"skip complete {target} ({expected} bytes)")
                continue
            # Preserve an incorrectly finalized partial download and resume it.
            if not partial.exists() or target.stat().st_size > partial.stat().st_size:
                target.replace(partial)
            else:
                target.unlink()
        if partial.exists() and partial.stat().st_size > expected:
            raise RuntimeError(f"partial file exceeds remote size: {partial}")

        attempts = 0
        while (partial.stat().st_size if partial.exists() else 0) < expected:
            attempts += 1
            if attempts > 200:
                raise RuntimeError(f"download remained incomplete after 200 attempts: {partial}")
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url)
            if offset:
                request.add_header("Range", f"bytes={offset}-")
            print(f"download {url} -> {target} (offset={offset}, expected={expected})", flush=True)
            try:
                with urllib.request.urlopen(request) as response:
                    resumed = offset > 0 and response.status == 206
                    mode = "ab" if resumed else "wb"
                    before = offset if resumed else 0
                    with partial.open(mode) as output:
                        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            except (urllib.error.URLError, OSError) as error:
                print(f"transient download error: {error}; retrying", flush=True)
                time.sleep(min(5 * attempts, 30))
                continue
            after = partial.stat().st_size
            print(f"received {after}/{expected} bytes", flush=True)
            if after <= before:
                raise RuntimeError(f"download made no progress: {partial}")
            if after > expected:
                raise RuntimeError(f"download exceeds remote size: {partial}")
        partial.replace(target)


if __name__ == "__main__":
    main()
