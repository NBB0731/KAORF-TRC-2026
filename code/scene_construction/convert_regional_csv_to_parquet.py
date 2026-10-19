"""Convert small regional Zstandard CSV extracts to analysis Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from extract_regions_from_noaa import SCHEMA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    for source in sorted(args.input_root.glob("*/*.csv.zst")):
        region = source.parent.name
        day = source.name.removesuffix(".csv.zst")
        output = args.output_root / region / f"{day}.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and output.stat().st_size > 0:
            print(f"skip {output}")
            continue
        frame = pl.read_csv(
            source,
            schema_overrides=SCHEMA,
            try_parse_dates=True,
            null_values=["", "NA", "None"],
            low_memory=True,
        ).sort(["base_date_time", "mmsi"])
        temporary = output.with_suffix(".parquet.part")
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.replace(output)
        print(f"write {output}: {frame.height:,} rows, {output.stat().st_size / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
