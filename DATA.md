# Data: source, download and scene construction

The package archives code, checkpoints and results. It does **not** archive the
AIS source files or the derived scene archives; both are rebuilt from a public
source with the scripts in `code/scene_construction/`.

## 1. Source

NOAA/BOEM Marine Cadastre, vessel traffic (AIS) data, calendar year 2025:

- landing page: <https://www.coast.noaa.gov/digitalcoast/data/vesseltraffic.html>
- data dictionary: <https://coast.noaa.gov/data/marinecadastre/ais/data-dictionary.pdf>

The 145 rows of `preregistered/ais_2025_preregistered_manifest.csv` are the
single source of partition truth: 48 dates for each of the three ports. New York
splits into `train` (32 dates), `val` (8) and `blind_test` (8); the two external
ports are `external_only` with `allowed_for_tuning=no`.

## 2. Directory convention

Scripts read scene archives from `$KAORF_DATA_DIR` when that variable is set,
and from `<package>/data` otherwise:

```
data/
  new_york/<YYYY-MM-DD>.npz
  houston_galveston/<YYYY-MM-DD>.npz
  chesapeake/<YYYY-MM-DD>.npz
```

## 3. Scene archive fields

Each per-date NPZ holds five arrays, one row per scene:

| Field | Shape | Meaning |
|---|---|---|
| `history` | `(N, 8, 20, 8)` | vessel, time step, feature; 20 steps at 1-minute spacing before the anchor time |
| `history_mask` | `(N, 8, 20)` | observed history flags; padded time steps are zero |
| `future` | `(N, 8, 6, 2)` | six future positions at 5-minute steps, local projection, metres |
| `node_mask` | `(N, 8)` | valid vessel slots; up to eight vessels per scene |
| `static` | `(N, 8, 2)` | length and width, normalised |
| `vessel_id` | `(N, 8)` | MMSI per slot, for auditing and for the VesselType extension discussed in the manuscript |
| `anchor_minute` | `(N,)` | minute of day of the anchor time |

Labels are derived at training time, not stored: the minimum separation of a
pair is taken over the six predicted 5-minute points (no interpolation), and the
two thresholds are 0.25 nm and 0.50 nm.

## 4. Build order

```bash
python code/scene_construction/download_noaa_ais.py          # 1. raw daily files for the manifest dates
python code/scene_construction/extract_regions_from_noaa.py  # 2. clip to the port AOI, filter
python code/scene_construction/convert_regional_csv_to_parquet.py
python code/scene_construction/build_interaction_scenes.py   # 3. scene semantics
python code/scene_construction/build_final_scenes.py         # 4. run over the (port, date) inventory
python code/scene_construction/build_experiment_manifest.py  # 5. regenerate the pre-registered manifest
python code/scene_construction/audit_final_dataset.py        # 6. schema and partition audit
```

Step 5 is a check, not a dependency: the archived manifest is the version used
for every reported run, and regenerating it should return an identical file.

## 5. Volume and licensing

A full year of daily AIS files for the three port regions is on the order of one
hundred gigabytes before extraction and a few gigabytes as scene archives, which
is why they are not shipped. The source is public and does not require
registration. Check the Marine Cadastre terms before redistributing any derived
extract.
