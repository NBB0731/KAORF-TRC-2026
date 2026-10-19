# Approved local re-evaluation, 2026-09-24 (Figures 1–2 of the manuscript)

## What this is

The per-event probability logs and per-node trajectory errors behind the
manuscript's Figure 1 (reliability diagrams, PR curves) and Figure 2
(per-horizon displacement error) required a new forward pass over the locked
New York blind partition. That action is reserved for explicit approval under
the one-evaluation rule of the pre-registration; approval was given on
2026-09-24 and the re-evaluation was executed on a local Windows CPU host.

## Protocol guarantees (identical to the archived pipeline)

- 20 locked checkpoints: 10 pre-registered seeds × the two Table I arms
  (`constrained_feedback`, `constrained_no_feedback`). The two
  `seed_17_cpu_imported_20260923` copies are NOT part of the locked set and
  are excluded.
- No training, no fine-tuning, no re-selection. Each checkpoint is rebuilt
  from its own stored `args` with a `strict=True` state-dict load.
- The 8 blind dates (2025-11-05/09/19/23, 12-03/14/17/28) are read in locked
  manifest order; the blind partition is never passed to a trainer.
- Guards ran before any figure was drawn; a guard failure aborts the run.

## Guard results (all 20 combinations passed)

| Quantity | Tolerance | Worst observed drift |
|---|---|---|
| Severe/moderate AUPRC | abs 2e-6, rel 2e-5 | abs 3.8e-6 (rel 8.3e-6) |
| Severe/moderate Brier | abs 2e-6, rel 2e-5 | abs 3.0e-7 |
| Positive rate | exact | 0 (bitwise) |
| Binned ECE | abs 1e-4 (see note) | abs 1.6e-5 |
| ADE / FDE (trajectory) | abs 5e-4 nm, rel 1e-4 | rel 3.9e-5 (≈ 11 cm) |
| Day scene counts sum == total | exact | exact (12,103 scenes, 301,011 events, 89,077 node rows per combination) |

**ECE tolerance note.** ECE is a discontinuous 10-bin statistic: GPU→CPU
float32 last-bit differences flip bin membership for events near bin edges.
The continuous metrics pass the original 2e-6/2e-5 guard unchanged; ECE alone
drifts ~1e-5. The 1e-4 tolerance is ~140× smaller than the smallest ECE effect
the manuscript interprets (0.0089 → 0.0822). The same reasoning applies to the
trajectory ADE/FDE tolerance: `vector_norm` under float32 differs in last bits
between GPU and CPU; the observed worst relative drift is 3.9e-5 (absolute
difference below 11 cm), and the tolerance is 5e-4 nm ≈ 0.9 m.

## Local host differences (documented deviations from the cloud original)

1. **PosixPath unpickle shim** (both dump scripts): the locked checkpoints
   were pickled on Linux and their stored `args` contain `pathlib.PosixPath`;
   Python 3.13 on Windows cannot instantiate PosixPath during unpickling, so
   the scripts remap `pathlib.PosixPath` to `pathlib.PurePosixPath` before
   `torch.load`. Deserialisation shim only — no stored path is ever read.
2. The model source is the same file that was deployed to the cloud instance
   for the ablation stage (`rfign_model.py`, sha256-verified at deploy time),
   re-exposed here as an `rfign` package shim.
3. `--workers 0` (Windows DataLoader spawn); `--device cpu`.

## Environment

- Windows 10 host, Python 3.13.12, PyTorch 2.14.0+cpu, NumPy (venv build),
  scikit-learn 1.9.1, matplotlib 3.11.2, SciPy (Wilcoxon).
- Runtime: ~12 s per checkpoint per pass (4 min per full 20-combination pass).

## Files

- `results/events_blind_20260924/seed_<s>/<arm>/events.npz` — per-event risk
  log (y/p for both channels, anchor, scene index), one row per vessel pair.
- `results/events_blind_20260924/seed_<s>/<arm>/traj_errors.npz` — per-node
  per-horizon displacement error in metres (6 horizons = 5..30 min).
- `results/events_blind_20260924/seed_<s>/<arm>/{event_log,traj}_meta.json` —
  per-run guard deltas, checkpoint sha256, recomputed aggregates.
- `results/events_blind_20260924/{EVENT_LOG,TRAJ_LOG}_INFO.json` — run-level
  records and the (empty) failure lists.
- `results/figures_20260924/` — Figures 1–2 (PDF/PNG) and the plotted numbers
  as JSON (per-seed AP/ECE, bin structure, PR grid, paired per-horizon stats).
- `results/figures_20260924/stratified_diagnostics.json` — occupancy and
  anchor-range stratification (manuscript Table X); bucket event counts are
  asserted identical across all twenty combinations before any metric is
  computed. Vessel-type stratification is not recoverable (MMSI-only scene
  files; the MMSI-to-type mapping was lost with the scene constructor).
- `results/figures_20260924/fig_TCYB_qualitative_scenes*` — manuscript Figure 3
  (three scenes, single-checkpoint forward, seed 17 / 2025-11-05; the same
  pass reproduces that date's displacement error, 0.8174 nm).
- `code/statistics/rerun_20260924/` — the six scripts used, including the
  two dumper variants with the tolerances and shims documented above,
  `stratify_event_log.py` (Table X) and `plot_qualitative.py` (Figure 3).
