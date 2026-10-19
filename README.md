# KAORF — reproducibility package

Code, locked checkpoints and result records for the manuscript

> **Joint vessel trajectory and near-encounter forecasting with kinematically
> anchored ordinal risk feedback**

submitted to *Transportation Research Part C: Emerging Technologies*.

KAORF (kinematically anchored ordinal risk feedback) predicts 30-minute vessel
trajectories together with future minimum-separation events at 0.25 and 0.5
nautical miles. A learned kinematic anchor and a bounded residual produce
ordinal event probabilities; those probabilities enter the trajectory decoder as
**detached** messages, so the motion loss cannot update the reported score. The
manuscript compares the proposed arm with an otherwise identical no-feedback
arm, with non-learned baselines, with three learned comparators (plain
Transformer C1, graph-convolution C3, LSTM C2), across a six-arm ablation, on a
locked temporal test and under zero-shot transfer to two further ports.

---

## 1. What is archived here

| Archived | Not archived |
|---|---|
| Model, training, evaluation and statistics code (`code/`) | Raw AIS source files (public, see `DATA.md`) |
| 32 locked checkpoints (`checkpoints/`) | Derived per-date scene archives (regenerable, `DATA.md`) |
| Per-seed result records and summaries (`results/`) | The pinned training container image |
| Per-event and per-node logs of the blind partition (`results/events_blind_20260924/`) | |
| The pre-registered date manifest (`preregistered/`) | |

Every number reported in the manuscript traces to a file under `results/`.
`MANIFEST.sha256` covers every file in this package:

```bash
sha256sum -c MANIFEST.sha256      # macOS / Linux
```

## 2. Layout

```
code/
  model/                 RFIGN (KAORF), plain_transformer (C1), plain_graph (C3), lstm_baseline (C2)
  training/              trainer, seed launcher, baseline trainers
  evaluation/            zero-shot external transfer, baseline evaluation, summarisers
  statistics/            event logs, calibration, per-day aggregation, figures
  analysis/              day-clustered bootstrap (NY and external), C1 external contrast,
                         pair-orientation sensitivity
  scene_construction/    NOAA AIS -> regional clips -> interaction scenes
  params/                parameter counting
  guards/                label-parity guard for the numpy/torch label paths
checkpoints/
  seed_<s>/<arm>/best.pt             KAORF arms, 10 seeds (feedback, no-feedback)
  seed_17_cpu_imported_20260923/     CPU-side re-import used for the local re-evaluation
  c1_external_sub12/seed_<s>/plain_transformer/best.pt
results/
  blind/                 per-seed blind-test metrics
  ablation_e/            six-arm ablation, per-seed, per-epoch records
  external_d/            external ports: full48 and sub12 (12 dates x 4 strata)
  events_blind_20260924/ per-event and per-node logs of the blind partition
  f4_f6/                 day-clustered aggregation and fitted calibration
  trc_20261018/          analyses added for the TRC revision (Section 4)
  params/, port_density/, figures_20260924/
preregistered/
  ais_2025_preregistered_manifest.csv    48 dates x 3 ports, partition truth
```

## 3. Locked configuration

| Item | Value |
|---|---|
| Architecture | `RFIGN` (`code/model/rfign_model.py`); hidden width 96, 4 heads, 2 temporal layers, FF width 384, dropout 0.1, norm-first |
| Inputs | 8 features, 20 history steps, at most 8 vessels per scene; 6 future offsets at 5-minute steps (30-minute horizon) |
| Proposed arm | `--detach-risk-feedback --constrained-risk-feedback --close-event-target --residual-bound 0.25 --selection-metric joint` |
| Paired comparator | the same flags with `--no-feedback` |
| Loss weights | severe BCE 0.2, anchor BCE 0.2, moderate BCE 0.1, calibration KL 0.1; masked Smooth-L1 trajectory loss |
| Model selection | joint validation loss, 12 epochs, batch 32 |
| Seeds (locked set) | 17, 29, 42, 71, 101, 131, 211, 307, 509, 887 |
| Thresholds | severe 0.25 nm, moderate 0.50 nm (nested) |
| Parameters | KAORF 353,295 = trunk 313,453 + risk-message 27,840 + anchor 10,657 + moderate increment 1,345; C1 313,453; C3 295,021; C2 354,445 |

`code/params/count_params.py` reproduces these counts exactly and needs no data.

## 4. Analyses added for this revision

`code/analysis/` holds the three analyses that the TRC revision added. Each one
reads only archived records, writes one JSON, and is deterministic.

| Script | Question | Output |
|---|---|---|
| `day_clustered_ade_fde.py` | Does the displacement gain survive when the **calendar date**, not the seed, is the sampling unit? | `results/trc_20261018/day_clustered_ade_fde_ny.json` |
| `day_clustered_external.py` | Same question at the two external ports (12 shared dates) | `results/trc_20261018/day_clustered_external.json` |
| `c1_vs_kaorf_external.py` | How do the two KAORF arms compare with the retrained plain Transformer at the external ports? | `results/trc_20261018/c1_external_stats.json` |
| `order_sensitivity.py` | How much does the pair probability depend on which vessel is written first? | `results/trc_20261018/order_sensitivity/order_sensitivity_summary.json` |

All four use the same inference convention: ten paired seeds, an equal-date
estimand, 200,000 date-clustered resamples, RNG seed 20260924 (20261018 for the
order test). The first three reproduce their archived JSON byte for byte:

```bash
python code/analysis/day_clustered_external.py
python code/analysis/c1_vs_kaorf_external.py
python code/analysis/day_clustered_ade_fde.py --scenes-dir <path-to-new_york-scenes>
```

`order_sensitivity.py` re-runs the checkpoints and needs both the scene archives
and the archived event logs. It checks labels and probabilities against
`results/events_blind_20260924/` before computing anything and stops on any
misalignment. A single-seed pass takes a few minutes on CPU:

```bash
python code/analysis/order_sensitivity.py --seeds 17 --out /tmp/order_check
```

Add `--save-arrays` to keep the per-seed pairwise probability arrays (about
6 MB each); they are omitted here because they are regenerable.

## 5. Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. parameter counts -- no data needed, exact
python code/params/count_params.py

# 2. archived statistics -- needs only files in this package
python code/analysis/day_clustered_external.py
python code/analysis/c1_vs_kaorf_external.py

# 3. re-evaluate a locked checkpoint on the New York blind dates
python code/evaluation/eval_external_zeroshot.py --help
```

Training and scene construction need the AIS data and the scene archives; see
`DATA.md` for the download and build path, and `code/scene_construction/` for
the scripts.

## 6. Environment

Training and the primary evaluation ran on one NVIDIA A10 (24 GB) instance with
torch 2.1–2.4 and CUDA 12.1. The local re-evaluation behind the archived
per-event logs ran on CPU: Python 3.13.12, torch 2.14.0+cpu, scikit-learn 1.9.1.
Checkpoints were written on Linux and store `pathlib.PosixPath` objects inside
their argument dictionaries; `code/analysis/order_sensitivity.py` remaps that
class before loading, which is why a Windows-side load works unchanged.

`requirements.txt` states the minimum versions that were tested. The training
container image itself was not archived.

## 7. Data

Public AIS data from NOAA/BOEM Marine Cadastre. The manifest lists the 145
(port, date, partition) rows; scene archives are rebuilt from the public source
rather than redistributed. Details and the exact build order are in `DATA.md`.

## 8. Licence and citation

Code released under the MIT licence (`LICENSE`). The AIS data remain subject to
the terms of the source. Citation metadata is in `CITATION.cff`; a DOI will be
minted on acceptance.
