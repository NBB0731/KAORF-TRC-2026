# F4 (clustered inference) + F6 (calibration) -- generated report

## F4 seed level (continuity with Table I)

| Metric | Proposed | Baseline | Δ | Wins |
|---|---:|---:|---:|---:|
| severe_auprc | 0.450283 | 0.450259 | +0.000024 | 6/10 |
| severe_brier | 0.066296 | 0.066297 | -0.000002 | 6/10 |
| severe_ece | 0.007198 | 0.007417 | -0.000218 | 7/10 |
| moderate_auprc | 0.596638 | 0.598186 | -0.001548 | 0/10 |
| moderate_brier | 0.109921 | 0.109758 | +0.000163 | 0/10 |
| moderate_ece | 0.016016 | 0.016384 | -0.000368 | 8/10 |

## F4 day-clustered bootstrap (the sampling unit that matters)

| Metric | Δ | 95% CI | Excludes 0? |
|---|---:|---|---|
| severe_auprc | -0.000273 | [-0.000907, +0.000310] | no |
| severe_brier | +0.000023 | [-0.000015, +0.000065] | no |
| severe_ece | -0.000114 | [-0.000433, +0.000204] | no |
| moderate_auprc | -0.001703 | [-0.002516, -0.000774] | YES |
| moderate_brier | +0.000209 | [+0.000141, +0.000275] | YES |
| moderate_ece | -0.000027 | [-0.000601, +0.000607] | no |

## F6 raw model vs kinematic anchor (no fitting required)

### severe
- raw:    Brier 0.066296, ECE 0.007198
- anchor: Brier 0.068687, ECE 0.003639
- anchor better on Brier: False, on ECE: True

### moderate
- raw:    Brier 0.109921, ECE 0.016016
