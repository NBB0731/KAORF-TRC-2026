# F6 -- Post-hoc calibration fitted on validation, applied to blind

Fit: per (seed, variant, channel) on the validation event log;
evaluate frozen maps on the existing blind event log (no new blind pass).

## Mean blind ECE

| channel | regime | feedback | no_feedback |
|---|---|---|---|
| severe | raw | 0.0072 | 0.0074 |
| severe | temperature | 0.0083 | 0.0086 |
| severe | platt | 0.0065 | 0.0064 |
| severe | isotonic | 0.0064 | 0.0063 |
| moderate | raw | 0.0160 | 0.0164 |
| moderate | temperature | 0.0112 | 0.0121 |
| moderate | platt | 0.0106 | 0.0109 |
| moderate | isotonic | 0.0070 | 0.0070 |

## Mean blind Brier

| channel | regime | feedback | no_feedback |
|---|---|---|---|
| severe | raw | 0.0663 | 0.0663 |
| severe | temperature | 0.0663 | 0.0663 |
| severe | platt | 0.0662 | 0.0662 |
| severe | isotonic | 0.0662 | 0.0662 |
| moderate | raw | 0.1099 | 0.1098 |
| moderate | temperature | 0.1098 | 0.1096 |
| moderate | platt | 0.1096 | 0.1093 |
| moderate | isotonic | 0.1095 | 0.1092 |

## Paired comparisons (feedback - no_feedback)

- feedback_vs_no_feedback/ece/severe/raw: mean fb 0.0072 vs nf 0.0074, delta -0.0002, dz -0.43, p=0.193, wins 7/10
- feedback_vs_no_feedback/ece/severe/temperature: mean fb 0.0083 vs nf 0.0086, delta -0.0003, dz -0.45, p=0.193, wins 7/10
- feedback_vs_no_feedback/ece/severe/platt: mean fb 0.0065 vs nf 0.0064, delta +0.0000, dz 0.03, p=0.625, wins 4/10
- feedback_vs_no_feedback/ece/severe/isotonic: mean fb 0.0064 vs nf 0.0063, delta +0.0001, dz 0.19, p=0.492, wins 3/10
- feedback_vs_no_feedback/ece/moderate/raw: mean fb 0.0160 vs nf 0.0164, delta -0.0004, dz -0.65, p=0.0645, wins 8/10
- feedback_vs_no_feedback/ece/moderate/temperature: mean fb 0.0112 vs nf 0.0121, delta -0.0009, dz -1.20, p=0.00586, wins 9/10
- feedback_vs_no_feedback/ece/moderate/platt: mean fb 0.0106 vs nf 0.0109, delta -0.0003, dz -0.60, p=0.0645, wins 8/10
- feedback_vs_no_feedback/ece/moderate/isotonic: mean fb 0.0070 vs nf 0.0070, delta -0.0000, dz -0.01, p=0.77, wins 5/10

## KAORF vs deterministic baselines (blind ECE, baseline calibrated/evaluated on its own fitted map)

| baseline | channel | regime | KAORF fb mean | KAORF nf mean | baseline | fb below? (of 10) | p (one-sample) |
|---|---|---|---|---|---|---|---|
| cv | moderate | isotonic | 0.0070 | 0.0070 | 0.0098 | 10 | 0.00195 |
| cv | moderate | platt | 0.0106 | 0.0109 | 0.0168 | 10 | 0.00195 |
| cv | moderate | raw | 0.0160 | 0.0164 | 0.1314 | 10 | 0.00195 |
| cv | moderate | temperature | 0.0112 | 0.0121 | 0.0183 | 10 | 0.00195 |
| cv | severe | isotonic | 0.0064 | 0.0063 | 0.0094 | 10 | 0.00195 |
| cv | severe | platt | 0.0065 | 0.0064 | 0.0122 | 10 | 0.00195 |
| cv | severe | raw | 0.0072 | 0.0074 | 0.0543 | 10 | 0.00195 |
| cv | severe | temperature | 0.0083 | 0.0086 | 0.0459 | 10 | 0.00195 |
| imm | moderate | isotonic | 0.0070 | 0.0070 | 0.0094 | 10 | 0.00195 |
| imm | moderate | platt | 0.0106 | 0.0109 | 0.0166 | 10 | 0.00195 |
| imm | moderate | raw | 0.0160 | 0.0164 | 0.1320 | 10 | 0.00195 |
| imm | moderate | temperature | 0.0112 | 0.0121 | 0.0182 | 10 | 0.00195 |
| imm | severe | isotonic | 0.0064 | 0.0063 | 0.0091 | 10 | 0.00195 |
| imm | severe | platt | 0.0065 | 0.0064 | 0.0129 | 10 | 0.00195 |
| imm | severe | raw | 0.0072 | 0.0074 | 0.0547 | 10 | 0.00195 |
| imm | severe | temperature | 0.0083 | 0.0086 | 0.0453 | 10 | 0.00195 |
