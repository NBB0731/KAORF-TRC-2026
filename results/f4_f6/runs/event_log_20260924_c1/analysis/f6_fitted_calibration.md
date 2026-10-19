# F6 -- Post-hoc calibration fitted on validation, applied to blind

Fit: per (seed, variant, channel) on the validation event log;
evaluate frozen maps on the existing blind event log (no new blind pass).

## Mean blind ECE

| channel | regime | feedback | no_feedback |
|---|---|---|---|
| severe | raw | 0.0235 | n/a |
| severe | temperature | 0.0171 | n/a |
| severe | platt | 0.0060 | n/a |
| severe | isotonic | 0.0059 | n/a |
| moderate | raw | 0.0934 | n/a |
| moderate | temperature | 0.0398 | n/a |
| moderate | platt | 0.0070 | n/a |
| moderate | isotonic | 0.0068 | n/a |

## Mean blind Brier

| channel | regime | feedback | no_feedback |
|---|---|---|---|
| severe | raw | 0.0635 | n/a |
| severe | temperature | 0.0631 | n/a |
| severe | platt | 0.0616 | n/a |
| severe | isotonic | 0.0616 | n/a |
| moderate | raw | 0.1180 | n/a |
| moderate | temperature | 0.1058 | n/a |
| moderate | platt | 0.1018 | n/a |
| moderate | isotonic | 0.1018 | n/a |

## Paired comparisons (feedback - no_feedback)

