# Build log

The scorecard in this repository began as a blind rebuild of an earlier pipeline: written without reading the
earlier code, and checked against it only once each step was finished. These four logs are the record of that
rebuild, kept as written at the time, including the mistakes and their corrections.

| Log | Step | What it records |
|---|---|---|
| [week6_log.md](week6_log.md) | Blind baseline | Application table only; the `DAYS_EMPLOYED` 365243 placeholder caught from the raw data; AUC 0.7487 |
| [week7_log.md](week7_log.md) | Statistics from scratch | WoE/IV, logistic regression, AUC, GINI, KS and PSI; a KS bug under tied scores, found and fixed |
| [week8_log.md](week8_log.md) | Full feature set and scorecard | Relational features, the collinearity investigation, the PDO scorecard; a class-weight mismatch in the comparison harness, found and fixed |
| [week9_log.md](week9_log.md) | Verification | The diff against the earlier pipeline's results |

The numbers in these logs describe the rebuild as it stood on 2026-08-06. The model has changed since (coefficient
signs enforced, coarse classing, a corrected prior), so the current figures are the ones in the main
[README](../README.md) and [`outputs/results.json`](../outputs/results.json). Paths such as `Credit_Risk/src/`
refer to the earlier pipeline, which is in this repository's history before commit `a8708a0`.

Edits since the logs were written: punctuation only, plus the last paragraph of week 9, which now states how the
publishing decision was resolved.
