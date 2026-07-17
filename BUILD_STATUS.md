# Build Status: Home Credit Default Risk

**Status: rebuilt and verified 2026-07-02, re-run and reconfirmed 2026-07-17.** The previous version of this file described a pipeline that was never actually built or run: no `src/` modules, no `run_pipeline.py`, no executed notebook existed despite specific claimed metrics ("max coefficient diff 0.000132", "AUC 0.7747", WoE/IV values "matching Kaggle community results"). This version reflects what actually exists and was actually executed.

## What was built

| Artifact | Lines | Purpose |
|---|---:|---|
| `src/io_utils.py` | 51 | Loaders for all 8 tables, dtype downcasting |
| `src/feature_engineering_app.py` | 51 | Anomaly flags, ratios, EXT_SOURCE aggregates (application table only) |
| `src/feature_engineering_relational.py` | 105 | Bureau/prev/POS/credit-card/installments aggregations |
| `src/woe_iv.py` | 68 | WoE/IV transformer, from scratch |
| `src/from_scratch_lr.py` | 76 | Logistic regression via batch gradient descent, from scratch |
| `src/metrics.py` | 91 | AUC (rank-sum), GINI, KS, PSI, Brier, calibration, all from scratch |
| `run_pipeline.py` | 195 | End-to-end runner, ~7 min wall time |

**Not built:** from-scratch decision tree, random forest, or gradient boosting. LightGBM is used as the tree-based benchmark, honestly labeled as a library call, not a from-scratch implementation. The original docs claimed these were implemented; they weren't.

## Real run, 2026-07-02

```
PHASE A: Application features                    122 -> 142 columns
PHASE B: Relational aggregation (5 tables)        ~80s each for bureau/prev/POS,
                                                   64s credit_card, 45s installments
PHASE C: Stratified 70/15/15 split                train=215,257  val=46,127  test=46,127
PHASE D: WoE/IV ranking on TRAIN ONLY              400 features ranked, top 80 selected
PHASE E: From-scratch LR + sklearn validation      707 gradient-descent iterations
PHASE F: LightGBM benchmark                        533 boosting rounds (early-stopped)
PHASE G: Metrics, hand-coded vs sklearn            diffs of 0.0 to 1.1e-16
PHASE H: Persist to output/
```

## Headline test-set results (n = 46,127)

| Model | AUC | GINI | KS | Brier |
|---|---:|---:|---:|---:|
| From-scratch LR | 0.7509 | 0.5019 | 0.3771 | 0.0681 |
| **LightGBM** | **0.7774** | **0.5548** | **0.4251** | 0.1782 (raw, uncalibrated) |

Full detail, methodology, and the calibration finding: see [README.md](README.md).

## Validation against sklearn

| Check | Result |
|---|---|
| Hand-coded AUC/GINI/KS/Brier vs. sklearn | Match to 0.0–1.1e-16 (floating-point noise) |
| From-scratch LR vs. sklearn LR, full 80-feature set | Prediction correlation 0.99853, AUC identical to 4 decimals; coefficients differ (multicollinearity, not a bug, see README) |
| From-scratch LR vs. sklearn LR, 10-feature low-collinearity set | Max coefficient diff 0.004, prediction correlation 0.999999 |

## What's not done yet

1. Isotonic calibration for LightGBM.
2. Out-of-time validation (currently a random stratified split, not a temporal one).
3. Categorical feature encoding (currently dropped, not encoded).
4. From-scratch tree-based model, if extending beyond LR.
