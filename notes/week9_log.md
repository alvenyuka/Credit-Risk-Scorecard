# Week 9 log — verification against the oracle, final write-up

**Business question:** does this rebuild actually hold up against the real,
previously-shipped pipeline — and is there anything in the oracle's own
numbers that changes how the rebuild's findings should be read?

## First thing verification caught: a bug in my own comparison, not in the domain

Before opening the oracle, I re-checked my own Week 7/8 harness (see the
correction sections now in those logs) and found `FromScratchLogisticRegression`
had no equivalent of sklearn's `class_weight="balanced"`. Fixed it, reran
both weeks. That's arguably the single most useful thing this rebuild
produced: a reminder to audit your own comparison's premises before trusting
what it appears to say about the domain.

## Oracle diff

| Metric | Oracle (`output/results.json`, test, n=46,127) | This rebuild (val, n≈61,502) |
|---|---:|---:|
| LR val/test AUC | 0.7509 | 0.7622 |
| KS | 0.377–0.381 | 0.394 |
| Max \|coef diff\| vs sklearn, full feature set | 0.298 (80 features) | 0.0031 (56 features) |
| Max \|coef diff\| vs sklearn, small/clean feature set | 0.004 (10 features) | 0.0028 (25 features, Week 7) |
| Prediction correlation vs sklearn | 0.9985 | 0.999997 |
| Candidate features before selection | 400 (from 418 joined columns) | 65 |
| Categorical features used | 0 — dropped, flagged as future work | 10, WoE-encoded |
| LightGBM benchmark | AUC 0.7774, GINI 0.5548, KS 0.4251, badly calibrated | not built (out of scope here) |

Three real things this comparison surfaced, not guessed at:

1. **The oracle's own README explains its 0.298 divergence, and it lines up
   with what Week 8 diagnosed independently.** Its top features include
   literal interaction terms of the same three `EXT_SOURCE_*` columns —
   `EXT_SOURCE_MEAN`, `EXT_SOURCE_1_x_2`, `EXT_SOURCE_2_x_3`,
   `EXT_SOURCE_MIN/MAX` — a much more extreme, deliberate multicollinearity
   design than anything in this rebuild's 56 features (whose worst offender,
   `BUREAU_OVERDUE_MAX`/`MEAN` at r=+1.000, is one pair, not four engineered
   variants of the same three source columns). That difference in *how much*
   redundancy each feature set carries is the actual explanation for why the
   two "full-set" divergences (0.298 vs 0.0031) aren't comparable numbers —
   not a sign either rebuild is wrong.
2. **The oracle's own small-set check (0.004) matches this rebuild's
   corrected number (0.0028–0.0031) closely.** That's the real
   apples-to-apples comparison, and it holds up: both implementations agree
   that a properly regularized logistic regression on a low-collinearity
   feature set converges to sklearn's answer almost exactly, independent of
   which specific features are in it.
3. **This rebuild used categorical features (occupation type, income type,
   education, gender — 10 in total, WoE-encoded); the oracle's pipeline
   dropped all ~18 categorical columns** and lists that as unfinished future
   work in its own README. That's a genuine, if modest, place this rebuild
   goes further than the oracle, not just a smaller copy of it. `OCCUPATION_TYPE`
   (IV 0.083) and `NAME_INCOME_TYPE` (IV 0.056) carried real signal here.

**AUC**: this rebuild's 0.7622 vs the oracle's 0.7509 isn't a clean
apples-to-apples win — different split (val here vs the oracle's held-out
test set) and radically different feature-engineering depth (65 candidates
here vs 400 there, mostly because the oracle built far more relational
aggregates per table and several multiplicative interaction terms this
rebuild didn't attempt). The honest read: a much smaller, blind-built
feature set reached comparable-to-slightly-better linear-model discrimination,
which says more about diminishing returns past a certain point of
feature-engineering effort on this dataset than it says this rebuild is
"better."

## One refinement worth naming, not built

The oracle calibrates its scorecard's base odds to the actual population's
default rate (~11.4:1 at 8.07% observed default). This rebuild used an
assumed base_odds=20 at base_score=600 — a defensible business choice, but
an arbitrary one rather than a data-driven one. Calibrating to the real
population odds is the natural next refinement, named here rather than
retrofitted after the fact.

## Decision on publishing

Not made unilaterally here — see the conversation. This rebuild lives in its
own repo (`Portfolio/Credit_Risk_Rebuild/`, local git history, not pushed
anywhere) specifically so publishing is a separate, deliberate decision.
