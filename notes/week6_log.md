# Week 6 log — blind build (application table only)

**Business question:** using only what's on the loan application itself —
before pulling bureau history, prior loans, or payment behavior — how well can
we separate applicants likely to default from those who aren't? This sets a
floor: whatever the relational data adds in Week 7+ has to beat this.

## What was built (blind — `Credit_Risk/src/` not opened)

1. `src/io_raw.py` — loads `application_train.csv` / `application_test.csv`
   as-is, no dtype tuning yet. Confirmed shapes (307,511 x 122 train,
   48,744 x 121 test) and target balance (91.9% / 8.1%) against what's publicly
   documented about this dataset — sanity check, not a peek at the oracle's code.
2. `src/baseline_features.py` — own feature set:
   - `AGE_YEARS`, `EMPLOYED_YEARS` from the negative `DAYS_*` columns.
   - `DAYS_EMPLOYED_ANOM` + NaN-out for the `365243` placeholder value
     (pensioners/unemployed applicants have no real employment history — left
     in, it would read as ~1000 years employed and wreck any ratio built on
     it). Caught this by looking at the raw `describe()` output, not by being
     told about it.
   - `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`,
     `CREDIT_GOODS_RATIO`, `INCOME_PER_FAM_MEMBER` — ratios instead of raw
     amounts, since `AMT_*` columns are heavily right-skewed and scale with
     income.
   - `EXT_SOURCE_MEAN/STD/COUNT` — treated missingness on the three external
     bureau-style scores as its own signal rather than just imputing it away.
3. `src/baseline_model.py` — impute (median/most-frequent) -> scale -> one-hot
   -> `LogisticRegression(class_weight="balanced")`, 80/20 stratified split.

## Result

**Validation AUC: 0.7487** (train AUC also 0.7487 — no overfit gap, which
makes sense for a linear model with this much regularization-via-scaling on
this much data).

## What this says to a credit committee

Application-only data gets you to a respectable but not final answer —
roughly in line with published Home Credit baselines that use only the
application table. `EXT_SOURCE_*` are almost certainly doing most of the
work here (they're the closest thing in this table to an existing credit
score) — Week 7's WoE/IV pass should confirm or challenge that with actual
Information Value numbers instead of a guess. The real lift is expected to
come from bureau/previous-application history in later weeks, which is
exactly what a committee would want confirmed before trusting a model that
ignores an applicant's borrowing history.

## Oracle check

Not yet — per the rules, the oracle (`Credit_Risk/src/`, `README.md`) stays
closed until Week 9's verification pass, or until genuinely stuck. Nothing
in Week 6 required opening it.
