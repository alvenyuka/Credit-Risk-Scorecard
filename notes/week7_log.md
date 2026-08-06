# Week 7 log — WoE/IV, from-scratch LR, metrics (all validated)

**Business question:** which specific, nameable pieces of information on a
loan application actually drive risk, and can a credit decision be explained
in terms a loan officer or committee can read directly — not just "the model
said no"?

## Primitives built and validated

- `woe_iv.py` — WoE/IV from scratch. Sanity-checked on synthetic data: a
  near-perfect separator scores IV=15.9 ("suspiciously strong — check for
  leakage"), pure noise scores IV=0.001 ("useless").
- `metrics_scratch.py` — AUC (rank-sum), GINI, KS, PSI. AUC matches
  `sklearn.metrics.roc_auc_score` to 1e-16. **Found a real bug**: the first
  KS implementation was off by ~1.6e-3 under tied scores, because a row-by-row
  cumulative sum evaluates the good/bad CDF gap *mid-tie-block* instead of at
  distinct score values — not a valid place to check a CDF. Fixed by
  aggregating to one row per distinct score before cumulative-summing; now
  matches `scipy.stats.ks_2samp` to 0.00e+00.
- `from_scratch_lr.py` — logistic regression via batch gradient descent.
  On a clean synthetic set: AUC diff 2e-6 vs sklearn, max coef diff 0.001,
  prediction correlation 1.000000.

## Applied to the real application table

IV ranking (top of list) is the actual business finding here:

| Feature | IV | Read |
|---|---|---|
| EXT_SOURCE_MEAN | 0.609 | suspiciously strong |
| EXT_SOURCE_3 | 0.330 | strong |
| EXT_SOURCE_2 | 0.303 | strong |
| EXT_SOURCE_1 | 0.151 | medium |
| EMPLOYED_YEARS | 0.112 | medium |
| CREDIT_TERM, AMT_GOODS_PRICE, AGE_YEARS, OCCUPATION_TYPE, ... | 0.04–0.09 | weak |

**What this says to a committee:** the three external bureau-style scores
(`EXT_SOURCE_1/2/3`) — not anything on the application form itself — carry
almost all the signal. `EXT_SOURCE_MEAN`'s "suspiciously strong" flag isn't
a leak, it's arithmetic: it's the mean of the other three, so of course it's
at least as separating as any one of them. Keeping all four in the same
model is mild redundancy, not new information — worth revisiting in Week 8.

25 of 37 candidate features cleared the IV >= 0.02 bar and were kept for
the model.

## Result: 0.7509 val AUC (vs 0.7487 in Week 6)

Barely moved. **That's a real finding, not a failure of the WoE approach** —
worth saying plainly rather than dressing it up: for a linear model that was
already properly scaled, curating to WoE-encoded features didn't buy
meaningfully more discrimination power on this table. What WoE *does* buy,
that raw+one-hot doesn't, is interpretability a committee can act on directly
(each bin's WoE is a log-odds contribution, readable without a model card)
and more graceful handling of missingness and outliers than median-imputing
raw values. That's the actual pitch for using it here, not a bigger AUC number.

**KS = 0.372** (37.2 on the usual 0–100 scale) — in the range retail credit
scoring practice generally treats as good separation, a concrete number to
quote in an interview.

**PSI (train -> val) = 0.0001** — confirms the metric works (train/val is a
random split of the same period, so near-zero drift is expected), but this
is *not* the check that matters for production stability. A real PSI check
needs an out-of-time sample, which isn't built yet.

## A preview of Week 8 (superseded — see correction below)

`max |coef diff| vs sklearn: 0.0233`, prediction correlation 0.9165 — both
noticeably looser than the clean-synthetic-data validation (0.001 / 1.000000).
Read at the time as an early multicollinearity signal. It wasn't, mostly —
see the Week 8 log's correction section: sklearn was fit with
`class_weight="balanced"` and the from-scratch model wasn't, so this was
largely comparing two different objectives, not two solvers of the same one.
After fixing `from_scratch_lr.py` to support balanced sample weighting and
rerunning: **max coef diff 0.0028, prediction correlation 0.999999** — the
two models agree almost exactly once they're actually solving the same
problem. Leaving the original numbers above rather than deleting them; the
mistake and the fix are both part of the record.

## Oracle check

Still not opened. Nothing here required it.
