# Credit Risk Scorecard — a blind, from-scratch rebuild

**This content replaced an earlier, more complete version of this repo**
(400 candidate features vs. this rebuild's 65, a LightGBM benchmark,
calibration figures, and a live demo site) **on purpose.** That earlier
pipeline was real and independently re-verified — it isn't gone, it's in
this repo's git history, and its full numbers are quoted throughout this
README for comparison. What's here now is a deliberate blind rebuild of it:
same dataset, own feature choices, own bugs found and fixed in the open,
built without reading the original code until the verification step. The
earlier version had polish this one doesn't (LightGBM, a live demo); this
one shows something the earlier version didn't: the actual process, mistakes
included — built as a mastery exercise rather than a copy, with a plainer
goal than a bigger AUC number: can this get scored, and can it explain
*why*, in language a credit committee and a hiring manager would both
accept?

`figs/` and `output/demo_applicants.json` are kept from the earlier version
on purpose too — the live demo at
[credit-risk-alven.vercel.app](https://credit-risk-alven.vercel.app) fetches
both directly from this repo's `master` branch at runtime, so removing them
would have quietly broken a working, deployed page. Everything else in this
README describes the new rebuild; those two paths are the one deliberate
exception.

## The business question

Home Credit's application data (307,511 applicants, 8 relational tables)
plus a decision to make: should this applicant get the loan, and if not,
what's the specific, defensible reason? A model that only outputs a
probability doesn't answer the second half.

## What was built, and how (four weeks, compressed)

1. **Blind baseline** ([`week6_log.md`](notes/week6_log.md)) — raw
   application data only, own feature choices (caught the `DAYS_EMPLOYED`
   365243-placeholder anomaly from the raw data, not from being told about
   it), a naive logistic regression. **0.7487 AUC**, a floor to beat.
2. **Primitives from scratch** ([`week7_log.md`](notes/week7_log.md)) — WoE/IV,
   logistic regression via batch gradient descent, and AUC/GINI/KS/PSI, all
   hand-coded and validated against sklearn/scipy to floating-point
   precision. Found and fixed a real KS-statistic bug along the way (ties
   evaluated mid-tie-block instead of at distinct score values).
3. **Relational features + the multicollinearity chase**
   ([`week8_log.md`](notes/week8_log.md)) — own aggregations from bureau,
   previous-application, POS/cash, credit-card, and installment history
   lifted AUC to **0.7622**, KS to **0.394**. Deliberately went looking for
   the sklearn-vs-scratch coefficient divergence the oracle documents, found
   a real one, and then found a bug in *how it was being measured*
   (`class_weight="balanced"` on one side, not the other) — fixing that
   collapsed the divergence from 0.019 to 0.0031. Built a PDO scorecard
   (base 600, base odds 20:1, PDO 40) with working reason codes: a sampled
   declined applicant's biggest point losses were `DAYS_EMPLOYED_ANOM`
   (-17.9), `PREV_REFUSED_SHARE` (-10.2), `BUREAU_DAYS_CREDIT_MIN` (-9.2).
4. **Verification** ([`week9_log.md`](notes/week9_log.md)) — diffed every
   number against the oracle's `output/results.json` and its README. The
   oracle's own small-feature-set check (0.004 coefficient diff) matches
   this rebuild's corrected number (0.0031) closely; the oracle's larger
   full-set divergence (0.298) comes from a much more extreme
   multicollinearity design (literal `EXT_SOURCE_1_x_2`-style interaction
   terms) than this rebuild's feature set has.

## Results

| | This rebuild | Oracle |
|---|---:|---:|
| AUC | 0.7622 (val) | 0.7509 (test) |
| KS | 0.394 | 0.377–0.381 |
| Candidate features | 65 | 400 |
| Categorical features used | 10 (WoE-encoded) | 0 (dropped, flagged as future work) |
| Coefficient diff vs sklearn, clean feature set | 0.0028 | 0.004 |

Not a controlled A/B — different splits, and the oracle went much deeper on
relational feature engineering. The honest takeaway: a much smaller,
independently-derived feature set reaches comparable discrimination, and
using the categorical columns the oracle dropped picked up real signal
(`OCCUPATION_TYPE` IV 0.083, `NAME_INCOME_TYPE` IV 0.056) for free.

## The actual recommendation

For a production credit decision, use logistic regression on WoE-encoded
features over a black-box GBM (LightGBM in the oracle scores 0.7774 AUC but
is badly miscalibrated — see the oracle's calibration-curve finding) —
**not** because it's more accurate, it isn't, but because:

- every declined applicant gets a specific, auditable reason (`reason_codes()`
  in `src/scorecard.py`), which many jurisdictions require and which a GBM
  makes materially harder to produce;
- each WoE bin's monotonicity is checkable in a table before the model is
  even fit, not just hoped for;
- this scorecard outputs PD only — a full IFRS9 `ECL = PD x LGD x EAD`
  provisioning number needs LGD and EAD models this rebuild doesn't attempt.

## What's next, if extended

- Calibrate the scorecard's base odds to this population's actual default
  rate (~8%) instead of an assumed 20:1, the way the oracle does.
- Out-of-time validation — everything here is a random split; Home Credit's
  `DAYS_DECISION` field would support a real temporal holdout.
- The Fraud Detection rebuild, same from-scratch method, is the natural next
  cycle (see the 12-week plan).

## Repository layout

```
├── src/
│   ├── io_raw.py                # application table loader
│   ├── baseline_features.py     # Week 6 application-level features
│   ├── baseline_model.py        # Week 6 naive LR baseline
│   ├── woe_iv.py                # Week 7 — WoE/IV from scratch
│   ├── metrics_scratch.py       # Week 7 — AUC/GINI/KS/PSI from scratch
│   ├── from_scratch_lr.py       # Week 7 — logistic regression via gradient descent
│   ├── week7_run.py             # Week 7 — WoE + from-scratch LR on real data
│   ├── relational_features.py   # Week 8 — bureau/prev/POS/CC/installments aggregation
│   ├── week8_run.py             # Week 8 — full feature set + collinearity diagnosis
│   ├── scorecard.py             # Week 8 — PDO scorecard + reason codes
│   └── week8_full.py            # Week 8 — scorecard build + example applicants
├── notes/                       # week6-9 logs — the actual decision record
└── CLAUDE.md                    # the rules this rebuild followed
```

## Getting the data

Place the 8 Home Credit CSVs in `data/` (gitignored, not shipped in this repo):
[Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk).
`data/` needs: `application_train.csv`, `application_test.csv`, `bureau.csv`,
`bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`,
`credit_card_balance.csv`, `installments_payments.csv`.

## Running it

```bash
cd src
python baseline_model.py    # Week 6
python week7_run.py         # Week 7
python relational_features.py  # Week 8, first run only (caches to a local parquet)
python week8_full.py        # Week 8
```
