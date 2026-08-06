# Week 8 log — full feature set, collinearity, PDO scorecard

**Business question:** now that bureau history, prior applications, and
payment behavior are in the model, does the extra signal actually change the
lending decision — and can the model still produce a scorecard a credit
committee can approve and a loan officer can hand-apply, not just a
black-box probability?

## Relational features (own design, from scratch)

Built aggregations from `bureau.csv` + `bureau_balance.csv`,
`previous_application.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`,
`installments_payments.csv` -- 28 new features rolled up to one row per
`SK_ID_CURR` (see `relational_features.py` for the full list and reasoning).
Missing-value rates confirm the joins are sane: bureau history missing for
13.6% of applicants (no bureau record), credit-card fields missing for ~71%
(most applicants never had a Home Credit card) -- these are legitimate
"no history" gaps, not a loading bug.

## Result: 0.7622 val AUC, KS 0.396 (up from Week 7's 0.7509 / 0.372)

Relational history is where the real lift was — bureau history and payment
behavior add information the application form alone doesn't have.
`BUREAU_DAYS_CREDIT_MEAN` (average age of bureau-reported credit lines)
lands at IV=0.12, "medium" strength, ahead of most application-only fields.

## Reproducing the coefficient divergence

Kept 56 of 65 candidate features (IV >= 0.01, deliberately permissive to
preserve correlated pairs instead of curating them away). Correlation matrix
on the WoE-encoded features turned up real, clean multicollinearity:

| pair | r |
|---|---|
| `BUREAU_OVERDUE_MAX` vs `BUREAU_OVERDUE_MEAN` | **+1.000** |
| `REGION_RATING_CLIENT_W_CITY` vs `REGION_RATING_CLIENT` | +0.945 |
| `INSTAL_SHORTFALL_MEAN` vs `INSTAL_SHORTFALL_SUM` | +0.927 |
| `PREV_AMT_APPLICATION_MEAN` vs `PREV_AMT_CREDIT_MEAN` | +0.914 |

`BUREAU_OVERDUE_MAX`/`MEAN` at r=+1.000 is the smoking gun: in WoE space
(after binning) they carry *identical* information for almost every
applicant, since most have either zero overdue days across all bureau
records (both stats = 0) or one dominant overdue event (max ≈ mean). A model
genuinely cannot tell them apart.

**Honest result, not forced to match the oracle's number:** max coefficient
diff vs sklearn is **0.0187** here (mean 0.0053), smaller than the 0.298 the
original oracle documented on its 80-feature set. That's a real finding, not
a failure to reproduce — it says the *magnitude* of this effect depends on
exactly which correlated features are in the set and how many, not just "80
features = big divergence." What *did* reproduce cleanly is the mechanism:
prediction correlation stayed high (0.905) and AUC moved by only 0.0002
between sklearn and the from-scratch fit, even as individual coefficients on
`INSTAL_SHORTFALL_MEAN`, `EXT_SOURCE_MEAN`, and `EXT_SOURCE_COUNT` shifted
by 0.01–0.02. That's the textbook multicollinearity signature: when two
inputs carry the same information, a model can trade weight between them
without changing what it predicts — sklearn's exact solver and gradient
descent's iterative walk land in two different, equally valid places on
that trade-off. `BUREAU_OVERDUE_MAX`/`MEAN` being r=1.000 makes this
mechanism something I can point to concretely, not just assert.

## The scorecard

PDO scorecard, base_score=600 at base_odds=20:1, PDO=40 (business choices,
documented in `scorecard.py` -- an institution would set these to match its
existing policy scale, not derive them statistically). Coefficients were
de-standardized back onto raw WoE values first (the model was fit on
standardized features for gradient-descent stability; the scorecard needs
the equivalent raw-WoE coefficients, algebra in `scorecard.destandardize_coefficients`).

- Score range on validation set: 391–798 (within the 300–850 policy range).
- Mean score, applicants who repaid: **595.9**. Mean score, applicants who
  defaulted: **539.4**. A 56-point gap in the expected direction.
- Point-biserial correlation(score, actual default) = **-0.264**, clearly
  negative as required.
- Reason codes work end-to-end: a sampled defaulted-loan applicant's biggest
  point losses were `DAYS_EMPLOYED_ANOM` (-16.7), `PREV_REFUSED_SHARE` (-9.9),
  `BUREAU_DAYS_CREDIT_MIN` (-9.5) -- a real adverse-action explanation, not
  a SHAP plot.

## Why LR + WoE over a black-box GBM, for a credit committee

- **Adverse action reasons.** Many jurisdictions require a lender to state
  *why* an application was declined. WoE + a linear model gives that for
  free (`reason_codes()` above); extracting an equivalent explanation from
  XGBoost/LightGBM means SHAP or similar, which is a second model most
  committees can't independently audit.
- **Monotonicity is checkable, not just observed.** Each WoE bin's sign is
  visible in a table before the model is even fit — a reviewer can confirm
  "more overdue days -> lower WoE -> fewer points" holds everywhere, not just
  hope a GBM learned it.
- **Stability under retraining.** A scorecard's point table changes gently
  when refit on new data (same bins, similar WoE); a tree ensemble's split
  structure can change more unpredictably, which matters when a regulator
  expects consistent treatment of similar applicants over time.

## PD -> IFRS9 ECL

This model outputs **PD** (probability of default) at the application. IFRS9
Expected Credit Loss is `ECL = PD x LGD x EAD` -- Loss Given Default (what
fraction of exposure is actually lost, after collateral/recovery) and
Exposure At Default (roughly the outstanding balance at the point of
default) are separate estimation problems, out of scope for this rebuild.
Worth being explicit about that scope boundary rather than implying this
scorecard alone produces a provisioning number.

## Oracle check

Still not opened. Nothing here required it — the "smaller divergence than
documented" finding above stands as reported, not adjusted to match.

## Correction (found during Week 9 verification, 2026-08-06)

Before diffing against the oracle, I audited my own comparison and found a
real bug: `sklearn`'s model was fit with `class_weight="balanced"`, but
`FromScratchLogisticRegression` had no equivalent -- it was minimizing plain
cross-entropy on an 8%/92% imbalanced target while sklearn was solving a
reweighted objective. Not the same problem, so "divergence" between them was
partly measuring that mismatch, not pure multicollinearity.

Added balanced-class sample weighting to the from-scratch fit (matching
sklearn's exact convention: `weight_c = n / (2 * count_c)`), reran Week 8:

| | before fix | after fix |
|---|---|---|
| prediction correlation vs sklearn | 0.905 | **0.999997** |
| max \|coef diff\| vs sklearn | 0.0187 | **0.0031** |
| val AUC (scratch) | 0.7624 | 0.7622 |
| val KS | 0.396 | 0.394 |

So the honest finding changes: once the two models are actually solving the
same objective, they agree almost perfectly on this 56-feature set, even
with `BUREAU_OVERDUE_MAX`/`MEAN` sitting at r=+1.000 in it. L2 regularization
(both sklearn's default and my `l2=1e-3`) is apparently enough to keep that
collinearity from destabilizing the fit at this scale. The oracle's larger
0.298 divergence on its 80-feature set may come from a different
regularization strength or a denser cluster of correlated features
compounding across more of them -- a real open question, not resolved here,
since the two feature sets aren't the same and comparing them isn't
apples-to-apples either.

**The actual lesson from this week isn't about credit risk at all: verify
your own comparison's premises before trusting what it says about the
domain.** A "the from-scratch model and sklearn disagree" finding was
mostly a bug in how the comparison was set up, not a discovery about
multicollinearity -- and it would have shipped in the write-up unnoticed
if Week 9's oracle check hadn't prompted a second look at the harness itself.
