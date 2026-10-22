# Model validation report: Credit Risk Scorecard

A validation of the scorecard in this repository, written the way a bank's model-validation team would review a
new application scorecard before it is used: is it built on sound principles, does it rank and price risk
correctly, is it implemented as specified, and what must be fixed before it could be relied on.

Every figure below is read from a file the pipeline writes: `outputs/results.json` (generated 2026-10-06 at
commit `75ef602`), `outputs/challengers.json` and `outputs/business_impact.json`. The report was prepared by the
model's author; an independent validator should repeat the tests in section 9 before relying on it.

## 1. Summary

| Area | Result | Rating |
|---|---|---|
| Conceptual soundness | 32 characteristics, each with monotonic Weight of Evidence and a coefficient of the expected sign; sex and marital status excluded | Satisfactory |
| Discrimination | AUC 0.754 (95% bootstrap interval 0.747 to 0.761), KS 0.381, Gini 0.508 on 61,503 held-out applicants | Satisfactory |
| Calibration | Predicted and observed default rates within 0.40 percentage points in every decile; 20.2:1 observed odds where 20:1 is designed | Satisfactory |
| Implementation | From-scratch logistic regression matches scikit-learn: prediction correlation 0.999999, largest coefficient gap 0.0023 | Satisfactory |
| Stability over time | Not testable: the data has no application dates | **Open finding** |
| Benchmarking | LightGBM reaches AUC 0.781 on the same holdout, 0.027 above the scorecard | Accepted trade-off |

**Conclusion:** fit for its stated purpose, ranking applicants and explaining declines on this dataset, with one
open finding. Before use on a live portfolio it needs an out-of-time test on dated applications from that
portfolio.

## 2. Scope and intended use

- **Use:** approve or decline consumer-credit applications at a chosen approval rate, with the three
  characteristics that cost the applicant most points given as reasons for each decline.
- **Not covered:** loss given default and exposure at default (the model estimates probability of default only),
  pricing, limit setting, and behavioural scoring of existing borrowers.
- **Data:** Home Credit Default Risk, 307,511 applications across 8 linked tables. Training and validation are a
  random stratified 80/20 split; the bad rate is 8.07% in both.

## 3. Conceptual soundness

| Check | Evidence | Result |
|---|---|---|
| Characteristic screening | 62 candidates; Information Value of at least 0.01 keeps 53; a correlation gate (r above 0.9) keeps 47; a coefficient-sign gate keeps 32 | Documented, rule-based |
| Monotonic risk ranking | 27 of 27 numeric characteristics have monotonic Weight of Evidence after coarse classing | Pass |
| Expected direction | Every coefficient is negative on Weight of Evidence (`all_coefficients_negative: true`): more risk evidence always means fewer points | Pass |
| Coarse classing decision | Adopted because it cost 0.0029 AUC on the validation slice, inside the 0.005 tolerance set before testing; it raises monotonic characteristics from 14 of 31 to 27 of 27 | Pass |
| Prohibited bases | Sex (IV 0.038) and marital status (IV 0.023) never enter the scorecard. Adding them back would raise AUC from 0.7540 to 0.7553, a gain too small to justify a fair-lending risk | Pass |
| Age | Age was removed by the sign gate; no scorecard bin treats applicants aged 62 or over differently (the US Regulation B test, used here as a documentation check) | Pass |

The sign gate removed 15 characteristics whose coefficients turned positive in the presence of others, including
the credit amount, the annuity and the applicant's age. Removing them costs some predictive power (section 6) but
guarantees that no single reason code contradicts common sense.

## 4. Discrimination

| Measure | Value |
|---|---|
| AUC | 0.754 (95% bootstrap interval 0.747 to 0.761) |
| KS | 0.381 |
| Gini | 0.508 |
| Mean score, repaid / defaulted | 594 / 542 |

Default rate by score decile on the holdout, lowest scores first:

| Decile | Score range | Default rate |
|---|---|---|
| 1 | 389 to 514 | 26.4% |
| 2 | 514 to 541 | 15.0% |
| 3 | 541 to 560 | 10.5% |
| 4 | 560 to 577 | 8.3% |
| 5 | 577 to 592 | 5.9% |
| 6 | 592 to 607 | 4.9% |
| 7 | 607 to 622 | 3.4% |
| 8 | 622 to 639 | 3.0% |
| 9 | 639 to 662 | 2.0% |
| 10 | 662 to 787 | 1.3% |

The default rate falls in every step from the riskiest decile to the safest, a 20-fold spread.

## 5. Calibration

| Decile of predicted PD | Applicants | Mean predicted PD | Observed default rate |
|---|---|---|---|
| 1 | 6,151 | 1.20% | 1.30% |
| 2 | 6,150 | 2.08% | 2.02% |
| 3 | 6,150 | 2.87% | 3.01% |
| 4 | 6,150 | 3.75% | 3.35% |
| 5 | 6,151 | 4.80% | 4.94% |
| 6 | 6,150 | 6.12% | 5.93% |
| 7 | 6,150 | 7.93% | 8.26% |
| 8 | 6,150 | 10.52% | 10.47% |
| 9 | 6,150 | 14.89% | 15.04% |
| 10 | 6,151 | 26.05% | 26.40% |

- The largest gap is 0.40 percentage points (decile 4). The mean predicted PD is 8.02% against an observed 8.07%.
- The points scale is set at 600 points for 20:1 good-to-bad odds, 40 points to double the odds. Applicants
  scoring 580 to 620 (16,383 of them) show 20.2:1, so the scale means what it says.
- The model was fitted on balanced classes and its intercept corrected back to the 8.07% prior. Without that
  correction the odds of default would be overstated about 11 times, so it is the step to re-check whenever the
  model is refitted.

## 6. Benchmarking

| Model | Holdout AUC | Characteristics | Explainable line by line |
|---|---|---|---|
| Scorecard (this model) | 0.754 | 32 | Yes |
| Scorecard without coarse classing | 0.759 | 37 (11 of 32 numeric monotonic) | Partly |
| LightGBM challenger | 0.781 | all | No |

The gradient-boosted challenger ranks better by 0.027 AUC. The scorecard is preferred because every decline must
carry a reason a customer and a regulator can read, and because its points can be audited by hand. A lender could
use the challenger as a second-line monitor: applicants it ranks very differently from the scorecard are the ones
to review.

## 7. Business use at the chosen cut-off

At an 80% approval rate (cut-off 541 points), on the holdout and with a 45% loss given default:

| Measure | Value |
|---|---|
| Credit loss avoided against approving everyone | 46.8% |
| Default rate among approved applicants | 4.9% (8.1% if everyone is approved) |
| Good borrowers declined | 9,752, or 17.2% of applicants who repaid |

The third line is the cost of the first: of the 12,301 applicants declined, 2,549 would have defaulted and 9,752
would have repaid, about 3.8 good borrowers turned away for every defaulter. The policy table in `outputs/business_impact.json` gives the same trade-off at 90%
and 70% approval.

## 8. Findings

| # | Finding | Severity | Action |
|---|---|---|---|
| F1 | No out-of-time validation. The data has no application dates, so stability on a later population is untested. The train-to-holdout PSI of 0.0002 only confirms that the random split is random. | High | Re-validate on dated applications from the target portfolio, with quarterly PSI on scores and on each characteristic. |
| F2 | One loss-given-default figure (45%, the Basel foundation-IRB value for senior unsecured exposures) for every loan. | Medium | Replace with a portfolio LGD estimate before quoting loss figures in money. |
| F3 | The sign gate removed 15 characteristics, including credit amount and annuity, which carry real information. | Low | Accepted for explainability; monitor through the LightGBM challenger (section 6). |
| F4 | The currency of the amounts is not stated in the source data. | Low | Quote losses as percentages, as the README does. |

## 9. Tests that reproduce this report

`python -m pytest` runs 87 tests in CI, with no data needed. The ones that carry the claims above:

| Claim | Test file |
|---|---|
| From-scratch logistic regression equals scikit-learn | `tests/test_from_scratch_lr.py` |
| AUC, KS and Gini computed by hand equal library values | `tests/test_metrics_scratch.py` |
| Weight of Evidence, Information Value and monotonic binning | `tests/test_woe_iv.py`, `tests/test_feature_screening.py` |
| Points scale, reason codes and score clipping | `tests/test_scorecard.py` |
| Prohibited bases and the age check | `tests/test_fair_lending.py` |
| Loss avoided and good borrowers declined | `tests/test_business_impact.py` |

Re-running `python src/run_pipeline.py` and `python src/challengers.py` regenerates `outputs/`; every table in this report can be rebuilt from those
files.
