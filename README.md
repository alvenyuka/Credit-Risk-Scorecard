# Credit Risk Scorecard

A credit scorecard that tells a lender which loan applicants are likely to default, and gives a specific reason
for every decline. Tested on 61,503 past applicants it had not seen, approving only the top 80% by score would
have **cut credit losses by 47%**. Built on 307,511 Home Credit applications with Weight of Evidence and a
logistic regression written from first principles (**AUC 0.761**).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](#how-to-run)
[![tests](https://github.com/alvenyuka/Credit-Risk-Scorecard/actions/workflows/ci.yml/badge.svg)](https://github.com/alvenyuka/Credit-Risk-Scorecard/actions/workflows/ci.yml)

![Default rate by score decile, falling from 27.0% in the lowest band to 1.2% in the highest](figures/default_rate_by_band.png)

## Contents

1. [Business problem](#business-problem)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Business impact](#business-impact)
6. [Key insights](#key-insights)
7. [Limitations](#limitations)
8. [Repository structure](#repository-structure)
9. [How to run](#how-to-run)

## Business problem

A consumer lender approving loans to applicants with thin or no credit history has to answer two questions
at once: who is likely to repay, and why was a declined applicant declined. Fair-lending rules require a
specific reason for every adverse decision, and model-risk teams expect to review the direction of each
variable before a model goes live.

A scorecard answers both. Each applicant characteristic adds or removes a fixed number of points, the total
sets the decision, and the largest point losses are the reasons given to the applicant. The questions for
this project are how well such a scorecard separates good and bad borrowers on real data, and what it is
worth to a lender as an approval policy.

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (Kaggle): 307,511
applications with a repaid/defaulted outcome, and seven linked tables of credit-bureau records, previous
applications, instalment payments and card balances.

| Property | Value |
|---|---:|
| Applicants with an outcome | 307,511 |
| Default rate | 8.1% |
| Linked tables | 8 |
| Holdout (random 20%, stratified) | 61,503 applicants |
| Candidate features | 63 |
| Features kept after screening | 54 |

Sex (`CODE_GENDER`) and marital status (`NAME_FAMILY_STATUS`) are prohibited bases under the US Equal
Credit Opportunity Act and Regulation B, so they are excluded from the candidate features, and a test fails
if either is reintroduced.

## Methodology

```mermaid
flowchart LR
    A[8 Home Credit tables] --> B[63 candidate features]
    B --> C[WoE binning, IV >= 0.01: 54 kept]
    C --> D[Logistic regression, from scratch]
    D --> E[Points table: 600 at 20:1 odds, 40 PDO]
    E --> F[Score, reason codes, approval cut-off]
```

1. **Feature engineering.** Application ratios (credit to income, annuity to income, credit term) and
   applicant-level aggregates of bureau history, previous applications, instalments and card balances.
   `DAYS_EMPLOYED` carries a placeholder of 365,243 days for pensioners and the unemployed; it becomes an
   explicit anomaly flag instead of a thousand years of employment.
2. **Weight of Evidence and Information Value.** Numeric features are binned in deciles and categoricals by
   level, with smoothing for sparse bins. Features with an information value below 0.01 are dropped.
3. **Model.** A class-balanced logistic regression fitted by gradient descent on standardised WoE values.
   WoE, IV, the solver and the AUC, KS, GINI and PSI metrics are implemented in `src/` and tested against
   scikit-learn and scipy.
4. **Scaling.** Coefficients become points with 600 at 20:1 good-to-bad odds and 40 points to double the
   odds. The bins where an applicant lost the most points become the reason codes.
5. **Business evaluation.** On the holdout, approve only the highest-scoring share of applicants and measure
   the default rate, the credit loss and the good borrowers turned away (`src/business_impact.py`).

Every metric is written by the pipeline to [`outputs/results.json`](outputs/results.json) and every money
figure to [`outputs/business_impact.json`](outputs/business_impact.json), with the commit and package
versions.

## Results

| Metric (61,503 held-out applicants) | Value |
|---|---:|
| AUC | **0.761** |
| KS | **0.393** |
| GINI | 0.522 |
| Default rate, lowest vs highest score decile | 27.0% vs 1.2% |
| Mean score, repaid vs defaulted (scale 300 to 648) | 455 vs 400 |
| Prediction correlation with scikit-learn's fit | 0.999997 |

![Score distributions for repaid and defaulted applicants, means 455.3 and 399.8](figures/score_distribution.png)

A declined applicant receives reasons in the terms a credit officer would use. For a defaulted applicant in
the holdout: *employment record flagged as anomalous, -18.4 points; high share of previous applications
refused, -10.3 points; low share of previous applications approved, -8.8 points.*

## Business impact

Approving only the highest-scoring applicants, on the same 61,503-applicant holdout. Credit loss is the credit
amount of each approved loan that defaulted, times a loss given default of 45% (the Basel foundation-IRB value
for senior unsecured lending). Home Credit does not state its currency, so amounts are in the dataset's units.

| Policy | Approved | Default rate | Credit loss | Loss avoided | Good borrowers declined |
|---|---:|---:|---:|---:|---:|
| Approve everyone | 61,503 | 8.1% | 1.25 bn | n/a | 0 |
| Approve top 90% (score ≥ 373) | 55,353 | 6.0% | 0.88 bn | 29% | 7.9% |
| Approve top 80% (score ≥ 400) | 49,202 | 4.8% | 0.66 bn | **47%** | 17.2% |
| Approve top 70% (score ≥ 420) | 43,052 | 4.0% | 0.49 bn | 61% | 26.9% |

![Credit loss on the holdout by approval policy, falling from 1.25 billion when approving everyone to 0.49 billion at a 70% approval rate](figures/lending_policy.png)

At a 90% approval rate the scorecard removes 29% of credit losses while turning away fewer than one good
borrower in twelve. Where to set the cut-off depends on the margin earned on a good loan against the loss on a
bad one, which this table lets a credit committee price directly.

## Key insights

- **External bureau scores dominate.** The three `EXT_SOURCE` scores and their average carry far more
  information than any field on the application form; the average (IV 0.61) was checked for leakage and kept,
  because all three scores exist when the application is made.
- **Behavioural history adds real signal.** Bureau credit age, the share of active bureau credit and the share
  of previous applications refused all rank among the top 15 features.
- **Removing prohibited bases costs almost nothing.** Excluding sex and marital status moved AUC from 0.7622
  to 0.7611.
- **A small, independent feature set is enough.** 54 features reach AUC 0.761, against 0.751 for an earlier
  400-candidate version of this pipeline, because near-duplicate features were not selected.

![Top 15 of 54 selected features by information value, led by EXT_SOURCE_MEAN at 0.61](figures/iv_top15.png)

## Limitations

- **Probabilities are not calibrated** to the 8% base rate, because the fit is class-balanced; scores rank
  risk well but are not probabilities of default.
- **Random, not out-of-time, validation.** The dataset carries no application dates, so a forward time split
  cannot be built.
- **Loss uses one LGD and ignores revenue.** The business-impact table applies a single 45% LGD and does not
  model the interest earned on good loans, so it shows the loss side of the cut-off decision only.
- **PD only.** A full IFRS 9 provision also needs loss-given-default and exposure-at-default models.

## Repository structure

```
src/                         features, WoE/IV, logistic regression, metrics, scorecard, figures,
                             business_impact.py (approval-policy analysis)
tests/                       34 tests: statistics vs scikit-learn and scipy, binning, fair-lending
                             exclusions, approval-policy arithmetic
outputs/results.json         every headline metric, written by the pipeline
outputs/business_impact.json approval-policy table with its assumptions
figures/                     charts drawn from the same model
Credit_Risk_Scorecard.ipynb  walkthrough notebook, executed end to end
docs/METHODOLOGY.md          full method, results and test descriptions
```

## How to run

```bash
pip install -r requirements.txt
python -m pytest                   # 34 tests, no data needed
# put the 8 Home Credit Default Risk CSVs from Kaggle in data/
python src/run_pipeline.py         # fit, score and write outputs/results.json (about 7 minutes)
python src/make_figures.py         # score distribution and default rate by band
python src/business_impact.py      # approval-policy table, lending-policy and IV charts
```

## Documentation

The walkthrough notebook is [`Credit_Risk_Scorecard.ipynb`](Credit_Risk_Scorecard.ipynb); the full method, every
result and the test suite are described in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## License

MIT. See [`LICENSE`](LICENSE). Data: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (Kaggle).

Alven Yuka · [LinkedIn](https://www.linkedin.com/in/alven-yuka-610b78174/) · [Email](mailto:alvenyuka2@gmail.com)
