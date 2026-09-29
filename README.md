# Credit Risk Scorecard

> Which loan applicants are likely to default, and can a lender tell a declined applicant exactly why? A points-based scorecard on 307,511 real Home Credit applicants: AUC 0.762, KS 0.394 on 61,503 held-out applicants.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](#how-it-works)
[![AUC](https://img.shields.io/badge/AUC-0.762-success)](#what-i-found)
[![tests](https://github.com/alvenyuka/Credit-Risk-Scorecard/actions/workflows/ci.yml/badge.svg)](https://github.com/alvenyuka/Credit-Risk-Scorecard/actions/workflows/ci.yml)

## The problem

A lender that declines a loan usually has to give the applicant a specific reason. US fair-lending rules
require it, and the EU AI Act classes credit scoring as high risk. A black-box model can rank applicants
well and still be unable to say why one was turned down. For a credit committee, a model it cannot explain
is a model it cannot approve.

This project builds the kind of model a regulated lender can defend: a scorecard where every applicant
characteristic adds or removes a fixed number of points, so a decline comes with its reasons attached, for
example "lost 18 points on employment history".

## What I found

| Measure (61,503 held-out applicants) | Result |
|---|---:|
| AUC, how well the score ranks good and bad borrowers | **0.7622** |
| KS, the widest gap between repaid and defaulted score distributions | **0.3940** |
| GINI | 0.5244 |
| Average score, applicants who repaid vs defaulted (scale 300 to 656) | 455.6 vs 399.5 |

- **The score separates risk sharply.** Applicants in the lowest-scoring tenth default at 27.5%, those in the
  highest-scoring tenth at 1.3%, against an average of 8.1%.

  ![Default rate by score decile, falling from 27.5% in the lowest band to 1.3% in the highest](figures/default_rate_by_band.png)

- **Explainability costs little accuracy here.** An earlier tree-based benchmark (LightGBM) reached AUC
  0.7774. The scorecard gives up about 1.5 points of AUC and in return every decision can be explained.
- **Fields other lenders drop carry signal.** Occupation type and income type, discarded by an earlier
  version of this pipeline, turned out to be predictive once encoded properly.
- **Built by hand, checked against the standard libraries.** The statistics are written from scratch and
  then compared with scikit-learn and scipy: predictions agree at a 0.999997 correlation.

**What I would recommend to a credit committee:** use the scorecard, not the tree model, for approval
decisions. The reasons it produces can be given to applicants, each bin's direction can be reviewed before
the model is trained, and its point values move only slightly when refitted, which supports consistent
treatment of similar applicants over time. Fix the first limitation below before any real use.

## How it works

1. **Data.** The loan application plus the applicant's credit-bureau and previous-application history,
   combined into 65 candidate features.
2. **Weight of Evidence.** Each feature is grouped into bins and each bin is scored by how strongly it
   separates repaid from defaulted loans; weak features are dropped, leaving 56.
3. **Logistic regression**, written from scratch and fitted on those bins.
4. **Points table.** Model coefficients become points, so each applicant's score is a sum of readable parts
   and the largest point losses become the decline reasons.
5. **Validation.** AUC, KS, GINI and PSI, each hand-coded and tested against library equivalents. Every
   headline number is written by the code to [`outputs/results.json`](outputs/results.json).

## Run it

```bash
pip install -r requirements.txt
# put the 8 Home Credit Default Risk CSVs from Kaggle in data/ (not shipped)
python src/week8_full.py      # full pipeline, about 7 minutes, writes outputs/results.json
python -m pytest              # tests on synthetic data, about 8 seconds, no dataset needed
```

The walkthrough notebook is [`Credit_Risk_Scorecard.ipynb`](Credit_Risk_Scorecard.ipynb). `python src/make_figures.py` redraws the charts in `figures/` from the same model.

## Limitations

- **Gender and marital status are in the candidate features**, and gender appears in the reason codes.
  These are prohibited bases for lending decisions, so they must be removed and the model refitted before
  any real use. This is the next fix.
- **Probability of default only**, not calibrated to this population's roughly 8% default rate, and not a
  full IFRS 9 loss estimate (loss given default and exposure at default are not built).
- **Random, not time-based, validation.** The dataset has no application dates, so an out-of-time test is
  not possible on this data.

## More detail

The full write-up, including every result, the bugs found along the way, the tests and all limitations, is
in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). The live page at
[credit-risk-alven.vercel.app](https://credit-risk-alven.vercel.app) shows applicants scored by an earlier
version of this pipeline.

## License

MIT. See [`LICENSE`](LICENSE). Data: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (Kaggle).

## Connect

Built by Alven Yuka, CPA Finalist and Accounting Specialist at GIZ, Nairobi.

📫 [alvenyuka2@gmail.com](mailto:alvenyuka2@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/alven-yuka-610b78174/) · 🐙 [GitHub](https://github.com/alvenyuka)
