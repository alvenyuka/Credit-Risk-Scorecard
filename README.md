# Credit Risk Scorecard

> From-scratch WoE/IV and logistic regression on Home Credit's 307,511 real loan applicants, checked against scikit-learn and scipy at every step, with a points-based scorecard that gives a declined applicant a specific reason.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](#tech-stack)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](#tech-stack)
[![AUC](https://img.shields.io/badge/AUC-0.762-success)](#results)

A credit scorecard for Home Credit's loan applicants, built from Weight of
Evidence, a from-scratch logistic regression, and a points-based scorecard
a loan officer could actually read.

Start with [`Credit_Risk_Scorecard.ipynb`](Credit_Risk_Scorecard.ipynb). It's
the real narrative, run end to end against the real 307,511-applicant
dataset, not a summary of one.

## Quick Start

1. Download the 8 CSVs from Home Credit Default Risk on Kaggle (see "Dataset" below) and put them in `data/`.
2. `pip install -r requirements.txt`
3. `python build_notebook.py`, then execute the generated notebook end to end (full commands under "Running it" below).
4. Read "What's in the notebook" below for the narrative, or jump to Results for the headline numbers.

## Features

- **Weight of Evidence / Information Value, logistic regression, and scoring metrics (AUC, GINI, KS, PSI) built from scratch**, not imported — each checked against scikit-learn and scipy before being trusted.
- **A points-based scorecard**: every WoE bin becomes a specific score contribution, so a declined applicant's biggest point losses are named, not just a number.
- **Application, bureau, and previous-application history** combined into a single feature set (65 candidate features, 10 categorical).
- **A live demo** ([credit-risk-alven.vercel.app](https://credit-risk-alven.vercel.app)) scoring the shipped model against real held-out applicants.

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| ML | scikit-learn, scipy (for cross-checking the from-scratch implementation) |
| Core logic | From-scratch WoE/IV, logistic regression (gradient descent), AUC/GINI/KS/PSI |
| Notebook | Jupyter, nbconvert |

## Why this version of the repo looks different

This used to be a bigger pipeline: 400 candidate features, a LightGBM
benchmark, calibration figures, a live demo site. That work was real and I
verified it end to end at the time. I replaced it with this rebuild on
purpose. I wanted to see if I could get to a working scorecard again on my
own, without opening the old code, and be honest in the notebook about what
went wrong along the way instead of only showing the finished version. The
old pipeline is still in this repo's git history if you want to see it.

Two things from the old version are still here deliberately: `figs/` and
`output/demo_applicants.json`. The live demo at
[credit-risk-alven.vercel.app](https://credit-risk-alven.vercel.app) reads
both straight from this repo, so removing them would have quietly broken a
page that still works. Everything else described below is the new build.

## What's in the notebook

I started with just the loan application form and a naive logistic
regression, to see how far that alone would get me (0.75 AUC, a floor to
beat). Then I built Weight of Evidence, logistic regression, and the usual
credit-scoring metrics (AUC, GINI, KS, PSI) from scratch instead of
importing them, and checked every one against scikit-learn and scipy before
trusting it. One of those checks failed the first time: my KS statistic was
off under tied scores because I was checking the good/bad gap in the middle
of a tied block instead of at a real threshold. Fixed it, and it matched
exactly.

After that I added bureau history and previous-application data, which
pushed AUC to 0.76, and built a points-based scorecard where every WoE bin
becomes a specific score contribution, so a declined applicant's biggest
point losses are things like "lost 18 points on employment history," not a
number with no explanation attached.

I also caught a mistake in my own comparison. I thought I'd found real
model instability from correlated features (my from-scratch model and
scikit-learn's only agreed on 90% of predictions), when the actual problem
was that scikit-learn was using `class_weight="balanced"` and my own
implementation wasn't. Once I fixed that, the two models agreed on
99.9997% of predictions. Worth remembering: check that you're comparing two
models solving the same problem before concluding they disagree for an
interesting reason.

## Results

| | This rebuild | Old pipeline |
|---|---:|---:|
| AUC | 0.762 | 0.751 |
| KS | 0.394 | 0.377 to 0.381 |
| Candidate features | 65 | 400 |
| Categorical features used | 10 | 0 |

Not a fair head-to-head. Different train/test splits, and the old pipeline
did a lot more relational feature engineering than I redid here. What I
take from it: a much smaller, independently built feature set gets
comparable results, and the categorical columns the old pipeline dropped
(occupation type, income type) turned out to carry real signal.

## What I'd actually recommend

Logistic regression on Weight of Evidence features, not a gradient-boosted
model, even though the tree model would probably score higher. A declined
applicant is often legally entitled to a specific reason, and this
scorecard gives one directly. Each WoE bin's direction can be checked in a
table before the model is even trained, instead of trusting that a tree
model learned the right pattern. And the point table only shifts a little
when retrained on new data, which matters if a regulator expects consistent
treatment of similar applicants over time.

This scorecard outputs a probability of default. A full loss provision
under IFRS 9 needs that multiplied by loss given default and exposure at
default, which are separate models I didn't build here.

## If I extended this

The scorecard's base odds (20 good borrowers per bad one, at a score of
600) is a reasonable default, not something calculated from this
population. Calibrating it to this dataset's real 8% default rate would be
the first fix. Second would be an actual out-of-time validation split
instead of a random one.

## Known Limitations

- **Base odds aren't calibrated to this population.** 20 good borrowers per bad one at a score of 600 is a reasonable default, not fit to this dataset's actual ~8% default rate (see "If I extended this" above).
- **Validation split is random, not out-of-time.** A regulator-facing scorecard should be validated on a forward time split; this one wasn't.
- **PD only, not a full IFRS 9 loss estimate.** This scorecard outputs a probability of default; loss given default and exposure at default are separate models not built here.
- **The rebuild-vs-old-pipeline comparison in Results isn't a controlled benchmark** — different train/test splits, not an apples-to-apples A/B.

## Project Structure

```
Credit_Risk_Scorecard.ipynb   the notebook, run end to end
build_notebook.py             generates the notebook, edit this not the .ipynb
src/
  io_raw.py                   loads the application table
  baseline_features.py        application-level features
  baseline_model.py           the naive baseline
  woe_iv.py                   Weight of Evidence / Information Value
  metrics_scratch.py          AUC, GINI, KS, PSI
  from_scratch_lr.py          logistic regression by gradient descent
  relational_features.py      bureau / previous application / payment history
  scorecard.py                the points scorecard and reason codes
```

## Dataset

Download the 8 CSVs from [Home Credit Default Risk on Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)
and put them in `data/` (not shipped in this repo).

## Running it

```bash
pip install -r requirements.txt
python build_notebook.py
jupyter nbconvert --to notebook --execute Credit_Risk_Scorecard.ipynb \
  --output Credit_Risk_Scorecard.ipynb --ExecutePreprocessor.timeout=900
```

## Roadmap

- [x] From-scratch WoE/IV, logistic regression, and scoring metrics, verified against scikit-learn/scipy
- [x] Bureau and previous-application relational features
- [x] Points-based scorecard with reason codes
- [ ] Calibrate base odds to this dataset's actual ~8% default rate
- [ ] Out-of-time validation split
- [ ] LGD/EAD models for a full IFRS 9 loss estimate

## License

MIT — see [`LICENSE`](LICENSE).

## Credits

Author: **Alven Yuka** — CPA Finalist (Kenya). Built on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset (Kaggle).

## Connect

📫 [alvenyuka2@gmail.com](mailto:alvenyuka2@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/alven-yuka-610b78174/) · 🐙 [GitHub](https://github.com/alvenyuka)
