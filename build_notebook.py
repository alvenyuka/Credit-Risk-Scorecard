"""
Generates Credit_Risk_Scorecard.ipynb cell by cell. Edit this file, not the
notebook directly, then regenerate:

    python build_notebook.py
    jupyter nbconvert --to notebook --execute Credit_Risk_Scorecard.ipynb \
        --output Credit_Risk_Scorecard.ipynb --ExecutePreprocessor.timeout=3600

The notebook builds the scorecard one small step at a time (load, explore,
split, baseline, Weight of Evidence, the three feature gates, the fit, the
points table, evaluation) so that a reader can follow and recreate it. Each
step is a numbered task with a short code cell and, where it matters, a
"check your work" assertion.

The step-by-step build is then compared with run_pipeline.build_validation_scores(),
the tested function that writes outputs/results.json, and the notebook asserts
that the two produce the same features, coefficients and scores. The last cell
asserts that every figure shown equals outputs/results.json.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ---------------------------------------------------------------------------
# Introduction

md("""
# Credit Risk Scorecard

A points-based credit scorecard for Home Credit's public dataset: 307,511 loan
applicants across 8 related tables (the application, credit-bureau records,
previous applications and repayment history).

**The question:** using an applicant's application and their history with other
lenders, can a model separate borrowers who repay from those who default well
enough for a lending decision, with a specific, readable reason behind every
score?

**How this notebook is organised.** It follows the same three stages as any
model-building project, and each stage is broken into small numbered tasks:

1. **Prepare data:** import the tables, explore them, and split off a holdout.
2. **Build model:** set a baseline, build the scorecard step by step, and
   evaluate it on the holdout.
3. **Communicate results:** turn the scores into a lending policy, explain an
   individual decision, and make a recommendation.

Each task has a short code cell. Cells marked **Check your work** contain
assertions: if a step went wrong, the notebook stops there instead of carrying
a wrong number forward. The helper functions imported from `src/` are covered
by the test suite (`tests/`), and the notebook's last cell checks that every
figure shown here equals the one the pipeline writes to `outputs/results.json`.
""")

code("""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# The project's own functions live in src/ (tested in tests/); make them importable.
sys.path.insert(0, "src")

pd.set_option("display.width", 120)
""")

# ---------------------------------------------------------------------------
# 1. Prepare data

md("""
---
## 1. Prepare data

### 1.1 Import

The main table is `application_train.csv`: one row per loan application, with
the outcome in `TARGET` (1 = the applicant defaulted, 0 = repaid).
""")

md("""
**Task 1.1.1:** Load the application table and look at its size and a few
columns.
""")

code("""
from io_raw import load_application

# Read application_train.csv from data/ (about 166 MB)
app = load_application("train")

print(f"rows: {len(app):,}   columns: {app.shape[1]}")
app[["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "DAYS_EMPLOYED"]].head()
""")

code("""
# Check your work: one row per applicant, and every applicant is in the file
assert len(app) == 307_511
assert app["SK_ID_CURR"].is_unique
""")

md("""
**Task 1.1.2:** How common is default? This decides which metrics make sense
later.
""")

code("""
# Share of applicants in each outcome class
app["TARGET"].value_counts(normalize=True).rename({0: "repaid", 1: "defaulted"})
""")

md("""
About 8% of applicants defaulted. With a target this imbalanced, accuracy is a
poor guide: a model that approved everyone would be 92% "accurate" and useless.
The model is judged on **ranking** instead: does it put defaulters below
repayers? AUC and KS, defined in section 2.4, measure exactly that.
""")

md("""
**Task 1.1.3:** Look for values that are not what they claim to be.
`DAYS_EMPLOYED` counts days before the application, so it should be negative.
""")

code("""
# How many applicants have a positive (impossible) employment length, and what is it?
positive = app["DAYS_EMPLOYED"] > 0
print(f"applicants with DAYS_EMPLOYED > 0: {positive.sum():,}")
print(f"distinct values among them: {app.loc[positive, 'DAYS_EMPLOYED'].unique()}")
""")

md("""
Every one of them holds the same placeholder, 365243 days (about a thousand
years). These are pensioners and unemployed applicants. Left in, the value would
give retirees a millennium of work history and break every ratio built on it, so
the next task sets it to missing.
""")

md("""
**Task 1.1.4:** Build the application features with `engineer_baseline` from
`src/baseline_features.py`. It converts day counts to years, replaces the
placeholder with a missing value, adds ratio features (credit to income,
annuity to income and so on), and summarises the three external scores.
""")

code("""
from baseline_features import engineer_baseline

# One row per applicant: the raw columns kept plus the engineered ones
feats = engineer_baseline(app)

print(f"feature table: {feats.shape[0]:,} rows x {feats.shape[1]} columns")
feats[["AGE_YEARS", "EMPLOYED_YEARS", "CREDIT_INCOME_RATIO", "EXT_SOURCE_MEAN"]].describe().round(2)
""")

code("""
from feature_lists import PROHIBITED_BASES

# Check your work: the placeholder is now missing for exactly the 55,374 flagged applicants
assert (feats["EMPLOYED_YEARS"].isna() == positive).all()
assert feats["EMPLOYED_YEARS"].isna().sum() == positive.sum() == 55_374
# Check your work: sex and marital status never enter the feature table
assert not set(PROHIBITED_BASES) & set(feats.columns)
""")

md("""
Sex (`CODE_GENDER`) and marital status (`NAME_FAMILY_STATUS`) are prohibited
bases for a credit decision under the US Equal Credit Opportunity Act and
Regulation B. They are excluded from every feature list (`PROHIBITED_BASES` in
`src/feature_lists.py`), and `tests/test_fair_lending.py` pins that.
""")

md("""
**Task 1.1.5:** Add each applicant's history from the other seven tables:
credit-bureau records, previous Home Credit applications and repayment
behaviour. `load_relational_features` aggregates them to one row per applicant
(counts, averages, the share of bureau credit overdue, how often past
applications were refused). The aggregation reads about 2.5 GB, so the result
is cached and the cache is checked against a fingerprint of the code and data.
""")

code("""
from relational_features import load_relational_features

# One row per applicant with their history at other lenders and at Home Credit
rel = load_relational_features()

# Left join: applicants with no history keep their row, with missing history values
full = feats.merge(rel, on="SK_ID_CURR", how="left")
print(f"history table: {rel.shape[1] - 1} features   combined table: {full.shape[0]:,} rows x {full.shape[1]} columns")
""")

code("""
# Check your work: the join added columns, not rows
assert len(full) == len(feats) and full["SK_ID_CURR"].is_unique
""")

md("""
### 1.2 Explore

Before any modelling: does the data hold signal at all? The three `EXT_SOURCE`
columns are external bureau-style scores supplied with the data. Group the
applicants into ten equal-sized bands of their average external score and look
at the default rate in each band. (This is description only; nothing here is
fitted or used by the model.)
""")

md("""
**Task 1.2.1:** Default rate by decile of `EXT_SOURCE_MEAN`.
""")

code("""
# Ten equal-sized bands of the average external score (lowest scores first)
band = pd.qcut(full["EXT_SOURCE_MEAN"], 10)

# Default rate and number of applicants in each band
full.groupby(band, observed=True)["TARGET"].agg(default_rate="mean", applicants="size").round(4)
""")

md("""
The default rate falls steadily from the lowest band to the highest, roughly
tenfold. That smooth, one-directional pattern is what a scorecard is built to
capture: each band can be given a number of points, and the points rise as the
risk falls.
""")

md("""
**Task 1.2.2:** Missing values matter too. How does the default rate differ for
applicants with and without an employment length (the pensioner group from Task
1.1.3)?
""")

code("""
# Default rate for applicants with and without an employment length
full.groupby(full["EMPLOYED_YEARS"].isna())["TARGET"].mean().rename(
    {False: "employment length known", True: "missing (pensioners and unemployed)"})
""")

md("""
The missing group defaults *less* often than average. So a missing value is
information, not noise to be filled in with an average. Weight of Evidence,
used below, gives missing values a bin of their own for exactly this reason.
""")

md("""
### 1.3 Split

The model is fitted on one part of the data and judged on another part it has
never seen. 20% of applicants are held out, stratified so the default rate is
the same in both parts.
""")

md("""
**Task 1.3.1:** Separate the target from the features and split 80/20.
""")

code("""
# Target vector (what we predict) and feature matrix (what we predict it from)
y = full["TARGET"]
X = full.drop(columns=["TARGET", "SK_ID_CURR"])

# Same seed and stratification as the pipeline, so both use identical rows
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"training: {len(X_train):,} applicants, default rate {y_train.mean():.2%}")
print(f"holdout:  {len(X_val):,} applicants, default rate {y_val.mean():.2%}")
""")

code("""
# Check your work: an 80/20 split with the same default rate on both sides
assert len(X_val) == 61_503
assert abs(y_train.mean() - y_val.mean()) < 1e-3
""")

# ---------------------------------------------------------------------------
# 2. Build model

md("""
---
## 2. Build model

### 2.1 Baseline

A model is only useful if it beats something simpler. Two baselines set the
floor.
""")

md("""
**Task 2.1.1:** The "dumb" baseline gives every applicant the same score.
""")

code("""
# Every applicant gets the same score, so the model cannot tell anyone apart
constant_score = np.zeros(len(y_val))
print(f"AUC of a constant score: {roc_auc_score(y_val, constant_score):.4f}")
""")

md("""
> **What's AUC?** Pick one defaulter and one repayer at random. AUC is the
> probability that the model gives the defaulter the riskier score. 0.5 is a
> coin toss; 1.0 separates every pair. A constant score ties every pair, so it
> scores exactly 0.5.
""")

md("""
**Task 2.1.2:** A sensible baseline uses the application form alone: a
standard scikit-learn logistic regression with missing values filled by the
median, features scaled, and categories one-hot encoded (`build_pipeline` in
`src/baseline_model.py`). The full scorecard has to beat this.
""")

code("""
from baseline_model import build_pipeline

# Impute, scale, one-hot encode, then logistic regression (application columns only)
pipe = build_pipeline()
pipe.fit(X_train, y_train)

baseline_auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
print(f"application-only baseline AUC: {baseline_auc:.4f}")
""")

md("""
The application form alone gives the AUC printed above. That is the benchmark.
The pipeline selects its own columns by name, so it reads only the application
features even though `X_train` also holds the history columns.
""")

md("""
### 2.2 Iterate: build the scorecard step by step

A scorecard does not feed raw values to the model. It first cuts each feature
into bins and replaces each bin with its **Weight of Evidence (WoE)**.

> **What's Weight of Evidence?** For one bin of one feature,
> WoE = ln(share of all repayers in the bin / share of all defaulters in the bin).
> A positive WoE means the bin is safer than average, negative means riskier,
> and zero means the bin tells you nothing. Because every bin becomes one number,
> a credit committee can read and check the effect of every feature, bin by bin.
>
> **What's Information Value (IV)?** The sum over bins of
> (share of repayers - share of defaulters) x WoE. It measures how well the
> feature separates the two groups: below 0.02 is useless, 0.1 to 0.3 medium,
> 0.3 to 0.5 strong, and above 0.5 is suspiciously strong (check for leakage).

The binning and the WoE values are fitted on the training split only, then
applied to the holdout.
""")

md("""
**Task 2.2.1:** Compute WoE and IV by hand for one feature, `EXT_SOURCE_MEAN`,
in five steps.
""")

code("""
col = "EXT_SOURCE_MEAN"
x = X_train[col]

# 1. Cut points: the 0%, 10%, ..., 100% quantiles of the non-missing values
edges = np.unique(np.quantile(x.dropna().astype(float), np.linspace(0, 1, 11)))
edges[0], edges[-1] = -np.inf, np.inf  # open the first and last bins

# 2. Put every applicant in a bin; missing values get a bin of their own
bins = pd.cut(x, bins=edges, include_lowest=True).astype(str).where(x.notna(), "Missing")

# 3. Count defaulters (bad) and repayers (good) in each bin
woe_table = pd.DataFrame({"bin": bins, "y": y_train}).groupby("bin")["y"].agg(n="count", bad="sum")
woe_table["good"] = woe_table["n"] - woe_table["bad"]

# 4. Each bin's share of all goods and of all bads (+0.5 so an empty bin never gives log(0))
eps, k = 0.5, len(woe_table)
dist_good = (woe_table["good"] + eps) / (woe_table["good"].sum() + eps * k)
dist_bad = (woe_table["bad"] + eps) / (woe_table["bad"].sum() + eps * k)

# 5. WoE per bin, and IV for the whole feature
woe_table["woe"] = np.log(dist_good / dist_bad)
iv_by_hand = ((dist_good - dist_bad) * woe_table["woe"]).sum()
print(f"IV of {col}, computed by hand: {iv_by_hand:.4f}")
""")

md("""
The same calculation is packaged as `fit_woe` in `src/woe_iv.py`, which the rest
of the notebook uses for all 62 candidate features.
""")

code("""
from woe_iv import fit_woe

reference = fit_woe(X_train[col], y_train, is_categorical=False, n_bins=10)

# Check your work: the hand calculation and the tested function agree on every bin
assert np.allclose(woe_table.loc[reference["table"].index, "woe"], reference["table"]["woe"])
assert abs(iv_by_hand - reference["iv"]) < 1e-12
reference["table"][["n", "bad", "woe"]].round(4)
""")

md("""
Read down the `woe` column: it rises from strongly negative in the lowest band
to strongly positive in the highest, the same pattern as Task 1.2.1, now on a
scale the model can use.
""")

md("""
**Task 2.2.2:** Fit WoE for every candidate feature: 54 numeric (cut into ten
bins) and 8 categorical (one bin per category).
""")

code("""
from feature_lists import IV_THRESHOLD, MODEL_CATEGORICAL, MODEL_NUMERIC

# One WoE table per feature, fitted on the training split only
woe_fits = {c: fit_woe(X_train[c], y_train, is_categorical=False, n_bins=10) for c in MODEL_NUMERIC}
woe_fits.update({c: fit_woe(X_train[c], y_train, is_categorical=True) for c in MODEL_CATEGORICAL})

# The IV of each feature, for the screening steps below
ivs = {c: fit["iv"] for c, fit in woe_fits.items()}
print(f"candidate features: {len(woe_fits)}")
""")

code("""
# Check your work: 62 candidates, and the prohibited bases are not among them
assert len(woe_fits) == 62 and not set(PROHIBITED_BASES) & set(woe_fits)
""")

md("""
The candidates now pass through three gates, each removing features for a
stated reason.

**Task 2.2.3: Gate 1, information value.** Rank the features by IV and drop
those below 0.01, which carry almost no signal.
""")

code("""
from woe_iv import iv_strength

# Highest IV first (ties broken by name, so the order never changes between runs)
iv_ranked = sorted(woe_fits, key=lambda c: (-ivs[c], c))
iv_kept = [c for c in iv_ranked if ivs[c] >= IV_THRESHOLD]

print(f"Gate 1: {len(iv_kept)} of {len(iv_ranked)} features have IV >= {IV_THRESHOLD}\\n")
for c in iv_ranked[:15]:
    print(f"  {c:30s} IV={ivs[c]:.4f}  ({iv_strength(ivs[c])})")
""")

md("""
The three external scores carry far more information than any field on the
application form. `EXT_SOURCE_MEAN` crosses the threshold that is labelled
"check for leakage", so it was checked rather than assumed: it is the average of
the three external scores, all of which exist when the application is made, so
its IV reflects their combined strength, not information from after the
decision.
""")

md("""
**Task 2.2.4:** One encoding problem to rule out. `engineer_baseline` also made
an anomaly flag, `DAYS_EMPLOYED_ANOM`, for the placeholder found in Task 1.1.3.
Compare it with the `Missing` bin of `EMPLOYED_YEARS`.
""")

code("""
# WoE of the flag, and of the Missing bin of employment length
flag_fit = fit_woe(X_train["DAYS_EMPLOYED_ANOM"], y_train, is_categorical=True)
emp_table = woe_fits["EMPLOYED_YEARS"]["table"]

print(f"DAYS_EMPLOYED_ANOM = 1:    n={flag_fit['table'].loc['1', 'n']:,}  WoE={flag_fit['table'].loc['1', 'woe']:+.4f}")
print(f"EMPLOYED_YEARS = Missing:  n={emp_table.loc['Missing', 'n']:,}  WoE={emp_table.loc['Missing', 'woe']:+.4f}")
print(f"same applicants in both:   {(X_train['DAYS_EMPLOYED_ANOM'] == 1).equals(X_train['EMPLOYED_YEARS'].isna())}")
""")

md("""
The flag and the `Missing` bin are the same applicants with the same WoE, and
the WoE is positive: this group, largely pensioners, defaults less often than
average. Offering both encodings to the model would let the solver give them
opposite signs, so one of them would appear to penalise a lower-risk group,
which is a wrong adverse-action reason and a fair-lending problem (Regulation B
restricts discounting pension income). The scorecard therefore reads the group
only through `EMPLOYED_YEARS`; the flag is never offered to the model (it is not
in the 62 candidates).
""")

md("""
**Task 2.2.5:** Replace every value with its bin's WoE, for the features that
passed Gate 1. This is the matrix the model is fitted on.
""")

code("""
from woe_iv import transform_woe


def woe_encode(frame, cols):
    # One column per feature: each value replaced by the WoE of its bin
    return np.column_stack([transform_woe(frame[c], woe_fits[c]).to_numpy(dtype=float) for c in cols])


W_iv = woe_encode(X_train, iv_kept)
print(f"WoE matrix: {W_iv.shape[0]:,} applicants x {W_iv.shape[1]} features")
""")

md("""
**Task 2.2.6: Gate 2, correlation.** Two features that move together almost
perfectly say the same thing twice, and the model cannot tell which one deserves
the weight. Walk down the features in IV order and keep a feature only if its
correlation with every feature already kept is at most 0.9.
""")

code("""
from feature_lists import CORRELATION_THRESHOLD

corr = np.corrcoef(W_iv, rowvar=False)        # correlation of every pair of WoE columns
pos = {c: i for i, c in enumerate(iv_kept)}   # column number of each feature

corr_kept, corr_dropped = [], []
for c in iv_kept:                              # highest IV first
    partner = next((k for k in corr_kept if abs(corr[pos[c], pos[k]]) > CORRELATION_THRESHOLD), None)
    if partner is None:
        corr_kept.append(c)                    # no near-duplicate kept yet: keep it
    else:
        corr_dropped.append((c, partner, float(corr[pos[c], pos[partner]])))

print(f"Gate 2: {len(corr_dropped)} dropped, {len(corr_kept)} kept")
for c, partner, r in corr_dropped:
    print(f"  dropped {c:28s} kept {partner:28s} r={r:+.3f}")
""")

md("""
In each pair the higher-IV feature survives. `BUREAU_OVERDUE_MAX` and
`BUREAU_OVERDUE_MEAN` correlate at 1.000 not because they mean the same thing but
because most applicants have no overdue bureau credit, so ten quantile bins
collapse both into the same two-bin split.
""")

md("""
**Task 2.2.7: Gate 3, coefficient sign.** WoE is ln(good/bad), so a higher WoE
always means safer. In a model of the probability of default, every coefficient
should therefore be **negative**. A positive one means the feature's effect has
reversed once the other features are in the model, and the scorecard would hand
points to the riskier bins. Fit the model, remove the feature with the largest
positive coefficient, refit, and repeat until none is left.
""")

code("""
# Standardise each WoE column (mean 0, spread 1) so the solver treats them evenly
W_corr = W_iv[:, [pos[c] for c in corr_kept]]
m, s = W_corr.mean(axis=0), W_corr.std(axis=0)
s[s == 0] = 1.0
Ws = (W_corr - m) / s

sign_kept, sign_removed = list(corr_kept), []
while True:
    cols_now = [corr_kept.index(c) for c in sign_kept]
    fit = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Ws[:, cols_now], y_train)
    wrong = [(c, float(b)) for c, b in zip(sign_kept, fit.coef_.ravel()) if b > 0]
    if not wrong:
        break                                              # every coefficient is negative
    victim = max(wrong, key=lambda t: (t[1], t[0]))        # the largest positive coefficient
    sign_removed.append(victim)
    sign_kept.remove(victim[0])

print(f"Gate 3: {len(sign_removed)} removed, {len(sign_kept)} kept")
for c, b in sign_removed:
    print(f"  removed {c:28s} coefficient={b:+.4f}")
""")

md("""
`AGE_YEARS` is among the features removed. Its coefficient reverses once
employment history and income type are in the model. Kept with a positive
coefficient, age would have given negative points to the oldest applicants,
which Regulation B does not allow for applicants aged 62 or over. Removing it,
rather than forcing its sign, keeps every remaining coefficient a genuine fitted
value.

`class_weight="balanced"` makes the 8% of defaulters count as much as the 92% of
repayers during fitting; Task 2.3.2 undoes its effect on the intercept.
""")

md("""
**Task 2.2.8:** Fit the final model with the project's own logistic regression,
written from first principles in `src/from_scratch_lr.py` (batch gradient
descent with a small L2 penalty). It penalises the weights slightly differently
from scikit-learn, so the sign check is repeated on this fit.
""")

code("""
from from_scratch_lr import FromScratchLogisticRegression
from metrics_scratch import auc_rank_sum

kept_cols = list(sign_kept)
while True:
    # WoE matrices for the training and holdout splits, standardised on training statistics
    Xw_train, Xw_val = woe_encode(X_train, kept_cols), woe_encode(X_val, kept_cols)
    mean, std = Xw_train.mean(axis=0), Xw_train.std(axis=0)
    std[std == 0] = 1.0
    Xw_train_s, Xw_val_s = (Xw_train - mean) / std, (Xw_val - mean) / std

    scratch = FromScratchLogisticRegression(lr=0.5, n_iter=3000, l2=1e-3)
    scratch.fit(Xw_train_s, y_train.to_numpy(), class_weight="balanced")
    wrong = [(c, float(b)) for c, b in zip(kept_cols, scratch.coef_) if b > 0]
    if not wrong:
        break
    victim = max(wrong, key=lambda t: (t[1], t[0]))
    sign_removed.append(victim)
    kept_cols.remove(victim[0])

full_auc = auc_rank_sum(y_val, scratch.predict_proba(Xw_val_s))
print(f"final features: {len(kept_cols)} of {len(woe_fits)} candidates")
print(f"solver: {scratch.n_iter_} iterations, converged={scratch.converged_}")
print(f"holdout AUC: {full_auc:.4f}  (application-only baseline: {baseline_auc:.4f})")
""")

code("""
# Check your work: the scorecard beats the application-only baseline, every sign is right,
# and the solver stopped because it converged, not because it ran out of iterations
assert full_auc > baseline_auc
assert (scratch.coef_ < 0).all() and scratch.converged_
""")

md("""
**Task 2.2.9:** Every step above is also packaged in one tested function,
`build_validation_scores` in `src/run_pipeline.py`, which is what writes
`outputs/results.json`. Run it and confirm it builds the same model.
""")

code("""
from run_pipeline import build_validation_scores, summarise

# The packaged pipeline: the same steps, plus the points table and the acceptance checks
result, sc, Xf_val, yf_val, scores = build_validation_scores(full, verbose=False)
""")

code("""
# Check your work: the step-by-step build and the pipeline agree exactly
assert result["kept_cols"] == kept_cols
assert [d[:2] for d in result["correlation_gate_dropped"]] == [d[:2] for d in corr_dropped]
assert [r[0] for r in result["sign_gate_removed"]] == [r[0] for r in sign_removed]
assert np.allclose([r[1] for r in result["sign_gate_removed"]], [r[1] for r in sign_removed])
assert np.allclose(result["model"].coef_, scratch.coef_)
assert abs(result["val_auc"] - full_auc) < 1e-12
print("step-by-step build matches the pipeline: same features, coefficients and AUC")
""")

md("""
### 2.3 Turn the model into points

The model outputs log-odds of default. A scorecard re-expresses them as points
that a loan officer can add up without the model.

**Task 2.3.1:** The model was fitted on standardised WoE. Convert its
coefficients back to raw WoE units, so that points can be read straight off the
WoE tables.
""")

code("""
# coef on raw WoE = coef on standardised WoE / spread; the intercept absorbs the means
coef_raw = scratch.coef_ / std
intercept_balanced = scratch.intercept_ - np.sum(scratch.coef_ * mean / std)

# Check your work: matches the pipeline's conversion
assert np.allclose(coef_raw, sc["coef_raw"]) and abs(intercept_balanced - sc["intercept_balanced"]) < 1e-12
print(f"intercept on raw WoE: {intercept_balanced:+.4f}")
""")

md("""
**Task 2.3.2:** Correct the intercept. Balanced class weights fit the model as if
half of all applicants defaulted. Adding logit(training default rate) -
logit(0.5) moves the intercept back to the real 8% without changing any
coefficient, so the ranking (AUC, KS) is unchanged but probabilities mean what
they say (the King and Zeng prior correction).
""")

code("""
def logit(p):
    return np.log(p / (1 - p))


train_bad_rate = y_train.mean()
intercept = intercept_balanced + logit(train_bad_rate) - logit(0.5)

# Check your work: matches the pipeline's prior-corrected intercept
assert abs(intercept - sc["intercept"]) < 1e-12
print(f"training default rate {train_bad_rate:.4f}: intercept {intercept_balanced:+.4f} -> {intercept:+.4f}")
""")

md("""
**Task 2.3.3:** Set the scale. Two business conventions fix it: 600 points means
odds of 20 repayers to 1 defaulter, and every 40 points doubles the odds (PDO,
"points to double the odds"). Then

- factor = PDO / ln(2)
- offset = 600 - factor x ln(20)
- points for a bin = -factor x coefficient x WoE of the bin
- score = offset - factor x intercept + the points of the applicant's bins
""")

code("""
factor = 40 / np.log(2)                     # points per unit of log-odds
offset = 600 - factor * np.log(20)          # puts 20:1 odds at 600 points
base_points = offset - factor * intercept   # points every applicant starts with

# Check your work: the pipeline uses the same scale
assert np.isclose(factor, sc["factor"]) and np.isclose(offset, sc["offset"]) and np.isclose(base_points, sc["base_points"])
print(f"factor {factor:.2f}   offset {offset:.1f}   base points {base_points:.1f}")
""")

md("""
**Task 2.3.4:** Read the points table for the highest-IV feature. Each bin
carries a fixed number of points, and a committee can review every table like
this one before the model goes live.
""")

code("""
top = kept_cols[0]
table = woe_fits[top]["table"][["n", "bad", "woe"]].copy()
table["default_rate"] = table["bad"] / table["n"]
table["points"] = -factor * coef_raw[0] * table["woe"]   # points for each bin of this feature

# Check your work: the same points as the pipeline's table
assert np.allclose(table["points"], sc["points_tables"][top])
print(f"WoE and points for {top}, in bin order:")
table.round(4)
""")

md("""
**Task 2.3.5:** Score one holdout applicant by hand: base points plus the points
of each of their bins.
""")

code("""
from woe_iv import _bin_labels

applicant = Xf_val.iloc[0]
total = base_points
for c in kept_cols:
    fit = woe_fits[c]
    label = _bin_labels(pd.Series([applicant[c]]), fit["is_categorical"], fit["edges"]).iloc[0]
    total += sc["points_tables"][c].get(label, 0.0)   # an unseen category scores 0 points

# Check your work: the hand total equals the pipeline's score for this applicant
assert np.isclose(min(max(total, 300), 850), scores.iloc[0])
print(f"applicant {applicant.name}: {total:.1f} points")
""")

md("""
The full points table for every kept feature is written to
`outputs/scorecard_points.json`. The full list of kept features with their
monotonicity:
""")

code("""
from woe_iv import monotonicity_report

report = monotonicity_report(result["woe_fits"], kept_cols)
numeric = report[report["type"] == "numeric"]
print(f"kept features: {len(kept_cols)}  (numeric: {len(numeric)}, categorical: {len(report) - len(numeric)})")
print(f"numeric features whose WoE moves in one direction across the bins: "
      f"{int(numeric['monotonic_woe'].astype(bool).sum())} of {len(numeric)}")
""")

md("""
### 2.4 Evaluate

**Task 2.4.1:** Before trusting the evaluation metrics, test them. AUC and KS
are implemented from first principles in `src/metrics_scratch.py`; check them
against scikit-learn and scipy on synthetic data with tied scores.

> **What's KS?** The Kolmogorov-Smirnov statistic is the largest gap between the
> share of defaulters and the share of repayers scoring below any cut-off. It is
> the standard credit-industry measure of how well a score separates the groups.
""")

code("""
from scipy.stats import ks_2samp

from metrics_scratch import ks_statistic

# Synthetic outcomes and scores; rounding to 2 decimals forces tied scores
rng = np.random.default_rng(0)
y_check = rng.integers(0, 2, 5000)
score_check = np.round(rng.normal(0, 1, 5000) + y_check * 0.8, 2)

scratch_auc, sklearn_auc = auc_rank_sum(y_check, score_check), roc_auc_score(y_check, score_check)
scratch_ks = ks_statistic(y_check, score_check)
scipy_ks = ks_2samp(score_check[y_check == 1], score_check[y_check == 0]).statistic
print(f"AUC: from scratch {scratch_auc:.6f}, scikit-learn {sklearn_auc:.6f}")
print(f"KS:  from scratch {scratch_ks:.6f}, scipy {scipy_ks:.6f}")

# Check your work: both agree to floating-point precision
assert abs(scratch_auc - sklearn_auc) < 1e-9 and abs(scratch_ks - scipy_ks) < 1e-9
""")

md("""
The tie test matters for KS. Computed row by row after sorting, KS would measure
the gap in the middle of a block of tied scores, which is not a point on the
curve. The implementation groups by distinct score first, and the rounding above
forces ties so that an ungrouped version would fail.
""")

md("""
**Task 2.4.2:** Compare the from-scratch model with scikit-learn's fit of the
same class-balanced objective.
""")

code("""
print(f"from-scratch vs scikit-learn: prediction correlation={result['pred_corr']:.6f}, "
      f"largest coefficient difference={result['coef_diff_max']:.4f}")
""")

md("""
Their predictions agree almost exactly. Coefficients differ slightly more than
predictions because several bureau features remain correlated, which leaves the
individual coefficients less determined than the combined score.
""")

md("""
**Task 2.4.3:** Measure the scorecard on the holdout. `summarise` computes every
holdout figure the README quotes.
""")

code("""
summary = summarise(result, sc, Xf_val, yf_val, scores)

lo, hi = summary["val_auc_ci95_bootstrap"]
print(f"AUC  {summary['val_auc']:.4f}  (95% bootstrap interval {lo:.4f} to {hi:.4f})")
print(f"KS   {summary['val_ks']:.4f}")
print(f"GINI {summary['val_gini']:.4f}  (= 2 x AUC - 1)")
print(f"average score, repaid: {summary['mean_score_repaid']:.1f}   defaulted: {summary['mean_score_defaulted']:.1f}")
print(f"score range on the holdout: {summary['score_min']:.0f} to {summary['score_max']:.0f}")
""")

md("""
The bootstrap interval resamples the holdout 500 times; the AUC moves within it,
and the whole interval sits above the baseline. Repayers average a clearly
higher score than defaulters.
""")

md("""
**Task 2.4.4:** Check calibration: does a predicted 5% default rate really mean
5%? Group the holdout into ten bands of predicted probability of default (PD).
""")

code("""
near = summary["observed_odds_near_base_score"]
print(f"odds the scorecard assigns at 600 points: {summary['implied_odds_at_base_score']:.1f}:1")
print(f"observed odds among holdout applicants scoring {near['score_band'][0]}-{near['score_band'][1]}: "
      f"{near['good_to_bad']:.1f}:1 (n={near['n']:,})")
print(f"mean predicted PD {summary['mean_predicted_pd_val']:.4f} vs observed default rate {summary['val_bad_rate']:.4f}\\n")
pd.DataFrame(summary["calibration_by_decile"]).round(4)
""")

md("""
Predicted and observed default rates are close in every band, and the observed
odds around 600 points are close to the 20:1 the scale promises. That is the
prior correction of Task 2.3.2 at work.
""")

md("""
**Task 2.4.5:** Check stability. The Population Stability Index (PSI) compares
the score distribution on the training split with the holdout.
""")

code("""
print(f"PSI, training scores vs holdout scores: {summary['psi_train_to_holdout_scores']:.4f}")
""")

md("""
A PSI near zero is expected for a random split; it is the baseline for
monitoring drift once the scorecard is in use (above 0.25 is the conventional
warning level).
""")

md("""
**Task 2.4.6:** Plot the score distributions of repayers and defaulters, and the
default rate by score band.
""")

code("""
from make_figures import default_rate_by_band, score_distribution

_ = score_distribution(scores, yf_val)
""")

code("""
_ = default_rate_by_band(scores, yf_val)
""")

md("""
**Task 2.4.7:** Run the fair-lending acceptance checks.
""")

code("""
age = summary["reg_b_age_62_plus"]
print(f"every coefficient negative (WoE sign check): {summary['all_coefficients_negative']}")
print(f"AGE_YEARS in the scorecard: {age['age_feature_in_scorecard']}; check passed: {age['passed']}")
print(f"points for the EMPLOYED_YEARS Missing bin (pensioners and the unemployed): "
      f"{summary['employed_years_missing_bin_points']:+.1f}")

# Check your work: the acceptance checks pass
assert summary["all_coefficients_negative"] and age["passed"]
assert summary["employed_years_missing_bin_points"] > 0
""")

md("""
The pensioner group receives positive points, consistent with its lower default
rate (Task 1.2.2), and no applicant aged 62 or over can receive negative points
for age.
""")

# ---------------------------------------------------------------------------
# 3. Communicate results

md("""
---
## 3. Communicate results

**Task 3.1:** Turn the scores into a lending decision. Approve only the
highest-scoring share of the holdout and measure what each policy would have
cost. Credit loss = credit amount of each approved loan that defaulted x a 45%
loss given default (an assumption: the Basel foundation-IRB value for senior
unsecured exposures).
""")

code("""
import json

from business_impact import lending_policy_table

policy = lending_policy_table(scores.to_numpy(), yf_val.to_numpy(), Xf_val["AMT_CREDIT"].to_numpy())
policy[["approval_rate", "cut_off_score", "default_rate_approved", "loss_avoided_pct",
        "good_borrowers_declined_pct"]].round(4)
""")

code("""
# Check your work: the same table as outputs/business_impact.json
published = pd.DataFrame(json.load(open("outputs/business_impact.json", encoding="utf-8"))["policies"])
assert np.allclose(policy["loss_avoided_pct"], published["loss_avoided_pct"])
assert (policy["good_borrowers_declined"] == published["good_borrowers_declined"]).all()
""")

md("""
Each row is a policy a credit committee could choose. Lowering the approval rate
avoids more of the credit loss but turns away more good borrowers; the table
puts both sides of that trade on one line. At 80% approval, the scorecard avoids
46% of the loss that approving everyone would have caused.
""")

md("""
**Task 3.2:** Explain an individual decision. Take a holdout applicant who
defaulted and scored below the 80% approval cut-off, and list the reasons: the
characteristics on which they lost the most points against the best bin.
""")

code("""
ex = summary["example_declined_applicant"]
print(f"applicant scoring {ex['score']:.0f}, below the 80% approval cut-off of "
      f"{ex['cut_off_score_at_80pct_approval']:.0f} points. Reasons:")
for r in ex["reasons_points_below_best_bin"]:
    print(f"  {r['feature']}: {r['points']:.1f} points below the best bin")
""")

md("""
This is what the scorecard format buys over a raw probability. Each reason is the
gap between the points the applicant received on a characteristic and the most
points that characteristic can give, so a reason is never a characteristic on
which the applicant already scored the maximum. A gradient-boosted model
explained with SHAP does not hand over a fixed, auditable point table in the
same way.
""")

md("""
### Recommendation to a credit committee

A gradient-boosted model would probably rank applicants somewhat better. The
scorecard is still the stronger choice for a lending decision, for reasons other
than raw accuracy:

- every point has a specific, checkable reason, which a declined applicant is
  entitled to under adverse-action rules;
- the direction of every characteristic is enforced (all coefficients negative
  on WoE inputs) and each WoE table can be reviewed before the model is used;
- the fair-lending conditions are tested, not asserted: no prohibited basis in
  any feature list, and no negative points for applicants aged 62 or over.

**Scope.** The model outputs a probability of default. An IFRS 9 provision also
needs loss given default and exposure at default, which are separate models not
built here.

**Next steps.** Validate on a later population when data with application dates
is available, and replace decile binning of the zero-inflated delinquency
columns with zero-aware bins.
""")

md("""
**Task 3.3:** Final check: every figure in this notebook must equal the one the
pipeline wrote to `outputs/results.json`.
""")

code("""
import math

from results_io import read_results

ref = read_results()["metrics"]


def same(a, b):
    # Compare nested results, allowing only floating-point rounding differences
    if isinstance(a, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, z) for x, z in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


mismatched = [k for k in summary if not same(summary[k], ref[k])]
assert not mismatched, f"notebook and results.json differ on: {mismatched}"
assert abs(full_auc - ref["val_auc"]) < 1e-9 and len(kept_cols) == ref["n_features_kept"]
print(f"notebook figures match outputs/results.json ({len(summary)} entries checked)")
""")

nb["cells"] = cells

with open("Credit_Risk_Scorecard.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"wrote Credit_Risk_Scorecard.ipynb with {len(cells)} cells")
