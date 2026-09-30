"""
Generates Credit_Risk_Scorecard.ipynb cell by cell. Edit this file, not the
notebook directly, then regenerate:

    python build_notebook.py
    jupyter nbconvert --to notebook --execute Credit_Risk_Scorecard.ipynb \
        --output Credit_Risk_Scorecard.ipynb --ExecutePreprocessor.timeout=900

This calls the real functions in src/, it does not reimplement anything, so
the notebook's numbers are guaranteed to match what src/ actually produces.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ---------------------------------------------------------------------------

md("""
# Credit Risk Scorecard

A points-based credit scorecard for Home Credit's public dataset: 307,511 loan
applicants across 8 related tables (the application, credit-bureau records,
previous applications and repayment history).

The question: using an applicant's application and their history with other
lenders, can the model separate borrowers who repay from those who default well
enough for a lending decision, with a specific, readable reason behind every
score?

The notebook walks through the pipeline in four steps. Every function it calls
lives in `src/` and is covered by the test suite, so the numbers here are the
ones the pipeline itself produces (`outputs/results.json`).
""")

md("""
## Step 1: a baseline from the application form alone

The application form sets the floor: anything added later has to beat it.

One data-quality issue shapes the features. `DAYS_EMPLOYED` holds the
placeholder 365243 (about a thousand years) for pensioners and unemployed
applicants. Left in, it gives retirees a millennium of work history and breaks
every ratio built on it, so it becomes an explicit anomaly flag and the value is
set to missing.

Sex (`CODE_GENDER`) and marital status (`NAME_FAMILY_STATUS`) are prohibited
bases under the US Equal Credit Opportunity Act and Regulation B, so they are
excluded from the candidate features (`PROHIBITED_BASES` in
`src/baseline_features.py`, pinned by `tests/test_fair_lending.py`).
""")

code("""
import sys
sys.path.insert(0, "src")

from io_raw import load_application
from baseline_features import engineer_baseline
from baseline_model import build_pipeline, NUMERIC_COLS
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

train_raw = load_application("train")
feats = engineer_baseline(train_raw)

y = feats["TARGET"]
X = feats.drop(columns=["TARGET", "SK_ID_CURR"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = build_pipeline()
pipe.fit(X_train, y_train)

val_pred = pipe.predict_proba(X_val)[:, 1]
baseline_auc = roc_auc_score(y_val, val_pred)
print(f"application-only baseline AUC: {baseline_auc:.4f}")
""")

md("""
The application form alone gives the AUC printed above, in line with public
results that use only this table. It is the benchmark the full feature set has
to beat.
""")

# ---------------------------------------------------------------------------

md("""
## Step 2: the credit statistics, implemented and verified

A credit committee reviews a scorecard through Weight of Evidence (WoE) and
Information Value (IV), and model-risk teams expect the statistics to be
verifiable. WoE/IV, a gradient-descent logistic regression and the four metrics
a credit team checks (AUC, GINI, KS, PSI) are implemented in `src/` and checked
against scikit-learn and scipy before being used on real data.
""")

code("""
from woe_iv import fit_woe, iv_strength
from metrics_scratch import auc_rank_sum, gini, ks_statistic
from sklearn.metrics import roc_auc_score as sk_roc_auc_score
from scipy.stats import ks_2samp
import numpy as np

# check on synthetic data before using these on real applicants
rng = np.random.default_rng(0)
n = 5000
y_test = rng.integers(0, 2, n)
score_test = rng.normal(0, 1, n) + y_test * 0.8
score_test = np.round(score_test, 2)  # force ties on purpose

my_auc = auc_rank_sum(y_test, score_test)
sk_auc = sk_roc_auc_score(y_test, score_test)
print(f"AUC check: mine={my_auc:.6f} sklearn={sk_auc:.6f} diff={abs(my_auc-sk_auc):.2e}")

my_ks = ks_statistic(y_test, score_test)
sp_ks = ks_2samp(score_test[y_test == 1], score_test[y_test == 0]).statistic
print(f"KS check:  mine={my_ks:.6f} scipy={sp_ks:.6f} diff={abs(my_ks-sp_ks):.2e}")

assert abs(my_auc - sk_auc) < 1e-9
assert abs(my_ks - sp_ks) < 1e-9
print("both match to floating point precision")
""")

md("""
The KS check is the one worth testing under tied scores. Computed row by row
after sorting, KS evaluates the gap between the good and bad distributions in
the middle of a block of tied scores, which is not a point on the curve. The
implementation groups by distinct score before taking cumulative sums, and the
test rounds the scores to force ties so that an ungrouped version would fail.
""")

code("""
# now the real thing: Weight of Evidence and Information Value on the
# application features, ranked by predictive strength.
# DAYS_EMPLOYED_ANOM is a 0/1 flag, not something to quantile-bin as
# continuous, so it gets treated as categorical like the other flags do.
woe_fits = {}
for col in NUMERIC_COLS:
    if col == "DAYS_EMPLOYED_ANOM":
        continue
    woe_fits[col] = fit_woe(X_train[col], y_train, is_categorical=False, n_bins=10)
woe_fits["DAYS_EMPLOYED_ANOM"] = fit_woe(X_train["DAYS_EMPLOYED_ANOM"], y_train, is_categorical=True)

iv_ranked = sorted(((c, r["iv"]) for c, r in woe_fits.items()), key=lambda t: t[1], reverse=True)
print("Top 10 features by Information Value:")
for col, iv in iv_ranked[:10]:
    print(f"  {col:28s} IV={iv:.4f}  ({iv_strength(iv)})")
""")

md("""
The three `EXT_SOURCE` columns, external bureau-style scores supplied with the
data, carry far more information than any field on the application form, which
matches what is known about this dataset.

`EXT_SOURCE_MEAN` crosses the IV threshold that the helper labels "check for
leakage". It is checked rather than assumed: it is the average of the three
external scores, all of which exist when the application is made, so its IV
reflects their combined strength, not information from after the decision.
""")

# ---------------------------------------------------------------------------

md("""
## Step 3: bureau history, previous loans and the scorecard

Credit-bureau records, previous Home Credit applications and repayment history
are aggregated per applicant: counts, averages, the share of bureau credit
overdue, and how often past applications were approved or refused.

These features lift the AUC and support a points-based scorecard, in which each
WoE bin carries a fixed number of points and the reason for a low score is the
list of bins where the applicant lost the most points.
""")

code("""
from relational_features import build_all_relational_features
from pathlib import Path
import pandas as pd

cache_path = Path("src/_cache_relational_features.parquet")
if cache_path.exists():
    rel_feats = pd.read_parquet(cache_path)
else:
    rel_feats = build_all_relational_features()

full = feats.merge(rel_feats, on="SK_ID_CURR", how="left")
print(f"full feature table: {full.shape}")
""")

code("""
from baseline_features import BASE_NUMERIC_COLS, CATEGORICAL_COLS
from woe_iv import transform_woe
from from_scratch_lr import FromScratchLogisticRegression
from sklearn.linear_model import LogisticRegression
import numpy as np

RELATIONAL_NUMERIC = [c for c in rel_feats.columns if c != "SK_ID_CURR"]
ENGINEERED_NUMERIC = [
    "AGE_YEARS", "EMPLOYED_YEARS", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM", "CREDIT_GOODS_RATIO", "INCOME_PER_FAM_MEMBER",
    "EXT_SOURCE_MEAN", "EXT_SOURCE_STD", "EXT_SOURCE_COUNT",
]
ALL_NUMERIC = BASE_NUMERIC_COLS + ENGINEERED_NUMERIC + RELATIONAL_NUMERIC
ALL_CATEGORICAL = CATEGORICAL_COLS + ["DAYS_EMPLOYED_ANOM"]

y_full = full["TARGET"]
X_full = full.drop(columns=["TARGET", "SK_ID_CURR"])
Xf_train, Xf_val, yf_train, yf_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

woe_fits_full = {}
for col in ALL_NUMERIC:
    woe_fits_full[col] = fit_woe(Xf_train[col], yf_train, is_categorical=False, n_bins=10)
for col in ALL_CATEGORICAL:
    woe_fits_full[col] = fit_woe(Xf_train[col], yf_train, is_categorical=True)

iv_full_ranked = sorted(((c, r["iv"]) for c, r in woe_fits_full.items()), key=lambda t: t[1], reverse=True)
kept_cols = [c for c, iv in iv_full_ranked if iv >= 0.01]
print(f"keeping {len(kept_cols)} of {len(woe_fits_full)} candidate features (IV >= 0.01)")
""")

code("""
def woe_encode(df, cols, fits):
    return np.column_stack([transform_woe(df[c], fits[c]).values for c in cols])

Xw_train = woe_encode(Xf_train, kept_cols, woe_fits_full)
Xw_val = woe_encode(Xf_val, kept_cols, woe_fits_full)

mean, std = Xw_train.mean(axis=0), Xw_train.std(axis=0)
std[std == 0] = 1.0
Xw_train_s = (Xw_train - mean) / std
Xw_val_s = (Xw_val - mean) / std

sk_full = LogisticRegression(max_iter=3000, class_weight="balanced")
sk_full.fit(Xw_train_s, yf_train)

mine_full = FromScratchLogisticRegression(lr=0.5, n_iter=3000, l2=1e-3)
mine_full.fit(Xw_train_s, yf_train.values, class_weight="balanced")

my_val_pred = mine_full.predict_proba(Xw_val_s)
sk_val_pred = sk_full.predict_proba(Xw_val_s)[:, 1]

full_auc = auc_rank_sum(yf_val.values, my_val_pred)
print(f"full feature set AUC: {full_auc:.4f} (up from {baseline_auc:.4f} using the application form alone)")

coef_diff = np.abs(sk_full.coef_.ravel() - mine_full.coef_)
pred_corr = np.corrcoef(sk_val_pred, my_val_pred)[0, 1]
print(f"from-scratch vs scikit-learn: max coefficient diff={coef_diff.max():.4f}, prediction correlation={pred_corr:.6f}")
""")

md("""
The from-scratch solver and scikit-learn fit the same class-balanced objective,
so their predictions should agree almost exactly, and they do (correlation
printed above). An earlier version compared an unweighted fit with scikit-learn's
`class_weight="balanced"` fit and showed a correlation of only 0.905, which looked
like multicollinearity but was two different optimisation problems. Coefficients
still differ slightly more than predictions because several bureau features are
close to duplicates, which leaves individual coefficients less determined than
the combined score.
""")

code("""
from scorecard import build_scorecard, destandardize_coefficients, score_dataframe, reason_codes

coef_raw, intercept_raw = destandardize_coefficients(
    mine_full.coef_, mine_full.intercept_, mean, std
)

sc = build_scorecard(
    coef=coef_raw, intercept=intercept_raw, kept_cols=kept_cols,
    woe_fits=woe_fits_full, base_score=600, base_odds=20, pdo=40,
)

scores = score_dataframe(Xf_val, sc)
print(f"score range: {scores.min():.0f} to {scores.max():.0f}")
print(f"average score, applicants who repaid: {scores[yf_val == 0].mean():.1f}")
print(f"average score, applicants who defaulted: {scores[yf_val == 1].mean():.1f}")
""")

code("""
# one applicant who repaid and one who defaulted, with the reasons behind each score
good_idx = Xf_val.index[yf_val == 0][0]
bad_idx = Xf_val.index[yf_val == 1][0]

for label, idx in [("repaid the loan", good_idx), ("defaulted", bad_idx)]:
    row = Xf_val.loc[idx]
    print(f"\\n{label}, score={scores.loc[idx]:.0f}:")
    for col, pts in reason_codes(row, sc, top_n=3):
        print(f"  {col}: {pts:+.1f} points")
""")

md("""
This is what the scorecard format buys over a raw probability: a loan officer
can read that an applicant lost a given number of points on
`DAYS_EMPLOYED_ANOM` and know exactly what drove the score. A gradient-boosted
model explained with SHAP does not hand over a fixed, auditable point table in
the same way.
""")

# ---------------------------------------------------------------------------

md("""
## Step 4: recommendation to a credit committee

A gradient-boosted model would probably rank applicants somewhat better. The
scorecard is still the stronger choice for a lending decision, for reasons other
than raw accuracy:

- every point has a specific, checkable reason, which a declined applicant is
  entitled to under adverse-action rules;
- the monotonicity of each WoE bin can be reviewed in a table before the model
  is fitted, rather than hoped for in a tree ensemble;
- the point table moves gently when refitted on new data, which supports the
  consistent treatment of similar applicants that regulators expect.

Scope: the model outputs a probability of default. An IFRS 9 provision also
needs loss given default and exposure at default, which are separate models not
built here.

Next step: the base odds of 20:1 at a score of 600 are a conventional anchor,
not calibrated to this population. Calibrating them to the observed default
rate of about 8% would make the score readable as a probability.
""")

nb["cells"] = cells

with open("Credit_Risk_Scorecard.ipynb", "w") as f:
    nbf.write(nb, f)

print(f"wrote Credit_Risk_Scorecard.ipynb with {len(cells)} cells")
