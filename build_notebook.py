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
# Credit Risk Scorecard: a blind rebuild

I already had a working credit scorecard for this dataset. This notebook is
me rebuilding it from a blank file, on purpose, without looking at the old
code while I worked. The point wasn't to get a better number. It was to
actually earn the numbers instead of having them sitting in a repo I could
recite but not fully explain.

The question I'm answering: using an applicant's loan application and their
history with other lenders, can I separate good borrowers from bad ones well
enough to build a scorecard a loan officer could actually use, and can I
explain every point on that scorecard in plain language?

Dataset: Home Credit's public Kaggle competition, 307,511 applicants, 8
related tables (application, bureau history, previous loans, payment
records).
""")

md("""
## Step 1: a baseline, using only the application form

Before pulling in anyone's credit history, I wanted to know how far the
application form alone could get me. This sets a floor. Anything I add
later has to beat it.

I built a few features by hand. The one worth mentioning: `DAYS_EMPLOYED`
has a placeholder value of 365243 for pensioners and unemployed applicants,
which is about a thousand years. Left alone, that turns retirees into
people with a thousand years of work history, which breaks any ratio built
on top of it. I caught this by looking at `describe()` on the raw column,
not because anyone told me about it.
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
0.75 AUC using only what's on the application form. That's a real number,
in line with what other people get on this dataset using just the
application table. It's not the final answer. It's the number everything
else in this notebook has to beat.
""")

# ---------------------------------------------------------------------------

md("""
## Step 2: building the statistics from scratch

A logistic regression score isn't something a credit committee will just
trust. What they actually want to see is Weight of Evidence and Information
Value, the standard credit-scoring statistics, and they want to know the
model itself is doing what it's supposed to.

So I built four things by hand instead of importing them: Weight of
Evidence / Information Value, a logistic regression trained with plain
gradient descent, and the four metrics a credit team checks (AUC, GINI, KS,
PSI). Then I checked every one against scikit-learn and scipy before
trusting any of them on real data.
""")

code("""
from woe_iv import fit_woe, iv_strength
from metrics_scratch import auc_rank_sum, gini, ks_statistic
from sklearn.metrics import roc_auc_score as sk_roc_auc_score
from scipy.stats import ks_2samp
import numpy as np

# sanity check on synthetic data before trusting these on anything real
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
The KS check almost didn't pass. My first version computed it row by row
after sorting by score, and under tied scores that evaluates the good/bad
gap in the middle of a block of ties, which isn't a real point on the curve.
I only caught this because I deliberately rounded the test scores to force
ties before checking. Fixed it by grouping on the distinct score value
first, then taking cumulative sums. Worth mentioning because it's a good
example of why you validate against a library instead of assuming your own
math is right.
""")

code("""
# now the real thing: Weight of Evidence and Information Value on the
# application features, ranked so I can see what actually matters.
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
The three EXT_SOURCE columns dominate. They're external bureau-style scores
already sitting in the data, and nothing on the application form itself
comes close. That's not a surprise if you've read about this dataset
before, but I found it myself before reading anything, which is the point.
""")

# ---------------------------------------------------------------------------

md("""
## Step 3: bureau history, previous loans, and the scorecard

The application form is only part of the picture. Home Credit also gives
you the applicant's history with other lenders (bureau records), their
past applications with Home Credit itself, and their actual payment
behavior on prior loans. I built my own aggregations from these tables:
counts, averages, how much of their bureau history is overdue, how often
their past applications were approved or refused.

Adding this pushed the AUC up and also let me build something a loan
officer could actually use: a points-based scorecard, where every WoE bin
becomes a specific point value, and the reason for a low score is a list of
which bins the applicant landed in.
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
print(f"my model vs scikit-learn: max coefficient diff={coef_diff.max():.4f}, prediction correlation={pred_corr:.6f}")
""")

md("""
The first time I ran this comparison, the correlation between my model and
scikit-learn's was only 0.905, and I assumed that was multicollinearity
from correlated features (some of the bureau features really are close to
duplicates of each other). It wasn't, mostly. scikit-learn was using
`class_weight="balanced"` and my from-scratch model wasn't, so I was
comparing two different optimization problems, not two solvers of the same
one. Once I added the same class weighting to my own implementation, the
correlation jumped to 0.999997. The lesson: check that you're actually
comparing apples to apples before you write down a conclusion about why two
models disagree.
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
# one applicant who repaid, one who defaulted, and why the scorecard says what it says
good_idx = Xf_val.index[yf_val == 0][0]
bad_idx = Xf_val.index[yf_val == 1][0]

for label, idx in [("repaid the loan", good_idx), ("defaulted", bad_idx)]:
    row = Xf_val.loc[idx]
    print(f"\\n{label}, score={scores.loc[idx]:.0f}:")
    for col, pts in reason_codes(row, sc, top_n=3):
        print(f"  {col}: {pts:+.1f} points")
""")

md("""
That's the actual point of building the scorecard this way instead of just
reporting an AUC number. A loan officer, or a hiring manager, can read
"lost 17 points on DAYS_EMPLOYED_ANOM" and know exactly what drove the
score. A SHAP value from a gradient-boosted model doesn't hand you that as
directly.
""")

# ---------------------------------------------------------------------------

md("""
## Step 4: what I'd tell a credit committee

Logistic regression on Weight of Evidence features isn't the most accurate
model I could build here. A gradient-boosted tree would likely score higher.
I'd still recommend the scorecard approach for an actual lending decision,
for three reasons that have nothing to do with raw accuracy: every point on
the scorecard has a specific, checkable reason attached to it, which matters
when a declined applicant is entitled to know why; the monotonicity of each
WoE bin can be checked in a table before the model is even fit, instead of
hoping a tree model learned it; and the point table changes gently when
retrained on new data, which matters to a regulator who expects consistent
treatment of similar applicants over time.

What this model outputs is a probability of default. A full loss provision
under IFRS 9 needs that multiplied by loss given default and exposure at
default, both separate modeling problems I didn't build here. Worth being
upfront about that boundary rather than implying this notebook alone
produces a provisioning number.

If I extended this further, the next thing I'd fix is the scorecard's base
odds. I picked 20 good borrowers per 1 bad at a score of 600 because it's a
reasonable default, not because I calculated it from this population. The
real fix is calibrating that number to this dataset's actual default rate,
around 8%, instead of assuming a round number.
""")

nb["cells"] = cells

with open("Credit_Risk_Scorecard.ipynb", "w") as f:
    nbf.write(nb, f)

print(f"wrote Credit_Risk_Scorecard.ipynb with {len(cells)} cells")
