"""
Generates Credit_Risk_Scorecard.ipynb cell by cell. Edit this file, not the
notebook directly, then regenerate:

    python build_notebook.py
    jupyter nbconvert --to notebook --execute Credit_Risk_Scorecard.ipynb \
        --output Credit_Risk_Scorecard.ipynb --ExecutePreprocessor.timeout=3600

The notebook teaches how a credit scorecard is built, for a reader who has not
built one: it shows the finished product first (a toy points table), then
introduces each idea on a small made-up example before applying it to the
307,511 real applicants, in the order a scorecard is developed and reviewed.

The step-by-step build is compared with run_pipeline.build_validation_scores(),
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
md("""
# Building a Credit Scorecard, Step by Step

A lender has to decide, for every loan application, whether to approve it, and if it declines, it has to tell
the applicant why. A **credit scorecard** does both: each fact about the applicant earns a fixed number of
points, the points add up to a score, and the score decides. This notebook builds one from 307,511 real loan
applications (Home Credit's public dataset) and shows every step, so you can follow and rebuild it.

**What you will be able to do by the end**

1. Explain what a scorecard is and read a points table.
2. Turn a characteristic such as income or employment length into **Weight of Evidence**, and measure its
   strength with **Information Value**.
3. Choose which characteristics enter the scorecard, with a reason a reviewer can check for every one.
4. Fit a logistic regression and convert it into points.
5. Check a scorecard the way a validation team would: ranking, calibration, stability and fair lending.
6. Turn scores into an approval policy and explain an individual decline.

**Before you start.** Put the eight Home Credit CSVs from Kaggle in `data/`, and run
`pip install -r requirements.txt`. The notebook takes about 15 minutes. Some helper functions are imported from
`src/` (each is tested in `tests/`); where it matters, the notebook first does the same calculation by hand and
checks that the two agree.

**How to read it.** Each step says what we are about to do and why, runs one short cell, and says what the
output shows. Boxes marked **Try it** suggest a change to rerun. Cells containing **Check** stop the notebook if
a step went wrong.
""")

md("""
## 0. Where we are heading: a scorecard in miniature

Before any data, here is the finished product in miniature. Suppose a scorecard used only three facts. Every
range ("bin") of every fact carries points:

| Characteristic | Bin | Points |
|---|---|---:|
| External credit score | below 0.40 | 5 |
| | 0.40 to 0.60 | 25 |
| | above 0.60 | 45 |
| Years in current job | under 2 | 10 |
| | 2 or more | 30 |
| | not employed / pensioner | 28 |
| Previous applications refused | none | 30 |
| | some | 12 |

Every applicant starts with **base points** (say 480) and adds one row from each characteristic. The points
here are made up for illustration; the real ones are estimated from data in sections 3 to 6.

**Step 0.1:** Score one made-up applicant and decide.
""")

code("""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "src")          # the project's tested helper functions
pd.set_option("display.width", 120)

toy_points = {"external score 0.40 to 0.60": 25, "2 or more years in job": 30, "some refusals": 12}
toy_score = 480 + sum(toy_points.values())
cut_off = 540
print(f"score = 480 + {' + '.join(str(p) for p in toy_points.values())} = {toy_score}")
print("decision:", "approve" if toy_score >= cut_off else "decline")
best = {"external score 0.40 to 0.60": 45, "2 or more years in job": 30, "some refusals": 30}
lost = {k: best[k] - v for k, v in toy_points.items() if best[k] > v}
print("points below the best bin, the reasons a decline would cite:", lost)
""")

md("""
That is the whole idea. The applicant scores 547 and is approved. Had they scored below 540, the reasons for the
decline would be the characteristics where they lost the most points against the best bin: here 20 points on the
external score and 18 on refusals. Everything below is about doing this properly: which characteristics, which
bins, how many points, and how to prove the scorecard works.
""")

# ---------------------------------------------------------------------------
md("""
---
## 1. Data and sample design

**Step 1.1:** Load the main table, `application_train.csv`: one row per application, with the outcome in
`TARGET` (1 = defaulted, 0 = repaid).
""")

code("""
from io_raw import load_application

app = load_application("train")
print(f"rows: {len(app):,}   columns: {app.shape[1]}")
app[["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "DAYS_EMPLOYED"]].head()
""")

code("""
# Check: one row per applicant, and every applicant is in the file
assert len(app) == 307_511
assert app["SK_ID_CURR"].is_unique
""")

md("""
**Step 1.2:** How common is default?
""")

code("""
app["TARGET"].value_counts(normalize=True).rename({0: "repaid", 1: "defaulted"})
""")

md("""
About 8% of applicants defaulted. With a target this lopsided, accuracy is useless: approving everyone would be
92% "accurate". What matters is **ranking**: does the score put defaulters below repayers? AUC and KS, defined
in sections 2 and 7, measure exactly that.

**Step 1.3:** Look for values that are not what they claim to be. `DAYS_EMPLOYED` counts days before the
application, so it should be negative.
""")

code("""
positive = app["DAYS_EMPLOYED"] > 0
print(f"applicants with DAYS_EMPLOYED > 0: {positive.sum():,}")
print(f"distinct values among them: {app.loc[positive, 'DAYS_EMPLOYED'].unique()}")
""")

md("""
Every one holds the same placeholder, 365243 days (about a thousand years). These are pensioners and unemployed
applicants. Left in, the value would give retirees a millennium of work history, so the next step turns it into
a missing value.

**Step 1.4:** Build the application features with `engineer_baseline` (in `src/baseline_features.py`): ages and
employment length in years, the placeholder set to missing, ratios such as credit to income, and a summary of
three external credit scores supplied with the data.
""")

code("""
from baseline_features import engineer_baseline
from feature_lists import PROHIBITED_BASES

feats = engineer_baseline(app)
print(f"feature table: {feats.shape[0]:,} rows x {feats.shape[1]} columns")
feats[["AGE_YEARS", "EMPLOYED_YEARS", "CREDIT_INCOME_RATIO", "EXT_SOURCE_MEAN"]].describe().round(2)
""")

code("""
# Check: the placeholder is now missing for exactly those 55,374 applicants,
# and sex and marital status never enter the feature table
assert (feats["EMPLOYED_YEARS"].isna() == positive).all() and positive.sum() == 55_374
assert not set(PROHIBITED_BASES) & set(feats.columns)
""")

md("""
Sex (`CODE_GENDER`) and marital status (`NAME_FAMILY_STATUS`) are **prohibited bases**: US fair-lending law
(the Equal Credit Opportunity Act and Regulation B) forbids using them in a credit decision. They are left out of
every feature list, and `tests/test_fair_lending.py` fails if either comes back.

**Step 1.5:** Add each applicant's history from the other seven tables: credit-bureau records, previous Home
Credit applications and repayment behaviour, summarised to one row per applicant (counts, averages, the share of
bureau credit overdue, how often past applications were refused). The summary reads about 2.5 GB, so it is
cached; the cache is checked against a fingerprint of the code and data.
""")

code("""
from relational_features import load_relational_features

rel = load_relational_features()
full = feats.merge(rel, on="SK_ID_CURR", how="left")   # applicants with no history keep their row
print(f"history: {rel.shape[1] - 1} features   combined table: {full.shape[0]:,} rows x {full.shape[1]} columns")
""")

code("""
# Check: the join added columns, not rows
assert len(full) == len(feats) and full["SK_ID_CURR"].is_unique
""")

md("""
**Step 1.6:** Before modelling, does the data hold signal at all? Split applicants into ten equal groups by
their average external credit score and look at the default rate in each. (Description only; nothing is fitted.)
""")

code("""
band = pd.qcut(full["EXT_SOURCE_MEAN"], 10)
full.groupby(band, observed=True)["TARGET"].agg(default_rate="mean", applicants="size").round(4)
""")

md("""
The default rate falls about tenfold from the lowest group to the highest, smoothly. That steady, one-direction
pattern is exactly what a scorecard captures: higher bins earn more points.

**Step 1.7:** Missing values can carry signal too. Compare applicants with and without an employment length.
""")

code("""
full.groupby(full["EMPLOYED_YEARS"].isna())["TARGET"].mean().rename(
    {False: "employment length known", True: "missing (pensioners and unemployed)"})
""")

md("""
The missing group defaults *less* often than average, so a missing value is information, not a gap to fill
with an average. The scorecard gives missing values a bin of their own.

**Step 1.8:** Hold back 20% of applicants for testing, stratified so both parts have the same default rate. The
model never sees them until section 7.
""")

code("""
y = full["TARGET"]
X = full.drop(columns=["TARGET", "SK_ID_CURR"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"training: {len(X_train):,} applicants, default rate {y_train.mean():.2%}")
print(f"holdout:  {len(X_val):,} applicants, default rate {y_val.mean():.2%}")
""")

code("""
# Check: an 80/20 split with the same default rate on both sides
assert len(X_val) == 61_503 and abs(y_train.mean() - y_val.mean()) < 1e-3
""")

md("""
**So far:** 307,511 applicants, 8% defaults, one data-quality problem fixed, prohibited bases excluded, credit
history added, and 61,503 applicants held back for testing.
""")

# ---------------------------------------------------------------------------
md("""
---
## 2. Benchmarks: what the scorecard has to beat

> **What's AUC?** Pick one defaulter and one repayer at random. AUC is the chance the model gives the defaulter
> the riskier score. 0.5 is a coin toss; 1.0 separates every pair.

**Step 2.1:** A model that gives everyone the same score cannot tell anyone apart.
""")

code("""
print(f"AUC of a constant score: {roc_auc_score(y_val, np.zeros(len(y_val))):.4f}")
""")

md("""
**Step 2.2:** A sensible benchmark uses the application form alone: an off-the-shelf logistic regression with
missing values filled and categories encoded (`build_pipeline` in `src/baseline_model.py`).
""")

code("""
from baseline_model import build_pipeline

pipe = build_pipeline()
pipe.fit(X_train, y_train)      # the pipeline picks its own application-form columns by name
baseline_auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
print(f"application-only benchmark AUC: {baseline_auc:.4f}")
""")

md("""
That is the bar. The scorecard adds credit history and must beat it while staying readable.
""")

# ---------------------------------------------------------------------------
md("""
---
## 3. Characteristic analysis: Weight of Evidence

A scorecard does not feed raw values to the model. It cuts each characteristic into bins and replaces each bin
with one number, its **Weight of Evidence (WoE)**:

> WoE of a bin = ln( share of all repayers who are in the bin / share of all defaulters who are in the bin )

Positive WoE: the bin holds relatively more repayers, so it is safer than average. Negative: riskier. Zero:
the bin tells you nothing. **Information Value (IV)** adds up, over the bins, (share of repayers - share of
defaulters) x WoE, and measures how well the whole characteristic separates the two groups.

**Step 3.1:** Work it out on 1,000 made-up applicants in three bins first.
""")

code("""
toy = pd.DataFrame({"bin": ["low", "middle", "high"],
                    "repaid": [200, 400, 320],         # 920 repayers in total
                    "defaulted": [50, 25, 5]})         # 80 defaulters in total
toy["share_of_repayers"] = toy["repaid"] / toy["repaid"].sum()
toy["share_of_defaulters"] = toy["defaulted"] / toy["defaulted"].sum()
toy["woe"] = np.log(toy["share_of_repayers"] / toy["share_of_defaulters"])
toy["iv_part"] = (toy["share_of_repayers"] - toy["share_of_defaulters"]) * toy["woe"]
print(toy.round(3).to_string(index=False))
print(f"\\nInformation Value: {toy['iv_part'].sum():.3f}")
""")

md("""
The "low" bin holds 22% of repayers but 62% of defaulters, so its WoE is strongly negative: risky. The "high" bin
holds 35% of repayers and only 6% of defaulters, so its WoE is strongly positive: safe. The IV printed is very
high, as you would expect from numbers chosen to separate well. On real data, an IV below 0.02 is useless, 0.1 to
0.3 medium, 0.3 to 0.5 strong, and above 0.5 suspicious (check for leakage).

**Step 3.2:** Now the same calculation by hand on a real characteristic, `EXT_SOURCE_MEAN`, with ten bins at
its deciles. A small constant (0.5) is added to every count so that an empty bin never gives ln(0).
""")

code("""
col = "EXT_SOURCE_MEAN"
x = X_train[col]

edges = np.unique(np.quantile(x.dropna().astype(float), np.linspace(0, 1, 11)))   # decile cut points
edges[0], edges[-1] = -np.inf, np.inf                                            # open first and last bins
bins = pd.cut(x, bins=edges, include_lowest=True).astype(str).where(x.notna(), "Missing")

woe_table = pd.DataFrame({"bin": bins, "y": y_train}).groupby("bin")["y"].agg(n="count", bad="sum")
woe_table["good"] = woe_table["n"] - woe_table["bad"]
eps, k = 0.5, len(woe_table)
dist_good = (woe_table["good"] + eps) / (woe_table["good"].sum() + eps * k)
dist_bad = (woe_table["bad"] + eps) / (woe_table["bad"].sum() + eps * k)
woe_table["woe"] = np.log(dist_good / dist_bad)
iv_by_hand = ((dist_good - dist_bad) * woe_table["woe"]).sum()
print(f"IV of {col}, by hand: {iv_by_hand:.4f}")
""")

code("""
from woe_iv import fit_woe

reference = fit_woe(X_train[col], y_train, is_categorical=False, n_bins=10)

# Check: the hand calculation and the tested function agree on every bin
assert np.allclose(woe_table.loc[reference["table"].index, "woe"], reference["table"]["woe"])
assert abs(iv_by_hand - reference["iv"]) < 1e-12
reference["table"][["n", "bad", "woe"]].round(4)
""")

md("""
Read down the `woe` column: it climbs steadily from the lowest band to the highest, the same pattern as Step 1.6,
now on a scale the model can use.

### 3.3 When bins zig-zag: coarse classing

Ten equal bins do not always give a pattern that steady. Random noise can make the WoE go up, down, then up
again, and a scorecard would then award *fewer* points to a safer bin than to a riskier one next to it. That is
hard to defend to a credit committee and looks arbitrary to an applicant.

The standard fix is **coarse classing**: start from the fine bins and merge neighbours until the default rate
moves in one direction only. `fit_monotone_bins` in `src/woe_iv.py` does it in three steps:

1. Fine bins at the deciles. A column that is mostly zero (most applicants have no overdue credit, for example)
   gets a bin of its own for 0, so its few non-zero values are not lost.
2. Find the overall direction: does the default rate generally rise or fall across the bins?
3. Walk through the bins in order, and whenever a bin goes against the direction, merge it with the bin before
   (the "pool adjacent violators" method).

**Step 3.3:** Find the strongest characteristic whose ten bins zig-zag, and compare it before and after.
""")

code("""
from feature_lists import IV_THRESHOLD, MODEL_CATEGORICAL, MODEL_NUMERIC
from woe_iv import is_monotonic_woe

decile_fits = {c: fit_woe(X_train[c], y_train, n_bins=10) for c in MODEL_NUMERIC}
zigzag = [c for c in sorted(decile_fits, key=lambda c: -decile_fits[c]["iv"]) if not is_monotonic_woe(decile_fits[c])]
example = zigzag[0]
coarse = fit_woe(X_train[example], y_train, n_bins=10, binning="monotone")

print(f"numeric characteristics whose ten-bin WoE zig-zags: {len(zigzag)} of {len(MODEL_NUMERIC)}")
print(f"\\n{example}, ten bins (WoE):")
print(decile_fits[example]["table"]["woe"].round(3).to_string())
print(f"\\n{example}, after coarse classing (WoE):")
print(coarse["table"]["woe"].round(3).to_string())
""")

md("""
After merging, the WoE moves one way across the bins, so every bin earns at least as many points as the riskier
bin next to it. The cost is a little detail (fewer bins) and so a little ranking power.

Is that cost worth paying? The project decided it with a rule fixed *before* looking at results:
`src/challengers.py` builds both versions on the training applicants only, scores them on a validation slice
of those applicants, and adopts coarse classing if it costs no more than 0.005 AUC there. Step 3.4 reads the
recorded comparison.
""")

code("""
import json

challengers = json.load(open("outputs/challengers.json", encoding="utf-8"))
v = challengers["validation"]
print(f"validation AUC, ten bins:        {v['champion']['auc']:.4f}   "
      f"({v['champion']['n_numeric_monotonic']} of {v['champion']['n_numeric_kept']} numeric characteristics monotonic)")
print(f"validation AUC, coarse classing: {v['coarse_classing']['auc']:.4f}   "
      f"({v['coarse_classing']['n_numeric_monotonic']} of {v['coarse_classing']['n_numeric_kept']} monotonic)")
print(f"cost: {challengers['coarse_classing_auc_loss_validation']:.4f} AUC   decision: {challengers['decision']}")
""")

md("""
Coarse classing costs about 0.003 AUC and makes every numeric characteristic monotonic, so it was adopted. The
rest of the notebook (and the published scorecard) uses it.

**Step 3.5:** Fit WoE for all 62 candidates: 54 numeric, coarse-classed, and 8 categorical, one bin per category.
""")

code("""
woe_fits = {c: fit_woe(X_train[c], y_train, is_categorical=False, n_bins=10, binning="monotone") for c in MODEL_NUMERIC}
woe_fits.update({c: fit_woe(X_train[c], y_train, is_categorical=True) for c in MODEL_CATEGORICAL})
ivs = {c: fit["iv"] for c, fit in woe_fits.items()}
print(f"candidate characteristics: {len(woe_fits)}")
""")

code("""
# Check: 62 candidates, the prohibited bases are not among them, and every numeric one is monotonic
assert len(woe_fits) == 62 and not set(PROHIBITED_BASES) & set(woe_fits)
assert all(is_monotonic_woe(woe_fits[c]) for c in MODEL_NUMERIC)
""")

md("""
> **Try it:** in Step 3.3, replace `zigzag[0]` with `zigzag[1]` and compare another characteristic before and
> after coarse classing.

**So far:** every candidate characteristic is binned, each bin has a WoE, every numeric characteristic moves in
one direction, and each characteristic has an IV.
""")

# ---------------------------------------------------------------------------
md("""
---
## 4. Variable selection: three gates

Not all 62 candidates should enter. Each passes three gates, and each removal has a reason a reviewer can check.

**Step 4.1: Gate 1, information value.** Drop characteristics with IV below 0.01: they carry almost no signal.
""")

code("""
from woe_iv import iv_strength

iv_ranked = sorted(woe_fits, key=lambda c: (-ivs[c], c))   # highest IV first; ties broken by name
iv_kept = [c for c in iv_ranked if ivs[c] >= IV_THRESHOLD]
print(f"Gate 1: {len(iv_kept)} of {len(iv_ranked)} characteristics have IV >= {IV_THRESHOLD}\\n")
for c in iv_ranked[:12]:
    print(f"  {c:30s} IV={ivs[c]:.4f}  ({iv_strength(ivs[c])})")
""")

md("""
The three external scores lead by far. `EXT_SOURCE_MEAN` is above the "check for leakage" line, so it was
checked: it is the average of three scores that all exist when the application is made, so its strength is real,
not information from after the decision.

**Step 4.2:** One encoding trap. Step 1.4 also made a flag, `DAYS_EMPLOYED_ANOM`, for the pensioner placeholder.
Compare it with the `Missing` bin of `EMPLOYED_YEARS`.
""")

code("""
flag_fit = fit_woe(X_train["DAYS_EMPLOYED_ANOM"], y_train, is_categorical=True)
emp_table = woe_fits["EMPLOYED_YEARS"]["table"]
print(f"DAYS_EMPLOYED_ANOM = 1:    n={flag_fit['table'].loc['1', 'n']:,}  WoE={flag_fit['table'].loc['1', 'woe']:+.4f}")
print(f"EMPLOYED_YEARS = Missing:  n={emp_table.loc['Missing', 'n']:,}  WoE={emp_table.loc['Missing', 'woe']:+.4f}")
print(f"same applicants in both:   {(X_train['DAYS_EMPLOYED_ANOM'] == 1).equals(X_train['EMPLOYED_YEARS'].isna())}")
""")

md("""
Same applicants, same WoE. Given both, the model could give them opposite signs, and one of them would then
appear to *penalise* pensioners, a lower-risk group, which is a wrong reason to give a declined applicant and a
fair-lending problem. So the flag is never offered to the model; pensioners are read only through
`EMPLOYED_YEARS`.

**Step 4.3:** Replace every value with its bin's WoE for the characteristics that passed Gate 1.
""")

code("""
from woe_iv import transform_woe


def woe_encode(frame, cols):
    # One column per characteristic: each value replaced by the WoE of its bin
    return np.column_stack([transform_woe(frame[c], woe_fits[c]).to_numpy(dtype=float) for c in cols])


W_iv = woe_encode(X_train, iv_kept)
print(f"WoE matrix: {W_iv.shape[0]:,} applicants x {W_iv.shape[1]} characteristics")
""")

md("""
**Step 4.4: Gate 2, correlation.** Two characteristics that move together almost perfectly say the same thing
twice, and the model cannot tell which deserves the weight. Walk down the list in IV order and keep a
characteristic only if its correlation with every one already kept is at most 0.9.
""")

code("""
from feature_lists import CORRELATION_THRESHOLD

corr = np.corrcoef(W_iv, rowvar=False)
pos = {c: i for i, c in enumerate(iv_kept)}
corr_kept, corr_dropped = [], []
for c in iv_kept:
    partner = next((k for k in corr_kept if abs(corr[pos[c], pos[k]]) > CORRELATION_THRESHOLD), None)
    if partner is None:
        corr_kept.append(c)
    else:
        corr_dropped.append((c, partner, float(corr[pos[c], pos[partner]])))

print(f"Gate 2: {len(corr_dropped)} dropped, {len(corr_kept)} kept")
for c, partner, r in corr_dropped:
    print(f"  dropped {c:28s} kept {partner:28s} r={r:+.3f}")
""")

md("""
**Step 4.5: Gate 3, the direction of each effect.** Higher WoE always means safer. So in a model of the chance of
default, every characteristic's coefficient should be **negative**: more WoE, lower risk. A positive coefficient
means that, once the other characteristics are in the model, this one's effect has flipped, and the scorecard
would hand points to its *riskier* bins. Fit the model, remove the characteristic with the largest positive
coefficient, refit, and repeat until every coefficient is negative.
""")

code("""
W_corr = W_iv[:, [pos[c] for c in corr_kept]]
m, s = W_corr.mean(axis=0), W_corr.std(axis=0)
s[s == 0] = 1.0
Ws = (W_corr - m) / s                       # put every column on the same scale for the solver

sign_kept, sign_removed = list(corr_kept), []
while True:
    cols_now = [corr_kept.index(c) for c in sign_kept]
    fit = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Ws[:, cols_now], y_train)
    wrong = [(c, float(b)) for c, b in zip(sign_kept, fit.coef_.ravel()) if b > 0]
    if not wrong:
        break
    victim = max(wrong, key=lambda t: (t[1], t[0]))     # the most strongly flipped characteristic
    sign_removed.append(victim)
    sign_kept.remove(victim[0])

print(f"Gate 3: {len(sign_removed)} removed, {len(sign_kept)} kept")
for c, b in sign_removed:
    print(f"  removed {c:28s} coefficient={b:+.4f}")
""")

md("""
`AGE_YEARS` is among those removed. On its own, older applicants default less; but once employment history and
income type are in the model, age's coefficient flips. Kept, it would have given the oldest applicants
*negative* points, which Regulation B forbids for applicants aged 62 or over. Removing it, rather than forcing
its sign, keeps every remaining coefficient a genuine estimate.

`class_weight="balanced"` makes the 8% of defaulters count as much as the 92% of repayers while fitting; Step 6.2
corrects the effect this has on the intercept.

**So far:** the three gates leave the characteristics printed above, each with a reason for being there.
""")

# ---------------------------------------------------------------------------
md("""
---
## 5. Model fit

**Step 5.1:** Fit the final logistic regression with the project's own implementation (`src/from_scratch_lr.py`,
plain gradient descent). It treats the weights slightly differently from scikit-learn, so the sign check runs
again on this fit.
""")

code("""
from from_scratch_lr import FromScratchLogisticRegression
from metrics_scratch import auc_rank_sum

kept_cols = list(sign_kept)
while True:
    Xw_train, Xw_val = woe_encode(X_train, kept_cols), woe_encode(X_val, kept_cols)
    mean, std = Xw_train.mean(axis=0), Xw_train.std(axis=0)
    std[std == 0] = 1.0
    Xw_train_s, Xw_val_s = (Xw_train - mean) / std, (Xw_val - mean) / std   # scaled on training statistics

    scratch = FromScratchLogisticRegression(lr=0.5, n_iter=3000, l2=1e-3)
    scratch.fit(Xw_train_s, y_train.to_numpy(), class_weight="balanced")
    wrong = [(c, float(b)) for c, b in zip(kept_cols, scratch.coef_) if b > 0]
    if not wrong:
        break
    victim = max(wrong, key=lambda t: (t[1], t[0]))
    sign_removed.append(victim)
    kept_cols.remove(victim[0])

full_auc = auc_rank_sum(y_val, scratch.predict_proba(Xw_val_s))
print(f"characteristics in the scorecard: {len(kept_cols)} of {len(woe_fits)}")
print(f"holdout AUC: {full_auc:.4f}  (application-only benchmark: {baseline_auc:.4f})")
""")

code("""
# Check: it beats the benchmark, every sign is right, and the solver converged
assert full_auc > baseline_auc
assert (scratch.coef_ < 0).all() and scratch.converged_
""")

md("""
**Step 5.2:** Every step so far is also packaged in one tested function, `build_validation_scores`
(`src/run_pipeline.py`), which writes the published results. Run it and confirm it builds the same model.
""")

code("""
from run_pipeline import build_validation_scores, summarise

result, sc, Xf_val, yf_val, scores = build_validation_scores(full, verbose=False)
""")

code("""
# Check: the step-by-step build and the pipeline agree exactly
assert result["kept_cols"] == kept_cols
assert [d[:2] for d in result["correlation_gate_dropped"]] == [d[:2] for d in corr_dropped]
assert [r[0] for r in result["sign_gate_removed"]] == [r[0] for r in sign_removed]
assert np.allclose(result["model"].coef_, scratch.coef_)
assert abs(result["val_auc"] - full_auc) < 1e-12
print("step-by-step build matches the pipeline: same characteristics, coefficients and AUC")
""")

# ---------------------------------------------------------------------------
md("""
---
## 6. Scaling: from model to points

The model gives the log-odds of default. Three small steps turn that into the points table of section 0.

**Step 6.1:** The model was fitted on scaled WoE (Step 5.1). Convert its coefficients back to plain WoE units.
""")

code("""
coef_raw = scratch.coef_ / std
intercept_balanced = scratch.intercept_ - np.sum(scratch.coef_ * mean / std)

# Check: matches the pipeline's conversion
assert np.allclose(coef_raw, sc["coef_raw"]) and abs(intercept_balanced - sc["intercept_balanced"]) < 1e-12
print(f"intercept on plain WoE: {intercept_balanced:+.4f}")
""")

md("""
**Step 6.2:** Correct the intercept. The balanced weighting fitted the model as if half of all applicants
defaulted. Adding ln(p / (1 - p)) for the real default rate p, minus the same for 50%, moves the predictions back
to the real 8% without changing any coefficient, so the ranking is untouched but the probabilities become honest.
""")

code("""
def logit(p):
    return np.log(p / (1 - p))


intercept = intercept_balanced + logit(y_train.mean()) - logit(0.5)

# Check: matches the pipeline
assert abs(intercept - sc["intercept"]) < 1e-12
print(f"training default rate {y_train.mean():.4f}: intercept {intercept_balanced:+.4f} -> {intercept:+.4f}")
""")

md("""
**Step 6.3:** Choose the scale. Two conventions fix it: **600 points means odds of 20 repayers to 1 defaulter**,
and **every 40 points doubles the odds** (PDO, "points to double the odds"). Then:

- factor = 40 / ln(2), about 57.7 points per unit of log-odds
- offset = 600 - factor x ln(20)
- points for a bin = -factor x coefficient x WoE of the bin
- score = offset - factor x intercept (the base points) + the points of the applicant's bins
""")

code("""
factor = 40 / np.log(2)
offset = 600 - factor * np.log(20)
base_points = offset - factor * intercept

# Check: the pipeline uses the same scale
assert np.isclose(factor, sc["factor"]) and np.isclose(offset, sc["offset"]) and np.isclose(base_points, sc["base_points"])
print(f"factor {factor:.2f}   offset {offset:.1f}   base points {base_points:.1f}")
""")

md("""
**Step 6.4:** The points table for the strongest characteristic. This is a real version of the table in
section 0.
""")

code("""
top = kept_cols[0]
table = woe_fits[top]["table"][["n", "bad", "woe"]].copy()
table["default_rate"] = table["bad"] / table["n"]
table["points"] = -factor * coef_raw[0] * table["woe"]

# Check: the same points as the pipeline's table
assert np.allclose(table["points"], sc["points_tables"][top])
print(f"{top}: default rate and points by bin")
table.round(3)
""")

md("""
Safer bins (lower default rate) earn more points, bin by bin, because the WoE is monotonic and the coefficient is
negative. A credit committee reviews a table like this for every characteristic before the scorecard goes live;
all of them are in `outputs/scorecard_points.json`.

**Step 6.5:** Score one real holdout applicant by hand: base points plus the points of each of their bins.
""")

code("""
from woe_iv import _bin_labels

applicant = Xf_val.iloc[0]
total = base_points
for c in kept_cols:
    fit = woe_fits[c]
    label = _bin_labels(pd.Series([applicant[c]]), fit["is_categorical"], fit["edges"]).iloc[0]
    total += sc["points_tables"][c].get(label, 0.0)   # a category never seen in training scores 0

# Check: the hand total equals the pipeline's score for this applicant
assert np.isclose(min(max(total, 300), 850), scores.iloc[0])
print(f"applicant {applicant.name}: {total:.1f} points")
""")

md("""
> **Try it:** change `iloc[0]` to `iloc[1]` in Step 6.5 and score a different applicant.

**So far:** a points table for every characteristic, on a scale where 600 points means 20:1 odds.
""")

# ---------------------------------------------------------------------------
md("""
---
## 7. Validation: does it work, and can it be trusted?

**Step 7.1:** Test the measuring tools before using them. AUC and KS are written from scratch in
`src/metrics_scratch.py`; check them against scikit-learn and scipy on made-up scores with ties.

> **What's KS?** Sort applicants by score. KS is the largest gap between the share of defaulters and the share of
> repayers below any cut-off: the credit industry's standard measure of separation.
""")

code("""
from scipy.stats import ks_2samp

from metrics_scratch import ks_statistic

rng = np.random.default_rng(0)
y_check = rng.integers(0, 2, 5000)
score_check = np.round(rng.normal(0, 1, 5000) + y_check * 0.8, 2)     # rounding creates tied scores

print(f"AUC: from scratch {auc_rank_sum(y_check, score_check):.6f}, scikit-learn {roc_auc_score(y_check, score_check):.6f}")
scipy_ks = ks_2samp(score_check[y_check == 1], score_check[y_check == 0]).statistic
print(f"KS:  from scratch {ks_statistic(y_check, score_check):.6f}, scipy {scipy_ks:.6f}")

# Check: both agree to floating-point precision
assert abs(auc_rank_sum(y_check, score_check) - roc_auc_score(y_check, score_check)) < 1e-9
assert abs(ks_statistic(y_check, score_check) - scipy_ks) < 1e-9
""")

md("""
**Step 7.2:** Measure the scorecard on the holdout.
""")

code("""
summary = summarise(result, sc, Xf_val, yf_val, scores)
lo, hi = summary["val_auc_ci95_bootstrap"]
print(f"AUC  {summary['val_auc']:.4f}  (95% bootstrap interval {lo:.4f} to {hi:.4f})")
print(f"KS   {summary['val_ks']:.4f}")
print(f"average score, repaid: {summary['mean_score_repaid']:.1f}   defaulted: {summary['mean_score_defaulted']:.1f}")
print(f"from-scratch fit vs scikit-learn: prediction correlation {result['pred_corr']:.6f}")
""")

md("""
The scorecard reaches an AUC of 0.754: given one defaulter and one repayer, it scores the defaulter lower
about three times in four. The bootstrap interval (0.7473 to 0.7613) shows how far that figure could move with a
different sample of 61,503 applicants; even its lower end is above the application-only benchmark of 0.7467
(Step 2.2). KS is 0.381: at the best cut-off, 38 percentage points more of the defaulters than of the repayers
fall below it. Repayers average 594 points and defaulters 542. The from-scratch fit agrees with scikit-learn's to
six decimal places, so the hand-written solver is not the weak link.

**Step 7.3:** Calibration: does a predicted 5% default rate really mean 5%? Compare predicted and observed rates
in ten groups of predicted probability of default (PD).
""")

code("""
near = summary["observed_odds_near_base_score"]
print(f"odds at 600 points: scorecard {summary['implied_odds_at_base_score']:.1f}:1, "
      f"observed {near['good_to_bad']:.1f}:1 among {near['n']:,} holdout applicants scoring {near['score_band'][0]}-{near['score_band'][1]}")
print(f"mean predicted PD {summary['mean_predicted_pd_val']:.4f} vs observed default rate {summary['val_bad_rate']:.4f}\\n")
pd.DataFrame(summary["calibration_by_decile"]).round(4)
""")

md("""
Predicted and observed default rates sit close together in every group, and the observed odds around 600
points are close to the promised 20:1. That is the intercept correction of Step 6.2 at work.

**Step 7.4:** Stability. The Population Stability Index (PSI) compares the spread of scores in training with the
holdout; below 0.1 is stable, above 0.25 a warning.
""")

code("""
print(f"PSI, training scores vs holdout scores: {summary['psi_train_to_holdout_scores']:.4f}")
""")

md("""
**Step 7.5:** Picture the separation: score distributions of repayers and defaulters, and the default rate by
score band.
""")

code("""
from make_figures import default_rate_by_band, score_distribution

_ = score_distribution(scores, yf_val)
""")

code("""
_ = default_rate_by_band(scores, yf_val)
""")

md("""
**Step 7.6:** What does readability cost? `src/challengers.py` also trained LightGBM, a gradient-boosted model
that cannot produce a points table, on the same 62 candidates, and recorded its holdout AUC.
""")

code("""
h = challengers["holdout"]
print(f"holdout AUC: scorecard {summary['val_auc']:.4f}, LightGBM {h['lightgbm']['auc']:.4f}")
print(f"price of a readable, reason-giving model: {h['lightgbm']['auc'] - summary['val_auc']:.4f} AUC")
""")

md("""
LightGBM ranks applicants better, by about 0.03 AUC. A lender gives that up for a model where every point has a
reason, every characteristic moves in one direction, and a declined applicant can be told why. Whether that is
the right trade is a business and regulatory decision; this measures its size.

**Step 7.7:** The fair-lending checks.
""")

code("""
age = summary["reg_b_age_62_plus"]
print(f"every coefficient negative: {summary['all_coefficients_negative']}")
print(f"AGE_YEARS in the scorecard: {age['age_feature_in_scorecard']}; age check passed: {age['passed']}")
print(f"points for pensioners and the unemployed (EMPLOYED_YEARS Missing bin): {summary['employed_years_missing_bin_points']:+.1f}")

# Check: the acceptance checks pass and pensioners are not penalised
assert summary["all_coefficients_negative"] and age["passed"] and summary["employed_years_missing_bin_points"] > 0
""")

md("""
Pensioners receive positive points, consistent with their lower default rate (Step 1.7), and no applicant aged 62
or over can lose points for age.
""")

# ---------------------------------------------------------------------------
md("""
---
## 8. Strategy and adverse action

**Step 8.1:** Turn scores into a lending policy. Approve only the highest-scoring share of the holdout and
measure each policy. Credit loss = credit amount of each approved loan that defaulted x 45% (an assumed loss
given default: the Basel foundation-IRB value for senior unsecured lending).
""")

code("""
from business_impact import lending_policy_table

policy = lending_policy_table(scores.to_numpy(), yf_val.to_numpy(), Xf_val["AMT_CREDIT"].to_numpy())
policy[["approval_rate", "cut_off_score", "default_rate_approved", "loss_avoided_pct",
        "good_borrowers_declined_pct"]].round(4)
""")

code("""
# Check: the same table as outputs/business_impact.json
published = pd.DataFrame(json.load(open("outputs/business_impact.json", encoding="utf-8"))["policies"])
assert np.allclose(policy["loss_avoided_pct"], published["loss_avoided_pct"])
assert (policy["good_borrowers_declined"] == published["good_borrowers_declined"]).all()
""")

md("""
Each row is a policy a credit committee could choose. Approving the top 80% (a score of 541 or above) avoids
47% of the credit loss that approving everyone would have caused, while turning away 17% of the applicants who
would have repaid. Tighter policies avoid more loss but decline more good borrowers. Where to stop depends on the
margin earned on a good loan against the loss on a bad one, which this table lets the committee price.

**Step 8.2:** Explain a decline. Take a holdout applicant who defaulted and scored below the 80% cut-off, and
list the characteristics where they lost the most points against the best bin, exactly as in section 0.
""")

code("""
ex = summary["example_declined_applicant"]
print(f"applicant scoring {ex['score']:.0f}, below the 80% approval cut-off of {ex['cut_off_score_at_80pct_approval']:.0f}. Reasons:")
for r in ex["reasons_points_below_best_bin"]:
    print(f"  {r['feature']}: {r['points']:.1f} points below the best bin")
""")

md("""
A characteristic where the applicant already scored the maximum is never a reason. This is what the scorecard
format buys over a raw probability: fixed, auditable reasons.

### Recommendation to a credit committee

The scorecard is the right tool for this lending decision even though a gradient-boosted model ranks applicants
better (Step 7.6): every point has a checkable reason, which a declined applicant is entitled to; every
characteristic moves in one direction and was kept or removed for a stated reason; and the fair-lending
conditions are tested, not asserted.

**Scope.** It estimates the probability of default only. An IFRS 9 provision also needs loss given default and
exposure at default, which need recovery data this dataset does not have. **Next step:** validate on a later
population when application dates are available.

## What you learned

- A scorecard is a points table; the work is in choosing characteristics, bins and points defensibly (section 0).
- WoE turns any characteristic into one safe-to-risky scale, and IV measures its strength (section 3).
- Coarse classing makes every characteristic move one way, at a small, measured cost (section 3.3).
- Each characteristic enters or leaves for a stated reason: signal, redundancy, or a flipped effect (section 4).
- Scaling turns log-odds into points, and the intercept correction makes the probabilities honest (section 6).
- A scorecard is validated on ranking, calibration, stability and fair lending, and its readability has a price
  that can be measured (section 7).

**Exercises**

1. In Step 4.4, set `CORRELATION_THRESHOLD = 0.8` on the line after the import and rerun to the end of section 5.
   Which characteristics drop out, and what happens to AUC?
2. In Step 6.3, change the scale to 30 points to double the odds. Which numbers in the points table change, and
   does the ranking (AUC) change?
3. In Step 8.1, add a 60% approval rate to the policy table
   (`lending_policy_table(..., approval_rates=(1.0, 0.9, 0.8, 0.7, 0.6))`). Is the extra loss avoided worth the
   good borrowers turned away?

## Glossary

- **Bin:** a range of a characteristic (for example, 2 to 5 years in the current job).
- **WoE (Weight of Evidence):** ln(share of repayers in a bin / share of defaulters in the bin); positive = safer.
- **IV (Information Value):** how strongly a characteristic separates repayers from defaulters.
- **Coarse classing:** merging neighbouring bins so the default rate moves in one direction.
- **AUC / KS:** measures of how well the score ranks defaulters below repayers.
- **PD:** probability of default. **PDO:** points to double the odds.
- **PSI:** population stability index, how much the score distribution has shifted.
- **Prohibited basis:** a characteristic the law forbids in a credit decision (here sex and marital status).
""")

md("""
**Final check:** every figure in this notebook must equal the one the pipeline wrote to `outputs/results.json`.
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
