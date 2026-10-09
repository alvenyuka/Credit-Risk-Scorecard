"""Generate Credit_Risk_Scorecard.ipynb.

This script is the source of the notebook: edit it, regenerate, and execute the
notebook. Never edit the .ipynb by hand.

    python build_notebook.py
    jupyter nbconvert --to notebook --execute Credit_Risk_Scorecard.ipynb \
        --output Credit_Risk_Scorecard.ipynb --ExecutePreprocessor.timeout=3600

The notebook is written to be followed and recreated step by step. It is
organised around the three questions a lender asks of a scorecard (which
applicants are likely to default, whom to approve, and how to explain a
decline), with each section broken into small numbered steps of one short code
cell each, and "Check" assertions that stop the run if a step goes wrong. The
step-by-step build is compared with run_pipeline.build_validation_scores(), the
tested function that writes outputs/results.json, and the notebook asserts that
the two produce the same features, coefficients and scores. The last section
asserts that every figure shown equals outputs/results.json.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


# =====================================================================
# TITLE
# =====================================================================
md("""
# Credit Risk Scorecard: from 307,511 applications to an approval policy

**Which loan applicants are likely to default, which of them should a lender approve, and how is a decline
explained to the applicant?**

A lender has to decide, for every loan application, whether to approve it, and if it declines, it has to tell
the applicant why. A **credit scorecard** does both: each fact about the applicant earns a fixed number of
points, the points add up to a score, and the score decides. This notebook builds one from 307,511 real loan
applications (Home Credit's public dataset on Kaggle, with seven linked tables of credit history) and shows
every step, so you can follow and rebuild it.

**How this notebook is organised.** The analysis is driven by the three questions above, so it is organised
around them rather than around a single model:

- **Part 1, what the data can support** (sections 1 to 6): the finished product in miniature, then the
  application table loaded and checked, explored, joined to the applicant's credit history from the other
  seven tables, and turned into candidate characteristics with a holdout kept back. Two facts surface here
  that shape everything later: the three external bureau scores carry far more signal than the application
  form, and several characteristics do not move in one direction, which is why bins are coarse-classed.
- **Part 2, building the scorecard** (sections 7 to 10): the benchmarks it has to beat, Weight of Evidence and
  Information Value by hand and then on every candidate, three selection gates with a stated reason for every
  removal, a logistic regression written from scratch, and the conversion into points.
- **Part 3, validation and use** (sections 11 and 12): ranking, calibration, stability and fair-lending checks
  on the holdout, the price of a readable model against gradient boosting, the approval policy a credit
  committee could choose, and a decline explained.
- **Part 4, limits and record** (sections 13 and 14): what the results cannot show, and a check that every
  figure shown equals the one the tested pipeline wrote to `outputs/results.json`.

Each step is one short code cell with an explanation of what it does and what its output shows. Boxes marked
**Try it** suggest a change to rerun. Cells containing **Check** assertions stop the notebook at the step that
went wrong instead of carrying a wrong number forward. The statistics (WoE, IV, the solver, AUC, KS, PSI) are
written out in `src/` and tested against scikit-learn and scipy; where it matters, the notebook first does the
same calculation by hand and checks that the two agree.
""")

# =====================================================================
# PART 1
# =====================================================================
md("""
---
# Part 1: What the data can support

## 1. Setup

**Step 1.1:** Import the libraries, fix the random seed so every run gives the same numbers, and set the
plotting style.
""")

code("""
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

RUN_STARTED = time.time()   # used for the runtime in the run record
SEED = 42                   # the same seed the pipeline uses for the holdout split
np.random.seed(SEED)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid", context="notebook")
BLUE, RED, INK = "#2b6cb0", "#c0392b", "#2d3748"
os.makedirs("figs", exist_ok=True)
print("Setup OK. pandas:", pd.__version__, "| numpy:", np.__version__)
""")

md("""
**Step 1.2:** Make the project's own functions importable. The statistics the scorecard rests on (Weight of
Evidence, the logistic-regression solver, AUC and KS, the points arithmetic and the fair-lending checks) live
in `src/` rather than in this notebook so that `tests/` can check them; the notebook and the tests therefore run
the same code. Each is imported in the section that uses it.
""")

code("""
sys.path.insert(0, "src")
from feature_lists import (
    BASE_NUMERIC_COLS,       # raw application columns offered as numeric characteristics
    CATEGORICAL_COLS,        # raw application columns binned by level
    ENGINEERED_NUMERIC,      # ratios, ages and external-score summaries built in section 6
    EDA_ONLY_COLS,           # the pensioner flag: explored in section 4, never offered to the model
    RELATIONAL_NUMERIC,      # applicant-level summaries of the credit-history tables (section 5)
    MODEL_NUMERIC,           # every numeric candidate
    MODEL_CATEGORICAL,       # every categorical candidate
    PROHIBITED_BASES,        # sex and marital status: never used
    IV_THRESHOLD,            # gate 1 in section 9
    CORRELATION_THRESHOLD,   # gate 2 in section 9
)
print(f"{len(MODEL_NUMERIC)} numeric and {len(MODEL_CATEGORICAL)} categorical candidate characteristics; "
      f"prohibited bases: {PROHIBITED_BASES}")
""")

# =====================================================================
# 2. MINIATURE
# =====================================================================
md("""
---
## 2. Where we are heading: a scorecard in miniature

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
here are made up for illustration; the real ones are estimated from data in sections 8 to 10.

**Step 2.1:** Score one made-up applicant and decide.
""")

code("""
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

# =====================================================================
# 3. DATA LOADING
# =====================================================================
md("""
---
## 3. Data loading

The eight Home Credit CSVs (2.5 GB) are not stored in the repository: download the
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) data from Kaggle and
put the CSVs in the `data` folder next to this notebook.

**Step 3.1:** Load the main table, `application_train.csv`: one row per application, with the outcome in
`TARGET` (1 = defaulted, 0 = repaid).
""")

code("""
app = pd.read_csv("data/application_train.csv")

print(f"{len(app):,} applications, {app.shape[1]} columns, {app.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory")
app[["SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_2"]].head()
""")

code("""
# Check: one row per applicant, and every applicant is in the file
assert len(app) == 307_511
assert app["SK_ID_CURR"].is_unique
""")

md("""
**Step 3.2:** Home Credit ships a data dictionary. Read it and look up the columns this notebook leans on most.
""")

code("""
dictionary = pd.read_csv("data/HomeCredit_columns_description.csv", index_col=0, encoding="latin-1")
key_columns = ["TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
               "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "NAME_INCOME_TYPE", "OCCUPATION_TYPE"]
application_rows = dictionary["Table"].str.startswith("application") & dictionary["Row"].isin(key_columns)
pd.set_option("display.max_colwidth", 120)
dictionary.loc[application_rows, ["Row", "Description"]].set_index("Row")
""")

md("""
`DAYS_BIRTH` and `DAYS_EMPLOYED` count days *before* the application, so they are negative; `AMT_GOODS_PRICE`
is the price of the goods a consumer loan paid for; and the three `EXT_SOURCE` columns are normalised credit
scores supplied by outside sources, which section 4 shows to be the strongest signal in the data.

### 3.1 Data-quality checks

**Step 3.3:** Which columns have missing values, and how badly?
""")

code("""
missing = app.isna().mean().sort_values(ascending=False)
print(f"columns with any missing value: {(missing > 0).sum()} of {app.shape[1]}; more than half missing: {(missing > 0.5).sum()}")

fig, ax = plt.subplots(figsize=(9, 6))
(missing.head(20) * 100).sort_values().plot(kind="barh", ax=ax, color=BLUE)
ax.set_xlabel("Share of applications missing the value (%)")
ax.set_title("The 20 most incomplete columns")
plt.tight_layout()
plt.savefig("figs/missing_values.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
Sixty-seven of the 122 columns have gaps, and 41 are more than half empty. Almost all of those describe the
applicant's building (common area, number of floors, apartment counts, in three statistical variants each),
which is information most applicants never supplied. None of them is a candidate characteristic.

**Step 3.4:** Look for values that are not what they claim to be. `DAYS_EMPLOYED` counts days before the
application, so it should be negative.
""")

code("""
positive = app["DAYS_EMPLOYED"] > 0
print(f"applicants with DAYS_EMPLOYED > 0: {positive.sum():,}")
print(f"distinct values among them: {app.loc[positive, 'DAYS_EMPLOYED'].unique()}")
""")

md("""
Every one holds the same placeholder, 365243 days (about a thousand years). These are pensioners and unemployed
applicants. Left in, the value would give retirees a millennium of work history, so section 6 turns it into a
missing value and section 4 shows that the gap itself carries information.

**Step 3.5:** Check the money columns for impossible values.
""")

code("""
money = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
print(app[money].describe().round(0).to_string())
print(f"\\napplicants with an annual income above 10 million: {(app['AMT_INCOME_TOTAL'] > 1e7).sum()}, "
      f"the largest {app['AMT_INCOME_TOTAL'].max():,.0f}")
""")

code("""
# Check: every income and credit amount is positive
assert (app["AMT_INCOME_TOTAL"] > 0).all() and (app["AMT_CREDIT"] > 0).all()
""")

md("""
The median applicant earns 147,150 a year and borrows 513,531 (Home Credit does not state the currency). Three
applicants report an income above ten million, one of them 117 million; these are almost certainly data-entry
errors, but they are left in: the scorecard bins each characteristic at its deciles (section 8), so an extreme
value lands in the top bin with everyone else above the ninetieth percentile and cannot pull anything.
""")

# =====================================================================
# 4. EDA
# =====================================================================
md("""
---
## 4. Exploratory data analysis

### 4.1 How common is default?

**Step 4.1:** Count the outcomes.
""")

code("""
outcome = app["TARGET"].value_counts().rename({0: "repaid", 1: "defaulted"})
fig, ax = plt.subplots(figsize=(6, 3.8))
outcome.plot(kind="bar", ax=ax, color=[BLUE, RED], rot=0)
for i, v in enumerate(outcome):
    ax.text(i, v, f"{v:,} ({v / len(app):.1%})", ha="center", va="bottom")
ax.set_ylabel("Applicants")
ax.set_title("Outcome of the 307,511 applications")
plt.tight_layout()
plt.savefig("figs/target_balance.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
About 8% of applicants defaulted. With a target this lopsided, accuracy is useless: approving everyone would be
92% "accurate". What matters is **ranking**: does the score put defaulters below repayers? AUC and KS, defined
in sections 7 and 11, measure exactly that.

### 4.2 Who applies, and who defaults?

**Step 4.2:** Age. Convert `DAYS_BIRTH` to years, compare the age distributions of repayers and defaulters,
and look at the default rate by five-year band.
""")

code("""
age = -app["DAYS_BIRTH"] / 365.25
age_band = pd.cut(age, range(20, 75, 5))
by_age = app.groupby(age_band, observed=True)["TARGET"].agg(applicants="size", default_rate="mean")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].hist(age[app["TARGET"] == 0], bins=25, density=True, alpha=0.6, color=BLUE, label="repaid")
axes[0].hist(age[app["TARGET"] == 1], bins=25, density=True, alpha=0.6, color=RED, label="defaulted")
axes[0].set_xlabel("Age at application (years)")
axes[0].set_ylabel("Density")
axes[0].set_title("Age distribution by outcome")
axes[0].legend()
(by_age["default_rate"] * 100).plot(kind="bar", ax=axes[1], color=RED, rot=45)
axes[1].axhline(app["TARGET"].mean() * 100, color=INK, ls="--", lw=1)
axes[1].set_xlabel("Age band")
axes[1].set_ylabel("Default rate (%)")
axes[1].set_title("Default rate falls with age")
plt.tight_layout()
plt.savefig("figs/age_vs_default.png", dpi=150, bbox_inches="tight")
plt.show()
by_age
""")

md("""
Applicants are 20 to 69 years old, with a median of 43. Defaulters skew young: the default rate falls steadily
from about 12% for applicants under 25 to under 4% for those over 65. That steady, one-direction pattern is what
a scorecard captures. Section 9 nevertheless removes age from the model, for a reason that only shows once the
other characteristics are in.

**Step 4.3:** Income and credit amount, on a log scale because both are right-skewed, and the default rate by
income decile.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, col, title in [(axes[0], "AMT_INCOME_TOTAL", "Annual income"), (axes[1], "AMT_CREDIT", "Credit amount")]:
    ax.hist(np.log10(app.loc[app["TARGET"] == 0, col]), bins=60, density=True, alpha=0.6, color=BLUE, label="repaid")
    ax.hist(np.log10(app.loc[app["TARGET"] == 1, col]), bins=60, density=True, alpha=0.6, color=RED, label="defaulted")
    ax.set_xlabel(f"log10({col})")
    ax.set_ylabel("Density")
    ax.set_title(f"{title} by outcome")
    ax.legend()
plt.tight_layout()
plt.savefig("figs/income_credit_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

income_decile = pd.qcut(app["AMT_INCOME_TOTAL"], 10, duplicates="drop")
app.groupby(income_decile, observed=True)["TARGET"].agg(applicants="size", default_rate="mean").round(4)
""")

md("""
Income on its own says little: the default rate sits between 8% and 9% for the bottom seven deciles and only
falls, to about 6%, in the top two. Credit amount looks the same for both outcomes. What matters more is how the
two relate.

**Step 4.4:** The credit-to-income ratio, by decile.
""")

code("""
credit_to_income = app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"]
ratio_decile = pd.qcut(credit_to_income, 10)
by_ratio = app.groupby(ratio_decile, observed=True)["TARGET"].agg(applicants="size", default_rate="mean")

fig, ax = plt.subplots(figsize=(10, 4))
(by_ratio["default_rate"] * 100).plot(kind="bar", ax=ax, color=RED, rot=45)
ax.set_xticklabels([f"{i.left:.1f} to {i.right:.1f}" for i in by_ratio.index])
ax.set_xlabel("Credit amount as a multiple of annual income (deciles)")
ax.set_ylabel("Default rate (%)")
ax.set_title("Default rate by credit-to-income ratio: a hump, not a line")
plt.tight_layout()
plt.savefig("figs/credit_to_income.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
The default rate rises from about 7% for loans under 1.3 times income to about 9% around three to four times
income, then falls again for the largest multiples (which are mostly long mortgages to well-paid applicants). A
pattern that goes up and then down is a problem for a points table: it is what section 8.1 calls a zig-zag, and
coarse classing is the standard fix.

### 4.3 Default rate by applicant group

**Step 4.5:** The default rate by income type, education, housing and occupation, for groups of at least 500
applicants.
""")

code("""
def default_rate_by_group(col, ax, min_applicants=500):
    table = app.groupby(col, observed=True)["TARGET"].agg(applicants="size", default_rate="mean")
    table = table[table["applicants"] >= min_applicants].sort_values("default_rate")
    (table["default_rate"] * 100).plot(kind="barh", ax=ax, color=BLUE)
    ax.axvline(app["TARGET"].mean() * 100, color=INK, ls="--", lw=1)
    ax.set_xlabel("Default rate (%)")
    ax.set_ylabel("")
    ax.set_title(col)


fig, axes = plt.subplots(2, 2, figsize=(15, 11))
for ax, col in zip(axes.ravel(), ["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]):
    default_rate_by_group(col, ax)
plt.tight_layout()
plt.savefig("figs/default_by_group.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
The dashed line is the overall rate. Pensioners and state servants default at about 5%, working applicants at
nearly 10%. Education runs from about 5% with higher education to 11% with lower secondary. Renting or living
with parents carries roughly 12% against 8% for home owners. Occupation spreads widest: accountants default at
under 5% and low-skill labourers at 17%, with drivers, security and catering staff above 10%. Each of these is
a categorical characteristic in section 8, binned by level.

### 4.4 The external scores

Three columns, `EXT_SOURCE_1` to `EXT_SOURCE_3`, are normalised credit scores from outside sources.

**Step 4.6:** How complete are they, and how do they separate the outcomes?
""")

code("""
ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
print("share missing:", (app[ext_cols].isna().mean() * 100).round(1).to_dict())
print("correlation with default:", app[ext_cols].corrwith(app["TARGET"]).round(3).to_dict())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ext_cols):
    ax.hist(app.loc[app["TARGET"] == 0, col].dropna(), bins=40, density=True, alpha=0.6, color=BLUE, label="repaid")
    ax.hist(app.loc[app["TARGET"] == 1, col].dropna(), bins=40, density=True, alpha=0.6, color=RED, label="defaulted")
    ax.set_title(col)
    ax.set_xlabel("Score (0 = riskiest, 1 = safest)")
    ax.legend()
plt.tight_layout()
plt.savefig("figs/external_scores.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
The second score is present for almost everyone, the third for four applicants in five, and the first for fewer
than half. All three push defaulters to the left, and each has the strongest correlation with the outcome of any
column in the table. Section 6 summarises them into a mean, a spread and a count of scores present.

**Step 4.7:** Before modelling, does that signal hold across the whole range? Split applicants into ten equal
groups by their average external score and look at the default rate in each. (Description only; nothing is
fitted.)
""")

code("""
ext_mean = app[ext_cols].mean(axis=1)
ext_band = pd.qcut(ext_mean, 10)
by_ext = app.groupby(ext_band, observed=True)["TARGET"].agg(applicants="size", default_rate="mean")

fig, ax = plt.subplots(figsize=(10, 4))
(by_ext["default_rate"] * 100).plot(kind="bar", ax=ax, color=RED, rot=45)
ax.set_xticklabels([f"{i.left:.2f} to {i.right:.2f}" for i in by_ext.index])
ax.set_xlabel("Average external score (deciles)")
ax.set_ylabel("Default rate (%)")
ax.set_title("Default rate by average external score")
plt.tight_layout()
plt.savefig("figs/ext_source_deciles.png", dpi=150, bbox_inches="tight")
plt.show()
by_ext.round(4)
""")

md("""
The default rate falls about tenfold from the lowest group to the highest, smoothly. That steady, one-direction
pattern is exactly what a scorecard captures: higher bins earn more points.

### 4.5 Missing values as signal

**Step 4.8:** The pensioner placeholder from Step 3.4 is a missing employment length. Compare the default rate
of applicants with and without one, and the rate by years employed for those who have one.
""")

code("""
employed_years = -app["DAYS_EMPLOYED"].where(~positive, np.nan) / 365.25
by_missing = app.groupby(employed_years.isna())["TARGET"].agg(applicants="size", default_rate="mean").rename(
    index={False: "employment length known", True: "missing (pensioners and unemployed)"})
print(by_missing.round(4).to_string())

employed_band = pd.cut(employed_years, [0, 1, 2, 3, 5, 10, 20, 50])
by_employed = app.groupby(employed_band, observed=True)["TARGET"].agg(applicants="size", default_rate="mean")
fig, ax = plt.subplots(figsize=(9, 4))
(by_employed["default_rate"] * 100).plot(kind="bar", ax=ax, color=RED, rot=0)
ax.axhline(app["TARGET"].mean() * 100, color=INK, ls="--", lw=1)
ax.set_xlabel("Years in current employment")
ax.set_ylabel("Default rate (%)")
ax.set_title("Default rate by employment length (applicants with a known length)")
plt.tight_layout()
plt.savefig("figs/employment_length.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
The missing group defaults *less* often than average (about 5% against 9% for applicants with a known
employment length), so a missing value is information, not a gap to fill with an average. The scorecard gives
missing values a bin of their own. Among those employed, the rate falls from about 11% in the first three years
to about 4% after twenty.

### 4.6 Which raw numbers move with default?

**Step 4.9:** Rank every numeric column of the application table by its correlation with the outcome. A
correlation is a crude measure for a binary target, but it shows where the signal is before any binning.
""")

code("""
numeric = app.select_dtypes("number").drop(columns=["SK_ID_CURR", "TARGET"])
correlation = numeric.corrwith(app["TARGET"]).dropna().sort_values()
extremes = pd.concat([correlation.head(10), correlation.tail(10)])

fig, ax = plt.subplots(figsize=(9, 6))
extremes.plot(kind="barh", ax=ax, color=[BLUE if v < 0 else RED for v in extremes])
ax.set_xlabel("Correlation with default (negative = safer as the value rises)")
ax.set_title("The ten most negative and ten most positive correlations with default")
plt.tight_layout()
plt.savefig("figs/correlation_with_default.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
The three external scores stand alone at about -0.16 to -0.18; nothing on the application form comes close.
`DAYS_BIRTH` is the strongest form field (positive, because the column counts days before the application, so
a larger value means younger). The rest sit below 0.06. The application form, then, is weak on its own, which
is why section 5 adds the applicant's credit history.

**So far:** 307,511 applicants, 8% defaults, one data-quality problem found, the external scores identified as
the main signal, and two characteristics (credit-to-income, and later others) seen to zig-zag.
""")

# =====================================================================
# 5. CREDIT HISTORY
# =====================================================================
md("""
---
## 5. Adding the applicant's credit history

The application form is one table. Home Credit supplies six more, each keyed by the applicant's `SK_ID_CURR`
(or, for the monthly bureau statuses, by the bureau record):

| Table | Rows | What it holds |
|---|---:|---|
| `bureau.csv` | 1.7 million | Credits reported by other lenders to the credit bureau, one row per credit |
| `bureau_balance.csv` | 27.3 million | Monthly status of each bureau credit (current, days past due, closed) |
| `previous_application.csv` | 1.7 million | The applicant's earlier applications to Home Credit, approved or refused |
| `POS_CASH_balance.csv` | 10.0 million | Monthly balances of earlier point-of-sale and cash loans |
| `credit_card_balance.csv` | 3.8 million | Monthly balances and limits of earlier credit cards |
| `installments_payments.csv` | 13.6 million | Every instalment due and what was paid, and when |

Each is read with `pd.read_csv`, only the columns needed, and rolled up to one row per applicant: counts,
shares, averages, worst cases. `src/relational_features.py` holds the same aggregation for the pipeline, and
Step 5.7 checks that the two agree.

**Step 5.1:** The credit bureau. First the monthly statuses: a status of 1 to 5 means some days past due;
`C`, `X` and `0` mean closed, unknown or current. Roll them up to "was this credit ever delinquent" and
"months of history", then join onto the credits and summarise per applicant.
""")

code("""
bureau_balance = pd.read_csv("data/bureau_balance.csv", usecols=["SK_ID_BUREAU", "STATUS"], dtype={"STATUS": "category"})
bureau_balance["delinquent"] = bureau_balance["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("int8")
balance_summary = bureau_balance.groupby("SK_ID_BUREAU").agg(
    BB_MONTHS_COUNT=("STATUS", "count"), BB_EVER_DELINQUENT=("delinquent", "max")).reset_index()
print(f"{len(bureau_balance):,} monthly statuses on {len(balance_summary):,} bureau credits; "
      f"{balance_summary['BB_EVER_DELINQUENT'].mean():.1%} of credits were ever past due")
del bureau_balance

bureau = pd.read_csv("data/bureau.csv", usecols=["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT",
                                                 "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"])
bureau = bureau.merge(balance_summary, on="SK_ID_BUREAU", how="left")
bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
bureau_features = bureau.groupby("SK_ID_CURR").agg(
    BUREAU_COUNT=("SK_ID_BUREAU", "count"),
    BUREAU_ACTIVE_SHARE=("IS_ACTIVE", "mean"),
    BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),               # the oldest credit, in days before the application
    BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
    BUREAU_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
    BUREAU_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
    BUREAU_CREDIT_SUM_TOTAL=("AMT_CREDIT_SUM", "sum"),
    BUREAU_CREDIT_SUM_DEBT_TOTAL=("AMT_CREDIT_SUM_DEBT", "sum"),
    BUREAU_EVER_DELINQUENT_SHARE=("BB_EVER_DELINQUENT", "mean"),
).reset_index()
print(f"{len(bureau):,} bureau credits summarised for {len(bureau_features):,} applicants")
del bureau, balance_summary
""")

md("""
**Step 5.2:** Earlier applications to Home Credit: how many, how many approved or refused, their amounts, and
how recent the last decision was.
""")

code("""
previous = pd.read_csv("data/previous_application.csv", usecols=["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS",
                                                                 "AMT_APPLICATION", "AMT_CREDIT", "DAYS_DECISION"])
print("decisions on earlier applications:", previous["NAME_CONTRACT_STATUS"].value_counts().to_dict())
previous["IS_APPROVED"] = (previous["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
previous["IS_REFUSED"] = (previous["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
previous_features = previous.groupby("SK_ID_CURR").agg(
    PREV_APP_COUNT=("SK_ID_PREV", "count"),
    PREV_APPROVED_SHARE=("IS_APPROVED", "mean"),
    PREV_REFUSED_SHARE=("IS_REFUSED", "mean"),
    PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
    PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
    PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),              # least negative = most recent decision
).reset_index()
print(f"{len(previous):,} earlier applications summarised for {len(previous_features):,} applicants")
del previous
""")

md("""
**Step 5.3:** Point-of-sale and cash loans: days past due, and the instalment counts.
""")

code("""
pos_cash = pd.read_csv("data/POS_CASH_balance.csv", usecols=["SK_ID_CURR", "SK_DPD", "SK_DPD_DEF", "CNT_INSTALMENT"])
pos_features = pos_cash.groupby("SK_ID_CURR").agg(
    POS_DPD_MAX=("SK_DPD", "max"),
    POS_DPD_MEAN=("SK_DPD", "mean"),
    POS_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
    POS_CNT_INSTALMENT_MEAN=("CNT_INSTALMENT", "mean"),
).reset_index()
print(f"{len(pos_cash):,} monthly records summarised for {len(pos_features):,} applicants; "
      f"{(pos_cash['SK_DPD'] > 0).mean():.1%} of months had days past due")
del pos_cash
""")

md("""
**Step 5.4:** Credit cards: balances, utilisation of the limit, and days past due.
""")

code("""
cards = pd.read_csv("data/credit_card_balance.csv", usecols=["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"])
cards["UTILIZATION"] = cards["AMT_BALANCE"] / cards["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
card_features = cards.groupby("SK_ID_CURR").agg(
    CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
    CC_BALANCE_MAX=("AMT_BALANCE", "max"),
    CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
    CC_DPD_MAX=("SK_DPD", "max"),
).reset_index()
print(f"{len(cards):,} monthly card records summarised for {len(card_features):,} applicants")
del cards
""")

md("""
**Step 5.5:** Instalments: the most direct record of whether the applicant paid on time. Both day columns
count days before the application, so payment day minus due day is positive when the payment was late.
""")

code("""
instalments = pd.read_csv("data/installments_payments.csv",
                          usecols=["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"],
                          dtype={"DAYS_INSTALMENT": "float32", "DAYS_ENTRY_PAYMENT": "float32",
                                 "AMT_INSTALMENT": "float32", "AMT_PAYMENT": "float32"})
instalments["DAYS_LATE"] = instalments["DAYS_ENTRY_PAYMENT"] - instalments["DAYS_INSTALMENT"]
instalments["SHORTFALL"] = instalments["AMT_INSTALMENT"] - instalments["AMT_PAYMENT"]
instalment_features = instalments.groupby("SK_ID_CURR").agg(
    INSTAL_COUNT=("AMT_INSTALMENT", "count"),
    INSTAL_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
    INSTAL_DAYS_LATE_MAX=("DAYS_LATE", "max"),
    INSTAL_SHORTFALL_MEAN=("SHORTFALL", "mean"),
    INSTAL_SHORTFALL_SUM=("SHORTFALL", "sum"),
).reset_index()
print(f"{len(instalments):,} instalments summarised for {len(instalment_features):,} applicants; "
      f"{(instalments['DAYS_LATE'] > 0).mean():.1%} were paid late")
del instalments
""")

md("""
**Step 5.6:** Join the five summaries into one history table, one row per applicant who appears in any of
them.
""")

code("""
history = bureau_features
for part in (previous_features, pos_features, card_features, instalment_features):
    history = history.merge(part, on="SK_ID_CURR", how="outer")
print(f"history: {len(history):,} applicants x {history.shape[1] - 1} features")
for label, part in [("a bureau record", bureau_features), ("an earlier Home Credit application", previous_features),
                    ("a credit card", card_features), ("an instalment record", instalment_features)]:
    print(f"applicants with {label}: {app['SK_ID_CURR'].isin(part['SK_ID_CURR']).mean():.1%}")
""")

code("""
# Check: one row per applicant, the 28 features the pipeline defines, and the same values as the pipeline's
# tested, fingerprinted build (src/relational_features.py)
from relational_features import load_relational_features

assert history["SK_ID_CURR"].is_unique and set(history.columns) == {"SK_ID_CURR", *RELATIONAL_NUMERIC}
pipeline_history = load_relational_features(verbose=False)
pd.testing.assert_frame_equal(history.sort_values("SK_ID_CURR").reset_index(drop=True),
                              pipeline_history.sort_values("SK_ID_CURR").reset_index(drop=True),
                              check_dtype=False, check_like=True)
del pipeline_history
print("the plain-pandas history table equals the pipeline's")
""")

md("""
About 86% of applicants have a bureau record and nearly all have an earlier Home Credit application; fewer
than a third have had a credit card. An applicant with no history in a table keeps a missing value for its features, and
section 8 gives missing values a bin of their own.

**Step 5.7:** Does history carry signal? Two quick looks: the default rate by number of bureau credits, and by
the share of earlier applications that were refused.
""")

code("""
with_history = app[["SK_ID_CURR", "TARGET"]].merge(history[["SK_ID_CURR", "BUREAU_COUNT", "PREV_REFUSED_SHARE"]], how="left")
bureau_band = pd.cut(with_history["BUREAU_COUNT"].fillna(0), [-1, 0, 1, 3, 5, 10, 1000], labels=["0", "1", "2-3", "4-5", "6-10", "over 10"])
refused_band = pd.cut(with_history["PREV_REFUSED_SHARE"], [-0.01, 0, 0.25, 0.5, 0.75, 1.0], labels=["none", "up to 25%", "25-50%", "50-75%", "over 75%"])

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
for ax, band, title, xlabel in [(axes[0], bureau_band, "Default rate by number of bureau credits", "Bureau credits on file"),
                                (axes[1], refused_band, "Default rate by share of earlier applications refused", "Share of earlier Home Credit applications refused")]:
    rate = with_history.groupby(band, observed=True)["TARGET"].mean() * 100
    rate.plot(kind="bar", ax=ax, color=RED, rot=0)
    ax.axhline(app["TARGET"].mean() * 100, color=INK, ls="--", lw=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Default rate (%)")
plt.tight_layout()
plt.savefig("figs/history_vs_default.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
Applicants with no bureau record default at about 10%, those with a few credits at 7% to 8%: a thin file is a
risk in itself. The refusal share is stronger still: applicants never refused before default at 7%, those
refused more than half the time at 16% to 18%. These are the "behavioural" characteristics that the application
form cannot supply.
""")

# =====================================================================
# 6. FEATURES AND SAMPLE
# =====================================================================
md("""
---
## 6. Feature engineering and sample design

**Step 6.1:** Build the application characteristics: ages and employment length in years, the placeholder set
to missing (with a flag kept for exploration only), ratios such as credit to income, and a summary of the three
external scores. Only the columns in `src/feature_lists.py` are kept, so every script, the notebook and the tests
work from one definition.
""")

code("""
feats = app.copy()
feats["AGE_YEARS"] = -feats["DAYS_BIRTH"] / 365.25
feats["DAYS_EMPLOYED_ANOM"] = (feats["DAYS_EMPLOYED"] == 365243).astype(int)
feats["EMPLOYED_YEARS"] = -feats["DAYS_EMPLOYED"].where(feats["DAYS_EMPLOYED"] != 365243, np.nan) / 365.25
feats["CREDIT_INCOME_RATIO"] = feats["AMT_CREDIT"] / feats["AMT_INCOME_TOTAL"]
feats["ANNUITY_INCOME_RATIO"] = feats["AMT_ANNUITY"] / feats["AMT_INCOME_TOTAL"]
feats["CREDIT_TERM"] = feats["AMT_ANNUITY"] / feats["AMT_CREDIT"]
feats["CREDIT_GOODS_RATIO"] = feats["AMT_CREDIT"] / feats["AMT_GOODS_PRICE"]
feats["INCOME_PER_FAM_MEMBER"] = feats["AMT_INCOME_TOTAL"] / feats["CNT_FAM_MEMBERS"].replace(0, np.nan)
feats["EXT_SOURCE_MEAN"] = feats[ext_cols].mean(axis=1)
feats["EXT_SOURCE_STD"] = feats[ext_cols].std(axis=1)
feats["EXT_SOURCE_COUNT"] = feats[ext_cols].notna().sum(axis=1)
feats = feats[["TARGET", "SK_ID_CURR"] + BASE_NUMERIC_COLS + CATEGORICAL_COLS + ENGINEERED_NUMERIC + EDA_ONLY_COLS]

print(f"feature table: {feats.shape[0]:,} rows x {feats.shape[1]} columns")
feats[["AGE_YEARS", "EMPLOYED_YEARS", "CREDIT_INCOME_RATIO", "EXT_SOURCE_MEAN"]].describe().round(2)
""")

code("""
# Check: the placeholder is now missing for exactly those 55,374 applicants, sex and marital status never
# enter the feature table, and the table equals the pipeline's tested function
from baseline_features import engineer_baseline

assert (feats["EMPLOYED_YEARS"].isna() == positive).all() and positive.sum() == 55_374
assert not set(PROHIBITED_BASES) & set(feats.columns)
pd.testing.assert_frame_equal(feats, engineer_baseline(app))
""")

md("""
Sex (`CODE_GENDER`) and marital status (`NAME_FAMILY_STATUS`) are **prohibited bases**: US fair-lending law
(the Equal Credit Opportunity Act and Regulation B) forbids using them in a credit decision. They are left out of
every feature list, and `tests/test_fair_lending.py` fails if either comes back.

**Step 6.2:** Join the credit history from section 5. Applicants with no history keep their row.
""")

code("""
full = feats.merge(history, on="SK_ID_CURR", how="left")
print(f"combined table: {full.shape[0]:,} rows x {full.shape[1]} columns "
      f"({len(BASE_NUMERIC_COLS) + len(ENGINEERED_NUMERIC) + len(RELATIONAL_NUMERIC)} numeric and {len(CATEGORICAL_COLS)} categorical candidates)")
""")

code("""
# Check: the join added columns, not rows
assert len(full) == len(feats) and full["SK_ID_CURR"].is_unique
""")

md("""
**Step 6.3:** Hold back 20% of applicants for testing, stratified so both parts have the same default rate.
The model never sees them until section 11.
""")

code("""
y = full["TARGET"]
X = full.drop(columns=["TARGET", "SK_ID_CURR"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"training: {len(X_train):,} applicants, default rate {y_train.mean():.2%}")
print(f"holdout:  {len(X_val):,} applicants, default rate {y_val.mean():.2%}")
""")

code("""
# Check: an 80/20 split with the same default rate on both sides
assert len(X_val) == 61_503 and abs(y_train.mean() - y_val.mean()) < 1e-3
""")

md("""
**So far:** 307,511 applicants, 8% defaults, one data-quality problem fixed, prohibited bases excluded, credit
history added from six tables, 62 candidate characteristics, and 61,503 applicants held back for testing.
""")

# =====================================================================
# PART 2
# =====================================================================
md("""
---
# Part 2: Building the scorecard

## 7. Benchmarks: what the scorecard has to beat

> **What's AUC?** Pick one defaulter and one repayer at random. AUC is the chance the model gives the defaulter
> the riskier score. 0.5 is a coin toss; 1.0 separates every pair.

**Step 7.1:** A model that gives everyone the same score cannot tell anyone apart.
""")

code("""
print(f"AUC of a constant score: {roc_auc_score(y_val, np.zeros(len(y_val))):.4f}")
""")

md("""
**Step 7.2:** A sensible benchmark uses the application form alone: an off-the-shelf logistic regression with
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

# =====================================================================
# 8. WOE
# =====================================================================
md("""
---
## 8. Characteristic analysis: Weight of Evidence

A scorecard does not feed raw values to the model. It cuts each characteristic into bins and replaces each bin
with one number, its **Weight of Evidence (WoE)**:

> WoE of a bin = ln( share of all repayers who are in the bin / share of all defaulters who are in the bin )

Positive WoE: the bin holds relatively more repayers, so it is safer than average. Negative: riskier. Zero:
the bin tells you nothing. **Information Value (IV)** adds up, over the bins, (share of repayers - share of
defaulters) x WoE, and measures how well the whole characteristic separates the two groups.

**Step 8.1:** Work it out on 1,000 made-up applicants in three bins first.
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

**Step 8.2:** Now the same calculation by hand on a real characteristic, `EXT_SOURCE_MEAN`, with ten bins at
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
**Step 8.3:** Read the `woe` column as a picture.
""")

code("""
fig, ax = plt.subplots(figsize=(10, 4))
reference["table"]["woe"].plot(kind="bar", ax=ax, color=[RED if v < 0 else BLUE for v in reference["table"]["woe"]], rot=45)
ax.axhline(0, color=INK, lw=1)
ax.set_xlabel("Bin of EXT_SOURCE_MEAN (training applicants)")
ax.set_ylabel("Weight of Evidence")
ax.set_title("WoE climbs steadily from the riskiest bin to the safest")
plt.tight_layout()
plt.savefig("figs/woe_ext_source_mean.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
The WoE climbs steadily from the lowest band to the highest, the same pattern as Step 4.7, now on a scale the
model can use. Missing scores sit near the riskier end: an applicant with no external score at all is a thin
file, like the applicants with no bureau record in Step 5.7.

### 8.1 When bins zig-zag: coarse classing

Ten equal bins do not always give a pattern that steady. Random noise can make the WoE go up, down, then up
again (Step 4.4 showed a real hump), and a scorecard would then award *fewer* points to a safer bin than to a
riskier one next to it. That is hard to defend to a credit committee and looks arbitrary to an applicant.

The standard fix is **coarse classing**: start from the fine bins and merge neighbours until the default rate
moves in one direction only. `fit_monotone_bins` in `src/woe_iv.py` does it in three steps:

1. Fine bins at the deciles. A column that is mostly zero (most applicants have no overdue credit, for example)
   gets a bin of its own for 0, so its few non-zero values are not lost.
2. Find the overall direction: does the default rate generally rise or fall across the bins?
3. Walk through the bins in order, and whenever a bin goes against the direction, merge it with the bin before
   (the "pool adjacent violators" method).

**Step 8.4:** Find the strongest characteristic whose ten bins zig-zag, and compare it before and after.
""")

code("""
from woe_iv import is_monotonic_woe

decile_fits = {c: fit_woe(X_train[c], y_train, n_bins=10) for c in MODEL_NUMERIC}
zigzag = [c for c in sorted(decile_fits, key=lambda c: -decile_fits[c]["iv"]) if not is_monotonic_woe(decile_fits[c])]
example = zigzag[0]
coarse = fit_woe(X_train[example], y_train, n_bins=10, binning="monotone")
print(f"numeric characteristics whose ten-bin WoE zig-zags: {len(zigzag)} of {len(MODEL_NUMERIC)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
for ax, fit, title in [(axes[0], decile_fits[example], f"{example}: ten bins"), (axes[1], coarse, f"{example}: after coarse classing")]:
    woe = fit["table"]["woe"].drop("Missing", errors="ignore")
    woe.plot(kind="bar", ax=ax, color=[RED if v < 0 else BLUE for v in woe], rot=45)
    ax.axhline(0, color=INK, lw=1)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Weight of Evidence")
plt.tight_layout()
plt.savefig("figs/coarse_classing.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
After merging, the WoE moves one way across the bins, so every bin earns at least as many points as the riskier
bin next to it. The cost is a little detail (fewer bins) and so a little ranking power.

Is that cost worth paying? The project decided it with a rule fixed *before* looking at results:
`src/challengers.py` builds both versions on the training applicants only, scores them on a validation slice
of those applicants, and adopts coarse classing if it costs no more than 0.005 AUC there. Step 8.5 reads the
recorded comparison.
""")

code("""
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

**Step 8.6:** Fit WoE for all 62 candidates: 54 numeric, coarse-classed, and 8 categorical, one bin per
category. Then rank them by information value.
""")

code("""
from woe_iv import iv_strength

woe_fits = {c: fit_woe(X_train[c], y_train, is_categorical=False, n_bins=10, binning="monotone") for c in MODEL_NUMERIC}
woe_fits.update({c: fit_woe(X_train[c], y_train, is_categorical=True) for c in MODEL_CATEGORICAL})
ivs = pd.Series({c: fit["iv"] for c, fit in woe_fits.items()}).sort_values()

fig, ax = plt.subplots(figsize=(9, 13))
ivs.plot(kind="barh", ax=ax, color=[BLUE if v >= IV_THRESHOLD else "#a0aec0" for v in ivs])
ax.axvline(IV_THRESHOLD, color=RED, ls="--", lw=1, label=f"gate 1: IV >= {IV_THRESHOLD}")
ax.axvline(0.1, color=INK, ls=":", lw=1, label="0.10: medium")
ax.set_xlabel("Information value (training applicants)")
ax.set_title(f"All {len(ivs)} candidate characteristics by information value")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figs/iv_all_candidates.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"candidate characteristics: {len(woe_fits)}; strongest: {ivs.index[-1]} (IV {ivs.iloc[-1]:.3f}, {iv_strength(ivs.iloc[-1])})")
""")

code("""
# Check: 62 candidates, the prohibited bases are not among them, and every numeric one is monotonic
assert len(woe_fits) == 62 and not set(PROHIBITED_BASES) & set(woe_fits)
assert all(is_monotonic_woe(woe_fits[c]) for c in MODEL_NUMERIC)
""")

md("""
> **Try it:** in Step 8.4, replace `zigzag[0]` with `zigzag[1]` and compare another characteristic before and
> after coarse classing.

**So far:** every candidate characteristic is binned, each bin has a WoE, every numeric characteristic moves in
one direction, and each characteristic has an IV. The external scores lead by a distance; the grey bars at the
bottom carry no usable signal.
""")

# =====================================================================
# 9. SELECTION
# =====================================================================
md("""
---
## 9. Variable selection: three gates

Not all 62 candidates should enter. Each passes three gates, and each removal has a reason a reviewer can check.

**Step 9.1: Gate 1, information value.** Drop characteristics with IV below 0.01: they carry almost no signal.
""")

code("""
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

**Step 9.2:** One encoding trap. Step 6.1 also made a flag, `DAYS_EMPLOYED_ANOM`, for the pensioner placeholder.
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

**Step 9.3:** Replace every value with its bin's WoE for the characteristics that passed Gate 1.
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
**Step 9.4: Gate 2, correlation.** Two characteristics that move together almost perfectly say the same thing
twice, and the model cannot tell which deserves the weight. Look at the correlations among the 15 strongest
first, then walk down the list in IV order and keep a characteristic only if its correlation with every one
already kept is at most 0.9.
""")

code("""
corr = np.corrcoef(W_iv, rowvar=False)
pos = {c: i for i, c in enumerate(iv_kept)}

fig, ax = plt.subplots(figsize=(11, 9))
top15 = iv_kept[:15]
sns.heatmap(pd.DataFrame(corr[:15, :15], index=top15, columns=top15), vmin=-1, vmax=1, cmap="RdBu_r",
            annot=True, fmt=".2f", square=True, cbar=False, ax=ax, annot_kws={"size": 8})
ax.set_title("Correlation of WoE values among the 15 strongest characteristics")
plt.tight_layout()
plt.savefig("figs/woe_correlation_top15.png", dpi=150, bbox_inches="tight")
plt.show()
""")

code("""
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
The heat map shows why the gate exists: the external-score summaries overlap with the scores they are built
from, and the two region ratings and the credit and goods-price columns are near-duplicates. Each dropped
characteristic is named with the one it duplicates.

**Step 9.5: Gate 3, the direction of each effect.** Higher WoE always means safer. So in a model of the chance
of default, every characteristic's coefficient should be **negative**: more WoE, lower risk. A positive
coefficient means that, once the other characteristics are in the model, this one's effect has flipped, and the
scorecard would hand points to its *riskier* bins. Fit the model, remove the characteristic with the largest
positive coefficient, refit, and repeat until every coefficient is negative.
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
`AGE_YEARS` is among those removed. On its own, older applicants default less (Step 4.2); but once employment
history and income type are in the model, age's coefficient flips. Kept, it would have given the oldest
applicants *negative* points, which Regulation B forbids for applicants aged 62 or over. Removing it, rather
than forcing its sign, keeps every remaining coefficient a genuine estimate.

`class_weight="balanced"` makes the 8% of defaulters count as much as the 92% of repayers while fitting;
Step 10.5 corrects the effect this has on the intercept.

**So far:** the three gates leave the characteristics printed above, each with a reason for being there.
""")

# =====================================================================
# 10. FIT AND SCALE
# =====================================================================
md("""
---
## 10. Model fit and scaling to points

**Step 10.1:** Fit the final logistic regression with the project's own implementation
(`src/from_scratch_lr.py`, plain gradient descent). It treats the weights slightly differently from
scikit-learn, so the sign check runs again on this fit.
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
**Step 10.2:** Every step so far is also packaged in one tested function, `build_validation_scores`
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

md("""
**Step 10.3:** Look at the coefficients. Every one is negative (more WoE, less risk), and their sizes say which
characteristics move the score most once the others are held constant.
""")

code("""
coefficients = pd.Series(scratch.coef_, index=kept_cols).sort_values()
fig, ax = plt.subplots(figsize=(9, 8))
coefficients.plot(kind="barh", ax=ax, color=BLUE)
ax.axvline(0, color=INK, lw=1)
ax.set_xlabel("Coefficient on standardised WoE (all negative: more WoE, lower risk)")
ax.set_title(f"The {len(kept_cols)} characteristics in the scorecard")
plt.tight_layout()
plt.savefig("figs/coefficients.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
### 10.1 From model to points

The model gives the log-odds of default. Three small steps turn that into the points table of section 2.

**Step 10.4:** The model was fitted on scaled WoE (Step 10.1). Convert its coefficients back to plain WoE units.
""")

code("""
coef_raw = scratch.coef_ / std
intercept_balanced = scratch.intercept_ - np.sum(scratch.coef_ * mean / std)

# Check: matches the pipeline's conversion
assert np.allclose(coef_raw, sc["coef_raw"]) and abs(intercept_balanced - sc["intercept_balanced"]) < 1e-12
print(f"intercept on plain WoE: {intercept_balanced:+.4f}")
""")

md("""
**Step 10.5:** Correct the intercept. The balanced weighting fitted the model as if half of all applicants
defaulted. Adding ln(p / (1 - p)) for the real default rate p, minus the same for 50%, moves the predictions back
to the real 8% without changing any coefficient, so the ranking is untouched but the probabilities become
trustworthy.
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
**Step 10.6:** Choose the scale. Two conventions fix it: **600 points means odds of 20 repayers to 1
defaulter**, and **every 40 points doubles the odds** (PDO, "points to double the odds"). Then:

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
**Step 10.7:** The points table for the strongest characteristic, and the points by bin for the three
strongest. These are real versions of the table in section 2.
""")

code("""
top = kept_cols[0]
table = woe_fits[top]["table"][["n", "bad", "woe"]].copy()
table["default_rate"] = table["bad"] / table["n"]
table["points"] = -factor * coef_raw[0] * table["woe"]

# Check: the same points as the pipeline's table
assert np.allclose(table["points"], sc["points_tables"][top])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, c in zip(axes, kept_cols[:3]):
    points = sc["points_tables"][c]
    points.plot(kind="bar", ax=ax, color=[RED if v < 0 else BLUE for v in points], rot=60)
    ax.axhline(0, color=INK, lw=1)
    ax.set_title(c)
    ax.set_xlabel("")
    ax.set_ylabel("Points")
plt.tight_layout()
plt.savefig("figs/points_by_bin.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"{top}: default rate and points by bin")
table.round(3)
""")

md("""
Safer bins (lower default rate) earn more points, bin by bin, because the WoE is monotonic and the coefficient is
negative. A credit committee reviews a table like this for every characteristic before the scorecard goes live;
all of them are in `outputs/scorecard_points.json`.

**Step 10.8:** Score one real holdout applicant by hand: base points plus the points of each of their bins.
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
> **Try it:** change `iloc[0]` to `iloc[1]` in Step 10.8 and score a different applicant.

**So far:** a points table for every characteristic, on a scale where 600 points means 20:1 odds.
""")

# =====================================================================
# PART 3
# =====================================================================
md("""
---
# Part 3: Validation and use

## 11. Validation: does it work, and can it be trusted?

**Step 11.1:** Test the measuring tools before using them. AUC and KS are written from scratch in
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
**Step 11.2:** Measure the scorecard on the holdout.
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
(Step 7.2). KS is 0.381: at the best cut-off, 38 percentage points more of the defaulters than of the repayers
fall below it. Repayers average 594 points and defaulters 542. The from-scratch fit agrees with scikit-learn's to
six decimal places, so the hand-written solver is not the weak link.

**Step 11.3:** Draw both measures: the ROC curve, and the two cumulative distributions whose largest gap is KS.
""")

code("""
holdout_pd = scratch.predict_proba(Xw_val_s)
fpr, tpr, _ = roc_curve(yf_val, holdout_pd)
order = np.argsort(scores.to_numpy())
sorted_scores, sorted_y = scores.to_numpy()[order], yf_val.to_numpy()[order]
cum_defaulters = np.cumsum(sorted_y) / sorted_y.sum()
cum_repayers = np.cumsum(1 - sorted_y) / (1 - sorted_y).sum()
ks_at = int(np.argmax(cum_defaulters - cum_repayers))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(fpr, tpr, color=BLUE, lw=2, label=f"scorecard (AUC {summary['val_auc']:.3f})")
axes[0].plot([0, 1], [0, 1], color="#a0aec0", ls="--", lw=1, label="constant score (AUC 0.5)")
axes[0].set_xlabel("Share of repayers scored below the cut-off")
axes[0].set_ylabel("Share of defaulters scored below the cut-off")
axes[0].set_title("ROC curve on the holdout")
axes[0].legend(loc="lower right")
axes[1].plot(sorted_scores, cum_defaulters, color=RED, lw=2, label="defaulters")
axes[1].plot(sorted_scores, cum_repayers, color=BLUE, lw=2, label="repayers")
axes[1].vlines(sorted_scores[ks_at], cum_repayers[ks_at], cum_defaulters[ks_at], color=INK, ls="--",
               label=f"KS = {cum_defaulters[ks_at] - cum_repayers[ks_at]:.3f} at {sorted_scores[ks_at]:.0f} points")
axes[1].set_xlabel("Scorecard points")
axes[1].set_ylabel("Cumulative share of applicants below the score")
axes[1].set_title("KS: the largest gap between the two cumulative distributions")
axes[1].legend(loc="lower right")
plt.tight_layout()
plt.savefig("figs/roc_and_ks.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
**Step 11.4:** Calibration: does a predicted 5% default rate really mean 5%? Compare predicted and observed
rates in ten groups of predicted probability of default (PD).
""")

code("""
near = summary["observed_odds_near_base_score"]
print(f"odds at 600 points: scorecard {summary['implied_odds_at_base_score']:.1f}:1, "
      f"observed {near['good_to_bad']:.1f}:1 among {near['n']:,} holdout applicants scoring {near['score_band'][0]}-{near['score_band'][1]}")
print(f"mean predicted PD {summary['mean_predicted_pd_val']:.4f} vs observed default rate {summary['val_bad_rate']:.4f}")

calibration = pd.DataFrame(summary["calibration_by_decile"])
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot(calibration["mean_predicted_pd"] * 100, calibration["observed_default_rate"] * 100, marker="o", color=BLUE, label="scorecard, by PD decile")
limit = max(calibration["mean_predicted_pd"].max(), calibration["observed_default_rate"].max()) * 100 * 1.1
ax.plot([0, limit], [0, limit], color="#a0aec0", ls="--", lw=1, label="perfect calibration")
ax.set_xlabel("Mean predicted probability of default (%)")
ax.set_ylabel("Observed default rate (%)")
ax.set_title("Predicted against observed default rate, holdout")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("figs/calibration.png", dpi=150, bbox_inches="tight")
plt.show()
calibration.round(4)
""")

md("""
Predicted and observed default rates sit close together in every group, and the observed odds around 600
points are close to the promised 20:1. That is the intercept correction of Step 10.5 at work.

**Step 11.5:** Stability. The Population Stability Index (PSI) compares the spread of scores in training with
the holdout; below 0.1 is stable, above 0.25 a warning.
""")

code("""
print(f"PSI, training scores vs holdout scores: {summary['psi_train_to_holdout_scores']:.4f}")
""")

md("""
**Step 11.6:** Picture the separation: score distributions of repayers and defaulters, and the default rate by
score band (the same charts the README shows, drawn by `src/make_figures.py`).
""")

code("""
from make_figures import default_rate_by_band, score_distribution

_ = score_distribution(scores, yf_val)
""")

code("""
_ = default_rate_by_band(scores, yf_val)
""")

md("""
**Step 11.7:** What does readability cost? `src/challengers.py` also trained LightGBM, a gradient-boosted
model that cannot produce a points table, on the same 62 candidates, and recorded its holdout AUC.
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

**Step 11.8:** The fair-lending checks.
""")

code("""
age_check = summary["reg_b_age_62_plus"]
print(f"every coefficient negative: {summary['all_coefficients_negative']}")
print(f"AGE_YEARS in the scorecard: {age_check['age_feature_in_scorecard']}; age check passed: {age_check['passed']}")
print(f"points for pensioners and the unemployed (EMPLOYED_YEARS Missing bin): {summary['employed_years_missing_bin_points']:+.1f}")

# Check: the acceptance checks pass and pensioners are not penalised
assert summary["all_coefficients_negative"] and age_check["passed"] and summary["employed_years_missing_bin_points"] > 0
""")

md("""
Pensioners receive positive points, consistent with their lower default rate (Step 4.8), and no applicant aged
62 or over can lose points for age.
""")

# =====================================================================
# 12. STRATEGY
# =====================================================================
md("""
---
## 12. Strategy and adverse action

**Step 12.1:** Turn scores into a lending policy. Approve only the highest-scoring share of the holdout and
measure each policy. Credit loss = credit amount of each approved loan that defaulted x 45% (an assumed loss
given default: the Basel foundation-IRB value for senior unsecured lending).
""")

code("""
from business_impact import lending_policy_table

policy = lending_policy_table(scores.to_numpy(), yf_val.to_numpy(), Xf_val["AMT_CREDIT"].to_numpy())
policy[["approval_rate", "cut_off_score", "default_rate_approved", "credit_loss", "loss_avoided_pct",
        "good_borrowers_declined_pct"]].round(4)
""")

code("""
# Check: the same table as outputs/business_impact.json
published = pd.DataFrame(json.load(open("outputs/business_impact.json", encoding="utf-8"))["policies"])
assert np.allclose(policy["loss_avoided_pct"], published["loss_avoided_pct"])
assert (policy["good_borrowers_declined"] == published["good_borrowers_declined"]).all()
""")

md("""
**Step 12.2:** Draw the trade-off every policy makes: credit loss avoided against good borrowers turned away.
""")

code("""
labels = [f"{r:.0%}" for r in policy["approval_rate"]]
fig, ax = plt.subplots(figsize=(9, 4.5))
width = 0.38
ax.bar(np.arange(len(labels)) - width / 2, policy["loss_avoided_pct"] * 100, width, color=BLUE, label="credit loss avoided")
ax.bar(np.arange(len(labels)) + width / 2, policy["good_borrowers_declined_pct"] * 100, width, color=RED, label="good borrowers declined")
for i, (a, d) in enumerate(zip(policy["loss_avoided_pct"], policy["good_borrowers_declined_pct"])):
    ax.text(i - width / 2, a * 100 + 1, f"{a:.0%}", ha="center", fontsize=9)
    ax.text(i + width / 2, d * 100 + 1, f"{d:.0%}", ha="center", fontsize=9)
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_xlabel("Share of applicants approved (highest scores first)")
ax.set_ylabel("Share (%)")
ax.set_title(f"What each approval policy buys and costs, {len(yf_val):,} held-out applicants")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("figs/policy_tradeoff.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
Each bar pair is a policy a credit committee could choose. Approving the top 80% (a score of 541 or above)
avoids 47% of the credit loss that approving everyone would have caused, while turning away 17% of the
applicants who would have repaid. Tighter policies avoid more loss but decline more good borrowers. Where to
stop depends on the margin earned on a good loan against the loss on a bad one, which this chart lets the
committee price.

**Step 12.3:** Explain a decline. Take a holdout applicant who defaulted and scored below the 80% cut-off, and
list the characteristics where they lost the most points against the best bin, exactly as in section 2.
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
better (Step 11.7): every point has a checkable reason, which a declined applicant is entitled to; every
characteristic moves in one direction and was kept or removed for a stated reason; and the fair-lending
conditions are tested, not asserted.

**Scope.** It estimates the probability of default only. An IFRS 9 provision also needs loss given default and
exposure at default, which need recovery data this dataset does not have. **Next step:** validate on a later
population when application dates are available.
""")

# =====================================================================
# PART 4
# =====================================================================
md("""
---
# Part 4: Limits and record

## 13. Limitations

- **Random, not out-of-time, validation.** The dataset carries no application dates, so the holdout is a
  random 20% rather than a later period, and nothing here shows how the score behaves as the applicant
  population drifts. The PSI of Step 11.5 compares two random halves of one population, which is the easy case.
- **One loss given default, no revenue.** The policy table applies a single 45% LGD and ignores the interest
  earned on good loans, so it shows the loss side of the cut-off decision only. A committee pricing the cut-off
  needs the margin on a good loan too.
- **Probability of default only.** A full IFRS 9 provision also needs loss-given-default and
  exposure-at-default models, and this dataset has no recovery data to build them from.
- **The currency is unstated.** Home Credit does not say what unit the money columns are in, so the credit loss
  figures are in the dataset's own units and cannot be read as a sum of money.
- **The history tables are summarised, not modelled.** Each linked table is rolled up to a handful of counts,
  shares and averages (section 5). Patterns within a history (a borrower who was late, then caught up) are
  invisible to the scorecard.
- **Coarse classing merges by default rate on the training split only.** Bin edges are fixed from the training
  applicants and applied to the holdout; a bin that is monotonic in training can be slightly off in a new
  population. The challenger test (Step 8.5) measured this on a validation slice, once.
- **The sign gate removes, it does not explain.** A characteristic whose coefficient flips is dropped with its
  coefficient recorded, but the notebook does not identify which other characteristic absorbed its effect.
- **Fairness is checked against two prohibited bases and one age rule.** Sex and marital status never enter,
  and applicants aged 62 or over cannot lose points for age; other protected characteristics in the data (none
  are present explicitly, but proxies may be) are not tested.
""")

# =====================================================================
# 14. RESULTS CHECK
# =====================================================================
md("""
---
## 14. Results check and run record

The pipeline (`python src/run_pipeline.py`) writes every headline metric to `outputs/results.json`, with the
package versions and the git commit of the run. The notebook built the same model by hand, so every figure it
shows must equal the one in that file.

**Step 14.1:** Compare every entry.
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

md("""
**Step 14.2:** Record this run: date, runtime, peak memory, code version and library versions, written to
`outputs/notebook_run.json` next to the pipeline's results.
""")

code("""
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def peak_memory_gb():
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)  # Windows
        if peak is None:
            import resource  # Linux and macOS
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return round(peak / 1e9, 1)
    except Exception:
        return None


uncommitted = git("status", "--porcelain", "--", ".", ":(exclude)*.ipynb", ":(exclude)figs", ":(exclude)outputs")
run_record = {
    "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "runtime_minutes": round((time.time() - RUN_STARTED) / 60),
    "peak_memory_gb": peak_memory_gb(),
    "git_commit": git("rev-parse", "HEAD"),
    "uncommitted_code_changes": None if uncommitted is None else bool(uncommitted),
    "python": platform.python_version(),
    "packages": {p: package_version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn")},
    "results_checked_against": "outputs/results.json",
    "entries_checked": len(summary),
    "holdout_auc": float(full_auc),
    "characteristics_kept": len(kept_cols),
}
with open("outputs/notebook_run.json", "w", encoding="utf-8") as f:
    json.dump(run_record, f, indent=2)
print(json.dumps(run_record, indent=2))
""")

md("""
## What you learned

- A scorecard is a points table; the work is in choosing characteristics, bins and points defensibly
  (section 2).
- The application form is weak on its own; the external scores and the applicant's credit history carry the
  signal (sections 4 and 5).
- WoE turns any characteristic into one safe-to-risky scale, and IV measures its strength (section 8).
- Coarse classing makes every characteristic move one way, at a small, measured cost (section 8.1).
- Each characteristic enters or leaves for a stated reason: signal, redundancy, or a flipped effect (section 9).
- Scaling turns log-odds into points, and the intercept correction makes the probabilities trustworthy
  (section 10).
- A scorecard is validated on ranking, calibration, stability and fair lending, and its readability has a price
  that can be measured (section 11).
- The score only matters through the policy it supports and the reasons it gives (section 12).

**Exercises**

1. In Step 9.4, set `CORRELATION_THRESHOLD = 0.8` on the line after the import and rerun to the end of
   section 10. Which characteristics drop out, and what happens to AUC?
2. In Step 10.6, change the scale to 30 points to double the odds. Which numbers in the points table change, and
   does the ranking (AUC) change?
3. In Step 12.1, add a 60% approval rate to the policy table
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

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
with open("Credit_Risk_Scorecard.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote Credit_Risk_Scorecard.ipynb with {len(cells)} cells")
