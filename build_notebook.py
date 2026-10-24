"""Generate Credit_Risk_Scorecard.ipynb.

This script is the source of the notebook: edit it, regenerate, and execute the
notebook. Never edit the .ipynb by hand.

    python build_notebook.py
    jupyter nbconvert --to notebook --execute Credit_Risk_Scorecard.ipynb \
        --output Credit_Risk_Scorecard.ipynb --ExecutePreprocessor.timeout=3600

The notebook is a guided build that a junior data scientist can follow cell by
cell. It has three parts (Prepare Data, Build Model, Communicate Results), and
each part is a list of numbered steps. Every step has one short instruction, one
"Concept" line that explains the idea behind it, one small code cell with a
comment above each line, a printout of what changed, and a "Check your work"
cell whose assertions stop the run if the step went wrong. The expected values
in those assertions come from an executed run.

The tested logic stays in src/. The notebook loads the raw CSVs with plain
pd.read_csv, walks through each calculation by hand, then calls the src/
function that does the same job and asserts the two agree. The last steps
assert that every figure the notebook shows equals outputs/results.json, which
the tested pipeline (src/run_pipeline.py) wrote.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    # Two markdown blocks in a row become one cell: the reading of a result and the next step's heading
    if cells and cells[-1]["cell_type"] == "markdown":
        cells[-1]["source"] = cells[-1]["source"] + "\n\n" + text.strip()
    else:
        cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


def step(number, title, instruction, concept_question, concept_answer):
    # A step heading: what to do and why, then the idea behind it in one or two sentences
    md(f"""
### Step {number}: {title}

{instruction.strip()}

**Concept: {concept_question}** {concept_answer.strip()}
""")


# =====================================================================
# OPENING
# =====================================================================
md("""
# Credit Risk Scorecard: from 307,511 applications to an approval policy

**The business question.** A lender has to decide, for every loan application, whether to approve it. If it
declines, it has to tell the applicant why. Which applicants are likely to default, which should the lender
approve, and how is a decline explained?

**The data.** Home Credit's public Default Risk dataset on Kaggle: 307,511 past applications with a known
outcome (`TARGET` = 1 if the applicant defaulted), plus six tables of each applicant's credit history. Each row
is a loan Home Credit granted, so applicants it refused are not in the data (the Limitations come back to this).

**What you will build.** A **credit scorecard**: a points table where each fact about an applicant earns a fixed
number of points, the points add up to a score, and the score decides. You will build it one small step at a
time, the way a credit-risk analyst at a bank would, and check every step before moving on.

**The finished result.** On 61,503 applicants held back from fitting, the scorecard ranks risk with an
**AUC of 0.754** and a **KS of 0.381**, using **32 of 62** candidate characteristics. If the lender approves only
the top-scoring **80%**, it avoids **47%** of the credit losses it would suffer by approving every loan in this book, at the cost
of turning away **17.2%** of the applicants who would have repaid. Step 3.7 checks each of these figures against
`outputs/results.json` and `outputs/business_impact.json`.

**How to read this notebook.**

- **Part 1, Prepare Data:** import the tables, explore them, add the credit history, build the candidate
  characteristics and split off a holdout.
- **Part 2, Build Model:** set a baseline, then build the scorecard (Weight of Evidence, three selection gates, a
  logistic regression, points), then evaluate it on the holdout.
- **Part 3, Communicate Results:** turn the score into an approval policy, explain a decline, list what the
  results cannot show, and record every decision.

Each step has a short instruction, a **Concept** line, one small code cell with a comment above each line, and a
**Check your work** cell. If a check fails, the notebook stops at the step that went wrong instead of carrying a
wrong number forward. Run the cells in order from the top; the whole notebook takes about 11 minutes.
""")

# =====================================================================
# PART 1: PREPARE DATA
# =====================================================================
md("""
---
# Part 1: Prepare Data

## 1A. Import

Before any analysis: load the libraries, see the finished product in miniature, then read the main table and
check its quality.
""")

step("1.1", "Import the libraries",
     "Load pandas and NumPy for tables and arrays, Matplotlib and seaborn for charts, and the few scikit-learn "
     "tools used as references.",
     "why import everything at the top?",
     "A reader can see in one place every library the project depends on, and a missing library fails here, "
     "not halfway through an eleven-minute run.")

code("""
# Standard library: reading JSON files, paths, the clock
import json
import os
import sys
# time: the clock that measures how long the whole notebook takes
import time

# Charts
import matplotlib.pyplot as plt
import seaborn as sns

# Arrays and tables
import numpy as np
import pandas as pd

# scikit-learn: a reference logistic regression, reference metrics, and the train/holdout split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# Print the versions so a reader can match them with requirements.txt
print("pandas", pd.__version__, "| numpy", np.__version__)
""")

step("1.2", "Fix the random seed and the chart style",
     "Set one seed for every random step and one look for every chart.",
     "what is a random seed?",
     "Splitting the data and bootstrapping use random numbers. Fixing the seed makes the computer draw the same "
     "numbers on every run, so every figure in this notebook can be reproduced exactly.")

code("""
# Start the clock for the run record written at the end
RUN_STARTED = time.time()
# The same seed the pipeline uses for the holdout split
SEED = 42
# Make NumPy's random draws repeatable
np.random.seed(SEED)
# Show wide tables without wrapping
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)
# One chart style for the whole notebook
sns.set_theme(style="whitegrid", context="notebook")
# Three colours: blue for repaid or safe, red for defaulted or risky, dark grey for reference lines
BLUE, RED, INK = "#2b6cb0", "#c0392b", "#2d3748"
# Charts are saved here so the README can show them
os.makedirs("figs", exist_ok=True)
print("seed:", SEED, "| figures folder exists:", os.path.isdir("figs"))
""")

code("""
# Check your work
assert SEED == 42, f"expected seed 42, got {SEED}"
assert os.path.isdir("figs"), "the figs folder was not created"
""")

step("1.3", "Write the colour rule once",
     "Many charts colour a bar red when its value is below zero (risky) and blue otherwise. Write that rule once "
     "as a small function so every chart uses the same one.",
     "why a function for something so small?",
     "A rule written once cannot drift: if the colours ever change, they change everywhere.")

code("""
# A helper for every WoE chart: one colour per bar
def bar_colours(values):
    # Red for a negative value (riskier than average), blue for zero or above
    colours = []
    for value in values:
        if value < 0:
            colours.append(RED)
        else:
            colours.append(BLUE)
    return colours


# Try it on three values
example_colours = bar_colours([-1.2, 0.0, 0.8])
print(type(example_colours).__name__, len(example_colours), example_colours)
""")

code("""
# Check your work
assert example_colours == [RED, BLUE, BLUE], f"unexpected colours {example_colours}"
""")

step("1.4", "Make the project's own functions importable",
     "The statistics the scorecard rests on live in `src/` so that `tests/` can check them. Add `src/` to "
     "Python's search path and import the lists of candidate characteristics.",
     "why keep logic in `src/` and not in the notebook?",
     "Code in `src/` is tested on every change (`python -m pytest`), and the pipeline that writes the published "
     "results runs the same code. The notebook walks through each calculation by hand, then asserts that the "
     "tested function gives the same answer.")

code("""
# Let Python find the modules in src/
sys.path.insert(0, "src")

# The candidate characteristics and the two selection thresholds, defined once for the whole project
from feature_lists import (
    BASE_NUMERIC_COLS,       # raw application columns offered as numeric characteristics
    CATEGORICAL_COLS,        # raw application columns binned by level
    ENGINEERED_NUMERIC,      # ratios, ages and external-score summaries built in Steps 1.33 to 1.35
    EDA_ONLY_COLS,           # the pensioner flag: explored, never offered to the model
    RELATIONAL_NUMERIC,      # applicant-level summaries of the credit-history tables
    MODEL_NUMERIC,           # every numeric candidate
    MODEL_CATEGORICAL,       # every categorical candidate
    PROHIBITED_BASES,        # sex and marital status: never used
    IV_THRESHOLD,            # gate 1 in Step 2.14
    CORRELATION_THRESHOLD,   # gate 2 in Step 2.18
)

# Print how many numeric and categorical candidates there are, the excluded bases and the two gate thresholds
print(type(MODEL_NUMERIC).__name__, "of", len(MODEL_NUMERIC), "numeric candidates;",
      len(MODEL_CATEGORICAL), "categorical candidates")
print("prohibited bases:", PROHIBITED_BASES)
print("IV threshold:", IV_THRESHOLD, "| correlation threshold:", CORRELATION_THRESHOLD)
""")

code("""
# Check your work
assert len(MODEL_NUMERIC) == 54, f"expected 54 numeric candidates, got {len(MODEL_NUMERIC)}"
assert len(MODEL_CATEGORICAL) == 8, f"expected 8 categorical candidates, got {len(MODEL_CATEGORICAL)}"
assert PROHIBITED_BASES == ["CODE_GENDER", "NAME_FAMILY_STATUS"], PROHIBITED_BASES
""")

md("""
**Prohibited bases.** Sex (`CODE_GENDER`) and marital status (`NAME_FAMILY_STATUS`) are prohibited bases: US
fair-lending law (the Equal Credit Opportunity Act and its Regulation B, 12 CFR 1002) forbids using them in a credit
decision. They are left out of every feature list above, and `tests/test_fair_lending.py` fails if either comes
back.
""")

step("1.5", "Score a made-up applicant with a three-line scorecard",
     "Before touching data, see the finished product in miniature. Suppose a scorecard used only three facts, "
     "each split into ranges (\"bins\") that carry points. Every applicant starts with base points (say 480) and "
     "adds one row from each fact. The points below are made up for illustration; Part 2 estimates the real ones.",
     "what is a credit scorecard?",
     "A table of points, one row per bin of each characteristic. The applicant's score is the base points plus the "
     "points of their bins, and a cut-off score decides approval.")

md("""
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
""")

code("""
# The applicant's bins and the points each one earns
toy_points = {"external score 0.40 to 0.60": 25, "2 or more years in job": 30, "some refusals": 12}
# Every applicant starts from the same base points
toy_base = 480
# Add up the points
toy_score = toy_base
for toy_bin in toy_points:
    toy_score = toy_score + toy_points[toy_bin]
# Approve at 540 points or more
toy_cut_off = 540
if toy_score >= toy_cut_off:
    toy_decision = "approve"
else:
    toy_decision = "decline"
print("score:", toy_score, "| decision:", toy_decision)
""")

code("""
# Check your work
assert toy_score == 547, f"expected 547 points, got {toy_score}"
assert toy_decision == "approve"
""")

step("1.6", "Find the reasons a decline would give",
     "If this applicant had been declined, the lender would cite the characteristics where they lost the most "
     "points against the best bin.",
     "what is a reason code?",
     "For each characteristic, the gap between the best bin's points and the applicant's points. The biggest gaps "
     "are the reasons for a decline, and they are fixed and checkable, which a raw probability cannot give.")

code("""
# The best points available in each characteristic
toy_best = {"external score 0.40 to 0.60": 45, "2 or more years in job": 30, "some refusals": 30}
# Points lost against the best bin, for each characteristic where something was lost
toy_reasons = {}
for toy_bin in toy_points:
    gap = toy_best[toy_bin] - toy_points[toy_bin]
    if gap > 0:
        toy_reasons[toy_bin] = gap
print(type(toy_reasons).__name__, "with", len(toy_reasons), "reasons:", toy_reasons)
""")

code("""
# Check your work
assert toy_reasons == {"external score 0.40 to 0.60": 20, "some refusals": 18}, toy_reasons
""")

md("""
The applicant scores 547 and is approved. Had they scored below 540, the reasons would be 20 points lost on the
external score and 18 on refusals. A characteristic where they already had the best bin (years in job) is never a
reason. Everything below is about doing this properly: which characteristics, which bins, how many points, and how
to prove the scorecard works.
""")

step("1.7", "Read the main table",
     "Download the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) "
     "CSVs from Kaggle into `data/` (2.5 GB, not stored in the repository). Read `application_train.csv` with a "
     "plain `pd.read_csv`.",
     "what is the target?",
     "`TARGET` is the outcome the model learns to predict: 1 if the applicant later had payment difficulties "
     "(here called a default), 0 if they repaid.")

code("""
# One row per application, the outcome in TARGET
app = pd.read_csv("data/application_train.csv")
print(type(app).__name__, app.shape)
print(f"{app.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory")
""")

code("""
# Look at the first rows of the columns this notebook leans on
app[["SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_2"]].head()
""")

code("""
# Check your work
assert app.shape == (307_511, 122), f"expected 307,511 rows and 122 columns, got {app.shape}"
assert app["SK_ID_CURR"].is_unique, "an applicant appears twice"
""")

step("1.8", "Look up the columns in the data dictionary",
     "Home Credit ships a data dictionary. Read it and look up the ten columns used most.",
     "why read the dictionary?",
     "A column name can mislead. The dictionary tells you, for example, that the `DAYS_` columns count days "
     "before the application and are therefore negative.")

code("""
# The dictionary is in Latin-1 encoding, one row per column of each table
dictionary = pd.read_csv("data/HomeCredit_columns_description.csv", index_col=0, encoding="latin-1")
# The columns to look up
key_columns = ["TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
               "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "NAME_INCOME_TYPE", "OCCUPATION_TYPE"]
# Keep the rows that describe the application table and name one of those columns
is_application_row = dictionary["Table"].str.startswith("application")
is_key_column = dictionary["Row"].isin(key_columns)
key_descriptions = dictionary.loc[is_application_row & is_key_column, ["Row", "Description"]].set_index("Row")
# Show long descriptions in full rather than cut off
pd.set_option("display.max_colwidth", 120)
print(type(key_descriptions).__name__, key_descriptions.shape)
key_descriptions
""")

code("""
# Check your work
assert len(key_descriptions) == 10, f"expected 10 dictionary rows, got {len(key_descriptions)}"
""")

md("""
`AMT_GOODS_PRICE` is the price of the goods a consumer loan paid for, and the three `EXT_SOURCE` columns are
normalised credit scores supplied by outside sources. Steps 1.18 to 1.21 show they are the strongest signal in
the data.
""")

step("1.9", "Count the missing values",
     "Which columns have gaps, and how badly?",
     "why does missing data matter?",
     "A model cannot multiply a missing value. You either fill it, drop it, or (as a scorecard does) give it a bin "
     "of its own. Which choice is right depends on whether the gap itself carries information.")

code("""
# Share of applications missing each column, largest first
missing = app.isna().mean().sort_values(ascending=False)
# How many columns have any gap, and how many are more than half empty
n_with_gaps = int((missing > 0).sum())
n_half_empty = int((missing > 0.5).sum())
print(type(missing).__name__, len(missing))
# Print how many columns have gaps and how many are more than half empty, then show the five emptiest
print(f"columns with any missing value: {n_with_gaps} of {app.shape[1]}; more than half missing: {n_half_empty}")
missing.head()
""")

code("""
# Check your work
assert n_with_gaps == 67, f"expected 67 columns with gaps, got {n_with_gaps}"
assert n_half_empty == 41, f"expected 41 columns more than half empty, got {n_half_empty}"
""")

step("1.10", "Chart the most incomplete columns",
     "Draw the 20 worst columns as a horizontal bar chart.",
     "why a chart when the numbers are printed?",
     "Twenty numbers are hard to compare by eye; twenty bars are not.")

code("""
# Draw the 20 most incomplete columns, longest bar at the top
fig, ax = plt.subplots(figsize=(9, 6))
(missing.head(20) * 100).sort_values().plot(kind="barh", ax=ax, color=BLUE)
ax.set_xlabel("Share of applications missing the value (%)")
# Title the chart, fit the layout and save it as figs/missing_values.png
ax.set_title("The 20 most incomplete columns")
plt.tight_layout()
plt.savefig("figs/missing_values.png", dpi=150, bbox_inches="tight")
# Show the chart
plt.show()
""")

md("""
Sixty-seven of the 122 columns have gaps, and 41 are more than half empty. Almost all of those describe the
applicant's building (common area, number of floors, apartment counts, in three statistical variants each), which
most applicants never supplied. None of them is a candidate characteristic.
""")

step("1.11", "Find values that are not what they claim to be",
     "`DAYS_EMPLOYED` counts days before the application, so it should be negative. Count the positive values.",
     "what is a placeholder value?",
     "A number a system writes when the real value is unknown. It looks like data but is not, so it has to be "
     "found and replaced before any calculation uses it.")

code("""
# True where DAYS_EMPLOYED is positive, which should be impossible
positive = app["DAYS_EMPLOYED"] > 0
# The distinct values among those rows
placeholder_values = app.loc[positive, "DAYS_EMPLOYED"].unique()
print(type(positive).__name__, len(positive))
print(f"applicants with DAYS_EMPLOYED > 0: {positive.sum():,}")
# Print the distinct positive values: one value means one placeholder
print("distinct values among them:", placeholder_values)
""")

code("""
# Check your work
assert positive.sum() == 55_374, f"expected 55,374 positive values, got {positive.sum():,}"
assert list(placeholder_values) == [365243], f"expected one placeholder, got {placeholder_values}"
""")

md("""
Every one holds the same placeholder, 365243 days (about a thousand years). These are pensioners and unemployed
applicants. Left in, the value would give retirees a millennium of work history, so Step 1.33 turns it into a
missing value, and Step 1.20 shows that the gap itself carries information.
""")

step("1.12", "Check the money columns for impossible values",
     "Summarise income, credit, annuity and goods price, and look for amounts that cannot be right.",
     "what is an outlier, and should it be removed?",
     "A value far from the rest. A scorecard bins each characteristic at its deciles (Step 2.5), so an extreme "
     "value lands in the top bin with everyone else above the 90th percentile and cannot pull anything. It can "
     "stay.")

code("""
# The four money columns
money = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
# Count, mean, spread and percentiles, rounded to whole units
money_summary = app[money].describe().round(0)
# Incomes above ten million a year
n_huge_income = int((app["AMT_INCOME_TOTAL"] > 1e7).sum())
print(type(money_summary).__name__, money_summary.shape)
print(f"applicants with an annual income above 10 million: {n_huge_income}, the largest {app['AMT_INCOME_TOTAL'].max():,.0f}")
# Display the table
money_summary
""")

code("""
# Check your work
assert (app["AMT_INCOME_TOTAL"] > 0).all() and (app["AMT_CREDIT"] > 0).all(), "a non-positive amount"
assert n_huge_income == 3 and app["AMT_INCOME_TOTAL"].max() == 117_000_000
assert app["AMT_INCOME_TOTAL"].median() == 147_150 and app["AMT_CREDIT"].median() == 513_531
""")

md("""
The median applicant earns 147,150 a year and borrows 513,531 (Home Credit does not state the currency). Three
applicants report an income above ten million, one of them 117 million: almost certainly data-entry errors, but
they stay in for the reason given in the Concept line.

## 1B. Explore

Six questions, each answered with a table or a chart and one sentence on what it shows.
""")

step("1.13", "Count the outcomes",
     "How common is default?",
     "why is accuracy the wrong measure here?",
     "When only 8% default, a model that approves everyone is 92% \"accurate\" and useless. What matters is "
     "ranking: does the score put defaulters below repayers? AUC and KS (Steps 2.1 and 2.30) measure that.")

code("""
# Count repaid (0) and defaulted (1), with readable labels
outcome = app["TARGET"].value_counts().rename({0: "repaid", 1: "defaulted"})
# The share that defaulted
default_rate = app["TARGET"].mean()
print(type(outcome).__name__, len(outcome))
print(outcome.to_string())
# Print the share of applicants who defaulted
print(f"default rate: {default_rate:.1%}")
""")

code("""
# Check your work
assert outcome["repaid"] == 282_686 and outcome["defaulted"] == 24_825, outcome.to_dict()
assert round(default_rate, 3) == 0.081, f"expected an 8.1% default rate, got {default_rate:.4f}"
""")

code("""
# Draw the two counts with their shares written on the bars
fig, ax = plt.subplots(figsize=(6, 3.8))
outcome.plot(kind="bar", ax=ax, color=[BLUE, RED], rot=0)
for bar_position in range(len(outcome)):
    count = outcome.iloc[bar_position]
    ax.text(bar_position, count, f"{count:,} ({count / len(app):.1%})", ha="center", va="bottom")
# Name the y axis, title the chart and fit the layout
ax.set_ylabel("Applicants")
ax.set_title("Outcome of the 307,511 applications")
plt.tight_layout()
# Save the chart as figs/target_balance.png and show it
plt.savefig("figs/target_balance.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
About 8% of applicants defaulted (24,825 of 307,511). With a target this lopsided, ranking is what counts.
""")

step("1.14", "Compare default rates by age",
     "Convert `DAYS_BIRTH` to years and work out the default rate in each five-year band.",
     "what is a default rate by group?",
     "The share of applicants in a group who defaulted. Comparing it across groups is the simplest way to see "
     "whether a characteristic carries signal before any model is fitted.")

code("""
# Days before the application, made positive and converted to years
age = -app["DAYS_BIRTH"] / 365.25
# Five-year bands from 20 to 70
age_band = pd.cut(age, range(20, 75, 5))
# Applicants and default rate in each band
by_age = app.groupby(age_band, observed=True)["TARGET"].agg(applicants="size", default_rate="mean")
print(type(by_age).__name__, by_age.shape, "| median age", round(age.median(), 1))
by_age
""")

code("""
# Check your work
assert by_age["default_rate"].iloc[0] > 0.12, "applicants under 25 should default at over 12%"
assert by_age["default_rate"].iloc[-1] < 0.04, "applicants over 65 should default at under 4%"
assert round(age.median()) == 43
""")

code("""
# Left: age distributions of repayers and defaulters; right: the default rate by band
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].hist(age[app["TARGET"] == 0], bins=25, density=True, alpha=0.6, color=BLUE, label="repaid")
axes[0].hist(age[app["TARGET"] == 1], bins=25, density=True, alpha=0.6, color=RED, label="defaulted")
# Left panel: name the x and y axes and add a title
axes[0].set_xlabel("Age at application (years)")
axes[0].set_ylabel("Density")
axes[0].set_title("Age distribution by outcome")
# Add the left panel's legend, draw the default rate of each age band as a bar on the right panel, and a dashed line at the average default rate
axes[0].legend()
(by_age["default_rate"] * 100).plot(kind="bar", ax=axes[1], color=RED, rot=45)
axes[1].axhline(app["TARGET"].mean() * 100, color=INK, ls="--", lw=1)
# Right panel: name the x and y axes and add a title
axes[1].set_xlabel("Age band")
axes[1].set_ylabel("Default rate (%)")
axes[1].set_title("Default rate falls with age")
# Fit the layout, save the chart as figs/age_vs_default.png and show it
plt.tight_layout()
plt.savefig("figs/age_vs_default.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
The default rate falls steadily from about 12% for applicants under 25 to under 4% for those over 65, a
one-direction pattern a scorecard captures well. Step 2.20 nevertheless removes age from the model, for a reason
that only shows once the other characteristics are in.
""")

step("1.15", "Compare income and credit amount by outcome",
     "Draw both on a log scale, because a few very large values stretch the axis, and tabulate the default rate "
     "by income decile.",
     "what is a decile?",
     "Sort applicants by a value and cut them into ten groups of equal size. The first decile holds the lowest "
     "10%, the tenth the highest 10%.")

code("""
# Histograms of log10 income and log10 credit, repayers in blue and defaulters in red
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, col, title in [(axes[0], "AMT_INCOME_TOTAL", "Annual income"), (axes[1], "AMT_CREDIT", "Credit amount")]:
    ax.hist(np.log10(app.loc[app["TARGET"] == 0, col]), bins=60, density=True, alpha=0.6, color=BLUE, label="repaid")
    ax.hist(np.log10(app.loc[app["TARGET"] == 1, col]), bins=60, density=True, alpha=0.6, color=RED, label="defaulted")
    ax.set_xlabel(f"log10({col})")
    ax.set_ylabel("Density")
    ax.set_title(f"{title} by outcome")
    ax.legend()
plt.tight_layout()
# Save the chart as figs/income_credit_distribution.png and show it
plt.savefig("figs/income_credit_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
""")

code("""
# Ten income groups (duplicate cut points merged), and the default rate in each
income_decile = pd.qcut(app["AMT_INCOME_TOTAL"], 10, duplicates="drop")
by_income = app.groupby(income_decile, observed=True)["TARGET"].agg(applicants="size", default_rate="mean").round(4)
print(type(by_income).__name__, by_income.shape)
# Display the table
by_income
""")

code("""
# Check your work
assert by_income["default_rate"].iloc[-1] < 0.065, "the top income group should default at about 6%"
assert by_income["default_rate"].iloc[:7].between(0.08, 0.0905).all(), "the bottom seven should sit between 8% and 9%"
""")

md("""
Income on its own says little: the default rate sits between 8% and 9% for the bottom seven groups and only falls,
to about 6%, at the top. Credit amount looks the same for both outcomes. What matters more is how the two relate.
""")

step("1.16", "Look at the credit-to-income ratio",
     "Divide credit by income and work out the default rate by decile of that ratio.",
     "why a ratio?",
     "A loan of 500,000 is a heavy burden on a small income and a light one on a large income. The ratio puts "
     "applicants with different incomes on one scale.")

code("""
# Credit as a multiple of annual income
credit_to_income = app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"]
# Ten equal groups by that multiple, and the default rate in each
ratio_decile = pd.qcut(credit_to_income, 10)
by_ratio = app.groupby(ratio_decile, observed=True)["TARGET"].agg(applicants="size", default_rate="mean")
print(type(by_ratio).__name__, by_ratio.shape)
# Display the table
by_ratio.round(4)
""")

code("""
# Check your work: the rate rises from the first group to a peak in the middle, then falls again
peak_position = int(np.argmax(by_ratio["default_rate"].to_numpy()))
assert 0 < peak_position < 9, f"expected the peak inside the range, got group {peak_position}"
assert by_ratio["default_rate"].iloc[-1] < by_ratio["default_rate"].max()
""")

code("""
# Write each ratio group's range as the label for its bar
ratio_labels = []
for ratio_interval in by_ratio.index:
    ratio_labels.append(f"{ratio_interval.left:.1f} to {ratio_interval.right:.1f}")
# Make one figure with one set of axes
fig, ax = plt.subplots(figsize=(10, 4))
# Draw the default rate of each credit-to-income decile as a bar, write each decile's range under its bar and name the x axis
(by_ratio["default_rate"] * 100).plot(kind="bar", ax=ax, color=RED, rot=45)
ax.set_xticklabels(ratio_labels)
ax.set_xlabel("Credit amount as a multiple of annual income (deciles)")
# Name the y axis, title the chart and fit the layout
ax.set_ylabel("Default rate (%)")
ax.set_title("Default rate by credit-to-income ratio: a hump, not a line")
plt.tight_layout()
# Save the chart as figs/credit_to_income.png and show it
plt.savefig("figs/credit_to_income.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
The default rate rises from about 7% for loans under 1.3 times income to about 9% around three to four times
income, then falls again for the largest multiples. A pattern that goes up and then down is a problem for a points
table, which Step 2.8 calls a zig-zag; coarse classing is the standard fix.
""")

step("1.17", "Compare default rates across applicant groups",
     "Income type, education, housing and occupation, for groups of at least 500 applicants.",
     "what is a categorical characteristic?",
     "One whose values are labels, not numbers (\"Pensioner\", \"Working\"). A scorecard gives each label its "
     "own bin.")

code("""
# A helper that draws the default rate of each level of one column
def default_rate_by_group(col, ax, min_applicants=500):
    # Applicants and default rate for each level of one column
    table = app.groupby(col, observed=True)["TARGET"].agg(applicants="size", default_rate="mean")
    # Keep the levels with enough applicants to trust the rate, safest first
    table = table[table["applicants"] >= min_applicants].sort_values("default_rate")
    # One horizontal bar per level, with the overall rate as a dashed line
    (table["default_rate"] * 100).plot(kind="barh", ax=ax, color=BLUE)
    ax.axvline(app["TARGET"].mean() * 100, color=INK, ls="--", lw=1)
    ax.set_xlabel("Default rate (%)")
    ax.set_ylabel("")
    ax.set_title(col)
    return table


# Four panels, one per column
group_columns = ["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]
group_tables = {}
# Make a 2 x 2 grid of panels, one per column
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
# One panel per column, keeping each table to read later
for ax, col in zip(axes.ravel(), group_columns):
    group_tables[col] = default_rate_by_group(col, ax)
# Fit the layout, save the chart as figs/default_by_group.png and show it
plt.tight_layout()
plt.savefig("figs/default_by_group.png", dpi=150, bbox_inches="tight")
plt.show()
# Print the type and shape of the occupation table
print(type(group_tables["OCCUPATION_TYPE"]).__name__, group_tables["OCCUPATION_TYPE"].shape)
""")

code("""
# Check your work: the extremes quoted below
occupation = group_tables["OCCUPATION_TYPE"]["default_rate"]
assert occupation["Accountants"] < 0.05 and round(occupation["Low-skill Laborers"], 2) == 0.17
income_type = group_tables["NAME_INCOME_TYPE"]["default_rate"]
# Stop the run if this does not hold
assert income_type["Pensioner"] < 0.06 and income_type["Working"] < 0.10
""")

md("""
The dashed line is the overall rate. Pensioners and state servants default at about 5%, working applicants at
nearly 10%. Education runs from about 5% with higher education to 11% with lower secondary. Renting or living with
parents carries roughly 12% against 8% for applicants living in a house or apartment (the `NAME_HOUSING_TYPE`
label; it records the type of housing, not ownership, which is `FLAG_OWN_REALTY`). Occupation spreads widest: accountants under 5%, low-skill
labourers 17%.
""")

step("1.18", "Inspect the three external scores",
     "`EXT_SOURCE_1` to `EXT_SOURCE_3` are credit scores from outside sources. How complete are they, and how "
     "well do they separate the outcomes?",
     "what is a correlation with the outcome?",
     "A number from -1 to 1 saying how much a column moves with default. Negative means the higher the value, the "
     "lower the risk. It is a rough first look for a 0/1 target, not a final measure.")

code("""
# The three external scores
ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
# Share missing, in percent
ext_missing = (app[ext_cols].isna().mean() * 100).round(1)
# Correlation of each with default
ext_correlation = app[ext_cols].corrwith(app["TARGET"]).round(3)
print(type(ext_missing).__name__, len(ext_missing))
print("share missing:", ext_missing.to_dict())
# Print each external score's correlation with default
print("correlation with default:", ext_correlation.to_dict())
""")

code("""
# Check your work
assert ext_missing.to_dict() == {"EXT_SOURCE_1": 56.4, "EXT_SOURCE_2": 0.2, "EXT_SOURCE_3": 19.8}, ext_missing.to_dict()
assert (ext_correlation < -0.15).all(), ext_correlation.to_dict()
""")

code("""
# One histogram per score, repayers in blue and defaulters in red
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ext_cols):
    ax.hist(app.loc[app["TARGET"] == 0, col].dropna(), bins=40, density=True, alpha=0.6, color=BLUE, label="repaid")
    ax.hist(app.loc[app["TARGET"] == 1, col].dropna(), bins=40, density=True, alpha=0.6, color=RED, label="defaulted")
    ax.set_title(col)
    ax.set_xlabel("Score (0 = riskiest, 1 = safest)")
    ax.legend()
plt.tight_layout()
# Save the chart as figs/external_scores.png and show it
plt.savefig("figs/external_scores.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
The second score is present for almost everyone, the third for four applicants in five, and the first for fewer
than half. All three push defaulters to the left, and each correlates with default more strongly than any other column in
the application table (Step 1.20).
""")

step("1.19", "Check the external-score signal across its whole range",
     "Split applicants into ten equal groups by their average external score and look at the default rate in "
     "each. Nothing is fitted; this only describes the data.",
     "what is a monotonic pattern?",
     "One that moves in a single direction: here, the higher the score, the lower the default rate in every "
     "group. A scorecard rewards monotonic patterns with points that rise bin by bin.")

code("""
# The average of whichever external scores the applicant has
ext_mean = app[ext_cols].mean(axis=1)
# Ten equal groups by that average, and the default rate in each
ext_band = pd.qcut(ext_mean, 10)
by_ext = app.groupby(ext_band, observed=True)["TARGET"].agg(applicants="size", default_rate="mean")
print(type(by_ext).__name__, by_ext.shape)
# Display the table
by_ext.round(4)
""")

code("""
# Check your work: the rate falls in every step, and about tenfold from bottom to top
assert by_ext["default_rate"].is_monotonic_decreasing, "the default rate should fall group by group"
assert by_ext["default_rate"].iloc[0] / by_ext["default_rate"].iloc[-1] > 10
""")

code("""
# Write each external-score group's range as the label for its bar
ext_labels = []
for ext_interval in by_ext.index:
    ext_labels.append(f"{ext_interval.left:.2f} to {ext_interval.right:.2f}")
# Make one figure with one set of axes
fig, ax = plt.subplots(figsize=(10, 4))
# Draw the default rate of each external-score decile as a bar, write each decile's range under its bar and name the x axis
(by_ext["default_rate"] * 100).plot(kind="bar", ax=ax, color=RED, rot=45)
ax.set_xticklabels(ext_labels)
ax.set_xlabel("Average external score (deciles)")
# Name the y axis, title the chart and fit the layout
ax.set_ylabel("Default rate (%)")
ax.set_title("Default rate by average external score")
plt.tight_layout()
# Save the chart as figs/ext_source_deciles.png and show it
plt.savefig("figs/ext_source_deciles.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
The default rate falls about tenfold from the lowest group (23%) to the highest (2%), smoothly. That is the shape a
scorecard captures: higher bins earn more points.
""")

step("1.20", "Treat a missing value as a signal",
     "The placeholder from Step 1.11 is a missing employment length. Compare the default rate of applicants with "
     "and without one.",
     "is missing data always a problem?",
     "No. If the applicants with a gap behave differently from the rest, the gap is information. Filling it with "
     "an average would throw that information away; a scorecard gives missing values a bin of their own.")

code("""
# Years employed, with the placeholder turned into a missing value
employed_years = -app["DAYS_EMPLOYED"].where(~positive, np.nan) / 365.25
# Default rate for applicants with and without a known employment length
by_missing = app.groupby(employed_years.isna())["TARGET"].agg(applicants="size", default_rate="mean")
by_missing = by_missing.rename(index={False: "employment length known", True: "missing (pensioners and unemployed)"})
print(type(by_missing).__name__, by_missing.shape)
# Display the table
by_missing.round(4)
""")

code("""
# Check your work
assert by_missing.loc["missing (pensioners and unemployed)", "applicants"] == 55_374
assert round(by_missing.loc["missing (pensioners and unemployed)", "default_rate"], 3) == 0.054
assert round(by_missing.loc["employment length known", "default_rate"], 3) == 0.087
""")

code("""
# Default rate by years employed, for applicants with a known length
employed_band = pd.cut(employed_years, [0, 1, 2, 3, 5, 10, 20, 50])
by_employed = app.groupby(employed_band, observed=True)["TARGET"].agg(applicants="size", default_rate="mean")
# Make one figure with one set of axes
fig, ax = plt.subplots(figsize=(9, 4))
# Draw the default rate of each employment-length band as a bar, add a dashed line at the average default rate and name the x axis
(by_employed["default_rate"] * 100).plot(kind="bar", ax=ax, color=RED, rot=0)
ax.axhline(app["TARGET"].mean() * 100, color=INK, ls="--", lw=1)
ax.set_xlabel("Years in current employment")
# Name the y axis, title the chart and fit the layout
ax.set_ylabel("Default rate (%)")
ax.set_title("Default rate by employment length (applicants with a known length)")
plt.tight_layout()
# Save the chart as figs/employment_length.png and show it
plt.savefig("figs/employment_length.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
The missing group defaults *less* often (5.4% against 8.7% for applicants with a known employment length), so the
gap is information. Among those employed, the rate falls from about 11% in the first years to about 4% after
twenty.
""")

step("1.21", "Rank every numeric column by its correlation with default",
     "See where the signal is before any binning.",
     "why look at all columns at once?",
     "It shows quickly whether the application table alone is strong or weak, which decides whether the credit "
     "history tables are worth the effort of joining.")

code("""
# Every numeric column except the ID and the outcome
numeric = app.select_dtypes("number").drop(columns=["SK_ID_CURR", "TARGET"])
# Correlation of each with default, sorted from most negative to most positive
correlation = numeric.corrwith(app["TARGET"]).dropna().sort_values()
# The ten most negative and the ten most positive
extremes = pd.concat([correlation.head(10), correlation.tail(10)])
print(type(extremes).__name__, len(extremes))
extremes.round(3)
""")

code("""
# Check your work: the three external scores are the three most negative, and the strongest form field is DAYS_BIRTH
assert set(correlation.index[:3]) == set(ext_cols), list(correlation.index[:3])
assert correlation.index[-1] == "DAYS_BIRTH", correlation.index[-1]
""")

code("""
# Draw the twenty, safer-as-it-rises in blue and riskier-as-it-rises in red
fig, ax = plt.subplots(figsize=(9, 6))
extremes.plot(kind="barh", ax=ax, color=bar_colours(-extremes))
ax.set_xlabel("Correlation with default (negative = safer as the value rises)")
# Title the chart, fit the layout and save it as figs/correlation_with_default.png
ax.set_title("The ten most negative and ten most positive correlations with default")
plt.tight_layout()
plt.savefig("figs/correlation_with_default.png", dpi=150, bbox_inches="tight")
# Show the chart
plt.show()
""")

md("""
The three external scores stand alone at about -0.16 to -0.18; nothing else in the application table comes close.
They are not answers on the application form but third-party scores supplied with the data. `DAYS_BIRTH` is the
strongest form field (positive, because a larger value means younger). The form fields are weak on their own,
which is why the next steps add the applicant's credit history.

## 1C. Add the credit history

Home Credit supplies six more tables, each keyed by the applicant's `SK_ID_CURR` (or, for the monthly bureau
statuses, by the bureau record):

| Table | Rows | What it holds |
|---|---:|---|
| `bureau.csv` | 1.7 million | Credits reported by other lenders to the credit bureau, one row per credit |
| `bureau_balance.csv` | 27.3 million | Monthly status of each bureau credit (current, days past due, closed) |
| `previous_application.csv` | 1.7 million | The applicant's earlier applications to Home Credit, approved or refused |
| `POS_CASH_balance.csv` | 10.0 million | Monthly balances of earlier point-of-sale and cash loans |
| `credit_card_balance.csv` | 3.8 million | Monthly balances and limits of earlier credit cards |
| `installments_payments.csv` | 13.6 million | Every instalment due and what was paid, and when |

Each is read with `pd.read_csv` (only the columns needed) and rolled up to one row per applicant.
`src/relational_features.py` holds the same roll-up for the pipeline, and Step 1.30 checks that the two agree.
""")

step("1.22", "Read the monthly bureau statuses",
     "A status of 1 to 5 means some days past due; `C`, `X` and `0` mean closed, unknown or current. Read the file "
     "and mark each month as delinquent or not.",
     "what is delinquency?",
     "Being behind on a payment. \"Days past due\" (DPD) counts how far behind. A borrower who has been "
     "delinquent before is more likely to be delinquent again.")

code("""
# One row per bureau credit per month; read the status as a category to save memory
bureau_balance = pd.read_csv("data/bureau_balance.csv", usecols=["SK_ID_BUREAU", "STATUS"], dtype={"STATUS": "category"})
# 1 if that month was past due (status 1 to 5), else 0
bureau_balance["delinquent"] = bureau_balance["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("int8")
print(type(bureau_balance).__name__, bureau_balance.shape)
bureau_balance.head()
""")

code("""
# Check your work
assert len(bureau_balance) == 27_299_925, f"expected 27,299,925 monthly statuses, got {len(bureau_balance):,}"
""")

step("1.23", "Roll the monthly statuses up to one row per bureau credit",
     "For each credit: how many months of history, and was it ever delinquent?",
     "what does \"roll up\" mean?",
     "Group many rows that share a key and replace them with one summary row: a count, a maximum, an average. "
     "This is how a table with millions of rows becomes one row per applicant.")

code("""
# Months of history and ever-delinquent, per bureau credit
balance_summary = bureau_balance.groupby("SK_ID_BUREAU").agg(
    BB_MONTHS_COUNT=("STATUS", "count"), BB_EVER_DELINQUENT=("delinquent", "max")).reset_index()
print(type(balance_summary).__name__, balance_summary.shape)
print(f"{balance_summary['BB_EVER_DELINQUENT'].mean():.1%} of credits were ever past due")
# Free the 27 million rows
del bureau_balance
balance_summary.head()
""")

code("""
# Check your work
assert len(balance_summary) == 817_395 and balance_summary["SK_ID_BUREAU"].is_unique
assert round(balance_summary["BB_EVER_DELINQUENT"].mean(), 3) == 0.126
""")

step("1.24", "Summarise each applicant's bureau credits",
     "Read the bureau credits, attach the monthly summary, and roll up to one row per applicant: how many credits, "
     "how many active, how old, how overdue, how large, and the share ever delinquent.",
     "what is a credit bureau?",
     "An agency that collects every lender's records of a borrower's credits, so a new lender can see debts and "
     "repayment behaviour held elsewhere.")

code("""
# One row per credit reported by another lender
bureau = pd.read_csv("data/bureau.csv", usecols=["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT",
                                                 "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"])
# Attach months of history and ever-delinquent from Step 1.23
bureau = bureau.merge(balance_summary, on="SK_ID_BUREAU", how="left")
# 1 if the credit is still open
bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
# One row per applicant
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
print(type(bureau_features).__name__, bureau_features.shape)
# Free the memory held by tables no longer needed
del bureau, balance_summary
""")

code("""
# Check your work
assert bureau_features.shape == (305_811, 10), bureau_features.shape
assert bureau_features["SK_ID_CURR"].is_unique
""")

step("1.25", "Summarise each applicant's earlier Home Credit applications",
     "How many, what share approved or refused, their amounts, and how recent the last decision was.",
     "why do earlier refusals matter?",
     "A lender that refused the same person before had a reason. A high refusal share is a behavioural signal the "
     "application table cannot supply.")

code("""
# One row per earlier application
previous = pd.read_csv("data/previous_application.csv", usecols=["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS",
                                                                 "AMT_APPLICATION", "AMT_CREDIT", "DAYS_DECISION"])
# How the earlier applications were decided
decisions_count = previous["NAME_CONTRACT_STATUS"].value_counts()
print("decisions on earlier applications:", decisions_count.to_dict())
# 1 for approved, 1 for refused
previous["IS_APPROVED"] = (previous["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
previous["IS_REFUSED"] = (previous["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
# One row per applicant
previous_features = previous.groupby("SK_ID_CURR").agg(
    PREV_APP_COUNT=("SK_ID_PREV", "count"),
    PREV_APPROVED_SHARE=("IS_APPROVED", "mean"),
    PREV_REFUSED_SHARE=("IS_REFUSED", "mean"),
    PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
    PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
    PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),              # least negative = most recent decision
).reset_index()
print(f"{len(previous):,} earlier applications summarised for {len(previous_features):,} applicants")
print(type(previous_features).__name__, previous_features.shape)
# Free the memory held by tables no longer needed
del previous
""")

code("""
# Check your work
assert previous_features.shape == (338_857, 7), previous_features.shape
assert decisions_count["Refused"] == 290_678
""")

step("1.26", "Summarise point-of-sale and cash loans",
     "Days past due, and the average number of instalments.",
     "what is a point-of-sale loan?",
     "Credit given at the shop counter to buy a specific item, such as a phone, repaid in instalments.")

code("""
# One row per loan per month
pos_cash = pd.read_csv("data/POS_CASH_balance.csv", usecols=["SK_ID_CURR", "SK_DPD", "SK_DPD_DEF", "CNT_INSTALMENT"])
# Share of months with any days past due
pos_late_share = (pos_cash["SK_DPD"] > 0).mean()
# One row per applicant
pos_features = pos_cash.groupby("SK_ID_CURR").agg(
    POS_DPD_MAX=("SK_DPD", "max"),
    POS_DPD_MEAN=("SK_DPD", "mean"),
    POS_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
    POS_CNT_INSTALMENT_MEAN=("CNT_INSTALMENT", "mean"),
).reset_index()
print(f"{len(pos_cash):,} monthly records summarised for {len(pos_features):,} applicants; "
      f"{pos_late_share:.1%} of months had days past due")
print(type(pos_features).__name__, pos_features.shape)
# Free the memory held by tables no longer needed
del pos_cash
""")

code("""
# Check your work
assert pos_features.shape == (337_252, 5) and round(pos_late_share, 3) == 0.030
""")

step("1.27", "Summarise credit cards",
     "Balances, how much of the limit is used, and days past due.",
     "what is utilisation?",
     "Balance divided by credit limit. A card used to its limit suggests a borrower short of money.")

code("""
# One row per card per month
cards = pd.read_csv("data/credit_card_balance.csv", usecols=["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"])
# Balance over limit; a zero limit becomes missing so nothing divides by zero
cards["UTILIZATION"] = cards["AMT_BALANCE"] / cards["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
# One row per applicant
card_features = cards.groupby("SK_ID_CURR").agg(
    CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
    CC_BALANCE_MAX=("AMT_BALANCE", "max"),
    CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
    CC_DPD_MAX=("SK_DPD", "max"),
).reset_index()
print(f"{len(cards):,} monthly card records summarised for {len(card_features):,} applicants")
print(type(card_features).__name__, card_features.shape)
# Free the memory held by tables no longer needed
del cards
""")

code("""
# Check your work
assert card_features.shape == (103_558, 5), card_features.shape
""")

step("1.28", "Summarise instalment payments",
     "The most direct record of paying on time. Both day columns count days before the application, so payment "
     "day minus due day is positive when the payment was late.",
     "what is a shortfall?",
     "The instalment due minus the amount paid. Positive means the borrower paid less than they owed.")

code("""
# One row per instalment; 32-bit floats halve the memory of 13.6 million rows
instalments = pd.read_csv("data/installments_payments.csv",
                          usecols=["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"],
                          dtype={"DAYS_INSTALMENT": "float32", "DAYS_ENTRY_PAYMENT": "float32",
                                 "AMT_INSTALMENT": "float32", "AMT_PAYMENT": "float32"})
# Days late (positive) or early (negative)
instalments["DAYS_LATE"] = instalments["DAYS_ENTRY_PAYMENT"] - instalments["DAYS_INSTALMENT"]
# Amount due minus amount paid
instalments["SHORTFALL"] = instalments["AMT_INSTALMENT"] - instalments["AMT_PAYMENT"]
# Share of instalments paid late
late_share = (instalments["DAYS_LATE"] > 0).mean()
# One row per applicant
instalment_features = instalments.groupby("SK_ID_CURR").agg(
    INSTAL_COUNT=("AMT_INSTALMENT", "count"),
    INSTAL_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
    INSTAL_DAYS_LATE_MAX=("DAYS_LATE", "max"),
    INSTAL_SHORTFALL_MEAN=("SHORTFALL", "mean"),
    INSTAL_SHORTFALL_SUM=("SHORTFALL", "sum"),
).reset_index()
print(f"{len(instalments):,} instalments summarised for {len(instalment_features):,} applicants; "
      f"{late_share:.1%} were paid late")
print(type(instalment_features).__name__, instalment_features.shape)
# Free the memory held by tables no longer needed
del instalments
""")

code("""
# Check your work
assert instalment_features.shape == (339_587, 6) and round(late_share, 3) == 0.084
""")

step("1.29", "Join the five summaries into one history table",
     "One row per applicant who appears in any of them.",
     "what is an outer join?",
     "A join that keeps every key found in either table. An applicant with a bureau record but no credit card keeps "
     "their row, with the card columns missing.")

code("""
# Start from the bureau summary and add the other four, keeping every applicant found in any of them
history = bureau_features
for history_part in (previous_features, pos_features, card_features, instalment_features):
    history = history.merge(history_part, on="SK_ID_CURR", how="outer")
print(type(history).__name__, history.shape)
# Print how many applicants and history features there are, then show the first rows
print(f"history: {len(history):,} applicants x {history.shape[1] - 1} features")
history.head()
""")

code("""
# Check your work: one row per applicant and exactly the 28 history features the pipeline defines
assert history["SK_ID_CURR"].is_unique
assert set(history.columns) == {"SK_ID_CURR", *RELATIONAL_NUMERIC}
assert history.shape == (353_895, 29), history.shape
""")

step("1.30", "Compare the history table with the pipeline's",
     "`load_relational_features` in `src/relational_features.py` builds the same table for the pipeline (and caches "
     "it). The two must be identical.",
     "why compare with a second implementation?",
     "Two independent builds that agree are strong evidence that both are right. If they ever disagree, this cell "
     "stops the notebook before a wrong table reaches the model.")

code("""
# The pipeline's tested, fingerprinted build of the same table
from relational_features import load_relational_features

# The pipeline's own history table, built by the tested function
pipeline_history = load_relational_features(verbose=False)
# Print the type and shape of the pipeline's history table
print(type(pipeline_history).__name__, pipeline_history.shape)
""")

code("""
# Check your work: same rows, same columns, same values (column order and integer/float types may differ)
history_sorted = history.sort_values("SK_ID_CURR").reset_index(drop=True)
pipeline_sorted = pipeline_history.sort_values("SK_ID_CURR").reset_index(drop=True)
pd.testing.assert_frame_equal(history_sorted, pipeline_sorted, check_dtype=False, check_like=True)
# Free the memory held by tables no longer needed
del pipeline_history, pipeline_sorted, history_sorted
print("the plain-pandas history table equals the pipeline's")
""")

step("1.31", "See how many applicants have each kind of history",
     "Work out the share of applicants in the main table who appear in each summary.",
     "what is a thin file?",
     "An applicant with little or no credit history. A lender knows less about them, which is itself a risk.")

code("""
# Share of the 307,511 applicants found in each summary
coverage = {}
for coverage_label, coverage_part in [("a bureau record", bureau_features), ("an earlier Home Credit application", previous_features),
                                      ("a credit card", card_features), ("an instalment record", instalment_features)]:
    coverage[coverage_label] = app["SK_ID_CURR"].isin(coverage_part["SK_ID_CURR"]).mean()
print(type(coverage).__name__, len(coverage))
# One line per history table
for coverage_label in coverage:
    print(f"applicants with {coverage_label}: {coverage[coverage_label]:.1%}")
""")

code("""
# Check your work
assert round(coverage["a bureau record"], 3) == 0.857 and round(coverage["a credit card"], 3) == 0.283
""")

step("1.32", "Check that the history carries signal",
     "The default rate by number of bureau credits, and by the share of earlier applications refused.",
     "what is a behavioural characteristic?",
     "One that describes how a person has handled credit before (late payments, refusals), as opposed to who they "
     "are (age, education). Behaviour is usually the stronger predictor.")

code("""
# The outcome with two history columns attached; applicants with no history get missing values
with_history = app[["SK_ID_CURR", "TARGET"]].merge(history[["SK_ID_CURR", "BUREAU_COUNT", "PREV_REFUSED_SHARE"]], how="left")
# Bands of bureau-credit count (no record counts as 0) and of refusal share
bureau_band = pd.cut(with_history["BUREAU_COUNT"].fillna(0), [-1, 0, 1, 3, 5, 10, 1000], labels=["0", "1", "2-3", "4-5", "6-10", "over 10"])
refused_band = pd.cut(with_history["PREV_REFUSED_SHARE"], [-0.01, 0, 0.25, 0.5, 0.75, 1.0], labels=["none", "up to 25%", "25-50%", "50-75%", "over 75%"])
# Default rate in each band
rate_by_bureau = with_history.groupby(bureau_band, observed=True)["TARGET"].mean()
rate_by_refused = with_history.groupby(refused_band, observed=True)["TARGET"].mean()
print(type(with_history).__name__, with_history.shape)
# Print the default rate in each bureau-credit band and each refusal-share band
print("by bureau credits:", rate_by_bureau.round(3).to_dict())
print("by refusal share:", rate_by_refused.round(3).to_dict())
""")

code("""
# Check your work
assert len(with_history) == 307_511, "the join must not add or drop applicants"
assert rate_by_bureau["0"] > 0.095 and rate_by_refused["none"] < 0.075 and rate_by_refused["over 75%"] > 0.15
""")

code("""
# Draw both as bar charts with the overall rate as a dashed line
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
# Save the chart as figs/history_vs_default.png and show it
plt.savefig("figs/history_vs_default.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
Applicants with no bureau record default at about 10%, those with a few credits at 7% to 8%: a thin file is a risk
in itself. The refusal share is stronger still: applicants never refused before default at about 7%, those refused
more than three times in four at over 15%.

## 1D. Build the candidate characteristics

The scorecard's inputs are built one at a time first, so you can see each change. Then the steps are folded into one
`wrangle()` function, which is compared with the pipeline's tested version.
""")

step("1.33", "Turn ages and employment into years",
     "Work on a copy of the application table so the original stays untouched.",
     "why years and not days?",
     "\"43.1 years\" reads at a glance; \"-15,750 days\" does not. The bins and points will be read by people.")

code("""
# A copy, so the raw table stays as loaded
draft = app.copy()
# Age in years
draft["AGE_YEARS"] = -draft["DAYS_BIRTH"] / 365.25
# The placeholder flag, kept for exploration only
draft["DAYS_EMPLOYED_ANOM"] = (draft["DAYS_EMPLOYED"] == 365243).astype(int)
# Years employed, with the placeholder set to missing
draft["EMPLOYED_YEARS"] = -draft["DAYS_EMPLOYED"].where(draft["DAYS_EMPLOYED"] != 365243, np.nan) / 365.25
print(type(draft).__name__, draft.shape)
draft[["AGE_YEARS", "DAYS_EMPLOYED_ANOM", "EMPLOYED_YEARS"]].describe().round(2)
""")

code("""
# Check your work: exactly the placeholder applicants are now missing
assert (draft["EMPLOYED_YEARS"].isna() == positive).all()
assert draft["DAYS_EMPLOYED_ANOM"].sum() == 55_374
""")

step("1.34", "Add the affordability ratios",
     "Credit to income, annuity to income, annuity to credit (the repayment speed), credit to goods price, and "
     "income per family member.",
     "what is an annuity?",
     "The regular payment on the loan. Annuity divided by income is the share of income the loan takes each period.")

code("""
# Credit as a multiple of income
draft["CREDIT_INCOME_RATIO"] = draft["AMT_CREDIT"] / draft["AMT_INCOME_TOTAL"]
# Share of income taken by each payment
draft["ANNUITY_INCOME_RATIO"] = draft["AMT_ANNUITY"] / draft["AMT_INCOME_TOTAL"]
# Payment over credit: a high value means a short loan
draft["CREDIT_TERM"] = draft["AMT_ANNUITY"] / draft["AMT_CREDIT"]
# Credit over the price of the goods it bought
draft["CREDIT_GOODS_RATIO"] = draft["AMT_CREDIT"] / draft["AMT_GOODS_PRICE"]
# Income per family member; a family size of 0 becomes missing
draft["INCOME_PER_FAM_MEMBER"] = draft["AMT_INCOME_TOTAL"] / draft["CNT_FAM_MEMBERS"].replace(0, np.nan)
print(type(draft).__name__, draft.shape)
draft[["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "CREDIT_GOODS_RATIO", "INCOME_PER_FAM_MEMBER"]].head()
""")

code("""
# Check your work
assert draft.shape == (307_511, 130), draft.shape
assert (draft["CREDIT_INCOME_RATIO"] > 0).all()
""")

step("1.35", "Summarise the three external scores",
     "Their mean, their spread, and how many the applicant has.",
     "why keep the count of scores present?",
     "Missing scores carry information (Step 1.20 showed the same for employment). The count lets the model see "
     "how thin the applicant's file is.")

code("""
# Mean and standard deviation of the scores the applicant has
draft["EXT_SOURCE_MEAN"] = draft[ext_cols].mean(axis=1)
draft["EXT_SOURCE_STD"] = draft[ext_cols].std(axis=1)
# Number of scores present, 0 to 3
draft["EXT_SOURCE_COUNT"] = draft[ext_cols].notna().sum(axis=1)
print(type(draft).__name__, draft.shape)
draft["EXT_SOURCE_COUNT"].value_counts().sort_index()
""")

code("""
# Check your work
assert draft["EXT_SOURCE_COUNT"].between(0, 3).all()
assert draft.shape == (307_511, 133), draft.shape
""")

step("1.36", "Fold the steps into one `wrangle()` function",
     "Put Steps 1.33 to 1.35 in one function, keep only the columns listed in `src/feature_lists.py`, and run it "
     "on the raw table.",
     "why a function?",
     "One function applied to the raw table is the whole recipe in one place. The same function can then be run on "
     "new applicants, and the next step compares it with the pipeline's.")

code("""
# Every cleaning step from Part 1 in one function, so it can run on any copy of the table
def wrangle(raw):
    # Work on a copy
    out = raw.copy()
    # Ages and employment in years, placeholder set to missing (Step 1.33)
    out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.25
    out["DAYS_EMPLOYED_ANOM"] = (out["DAYS_EMPLOYED"] == 365243).astype(int)
    out["EMPLOYED_YEARS"] = -out["DAYS_EMPLOYED"].where(out["DAYS_EMPLOYED"] != 365243, np.nan) / 365.25
    # Affordability ratios (Step 1.34)
    out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"]
    out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"]
    out["CREDIT_TERM"] = out["AMT_ANNUITY"] / out["AMT_CREDIT"]
    out["CREDIT_GOODS_RATIO"] = out["AMT_CREDIT"] / out["AMT_GOODS_PRICE"]
    out["INCOME_PER_FAM_MEMBER"] = out["AMT_INCOME_TOTAL"] / out["CNT_FAM_MEMBERS"].replace(0, np.nan)
    # External-score summary (Step 1.35)
    out["EXT_SOURCE_MEAN"] = out[ext_cols].mean(axis=1)
    out["EXT_SOURCE_STD"] = out[ext_cols].std(axis=1)
    out["EXT_SOURCE_COUNT"] = out[ext_cols].notna().sum(axis=1)
    # Keep the outcome, the ID and the listed candidates; sex and marital status are not in the lists
    keep = ["TARGET", "SK_ID_CURR"] + BASE_NUMERIC_COLS + CATEGORICAL_COLS + ENGINEERED_NUMERIC + EDA_ONLY_COLS
    return out[keep]


# Run it on the full application table
feats = wrangle(app)
# Free the memory held by tables no longer needed
del draft
print(type(feats).__name__, feats.shape)
feats[["AGE_YEARS", "EMPLOYED_YEARS", "CREDIT_INCOME_RATIO", "EXT_SOURCE_MEAN"]].describe().round(2)
""")

code("""
# Check your work: 37 columns, the prohibited bases absent, and identical to the pipeline's tested function
from baseline_features import engineer_baseline

# Stop the run if this does not hold
assert feats.shape == (307_511, 37), feats.shape
assert not set(PROHIBITED_BASES) & set(feats.columns), "a prohibited basis reached the feature table"
pd.testing.assert_frame_equal(feats, engineer_baseline(app))
# Say that the two functions agreed (the assert above would have stopped the run otherwise)
print("wrangle(app) equals engineer_baseline(app)")
""")

step("1.37", "Attach the credit history",
     "Join the history table from Step 1.29. Applicants with no history keep their row.",
     "what is a left join?",
     "A join that keeps every row of the left table (here, every applicant) and attaches matching rows from the "
     "right table where they exist.")

code("""
# Every applicant keeps their row; history columns are missing where there is no history
full = feats.merge(history, on="SK_ID_CURR", how="left")
n_numeric = len(BASE_NUMERIC_COLS) + len(ENGINEERED_NUMERIC) + len(RELATIONAL_NUMERIC)
print(type(full).__name__, full.shape)
# Print how many numeric and categorical candidates the model starts from
print(f"{n_numeric} numeric and {len(CATEGORICAL_COLS)} categorical candidates")
""")

code("""
# Check your work: the join added columns, not rows
assert full.shape == (307_511, 65), full.shape
assert full["SK_ID_CURR"].is_unique
assert n_numeric + len(CATEGORICAL_COLS) == 62
""")

md("""
## 1E. Split
""")

step("1.38", "Separate the outcome from the characteristics",
     "`y` is what the model predicts; `X` is what it may use.",
     "what is the feature matrix?",
     "The table of inputs, one row per applicant and one column per characteristic. The ID and the outcome are "
     "removed so the model cannot read the answer.")

code("""
# The outcome
y = full["TARGET"]
# The inputs, without the outcome or the ID
X = full.drop(columns=["TARGET", "SK_ID_CURR"])
print(type(X).__name__, X.shape, "|", type(y).__name__, y.shape)
""")

code("""
# Check your work
assert "TARGET" not in X.columns and "SK_ID_CURR" not in X.columns
assert X.shape == (307_511, 63), X.shape
""")

step("1.39", "Hold back 20% of applicants for the final test",
     "Split at random, stratified so both parts have the same default rate. The model never sees the holdout until "
     "Step 2.22.",
     "what is a holdout, and why stratify?",
     "Applicants kept aside to test the finished model, so its score is measured on people it has never seen. "
     "Stratifying keeps the 8% default rate the same in both parts, so neither part is luckier than the other.")

code("""
# 80% to build the model, 20% to test it, the same default rate in both
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(type(X_train).__name__, X_train.shape, X_val.shape)
print(f"training: {len(X_train):,} applicants, default rate {y_train.mean():.2%}")
# Print the size and default rate of the holdout
print(f"holdout:  {len(X_val):,} applicants, default rate {y_val.mean():.2%}")
""")

code("""
# Check your work: an 80/20 split with the same default rate on both sides
assert len(X_train) == 246_008 and len(X_val) == 61_503, (len(X_train), len(X_val))
assert abs(y_train.mean() - y_val.mean()) < 1e-3
""")

md("""
**Part 1 in one line:** 307,511 applicants, 8% defaults, one placeholder found and fixed, prohibited bases excluded,
credit history added from six tables, 62 candidate characteristics, and 61,503 applicants held back for testing.
""")

# =====================================================================
# PART 2: BUILD MODEL
# =====================================================================
md("""
---
# Part 2: Build Model

## 2A. Baseline

Every model needs a bar to clear. Two are set here: a naive one that knows nothing, and a sensible one built from
the application table alone.
""")

step("2.1", "Measure a naive baseline",
     "Give every holdout applicant the same score, and also count how \"accurate\" approving everyone would be.",
     "what is AUC?",
     "Pick one defaulter and one repayer at random. AUC is the chance the model gives the defaulter the riskier "
     "score. 0.5 is a coin toss; 1.0 separates every pair. It measures ranking, which is what a lender needs.")

code("""
# AUC of a score that is the same for everyone
naive_auc = roc_auc_score(y_val, np.zeros(len(y_val)))
# "Accuracy" of approving everyone: the share who repaid
naive_accuracy = 1 - y_val.mean()
print(type(naive_auc).__name__, f"AUC of a constant score: {naive_auc:.4f}")
print(f"accuracy of approving everyone: {naive_accuracy:.1%}")
""")

code("""
# Check your work
assert naive_auc == 0.5, naive_auc
assert round(naive_accuracy, 3) == 0.919, naive_accuracy
""")

md("""
A constant score has an AUC of 0.5 and approving everyone is 91.9% "accurate", which shows why accuracy is not the
measure: it praises a model that cannot tell anyone apart.
""")

step("2.2", "See why the raw table cannot go straight into a model",
     "Try to fit a plain logistic regression on two raw columns, one of which holds text. This is a common first "
     "error, so it is shown on purpose.",
     "what is logistic regression?",
     "A model that adds up weighted inputs and turns the total into a probability between 0 and 1. It needs "
     "numbers, with no gaps.")

code("""
# Two raw columns from the first 1,000 training rows: one numeric, one text
trap_X = X_train[["AMT_CREDIT", "NAME_CONTRACT_TYPE"]].head(1000)
trap_y = y_train.head(1000)
# Try to fit and print the error instead of stopping
try:
    LogisticRegression().fit(trap_X, trap_y)
    trap_error = None
except ValueError as error:
    trap_error = str(error)
print(type(trap_X).__name__, trap_X.shape)
print("error:", trap_error)
""")

code("""
# Check your work: the fit failed on the text column
assert trap_error is not None and "could not convert string to float" in trap_error, trap_error
""")

md("""
**The fix.** Text has to be turned into numbers (one-hot encoding gives each label its own 0/1 column) and gaps
have to be filled or binned. The benchmark in the next step does the first; the scorecard does both through Weight of
Evidence (Step 2.4).
""")

step("2.3", "Fit the application-only benchmark",
     "`build_pipeline` in `src/baseline_model.py` fills missing values with the median, scales the numbers, "
     "one-hot encodes the text, and fits a logistic regression on the application table alone.",
     "what is a benchmark model?",
     "A simple, standard model the real one must beat. If the scorecard with credit history cannot beat the "
     "application table alone, the extra work bought nothing.")

code("""
# The benchmark pipeline: impute, scale, encode, then a balanced logistic regression
from baseline_model import build_pipeline

# An unfitted copy of the benchmark
benchmark = build_pipeline()
# It picks its own application-form columns by name
benchmark.fit(X_train, y_train)
# Probability of default for each holdout applicant, and the AUC
benchmark_pd = benchmark.predict_proba(X_val)[:, 1]
baseline_auc = roc_auc_score(y_val, benchmark_pd)
print(type(benchmark_pd).__name__, benchmark_pd.shape)
# Print the benchmark's AUC on the holdout
print(f"application-only benchmark AUC: {baseline_auc:.4f}")
""")

code("""
# Check your work
assert round(baseline_auc, 4) == 0.7467, f"expected AUC 0.7467, got {baseline_auc:.4f}"
""")

md("""
That is the bar: **0.7467**. The scorecard adds credit history and must beat it while staying readable.

## 2B. Iterate: Weight of Evidence

A scorecard does not feed raw values to the model. It cuts each characteristic into bins and replaces each bin with
one number, its Weight of Evidence. The idea comes from credit-scoring practice; Naeem Siddiqi's *Credit Risk
Scorecards* (Wiley, 2006) is the standard reference.
""")

step("2.4", "Work out Weight of Evidence on a toy table",
     "1,000 made-up applicants in three bins.",
     "what are WoE and IV?",
     "WoE of a bin = ln(share of all repayers in the bin / share of all defaulters in the bin). Positive means "
     "safer than average, negative riskier, zero no information. Information Value (IV) adds up (share of repayers "
     "- share of defaulters) x WoE over the bins and measures how well a whole characteristic separates the two "
     "groups.")

code("""
# Three bins with their repayers and defaulters
toy = pd.DataFrame({"bin": ["low", "middle", "high"],
                    "repaid": [200, 400, 320],         # 920 repayers in total
                    "defaulted": [50, 25, 5]})         # 80 defaulters in total
# Each bin's share of all repayers and of all defaulters
toy["share_of_repayers"] = toy["repaid"] / toy["repaid"].sum()
toy["share_of_defaulters"] = toy["defaulted"] / toy["defaulted"].sum()
# WoE, and each bin's part of the IV
toy["woe"] = np.log(toy["share_of_repayers"] / toy["share_of_defaulters"])
toy["iv_part"] = (toy["share_of_repayers"] - toy["share_of_defaulters"]) * toy["woe"]
toy_iv = toy["iv_part"].sum()
# Print the type and shape of the toy table and its Information Value, then show the table
print(type(toy).__name__, toy.shape, f"| Information Value: {toy_iv:.3f}")
toy.round(3)
""")

code("""
# Check your work
assert round(toy_iv, 3) == 0.961, toy_iv
assert toy["woe"].iloc[0] < 0 < toy["woe"].iloc[2], "low should be risky and high safe"
""")

md("""
The "low" bin holds 22% of repayers but 62% of defaulters, so its WoE is strongly negative: risky. The "high" bin
holds 35% of repayers and only 6% of defaulters: safe. On real data, an IV below 0.02 is useless, 0.1 to 0.3 medium,
0.3 to 0.5 strong, and above 0.5 suspicious (check for leakage).
""")

step("2.5", "Cut a real characteristic into ten bins",
     "Take `EXT_SOURCE_MEAN` on the training applicants and cut it at its deciles. Missing values get a bin of "
     "their own.",
     "why open the first and last bins?",
     "A new applicant may score below the lowest or above the highest value seen in training. Open ends (minus "
     "and plus infinity) make sure every value lands in a bin.")

code("""
# The characteristic, on the training applicants only
col = "EXT_SOURCE_MEAN"
x = X_train[col]
# Decile cut points, duplicates removed
edges = np.unique(np.quantile(x.dropna().astype(float), np.linspace(0, 1, 11)))
# Open the first and last bins
edges[0], edges[-1] = -np.inf, np.inf
# Each applicant's bin as text, with "Missing" for a missing score
bins = pd.cut(x, bins=edges, include_lowest=True).astype(str).where(x.notna(), "Missing")
print(type(bins).__name__, len(bins), "|", len(edges) - 1, "bins plus Missing")
bins.value_counts().head()
""")

code("""
# Check your work
assert len(edges) == 11 and (bins == "Missing").sum() == 139
""")

step("2.6", "Count repayers and defaulters in each bin and compute WoE by hand",
     "Add a small constant (0.5) to every count so an empty bin never gives ln(0).",
     "what is smoothing?",
     "Adding a small number to every count before dividing. It changes the result very little but keeps a bin "
     "with no defaulters from producing an infinite WoE.")

code("""
# Applicants and defaulters per bin
woe_table = pd.DataFrame({"bin": bins, "y": y_train}).groupby("bin")["y"].agg(n="count", bad="sum")
woe_table["good"] = woe_table["n"] - woe_table["bad"]
# Smoothed shares of repayers and defaulters
eps = 0.5
k = len(woe_table)
dist_good = (woe_table["good"] + eps) / (woe_table["good"].sum() + eps * k)
# The same for the defaulters
dist_bad = (woe_table["bad"] + eps) / (woe_table["bad"].sum() + eps * k)
# WoE and IV
woe_table["woe"] = np.log(dist_good / dist_bad)
iv_by_hand = ((dist_good - dist_bad) * woe_table["woe"]).sum()
print(type(woe_table).__name__, woe_table.shape, f"| IV of {col}, by hand: {iv_by_hand:.4f}")
# Display the table
woe_table.round(4)
""")

code("""
# Check your work
assert round(iv_by_hand, 4) == 0.6091, iv_by_hand
""")

step("2.7", "Compare with the tested `fit_woe`",
     "`fit_woe` in `src/woe_iv.py` does the same calculation for the pipeline.",
     "why check a hand calculation against a library function?",
     "If both agree on every bin, the hand version proves you understand the function, and the function's tests "
     "prove the hand version is right.")

code("""
# The tested function, ten decile bins
from woe_iv import fit_woe

# The tested function on the same column and the same ten bins
reference = fit_woe(X_train[col], y_train, is_categorical=False, n_bins=10)
# Print the type and keys of the tested function's result, then show its bin table
print(type(reference).__name__, sorted(reference))
reference["table"][["n", "bad", "woe"]].round(4)
""")

code("""
# Check your work: the hand calculation and the tested function agree on every bin
assert np.allclose(woe_table.loc[reference["table"].index, "woe"], reference["table"]["woe"])
assert abs(iv_by_hand - reference["iv"]) < 1e-12
""")

code("""
# Draw the WoE by bin: red below zero (riskier), blue above (safer)
fig, ax = plt.subplots(figsize=(10, 4))
reference["table"]["woe"].plot(kind="bar", ax=ax, color=bar_colours(reference["table"]["woe"]), rot=45)
ax.axhline(0, color=INK, lw=1)
# Name the x and y axes and title the chart
ax.set_xlabel("Bin of EXT_SOURCE_MEAN (training applicants)")
ax.set_ylabel("Weight of Evidence")
ax.set_title("WoE climbs steadily from the riskiest bin to the safest")
# Fit the layout, save the chart as figs/woe_ext_source_mean.png and show it
plt.tight_layout()
plt.savefig("figs/woe_ext_source_mean.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
The WoE climbs steadily from the lowest band to the highest, the pattern of Step 1.19 on a scale the model can use.
The Missing bin is only slightly riskier than average, far from the riskiest band: having no external score
says little on its own.
""")

step("2.8", "Find the characteristics whose bins zig-zag",
     "Fit ten-bin WoE for every numeric candidate and list those whose WoE does not move in one direction, "
     "strongest first.",
     "why does a zig-zag matter?",
     "If the WoE goes up, down, then up again, the scorecard awards fewer points to a safer bin than to the riskier "
     "bin next to it. That is hard to defend to a credit committee and looks arbitrary to an applicant.")

code("""
# The tested check for a WoE that moves one way
from woe_iv import is_monotonic_woe

# Ten-bin WoE for every numeric candidate
decile_fits = {}
for decile_col in MODEL_NUMERIC:
    decile_fits[decile_col] = fit_woe(X_train[decile_col], y_train, n_bins=10)


# A sort key, written out so the ordering is plain to read
def minus_iv(name):
    # Sort key: the largest IV first
    return -decile_fits[name]["iv"]


# The characteristics that zig-zag, strongest IV first
zigzag = []
for zigzag_col in sorted(decile_fits, key=minus_iv):
    if not is_monotonic_woe(decile_fits[zigzag_col]):
        zigzag.append(zigzag_col)
print(type(zigzag).__name__, len(zigzag))
# Print how many numeric characteristics zig-zag and which one is strongest
print(f"numeric characteristics whose ten-bin WoE zig-zags: {len(zigzag)} of {len(MODEL_NUMERIC)}; strongest: {zigzag[0]}")
""")

code("""
# Check your work
assert len(zigzag) == 32, f"expected 32 zig-zagging characteristics, got {len(zigzag)}"
""")

step("2.9", "Fix the zig-zag with coarse classing",
     "`fit_monotone_bins` in `src/woe_iv.py` starts from the ten bins and merges neighbours until the default rate "
     "moves one way. A column that is mostly zero gets a bin of its own for 0 first.",
     "what is coarse classing?",
     "Merging neighbouring fine bins into wider ones. Here it uses the \"pool adjacent violators\" method: walk "
     "through the bins in order and merge any bin that goes against the overall direction with the one before it.")

code("""
# The strongest zig-zagging characteristic, binned again with coarse classing
example = zigzag[0]
coarse = fit_woe(X_train[example], y_train, n_bins=10, binning="monotone")
print(type(coarse).__name__, f"{example}: {len(decile_fits[example]['table'])} bins before, {len(coarse['table'])} after")
# Display the merged bins
coarse["table"][["n", "bad", "woe"]].round(4)
""")

code("""
# Check your work: monotonic after merging, and fewer bins
assert is_monotonic_woe(coarse) and not is_monotonic_woe(decile_fits[example])
assert len(coarse["table"]) < len(decile_fits[example]["table"])
""")

code("""
# Before and after, side by side (the Missing bin is left off the chart)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
for ax, fit, title in [(axes[0], decile_fits[example], f"{example}: ten bins"), (axes[1], coarse, f"{example}: after coarse classing")]:
    woe = fit["table"]["woe"].drop("Missing", errors="ignore")
    woe.plot(kind="bar", ax=ax, color=bar_colours(woe), rot=45)
    ax.axhline(0, color=INK, lw=1)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Weight of Evidence")
plt.tight_layout()
# Save the chart as figs/coarse_classing.png and show it
plt.savefig("figs/coarse_classing.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
After merging, every bin earns at least as many points as the riskier bin next to it. The cost is a little detail
(fewer bins) and so a little ranking power.
""")

step("2.10", "Read the rule that decided coarse classing",
     "`src/challengers.py` built both versions on the training applicants only, scored them on a validation slice of "
     "those applicants, and adopted coarse classing if it cost no more than 0.005 AUC there. Read the recorded "
     "result.",
     "why fix a decision rule before looking at results?",
     "If the rule is chosen after seeing the numbers, it is easy to choose the one that flatters the preferred "
     "answer. Fixing it first keeps the choice honest.")

code("""
# The challenger comparison written by src/challengers.py
challengers = json.load(open("outputs/challengers.json", encoding="utf-8"))
v = challengers["validation"]
print(type(challengers).__name__, sorted(challengers))
# Print the validation AUC with ten bins and with coarse classing, and what coarse classing cost
print(f"validation AUC, ten bins:        {v['champion']['auc']:.4f}   "
      f"({v['champion']['n_numeric_monotonic']} of {v['champion']['n_numeric_kept']} numeric characteristics monotonic)")
print(f"validation AUC, coarse classing: {v['coarse_classing']['auc']:.4f}   "
      f"({v['coarse_classing']['n_numeric_monotonic']} of {v['coarse_classing']['n_numeric_kept']} monotonic)")
print(f"cost: {challengers['coarse_classing_auc_loss_validation']:.4f} AUC   decision: {challengers['decision']}")
""")

code("""
# Check your work
assert challengers["decision"] == "adopt coarse classing"
assert challengers["coarse_classing_auc_loss_validation"] <= 0.005
""")

md("""
Coarse classing costs about 0.003 AUC and makes every numeric characteristic monotonic, so it was adopted. The rest
of the notebook (and the published scorecard) uses it.
""")

step("2.11", "Fit WoE for all 62 candidates",
     "54 numeric with coarse classing, and 8 categorical with one bin per level.",
     "how is a categorical characteristic binned?",
     "Each label is its own bin, so no merging is needed; the WoE simply compares the label's share of repayers "
     "with its share of defaulters.")

code("""
# The usual labels for IV strength
from woe_iv import iv_strength

# Coarse-classed WoE for each numeric candidate
woe_fits = {}
for woe_col in MODEL_NUMERIC:
    woe_fits[woe_col] = fit_woe(X_train[woe_col], y_train, is_categorical=False, n_bins=10, binning="monotone")
# One bin per level for each categorical candidate
for woe_col in MODEL_CATEGORICAL:
    woe_fits[woe_col] = fit_woe(X_train[woe_col], y_train, is_categorical=True)
print(type(woe_fits).__name__, len(woe_fits))
""")

code("""
# Check your work: 62 candidates, no prohibited basis, every numeric one monotonic
assert len(woe_fits) == 62 and not set(PROHIBITED_BASES) & set(woe_fits)
for check_col in MODEL_NUMERIC:
    assert is_monotonic_woe(woe_fits[check_col]), f"{check_col} is not monotonic"
""")

step("2.12", "Rank the candidates by information value",
     "Collect each candidate's IV and draw all 62.",
     "how strong is strong?",
     "The usual reading of IV: below 0.02 useless, 0.02 to 0.1 weak, 0.1 to 0.3 medium, 0.3 to 0.5 strong, above "
     "0.5 suspiciously strong (check that the characteristic was known when the application was made).")

code("""
# Each candidate's IV, weakest first
iv_values = {}
for iv_col in woe_fits:
    iv_values[iv_col] = woe_fits[iv_col]["iv"]
ivs = pd.Series(iv_values).sort_values()
# Print the type and length of the IV series and the strongest candidate
print(type(ivs).__name__, len(ivs))
print(f"strongest: {ivs.index[-1]} (IV {ivs.iloc[-1]:.3f}, {iv_strength(ivs.iloc[-1])})")
""")

code("""
# Check your work
assert ivs.index[-1] == "EXT_SOURCE_MEAN" and round(ivs.iloc[-1], 3) == 0.609
""")

code("""
# One bar per candidate: blue if it passes gate 1, grey if not
iv_colours = []
for iv_value in ivs:
    if iv_value >= IV_THRESHOLD:
        iv_colours.append(BLUE)
    else:
        iv_colours.append("#a0aec0")
# Make one tall figure with one set of axes
fig, ax = plt.subplots(figsize=(9, 13))
# Draw each candidate's IV as a bar, and mark the gate-1 threshold and the 0.10 medium-strength line
ivs.plot(kind="barh", ax=ax, color=iv_colours)
ax.axvline(IV_THRESHOLD, color=RED, ls="--", lw=1, label=f"gate 1: IV >= {IV_THRESHOLD}")
ax.axvline(0.1, color=INK, ls=":", lw=1, label="0.10: medium")
# Name the x axis, title the chart and add the legend for the two lines
ax.set_xlabel("Information value (training applicants)")
ax.set_title(f"All {len(ivs)} candidate characteristics by information value")
ax.legend(loc="lower right")
# Fit the layout, save the chart as figs/iv_all_candidates.png and show it
plt.tight_layout()
plt.savefig("figs/iv_all_candidates.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
The external scores lead by a distance; the grey bars at the bottom carry no usable signal. `EXT_SOURCE_MEAN` is
above the "check for leakage" line. Nothing in this notebook proves it is free of leakage: the data dictionary
describes each external score only as "Normalized score from external data source" and does not say when it was
obtained. The notebook assumes, as for the other application fields, that the three scores exist when the
application is made; the Limitations list this assumption.

## 2C. Iterate: three selection gates

Not all 62 candidates should enter the model. Each passes three gates, and each removal has a reason a reviewer can
check.
""")

step("2.13", "Sort the candidates by IV",
     "Highest first, with ties broken by name, so the gates always see the candidates in the same order.",
     "why does order matter?",
     "Gate 2 keeps the first of two near-duplicates it meets. Walking down in IV order means the stronger of the two "
     "is the one kept.")

code("""
# Pairs of (negative IV, name), so a plain sort puts the highest IV first and breaks ties by name
pairs = []
for pair_col in woe_fits:
    pairs.append((-ivs[pair_col], pair_col))
pairs.sort()
# The names in that order
iv_ranked = []
for neg_iv, ranked_col in pairs:
    iv_ranked.append(ranked_col)
print(type(iv_ranked).__name__, len(iv_ranked), iv_ranked[:5])
""")

code("""
# Check your work
assert len(iv_ranked) == 62 and iv_ranked[0] == "EXT_SOURCE_MEAN"
""")

step("2.14", "Apply gate 1: information value",
     "Drop characteristics whose IV is below the threshold: they carry almost no signal.",
     "why such a low threshold?",
     "Gate 1 is permissive on purpose. It only removes the useless; gates 2 and 3 deal with redundancy and with "
     "effects that flip.")

code("""
# Keep the candidates at or above the IV threshold, in IV order
iv_kept = []
for gate1_col in iv_ranked:
    if ivs[gate1_col] >= IV_THRESHOLD:
        iv_kept.append(gate1_col)
print(type(iv_kept).__name__, len(iv_kept))
# Print how many characteristics pass gate 1
print(f"Gate 1: {len(iv_kept)} of {len(iv_ranked)} characteristics have IV >= {IV_THRESHOLD}")
# The twelve strongest, with their strength
for top_col in iv_ranked[:12]:
    print(f"  {top_col:30s} IV={ivs[top_col]:.4f}  ({iv_strength(ivs[top_col])})")
""")

code("""
# Check your work
assert len(iv_kept) == 53, f"expected 53 after gate 1, got {len(iv_kept)}"
""")

step("2.15", "Spot an encoding trap",
     "Step 1.33 also made a flag, `DAYS_EMPLOYED_ANOM`, for the pensioner placeholder. Compare it with the "
     "`Missing` bin of `EMPLOYED_YEARS`.",
     "what is an encoding trap?",
     "Two columns that describe the same people in different ways. Given both, a model can split the effect "
     "between them with opposite signs, and one of them then seems to penalise a group it should not.")

code("""
# WoE of the flag, and of the Missing bin of years employed
flag_fit = fit_woe(X_train["DAYS_EMPLOYED_ANOM"], y_train, is_categorical=True)
emp_table = woe_fits["EMPLOYED_YEARS"]["table"]
# Are they the same applicants?
same_applicants = (X_train["DAYS_EMPLOYED_ANOM"] == 1).equals(X_train["EMPLOYED_YEARS"].isna())
print(type(flag_fit).__name__)
print(f"DAYS_EMPLOYED_ANOM = 1:    n={flag_fit['table'].loc['1', 'n']:,}  WoE={flag_fit['table'].loc['1', 'woe']:+.4f}")
# Print the Missing bin's size and WoE, and whether it holds the same applicants as the flag
print(f"EMPLOYED_YEARS = Missing:  n={emp_table.loc['Missing', 'n']:,}  WoE={emp_table.loc['Missing', 'woe']:+.4f}")
print(f"same applicants in both:   {same_applicants}")
""")

code("""
# Check your work: same applicants, and the flag is not a candidate
assert same_applicants and flag_fit["table"].loc["1", "n"] == 44_143
assert "DAYS_EMPLOYED_ANOM" not in woe_fits
""")

md("""
Same applicants, same WoE. Given both, the model could give them opposite signs, and one would then appear to
*penalise* pensioners, a lower-risk group: a wrong reason to give a declined applicant and a fair-lending problem.
So the flag is never offered to the model; pensioners are read only through `EMPLOYED_YEARS`.
""")

step("2.16", "Replace every value with its bin's WoE",
     "Build the WoE matrix for the 53 characteristics that passed gate 1.",
     "what does WoE encoding do?",
     "Every value, numeric or text, becomes one number on the same safe-to-risky scale. Text and missing values are "
     "handled at the same time, which fixes the trap of Step 2.2.")

code("""
# The tested function that maps values to their bin's WoE
from woe_iv import transform_woe


# A helper that turns raw characteristics into WoE columns
def woe_encode(frame, cols):
    # One column per characteristic, each value replaced by the WoE of its bin
    columns = []
    for encode_col in cols:
        woe_values = transform_woe(frame[encode_col], woe_fits[encode_col])
        columns.append(woe_values.to_numpy(dtype=float))
    return np.column_stack(columns)


# WoE-encode the training rows for the characteristics that passed gate 1
W_iv = woe_encode(X_train, iv_kept)
# Print the type and shape of the WoE matrix
print(type(W_iv).__name__, W_iv.shape)
print(f"WoE matrix: {W_iv.shape[0]:,} applicants x {W_iv.shape[1]} characteristics")
""")

code("""
# Check your work: one row per training applicant, one column per gate-1 survivor, no gaps
assert W_iv.shape == (246_008, 53) and not np.isnan(W_iv).any()
""")

step("2.17", "Look at the correlations among the strongest characteristics",
     "Draw the correlation of the WoE values among the 15 strongest.",
     "what is multicollinearity?",
     "Two inputs that move together almost perfectly say the same thing twice, and the model cannot tell which "
     "deserves the weight. Its coefficients then become unstable.")

code("""
# Correlation of every pair of WoE columns
corr = np.corrcoef(W_iv, rowvar=False)
# Each characteristic's column number in the matrix
pos = {}
for col_number, pos_col in enumerate(iv_kept):
    pos[pos_col] = col_number
print(type(corr).__name__, corr.shape)
""")

code("""
# Check your work
assert corr.shape == (53, 53) and np.allclose(np.diag(corr), 1.0)
""")

code("""
# Heat map of the 15 strongest
fig, ax = plt.subplots(figsize=(11, 9))
top15 = iv_kept[:15]
sns.heatmap(pd.DataFrame(corr[:15, :15], index=top15, columns=top15), vmin=-1, vmax=1, cmap="RdBu_r",
            annot=True, fmt=".2f", square=True, cbar=False, ax=ax, annot_kws={"size": 8})
# Title the chart, fit the layout and save it as figs/woe_correlation_top15.png
ax.set_title("Correlation of WoE values among the 15 strongest characteristics")
plt.tight_layout()
plt.savefig("figs/woe_correlation_top15.png", dpi=150, bbox_inches="tight")
# Show the chart
plt.show()
""")

md("""
The external-score summaries overlap with the scores they are built from, and the two region ratings and the credit
and goods-price columns are near-duplicates.
""")

step("2.18", "Apply gate 2: correlation",
     "Walk down the list in IV order and keep a characteristic only if its correlation with every one already kept "
     "is at most the threshold.",
     "why keep the higher-IV one of a pair?",
     "Of two characteristics that say the same thing, the one with more signal is the better messenger. The dropped "
     "one is named with its partner so a reviewer can check.")

code("""
# Gate 2 by hand: go through the characteristics from the highest IV down
corr_kept = []
corr_dropped = []
# Keep a characteristic unless an already-kept one says the same thing
for gate2_col in iv_kept:
    # The first already-kept characteristic this one is too correlated with, if any
    partner = None
    for kept_col in corr_kept:
        if abs(corr[pos[gate2_col], pos[kept_col]]) > CORRELATION_THRESHOLD:
            partner = kept_col
            break
    # Keep it, or record what it duplicates
    if partner is None:
        corr_kept.append(gate2_col)
    else:
        corr_dropped.append((gate2_col, partner, float(corr[pos[gate2_col], pos[partner]])))

# Print how many characteristics gate 2 dropped and kept, then each dropped one with its partner
print(type(corr_kept).__name__, len(corr_kept))
print(f"Gate 2: {len(corr_dropped)} dropped, {len(corr_kept)} kept")
for dropped_col, partner_col, r in corr_dropped:
    print(f"  dropped {dropped_col:28s} kept {partner_col:28s} r={r:+.3f}")
""")

code("""
# Check your work
assert len(corr_kept) == 47 and len(corr_dropped) == 6, (len(corr_kept), len(corr_dropped))
""")

step("2.19", "Scale the surviving columns",
     "Subtract each column's mean and divide by its standard deviation, so the solver treats every column alike.",
     "what is standardising?",
     "Rescaling a column to mean 0 and standard deviation 1. It does not change what the column says, only its "
     "units, and helps the solver converge.")

code("""
# Column numbers of the gate-2 survivors in the WoE matrix
surviving_columns = []
for surviving_col in corr_kept:
    surviving_columns.append(pos[surviving_col])
W_corr = W_iv[:, surviving_columns]
# Mean and standard deviation of each column; a constant column keeps a divisor of 1
m = W_corr.mean(axis=0)
s = W_corr.std(axis=0)
s[s == 0] = 1.0
# Each column on the same scale
Ws = (W_corr - m) / s
# Print the type and shape of the standardised WoE matrix
print(type(Ws).__name__, Ws.shape)
""")

code("""
# Check your work: every column now has mean about 0
assert Ws.shape == (246_008, 47) and np.allclose(Ws.mean(axis=0), 0, atol=1e-9)
""")

step("2.20", "Apply gate 3: the direction of each effect",
     "Higher WoE always means safer, so in a model of the chance of default every coefficient should be negative. "
     "Fit the model, remove the characteristic with the largest positive coefficient, refit, and repeat until "
     "every coefficient is negative.",
     "what does a positive coefficient mean here?",
     "That, once the other characteristics are in the model, this one's effect has flipped: the scorecard would "
     "hand points to its riskier bins. Removing it, rather than forcing its sign, keeps every remaining coefficient "
     "a genuine estimate.")

code("""
# find_worst_positive returns the characteristic with the largest positive coefficient and that coefficient
from fit_model import find_worst_positive

# Gate 3 by hand: start from the gate-2 survivors and refit until no sign is wrong
sign_kept = list(corr_kept)
sign_removed = []
# Repeat: fit, find the worst wrong sign, remove it
while True:
    # Column numbers of the characteristics still in
    cols_now = []
    for still_in in sign_kept:
        cols_now.append(corr_kept.index(still_in))
    # A balanced logistic regression on those columns
    fit = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Ws[:, cols_now], y_train)
    # Stop when no coefficient is positive; otherwise remove the worst and refit
    worst_name, worst_coef = find_worst_positive(sign_kept, fit.coef_.ravel())
    if worst_name is None:
        break
    sign_removed.append((worst_name, worst_coef))
    sign_kept.remove(worst_name)

# Print how many characteristics gate 3 removed and kept, then each removed one with its coefficient
print(type(sign_kept).__name__, len(sign_kept))
print(f"Gate 3: {len(sign_removed)} removed, {len(sign_kept)} kept")
for removed_col, removed_coef in sign_removed:
    print(f"  removed {removed_col:28s} coefficient={removed_coef:+.4f}")
""")

code("""
# Check your work: age is among the removed, and every remaining coefficient is negative
removed_names = []
for removed_record in sign_removed:
    removed_names.append(removed_record[0])
assert "AGE_YEARS" in removed_names
# Stop the run if this does not hold
assert (fit.coef_ < 0).all()
""")

md("""
`AGE_YEARS` is among those removed. On its own, older applicants default less (Step 1.14), but once correlated
characteristics such as employment length are in the model, age's coefficient flips (the notebook does not isolate
which one causes it). Kept, it would have given the oldest applicants
*negative* points, which Regulation B forbids for applicants aged 62 or over (12 CFR 1002.6(b)(2)).

`class_weight="balanced"` makes the 8% of defaulters count as much as the 92% of repayers while fitting; Step 2.26
corrects the effect this has on the intercept.

## 2D. Iterate: the final model and its points
""")

step("2.21", "Fit the final model with the project's own solver",
     "`src/from_scratch_lr.py` fits logistic regression by plain gradient descent. It treats the weights slightly "
     "differently from scikit-learn, so the sign check runs again on this fit.",
     "what is gradient descent?",
     "A way to fit a model by taking many small steps downhill on its error, each step moving the coefficients in "
     "the direction that reduces the error most, until a step no longer helps.")

code("""
# The hand-written solver and the hand-written AUC
from from_scratch_lr import FromScratchLogisticRegression
from metrics_scratch import auc_rank_sum

# The final fit: start from the gate-3 survivors and stop when every sign is right
kept_cols = list(sign_kept)
# Repeat: encode, scale, fit, and drop the worst wrong sign until none is left
while True:
    # WoE matrices for training and holdout
    Xw_train = woe_encode(X_train, kept_cols)
    Xw_val = woe_encode(X_val, kept_cols)
    # Scale both on the training statistics only
    mean = Xw_train.mean(axis=0)
    std = Xw_train.std(axis=0)
    std[std == 0] = 1.0
    Xw_train_s = (Xw_train - mean) / std
    Xw_val_s = (Xw_val - mean) / std
    # Fit by gradient descent with balanced class weights
    scratch = FromScratchLogisticRegression(lr=0.5, n_iter=3000, l2=1e-3)
    scratch.fit(Xw_train_s, y_train.to_numpy(), class_weight="balanced")
    # Same sign rule as gate 3
    worst_name, worst_coef = find_worst_positive(kept_cols, scratch.coef_)
    if worst_name is None:
        break
    sign_removed.append((worst_name, worst_coef))
    kept_cols.remove(worst_name)

# Print the type and coefficient shape of the fitted model, and how many characteristics it keeps
print(type(scratch).__name__, scratch.coef_.shape)
print(f"characteristics in the scorecard: {len(kept_cols)} of {len(woe_fits)}")
""")

code("""
# Check your work: 32 characteristics, every sign right, the solver converged
assert len(kept_cols) == 32, len(kept_cols)
assert (scratch.coef_ < 0).all() and scratch.converged_
""")

step("2.22", "Score the holdout and compare with the benchmark",
     "Compute the holdout AUC with the project's own AUC function.",
     "why measure on the holdout?",
     "Every choice so far used training applicants only. The holdout shows how the model ranks people it has never "
     "seen, which is what a lender will face.")

code("""
# Probability of default for each holdout applicant, and the AUC
holdout_pd = scratch.predict_proba(Xw_val_s)
full_auc = auc_rank_sum(y_val, holdout_pd)
print(type(holdout_pd).__name__, holdout_pd.shape)
# Print the holdout AUC next to the benchmark's
print(f"holdout AUC: {full_auc:.4f}  (application-only benchmark: {baseline_auc:.4f})")
""")

code("""
# Check your work: it beats the benchmark
assert full_auc > baseline_auc
assert round(full_auc, 4) == 0.7540, full_auc
""")

step("2.23", "Confirm the pipeline builds the same model",
     "`build_validation_scores` in `src/run_pipeline.py` packages every step so far and writes the published "
     "results. Run it.",
     "what is a pipeline?",
     "One function that runs every step in order, so the published numbers come from code, not from a notebook "
     "someone ran by hand.")

code("""
# The pipeline's own fit and holdout summary
from run_pipeline import build_validation_scores, summarise

# result: the fitted model and gates; sc: the scorecard; Xf_val, yf_val: the holdout; scores: holdout points
result, sc, Xf_val, yf_val, scores = build_validation_scores(full, verbose=False)
print(type(result).__name__, type(sc).__name__, type(scores).__name__, scores.shape)
""")

code("""
# Check your work: the step-by-step build and the pipeline agree exactly
assert result["kept_cols"] == kept_cols
pipeline_dropped = []
for dropped_record in result["correlation_gate_dropped"]:
    pipeline_dropped.append(dropped_record[:2])
# The same pairs from the hand-built gate 2
notebook_dropped = []
for dropped_record in corr_dropped:
    notebook_dropped.append(dropped_record[:2])
# Stop the run if this does not hold
assert pipeline_dropped == notebook_dropped
pipeline_removed = []
for removed_record in result["sign_gate_removed"]:
    pipeline_removed.append(removed_record[0])
# The same names from the hand-built gate 3
notebook_removed = []
for removed_record in sign_removed:
    notebook_removed.append(removed_record[0])
# Stop the run if this does not hold
assert pipeline_removed == notebook_removed
assert np.allclose(result["model"].coef_, scratch.coef_)
assert abs(result["val_auc"] - full_auc) < 1e-12
# Say that the step-by-step build matched the pipeline (the asserts above would have stopped the run otherwise)
print("step-by-step build matches the pipeline: same characteristics, coefficients and AUC")
""")

step("2.24", "Read the coefficients",
     "Every one is negative (more WoE, less risk), and their sizes say which characteristics move the score most "
     "once the others are held constant.",
     "what does a coefficient's size mean?",
     "On standardised inputs, a larger coefficient (in absolute value) means the characteristic shifts the "
     "predicted risk more for the same change in its WoE.")

code("""
# Coefficients by characteristic, most negative first
coefficients = pd.Series(scratch.coef_, index=kept_cols).sort_values()
print(type(coefficients).__name__, len(coefficients))
coefficients.head().round(4)
""")

code("""
# Check your work
assert (coefficients < 0).all() and len(coefficients) == 32
""")

code("""
# Draw all 32
fig, ax = plt.subplots(figsize=(9, 8))
coefficients.plot(kind="barh", ax=ax, color=BLUE)
ax.axvline(0, color=INK, lw=1)
# Name the x axis, title the chart and fit the layout
ax.set_xlabel("Coefficient on standardised WoE (all negative: more WoE, lower risk)")
ax.set_title(f"The {len(kept_cols)} characteristics in the scorecard")
plt.tight_layout()
# Save the chart as figs/coefficients.png and show it
plt.savefig("figs/coefficients.png", dpi=150, bbox_inches="tight")
plt.show()
""")

step("2.25", "Convert the coefficients back to plain WoE units",
     "The model was fitted on scaled WoE (Step 2.21). Undo the scaling.",
     "why undo the scaling?",
     "The points table multiplies each coefficient by a bin's WoE. That only works if the coefficient is in WoE "
     "units, not in standard deviations.")

code("""
# Coefficients per unit of WoE
coef_raw = scratch.coef_ / std
# The intercept moves by the part the scaling had absorbed
intercept_balanced = scratch.intercept_ - np.sum(scratch.coef_ * mean / std)
print(type(coef_raw).__name__, coef_raw.shape, f"| intercept on plain WoE: {intercept_balanced:+.4f}")
""")

code("""
# Check your work: matches the pipeline's conversion
assert np.allclose(coef_raw, sc["coef_raw"]) and abs(intercept_balanced - sc["intercept_balanced"]) < 1e-12
""")

step("2.26", "Correct the intercept to the real default rate",
     "The balanced weighting fitted the model as if half of all applicants defaulted. Add ln(p / (1 - p)) for the "
     "real default rate p, minus the same for 50%.",
     "what is a prior correction?",
     "A shift of the intercept that moves predicted probabilities back to the real default rate without changing "
     "any coefficient. The ranking (and so the AUC) is untouched, but the probabilities become trustworthy.")

code("""
# A helper: probability to log-odds
def logit(p):
    # Log-odds of a probability
    return np.log(p / (1 - p))


# Shift from a 50% prior to the training default rate
intercept = intercept_balanced + logit(y_train.mean()) - logit(0.5)
print(type(intercept).__name__, f"training default rate {y_train.mean():.4f}: intercept {intercept_balanced:+.4f} -> {intercept:+.4f}")
""")

code("""
# Check your work: matches the pipeline
assert abs(intercept - sc["intercept"]) < 1e-12
assert round(intercept, 4) == -2.4379
""")

step("2.27", "Choose the points scale",
     "Two conventions fix it: 600 points means odds of 20 repayers to 1 defaulter, and every 40 points doubles the "
     "odds. Then factor = 40 / ln(2), offset = 600 - factor x ln(20), and base points = offset - factor x intercept.",
     "what is PDO?",
     "\"Points to double the odds\": the number of extra points that doubles an applicant's good-to-bad odds. "
     "Here it is 40, so odds of 20:1 at 600 points become 40:1 at 640 and 80:1 at 680. It is a lender's convention, "
     "not a statistic; 20:1 at 600 and a PDO of 40 are common choices.")

code("""
# Points per unit of log-odds
factor = 40 / np.log(2)
# The constant that puts 20:1 odds at 600 points
offset = 600 - factor * np.log(20)
# The points every applicant starts with
base_points = offset - factor * intercept
print(type(factor).__name__, f"factor {factor:.2f}   offset {offset:.1f}   base points {base_points:.1f}")
""")

code("""
# Check your work: the pipeline uses the same scale
assert np.isclose(factor, sc["factor"]) and np.isclose(offset, sc["offset"]) and np.isclose(base_points, sc["base_points"])
""")

step("2.28", "Build the points table for the strongest characteristic",
     "Points for a bin = -factor x coefficient x WoE of the bin.",
     "why the minus sign?",
     "The coefficient is negative and the scale rewards safety, so a safer bin (positive WoE) must earn positive "
     "points. The minus sign turns negative x positive into a positive number.")

code("""
# The first kept characteristic and its bins
top = kept_cols[0]
points_table = woe_fits[top]["table"][["n", "bad", "woe"]].copy()
# Default rate and points per bin
points_table["default_rate"] = points_table["bad"] / points_table["n"]
points_table["points"] = -factor * coef_raw[0] * points_table["woe"]
print(type(points_table).__name__, points_table.shape, "|", top)
# Display the table
points_table.round(3)
""")

code("""
# Check your work: the same points as the pipeline's table
assert np.allclose(points_table["points"], sc["points_tables"][top])
""")

code("""
# Points by bin for the three strongest characteristics
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, chart_col in zip(axes, kept_cols[:3]):
    points = sc["points_tables"][chart_col]
    points.plot(kind="bar", ax=ax, color=bar_colours(points), rot=60)
    ax.axhline(0, color=INK, lw=1)
    ax.set_title(chart_col)
    ax.set_xlabel("")
    ax.set_ylabel("Points")
plt.tight_layout()
# Save the chart as figs/points_by_bin.png and show it
plt.savefig("figs/points_by_bin.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
Safer bins (lower default rate) earn more points, bin by bin, because the WoE is monotonic and the coefficient is
negative. A credit committee reviews a table like this for every characteristic before a scorecard goes live; all of
them are in `outputs/scorecard_points.json`.
""")

step("2.29", "Score one real holdout applicant by hand",
     "Base points plus the points of each of their bins.",
     "why score one applicant by hand?",
     "It proves the points table is the whole model: nothing hidden, every point traceable to a bin.")

code("""
# The public function that gives a value's bin label, the key of the points table
from woe_iv import bin_labels

# The first holdout applicant
applicant = Xf_val.iloc[0]
total = base_points
for score_col in kept_cols:
    # The bin this applicant falls in
    score_fit = woe_fits[score_col]
    label = bin_labels(pd.Series([applicant[score_col]]), score_fit).iloc[0]
    # Add that bin's points; a category never seen in training scores 0
    total += sc["points_tables"][score_col].get(label, 0.0)
print(type(applicant).__name__, f"applicant {applicant.name}: {total:.1f} points")
""")

code("""
# Check your work: the hand total equals the pipeline's score for this applicant (scores are kept within 300 to 850)
assert np.isclose(min(max(total, 300), 850), scores.iloc[0])
assert round(total, 1) == 608.4
""")

md("""
## 2E. Evaluate
""")

step("2.30", "Test the measuring tools first",
     "AUC and KS are written from scratch in `src/metrics_scratch.py`. Check them against scikit-learn and scipy on "
     "made-up scores with ties.",
     "what is KS?",
     "Sort applicants by score. KS (Kolmogorov-Smirnov) is the largest gap between the share of defaulters and the "
     "share of repayers scoring below any cut-off: the credit industry's standard measure of separation.")

code("""
# KS by hand and from scipy, to compare
from scipy.stats import ks_2samp

# The hand-written KS from src/metrics_scratch.py
from metrics_scratch import ks_statistic

# 5,000 made-up outcomes and scores; rounding creates tied scores
rng = np.random.default_rng(0)
y_check = rng.integers(0, 2, 5000)
score_check = np.round(rng.normal(0, 1, 5000) + y_check * 0.8, 2)
# Each metric two ways
auc_mine = auc_rank_sum(y_check, score_check)
auc_sklearn = roc_auc_score(y_check, score_check)
ks_mine = ks_statistic(y_check, score_check)
# KS from scipy, the reference
ks_scipy = ks_2samp(score_check[y_check == 1], score_check[y_check == 0]).statistic
# Print the type and shape of the scores, then AUC and KS by hand next to the library values
print(type(score_check).__name__, score_check.shape)
print(f"AUC: from scratch {auc_mine:.6f}, scikit-learn {auc_sklearn:.6f}")
print(f"KS:  from scratch {ks_mine:.6f}, scipy {ks_scipy:.6f}")
""")

code("""
# Check your work: both agree to floating-point precision
assert abs(auc_mine - auc_sklearn) < 1e-9 and abs(ks_mine - ks_scipy) < 1e-9
""")

step("2.31", "Measure the scorecard on the holdout",
     "`summarise` in `src/run_pipeline.py` computes every holdout metric the README reports.",
     "what is a bootstrap interval?",
     "Resample the holdout with replacement many times and recompute AUC each time. The middle 95% of those values "
     "shows how far the AUC could move with a different sample of applicants.")

code("""
# Every holdout metric, as the pipeline computes it
summary = summarise(result, sc, Xf_val, yf_val, scores)
lo, hi = summary["val_auc_ci95_bootstrap"]
print(type(summary).__name__, len(summary), "entries")
# Print the AUC with its interval, the KS and the average score of each group
print(f"AUC  {summary['val_auc']:.4f}  (95% bootstrap interval {lo:.4f} to {hi:.4f})")
print(f"KS   {summary['val_ks']:.4f}")
print(f"average score, repaid: {summary['mean_score_repaid']:.1f}   defaulted: {summary['mean_score_defaulted']:.1f}")
# Print how closely the from-scratch fit matches scikit-learn
print(f"from-scratch fit vs scikit-learn: prediction correlation {result['pred_corr']:.6f}")
""")

code("""
# Check your work
assert round(summary["val_auc"], 3) == 0.754 and round(summary["val_ks"], 3) == 0.381
assert round(lo, 4) == 0.7473 and round(hi, 4) == 0.7613
""")

md("""
The scorecard reaches an AUC of 0.754: given one defaulter and one repayer, it scores the defaulter lower about three
times in four. The bootstrap interval (0.7473 to 0.7613) shows how far that figure could move with a different
sample of 61,503 applicants. KS is 0.381: at the best cut-off, 38 percentage points more of the defaulters than of the
repayers fall below it. Repayers average 594 points and defaulters 542. The from-scratch fit agrees with
scikit-learn's to six decimal places, so the hand-written solver is not the weak link.

**Is the lift over the benchmark real?** It is tempting to say yes because the scorecard's interval sits above the
benchmark's 0.7467, but that compares an interval with a single number. Both models were scored on the same
61,503 applicants, so a resample that is hard for one is usually hard for the other. The fair test resamples once,
scores both models on the same rows, and keeps the difference; then repeats.
""")

code("""
# Both models' predicted PDs on the holdout, in the same row order (both were scored on X_val)
scorecard_pd_val = np.asarray(holdout_pd)
benchmark_pd_val = np.asarray(benchmark_pd)
y_val_array = y_val.to_numpy()
# One resample by hand: 61,503 row numbers drawn with replacement
rng_demo = np.random.default_rng(SEED)
rows = rng_demo.integers(0, len(y_val_array), len(y_val_array))
# Both AUCs on exactly those rows, and the difference
one_scorecard = auc_rank_sum(y_val_array[rows], scorecard_pd_val[rows])
one_benchmark = auc_rank_sum(y_val_array[rows], benchmark_pd_val[rows])
# Print the type and shape of one resample and the two AUCs on it
print(type(rows).__name__, rows.shape)
print(f"one resample: scorecard {one_scorecard:.4f}, benchmark {one_benchmark:.4f}, "
      f"difference {one_scorecard - one_benchmark:.4f}")
""")

code("""
# Check your work: the resample has as many rows as the holdout, and each AUC is a valid AUC
assert len(rows) == len(y_val_array) == 61_503
assert round(one_scorecard, 4) == 0.7484 and round(one_benchmark, 4) == 0.7373
""")

code("""
# The tested function repeats that 500 times with the pipeline's seed and keeps the middle 95%
from metrics_scratch import bootstrap_auc_difference

# The lift and its paired interval
auc_lift, lift_lo, lift_hi = bootstrap_auc_difference(y_val_array, scorecard_pd_val, benchmark_pd_val,
                                                      n_boot=500, seed=SEED)
# Print the type of the lift, then the lift and its 95% interval
print(type(auc_lift).__name__)
print(f"scorecard minus benchmark: {auc_lift:.4f} AUC (95% paired bootstrap interval {lift_lo:.4f} to {lift_hi:.4f})")
""")

code("""
# Check your work: the point estimate is the plain difference of the two AUCs, inside its own interval
assert abs(auc_lift - (full_auc - baseline_auc)) < 1e-12
assert lift_lo < auc_lift < lift_hi
assert round(auc_lift, 4) == 0.0073 and round(lift_lo, 4) == 0.0047 and round(lift_hi, 4) == 0.0101
""")

md("""
The scorecard is 0.0073 AUC above the benchmark, and the paired interval, 0.0047 to 0.0101, stays above zero: the
lift is real, but small. Most of the ranking power is already in the application data, above all in the external
scores, which are third-party model outputs whose construction is not published: the scorecard leans on outside
vendors' models. So the case for the scorecard over the benchmark is a stated reason for every decline, points that move one
way, and control over which characteristics may enter, not better ranking.
""")

step("2.32", "Draw the ROC curve and the KS gap",
     "Prepare the two curves first, then draw them.",
     "what is a ROC curve?",
     "For every possible cut-off, the share of defaulters caught against the share of repayers wrongly caught. The "
     "further it bows above the diagonal, the better the ranking; the area under it is the AUC.")

code("""
# ROC points from the holdout probabilities
fpr, tpr, _ = roc_curve(yf_val, holdout_pd)
# Applicants sorted by score, lowest first
order = np.argsort(scores.to_numpy())
sorted_scores = scores.to_numpy()[order]
sorted_y = yf_val.to_numpy()[order]
# Cumulative share of defaulters and of repayers at or below each score
cum_defaulters = np.cumsum(sorted_y) / sorted_y.sum()
cum_repayers = np.cumsum(1 - sorted_y) / (1 - sorted_y).sum()
# Where the gap is largest
ks_at = int(np.argmax(cum_defaulters - cum_repayers))
ks_gap = cum_defaulters[ks_at] - cum_repayers[ks_at]
print(type(sorted_scores).__name__, sorted_scores.shape, f"| KS = {ks_gap:.3f} at {sorted_scores[ks_at]:.0f} points")
""")

code("""
# Check your work: the gap drawn matches the reported KS to the three decimals the chart prints
assert round(ks_gap, 3) == round(summary["val_ks"], 3)
""")

code("""
# Two panels side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Left: the ROC curve against a coin toss
axes[0].plot(fpr, tpr, color=BLUE, lw=2, label=f"scorecard (AUC {summary['val_auc']:.3f})")
axes[0].plot([0, 1], [0, 1], color="#a0aec0", ls="--", lw=1, label="constant score (AUC 0.5)")
axes[0].set_xlabel("Share of repayers scored below the cut-off")
# Left panel: name the y axis, add a title and the legend
axes[0].set_ylabel("Share of defaulters scored below the cut-off")
axes[0].set_title("ROC curve on the holdout")
axes[0].legend(loc="lower right")
# Right: the two cumulative distributions and their largest gap
axes[1].plot(sorted_scores, cum_defaulters, color=RED, lw=2, label="defaulters")
axes[1].plot(sorted_scores, cum_repayers, color=BLUE, lw=2, label="repayers")
axes[1].vlines(sorted_scores[ks_at], cum_repayers[ks_at], cum_defaulters[ks_at], color=INK, ls="--",
               label=f"KS = {cum_defaulters[ks_at] - cum_repayers[ks_at]:.3f} at {sorted_scores[ks_at]:.0f} points")
# Right panel: name the x and y axes and add a title
axes[1].set_xlabel("Scorecard points")
axes[1].set_ylabel("Cumulative share of applicants below the score")
axes[1].set_title("KS: the largest gap between the two cumulative distributions")
# Add the right panel's legend, fit the layout and save the chart as figs/roc_and_ks.png
axes[1].legend(loc="lower right")
plt.tight_layout()
plt.savefig("figs/roc_and_ks.png", dpi=150, bbox_inches="tight")
# Show the chart
plt.show()
""")

step("2.33", "Check calibration",
     "Does a predicted 5% default rate really mean 5%? Compare predicted and observed rates in ten groups of "
     "predicted probability of default.",
     "what are PD and calibration?",
     "PD is the probability of default. Here \"default\" is Home Credit's own payment-difficulty flag (`TARGET`), "
     "not the 90-days-past-due backstop of Basel's default definition (alongside unlikeliness to pay) and of the "
     "rebuttable presumption in IFRS 9, the accounting standard that sets expected-credit-loss provisions. So this "
     "PD is a probability of that flag. A model is calibrated when its predicted PDs match the default rates actually "
     "observed.")

code("""
# Odds near 600 points, and the average predicted against the observed default rate
near = summary["observed_odds_near_base_score"]
print(f"odds at 600 points: scorecard {summary['implied_odds_at_base_score']:.1f}:1, "
      f"observed {near['good_to_bad']:.1f}:1 among {near['n']:,} holdout applicants scoring {near['score_band'][0]}-{near['score_band'][1]}")
print(f"mean predicted PD {summary['mean_predicted_pd_val']:.4f} vs observed default rate {summary['val_bad_rate']:.4f}")
# Predicted against observed in ten groups
calibration = pd.DataFrame(summary["calibration_by_decile"])
print(type(calibration).__name__, calibration.shape)
calibration.round(4)
""")

code("""
# Check your work: no group is off by more than 0.4 percentage points
assert summary["calibration_max_abs_gap"] < 0.004, summary["calibration_max_abs_gap"]
assert round(near["good_to_bad"], 1) == 20.2
""")

code("""
# Predicted against observed, with the perfect-calibration diagonal
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot(calibration["mean_predicted_pd"] * 100, calibration["observed_default_rate"] * 100, marker="o", color=BLUE, label="scorecard, by PD decile")
limit = max(calibration["mean_predicted_pd"].max(), calibration["observed_default_rate"].max()) * 100 * 1.1
# Add the perfect-calibration diagonal, and name the x and y axes
ax.plot([0, limit], [0, limit], color="#a0aec0", ls="--", lw=1, label="perfect calibration")
ax.set_xlabel("Mean predicted probability of default (%)")
ax.set_ylabel("Observed default rate (%)")
# Title the chart, add the legend and fit the layout
ax.set_title("Predicted against observed default rate, holdout")
ax.legend(loc="upper left")
plt.tight_layout()
# Save the chart as figs/calibration.png and show it
plt.savefig("figs/calibration.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
Predicted and observed default rates sit close together in every group, and the observed odds around 600 points
(20.2:1) are close to the designed 20:1. That is the intercept correction of Step 2.26 at work.
""")

step("2.34", "Check stability",
     "Compare the spread of scores in training with the holdout.",
     "what is PSI?",
     "The Population Stability Index measures how far a score distribution has moved between two groups. Below 0.1 "
     "is usually read as stable, above 0.25 as a warning.")

code("""
# PSI between training and holdout scores
psi = summary["psi_train_to_holdout_scores"]
print(type(psi).__name__, f"PSI, training scores vs holdout scores: {psi:.4f}")
""")

code("""
# Check your work
assert psi < 0.1 and round(psi, 4) == 0.0002
""")

md("""
A PSI near zero is expected here: both groups are random slices of one file, so this says nothing about drift over
time (see the limitations).
""")

step("2.35", "Picture the separation",
     "Score distributions of repayers and defaulters, and the default rate by score band, drawn by "
     "`src/make_figures.py` (the same charts the README shows).",
     "what should a good score distribution look like?",
     "Two humps, defaulters to the left of repayers, overlapping as little as possible.")

code("""
# The two charts the README uses, drawn by the tested functions
from make_figures import default_rate_by_band, score_distribution

# Repayers and defaulters by score
_ = score_distribution(scores, yf_val)
""")

code("""
# Default rate in each tenth of applicants by score
_ = default_rate_by_band(scores, yf_val)
""")

code("""
# Check your work: the lowest-scoring tenth defaults about 20 times as often as the highest
assert round(summary["default_rate_lowest_decile"], 3) == 0.264 and round(summary["default_rate_highest_decile"], 3) == 0.013
""")

step("2.36", "Measure the price of readability",
     "`src/challengers.py` also trained LightGBM, a gradient-boosted tree model that cannot produce a points table, "
     "on the same 62 candidates. Read its holdout AUC.",
     "what is a challenger model?",
     "A stronger but less explainable model fitted on the same data. The gap between it and the scorecard is the "
     "price the lender pays for a model it can explain.")

code("""
# The challenger's holdout AUC, and the gap to the scorecard
h = challengers["holdout"]
readability_price = h["lightgbm"]["auc"] - summary["val_auc"]
print(type(h).__name__, sorted(h))
# Print both holdout AUCs and the AUC given up for a readable model
print(f"holdout AUC: scorecard {summary['val_auc']:.4f}, LightGBM {h['lightgbm']['auc']:.4f}")
print(f"price of a readable, reason-giving model: {readability_price:.4f} AUC")
""")

code("""
# Check your work
assert round(h["lightgbm"]["auc"], 4) == 0.7808 and round(readability_price, 3) == 0.027
""")

md("""
LightGBM ranks applicants better, by about 0.03 AUC. A lender gives that up for a model where every point has a
reason, every characteristic moves in one direction, and a declined applicant can be told why.
""")

step("2.37", "Run the fair-lending checks",
     "Every coefficient negative, age not in the scorecard, and pensioners not penalised.",
     "what does Regulation B require on age?",
     "A scoring system may use age only if applicants aged 62 or over are not given fewer points because of their "
     "age (12 CFR 1002.6(b)(2)).")

code("""
# The age check and the points for the missing employment bin
age_check = summary["reg_b_age_62_plus"]
print(type(age_check).__name__)
print(f"every coefficient negative: {summary['all_coefficients_negative']}")
# Print whether age is in the scorecard and passes the age check, and the pensioners' points
print(f"AGE_YEARS in the scorecard: {age_check['age_feature_in_scorecard']}; age check passed: {age_check['passed']}")
print(f"points for pensioners and the unemployed (EMPLOYED_YEARS Missing bin): {summary['employed_years_missing_bin_points']:+.1f}")
""")

code("""
# Check your work: the acceptance checks pass and pensioners are not penalised
assert summary["all_coefficients_negative"] and age_check["passed"]
assert summary["employed_years_missing_bin_points"] > 0
""")

md("""
Pensioners receive positive points (+8.5), consistent with their lower default rate (Step 1.20), and no applicant
aged 62 or over can lose points for age.
""")

# =====================================================================
# PART 3: COMMUNICATE RESULTS
# =====================================================================
md("""
---
# Part 3: Communicate Results

This part speaks to a credit committee: what the score lets them do, what it costs, and why each decline has a
reason.
""")

step("3.1", "Turn scores into lending policies",
     "Approve only the highest-scoring share of the holdout and measure each policy. Credit loss = credit amount of "
     "each approved loan that defaulted x 45%.",
     "what is LGD?",
     "Loss given default: the share of the amount lent that is lost when a borrower defaults. 45% is an "
     "illustrative flat figure, borrowed from the Basel foundation internal-ratings-based (F-IRB) supervisory value "
     "for senior unsecured claims on banks and other financial institutions. Basel sets no supervisory LGD for "
     "consumer loans: a bank using internal ratings estimates its own. One flat LGD multiplies every policy's loss "
     "by the same number, so the share of loss avoided does not depend on it; only the amounts do. The \"approve "
     "everyone\" row means every loan in this book, all already granted by Home Credit. \"Defaulted\" is the "
     "payment-difficulty flag, and many such loans cure, so the amounts are an upper bound; use the percentages.")

code("""
# The tested function that builds the money table
from business_impact import lending_policy_table

# One row per approval rate: 100%, 90%, 80%, 70%
policy = lending_policy_table(scores.to_numpy(), yf_val.to_numpy(), Xf_val["AMT_CREDIT"].to_numpy())
print(type(policy).__name__, policy.shape)
policy[["approval_rate", "cut_off_score", "default_rate_approved", "credit_loss", "loss_avoided_pct",
        "good_borrowers_declined_pct"]].round(4)
""")

code("""
# Check your work: the same table as outputs/business_impact.json
published = pd.DataFrame(json.load(open("outputs/business_impact.json", encoding="utf-8"))["policies"])
assert np.allclose(policy["loss_avoided_pct"], published["loss_avoided_pct"])
assert (policy["good_borrowers_declined"] == published["good_borrowers_declined"]).all()
""")

step("3.2", "Draw what each policy buys and costs",
     "Credit loss avoided against good borrowers turned away, side by side.",
     "what is the trade-off?",
     "A tighter policy avoids more losses but also turns away more people who would have repaid, and the profit on "
     "their loans with them.")

code("""
# Bar labels such as "80%"
policy_labels = []
for approval_rate in policy["approval_rate"]:
    policy_labels.append(f"{approval_rate:.0%}")
# Make one figure with one set of axes
fig, ax = plt.subplots(figsize=(9, 4.5))
# Width of each bar, so two fit side by side
width = 0.38
# Blue: loss avoided; red: good borrowers declined
ax.bar(np.arange(len(policy_labels)) - width / 2, policy["loss_avoided_pct"] * 100, width, color=BLUE, label="credit loss avoided")
ax.bar(np.arange(len(policy_labels)) + width / 2, policy["good_borrowers_declined_pct"] * 100, width, color=RED, label="good borrowers declined")
# Write each value above its bar
for bar_number in range(len(policy)):
    avoided = policy["loss_avoided_pct"].iloc[bar_number]
    declined = policy["good_borrowers_declined_pct"].iloc[bar_number]
    ax.text(bar_number - width / 2, avoided * 100 + 1, f"{avoided:.0%}", ha="center", fontsize=9)
    ax.text(bar_number + width / 2, declined * 100 + 1, f"{declined:.0%}", ha="center", fontsize=9)
# Put one tick under each pair of bars and label it with its approval rate
ax.set_xticks(np.arange(len(policy_labels)))
ax.set_xticklabels(policy_labels)
# Name the x and y axes and title the chart
ax.set_xlabel("Share of applicants approved (highest scores first)")
ax.set_ylabel("Share (%)")
ax.set_title(f"What each approval policy buys and costs, {len(yf_val):,} held-out applicants")
# Add the legend, fit the layout and save the chart as figs/policy_tradeoff.png
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("figs/policy_tradeoff.png", dpi=150, bbox_inches="tight")
# Show the chart
plt.show()
""")

code("""
# Check your work: the 80% policy quoted below
top80 = policy.loc[np.isclose(policy["approval_rate"], 0.8)].iloc[0]
assert round(top80["loss_avoided_pct"], 2) == 0.47 and round(top80["good_borrowers_declined_pct"], 3) == 0.172
assert round(top80["cut_off_score"]) == 541
""")

md("""
Approving the top 80% (a score of 541 or above) avoids 47% of the credit loss that approving everyone would have
caused, while turning away 17.2% of the applicants who would have repaid. Tighter policies avoid more loss but
decline more good borrowers. Where to stop depends on the margin earned on a good loan against the loss on a bad
one, which this chart lets the committee price.
""")

step("3.3", "Explain a decline",
     "Take a holdout applicant who defaulted and scored below the 80% cut-off, and list the characteristics where "
     "they lost the most points against the best bin, exactly as in Step 1.6.",
     "what is adverse action?",
     "Declining an application. Under the US Equal Credit Opportunity Act the lender must give the applicant the "
     "principal reasons, which a scorecard's points gaps supply directly.")

code("""
# The example the pipeline picked, with its three main reasons
ex = summary["example_declined_applicant"]
print(type(ex).__name__)
print(f"applicant scoring {ex['score']:.0f}, below the 80% approval cut-off of {ex['cut_off_score_at_80pct_approval']:.0f}. Reasons:")
# One line per reason
for reason in ex["reasons_points_below_best_bin"]:
    print(f"  {reason['feature']}: {reason['points']:.1f} points below the best bin")
""")

code("""
# Check your work: three reasons, largest gap first
assert len(ex["reasons_points_below_best_bin"]) == 3
assert ex["reasons_points_below_best_bin"][0]["feature"] == "EXT_SOURCE_3"
""")

step("3.4", "Show the drivers to a manager",
     "Rank the characteristics by the spread of points they can award (best bin minus worst bin) and draw the top "
     "ten.",
     "why the points range?",
     "It is the most an applicant's score can move on that one characteristic, which is the plainest measure of how "
     "much it drives a decision.")

code("""
# Best-bin points minus worst-bin points for each characteristic
points_range = {}
for driver_col in kept_cols:
    driver_points = sc["points_tables"][driver_col]
    points_range[driver_col] = driver_points.max() - driver_points.min()
drivers = pd.Series(points_range).sort_values()
# Print the type and length of the points ranges, then show the ten largest
print(type(drivers).__name__, len(drivers))
drivers.tail(10).round(1)
""")

code("""
# Check your work: one range per characteristic, all positive
assert len(drivers) == 32 and (drivers > 0).all()
""")

code("""
# The ten characteristics that move a score most
fig, ax = plt.subplots(figsize=(9, 5))
drivers.tail(10).plot(kind="barh", ax=ax, color=BLUE)
ax.set_xlabel("Points between the worst and the best bin")
# Title the chart, fit the layout and show it
ax.set_title("What moves an applicant's score most")
plt.tight_layout()
plt.show()
""")

md("""
**The model in words.** An applicant starts with the base points (Step 2.27) and gains or loses points for each of 32
characteristics. Two of the external scores (EXT_SOURCE_3 and EXT_SOURCE_2) move a score most; education level is third,
ahead of the other external score and the average of all three. Next come the type of organisation the applicant works for, how much of their credit-card limit they use, the
ratio of the credit to the price of the goods, their occupation and the share of their earlier applications that
were refused. Late instalments and the age of their credits play only a small part. Education, occupation and
organisation type are candidates for the proxy (fair-lending) test listed in Next steps. Every characteristic
gives more points to its safer bins, and none of them is sex, marital status or age.

**Recommendation to a credit committee.** Use the scorecard for this lending decision even though a gradient-boosted
model ranks applicants better (Step 2.36): every point has a checkable reason, which a declined applicant is
entitled to; every characteristic moves in one direction and was kept or removed for a stated reason; and the
fair-lending conditions are tested, not asserted. Start the cut-off discussion from the 80% policy, then price it
with the margin on a good loan, which this data does not contain.

**Scope.** It estimates the probability of Home Credit's payment-difficulty flag, over one outcome window. An
IFRS 9 provision also needs lifetime PD curves, a test for a significant increase in credit risk to stage loans,
forward-looking economic scenarios, and loss given default and exposure at default, which need recovery data this
dataset does not have.
""")

step("3.5", "Record every modelling decision",
     "Build the decision log from this run's results, so the evidence column cannot drift from the numbers above.",
     "what is a decision log?",
     "A table of every choice that shaped the model, the alternative weighed against it, the evidence that settled "
     "it and the trade-off accepted. A model validator reviews a scorecard through exactly this record.")

code("""
# Tools to show the log as a table, and to read the saved results
from IPython.display import Markdown, display

# The reader for outputs/results.json
from results_io import read_results

# The fit with sex and marital status added is run only by the pipeline (src/fit_model.py), so read its result
ablation = read_results()["metrics"]["prohibited_basis_ablation"]
# The validation-slice results of the challengers, and the points that double the odds
ch_val = challengers["validation"]
pdo = summary["scorecard_factor"] * np.log(2)
# Information value of the two prohibited bases
sex_iv = ablation["prohibited_bases_iv"]["CODE_GENDER"]
marital_iv = ablation["prohibited_bases_iv"]["NAME_FAMILY_STATUS"]
# The correlation-gate example quoted in the log
overdue = None
for dropped_record in summary["correlation_gate_dropped"]:
    if dropped_record["dropped"] == "BUREAU_OVERDUE_MEAN":
        overdue = dropped_record
        break
print(type(ablation).__name__, "| PDO", round(pdo, 6), "| example:", overdue["dropped"], "->", overdue["kept"])
""")

code("""
# Check your work
assert overdue is not None and abs(pdo - 40) < 1e-6
""")

code("""
# A short name for the holdout summary used in every row
m = summary
# One row per decision: what I chose, the alternative, the evidence, the cost
decisions = [
    # Fair lending: which characteristics may enter at all
    ("Leave out sex and marital status (1.4)",
     f"Use them: IV {sex_iv:.3f} and {marital_iv:.3f}",
     f"holdout AUC {ablation['val_auc_sklearn_final_features']:.4f} without them, "
     f"{ablation['val_auc_sklearn_with_prohibited_bases']:.4f} with them",
     f"{ablation['val_auc_sklearn_with_prohibited_bases'] - ablation['val_auc_sklearn_final_features']:.4f} AUC "
     "given up; required by ECOA and Regulation B"),
    # Data cleaning: the 1,000-year placeholder
    ("Read DAYS_EMPLOYED = 365243 as missing, with its own bin (1.11, 1.33)",
     "Keep the raw value (about 1,000 years in work)",
     f"{int(positive.sum()):,} applicants carry the placeholder; their missing bin earns "
     f"{m['employed_years_missing_bin_points']:.1f} points, so pensioners are not penalised",
     "none found; ratios built on employment stay sane"),
    # Validation design
    ("Random 80/20 holdout, stratified on default (1.39)",
     "Out-of-time holdout (impossible: no application dates)",
     f"{len(X_val):,} held out; default rate {m['train_bad_rate']:.2%} training, {m['val_bad_rate']:.2%} holdout",
     "drift over time is untested (limitations)"),
    # Binning
    ("Coarse classing over plain decile bins (2.9, 2.10)",
     "Decile bins, the champion",
     f"monotonic numeric characteristics {ch_val['coarse_classing']['n_numeric_monotonic']} of "
     f"{ch_val['coarse_classing']['n_numeric_kept']}, against {ch_val['champion']['n_numeric_monotonic']} of "
     f"{ch_val['champion']['n_numeric_kept']}; rule: {challengers['decision_rule']}",
     f"{challengers['coarse_classing_auc_loss_validation']:.4f} validation AUC"),
    # Selection gate 1
    (f"Gate 1: keep characteristics with IV >= {IV_THRESHOLD} (2.14)",
     "A stricter screen",
     f"{m['n_after_iv_screen']} of {m['n_candidate_features']} candidates pass",
     "permissive on purpose; gates 2 and 3 remove redundancy and flipped effects"),
    # Selection gate 2
    (f"Gate 2: of any WoE pair with |r| > {CORRELATION_THRESHOLD}, keep the higher-IV one (2.18)",
     "Keep both and let the fit split the weight",
     f"{len(m['correlation_gate_dropped'])} dropped, e.g. {overdue['dropped']} (r = {overdue['r']:.3f} "
     f"with {overdue['kept']})",
     "a little signal in the dropped partner"),
    # Selection gate 3
    ("Gate 3: drop any characteristic whose coefficient flips sign (2.20)",
     "Keep it for its AUC contribution",
     f"{len(m['sign_gate_removed'])} removed; all {m['n_features_kept']} kept coefficients negative: "
     f"{m['all_coefficients_negative']}",
     "the gate does not show which characteristic absorbed the effect (limitations)"),
    # The solver
    ("Logistic regression written from scratch, checked against scikit-learn (2.21)",
     "scikit-learn alone",
     f"max coefficient difference {m['coef_diff_max_vs_sklearn']:.4f}, prediction correlation "
     f"{m['pred_corr_vs_sklearn']:.6f}",
     "more code to test (tests/test_from_scratch_lr.py)"),
    # Calibration
    ("Class-balanced fit, then correct the intercept to the true default rate (2.26)",
     "Unweighted fit, or no correction",
     f"intercept {m['intercept_balanced']:+.3f} -> {m['intercept_prior_corrected']:+.3f}; mean predicted PD "
     f"{m['mean_predicted_pd_val']:.2%} against {m['val_bad_rate']:.2%} observed; largest decile gap "
     f"{m['calibration_max_abs_gap']:.4f}",
     "one extra step that must be repeated on every refit"),
    # The points scale
    (f"Scale: 600 points at 20:1 good-to-bad odds, {pdo:.0f} points to double the odds (2.27)",
     f"Anchor 600 to the training population's own odds ({m['train_good_to_bad_odds']:.1f}:1)",
     f"implied odds at 600: {m['implied_odds_at_base_score']:.1f}:1; observed in the {near['score_band'][0]} to "
     f"{near['score_band'][1]} band: {near['good_to_bad']:.1f}:1 ({near['n']:,} applicants)",
     "a policy convention, not a statistic; a lender would use its own scale"),
    # Model family
    ("A points scorecard rather than gradient boosting (2.36, 3.4)",
     "LightGBM on the same candidates",
     f"holdout AUC {m['val_auc']:.4f} against LightGBM {challengers['holdout']['lightgbm']['auc']:.4f}",
     f"{challengers['holdout']['lightgbm']['auc'] - m['val_auc']:.4f} AUC paid for a fixed reason on every decline"),
    # Pricing the policies
    ("Price policies with one 45% loss given default (3.1)",
     "A modelled LGD (impossible: no recovery data)",
     f"approving the top 80% (score {top80['cut_off_score']:.0f}+) avoids {top80['loss_avoided_pct']:.0%} of loss "
     f"and declines {top80['good_borrowers_declined_pct']:.0%} of good borrowers",
     "loss side only, margin on good loans ignored; the flat LGD moves the amounts, not the shares"),
]
# The rows as a table
log = pd.DataFrame(decisions, columns=["Decision", "Alternative considered", "Evidence from this run", "Trade-off accepted"])
# Print the type and shape of the decision log
print(type(log).__name__, log.shape)
""")

code("""
# Check your work: every decision rests on a passing result, not just a description
assert len(log) == 12 and log.notna().all().all()
assert ablation["val_auc_sklearn_with_prohibited_bases"] - ablation["val_auc_sklearn_final_features"] < 0.005
assert m["all_coefficients_negative"] and abs(m["implied_odds_at_base_score"] - 20) < 1e-6
# Stop the run if this does not hold
assert m["calibration_max_abs_gap"] < 0.01
""")

code("""
# A helper that makes a cell value safe inside a markdown table
def clean(value):
    # Pipes inside cells (as in |r|) are escaped so they don't split columns
    return str(value).replace("|", "\\\\|")


# A helper that writes a table as markdown text
def markdown_table(df):
    # A plain markdown table: header, divider, then one line per row
    lines = []
    header = []
    for name in df.columns:
        header.append(clean(name))
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in df.itertuples(index=False):
        row_cells = []
        for value in row:
            row_cells.append(clean(value))
        lines.append("| " + " | ".join(row_cells) + " |")
    return "\\n".join(lines)


# Show the decision log as a table
display(Markdown(markdown_table(log)))
""")

md("""
Read down the trade-off column and the pattern is consistent: wherever readability, fairness or a defensible reason
competed with ranking power, the scorecard gave up some AUC, and the log records exactly how much.
""")

step("3.6", "Check every figure against the pipeline's results file",
     "The pipeline (`python src/run_pipeline.py`) wrote every headline metric to `outputs/results.json`. The "
     "notebook rebuilt the same model, so every entry of its summary must equal that file.",
     "why check the notebook against a file?",
     "A number typed into prose can drift from the code. A number compared with the file the tested code wrote "
     "cannot.")

code("""
# math.isclose for comparing decimals
import math

# The published metrics
ref = read_results()["metrics"]


# A helper that compares two results value by value
def same(a, b):
    # Compare nested results, allowing only floating-point rounding differences
    if isinstance(a, dict):
        if set(a) != set(b):
            return False
        for key in a:
            if not same(a[key], b[key]):
                return False
        return True
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        for item_a, item_b in zip(a, b):
            if not same(item_a, item_b):
                return False
        return True
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


# Every summary entry that differs from the file
mismatched = []
for summary_key in summary:
    if not same(summary[summary_key], ref[summary_key]):
        mismatched.append(summary_key)
print(type(ref).__name__, len(ref), "entries in results.json;", len(mismatched), "mismatched")
""")

code("""
# Check your work
assert not mismatched, f"notebook and results.json differ on: {mismatched}"
assert abs(full_auc - ref["val_auc"]) < 1e-9 and len(kept_cols) == ref["n_features_kept"]
print(f"notebook figures match outputs/results.json ({len(summary)} entries checked)")
""")

step("3.7", "Check the headline in the opening cell",
     "The opening cell quotes AUC 0.754, KS 0.381, 32 of 62 characteristics, 61,503 held-out applicants, and the "
     "80% policy's 47% loss cut and 17.2% good borrowers declined. Format each from the results files at the "
     "precision the text uses.",
     "why check prose?",
     "The opening cell is what most readers see. If a re-run moved a number, this cell fails instead of letting "
     "the text go stale.")

code("""
# The opening cell's figures, formatted from the files
headline = {
    "AUC": f"{ref['val_auc']:.3f}",
    "KS": f"{ref['val_ks']:.3f}",
    "kept": f"{ref['n_features_kept']} of {ref['n_candidate_features']}",
    "holdout": f"{ref['n_validation_rows']:,}",
    "loss cut at 80%": f"{published.loc[np.isclose(published['approval_rate'], 0.8), 'loss_avoided_pct'].iloc[0]:.0%}",
    "good declined at 80%": f"{published.loc[np.isclose(published['approval_rate'], 0.8), 'good_borrowers_declined_pct'].iloc[0]:.1%}",
}
print(type(headline).__name__, headline)
""")

code("""
# Check your work
assert headline == {"AUC": "0.754", "KS": "0.381", "kept": "32 of 62", "holdout": "61,503",
                    "loss cut at 80%": "47%", "good declined at 80%": "17.2%"}, headline
""")

step("3.8", "Write the run record",
     "Date, runtime, code version and library versions, saved to `outputs/notebook_run.json` next to the "
     "pipeline's results.",
     "what is a run record?",
     "A note of exactly which code and libraries produced the figures, so anyone can check them rather than take "
     "them on trust.")

code("""
# Tools to record the Python version, the commit, the date and the library versions
import platform
import subprocess
from datetime import datetime, timezone
# Library versions as installed
from importlib.metadata import PackageNotFoundError, version

# The commit the code was at, and whether the code had edits not yet committed.
# If git is missing the values are recorded as None instead of stopping the notebook.
try:
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    changed = subprocess.run(["git", "status", "--porcelain", "--", "src", "tests", "build_notebook.py"],
                             capture_output=True, text=True).stdout.strip()
    uncommitted_code_changes = changed != ""
except OSError:
    git_commit = None
    uncommitted_code_changes = None

# Library versions; a missing package is recorded as None
package_versions = {}
for package in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn"]:
    try:
        package_versions[package] = version(package)
    except PackageNotFoundError:
        package_versions[package] = None

# Everything needed to reproduce this run
run_record = {
    "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "runtime_minutes": round((time.time() - RUN_STARTED) / 60),
    "git_commit": git_commit,
    "uncommitted_code_changes": uncommitted_code_changes,
    "python": platform.python_version(),
    "packages": package_versions,
    "results_checked_against": "outputs/results.json",
    "entries_checked": len(summary),
    "holdout_auc": float(full_auc),
    "benchmark_auc": float(baseline_auc),
    "auc_lift_over_benchmark": float(auc_lift),
    "auc_lift_ci95_paired_bootstrap": [float(lift_lo), float(lift_hi)],
    "characteristics_kept": len(kept_cols),
}
# Save it next to the pipeline's results
with open("outputs/notebook_run.json", "w", encoding="utf-8") as f:
    json.dump(run_record, f, indent=2)
print(type(run_record).__name__)
print(json.dumps(run_record, indent=2))
""")

code("""
# Check your work
saved = json.load(open("outputs/notebook_run.json", encoding="utf-8"))
assert saved["entries_checked"] == 46 and saved["characteristics_kept"] == 32
assert round(saved["benchmark_auc"], 4) == 0.7467 and abs(saved["auc_lift_over_benchmark"] - auc_lift) < 1e-12
""")

md("""
## What you built

- **Part 1, Prepare Data.** You loaded 307,511 applications, found and fixed a placeholder that gave pensioners a
  thousand years of work, saw that the external scores carry far more signal than the other application fields, rolled six
  credit-history tables up to one row per applicant, built 62 candidate characteristics with one `wrangle()`
  function, and held back 61,503 applicants.
- **Part 2, Build Model.** You set two baselines (a constant score at AUC 0.5 and an application-only model at
  0.7467), computed Weight of Evidence by hand and checked it against the tested function, fixed zig-zagging bins
  with coarse classing, passed the candidates through three gates to 32, fitted a logistic regression written from
  scratch, turned it into points on a 600 = 20:1 scale, and evaluated it on the holdout: AUC 0.754, KS 0.381, close
  calibration, a stable score and passing fair-lending checks.
- **Part 3, Communicate Results.** You turned the score into approval policies (the top 80% avoids 47% of the
  credit loss), explained a decline with fixed reasons, showed the drivers, logged every decision and checked every
  figure against the pipeline's results file.

## Limitations

- **An already-approved book (selection bias).** Every row is a loan Home Credit granted. Applicants it refused have
  no outcome and are not in the data, and their default rate is likely higher. The 47% loss cut and the "approve
  everyone" row are measured within this approved book. Using the score on new applicants would need reject
  inference (estimating how refused applicants would have performed) or a test on through-the-door data (everyone
  who applies).
- **Random, not out-of-time, validation.** The dataset carries no application dates, so the holdout is a random 20%
  rather than a later period, and nothing here shows how the score behaves as the applicant population drifts. The
  PSI of Step 2.34 compares two random slices of one population, which is the easy case.
- **One loss given default, no revenue.** The policy table applies a single 45% LGD and ignores the interest earned
  on good loans, so it shows the loss side of the cut-off decision only.
- **Cut-offs are set on the same holdout** that measures the loss avoided.
- **Home Credit's default flag, not a regulatory default.** `TARGET` marks payment difficulties on early
  instalments, not the 90-days-past-due backstop of Basel's default definition (with unlikeliness to pay) and of
  IFRS 9's rebuttable presumption.
- **Probability of default only, over one horizon.** A full IFRS 9 provision also needs lifetime PD term
  structures, significant-increase-in-credit-risk staging, forward-looking scenarios, and loss-given-default and
  exposure-at-default models; this dataset has no recovery data for the last two.
- **The external scores are opaque, and their timing is assumed.** The strongest characteristics are third-party
  scores whose construction is not published, so a reason that points at one says where the applicant lost points,
  not why. When they were obtained is not documented; that they exist at application time is an assumption.
- **The currency is unstated.** Credit-loss figures are in the dataset's own units.
- **The history tables are summarised, not modelled.** Patterns within a history (a borrower who was late, then
  caught up) are invisible to the scorecard.
- **Coarse classing is fitted on the training split only.** A bin that is monotonic in training can be slightly off
  in a new population. The challenger test (Step 2.10) measured this on a validation slice, once.
- **The sign gate removes, it does not explain.** It does not identify which other characteristic absorbed a removed
  characteristic's effect.
- **Fairness is checked against two prohibited bases and one age rule.** Other protected characteristics may have
  proxies in the data and are not tested.

## Next steps

1. Validate on a later population once application dates are available, and track PSI month by month.
2. Add the margin on a good loan to the policy table, so the cut-off can be set on profit, not loss alone.
3. Build loss-given-default and exposure-at-default models if recovery data becomes available, to move from PD to
   expected credit loss.
4. Test the scorecard for proxy discrimination (for example, education, occupation and organisation type acting as proxies for sex, marital status or age) with a
   formal disparate-impact analysis.

**Sensitivity checks to try**

1. In Step 2.18, set `CORRELATION_THRESHOLD = 0.8` at the top of the cell and rerun to Step 2.22. Which
   characteristics drop out, and what happens to AUC?
2. In Step 2.27, change the scale to 30 points to double the odds. Which numbers in the points table change, and
   does the ranking (AUC) change?
3. In Step 3.1, add a 60% approval rate (`lending_policy_table(..., approval_rates=(1.0, 0.9, 0.8, 0.7, 0.6))`). Is
   the extra loss avoided worth the good borrowers turned away?

## Glossary

- **Bin:** a range of a characteristic (for example, 2 to 5 years in the current job), or one label of a text column.
- **WoE (Weight of Evidence):** ln(share of repayers in a bin / share of defaulters in the bin); positive = safer.
- **IV (Information Value):** how strongly a characteristic separates repayers from defaulters.
- **Coarse classing:** merging neighbouring bins so the default rate moves in one direction.
- **AUC / KS:** measures of how well the score ranks defaulters below repayers.
- **PD:** probability of default. **LGD:** loss given default. **PDO:** points to double the odds.
- **PSI:** population stability index, how much the score distribution has shifted.
- **Prohibited basis:** a characteristic the law forbids in a credit decision (here sex and marital status, under
  ECOA and Regulation B).
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
with open("Credit_Risk_Scorecard.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote Credit_Risk_Scorecard.ipynb with {len(cells)} cells")
