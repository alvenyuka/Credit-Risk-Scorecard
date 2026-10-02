"""
Generates Credit_Risk_Scorecard.ipynb cell by cell. Edit this file, not the
notebook directly, then regenerate:

    python build_notebook.py
    jupyter nbconvert --to notebook --execute Credit_Risk_Scorecard.ipynb \
        --output Credit_Risk_Scorecard.ipynb --ExecutePreprocessor.timeout=3600

The scorecard is fitted by calling run_pipeline.build_validation_scores(), the
same function that writes outputs/results.json, and the last cell asserts that
the notebook's figures equal that file. The notebook adds narrative, tables and
charts; it does not reimplement the pipeline.
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
lives in `src/` and is covered by the test suite, and the final cell checks that
the figures shown here equal the ones the pipeline writes to
`outputs/results.json`.
""")

md("""
## Step 1: a baseline from the application form alone

The application form sets the floor: anything added later has to beat it.

One data-quality issue shapes the features. `DAYS_EMPLOYED` holds the
placeholder 365243 (about a thousand years) for pensioners and unemployed
applicants. Left in, it gives retirees a millennium of work history and breaks
every ratio built on it, so the value is set to missing and an anomaly flag
records it for exploration.

Sex (`CODE_GENDER`) and marital status (`NAME_FAMILY_STATUS`) are prohibited
bases under the US Equal Credit Opportunity Act and Regulation B, so they are
excluded from every feature list (`PROHIBITED_BASES` in `src/feature_lists.py`,
pinned by `tests/test_fair_lending.py`).
""")

code("""
import sys
sys.path.insert(0, "src")

from io_raw import load_application
from baseline_features import engineer_baseline
from baseline_model import build_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

train_raw = load_application("train")
feats = engineer_baseline(train_raw)

y = feats["TARGET"]
X = feats.drop(columns=["TARGET", "SK_ID_CURR"])
print(f"applicants: {len(y):,}   default rate: {y.mean():.2%}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = build_pipeline()
pipe.fit(X_train, y_train)

baseline_auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
print(f"application-only baseline AUC: {baseline_auc:.4f}")
""")

md("""
The application form alone gives the AUC printed above. It is the benchmark the
full feature set has to beat.
""")

# ---------------------------------------------------------------------------

md("""
## Step 2: the credit statistics, implemented and verified

A credit committee reviews a scorecard through Weight of Evidence (WoE) and
Information Value (IV), and model-risk teams expect the statistics to be
verifiable. WoE and IV, a gradient-descent logistic regression and the metrics a
credit team checks (AUC, GINI, KS and PSI) are implemented in `src/` and checked
against scikit-learn and scipy before being used on real data.
""")

code("""
from metrics_scratch import auc_rank_sum, ks_statistic
from scipy.stats import ks_2samp
import numpy as np

# check on synthetic data before using these on real applicants
rng = np.random.default_rng(0)
n = 5000
y_check = rng.integers(0, 2, n)
score_check = np.round(rng.normal(0, 1, n) + y_check * 0.8, 2)  # rounding forces ties

scratch_auc = auc_rank_sum(y_check, score_check)
sklearn_auc = roc_auc_score(y_check, score_check)
print(f"AUC check: from-scratch={scratch_auc:.6f} sklearn={sklearn_auc:.6f} diff={abs(scratch_auc - sklearn_auc):.2e}")

scratch_ks = ks_statistic(y_check, score_check)
scipy_ks = ks_2samp(score_check[y_check == 1], score_check[y_check == 0]).statistic
print(f"KS check:  from-scratch={scratch_ks:.6f} scipy={scipy_ks:.6f} diff={abs(scratch_ks - scipy_ks):.2e}")

assert abs(scratch_auc - sklearn_auc) < 1e-9
assert abs(scratch_ks - scipy_ks) < 1e-9
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
# Weight of Evidence and Information Value on the application features
from feature_lists import APPLICATION_NUMERIC, CATEGORICAL_COLS
from woe_iv import fit_woe, iv_strength

woe_fits = {c: fit_woe(X_train[c], y_train, is_categorical=False, n_bins=10) for c in APPLICATION_NUMERIC}
woe_fits.update({c: fit_woe(X_train[c], y_train, is_categorical=True) for c in CATEGORICAL_COLS})

iv_ranked = sorted(woe_fits, key=lambda c: -woe_fits[c]["iv"])
print("Top 10 application features by Information Value:")
for col in iv_ranked[:10]:
    print(f"  {col:28s} IV={woe_fits[col]['iv']:.4f}  ({iv_strength(woe_fits[col]['iv'])})")
""")

md("""
The three `EXT_SOURCE` columns, external bureau-style scores supplied with the
data, carry far more information than any field on the application form.

`EXT_SOURCE_MEAN` crosses the IV threshold that the helper labels "check for
leakage". It is checked rather than assumed: it is the average of the three
external scores, all of which exist when the application is made, so its IV
reflects their combined strength, not information from after the decision.
""")

code("""
# The anomaly flag duplicates the Missing bin of EMPLOYED_YEARS exactly.
flag_fit = fit_woe(X_train["DAYS_EMPLOYED_ANOM"], y_train, is_categorical=True)
emp_table = woe_fits["EMPLOYED_YEARS"]["table"]
print(f"DAYS_EMPLOYED_ANOM = 1:     n={flag_fit['table'].loc['1', 'n']:,}  WoE={flag_fit['table'].loc['1', 'woe']:+.4f}")
print(f"EMPLOYED_YEARS = Missing:   n={emp_table.loc['Missing', 'n']:,}  WoE={emp_table.loc['Missing', 'woe']:+.4f}")
same_rows = (X_train["DAYS_EMPLOYED_ANOM"] == 1).equals(X_train["EMPLOYED_YEARS"].isna())
print(f"same applicants in both: {same_rows}")
""")

md("""
The flag and the `Missing` bin of `EMPLOYED_YEARS` are the same applicants with
the same WoE, and the WoE is positive: this group, largely pensioners, defaults
less often than average. Offering both encodings to the model lets the solver
give them opposite signs, so one of them appears to penalise a lower-risk group,
which is a wrong adverse-action reason and a fair-lending problem (Regulation B
restricts discounting pension income). The scorecard therefore reads the group
only through `EMPLOYED_YEARS`; the flag stays an exploration column.
""")

# ---------------------------------------------------------------------------

md("""
## Step 3: bureau history, previous loans and the scorecard

Credit-bureau records, previous Home Credit applications and repayment history
are aggregated per applicant: counts, averages, the share of bureau credit
overdue, and how often past applications were approved or refused.

The candidate features then pass three gates on the training split:

1. **Information value**: features with IV below 0.01 are dropped.
2. **Correlation**: of any pair of WoE columns correlated above 0.9, only the
   higher-IV feature is kept.
3. **Coefficient sign**: WoE is ln(good/bad), so in a model of the probability
   of default every coefficient must be negative. A positive coefficient means
   the feature's effect has reversed once the others are in the model, which
   would award points to the riskier bins. The feature with the largest positive
   coefficient is removed and the model refitted until none remain.

The cell below runs the same function that writes `outputs/results.json`, and
prints each gate's decisions.
""")

code("""
from run_pipeline import build_validation_scores, summarise

result, sc, Xf_val, yf_val, scores = build_validation_scores()
kept_cols = sc["kept_cols"]
""")

md("""
`AGE_YEARS` is among the features the sign gate removes when its coefficient
reverses once employment history and income type are in the model. Kept with a
positive coefficient, age would have assigned negative points to the oldest
applicants, which Regulation B does not allow for applicants aged 62 or over.
Removing it, rather than forcing its sign, keeps every remaining coefficient a
genuine fitted value.
""")

code("""
import pandas as pd
from woe_iv import monotonicity_report

report = monotonicity_report(result["woe_fits"], kept_cols)
numeric = report[report["type"] == "numeric"]
print(f"kept features: {len(kept_cols)}  (numeric: {len(numeric)}, categorical: {len(report) - len(numeric)})")
print(f"numeric features with monotonic WoE across their bins: {int(numeric['monotonic_woe'].astype(bool).sum())} of {len(numeric)}")

top = kept_cols[0]
table = result["woe_fits"][top]["table"][["n", "bad", "woe"]].copy()
table["default_rate"] = table["bad"] / table["n"]
table["points"] = sc["points_tables"][top]
print(f"\\nWoE and points for the highest-IV feature, {top}, in bin order:")
table.round(4)
""")

md("""
Every bin of every feature carries a fixed number of points, and a committee can
review each table like the one above before the model goes live. The full points
table for every kept feature is written to `outputs/scorecard_points.json`.
""")

code("""
full_auc = result["val_auc"]
print(f"full feature set AUC: {full_auc:.4f} (up from {baseline_auc:.4f} using the application form alone)")
print(f"from-scratch vs scikit-learn: max coefficient diff={result['coef_diff_max']:.4f}, "
      f"prediction correlation={result['pred_corr']:.6f}")
m = result["model"]
print(f"from-scratch solver: {m.n_iter_} iterations, converged={m.converged_}, last cost change={m.final_cost_change_:.2e}")
""")

md("""
The from-scratch solver and scikit-learn fit the same class-balanced objective,
so their predictions should agree almost exactly, and they do (correlation
printed above). Coefficients differ slightly more than predictions because
several bureau features remain correlated, which leaves individual coefficients
less determined than the combined score.
""")

code("""
summary = summarise(result, sc, Xf_val, yf_val, scores)
print(f"training default rate: {summary['train_bad_rate']:.4f}")
print(f"intercept fitted under the balanced (50/50) weighting: {summary['intercept_balanced']:+.4f}")
print(f"intercept after the prior correction:                  {summary['intercept_prior_corrected']:+.4f}")
print(f"odds the scorecard assigns at 600 points: {summary['implied_odds_at_base_score']:.1f}:1")
near = summary["observed_odds_near_base_score"]
print(f"observed good:bad odds among holdout applicants scoring {near['score_band'][0]}-{near['score_band'][1]}: "
      f"{near['good_to_bad']:.1f}:1 (n={near['n']:,})")
print(f"mean predicted PD {summary['mean_predicted_pd_val']:.4f} vs observed default rate {summary['val_bad_rate']:.4f}")
print(f"observed score range: {summary['score_min']:.0f} to {summary['score_max']:.0f}; "
      f"share clipped at 300: {summary['share_clipped_at_300']:.2%}")
print(f"average score, repaid: {summary['mean_score_repaid']:.1f}   defaulted: {summary['mean_score_defaulted']:.1f}")
lo, hi = summary["val_auc_ci95_bootstrap"]
print(f"AUC {summary['val_auc']:.4f} (95% bootstrap CI {lo:.4f} to {hi:.4f}), KS {summary['val_ks']:.4f}, "
      f"GINI {summary['val_gini']:.4f}")
print(f"PSI, training scores vs holdout scores: {summary['psi_train_to_holdout_scores']:.4f}")
""")

md("""
The model is fitted with class-balanced weights, which centre the intercept as
if half of all applicants defaulted. Before the points table is built, the
intercept is shifted back to the training default rate (the King and Zeng prior
correction). Ranking is unchanged, but the scale now means what it says: 600
points corresponds to 20:1 good:bad odds, which the observed odds around 600
confirm, and the score converts to a probability of default. A PSI near zero
between training and holdout scores is expected for a random split; it is the
baseline for monitoring drift once the scorecard is in use.
""")

code("""
calib = pd.DataFrame(summary["calibration_by_decile"])
print("predicted PD against observed default rate, by decile of predicted PD:")
calib.round(4)
""")

code("""
from make_figures import score_distribution, default_rate_by_band

_ = score_distribution(scores, yf_val)
""")

code("""
_ = default_rate_by_band(scores, yf_val)
""")

code("""
ex = summary["example_declined_applicant"]
print(f"a defaulted holdout applicant below the 80% approval cut-off ({ex['cut_off_score_at_80pct_approval']:.0f} points), "
      f"score {ex['score']:.0f}. Reasons, as points below the best bin:")
for r in ex["reasons_points_below_best_bin"]:
    print(f"  {r['feature']}: {r['points']:.1f} points below the best bin")
""")

md("""
This is what the scorecard format buys over a raw probability. Each reason is
the gap between the points the applicant received on a characteristic and the
most points that characteristic can give, so a reason is never a characteristic
on which the applicant already scored the maximum, and the largest gaps are the
ones to report. A gradient-boosted model explained with SHAP does not hand over
a fixed, auditable point table in the same way.
""")

code("""
age = summary["reg_b_age_62_plus"]
print(f"every coefficient negative (WoE sign check): {summary['all_coefficients_negative']}")
print(f"AGE_YEARS in the scorecard: {age['age_feature_in_scorecard']}; "
      f"bins covering age 62+: {age['bins_covering_62_plus'] or 'none'}; check passed: {age['passed']}")
print(f"points for the EMPLOYED_YEARS Missing bin (pensioners and the unemployed): "
      f"{summary['employed_years_missing_bin_points']:+.1f}")
""")

# ---------------------------------------------------------------------------

md("""
## Step 4: recommendation to a credit committee

A gradient-boosted model would probably rank applicants somewhat better. The
scorecard is still the stronger choice for a lending decision, for reasons other
than raw accuracy:

- every point has a specific, checkable reason, which a declined applicant is
  entitled to under adverse-action rules;
- the direction of every characteristic is enforced (all coefficients negative
  on WoE inputs) and each WoE table can be reviewed before the model is used,
  rather than hoped for in a tree ensemble;
- the fair-lending conditions are tested, not asserted: no prohibited basis in
  any feature list, and no negative points for applicants aged 62 or over.

Scope: the model outputs a probability of default. An IFRS 9 provision also
needs loss given default and exposure at default, which are separate models not
built here.

Next steps: validate on a later population when data with application dates is
available, and replace decile binning of the zero-inflated delinquency columns
with zero-aware bins.
""")

code("""
# The figures above must equal the ones the pipeline wrote to outputs/results.json.
import math
from results_io import read_results

ref = read_results()["metrics"]

def same(a, b):
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
