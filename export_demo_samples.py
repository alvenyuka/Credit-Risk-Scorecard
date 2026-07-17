"""
Export a small set of real, held-out test-set applicants for the Vercel
case-study site's "explore real applicants" demo.

Reloads application_train.csv and reruns only the (fast, application-level)
feature engineering, then reproduces the exact same stratified train/val/test
split used by run_pipeline.py (same SEED, same row order, same input length),
so the resulting test-set row order lines up positionally with
output/predictions.npz. No relational joins needed for this script, so it
runs in seconds rather than minutes.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.io_utils import load_application
from src.feature_engineering_app import engineer_application_features

SEED = 42
ROOT = Path(__file__).resolve().parent

app = load_application("train")
app = engineer_application_features(app)

target_col = "TARGET"
y_all = app[target_col]

_, temp_idx = train_test_split(
    app.index, test_size=0.30, stratify=y_all, random_state=SEED
)
y_temp = y_all.loc[temp_idx]
_, test_idx = train_test_split(
    temp_idx, test_size=0.50, stratify=y_temp, random_state=SEED
)

test_app = app.loc[test_idx].reset_index(drop=True)

preds = np.load(ROOT / "output" / "predictions.npz")
assert len(test_app) == len(preds["y_test"]), "Split mismatch: row counts differ"
assert (test_app["TARGET"].values == preds["y_test"]).all(), "Split mismatch: TARGET order differs"

test_app["LGBM_PROB"] = preds["lightgbm"]
test_app["FS_LR_PROB"] = preds["fs_lr"]

# Score band thresholds from output/score_band_table.csv (LightGBM prob scaled to 300-850 style)
def score_band(p):
    if p < 0.20:
        return "640+"
    if p < 0.35:
        return "600-640"
    if p < 0.55:
        return "560-600"
    if p < 0.75:
        return "520-560"
    return "<520"

DISPLAY_COLS = [
    "SK_ID_CURR", "NAME_CONTRACT_TYPE", "CODE_GENDER", "AGE_YEARS", "CNT_CHILDREN",
    "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "EMPLOYED_YEARS",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "TARGET", "LGBM_PROB", "FS_LR_PROB",
]

# Pick applicants spanning risk deciles, deliberately including some real defaults
# (an unweighted sample at an 8% base rate would likely show none at all).
test_app["decile"] = pd.qcut(test_app["LGBM_PROB"], 10, labels=False, duplicates="drop")

samples = []
for d in sorted(test_app["decile"].unique()):
    bucket = test_app[test_app["decile"] == d]
    repaid = bucket[bucket["TARGET"] == 0]
    defaulted = bucket[bucket["TARGET"] == 1]
    n_repaid = 1
    n_defaulted = 1 if len(defaulted) > 0 else 0
    if n_repaid:
        samples.append(repaid.sample(n=min(n_repaid, len(repaid)), random_state=SEED))
    if n_defaulted:
        samples.append(defaulted.sample(n=min(n_defaulted, len(defaulted)), random_state=SEED))

sample_df = pd.concat(samples).drop_duplicates(subset="SK_ID_CURR")
sample_df = sample_df.sort_values("LGBM_PROB").reset_index(drop=True)

records = []
for _, row in sample_df.iterrows():
    rec = {}
    for col in DISPLAY_COLS:
        val = row[col]
        if isinstance(val, (np.floating, float)):
            val = None if pd.isna(val) else round(float(val), 4)
        elif isinstance(val, (np.integer, int)):
            val = int(val)
        else:
            val = str(val)
        rec[col] = val
    rec["SCORE_BAND"] = score_band(rec["LGBM_PROB"])
    rec["OUTCOME"] = "Defaulted" if rec["TARGET"] == 1 else "Repaid"
    records.append(rec)

out_path = ROOT / "output" / "demo_applicants.json"
with open(out_path, "w") as f:
    json.dump(records, f, indent=2)

print(f"Exported {len(records)} demo applicants to {out_path}")
print(f"Outcomes: {sum(r['TARGET'] for r in records)} defaulted / {len(records)} total")
