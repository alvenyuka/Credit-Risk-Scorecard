"""
Regenerate figs/ from the artifacts run_pipeline.py writes to output/.

Run this after run_pipeline.py, never before -- every chart here reads
predictions.npz, feature_iv_ranking.csv, or score_band_table.csv, so it only
plots what the pipeline actually produced.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import metrics as m

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
FIGS_DIR = ROOT / "figs"
FIGS_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

preds = np.load(OUT_DIR / "predictions.npz")
y_test = preds["y_test"]
fs_lr_probs = preds["fs_lr"]
lgbm_probs = preds["lightgbm"]

# =============================================================================
# 1. IV top 20
# =============================================================================
iv_table = pd.read_csv(OUT_DIR / "feature_iv_ranking.csv").head(20)

fig, ax = plt.subplots(figsize=(8.8, 7.2))
ax.barh(iv_table["feature"][::-1], iv_table["iv"][::-1], color="#2f6f9f")
ax.set_xlabel("Information Value")
ax.set_title("Top 20 features by Information Value")
fig.tight_layout()
fig.savefig(FIGS_DIR / "iv_top20.png", dpi=130)
plt.close(fig)

# =============================================================================
# 2. Calibration curve -- from-scratch LR vs LightGBM
# =============================================================================
cal_lr = m.calibration_curve(y_test, fs_lr_probs, n_bins=10)
cal_lgbm = m.calibration_curve(y_test, lgbm_probs, n_bins=10)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(cal_lr["mean_predicted"], cal_lr["observed_rate"], "o-", label="From-scratch LR")
ax.plot(cal_lgbm["mean_predicted"], cal_lgbm["observed_rate"], "o-", label="LightGBM", color="#e0703d")
ax.plot([0, 0.5], [0, 0.5], "--", color="gray", label="Perfect calibration")
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 0.5)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed default rate")
ax.set_title("Calibration curve")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS_DIR / "calibration_curve.png", dpi=130)
plt.close(fig)

# =============================================================================
# 3. Lift chart -- LightGBM, by score decile
# =============================================================================
decile = pd.qcut(lgbm_probs, 10, labels=False, duplicates="drop") + 1
overall_rate = y_test.mean()
lift_df = pd.DataFrame({"decile": decile, "y": y_test}).groupby("decile")["y"].mean()
lift = lift_df / overall_rate

fig, ax = plt.subplots(figsize=(8.8, 5.3))
ax.bar(lift.index, lift.values, color="#1c3f4d")
ax.axhline(1.0, color="red", linestyle="--", label="Baseline (no model)")
ax.set_xlabel("Score decile (1 = lowest risk, 10 = highest risk)")
ax.set_ylabel("Lift over average default rate")
ax.set_title("Lift chart -- LightGBM")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS_DIR / "lift_chart.png", dpi=130)
plt.close(fig)

# =============================================================================
# 4. Predicted probability distribution by outcome -- LightGBM
# =============================================================================
fig, ax = plt.subplots(figsize=(8.8, 5.3))
ax.hist(lgbm_probs[y_test == 0], bins=50, density=True, alpha=0.6, color="#5aa9a3", label="Non-default")
ax.hist(lgbm_probs[y_test == 1], bins=50, density=True, alpha=0.6, color="#e07b7b", label="Default")
ax.set_xlabel("Predicted probability of default (LightGBM)")
ax.set_ylabel("Density")
ax.set_title("Predicted probability distribution by outcome")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS_DIR / "score_distribution.png", dpi=130)
plt.close(fig)

print(f"Wrote 4 figures to {FIGS_DIR}")
