"""
End-to-end Home Credit Default Risk pipeline.

PHASE A: Load + engineer application-level features
PHASE B: Aggregate relational tables (bureau, previous_application, POS_CASH,
         credit_card_balance, installments_payments) and merge
PHASE C: Train/val/test split (stratified)
PHASE D: WoE/IV feature selection on TRAIN ONLY
PHASE E: From-scratch logistic regression, validated against sklearn
PHASE F: LightGBM benchmark
PHASE G: Metrics (hand-coded AUC/GINI/KS/PSI/Brier/calibration), validated against sklearn
PHASE H: Persist results

Every number in README.md, BUILD_STATUS.md, and Credit_Risk_Scorecard.ipynb
traces back to a run of this exact script -- nothing is hand-restated. If a
claim in those docs doesn't match a fresh run of this file, the docs are
wrong, not the other way around (see README.md's opening paragraph for why
that guarantee matters more than usual on this particular project).
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import roc_auc_score as sk_roc_auc, brier_score_loss as sk_brier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.io_utils import (
    load_application, load_bureau, load_bureau_balance, load_previous_application,
    load_pos_cash, load_credit_card_balance, load_installments,
)
from src.feature_engineering_app import engineer_application_features
from src.feature_engineering_relational import (
    aggregate_bureau, aggregate_previous_application, aggregate_pos_cash,
    aggregate_credit_card_balance, aggregate_installments,
)
from src.woe_iv import iv_ranking
from src.from_scratch_lr import FromScratchLogisticRegression
from src import metrics as m

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

t_start = time.time()


def phase(label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")


# =============================================================================
phase("PHASE A: Application-level features")
# =============================================================================
app = load_application("train")
print(f"application_train: {app.shape}")
app = engineer_application_features(app)
print(f"After engineering: {app.shape}")

# =============================================================================
phase("PHASE B: Relational table aggregation")
# =============================================================================
t0 = time.time()
bureau = load_bureau()
bureau_balance = load_bureau_balance()
bureau_agg = aggregate_bureau(bureau, bureau_balance)
print(f"  bureau -> {bureau_agg.shape}  ({time.time()-t0:.0f}s)")
del bureau, bureau_balance

t0 = time.time()
prev = load_previous_application()
prev_agg = aggregate_previous_application(prev)
print(f"  previous_application -> {prev_agg.shape}  ({time.time()-t0:.0f}s)")
del prev

t0 = time.time()
pos = load_pos_cash()
pos_agg = aggregate_pos_cash(pos)
print(f"  POS_CASH_balance -> {pos_agg.shape}  ({time.time()-t0:.0f}s)")
del pos

t0 = time.time()
cc = load_credit_card_balance()
cc_agg = aggregate_credit_card_balance(cc)
print(f"  credit_card_balance -> {cc_agg.shape}  ({time.time()-t0:.0f}s)")
del cc

t0 = time.time()
inst = load_installments()
inst_agg = aggregate_installments(inst)
print(f"  installments_payments -> {inst_agg.shape}  ({time.time()-t0:.0f}s)")
del inst

df = app
for agg in [bureau_agg, prev_agg, pos_agg, cc_agg, inst_agg]:
    df = df.merge(agg, on="SK_ID_CURR", how="left")
print(f"\nFinal joined shape: {df.shape}")

# =============================================================================
phase("PHASE C: Train/val/test split (stratified)")
# =============================================================================
target_col = "TARGET"
id_col = "SK_ID_CURR"
feature_cols = [c for c in df.columns if c not in (target_col, id_col)]

# Drop non-numeric (categorical raw strings) for this pass -- WoE/IV ranking
# handles them, but the from-scratch LR needs numeric input only.
numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

X_all = df[numeric_feature_cols]
y_all = df[target_col]

# Two-stage split (70/30, then 50/50 of the 30) rather than a single
# three-way split because sklearn's train_test_split only splits in two;
# stratifying twice on the same target keeps the ~8.07% default rate
# consistent across all three resulting sets (see the printed rates below --
# train/val/test land within 0.001 of each other). val exists to give
# LightGBM's early stopping (Phase F) a held-out set to watch that isn't
# the final test set it gets scored against.
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.30, stratify=y_all, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)
print(f"Train: {X_train.shape}  default rate {y_train.mean():.4f}")
print(f"Val:   {X_val.shape}  default rate {y_val.mean():.4f}")
print(f"Test:  {X_test.shape}  default rate {y_test.mean():.4f}")

# =============================================================================
phase("PHASE D: WoE/IV feature selection (TRAIN ONLY)")
# =============================================================================
t0 = time.time()
train_with_target = X_train.copy()
train_with_target[target_col] = y_train.values
iv_table = iv_ranking(train_with_target, target_col, numeric_feature_cols, bins=10)
print(f"IV computed for {len(iv_table)} features in {time.time()-t0:.0f}s")
print(iv_table.head(15).to_string(index=False))

# 80 of 400 candidate features: a round-number cutoff, not a value chosen by
# tuning against test-set performance (which would itself be a form of
# leakage). IV has already fallen from >0.6 (rank 1) to "Weak" territory
# (~0.046, see woe_iv.py's IV_STRENGTH_BANDS) by rank 80 -- the top ~15-20
# features (the EXT_SOURCE cluster and its interactions, see
# figs/iv_top20.png) carry most of the real signal.
N_FEATURES = 80
selected_features = iv_table.head(N_FEATURES)["feature"].tolist()
print(f"\nSelected top {len(selected_features)} features by IV")

# =============================================================================
phase("PHASE E: From-scratch logistic regression, validated against sklearn")
# =============================================================================
X_train_sel = X_train[selected_features].fillna(X_train[selected_features].median())
X_val_sel = X_val[selected_features].fillna(X_train[selected_features].median())
X_test_sel = X_test[selected_features].fillna(X_train[selected_features].median())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_val_scaled = scaler.transform(X_val_sel)
X_test_scaled = scaler.transform(X_test_sel)

print("Training from-scratch LR...")
t0 = time.time()
fs_lr = FromScratchLogisticRegression(learning_rate=0.5, n_iterations=1500, l2=1.0, verbose=True)
fs_lr.fit(X_train_scaled, y_train.values)
print(f"  done in {time.time()-t0:.0f}s, {len(fs_lr.loss_history_)} iterations")

print("Training sklearn LR (same data, same L2) for validation...")
sk_lr = SklearnLR(C=1.0, max_iter=1500, solver="lbfgs")
sk_lr.fit(X_train_scaled, y_train.values)

fs_probs_test = fs_lr.predict_proba(X_test_scaled)[:, 1]
sk_probs_test = sk_lr.predict_proba(X_test_scaled)[:, 1]

max_coef_diff = float(np.max(np.abs(fs_lr.coef_ - sk_lr.coef_[0])))
prob_correlation = float(np.corrcoef(fs_probs_test, sk_probs_test)[0, 1])
fs_auc_sk_metric = float(sk_roc_auc(y_test, fs_probs_test))
sk_auc_sk_metric = float(sk_roc_auc(y_test, sk_probs_test))

print(f"\nValidation vs sklearn:")
print(f"  max |coef diff|      : {max_coef_diff:.6f}")
print(f"  probability corr     : {prob_correlation:.6f}")
print(f"  from-scratch test AUC: {fs_auc_sk_metric:.4f}")
print(f"  sklearn test AUC     : {sk_auc_sk_metric:.4f}")

# =============================================================================
phase("PHASE F: LightGBM benchmark")
# =============================================================================
X_train_lgb = X_train[selected_features]
X_val_lgb = X_val[selected_features]
X_test_lgb = X_test[selected_features]

lgb_train = lgb.Dataset(X_train_lgb, label=y_train)
lgb_val = lgb.Dataset(X_val_lgb, label=y_val, reference=lgb_train)

lgb_params = {
    "objective": "binary", "metric": "auc", "is_unbalance": True,
    "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6,
    "min_child_samples": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 5, "lambda_l2": 1.0, "verbose": -1, "seed": SEED,
}
t0 = time.time()
lgb_model = lgb.train(
    lgb_params, lgb_train, num_boost_round=2000, valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)],
)
print(f"LightGBM trained in {time.time()-t0:.0f}s, best iter {lgb_model.best_iteration}")

lgb_probs_test = lgb_model.predict(X_test_lgb, num_iteration=lgb_model.best_iteration)

# =============================================================================
phase("PHASE G: Metrics -- hand-coded, validated against sklearn")
# =============================================================================
def report_model(name, y_true, y_prob):
    y_true_arr = y_true.values if hasattr(y_true, "values") else y_true
    auc_hc = m.roc_auc(y_true_arr, y_prob)
    auc_sk = float(sk_roc_auc(y_true_arr, y_prob))
    gini_hc = m.gini(y_true_arr, y_prob)
    ks_hc = m.ks_statistic(y_true_arr, y_prob)
    brier_hc = m.brier_score(y_true_arr, y_prob)
    brier_sk = float(sk_brier(y_true_arr, y_prob))
    print(f"\n{name}")
    print(f"  AUC   hand-coded={auc_hc:.6f}  sklearn={auc_sk:.6f}  diff={abs(auc_hc-auc_sk):.2e}")
    print(f"  GINI  = {gini_hc:.4f}")
    print(f"  KS    = {ks_hc:.4f}")
    print(f"  Brier hand-coded={brier_hc:.6f}  sklearn={brier_sk:.6f}  diff={abs(brier_hc-brier_sk):.2e}")
    return {"auc": auc_hc, "auc_sklearn": auc_sk, "gini": gini_hc, "ks": ks_hc,
            "brier": brier_hc, "brier_sklearn": brier_sk}

results = {}
results["from_scratch_lr"] = report_model("From-scratch LR", y_test, fs_probs_test)
results["sklearn_lr"] = report_model("sklearn LR (validation reference)", y_test, sk_probs_test)
results["lightgbm"] = report_model("LightGBM", y_test, lgb_probs_test)

psi = m.population_stability_index(fs_probs_test[:len(fs_probs_test)//2], fs_probs_test[len(fs_probs_test)//2:])
print(f"\nPSI (test split in half, sanity check): {psi:.5f}")
results["psi_test_split_sanity"] = psi

# =============================================================================
phase("PHASE H: Persist results")
# =============================================================================
out_dir = Path(__file__).resolve().parent / "output"
out_dir.mkdir(exist_ok=True)

results["validation"] = {
    "max_coef_diff_vs_sklearn": max_coef_diff,
    "probability_correlation_vs_sklearn": prob_correlation,
}
results["dataset"] = {
    "train_rows": int(len(X_train)), "val_rows": int(len(X_val)), "test_rows": int(len(X_test)),
    "train_default_rate": float(y_train.mean()), "test_default_rate": float(y_test.mean()),
    "n_features_selected": len(selected_features), "n_features_available": len(numeric_feature_cols),
}

with open(out_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

iv_table.to_csv(out_dir / "feature_iv_ranking.csv", index=False)
with open(out_dir / "top_features.json", "w") as f:
    json.dump(selected_features, f, indent=2)

np.savez(out_dir / "predictions.npz",
         y_test=y_test.values, fs_lr=fs_probs_test, sklearn_lr=sk_probs_test, lightgbm=lgb_probs_test)

print(f"\nResults written to {out_dir}")
print(f"\nTOTAL PIPELINE TIME: {time.time()-t_start:.0f}s")
