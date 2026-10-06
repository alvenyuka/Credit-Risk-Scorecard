"""
Champion against challengers, decided without touching the holdout.

The published scorecard (decile WoE bins) is the champion. Two challengers act on
recommendations the project itself made:

1. Coarse classing (binning="monotone"): bins merged until the default rate moves in one
   direction, with a zero bin for zero-inflated columns. Only 11 of the champion's 32
   numeric characteristics have monotonic WoE, and a reviewer expects all of them to.
2. LightGBM on the same 62 candidate characteristics: the notebook said a gradient-boosted
   model "would probably rank applicants somewhat better". This measures by how much, which
   is the price of the scorecard's interpretability.

Decision rule, fixed before any result was seen: the coarse-classed scorecard replaces the
champion if its AUC on the validation split is at most 0.005 below the champion's. LightGBM
is a benchmark only; it cannot replace a scorecard that must give reasons for declines.

The validation split is the last 20% of the development applicants (the 80% of applicants
that are not the holdout), drawn by fit_full_model's own stratified split. The holdout AUC
of every candidate is reported for information after the decision.

Writes outputs/challengers.json. Run: python src/challengers.py   (about 15 minutes)
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from feature_lists import MODEL_CATEGORICAL, MODEL_NUMERIC
from fit_model import SEED, build_full_feature_table, fit_full_model
from results_io import OUTPUT_PATH
from woe_iv import is_monotonic_woe

MAX_AUC_LOSS = 0.005
OUT_PATH = OUTPUT_PATH.parent / "challengers.json"


def scorecard_summary(result: dict) -> dict:
    """AUC, KS, feature counts and monotonicity for one fit_full_model result."""
    numeric = [c for c in result["kept_cols"] if not result["woe_fits"][c]["is_categorical"]]
    return {
        "auc": result["val_auc"],
        "ks": result["val_ks"],
        "n_features_kept": len(result["kept_cols"]),
        "n_numeric_kept": len(numeric),
        "n_numeric_monotonic": int(sum(bool(is_monotonic_woe(result["woe_fits"][c])) for c in numeric)),
        "sign_gate_removed": [c for c, _ in result["sign_gate_removed"]],
    }


def lightgbm_auc(train: pd.DataFrame, test: pd.DataFrame, n_estimators: int | None = None) -> tuple:
    """Fit LightGBM on `train` and return (AUC on `test`, number of trees).

    With n_estimators=None the tree count is chosen by early stopping on the last 20% of
    `train` (stratified), so `test` is never used to choose it.
    """
    cols = MODEL_NUMERIC + MODEL_CATEGORICAL

    def prep(frame):
        X = frame[cols].copy()
        for c in MODEL_CATEGORICAL:
            X[c] = X[c].astype("category")
        return X

    params = dict(learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=4)
    if n_estimators is None:
        fit, stop = train_test_split(train, test_size=0.2, random_state=SEED, stratify=train["TARGET"])
        model = lgb.LGBMClassifier(n_estimators=3000, **params)
        model.fit(prep(fit), fit["TARGET"], eval_set=[(prep(stop), stop["TARGET"])],
                  eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        n_estimators = int(model.best_iteration_)
    model = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
    model.fit(prep(train), train["TARGET"])
    return float(roc_auc_score(test["TARGET"], model.predict_proba(prep(test))[:, 1])), n_estimators


def main():
    full = build_full_feature_table()
    # The same holdout as the published pipeline: fit_full_model splits on these rows, seed and target.
    dev_idx, holdout_idx = train_test_split(full.index, test_size=0.2, random_state=SEED, stratify=full["TARGET"])
    dev = full.loc[dev_idx]

    # Decision stage: fit_full_model on the development rows splits them again, 80/20, so its
    # "val_auc" here is a validation AUC inside the development data.
    print("fitting the champion and the coarse-classed challenger on the development split ...")
    champion_val = scorecard_summary(fit_full_model(dev, verbose=False, binning="decile"))
    coarse_val = scorecard_summary(fit_full_model(dev, verbose=False, binning="monotone"))
    v_fit, v_val = train_test_split(dev, test_size=0.2, random_state=SEED, stratify=dev["TARGET"])
    lgbm_val, lgbm_trees = lightgbm_auc(v_fit, v_val)

    auc_loss = champion_val["auc"] - coarse_val["auc"]
    adopt = auc_loss <= MAX_AUC_LOSS
    print(f"validation AUC: champion {champion_val['auc']:.4f}, coarse classing {coarse_val['auc']:.4f} "
          f"(loss {auc_loss:+.4f}), LightGBM {lgbm_val:.4f}")
    print(f"decision: {'adopt coarse classing' if adopt else 'keep the champion'} (rule: loss <= {MAX_AUC_LOSS})")

    # Information stage: the holdout, after the decision.
    print("scoring every candidate on the holdout ...")
    champion_hold = scorecard_summary(fit_full_model(full, verbose=False, binning="decile"))
    coarse_hold = scorecard_summary(fit_full_model(full, verbose=False, binning="monotone"))
    lgbm_hold, _ = lightgbm_auc(dev, full.loc[holdout_idx], n_estimators=lgbm_trees)

    out = {
        "decision_rule": f"adopt coarse classing if its validation AUC is at most {MAX_AUC_LOSS} below the champion's",
        "validation": {"rows_fit": int(len(dev) * 0.8), "champion": champion_val, "coarse_classing": coarse_val,
                       "lightgbm": {"auc": lgbm_val, "n_trees": lgbm_trees}},
        "coarse_classing_auc_loss_validation": auc_loss,
        "decision": "adopt coarse classing" if adopt else "keep the champion",
        "holdout": {"champion": champion_hold, "coarse_classing": coarse_hold, "lightgbm": {"auc": lgbm_hold}},
        "interpretability_price_holdout_auc": lgbm_hold - champion_hold["auc"],
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"holdout AUC: champion {champion_hold['auc']:.4f}, coarse classing {coarse_hold['auc']:.4f}, "
          f"LightGBM {lgbm_hold:.4f}")
    print(f"wrote {OUT_PATH}")
    return out


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
