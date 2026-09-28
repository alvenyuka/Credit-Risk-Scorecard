"""
Week 8, final step: build the PDO scorecard from the Week 8 model and show
it actually being read the way a loan officer would -- a score plus reason
codes for a couple of real applicants, not just a validation-set AUC number.
"""
import numpy as np

import week8_run
from results_io import write_results
from scorecard import build_scorecard, destandardize_coefficients, reason_codes, score_dataframe


def main():
    result = week8_run.main()

    coef_raw, intercept_raw = destandardize_coefficients(
        result["model"].coef_, result["model"].intercept_,
        result["standardize_mean"], result["standardize_std"],
    )

    sc = build_scorecard(
        coef=coef_raw, intercept=intercept_raw, kept_cols=result["kept_cols"],
        woe_fits=result["woe_fits"], base_score=600, base_odds=20, pdo=40,
    )

    X_val = result["X_val_raw"]
    y_val = result["y_val"]
    scores = score_dataframe(X_val, sc)

    print("\n=== Scorecard ===")
    print(f"base_points={sc['base_points']:.1f}  factor={sc['factor']:.2f}  offset={sc['offset']:.1f}")
    print(f"score range on validation set: {scores.min():.0f} - {scores.max():.0f}")
    print(f"mean score, actually-good applicants: {scores[y_val == 0].mean():.1f}")
    print(f"mean score, actually-defaulted applicants: {scores[y_val == 1].mean():.1f}")

    # sanity check the scorecard is monotonic with real risk: correlate score with actual outcome
    from scipy.stats import pointbiserialr
    corr, _ = pointbiserialr(y_val, scores)
    print(f"point-biserial correlation(score, default): {corr:.3f}  (should be clearly negative: higher score = lower risk)")
    assert corr < -0.2, "scorecard should be meaningfully negatively correlated with default"

    print("\n=== Two example applicants, scored and explained ===")
    good_idx = X_val.index[y_val == 0][0]
    bad_idx = X_val.index[y_val == 1][0]

    for label, idx in [("a repaid loan", good_idx), ("a defaulted loan", bad_idx)]:
        row = X_val.loc[idx]
        s = scores.loc[idx]
        print(f"\n{label} (score={s:.0f}):")
        for col, pts in reason_codes(row, sc, top_n=3):
            print(f"  {col:28s} {pts:+.1f} points")

    # Every number the README quotes is written here by the code that computed
    # it, never transcribed by hand. See src/results_io.py for why.
    path = write_results(
        {
            "val_auc": result["val_auc"],
            "val_ks": result["val_ks"],
            "val_gini": 2 * result["val_auc"] - 1,
            "pred_corr_vs_sklearn": result["pred_corr"],
            "coef_diff_max_vs_sklearn": result["coef_diff_max"],
            "n_features_kept": len(result["kept_cols"]),
            "n_validation_rows": int(len(y_val)),
            "point_biserial_score_vs_default": float(corr),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "mean_score_repaid": float(scores[y_val == 0].mean()),
            "mean_score_defaulted": float(scores[y_val == 1].mean()),
            "scorecard_base_points": float(sc["base_points"]),
            "scorecard_factor": float(sc["factor"]),
        },
        data_source="real",
        notes=(
            "Home Credit Default Risk, 8 relational tables. Validation is a holdout "
            "split, not the Kaggle test set, so these figures are not leaderboard "
            "scores. coef_diff_max_vs_sklearn is expected to be larger on the full "
            "selected feature set than on a small clean one; that is multicollinearity "
            "among near-duplicate selected features, not a solver defect."
        ),
    )
    print(f"\nwrote {path}")

    return sc, scores


if __name__ == "__main__":
    main()
