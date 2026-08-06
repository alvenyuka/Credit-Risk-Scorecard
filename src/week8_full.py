"""
Week 8, final step: build the PDO scorecard from the Week 8 model and show
it actually being read the way a loan officer would -- a score plus reason
codes for a couple of real applicants, not just a validation-set AUC number.
"""
import numpy as np

import week8_run
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

    return sc, scores


if __name__ == "__main__":
    main()
