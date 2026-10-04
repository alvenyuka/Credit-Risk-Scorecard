"""
The release pipeline: fit the model once, build the PDO scorecard, run the
acceptance checks, measure it on the holdout, and write every figure the README
quotes.

Writes:
- outputs/results.json          headline metrics, checks and provenance
- outputs/scorecard_points.json the points table: every bin of every feature
- outputs/val_scores.parquet    holdout scores, PD, outcome and credit amount
                                (gitignored; read by make_figures.py and
                                business_impact.py so a release needs one fit)

Run: python src/run_pipeline.py   (needs the 8 Home Credit CSVs in data/, about 7 minutes)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from business_impact import lending_policy_table
from feature_lists import PROHIBITED_BASES
from fit_model import fit_full_model, prohibited_basis_ablation
from metrics_scratch import (
    auc_rank_sum,
    bootstrap_auc_ci,
    calibration_by_decile,
    default_rate_by_score_decile,
    psi,
)
from results_io import OUTPUT_PATH, read_results, write_results
from scorecard import (
    assert_no_negative_points_for_age_62_plus,
    assert_woe_coefficient_signs,
    build_scorecard,
    destandardize_coefficients,
    odds_at_score,
    prior_correct_intercept,
    reason_codes,
    score_dataframe,
    score_to_pd,
)
from woe_iv import MISSING_LABEL, is_monotonic_woe

OUTPUTS = OUTPUT_PATH.parent
VAL_SCORES_PATH = OUTPUTS / "val_scores.parquet"
POINTS_PATH = OUTPUTS / "scorecard_points.json"
BASE_SCORE, BASE_ODDS, PDO = 600, 20, 40
EXAMPLE_APPROVAL_RATE = 0.8


def build_validation_scores(full=None, verbose: bool = True):
    """Fit the model, build the prior-corrected scorecard, run the acceptance
    checks, and score the holdout.

    Used by main() and by the notebook, so both describe exactly one model.
    Returns (result, sc, X_val, y_val, scores).
    """
    result = fit_full_model(full, verbose=verbose)
    kept = result["kept_cols"]
    if set(kept) & set(PROHIBITED_BASES):
        raise ValueError(f"prohibited bases reached the scorecard: {set(kept) & set(PROHIBITED_BASES)}")

    coef_raw, intercept_balanced = destandardize_coefficients(
        result["model"].coef_, result["model"].intercept_,
        result["standardize_mean"], result["standardize_std"],
    )
    assert_woe_coefficient_signs(coef_raw, kept)

    # class_weight="balanced" fits the intercept to a 50/50 prior; restore the
    # training default rate so that 600 points really means 20:1 good:bad odds.
    intercept = prior_correct_intercept(intercept_balanced, result["train_bad_rate"])

    sc = build_scorecard(coef=coef_raw, intercept=intercept, kept_cols=kept,
                         woe_fits=result["woe_fits"], base_score=BASE_SCORE,
                         base_odds=BASE_ODDS, pdo=PDO)
    sc["coef_raw"] = coef_raw
    sc["intercept_balanced"] = float(intercept_balanced)
    sc["intercept"] = float(intercept)
    result["age_check"] = assert_no_negative_points_for_age_62_plus(sc)

    X_val, y_val = result["X_val_raw"], result["y_val"]
    scores = score_dataframe(X_val, sc)
    return result, sc, X_val, y_val, scores


def _matrix_scores(Xw: np.ndarray, sc: dict) -> np.ndarray:
    """Unclipped scores from a raw WoE matrix (same arithmetic as score_dataframe)."""
    return sc["offset"] - sc["factor"] * (sc["intercept"] + Xw @ sc["coef_raw"])


def _records(df: pd.DataFrame) -> list:
    """DataFrame rows as plain Python dicts (JSON-safe types, NaN as None)."""
    return json.loads(df.to_json(orient="records"))


def select_declined_example(scores: pd.Series, y_val: pd.Series, cut_off: float):
    """First defaulted holdout applicant scoring below the cut-off (holdout order)."""
    below = scores[(y_val == 1) & (scores < cut_off)]
    return below.index[0]


def summarise(result, sc, X_val, y_val, scores) -> dict:
    """Every holdout figure the README, METHODOLOGY and notebook report."""
    y = y_val.to_numpy()
    raw_val = score_dataframe(X_val, sc, clip=False)
    if not np.allclose(raw_val.to_numpy(), _matrix_scores(result["Xw_val"], sc)):
        raise AssertionError("points-table scores and matrix scores disagree")
    pd_val = score_to_pd(raw_val, sc)
    auc_from_pd = auc_rank_sum(y, pd_val)
    if abs(auc_from_pd - result["val_auc"]) > 1e-9:
        raise AssertionError("scorecard ranking differs from the fitted model's ranking")

    train_scores = np.clip(_matrix_scores(result["Xw_train"], sc), 300, 850)
    auc_lo, auc_hi = bootstrap_auc_ci(y, pd_val, n_boot=500, seed=42)
    deciles = default_rate_by_score_decile(scores.to_numpy(), y)
    calib = calibration_by_decile(pd_val, y)

    near = (scores >= BASE_SCORE - 20) & (scores < BASE_SCORE + 20)
    n_bad_near = int((y_val[near] == 1).sum())
    p = result["train_bad_rate"]

    cut = lending_policy_table(scores.to_numpy(), y, X_val["AMT_CREDIT"].to_numpy(),
                               approval_rates=(EXAMPLE_APPROVAL_RATE,)).iloc[0]["cut_off_score"]
    ex_idx = select_declined_example(scores, y_val, cut)
    ex_reasons = reason_codes(X_val.loc[ex_idx], sc, top_n=3)

    from scipy.stats import pointbiserialr
    corr, _ = pointbiserialr(y, scores)
    if corr >= -0.2:
        raise AssertionError("scorecard should be meaningfully negatively correlated with default")

    numeric_kept = [c for c in sc["kept_cols"] if not result["woe_fits"][c]["is_categorical"]]
    model = result["model"]
    emp_pts = sc["points_tables"].get("EMPLOYED_YEARS")

    return {
        "val_auc": float(result["val_auc"]),
        "val_auc_ci95_bootstrap": [auc_lo, auc_hi],
        "val_ks": result["val_ks"],
        "val_gini": 2 * result["val_auc"] - 1,
        "pred_corr_vs_sklearn": result["pred_corr"],
        "coef_diff_max_vs_sklearn": result["coef_diff_max"],
        "n_candidate_features": result["n_candidate_features"],
        "n_categorical_features": result["n_categorical_features"],
        "n_after_iv_screen": result["n_after_iv_screen"],
        "n_after_correlation_gate": result["n_after_correlation_gate"],
        "n_features_kept": len(sc["kept_cols"]),
        "correlation_gate_dropped": [
            {"dropped": c, "kept": k, "r": r} for c, k, r in result["correlation_gate_dropped"]],
        "sign_gate_removed": [
            {"feature": c, "coefficient_at_removal": b} for c, b in result["sign_gate_removed"]],
        "all_coefficients_negative": bool((sc["coef_raw"] < 0).all()),
        "n_numeric_features_monotonic_woe": int(sum(bool(is_monotonic_woe(result["woe_fits"][c]))
                                                    for c in numeric_kept)),
        "n_numeric_features_kept": len(numeric_kept),
        "n_validation_rows": int(len(y_val)),
        "train_bad_rate": p,
        "val_bad_rate": float(y.mean()),
        "intercept_balanced": sc["intercept_balanced"],
        "intercept_prior_corrected": sc["intercept"],
        "implied_odds_at_base_score": odds_at_score(BASE_SCORE, sc),
        "observed_odds_near_base_score": {
            "score_band": [BASE_SCORE - 20, BASE_SCORE + 20],
            "n": int(near.sum()),
            "good_to_bad": float((near.sum() - n_bad_near) / n_bad_near) if n_bad_near else None,
        },
        "base_points_implied_odds": odds_at_score(sc["base_points"], sc),
        "train_good_to_bad_odds": float((1 - p) / p),
        "mean_predicted_pd_val": float(pd_val.mean()),
        "calibration_by_decile": _records(calib),
        "calibration_max_abs_gap": float(calib["gap"].abs().max()),
        "default_rate_by_score_decile": _records(deciles),
        "default_rate_lowest_decile": float(deciles["default_rate"].iloc[0]),
        "default_rate_highest_decile": float(deciles["default_rate"].iloc[-1]),
        "psi_train_to_holdout_scores": float(psi(train_scores, scores.to_numpy())),
        "point_biserial_score_vs_default": float(corr),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "share_clipped_at_300": float((scores <= 300).mean()),
        "share_clipped_at_850": float((scores >= 850).mean()),
        "mean_score_repaid": float(scores[y_val == 0].mean()),
        "mean_score_defaulted": float(scores[y_val == 1].mean()),
        "scorecard_base_points": float(sc["base_points"]),
        "scorecard_factor": float(sc["factor"]),
        "scorecard_offset": float(sc["offset"]),
        "example_declined_applicant": {
            "holdout_row": int(ex_idx),
            "outcome": "defaulted",
            "score": float(scores.loc[ex_idx]),
            "cut_off_score_at_80pct_approval": float(cut),
            "reasons_points_below_best_bin": [{"feature": c, "points": v} for c, v in ex_reasons],
        },
        "reg_b_age_62_plus": result["age_check"],
        "employed_years_missing_bin_points": (
            float(emp_pts[MISSING_LABEL]) if emp_pts is not None and MISSING_LABEL in emp_pts.index else None),
        "solver": {
            "n_iter": int(model.n_iter_),
            "max_iter": int(model.n_iter),
            "tol": float(model.tol),
            "converged": bool(model.converged_),
            "final_cost_change": model.final_cost_change_,
        },
    }


def points_table_records(result, sc) -> list:
    """The scorecard as data: one entry per feature with its ordered bins and points."""
    out = []
    for i, col in enumerate(sc["kept_cols"]):
        fit = result["woe_fits"][col]
        table = fit["table"]
        pts = sc["points_tables"][col]
        out.append({
            "feature": col,
            "type": "categorical" if fit["is_categorical"] else "numeric",
            "iv": fit["iv"],
            "coefficient_raw_woe": float(sc["coef_raw"][i]),
            "monotonic_woe": is_monotonic_woe(fit),
            "bins": [{"bin": str(b), "n_train": int(table.loc[b, "n"]),
                      "woe": float(table.loc[b, "woe"]), "points": float(pts.loc[b])}
                     for b in table.index],
        })
    return out


def load_validation_scores() -> pd.DataFrame:
    """Holdout scores written by main(), checked against results.json."""
    if not VAL_SCORES_PATH.exists():
        raise FileNotFoundError(f"{VAL_SCORES_PATH} not found. Run `python src/run_pipeline.py` first.")
    df = pd.read_parquet(VAL_SCORES_PATH)
    m = read_results()["metrics"]
    if len(df) != m["n_validation_rows"] or abs(df["score"].mean() - (
            m["mean_score_repaid"] * (1 - m["val_bad_rate"]) + m["mean_score_defaulted"] * m["val_bad_rate"])) > 1e-6:
        raise ValueError("val_scores.parquet does not match outputs/results.json; rerun run_pipeline.py")
    return df


def main():
    """Fit once, print the release summary, and write results.json, the points table and the holdout scores."""
    result, sc, X_val, y_val, scores = build_validation_scores()
    summary = summarise(result, sc, X_val, y_val, scores)

    print("\n=== Scorecard ===")
    print(f"train bad rate={summary['train_bad_rate']:.4f}  intercept balanced={summary['intercept_balanced']:+.4f}"
          f" -> prior-corrected={summary['intercept_prior_corrected']:+.4f}")
    print(f"base_points={sc['base_points']:.1f}  factor={sc['factor']:.2f}  offset={sc['offset']:.1f}")
    print(f"implied odds at {BASE_SCORE}: {summary['implied_odds_at_base_score']:.2f}:1; observed good:bad "
          f"among holdout applicants scoring {BASE_SCORE - 20}-{BASE_SCORE + 20}: "
          f"{summary['observed_odds_near_base_score']['good_to_bad']:.2f}:1")
    print(f"mean predicted PD (holdout) {summary['mean_predicted_pd_val']:.4f} vs observed {summary['val_bad_rate']:.4f}")
    print(f"observed score range on the holdout: {summary['score_min']:.0f} - {summary['score_max']:.0f} "
          f"(share clipped at 300: {summary['share_clipped_at_300']:.4%})")
    print(f"mean score, repaid: {summary['mean_score_repaid']:.1f}   defaulted: {summary['mean_score_defaulted']:.1f}")
    print(f"AUC {summary['val_auc']:.4f} (95% bootstrap CI {summary['val_auc_ci95_bootstrap'][0]:.4f}"
          f" to {summary['val_auc_ci95_bootstrap'][1]:.4f})  KS {summary['val_ks']:.4f}"
          f"  PSI train->holdout {summary['psi_train_to_holdout_scores']:.4f}")
    print(f"point-biserial correlation(score, default): {summary['point_biserial_score_vs_default']:.3f}")
    print(f"Reg B age check: {summary['reg_b_age_62_plus']}")

    print("\ncalibration by predicted-PD decile:")
    print(pd.DataFrame(summary["calibration_by_decile"]).to_string(index=False))
    print("\ndefault rate by score decile:")
    print(pd.DataFrame(summary["default_rate_by_score_decile"]).to_string(index=False))

    ex = summary["example_declined_applicant"]
    print(f"\nexample: defaulted applicant below the 80% cut-off ({ex['cut_off_score_at_80pct_approval']:.0f}),"
          f" score {ex['score']:.0f}:")
    for r in ex["reasons_points_below_best_bin"]:
        print(f"  {r['feature']:28s} {r['points']:.1f} points below the best bin")

    print("\nprohibited-basis ablation (scikit-learn, same split):")
    ablation = prohibited_basis_ablation(result)
    print(f"  final features: AUC {ablation['val_auc_sklearn_final_features']:.4f}; "
          f"with sex and marital status added: {ablation['val_auc_sklearn_with_prohibited_bases']:.4f}")
    summary["prohibited_basis_ablation"] = ablation

    path = write_results(
        summary,
        data_source="real",
        notes=(
            "Home Credit Default Risk, 8 relational tables. Validation is a random stratified "
            "20% holdout, not the Kaggle test set, so these figures are not leaderboard scores. "
            "Scores use the prior-corrected intercept, so 600 points corresponds to 20:1 good:bad "
            "odds. The prohibited-basis ablation is a documentation check; sex and marital status "
            "never enter the scorecard."
        ),
    )
    POINTS_PATH.write_text(json.dumps(points_table_records(result, sc), indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    pd.DataFrame({
        "score": scores.to_numpy(),
        "pd": score_to_pd(score_dataframe(X_val, sc, clip=False), sc),
        "TARGET": y_val.to_numpy(),
        "AMT_CREDIT": X_val["AMT_CREDIT"].to_numpy(),
    }, index=X_val.index).to_parquet(VAL_SCORES_PATH)
    print(f"\nwrote {path}, {POINTS_PATH.name} and {VAL_SCORES_PATH.name}")
    return sc, scores


if __name__ == "__main__":
    main()
