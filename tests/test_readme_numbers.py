"""Every figure the README quotes, checked against the files the pipeline wrote.

The README is typed by hand, but its numbers come from outputs/results.json,
outputs/business_impact.json and outputs/challengers.json. These tests format
each number from those files at the precision the README prints it (0.754, not
0.7539670402) and check that exact text is in the README. A re-run that moves a
number, or an edit that mistypes one, fails here instead of reaching a reader.

No data is needed: the outputs are committed, so these run in CI.
"""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def load_json(name):
    with open(ROOT / "outputs" / name, encoding="utf-8") as f:
        return json.load(f)


METRICS = load_json("results.json")["metrics"]
POLICIES = load_json("business_impact.json")["policies"]
CHALLENGERS = load_json("challengers.json")


def assert_readme_says(phrase):
    assert phrase in README, f"README does not contain {phrase!r}"


def policy_at(approval_rate):
    # The row of the money table for one approval rate
    for row in POLICIES:
        if abs(row["approval_rate"] - approval_rate) < 1e-9:
            return row
    raise AssertionError(f"no policy row for approval rate {approval_rate}")


def count_test_functions():
    # Count every test function in tests/. No test here is parametrized, so this
    # equals the number pytest collects.
    total = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                total += 1
    return total


def test_headline_ranking_figures():
    auc = METRICS["val_auc"]
    low, high = METRICS["val_auc_ci95_bootstrap"]
    assert f"{auc:.3f}" == "0.754"
    assert_readme_says(f"AUC {auc:.3f}")
    assert_readme_says(f"95% bootstrap interval {low:.3f} to {high:.3f}")
    assert_readme_says(f"| AUC | {auc:.3f} |")
    assert_readme_says(f"| KS | {METRICS['val_ks']:.3f} |")
    assert_readme_says(f"| Gini | {METRICS['val_gini']:.3f} |")
    assert f"{METRICS['val_ks']:.3f}" == "0.381"


def test_holdout_size_and_default_rate():
    n_holdout = METRICS["n_validation_rows"]
    assert n_holdout == 61_503
    assert_readme_says(f"({n_holdout:,})")
    assert_readme_says(f"| Measure (holdout of {n_holdout:,}) |")
    assert_readme_says(f"({METRICS['val_bad_rate']:.1%} defaulted)")


def test_default_rate_by_score_decile():
    lowest = METRICS["default_rate_lowest_decile"]
    highest = METRICS["default_rate_highest_decile"]
    assert_readme_says(f"defaulted {lowest:.1%} of the time, the highest-scoring tenth {highest:.1%}")


def test_money_table_matches_business_impact():
    everyone = policy_at(1.0)
    assert_readme_says(
        f"| Approve everyone | {everyone['approved']:,} | {everyone['default_rate_approved']:.1%} | "
        f"{everyone['credit_loss'] / 1e9:.2f} bn | n/a | 0% |"
    )
    for rate in (0.9, 0.8, 0.7):
        row = policy_at(rate)
        assert_readme_says(
            f"| Approve top {rate:.0%} (score {row['cut_off_score']:.1f} or more) | {row['approved']:,} | "
            f"{row['default_rate_approved']:.1%} | {row['credit_loss'] / 1e9:.2f} bn | "
            f"{row['loss_avoided_pct']:.0%} | {row['good_borrowers_declined_pct']:.1%} |"
        )


def test_eighty_percent_policy_headline():
    row = policy_at(0.8)
    assert f"{row['loss_avoided_pct']:.0%}" == "47%"
    assert_readme_says(f"Approving only the top 80% cuts credit losses by {row['loss_avoided_pct']:.0%}.")
    assert_readme_says(f"turning away {row['good_borrowers_declined_pct']:.1%} of the borrowers who would have repaid")


def test_score_scale_and_stability_table():
    assert_readme_says(f"| {METRICS['mean_score_repaid']:.0f} vs {METRICS['mean_score_defaulted']:.0f} |")
    assert_readme_says(f"Scores run from {METRICS['score_min']:.0f} to {METRICS['score_max']:.0f} on the holdout")
    observed = METRICS["observed_odds_near_base_score"]
    assert_readme_says(f"| {observed['good_to_bad']:.1f} observed, {METRICS['implied_odds_at_base_score']:.1f} designed |")
    assert_readme_says(f"Applicants scoring {observed['score_band'][0]} to {observed['score_band'][1]}")
    assert_readme_says(f"| {METRICS['psi_train_to_holdout_scores']:.4f} |")


def test_calibration_gap():
    gap_points = METRICS["calibration_max_abs_gap"] * 100
    assert_readme_says(f"agree within {gap_points:.1f} percentage points")


def test_readability_price():
    scorecard_auc = METRICS["val_auc"]
    lightgbm_auc = CHALLENGERS["holdout"]["lightgbm"]["auc"]
    assert_readme_says(
        f"reaches AUC {lightgbm_auc:.3f} on the same applicants, {lightgbm_auc - scorecard_auc:.3f} above "
        f"the scorecard's {scorecard_auc:.3f}"
    )


def test_challenger_figures_in_step_11():
    assert_readme_says(f"It cost {CHALLENGERS['coarse_classing_auc_loss_validation']:.4f}")
    holdout = CHALLENGERS["holdout"]
    assert_readme_says(f"equal-size bins scored AUC {holdout['champion']['auc']:.4f}")
    assert_readme_says(f"coarse-classed scorecard {holdout['coarse_classing']['auc']:.4f}")
    assert_readme_says(f"LightGBM reached {holdout['lightgbm']['auc']:.4f}")
    assert_readme_says(f"the gap to the equal-size-bin scorecard is {CHALLENGERS['interpretability_price_holdout_auc']:.3f}")


def test_selection_gates():
    assert_readme_says(f"{METRICS['n_candidate_features']} candidate characteristics")
    assert_readme_says(
        f"Three gates cut {METRICS['n_candidate_features']} characteristics to {METRICS['n_features_kept']}"
    )
    assert_readme_says(f"This leaves {METRICS['n_after_iv_screen']}.")
    assert_readme_says(f"This leaves {METRICS['n_after_correlation_gate']}.")
    assert_readme_says(f"This leaves {METRICS['n_features_kept']}, and every remaining coefficient is negative")
    assert_readme_says(f"{len(METRICS['sign_gate_removed'])} characteristics, including age")
    assert METRICS["all_coefficients_negative"] is True


def test_prohibited_basis_ablation():
    ablation = METRICS["prohibited_basis_ablation"]
    assert_readme_says(
        f"raise AUC only from {ablation['val_auc_sklearn_final_features']:.4f} to "
        f"{ablation['val_auc_sklearn_with_prohibited_bases']:.4f}"
    )


def test_from_scratch_solver_agreement():
    assert_readme_says(f"correlate with scikit-learn's at {METRICS['pred_corr_vs_sklearn']:.6f}")
    assert_readme_says(f"largest coefficient difference is {METRICS['coef_diff_max_vs_sklearn']:.4f}")


def test_intercept_correction_and_odds():
    assert_readme_says(
        f"from {METRICS['intercept_balanced']:.4f} to {METRICS['intercept_prior_corrected']:.4f}"
    )
    assert_readme_says(f"about {METRICS['train_good_to_bad_odds']:.1f} times as much as a repayer")


def test_example_decline_reasons():
    example = METRICS["example_declined_applicant"]
    assert_readme_says(
        f"who scored {example['score']:.0f}, below the {example['cut_off_score_at_80pct_approval']:.1f} cut-off"
    )
    reasons = example["reasons_points_below_best_bin"]
    assert_readme_says(f"`{reasons[0]['feature']}` ({reasons[0]['points']:.1f} points below the best bin)")
    assert_readme_says(f"`{reasons[1]['feature']}` ({reasons[1]['points']:.1f})")
    assert_readme_says(f"`{reasons[2]['feature']}` ({reasons[2]['points']:.1f})")


def test_monotonic_characteristics():
    validation = CHALLENGERS["validation"]["champion"]
    assert_readme_says(
        f"only {validation['n_numeric_monotonic']} of {validation['n_numeric_kept']} numeric characteristics"
    )
    assert METRICS["n_numeric_features_monotonic_woe"] == METRICS["n_numeric_features_kept"]
    assert_readme_says(f"all {METRICS['n_numeric_features_kept']} numeric characteristics in the final model")


def test_readme_test_count_matches_tests_folder():
    stated = re.findall(r"(\d+) tests", README)
    assert stated, "README states no test count"
    actual = count_test_functions()
    for number in stated:
        assert int(number) == actual, f"README says {number} tests, tests/ holds {actual}"
