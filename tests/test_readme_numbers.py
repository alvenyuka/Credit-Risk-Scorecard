"""Every number in the README, in order, checked against where it comes from.

The README is typed by hand. This file pins every number in its prose, tables
and image descriptions, one occurrence at a time, in the order they appear.
Each expected number is one of four kinds:

- a pipeline result, formatted from outputs/results.json,
  outputs/business_impact.json or outputs/challengers.json at the precision the
  README prints it (0.754, not 0.7539670402);
- a figure the executed notebook prints, checked against the notebook's saved
  output;
- a figure from the week-by-week build log, checked against that file in notes/;
- a fixed number that is part of the design or the wording (the 600-point
  scale, the 0.01 IV threshold, a step number), listed with the reason it is
  fixed.

Code blocks, inline code and link targets are skipped; the test count in the
How to run block has its own test at the end. A re-run that moves a result, or
an edit that mistypes any number, changes the sequence and fails here. Adding
or rewording a number in the README means adding it here too, with its source.

No data is needed: the outputs and the executed notebook are committed, so
these run in CI.
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
NOTEBOOK_RUN = load_json("notebook_run.json")


def notebook_printed_text():
    # Join every text output the executed notebook saved, so a figure the
    # README takes from the notebook can be found in what the notebook printed
    notebook = json.loads((ROOT / "Credit_Risk_Scorecard.ipynb").read_text(encoding="utf-8"))
    pieces = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            text = output.get("text") or output.get("data", {}).get("text/plain", "")
            pieces.append("".join(text))
    return "\n".join(pieces)


NOTEBOOK_TEXT = notebook_printed_text()


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


# ---------------------------------------------------------------------------
# Reading the numbers out of the README
# ---------------------------------------------------------------------------

DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
NUMBER = re.compile(r"(?<![\w.,/-])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)|(?<=/)\d+(?!\w)")


def readme_prose():
    # Drop what is not prose: fenced code, inline code, link targets, HTML tags
    # and the "1." markers of the numbered build steps
    text = re.sub(r"```.*?```", "", README, flags=re.S)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(?m)^\s*\d+\.\s", "", text)
    return text


def readme_numbers():
    # Every number in reading order, with its line for the error message.
    # A date counts as one token.
    found = []
    for line_no, line in enumerate(readme_prose().splitlines(), start=1):
        position = 0
        while position < len(line):
            date = DATE.match(line, position)
            if date and (position == 0 or not line[position - 1].isalnum()):
                found.append((date.group(0), line_no, line))
                position = date.end()
                continue
            number = NUMBER.match(line, position)
            if number:
                found.append((number.group(0), line_no, line))
                position = number.end()
                continue
            position += 1
    return found


# ---------------------------------------------------------------------------
# The expected sequence, one entry per occurrence
# ---------------------------------------------------------------------------

NOTEBOOK_FIGURES = []   # (README text, text the notebook prints)
NOTE_FIGURES = []       # (README text, notes file, text in that file)


def result(text):
    # A pipeline result, already formatted from outputs/*.json
    return text


def from_notebook(text, printed=None):
    # A figure the executed notebook prints. Thousands separators may differ.
    NOTEBOOK_FIGURES.append((text, printed if printed is not None else text))
    return text


def from_notes(text, file_name, needle):
    # A figure from the build log in notes/
    NOTE_FIGURES.append((text, file_name, needle))
    return text


def fixed(text, why):
    # A number that is part of the design or the wording, not a result
    assert why
    return text


def pct(x, places=1):
    return f"{x * 100:.{places}f}"


def bn(x):
    return f"{x / 1e9:.2f}"


def expected_numbers():
    m = METRICS
    auc = f"{m['val_auc']:.3f}"
    auc4 = f"{m['val_auc']:.4f}"
    ks = f"{m['val_ks']:.3f}"
    low, high = m["val_auc_ci95_bootstrap"]
    n_holdout = f"{m['n_validation_rows']:,}"
    bad_rate = pct(m["train_bad_rate"])
    good_rate = pct(1 - m["train_bad_rate"])
    lowest = pct(m["default_rate_lowest_decile"])
    highest = pct(m["default_rate_highest_decile"])
    p100, p90, p80, p70 = policy_at(1.0), policy_at(0.9), policy_at(0.8), policy_at(0.7)
    lightgbm = CHALLENGERS["holdout"]["lightgbm"]["auc"]
    observed = m["observed_odds_near_base_score"]
    example = m["example_declined_applicant"]
    reasons = example["reasons_points_below_best_bin"]
    ablation = m["prohibited_basis_ablation"]
    holdout = CHALLENGERS["holdout"]
    rows = from_notebook("307,511", "307511")
    baseline = from_notebook("0.7467", "application-only benchmark AUC: 0.7467")
    assert f"{NOTEBOOK_RUN['benchmark_auc']:.4f}" == baseline
    lift_lo, lift_hi = NOTEBOOK_RUN["auc_lift_ci95_paired_bootstrap"]
    placeholder = from_notebook("365,243", "365243")

    seq = []
    # Opening and data line
    seq += [rows]
    seq += [result(lowest), result(highest)]                     # hero image description
    seq += [rows, result(bad_rate), fixed("8", "tables in the Kaggle download"),
            fixed("7", "tables the model uses")]
    # Results
    seq += [fixed("20", "holdout share of the split"), result(n_holdout)]
    seq += [result(auc), result(f"{m['val_auc'] * 100:.0f}"),   # "about 75% of the time"
            fixed("95", "interval level"), result(f"{low:.3f}"), result(f"{high:.3f}"),
            fixed("95", "interval level")]
    seq += [result(lowest), result(highest)]
    seq += [result(f"{p80['approval_rate']:.0%}".rstrip("%")), result(f"{p80['loss_avoided_pct']:.0%}".rstrip("%")),
            result(pct(p80["good_borrowers_declined_pct"]))]
    seq += [result(f"{m['calibration_max_abs_gap'] * 100:.1f}")]
    seq += [result(f"{lightgbm:.3f}"), result(f"{lightgbm - m['val_auc']:.3f}"), result(auc),
            fixed("11", "step number")]
    seq += [baseline, result(f"{NOTEBOOK_RUN['auc_lift_over_benchmark']:.3f}"), result(auc4),
            fixed("95", "interval level"), result(f"{lift_lo:.3f}"), result(f"{lift_hi:.3f}")]
    seq += [result(str(m["n_features_kept"]))]                   # demo line
    # Measures table
    seq += [result(n_holdout)]
    seq += [result(auc), fixed("0.5", "AUC of a coin flip"), fixed("1.0", "AUC of a perfect ranking")]
    seq += [result(ks)]
    seq += [result(f"{m['val_gini']:.3f}"), fixed("0", "Gini range"), fixed("1", "Gini range"),
            fixed("2", "Gini formula"), fixed("1", "Gini formula")]
    seq += [result(f"{m['mean_score_repaid']:.0f}"), result(f"{m['mean_score_defaulted']:.0f}"),
            result(f"{m['score_min']:.0f}"), result(f"{m['score_max']:.0f}")]
    seq += [fixed("600", "base score of the scale"), result(f"{observed['good_to_bad']:.1f}"),
            result(f"{m['implied_odds_at_base_score']:.1f}"), result(str(observed["score_band"][0])),
            result(str(observed["score_band"][1])), result(f"{observed['good_to_bad']:.0f}"),
            fixed("1", "odds wording")]
    seq += [result(f"{m['psi_train_to_holdout_scores']:.4f}"), fixed("0.1", "usual PSI stability threshold"),
            fixed("0", "PSI wording")]
    # Money table paragraph and table
    seq += [fixed("45", "assumed loss given default"), fixed("45", "assumed loss given default")]
    seq += [result(f"{p100['approved']:,}"), result(pct(p100["default_rate_approved"])),
            result(bn(p100["credit_loss"])), result(f"{p100['good_borrowers_declined_pct']:.0%}".rstrip("%"))]
    for row in (p90, p80, p70):
        seq += [result(f"{row['approval_rate']:.0%}".rstrip("%")), result(f"{row['cut_off_score']:.1f}"),
                result(f"{row['approved']:,}"), result(pct(row["default_rate_approved"])),
                result(bn(row["credit_loss"])), result(f"{row['loss_avoided_pct']:.0%}".rstrip("%")),
                result(pct(row["good_borrowers_declined_pct"]))]
    seq += [result(bn(p100["credit_loss"])), result(bn(p70["credit_loss"])),
            result(f"{p70['approval_rate']:.0%}".rstrip("%"))]  # policy chart description
    seq += [result(f"{p80['approval_rate']:.0%}".rstrip("%"))]
    # Step 1
    seq += [result(bad_rate), result(good_rate), placeholder,
            result(f"{int(placeholder.replace(',', '')) / 365.25:,.0f}")]
    seq += [from_notebook("282,686", "282686"), from_notebook("24,825", "24825"),
            result(good_rate), result(bad_rate)]
    # Step 2
    seq += [baseline, fixed("2026", "year of the build log"),
            from_notes("0.7487", "week6_log.md", "Validation AUC: 0.7487")]
    # Steps 5 and 6
    seq += [result(str(m["n_candidate_features"]))]
    seq += [result(str(m["n_candidate_features"])), result(str(m["n_features_kept"]))]
    seq += [fixed("0.01", "IV gate threshold"), result(str(m["n_after_iv_screen"]))]
    seq += [fixed("0.9", "correlation gate threshold"), result(str(m["n_after_correlation_gate"]))]
    seq += [result(str(m["n_features_kept"]))]
    seq += [result(f"{ablation['val_auc_sklearn_final_features']:.4f}"),
            result(f"{ablation['val_auc_sklearn_with_prohibited_bases']:.4f}")]
    # Step 7
    seq += [result(f"{m['pred_corr_vs_sklearn']:.6f}"), result(f"{m['coef_diff_max_vs_sklearn']:.4f}")]
    # Step 8: the scale is a design choice, the odds pair follows from it
    seq += [fixed("600", "base score"), fixed("20", "odds at the base score"), fixed("1", "odds wording"),
            fixed("40", "points to double the odds"), fixed("20", "odds at 600"), fixed("1", "odds wording"),
            fixed("600", "base score"), fixed("40", "doubled odds"), fixed("1", "odds wording"),
            fixed("640", "600 plus one PDO")]
    # Step 9
    seq += [fixed("20", "holdout share of the split")]
    seq += [result(auc), result(ks), from_notebook("571", "at 571 points")]
    # Step 10
    seq += [result(f"{example['score']:.0f}"), result(f"{example['cut_off_score_at_80pct_approval']:.1f}"),
            result(f"{p80['approval_rate']:.0%}".rstrip("%")), result(f"{reasons[0]['points']:.1f}"),
            result(f"{reasons[1]['points']:.1f}"), result(f"{reasons[2]['points']:.1f}")]
    # Step 11
    seq += [result(str(m["n_candidate_features"])), fixed("0.005", "coarse-classing decision rule"),
            result(f"{CHALLENGERS['coarse_classing_auc_loss_validation']:.4f}"),
            result(f"{holdout['champion']['auc']:.4f}"), result(f"{holdout['coarse_classing']['auc']:.4f}"),
            result(f"{holdout['lightgbm']['auc']:.4f}"), result(f"{lightgbm - m['val_auc']:.3f}"),
            result(f"{CHALLENGERS['interpretability_price_holdout_auc']:.3f}")]
    # Problems I hit
    seq += [from_notes("0.0016", "week7_log.md", "~1.6e-3")]
    seq += [result(f"{m['train_good_to_bad_odds']:.1f}"), from_notes("0.905", "week8_log.md", "0.905"),
            from_notes("0.999997", "week8_log.md", "0.999997"), result(f"{m['pred_corr_vs_sklearn']:.6f}")]
    seq += [placeholder, result(str(len(m["sign_gate_removed"])))]
    validation = CHALLENGERS["validation"]["champion"]
    seq += [result(str(validation["n_numeric_monotonic"])), result(str(validation["n_numeric_kept"])),
            result(str(m["n_numeric_features_monotonic_woe"]))]
    seq += [fixed("600", "base score"), fixed("20", "designed odds"), fixed("1", "odds wording"),
            fixed("9", "week number"), fixed("20", "designed odds"), fixed("1", "odds wording"),
            fixed("600", "base score"), from_notes("11.4", "week9_log.md", "11.4:1"), fixed("1", "odds wording"),
            fixed("50", "balanced weighting"), fixed("50", "balanced weighting"), result(bad_rate),
            result(f"{m['intercept_balanced']:.4f}"), result(f"{m['intercept_prior_corrected']:.4f}"),
            fixed("600", "base score"), result(f"{observed['good_to_bad']:.1f}"), fixed("1", "odds wording"),
            result(f"{m['implied_odds_at_base_score']:.1f}")]
    # Data
    seq += [rows, fixed("8", "tables in the Kaggle download"), fixed("7", "tables the model uses")]
    seq += [fixed("20", "holdout share of the split")]
    # Limits
    seq += [fixed("45", "assumed loss given default")]
    seq += [fixed("90", "regulatory days-past-due backstop"), fixed("9", "IFRS 9"), fixed("9", "IFRS 9")]
    seq += [fixed("9", "IFRS 9")]
    # How to run: provenance of the committed outputs
    seq += [fixed("2026-10-19", "date of the run that wrote outputs/results.json; checked below"),
            fixed("2026-10-06", "date of the earlier run with the same results")]
    return seq


def test_every_readme_number_in_order():
    expected = expected_numbers()
    found = readme_numbers()
    for index in range(min(len(expected), len(found))):
        text, line_no, line = found[index]
        assert text == expected[index], (
            f"number {index + 1} in the README is {text!r}, expected {expected[index]!r}; "
            f"line {line_no}: {line.strip()[:160]}"
        )
    if len(found) > len(expected):
        extra_text, extra_line, _ = found[len(expected)]
        raise AssertionError(f"README has a number the test does not expect: {extra_text!r} on line {extra_line}")
    assert len(found) == len(expected), f"README ends before the expected number {expected[len(found)]!r}"


def test_notebook_sourced_numbers_are_printed_by_the_notebook():
    expected_numbers()
    assert NOTEBOOK_FIGURES
    for text, printed in NOTEBOOK_FIGURES:
        assert printed in NOTEBOOK_TEXT, f"README figure {text} is not in the notebook's output ({printed!r})"


def test_notes_sourced_numbers_are_in_the_build_log():
    expected_numbers()
    assert NOTE_FIGURES
    for text, file_name, needle in NOTE_FIGURES:
        log = (ROOT / "notes" / file_name).read_text(encoding="utf-8")
        assert needle in log, f"README figure {text} is not in notes/{file_name} ({needle!r})"


def test_target_counts_add_up():
    # The bar labels in step 1 must sum to the row count and give the stated rates
    repaid, defaulted = 282_686, 24_825
    assert repaid + defaulted == 307_511
    assert f"{defaulted / (repaid + defaulted):.1%}" == f"{METRICS['train_bad_rate']:.1%}" == "8.1%"


def test_run_provenance_matches_results_file():
    # Pinned on purpose: the README names the run that wrote outputs/results.json.
    # After an honest re-run of src/run_pipeline.py this fails until the
    # provenance sentence under How to run (date and commit) and the date below
    # are updated to the new run, so the README cannot describe a run that is
    # no longer the one committed.
    stamped = load_json("results.json")
    assert stamped["generated_at_utc"].startswith("2026-10-19")
    assert_readme_says(f"(commit `{stamped['git_commit']}`)")


def test_headline_sentences():
    # The four headline sentences, pinned whole so a stray copy elsewhere cannot stand in
    m = METRICS
    low, high = m["val_auc_ci95_bootstrap"]
    assert f"{m['val_auc']:.3f}" == "0.754" and f"{m['val_ks']:.3f}" == "0.381"
    assert_readme_says(f"**The score ranks risk well.** AUC {m['val_auc']:.3f}: pick one")
    assert_readme_says(f"(95% bootstrap interval {low:.3f} to {high:.3f}:")
    row = policy_at(0.8)
    assert f"{row['loss_avoided_pct']:.0%}" == "47%"
    assert_readme_says(f"**Approving only the top 80% cuts credit losses by {row['loss_avoided_pct']:.0%}.**")
    assert_readme_says(f"| AUC | {m['val_auc']:.3f} |")
    assert_readme_says(f"| KS | {m['val_ks']:.3f} |")


def test_holdout_size_and_default_rate():
    n_holdout = METRICS["n_validation_rows"]
    assert n_holdout == 61_503
    assert_readme_says(f"({n_holdout:,})")
    assert_readme_says(f"| Measure (holdout of {n_holdout:,}) |")


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


def test_loss_avoided_does_not_depend_on_the_flat_lgd():
    # The README says one flat LGD leaves the loss-avoided shares unchanged.
    # Loss avoided is 1 - loss(policy) / loss(everyone), and the LGD cancels.
    everyone = policy_at(1.0)
    for rate in (0.9, 0.8, 0.7):
        row = policy_at(rate)
        share = 1 - row["credit_loss"] / everyone["credit_loss"]
        assert abs(share - row["loss_avoided_pct"]) < 1e-9


def test_selection_gates():
    assert METRICS["all_coefficients_negative"] is True
    assert_readme_says(f"This leaves {METRICS['n_features_kept']}, and every remaining coefficient is negative")
    assert METRICS["n_numeric_features_monotonic_woe"] == METRICS["n_numeric_features_kept"]


def test_example_decline_reasons():
    example = METRICS["example_declined_applicant"]
    reasons = example["reasons_points_below_best_bin"]
    assert_readme_says(f"`{reasons[0]['feature']}` ({reasons[0]['points']:.1f} points below the best bin)")
    assert_readme_says(f"`{reasons[1]['feature']}` ({reasons[1]['points']:.1f})")
    assert_readme_says(f"`{reasons[2]['feature']}` ({reasons[2]['points']:.1f})")


def test_readme_test_count_matches_tests_folder():
    stated = re.findall(r"(\d+) tests", README)
    assert stated, "README states no test count"
    actual = count_test_functions()
    for number in stated:
        assert int(number) == actual, f"README says {number} tests, tests/ holds {actual}"
