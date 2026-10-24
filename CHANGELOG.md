# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). A change to any
figure in `outputs/` is a new minor version at least, and the README states the
run that produced it. A version's date is the day its code was finished; its
tag is created when the pull request that carries it is merged.

## [Unreleased]

## [1.1.0] - 2026-10-24

The notebook becomes a guided build a newcomer can follow cell by cell. No
model figure changes: `outputs/results.json`, `challengers.json` and
`scorecard_points.json` are identical to 1.0.0, and `business_impact.json`
differs only in its `lgd_basis` text.

### Changed

- Notebook comments describe their own lines: 76 generic, repeated comments
  are replaced, three that described the wrong line are corrected, and
  subplot and tick lines get their own comments.
- Notebook text: a Limitations entry on selection bias (every row is a loan
  already granted; new applicants need reject inference or through-the-door
  data), the external scores as third-party values whose timing is assumed
  rather than checked for leakage, IFRS 9 defined on first use with the
  90-days-past-due backstop worded as Basel's and IFRS 9's, and age's sign
  flip no longer traced to named characteristics.
- `src/business_impact.py` writes the corrected `lgd_basis` wording, and
  `outputs/business_impact.json` is regenerated from the saved holdout
  scores (the other 44 leaves and both figures are unchanged).
- The validation report gives the paired lift interval for the benchmark.

- `Credit_Risk_Scorecard.ipynb` (written by `build_notebook.py`) is rebuilt in
  three parts, Prepare Data, Build Model and Communicate Results, of numbered
  steps. Each step has one instruction, a one-line explanation of the idea
  behind it, one small code cell with a comment on each line, printed output,
  and a "Check your work" cell that stops the run if the step went wrong.
- The notebook reads the raw CSVs with `pd.read_csv`, works each calculation
  by hand, asserts that the tested function in `src/` gives the same answer,
  and checks every figure against `outputs/results.json`.
- Every top-level statement in a code cell has a plain-words comment above
  it (or sits in a group of at most three under one).
- The lift over the application-only benchmark is tested with a paired
  bootstrap of the AUC difference (`bootstrap_auc_difference` in
  `src/metrics_scratch.py`): 0.0073 AUC, 95% interval 0.0047 to 0.0101.
- The example applicant is scored with the public `bin_labels` in
  `src/woe_iv.py`; sections are lettered (1A, 2B) so they no longer collide
  with step numbers.
- Domain wording: the flat 45% LGD, the PDO definition, the default flag
  behind PD and the IFRS 9 limitation, as in 1.0.0's README fixes.
- README paragraph on the notebook updated to describe the guided build, and
  Results gives the paired interval for the lift over the benchmark.
- `outputs/notebook_run.json` records the new run, plus the benchmark AUC, the
  lift and its interval. `tests/test_readme_numbers.py` checks the README's
  lift interval against it, and the bar labels and KS cut-off against the
  notebook's printed output.

## [1.0.0] - 2026-10-24

The first tagged release: the scorecard as published, with its figures tested
against the files the pipeline wrote.

### Added

- Points scorecard on 307,511 Home Credit applications: Weight of Evidence
  binning with coarse classing, three selection gates (information value,
  correlation, coefficient sign), a logistic regression written from scratch
  and checked against scikit-learn, and a 600 points = 20:1 odds scale.
- Holdout results in `outputs/results.json`: AUC 0.754, KS 0.381 on 61,503
  held-back applicants; 32 characteristics kept from 62 candidates.
- Approval-policy table in `outputs/business_impact.json`: approving the top
  80% avoids 47% of the credit loss and declines 17.2% of good borrowers.
- LightGBM and coarse-classing comparison in `outputs/challengers.json`.
- Model validation report (`docs/VALIDATION_REPORT.md`) and methodology
  (`docs/METHODOLOGY.md`).
- Live case-study page in `demo/`.
- `tests/test_readme_numbers.py`: every number in the README's prose, tables
  and image descriptions is pinned in reading order. Each is formatted from
  `outputs/*.json` at the printed precision, found in the executed notebook's
  output, found in the build log in `notes/`, or listed as a fixed design or
  wording number with its reason. The README's test count must match `tests/`.
- Pull request template.

### Fixed

- README run provenance and table count (2026-10-21 review).
- README baseline: the executed application-only benchmark, AUC 0.7467 on the
  holdout, replaces the August build-log figure, and Results states what the
  0.007 lift over it means.
- The 45% loss given default is described as an illustrative flat figure
  borrowed from the Basel foundation-IRB value for senior unsecured claims on
  financial institutions (Basel sets no retail value), in the README, the
  validation report, the demo note and `src/business_impact.py`.
- Limits: `TARGET` is Home Credit's payment-difficulty flag, not a regulatory
  default, and the IFRS 9 limit names lifetime PD, staging and scenarios.
- Limits and validation finding F5: every row is a loan Home Credit already
  granted, so the loss cuts are measured within an approved book; scoring new
  applicants needs reject inference or through-the-door data.
- The money table says the amounts are an upper bound (many flagged loans
  cure) and that "approve everyone" means every loan in this book.
- The external scores are described as third-party values in the application
  table, not answers on the form; their timing is an assumption (F7).
- The default-flag limit names the 90-days-past-due backstop as Basel's
  definition (with unlikeliness to pay) and IFRS 9's rebuttable presumption.
- Validation report and methodology cite the 2026-10-19 run at `e683632`; the
  report adds the application-only benchmark and the default definition (F6).
