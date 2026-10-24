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
model figure changes: `outputs/results.json`, `business_impact.json`,
`challengers.json` and `scorecard_points.json` are identical to 1.0.0.

### Changed

- `Credit_Risk_Scorecard.ipynb` (written by `build_notebook.py`) is rebuilt in
  three parts, Prepare Data, Build Model and Communicate Results, of numbered
  steps. Each step has one instruction, a one-line explanation of the idea
  behind it, one small code cell with a comment on each line, printed output,
  and a "Check your work" cell that stops the run if the step went wrong.
- The notebook reads the raw CSVs with `pd.read_csv`, works each calculation
  by hand, asserts that the tested function in `src/` gives the same answer,
  and checks every figure against `outputs/results.json`.
- README paragraph on the notebook updated to describe the guided build.
- `outputs/notebook_run.json` records the new run.

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
