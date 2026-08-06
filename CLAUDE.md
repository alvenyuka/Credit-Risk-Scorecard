# Credit_Risk_Rebuild — instructions

## What this is

A **blind, from-scratch rebuild** of `Portfolio/Credit_Risk/` — a mastery exercise,
not a copy job. It's the first project in a 12-week interview-readiness plan
(see `Data_Analytics_Mastery_Plan.xlsx` / the published plan artifact), Weeks 6-9,
compressed to start now instead of in September.

Two prior videos set the *process*:
- Ali Abdaal, "Success Is Hard Until You Build Systems Like This" — fixed time
  blocks and milestone commits, not motivation.
- Kedeisha Bryan, "Don't Become a Data Analyst (Do THIS Instead)" — the point of
  this project is a differentiated fintech-risk story, not a generic tutorial clone.

A third video sets the *shape of the output*:
- Lore So What, "How to Create Data Analytics Projects that Get You Hired" — a
  hireable project isn't "I trained a model," it's a business question, documented
  cleaning decisions, analysis tied to a real decision, and a written recommendation
  a non-technical stakeholder could act on. Every milestone below should be
  answerable in those terms, not just "AUC went up."

## Hard rules

1. **Don't open `Portfolio/Credit_Risk/src/*.py` or `README.md` while building.**
   It's the oracle/answer key, not a starting point. Only open it after being
   stuck 30+ minutes on something specific — and note in the week's log file
   *what* forced the look, so the gap is visible later.
2. **Commit at every real milestone**, not at the end of a session. The git log
   in this repo is itself a study artifact — it should show the actual order
   things were figured out in.
3. **Validate against the oracle only after finishing a step**, never mid-build.
   Diffing early turns this into copying with extra steps.
4. **Every milestone's log entry answers**: what business question does this
   serve, what did the data/result actually say, and what would you tell a
   credit committee or a hiring manager because of it.

## Weekly shape (compressed, started early)

- **Week 6 — blind build**: raw data in, own baseline features, a naive but
  honest baseline model. Business framing: "using only what's on the loan
  application itself, how well can we separate good from bad borrowers before
  pulling in bureau history?"
- **Week 7 — primitives from scratch**: WoE/IV, logistic regression via batch
  gradient descent, AUC/GINI/KS/PSI by hand, validated against sklearn to
  floating-point precision, then against the oracle. Business framing: these
  are the exact statistics a credit risk committee expects to see and can
  challenge — building them by hand is what makes them defensible in an
  interview, not just runnable.
- **Week 8 — chase the nuance, build the scorecard**: reproduce the documented
  sklearn-vs-scratch coefficient divergence on the full feature set, diagnose
  it independently, then build the PDO scorecard (300-850). Business framing:
  a scorecard a loan officer can actually use, plus an honest account of where
  the model is unstable and why (multicollinearity among `EXT_SOURCE_*`
  derivatives) — the kind of caveat a hiring manager wants to hear you volunteer.
- **Week 9 — verify, write up, decide on publishing**: full pipeline re-run,
  diff every headline number against `Credit_Risk/output/results.json`, README
  written as a hireable-project narrative (business question -> cleaning ->
  analysis -> recommendation), then decide with the user whether/how this
  gets published.

## Where things live

- `src/` — the rebuild code, own design.
- `notes/weekN_log.md` — one log per week: what was built, what was checked
  against the oracle and when, what the business framing was.
- Raw data goes in `data/` (gitignored, not shipped) — see README.md for the
  Kaggle source.
