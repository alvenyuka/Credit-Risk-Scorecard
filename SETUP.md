# SETUP: Run this project on your machine

## 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate            # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

Verify:
```bash
python3 -c "import numpy, pandas, sklearn, lightgbm, scipy; print('OK')"
```

## 2. Place the data files

Create a `data/` folder in the project root with all 8 Home Credit CSVs, see [SCHEMA.md](SCHEMA.md) for exact filenames and what each one is. Get them from [the Kaggle competition page](https://www.kaggle.com/competitions/home-credit-default-risk) (requires a free Kaggle account).

```
Credit_Risk/
├── data/
│   ├── application_train.csv
│   ├── application_test.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── previous_application.csv
│   ├── POS_CASH_balance.csv
│   ├── credit_card_balance.csv
│   └── HomeCredit_columns_description.csv
├── src/
├── output/
├── figs/
└── ...
```

Total size on disk: ~2.5 GB.

> `HomeCredit_columns_description.csv` is Latin-1 encoded, not UTF-8: a documented Kaggle export quirk. `src/io_utils.load_columns_description()` handles it.

## 3. Run the pipeline

```bash
python3 run_pipeline.py
```

Takes **~7 minutes** on a laptop, most of it is loading and aggregating the large relational tables (installments_payments alone is 13.6M rows / 723 MB). Phase-by-phase timing from an actual run:

```
PHASE A: Application features            ~5s     (307,511 rows, 122 -> 142 columns)
PHASE B: Relational aggregation          ~5 min  (bureau ~80s, previous_application ~75s,
                                                   POS_CASH ~80s, credit_card ~64s, installments ~45s)
PHASE C: Train/val/test split            instant (215,257 / 46,127 / 46,127)
PHASE D: WoE/IV feature selection        ~19s    (400 features ranked, top 80 selected)
PHASE E: From-scratch LR + validation    ~30s    (707 gradient-descent iterations)
PHASE F: LightGBM benchmark              ~45s    (533 boosting rounds)
PHASE G: Metrics + sklearn validation    ~1s
PHASE H: Persist to output/              instant
```

Real headline results from this run, see [README.md](README.md) for full detail:

```
From-scratch LR : AUC 0.7509, GINI 0.5019, KS 0.3771
LightGBM        : AUC 0.7774, GINI 0.5548, KS 0.4251
```

**What gets created in `output/`:**

| File | What it is |
|---|---|
| `results.json` | Every metric, hand-coded and sklearn-validated |
| `predictions.npz` | Test-set predictions for all 3 models (from-scratch LR, sklearn LR, LightGBM) |
| `feature_iv_ranking.csv` | All 400 candidate features ranked by Information Value |
| `top_features.json` | The 80 features actually selected |
| `score_band_table.csv` | Population-calibrated score bands |

And in `figs/`: `iv_top20.png`, `score_distribution.png`, `calibration_curve.png`, `lift_chart.png`, regenerated from the actual `output/predictions.npz` via `make_figs.py`, not hand-drawn.

**If it can't find the data:** `src/io_utils.py`'s `DATA_DIR` resolves to `<project root>/data/`. Move your CSVs there.

**If memory is tight:** the biggest single load is `installments_payments.csv` (723 MB on disk, several GB in memory before aggregation). Close other applications, or reduce `N_FEATURES` in `run_pipeline.py` to select fewer features before the LR/LightGBM training step (this doesn't reduce memory during Phase B, which is the actual bottleneck).

## 4. Regenerating the figures

After `run_pipeline.py` finishes, run `python3 make_figs.py` to regenerate the four charts in `figs/` from the fresh `output/` artifacts.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'lightgbm'` | `pip install -r requirements.txt` inside the activated venv |
| `FileNotFoundError: ... application_train.csv` | CSVs aren't in `data/`, see step 2 |
| `UnicodeDecodeError` on `HomeCredit_columns_description.csv` | Use `load_columns_description()` from `src/io_utils.py`, not raw `pd.read_csv` |
| Pipeline is slow | Expected: Phase B (relational aggregation) is genuinely the bottleneck on tables this size, not a bug |
