# Data schema: Home Credit Default Risk

8 CSVs, joined on `SK_ID_CURR` (application) and `SK_ID_PREV` (prior loan reference).

| Table | Rows | Grain | What it is |
|---|---:|---|---|
| `application_train.csv` | 307,511 | 1 row / applicant | The target (`TARGET`) plus applicant demographics, income, employment, external credit-bureau scores (`EXT_SOURCE_1/2/3`) |
| `application_test.csv` | 48,744 | 1 row / applicant | Same schema, no `TARGET` (Kaggle leaderboard holdout, not used in this pipeline) |
| `bureau.csv` | 1,716,428 | 1 row / prior credit-bureau record | Credit history reported by *other* lenders, keyed to `SK_ID_CURR` |
| `bureau_balance.csv` | 27,299,925 | 1 row / bureau-record / month | Monthly delinquency status (`STATUS`) for each `bureau.csv` record |
| `previous_application.csv` | 1,670,214 | 1 row / prior Home Credit application | The applicant's loan history *with Home Credit itself* |
| `POS_CASH_balance.csv` | 10,001,358 | 1 row / prev. loan / month | Monthly point-of-sale/cash loan balance snapshots |
| `credit_card_balance.csv` | 3,840,312 | 1 row / prev. loan / month | Monthly revolving credit-card balance snapshots |
| `installments_payments.csv` | 13,605,401 | 1 row / scheduled installment | Actual vs. scheduled repayment per installment, the strongest behavioural signal (see [README](README.md#feature-engineering)) |

## Known quirks (handled in `src/`)

- **`DAYS_EMPLOYED` anomaly**: unemployed/pensioner applicants carry a placeholder value of `365243` instead of a real day-count. Flagged (`ANOM_DAYS_EMPLOYED`) and set to `NaN` before use, see `src/feature_engineering_app.py`.
- **`DAYS_*` columns are negative**: days *before* the application date. Converted to positive years (`AGE_YEARS`, `EMPLOYED_YEARS`).
- **`HomeCredit_columns_description.csv` is Latin-1 encoded**, not UTF-8: a documented Kaggle export quirk. `src/io_utils.load_columns_description()` handles it.
- **Not every applicant has every supplementary table.** Left joins on `SK_ID_CURR` leave `NaN` for applicants with no bureau history, no prior Home Credit loans, etc.; this is itself informative (new-to-credit applicants), not missing data to impute away.
