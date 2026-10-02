"""
The single definition of every feature list and screening threshold.

Every script, the notebook and the tests import from here, so the baseline, the
scorecard pipeline and the walkthrough cannot drift onto different feature sets.
"""

# Raw application-table columns used as numeric candidates.
BASE_NUMERIC_COLS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE", "CNT_FAM_MEMBERS", "CNT_CHILDREN",
    "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
    "HOUR_APPR_PROCESS_START",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]

# Raw application-table columns WoE-encoded by level.
CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
]

# Sex and marital status are prohibited bases for a credit decision under the US Equal
# Credit Opportunity Act and Regulation B (12 CFR 1002.6(b)), so they never enter the
# candidate set. Age is a permitted input in an empirically derived scoring system
# provided applicants aged 62 or over are not assigned a negative factor; that condition
# is enforced by scorecard.assert_no_negative_points_for_age_62_plus.
PROHIBITED_BASES = ["CODE_GENDER", "NAME_FAMILY_STATUS"]

# Features derived in baseline_features.engineer_baseline and offered to the scorecard.
ENGINEERED_NUMERIC = [
    "AGE_YEARS", "EMPLOYED_YEARS",
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
    "CREDIT_GOODS_RATIO", "INCOME_PER_FAM_MEMBER",
    "EXT_SOURCE_MEAN", "EXT_SOURCE_STD", "EXT_SOURCE_COUNT",
]

# Engineered for exploration only, never offered to the scorecard. DAYS_EMPLOYED_ANOM
# is 1 exactly when EMPLOYED_YEARS is missing, so it duplicates the Missing bin of
# EMPLOYED_YEARS. Fitting both lets the solver give the two encodings opposite signs,
# which produced a reason code that penalised a lower-risk (largely pensioner) group.
EDA_ONLY_COLS = ["DAYS_EMPLOYED_ANOM"]

# Applicant-level aggregates built by relational_features.build_all_relational_features.
RELATIONAL_NUMERIC = [
    "BUREAU_COUNT", "BUREAU_ACTIVE_SHARE", "BUREAU_DAYS_CREDIT_MIN",
    "BUREAU_DAYS_CREDIT_MEAN", "BUREAU_OVERDUE_MAX", "BUREAU_OVERDUE_MEAN",
    "BUREAU_CREDIT_SUM_TOTAL", "BUREAU_CREDIT_SUM_DEBT_TOTAL",
    "BUREAU_EVER_DELINQUENT_SHARE",
    "PREV_APP_COUNT", "PREV_APPROVED_SHARE", "PREV_REFUSED_SHARE",
    "PREV_AMT_APPLICATION_MEAN", "PREV_AMT_CREDIT_MEAN", "PREV_DAYS_DECISION_MAX",
    "POS_DPD_MAX", "POS_DPD_MEAN", "POS_DPD_DEF_MAX", "POS_CNT_INSTALMENT_MEAN",
    "CC_BALANCE_MEAN", "CC_BALANCE_MAX", "CC_UTILIZATION_MEAN", "CC_DPD_MAX",
    "INSTAL_COUNT", "INSTAL_DAYS_LATE_MEAN", "INSTAL_DAYS_LATE_MAX",
    "INSTAL_SHORTFALL_MEAN", "INSTAL_SHORTFALL_SUM",
]

# Application-only numeric set (the WoE baseline in woe_baseline.py).
APPLICATION_NUMERIC = BASE_NUMERIC_COLS + ENGINEERED_NUMERIC

# The naive impute-scale-one-hot baseline in baseline_model.py also reads the anomaly
# flag as a numeric input. It produces no points or reason codes, so the duplication
# described above does not affect any decision.
BASELINE_NUMERIC = APPLICATION_NUMERIC + EDA_ONLY_COLS

# Candidate features offered to the scorecard.
MODEL_NUMERIC = BASE_NUMERIC_COLS + ENGINEERED_NUMERIC + RELATIONAL_NUMERIC
MODEL_CATEGORICAL = list(CATEGORICAL_COLS)

# Screening thresholds.
IV_THRESHOLD = 0.01            # drop features with information value below this
CORRELATION_THRESHOLD = 0.9    # of any WoE pair with |r| above this, keep the higher-IV one
