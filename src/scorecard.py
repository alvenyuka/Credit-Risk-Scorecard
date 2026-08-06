"""
Week 8 — PDO (points-to-double-odds) scorecard, from scratch.

The scorecard formula turns a logistic regression's log-odds into a points
scale a loan officer can hand-apply without touching a model:

    score = Offset + Factor * ln(odds_of_being_good)

where Factor = PDO / ln(2) (points needed to double the odds of being good)
and Offset is chosen so a reference score (base_score) corresponds to a
chosen reference odds (base_odds).

Because the model's log-odds is a *sum* of (coef_i * WoE_i) terms across
WoE-encoded features, the score decomposes additively into a base score plus
one point-allocation per feature bin -- that additive structure is the whole
point: "applicant lost 18 points on EXT_SOURCE_2 because they fall in the
lowest decile" is a sentence a credit committee can act on; "SHAP value
-0.31" is not.

Assumptions made here (documented, not hidden): base_score=600, base_odds=20
(20 good : 1 bad at 600), PDO=40. These are business choices, not statistical
ones -- an institution sets them to match whatever score range/sensitivity
their existing policy already uses. Final scores are clipped to [300, 850],
the conventional retail-credit range, since a handful of extreme outliers
would otherwise blow past it.
"""
import numpy as np
import pandas as pd

from woe_iv import _bin_labels


def build_scorecard(coef: np.ndarray, intercept: float, kept_cols: list,
                     woe_fits: dict, base_score: int = 600, base_odds: float = 20,
                     pdo: float = 40) -> dict:
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    # score = offset - factor * (intercept + sum(coef_i * woe_i))
    # (minus sign: our model predicts P(bad); the scorecard convention scores odds of GOOD)
    base_points = offset - factor * intercept

    points_tables = {}
    for i, col in enumerate(kept_cols):
        woe = woe_fits[col]["table"]["woe"]
        points_tables[col] = -factor * coef[i] * woe

    return {
        "factor": factor, "offset": offset, "base_points": base_points,
        "points_tables": points_tables, "kept_cols": kept_cols, "woe_fits": woe_fits,
    }


def score_dataframe(df: pd.DataFrame, sc: dict) -> pd.Series:
    total = pd.Series(sc["base_points"], index=df.index, dtype=float)
    for col in sc["kept_cols"]:
        fit = sc["woe_fits"][col]
        bins = _bin_labels(df[col], fit["is_categorical"], fit["edges"])
        points = bins.map(sc["points_tables"][col]).fillna(0.0)
        total += points
    return total.clip(300, 850)


def reason_codes(row: pd.Series, sc: dict, top_n: int = 3) -> list:
    """The n biggest point deductions for one applicant -- an adverse-action
    reason list, the thing a declined applicant is legally entitled to in
    many jurisdictions (and the thing a black-box GBM makes awkward to produce)."""
    contributions = []
    for col in sc["kept_cols"]:
        fit = sc["woe_fits"][col]
        bins = _bin_labels(pd.Series([row[col]]), fit["is_categorical"], fit["edges"])
        pts = sc["points_tables"][col].get(bins.iloc[0], 0.0)
        contributions.append((col, pts))
    contributions.sort(key=lambda t: t[1])  # most negative (biggest point loss) first
    return contributions[:top_n]


def destandardize_coefficients(coef_std: np.ndarray, intercept_std: float,
                                mean: np.ndarray, std: np.ndarray) -> tuple:
    """The model was fit on standardized WoE features ((x-mean)/std) for
    gradient-descent stability. The scorecard needs coefficients that apply
    directly to raw WoE values instead -- algebraically equivalent, just
    re-expressed:

        z = intercept + sum(coef_i * (x_i - mean_i)/std_i)
          = [intercept - sum(coef_i * mean_i/std_i)] + sum((coef_i/std_i) * x_i)

    so coef_raw_i = coef_i / std_i, intercept_raw = intercept - sum(coef_i * mean_i / std_i).
    """
    coef_raw = coef_std / std
    intercept_raw = intercept_std - np.sum(coef_std * mean / std)
    return coef_raw, intercept_raw
