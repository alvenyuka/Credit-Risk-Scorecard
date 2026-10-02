"""
PDO (points to double the odds) scorecard and its acceptance checks.

The scorecard turns a logistic regression's log-odds into points a loan officer
can apply without the model:

    score = Offset + Factor * ln(odds of being good)

where Factor = PDO / ln(2) and Offset places base_score at base_odds. Because
the model's log-odds is a sum of coef_i * WoE_i terms, the score decomposes into
base points plus one point value per feature bin, so "18 points below the best
bin on EXT_SOURCE_2" is a reason a credit committee can act on.

Scaling choices (business conventions, not estimates): base_score 600 at
base_odds 20 good to 1 bad, PDO 40. Scores are clipped to the conventional
300 to 850 range.

The model is fitted with class_weight="balanced", which fits the intercept to a
50/50 prior. prior_correct_intercept restores the training default rate before
the points table is built (King and Zeng, 2001), so that 600 really corresponds
to 20:1 good:bad odds and the score can be read as a probability of default.

Acceptance checks run on every build:
- assert_woe_coefficient_signs: WoE = ln(good/bad), so in a P(bad) model every
  coefficient must be negative; a positive one reverses the direction of a
  characteristic and produces wrong reason codes.
- assert_no_negative_points_for_age_62_plus: Regulation B (12 CFR 1002.6(b)(2))
  allows age in an empirically derived system only if applicants aged 62 or
  over are not assigned a negative factor.
"""
import numpy as np
import pandas as pd

from woe_iv import MISSING_LABEL, _bin_labels


def prior_correct_intercept(intercept: float, train_bad_rate: float,
                            fitted_bad_rate: float = 0.5) -> float:
    """Shift a logistic intercept from the prior it was fitted to onto the true prior.

    A weighted fit whose weights make the classes count equally behaves as if the
    bad rate were fitted_bad_rate (0.5 for class_weight="balanced"). Adding
    logit(train_bad_rate) - logit(fitted_bad_rate) moves predicted probabilities
    back to the observed default rate without changing any coefficient, so the
    ranking (AUC, KS) is unchanged.
    """
    def logit(p):
        return np.log(p / (1 - p))
    return float(intercept + logit(train_bad_rate) - logit(fitted_bad_rate))


def assert_woe_coefficient_signs(coef_raw, kept_cols) -> None:
    """WoE = ln(good/bad), so in a P(bad) model every coefficient must be negative."""
    wrong = [c for c, b in zip(kept_cols, np.asarray(coef_raw, dtype=float)) if b > 0]
    if wrong:
        raise ValueError(f"coefficients with the wrong sign for WoE inputs: {wrong}")


def build_scorecard(coef: np.ndarray, intercept: float, kept_cols: list,
                    woe_fits: dict, base_score: int = 600, base_odds: float = 20,
                    pdo: float = 40) -> dict:
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    # score = offset - factor * (intercept + sum(coef_i * woe_i))
    # (minus sign: the model predicts P(bad); the scorecard scores the odds of good)
    base_points = offset - factor * intercept

    points_tables = {}
    for i, col in enumerate(kept_cols):
        woe = woe_fits[col]["table"]["woe"]
        points_tables[col] = -factor * coef[i] * woe

    return {
        "factor": factor, "offset": offset, "base_points": base_points,
        "base_score": base_score, "base_odds": base_odds, "pdo": pdo,
        "points_tables": points_tables, "kept_cols": kept_cols, "woe_fits": woe_fits,
    }


def score_dataframe(df: pd.DataFrame, sc: dict, clip: bool = True) -> pd.Series:
    """Total points per row; clip=False returns the unclipped score."""
    total = pd.Series(sc["base_points"], index=df.index, dtype=float)
    for col in sc["kept_cols"]:
        fit = sc["woe_fits"][col]
        bins = _bin_labels(df[col], fit["is_categorical"], fit["edges"])
        total += bins.map(sc["points_tables"][col]).astype(float).fillna(0.0)
    return total.clip(300, 850) if clip else total


def score_to_pd(score, sc: dict) -> np.ndarray:
    """Probability of default implied by an (unclipped) score."""
    log_odds_good = (np.asarray(score, dtype=float) - sc["offset"]) / sc["factor"]
    return 1.0 / (1.0 + np.exp(log_odds_good))


def odds_at_score(score: float, sc: dict) -> float:
    """Good:bad odds the scorecard assigns to a score."""
    return float(np.exp((score - sc["offset"]) / sc["factor"]))


def reason_codes(row: pd.Series, sc: dict, top_n: int = 3) -> list:
    """Adverse-action reasons by the points-below-maximum method.

    For each characteristic, points lost = (points of its best bin) - (points of
    the applicant's bin). Characteristics are ranked by points lost, and only
    those where the applicant actually lost points are returned, so a reason is
    never a characteristic on which the applicant scored the maximum. Each entry
    is (feature, points_lost) with points_lost > 0.
    """
    lost = []
    for col in sc["kept_cols"]:
        fit = sc["woe_fits"][col]
        pts = sc["points_tables"][col]
        label = _bin_labels(pd.Series([row[col]]), fit["is_categorical"], fit["edges"]).iloc[0]
        lost.append((col, float(pts.max() - pts.get(label, 0.0))))
    lost.sort(key=lambda t: (-t[1], t[0]))
    return [(c, v) for c, v in lost if v > 0][:top_n]


def _bin_upper_edge(label: str) -> float:
    """Upper edge of a continuous bin label such as '(56.021, 60.709]'."""
    return float(label.split(",")[1].strip(" ])"))


def age_62_plus_points(sc: dict, col: str = "AGE_YEARS", age: float = 62.0) -> pd.Series:
    """Points of every bin of `col` that contains applicants aged `age` or over.

    A bin covers 62+ when its upper edge exceeds 62, which includes a bin that
    straddles 62. Returns an empty Series when `col` is not in the scorecard.
    """
    if col not in sc["kept_cols"]:
        return pd.Series(dtype=float)
    pts = sc["points_tables"][col]
    covering = [b for b in pts.index if b != MISSING_LABEL and _bin_upper_edge(b) > age]
    return pts.loc[covering]


def assert_no_negative_points_for_age_62_plus(sc: dict, col: str = "AGE_YEARS") -> dict:
    """Regulation B: no negative factor for applicants aged 62 or over.

    Returns a summary for results.json and raises ValueError on a violation.
    """
    pts = age_62_plus_points(sc, col)
    summary = {
        "age_feature_in_scorecard": col in sc["kept_cols"],
        "bins_covering_62_plus": {str(k): float(v) for k, v in pts.items()},
        "min_points_62_plus": float(pts.min()) if len(pts) else None,
    }
    if len(pts) and (pts < 0).any():
        raise ValueError(f"Regulation B: negative points for bins covering age 62+: {pts[pts < 0].to_dict()}")
    summary["passed"] = True
    return summary


def destandardize_coefficients(coef_std: np.ndarray, intercept_std: float,
                               mean: np.ndarray, std: np.ndarray) -> tuple:
    """Re-express coefficients fitted on standardised WoE ((x - mean) / std) on raw WoE.

        z = intercept + sum(coef_i * (x_i - mean_i) / std_i)
          = [intercept - sum(coef_i * mean_i / std_i)] + sum((coef_i / std_i) * x_i)

    so coef_raw_i = coef_i / std_i and intercept_raw = intercept - sum(coef_i * mean_i / std_i).
    """
    coef_raw = coef_std / std
    intercept_raw = intercept_std - np.sum(coef_std * mean / std)
    return coef_raw, intercept_raw
