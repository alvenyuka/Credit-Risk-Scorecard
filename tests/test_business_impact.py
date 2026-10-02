"""The lending-policy arithmetic behind the README's business-impact table."""
import numpy as np
import pytest

from business_impact import lending_policy_table


def small_book():
    # scores: higher is safer. Two defaults sit at the bottom of the score range.
    scores = np.array([700, 650, 600, 550, 500, 450, 400, 350, 300, 250])
    defaulted = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    exposure = np.full(10, 1_000.0)
    return scores, defaulted, exposure


def test_approve_all_carries_the_whole_loss():
    s, d, e = small_book()
    row = lending_policy_table(s, d, e, lgd=0.5, approval_rates=(1.0,)).iloc[0]
    assert row["credit_loss"] == pytest.approx(2 * 1_000 * 0.5)
    assert row["loss_avoided_vs_approve_all"] == 0
    assert row["good_borrowers_declined"] == 0


def test_cut_off_approves_the_highest_scores_first():
    s, d, e = small_book()
    row = lending_policy_table(s, d, e, lgd=0.5, approval_rates=(0.8,)).iloc[0]
    # the top 8 are all good, so every default is declined
    assert row["approved"] == 8
    assert row["credit_loss"] == 0
    assert row["loss_avoided_pct"] == pytest.approx(1.0)
    assert row["default_rate_approved"] == 0
    assert row["cut_off_score"] == 350


def test_tighter_policy_turns_away_good_borrowers():
    s, d, e = small_book()
    row = lending_policy_table(s, d, e, lgd=0.5, approval_rates=(0.5,)).iloc[0]
    assert row["good_borrowers_declined"] == 3
    assert row["good_borrowers_declined_pct"] == pytest.approx(3 / 8)


def test_loss_scales_with_exposure_not_head_count():
    s, d, _ = small_book()
    exposure = np.array([1, 1, 1, 1, 1, 1, 1, 1, 10_000, 1], dtype=float)
    row = lending_policy_table(s, d, exposure, lgd=1.0, approval_rates=(1.0,)).iloc[0]
    assert row["credit_loss"] == pytest.approx(10_001)


def test_tied_scores_still_approve_exactly_the_requested_share():
    """Ties at the cut-off are broken by position, so the approved count is exact."""
    scores = np.array([500, 400, 400, 400, 400, 300, 300, 300, 300, 200])
    defaulted = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
    row = lending_policy_table(scores, defaulted, np.ones(10), lgd=1.0, approval_rates=(0.3,)).iloc[0]
    assert row["approved"] == 3
    assert row["cut_off_score"] == 400


def test_policy_records_are_strict_json():
    """The approve-everyone row has no cut-off; it must serialise as null, not NaN."""
    import json

    from business_impact import policies_to_records

    s, d, e = small_book()
    records = policies_to_records(lending_policy_table(s, d, e))
    assert records[0]["cut_off_score"] is None
    text = json.dumps(records, allow_nan=False)
    assert "NaN" not in text
