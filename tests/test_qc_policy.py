from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from qc.config import DatasetConfig
from qc.conformal import benjamini_hochberg
from qc.policy import (
    HARD_FAILURE,
    HUMAN_APPROVED,
    INFORMATIONAL,
    PASSING,
    UNAVAILABLE_EVIDENCE,
    UNEXPLAINED_ANOMALY,
    disposition_for,
    finding,
    status_for,
)
from qc.recurrence import assess_recurrence
from qc.temporal import (
    share_shift_test,
    student_t_two_sided_p,
)

CONFIG = DatasetConfig()


def test_disposition_mapping():
    assert disposition_for("CONTRACT_FAILURE", True, ()) == HARD_FAILURE
    assert disposition_for("FAIL", True, ()) == UNEXPLAINED_ANOMALY
    assert disposition_for("FAIL", True, ("evt",)) == HUMAN_APPROVED
    assert disposition_for("UNAVAILABLE", True, ()) == UNAVAILABLE_EVIDENCE
    assert disposition_for("UNAVAILABLE", False, ()) == INFORMATIONAL
    assert disposition_for("PASS", True, ()) == PASSING
    assert disposition_for("PASS", False, ()) == INFORMATIONAL


def test_status_precedence_and_human_approval():
    unexplained = finding("temporal", "national", "FAIL")
    assert status_for([unexplained]) == "INVESTIGATE"

    approved = replace(
        unexplained, disposition=HUMAN_APPROVED, approval_ids=("evt-1",)
    )
    assert status_for([approved]) == "PASS_WITH_EXPLANATION"

    contract = finding("contracts:nulls", "ds", "CONTRACT_FAILURE")
    assert status_for([contract, approved]) == "DATA_CONTRACT_FAILURE"
    unavailable = finding("temporal", "ds", "UNAVAILABLE")
    assert status_for([unavailable, approved]) == "INCOMPLETE"
    assert status_for([unexplained, unavailable]) == "INVESTIGATE"


def test_benjamini_hochberg_never_claims_discovery_on_empty_input():
    adjusted, significant = benjamini_hochberg([], 0.05)
    assert adjusted == [] and significant == []
    adjusted, significant = benjamini_hochberg([0.001, 0.4, 0.6], 0.05)
    assert significant[0] is True
    assert significant[1:] == [False, False]
    assert adjusted[0] <= 0.05
    with pytest.raises(ValueError, match="p-values"):
        benjamini_hochberg([1.5])


def _recurrence_finding(finding_id, impact, materiality=50.0):
    from qc.policy import finding, stable_key_for

    item = finding(
        "temporal",
        "banner_id:B1",
        "FAIL",
        scope_type="temporal_series",
        metric="dollar",
        level="banner_id",
        period=10,
        impact=impact,
        materiality=materiality,
    )
    payload = {
        "finding_id": finding_id,
        "stable_key": stable_key_for(
            "temporal", "temporal_series", "dollar", "banner_id", "banner_id:B1"
        ),
        "check": "temporal",
        "scope": "banner_id:B1",
        "scope_type": "temporal_series",
        "metric": "dollar",
        "level": "banner_id",
        "outcome": "FAIL",
        "disposition": UNEXPLAINED_ANOMALY,
        "approval_ids": [],
        "impact": impact,
        "materiality": materiality,
    }
    assert item.stable_key == payload["stable_key"]
    return payload


def test_recurrence_requires_repetition_and_material_cumulative_impact():
    current = [_recurrence_finding("abc", 60.0)]
    prior = [{"findings": [_recurrence_finding("def", 60.0)]}]
    config = replace(CONFIG, recurrence_window=3, recurrence_minimum=2)
    escalated = assess_recurrence(current, prior, config)
    assert len(escalated) == 1
    assert escalated[0].repeated
    assert escalated[0].material
    assert escalated[0].occurrences == 2
    assert escalated[0].cumulative_impact == pytest.approx(120.0)

    small_prior = [{"findings": [_recurrence_finding("def", 5.0)]}]
    small = assess_recurrence(
        [_recurrence_finding("abc", 5.0)], small_prior, config
    )
    assert not small[0].material

    single = assess_recurrence(current, [], config)
    assert single == []

    zero_budget = assess_recurrence(
        current,
        prior,
        replace(config, recurrence_budget_ratio=0.0),
    )
    assert not zero_budget[0].material

    # Two periods in one logical refresh count once, with combined impact.
    repeated = assess_recurrence(
        [_recurrence_finding("abc", 30.0), _recurrence_finding("abd", 30.0)],
        prior,
        config,
    )
    assert repeated[0].occurrences == 2
    assert repeated[0].cumulative_impact == pytest.approx(120.0)

    approved_prior = [{"findings": [
        {
            **_recurrence_finding("def", 60.0),
            "disposition": HUMAN_APPROVED,
            "approval_ids": ["evt-1"],
        }
    ]}]
    assert assess_recurrence(current, approved_prior, config) == []

    # Signed impacts that net to zero still escalate on gross persistence.
    alternating = assess_recurrence(
        [_recurrence_finding("abc", 40.0)],
        [{"findings": [_recurrence_finding("def", -40.0)]}],
        config,
    )
    assert alternating[0].cumulative_impact == pytest.approx(0.0)
    assert alternating[0].cumulative_gross_impact == pytest.approx(80.0)
    assert alternating[0].material


@pytest.mark.parametrize(
    ("t", "df", "expected"),
    [(0.0, 10, 1.0), (2.0, 10, 0.07339), (3.0, 5, 0.03010), (-2.228, 10, 0.05)],
)
def test_student_t_two_sided_p_matches_reference_values(t, df, expected):
    assert student_t_two_sided_p(t, df) == pytest.approx(expected, abs=2e-4)


def test_share_shift_ignores_common_movement_and_flags_leaf_shift():
    rng = np.random.default_rng(3)
    weeks = list(range(1, 31))
    parent = [1000.0 * (1.0 + 0.1 * rng.standard_normal()) for _ in weeks]
    share = [0.3 + 0.005 * rng.standard_normal() for _ in weeks]
    child = [p * s for p, s in zip(parent, share, strict=True)]
    # A 15% market-wide drop keeps the leaf's share: not a leaf anomaly.
    common = share_shift_test(weeks, child, weeks, parent, 0.85 * 0.3 * 1000, 0.85 * 1000, 26)
    assert common is not None and common[1] > 0.05
    # The same leaf alone dropping 20% moves its share far outside its history.
    shifted = share_shift_test(weeks, child, weeks, parent, 0.8 * 0.3 * 1000, 1000.0, 26)
    assert shifted is not None and shifted[1] < 1e-6
    assert shifted[2] < 0.0
    # Too little aligned history is not evaluated rather than guessed.
    assert share_shift_test(weeks[:5], child[:5], weeks[:5], parent[:5], 300, 1000, 26) is None
