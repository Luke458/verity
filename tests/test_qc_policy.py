from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from qc.config import DatasetConfig
from qc.conformal import benjamini_hochberg
from qc.evidence_package import ExplanationCertificate
from qc.policy import (
    HARD_FAILURE,
    HUMAN_APPROVED,
    INFORMATIONAL,
    PASSING,
    STATISTICALLY_EXPLAINED,
    UNAVAILABLE_EVIDENCE,
    UNEXPLAINED_ANOMALY,
    collect_findings,
    disposition_for,
    finalize_policy,
    finding,
    status_for,
)
from qc.recurrence import assess_recurrence
from qc.temporal import (
    SeriesTemporalEvidence,
    share_shift_test,
    student_t_two_sided_p,
)

CONFIG = DatasetConfig()


def _certificate_for(item, *, status="VERIFIED", assessment_id="run-1", **overrides):
    kwargs = dict(
        certificate_id=f"cert-{item.scope}",
        assessment_id=assessment_id,
        scope=item.scope,
        finding_ids=(item.finding_id,),
        bases=("historically_calibrated_interval",),
        status=status,
        reasons=() if status == "VERIFIED" else ("rejected",),
        support={},
        coverage={"finding_ids": [item.finding_id]},
        net_unexplained=0.0,
        gross_unexplained=0.0,
        materiality_threshold=1.0,
        evidence_ids=(),
        approval_ids=(),
        clearance_basis="statistical",
        evidence_digest="evidence-1",
        qualification_digest="qualification-1",
        scope_type=item.scope_type,
        period=item.period,
        metric=item.metric,
        level=item.level,
    )
    kwargs.update(overrides)
    return ExplanationCertificate(**kwargs)


def test_disposition_mapping():
    assert disposition_for("CONTRACT_FAILURE", True, ()) == HARD_FAILURE
    assert disposition_for("FAIL", True, ()) == UNEXPLAINED_ANOMALY
    assert disposition_for("FAIL", True, ("evt",)) == HUMAN_APPROVED
    assert disposition_for("UNAVAILABLE", True, ()) == UNAVAILABLE_EVIDENCE
    assert disposition_for("UNAVAILABLE", False, ()) == INFORMATIONAL
    assert disposition_for("PASS", True, ()) == PASSING
    assert disposition_for("PASS", False, ()) == INFORMATIONAL


def test_status_for_distinguishes_statistical_clearance():
    unexplained = finding("temporal", "national", "FAIL")
    assert status_for([unexplained]) == "INVESTIGATE"

    cleared = replace(
        unexplained,
        disposition=STATISTICALLY_EXPLAINED,
        clearance_basis="statistical",
        certificate_id="cert-1",
    )
    assert status_for([cleared]) == "PASS_WITH_EXPLANATION"

    approved = replace(
        unexplained, disposition=HUMAN_APPROVED, approval_ids=("evt-1",)
    )
    assert status_for([approved]) == "PASS_WITH_EXPLANATION"

    contract = finding("contracts:nulls", "ds", "CONTRACT_FAILURE")
    assert status_for([contract, cleared]) == "DATA_CONTRACT_FAILURE"
    unavailable = finding("temporal", "ds", "UNAVAILABLE")
    assert status_for([unavailable, cleared]) == "INCOMPLETE"


def _policy_result(**overrides):
    base = dict(
        run_id="run-1",
        dataset="ds",
        status="INVESTIGATE",
        contracts=SimpleNamespace(checks=[]),
        input_findings=[],
        machine={
            "historical_revision": {"status": "INVESTIGATE"},
            "evidence_digests": ["evidence-1"],
        },
        attribution=SimpleNamespace(
            matched_event_ids=[],
            approval_coverage=[],
        ),
        reconciliation=None,
        counterfactual=None,
        cubes={},
        reference=None,
        expectations=(),
        observed_at=None,
        temporal_required=False,
        temporal=None,
        ledger=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_dataset_certificate_cannot_clear_a_historical_revision():
    result = _policy_result()
    findings = collect_findings(result, CONFIG)
    historical = next(
        item for item in findings if item.check == "historical_revision"
    )
    certificate = _certificate_for(
        historical,
        scope="dataset",
        scope_type="dataset",
        period=None,
    )
    finalize_policy(result, findings=findings, certificates=(certificate,))
    stored = next(
        item for item in result.machine["findings"] if item["check"] == "historical_revision"
    )
    assert stored["disposition"] == UNEXPLAINED_ANOMALY
    assert result.status == "INVESTIGATE"
    assert not result.machine["clearance"]


def test_rejected_certificate_does_not_clear():
    result = _policy_result()
    findings = collect_findings(result, CONFIG)
    historical = next(
        item for item in findings if item.check == "historical_revision"
    )
    certificate = _certificate_for(
        historical,
        status="REJECTED",
        scope="dataset",
        scope_type="dataset",
        period=None,
    )
    finalize_policy(result, findings=findings, certificates=(certificate,))
    stored = next(
        item for item in result.machine["findings"] if item["check"] == "historical_revision"
    )
    assert stored["disposition"] == UNEXPLAINED_ANOMALY
    assert result.status == "INVESTIGATE"


def test_temporal_certificate_clears_matching_series_only():
    evidence = SeriesTemporalEvidence(
        series_id="banner_id:B1",
        target_week=10,
        actual=100.0,
        adjusted_actual=100.0,
        adjustment=0.0,
        forecast_median=100.0,
        forecast_quantiles={},
        residual=0.0,
        relative_residual=0.0,
        nominal_percentile=0.5,
        calibrated_percentile=0.5,
        standardized_residual=0.0,
        robust_z=0.0,
        seasonal_z=0.0,
        ewma_z=0.0,
        change_point_score=0.0,
        anomaly=True,
        flags=["forecast_lower"],
        metric="dollar",
        level="banner_id",
    )
    result = _policy_result(
        temporal=SimpleNamespace(
            series=[evidence],
            unavailable_series=[],
            coordinated=[],
        ),
        temporal_required=True,
    )
    findings = collect_findings(result, CONFIG)
    temporal = next(item for item in findings if item.check == "temporal")
    assert temporal.period == 10
    certificate = _certificate_for(temporal)
    finalize_policy(result, findings=findings, certificates=(certificate,))
    stored = next(
        item for item in result.machine["findings"] if item["check"] == "temporal"
    )
    assert stored["scope"] == "banner_id:B1"
    assert stored["disposition"] == STATISTICALLY_EXPLAINED
    assert result.machine["schema_version"] == 5
    assert result.machine["clearance"][0]["period"] == 10
    assert result.machine["clearance_bases"][stored["finding_id"]] == "statistical"


def test_foreign_assessment_certificate_is_ignored():
    evidence = SeriesTemporalEvidence(
        series_id="national",
        target_week=10,
        actual=100.0,
        adjusted_actual=100.0,
        adjustment=0.0,
        forecast_median=100.0,
        forecast_quantiles={},
        residual=0.0,
        relative_residual=0.0,
        nominal_percentile=0.5,
        calibrated_percentile=0.5,
        standardized_residual=0.0,
        robust_z=0.0,
        seasonal_z=0.0,
        ewma_z=0.0,
        change_point_score=0.0,
        anomaly=True,
        flags=["forecast_lower"],
        metric="dollar",
        level="national",
    )
    result = _policy_result(
        temporal=SimpleNamespace(
            series=[evidence], unavailable_series=[], coordinated=[]
        ),
        temporal_required=True,
    )
    findings = collect_findings(result, CONFIG)
    temporal = next(item for item in findings if item.check == "temporal")
    certificate = _certificate_for(temporal, assessment_id="other-run")
    finalize_policy(result, findings=findings, certificates=(certificate,))
    stored = next(
        item for item in result.machine["findings"] if item["check"] == "temporal"
    )
    assert stored["disposition"] == UNEXPLAINED_ANOMALY
    assert not result.machine["clearance"]


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
        "clearance_basis": "",
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

    cleared_prior = [{"findings": [
        {**_recurrence_finding("def", 60.0), "disposition": "STATISTICALLY_EXPLAINED"}
    ]}]
    assert assess_recurrence(current, cleared_prior, config) == []

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
