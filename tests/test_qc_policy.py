from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from qc.config import DatasetConfig
from qc.conformal import benjamini_hochberg
from qc.policy import (
    HARD_FAILURE,
    HUMAN_APPROVED,
    INFORMATIONAL,
    PASSING,
    STATISTICALLY_EXPLAINED,
    UNAVAILABLE_EVIDENCE,
    UNEXPLAINED_ANOMALY,
    disposition_for,
    finalize_policy,
    finding,
    status_for,
)
from qc.recurrence import assess_recurrence
from qc.temporal import SeriesTemporalEvidence, coordinated_groups

CONFIG = DatasetConfig()


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
        dataset="ds",
        status="INVESTIGATE",
        contracts=SimpleNamespace(checks=[]),
        input_findings=[],
        machine={"historical_revision": {"status": "INVESTIGATE"}},
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


def _certificate(scope: str, status: str = "VERIFIED"):
    return SimpleNamespace(
        certificate_id=f"cert-{scope}",
        scope=scope,
        status=status,
        reasons=() if status == "VERIFIED" else ("rejected",),
    )


def test_certificate_clears_only_its_scope():
    result = _policy_result()
    finalize_policy(result, (), (_certificate("dataset"),))
    historical = next(
        item for item in result.machine["findings"] if item["check"] == "historical_revision"
    )
    assert historical["disposition"] == STATISTICALLY_EXPLAINED
    assert historical["clearance_basis"] == "statistical"
    assert historical["certificate_id"] == "cert-dataset"
    assert result.status == "PASS_WITH_EXPLANATION"
    assert result.machine["clearance_bases"][historical["finding_id"]] == "statistical"
    assert result.machine["schema_version"] == 3
    assert result.machine["clearance"][0]["basis"] == "statistical"


def test_rejected_certificate_does_not_clear():
    result = _policy_result()
    finalize_policy(result, (), (_certificate("dataset", status="REJECTED"),))
    historical = next(
        item for item in result.machine["findings"] if item["check"] == "historical_revision"
    )
    assert historical["disposition"] == UNEXPLAINED_ANOMALY
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
    )
    result = _policy_result(
        temporal=SimpleNamespace(
            series=[evidence],
            unavailable_series=[],
            coordinated=[],
        ),
        temporal_required=True,
    )
    finalize_policy(result, (), (_certificate("banner_id:B1"),))
    temporal = next(
        item for item in result.machine["findings"] if item["check"] == "temporal"
    )
    assert temporal["scope"] == "banner_id:B1"
    assert temporal["disposition"] == STATISTICALLY_EXPLAINED


def test_benjamini_hochberg_never_claims_discovery_on_empty_input():
    adjusted, significant = benjamini_hochberg([], 0.05)
    assert adjusted == [] and significant == []
    adjusted, significant = benjamini_hochberg([0.001, 0.4, 0.6], 0.05)
    assert significant[0] is True
    assert significant[1:] == [False, False]
    assert adjusted[0] <= 0.05
    with pytest.raises(ValueError, match="p-values"):
        benjamini_hochberg([1.5])


def test_recurrence_requires_repetition_and_material_cumulative_impact():
    current = [
        {
            "finding_id": "abc",
            "check": "temporal",
            "scope": "banner_id:B1",
            "outcome": "FAIL",
            "disposition": UNEXPLAINED_ANOMALY,
            "approval_ids": [],
            "clearance_basis": "",
        }
    ]
    prior = [
        {
            "findings": [{"finding_id": "abc"}],
            "ledger": {"net_unexplained": 60.0},
            "materiality_threshold": 50.0,
        }
    ]
    config = replace(CONFIG, recurrence_window=3, recurrence_minimum=2)
    escalated = assess_recurrence(
        current, prior, config, current_impact=60.0, current_threshold=50.0
    )
    assert len(escalated) == 1
    assert escalated[0].repeated
    assert escalated[0].material
    assert escalated[0].occurrences == 2

    small = assess_recurrence(
        current, prior, config, current_impact=5.0, current_threshold=50.0
    )
    assert not small[0].material

    single = assess_recurrence(
        current, [], config, current_impact=60.0, current_threshold=50.0
    )
    assert single == []


def test_coordinated_groups_combine_same_direction_residuals():
    def item(series_id, level, residual, relative):
        return SeriesTemporalEvidence(
            series_id=series_id,
            target_week=41,
            actual=100.0,
            adjusted_actual=100.0,
            adjustment=0.0,
            forecast_median=100.0,
            forecast_quantiles={},
            residual=residual,
            relative_residual=relative,
            nominal_percentile=0.2,
            calibrated_percentile=None,
            standardized_residual=1.0,
            robust_z=1.0,
            seasonal_z=1.0,
            ewma_z=1.0,
            change_point_score=0.0,
            anomaly=False,
            flags=[],
            level=level,
        )

    evidence = [
        item("national", "national", -300.0, -0.03),
        item("banner_id:B1", "banner_id", -80.0, -0.012),
        item("banner_id:B2", "banner_id", -70.0, -0.011),
        item("banner_id:B3", "banner_id", 40.0, 0.006),
    ]
    groups = coordinated_groups(evidence, CONFIG)
    assert len(groups) == 1
    assert groups[0]["level"] == "banner_id"
    assert groups[0]["direction"] == "decrease"
    assert groups[0]["series_ids"] == ["banner_id:B1", "banner_id:B2"]
    assert groups[0]["combined_ratio"] >= CONFIG.temporal_min_relative_residual
    assert groups[0]["individually_flagged"] is False

    quiet = coordinated_groups(evidence[:1], CONFIG)
    assert quiet == []
