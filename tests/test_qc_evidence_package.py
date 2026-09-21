from __future__ import annotations

import math

import numpy as np
import pandas as pd

from qc.config import DatasetConfig
from qc.evidence_package import (
    EVIDENCE_PACKAGE_SCHEMA,
    AssessmentEvidence,
    build_assessment_evidence,
    verify_explanations,
)
from qc.run import run_qc

CONFIG = DatasetConfig(
    report_grain=(),
    forecaster="auto",
    calendar_anchor_date="2019-01-07",
    calendar_events=(("christmas", "12-25", 1, 1), ("easter", "easter", 1, 1)),
)


class FrameSource:
    def __init__(self, frames):
        self.frames = frames

    def list_versions(self):
        return ["v1", "v2"]

    def available_stages(self, version):
        return ["warehouse", "report"]

    def read_fact(self, version, stage):
        return self.frames[version, stage].copy()

    def read_dim(self, version, name):
        return None


def _source(target_delta: float = 0.0) -> FrameSource:
    rng = np.random.default_rng(11)
    values = {
        week: 300.0
        + 2.0 * week
        + 30.0 * math.sin(2.0 * math.pi * week / 52.0)
        + float(rng.normal(0.0, 3.0))
        for week in range(1, 121)
    }
    values[121] = values[120] + target_delta

    frames = {}
    for version, weeks in (("v1", range(1, 121)), ("v2", range(1, 122))):
        rows = [
            {
                "week": week,
                "store_id": store,
                "product_id": "p",
                "dollar": values[week],
                "units": values[week] / 10.0,
            }
            for week in weeks
            for store in ("a", "b")
        ]
        fact = pd.DataFrame(rows)
        report = fact.groupby("week", as_index=False)[["dollar", "units"]].sum()
        frames[version, "warehouse"] = fact
        frames[version, "report"] = report
    return FrameSource(frames)


def _evidence(target_delta: float = 0.0):
    result = run_qc(_source(target_delta), "v2", "v1", CONFIG)
    return result, build_assessment_evidence(
        result, CONFIG, assessment_id="assessment-1"
    )


def test_assessment_evidence_roundtrip_and_provider_view():
    _, evidence = _evidence()
    assert evidence.schema_version == EVIDENCE_PACKAGE_SCHEMA
    national = next(
        item for item in evidence.predictions if item.series_id == "national"
    )
    assert national.interval_lower is not None
    assert national.interval_upper is not None
    assert national.calibration_status == "OK"
    assert national.calibration_n > 0
    assert national.history_observed == 120
    assert evidence.calendar_identity["declared"] is True

    serialized = evidence.to_dict()
    assert AssessmentEvidence.from_dict(serialized).to_dict() == serialized

    view = evidence.provider_view(1_000_000)
    assert view.get("abstain") is not True
    for key in ("assessment_id", "calendar_identity", "net_unexplained"):
        assert key in view

    abstained = evidence.provider_view(40)
    assert abstained["abstain"] is True
    assert "budget" in abstained["reason"]


def _qualification(evidence, *, required_coverage=0.0, max_width_ratio=1e9, status="QUALIFIED"):
    from qc.qualification import QualificationArtifact, QualifiedCombination

    combinations = tuple(
        QualifiedCombination(
            model=prediction.selected_model,
            metric=prediction.metric,
            level=prediction.level,
            horizon=prediction.horizon,
            development_groups=100,
            test_groups=100,
            development_coverage=1.0,
            test_coverage=1.0,
            development_lower_bound=1.0,
            test_lower_bound=1.0,
            width_ratio=0.01,
            qualified=True,
        )
        for prediction in evidence.predictions
    )
    return QualificationArtifact(
        provenance="synthetic",
        required_coverage=required_coverage,
        max_width_ratio=max_width_ratio,
        minimum_groups=1,
        combinations=combinations,
        development_digest="dev",
        test_digest="test",
        status=status,
    )


def _finding_for(prediction, *, materiality=1e9, outcome="FAIL"):
    from qc.policy import finding

    return finding(
        "temporal",
        prediction.series_id,
        outcome,
        scope_type="temporal_series",
        metric=prediction.metric,
        level=prediction.level,
        period=prediction.target_week,
        impact=abs(prediction.residual),
        materiality=materiality,
    )


def test_certificates_verify_calibrated_explanation():
    _, evidence = _evidence()
    national = next(
        item for item in evidence.predictions if item.series_id == "national"
    )
    certificates = verify_explanations(
        evidence,
        CONFIG,
        materiality_threshold=1e9,
        findings=(_finding_for(national),),
        qualification=_qualification(evidence),
    )
    certificate = next(item for item in certificates if item.scope == "national")
    assert certificate.status == "VERIFIED"
    assert certificate.clearance_basis == "statistical"
    assert "historically_calibrated_interval" in certificate.bases
    assert certificate.support["calibration_n"] > 0
    assert certificate.finding_ids
    assert certificate.evidence_digest == evidence.digest
    assert certificate.to_dict()["schema_version"] == 3

    without_qualification = verify_explanations(
        evidence,
        CONFIG,
        materiality_threshold=1e9,
        findings=(_finding_for(national),),
    )
    rejected = next(
        item for item in without_qualification if item.scope == "national"
    )
    assert rejected.status == "REJECTED"
    assert any("qualification" in reason for reason in rejected.reasons)


def _revision_source(*, dollar_factor: float = 1.0, unit_factor: float = 1.0):
    rng = np.random.default_rng(21)
    values = {
        week: 300.0
        + 2.0 * week
        + 30.0 * math.sin(2.0 * math.pi * week / 52.0)
        + float(rng.normal(0.0, 2.0))
        for week in range(1, 122)
    }
    values[121] = 300.0 + 2.0 * 121 + 30.0 * math.sin(2.0 * math.pi * 121.0 / 52.0)
    frames = {}
    for version, weeks in (("v1", range(1, 121)), ("v2", range(1, 122))):
        scaled = version == "v2"
        rows = [
            {
                "week": week,
                "store_id": store,
                "product_id": "p",
                "dollar": values[week]
                * (dollar_factor if scaled and week <= 120 else 1.0),
                "units": (values[week] / 10.0)
                * (unit_factor if scaled and week <= 120 else 1.0),
            }
            for week in weeks
            for store in ("a", "b")
        ]
        fact = pd.DataFrame(rows)
        report = fact.groupby("week", as_index=False)[["dollar", "units"]].sum()
        frames[version, "warehouse"] = fact
        frames[version, "report"] = report
    return FrameSource(frames)


def test_broad_immaterial_revision_is_not_statistically_cleared():
    result = run_qc(_revision_source(dollar_factor=1.0002), "v2", "v1", CONFIG)
    historical = next(
        item
        for item in result.machine["findings"]
        if item["check"] == "historical_revision"
    )
    assert historical["outcome"] == "FAIL"
    assert historical["disposition"] == "UNEXPLAINED_ANOMALY"
    assert result.status in ("INVESTIGATE", "DATA_CONTRACT_FAILURE")
    assert not result.machine["clearance"]
    dataset_certificate = next(
        item for item in result.certificates if item.scope == "dataset"
    )
    assert dataset_certificate.status == "REJECTED"
    assert "does not authorize clearance" in dataset_certificate.reasons[0]
    ledger = result.machine["ledger"]
    assert abs(ledger["conservation"]["difference"]) <= 1e-6 * max(
        1.0, abs(ledger["raw_movement"])
    )


def test_material_price_change_is_not_cleared_without_support():
    result = run_qc(_revision_source(dollar_factor=1.05), "v2", "v1", CONFIG)
    historical = next(
        item
        for item in result.machine["findings"]
        if item["check"] == "historical_revision"
    )
    assert historical["disposition"] == "UNEXPLAINED_ANOMALY"
    assert result.status in ("INVESTIGATE", "DATA_CONTRACT_FAILURE")
    assert not result.machine["clearance"]


def test_certificates_reject_unexplained_movement():
    result, evidence = _evidence(target_delta=5000.0)
    national = next(
        item for item in evidence.predictions if item.series_id == "national"
    )
    certificates = verify_explanations(
        evidence,
        CONFIG,
        materiality_threshold=result.machine["materiality_threshold"],
        findings=(_finding_for(national, materiality=1.0),),
        qualification=_qualification(evidence),
    )
    certificate = next(item for item in certificates if item.scope == "national")
    assert certificate.status == "REJECTED"
    assert certificate.reasons
    assert any(
        "materiality" in reason or "interval" in reason
        for reason in certificate.reasons
    )
