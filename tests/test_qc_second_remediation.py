"""Regression acceptance for the second remediation plan.

Each test converts a demonstrated reproduction or required behavior from
``docs/qc-second-remediation-plan.md`` into a regression guard that runs
through the real orchestration paths where the defect was observed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from qc.cohort import CohortCase, _common_gate_results, _evaluation_cases
from qc.config import DatasetConfig
from qc.decisions import FEATURE_VERSION
from qc.evaluation import EvaluationCase, evaluation_gates
from qc.evidence_package import (
    AssessmentEvidence,
    ExplanationCertificate,
    PredictionEvidence,
    verify_explanations,
)
from qc.labels import records_from_store
from qc.policy import finding
from qc.qualification import (
    QualificationArtifact,
    QualifiedCombination,
    build_qualification,
    pin_qualification,
)
from qc.recurrence import assess_recurrence
from qc.run import run_qc
from qc.store import SqliteStore
from qc.systemone import ProviderAbstention, build_provider_state
from qc.weekly import run_weekly

STAGES = ("warehouse", "report")
BASE = DatasetConfig(
    name="second-remediation",
    report_grain=(),
    calendar_anchor_date="2019-01-07",
    calendar_events=(("christmas", "12-25", 1, 1), ("easter", "easter", 1, 1)),
)


class FrameSource:
    def __init__(self, frames):
        self.frames = frames

    def list_versions(self):
        return ["V0001", "V0002"]

    def available_stages(self, version):
        return list(STAGES)

    def read_fact(self, version, stage):
        return self.frames[version, stage].copy()

    def read_dim(self, version, name):
        return None


def _periodic(week: int) -> float:
    return (
        500.0
        + 100.0 * math.sin(2.0 * math.pi * week / 52.0)
        + 5.0 * math.sin(4.0 * math.pi * week / 52.0)
    )


def _frame(
    dollar: dict[int, float],
    units: dict[int, float],
    weeks: range,
    *,
    stock: float | None = None,
) -> pd.DataFrame:
    rows = []
    for week in weeks:
        for store in ("S1", "S2"):
            row = {
                "week": week,
                "store_id": store,
                "product_id": "P1",
                "dollar": dollar[week],
                "units": units[week],
            }
            if stock is not None:
                row["stock"] = stock
            rows.append(row)
    return pd.DataFrame(rows)


def _two_week_source(*, unit_factor: float = 1.0, stock: bool = False) -> FrameSource:
    dollar = {week: _periodic(week) for week in range(1, 123)}
    units = {week: _periodic(week) / 10.0 for week in range(1, 123)}
    units[121] = units[121] * unit_factor
    frames = {}
    metrics = ["dollar", "units"] + (["stock"] if stock else [])
    for version, weeks in (
        ("V0001", range(1, 121)),
        ("V0002", range(1, 123)),
    ):
        fact = _frame(dollar, units, weeks, stock=100.0 if stock else None)
        report = fact.groupby("week", as_index=False)[metrics].sum()
        frames[version, "warehouse"] = fact
        frames[version, "report"] = report
    return FrameSource(frames)


def _dollar_only_source() -> FrameSource:
    dollar = {week: _periodic(week) for week in range(1, 123)}
    frames = {}
    for version, weeks in (
        ("V0001", range(1, 121)),
        ("V0002", range(1, 123)),
    ):
        fact = _frame(dollar, {week: 1.0 for week in weeks}, weeks)
        fact = fact.drop(columns=["units"])
        report = fact.groupby("week", as_index=False)[["dollar"]].sum()
        frames[version, "warehouse"] = fact
        frames[version, "report"] = report
    return FrameSource(frames)


# ---------------------------------------------------------------------------
# 1. Secondary measures are checked
# ---------------------------------------------------------------------------


def test_units_collapse_with_unchanged_dollars_requires_review():
    result = run_qc(_two_week_source(unit_factor=0.01), "V0002", "V0001", BASE)
    assert result.temporal is not None
    failures = [
        item
        for item in result.machine["findings"]
        if item["check"] == "temporal"
        and item["metric"] == "units"
        and item["outcome"] == "FAIL"
    ]
    assert failures
    assert result.status == "INVESTIGATE"
    assert result.machine["requires_investigation"] is True
    assert all(
        item["disposition"] != "STATISTICALLY_EXPLAINED" for item in failures
    )


def test_absent_optional_metric_is_reported_and_missing_required_is_incomplete():
    config = DatasetConfig(
        name="metrics",
        report_grain=(),
        metric_columns=("dollar", "units"),
        required_columns=("week", "dollar"),
        temporal_required_metrics=("dollar",),
        temporal_optional_metrics=("units",),
    )
    result = run_qc(_dollar_only_source(), "V0002", "V0001", config)
    assert result.temporal is not None
    assert result.temporal.metric_status == {
        "dollar": "ASSESSED",
        "units": "ABSENT",
    }
    absent = next(
        item
        for item in result.machine["findings"]
        if item["check"] == "input:metric:units"
    )
    assert absent["outcome"] == "UNAVAILABLE"
    assert absent["required"] is False
    assert result.status != "INCOMPLETE"

    required = replace(
        config,
        temporal_required_metrics=("dollar", "units"),
        temporal_optional_metrics=(),
    )
    incomplete = run_qc(_dollar_only_source(), "V0002", "V0001", required)
    assert incomplete.status == "INCOMPLETE"


def test_snapshot_measure_is_compared_within_period_not_summed_across_time():
    result = run_qc(_two_week_source(stock=True), "V0002", "V0001", BASE)
    assert result.temporal is not None
    pytest.importorskip("qc")
    package = next(
        package
        for package in result.evidence_packages
        if any(item.metric == "stock" for item in package.predictions)
    )
    stock = next(
        item for item in package.predictions if item.metric == "stock"
    )
    assert stock.aggregation == "snapshot"
    # Two stores hold 100 each at that week: the level is compared within the
    # period, never accumulated across weeks (which would grow without bound).
    assert stock.actual == pytest.approx(200.0)
    assert stock.residual == pytest.approx(0.0, abs=1e-9)
    stock_failures = [
        item
        for item in result.machine["findings"]
        if item["check"] == "temporal" and item["metric"] == "stock"
    ]
    assert all(item["outcome"] != "FAIL" for item in stock_failures)


def test_metric_ledgers_are_keyed_by_measure():
    result = run_qc(_two_week_source(unit_factor=0.01), "V0002", "V0001", BASE)
    package = result.evidence_packages[0]
    assert {"dollar", "units"}.issubset(set(package.metric_ledgers))
    view = package.provider_view(1_000_000)
    assert "metric_ledgers" in view
    assert "units" in view["metric_ledgers"]


# ---------------------------------------------------------------------------
# 2. Certificate binding fails closed
# ---------------------------------------------------------------------------


def _prediction(**overrides) -> PredictionEvidence:
    base = dict(
        series_id="national",
        metric="dollar",
        level="national",
        target_week=121,
        actual=100.0,
        expected=100.0,
        residual=0.0,
        relative_residual=0.0,
        standardized_residual=0.0,
        nominal_percentile=0.5,
        calibrated_percentile=0.5,
        interval_lower=90.0,
        interval_upper=110.0,
        interval_alpha=0.1,
        interval_width=20.0,
        interval_coverage=0.9,
        calibration_status="OK",
        calibration_pool="dollar|national|ridge:v1|h1",
        calibration_n=20,
        forecast_error=3.0,
        history_weeks=120,
        history_observed=120,
        history_missing=0,
        missing_weeks=(),
        selected_model="ridge:v1",
        anomaly=False,
        flags=(),
        horizon=1,
        support="model",
        materiality=10.0,
        training_endpoint_week=120,
        forecast_origin_week=120,
    )
    base.update(overrides)
    return PredictionEvidence(**base)


def _package(prediction: PredictionEvidence, **overrides) -> AssessmentEvidence:
    base = dict(
        assessment_id="assessment-1",
        run_id="run-1",
        dataset="ds",
        observation_cutoff="2026-01-01T00:00:00+00:00",
        snapshot_identity={},
        calendar_identity={},
        model_identity={},
        config_identity={},
        predictions=(prediction,),
        contracts=({"name": "keys", "status": "PASS", "detail": ""},),
        materiality_threshold=100.0,
        provenance="synthetic",
    )
    base.update(overrides)
    return AssessmentEvidence(**base)


def _qualification(prediction: PredictionEvidence, **overrides) -> QualificationArtifact:
    kwargs = dict(
        provenance="synthetic",
        required_coverage=0.0,
        max_width_ratio=1e9,
        minimum_groups=1,
        combinations=(
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
            ),
        ),
        development_digest="dev",
        test_digest="test",
        status="QUALIFIED",
    )
    kwargs.update(overrides)
    return QualificationArtifact(**kwargs)


def _temporal_finding(period: int = 121, **overrides):
    kwargs = dict(
        scope="national",
        outcome="FAIL",
        scope_type="temporal_series",
        metric="dollar",
        level="national",
        period=period,
        impact=1.0,
        materiality=1e9,
    )
    kwargs.update(overrides)
    return finding("temporal", **kwargs)


def _certificate(item, **overrides) -> ExplanationCertificate:
    kwargs = dict(
        certificate_id=f"cert-{item.finding_id[:6]}",
        assessment_id="assessment-1",
        scope=item.scope,
        finding_ids=(item.finding_id,),
        bases=("historically_calibrated_interval",),
        status="VERIFIED",
        reasons=(),
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


def _policy_result():
    return SimpleNamespace(
        run_id="run-1",
        assessment_id="assessment-1",
        dataset="ds",
        status="INVESTIGATE",
        machine={"evidence_digests": ["evidence-1"]},
        attribution=None,
    )


def test_blank_stale_wrong_scope_and_unsupported_certificates_cannot_clear():
    variants = {
        "blank digest": _certificate(_temporal_finding(), evidence_digest=""),
        "wrong assessment": _certificate(
            _temporal_finding(), assessment_id="other"
        ),
        "stale schema": _certificate(_temporal_finding(), schema_version=2),
        "wrong policy": _certificate(_temporal_finding(), policy_version="2"),
        "wrong metric": _certificate(_temporal_finding(), metric="units"),
        "wrong period": _certificate(_temporal_finding(), period=122),
        "wrong scope": _certificate(_temporal_finding(), scope="banner_id:B1"),
        "missing qualification": _certificate(
            _temporal_finding(), qualification_digest=""
        ),
        "unknown evidence digest": _certificate(
            _temporal_finding(), evidence_digest="not-recorded"
        ),
    }
    from qc.policy import apply_policy

    for label, certificate in variants.items():
        result = _policy_result()
        apply_policy(result, [_temporal_finding()], (certificate,))
        assert result.status == "INVESTIGATE", label
        assert not result.machine["clearance"], label


# ---------------------------------------------------------------------------
# 3. Applicable failures prevent statistical authorization
# ---------------------------------------------------------------------------


def test_failed_reconciliation_and_required_inputs_block_clearance():
    prediction = _prediction()
    temporal = _temporal_finding()
    blocking_findings = [
        finding(
            "reconciliation:national",
            "ds",
            "FAIL",
            scope_type="reconciliation",
            impact=50.0,
            materiality=1.0,
        ),
        finding(
            "input:warehouse",
            "ds",
            "UNAVAILABLE",
            required=True,
        ),
        finding(
            "hierarchy_total:banner_id",
            "banner_id",
            "FAIL",
        ),
        finding("lineage", "ds", "FAIL", required=False),
    ]
    certificates = verify_explanations(
        _package(prediction),
        BASE,
        materiality_threshold=100.0,
        findings=(temporal, *blocking_findings),
        qualification=_qualification(prediction),
    )
    for item in blocking_findings:
        certificate = next(
            entry for entry in certificates if entry.finding_ids == (item.finding_id,)
        ) if any(entry.finding_ids == (item.finding_id,) for entry in certificates) else None
        # Blocking findings are not clearable at all and never verified.
        assert certificate is None or certificate.status == "REJECTED"


def test_temporal_clearance_is_rejected_when_any_applicable_failure_exists():
    prediction = _prediction()
    temporal = _temporal_finding()
    certificates = verify_explanations(
        _package(
            prediction,
            contracts=({"name": "keys", "status": "PASS", "detail": ""},),
        ),
        BASE,
        materiality_threshold=100.0,
        findings=(
            temporal,
            finding("reconciliation:national", "ds", "FAIL"),
            finding("input:warehouse", "ds", "UNAVAILABLE", required=True),
        ),
        qualification=_qualification(prediction),
    )
    certificate = next(
        entry
        for entry in certificates
        if entry.clearance_basis == "statistical"
        and entry.finding_ids == (temporal.finding_id,)
    )
    assert certificate.status == "REJECTED"
    assert any("integrity failures" in reason for reason in certificate.reasons)


# ---------------------------------------------------------------------------
# 4. Qualification identity, revocation and replay
# ---------------------------------------------------------------------------


def _write_delta_versions(path: Path, targets: tuple[int, ...]) -> None:
    pytest.importorskip("deltalake")
    from deltalake import write_deltalake

    for index, target in enumerate(targets):
        rows = [
            {
                "week": week,
                "store_id": store,
                "product_id": "P1",
                "dollar": float(week),
                "units": 1.0,
            }
            for week in range(1, target + 1)
            for store in ("S1", "S2")
        ]
        write_deltalake(
            str(path), pd.DataFrame(rows), mode="overwrite" if index else "error"
        )


def _pinned_qualification(
    path: Path,
    *,
    status: str = "QUALIFIED",
    created: str = "2020-01-01T00:00:00+00:00",
) -> str:
    artifact = QualificationArtifact(
        provenance="synthetic",
        required_coverage=0.8,
        max_width_ratio=0.25,
        minimum_groups=9,
        combinations=(),
        development_digest="dev",
        test_digest="test",
        status=status,
    )
    return pin_qualification(artifact, path, created=created)


def test_qualification_replacement_changes_identity_and_replay(tmp_path):
    pytest.importorskip("deltalake")
    path = tmp_path / "fact"
    _write_delta_versions(path, (30, 31, 32))
    qualification = tmp_path / "qualification.json"
    digest_a = _pinned_qualification(qualification)
    config = replace(
        DatasetConfig(),
        name="qualification-identity",
        qualification_path=str(qualification),
    )
    first = run_weekly(
        str(path),
        config=config,
        store_path=tmp_path / "store.db",
        out_root=tmp_path / "weekly",
    )
    assert first.skipped is False

    reused = run_weekly(
        str(path),
        config=config,
        store_path=tmp_path / "store.db",
        out_root=tmp_path / "weekly",
    )
    assert reused.skipped is True
    assert reused.assessment_id == first.assessment_id

    # Revocation at the same path must change identity, never reuse the cache.
    digest_b = _pinned_qualification(
        qualification,
        status="UNQUALIFIED",
        created="2020-01-02T00:00:00+00:00",
    )
    assert digest_b != digest_a
    revoked = run_weekly(
        str(path),
        config=config,
        store_path=tmp_path / "store.db",
        out_root=tmp_path / "weekly",
    )
    assert revoked.skipped is False
    assert revoked.assessment_id != first.assessment_id

    # Losing the artifact entirely is also a distinct, unavailable state.
    qualification.unlink()
    qualification.with_name(qualification.name + ".sha256").unlink()
    lost = run_weekly(
        str(path),
        config=config,
        store_path=tmp_path / "store.db",
        out_root=tmp_path / "weekly",
    )
    assert lost.skipped is False
    assert lost.assessment_id not in (first.assessment_id, revoked.assessment_id)

    # Historical replay still sees its original frozen identity.
    with SqliteStore(tmp_path / "store.db") as store:
        original = store.assessment(first.assessment_id)
        assert original is not None
        assert original["identity"]["qualification"]["digest"] == digest_a
        assert original["result"]["assessment_id"] == first.assessment_id


def test_future_qualification_cannot_affect_a_historical_assessment(tmp_path):
    pytest.importorskip("deltalake")
    path = tmp_path / "fact"
    _write_delta_versions(path, (30, 31, 32))
    qualification = tmp_path / "qualification.json"
    _pinned_qualification(
        qualification, created="2099-01-01T00:00:00+00:00"
    )
    config = replace(
        DatasetConfig(),
        name="qualification-future",
        qualification_path=str(qualification),
    )
    first = run_weekly(
        str(path),
        config=config,
        store_path=tmp_path / "store.db",
        out_root=tmp_path / "weekly",
    )
    with SqliteStore(tmp_path / "store.db") as store:
        identity = store.assessment(first.assessment_id)["identity"]
    assert identity["qualification"]["status"] == "UNAVAILABLE"
    assert "cutoff" in identity["qualification"]["error"]

    _pinned_qualification(
        qualification, created="2020-01-01T00:00:00+00:00"
    )
    available = run_weekly(
        str(path),
        config=config,
        store_path=tmp_path / "store.db",
        out_root=tmp_path / "weekly",
    )
    assert available.skipped is False
    assert available.assessment_id != first.assessment_id


# ---------------------------------------------------------------------------
# 5. Provider evidence is mandatory-aware
# ---------------------------------------------------------------------------


def test_oversized_failed_check_evidence_compresses_or_abstains():
    check = {
        "check": "contracts:keys",
        "scope": "ds",
        "scope_type": "contract",
        "outcome": "CONTRACT_FAILURE",
        "detail": "x" * 5000,
        "finding_id": "f1",
    }
    package = _package(_prediction(), failed_checks=(check,))
    view = package.provider_view(1500)
    assert view.get("abstain") is not True
    assert view["failed_checks"]
    assert "x" * 1000 not in json.dumps(view)
    assert view["omitted"]
    assert len(view["failed_checks"][0]["detail"]) <= 200

    tiny = package.provider_view(200)
    assert tiny["abstain"] is True
    assert "budget" in tiny["reason"]

    result = SimpleNamespace(evidence_packages=[package], evidence_package=package)
    with pytest.raises(ProviderAbstention):
        build_provider_state(result, max_chars=200)


def test_provider_capture_contains_required_evidence_and_no_policy_answers():
    result = run_qc(_two_week_source(unit_factor=0.01), "V0002", "V0001", BASE)
    state = build_provider_state(result, max_chars=400_000)
    assert "failed_checks" in state
    assert "missing_required" in state
    assert "disposition" not in state
    assert "requires_investigation" not in state
    assert "certificate" not in state
    assert "clearance" not in state


# ---------------------------------------------------------------------------
# 6. Qualification cannot be gamed
# ---------------------------------------------------------------------------


def _entry(
    group: str,
    split: str,
    *,
    covered: bool = True,
    width: float = 0.01,
    model: str = "ridge:v1",
    assessment_id: str = "assessment",
) -> dict:
    return {
        "assessment_id": assessment_id,
        "group": group,
        "cutoff": "2026-01-01T00:00:00+00:00",
        "split": split,
        "basis": "held_out_target",
        "model": model,
        "metric": "dollar",
        "level": "national",
        "horizon": 1,
        "covered": covered,
        "width_ratio": width,
    }


def _passing_gates() -> dict:
    return {
        "status": "PASS",
        "gates": {
            "detection_rate": {"status": "PASS"},
            "false_positive_rate": {"status": "PASS"},
            "false_clearance": {"status": "PASS"},
        },
    }


def test_qualification_rejects_duplicates_overlap_and_thin_evidence():
    duplicate = [_entry("g1", "development")] * 2 + [
        _entry("g2", "development")
    ]
    with pytest.raises(ValueError, match="duplicate"):
        build_qualification(
            duplicate,
            [_entry("t1", "test")],
            provenance="synthetic",
            gates=_passing_gates(),
        )

    with pytest.raises(ValueError, match="disjoint"):
        build_qualification(
            [_entry("g1", "development")],
            [_entry("g1", "test")],
            provenance="synthetic",
            gates=_passing_gates(),
        )

    thin = build_qualification(
        [_entry("g1", "development"), _entry("g2", "development")],
        [_entry("t1", "test")],
        provenance="synthetic",
        minimum_groups=2,
        required_coverage=0.4,
        gates=_passing_gates(),
    )
    assert thin.status == "UNQUALIFIED"
    assert not thin.supports("ridge:v1", "dollar", "national", 1)

    missing_gates = build_qualification(
        [_entry(f"g{i}", "development") for i in range(3)],
        [_entry(f"t{i}", "test") for i in range(3)],
        provenance="synthetic",
        minimum_groups=3,
        required_coverage=0.4,
        gates=None,
    )
    assert missing_gates.status == "UNQUALIFIED"
    assert any("gate" in reason for reason in missing_gates.reasons)


def test_real_qualification_requires_linked_labels():
    with pytest.raises(ValueError, match="analyst-labelled"):
        build_qualification(
            [_entry(f"g{i}", "development") for i in range(3)],
            [_entry(f"t{i}", "test") for i in range(3)],
            provenance="real",
            minimum_groups=3,
            required_coverage=0.4,
            gates=_passing_gates(),
        )


def test_qualification_with_independent_groups_and_gates_is_qualified():
    artifact = build_qualification(
        [_entry(f"g{i}", "development") for i in range(3)],
        [_entry(f"t{i}", "test") for i in range(3)],
        provenance="synthetic",
        minimum_groups=3,
        required_coverage=0.4,
        gates=_passing_gates(),
        config_hash="config",
        calendar_hash="calendar",
        evidence_policy_version="3",
    )
    assert artifact.status == "QUALIFIED"
    assert artifact.supports("ridge:v1", "dollar", "national", 1)
    assert artifact.config_hash == "config"


# ---------------------------------------------------------------------------
# 7. Recurrence survives serialization, storage and replay
# ---------------------------------------------------------------------------


def _recurrence_payload() -> dict:
    item = finding(
        "temporal",
        "national",
        "FAIL",
        scope_type="temporal_series",
        metric="dollar",
        level="national",
        period=121,
        impact=10.0,
        materiality=5.0,
    )
    return json.loads(json.dumps(asdict(item)))


def test_recurrence_survives_serialization_and_storage(tmp_path):
    serialized = _recurrence_payload()
    assert serialized["stable_key"]
    prior = [{"findings": [serialized], "ledger": {}, "materiality_threshold": 0.0}]
    config = replace(
        DatasetConfig(), recurrence_window=3, recurrence_minimum=2
    )
    assessments = assess_recurrence([serialized], prior, config)
    assert assessments
    assert assessments[0].stable_key == serialized["stable_key"]
    assert assessments[0].material
    assert assessments[0].cumulative_impact == 20.0

    store = SqliteStore(tmp_path / "store.db")
    store.record_run(
        "run-a",
        "recurrence",
        "INVESTIGATE",
        {
            "run_id": "run-a",
            "status": "INVESTIGATE",
            "version_pair": {"previous_id": "1", "current_id": "2"},
            "findings": [serialized],
            "materiality_threshold": 1.0,
        },
        created="2026-01-10T00:00:00+00:00",
    )
    entries = store.recurrence_inputs(
        "recurrence",
        cutoff="2026-02-01T00:00:00+00:00",
        window=3,
        current_version="3",
    )
    refreshes = store.recurrence_refreshes(entries)
    store.close()
    replayed = assess_recurrence([serialized], refreshes, config)
    assert replayed and replayed[0].stable_key == serialized["stable_key"]


# ---------------------------------------------------------------------------
# 8. One evaluation contract everywhere
# ---------------------------------------------------------------------------


def test_store_and_synthetic_evaluation_produce_identical_gates():
    cases = []
    for index in range(12):
        for actionable in (True, False):
            cases.append(
                EvaluationCase(
                    case_id=f"{index}-{actionable}",
                    family="missing_stores" if actionable else "market_movement",
                    group=f"incident-{index}-{actionable}",
                    actionable=actionable,
                    verifier_status="INVESTIGATE" if actionable else "PASS",
                    challenger_status="INVESTIGATE" if actionable else "PASS",
                    effective_status="INVESTIGATE" if actionable else "PASS",
                    verifier_review=actionable,
                    challenger_review=actionable,
                    effective_review=actionable,
                    expected_review=actionable,
                )
            )
    synthetic = evaluation_gates(cases)

    cohort_cases = [
        CohortCase(
            case_id=case.case_id,
            split="heldout",
            seed=index,
            family=case.family,
            is_control=not case.actionable,
            engine_status=case.verifier_status,
            historical_status=None,
            expected_status="INVESTIGATE" if case.actionable else "PASS",
            expected_class=None,
            injection_stage=None,
            first_divergence=None,
            reconstruction_score=None,
            explained_fraction=1.0,
            detected=case.effective_review,
            expected_match=True,
            false_positive=False,
        )
        for index, case in enumerate(cases)
    ]
    mapped = _evaluation_cases(cohort_cases)
    assert evaluation_gates(mapped) == synthetic
    checks, common = _common_gate_results(cohort_cases, {
        "min_detection_rate": 0.9,
        "max_false_positive_rate": 0.1,
    })
    assert common == evaluation_gates(mapped, minimum_detection_rate=0.9)
    assert any(check["gate"] == "false_clearance_upper_bound" for check in checks)


# ---------------------------------------------------------------------------
# 9. Legacy evidence keeps its versions through migration
# ---------------------------------------------------------------------------


def test_legacy_records_keep_versions_and_provenance_through_migration(tmp_path):
    path = tmp_path / "legacy.db"
    store = SqliteStore(path)
    for index in range(4):
        store.record_run(
            f"run-legacy-{index}",
            "ds",
            "INVESTIGATE",
            {"run_id": f"run-legacy-{index}", "status": "INVESTIGATE"},
            features=[0.0],
            feature_version=2,
            evidence_text="legacy evidence",
            created="2026-01-01T00:00:00+00:00",
        )
        store.record_outcome(
            f"run-legacy-{index}",
            "MISSING_STORES",
            confirmed=True,
            analyst="analyst-1",
            provenance="analyst",
            requires_investigation=True,
            created="2026-01-02T00:00:00+00:00",
        )
    store.close()
    migrated = SqliteStore(path)
    assert migrated.connection.execute("PRAGMA user_version").fetchone()[0] == 6
    migrated.close()

    records = records_from_store(path)
    assert len(records) == 4
    assert all(record.source == "analyst" for record in records)
    assert all(record.feature_version == 2 for record in records)
    assert all(record.metadata["text_version"] is None for record in records)
    assert all(record.feature_version != FEATURE_VERSION for record in records)

    from qc.training import train_decision_provider

    with pytest.raises(ValueError, match="feature version"):
        train_decision_provider(records, BASE, epochs=1)


# ---------------------------------------------------------------------------
# 10. Integrity results are policy findings
# ---------------------------------------------------------------------------


def test_hierarchy_and_lineage_results_become_policy_findings():
    result = run_qc(_two_week_source(), "V0002", "V0001", BASE)
    checks = {item["check"] for item in result.machine["findings"]}
    assert "lineage" in checks
    assert any(check.startswith("hierarchy") for check in checks)
