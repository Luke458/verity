"""Regression acceptance for demonstrated failures.

Each test converts a previously demonstrated defect or required acceptance
scenario into a regression guard.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from qc.config import DatasetConfig
from qc.conformal import minimum_samples
from qc.evaluation import EvaluationCase, evaluation_gates, group_metrics
from qc.reference import ReferenceSpec
from qc.run import run_qc
from qc.store import SqliteStore
from qc.temporal import (
    NAIVE_KIND,
    CandidateSpec,
    ResidualPool,
    _rolling_evaluations,
    rolling_origin_errors,
    select_candidate,
)
from qc.weekly import run_weekly

STAGES = ("warehouse", "report")
CONFIG = DatasetConfig(
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


def _frame(values: dict[int, float], weeks: range) -> pd.DataFrame:
    rows = [
        {
            "week": week,
            "store_id": store,
            "product_id": "P1",
            "dollar": values[week],
            "units": values[week] / 10.0,
        }
        for week in weeks
        for store in ("S1", "S2")
    ]
    fact = pd.DataFrame(rows)
    return fact


def _two_week_source(first_factor: float = 0.4) -> FrameSource:
    values = {week: _periodic(week) for week in range(1, 123)}
    values[121] = first_factor * _periodic(121)
    values[122] = _periodic(122)
    frames = {}
    for version, weeks in (("V0001", range(1, 121)), ("V0002", range(1, 123))):
        fact = _frame(values, weeks)
        report = fact.groupby("week", as_index=False)[["dollar", "units"]].sum()
        frames[version, "warehouse"] = fact
        frames[version, "report"] = report
    return FrameSource(frames)


def _single_week_source(first_factor: float = 0.4) -> FrameSource:
    values = {week: _periodic(week) for week in range(1, 122)}
    values[121] = first_factor * _periodic(121)
    frames = {}
    for version, weeks in (("V0001", range(1, 121)), ("V0002", range(1, 122))):
        fact = _frame(values, weeks)
        report = fact.groupby("week", as_index=False)[["dollar", "units"]].sum()
        frames[version, "warehouse"] = fact
        frames[version, "report"] = report
    return FrameSource(frames)


def test_doubled_reference_totals_force_review():
    source = _two_week_source()
    reference = source.read_fact("V0002", "report").copy()
    reference["dollar"] = reference["dollar"] * 2.0
    spec = ReferenceSpec("national", CONFIG.name, ("dollar",))
    result = run_qc(
        source,
        "V0002",
        "V0001",
        CONFIG,
        reference_frame=reference,
        reference_spec=spec,
    )
    assert result.machine["reference"]["status"] == "MISMATCH"
    reference_findings = [
        item
        for item in result.machine["findings"]
        if item["check"] == "reference"
        and item["outcome"] == "FAIL"
    ]
    assert reference_findings
    assert all(not item["approval_ids"] for item in reference_findings)
    assert result.status in ("INVESTIGATE", "DATA_CONTRACT_FAILURE")
    assert result.machine["requires_investigation"] is True


# ---------------------------------------------------------------------------
# 2. 60% drop in the first of two appended weeks
# ---------------------------------------------------------------------------


def test_first_of_two_appended_weeks_is_detected_and_both_assessed():
    result = run_qc(_two_week_source(first_factor=0.4), "V0002", "V0001", CONFIG)
    assert result.temporal is not None
    assert result.temporal.targets == [121, 122]
    national = {
        item.target_week: item
        for item in result.temporal.series
        if item.series_id == "national"
    }
    assert set(national) == {121, 122}
    assert national[121].anomaly is True
    assert national[121].residual < 0
    assert national[122].anomaly is False
    assert national[121].horizon == 1
    assert national[122].horizon == 2
    assert national[122].training_endpoint_week == 120
    assert national[122].forecast_origin_week == 120
    findings = [
        item
        for item in result.machine["findings"]
        if item["check"] == "temporal"
    ]
    first = next(
        item for item in findings if item["scope"] == "national" and item["period"] == 121
    )
    second = next(
        item for item in findings if item["scope"] == "national" and item["period"] == 122
    )
    assert first["outcome"] == "FAIL"
    assert second["outcome"] in ("PASS", "FAIL")
    # Period-specific identities: the two findings are distinct.
    assert first["finding_id"] != second["finding_id"]


# ---------------------------------------------------------------------------
# 3. Interval isolation
# ---------------------------------------------------------------------------


def test_large_sibling_raw_residuals_do_not_inflate_target_interval():
    z_errors = [0.4, -0.6, 0.2, -0.3, 0.5, -0.1, 0.3, -0.4, 0.6, -0.2]
    small = ResidualPool("dollar", "national", "ridge:v1", 1)
    large = ResidualPool("dollar", "national", "ridge:v1", 1)
    for z in z_errors:
        small.residuals.append(z)
        small.raw_residuals.append(z * 1.0)
        large.residuals.append(z)
        large.raw_residuals.append(z * 1_000_000.0)
    interval_small = small.conformal(center=100.0, alpha=0.1, scale=2.0, series_errors=z_errors)
    interval_large = large.conformal(center=100.0, alpha=0.1, scale=2.0, series_errors=z_errors)
    assert interval_small.width is not None
    assert interval_large.width == pytest.approx(interval_small.width)
    assert interval_large.width < 1000.0


# ---------------------------------------------------------------------------
# 5/6. Frozen recurrence inputs and identity
# ---------------------------------------------------------------------------


def _run_payload(run_id: str, previous: str, current: str, net: float = 5.0) -> dict:
    return {
        "run_id": run_id,
        "status": "INVESTIGATE",
        "version_pair": {
            "previous_id": previous,
            "current_id": current,
            "current_max_week": int(current.lstrip("V")),
        },
        "findings": [],
        "ledger": {"net_unexplained": net},
        "materiality_threshold": 1.0,
        "historical_revision": {"unexplained_delta": net},
    }


def _seed_run(store: SqliteStore, run_id: str, previous: str, current: str, created: str):
    store.record_run(
        run_id,
        "recurrence",
        "INVESTIGATE",
        _run_payload(run_id, previous, current),
        created=created,
    )


def test_future_dated_recurrence_is_excluded_at_the_cutoff(tmp_path):
    store = SqliteStore(tmp_path / "store.db")
    _seed_run(store, "old", "V0001", "V0002", "2026-01-10T00:00:00+00:00")
    _seed_run(store, "future", "V0002", "V0003", "2027-01-10T00:00:00+00:00")
    entries = store.recurrence_inputs(
        "recurrence",
        cutoff="2026-06-01T00:00:00+00:00",
        window=3,
        current_version="V0003",
    )
    assert [entry["run_id"] for entry in entries] == ["old"]
    store.close()


def test_recurrence_inputs_deduplicate_retries_and_latest_revision_wins(tmp_path):
    store = SqliteStore(tmp_path / "store.db")
    _seed_run(store, "run-a", "V0001", "V0002", "2026-01-10T00:00:00+00:00")
    _seed_run(store, "run-a-retry", "V0001", "V0002", "2026-01-11T00:00:00+00:00")
    _seed_run(store, "run-b", "V0002", "V0003", "2026-01-12T00:00:00+00:00")
    entries = store.recurrence_inputs(
        "recurrence",
        cutoff="2026-02-01T00:00:00+00:00",
        window=3,
        current_version="V0004",
    )
    assert [entry["run_id"] for entry in entries] == ["run-a-retry", "run-b"]
    store.close()


def _write_delta_versions(path: Path, targets: tuple[int, ...]) -> None:
    pytest.importorskip("deltalake")
    from deltalake import write_deltalake

    for index, target in enumerate(targets):
        rows = []
        for week in range(1, target + 1):
            value = _periodic(week)
            if week == target:
                value = _periodic(target - 1)
            for store in ("S1", "S2"):
                rows.append(
                    {
                        "week": week,
                        "store_id": store,
                        "product_id": "P1",
                        "dollar": value,
                        "units": 10.0,
                    }
                )
        frame = pd.DataFrame(rows)
        mode = "overwrite" if index else "error"
        write_deltalake(str(path), frame, mode=mode)


def test_future_dated_refresh_cannot_change_a_historical_assessment(tmp_path):
    pytest.importorskip("deltalake")
    path = tmp_path / "fact"
    _write_delta_versions(path, (30, 31, 32))
    config = replace(DatasetConfig(), name="recurrence-identity")
    first = run_weekly(
        str(path),
        config=config,
        store_path=tmp_path / "store.db",
        out_root=tmp_path / "weekly",
    )
    assert first.skipped is False
    assert first.assessment_id

    with SqliteStore(tmp_path / "store.db") as store:
        _seed_run(
            store,
            "future-refresh",
            "V0001",
            "V0002",
            "2099-01-01T00:00:00+00:00",
        )
    again = run_weekly(
        str(path),
        config=config,
        store_path=tmp_path / "store.db",
        out_root=tmp_path / "weekly",
    )
    assert again.skipped is True
    assert again.assessment_id == first.assessment_id
    assert again.status == first.status


def test_eligible_recurrence_evidence_change_creates_a_new_assessment(tmp_path):
    pytest.importorskip("deltalake")
    path = tmp_path / "fact"
    _write_delta_versions(path, (30, 31, 32))
    config = replace(DatasetConfig(), name="recurrence-change")
    first = run_weekly(
        str(path),
        config=config,
        store_path=tmp_path / "store.db",
        out_root=tmp_path / "weekly",
    )
    assert first.skipped is False

    with SqliteStore(tmp_path / "store.db") as store:
        store.record_run(
            "prior-refresh",
            "recurrence-change",
            "INVESTIGATE",
            _run_payload("prior-refresh", "0", "1"),
            created="2020-01-01T00:00:00+00:00",
        )
    changed = run_weekly(
        str(path),
        config=config,
        store_path=tmp_path / "store.db",
        out_root=tmp_path / "weekly",
    )
    assert changed.assessment_id != first.assessment_id
    assert changed.skipped is False


# ---------------------------------------------------------------------------
# 7. Repeated incident copies / no controls -> insufficient evidence
# ---------------------------------------------------------------------------


def test_repeated_incident_copies_without_controls_are_insufficient():
    cases = [
        EvaluationCase(
            case_id=f"copy-{index}",
            family="missing_stores",
            group="incident-1",
            actionable=True,
            verifier_status="INVESTIGATE",
            challenger_status="INVESTIGATE",
            effective_status="INVESTIGATE",
            verifier_review=True,
            challenger_review=True,
            effective_review=True,
            expected_review=True,
        )
        for index in range(12)
    ]
    gates = evaluation_gates(cases)
    assert gates["status"] == "INSUFFICIENT_EVIDENCE"
    assert gates["gates"]["detection_rate"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert gates["gates"]["false_positive_rate"]["status"] == "INSUFFICIENT_EVIDENCE"
    grouped = group_metrics(cases)
    assert grouped["actionable_groups"] == 1
    assert grouped["control_groups"] == 0


# ---------------------------------------------------------------------------
# 8. Disjoint selection/calibration partitions and correct horizons
# ---------------------------------------------------------------------------


def test_selection_and_calibration_partitions_are_disjoint_with_exact_horizons():
    weeks = list(range(1, 61))
    values = [float(_periodic(week)) for week in weeks]
    config = replace(CONFIG, temporal_calibration_origins=6)
    horizon = 2
    eligible = _rolling_evaluations(
        list(zip(weeks, values, strict=True)), horizon, config, None
    )
    selection = eligible[:-6]
    calibration = eligible[-6:]
    selection_targets = {index for index, _ in selection}
    calibration_targets = {index for index, _ in calibration}
    assert selection_targets
    assert calibration_targets
    assert selection_targets.isdisjoint(calibration_targets)
    for target_index, endpoint_index in eligible:
        endpoint_week = weeks[endpoint_index]
        target_week = weeks[target_index]
        assert target_week - endpoint_week == horizon
    spec = CandidateSpec(NAIVE_KIND)
    selection_errors = rolling_origin_errors(
        weeks,
        values,
        spec,
        horizon,
        None,
        config,
        reserved_tail=6,
        evaluation="selection",
    )
    calibration_errors = rolling_origin_errors(
        weeks,
        values,
        spec,
        horizon,
        None,
        config,
        reserved_tail=6,
        evaluation="calibration",
    )
    assert len(selection_errors) == len(selection)
    assert len(calibration_errors) == len(calibration)
    assert len(calibration_errors) >= minimum_samples(0.1) - 3


def test_insufficient_partitions_return_insufficient_evidence():
    weeks = list(range(1, 31))
    values = [float(_periodic(week)) for week in weeks]
    config = replace(
        CONFIG,
        forecaster="auto",
        temporal_backtest_origins=5,
        temporal_calibration_origins=20,
    )
    outcome = select_candidate(weeks, values, 1, None, config)
    assert outcome.mode == "insufficient"
    assert all(not item.eligible for item in outcome.evaluations)


def test_initial_assessment_without_predecessors_is_a_cold_start():
    result = run_qc(_single_week_source(), "V0002", "V0001", CONFIG)
    assert result.machine["recurrence_status"] == "COLD_START"
    assert "recurrence" not in result.machine


def test_tampered_recurrence_predecessor_is_rejected(tmp_path):
    store = SqliteStore(tmp_path / "store.db")
    _seed_run(store, "run-a", "1", "2", "2026-01-10T00:00:00+00:00")
    entries = store.recurrence_inputs(
        "recurrence",
        cutoff="2026-02-01T00:00:00+00:00",
        window=3,
        current_version="3",
    )
    assert len(entries) == 1
    store.connection.execute(
        "UPDATE runs SET payload = ? WHERE run_id = 'run-a'",
        (json.dumps(_run_payload("run-a", "1", "2", net=999.0)),),
    )
    store.connection.commit()
    with pytest.raises(ValueError, match="changed"):
        store.recurrence_refreshes(entries)
    store.close()
