"""Regression acceptance for multi-measure assessment,
the shared evaluation contract and integrity findings.

Each test runs through the real orchestration paths where a defect was
observed.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd
import pytest

from qc.cohort import CohortCase, _common_gate_results, _evaluation_cases
from qc.config import DatasetConfig
from qc.evaluation import EvaluationCase, evaluation_gates
from qc.run import run_qc

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
    stock = next(
        item
        for item in result.temporal.series
        if item.metric == "stock" and item.series_id == "national"
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
    shared = evaluation_gates(mapped, minimum_detection_rate=0.9)["gates"]
    for name in ("detection_rate", "false_positive_rate"):
        assert common["gates"][name] == shared[name]
    # The false-clearance bound is reported by the cohort, not gated.
    assert common["false_clearance_upper_bound"]["incidents"] == 12
    assert [check["gate"] for check in checks] == [
        "min_detection_rate",
        "max_false_positive_rate",
    ]


# ---------------------------------------------------------------------------
# 9. Integrity results are policy findings
# ---------------------------------------------------------------------------


def test_hierarchy_and_lineage_results_become_policy_findings():
    result = run_qc(_two_week_source(), "V0002", "V0001", BASE)
    checks = {item["check"] for item in result.machine["findings"]}
    assert "lineage" in checks
    assert any(check.startswith("hierarchy") for check in checks)
