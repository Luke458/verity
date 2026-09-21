"""Negative controls: every check must be able to fail.

A check that cannot be made to fail is not a check. Each registered control
feeds deliberately corrupted input to a check and asserts that it reports
failure. When a new check is added, register a control here; when a control
stops failing, the check has become vacuous.

The registry is intentionally non-empty from the start so the meta-test cannot
pass by having nothing to run.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from qc.attribution import AttributionResult, classify_run, explain_revision
from qc.config import DatasetConfig
from qc.contracts import validate_contracts
from qc.expectations import RatioExpectation, apply_expectations
from qc.lifecycle import (
    NEW_BACKFILL,
    TRUNCATED,
    LifecycleEvent,
    classify_entity_changes,
)
from qc.lineage import analyze_lineage
from qc.reconciliation import run_reconciliation
from qc.relationships import _benjamini_hochberg
from qc.versions import build_version_pair

NegativeControl = Callable[[], None]

NEGATIVE_CONTROLS: list[tuple[str, NegativeControl]] = []


def negative_control(name: str) -> Callable[[NegativeControl], NegativeControl]:
    def decorate(function: NegativeControl) -> NegativeControl:
        NEGATIVE_CONTROLS.append((name, function))
        return function

    return decorate


def _failed(result, name: str) -> bool:
    return any(check.name == name and not check.passed for check in result.checks)


@negative_control("contracts:required_columns")
def _required_columns_fails() -> None:
    frame = pd.DataFrame({"week": [1], "dollar": [1.0]})
    result = validate_contracts(frame, frame, DatasetConfig())
    assert result.status == "DATA_CONTRACT_FAILURE"
    assert _failed(result, "required_columns")


@negative_control("contracts:metric_dtypes")
def _metric_dtypes_fails() -> None:
    frame = pd.DataFrame({"week": [1], "dollar": ["not-a-number"], "units": [1]})
    result = validate_contracts(frame, frame, DatasetConfig())
    assert result.status == "DATA_CONTRACT_FAILURE"
    assert _failed(result, "metric_dtypes")


@negative_control("contracts:week_progression")
def _week_progression_fails() -> None:
    frame = pd.DataFrame(
        {"week": [1, 3], "dollar": [1.0, 2.0], "units": [1, 2]}
    )
    result = validate_contracts(frame, frame, DatasetConfig())
    assert result.status == "DATA_CONTRACT_FAILURE"
    assert _failed(result, "week_progression")


@negative_control("contracts:null_fraction")
def _null_fraction_fails() -> None:
    frame = pd.DataFrame(
        {"week": [1, 2], "dollar": [float("nan"), float("nan")], "units": [1, 2]}
    )
    result = validate_contracts(frame, frame, DatasetConfig())
    assert result.status == "DATA_CONTRACT_FAILURE"
    assert _failed(result, "null_fraction")


@negative_control("contracts:duplicate_fraction")
def _duplicate_fraction_fails() -> None:
    frame = pd.DataFrame(
        {
            "week": [1, 1, 2],
            "store_id": ["s1", "s1", "s2"],
            "dollar": [1.0, 1.0, 2.0],
            "units": [1, 1, 2],
        }
    )
    result = validate_contracts(frame, frame, DatasetConfig(entity_key_columns=("store_id",)))
    assert result.status == "DATA_CONTRACT_FAILURE"
    assert _failed(result, "duplicate_fraction")


@negative_control("reconciliation:mass_balance")
def _mass_balance_fails() -> None:
    base = pd.DataFrame(
        {
            "week": [1, 2],
            "dollar": [60.0, 40.0],
            "units": [1, 1],
            "store_id": ["s1", "s2"],
        }
    )
    report = base.copy()
    report["dollar"] = [30.0, 20.0]
    result = run_reconciliation(base, report, DatasetConfig(report_grain=()))
    assert result.status == "RECONCILIATION_FAILURE"
    assert _failed(result, "mass_balance:dollar")


@negative_control("reconciliation:mass_balance_per_key")
def _mass_balance_per_key_fails() -> None:
    # Equal grand totals, different weekly distribution. A global-sum check
    # passes this; the per-key comparison must fail it.
    base = pd.DataFrame(
        {"week": [1, 2], "dollar": [100.0, 0.0], "units": [1, 1]}
    )
    report = pd.DataFrame(
        {"week": [1, 2], "dollar": [50.0, 50.0], "units": [1, 1]}
    )
    result = run_reconciliation(base, report, DatasetConfig(report_grain=()))
    assert result.status == "RECONCILIATION_FAILURE"
    assert _failed(result, "mass_balance:dollar")


@negative_control("reconciliation:self_comparison_is_skipped")
def _self_comparison_is_skipped() -> None:
    frame = pd.DataFrame(
        {"week": [1, 2], "dollar": [1.0, 2.0], "units": [1, 2]}
    )
    result = run_reconciliation(frame, frame, DatasetConfig())
    assert result.status == "NOT_EVALUATED"
    assert result.checks
    assert all(check.status == "SKIPPED" for check in result.checks)


@negative_control("attribution:over_explained_blocks_pass")
def _over_explained_blocks_pass() -> None:
    attribution = AttributionResult(
        raw_delta=100.0,
        explained_delta=200.0,
        explained_fraction=1.0,
        over_explained=True,
        structural_events=[
            LifecycleEvent("store", "S1", NEW_BACKFILL, historical_value_added=200.0)
        ],
        matched_event_ids=["evt-1"],
    )
    events = list(attribution.structural_events)
    assert (
        classify_run(
            "PASS", events, attribution, DatasetConfig(), reconstruction_score=1.0
        )
        == "INVESTIGATE"
    )


@negative_control("attribution:offsetting_is_flagged")
def _offsetting_is_flagged() -> None:
    cube = pd.DataFrame(
        {
            "period": ["overlap"],
            "week": [1],
            "dollar_delta": [0.0],
            "dollar_previous": [100.0],
        }
    )
    events = [
        LifecycleEvent(
            "store",
            "S1",
            NEW_BACKFILL,
            historical_weeks_added=(1,),
            historical_value_added=50.0,
        ),
        LifecycleEvent(
            "store",
            "S2",
            TRUNCATED,
            historical_weeks_removed=(1,),
            historical_value_removed=50.0,
        ),
    ]
    result = explain_revision(cube, events, [], DatasetConfig())
    assert result.offsetting is True
    assert result.explained_added == 50.0
    assert result.explained_removed == 50.0
    # Unregistered offsetting structure must still be investigated.
    assert classify_run("PASS", events, result, DatasetConfig()) == "INVESTIGATE"


@negative_control("lifecycle:all_null_metric_is_not_presence")
def _all_null_metric_is_not_presence() -> None:
    previous = pd.DataFrame(
        {"week": [1], "store_id": ["S1"], "dollar": [10.0], "units": [1]}
    )
    current = pd.concat(
        [
            previous,
            pd.DataFrame(
                {
                    "week": [1],
                    "store_id": ["S9"],
                    "dollar": [float("nan")],
                    "units": [1],
                }
            ),
        ],
        ignore_index=True,
    )
    pair = build_version_pair(
        previous, current, "V1", "V2", "warehouse", "warehouse", DatasetConfig()
    )
    events = classify_entity_changes(previous, current, pair, DatasetConfig())
    assert not any(event.entity_id == "S9" for event in events)


class _StageSource:
    def __init__(self, frames):
        self.frames = frames

    def read_fact(self, version, stage):
        return self.frames[(version, stage)]


@negative_control("lineage:unmapped_stages_are_unknown")
def _unmapped_stages_are_unknown() -> None:
    frame = pd.DataFrame({"week": [1], "dollar": [1.0], "units": [1]})
    source = _StageSource({("V1", "mystery"): frame, ("V2", "mystery"): frame})
    pair = build_version_pair(
        frame, frame, "V1", "V2", "mystery", "mystery", DatasetConfig()
    )
    result = analyze_lineage(
        source, "V1", "V2", ["mystery"], DatasetConfig(), pair
    )
    assert result.status == "UNKNOWN"
    assert result.first_divergence is None


@negative_control("relationships:multiple_comparison_rejects_noise")
def _multiple_comparison_rejects_noise() -> None:
    assert not any(_benjamini_hochberg([0.4, 0.5, 0.6, 0.7, 0.8]))


@negative_control("expectations:no_ratio_never_matches")
def _no_ratio_never_matches() -> None:
    expectation = RatioExpectation(
        expectation_id="e1",
        dataset="d",
        metric="dollar_per_unit",
        min_ratio=0.0,
        max_ratio=10.0,
        week=6,
        approved_by="analyst",
    )
    report = apply_expectations(
        [{"metric": "dollar_per_unit", "week": 6}],
        [expectation],
        {"dataset": "d", "as_of": "2026-06-01"},
    )
    assert report["unexpected"]


@negative_control("expectations:incomparable_window_fails_closed")
def _incomparable_window_fails_closed() -> None:
    expectation = RatioExpectation(
        expectation_id="e1",
        dataset="d",
        metric="dollar_per_unit",
        min_ratio=0.0,
        max_ratio=10.0,
        effective_from="2026-01-01",
        effective_to="2026-12-31",
        approved_by="analyst",
    )
    report = apply_expectations(
        [{"metric": "dollar_per_unit", "week": 6, "ratio": 5.0}],
        [expectation],
        {"dataset": "d", "as_of": "30"},
    )
    assert report["unexpected"]
    assert any(
        "incomparable as_of window" in reason
        for entry in report["audit"]
        for reason in entry["reasons"]
    )


def test_negative_controls_are_registered() -> None:
    assert len(NEGATIVE_CONTROLS) >= 14, (
        "the negative-control registry shrank; every check must keep a control"
    )


def test_every_negative_control_fails() -> None:
    for name, control in NEGATIVE_CONTROLS:
        try:
            control()
        except AssertionError as error:  # pragma: no cover - diagnostic path
            raise AssertionError(f"negative control {name!r} did not fail") from error
