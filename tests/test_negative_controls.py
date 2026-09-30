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


@negative_control("drift:redistribution_is_detected")
def _drift_detects_invariant_redistribution() -> None:
    # A value redistribution preserves every key the deterministic layer reads
    # (rows, nulls, duplicates, entity set, column sums). If the drift check
    # cannot see it, the check is vacuous.
    import pandas as pd

    from qc.config import DatasetConfig
    from qc.distribution_drift import assess_distribution_drift

    config = DatasetConfig()
    weeks = list(range(1, 15))
    rows = []
    for week in weeks:
        for index in range(60):
            rows.append(
                {
                    "week": week,
                    "store_id": f"S{index % 4}",
                    "product_id": f"P{index}",
                    "dollar": float(10 + (index % 7)),
                }
            )
    frame = pd.DataFrame(rows)
    target = max(weeks)
    period_rows = frame.index[frame["week"] == target]
    donate = period_rows[:30]
    receive = period_rows[30:]
    moved = float(frame.loc[donate, "dollar"].sum()) * 0.5
    frame.loc[donate, "dollar"] -= moved / len(donate)
    frame.loc[receive, "dollar"] += moved / len(receive)
    # The invariant the deterministic layer keys on must hold, or the control
    # is testing something else entirely.
    original_total = float(
        sum(10 + (index % 7) for index in range(60))
    )
    assert abs(float(frame.loc[period_rows, "dollar"].sum()) - original_total) < 1e-6
    result = assess_distribution_drift(frame, config, target)
    assert result.evaluated
    assert result.any_drift, "invariant-preserving redistribution went undetected"


@negative_control("drift:stable_period_does_not_fire")
def _drift_silent_on_stable_period() -> None:
    # The complement: a period drawn from the same process must stay quiet, or
    # the check is a false-positive generator.
    import pandas as pd

    from qc.config import DatasetConfig
    from qc.distribution_drift import assess_distribution_drift

    config = DatasetConfig()
    rows = []
    for week in range(1, 15):
        for index in range(60):
            rows.append(
                {
                    "week": week,
                    "store_id": f"S{index % 4}",
                    "product_id": f"P{index}",
                    "dollar": float(10 + (index % 7)),
                }
            )
    result = assess_distribution_drift(pd.DataFrame(rows), config, 14)
    assert result.evaluated
    assert not result.any_drift, "drift fired on an unchanged period"


@negative_control("drift:volatile_scope_uses_own_threshold")
def _drift_uses_scope_specific_threshold() -> None:
    # A global PSI constant is wrong because PSI is sample-size dependent: the
    # same number means different things in a volatile scope and a stable one.
    # This scope is deliberately noisy, so its ordinary week-to-week movement
    # is large. A period that is *typical for this scope* must stay quiet, which
    # only a per-scope threshold can deliver.
    import numpy as np
    import pandas as pd

    from qc.config import DatasetConfig
    from qc.distribution_drift import assess_distribution_drift

    config = DatasetConfig()
    rng = np.random.default_rng(11)
    rows = []
    for week in range(1, 15):
        for index in range(60):
            rows.append(
                {
                    "week": week,
                    "store_id": f"S{index % 4}",
                    "product_id": f"P{index}",
                    "dollar": float(100 + rng.normal(0, 45)),
                }
            )
    result = assess_distribution_drift(pd.DataFrame(rows), config, 14)
    assert result.evaluated
    national = next(
        scope for scope in result.scopes if scope.scope_type == "national"
    )
    # The scope's own history is genuinely volatile: the derived threshold must
    # reflect that, and must exceed the flat 0.25 a global constant would use.
    assert national.threshold is not None
    assert national.threshold > 0.25, (
        f"derived threshold {national.threshold} did not rise for a volatile scope"
    )
    assert not national.drifted, (
        "a period typical of a volatile scope was alarmed"
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


@negative_control("coverage:aggregate_drop_is_detected")
def _coverage_detects_aggregate_drop() -> None:
    # The gate's whole purpose is to catch a genuine loss of counterpart keys.
    # If it cannot, the gate is vacuous and must not be wired.
    import pandas as pd

    from qc.config import DatasetConfig
    from qc.coverage import assess_aggregate_coverage

    config = DatasetConfig(
        entity_key_columns=("store_id", "product_id"),
        entity_columns=("store_id", "product_id"),
    )
    rows = []
    for week in range(1, 16):
        for store in range(6):
            for product in range(6):
                rows.append(
                    {
                        "week": week,
                        "store_id": f"S{store}",
                        "product_id": f"P{product}",
                        "dollar": 1.0,
                    }
                )
    frame = pd.DataFrame(rows)
    baseline = assess_aggregate_coverage(frame, config, 15)
    assert baseline["product_id"].regressed is False

    # Drop two thirds of the stores carrying products in the appended period.
    period_rows = frame.index[frame["week"] == 15]
    doomed = [
        index
        for index in period_rows
        if frame.at[index, "store_id"] in ("S4", "S5")
    ]
    mutated = frame.drop(index=doomed)
    result = assess_aggregate_coverage(mutated, config, 15)
    assert result["product_id"].regressed, (
        "aggregate coverage drop went undetected"
    )


@negative_control("coverage:normal_period_does_not_fire")
def _coverage_quiet_on_normal_period() -> None:
    # The complement: an unchanged period must stay quiet, or the gate is a
    # false-positive generator rather than a gate.
    import pandas as pd

    from qc.config import DatasetConfig
    from qc.coverage import assess_aggregate_coverage

    config = DatasetConfig(
        entity_key_columns=("store_id", "product_id"),
        entity_columns=("store_id", "product_id"),
    )
    rows = []
    for week in range(1, 16):
        for store in range(5):
            for product in range(5):
                rows.append(
                    {
                        "week": week,
                        "store_id": f"S{store}",
                        "product_id": f"P{product}",
                        "dollar": 1.0,
                    }
                )
    result = assess_aggregate_coverage(pd.DataFrame(rows), config, 15)
    assert not any(entry.regressed for entry in result.values())
