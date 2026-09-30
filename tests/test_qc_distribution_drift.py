"""Distribution drift behaviour.

The properties under test are the ones that decide whether this check is safe
to switch on: it must fire on a redistribution that preserves every invariant
the deterministic layer reads, it must stay quiet on a period drawn from the
same process, and it must derive its threshold from the scope's own history
rather than a constant. The ``None`` path matters as much as the firing path: a
scope that cannot be evaluated must report that, never zero drift.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qc.config import DatasetConfig
from qc.distribution_drift import (
    DRIFT_CHECK_VERSION,
    assess_distribution_drift,
    population_stability_index,
)


def _frame(values_by_week: dict[int, list[float]]) -> pd.DataFrame:
    rows = []
    for week, values in values_by_week.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "week": week,
                    "store_id": f"S{index % 4}",
                    "product_id": f"P{index}",
                    "dollar": float(value),
                }
            )
    return pd.DataFrame(rows)


def _steady(weeks: int = 15, count: int = 60) -> pd.DataFrame:
    return _frame(
        {week: [10 + (index % 7) for index in range(count)] for week in range(1, weeks + 1)}
    )


CONFIG = DatasetConfig()


def test_bin_edges_come_from_the_reference_only() -> None:
    # The comparison is directional on purpose: a threshold must not move
    # because the period under test moved, so edges are always taken from the
    # reference. Swapping the arguments changes the discretisation, so PSI here
    # is deliberately not symmetric and must not be asserted to be.
    left = np.random.default_rng(3).normal(0, 1, 500)
    right = np.random.default_rng(4).normal(1.5, 1, 500)
    forward = population_stability_index(left, right)
    backward = population_stability_index(right, left)
    assert forward is not None and forward > 0
    assert backward is not None and backward > 0
    # A reference drawn from the shifted sample must score higher against that
    # same shifted sample than the stable reference does, because the bin
    # structure is inherited from whichever sample is named first.
    same = population_stability_index(right, right)
    assert same is not None and same < forward


def test_entity_thresholds_are_derived_from_their_own_rows() -> None:
    # Regression: the baseline must be computed within the scope's own rows.
    # Deriving every scope's threshold from the pooled frame hands each entity
    # the national threshold, which is both wrong by construction and a
    # false-positive generator for narrow entities.
    rng = np.random.default_rng(21)
    rows = []
    for week in range(1, 16):
        for entity in ("WIDE", "NARROW"):
            count = 200 if entity == "WIDE" else 200
            for index in range(count):
                spread = 45.0 if entity == "WIDE" else 2.0
                base = 100.0 if entity == "WIDE" else 500.0
                rows.append(
                    {
                        "week": week,
                        "store_id": entity,
                        "product_id": f"{entity}-{index}",
                        "dollar": float(base + rng.normal(0, spread)),
                    }
                )
    frame = pd.DataFrame(rows)
    result = assess_distribution_drift(frame, CONFIG, 15, entity_columns=["store_id"])
    evaluated = {scope.scope: scope for scope in result.scopes if scope.psi is not None}
    assert "store_id:WIDE" in evaluated, sorted(evaluated)
    assert "store_id:NARROW" in evaluated, sorted(evaluated)
    wide = evaluated["store_id:WIDE"]
    narrow = evaluated["store_id:NARROW"]
    # A volatile scope must not inherit a stable scope's threshold.
    assert wide.threshold != narrow.threshold, (
        f"both scopes derived the same threshold {wide.threshold}"
    )
    # The narrow, quiet scope must not be alarmed by its own ordinary movement.
    assert not narrow.drifted, (
        f"narrow scope alarmed: psi={narrow.psi} thr={narrow.threshold}"
    )


def test_psi_grows_with_separation() -> None:
    rng = np.random.default_rng(5)
    base = rng.normal(0, 1, 800)
    scores = [
        population_stability_index(base, rng.normal(shift, 1, 800))
        for shift in (0.0, 0.5, 1.0, 2.0)
    ]
    assert scores == sorted(scores), scores
    assert scores[-1] > scores[0]


def test_psi_returns_none_for_tiny_samples() -> None:
    # Not evaluable must never be reported as zero drift.
    assert population_stability_index([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) is None


def test_psi_returns_none_for_constant_reference() -> None:
    assert population_stability_index([5.0] * 100, list(range(100))) is None


def test_psi_ignores_non_finite_values() -> None:
    sample = np.linspace(0, 1, 200)
    dirty = np.append(sample, [np.nan, np.inf, -np.inf])
    assert population_stability_index(sample, dirty) is not None
    assert math.isfinite(population_stability_index(sample, dirty))


def test_steady_series_reports_no_drift() -> None:
    result = assess_distribution_drift(_steady(), CONFIG, 15)
    assert result.evaluated
    assert not result.any_drift


def test_invariant_preserving_redistribution_is_detected() -> None:
    frame = _steady()
    period_rows = frame.index[frame["week"] == 15]
    donate, receive = period_rows[:30], period_rows[30:]
    total_before = float(frame.loc[period_rows, "dollar"].sum())
    moved = float(frame.loc[donate, "dollar"].sum()) * 0.5
    frame.loc[donate, "dollar"] -= moved / len(donate)
    frame.loc[receive, "dollar"] += moved / len(receive)
    # The invariant holds: every deterministic key is unchanged.
    assert frame.loc[period_rows, "dollar"].sum() == pytest.approx(total_before)
    assert len(frame) == len(_steady())
    assert frame["store_id"].nunique() == 4
    result = assess_distribution_drift(frame, CONFIG, 15)
    assert result.any_drift


def test_threshold_is_derived_per_scope_not_constant() -> None:
    # Two scopes with different intrinsic volatility, same check.
    rng = np.random.default_rng(7)
    stable = _steady()
    volatile = _frame(
        {
            week: list(rng.normal(100, 45, 60))
            for week in range(1, 16)
        }
    )
    stable_result = assess_distribution_drift(stable, CONFIG, 15)
    volatile_result = assess_distribution_drift(volatile, CONFIG, 15)
    stable_threshold = next(
        s.threshold for s in stable_result.scopes if s.scope_type == "national"
    )
    volatile_threshold = next(
        s.threshold for s in volatile_result.scopes if s.scope_type == "national"
    )
    assert volatile_threshold > stable_threshold
    assert volatile_threshold > 0.25, "volatile scope must not use a flat constant"


def test_short_reference_is_not_evaluated_rather_than_alarmed() -> None:
    frame = _steady(weeks=4)
    result = assess_distribution_drift(frame, CONFIG, 4)
    national = next(s for s in result.scopes if s.scope_type == "national")
    assert national.psi is None
    assert national.threshold is None
    assert not national.drifted
    assert "insufficient reference history" in national.detail


def test_missing_period_is_not_evaluated() -> None:
    result = assess_distribution_drift(_steady(), CONFIG, 999)
    assert not result.evaluated
    assert not result.scopes


def test_abs_floor_applies_to_very_stable_scopes() -> None:
    frame = _frame({week: list(np.linspace(0, 1, 60)) for week in range(1, 16)})
    result = assess_distribution_drift(frame, CONFIG, 15)
    national = next(s for s in result.scopes if s.scope_type == "national")
    assert national.threshold is not None
    assert national.threshold >= 0.01


def test_abs_floor_is_respected_even_with_quiet_history() -> None:
    frame = _steady()
    result = assess_distribution_drift(frame, CONFIG, 15, abs_floor=0.5)
    national = next(s for s in result.scopes if s.scope_type == "national")
    assert national.threshold == pytest.approx(0.5)


def test_entity_scopes_are_evaluated_when_large_enough() -> None:
    frame = _frame(
        {week: [10 + (index % 7) for index in range(200)] for week in range(1, 16)}
    )
    result = assess_distribution_drift(frame, CONFIG, 15, entity_columns=["store_id"])
    entity_scopes = [s for s in result.scopes if s.scope_type == "entity:store_id"]
    assert entity_scopes
    assert all(scope.scope.startswith("store_id:") for scope in entity_scopes)


def test_to_dict_is_serialisable_and_versioned() -> None:
    result = assess_distribution_drift(_steady(), CONFIG, 15)
    payload = result.to_dict()
    assert payload["version"] == DRIFT_CHECK_VERSION
    assert payload["period"] == 15
    assert payload["evaluated"] is True
    assert payload["any_drift"] is False
    assert isinstance(payload["scopes"], list)


def test_invalid_arguments_rejected() -> None:
    frame = _steady()
    with pytest.raises(ValueError, match="threshold_quantile"):
        assess_distribution_drift(frame, CONFIG, 15, threshold_quantile=0.0)
    with pytest.raises(ValueError, match="abs_floor"):
        assess_distribution_drift(frame, CONFIG, 15, abs_floor=-1.0)
    with pytest.raises(ValueError, match="reference_weeks"):
        assess_distribution_drift(frame, CONFIG, 15, reference_weeks=0)


def test_check_is_disabled_by_default() -> None:
    # A detector that is not qualified on real data must not run implicitly.
    assert CONFIG.distribution_drift_enabled is False


def test_enabled_drift_escalates_through_run_qc(tmp_path):
    """Regression: drift used to set a status that apply_policy then discarded.

    The value redistribution below preserves every row, key, null cell, every
    per-store weekly sum and therefore every reconciliation total; only the
    distribution drift check can see it, and with the check enabled its
    finding must reach the final status.
    """
    from qc.run import run_qc
    from qcgen.config import suite_config
    from qcgen.scenarios import build_scenario
    from qcgen.sources import ScenarioSource

    built = build_scenario(
        suite_config("small"), 0, tmp_path, "clean",
        ("source", "coded", "warehouse", "report"),
    )

    class Redistributed:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def read_fact(self, version, stage):
            frame = self.inner.read_fact(version, stage)
            if version == "V0002" and stage != "report":
                latest = frame["week"] == frame["week"].max()
                frame.loc[latest, "dollar"] = (
                    frame.loc[latest]
                    .groupby("store_id")["dollar"]
                    .transform(lambda values: np.sort(values.to_numpy()))
                    .to_numpy()
                )
            return frame

    base = replace(DatasetConfig(), temporal_enabled=False)
    source = Redistributed(ScenarioSource(built.directory))
    off = run_qc(source, "V0002", "V0001", base)
    on = run_qc(source, "V0002", "V0001", replace(base, distribution_drift_enabled=True))
    assert off.status == "PASS"
    assert on.distribution_drift is not None and on.distribution_drift.any_drift
    assert on.status == "INVESTIGATE"
    assert any(
        item["check"] == "distribution_drift" and item["outcome"] == "FAIL"
        for item in on.machine["findings"]
    )
