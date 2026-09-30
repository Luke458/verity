"""Scope-level coverage baseline.

The properties under test are the ones that make a coverage gate usable on real
sparse data: a scope's coverage must be measured against its *own* trailing
experience rather than a global constant, coverage must be one-sided so a rise is
never reported as a regression, and a scope without enough history must report
not-evaluated rather than guessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qc.config import DatasetConfig
from qc.coverage import (
    assess_scope_coverage,
)


def _frame(coverage_by_week: dict[int, int], seed: int = 3) -> pd.DataFrame:
    """week -> number of store/product pairs present that week."""
    rng = np.random.default_rng(seed)
    rows = []
    for week, count in coverage_by_week.items():
        for index in range(count):
            rows.append(
                {
                    "week": week,
                    "store_id": f"S{index % 3}",
                    "product_id": f"P{index}",
                    "dollar": float(10 + rng.integers(0, 5)),
                }
            )
    return pd.DataFrame(rows)


CONFIG = DatasetConfig(
    name="coverage",
    entity_key_columns=("store_id", "product_id"),
    entity_columns=("store_id", "product_id"),
)


def test_stable_coverage_does_not_regress() -> None:
    frame = _frame({week: 30 for week in range(1, 16)})
    result = assess_scope_coverage(frame, CONFIG, 15)
    assert result.evaluated
    assert not result.any_regression


def test_genuine_drop_is_detected() -> None:
    frame = _frame({**{week: 30 for week in range(1, 16)}, 15: 5})
    result = assess_scope_coverage(frame, CONFIG, 15)
    national = next(s for s in result.scopes if s.scope == "national")
    assert national.regressed
    assert national.covered == 5
    assert national.threshold is not None


def test_coverage_is_one_sided() -> None:
    # A rise is never a regression, however large.
    frame = _frame({**{week: 10 for week in range(1, 16)}, 15: 200})
    result = assess_scope_coverage(frame, CONFIG, 15)
    national = next(s for s in result.scopes if s.scope == "national")
    assert national.covered == 200
    assert not national.regressed


def test_threshold_comes_from_the_scope_own_history() -> None:
    # A scope with a much higher baseline must get a much higher floor; a global
    # constant would either swamp it or fire on it constantly.
    small = _frame({week: 5 for week in range(1, 16)}, seed=1)
    large = _frame({week: 500 for week in range(1, 16)}, seed=2)
    small_result = assess_scope_coverage(small, CONFIG, 15)
    large_result = assess_scope_coverage(large, CONFIG, 15)
    small_national = next(s for s in small_result.scopes if s.scope == "national")
    large_national = next(s for s in large_result.scopes if s.scope == "national")
    assert small_national.threshold != large_national.threshold
    assert large_national.threshold > small_national.threshold


def test_floor_tracks_the_scope_own_recent_level() -> None:
    # The floor is read off the scope's own trailing history. A scope that has
    # been running low is judged against that low level, not against a constant
    # borrowed from a busier scope. (A bimodal scope is the known weakness of a
    # lower-tail quantile and is recorded as such, not asserted away.)
    low = _frame({week: 6 for week in range(1, 16)}, seed=9)
    high = _frame({week: 90 for week in range(1, 16)}, seed=10)
    low_floor = next(
        s for s in assess_scope_coverage(low, CONFIG, 15).scopes
        if s.scope == "national"
    ).threshold
    high_floor = next(
        s for s in assess_scope_coverage(high, CONFIG, 15).scopes
        if s.scope == "national"
    ).threshold
    assert low_floor is not None and high_floor is not None
    assert high_floor > low_floor


def test_insufficient_history_is_not_evaluated() -> None:
    frame = _frame({1: 30, 2: 30, 3: 30, 4: 5})
    result = assess_scope_coverage(frame, CONFIG, 4)
    national = next(s for s in result.scopes if s.scope == "national")
    assert national.threshold is None
    assert not national.regressed
    assert "insufficient reference history" in national.detail


def test_missing_period_is_not_evaluated() -> None:
    frame = _frame({week: 30 for week in range(1, 16)})
    result = assess_scope_coverage(frame, CONFIG, 999)
    assert not result.evaluated
    assert not result.scopes


def test_empty_frame_is_safe() -> None:
    result = assess_scope_coverage(pd.DataFrame(), CONFIG, 5)
    assert not result.evaluated


def test_scope_coverage_measures_the_other_key() -> None:
    # Regression: measuring a store scope by its own column yields 1 for every
    # store. Coverage for a store must be the number of products it carried.
    frame = _frame({week: 30 for week in range(1, 16)})
    result = assess_scope_coverage(frame, CONFIG, 15)
    stores = [s for s in result.scopes if s.scope_type == "entity:store_id"]
    assert stores
    assert all(s.covered > 1 for s in stores), "store coverage collapsed to 1"


def test_absent_scope_in_a_period_counts_zero() -> None:
    # A store that stops trading entirely must read zero coverage, not NaN.
    rows = []
    for week in range(1, 16):
        # Store S0 is absent from the appended period only; it traded every
        # earlier period, so its absence is a real coverage drop.
        stores = ("S1",) if week == 15 else ("S0", "S1")
        for store in stores:
            for product in ("P1", "P2", "P3"):
                rows.append(
                    {"week": week, "store_id": store, "product_id": product, "dollar": 1.0}
                )
    frame = pd.DataFrame(rows)
    result = assess_scope_coverage(frame, CONFIG, 15)
    target = next(
        (s for s in result.scopes if s.scope == "store_id:S0"),
        None,
    )
    assert target is not None
    assert target.covered == 0
    assert target.regressed


def test_to_dict_is_serialisable_and_versioned() -> None:
    frame = _frame({week: 30 for week in range(1, 16)})
    payload = assess_scope_coverage(frame, CONFIG, 15).to_dict()
    assert payload["version"] == 1
    assert payload["period"] == 15
    assert payload["evaluated"] is True
    assert isinstance(payload["scopes"], list)


def test_invalid_arguments_rejected() -> None:
    frame = _frame({week: 30 for week in range(1, 16)})
    with pytest.raises(ValueError, match="quantile"):
        assess_scope_coverage(frame, CONFIG, 15, quantile=1.5)
    with pytest.raises(ValueError, match="reference_periods"):
        assess_scope_coverage(frame, CONFIG, 15, reference_periods=0)


def test_regressed_scope_types_available_for_gating() -> None:
    frame = _frame({**{week: 30 for week in range(1, 16)}, 15: 2})
    result = assess_scope_coverage(frame, CONFIG, 15)
    assert result.regressed_scope_types()
