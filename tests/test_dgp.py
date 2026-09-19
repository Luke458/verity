from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from qcgen.dgp import generate_history, week_end_dates
from qcgen.universe import generate_universe


def test_week_end_dates_are_saturdays():
    dates = week_end_dates("2024-01-06", 5)
    assert len(dates) == 5
    assert all(date.weekday() == 5 for date in dates)


def test_week_end_dates_reject_non_saturday():
    with pytest.raises(ValueError):
        week_end_dates("2024-01-01", 2)


def test_history_shape_and_invariants(tiny_config):
    universe = generate_universe(tiny_config.universe, np.random.default_rng(3), 30)
    fact = generate_history(universe, tiny_config.history, np.random.default_rng(4))

    assert set(fact.columns) == {
        "week",
        "store_id",
        "product_id",
        "units",
        "dollar",
        "scripts",
        "stock",
    }
    assert fact["week"].min() == 1
    assert fact["week"].max() == tiny_config.history.n_weeks
    assert (fact["units"] >= 0).all()
    assert (fact["dollar"] >= 0).all()
    assert (fact["stock"] >= 0).all()

    merged = fact.merge(universe.stores[["store_id", "open_week"]], on="store_id")
    assert (merged["week"] >= merged["open_week"]).all()

    script_products = set(
        universe.products.loc[universe.products["is_script"], "product_id"]
    )
    non_script = fact.loc[~fact["product_id"].isin(script_products), "scripts"]
    assert (non_script == 0).all()
    script_rows = fact.loc[fact["product_id"].isin(script_products), "scripts"]
    assert (script_rows > 0).any()


def test_history_is_deterministic(tiny_config):
    universe = generate_universe(tiny_config.universe, np.random.default_rng(7), 30)
    first = generate_history(universe, tiny_config.history, np.random.default_rng(8))
    second = generate_history(universe, tiny_config.history, np.random.default_rng(8))
    assert first.equals(second)


def test_christmas_week_is_lifted(tiny_config):
    history = replace(
        tiny_config.history, n_weeks=60, start_week="2024-01-06"
    )
    universe = generate_universe(tiny_config.universe, np.random.default_rng(9), 60)
    fact = generate_history(universe, history, np.random.default_rng(10))

    dates = week_end_dates(history.start_week, history.n_weeks)
    starts = dates - pd.Timedelta(days=6)
    christmas = [
        i + 1
        for i in range(len(dates))
        if starts[i].date() <= date(2024, 12, 25) <= dates[i].date()
    ]
    assert len(christmas) == 1

    weekly = fact.groupby("week")["units"].mean()
    baseline = weekly.drop(index=christmas).mean()
    assert weekly.loc[christmas[0]] > 1.15 * baseline
