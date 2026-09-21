from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qc.config import DatasetConfig
from qc.lifecycle import NEW_RECENT, LifecycleEvent
from qc.temporal import (
    BaselineForecaster,
    CalibrationMap,
    CalibrationPools,
    CandidateSpec,
    backtest_forecaster,
    build_temporal_series,
    candidate_grid,
    new_period_adjustments,
    rolling_origin_errors,
    run_temporal_qc,
    select_candidate,
)
from qc.versions import build_version_pair

CONFIG = DatasetConfig()


def _fact(rows):
    return pd.DataFrame(rows)


def _noisy_values(n: int = 40, seed: int = 0, sigma: float = 3.0) -> dict[int, float]:
    rng = np.random.default_rng(seed)
    return {
        week: 100.0 + 10.0 * math.sin(2 * math.pi * week / 52.0)
        + float(rng.normal(0.0, sigma))
        for week in range(1, n + 1)
    }


def _national_frame(values: dict[int, float]) -> pd.DataFrame:
    return _fact(
        [{"week": week, "dollar": value, "units": 1} for week, value in values.items()]
    )


def _temporal_pair(previous_values, target_value):
    previous = _national_frame(previous_values)
    current = _national_frame({**previous_values, 41: target_value})
    pair = build_version_pair(
        previous, current, "V1", "V2", "warehouse", "warehouse", CONFIG
    )
    return previous, current, pair


def test_baseline_forecaster_quantiles_are_ordered():
    forecaster = BaselineForecaster(season=52)
    predictions = forecaster.predict([10, 11, 12, 13, 14], 1, [0.01, 0.5, 0.99])
    assert predictions.shape == (1, 3)
    assert predictions[0][0] <= predictions[0][1] <= predictions[0][2]


def test_build_temporal_series():
    fact = _fact(
        [
            {"week": 1, "banner_id": "B1", "commodity_id": "C1", "dollar": 10.0},
            {"week": 1, "banner_id": "B2", "commodity_id": "C2", "dollar": 5.0},
            {"week": 2, "banner_id": "B1", "commodity_id": "C1", "dollar": 11.0},
        ]
    )
    series = build_temporal_series(fact, CONFIG)
    assert set(series["series_id"]) == {
        "national",
        "banner_id:B1",
        "banner_id:B2",
        "commodity_id:C1",
        "commodity_id:C2",
    }
    national = series.loc[series["series_id"] == "national"].sort_values("week")
    assert national["value"].tolist() == [15.0, 11.0]


def test_calibration_map_percentile():
    calibration = CalibrationMap(residuals=[-2, -1, 0, 1, 2], coverage={"0.5": 0.5}, n=5)
    assert calibration.percentile(-3) == 0.0
    assert calibration.percentile(0) == pytest.approx(0.4)
    assert calibration.percentile(3) == 1.0
    assert CalibrationMap().percentile(0) is None


def test_temporal_flags_a_drop():
    values = _noisy_values(40, seed=1)
    previous, current, pair = _temporal_pair(values, values[40] - 30.0)
    result = run_temporal_qc(previous, current, pair, CONFIG)

    assert result is not None
    assert result.anomaly
    national = next(s for s in result.series if s.series_id == "national")
    assert national.actual == pytest.approx(values[40] - 30.0)
    assert national.adjusted_actual == pytest.approx(national.actual)
    assert national.anomaly
    assert "forecast_lower" in national.flags
    assert result.calibration["n"] > 0


def test_temporal_is_quiet_on_normal_target():
    values = _noisy_values(40, seed=1)
    previous, current, pair = _temporal_pair(values, values[40])
    result = run_temporal_qc(previous, current, pair, CONFIG)

    assert result is not None
    national = next(s for s in result.series if s.series_id == "national")
    assert not national.anomaly
    assert not result.anomaly


def test_temporal_adjusts_new_entity_target_contribution_once():
    values = _noisy_values(40, seed=2)
    previous = _fact(
        [
            {"week": week, "banner_id": "B1", "commodity_id": "C1", "dollar": value}
            for week, value in values.items()
        ]
    )
    target_value = values[40]
    new_rows = _fact(
        [
            {"week": 41, "store_id": "S9", "product_id": "P9", "banner_id": "B1", "commodity_id": "C1", "dollar": 30.0, "units": 1},
            {"week": 41, "store_id": "S1", "product_id": "P1", "banner_id": "B1", "commodity_id": "C1", "dollar": target_value, "units": 1},
        ]
    )
    current = pd.concat(
        [
            previous,
            new_rows,
        ],
        ignore_index=True,
    )
    pair = build_version_pair(
        previous, current, "V1", "V2", "warehouse", "warehouse", CONFIG
    )
    events = [
        LifecycleEvent("store", "S9", NEW_RECENT),
        LifecycleEvent("product", "P9", NEW_RECENT),
    ]
    adjustments = new_period_adjustments(current, events, pair, CONFIG)
    # Store-level adjustment wins; the product event must not double count.
    assert adjustments["national"] == pytest.approx(30.0)
    assert adjustments["banner_id:B1"] == pytest.approx(30.0)

    result = run_temporal_qc(previous, current, pair, CONFIG, events)
    national = next(s for s in result.series if s.series_id == "national")
    assert national.adjustment == pytest.approx(30.0)
    assert national.adjusted_actual == pytest.approx(target_value)
    assert not national.anomaly


def test_backtest_uses_previous_version_only():
    values = _noisy_values(40, seed=3)
    previous = _national_frame(values)
    series = build_temporal_series(previous, CONFIG)
    forecaster = BaselineForecaster(season=52)
    calibration = backtest_forecaster(series, CONFIG, forecaster)
    # Residuals come from history only; no extreme target value can appear.
    assert calibration.n > 0
    assert max(abs(value) for value in calibration.residuals) < 20.0


def _long_pair(n: int = 60, seed: int = 7, target_delta: float = 0.0):
    values = _noisy_values(n, seed=seed)
    previous = _national_frame(values)
    target = n + 1
    current = _national_frame({**values, target: values[n] + target_delta})
    pair = build_version_pair(
        previous, current, "V1", "V2", "warehouse", "warehouse", CONFIG
    )
    return previous, current, pair


def test_target_observation_never_enters_calibration_pool():
    quiet_previous, quiet_current, quiet_pair = _long_pair(target_delta=0.0)
    spike_previous, spike_current, spike_pair = _long_pair(target_delta=400.0)
    quiet = run_temporal_qc(quiet_previous, quiet_current, quiet_pair, CONFIG)
    spike = run_temporal_qc(spike_previous, spike_current, spike_pair, CONFIG)

    quiet_pools = {pool["key"]: pool["residuals"] for pool in quiet.calibration["pools"]}
    spike_pools = {pool["key"]: pool["residuals"] for pool in spike.calibration["pools"]}
    assert quiet_pools == spike_pools
    assert all(residuals for residuals in quiet_pools.values())

    national = next(item for item in spike.series if item.series_id == "national")
    pool = next(
        pool
        for pool in spike.calibration["pools"]
        if pool["key"] == national.calibration_pool
    )
    assert national.standardized_residual not in pool["residuals"]


def test_rolling_origin_forecast_precedes_evaluation():
    class RecordingForecaster:
        name = "recording"

        def __init__(self):
            self.lengths: list[int] = []

        def predict(self, history, horizon, quantiles):
            self.lengths.append(len(history))
            return BaselineForecaster().predict(history, horizon, quantiles)

    values = [100.0 + 5.0 * math.sin(week / 4.0) for week in range(1, 61)]
    forecaster = RecordingForecaster()
    spec = CandidateSpec("fixed", fixed="baseline")
    errors = rolling_origin_errors(
        list(range(1, 61)),
        values,
        spec,
        horizon=3,
        calendar=None,
        config=CONFIG,
        origins=5,
        forecaster=forecaster,
    )
    assert len(errors) == 5
    assert max(forecaster.lengths) <= len(values) - 3
    assert len(set(forecaster.lengths)) > 1


def test_missing_weeks_are_preserved_not_compressed():
    fact = _fact(
        [
            {"week": 1, "dollar": 10.0},
            {"week": 2, "dollar": 11.0},
            {"week": 4, "dollar": 13.0},
        ]
    )
    series = build_temporal_series(fact, CONFIG)
    national = series.loc[series["series_id"] == "national"].sort_values("week")
    assert national["week"].tolist() == [1, 2, 3, 4]
    assert national["missing"].tolist() == [False, False, True, False]
    assert math.isnan(national.loc[national["week"] == 3, "value"].iloc[0])


def test_run_reports_missing_history_explicitly():
    values = _noisy_values(60, seed=8)
    for week in (10, 11, 12):
        del values[week]
    previous = _national_frame(values)
    target = 61
    current = _national_frame({**values, target: values[60]})
    pair = build_version_pair(
        previous, current, "V1", "V2", "warehouse", "warehouse", CONFIG
    )
    result = run_temporal_qc(previous, current, pair, CONFIG)
    national = next(item for item in result.series if item.series_id == "national")
    assert national.history_missing == 3
    assert {10, 11, 12} <= set(national.missing_weeks)
    assert national.history_observed == len(values)


def test_calibration_pools_partition_by_key():
    pools = CalibrationPools("dollar", 1, 0.1)
    pools.add("national", "ridge:trend_only:p1", [(0.5, 5.0)])
    pools.add("banner_id", "ridge:trend_only:p1", [(0.7, 7.0)])
    pools.add("national", "seasonal_naive:v1", [(-0.2, -2.0)])
    other = CalibrationPools("units", 1, 0.1)
    other.add("national", "seasonal_naive:v1", [(0.1, 1.0)])

    assert set(pools.pools) == {
        "dollar|national|ridge:trend_only:p1|h1",
        "dollar|banner_id|ridge:trend_only:p1|h1",
        "dollar|national|seasonal_naive:v1|h1",
    }
    assert pools.get("banner_id", "ridge:trend_only:p1").n == 1
    assert pools.get("banner_id", "seasonal_naive:v1") is None
    assert other.get("national", "seasonal_naive:v1").n == 1
    assert pools.aggregate(revision="ridge:trend_only:p1").n == 2


def test_candidate_grid_requires_two_retail_years():
    short = candidate_grid(CONFIG, CONFIG.temporal_min_annual_history - 1)
    assert all(not spec.annual for spec in short)
    assert any(spec.reduced for spec in short)

    long = candidate_grid(CONFIG, CONFIG.temporal_min_annual_history)
    annual = [spec for spec in long if spec.annual]
    assert len(annual) == 9
    assert {(spec.fourier_order, spec.penalty) for spec in annual} == {
        (order, penalty)
        for order in (1, 2, 3)
        for penalty in (0.1, 1.0, 10.0)
    }


def test_selection_breaks_ties_toward_simpler_model():
    weeks = list(range(1, 121))
    values = [100.0] * len(weeks)
    outcome = select_candidate(
        weeks, values, 1, None, replace(CONFIG, forecaster="auto")
    )
    assert outcome.frozen
    assert outcome.spec.kind == "seasonal_naive"


def test_auto_selection_freezes_ridge_and_calibrates_intervals():
    rng = np.random.default_rng(3)
    values = {
        week: 500.0
        + 2.0 * week
        + 40.0 * math.sin(2.0 * math.pi * week / 52.0)
        + float(rng.normal(0.0, 4.0))
        for week in range(1, 121)
    }
    previous = _national_frame(values)
    current = _national_frame({**values, 121: values[120]})
    pair = build_version_pair(
        previous, current, "V1", "V2", "warehouse", "warehouse", CONFIG
    )
    config = replace(
        CONFIG,
        forecaster="auto",
        calendar_anchor_date="2019-01-07",
        calendar_events=(("christmas", "12-25", 1, 1), ("easter", "easter", 1, 1)),
    )
    result = run_temporal_qc(previous, current, pair, config)
    national = next(item for item in result.series if item.series_id == "national")

    assert national.selected_model.startswith("ridge")
    assert national.calibration_status == "OK"
    assert national.calibration_n > 0
    assert national.interval_lower is not None
    assert national.interval_upper is not None
    assert national.interval_width is not None and national.interval_width > 0
    assert national.interval_coverage is not None
    assert national.forecast_error is not None and national.forecast_error > 0
    assert national.history_weeks == 121
    assert national.history_observed == 120

    selection = result.calibration["selection"]["national"]
    assert selection["frozen"] is True
    assert selection["mode"] == "auto"
    assert len([c for c in selection["candidates"] if c["eligible"]]) >= 2


def test_sparse_leaf_uses_labelled_parent_share_fallback():
    previous = pd.DataFrame(
        [
            {"week": week, "banner_id": "b1", "dollar": 100.0, "units": 1}
            for week in range(1, 41)
        ]
        + [
            {"week": week, "banner_id": "b2", "dollar": 50.0, "units": 1}
            for week in range(31, 41)
        ]
    )
    current = pd.concat(
        [
            previous,
            pd.DataFrame(
                [
                    {"week": 41, "banner_id": "b1", "dollar": 100.0, "units": 1},
                    {"week": 41, "banner_id": "b2", "dollar": 50.0, "units": 1},
                ]
            ),
        ],
        ignore_index=True,
    )
    pair = build_version_pair(
        previous, current, "V1", "V2", "warehouse", "warehouse", CONFIG
    )
    result = run_temporal_qc(previous, current, pair, CONFIG)

    sparse = next(item for item in result.series if item.series_id == "banner_id:b2")
    assert sparse.support == "parent_share"
    assert sparse.calibration_status == "PARENT_SHARE_FALLBACK"
    assert sparse.calibration_n == 0
    assert sparse.interval_lower is None
    assert sparse.parent_series == "national"
    assert sparse.share == pytest.approx(500.0 / 1500.0)
    assert "banner_id:b2" not in result.unavailable_series
    parent_share = result.calibration["parent_share"]
    assert len(parent_share) == 1
    assert parent_share[0]["series_id"] == "banner_id:b2"
    assert parent_share[0]["parent_series"] == "national"
    assert parent_share[0]["share"] == pytest.approx(1.0 / 3.0)
    assert parent_share[0]["basis"] == "historical_share"


def test_sarimax_forecaster_is_available():
    pytest.importorskip("statsmodels")
    from qc.temporal import SarimaxForecaster

    values = list(_noisy_values(60, seed=4, sigma=1.0).values())
    forecaster = SarimaxForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    predictions = forecaster.predict(values, 1, (0.01, 0.5, 0.99))
    assert predictions.shape == (1, 3)
    assert predictions[0][0] <= predictions[0][1] <= predictions[0][2]


def test_get_forecaster_selects_sarimax():
    pytest.importorskip("statsmodels")
    from dataclasses import replace

    from qc.temporal import SarimaxForecaster, get_forecaster

    config = replace(CONFIG, forecaster="sarimax")
    assert isinstance(get_forecaster(config), SarimaxForecaster)
