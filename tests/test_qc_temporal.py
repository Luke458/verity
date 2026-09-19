from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qc.config import DatasetConfig
from qc.lifecycle import NEW_RECENT, LifecycleEvent
from qc.temporal import (
    BaselineForecaster,
    CalibrationMap,
    backtest_forecaster,
    build_temporal_series,
    new_period_adjustments,
    run_temporal_qc,
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
