from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qc import share_model
from qc.config import DatasetConfig
from qc.temporal import run_temporal_qc
from qc.versions import build_version_pair

LEVELS = np.array([0.01, 0.05, *np.round(np.arange(0.1, 0.91, 0.05), 2), 0.95, 0.99])
CONFIG = replace(
    DatasetConfig(), temporal_entity_columns=("commodity_id",), share_baseline="timestral"
)


def _normal_quantiles(median: float, sigma: float) -> np.ndarray:
    from statistics import NormalDist

    return np.array([median + sigma * NormalDist().inv_cdf(level) for level in LEVELS])


class _LastShareModel:
    """Stub: the last observed share with a 2% relative sigma, recording each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def forecast(self, context: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append((context.shape, horizon))
        rows = []
        for row in context:
            last = float(row[np.isfinite(row)][-1])
            rows.append(_normal_quantiles(last, 0.02 * last))
        return np.array(rows), LEVELS


def test_quantile_p_reads_the_side_the_observation_falls_on():
    quantiles = _normal_quantiles(0.3, 0.003)
    z, p, median, relative = share_model.quantile_p(0.3, quantiles, LEVELS)
    assert (z, p, median) == pytest.approx((0.0, 1.0, 0.3))
    assert relative == pytest.approx(0.01)
    low = share_model.quantile_p(0.3 - 3 * 0.003, quantiles, LEVELS)
    assert low[0] == pytest.approx(-3.0) and low[1] == pytest.approx(0.0027, abs=1e-4)
    # A forecast skewed upward judges a fall on its tighter lower side.
    skewed = quantiles + np.where(LEVELS > 0.5, 3 * (quantiles - 0.3), 0.0)
    assert share_model.quantile_p(0.3 - 0.006, skewed, LEVELS)[0] == pytest.approx(-0.006 / (0.003 * 1.0), rel=0.01)
    assert share_model.quantile_p(0.3, np.full(len(LEVELS), 0.3), LEVELS) is None
    assert share_model.quantile_p(math.nan, quantiles, LEVELS) is None


def test_share_rows_keep_gaps_on_the_parent_grid():
    rows, counts, last = share_model.share_rows(
        [([1, 2, 4], [1.0, 2.0, 4.0]), ([3], [3.0])], [1, 2, 3, 4], [10.0, 10.0, 10.0, 10.0]
    )
    assert last == 4 and counts == [3, 1]
    np.testing.assert_allclose(rows[0], [0.1, 0.2, np.nan, 0.4])
    np.testing.assert_allclose(rows[1], [np.nan, np.nan, 0.3, np.nan])
    assert share_model.share_rows([([1], [1.0])], [1], [0.0]) is None


def _three_leaf_pair(target_c2_factor: float):
    rng = np.random.default_rng(11)
    rows = []
    for week in range(1, 62):
        for commodity, share in (("C1", 0.5), ("C2", 0.3), ("C3", 0.2)):
            value = 1000.0 * share * (1.0 + 0.005 * rng.standard_normal())
            if week == 61 and commodity == "C2":
                value *= target_c2_factor
            rows.append({"week": week, "commodity_id": commodity, "dollar": value, "units": 1})
    current = pd.DataFrame(rows)
    previous = current.loc[current["week"] <= 60]
    pair = build_version_pair(previous, current, "V1", "V2", "warehouse", "warehouse", CONFIG)
    return previous, current, pair


def test_timestral_baseline_scores_every_leaf_in_one_batch(monkeypatch):
    stub = _LastShareModel()
    monkeypatch.setattr(share_model, "load_share_model", lambda name, revision="": stub)
    previous, current, pair = _three_leaf_pair(0.8)
    result = run_temporal_qc(previous, current, pair, CONFIG)
    leaves = {s.series_id: s for s in result.series if s.metric == "dollar" and s.level != "national"}
    assert {s.share_method for s in leaves.values()} == {"timestral"}
    # One forecast per measure, every leaf in it, one week ahead.
    assert ((3, 60), 1) in stub.calls and all(shape[0] == 3 for shape, _ in stub.calls)
    assert leaves["commodity_id:C2"].anomaly and leaves["commodity_id:C2"].share_impact < 0.0
    # Siblings' shares rise when C2 falls; C2 itself is the strongest signal.
    assert min(leaves.values(), key=lambda s: s.share_p).series_id == "commodity_id:C2"


def test_timestral_baseline_passes_an_unchanged_refresh(monkeypatch):
    monkeypatch.setattr(share_model, "load_share_model", lambda name, revision="": _LastShareModel())
    previous, current, pair = _three_leaf_pair(1.0)
    result = run_temporal_qc(previous, current, pair, CONFIG)
    assert not any(s.anomaly for s in result.series if s.level != "national")


def test_engine_baseline_never_loads_a_share_model(monkeypatch):
    def refuse(name, revision=""):
        raise AssertionError("the engine baseline loaded a share model")

    monkeypatch.setattr(share_model, "load_share_model", refuse)
    previous, current, pair = _three_leaf_pair(0.8)
    result = run_temporal_qc(previous, current, pair, replace(CONFIG, share_baseline="engine"))
    assert {s.share_method for s in result.series if s.share_p is not None} <= {"trailing", "seasonal"}


def test_share_baseline_is_validated():
    with pytest.raises(ValueError, match="share_baseline"):
        DatasetConfig(share_baseline="chronos")
    with pytest.raises(ValueError, match="share_model"):
        DatasetConfig(share_baseline="timestral", share_model="")


@pytest.mark.slow
def test_real_timestral_forecast_has_the_stub_shape():
    pytest.importorskip("timestral")
    model = share_model.load_share_model("mistralai/Timestral-5M")
    context = np.tile(np.linspace(0.29, 0.31, 60), (2, 1))
    context[1, 10:20] = np.nan
    quantiles, levels = model.forecast(context, 2)
    assert quantiles.shape == (2, len(levels)) and np.isfinite(quantiles).all()
    assert (np.diff(quantiles, axis=1) >= 0).all()
