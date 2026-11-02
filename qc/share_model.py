"""Pretrained share baseline: a Timestral forecast of each leaf's share of its parent.

Experimental, selected with ``share_baseline: timestral`` (docs/timestral-share.md).
Every leaf of one measure and period is forecast in one batch from its share
history on the parent's week grid (a missing week stays NaN, never compressed),
and the observed share is scored against the forecast quantiles. Timestral and
torch are an optional dependency (``pip install -e '.[timestral]'``), imported
only when this baseline is configured; the default engine never loads them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from statistics import NormalDist
from typing import Any, Protocol

import numpy as np

_NORMAL = NormalDist()
_Z90 = _NORMAL.inv_cdf(0.9)


class ShareModel(Protocol):
    def forecast(self, context: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """``context`` is (series, weeks) with NaN for missing weeks.

        Returns the quantiles at the last horizon step, (series, levels), and
        the ascending quantile levels.
        """
        ...


@dataclass(frozen=True)
class TimestralShareModel:
    forecaster: Any

    def forecast(self, context: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        result = self.forecaster.forecast(np.asarray(context, dtype=float), prediction_length=horizon)
        quantiles = np.asarray(result.quantiles, dtype=float)[:, 0, -1, :]
        return quantiles, np.asarray(result.quantile_levels, dtype=float)


@lru_cache(maxsize=4)
def load_share_model(name: str, revision: str = "") -> ShareModel:
    try:
        from timestral import TimestralForecaster
    except ImportError as error:
        raise ImportError(
            "share_baseline timestral needs the optional dependency: pip install -e '.[timestral]'"
        ) from error
    forecaster = TimestralForecaster.from_pretrained(name, device="cpu", revision=revision or None)
    return TimestralShareModel(forecaster)


def share_rows(
    leaves: Sequence[tuple[Sequence[int], Sequence[float]]],
    parent_weeks: Sequence[int],
    parent_values: Sequence[float],
) -> tuple[np.ndarray, list[int], int] | None:
    """Each leaf's share on the parent's dense week grid, its observed count and the grid's last week.

    A week where the leaf or the parent is unobserved (or the parent is not
    positive) is NaN, so the model sees the gap instead of a shorter history.
    """
    parent = {
        int(week): float(value)
        for week, value in zip(parent_weeks, parent_values, strict=True)
        if math.isfinite(float(value)) and float(value) > 0.0
    }
    if not parent:
        return None
    grid = range(min(parent), max(parent) + 1)
    rows = np.full((len(leaves), len(grid)), np.nan)
    counts: list[int] = []
    for index, (weeks, values) in enumerate(leaves):
        for week, value in zip(weeks, values, strict=True):
            week = int(week)
            if week in parent and math.isfinite(float(value)):
                rows[index, week - grid.start] = float(value) / parent[week]
        counts.append(int(np.isfinite(rows[index]).sum()))
    return rows, counts, grid.stop - 1


def quantile_p(
    observed: float, quantiles: np.ndarray, levels: np.ndarray
) -> tuple[float, float, float, float] | None:
    """(z, two-sided p, median, relative spread) of ``observed`` against forecast quantiles.

    The scale is the 10-50 or 50-90 quantile gap on the side the observation
    falls, read as a normal sigma, so a skewed forecast is judged on its own
    side; the tail beyond the outermost quantiles is the normal tail at that
    scale. The relative spread (80% half-width as a sigma, over the median) is
    the same relative scale the ``trailing`` baseline reports. Returns None for
    a degenerate forecast.
    """
    quantiles = np.asarray(quantiles, dtype=float)
    if not (math.isfinite(observed) and np.isfinite(quantiles).all()):
        return None
    median = float(np.interp(0.5, levels, quantiles))
    lower = float(np.interp(0.1, levels, quantiles))
    upper = float(np.interp(0.9, levels, quantiles))
    if median <= 0.0 or upper <= lower:
        return None
    gap = upper - median if observed >= median else median - lower
    sigma = max(gap, (upper - lower) / 2.0 * 1e-3) / _Z90
    z = (observed - median) / sigma
    p = 2.0 * (1.0 - _NORMAL.cdf(abs(z)))
    relative = (upper - lower) / (2.0 * _Z90) / median
    return z, min(1.0, max(0.0, p)), median, relative
