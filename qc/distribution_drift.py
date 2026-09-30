"""Distribution drift over the appended period.

The deterministic layer keys on structure: row counts, null cells, duplicate
keys, the entity-set hash, metric sums and distinct counts. A value
redistribution can preserve every one of those while moving the joint
distribution substantially, and no arithmetic on totals or keys will see it.
Per-entity temporal series may also stay inside their own materiality floors,
because the fault can be spread thinly enough that no single entity moves far.

This module adds the missing shape. For a scope and metric it compares the
appended period's value distribution against a trailing reference window of the
same scope, using the population stability index, and it decides whether the
observed movement is larger than that scope's own history of ordinary
week-to-week movement.

The threshold is the load-bearing part. A single global PSI cut-off is wrong:
PSI is a sample-size-dependent statistic whose distribution depends on the bin
count and on both sample sizes, so a fixed number means different things at
five hundred rows and at five hundred thousand. The threshold here is therefore
derived per scope from the reference window's *own* consecutive-week PSI
distribution. A scope that is normally volatile is only alarmed when the
appended period is more volatile than it has ever been; a scope that is
normally stable is alarmed much earlier.

Nothing here is a clearance. A drift finding is evidence that the distribution
moved, not evidence of why, and it never authorises a statistical clearance or
overrides a deterministic finding.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import DatasetConfig

DRIFT_CHECK_VERSION = 1

DEFAULT_BINS = 10
DEFAULT_REFERENCE_WEEKS = 8
# Floor for a scope whose reference history is too short to estimate a
# threshold. A scope with no measurable own-variance must not be alarmed by a
# single ratio, so it stays silent rather than guessing.
DEFAULT_ABS_FLOOR = 0.01
DEFAULT_THRESHOLD_QUANTILE = 0.95
DEFAULT_MIN_REFERENCE_WEEKS = 4
DEFAULT_MIN_OBSERVATIONS = 50


def _edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Quantile edges from the reference, with open outer bounds.

    Edges come from the reference only: a threshold must not move because the
    period under test moved.
    """
    quantiles = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(quantiles)
    if edges.size < 2:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(
    reference: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    bins: int = DEFAULT_BINS,
) -> float | None:
    """PSI between two samples, or ``None`` when it cannot be computed.

    Returns ``None`` rather than a number whenever the comparison is not
    meaningful - too few observations, no finite values, or a reference that
    collapses to a single value. A caller must treat ``None`` as *not
    evaluated*, never as zero drift.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size < DEFAULT_MIN_OBSERVATIONS or cur.size < DEFAULT_MIN_OBSERVATIONS:
        return None
    if np.unique(ref).size < 2:
        # A degenerate reference (an all-zero or otherwise constant window)
        # cannot support a distribution comparison. Returning 0.0 here would
        # report "no drift" for a period that may be arbitrarily far away, so
        # this is a hard not-evaluated rather than a zero.
        return None
    limits = _edges(ref, bins)
    ref_counts = np.histogram(ref, limits)[0].astype(np.float64)
    cur_counts = np.histogram(cur, limits)[0].astype(np.float64)
    ref_share = ref_counts / ref.size
    cur_share = cur_counts / cur.size
    # Additive smoothing keeps an empty bin finite; without it a single empty
    # bin yields an infinite score that no threshold can be compared against.
    smoothing = 1.0 / (2.0 * ref.size)
    ref_share = ref_share + smoothing
    cur_share = cur_share + smoothing
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def _consecutive_psi(
    frame: pd.DataFrame, week_column: str, metric: str, bins: int
) -> list[float]:
    """PSI of each week in the reference against the week before it."""
    weekly = frame.groupby(week_column, observed=True)[metric].apply(
        lambda values: values.to_numpy(dtype=np.float64)
    )
    weeks = sorted(weekly.index)
    scores: list[float] = []
    for previous, current in zip(weeks, weeks[1:]):
        score = population_stability_index(weekly[previous], weekly[current], bins)
        if score is not None and math.isfinite(score):
            scores.append(score)
    return scores


@dataclass(frozen=True)
class DriftScope:
    """One evaluated scope, with the evidence for its own decision."""

    scope: str
    scope_type: str
    metric: str
    period: int
    psi: float | None
    threshold: float | None
    drifted: bool
    reference_weeks: int
    reference_observations: int
    period_observations: int
    baseline_psi: float | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "scope_type": self.scope_type,
            "metric": self.metric,
            "period": self.period,
            "psi": self.psi,
            "threshold": self.threshold,
            "drifted": self.drifted,
            "reference_weeks": self.reference_weeks,
            "reference_observations": self.reference_observations,
            "period_observations": self.period_observations,
            "baseline_psi": self.baseline_psi,
            "detail": self.detail,
        }


@dataclass
class DistributionDriftResult:
    """Drift assessment for one appended period, across all scopes."""

    period: int | None = None
    evaluated: bool = False
    scopes: list[DriftScope] = field(default_factory=list)

    @property
    def drifted(self) -> list[DriftScope]:
        return [scope for scope in self.scopes if scope.drifted]

    @property
    def not_evaluated(self) -> list[DriftScope]:
        return [scope for scope in self.scopes if scope.psi is None]

    @property
    def any_drift(self) -> bool:
        return bool(self.drifted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": DRIFT_CHECK_VERSION,
            "period": self.period,
            "evaluated": self.evaluated,
            "any_drift": self.any_drift,
            "scopes": [scope.to_dict() for scope in self.scopes],
            "not_evaluated_scopes": [scope.scope for scope in self.not_evaluated],
        }


def _assess_scope(
    period_values: np.ndarray,
    reference: pd.DataFrame,
    week_column: str,
    metric: str,
    scope: str,
    scope_type: str,
    period: int,
    bins: int,
    reference_weeks: int,
    abs_floor: float,
    threshold_quantile: float,
    min_reference_weeks: int,
) -> DriftScope:
    baseline = _consecutive_psi(reference, week_column, metric, bins)
    reference_observations = int(len(reference))
    period_observations = int(period_values.size)

    def unresolved(reason: str) -> DriftScope:
        return DriftScope(
            scope=scope,
            scope_type=scope_type,
            metric=metric,
            period=period,
            psi=None,
            threshold=None,
            drifted=False,
            reference_weeks=len(baseline),
            reference_observations=reference_observations,
            period_observations=period_observations,
            detail=reason,
        )

    if len(baseline) < min_reference_weeks:
        return unresolved(
            f"insufficient reference history: {len(baseline)} consecutive-week "
            f"comparisons, need {min_reference_weeks}"
        )
    # A scope whose own weeks are already this volatile cannot be alarmed by a
    # ratio in this range; the threshold is its own observed maximum.
    threshold = max(abs_floor, float(np.quantile(baseline, threshold_quantile)))
    score = population_stability_index(
        reference[metric].to_numpy(dtype=np.float64), period_values, bins
    )
    if score is None or not math.isfinite(score):
        return unresolved("distribution not comparable to the reference")
    return DriftScope(
        scope=scope,
        scope_type=scope_type,
        metric=metric,
        period=period,
        psi=score,
        threshold=threshold,
        drifted=score > threshold,
        reference_weeks=len(baseline),
        reference_observations=reference_observations,
        period_observations=period_observations,
        baseline_psi=float(np.quantile(baseline, threshold_quantile)),
        detail=(
            f"psi {score:.5f} exceeds scope threshold {threshold:.5f} derived "
            f"from {len(baseline)} of this scope's own consecutive-week values"
        ),
    )


def assess_distribution_drift(
    fact: pd.DataFrame,
    config: DatasetConfig,
    period: int,
    metric: str | None = None,
    entity_columns: Sequence[str] | None = None,
    bins: int = DEFAULT_BINS,
    reference_weeks: int = DEFAULT_REFERENCE_WEEKS,
    abs_floor: float = DEFAULT_ABS_FLOOR,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    min_reference_weeks: int = DEFAULT_MIN_REFERENCE_WEEKS,
) -> DistributionDriftResult:
    """Compare the appended period's distribution against trailing reference.

    The reference window is the ``reference_weeks`` business periods immediately
    before ``period`` in the same frame, so the comparison never reaches forward
    and never uses the period under test. Scopes are evaluated independently and
    each carries its own threshold.
    """
    week_column = config.week_column
    metric = metric or config.primary_metric
    result = DistributionDriftResult(period=int(period))
    if fact.empty or week_column not in fact.columns or metric not in fact.columns:
        return result
    if not 0.0 < threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be in (0, 1]")
    if abs_floor < 0.0:
        raise ValueError("abs_floor must not be negative")
    if reference_weeks < 1:
        raise ValueError("reference_weeks must be positive")

    period_mask = fact[week_column] == period
    if not bool(period_mask.any()):
        result.evaluated = False
        return result

    earlier = fact[fact[week_column] < period]
    available = sorted(set(earlier[week_column].unique().tolist()))
    window = available[-reference_weeks:]
    reference = earlier[earlier[week_column].isin(window)]
    if not window:
        result.evaluated = False
        return result
    result.evaluated = True

    def assess(
        frame: pd.DataFrame,
        scope: str,
        scope_type: str,
        column: str | None = None,
        value: Any = None,
    ) -> DriftScope:
        values = frame.loc[period_mask[frame.index], metric].to_numpy(dtype=np.float64)
        # The reference must be the scope's *own* rows, not the whole frame.
        # Deriving a scope's threshold from the aggregate history would hand
        # every entity the national threshold, which is wrong by construction
        # and a false-positive generator: a narrow entity scored against a
        # pooled baseline reads high against a threshold that was never
        # calibrated for it.
        own = reference
        if column is not None and column in reference.columns:
            own = reference[reference[column].eq(value)]
        return _assess_scope(
            values,
            own,
            week_column,
            metric,
            scope,
            scope_type,
            int(period),
            bins,
            reference_weeks,
            abs_floor,
            threshold_quantile,
            min_reference_weeks,
        )

    result.scopes.append(assess(fact, f"national:{metric}", "national"))
    for column in (entity_columns if entity_columns is not None else config.entity_columns):
        if column not in fact.columns:
            continue
        for value, group in fact.groupby(column, observed=True, dropna=False):
            label = f"{column}:{value}"
            result.scopes.append(
                assess(group, label, f"entity:{column}", column=column, value=value)
            )
    return result
