"""Scope-level coverage baseline.

A per-entity rule that escalates whenever an entity is absent from the appended
period is only meaningful on a dense source. On a real transactional source it
is not a rule at all: the measured 27-refresh replay found 80-89% of
previously-seen entity pairs absent in any given week, with a median dropped
entity having traded in only two of the last eight weeks. Every refresh
escalated, which is the same as not escalating.

This module supplies the missing half. It answers a scope-level question -
*did this scope lose coverage relative to its own recent experience?* - and that
answer is what a per-entity escalation should be gated on. Per-entity absences
remain facts and remain available as evidence; they simply must not escalate on
their own.

The threshold is derived from each scope's own trailing coverage counts, for the
same reason the distribution-drift check does: real scopes differ by orders of
magnitude in size and volatility, so an absolute expected-entity count is
meaningless and a global constant is wrong. Coverage is deliberately
one-sided. Only a fall below a scope's own recent experience is evidence of a
regression; a rise is never a regression and must not be reported as one.

**Status: the pre-registered gate in `config/coverage-gate.json` FAILED** and is
recorded as failed in `docs/claims.md`. This module is a measured, documented
challenger, not an active check. See the claims matrix for the numbers and for
why the threshold was not moved to make it pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import DatasetConfig

COVERAGE_CHECK_VERSION = 1

DEFAULT_REFERENCE_PERIODS = 8
DEFAULT_QUANTILE = 0.05
# Below this many reference periods the scope's own experience cannot be
# characterised, and the scope is reported as not evaluated rather than assumed.
DEFAULT_MIN_REFERENCE_PERIODS = 4


@dataclass(frozen=True)
class CoverageScope:
    """One scope's coverage evidence for the appended period."""

    scope: str
    scope_type: str
    period: int
    covered: int
    threshold: int | None
    regressed: bool
    reference_periods: int
    baseline_min: int | None = None
    baseline_max: int | None = None
    baseline_median: float | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "scope_type": self.scope_type,
            "period": self.period,
            "covered": self.covered,
            "threshold": self.threshold,
            "regressed": self.regressed,
            "reference_periods": self.reference_periods,
            "baseline_min": self.baseline_min,
            "baseline_max": self.baseline_max,
            "baseline_median": self.baseline_median,
            "detail": self.detail,
        }


@dataclass
class CoverageResult:
    """Coverage assessment for one appended period across all scopes."""

    period: int | None = None
    evaluated: bool = False
    scopes: list[CoverageScope] = field(default_factory=list)

    @property
    def regressed(self) -> list[CoverageScope]:
        return [scope for scope in self.scopes if scope.regressed]

    @property
    def not_evaluated(self) -> list[CoverageScope]:
        return [scope for scope in self.scopes if scope.threshold is None]

    @property
    def any_regression(self) -> bool:
        return bool(self.regressed)

    def regressed_scope_types(self) -> frozenset[str]:
        """Scope *types* that regressed, for gating per-entity escalations."""
        return frozenset(scope.scope_type for scope in self.regressed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": COVERAGE_CHECK_VERSION,
            "period": self.period,
            "evaluated": self.evaluated,
            "any_regression": self.any_regression,
            "scopes": [scope.to_dict() for scope in self.scopes],
            "not_evaluated_scopes": [scope.scope for scope in self.not_evaluated],
        }


def _scope_label(column: str, value: Any) -> str:
    return f"{column}:{value}"


def _coverage_table(
    fact: pd.DataFrame,
    week_column: str,
    column: str | None,
    leaf: Sequence[str],
) -> pd.DataFrame:
    """Distinct-leaf coverage per period, as a period x scope table.

    One groupby over the whole frame, rather than a masked pass per scope per
    period: the obvious implementation rescans the frame 38 scopes x 9 periods
    times, which is what made the first gate run exceed its budget.
    """
    keys = [week_column, *leaf]
    if column is not None:
        keys.insert(1, column)
    grouped = (
        fact.groupby(keys, observed=True, dropna=False).size().reset_index()
    )
    if column is None:
        counts = grouped.groupby(week_column, observed=True)[keys[1:]].size()
        return counts.to_frame("n")
    counted = grouped.groupby([week_column, column], observed=True)[list(leaf)].size()
    return counted.unstack(column)


def assess_scope_coverage(
    fact: pd.DataFrame,
    config: DatasetConfig,
    period: int,
    entity_columns: Sequence[str] | None = None,
    reference_periods: int = DEFAULT_REFERENCE_PERIODS,
    quantile: float = DEFAULT_QUANTILE,
    min_reference_periods: int = DEFAULT_MIN_REFERENCE_PERIODS,
) -> CoverageResult:
    """Compare the appended period's coverage against each scope's own history.

    A scope is a value of one configured entity column and its coverage is the
    number of distinct *leaf* keys present for it: for a store scope the leaves
    are the products it carried, for a product scope the stores that carried it.
    The national scope covers the full key tuple.

    ``quantile`` is the lower tail. A scope regresses when its coverage falls
    below the level it has already experienced in ``quantile`` of its own
    reference periods.
    """
    week_column = config.week_column
    result = CoverageResult(period=int(period))
    if fact.empty or week_column not in fact.columns:
        return result
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if reference_periods < 1:
        raise ValueError("reference_periods must be positive")

    earlier = fact[fact[week_column] < period]
    if not bool((fact[week_column] == period).any()) or earlier.empty:
        result.evaluated = False
        return result
    result.evaluated = True

    keys = [key for key in config.entity_key_columns if key in fact.columns]
    if not keys:
        return result
    window = sorted(earlier[week_column].unique().tolist())[-reference_periods:]

    columns = (
        list(entity_columns)
        if entity_columns is not None
        else [column for column in config.entity_columns if column in fact.columns]
    )
    tables: dict[str, pd.DataFrame] = {
        "national": _coverage_table(fact, week_column, None, keys)
    }
    for column in columns:
        if column in fact.columns:
            leaf = [key for key in keys if key != column]
            tables[f"entity:{column}"] = _coverage_table(
                fact, week_column, column, leaf
            )

    def evaluate(label: str, scope_type: str, column: str | None, value: Any) -> None:
        table = tables.get(scope_type)
        if table is None:
            return
        if column is None:
            series = table.iloc[:, 0]
        elif value in table.columns:
            series = table[value]
        else:
            return
        # An unstack leaves NaN where a scope never appeared in a period; a
        # scope absent from a period genuinely covered nothing.
        series = series.reindex(sorted(fact[week_column].unique())).astype("float64").fillna(0.0)
        history = [int(series.get(week_id, 0.0)) for week_id in window]
        covered = int(series.get(period, 0.0))
        if len(history) < min_reference_periods:
            result.scopes.append(
                CoverageScope(
                    scope=label,
                    scope_type=scope_type,
                    period=int(period),
                    covered=covered,
                    threshold=None,
                    regressed=False,
                    reference_periods=len(history),
                    detail=(
                        f"insufficient reference history: {len(history)} periods, "
                        f"need {min_reference_periods}"
                    ),
                )
            )
            return
        floor = int(np.quantile(history, quantile))
        result.scopes.append(
            CoverageScope(
                scope=label,
                scope_type=scope_type,
                period=int(period),
                covered=covered,
                threshold=floor,
                regressed=covered < floor,
                reference_periods=len(history),
                baseline_min=int(min(history)),
                baseline_max=int(max(history)),
                baseline_median=float(np.median(history)),
                detail=(
                    f"covered {covered} against this scope's own "
                    f"{quantile:.0%} lower bound {floor} "
                    f"(own range {min(history)}-{max(history)})"
                ),
            )
        )

    evaluate("national", "national", None, None)
    for column in columns:
        if column not in fact.columns:
            continue
        for value in fact[column].dropna().unique().tolist():
            evaluate(
                _scope_label(column, value), f"entity:{column}", column, value
            )
    return result


@dataclass(frozen=True)
class AggregateCoverage:
    """Aggregate coverage for one entity key, over the appended period.

    This is the decision-grade view. The per-scope check asks "did this one
    scope twitch"; the escalation it gates asks "did we lose products / did we
    lose stores". Only the aggregate answers that question, and it is the only
    one of the two whose flag volume is small enough to sit in a decision path.
    """

    entity_column: str
    counterpart_column: str
    period: int
    covered: int
    threshold: int | None
    regressed: bool
    reference_periods: int
    baseline_min: int | None = None
    baseline_max: int | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_column": self.entity_column,
            "counterpart_column": self.counterpart_column,
            "period": self.period,
            "covered": self.covered,
            "threshold": self.threshold,
            "regressed": self.regressed,
            "reference_periods": self.reference_periods,
            "baseline_min": self.baseline_min,
            "baseline_max": self.baseline_max,
            "detail": self.detail,
        }


def assess_aggregate_coverage(
    fact: pd.DataFrame,
    config: DatasetConfig,
    period: int,
    entity_columns: Sequence[str] | None = None,
    reference_periods: int = DEFAULT_REFERENCE_PERIODS,
    quantile: float = DEFAULT_QUANTILE,
    min_reference_periods: int = DEFAULT_MIN_REFERENCE_PERIODS,
) -> dict[str, AggregateCoverage]:
    """Distinct counterpart keys per period, against each key's own history.

    For ``product_id`` the counterpart is ``store_id``: the question is how many
    distinct stores carried at least one of these products, not how many
    individual products traded. That is the quantity a coverage regression
    actually means, and it is one number per entity key rather than one per
    entity value.
    """
    week_column = config.week_column
    if fact.empty or week_column not in fact.columns:
        return {}
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    earlier = fact[fact[week_column] < period]
    if not bool((fact[week_column] == period).any()) or earlier.empty:
        return {}
    window = sorted(earlier[week_column].unique().tolist())[-reference_periods:]
    keys = [key for key in config.entity_key_columns if key in fact.columns]
    columns = (
        list(entity_columns)
        if entity_columns is not None
        else [column for column in config.entity_columns if column in fact.columns]
    )
    results: dict[str, AggregateCoverage] = {}
    for column in columns:
        counterparts = [key for key in keys if key != column]
        if not counterparts:
            continue
        target = counterparts[0]
        counts = {
            int(week_id): int(fact.loc[fact[week_column] == week_id, target].nunique())
            for week_id in window
        }
        history = [counts[week_id] for week_id in window]
        covered = int(fact.loc[fact[week_column] == period, target].nunique())
        if len(history) < min_reference_periods:
            results[column] = AggregateCoverage(
                entity_column=column,
                counterpart_column=target,
                period=int(period),
                covered=covered,
                threshold=None,
                regressed=False,
                reference_periods=len(history),
                detail=(
                    f"insufficient reference history: {len(history)} periods, "
                    f"need {min_reference_periods}"
                ),
            )
            continue
        floor = int(np.quantile(history, quantile))
        results[column] = AggregateCoverage(
            entity_column=column,
            counterpart_column=target,
            period=int(period),
            covered=covered,
            threshold=floor,
            regressed=covered < floor,
            reference_periods=len(history),
            baseline_min=int(min(history)),
            baseline_max=int(max(history)),
            detail=(
                f"{covered} distinct {target} against this column's own "
                f"{quantile:.0%} lower bound {floor} "
                f"(own range {min(history)}-{max(history)})"
            ),
        )
    return results
