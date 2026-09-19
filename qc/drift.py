"""Drift monitoring over the prequential calibration pool.

Calibration assumes residuals stay exchangeable with future errors. This check
compares an earlier baseline window with the most recent window at a chosen
as-of point. Baseline and recent windows are restricted to the series present
in both, so a change in entity mix is not mistaken for drift. Coverage is only
claimed when a conformal interval can actually be built and the recent window
is large enough for its Wilson interval to be informative; otherwise the
report says ``NOT_EVALUATED`` rather than ``STABLE``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from .conformal import conformal_interval, minimum_samples, wilson_interval
from .prequential import CalibrationPool


@dataclass
class DriftReport:
    status: str
    as_of: int
    target_week: int
    usable: int
    baseline_n: int
    recent_n: int
    baseline_median_abs: float | None
    recent_median_abs: float | None
    scale_ratio: float | None
    recent_coverage: float | None
    coverage_ci_low: float | None
    coverage_ci_high: float | None
    expected_coverage: float
    common_series: int = 0
    flags: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "as_of": self.as_of,
            "target_week": self.target_week,
            "usable": self.usable,
            "baseline_n": self.baseline_n,
            "recent_n": self.recent_n,
            "common_series": self.common_series,
            "baseline_median_abs": self.baseline_median_abs,
            "recent_median_abs": self.recent_median_abs,
            "scale_ratio": self.scale_ratio,
            "recent_coverage": self.recent_coverage,
            "coverage_ci_low": self.coverage_ci_low,
            "coverage_ci_high": self.coverage_ci_high,
            "expected_coverage": self.expected_coverage,
            "flags": list(self.flags),
            "detail": self.detail,
        }


def monitor_drift(
    pool: CalibrationPool,
    as_of: int,
    target_week: int,
    alpha: float = 0.05,
    ratio_threshold: float = 1.5,
    coverage_margin: float = 0.1,
    scope: str | None = None,
) -> DriftReport:
    records = sorted(
        pool.usable(as_of, target_week, scope=scope),
        key=lambda record: (record.target_week, record.available_on),
    )
    expected_coverage = 1.0 - alpha
    minimum = max(pool.min_samples, 2)
    if len(records) < 2 * minimum:
        return DriftReport(
            status="INSUFFICIENT",
            as_of=as_of,
            target_week=target_week,
            usable=len(records),
            baseline_n=0,
            recent_n=0,
            baseline_median_abs=None,
            recent_median_abs=None,
            scale_ratio=None,
            recent_coverage=None,
            coverage_ci_low=None,
            coverage_ci_high=None,
            expected_coverage=expected_coverage,
            detail=(
                f"need at least {2 * minimum} usable residuals for a "
                "baseline/recent comparison"
            ),
        )

    split = len(records) // 2
    baseline_all = records[:split]
    recent_all = records[split:]
    common = {record.series_id for record in baseline_all} & {
        record.series_id for record in recent_all
    }
    if not common:
        return DriftReport(
            status="NOT_EVALUATED",
            as_of=as_of,
            target_week=target_week,
            usable=len(records),
            baseline_n=len(baseline_all),
            recent_n=len(recent_all),
            baseline_median_abs=None,
            recent_median_abs=None,
            scale_ratio=None,
            recent_coverage=None,
            coverage_ci_low=None,
            coverage_ci_high=None,
            expected_coverage=expected_coverage,
            detail="baseline and recent windows share no series",
        )
    baseline = [record for record in baseline_all if record.series_id in common]
    recent = [record for record in recent_all if record.series_id in common]
    baseline_abs = [abs(record.residual) for record in baseline]
    recent_abs = [abs(record.residual) for record in recent]
    baseline_median = statistics.median(baseline_abs)
    recent_median = statistics.median(recent_abs)

    flags: list[str] = []
    ratio = None
    if baseline_median > 1e-12:
        ratio = recent_median / baseline_median
        if ratio > ratio_threshold:
            flags.append("residual_scale_drift")
    else:
        flags.append("scale_ratio_undefined")

    coverage: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    required = max(minimum, minimum_samples(alpha))
    if len(baseline_abs) >= required:
        interval = conformal_interval(baseline_abs, center=0.0, alpha=alpha)
        if interval.status == "OK" and interval.upper is not None:
            covered = sum(1 for value in recent_abs if value <= interval.upper)
            coverage = covered / len(recent_abs)
            ci_low, ci_high = wilson_interval(covered, len(recent_abs))
            if ci_high is not None and ci_high < expected_coverage - coverage_margin:
                flags.append("coverage_drift")
        else:
            flags.append("coverage_not_evaluated")
    else:
        flags.append("coverage_not_evaluated")

    if "residual_scale_drift" in flags or "coverage_drift" in flags:
        status = "DRIFT"
    elif "coverage_not_evaluated" in flags or "scale_ratio_undefined" in flags:
        status = "NOT_EVALUATED"
    else:
        status = "STABLE"

    if status == "DRIFT":
        detail = (
            "recent residuals inflate; refit calibration"
            if "residual_scale_drift" in flags
            else "recent coverage below nominal; check drift"
        )
    elif status == "NOT_EVALUATED":
        detail = "drift could not be evaluated: " + ", ".join(flags)
    else:
        detail = "baseline and recent windows are consistent"

    return DriftReport(
        status=status,
        as_of=as_of,
        target_week=target_week,
        usable=len(records),
        baseline_n=len(baseline),
        recent_n=len(recent),
        baseline_median_abs=baseline_median,
        recent_median_abs=recent_median,
        scale_ratio=ratio,
        recent_coverage=coverage,
        coverage_ci_low=ci_low,
        coverage_ci_high=ci_high,
        expected_coverage=expected_coverage,
        common_series=len(common),
        flags=flags,
        detail=detail,
    )
