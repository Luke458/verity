"""Drift monitoring over the prequential calibration pool.

Calibration assumes residuals stay exchangeable with future errors. This check
compares an earlier baseline window with the most recent window at a chosen
as-of point: if the residual scale inflates beyond a ratio threshold, or recent
coverage of the baseline conformal interval falls below nominal, the status is
``DRIFT`` and the calibration should be refitted before it is trusted.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from .conformal import conformal_interval
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
    expected_coverage: float
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
            "baseline_median_abs": self.baseline_median_abs,
            "recent_median_abs": self.recent_median_abs,
            "scale_ratio": self.scale_ratio,
            "recent_coverage": self.recent_coverage,
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
) -> DriftReport:
    records = sorted(
        pool.usable(as_of, target_week),
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
            expected_coverage=expected_coverage,
            detail=(
                f"need at least {2 * minimum} usable residuals for a "
                "baseline/recent comparison"
            ),
        )

    split = len(records) // 2
    baseline = records[:split]
    recent = records[split:]
    baseline_abs = [abs(record.residual) for record in baseline]
    recent_abs = [abs(record.residual) for record in recent]
    baseline_median = statistics.median(baseline_abs)
    recent_median = statistics.median(recent_abs)
    ratio = (
        recent_median / baseline_median
        if baseline_median > 1e-12
        else (float("inf") if recent_median > 1e-12 else 1.0)
    )

    interval = conformal_interval(baseline_abs, center=0.0, alpha=alpha)
    coverage: float | None = None
    if interval.status == "OK" and interval.upper is not None:
        covered = sum(1 for value in recent_abs if value <= interval.upper)
        coverage = covered / len(recent_abs)

    flags: list[str] = []
    if ratio > ratio_threshold:
        flags.append("residual_scale_drift")
    if coverage is not None and coverage < expected_coverage - coverage_margin:
        flags.append("coverage_drift")

    return DriftReport(
        status="DRIFT" if flags else "STABLE",
        as_of=as_of,
        target_week=target_week,
        usable=len(records),
        baseline_n=len(baseline),
        recent_n=len(recent),
        baseline_median_abs=baseline_median,
        recent_median_abs=recent_median,
        scale_ratio=ratio,
        recent_coverage=coverage,
        expected_coverage=expected_coverage,
        flags=flags,
        detail=(
            "recent residuals inflate; refit calibration"
            if "residual_scale_drift" in flags
            else "recent coverage below nominal; check drift"
            if "coverage_drift" in flags
            else "baseline and recent windows are consistent"
        ),
    )
