from __future__ import annotations

from qc.drift import monitor_drift
from qc.prequential import CalibrationPool, CalibrationRecord


def _record(target: int, residual: float) -> CalibrationRecord:
    return CalibrationRecord(
        series_id="national",
        target_week=target,
        available_on=target,
        residual=residual,
    )


def test_insufficient_pool_reports_insufficient():
    pool = CalibrationPool(min_samples=5)
    for target in range(1, 5):
        pool.add(_record(target, 0.1))
    report = monitor_drift(pool, as_of=10, target_week=11)
    assert report.status == "INSUFFICIENT"
    assert "need at least" in report.detail


def test_stable_pool_reports_stable():
    pool = CalibrationPool(min_samples=5)
    pattern = [-0.2, 0.1, 0.0, 0.3, -0.1]
    for index in range(20):
        pool.add(_record(index + 1, pattern[index % len(pattern)]))
    report = monitor_drift(pool, as_of=25, target_week=26)
    assert report.status == "STABLE"
    assert report.flags == []
    assert report.baseline_n == 10
    assert report.recent_n == 10


def test_scale_inflation_is_detected():
    pool = CalibrationPool(min_samples=5)
    for index in range(20):
        residual = 0.1 if index < 10 else 2.0 + 0.1 * index
        pool.add(_record(index + 1, residual))
    report = monitor_drift(pool, as_of=25, target_week=26)
    assert report.status == "DRIFT"
    assert "residual_scale_drift" in report.flags
    assert report.scale_ratio is not None and report.scale_ratio > 1.5
    assert "refit" in report.detail
