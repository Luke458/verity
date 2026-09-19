from __future__ import annotations

import math
import random

import pytest

from qc.conformal import (
    conformal_interval,
    leave_one_out_coverage,
    minimum_samples,
)


def test_interval_exists_with_enough_residuals():
    residuals = [float(value) for value in range(1, 41)]
    interval = conformal_interval(residuals, center=0.0, alpha=0.05)
    assert interval.status == "OK"
    assert interval.n == 40
    assert interval.rank == math.ceil(41 * 0.95)
    assert interval.lower == pytest.approx(-39.0)
    assert interval.upper == pytest.approx(39.0)
    assert interval.width == pytest.approx(78.0)


def test_interval_is_insufficient_for_small_samples():
    interval = conformal_interval([1.0, 2.0, 3.0], center=0.0, alpha=0.05)
    assert interval.status == "INSUFFICIENT_CALIBRATION"
    assert interval.lower is None and interval.upper is None
    assert "need at least" in interval.detail
    assert conformal_interval([], center=0.0).status == "INSUFFICIENT_CALIBRATION"


def test_alpha_validation():
    with pytest.raises(ValueError):
        conformal_interval([1.0, 2.0], center=0.0, alpha=0.0)
    with pytest.raises(ValueError):
        conformal_interval([1.0, 2.0], center=0.0, alpha=1.0)


def test_leave_one_out_coverage_matches_nominal():
    residuals = [float(value) for value in range(1, 41)]
    report = leave_one_out_coverage(residuals, alpha=0.05)
    assert report["n"] == 40
    assert report["coverage"] == pytest.approx(0.95, abs=0.03)


def test_minimum_samples_formula():
    assert minimum_samples(0.05) == 19
    assert minimum_samples(0.1) == 9
    assert minimum_samples(0.2) == 4
    assert minimum_samples(0.5) == 1
    assert minimum_samples(0.01) == 99
    with pytest.raises(ValueError):
        minimum_samples(0.0)


def test_non_finite_residuals_are_rejected():
    with pytest.raises(ValueError):
        conformal_interval([1.0, float("nan")], center=0.0)
    with pytest.raises(ValueError):
        conformal_interval([1.0, float("inf")], center=0.0)
    with pytest.raises(ValueError):
        leave_one_out_coverage([1.0, float("nan")])


def test_leave_one_out_is_not_evaluable_not_zero():
    report = leave_one_out_coverage([1.0, 2.0, 3.0], alpha=0.05)
    assert report["evaluated"] is False
    assert report["coverage"] is None
    assert report["n"] == 0


def test_exchangeable_coverage_is_close_to_nominal():
    rng = random.Random(7)
    residuals = [rng.gauss(0.0, 1.0) for _ in range(200)]
    report = leave_one_out_coverage(residuals, alpha=0.1)
    assert report["evaluated"] is True
    assert report["n"] == 200
    assert 0.82 <= float(report["coverage"]) <= 1.0
