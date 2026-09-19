from __future__ import annotations

import pytest

from qc.prequential import CalibrationPool, CalibrationRecord, PrequentialStore


def _record(target: int, residual: float, series: str = "national") -> CalibrationRecord:
    return CalibrationRecord(
        series_id=series,
        target_week=target,
        available_on=target,
        residual=residual,
        run_id=f"run-{target}",
    )


def test_pool_latest_wins_and_bounds_size():
    pool = CalibrationPool(max_records=3, min_samples=2)
    assert pool.add(_record(10, 1.0)) is True
    # A corrected residual supersedes the stale one instead of being rejected.
    assert pool.add(_record(10, 5.0)) is True
    assert [record.residual for record in pool.records] == [5.0]
    for target in (11, 12, 13):
        pool.add(_record(target, float(target)))
    assert len(pool.records) == 3
    assert [record.target_week for record in pool.records] == [11, 12, 13]


def test_pool_rejects_non_finite_residuals():
    pool = CalibrationPool(min_samples=1)
    with pytest.raises(ValueError):
        pool.add(_record(10, float("nan")))
    with pytest.raises(ValueError):
        pool.add(_record(11, float("inf")))


def test_scope_partition_is_respected():
    pool = CalibrationPool(min_samples=1)
    pool.add(
        CalibrationRecord("national", 10, 10, 1.0, run_id="a", scope="alpha")
    )
    pool.add(
        CalibrationRecord("national", 10, 10, 9.0, run_id="b", scope="beta")
    )
    assert [r.scope for r in pool.usable(20, 20, scope="alpha")] == ["alpha"]
    # z=5 is above alpha's single residual (extreme=1 -> 1.0) but below beta's
    # (extreme=0 -> 0.5): the pools are genuinely separate.
    assert pool.p_value(5.0, 20, 20, tail="lower", scope="alpha") == pytest.approx(1.0)
    assert pool.p_value(5.0, 20, 20, tail="lower", scope="beta") == pytest.approx(0.5)


def test_conformal_p_value_formula():
    pool = CalibrationPool(min_samples=1)
    for target, residual in ((1, -1.0), (2, 0.0), (3, 1.0), (4, 2.0)):
        pool.add(_record(target, residual))
    # 1 + #{r <= -1} = 2 over n + 1 = 5.
    assert pool.p_value(-1.0, 10, 10, tail="lower") == pytest.approx(0.4)
    # 1 + #{r >= 2} = 2 over 5.
    assert pool.p_value(2.0, 10, 10, tail="upper") == pytest.approx(0.4)
    assert pool.p_value(100.0, 10, 10, tail="upper") == pytest.approx(0.2)


def test_leakage_rules_exclude_current_and_future():
    pool = CalibrationPool(min_samples=1)
    for target in (10, 11, 12):
        pool.add(_record(target, 0.1 * target))
    usable = pool.usable(as_of=12, target_week=12)
    assert [record.target_week for record in usable] == [10, 11]


def test_percentile_requires_min_samples():
    pool = CalibrationPool(min_samples=3)
    pool.add(_record(10, -1.0))
    pool.add(_record(11, 0.0))
    assert pool.percentile(0.5, as_of=12, target_week=12) is None
    pool.add(_record(12, 1.0))
    percentile = pool.percentile(0.5, as_of=13, target_week=13)
    assert percentile is not None and 0.0 <= percentile <= 1.0


def test_interval_uses_conformal_rank():
    pool = CalibrationPool(min_samples=3)
    for target, residual in ((1, -2.0), (2, -1.0), (3, 0.0), (4, 1.0)):
        pool.add(_record(target, residual))
    interval = pool.interval(0.0, as_of=10, target_week=10, alpha=0.25)
    assert interval.status == "OK"
    assert interval.n == 4
    assert interval.upper == pytest.approx(2.0)

    strict = pool.interval(0.0, as_of=10, target_week=10, alpha=0.05)
    assert strict.status == "INSUFFICIENT_CALIBRATION"


def test_store_roundtrip(tmp_path):
    store = PrequentialStore(tmp_path / "calibration.jsonl")
    store.add([_record(10, -1.0), _record(11, 0.5)])
    pool = store.pool(min_samples=2)
    assert len(pool.records) == 2
    assert pool.records[0].residual == pytest.approx(-1.0)
