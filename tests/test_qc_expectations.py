from __future__ import annotations

import pytest

from qc.expectations import (
    RatioExpectation,
    apply_expectations,
    load_expectations,
    save_expectations,
)


def _flag(week: int = 6, ratio: float = 100.0, metric: str = "dollar_per_unit"):
    return {"metric": metric, "week": week, "ratio": ratio}


def _expectation(**overrides) -> RatioExpectation:
    data = {
        "expectation_id": "e1",
        "dataset": "synthetic-retail",
        "metric": "dollar_per_unit",
        "min_ratio": 50.0,
        "max_ratio": 150.0,
        "week": 6,
        "approved_by": "analyst",
    }
    data.update(overrides)
    return RatioExpectation(**data)


def test_approved_expectation_explains_only_its_alert():
    report = apply_expectations(
        [_flag(6, 100.0), _flag(7, 100.0)],
        [_expectation()],
        {"dataset": "synthetic-retail", "as_of": "2026-06-01"},
    )
    assert [item["week"] for item in report["expected"]] == [6]
    assert report["expected"][0]["explained_by"] == "e1"
    assert report["expected"][0]["ambiguous"] is False
    assert [item["week"] for item in report["unexpected"]] == [7]


def test_ambiguous_matches_are_flagged():
    report = apply_expectations(
        [_flag(6, 100.0)],
        [_expectation(expectation_id="e1"), _expectation(expectation_id="e2")],
        {"dataset": "synthetic-retail", "as_of": "2026-06-01"},
    )
    assert len(report["expected"]) == 1
    assert report["expected"][0]["ambiguous"] is True
    assert report["expected"][0]["match_count"] == 2


def test_unapproved_expired_and_wrong_scope_are_ignored():
    expectations = [
        _expectation(expectation_id="unapproved", approved_by=None),
        _expectation(expectation_id="wrong-dataset", dataset="other"),
        _expectation(
            expectation_id="expired",
            effective_to="2025-01-01",
        ),
        _expectation(
            expectation_id="out-of-band",
            min_ratio=200.0,
            max_ratio=300.0,
        ),
        _expectation(expectation_id="wrong-week", week=9),
    ]
    report = apply_expectations(
        [_flag(6, 100.0)],
        expectations,
        {"dataset": "synthetic-retail", "as_of": "2026-06-01"},
    )
    assert report["expected"] == []
    assert len(report["unexpected"]) == 1
    reasons = " ".join(
        reason for entry in report["audit"] for reason in entry["reasons"]
    )
    assert "unapproved" in reasons
    assert "wrong dataset" in reasons
    assert "expired" in reasons
    assert "outside approved band" in reasons
    assert "wrong week" in reasons


def test_expectation_validation():
    with pytest.raises(ValueError):
        RatioExpectation("e", "d", "m")
    with pytest.raises(ValueError):
        RatioExpectation("e", "d", "m", min_ratio=10, max_ratio=1)
    with pytest.raises(ValueError):
        RatioExpectation(
            "e", "d", "m", min_ratio=1, max_ratio=10,
            effective_from="2026-02-01", effective_to="2026-01-01",
        )


def test_roundtrip(tmp_path):
    path = tmp_path / "expectations.json"
    expectations = [_expectation(), _expectation(expectation_id="e2", week=7)]
    save_expectations(expectations, path)
    loaded = load_expectations(path)
    assert loaded == expectations
