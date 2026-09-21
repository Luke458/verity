from __future__ import annotations

import datetime as dt
from dataclasses import replace

import pytest

from qc.calendar import (
    BusinessCalendar,
    build_calendar,
    calendar_identity,
    easter_sunday,
)
from qc.config import DatasetConfig

CONFIG = DatasetConfig(
    calendar_anchor_date="2024-01-01",
    calendar_anchor_week=1,
    calendar_events=(
        ("christmas", "12-25", 1, 1),
        ("easter", "easter", 1, 1),
        ("eofy", "06-30", 0, 0),
    ),
)


def test_declared_calendar_maps_weeks_to_dates():
    calendar = build_calendar(CONFIG)
    assert calendar is not None
    assert calendar.week_start(1) == dt.date(2024, 1, 1)
    assert calendar.week_start(2) == dt.date(2024, 1, 8)
    assert calendar.week_for_date(dt.date(2024, 1, 9)) == 2
    year, week = calendar.retail_year_week(1)
    assert (year, week) == (2024, 1)


def test_moving_easter_aligns_to_declared_weeks():
    calendar = build_calendar(CONFIG)
    assert calendar is not None
    assert easter_sunday(2024) == dt.date(2024, 3, 31)
    assert easter_sunday(2025) == dt.date(2025, 4, 20)

    easter_2024 = calendar.week_for_date(dt.date(2024, 3, 31))
    easter_2025 = calendar.week_for_date(dt.date(2025, 4, 20))
    assert easter_2025 - easter_2024 == dt.timedelta(days=385).days // 7

    weeks = calendar.event_weeks_for_range(list(range(1, 110)))
    assert easter_2024 in weeks["easter"]
    assert easter_2024 - 1 in weeks["easter"]
    assert easter_2024 + 1 in weeks["easter"]
    assert easter_2024 + 2 not in weeks["easter"]
    assert easter_2025 in weeks["easter"]
    # Christmas and financial year end are fixed dates with declared windows.
    assert calendar.week_for_date(dt.date(2024, 12, 25)) in weeks["christmas"]
    assert calendar.week_for_date(dt.date(2024, 6, 30)) in weeks["eofy"]


def test_53_week_retail_years_are_detected():
    assert BusinessCalendar.weeks_in_retail_year(2015) == 53
    assert BusinessCalendar.is_53_week_year(2015)
    assert BusinessCalendar.is_53_week_year(2026)
    assert not BusinessCalendar.is_53_week_year(2024)

    calendar = BusinessCalendar(1, dt.date(2014, 12, 29))
    # The 53rd week has a well-defined retail year/week instead of wrapping.
    year, week = calendar.retail_year_week(53)
    assert (year, week) == (2015, 53)
    year, week = calendar.retail_year_week(54)
    assert (year, week) == (2016, 1)


def test_undecalred_calendar_is_explicit_and_identity_is_stable():
    config = DatasetConfig()
    assert build_calendar(config) is None
    identity = calendar_identity(config)
    assert identity == {"version": 1, "declared": False}

    identity = calendar_identity(CONFIG)
    assert identity["declared"] is True
    assert identity["anchor_date"] == "2024-01-01"
    assert [event["name"] for event in identity["events"]] == [
        "christmas",
        "easter",
        "eofy",
    ]


def test_synthetic_profile_calendar_matches_generator_events():
    from qcgen.config import dataset_calendar, profile_history
    from qcgen.dgp import holiday_boosts, week_end_dates

    history = profile_history("full")
    config = replace(DatasetConfig(), **dataset_calendar("full"))
    calendar = build_calendar(config)
    assert calendar is not None
    dates = week_end_dates(history.start_week, history.n_weeks)
    boosts = holiday_boosts(dates, history)
    generated = {index + 1 for index, value in enumerate(boosts) if value > 1.0}
    declared: set[int] = set()
    for weeks in calendar.event_weeks_for_range(
        list(range(1, history.n_weeks + 1))
    ).values():
        declared.update(weeks)
    assert declared == generated


def test_calendar_events_require_an_anchor():
    with pytest.raises(ValueError, match="require calendar_anchor_date"):
        replace(CONFIG, calendar_anchor_date="")
    with pytest.raises(ValueError, match="ISO date"):
        replace(CONFIG, calendar_anchor_date="01/01/2024")
    with pytest.raises(ValueError, match="easter or MM-DD"):
        replace(CONFIG, calendar_events=(("event", "not-a-rule", 0, 0),))
    with pytest.raises(ValueError, match="nonnegative"):
        replace(CONFIG, calendar_events=(("event", "12-25", -1, 0),))
    with pytest.raises(ValueError, match="interval_alpha"):
        replace(CONFIG, temporal_interval_alpha=0.0)
