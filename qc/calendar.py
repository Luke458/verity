"""Immutable declared business calendar and retail event windows.

The engine never assumes a geography: a dataset declares the first day of one
business week and, optionally, the annual events that move or fix demand.
Business periods stay integer week indices; the calendar maps them to dates,
retail (ISO) year/week and event indicators. Retail years with 53 weeks are
handled by the ISO calendar rather than by a fixed 52-week assumption.

Event rules are explicit:

* ``easter`` - the moving Easter Sunday for a Gregorian year.
* ``MM-DD`` - a fixed annual date (for example ``12-25`` or ``06-30``).

``lead_weeks`` and ``lag_weeks`` extend the event window to adjacent business
weeks, which is how retail calibrations cover the weeks around Christmas,
Easter and financial year end.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from .config import DatasetConfig

CALENDAR_VERSION = 1

_EVENT_RULE_EASTER = "easter"


def easter_sunday(year: int) -> dt.date:
    """Anonymous Gregorian algorithm for the date of Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def _fixed_rule_date(rule: str, year: int) -> dt.date | None:
    try:
        month_text, day_text = rule.split("-", 1)
        month, day = int(month_text), int(day_text)
    except ValueError:
        return None
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


@dataclass(frozen=True)
class BusinessCalendar:
    """Declared mapping from business week indices to dates and events."""

    anchor_week: int
    anchor_date: dt.date
    events: tuple[tuple[str, str, int, int], ...] = ()

    def week_start(self, week: int) -> dt.date:
        return self.anchor_date + dt.timedelta(days=7 * (int(week) - self.anchor_week))

    def contains(self, week: int, when: dt.date) -> bool:
        start = self.week_start(week)
        return start <= when < start + dt.timedelta(days=7)

    def week_for_date(self, when: dt.date) -> int:
        offset = (when - self.anchor_date).days
        return self.anchor_week + offset // 7

    def retail_year_week(self, week: int) -> tuple[int, int]:
        iso = self.week_start(week).isocalendar()
        return int(iso.year), int(iso.week)

    @staticmethod
    def weeks_in_retail_year(year: int) -> int:
        return dt.date(year, 12, 28).isocalendar().week

    @classmethod
    def is_53_week_year(cls, year: int) -> bool:
        return cls.weeks_in_retail_year(year) == 53

    def event_dates(self, name: str, rule: str, year: int) -> list[dt.date]:
        if rule == _EVENT_RULE_EASTER:
            return [easter_sunday(year)]
        fixed = _fixed_rule_date(rule, year)
        return [fixed] if fixed else []

    def event_weeks(self, name: str, rule: str, lead: int, lag: int, week: int) -> set[int]:
        """Event window containing ``week`` for every calendar year nearby."""
        when = self.week_start(week)
        weeks: set[int] = set()
        for year in (when.year - 1, when.year, when.year + 1):
            for date in self.event_dates(name, rule, year):
                centre = self.week_for_date(date)
                weeks.update(range(centre - lead, centre + lag + 1))
        return weeks

    def event_windows(self, week: int) -> dict[str, bool]:
        return {
            name: week in self.event_weeks(name, rule, lead, lag, week)
            for name, rule, lead, lag in self.events
        }

    def event_weeks_for_range(self, weeks: list[int] | tuple[int, ...]) -> dict[str, set[int]]:
        span = {int(week) for week in weeks}
        if not span:
            return {name: set() for name, _, _, _ in self.events}
        first_year = self.week_start(min(span)).year - 1
        last_year = self.week_start(max(span)).year + 1
        windows: dict[str, set[int]] = {}
        for name, rule, lead, lag in self.events:
            found: set[int] = set()
            for year in range(first_year, last_year + 1):
                for date in self.event_dates(name, rule, year):
                    centre = self.week_for_date(date)
                    found.update(range(centre - lead, centre + lag + 1))
            windows[name] = found & span
        return windows

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CALENDAR_VERSION,
            "anchor_week": self.anchor_week,
            "anchor_date": self.anchor_date.isoformat(),
            "events": [
                {
                    "name": name,
                    "rule": rule,
                    "lead_weeks": lead,
                    "lag_weeks": lag,
                }
                for name, rule, lead, lag in self.events
            ],
        }

    def identity(self) -> dict[str, Any]:
        return self.to_dict()


def build_calendar(config: DatasetConfig) -> BusinessCalendar | None:
    """Return the declared calendar, or ``None`` when no anchor is declared.

    A missing calendar is an explicit unsupported-support state, never a silent
    assumption of a particular country or retail convention.
    """
    if not config.calendar_anchor_date:
        return None
    return BusinessCalendar(
        anchor_week=config.calendar_anchor_week,
        anchor_date=dt.date.fromisoformat(config.calendar_anchor_date),
        events=config.calendar_events,
    )


def calendar_identity(config: DatasetConfig) -> dict[str, Any]:
    calendar = build_calendar(config)
    if calendar is None:
        return {"version": CALENDAR_VERSION, "declared": False}
    identity = calendar.identity()
    identity["declared"] = True
    return identity
