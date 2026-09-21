"""Version pair resolution.

Identifies previous/current snapshots, the overlapping period and the new
periods, and flags unexpected version shapes (no new week, multiple new weeks,
shortened or extended history).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import DatasetConfig
from .source import VersionSource


@dataclass(frozen=True)
class VersionPair:
    previous_id: str
    current_id: str
    previous_stage: str
    current_stage: str
    previous_min_week: int
    previous_max_week: int
    current_min_week: int
    current_max_week: int
    overlap_start: int
    overlap_end: int
    new_periods: tuple[int, ...]
    shape: str

    @property
    def overlap_weeks(self) -> tuple[int, ...]:
        return tuple(range(self.overlap_start, self.overlap_end + 1))

    def to_dict(self) -> dict:
        return {
            "previous_id": self.previous_id,
            "current_id": self.current_id,
            "previous_stage": self.previous_stage,
            "current_stage": self.current_stage,
            "previous_min_week": self.previous_min_week,
            "previous_max_week": self.previous_max_week,
            "current_min_week": self.current_min_week,
            "current_max_week": self.current_max_week,
            "overlap_start": self.overlap_start,
            "overlap_end": self.overlap_end,
            "new_periods": list(self.new_periods),
            "shape": self.shape,
        }


def build_version_pair(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    previous_id: str,
    current_id: str,
    previous_stage: str,
    current_stage: str,
    config: DatasetConfig,
) -> VersionPair:
    week = config.week_column
    if week not in previous.columns or week not in current.columns:
        raise ValueError(f"week column {week!r} missing from a version")
    if previous.empty or current.empty:
        raise ValueError("cannot compare empty versions")

    previous_min, previous_max = int(previous[week].min()), int(previous[week].max())
    current_min, current_max = int(current[week].min()), int(current[week].max())
    new_periods = tuple(
        sorted(w for w in set(int(v) for v in current[week].unique()) if w > previous_max)
    )
    overlap_start = max(previous_min, current_min)
    overlap_end = min(previous_max, current_max)

    if current_max <= previous_max:
        shape = "NO_NEW_WEEK"
    elif current_min < previous_min:
        shape = "HISTORY_EXTENDED"
    elif current_min > previous_min:
        shape = "HISTORY_SHORTENED"
    elif len(new_periods) > 1:
        shape = "MULTIPLE_NEW_WEEKS"
    else:
        shape = "NORMAL"

    return VersionPair(
        previous_id=previous_id,
        current_id=current_id,
        previous_stage=previous_stage,
        current_stage=current_stage,
        previous_min_week=previous_min,
        previous_max_week=previous_max,
        current_min_week=current_min,
        current_max_week=current_max,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        new_periods=new_periods,
        shape=shape,
    )


def resolve_versions(
    source: VersionSource,
    current_id: str,
    previous_id: str,
    config: DatasetConfig,
) -> VersionPair:
    previous_stage = source.resolve_stage(previous_id, config.analysis_stages())
    current_stage = source.resolve_stage(current_id, config.analysis_stages())
    previous = source.read_fact(previous_id, previous_stage)
    current = source.read_fact(current_id, current_stage)
    return build_version_pair(
        previous,
        current,
        previous_id,
        current_id,
        previous_stage,
        current_stage,
        config,
    )


def select_versions(versions: list[str], current: str | None = None,
                    previous: str | None = None) -> tuple[str, str]:
    """Select a predecessor relative to current using source history order."""
    if not versions:
        raise ValueError("no available snapshots")
    current = current or versions[-1]
    if current not in versions:
        raise ValueError(f"unavailable current snapshot: {current}")
    index = versions.index(current)
    if previous is None:
        if index == 0:
            raise ValueError("selected snapshot has no predecessor")
        previous = versions[index - 1]
    if previous not in versions or versions.index(previous) >= index:
        raise ValueError("previous snapshot must exist and precede current")
    return current, previous
