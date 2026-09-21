from __future__ import annotations

import datetime as dt
import math
from dataclasses import replace

import numpy as np
import pandas as pd

from qc.calendar import build_calendar
from qc.config import DatasetConfig
from qc.run import run_qc

BASE = DatasetConfig(report_grain=(), forecaster="auto")

CHRISTMAS = (("christmas", "12-25", 0, 0),)


class FrameSource:
    def __init__(self, frames):
        self.frames = frames

    def list_versions(self):
        return ["v1", "v2"]

    def available_stages(self, version):
        return ["warehouse", "report"]

    def read_fact(self, version, stage):
        return self.frames[version, stage].copy()

    def read_dim(self, version, name):
        return None


def _calendar_config(target_date: dt.date, target_week: int = 121) -> DatasetConfig:
    anchor = target_date - dt.timedelta(days=7 * (target_week - 1))
    return replace(
        BASE,
        calendar_anchor_date=anchor.isoformat(),
        calendar_events=CHRISTMAS,
    )


def _seasonal_frames(
    config: DatasetConfig,
    target_factor: float,
    seed: int = 5,
    commodities: tuple[str, ...] = (),
):
    calendar = build_calendar(config)
    rng = np.random.default_rng(seed)
    values: dict[int, float] = {}
    for week in range(1, 122):
        base = 1000.0 + 5.0 * week + 100.0 * math.sin(2.0 * math.pi * week / 52.0)
        values[week] = base + float(rng.normal(0.0, 10.0))
    windows = calendar.event_weeks_for_range(list(range(1, 122)))
    for week in windows["christmas"]:
        values[week] *= 1.4
    target = 1000.0 + 5.0 * 121 + 100.0 * math.sin(2.0 * math.pi * 121.0 / 52.0)
    values[121] = target * target_factor

    frames = {}
    for version, weeks in (("v1", range(1, 121)), ("v2", range(1, 122))):
        rows = []
        for week in weeks:
            if commodities:
                share = values[week] / len(commodities)
                for index, commodity in enumerate(commodities):
                    if week == 121:
                        factor = 1.3 if index == 0 else 0.7
                    else:
                        factor = 1.0
                    rows.append(
                        {
                            "week": week,
                            "store_id": "s1",
                            "product_id": f"p{index}",
                            "commodity_id": commodity,
                            "dollar": share * factor,
                            "units": share * factor / 10.0,
                        }
                    )
            else:
                rows.append(
                    {
                        "week": week,
                        "store_id": "s1",
                        "product_id": "p1",
                        "dollar": values[week],
                        "units": values[week] / 10.0,
                    }
                )
        fact = pd.DataFrame(rows)
        report = fact.groupby("week", as_index=False)[["dollar", "units"]].sum()
        frames[version, "warehouse"] = fact
        frames[version, "report"] = report
    return FrameSource(frames)


def test_expected_post_christmas_decline_is_explained_without_review():
    config = _calendar_config(dt.date(2021, 12, 28))
    result = run_qc(_seasonal_frames(config, target_factor=1.0), "v2", "v1", config)

    assert result.machine["calendar"]["declared"] is True
    assert result.temporal is not None
    assert not result.temporal.anomaly
    assert result.status in ("PASS", "PASS_WITH_EXPLANATION")
    assert result.machine["requires_investigation"] is False
    verified = [item for item in result.certificates if item.status == "VERIFIED"]
    assert any(item.scope == "dataset" for item in verified)


def test_decline_outside_seasonal_expectations_requires_review():
    config = _calendar_config(dt.date(2021, 7, 15))
    result = run_qc(
        _seasonal_frames(config, target_factor=0.6), "v2", "v1", config
    )

    assert result.temporal is not None
    assert result.temporal.anomaly
    national = next(
        item for item in result.temporal.series if item.series_id == "national"
    )
    assert national.residual < 0
    assert result.status == "INVESTIGATE"
    assert result.machine["requires_investigation"] is True


def test_opposing_commodity_errors_leave_gross_residual_visible():
    config = replace(
        _calendar_config(dt.date(2021, 12, 28)),
        temporal_entity_columns=("commodity_id",),
    )
    result = run_qc(
        _seasonal_frames(config, target_factor=1.0, commodities=("c1", "c2")),
        "v2",
        "v1",
        config,
    )
    national = next(
        item for item in result.temporal.series if item.series_id == "national"
    )
    first = next(
        item for item in result.temporal.series if item.series_id == "commodity_id:c1"
    )
    second = next(
        item for item in result.temporal.series if item.series_id == "commodity_id:c2"
    )
    assert first.residual > 0
    assert second.residual < 0
    assert abs(national.residual) < 0.1 * abs(first.residual)
    assert first.anomaly and second.anomaly
    assert any(
        finding["check"] == "temporal" and finding["outcome"] == "FAIL"
        for finding in result.machine["findings"]
    )
