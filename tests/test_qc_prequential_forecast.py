from __future__ import annotations

import json

import pandas as pd
import pytest

from qc.cli import main
from qc.config import DatasetConfig
from qc.prequential import (
    LoadSnapshot,
    forecast_across_loads,
    prequential_sequence,
)


def _frame(target: int, final_value: float | None = None) -> pd.DataFrame:
    rows = []
    for week in range(1, target + 1):
        value = 100.0 + 0.1 * week
        if week == target and final_value is not None:
            value = final_value
        rows.append({"week": week, "dollar": value, "units": 10})
    return pd.DataFrame(rows)


def _load(target: int, value: float | None = None, available_on: int | None = None):
    return LoadSnapshot(
        load_id=f"L{target}",
        target_week=target,
        frame=_frame(target, value),
        available_on=target if available_on is None else available_on,
    )


def test_point_in_time_pool_growth():
    loads = [_load(target) for target in range(30, 35)]
    result = forecast_across_loads(loads, DatasetConfig(), min_samples=2)

    assert result.loads == 5
    assert result.records_committed == 5  # one national series per load
    by_load: dict[str, list] = {}
    for item in result.evidence:
        by_load.setdefault(item.load_id, []).append(item)
    assert all(len(items) == 1 for items in by_load.values())

    first = by_load["L30"][0]
    assert first.pool_records == 0
    assert first.pool_percentile is None

    second = by_load["L31"][0]
    assert second.pool_records == 1  # L30 only, still below min_samples
    assert second.pool_percentile is None

    third = by_load["L32"][0]
    assert third.pool_records == 2
    assert third.pool_percentile is not None


def test_same_week_availability_is_excluded():
    loads = [_load(30, available_on=31), _load(31, available_on=31)]
    result = forecast_across_loads(loads, min_samples=1)
    second = next(item for item in result.evidence if item.load_id == "L31")
    assert second.pool_records == 0


def test_sequence_validation():
    with pytest.raises(ValueError):
        forecast_across_loads([_load(30), _load(30)])
    with pytest.raises(ValueError):
        forecast_across_loads([_load(31), _load(32, available_on=30)])


def test_insufficient_history_is_recorded_not_scored():
    result = forecast_across_loads([_load(10)], min_samples=1)
    item = result.evidence[0]
    assert item.status == "INSUFFICIENT_HISTORY"
    assert result.records_committed == 0


def test_prequential_flag_on_anomalous_load():
    loads = [_load(target) for target in range(30, 34)]
    loads.append(_load(34, value=40.0))
    result = forecast_across_loads(loads, min_samples=2)
    last = next(item for item in result.evidence if item.load_id == "L34")
    assert last.pool_records == 4
    assert last.pool_percentile == pytest.approx(0.0)
    assert last.flag == "prequential_lower"


class _DictSource:
    def __init__(self, frames, stage: str = "warehouse"):
        self.frames = frames
        self.stage = stage

    def resolve_stage(self, version, preferred):
        return self.stage

    def available_stages(self, version):
        return [self.stage]

    def list_versions(self):
        return list(self.frames)

    def read_fact(self, version, stage):
        return self.frames[version]


def test_prequential_sequence_builds_loads_from_versions():
    frames = {str(index): _frame(target) for index, target in enumerate((30, 31, 32))}
    result = prequential_sequence(
        _DictSource(frames), ["0", "1", "2"], DatasetConfig(), min_samples=1
    )
    assert result.loads == 3
    assert [item.load_id for item in result.evidence] == [
        "synthetic-retail:0",
        "synthetic-retail:1",
        "synthetic-retail:2",
    ]


def test_prequential_sequence_rejects_decreasing_targets():
    frames = {"0": _frame(31), "1": _frame(30)}
    with pytest.raises(ValueError):
        prequential_sequence(_DictSource(frames), ["0", "1"], DatasetConfig())


def test_cli_prequential_forecast(tmp_path, capsys):
    pytest.importorskip("deltalake")
    from deltalake import write_deltalake

    path = tmp_path / "fact"
    for index, target in enumerate((30, 31, 32)):
        frame = _frame(target)
        if index == 0:
            write_deltalake(str(path), frame)
        else:
            write_deltalake(str(path), frame, mode="overwrite")
    assert (
        main(
            [
                "prequential-forecast",
                "--uri",
                str(path),
                "--versions",
                "0,1,2",
                "--min-samples",
                "1",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["loads"] == 3
    assert payload["records_committed"] == 3
