from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from qc.config import DatasetConfig
from qc.delta import DeltaSource, describe_delta_table
from qc.run import run_qc

pytest.importorskip("deltalake")

from deltalake import write_deltalake  # noqa: E402


def _rows(weeks, stores, metric=10.0):
    return [
        {
            "week": week,
            "store_id": store,
            "product_id": "P1",
            "dollar": metric,
            "units": 1,
        }
        for week in weeks
        for store in stores
    ]


def _write_versions(tmp_path):
    path = tmp_path / "fact"
    write_deltalake(str(path), pd.DataFrame(_rows(range(1, 5), ("S1", "S2"))))
    # Current version drops S2 from the new week 5.
    current = _rows(range(1, 6), ("S1",)) + _rows(range(1, 5), ("S2",))
    write_deltalake(str(path), pd.DataFrame(current), mode="overwrite")
    return path


def test_describe_delta_table(tmp_path):
    path = _write_versions(tmp_path)
    info = describe_delta_table(str(path))
    assert info["current_version"] == 1
    assert [entry["version"] for entry in info["history"]] == [0, 1]
    assert info["rows_at_current"] == len(_rows(range(1, 6), ("S1",))) + len(
        _rows(range(1, 5), ("S2",))
    )
    names = {field["name"] for field in info["schema"]}
    assert {"week", "store_id", "product_id", "dollar", "units"} <= names


def test_delta_source_reads_versions(tmp_path):
    path = _write_versions(tmp_path)
    source = DeltaSource(uri=str(path))
    assert source.list_versions() == ["0", "1"]
    assert source.available_stages("1") == ["warehouse"]
    assert source.resolve_stage("1", ("report", "warehouse")) == "warehouse"
    previous = source.read_fact("0", "warehouse")
    current = source.read_fact("1", "warehouse")
    assert previous["week"].max() == 4
    assert current["week"].max() == 5
    assert set(current.loc[current["week"] == 5, "store_id"]) == {"S1"}


def test_run_qc_over_delta_versions(tmp_path):
    path = _write_versions(tmp_path)
    source = DeltaSource(uri=str(path))
    config = replace(DatasetConfig(), temporal_enabled=False)
    result = run_qc(source, "1", "0", config)

    assert result.status == "INVESTIGATE"
    assert result.decisions is not None
    assert result.decisions.get("likely_cause").value == "MISSING_STORES"
    assert result.decisions.requires_investigation is True
    assert any(
        event.classification == "LATEST_WEEK_MISSING" for event in result.events
    )
    assert result.machine["version_pair"]["shape"] == "NORMAL"
