from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("deltalake")

from deltalake import write_deltalake

from qc.cli import main
from qc.config import load_dataset_config
from qc.onboard import (
    assess_versions,
    config_from_proposal,
    profile_table,
)


def _rows(weeks):
    rows = []
    for week in weeks:
        for store, banner, state in (("S1", "B1", "NSW"), ("S2", "B2", "VIC")):
            rows.append(
                {
                    "week": week,
                    "store_id": store,
                    "product_id": "P1",
                    "banner_id": banner,
                    "state_id": state,
                    "dollar": 100.0 + week,
                    "units": 5,
                }
            )
    return rows


def _write_pair(path):
    write_deltalake(str(path), pd.DataFrame(_rows(range(1, 5))))
    write_deltalake(str(path), pd.DataFrame(_rows(range(1, 6))), mode="overwrite")


def test_profile_and_proposal(tmp_path):
    path = tmp_path / "fact"
    _write_pair(path)
    profile = profile_table(str(path))
    assert profile["week_column"] == "week"
    assert "dollar" in profile["metric_columns"]
    assert {"store_id", "product_id"} <= set(profile["entity_columns"])
    proposal = profile["proposed"]
    assert proposal["entity_key_columns"] == ["store_id", "product_id"]
    assert proposal["report_grain"] == ["banner_id", "state_id"]
    assert proposal["primary_metric"] == "dollar"
    blockers = [
        finding
        for finding in profile["findings"]
        if finding["severity"] == "BLOCKER"
    ]
    assert not blockers, blockers


def test_assess_versions_passes(tmp_path):
    path = tmp_path / "fact"
    _write_pair(path)
    profile = profile_table(str(path))
    config = config_from_proposal(profile["proposed"])
    assessment = assess_versions(str(path), config, 0, 1)
    assert assessment["contracts"]["status"] == "PASS"
    assert assessment["version_pair"]["shape"] == "NORMAL"
    assert not [
        finding
        for finding in assessment["findings"]
        if finding["severity"] == "BLOCKER"
    ]


def test_bad_table_reports_blockers(tmp_path):
    path = tmp_path / "bad"
    frame = pd.DataFrame(
        {
            "week": [1, 2, 4],
            "store_id": ["S1", "S1", "S1"],
            "dollar": ["a", "b", "c"],
        }
    )
    write_deltalake(str(path), frame)
    profile = profile_table(str(path))
    codes = {
        finding["code"]
        for finding in profile["findings"]
        if finding["severity"] == "BLOCKER"
    }
    assert "WEEK_NOT_CONTIGUOUS" in codes
    assert "MISSING_METRIC_COLUMNS" in codes


def test_duplicate_keys_reported(tmp_path):
    path = tmp_path / "dupes"
    frame = pd.DataFrame(
        {
            "week": [1, 1],
            "store_id": ["S1", "S1"],
            "dollar": [10.0, 11.0],
            "units": [1, 1],
        }
    )
    write_deltalake(str(path), frame)
    profile = profile_table(str(path))
    codes = {finding["code"] for finding in profile["findings"]}
    assert "DUPLICATE_KEYS" in codes


def test_cli_onboard_writes_loadable_config(tmp_path, capsys):
    path = tmp_path / "fact"
    _write_pair(path)
    out = tmp_path / "configs" / "dataset.yaml"
    assert (
        main(
            [
                "onboard",
                "--uri",
                str(path),
                "--previous",
                "0",
                "--current",
                "1",
                "--out",
                str(out),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"]["week_column"] == "week"
    assert payload["assessment"]["contracts"]["status"] == "PASS"

    config = load_dataset_config(out)
    assert config.week_column == "week"
    assert "dollar" in config.metric_columns
    assert tuple(config.entity_key_columns) == ("store_id", "product_id")
