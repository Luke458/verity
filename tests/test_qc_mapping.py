from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("deltalake")

from deltalake import write_deltalake

from qc.cli import main
from qc.config import load_dataset_config
from qc.delta import DeltaSource
from qc.onboard import (
    assess_versions,
    config_from_proposal,
    profile_table,
)
from qc.run import run_qc
from qc.source import mapped

MAPPING = (
    "column_map:\n"
    "  week: wk\n"
    "  store_id: sty\n"
    "  product_id: pfc\n"
    "  dollar: dol\n"
)


def _config(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return load_dataset_config(path)


def _canonical_frame(target: int, latest_stores=("S1", "S2")) -> pd.DataFrame:
    rows = []
    for week in range(1, target + 1):
        value = 100.0 + 0.5 * ((week * 3) % 4)
        if week == target:
            value = 100.0 + 0.5 * (((target - 1) * 3) % 4)
        stores = ("S1", "S2") if week < target else latest_stores
        for store in stores:
            rows.append(
                {
                    "week": week,
                    "store_id": store,
                    "product_id": "P1",
                    "dollar": value,
                    "units": 10,
                }
            )
    return pd.DataFrame(rows)


def _production_frame(target: int, latest_stores=("S1", "S2")) -> pd.DataFrame:
    return _canonical_frame(target, latest_stores).rename(
        columns={
            "week": "wk",
            "store_id": "sty",
            "product_id": "pfc",
            "dollar": "dol",
        }
    )


def _write_versions(path, frames):
    for index, frame in enumerate(frames):
        if index == 0:
            write_deltalake(str(path), frame)
        else:
            write_deltalake(str(path), frame, mode="overwrite")


def test_config_column_map_roundtrip(tmp_path):
    config = _config(tmp_path, MAPPING)
    assert config.column_map_dict() == {
        "week": "wk",
        "store_id": "sty",
        "product_id": "pfc",
        "dollar": "dol",
    }
    assert config.source_rename() == {
        "wk": "week",
        "sty": "store_id",
        "pfc": "product_id",
        "dol": "dollar",
    }


def test_config_column_map_duplicates_rejected(tmp_path):
    with pytest.raises(ValueError):
        _config(tmp_path, "column_map:\n  product_id: pfc\n  store_id: pfc\n")


def test_mapped_source_renames_columns(tmp_path):
    path = tmp_path / "prod"
    write_deltalake(str(path), _production_frame(30))
    source = mapped(
        DeltaSource(uri=str(path)),
        {
            "week": "wk",
            "store_id": "sty",
            "product_id": "pfc",
            "dollar": "dol",
        },
    )
    frame = source.read_fact("0", "warehouse")
    assert {"week", "store_id", "product_id", "dollar"} <= set(frame.columns)
    assert "wk" not in frame.columns
    assert source.read_dim("0", "stores") is None
    assert source.warnings == []


def test_mapping_integration_matches_canonical(tmp_path):
    canonical_path = tmp_path / "canonical"
    production_path = tmp_path / "prod"
    _write_versions(
        canonical_path, [_canonical_frame(target) for target in (30, 31, 32)]
    )
    _write_versions(
        production_path, [_production_frame(target) for target in (30, 31, 32)]
    )

    canonical = run_qc(DeltaSource(uri=str(canonical_path)), "2", "1")
    config = _config(tmp_path, MAPPING)
    production_source = mapped(
        DeltaSource(uri=str(production_path)), config.column_map_dict()
    )
    production = run_qc(production_source, "2", "1")

    assert production.status == canonical.status
    assert (
        production.decisions.get("likely_cause").value
        == canonical.decisions.get("likely_cause").value
    )
    assert production.machine["historical_revision"]["status"] == canonical.machine[
        "historical_revision"
    ]["status"]


def test_mapping_fails_loud_when_production_column_missing(tmp_path):
    path = tmp_path / "no-metric"
    frames = [
        _canonical_frame(30).rename(columns={"dollar": "value"}),
        _canonical_frame(31).rename(columns={"dollar": "value"}),
    ]
    _write_versions(path, frames)
    source = mapped(DeltaSource(uri=str(path)), {"dollar": "dol"})
    result = run_qc(source, "1", "0")
    assert result.status == "DATA_CONTRACT_FAILURE"
    assert any("dol" in warning for warning in source.warnings)
    assert any(
        check.name.endswith("required_columns") for check in result.contracts.failed
    )


def test_onboard_aliases_detect_roles_and_emit_column_map(tmp_path):
    path = tmp_path / "prod"
    _write_versions(path, [_production_frame(30), _production_frame(31)])
    profile = profile_table(
        str(path),
        aliases={
            "wk": "week",
            "sty": "store_id",
            "pfc": "product_id",
            "dol": "dollar",
        },
    )
    assert profile["week_column"] == "week"
    assert "product_id" in profile["entity_columns"]
    assert "dollar" in profile["metric_columns"]
    proposal = profile["proposed"]
    assert proposal["column_map"] == {
        "week": "wk",
        "store_id": "sty",
        "product_id": "pfc",
        "dollar": "dol",
    }

    config = config_from_proposal(proposal)
    assessment = assess_versions(str(path), config, 0, 1)
    assert assessment["contracts"]["status"] == "PASS"


def test_cli_onboard_alias_writes_loadable_mapping(tmp_path, capsys):
    path = tmp_path / "prod"
    write_deltalake(str(path), _production_frame(30))
    out = tmp_path / "configs" / "prod.yaml"
    assert (
        main(
            [
                "onboard",
                "--uri",
                str(path),
                "--alias",
                "wk=week",
                "--alias",
                "sty=store_id",
                "--alias",
                "pfc=product_id",
                "--alias",
                "dol=dollar",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "column_map=" in output
    config = load_dataset_config(out)
    assert config.column_map_dict()["dollar"] == "dol"
    assert config.week_column == "week"


def test_cli_weekly_with_mapping(tmp_path):
    path = tmp_path / "prod"
    _write_versions(path, [_production_frame(target) for target in (30, 31, 32)])
    config_path = tmp_path / "prod-config.yaml"
    config_path.write_text("name: prod-mapped\n" + MAPPING)
    assert (
        main(
            [
                "weekly",
                "--uri",
                str(path),
                "--config",
                str(config_path),
                "--out",
                str(tmp_path / "weekly"),
            ]
        )
        == 0
    )
