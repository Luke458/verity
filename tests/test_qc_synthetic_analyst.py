from __future__ import annotations

import json
from pathlib import Path

import pytest

from qc.champion import eval_cases_from_suite, run_champion
from qc.cli import main
from qc.config import DatasetConfig
from qc.labels import CAUSE_BY_ORACLE, records_from_store
from qc.store import SqliteStore
from qc.synthetic_analyst import AnalystProfile, simulate_analyst
from qcgen.config import suite_config
from qcgen.scenarios import generate_suite

STAGES = ("source", "coded", "warehouse", "report")


@pytest.fixture(scope="module")
def suites(tmp_path_factory):
    root = tmp_path_factory.mktemp("analyst")
    train = generate_suite(suite_config("tiny", seed=41), root, 4, "train", STAGES)
    eval_dir = generate_suite(suite_config("tiny", seed=51), root, 2, "eval", STAGES)
    return train, eval_dir


def _oracle_cause(suite_dir: Path, run_id: str) -> str:
    scenario = run_id.split(":", 1)[1]
    manifest = json.loads(
        (Path(suite_dir) / scenario / "manifest.json").read_text()
    )
    expected_class = manifest["cases"][0]["expected_class"]
    return CAUSE_BY_ORACLE.get(expected_class, "UNKNOWN")


def test_perfect_profile_matches_oracle(tmp_path, suites):
    train_suite, _ = suites
    store = tmp_path / "perfect.db"
    summary = simulate_analyst(train_suite, store, profile="perfect", seed=1)

    assert summary["runs"] == 4
    assert summary["confirmed"] == 4
    assert summary["drafts"] == 0
    assert summary["corrections"] == 0

    records = records_from_store(store)
    assert len(records) == 4
    assert all(record.source == "synthetic" for record in records)
    assert all(record.text and "status=" in record.text for record in records)
    for record in records:
        assert record.labels["likely_cause"] == _oracle_cause(train_suite, record.run_id)

    with SqliteStore(store) as database:
        for record in records:
            latest = database.latest_outcome(record.run_id)
            assert bool(latest["confirmed"]) is True
            assert latest["provenance"] == "synthetic"


def test_corrections_replace_first_pass(tmp_path, suites):
    train_suite, _ = suites
    correcting = AnalystProfile(
        "correcting",
        cause_accuracy=0.5,
        origin_accuracy=1.0,
        severity_accuracy=1.0,
        confirm_rate=0.5,
        unknown_rate=0.0,
        correction_rate=1.0,
        investigation_false_positive=0.0,
        investigation_false_negative=0.0,
        max_delay_weeks=0,
    )
    store = tmp_path / "correcting.db"
    summary = simulate_analyst(train_suite, store, profile=correcting, seed=2)

    assert summary["corrections"] == summary["runs"]
    for record in records_from_store(store):
        assert record.labels["likely_cause"] == _oracle_cause(train_suite, record.run_id)


def test_sloppy_profile_produces_drafts(tmp_path, suites):
    train_suite, _ = suites
    store = tmp_path / "sloppy.db"
    summary = simulate_analyst(train_suite, store, profile="sloppy", seed=2)
    assert summary["drafts"] > 0
    records = records_from_store(store)
    assert 0 < len(records) <= summary["runs"]


def test_champion_with_synthetic_store_stays_ineligible(tmp_path, suites):
    train_suite, eval_dir = suites
    store = tmp_path / "champion.db"
    simulate_analyst(train_suite, store, profile="careful", seed=3)

    result = run_champion(
        records_from_store(store),
        eval_cases_from_suite(eval_dir),
        DatasetConfig(),
        gates={"min_overall_accuracy": 0.0, "min_cause_accuracy": 0.0},
    )
    assert result.champion is not None
    assert result.production_eligible is False
    assert result.provenance["labels"] == "synthetic"
    assert result.provenance["train_sources"] == ["synthetic"]


def test_cli_simulate_analyst(tmp_path, suites, capsys):
    train_suite, _ = suites
    store = tmp_path / "cli.db"
    assert (
        main(
            [
                "simulate-analyst",
                "--suite-dir",
                str(train_suite),
                "--store",
                str(store),
                "--profile",
                "typical",
                "--seed",
                "4",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "simulated analyst" in output
    assert "provenance: synthetic" in output
    assert store.exists()
