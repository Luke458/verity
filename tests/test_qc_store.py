from __future__ import annotations

import json
import sqlite3

import pytest

from qc.store import SCHEMA_VERSION, SqliteStore


@pytest.fixture()
def store(tmp_path):
    with SqliteStore(tmp_path / "qc.db") as database:
        yield database


def test_runs_are_immutable(store):
    store.record_run("run-1", "demo", "PASS", {"status": "PASS"})
    with pytest.raises(ValueError):
        store.record_run("run-1", "demo", "PASS", {"status": "PASS"})
    rows = store.list_runs()
    assert [row["run_id"] for row in rows] == ["run-1"]
    assert rows[0]["status"] == "PASS"


def test_record_result_end_to_end(tmp_path):
    from qc.run import run_qc
    from qcgen.config import suite_config
    from qcgen.scenarios import build_scenario
    from qcgen.sources import ScenarioSource

    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    result = run_qc(ScenarioSource(built.directory), "V0002", "V0001")

    with SqliteStore(tmp_path / "runs.db") as database:
        database.record_result(result)
        row = database.connection.execute(
            "SELECT status, payload FROM runs WHERE run_id = ?", (result.run_id,)
        ).fetchone()
    assert row["status"] == result.status
    assert json.loads(row["payload"])["findings"]


def test_older_store_is_backed_up_and_upgraded(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as legacy:
        legacy.execute(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, dataset TEXT, created TEXT, "
            "status TEXT, features TEXT, feature_version INTEGER, evidence_text TEXT, "
            "text_version INTEGER, payload TEXT)"
        )
        legacy.execute(
            "INSERT INTO runs (run_id, dataset, created, status, payload) "
            "VALUES ('old', 'demo', '2026-01-01T00:00:00+00:00', 'PASS', '{}')"
        )
        legacy.execute("PRAGMA user_version = 6")
    with SqliteStore(path) as database:
        assert [row["run_id"] for row in database.list_runs()] == ["old"]
        database.record_run("new", "demo", "PASS", {})
        version = database.connection.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION
    assert list(tmp_path.glob("legacy.db.v6.backup-*"))


def test_newer_store_is_refused(tmp_path):
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as future:
        future.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(ValueError, match="unsupported store schema"):
        SqliteStore(path)
