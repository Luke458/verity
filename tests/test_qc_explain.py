from __future__ import annotations

import json

from qc.cli import main
from qc.run import run_qc
from qc.store import SqliteStore
from qc.versions import select_versions
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario
from qcgen.sources import ScenarioSource

STAGES = ("source", "coded", "warehouse", "report")


def _report(tmp_path):
    built = build_scenario(suite_config("tiny"), 0, tmp_path, "missing_stores", STAGES)
    source = ScenarioSource(built.directory)
    current, previous = select_versions(source.list_versions(), None, None)
    result = run_qc(source, current, previous)
    directory = tmp_path / "report"
    directory.mkdir()
    (directory / "report.json").write_text(json.dumps(result.machine, default=str))
    return directory, result


def test_explain_reads_report_directory(tmp_path, capsys):
    directory, result = _report(tmp_path)
    assert main(["explain", "--report", str(directory), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["machine"]["run_id"] == result.run_id

    assert main(["explain", "--report", str(directory)]) == 0
    output = capsys.readouterr().out
    assert f"status: {result.status}" in output
    # Only non-passing findings are listed, each with its disposition.
    assert "-> UNEXPLAINED_ANOMALY" in output


def test_explain_reads_sqlite_journal(tmp_path, capsys):
    directory, result = _report(tmp_path)
    store_path = tmp_path / "assessments.sqlite"
    with SqliteStore(store_path) as store:
        store.connection.execute(
            "INSERT INTO assessments VALUES (?, ?, ?, ?, ?, ?)",
            (
                "assessment-1",
                json.dumps({"config": {"name": result.dataset}}),
                result.status,
                json.dumps({"report.json": (directory / "report.json").read_text()}),
                json.dumps({}),
                json.dumps({"dataset": result.dataset, "status": result.status}),
            ),
        )
        store.connection.commit()
    assert main(["explain", "--store", str(store_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["assessment_id"] == "assessment-1"
    assert payload["machine"]["status"] == result.status
    assert payload["machine"]["findings"]


def test_explain_without_a_match_fails_cleanly(tmp_path, capsys):
    store_path = tmp_path / "empty.sqlite"
    SqliteStore(store_path).close()
    assert main(["explain", "--store", str(store_path)]) == 2
    assert "no matching assessment" in capsys.readouterr().err
