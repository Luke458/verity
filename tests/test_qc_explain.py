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
    periods = []
    for package in result.evidence_packages:
        payload = package.to_dict()
        payload["certificates"] = [
            certificate.to_dict()
            for certificate in result.certificates
            if certificate.evidence_digest == package.digest
        ]
        periods.append(payload)
    (directory / "evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "assessment_id": result.assessment_id,
                "periods": periods,
            },
            default=str,
        )
    )
    return directory, result


def test_explain_reads_report_directory(tmp_path, capsys):
    directory, result = _report(tmp_path)
    assert (
        main(["explain", "--report", str(directory), "--json"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["machine"]["run_id"] == result.run_id
    assert payload["evidence"]["schema_version"] == 2
    assert payload["evidence"]["periods"]

    assert main(["explain", "--report", str(directory)]) == 0
    output = capsys.readouterr().out
    assert "status:" in output
    assert "ledger:" in output


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
                json.dumps(
                    {"evidence.json": (directory / "evidence.json").read_text()}
                ),
                json.dumps({}),
                json.dumps(result.machine, default=str),
            ),
        )
        store.connection.commit()

    assert (
        main(
            [
                "explain",
                "--store",
                str(store_path),
                "--assessment",
                "assessment-1",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["assessment_id"] == "assessment-1"
    assert payload["evidence"]["periods"]


def test_explain_reports_missing_assessment(tmp_path, capsys):
    store_path = tmp_path / "assessments.sqlite"
    SqliteStore(store_path).close()
    assert (
        main(
            [
                "explain",
                "--store",
                str(store_path),
                "--dataset",
                "unknown",
                "--json",
            ]
        )
        == 2
    )
    assert "no matching assessment" in capsys.readouterr().err
