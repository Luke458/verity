from __future__ import annotations

import json
import shlex
import sys

from qc.cli import main
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario, generate_suite

AGENT_STUB = """
import json
import sys

brief = json.load(sys.stdin)
cause = brief["candidate_causes"][0]["cause"]
print(json.dumps({
    "root_cause": cause,
    "confidence": 0.8,
    "evidence_ids": brief["evidence_ids"][:1],
    "recommended_actions": ["Check the source extract."],
    "summary": f"Stub concluded {cause}.",
    "follow_up_questions": [],
}))
"""


def _scenario(tmp_path):
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    return built.directory


def test_cli_run_human_output(tmp_path, capsys):
    scenario_dir = _scenario(tmp_path)
    assert main(["run", "--scenario-dir", str(scenario_dir)]) == 0
    output = capsys.readouterr().out
    assert "QC RUN V0001->V0002" in output
    assert "STATUS INVESTIGATE" in output
    assert "latest_week_missing" in output


def test_cli_run_json_output(tmp_path, capsys):
    scenario_dir = _scenario(tmp_path)
    assert main(["run", "--scenario-dir", str(scenario_dir), "--json"]) == 0
    machine = json.loads(capsys.readouterr().out)
    assert machine["status"] == "INVESTIGATE"
    assert machine["contracts"]["status"] == "PASS"
    assert machine["version_pair"]["overlap_end"] == 29
    assert machine["version_pair"]["new_periods"] == [30]
    assert machine["decision"]["values"]["likely_cause"]["value"]


def test_cli_train(tmp_path, capsys):
    suite_dir = generate_suite(
        suite_config("tiny"),
        tmp_path,
        4,
        "cli-train",
        ("source", "coded", "warehouse", "report"),
    )
    out = tmp_path / "artifact"
    assert (
        main(
            [
                "train",
                "--suite-dir",
                str(suite_dir),
                "--out",
                str(out),
                "--validation-fraction",
                "0.25",
            ]
        )
        == 0
    )
    assert (out / "provider.json").exists()
    assert (out / "metrics.json").exists()
    assert "trained decision provider" in capsys.readouterr().out


def test_cli_investigate_and_incidents(tmp_path, capsys):
    scenario_dir = _scenario(tmp_path)
    script = tmp_path / "agent.py"
    script.write_text(AGENT_STUB)
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    store = tmp_path / "incidents.jsonl"

    assert (
        main(
            [
                "investigate",
                "--scenario-dir",
                str(scenario_dir),
                "--agent-cmd",
                command,
                "--incident-store",
                str(store),
                "--save-draft",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "INVESTIGATION" in output
    assert "MISSING_STORES" in output
    assert store.exists()

    assert main(["incidents", "list", "--store", str(store)]) == 0
    assert "incidents=1" in capsys.readouterr().out


def test_cli_decide_json(tmp_path, capsys):
    scenario_dir = _scenario(tmp_path)
    assert main(["decide", "--scenario-dir", str(scenario_dir), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"]["requires_investigation"] is True
    assert payload["decision"]["provider"] == "rule"


def test_cli_store_and_drift(tmp_path, capsys):
    from qc.prequential import CalibrationRecord, PrequentialStore

    scenario_dir = _scenario(tmp_path)
    database = tmp_path / "store.db"
    assert (
        main(
            [
                "store",
                "add-run",
                "--store",
                str(database),
                "--scenario-dir",
                str(scenario_dir),
                "--root-cause",
                "MISSING_STORES",
                "--confirmed",
                "--resolution",
                "supplier extract fixed",
            ]
        )
        == 0
    )
    assert main(["store", "list", "--store", str(database)]) == 0
    listing = capsys.readouterr().out
    assert "runs=1" in listing
    assert "confirmed" in listing

    assert (
        main(
            [
                "store",
                "incidents",
                "--store",
                str(database),
                "--scenario-dir",
                str(scenario_dir),
            ]
        )
        == 0
    )
    incidents = capsys.readouterr().out
    assert "similarity=" in incidents
    assert "MISSING_STORES" in incidents
    assert "SOURCE" in incidents  # inherited from the run's decisions

    calibration_path = tmp_path / "calibration.jsonl"
    PrequentialStore(calibration_path).add(
        [
            CalibrationRecord(
                series_id="national",
                target_week=index + 1,
                available_on=index + 1,
                residual=0.1,
            )
            for index in range(20)
        ]
    )
    assert (
        main(
            [
                "drift",
                "--store",
                str(calibration_path),
                "--as-of",
                "25",
                "--target-week",
                "26",
            ]
        )
        == 0
    )
    drift_output = capsys.readouterr().out
    assert "drift status=STABLE" in drift_output
