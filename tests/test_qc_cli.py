from __future__ import annotations

import json

from qc.cli import main
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario


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

