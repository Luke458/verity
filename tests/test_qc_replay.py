from __future__ import annotations

import json

import pytest

from qc.cli import main
from qc.replay import ReplayCase, ReplayPlan, replay
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path_factory.mktemp("replay"),
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    return built.directory


def test_replay_single_case(scenario, tmp_path):
    plan = ReplayPlan(
        cases=(
            ReplayCase(
                "case-1", "V0001", "V0002", "2026-01-01", uri=str(scenario)
            ),
        )
    )
    report = replay(plan, tmp_path / "out")
    assert report.passed
    assert report.cases[0]["status"] == "INVESTIGATE"
    assert (tmp_path / "out" / "replay.json").exists()
    assert (tmp_path / "out" / "cases" / "case-1.json").exists()
    payload = json.loads((tmp_path / "out" / "replay.json").read_text())
    assert payload["cases"][0]["case_id"] == "case-1"
    assert payload["plan_sha256"]


def test_replay_records_missing_version(scenario, tmp_path):
    plan = ReplayPlan(
        cases=(
            ReplayCase(
                "case-bad", "V0001", "V9999", "2026-01-01", uri=str(scenario)
            ),
        )
    )
    report = replay(plan, tmp_path / "out-bad")
    assert not report.passed
    assert report.failed[0]["case_id"] == "case-bad"
    assert "not present" in report.failed[0]["error"]


def test_replay_case_validation():
    with pytest.raises(ValueError):
        ReplayCase("x", "V0001", "V0001", "2026-01-01")
    with pytest.raises(ValueError):
        ReplayPlan(
            cases=(
                ReplayCase("a", "V0001", "V0002", "2026-01-01"),
                ReplayCase("a", "V0002", "V0003", "2026-01-02"),
            )
        )


def test_replay_order_enforced():
    with pytest.raises(ValueError):
        ReplayPlan(
            cases=(
                ReplayCase("a", "V0003", "V0005", "2026-01-01"),
                ReplayCase("b", "V0004", "V0006", "2026-01-02"),
            )
        )
    with pytest.raises(ValueError):
        ReplayPlan(
            cases=(
                ReplayCase("a", "V0001", "V0002", "2026-02-01"),
                ReplayCase("b", "V0002", "V0003", "2026-01-01"),
            )
        )


def test_cli_replay(scenario, tmp_path, capsys):
    plan_path = tmp_path / "plan.json"
    ReplayPlan(
        cases=(
            ReplayCase("case-1", "V0001", "V0002", "2026-01-01", uri=str(scenario)),
        )
    ).save(plan_path)
    assert (
        main(["replay", "--plan", str(plan_path), "--out", str(tmp_path / "cli")])
        == 0
    )
    output = capsys.readouterr().out
    assert "replay cases=1 failed=0" in output
    assert "INVESTIGATE" in output
