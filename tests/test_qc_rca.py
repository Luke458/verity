from __future__ import annotations

import pytest

from qc.cli import main
from qc.rca import investigate
from qc.run import run_qc
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario
from qcgen.sources import ScenarioSource


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path_factory.mktemp("rca"),
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    return run_qc(ScenarioSource(built.directory), "V0002", "V0001")


def test_default_trace_is_bounded_and_cited(case):
    result = investigate(case, max_steps=2)
    assert result.steps
    assert result.steps[0].tool == "lifecycle_changes"
    assert result.steps[0].args.get("classification") == "LATEST_WEEK_MISSING"
    assert len(result.steps) <= 2
    assert result.confirmed is False
    assert result.stop_reason in ("completed", "max_steps", "selector_done")
    assert result.steps[0].total_rows >= 1


def test_selector_can_choose_another_tool(case):
    def selector(result, trace, remaining):
        if "contract_failures" in remaining:
            return "contract_failures", {}
        return None

    result = investigate(case, selector=selector, max_steps=1)
    assert [step.tool for step in result.steps] == ["contract_failures"]
    assert result.stop_reason == "max_steps"


def test_selector_cannot_repeat_or_invent_tools(case):
    def repeating(result, trace, remaining):
        return "lifecycle_changes", {}

    with pytest.raises(ValueError):
        investigate(case, selector=repeating, max_steps=3)

    def inventing(result, trace, remaining):
        return "drop_table", {}

    with pytest.raises(ValueError):
        investigate(case, selector=inventing, max_steps=1)


def test_cli_rca(case, tmp_path, capsys):
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    assert (
        main(["rca", "--scenario-dir", str(built.directory), "--steps", "2"]) == 0
    )
    output = capsys.readouterr().out
    assert "RCA V0001->V0002" in output
    assert "lifecycle_changes" in output
