from __future__ import annotations

import json

import pytest

from qc.reporting import render_markdown, write_report
from qc.run import run_qc
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario
from qcgen.sources import ScenarioSource


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path_factory.mktemp("reporting"),
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    return run_qc(ScenarioSource(built.directory), "V0002", "V0001")


def test_render_markdown_contains_sections(case):
    result = case
    markdown = render_markdown(result)
    for heading in (
        f"# QC RUN {result.run_id}",
        "**STATUS INVESTIGATE**",
        "## Versions",
        "## Historical revision",
        "## Latest week",
        "## Decision",
        "## Reasons",
    ):
        assert heading in markdown
    assert "MISSING_STORES" in markdown
    assert "store:S" in markdown or "store" in markdown


def test_write_report_creates_files_and_refuses_overwrite(case, tmp_path):
    result = case
    directory = tmp_path / "run-1"
    paths = write_report(result, directory)
    assert paths["markdown"].exists()
    assert paths["machine"].exists()

    machine = json.loads(paths["machine"].read_text())
    assert machine["status"] == result.status
    assert "decision" in machine
    assert (paths["machine"].parent / "report.md").read_text() == paths[
        "markdown"
    ].read_text()

    with pytest.raises(FileExistsError):
        write_report(result, directory)
