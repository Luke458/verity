from __future__ import annotations

import json
import re

import pytest

from qc.reporting import render_machine, render_markdown, write_report
from qc.run import run_qc
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario
from qcgen.sources import ScenarioSource

STAGES = ("source", "coded", "warehouse", "report")


def _run(tmp_path_factory, family, registry=None, profile="small"):
    built = build_scenario(
        suite_config(profile), 0, tmp_path_factory.mktemp(family), family, STAGES
    )
    return run_qc(ScenarioSource(built.directory), "V0002", "V0001", registry=registry)


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    return _run(tmp_path_factory, "missing_stores")


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return _run(tmp_path_factory, "clean")


def test_report_leads_with_status_and_what_needs_a_decision(case):
    markdown = render_markdown(case)
    lines = markdown.splitlines()
    assert lines[0] == "# Verity QC: synthetic-retail, V0001 -> V0002"
    assert lines[2].startswith("**INVESTIGATE**")
    assert re.search(r"- \*\*Missing:\*\* \d+ stores? absent from week 104", markdown)
    assert "## Needs a decision" in markdown
    assert "| Missing from the latest week |" in markdown
    assert "Approve each closure (LATEST_WEEK_MISSING)" in markdown
    assert markdown.index("## Needs a decision") < markdown.index("## Structural changes")
    assert "## What this refresh could not see" in markdown
    # Rule labels are advisory and carry no invented probabilities.
    assert "MISSING_STORES" in markdown
    assert "probability" not in markdown.lower()


def test_machine_records_the_rule_that_decided_history(case):
    history = case.machine["historical_revision"]
    assert history["status"] == "INVESTIGATE"
    assert history["reason"] == "missing_entities"
    missing = case.machine["missing_entities"]["store"]
    assert missing["material"] and missing["entities"] >= 1


@pytest.mark.parametrize(
    ("family", "status", "reason"),
    [
        ("clean", "PASS", "none"),
        ("new_store_backfill", "INVESTIGATE", "unmatched_events"),
        ("coding_error", "INVESTIGATE", "unexplained_value_change"),
    ],
)
def test_history_reason_names_the_deciding_rule(tmp_path_factory, family, status, reason):
    history = _run(tmp_path_factory, family).machine["historical_revision"]
    assert (history["status"], history["reason"]) == (status, reason)


def test_stored_machine_output_renders_the_same_report(case, tmp_path):
    paths = write_report(case, tmp_path / "run")
    stored = json.loads(paths["machine"].read_text())
    assert render_machine(stored) == paths["markdown"].read_text()


def test_clean_refresh_report_is_short(clean):
    markdown = render_markdown(clean)
    assert markdown.splitlines()[2] == "**PASS**: No check failed."
    assert "- **History:** unchanged." in markdown
    assert "## Needs a decision" not in markdown
    assert "## Cause labels" not in markdown
    assert "No series moved outside its expected range" in markdown


def test_approved_closure_is_shown_with_its_approval(case, tmp_path_factory):
    stores = [
        event.entity_id for event in case.events
        if event.classification == "LATEST_WEEK_MISSING" and event.entity_type == "store"
    ]
    entry = {
        "dataset": "synthetic-retail", "confirmed": True, "approved_by": "analyst",
        "approved_at": "2000-01-01T00:00:00Z",
        "event_id": "close-stores", "classification": "LATEST_WEEK_MISSING",
        "entity_type": "store", "entity_ids": stores,
        "effective_from_week": 104, "effective_to_week": 104,
    }
    approved = _run(tmp_path_factory, "missing_stores", registry=[entry])
    markdown = render_markdown(approved)
    assert approved.status == "PASS_WITH_EXPLANATION"
    assert "Every failed check is covered by an approval" in markdown
    assert "explained by `close-stores`" in markdown
    assert f"| store {stores[0]} | missing from the latest week |" in markdown
    assert "| close-stores |" in markdown
    assert "## Needs a decision" not in markdown


def test_untrusted_identifiers_cannot_break_tables(case):
    machine = json.loads(json.dumps(case.machine, default=str))
    machine["events"][0]["entity_id"] = "S1 | injected\n# heading"
    markdown = render_machine(machine)
    assert "S1 \\| injected # heading" in markdown
    assert "\n# heading" not in markdown


def test_write_report_creates_files_and_refuses_overwrite(case, tmp_path):
    directory = tmp_path / "run-1"
    paths = write_report(case, directory)
    machine = json.loads(paths["machine"].read_text())
    assert machine["status"] == case.status
    assert "decision" in machine
    assert all(
        isinstance(check["passed"], bool) for check in machine["contracts"]["checks"]
    )
    with pytest.raises(FileExistsError):
        write_report(case, directory)
