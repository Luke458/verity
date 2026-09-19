from __future__ import annotations

import json
import shlex
import sys

import pytest

from qc.agent import (
    CommandAgent,
    InvestigationBrief,
    NullAgent,
    build_investigation_brief,
    parse_investigation_result,
)
from qc.decisions import FeatureEncoder
from qc.evidence_query import ALLOWED_QUERIES
from qc.run import run_qc
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario
from qcgen.sources import ScenarioSource

STUB = """
import json
import sys

brief = json.load(sys.stdin)
cause = brief["candidate_causes"][0]["cause"]
print(json.dumps({
    "root_cause": cause,
    "confidence": 0.82,
    "evidence_ids": brief["evidence_ids"][:2],
    "recommended_actions": ["Compare the source extract."],
    "summary": f"Stub concluded {cause}.",
    "follow_up_questions": brief["open_questions"][:1],
}))
"""


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path_factory.mktemp("agent"),
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    result = run_qc(ScenarioSource(built.directory), "V0002", "V0001")
    return result, build_investigation_brief(result, result.decisions)


def test_brief_is_actionable(case):
    result, brief = case
    assert brief.run_id == result.run_id
    assert brief.candidate_causes
    assert brief.candidate_causes[0]["cause"] == "MISSING_STORES"
    assert brief.findings
    assert brief.open_questions
    assert brief.recommended_queries
    assert brief.evidence_ids
    assert brief.constraints["read_only"] is True
    assert "root_cause" in brief.response_schema
    assert brief.tool_calls
    assert brief.tool_calls[0]["tool"] == "lifecycle_changes"
    assert all(call["tool"] in ALLOWED_QUERIES for call in brief.tool_calls)
    json.dumps(brief.to_dict())


def test_null_agent_returns_brief_only(case):
    _, brief = case
    result = NullAgent().investigate(brief)
    assert result.root_cause == "UNKNOWN"
    assert result.confidence == 0.0
    assert result.follow_up_questions == brief.open_questions


def test_command_agent_roundtrip(case, tmp_path):
    _, brief = case
    script = tmp_path / "agent.py"
    script.write_text(STUB)
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    result = CommandAgent(command).investigate(brief)
    assert result.agent == "command"
    assert result.root_cause == "MISSING_STORES"
    assert result.confidence == pytest.approx(0.82)
    assert result.evidence_ids
    assert result.recommended_actions


def test_command_agent_failure(case):
    _, brief = case
    with pytest.raises(RuntimeError):
        CommandAgent("exit 1").investigate(brief)


def test_parse_rejects_invalid_payloads():
    with pytest.raises(ValueError):
        parse_investigation_result({"confidence": 0.5}, "run", "agent")
    with pytest.raises(ValueError):
        parse_investigation_result(
            {"root_cause": "X", "confidence": 2.0, "summary": "s"}, "run", "agent"
        )


def test_feature_vector_matches_encoder(case):
    result, _ = case
    encoder = FeatureEncoder()
    assert len(encoder.encode(result)) == len(encoder.feature_names)
