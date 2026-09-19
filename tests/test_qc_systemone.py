from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from qc.decisions import (
    DecisionSet,
    DecisionValue,
    RuleDecisionProvider,
    default_fields,
)
from qc.run import run_qc
from qc.systemone import (
    FallbackDecisionProvider,
    SystemOneDecisionProvider,
    answers_to_decisions,
    build_evidence_state,
    decision_to_systemone_answers,
)
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario
from qcgen.sources import ScenarioSource

CANNED_RESPONSE = {
    "model": "jev-test",
    "answers": {
        "likely_cause": {
            "type": "choice",
            "choice": "MISSING_STORES",
            "probabilities": {
                "MISSING_STORES": 0.9,
                "BACKFILL": 0.05,
                "UNKNOWN": 0.05,
            },
            "confidence": 0.9,
        },
        "likely_origin": {
            "type": "choice",
            "choice": "SOURCE",
            "probabilities": {"SOURCE": 0.8, "UNKNOWN": 0.2},
        },
        "severity": {
            "type": "score",
            "score": 1.6,
            "legend": {"0": "LOW", "1": "MEDIUM", "2": "HIGH", "3": "CRITICAL"},
            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.6, "3": 0.1},
            "confidence": 0.6,
        },
        "requires_investigation": {"type": "noul", "noul": 0.93},
    },
}


class _Handler(BaseHTTPRequestHandler):
    response = CANNED_RESPONSE
    status = 200

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.response).encode())

    def log_message(self, *args):  # silence test server
        return


@pytest.fixture()
def endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _Handler.response = CANNED_RESPONSE
    _Handler.status = 200
    yield f"http://127.0.0.1:{server.server_address[1]}/v1/systemone"
    server.shutdown()


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path_factory.mktemp("systemone"),
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    return run_qc(ScenarioSource(built.directory), "V0002", "V0001")


def test_systemone_provider_maps_answers(endpoint, case):
    decisions = SystemOneDecisionProvider(url=endpoint).decide(case)
    assert decisions.provider == "systemone"
    assert decisions.get("likely_cause").value == "MISSING_STORES"
    assert decisions.get("likely_cause").probability_kind == "provider_reported"
    flag = decisions.get("requires_investigation")
    assert flag.value is True
    assert flag.probabilities["True"] == pytest.approx(0.93)
    assert decisions.requires_investigation is True

    severity = decisions.get("severity")
    assert severity.kind == "score"
    assert severity.value == "HIGH"
    assert severity.index == pytest.approx(1.6)
    assert severity.probabilities["HIGH"] == pytest.approx(0.6)


def test_wire_format_roundtrip(case):
    rule = RuleDecisionProvider().decide(case)
    answers = decision_to_systemone_answers(rule)

    assert answers["requires_investigation"]["type"] == "noul"
    assert answers["likely_cause"]["type"] == "choice"
    assert answers["severity"]["type"] == "score"
    assert answers["severity"]["legend"]["2"] == "HIGH"
    assert set(answers["severity"]["probabilities"]) == {"0", "1", "2", "3"}

    parsed = answers_to_decisions(answers, default_fields(), case.run_id)
    for name in ("likely_cause", "likely_origin", "severity", "requires_investigation"):
        assert parsed.get(name).value == rule.get(name).value
    assert parsed.get("severity").index == pytest.approx(rule.get("severity").index)
    assert parsed.requires_investigation == rule.requires_investigation


def test_systemone_accepts_choice_format_for_score_field(endpoint, case):
    legacy = json.loads(json.dumps(CANNED_RESPONSE))
    legacy["answers"]["severity"] = {
        "type": "choice",
        "choice": "MEDIUM",
        "probabilities": {"LOW": 0.1, "MEDIUM": 0.7, "HIGH": 0.2},
    }
    _Handler.response = legacy
    decisions = SystemOneDecisionProvider(url=endpoint).decide(case)
    severity = decisions.get("severity")
    assert severity.kind == "score"
    assert severity.value == "MEDIUM"
    assert severity.index == pytest.approx(1.0)
    assert severity.probabilities["MEDIUM"] == pytest.approx(0.7)
    _Handler.response = CANNED_RESPONSE


def test_systemone_rejects_out_of_range_score(endpoint, case):
    broken = json.loads(json.dumps(CANNED_RESPONSE))
    broken["answers"]["severity"]["score"] = 9.0
    _Handler.response = broken
    with pytest.raises(ValueError):
        SystemOneDecisionProvider(url=endpoint).decide(case)
    _Handler.response = CANNED_RESPONSE


def test_systemone_http_error_raises(endpoint, case):
    _Handler.status = 500
    with pytest.raises(RuntimeError):
        SystemOneDecisionProvider(url=endpoint).decide(case)
    _Handler.status = 200


def test_systemone_missing_answer_raises(endpoint, case):
    _Handler.response = {"answers": {}}
    with pytest.raises(ValueError):
        SystemOneDecisionProvider(url=endpoint).decide(case)
    _Handler.response = CANNED_RESPONSE


def test_systemone_rejects_unknown_choice(endpoint, case):
    broken = json.loads(json.dumps(CANNED_RESPONSE))
    broken["answers"]["likely_cause"]["choice"] = "NOT_A_CLASS"
    _Handler.response = broken
    with pytest.raises(ValueError):
        SystemOneDecisionProvider(url=endpoint).decide(case)
    _Handler.response = CANNED_RESPONSE


def test_evidence_state_is_bounded(case):
    state = build_evidence_state(case, max_chars=2000)
    assert len(state) <= 2000
    payload = json.loads(state)
    assert "historical_revision" in payload
    assert "events" in payload
    assert "raw" not in payload


class _StubProvider:
    name = "stub"

    def __init__(self, decision: DecisionSet):
        self._decision = decision

    def decide(self, result):
        return self._decision


def _decision_set(cause: str, probability: float, requires: bool, provider: str):
    values = {
        "likely_cause": DecisionValue(
            "likely_cause",
            "choice",
            cause,
            {cause: probability, "UNKNOWN": 1.0 - probability},
            provider,
            "heuristic",
        ),
        "likely_origin": DecisionValue(
            "likely_origin", "choice", "UNKNOWN", {"UNKNOWN": 1.0}, provider, "heuristic"
        ),
        "severity": DecisionValue(
            "severity", "choice", "HIGH", {"HIGH": 1.0}, provider, "heuristic"
        ),
        "requires_investigation": DecisionValue(
            "requires_investigation",
            "boolean",
            requires,
            {"True": 0.9, "False": 0.1},
            provider,
            "heuristic",
        ),
    }
    return DecisionSet("run", provider, values, requires)


def test_fallback_escalates_when_uncertain(case):
    local = _StubProvider(_decision_set("UNKNOWN", 0.4, True, "local"))
    remote = _StubProvider(_decision_set("MISSING_STORES", 0.9, True, "remote"))
    provider = FallbackDecisionProvider(
        local=local, remote=remote, escalate_below=0.85
    )
    decision = provider.decide(case)
    assert decision.get("likely_cause").value == "MISSING_STORES"
    assert decision.provider == "stub->remote"


def test_fallback_keeps_confident_local(case):
    local = _StubProvider(_decision_set("MISSING_STORES", 0.95, True, "local"))
    remote = _StubProvider(_decision_set("BACKFILL", 0.9, True, "remote"))
    provider = FallbackDecisionProvider(local=local, remote=remote)
    assert provider.decide(case).get("likely_cause").value == "MISSING_STORES"


def test_fallback_does_not_escalate_clean_case(case):
    local = _StubProvider(_decision_set("UNKNOWN", 0.4, False, "local"))
    remote = _StubProvider(_decision_set("MISSING_STORES", 0.9, True, "remote"))
    provider = FallbackDecisionProvider(local=local, remote=remote)
    assert provider.decide(case).get("likely_cause").value == "UNKNOWN"
