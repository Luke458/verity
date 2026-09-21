from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from optional.laya_service import pack
from qc.laya import MODEL, REVISION, LayaDecisionProvider
from qc.run import run_qc
from tests.test_reliability import CONFIG, HandSource


class Tokenizer:
    mask_token = "[MASK]"

    def __call__(self, text, **kwargs):
        return {"input_ids": text.split()}


class Agent:
    tok = Tokenizer()
    cfg = {"max_len": 64, "head_max_len": 32}

    def _to_internal(self, q):
        return {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria")}


def test_laya_failure_retains_integrity_review(monkeypatch):
    provider = LayaDecisionProvider()
    monkeypatch.setattr(
        provider,
        "_post",
        lambda payload: {"metadata": {"model": MODEL, "revision": "wrong"}},
    )
    source = HandSource()
    source.frames["v2", "report"]["dollar"] *= 2
    result = run_qc(source, "v2", "v1", CONFIG, decision_provider=provider)
    assert result.status == "INVESTIGATE"
    assert result.decisions.requires_investigation
    assert result.machine["provider_availability"]["status"] == "UNAVAILABLE"
    assert provider.artifact_identity["revision"] == REVISION


def test_pack_refuses_truncated_or_missing_mandatory_evidence(monkeypatch):
    # Keep mandatory tests independent of Torch/Laya installation.
    import sys

    monkeypatch.setitem(
        sys.modules,
        "laya.common",
        SimpleNamespace(
            render_options=lambda q: list(q.get("crit") or ["false", "true"])
        ),
    )
    questions = {
        "q": {
            "type": "choice",
            "instructions": "Choose",
            "criteria": {"yes": "yes", "no": "no"},
        }
    }
    for state in (
        {"truncated": True},
        {"status": "PASS"},
        {"findings": [{"outcome": "FAIL", "detail": "word " * 100}]},
    ):
        with pytest.raises(ValueError):
            pack(Agent(), json.dumps(state), questions)
    state, omitted = pack(
        Agent(),
        json.dumps({"status": "PASS", "findings": [], "events": ["word " * 100]}),
        questions,
    )
    assert json.loads(state)["status"] == "PASS"
    assert "events" in omitted


def test_worker_timeout_terminates_and_refuses_reuse():
    from optional.laya_service import exchange

    class Pipe:
        def send(self, value):
            self.value = value

        def poll(self, timeout):
            assert timeout == 0.01
            return False

    class Process:
        alive = True
        killed = False

        def is_alive(self):
            return self.alive

        def terminate(self):
            pass

        def join(self, timeout):
            assert timeout == 5

        def kill(self):
            self.alive = False
            self.killed = True

    process = Process()
    with pytest.raises(TimeoutError, match="deadline"):
        exchange(Pipe(), process, {}, 0.01)
    assert process.killed
    with pytest.raises(RuntimeError, match="unavailable"):
        exchange(Pipe(), process, {}, 0.01)
