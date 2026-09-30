"""Notification sink behaviour.

These tests pin the properties that make a sink safe to add to a scheduled run:
it is bounded, deterministic, idempotent for cached assessments, and it can
never change the assessment outcome. A sink that raised into the orchestrator, or
that silently truncated a payload, would be a correctness regression, not a
convenience issue.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qc.notify import (
    DEFAULT_NOTIFY_STATUSES,
    NotificationSpec,
    build_payload,
    dispatch_all,
    load_notification_specs,
)
from qc.weekly import WeeklyResult


def _result(**overrides) -> WeeklyResult:
    base = {
        "dataset": "retail",
        "current_version": "42",
        "previous_version": "41",
        "stage": "warehouse",
        "status": "INVESTIGATE",
        "assessment_id": "assessment-1",
        "attempt_id": "attempt-1",
        "elapsed_seconds": 3.5,
    }
    base.update(overrides)
    return WeeklyResult(**base)


def test_payload_carries_identity_and_status() -> None:
    payload = build_payload(_result())
    assert payload["notify_schema"] == 1
    assert payload["status"] == "INVESTIGATE"
    assert payload["assessment_id"] == "assessment-1"
    assert payload["attempt_id"] == "attempt-1"
    assert payload["previous_version"] == "41"
    assert payload["current_version"] == "42"
    assert payload["payload_digest"]


def test_payload_is_deterministic() -> None:
    # A redelivery of the same assessment must be byte-identical so an operator
    # can tell a repeat apart from a genuinely new finding.
    assert build_payload(_result()) == build_payload(_result())


def test_digest_tracks_content_not_key_order() -> None:
    first = build_payload(_result(notes=["a"]))
    second = build_payload(_result(notes=["b"]))
    assert first["payload_digest"] != second["payload_digest"]


def test_clean_status_is_suppressed() -> None:
    # A sink that pages on every clean week is a sink people learn to ignore.
    payload = build_payload(_result(status="PASS"))
    assert payload["suppressed"] == "status not in notify_on"
    assert "PASS" not in payload["notify_on"]


def test_pass_with_explanation_is_suppressed_by_default() -> None:
    payload = build_payload(_result(status="PASS_WITH_EXPLANATION"))
    assert payload["suppressed"] == "status not in notify_on"


def test_custom_status_list_is_honoured() -> None:
    payload = build_payload(_result(status="INCOMPLETE"), statuses=("INCOMPLETE",))
    assert "suppressed" not in payload
    assert payload["notify_on"] == ["INCOMPLETE"]


def test_cached_assessment_self_suppresses() -> None:
    # `qc weekly` is idempotent; a retry must not re-page for a cached result.
    payload = build_payload(_result(skipped=True))
    assert payload["suppressed"] == "assessment was cached; no new notification"


def test_optional_sections_are_dropped_not_truncated() -> None:
    result = _result(artifacts={f"key-{index}": "x" * 500 for index in range(40)})
    payload = build_payload(result, max_bytes=2048)
    assert len(json.dumps(payload)) <= 2048 + 128
    # Dropping is reported, so a reader can tell "did not fit" from "absent".
    assert "artifacts" in payload["omitted"]
    # Required identity survives the shrink.
    assert payload["assessment_id"] == "assessment-1"
    assert payload["status"] == "INVESTIGATE"


def test_required_fields_survive_shrink() -> None:
    payload = build_payload(_result(decision={"blob": "y" * 4000}), max_bytes=3000)
    assert payload["dataset"] == "retail"
    assert payload["payload_digest"]


def test_impossible_bound_refuses_rather_than_oversizing() -> None:
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        build_payload(_result(), max_bytes=64)


def test_non_positive_max_bytes_rejected() -> None:
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        build_payload(_result(), max_bytes=0)


def test_all_notes_are_carried() -> None:
    # Regression: an early version appended only the first note and stopped.
    payload = build_payload(_result(notes=["first", "second", "third"]))
    assert payload["notes"] == ["first", "second", "third"]


def test_file_sink_appends_one_line_per_delivery(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "weekly.jsonl"
    spec = NotificationSpec("file", str(target))
    for _ in range(3):
        outcome = dispatch_all([spec], _result())[0]
        assert outcome.delivered
    lines = target.read_text().strip().split("\n")
    assert len(lines) == 3
    digests = {json.loads(line)["payload_digest"] for line in lines}
    assert len(digests) == 1


def test_file_sink_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    # A path that cannot be created must not break the caller.
    spec = NotificationSpec("file", str(tmp_path / "afile" / "x.jsonl"))
    (tmp_path / "afile").write_text("not a directory")
    outcome = dispatch_all([spec], _result())[0]
    assert outcome.delivered is False
    assert outcome.detail


def test_unreachable_webhook_is_recorded_not_raised() -> None:
    spec = NotificationSpec("webhook", "http://127.0.0.1:9/never", timeout=1.0)
    outcome = dispatch_all([spec], _result())[0]
    assert outcome.delivered is False
    assert outcome.detail


def test_missing_token_environment_is_refused() -> None:
    # Credentials come from the environment; an unset token is a hard stop
    # rather than an unauthenticated request.
    spec = NotificationSpec(
        "webhook", "http://127.0.0.1:9/x", token_env="QC_NOTIFY_TOKEN_UNSET"
    )
    outcome = dispatch_all([spec], _result())[0]
    assert outcome.delivered is False
    assert "QC_NOTIFY_TOKEN_UNSET" in outcome.detail


def test_token_is_not_placed_in_the_payload() -> None:
    payload = build_payload(_result())
    assert "token" not in json.dumps(payload).lower()
    assert "authorization" not in json.dumps(payload).lower()


def test_fanout_reports_each_sink_independently(tmp_path: Path) -> None:
    good = NotificationSpec("file", str(tmp_path / "ok.jsonl"))
    bad = NotificationSpec("webhook", "http://127.0.0.1:9/x", timeout=1.0)
    outcomes = dispatch_all([good, bad], _result())
    assert [outcome.delivered for outcome in outcomes] == [True, False]


def test_empty_sink_list_does_nothing() -> None:
    assert dispatch_all([], _result()) == []


def test_unusable_payload_is_reported_per_sink() -> None:
    # The bound is impossible, so no sink is attempted and nothing raises.
    outcomes = dispatch_all(
        [NotificationSpec("file", "/tmp/unused.jsonl")], _result(), max_bytes=32
    )
    assert len(outcomes) == 1
    assert outcomes[0].delivered is False
    assert "payload error" in outcomes[0].detail


def test_unknown_sink_kind_rejected() -> None:
    with pytest.raises(ValueError, match="unknown notification sink kind"):
        NotificationSpec.from_dict({"kind": "carrier-pigeon", "target": "x"})


def test_webhook_requires_target() -> None:
    with pytest.raises(ValueError, match="requires a target"):
        NotificationSpec.from_dict({"kind": "webhook"})


def test_non_positive_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        NotificationSpec.from_dict(
            {"kind": "webhook", "target": "https://x", "timeout": 0}
        )


def test_suppressed_payload_reaches_no_sink(tmp_path: Path) -> None:
    # Regression: suppression used to be a payload field only, and the payload
    # was still delivered, so every clean week and cached retry re-paged.
    target = tmp_path / "weekly.jsonl"
    spec = NotificationSpec("file", str(target))
    for result in (_result(status="PASS"), _result(skipped=True)):
        outcome = dispatch_all([spec], result)[0]
        assert outcome.suppressed and not outcome.delivered
    assert not target.exists()


def test_shrink_drops_as_many_sections_as_needed() -> None:
    # Regression: a payload that needed two sections dropped used to drop none
    # and then refuse, losing the alert entirely.
    result = _result(
        decision={"x": "a" * 3000},
        drift={"y": "b" * 3000},
        artifacts={"c": "c" * 3000},
    )
    payload = build_payload(result, max_bytes=5000)
    assert len(json.dumps(payload, sort_keys=True)) <= 5000
    assert {"artifacts", "decision"} <= set(payload["omitted"])
    assert "drift" in payload and "drift" not in payload["omitted"]


def test_webhook_refuses_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    # A redirect would forward the Authorization header to another origin.
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server API
            seen.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("QC_NOTIFY_TEST_TOKEN", "secret")
        spec = NotificationSpec(
            "webhook",
            f"http://127.0.0.1:{server.server_port}/hook",
            timeout=5.0,
            token_env="QC_NOTIFY_TEST_TOKEN",
        )
        outcome = dispatch_all([spec], _result())[0]
    finally:
        server.shutdown()
    assert outcome.delivered is False
    assert "302" in outcome.detail
    assert seen == ["/hook"]


@pytest.mark.parametrize(
    "target",
    ["http://hooks.example.com/x", "file:///etc/passwd", "ftp://example.com/x"],
)
def test_webhook_requires_https_off_loopback(target: str) -> None:
    with pytest.raises(ValueError, match="https"):
        NotificationSpec.from_dict({"kind": "webhook", "target": target})


def test_spec_file_accepts_both_shapes(tmp_path: Path) -> None:
    listed = tmp_path / "a.json"
    listed.write_text(
        json.dumps({"sinks": [{"kind": "file", "target": str(tmp_path / "x")}]})
    )
    wrapped = tmp_path / "b.json"
    wrapped.write_text(json.dumps([{"kind": "stdout"}]))
    assert len(load_notification_specs(listed)) == 1
    assert len(load_notification_specs(wrapped)) == 1


def test_spec_file_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"sinks": "not-a-list"}))
    with pytest.raises(ValueError, match="must be a list"):
        load_notification_specs(path)


def test_default_statuses_cover_actionable_outcomes() -> None:
    # Regression guard: the default must keep paging on a contract failure and
    # on an incomplete assessment, and stay silent on a clean week.
    assert "INVESTIGATE" in DEFAULT_NOTIFY_STATUSES
    assert "DATA_CONTRACT_FAILURE" in DEFAULT_NOTIFY_STATUSES
    assert "INCOMPLETE" in DEFAULT_NOTIFY_STATUSES
    assert "PASS" not in DEFAULT_NOTIFY_STATUSES
    assert "PASS_WITH_EXPLANATION" not in DEFAULT_NOTIFY_STATUSES
