"""Notice matching and draft registry entries (qc.notices, `qc notices`)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from qc.cli import main
from qc.notices import NONE, Candidate, SystemOneMatcher, load_notices
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario


def _scenario(tmp_path, family):
    built = build_scenario(suite_config("tiny"), 0, tmp_path / family, family, ("source", "coded", "warehouse", "report"))
    return built, built.oracle["cases"][0]


def _backfill_dispositions(capsys, scenario_dir, *extra):
    """Dispositions of the findings the backfill raises (other findings are not its to clear)."""
    capsys.readouterr()
    assert main(["run", "--scenario-dir", str(scenario_dir), "--json", "--no-decisions", *extra]) == 0
    findings = json.loads(capsys.readouterr().out)["findings"]
    return {
        f["check"]: f["disposition"]
        for f in findings
        if f["check"] in ("historical_revision", "historical_event") and f["outcome"] != "PASS"
    }


def test_backfill_notice_drafts_an_entry_that_explains_once_approved(tmp_path, capsys):
    built, case = _scenario(tmp_path, "new_store_backfill")
    store = case["affected"]["stores"][0]
    start, end = case["details"]["backfill_start"], case["details"]["backfill_end"]
    notices = tmp_path / "notices.txt"
    notices.write_text(
        f"New store {store} added to the feed with back history (weeks {start}-{end}).\n"
        "Reminder: month-end close moves to the 3rd.\n"
    )
    drafts = tmp_path / "drafts.json"
    capsys.readouterr()
    assert main(["notices", "--scenario-dir", str(built.directory), "--notices", str(notices),
                 "--out", str(drafts), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [m["action"] for m in report["matches"]] == ["registry_draft", "none"]
    entries = json.loads(drafts.read_text())["events"]
    assert len(entries) == 1 and entries[0]["entity_ids"] == [store]
    assert entries[0]["confirmed"] is False and entries[0]["approved_by"] is None

    # A draft explains nothing until a person approves it.
    unexplained = {"historical_revision": "UNEXPLAINED_ANOMALY"}
    assert _backfill_dispositions(capsys, built.directory) == unexplained
    assert _backfill_dispositions(capsys, built.directory, "--registry", str(drafts)) == unexplained
    entries[0].update(approved_by="analyst", approved_at="2000-01-01T00:00:00+00:00", confirmed=True)
    drafts.write_text(json.dumps({"events": entries}))
    assert _backfill_dispositions(capsys, built.directory, "--registry", str(drafts)) == {
        "historical_revision": "HUMAN_APPROVED",
        "historical_event": "HUMAN_APPROVED",
    }


def _run(capsys, scenario_dir, *extra):
    capsys.readouterr()
    assert main(["run", "--scenario-dir", str(scenario_dir), "--json", "--no-decisions", *extra]) in (0, 1)
    return json.loads(capsys.readouterr().out)


def _approve(path):
    entries = json.loads(path.read_text())["events"]
    for entry in entries:
        entry.update(approved_by="analyst", approved_at="2000-01-01T00:00:00+00:00", confirmed=True)
    path.write_text(json.dumps({"events": entries}))
    return entries


@pytest.mark.parametrize(
    ("family", "text", "explained_check"),
    [
        ("missing_stores", "{stores} closed for refit from week {week}.", "absence_event"),
        ("commodity_remap", "Products in {commodities} reclassified (range review).",
         "historical_event"),
    ],
)
def test_closure_and_move_notices_draft_entries_that_explain_once_approved(tmp_path, capsys, family, text, explained_check):
    built, case = _scenario(tmp_path, family)
    affected = case["affected"]
    notices = tmp_path / "notices.json"
    notices.write_text(json.dumps([{"id": "n1", "text": text.format(
        stores=" and ".join(affected.get("stores", [])),
        commodities=" and ".join(affected.get("commodities", [])),
        week=(case.get("weeks") or [""])[0],
    )}]))
    drafts = tmp_path / "drafts.json"
    capsys.readouterr()
    assert main(["notices", "--scenario-dir", str(built.directory), "--notices", str(notices),
                 "--out", str(drafts), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["matches"][0]["action"] == "registry_draft"
    blind = _run(capsys, built.directory)
    assert blind["status"] == "INVESTIGATE"
    assert _run(capsys, built.directory, "--registry", str(drafts))["status"] == "INVESTIGATE"

    entries = _approve(drafts)
    assert entries[0]["effective_from_week"] == entries[0]["effective_to_week"] == report["latest_week"]
    approved = _run(capsys, built.directory, "--registry", str(drafts))

    def unexplained(report):
        return {(f["check"], f["scope"], f["metric"]) for f in report["findings"] if f["disposition"] == "UNEXPLAINED_ANOMALY"}

    # The approval clears the change's own findings (its approval finding, the
    # historical status and the affected banners' or categories' latest week)
    # and nothing it did not cause; what remains was already flagged blind.
    affected = {f"banner_id:{b}" for b in affected.get("banners", [])} | {
        f"commodity_id:{c}" for c in affected.get("commodities", [])}
    remaining = unexplained(approved)
    assert remaining <= unexplained(blind)
    assert not {item for item in remaining if item[0] == "historical_revision" or item[1] in affected}
    assert any(f["check"] == explained_check and f["disposition"] == "HUMAN_APPROVED" for f in approved["findings"])
    assert approved["status"] == ("INVESTIGATE" if remaining else "PASS_WITH_EXPLANATION")


class _Service(BaseHTTPRequestHandler):
    """A canned System One service: picks c0, fails the dataset check for "[Other]" notices."""

    requests: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Service.requests.append(body)
        if "match" in body["questions"]:
            answers = {"match": {"type": "choice", "choice": "c0", "probabilities": {"c0": 0.8, "none": 0.2}}}
        else:
            other = body["state"]["notice"].startswith("[Other]")
            answers = {q: {"type": "noul", "noul": 0.1 if (other and q == "same_dataset") else 0.9}
                       for q in body["questions"]}
        payload = json.dumps({"model": body["model"], "answers": answers}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_systemone_matcher_speaks_the_protocol_and_checks_veto():
    server = HTTPServer(("127.0.0.1", 0), _Service)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        matcher = SystemOneMatcher(f"http://127.0.0.1:{server.server_port}", "lux")
        cands = [Candidate("c0", "LATEST_WEEK_MISSING", "store", ["S005"], [104], "stores missing: S005")]
        assert matcher.match("S005 closed for refit.", cands, "ds", 104, {}).candidate == "c0"
        assert matcher.match("[Other] S005 closed for refit.", cands, "ds", 104, {}).candidate == NONE
        choice = _Service.requests[0]
        assert choice["model"] == "lux" and set(choice["questions"]["match"]["criteria"]) == {"c0", NONE}
        assert set(_Service.requests[1]["questions"]) == {"same_dataset", "in_effect", "same_kind", "weeks_cover"}
    finally:
        server.shutdown()
    with pytest.raises(ValueError):
        SystemOneMatcher("http://decisions.example.com", "lux")


def test_notice_files_in_three_formats(tmp_path):
    for name, body in (
        ("a.txt", "first notice\n\nsecond notice\n"),
        ("a.jsonl", '{"id": "x", "text": "first notice"}\n{"text": "second notice"}\n'),
        ("a.json", '["first notice", {"id": "y", "text": "second notice"}]'),
    ):
        path = tmp_path / name
        path.write_text(body)
        assert [n["text"] for n in load_notices(path)] == ["first notice", "second notice"]
