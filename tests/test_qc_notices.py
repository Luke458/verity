"""Notice matching and draft registry entries (qc.notices, `qc notices`)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from qc.cli import main
from qc.notices import (
    NONE,
    Candidate,
    ExtractMatcher,
    PatternMatcher,
    load_notices,
    reading_from_answers,
)
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


@pytest.mark.parametrize(
    ("factor", "pair_covered", "revision_explained"),
    [
        (1.0, True, True),  # history carried across unchanged
        (0.95, True, False),  # the pair conserves value, the 5% it lost is still material
        (0.8, False, False),  # not conserved: the approval does not cover the pair
    ],
)
def test_a_merge_notice_drafts_one_replacement_that_explains_only_conserved_history(
    tmp_path, capsys, factor, pair_covered, revision_explained
):
    stages = ("source", "coded", "warehouse", "report")
    built = build_scenario(suite_config("tiny"), 0, tmp_path / "merge", "entity_merge", stages,
                           params={"entity_merge": {"factor": factor}})
    case = built.oracle["cases"][0]
    old, new = case["details"]["source_store"], case["details"]["target_store"]
    notices = tmp_path / "notices.txt"
    notices.write_text(f"{old} merged into {new}; its history was carried across.\n")
    drafts = tmp_path / "drafts.json"
    capsys.readouterr()
    assert main(["notices", "--scenario-dir", str(built.directory), "--notices", str(notices),
                 "--out", str(drafts), "--json"]) == 0
    match = json.loads(capsys.readouterr().out)["matches"][0]
    assert match["candidate"]["classification"] == "ENTITY_REPLACED" and len(match["drafts"]) == 1
    (entry,) = _approve(drafts)
    assert (entry["classification"], entry["entity_ids"]) == ("ENTITY_REPLACED", [f"{old}->{new}"])

    blind = _run(capsys, built.directory)
    approved = _run(capsys, built.directory, "--registry", str(drafts))

    def dispositions(report, check):
        return sorted({f["disposition"] for f in report["findings"] if f["check"] == check and f["outcome"] != "PASS"})

    def unexplained(report):
        return {(f["check"], f["scope"], f["metric"]) for f in report["findings"] if f["disposition"] == "UNEXPLAINED_ANOMALY"}

    assert dispositions(blind, "historical_revision") == ["UNEXPLAINED_ANOMALY"]
    covered = dispositions(approved, "historical_event")
    assert covered == (["HUMAN_APPROVED"] if pair_covered else [])
    if revision_explained:
        # Approved, the merge leaves only its world's own flags: the new store
        # is compared with its restated history, not set aside as new.
        clean = build_scenario(suite_config("tiny"), 0, tmp_path / "clean", "clean", stages)
        assert dispositions(approved, "historical_revision") == ["HUMAN_APPROVED"]
        assert unexplained(approved) <= unexplained(_run(capsys, clean.directory))
    else:
        assert dispositions(approved, "historical_revision") == ["UNEXPLAINED_ANOMALY"]
        assert approved["status"] == "INVESTIGATE"


def _answers(**values):
    return {name: {"value": value, "confidence": 0.9} for name, value in values.items()}


def test_reading_from_extracted_fields():
    names = {"C04": "First Aid"}
    reading = reading_from_answers(
        _answers(data_source="the sales feed", change="store closed, not trading or sending no sales", planned=False,
                 stores=["Store 16", "S004"], products=[], categories=["First Aid"], first_week=104, last_week=None),
        "synthetic-retail", names,
    )
    assert not reading.other_dataset and not reading.planned
    assert reading.change_type == "missing_stores"
    assert reading.ids == {"S016", "S004", "C04"} and reading.spans == [(104, None)]
    # A named source sharing no word with the dataset is another dataset; none named is not.
    other = reading_from_answers(_answers(data_source="Wholesale shipments"), "synthetic-retail", {})
    assert other.other_dataset and other.change_type is None
    assert not reading_from_answers(_answers(data_source=None), "synthetic-retail", {}).other_dataset


def test_a_merge_explains_a_new_store_only_through_its_new_id():
    merge = [
        Candidate("c0", "ENTITY_REMOVED", "store", ["S005"], [1, 103], "stores removed entirely: S005"),
        Candidate("c1", "ENTITY_REPLACED", "store", ["S005", "S900"], [], "S005 replaced by S900"),
        Candidate("c2", "NEW_ENTITY_HISTORICAL_BACKFILL", "store", ["S900"], [1, 103], "new: S900"),
    ]
    backfill = [merge[2]]
    patterns = PatternMatcher()
    # The right direction explains the merge's own changes, as one replacement
    # when the refresh shows it (drafted as its removal and its backfill).
    assert patterns.match("Consolidation - S005 replaced by S900.", merge, "ds", 104, {}).candidate == "c1"
    assert patterns.match("Consolidation - S005 replaced by S900.", merge[:1], "ds", 104, {}).candidate == "c0"
    assert patterns.match("Consolidation - S005 replaced by S900.", backfill, "ds", 104, {}).candidate == "c2"
    # S900 named as the old id cannot explain S900 arriving with history (the confirm run's defect).
    assert patterns.match("Consolidation - S900 replaced by S969.", backfill, "ds", 104, {}).candidate == NONE
    assert patterns.match("S900 now trades as S969.", merge, "ds", 104, {}).candidate == NONE
    # No direction read (no forward phrase): the old rule, any shared id.
    assert patterns.match("Consolidation of S900 and S969 in the store master.", backfill, "ds", 104, {}).candidate == "c2"

    # The extraction reading has no direction field: it keeps the any-shared-id rule.
    answers = _answers(change="store merged into or replaced by another store", planned=False, stores=["S900", "S969"])
    assert reading_from_answers(answers, "ds", {}, latest_week=104).merge_targets is None


class _Extract(BaseHTTPRequestHandler):
    """A canned Extract service, wrapped like Workers AI ("result")."""

    requests: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Extract.requests.append({"body": body, "auth": self.headers.get("Authorization")})
        other = body["state"].startswith("[Other]")
        answers = _answers(data_source="Other feed" if other else None, change="store closed, not trading or sending no sales",
                           planned=False, stores=["S005"], first_week=104)
        payload = json.dumps({"result": {"model": "clairic-flash", "answers": answers}, "success": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_extract_matcher_reads_then_decides_deterministically():
    server = HTTPServer(("127.0.0.1", 0), _Extract)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        matcher = ExtractMatcher(f"http://127.0.0.1:{server.server_port}", token="t0ken")
        cands = [Candidate("c0", "LATEST_WEEK_MISSING", "store", ["S005"], [104], "stores missing: S005")]
        found = matcher.match("S005 closed for refit.", cands, "ds", 104, {})
        assert found.candidate == "c0" and found.details["extracted"]["stores"]["value"] == ["S005"]
        assert matcher.match("[Other] S005 closed.", cands, "ds", 104, {}).candidate == NONE
        # Another store's closure: the rules find no observed change that fits.
        other_store = [Candidate("c0", "LATEST_WEEK_MISSING", "store", ["S009"], [104], "stores missing: S009")]
        assert matcher.match("S005 closed for refit.", other_store, "ds", 104, {}).candidate == NONE
        sent = _Extract.requests[0]
        assert sent["auth"] == "Bearer t0ken" and sent["body"]["model"] == "clairic-flash"
        assert "candidates" not in json.dumps(sent["body"]) and sent["body"]["state"] == "S005 closed for refit."
    finally:
        server.shutdown()
    with pytest.raises(ValueError):
        ExtractMatcher("http://extract.example.com")


def test_notice_files_in_three_formats(tmp_path):
    for name, body in (
        ("a.txt", "first notice\n\nsecond notice\n"),
        ("a.jsonl", '{"id": "x", "text": "first notice"}\n{"text": "second notice"}\n'),
        ("a.json", '["first notice", {"id": "y", "text": "second notice"}]'),
    ):
        path = tmp_path / name
        path.write_text(body)
        assert [n["text"] for n in load_notices(path)] == ["first notice", "second notice"]
    # A text file whose first notice has a [tag] is lines, not a broken JSON list.
    tagged = tmp_path / "inbox.txt"
    tagged.write_text("[Wholesale shipments] S004 closed.\n{braces} in a second notice\n")
    assert [n["text"] for n in load_notices(tagged)] == ["[Wholesale shipments] S004 closed.", "{braces} in a second notice"]
    broken = tmp_path / "broken.json"
    broken.write_text("[not json")
    with pytest.raises(json.JSONDecodeError):
        load_notices(broken)
