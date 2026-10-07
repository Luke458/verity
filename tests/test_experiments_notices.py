"""Notice-matching sandbox: oracle answers, the pattern baseline and the rewrite fact check."""

from __future__ import annotations

from experiments.notices.match import NONE, checked_choice, pattern_choice, score
from experiments.notices.notices import Candidate, gold_for
from experiments.notices.rewrite import faithful


def _row(candidates):
    return {
        "dataset": "synthetic-retail",
        "latest_week": 104,
        "commodities": {"C01": "Analgesics", "C04": "First Aid"},
        "candidates": candidates,
    }


MISSING = {"key": "c0", "classification": "LATEST_WEEK_MISSING", "entity_type": "store",
           "entity_ids": ["S005", "S010"], "weeks": [104]}
DECOY = {"key": "c1", "classification": "LATEST_WEEK_MISSING", "entity_type": "product",
         "entity_ids": ["P0075"], "weeks": [104]}
BACKFILL = {"key": "c0", "classification": "NEW_ENTITY_HISTORICAL_BACKFILL", "entity_type": "store",
            "entity_ids": ["S900"], "weeks": list(range(85, 104))}


def _notice(text, gold=()):
    return {"text": text, "kind": "true" if gold else "wrong_entity", "gold": list(gold)}


def test_gold_needs_the_right_classification_and_a_shared_entity():
    cands = [Candidate(**MISSING), Candidate(**DECOY)]
    assert gold_for("missing_stores", {"S005", "S024"}, cands) == ["c0"]
    assert gold_for("missing_stores", {"S011"}, cands) == []
    assert gold_for("history_truncation", {"S005"}, cands) == []


def test_pattern_baseline_reads_ids_cues_weeks_and_dataset_tags():
    row = _row([MISSING, DECOY])
    assert pattern_choice(row, _notice("Store closures: store 5 and S010 did not trade in wk 104.")) == "c0"
    assert pattern_choice(row, _notice("S005 will close for refit from week 110.")) == NONE
    assert pattern_choice(row, _notice("[Loyalty panel] S005 closed for refit from week 104.")) == NONE
    assert pattern_choice(row, _notice("Vendor withdrew weeks 98-103 for S005 pending an audit.")) == NONE
    backfill = _row([BACKFILL])
    assert pattern_choice(backfill, _notice("S900 added to the feed with back history (weeks 85-104).")) == "c0"
    assert pattern_choice(backfill, _notice("S900: history for weeks 25 through 44 only has been loaded.")) == NONE


def test_rewrite_must_keep_numbers_and_the_other_dataset():
    template = "[Loyalty panel] S005 and store 10 closed for refit from wk 104."
    assert faithful(template, "Loyalty team: stores 5 and 10 shut for a refit as of week 104.")
    assert not faithful(template, "Stores 5 and 10 shut for a refit as of week 104.")
    assert not faithful(template, "Loyalty: store 5 shut for a refit as of week 104.")


def test_checks_veto_a_match_and_scoring_counts_false_explanations():
    assert checked_choice({"c0": 0.8, NONE: 0.2}, {"same_dataset": 0.9, "in_effect": 0.3}) == NONE
    assert checked_choice({"c0": 0.8, NONE: 0.2}, {"same_dataset": 0.9, "in_effect": 0.7}) == "c0"
    row = _row([MISSING])
    items = [
        (row, _notice("true", ["c0"]), "c0"),
        (row, _notice("distractor"), "c0"),
        (row, _notice("distractor"), NONE),
        ({**row, "candidates": []}, _notice("ignored"), NONE),
    ]
    report = score(items)
    assert report["notices"] == 3
    assert report["false_explanations"]["hits"] == 1 and report["recall"]["hits"] == 1
