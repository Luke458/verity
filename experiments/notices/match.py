"""Matching a notice to one observed change: a pattern baseline, a decision-model
question, and the scoring both are judged by.

A method answers one candidate key per notice, or ``"none"``. The dangerous
error is a *false explanation*: answering a candidate the notice does not
explain, which could get a real fault approved away.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from qc.conformal import wilson_interval

from .notices import ACCEPTS, DISTRACTOR_KINDS

NONE = "none"

# Change-type cues, most specific first.
CUES: tuple[tuple[str, str], ...] = (
    (r"\bmerged\b|\breplaced by\b|\bnow trades as\b|\brebrand\b|\bconsolidation\b", "entity_merge"),
    (r"\bdiscontinued\b|\bout of stock\b|\bdelisted\b", "missing_products"),
    (r"\bmoving from\b|\breclassified\b|\bnow sit under\b|\bwill move to\b|\bhierarchy\b", "commodity_remap"),
    (r"\bwithdrew\b|\bdeleted at source\b|\bremoving history\b|\bwithdrawn\b", "history_truncation"),
    (r"\bnew store\b|\bjoins the panel\b|\badded to the feed\b|\bback history\b|\bhistory for\b.*\bloaded\b", "new_store_backfill"),
    (r"\bclosed?\b|\bclosures?\b|\bdid not trade\b|\bno sales feed\b|\brefit\b|\brenovations?\b", "missing_stores"),
)
FUTURE = re.compile(r"\bwill\b|\bto be\b|\bplanned\b", re.IGNORECASE)
WEEKS = re.compile(r"\b(?:weeks?|wk)\s*(\d+)(?:\s*(?:-|to|through)\s*(\d+))?", re.IGNORECASE)


def _ids(text: str, commodities: dict[str, str]) -> set[str]:
    found = {f"S{int(n):03d}" for n in re.findall(r"\bS(\d{3})\b", text)}
    found |= {f"S{int(n):03d}" for n in re.findall(r"\bstore\s+S?(\d{1,3})\b", text, re.IGNORECASE)}
    found |= {f"P{int(n):04d}" for n in re.findall(r"\bP(\d{4})\b", text)}
    found |= {f"P{int(n):04d}" for n in re.findall(r"\bproduct\s+(\d{1,4})\b", text, re.IGNORECASE)}
    found |= set(re.findall(r"\bC\d{2}\b", text))
    found |= {cid for cid, name in commodities.items() if name.lower() in text.lower()}
    return found


def _change_type(text: str) -> str | None:
    for pattern, change_type in CUES:
        if re.search(pattern, text, re.IGNORECASE):
            return change_type
    return None


def pattern_choice(row: dict[str, Any], notice: dict[str, Any]) -> str:
    """The deterministic baseline: dataset tag, change cue, shared entity, consistent weeks."""
    text = notice["text"]
    tag = re.match(r"^\[(.+?)\]", text)
    if tag and tag.group(1) != row["dataset"]:
        return NONE
    change_type = _change_type(text)
    if change_type is None or FUTURE.search(text):
        return NONE
    ids = _ids(text, row.get("commodities", {}))
    spans = [(int(a), int(b) if b else None) for a, b in WEEKS.findall(text)]
    latest = int(row["latest_week"])
    for cand in row["candidates"]:
        if cand["classification"] not in ACCEPTS[change_type]:
            continue
        if not ids & {part for e in cand["entity_ids"] for part in e.split("->")}:
            continue
        observed = cand["weeks"]
        if spans and observed and change_type in ("new_store_backfill", "history_truncation"):
            start, end = spans[0]
            end = end if end is not None else latest
            if not (start <= min(observed) and max(observed) <= end + 1):
                continue
        if spans and change_type in ("missing_stores", "missing_products", "commodity_remap"):
            if spans[0][0] > latest:
                continue
        return str(cand["key"])
    return NONE


DATASET_DESCRIPTION = "weekly point-of-sale sales of a pharmacy chain, by store and product"
QUESTION = (
    "Which of the changes observed in this refresh does the notice describe? Choose none if the "
    "notice does not account for any of them: it is about other stores or products, a different "
    "kind of change, other weeks, something that has not happened yet, or another dataset."
)


def lux_request(row: dict[str, Any], notice: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """System One state and question for one notice (None when there is nothing to choose)."""
    state = {
        "dataset being checked": row["dataset"],
        "latest week in this refresh": row["latest_week"],
        "notice": notice["text"],
    }
    criteria = {c["key"]: c["description"] for c in row["candidates"]}
    criteria[NONE] = "None of the observed changes"
    return state, {"match": {"type": "choice", "instructions": QUESTION, "criteria": criteria}}


def lux_checks(
    row: dict[str, Any], notice: dict[str, Any], candidate: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Direct yes/no checks on a chosen candidate; code requires all of them to hold."""
    state = {
        "dataset being checked": f"{row['dataset']}: {DATASET_DESCRIPTION}",
        "latest week in this refresh": row["latest_week"],
        "notice": notice["text"],
        "observed change": candidate["description"],
    }
    yes_no = {"false": "No", "true": "Yes"}
    questions = {
        "same_dataset": {
            "type": "noul",
            "instructions": f"Is the notice about the dataset being checked ({row['dataset']})? "
            "Answer no if it names a different dataset or data source.",
            "criteria": yes_no,
        },
        "in_effect": {
            "type": "noul",
            "instructions": f"Had the change in the notice already happened by week {row['latest_week']}? "
            "Answer no if it is planned for a later week.",
            "criteria": yes_no,
        },
        "same_kind": {
            "type": "noul",
            "instructions": "Does the notice describe the same kind of change as the observed change "
            "(for example a closure, a new store with history, deleted history, a category move or a merge)?",
            "criteria": yes_no,
        },
        "weeks_cover": {
            "type": "noul",
            "instructions": "Do the weeks stated in the notice cover the weeks of the observed change? "
            "Answer yes if either states no weeks.",
            "criteria": yes_no,
        },
    }
    return state, questions


def checked_choice(probabilities: dict[str, float], checks: dict[str, float] | None, threshold: float = 0.5) -> str:
    """The best option, kept only when every check on it clears ``threshold``."""
    best = max(probabilities, key=probabilities.__getitem__)
    if best == NONE or not checks or min(checks.values()) < threshold:
        return NONE
    return best


def _rate(hits: int, n: int) -> dict[str, Any]:
    low, high = wilson_interval(hits, n)
    return {"hits": hits, "n": n, "rate": hits / n if n else None, "ci_low": low if n else None, "ci_high": high if n else None}


def score(items: Sequence[tuple[dict[str, Any], dict[str, Any], str]]) -> dict[str, Any]:
    """``items`` are (row, notice, choice). Only rows with at least one candidate count."""
    items = [item for item in items if item[0]["candidates"]]
    correct = [choice in notice["gold"] if notice["gold"] else choice == NONE for _, notice, choice in items]
    false_explanation = [choice != NONE and choice not in notice["gold"] for _, notice, choice in items]
    true_with_answer = [(n, c) for _, n, c in items if n["kind"] == "true" and n["gold"]]
    report: dict[str, Any] = {
        "notices": len(items),
        "accuracy": _rate(sum(correct), len(items)),
        "false_explanations": _rate(sum(false_explanation), len(items)),
        "recall": _rate(sum(c in n["gold"] for n, c in true_with_answer), len(true_with_answer)),
        "false_explanations_by_kind": {},
    }
    for kind in DISTRACTOR_KINDS:
        chosen = [choice for _, notice, choice in items if notice["kind"] == kind]
        report["false_explanations_by_kind"][kind] = _rate(sum(c != NONE for c in chosen), len(chosen))
    return report
