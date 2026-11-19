"""Matching a notice to one observed change: a pattern baseline, a decision-model
question, and the scoring both are judged by.

A method answers one candidate key per notice, or ``"none"``. The dangerous
error is a *false explanation*: answering a candidate the notice does not
explain, which could get a real fault approved away.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qc.conformal import wilson_interval
from qc.notices import (
    DATASET_DESCRIPTION,
    NONE,
    Candidate,
    PatternMatcher,
    decide,
    reading_from_answers,
)

from .notices import DISTRACTOR_KINDS

# The Lux (System One decision) requests, as tested on 2026-10-07 and in the
# confirm run; formerly qc.notices.SystemOneMatcher's.
QUESTION = (
    "Which of the changes observed in this refresh does the notice describe? Choose none if the "
    "notice does not account for any of them: it is about other stores or products, a different "
    "kind of change, other weeks, something that has not happened yet, or another dataset."
)


def choice_request(
    text: str, cands: Sequence[Candidate], dataset: str, latest_week: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """System One state and question: which observed change does the notice describe?"""
    state = {"dataset being checked": dataset, "latest week in this refresh": latest_week, "notice": text}
    criteria = {c.key: c.description for c in cands}
    criteria[NONE] = "None of the observed changes"
    return state, {"match": {"type": "choice", "instructions": QUESTION, "criteria": criteria}}


def check_request(
    text: str, cand: Candidate, dataset: str, latest_week: int, description: str = DATASET_DESCRIPTION
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Direct yes/no checks on a chosen change; code requires all of them to hold."""
    state = {
        "dataset being checked": f"{dataset}: {description}",
        "latest week in this refresh": latest_week,
        "notice": text,
        "observed change": cand.description,
    }
    yes_no = {"false": "No", "true": "Yes"}
    questions = {
        "same_dataset": {
            "type": "noul",
            "instructions": f"Is the notice about the dataset being checked ({dataset})? "
            "Answer no if it names a different dataset or data source.",
            "criteria": yes_no,
        },
        "in_effect": {
            "type": "noul",
            "instructions": f"Had the change in the notice already happened by week {latest_week}? "
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


def _cands(row: dict[str, Any]) -> list[Candidate]:
    return [Candidate(**c) for c in row["candidates"]]


def pattern_choice(row: dict[str, Any], notice: dict[str, Any]) -> str:
    """The deterministic baseline (``qc.notices.PatternMatcher``)."""
    names = row.get("commodities", {})
    return PatternMatcher().match(notice["text"], _cands(row), row["dataset"], int(row["latest_week"]), names).candidate


def extract_choice(
    row: dict[str, Any], notice: dict[str, Any], answers: dict[str, Any] | None, pattern_ids: bool = False
) -> str:
    """An extraction model's reading through the baseline's rules; no reading answers none.
    ``pattern_ids`` adds the ids the patterns find in the notice text to the model's."""
    if answers is None or not row["candidates"]:
        return NONE
    names = row.get("commodities", {})
    text = notice["text"] if pattern_ids else None
    reading = reading_from_answers(answers, row["dataset"], names, latest_week=int(row["latest_week"]), text=text)
    return decide(reading, _cands(row), int(row["latest_week"])).candidate


def lux_request(row: dict[str, Any], notice: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return choice_request(notice["text"], _cands(row), row["dataset"], row["latest_week"])


def lux_checks(
    row: dict[str, Any], notice: dict[str, Any], candidate: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return check_request(notice["text"], Candidate(**candidate), row["dataset"], row["latest_week"])


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
