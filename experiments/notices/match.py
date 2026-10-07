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
from qc.notices import NONE, Candidate, PatternMatcher, check_request, choice_request

from .notices import DISTRACTOR_KINDS


def _cands(row: dict[str, Any]) -> list[Candidate]:
    return [Candidate(**c) for c in row["candidates"]]


def pattern_choice(row: dict[str, Any], notice: dict[str, Any]) -> str:
    """The deterministic baseline (``qc.notices.PatternMatcher``)."""
    names = row.get("commodities", {})
    return PatternMatcher().match(notice["text"], _cands(row), row["dataset"], int(row["latest_week"]), names).candidate


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
