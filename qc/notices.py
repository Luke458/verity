"""Draft explanations from free-text change notices.

Operators hear about structural changes as text ("S012 closed for refit from
week 118"). ``draft`` matches each notice to one of the refresh's observed
lifecycle changes (``candidates``). Where the expected-event registry can
express that change, it writes an **unapproved** registry entry that cites the
notice. A person sets ``approved_by``/``approved_at`` and ``confirmed`` before
the engine will use it; nothing here changes a status.

Two matchers: ``PatternMatcher`` (deterministic, the default) and
``SystemOneMatcher``, which asks any decision service speaking the System One
protocol (``POST /v1/systemone``) one choice question and four yes/no checks,
and keeps a match only if every check holds. docs/notice-matching.md
measures both.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .lifecycle import (
    EXTENDED,
    LATEST_MISSING,
    NEW_BACKFILL,
    RECLASSIFIED,
    REMOVED,
    TRUNCATED,
)

NONE = "none"
REPLACED = "ENTITY_REPLACED"
# Historical changes: the registry scopes them by the weeks they rewrite.
HISTORY_CLASSES = frozenset({NEW_BACKFILL, REMOVED, EXTENDED, TRUNCATED})
# Closures and category moves: scoped by the weeks they are in effect.
EFFECT_CLASSES = frozenset({LATEST_MISSING, RECLASSIFIED})
REGISTRY_CLASSES = HISTORY_CLASSES | EFFECT_CLASSES
PLURALS = {"store": "stores", "product": "products", "commodity": "categories"}
CLASS_PHRASES = {
    "LATEST_WEEK_MISSING": "missing in the latest week",
    "NEW_ENTITY_HISTORICAL_BACKFILL": "new, arrived with past history",
    "NEW_ENTITY_RECENT": "new, recent weeks only",
    "ENTITY_REMOVED": "removed entirely",
    "ENTITY_HISTORY_EXTENDED": "history extended",
    "ENTITY_HISTORY_TRUNCATED": "history truncated",
    "POSSIBLE_RECLASSIFICATION": "products moved between categories",
    REPLACED: "id replaced by another",
}
# Which observed classifications a kind of notice can explain.
ACCEPTS: dict[str, frozenset[str]] = {
    "missing_stores": frozenset({"LATEST_WEEK_MISSING"}),
    "missing_products": frozenset({"LATEST_WEEK_MISSING"}),
    "new_store_backfill": frozenset({NEW_BACKFILL}),
    "history_truncation": frozenset({TRUNCATED, REMOVED}),
    "commodity_remap": frozenset({"POSSIBLE_RECLASSIFICATION"}),
    "entity_merge": frozenset({REPLACED, REMOVED, NEW_BACKFILL}),
}


@dataclass
class Candidate:
    """One observed change: a classification, its entities and weeks."""

    key: str
    classification: str
    entity_type: str
    entity_ids: list[str]
    weeks: list[int]
    description: str = ""
    product_ids: list[str] = field(default_factory=list)  # category moves: the products moved


def _span(weeks: Sequence[int]) -> str:
    if not weeks:
        return ""
    return f"week {weeks[0]}" if len(weeks) == 1 else f"weeks {min(weeks)}-{max(weeks)}"


def _label(entity_id: str, names: Mapping[str, str]) -> str:
    parts = entity_id.split("->")
    return " -> ".join(f"{p} ({names[p]})" if p in names else p for p in parts)


def candidates(result: Any, names: Mapping[str, str] | None = None) -> list[Candidate]:
    """The refresh's lifecycle changes grouped by classification, plus id replacements.

    ``names`` optionally maps ids to display names (stores, categories).
    """
    names = names or {}
    latest = result.version_pair.current_max_week if result.version_pair is not None else None
    groups: dict[tuple[str, str], Candidate] = {}
    for event in result.events or ():
        group = groups.setdefault(
            (event.classification, event.entity_type),
            Candidate("", event.classification, event.entity_type, [], []),
        )
        group.entity_ids.append(str(event.entity_id))
        group.product_ids += [str(p) for p in event.details.get("products", [])]
        weeks = list(event.historical_weeks_added) + list(event.historical_weeks_removed)
        if event.classification == "LATEST_WEEK_MISSING" and latest is not None:
            weeks = [latest]
        group.weeks = sorted(set(group.weeks) | {int(w) for w in weeks})
    for relationship in result.relationships or ():
        if relationship.relationship in ("replaced_by", "merged_into", "superseded_by"):
            group = groups.setdefault((REPLACED, "store"), Candidate("", REPLACED, "store", [], []))
            group.entity_ids += [str(relationship.source_id), str(relationship.target_id)]
    out = []
    for index, (_, group) in enumerate(sorted(groups.items())):
        group.key = f"c{index}"
        labels = [_label(e, names) for e in group.entity_ids]
        if group.classification == REPLACED:
            listed = " replaced by ".join(labels)
        else:
            listed = ", ".join(labels[:6]) + (f" and {len(labels) - 6} more" if len(labels) > 6 else "")
        span = _span(group.weeks)
        group.description = (
            f"{PLURALS.get(group.entity_type, group.entity_type + 's')} "
            f"{CLASS_PHRASES.get(group.classification, group.classification)}: {listed}"
            + (f" ({span})" if span else "")
        )
        out.append(group)
    return out


# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------


@dataclass
class Match:
    candidate: str  # a candidate key, or NONE
    details: dict[str, Any] = field(default_factory=dict)


class Matcher(Protocol):
    name: str

    def match(
        self, text: str, cands: Sequence[Candidate], dataset: str, latest_week: int, names: Mapping[str, str]
    ) -> Match: ...


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


def _ids(text: str, names: Mapping[str, str]) -> set[str]:
    found = {f"S{int(n):03d}" for n in re.findall(r"\bS(\d{3})\b", text)}
    found |= {f"S{int(n):03d}" for n in re.findall(r"\bstore\s+S?(\d{1,3})\b", text, re.IGNORECASE)}
    found |= {f"P{int(n):04d}" for n in re.findall(r"\bP(\d{4})\b", text)}
    found |= {f"P{int(n):04d}" for n in re.findall(r"\bproduct\s+(\d{1,4})\b", text, re.IGNORECASE)}
    found |= set(re.findall(r"\bC\d{2}\b", text))
    found |= {cid for cid, name in names.items() if cid.startswith("C") and name.lower() in text.lower()}
    return found


class PatternMatcher:
    """Dataset tag, change-type cue, a shared entity id and consistent weeks.

    Written for the synthetic id formats (S###, P####, C##); a deployment
    would adapt the id patterns to its own keys.
    """

    name = "patterns"

    def match(
        self, text: str, cands: Sequence[Candidate], dataset: str, latest_week: int, names: Mapping[str, str]
    ) -> Match:
        tag = re.match(r"^\[(.+?)\]", text)
        if tag and tag.group(1) != dataset:
            return Match(NONE, {"reason": "another dataset"})
        change_type = next((t for pattern, t in CUES if re.search(pattern, text, re.IGNORECASE)), None)
        if change_type is None:
            return Match(NONE, {"reason": "no change cue"})
        if FUTURE.search(text):
            return Match(NONE, {"reason": "not yet in effect"})
        ids = _ids(text, names)
        spans = [(int(a), int(b) if b else None) for a, b in WEEKS.findall(text)]
        for cand in cands:
            if cand.classification not in ACCEPTS[change_type]:
                continue
            if not ids & {part for e in cand.entity_ids for part in e.split("->")}:
                continue
            if spans and cand.weeks and change_type in ("new_store_backfill", "history_truncation"):
                start, end = spans[0]
                end = end if end is not None else latest_week
                if not (start <= min(cand.weeks) and max(cand.weeks) <= end + 1):
                    continue
            if spans and change_type in ("missing_stores", "missing_products", "commodity_remap"):
                if spans[0][0] > latest_week:
                    continue
            return Match(cand.key, {"change_type": change_type})
        return Match(NONE, {"reason": "no observed change fits", "change_type": change_type})


DATASET_DESCRIPTION = "weekly point-of-sale sales of a pharmacy chain, by store and product"
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


def _endpoint_url(base: str) -> str:
    from .notify import _validate_webhook_target

    url = base.rstrip("/")
    url = url if url.endswith("/v1/systemone") else url + "/v1/systemone"
    _validate_webhook_target(url)  # https, or http to loopback only
    return url


class SystemOneMatcher:
    """A System One decision service: one choice, then yes/no checks on the chosen change."""

    name = "systemone"

    def __init__(
        self,
        endpoint: str,
        model: str,
        threshold: float = 0.5,
        timeout: float = 60.0,
        dataset_description: str = DATASET_DESCRIPTION,
    ):
        self.url = _endpoint_url(endpoint)
        self.model, self.threshold, self.timeout = model, threshold, timeout
        self.dataset_description = dataset_description

    def _ask(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        import urllib.request

        from .notify import _OPENER

        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        request = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with _OPENER.open(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        answers = payload.get("answers", payload)
        if not isinstance(answers, dict):
            raise ValueError("decision service returned no answers")
        return answers

    def match(
        self, text: str, cands: Sequence[Candidate], dataset: str, latest_week: int, names: Mapping[str, str]
    ) -> Match:
        if not cands:
            return Match(NONE)
        probabilities = self._ask(*choice_request(text, cands, dataset, latest_week))["match"]["probabilities"]
        best = max(probabilities, key=probabilities.__getitem__)
        if best == NONE:
            return Match(NONE, {"probabilities": probabilities})
        cand = next(c for c in cands if c.key == best)
        answers = self._ask(*check_request(text, cand, dataset, latest_week, self.dataset_description))
        checks = {name: float(answer["noul"]) for name, answer in answers.items()}
        details = {"probabilities": probabilities, "checks": checks}
        if min(checks.values()) < self.threshold:
            return Match(NONE, {**details, "reason": "a check failed"})
        return Match(best, details)


# ---------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------


def load_notices(path: str | Path) -> list[dict[str, str]]:
    """Notices from JSON (a list of strings or {"id", "text"}), JSONL, or plain text lines."""
    raw = Path(path).read_text()
    stripped = raw.lstrip()
    items: list[Any]
    if stripped.startswith("["):
        items = json.loads(raw)
    elif stripped.startswith("{"):
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        items = [line.strip() for line in raw.splitlines() if line.strip()]
    notices = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            item = {"text": item}
        text = str(item["text"]).strip()
        notices.append({"id": str(item.get("id") or f"notice-{index + 1}"), "text": text})
    return notices


def registry_draft(cand: Candidate, dataset: str, latest_week: int | None = None) -> dict[str, Any] | None:
    """An unapproved registry entry for an observed change the registry can express.

    A historical change is scoped to the weeks it rewrote. A closure or a
    category move is scoped to the refresh's latest week; the reviewer widens
    ``effective_to_week`` if the change is known to last. A move also lists
    the products observed moving, so the approval covers no others.
    """
    if cand.classification in HISTORY_CLASSES and cand.weeks:
        scope: dict[str, Any] = {"expected_history_start": min(cand.weeks), "expected_history_end": max(cand.weeks)}
    elif cand.classification in EFFECT_CLASSES and latest_week is not None:
        scope = {"effective_from_week": latest_week, "effective_to_week": latest_week}
        if cand.classification == RECLASSIFIED:
            scope["product_ids"] = sorted(cand.product_ids)
    else:
        return None
    identity = f"{dataset}|{cand.classification}|{cand.entity_type}|{sorted(cand.entity_ids)}|{cand.weeks}|{latest_week}"
    return {
        "event_id": f"draft-{hashlib.sha256(identity.encode()).hexdigest()[:12]}",
        "dataset": dataset,
        "classification": cand.classification,
        "entity_type": cand.entity_type,
        "entity_ids": list(cand.entity_ids),
        **scope,
        "description": cand.description,
        "notices": [],
        "confirmed": False,
        "approved_by": None,
        "approved_at": None,
    }


def draft(
    result: Any,
    notices: Sequence[dict[str, str]],
    matcher: Matcher,
    names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Match every notice; draft registry entries where the registry can use them."""
    names = names or {}
    cands = candidates(result, names)
    latest = result.version_pair.current_max_week if result.version_pair is not None else 0
    by_key = {c.key: c for c in cands}
    matches: list[dict[str, Any]] = []
    drafts: dict[str, dict[str, Any]] = {}  # one per observed change, citing every supporting notice
    for notice in notices:
        found = matcher.match(notice["text"], cands, result.dataset, latest, names) if cands else Match(NONE)
        cand = by_key.get(found.candidate)
        targets: list[Candidate] = []
        if cand is not None and cand.classification == REPLACED:
            # A replacement is explained by its removal and its backfill.
            ids = {part for e in cand.entity_ids for part in e.split("->")}
            targets = [c for c in cands if c.classification in HISTORY_CLASSES and set(c.entity_ids) & ids]
        elif cand is not None:
            targets = [cand]
        drafted = []
        for target in targets:
            entry = registry_draft(target, result.dataset, latest)
            if entry is None:
                continue
            entry = drafts.setdefault(entry["event_id"], entry)
            entry["notices"].append({"id": notice["id"], "text": notice["text"], "matched_by": matcher.name})
            drafted.append(entry["event_id"])
        matches.append(
            {
                "notice": notice,
                "candidate": asdict(cand) if cand is not None else None,
                "action": "registry_draft" if drafted else ("annotation" if cand is not None else "none"),
                "drafts": drafted,
                "details": found.details,
            }
        )
    return {
        "dataset": result.dataset,
        "status": result.status,
        "latest_week": latest,
        "matcher": matcher.name,
        "candidates": [asdict(c) for c in cands],
        "matches": matches,
        "registry_drafts": list(drafts.values()),
    }
