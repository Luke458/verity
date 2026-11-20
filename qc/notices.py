"""Draft explanations from free-text change notices.

Operators hear about structural changes as text ("S012 closed for refit from
week 118"). ``draft`` matches each notice to one of the refresh's observed
lifecycle changes (``candidates``). Where the expected-event registry can
express that change, it writes an **unapproved** registry entry that cites the
notice. A person sets ``approved_by``/``approved_at`` and ``confirmed`` before
the engine will use it; nothing here changes a status.

Two matchers share the same deterministic rules (``decide``) and differ in how
they read a notice: ``PatternMatcher`` (regexes, the default) and
``ExtractMatcher``, which asks an extraction service (System One: Extract, for
example Clairic-flash) to fill ``NOTICE_FIELDS``. docs/notice-matching.md
measures both, and a decision model that chose among the candidates itself
(removed after it lost on false explanations).
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
    REPLACED,
    TRUNCATED,
)

NONE = "none"
# Historical changes: the registry scopes them by the weeks they rewrite.
HISTORY_CLASSES = frozenset({NEW_BACKFILL, REMOVED, EXTENDED, TRUNCATED})
# Closures and category moves: scoped by the weeks they are in effect.
EFFECT_CLASSES = frozenset({LATEST_MISSING, RECLASSIFIED})
REGISTRY_CLASSES = HISTORY_CLASSES | EFFECT_CLASSES | {REPLACED}
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
# Phrases that name the old id before the new one ("S005 merged into S900").
MERGE_FORWARD = re.compile(
    r"\b(?:merged into|replaced by|now trades as|rebranded (?:to|as)|consolidated into)\b", re.IGNORECASE
)
WEEKS = re.compile(r"\b(?:weeks?|wk)\s*(\d+)(?:\s*(?:-|to|through)\s*(\d+))?", re.IGNORECASE)


def _ids(text: str, names: Mapping[str, str]) -> set[str]:
    found = {f"S{int(n):03d}" for n in re.findall(r"\bS(\d{3})\b", text)}
    found |= {f"S{int(n):03d}" for n in re.findall(r"\bstore\s+S?(\d{1,3})\b", text, re.IGNORECASE)}
    found |= {f"P{int(n):04d}" for n in re.findall(r"\bP(\d{4})\b", text)}
    found |= {f"P{int(n):04d}" for n in re.findall(r"\bproduct\s+(\d{1,4})\b", text, re.IGNORECASE)}
    found |= set(re.findall(r"\bC\d{2}\b", text))
    found |= {cid for cid, name in names.items() if cid.startswith("C") and name.lower() in text.lower()}
    return found


@dataclass
class Reading:
    """What a notice says, in the terms the matching rules use."""

    other_dataset: bool
    change_type: str | None  # an ACCEPTS key, or None when no known kind of change
    planned: bool
    ids: set[str]
    spans: list[tuple[int, int | None]]  # stated week ranges, first one used
    merge_targets: set[str] | None = None  # a merge's new ids; None when the direction is not read


def _merge_fits(reading: Reading, cand: Candidate) -> bool:
    """A merge explains a new entity only through its new id, a removal only through an old one."""
    targets = reading.merge_targets
    if targets is None:
        return True
    shared = reading.ids & {part for e in cand.entity_ids for part in e.split("->")}
    if cand.classification == NEW_BACKFILL:
        return bool(shared & targets)
    if cand.classification == REMOVED:
        return bool(shared - targets)
    if cand.classification == REPLACED:  # entity_ids are (old, new) pairs
        pairs = zip(cand.entity_ids[0::2], cand.entity_ids[1::2], strict=False)
        return any((old in shared and old not in targets) or new in targets for old, new in pairs)
    return True


def decide(reading: Reading, cands: Sequence[Candidate], latest_week: int) -> Match:
    """Deterministic rules: another dataset, kind of change, tense, a shared entity id, weeks."""
    if reading.other_dataset:
        return Match(NONE, {"reason": "another dataset"})
    change_type = reading.change_type
    if change_type is None:
        return Match(NONE, {"reason": "no change cue"})
    if reading.planned:
        return Match(NONE, {"reason": "not yet in effect"})
    spans = reading.spans
    if change_type == "entity_merge":
        # A merge is one replacement: drafting it covers both the removal and
        # the backfill, where either alone would leave the other unexplained.
        cands = sorted(cands, key=lambda c: c.classification != REPLACED)
    for cand in cands:
        if cand.classification not in ACCEPTS[change_type]:
            continue
        if not reading.ids & {part for e in cand.entity_ids for part in e.split("->")}:
            continue
        if change_type == "entity_merge" and not _merge_fits(reading, cand):
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


def read_patterns(text: str, dataset: str, names: Mapping[str, str]) -> Reading:
    tag = re.match(r"^\[(.+?)\]", text)
    forward = MERGE_FORWARD.search(text)
    after = _ids(text[forward.end():], names) if forward else set()
    return Reading(
        other_dataset=bool(tag and tag.group(1) != dataset),
        change_type=next((t for pattern, t in CUES if re.search(pattern, text, re.IGNORECASE)), None),
        planned=bool(FUTURE.search(text)),
        ids=_ids(text, names),
        spans=[(int(a), int(b) if b else None) for a, b in WEEKS.findall(text)],
        merge_targets=after or None,
    )


class PatternMatcher:
    """Dataset tag, change-type cue, a shared entity id and consistent weeks.

    Written for the synthetic id formats (S###, P####, C##); a deployment
    would adapt the id patterns to its own keys.
    """

    name = "patterns"

    def match(
        self, text: str, cands: Sequence[Candidate], dataset: str, latest_week: int, names: Mapping[str, str]
    ) -> Match:
        return decide(read_patterns(text, dataset, names), cands, latest_week)


DATASET_DESCRIPTION = "weekly point-of-sale sales of a pharmacy chain, by store and product"


# An extraction model reads the notice into the fields ``decide`` needs; the
# matching rules stay deterministic and identical to the pattern baseline's.
NOTICE_SCHEMA_VERSION = 2
CHANGE_PHRASES = {
    "store closed, not trading or sending no sales": "missing_stores",
    "products discontinued, delisted or out of stock": "missing_products",
    "new store added with past history": "new_store_backfill",
    "sales history deleted or withdrawn": "history_truncation",
    "products moved to another category": "commodity_remap",
    "store merged into or replaced by another store": "entity_merge",
    "something else": None,
}
NOTICE_FIELDS: dict[str, Any] = {
    "data_source": {
        "type": "text",
        "instructions": "The name of the dataset, data feed or data source the notice says is affected, as "
        "written, including a [tag] at the start of the notice (for example 'Wholesale shipments', 'loyalty "
        "panel' or 'sales feed'). Null if the notice names none; words such as data, history or records alone "
        "are not a name.",
    },
    "change": {
        "type": "enum",
        "enum": list(CHANGE_PHRASES),
        "instructions": "The kind of change the notice reports.",
    },
    "planned": {
        "type": "boolean",
        "instructions": "True only if the change itself has not happened yet (planned or upcoming). False if "
        "it has happened or is in effect; a later reopening, reversal or resupply does not make it planned.",
    },
    "stores": {
        "type": "list",
        "items": "text",
        "instructions": "Every store the change applies to, as written (for example 'S012' or 'store 12').",
    },
    "products": {
        "type": "list",
        "items": "text",
        "instructions": "Every product the change applies to, as written (for example 'P0042' or 'item 42').",
    },
    "categories": {
        "type": "list",
        "items": "text",
        "instructions": "Every product category the notice mentions, as written (a code such as 'C04' or a name).",
    },
    "first_week": {"type": "number", "integer": True, "instructions": "The first week number the change applies to."},
    "last_week": {"type": "number", "integer": True, "instructions": "The last week number the change applies to."},
}
_GENERIC = frozenset({
    "a", "and", "by", "data", "dataset", "entry", "feed", "file", "for", "history", "in", "information",
    "of", "on", "record", "table", "the", "trading",
})


def _words(text: str) -> set[str]:
    words = (w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", text.lower()))
    return {w for w in words if w not in _GENERIC}


def extract_request(text: str) -> dict[str, Any]:
    """A System One: Extract body for one notice; the candidates are not shown to the model."""
    return {"state": text, "fields": NOTICE_FIELDS}


def reading_from_answers(
    answers: Mapping[str, Any],
    dataset: str,
    names: Mapping[str, str],
    description: str = DATASET_DESCRIPTION,
    latest_week: int | None = None,
    text: str | None = None,
) -> Reading:
    """Extracted values -> ``Reading``.

    A named source is another dataset when it shares no word with the dataset's
    name or description; ids normalize as in the patterns. Stated weeks override
    tense: a change that starts by ``latest_week`` is in effect. With ``text``,
    ids the patterns find in the notice are added to the model's.
    """
    value = {name: (answer or {}).get("value") for name, answer in answers.items()}
    source = value.get("data_source")
    words = _words(source) if isinstance(source, str) else set()
    listed = [
        *(f"store {s}" for s in value.get("stores") or ()),
        *(f"product {p}" for p in value.get("products") or ()),
        *(str(c) for c in value.get("categories") or ()),
    ]
    first, last = value.get("first_week"), value.get("last_week")
    start = first if first is not None else last
    started = start is not None and latest_week is not None and int(start) <= latest_week
    return Reading(
        other_dataset=bool(words) and not words & _words(f"{dataset} {description}"),
        change_type=CHANGE_PHRASES.get(value.get("change") or "something else"),
        planned=value.get("planned") is True and not started,
        ids=_ids(" ; ".join(listed), names) | (_ids(text, names) if text else set()),
        spans=[(int(start), int(last) if last is not None else None)] if start is not None else [],
        # NOTICE_FIELDS has no new-id field, so a merge's direction is not read
        # (the model lists one store for most merges): the old any-shared-id rule.
    )


class ExtractMatcher:
    """An extraction service (System One: Extract, e.g. Clairic-flash) reads the notice; ``decide`` matches."""

    name = "extract"

    def __init__(
        self,
        endpoint: str,
        model: str = "clairic-flash",
        timeout: float = 120.0,
        dataset_description: str = DATASET_DESCRIPTION,
        token: str | None = None,
    ):
        url = endpoint.rstrip("/")
        self.url = url if "/v1/" in url or "/ai/run/" in url else url + "/v1/extract"
        from .notify import _validate_webhook_target

        _validate_webhook_target(self.url)  # https, or http to loopback only
        self.model, self.timeout, self.token = model, timeout, token
        self.dataset_description = dataset_description

    def _extract(self, text: str) -> dict[str, Any]:
        import urllib.request

        from .notify import _OPENER

        body = json.dumps({"model": self.model, **extract_request(text)}).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(self.url, data=body, headers=headers)
        with _OPENER.open(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        payload = payload.get("result", payload)  # Workers AI wraps the body in "result"
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("extraction service returned no answers")
        return answers

    def match(
        self, text: str, cands: Sequence[Candidate], dataset: str, latest_week: int, names: Mapping[str, str]
    ) -> Match:
        if not cands:
            return Match(NONE)
        answers = self._extract(text)
        reading = reading_from_answers(answers, dataset, names, self.dataset_description, latest_week)
        result = decide(reading, cands, latest_week)
        extracted = {
            name: {k: a.get(k) for k in ("value", "confidence", "p_null", "span", "spans") if k in a}
            for name, a in answers.items()
            if isinstance(a, dict)
        }
        return Match(result.candidate, {**result.details, "extracted": extracted})


# ---------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------


def load_notices(path: str | Path) -> list[dict[str, str]]:
    """Notices from JSON (a list of strings or {"id", "text"}), JSONL, or plain text lines.

    ``.json`` and ``.jsonl`` files must parse. Any other file is read as JSON
    only if it parses as such, so a text file whose first notice starts with a
    ``[tag]`` is read as lines.
    """
    source = Path(path)
    raw = source.read_text()
    stripped = raw.lstrip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    items: list[Any] = lines
    if source.suffix == ".json" or (source.suffix != ".jsonl" and stripped.startswith("[")):
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            if source.suffix == ".json":
                raise
    elif source.suffix == ".jsonl" or stripped.startswith("{"):
        try:
            items = [json.loads(line) for line in lines]
        except json.JSONDecodeError:
            if source.suffix == ".jsonl":
                raise
            items = lines
    notices = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            item = {"text": item}
        text = str(item["text"]).strip()
        notices.append({"id": str(item.get("id") or f"notice-{index + 1}"), "text": text})
    return notices


def registry_draft(cand: Candidate, dataset: str, latest_week: int | None = None) -> dict[str, Any] | None:
    """An unapproved registry entry for an observed change the registry can express.

    A historical change, and a replacement (one ``old->new`` pair, see
    ``replacement_pairs``), is scoped to the weeks it rewrote. A closure or a
    category move is scoped to the refresh's latest week; the reviewer widens
    ``effective_to_week`` if the change is known to last. A move also lists
    the products observed moving, so the approval covers no others.
    """
    if cand.classification in HISTORY_CLASSES | {REPLACED} and cand.weeks:
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


def replacement_pairs(result: Any, cand: Candidate) -> list[Candidate]:
    """A replacement candidate as one ``old->new`` candidate per pair.

    Each pair carries the weeks its removal and its backfill rewrote, which
    scope its registry entry.
    """
    weeks: dict[str, set[int]] = {}
    for event in result.events or ():
        if event.entity_type == cand.entity_type:
            weeks.setdefault(str(event.entity_id), set()).update(
                int(w) for w in (*event.historical_weeks_added, *event.historical_weeks_removed))
    return [
        Candidate(cand.key, REPLACED, cand.entity_type, [f"{old}->{new}"],
                  sorted(weeks.get(old, set()) | weeks.get(new, set())), cand.description)
        for old, new in zip(cand.entity_ids[0::2], cand.entity_ids[1::2], strict=False)
    ]


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
        found = (
            matcher.match(notice["text"], cands, result.dataset, latest, names)
            if cands
            else Match(NONE, {"reason": "the refresh shows no lifecycle change"})
        )
        cand = by_key.get(found.candidate)
        targets: list[Candidate] = []
        if cand is not None and cand.classification == REPLACED:
            # One entry per pair: it explains the removal and the backfill
            # together, and only if the history carried across is conserved.
            targets = replacement_pairs(result, cand)
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
