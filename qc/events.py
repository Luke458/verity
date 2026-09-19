"""Expected-event registry workflow.

Registry entries describe known changes (new store onboarding, mapping
corrections, source migrations) with the entity and history window they are
expected to affect. Attribution matches structural events against the registry;
unmatched structure remains ``INVESTIGATE``. Proposed entries from observed
backfills are drafts and must be confirmed before they are trusted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lifecycle import NEW_BACKFILL, LifecycleEvent


def load_registry(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("events", [])
    if not isinstance(data, list):
        raise ValueError("registry must be a list or an object with an 'events' list")
    return [dict(item) for item in data]


def save_registry(entries: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps({"events": list(entries)}, indent=2, sort_keys=True)
    )


def propose_expected_events(
    events: list[LifecycleEvent], source: str = "shadow"
) -> list[dict[str, Any]]:
    """Draft registry entries from observed historical backfills."""
    proposals: list[dict[str, Any]] = []
    for event in events:
        if event.classification != NEW_BACKFILL:
            continue
        weeks = list(event.historical_weeks_added)
        proposals.append(
            {
                "event_id": f"proposed-{event.entity_type}-{event.entity_id}",
                "event_type": (
                    "new_store_historical_backfill"
                    if event.entity_type == "store"
                    else "new_entity_historical_backfill"
                ),
                "entity_type": event.entity_type,
                "entity_ids": [event.entity_id],
                "effective_week": max(weeks) if weeks else None,
                "expected_history_start": min(weeks) if weeks else None,
                "expected_history_end": max(weeks) if weeks else None,
                "description": (
                    f"Proposed from an observed historical backfill of "
                    f"{event.entity_type} {event.entity_id}."
                ),
                "source": source,
                "confirmed": False,
            }
        )
    return proposals
