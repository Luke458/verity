"""Confirmed-only operational store (architecture sections 11, 33).

A single SQLite database holds:

- ``runs``: immutable machine payloads plus optional feature vectors;
- ``outcomes``: append-only analyst confirmations; retrieval only ever uses
  the latest outcome and only when it is confirmed;
- ``registry``: revisioned expected-event entries (latest revision per event);
- ``relationships``: confirmed or candidate entity relationships.

The flat JSONL stores remain for lightweight use; this is the durable store
with history and confirmation semantics.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .decisions import FeatureEncoder
from .evidence_text import evidence_text as build_evidence_text
from .incidents import cosine_similarity
from .jsonutil import dumps as json_dumps
from .relationships import EntityRelationship

SCHEMA_VERSION = 2
PROVENANCE_VALUES = (
    "analyst",
    "synthetic",
    "imported",
    "agent_draft",
    "unknown",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    created TEXT NOT NULL,
    status TEXT NOT NULL,
    features TEXT,
    feature_version INTEGER,
    evidence_text TEXT,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    created TEXT NOT NULL,
    analyst TEXT,
    confirmed INTEGER NOT NULL DEFAULT 0,
    requires_investigation INTEGER,
    provenance TEXT NOT NULL
        CHECK (provenance IN ('analyst','synthetic','imported','agent_draft','unknown')),
    root_cause TEXT,
    likely_origin TEXT,
    severity TEXT,
    resolution TEXT,
    summary TEXT,
    symptom_tags TEXT
);
CREATE TABLE IF NOT EXISTS registry (
    event_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    created TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (event_id, revision)
);
CREATE TABLE IF NOT EXISTS relationships (
    relationship_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    relationship TEXT NOT NULL,
    confirmed INTEGER NOT NULL DEFAULT 0,
    created TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE (source_id, target_id, entity_type, relationship)
);
"""


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "confirmed")


def import_outcomes(
    store: SqliteStore,
    rows: Sequence[dict[str, Any]],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import analyst outcomes from flat rows (CSV or Delta).

    Required: ``run_id`` and ``root_cause``. Optional: ``likely_origin``
    (or ``origin``), ``severity``, ``resolution``, ``summary``, ``analyst``,
    ``confirmed``, ``requires_investigation``, ``symptom_tags`` (comma
    separated), ``created``. Unknown runs are reported, never created.
    """
    imported = 0
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        run_id = str(row.get("run_id") or "").strip()
        root_cause = str(row.get("root_cause") or "").strip()
        if not run_id or not root_cause:
            errors.append(
                {
                    "row": index,
                    "error": "run_id and root_cause are required",
                }
            )
            continue
        tags_raw = row.get("symptom_tags") or ""
        if isinstance(tags_raw, str):
            tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]
        else:
            tags = [str(tag) for tag in tags_raw]
        requires_raw = row.get("requires_investigation")
        requires = (
            None
            if requires_raw is None or str(requires_raw).strip() == ""
            else _as_bool(requires_raw)
        )
        if dry_run:
            if store._run_row(run_id) is None:
                errors.append({"row": index, "error": f"unknown run {run_id!r}"})
                continue
            imported += 1
            continue
        try:
            store.record_outcome(
                run_id,
                root_cause=root_cause,
                confirmed=_as_bool(row.get("confirmed", False)),
                likely_origin=str(
                    row.get("likely_origin") or row.get("origin") or "UNKNOWN"
                ),
                severity=str(row.get("severity") or "MEDIUM"),
                resolution=str(row.get("resolution") or ""),
                summary=str(row.get("summary") or ""),
                analyst=(
                    str(row["analyst"]) if row.get("analyst") not in (None, "") else None
                ),
                symptom_tags=tags,
                requires_investigation=requires,
                provenance=str(row.get("provenance") or "imported"),
                created=(
                    str(row["created"]) if row.get("created") not in (None, "") else None
                ),
            )
        except ValueError as error:
            errors.append({"row": index, "error": str(error)})
            continue
        imported += 1
    return {"imported": imported, "errors": errors, "dry_run": dry_run}


class SqliteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        self.connection.commit()

    def _migrate(self) -> None:
        """Create or version-check the schema.

        A store from an older breaking schema is refused rather than silently
        altered: no production data exists yet, and a silent ALTER cannot add
        the provenance CHECK constraint.
        """
        row = self.connection.execute("PRAGMA user_version").fetchone()
        version = int(row[0]) if row is not None else 0
        if version == 0:
            self.connection.executescript(SCHEMA)
            self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"store schema version {version} is not supported by this "
                f"build (expected {SCHEMA_VERSION}); recreate the store"
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def record_run(
        self,
        run_id: str,
        dataset: str,
        status: str,
        payload: dict[str, Any],
        features: Sequence[float] | None = None,
        feature_version: int | None = None,
        evidence_text: str | None = None,
        created: str | None = None,
    ) -> None:
        try:
            self.connection.execute(
                "INSERT INTO runs (run_id, dataset, created, status, features, "
                "feature_version, evidence_text, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    dataset,
                    created or "",
                    status,
                    json.dumps(list(features)) if features is not None else None,
                    feature_version,
                    evidence_text,
                    json_dumps(payload, sort_keys=True, default=str),
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"run {run_id!r} is already recorded; runs are immutable"
            ) from error
        self.connection.commit()

    def record_result(
        self, result: Any, created: str | None = None
    ) -> str:
        encoder = FeatureEncoder()
        self.record_run(
            run_id=result.run_id,
            dataset=getattr(result, "dataset", "unknown"),
            status=result.status,
            payload=result.machine,
            features=encoder.encode(result).tolist(),
            feature_version=encoder.feature_version,
            evidence_text=build_evidence_text(result),
            created=created,
        )
        return str(result.run_id)

    def _run_row(self, run_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT r.run_id, r.dataset, r.status, r.created,
                   o.root_cause, o.confirmed, o.resolution
            FROM runs AS r
            LEFT JOIN outcomes AS o ON o.outcome_id = (
                SELECT outcome_id FROM outcomes
                WHERE run_id = r.run_id
                ORDER BY outcome_id DESC LIMIT 1
            )
            ORDER BY r.created, r.run_id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Outcomes
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        run_id: str,
        root_cause: str,
        confirmed: bool = False,
        likely_origin: str = "UNKNOWN",
        severity: str = "MEDIUM",
        resolution: str = "",
        summary: str = "",
        analyst: str | None = None,
        symptom_tags: Sequence[str] = (),
        requires_investigation: bool | None = None,
        provenance: str = "unknown",
        created: str | None = None,
    ) -> int:
        if provenance not in PROVENANCE_VALUES:
            raise ValueError(
                f"provenance {provenance!r} is not one of {PROVENANCE_VALUES}"
            )
        if self._run_row(run_id) is None:
            raise ValueError(f"unknown run {run_id!r}; record the run first")
        cursor = self.connection.execute(
            "INSERT INTO outcomes (run_id, created, analyst, confirmed, "
            "requires_investigation, provenance, root_cause, likely_origin, "
            "severity, resolution, summary, symptom_tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                created or "",
                analyst,
                int(confirmed),
                None if requires_investigation is None else int(requires_investigation),
                provenance,
                root_cause,
                likely_origin,
                severity,
                resolution,
                summary,
                json.dumps(list(symptom_tags)),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid or 0)

    def latest_outcome(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM outcomes WHERE run_id = ? "
            "ORDER BY outcome_id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def confirmed_runs(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT r.*, o.root_cause, o.likely_origin, o.severity,
                   o.resolution, o.summary, o.symptom_tags, o.analyst,
                   o.requires_investigation, o.provenance
            FROM runs AS r
            JOIN outcomes AS o ON o.outcome_id = (
                SELECT outcome_id FROM outcomes
                WHERE run_id = r.run_id AND confirmed = 1
                ORDER BY outcome_id DESC LIMIT 1
            )
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def retrieve(
        self,
        features: Sequence[float] | np.ndarray,
        k: int = 3,
        tags: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Confirmed-only similarity retrieval, blending stored features."""
        query = np.asarray(features, dtype=float)
        query_tags = {str(tag) for tag in (tags or [])}
        scored: list[dict[str, Any]] = []
        for row in self.confirmed_runs():
            if not row.get("features"):
                continue
            vector = np.asarray(json.loads(row["features"]), dtype=float)
            if len(vector) != len(query):
                continue
            score = cosine_similarity(query, vector)
            record_tags = set(json.loads(row.get("symptom_tags") or "[]"))
            if query_tags and record_tags:
                union = query_tags | record_tags
                score += 0.1 * (len(query_tags & record_tags) / len(union))
            scored.append(
                {
                    "run_id": row["run_id"],
                    "dataset": row["dataset"],
                    "root_cause": row["root_cause"],
                    "likely_origin": row["likely_origin"],
                    "severity": row["severity"],
                    "resolution": row["resolution"],
                    "summary": row["summary"],
                    "similarity": score,
                }
            )
        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------
    # Revisioned expected-event registry
    # ------------------------------------------------------------------

    def save_registry(
        self,
        entries: Sequence[dict[str, Any]],
        revision: int | None = None,
        created: str | None = None,
    ) -> int:
        if revision is None:
            row = self.connection.execute(
                "SELECT COALESCE(MAX(revision), 0) AS current FROM registry"
            ).fetchone()
            revision = int(row["current"]) + 1
        for entry in entries:
            event_id = str(entry["event_id"])
            self.connection.execute(
                "INSERT OR REPLACE INTO registry (event_id, revision, created, "
                "payload) VALUES (?, ?, ?, ?)",
                (
                    event_id,
                    int(revision),
                    created or "",
                    json.dumps(entry, sort_keys=True),
                ),
            )
        self.connection.commit()
        return int(revision)

    def load_registry(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT payload FROM registry AS current
            WHERE revision = (
                SELECT MAX(revision) FROM registry
                WHERE event_id = current.event_id
            )
            ORDER BY event_id
            """
        ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def registry_revisions(self) -> list[int]:
        rows = self.connection.execute(
            "SELECT DISTINCT revision FROM registry ORDER BY revision"
        ).fetchall()
        return [int(row["revision"]) for row in rows]

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def add_relationship(
        self, relationship: EntityRelationship, created: str | None = None
    ) -> bool:
        cursor = self.connection.execute(
            "INSERT INTO relationships (source_id, target_id, "
            "entity_type, relationship, confirmed, created, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (source_id, target_id, entity_type, relationship) "
            "DO UPDATE SET confirmed = excluded.confirmed, "
            "created = excluded.created, payload = excluded.payload",
            (
                relationship.source_id,
                relationship.target_id,
                relationship.entity_type,
                relationship.relationship,
                int(relationship.confirmed),
                created or "",
                json_dumps(relationship.to_dict(), sort_keys=True),
            ),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def relationships(
        self, confirmed_only: bool = False
    ) -> list[EntityRelationship]:
        query = "SELECT payload, confirmed FROM relationships"
        if confirmed_only:
            query += " WHERE confirmed = 1"
        rows = self.connection.execute(query).fetchall()
        return [EntityRelationship.from_dict(json.loads(row["payload"])) for row in rows]

    def for_entity(
        self, entity_type: str, entity_ids: Sequence[str]
    ) -> list[EntityRelationship]:
        wanted = {str(entity) for entity in entity_ids}
        return [
            relationship
            for relationship in self.relationships()
            if relationship.entity_type == entity_type
            and (
                relationship.source_id in wanted
                or relationship.target_id in wanted
            )
        ]
