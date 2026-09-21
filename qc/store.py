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

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .decisions import FeatureEncoder
from .evidence_text import evidence_text as build_evidence_text
from .incidents import cosine_similarity
from .jsonutil import dumps as json_dumps
from .relationships import EntityRelationship

SCHEMA_VERSION = 6
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
    text_version INTEGER,
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
    symptom_tags TEXT,
    incident_group TEXT
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


JOURNAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
 assessment_id TEXT PRIMARY KEY, identity TEXT NOT NULL, status TEXT NOT NULL,
 artifacts TEXT NOT NULL, checksums TEXT NOT NULL, result TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
 attempt_id TEXT PRIMARY KEY, assessment_id TEXT NOT NULL,
 observed_at TEXT NOT NULL, state TEXT NOT NULL, error TEXT
);
CREATE TABLE IF NOT EXISTS calibration_revisions (
 assessment_id TEXT NOT NULL, series_id TEXT NOT NULL, target_week INTEGER NOT NULL,
 observed_at TEXT NOT NULL, payload TEXT NOT NULL,
 PRIMARY KEY (assessment_id, series_id, target_week)
);
"""


def observation_time(value: str | None = None) -> str:
    timestamp = datetime.fromisoformat(value) if value else datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).isoformat()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "confirmed")


def _version_order(value: Any) -> tuple[int, object]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


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
    if dry_run:
        clone = object.__new__(SqliteStore)
        clone.connection = sqlite3.connect(":memory:")
        clone.connection.row_factory = sqlite3.Row
        clone._transaction_depth = 0
        store.connection.backup(clone.connection)
        try:
            report = import_outcomes(clone, rows, dry_run=False)
            return {**report, "dry_run": True}
        finally:
            clone.close()
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
                incident_group=row.get("incident_group") or None,
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
        self._transaction_depth = 0
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        self._commit()

    def _migrate(self) -> None:
        """Create or version-check the schema.

        A store from an older breaking schema is refused rather than silently
        altered: no production data exists yet, and a silent ALTER cannot add
        the provenance CHECK constraint.
        """
        row = self.connection.execute("PRAGMA user_version").fetchone()
        version = int(row[0]) if row is not None else 0
        if version not in (0, 1, 2, 3, 4, 5, SCHEMA_VERSION):
            raise ValueError(f"unsupported store schema version {version}")
        existing = self.connection.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'runs'").fetchone()
        if version in (1, 2, 3, 4, 5) or (version == 0 and existing is not None):
            backup = self.path.with_name(self.path.name + f".v{version}.backup-{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}")
            with sqlite3.connect(backup) as target:
                self.connection.backup(target)
        self.connection.execute("BEGIN IMMEDIATE")
        with self.connection:
            for statement in (SCHEMA + JOURNAL_SCHEMA).split(";"):
                if statement.strip():
                    self.connection.execute(statement)
            columns = {row[1] for row in self.connection.execute("PRAGMA table_info(outcomes)")}
            for name, definition in {
                "provenance": "TEXT NOT NULL DEFAULT 'unknown'",
                "requires_investigation": "INTEGER",
                "incident_group": "TEXT",
            }.items():
                if name not in columns:
                    self.connection.execute(f"ALTER TABLE outcomes ADD COLUMN {name} {definition}")
            run_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(runs)")}
            if "text_version" not in run_columns:
                self.connection.execute("ALTER TABLE runs ADD COLUMN text_version INTEGER")
            self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _commit(self) -> None:
        if not self._transaction_depth:
            self.connection.commit()

    @contextmanager
    def transaction(self):
        self._transaction_depth += 1
        try:
            with self.connection:
                yield
        finally:
            self._transaction_depth -= 1

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
        text_version: int | None = None,
        created: str | None = None,
    ) -> None:
        try:
            self.connection.execute(
                "INSERT INTO runs (run_id, dataset, created, status, features, "
                "feature_version, evidence_text, text_version, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    dataset,
                    observation_time(created),
                    status,
                    json_dumps(list(features)) if features is not None else None,
                    feature_version,
                    evidence_text,
                    text_version,
                    json_dumps(payload, sort_keys=True, default=str),
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"run {run_id!r} is already recorded; runs are immutable"
            ) from error
        self._commit()

    def record_result(
        self, result: Any, created: str | None = None
    ) -> str:
        from .evidence_text import EVIDENCE_TEXT_VERSION

        encoder = FeatureEncoder()
        self.record_run(
            run_id=result.run_id,
            dataset=getattr(result, "dataset", "unknown"),
            status=result.status,
            payload=result.machine,
            features=encoder.encode(result).tolist(),
            feature_version=encoder.feature_version,
            evidence_text=build_evidence_text(result),
            text_version=EVIDENCE_TEXT_VERSION,
            created=created,
        )
        return str(result.run_id)

    def _run_row(self, run_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()

    def assessment(self, assessment_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM assessments WHERE assessment_id = ?", (assessment_id,)
        ).fetchone()
        return self._assessment_payload(row)

    def latest_assessment(self, dataset: str | None = None) -> dict[str, Any] | None:
        rows = self.connection.execute(
            "SELECT * FROM assessments ORDER BY rowid DESC"
        ).fetchall()
        for row in rows:
            payload = self._assessment_payload(row)
            if payload is None:
                continue
            if dataset is None or payload["result"].get("dataset") == dataset:
                return payload
        return None

    @staticmethod
    def _assessment_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "assessment_id": row["assessment_id"],
            "identity": json.loads(row["identity"]),
            "status": row["status"],
            "artifacts": json.loads(row["artifacts"]),
            "checksums": json.loads(row["checksums"]),
            "result": json.loads(row["result"]),
        }

    def recurrence_inputs(
        self,
        dataset: str,
        *,
        cutoff: str,
        window: int = 2,
        current_version: str | None = None,
    ) -> list[dict[str, Any]]:
        """Frozen recurrence predecessors strictly before the cutoff.

        Distinct logical refreshes are identified by their version pair; the
        latest eligible revision of each refresh is selected at the cutoff and
        retries or alternative attempts of the same refresh are excluded as
        duplicate evidence.
        """
        rows = self.connection.execute(
            "SELECT run_id, created, payload FROM runs "
            "WHERE dataset = ? AND created <= ? "
            "ORDER BY created DESC, run_id DESC",
            (dataset, observation_time(cutoff)),
        ).fetchall()
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            payload = json.loads(row["payload"])
            pair = payload.get("version_pair") or {}
            previous_id = pair.get("previous_id")
            current_id = pair.get("current_id")
            key = (str(previous_id), str(current_id))
            if key in latest:
                continue
            if current_version is not None and _version_order(current_id) >= _version_order(current_version):
                continue
            latest[key] = {
                "run_id": row["run_id"],
                "created": row["created"],
                "previous_version": previous_id,
                "current_version": current_id,
                "previous_max_week": pair.get("previous_max_week"),
                "current_max_week": pair.get("current_max_week"),
                "payload_hash": hashlib.sha256(
                    str(row["payload"]).encode()
                ).hexdigest(),
            }
        ordered = sorted(
            latest.values(),
            key=lambda entry: (
                # Snapshot/business-period metadata orders predecessors; version
                # string ordering is only a fallback for legacy payloads.
                int(entry["previous_max_week"] or 0),
                int(entry["current_max_week"] or 0),
                _version_order(entry["previous_version"]),
                _version_order(entry["current_version"]),
                str(entry["created"]),
            ),
        )
        limit = max(0, int(window) - 1)
        return ordered[-limit:] if limit else []

    def recurrence_refreshes(
        self, entries: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Load the frozen predecessor payloads and verify their hashes."""
        refreshes: list[dict[str, Any]] = []
        for entry in entries:
            row = self.connection.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (entry["run_id"],)
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"recurrence predecessor {entry['run_id']!r} is unavailable"
                )
            digest = hashlib.sha256(str(row["payload"]).encode()).hexdigest()
            if digest != entry.get("payload_hash"):
                raise ValueError(
                    f"recurrence predecessor {entry['run_id']!r} evidence changed"
                )
            payload = json.loads(row["payload"])
            historical = payload.get("historical_revision") or {}
            refreshes.append(
                {
                    "run_id": payload.get("run_id"),
                    "status": payload.get("status"),
                    "findings": payload.get("findings", []),
                    "ledger": payload.get("ledger"),
                    "unexplained_delta": historical.get("unexplained_delta", 0.0),
                    "materiality_threshold": payload.get("materiality_threshold", 0.0),
                    "payload_hash": digest,
                }
            )
        return refreshes

    def recent_refreshes(
        self, dataset: str, limit: int = 2
    ) -> list[dict[str, Any]]:
        """Most recent stored machine payloads for recurrence assessment."""
        rows = self.connection.execute(
            "SELECT payload FROM runs WHERE dataset = ? "
            "ORDER BY created DESC, run_id DESC LIMIT ?",
            (dataset, int(limit)),
        ).fetchall()
        refreshes: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload"])
            historical = payload.get("historical_revision") or {}
            refreshes.append(
                {
                    "run_id": payload.get("run_id"),
                    "status": payload.get("status"),
                    "findings": payload.get("findings", []),
                    "ledger": payload.get("ledger"),
                    "unexplained_delta": historical.get("unexplained_delta", 0.0),
                    "materiality_threshold": payload.get("materiality_threshold", 0.0),
                }
            )
        return refreshes

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
        incident_group: str | None = None,
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
            "severity, resolution, summary, symptom_tags, incident_group) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                observation_time(created),
                analyst,
                int(confirmed),
                None if requires_investigation is None else int(requires_investigation),
                provenance,
                root_cause,
                likely_origin,
                severity,
                resolution,
                summary,
                json_dumps(list(symptom_tags)),
                incident_group,
            ),
        )
        self._commit()
        return int(cursor.lastrowid or 0)

    def latest_outcome(self, run_id: str, as_of: str | None = None) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM outcomes WHERE run_id = ? AND created != '' AND created <= ? "
            "ORDER BY created DESC, outcome_id DESC LIMIT 1",
            (run_id, observation_time(as_of)),
        ).fetchone()
        return dict(row) if row is not None else None

    def confirmed_runs(self, as_of: str | None = None) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT r.*, o.root_cause, o.likely_origin, o.severity,
                   o.resolution, o.summary, o.symptom_tags, o.analyst,
                   o.requires_investigation, o.provenance, o.incident_group
            FROM runs AS r
            JOIN outcomes AS o ON o.outcome_id = (
                SELECT outcome_id FROM outcomes
                WHERE run_id = r.run_id AND created != '' AND created <= ?
                ORDER BY created DESC, outcome_id DESC LIMIT 1
            ) WHERE o.confirmed = 1 AND r.created != '' AND r.created <= ?
            """, (observation_time(as_of), observation_time(as_of))
        ).fetchall()
        return [dict(row) for row in rows]

    def retrieve(
        self,
        features: Sequence[float] | np.ndarray,
        k: int = 3,
        tags: Sequence[str] | None = None,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """Confirmed-only similarity retrieval, blending stored features."""
        query = np.asarray(features, dtype=float)
        query_tags = {str(tag) for tag in (tags or [])}
        scored: list[dict[str, Any]] = []
        for row in self.confirmed_runs(as_of):
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
                "INSERT INTO registry (event_id, revision, created, "
                "payload) VALUES (?, ?, ?, ?)",
                (
                    event_id,
                    int(revision),
                    observation_time(created),
                    json_dumps(entry, sort_keys=True),
                ),
            )
        self._commit()
        return int(revision)

    def load_registry(self, as_of: str | None = None) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT payload FROM registry AS current
            WHERE revision = (
                SELECT MAX(revision) FROM registry
                WHERE event_id = current.event_id AND created != '' AND created <= ?
            )
            ORDER BY event_id
            """, (observation_time(as_of),)
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
                observation_time(created),
                json_dumps(relationship.to_dict(), sort_keys=True),
            ),
        )
        self._commit()
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
