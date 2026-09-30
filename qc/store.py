"""Durable SQLite journal for weekly assessments.

One database holds:

- ``runs``: immutable machine payloads, the frozen inputs to recurrence;
- ``assessments``: content-addressed weekly assessments with their published
  artifacts and checksums;
- ``attempts``: one row per invocation, recording its state and any error.

Opening a store from an older schema backs it up first. Retired tables from
earlier schemas (analyst outcomes, calibration revisions, registry and
relationship copies) are left in place and never read.
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

from .jsonutil import dumps as json_dumps

SCHEMA_VERSION = 7

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    created TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assessments (
 assessment_id TEXT PRIMARY KEY, identity TEXT NOT NULL, status TEXT NOT NULL,
 artifacts TEXT NOT NULL, checksums TEXT NOT NULL, result TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
 attempt_id TEXT PRIMARY KEY, assessment_id TEXT NOT NULL,
 observed_at TEXT NOT NULL, state TEXT NOT NULL, error TEXT
);
"""


def observation_time(value: str | None = None) -> str:
    timestamp = datetime.fromisoformat(value) if value else datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).isoformat()


def _version_order(value: Any) -> tuple[int, object]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


class SqliteStore:
    def __init__(self, path: str | Path):
        self._transaction_depth = 0
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self._migrate()
        self._commit()

    def _migrate(self) -> None:
        """Create the schema, backing up any older store first."""
        row = self.connection.execute("PRAGMA user_version").fetchone()
        version = int(row[0]) if row is not None else 0
        if version > SCHEMA_VERSION:
            raise ValueError(f"unsupported store schema version {version}")
        existing = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'runs'"
        ).fetchone()
        if existing is not None and version < SCHEMA_VERSION:
            stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
            backup = self.path.with_name(f"{self.path.name}.v{version}.backup-{stamp}")
            with sqlite3.connect(backup) as target:
                self.connection.backup(target)
        self.connection.execute("BEGIN IMMEDIATE")
        with self.connection:
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    self.connection.execute(statement)
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

    def record_run(
        self,
        run_id: str,
        dataset: str,
        status: str,
        payload: dict[str, Any],
        created: str | None = None,
    ) -> None:
        try:
            self.connection.execute(
                "INSERT INTO runs (run_id, dataset, created, status, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    dataset,
                    observation_time(created),
                    status,
                    json_dumps(payload, sort_keys=True, default=str),
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"run {run_id!r} is already recorded; runs are immutable"
            ) from error
        self._commit()

    def record_result(self, result: Any, created: str | None = None) -> str:
        self.record_run(
            run_id=result.run_id,
            dataset=getattr(result, "dataset", "unknown"),
            status=result.status,
            payload=result.machine,
            created=created,
        )
        return str(result.run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT run_id, dataset, status, created FROM runs ORDER BY created, run_id"
        ).fetchall()
        return [dict(row) for row in rows]

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
                    "unexplained_delta": historical.get("unexplained_delta", 0.0),
                    "materiality_threshold": payload.get("materiality_threshold", 0.0),
                    "payload_hash": digest,
                }
            )
        return refreshes
