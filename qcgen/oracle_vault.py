"""Ground-truth vault.

Scenario *data* (Parquet versions plus a data-only ``manifest.json``) must be
readable by the engine without exposing the fault oracle. Ground truth is
written to a separate root through this API, and only scoring harnesses may
read it. The vault deliberately knows nothing about the engine: it is a plain
file store for oracle payloads.

Layout::

    <oracle_root>/<scenario_id>.json

Each payload contains the scenario's family, fault spec, cases and any
registry entries that were declared as expected events.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .oracle import json_default

ORACLE_KEYS = ("family", "fault", "cases", "expected_events")


class OracleVault:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, scenario_id: str) -> Path:
        return self.root / f"{scenario_id}.json"

    def write(self, scenario_id: str, payload: dict[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path(scenario_id)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=json_default)
        )
        return path

    def read(self, scenario_id: str) -> dict[str, Any] | None:
        path = self.path(scenario_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def require(self, scenario_id: str) -> dict[str, Any]:
        payload = self.read(scenario_id)
        if payload is None:
            raise FileNotFoundError(
                f"no oracle payload for {scenario_id!r} under {self.root}"
            )
        return payload

    def exists(self, scenario_id: str) -> bool:
        return self.path(scenario_id).exists()

    def scenario_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(path.stem for path in self.root.glob("*.json"))

    def family(self, scenario_id: str) -> str | None:
        payload = self.read(scenario_id)
        return payload.get("family") if payload else None

    def cases(self, scenario_id: str) -> list[dict[str, Any]]:
        payload = self.read(scenario_id) or {}
        return list(payload.get("cases", []))

    def first_case(self, scenario_id: str) -> dict[str, Any]:
        cases = self.cases(scenario_id)
        return cases[0] if cases else {}

    def expected_events(self, scenario_id: str) -> list[dict[str, Any]]:
        payload = self.read(scenario_id) or {}
        return list(payload.get("expected_events", []))


def default_oracle_root(suite_dir: str | Path) -> Path:
    """Oracle root adjacent to a suite directory.

    ``data/suites/demo`` -> ``data/suites/oracle/demo``. Callers that generate
    suites should pass the root explicitly; this helper exists for harnesses
    reading existing suites and is used as the generator default.
    """
    suite_dir = Path(suite_dir)
    return suite_dir.parent / "oracle" / suite_dir.name


def default_registry_root(suite_dir: str | Path) -> Path:
    suite_dir = Path(suite_dir)
    return suite_dir.parent / "registry" / suite_dir.name
