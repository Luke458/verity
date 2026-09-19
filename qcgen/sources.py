"""Development VersionSource over a generated scenario directory.

The QC engine depends only on the ``qc.source.VersionSource`` protocol; this
adapter is how local suites and tests satisfy it. A Delta-backed source will
implement the same methods against Unity Catalog tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .snapshots import SnapshotStore


class ScenarioSource:
    def __init__(
        self,
        scenario_dir: str | Path,
        stage_preference: Sequence[str] = ("warehouse", "coded", "source", "report"),
    ) -> None:
        self.scenario_dir = Path(scenario_dir)
        self.store = SnapshotStore(self.scenario_dir)
        self.stage_preference = tuple(stage_preference)

    def list_versions(self) -> list[str]:
        versions_dir = self.scenario_dir / "versions"
        if not versions_dir.exists():
            return []
        return sorted(path.name for path in versions_dir.iterdir() if path.is_dir())

    def available_stages(self, version: str) -> list[str]:
        return self.store.available_stages(self.scenario_dir, version)

    def resolve_stage(self, version: str, preferred: Sequence[str]) -> str:
        available = self.available_stages(version)
        for stage in preferred:
            if stage in available:
                return stage
        if not available:
            raise FileNotFoundError(
                f"version {version!r} has no stages under {self.scenario_dir}"
            )
        return available[-1]

    def read_fact(self, version: str, stage: str) -> pd.DataFrame:
        return self.store.read_fact(self.scenario_dir, version, stage)

    def read_dim(self, version: str, name: str) -> pd.DataFrame | None:
        dims = self.store.read_dims(self.scenario_dir, version)
        return dims.get(name)
