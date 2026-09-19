"""Version source protocol and a production-column mapping decorator.

The engine reads versioned snapshots through ``VersionSource``. Local scenarios
use ``qcgen.sources.ScenarioSource``; Delta tables use ``qc.delta.DeltaSource``.
``MappedSource`` wraps any source and renames production columns to canonical
names (for example ``pfc`` -> ``product_id``) so every layer above the source
sees the same vocabulary.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class VersionSource(Protocol):
    def list_versions(self) -> list[str]: ...

    def available_stages(self, version: str) -> list[str]: ...

    def resolve_stage(self, version: str, preferred: Sequence[str]) -> str: ...

    def read_fact(self, version: str, stage: str) -> pd.DataFrame: ...

    def read_dim(self, version: str, name: str) -> pd.DataFrame | None: ...


class MappedSource:
    """Rename production columns to canonical names on read.

    ``column_map`` is canonical -> production. Columns that are missing from a
    frame are skipped and recorded in ``warnings`` (once per column); the
    engine's own contract checks then fail with a clear message if a required
    canonical column is absent.
    """

    def __init__(
        self,
        source: VersionSource,
        column_map: dict[str, str],
    ) -> None:
        self._source = source
        self.column_map = dict(column_map)
        self._rename = {
            str(production): str(canonical)
            for canonical, production in column_map.items()
            if str(canonical) != str(production)
        }
        self.warnings: list[str] = []
        self._reported_missing: set[str] = set()

    @property
    def inner(self) -> VersionSource:
        return self._source

    def _apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._rename or frame is None or len(frame.columns) == 0:
            return frame
        missing = [
            production for production in self._rename if production not in frame.columns
        ]
        for production in missing:
            if production not in self._reported_missing:
                self._reported_missing.add(production)
                self.warnings.append(
                    f"production column {production!r} not present in a read frame"
                )
        return frame.rename(columns=self._rename)

    def list_versions(self) -> list[str]:
        return self._source.list_versions()

    def available_stages(self, version: str) -> list[str]:
        return self._source.available_stages(version)

    def resolve_stage(self, version: str, preferred: Sequence[str]) -> str:
        return self._source.resolve_stage(version, preferred)

    def read_fact(self, version: str, stage: str) -> pd.DataFrame:
        return self._apply(self._source.read_fact(version, stage))

    def read_dim(self, version: str, name: str) -> pd.DataFrame | None:
        frame = self._source.read_dim(version, name)
        return self._apply(frame) if frame is not None else None


def mapped(source: VersionSource, column_map: dict[str, str] | None) -> VersionSource:
    """Wrap a source only when a non-empty mapping is configured."""
    if not column_map:
        return source
    return MappedSource(source, column_map)
