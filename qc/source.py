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

    def snapshot_metadata(self, version):
        method = getattr(self._source, "snapshot_metadata", None)
        return method(version) if method else {}

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
        renamed = frame.rename(columns=self._rename)
        if not renamed.columns.is_unique:
            raise ValueError("column_map creates duplicate canonical columns")
        return renamed

    def list_versions(self) -> list[str]:
        return self._source.list_versions()

    def available_stages(self, version: str) -> list[str]:
        return self._source.available_stages(version)

    def resolve_stage(self, version: str, preferred: Sequence[str]) -> str:
        return self._source.resolve_stage(version, preferred)

    def read_fact(self, version: str, stage: str) -> pd.DataFrame:
        return self._apply(self._source.read_fact(version, stage))

    def read_fact_columns(self, version, stage, columns):
        method = getattr(self._source, "read_fact_columns", None)
        if method is None:
            return self.read_fact(version, stage)
        production = [self.column_map.get(column, column) for column in columns]
        # Read canonical columns too so a rename collision cannot be hidden by projection.
        return self._apply(method(version, stage, list(dict.fromkeys([*production, *columns]))))

    def read_dim(self, version: str, name: str) -> pd.DataFrame | None:
        frame = self._source.read_dim(version, name)
        return self._apply(frame) if frame is not None else None


def mapped(source: VersionSource, column_map: dict[str, str] | None) -> VersionSource:
    """Wrap a source only when a non-empty mapping is configured."""
    if not column_map:
        return source
    return MappedSource(source, column_map)


class CachedSource:
    """Per-assessment immutable snapshot cache and explicit calendar normalization."""
    def __init__(self, source, config):
        self.source, self.config = source, config
        self.facts = {}
        self.dims = {}
        self.read_count = 0
        self.weekday = None

    def __getattr__(self, name):
        return getattr(self.source, name)

    def read_fact(self, version, stage):
        key = (version, stage)
        if key not in self.facts:
            projected = getattr(self.source, "read_fact_columns", None)
            columns = list(dict.fromkeys((self.config.week_column, *self.config.required_columns,
                *self.config.entity_columns, *self.config.entity_key_columns, *self.config.report_grain,
                *self.config.metric_columns, *self.config.temporal_entity_columns,
                *(c for hierarchy in self.config.hierarchy_columns for c in hierarchy),
                *(c for pair in self.config.count_entity_map for c in pair),
                *(c for _, keys in self.config.stage_keys for c in keys))))
            frame = (projected(version, stage, columns) if projected else self.source.read_fact(version, stage)).copy()
            week = self.config.week_column
            if week in frame and self.config.calendar == "weekly_date":
                dates = pd.to_datetime(frame[week], errors="raise", utc=True)
                if (dates != dates.dt.normalize()).any() or dates.dt.dayofweek.nunique() != 1:
                    raise ValueError("weekly dates must be midnight on a consistent weekday")
                weekday = int(dates.dt.dayofweek.iloc[0]) if len(dates) else None
                if self.weekday is not None and weekday != self.weekday:
                    raise ValueError("weekly dates must use the same weekday across snapshots")
                self.weekday = weekday
                frame[week] = ((dates - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(days=7)).astype(int)
            elif week in frame and self.config.calendar == "mapped":
                frame[week] = frame[week].astype(str).map(dict(self.config.period_map))
            self.facts[key] = frame
            self.read_count += 1
        return self.facts[key].copy()

    def read_dim(self, version, name):
        key = (version, name)
        if key not in self.dims:
            self.dims[key] = self.source.read_dim(version, name)
        value = self.dims[key]
        return value.copy() if value is not None else None


class ParquetManifestSource:
    """Explicit local snapshot versions; no directory-order or cross-table inference.

    Manifest v1: source_id plus ordered snapshots, each with version, observed_at,
    stages {name: relative_parquet_path}, and dimensions {name: relative_path}.
    """
    def __init__(self, manifest):
        import json
        from pathlib import Path
        self.path = Path(manifest).resolve()
        payload = json.loads(self.path.read_text())
        if payload.get("schema_version") != 1 or not payload.get("source_id"):
            raise ValueError("Parquet manifest requires schema_version=1 and source_id")
        self.source_id = payload["source_id"]
        self.snapshots = payload["snapshots"]
        versions = [str(item["version"]) for item in self.snapshots]
        if not versions or len(set(versions)) != len(versions):
            raise ValueError("snapshot versions must be nonempty and unique")
        self.by_version = dict(zip(versions, self.snapshots, strict=True))
        from .store import observation_time
        times = [observation_time(item["observed_at"]) for item in self.snapshots]
        if times != sorted(times):
            raise ValueError("manifest observations must be chronologically ordered")

    def _path(self, relative):
        path = (self.path.parent / relative).resolve()
        if not path.is_relative_to(self.path.parent) or path.suffix != ".parquet":
            raise ValueError("snapshot files must be Parquet files within the manifest directory")
        return path

    def list_versions(self):
        return list(self.by_version)

    def available_stages(self, version):
        return [name for name, relative in self.by_version[version]["stages"].items()
                if self._path(relative).is_file()]

    def resolve_stage(self, version, preferred):
        available = self.available_stages(version)
        for stage in preferred:
            if stage in available:
                return stage
        raise FileNotFoundError("configured stage unavailable")

    def read_fact(self, version, stage):
        return pd.read_parquet(self._path(self.by_version[version]["stages"][stage]))

    def read_fact_columns(self, version, stage, columns):
        import pyarrow.parquet as pq
        path = self._path(self.by_version[version]["stages"][stage])
        names = pq.read_schema(path).names
        return pd.read_parquet(path, columns=[name for name in columns if name in names])

    def read_dim(self, version, name):
        path = self.by_version[version].get("dimensions", {}).get(name)
        if not path or not self._path(path).is_file():
            return None
        return pd.read_parquet(self._path(path))

    def snapshot_metadata(self, version):
        from datetime import datetime

        from .store import observation_time
        observed = observation_time(self.by_version[version]["observed_at"])
        return {"table_id": self.source_id, "version": version,
                "committed_at_ms": datetime.fromisoformat(observed).timestamp() * 1000,
                "manifest": self.by_version[version]}
