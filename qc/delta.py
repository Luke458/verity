"""Delta table source via delta-rs (no Spark or cloud compute required).

`DeltaSource` reads Delta tables and their versions directly, so the engine can
assess datalake table versions locally or from any storage delta-rs supports
(local paths, S3, ADLS, GCS via ``storage_options``).

Single-table mode models one table as one pipeline stage (default
``warehouse``). Multi-table mode maps stage names and dimension names to their
own tables; version numbers are assumed to be commit-version aligned across
stage tables, which is the contract for a pipeline that commits stages
together. Dimension tables are slowly changing and often have fewer versions,
so a missing dimension version falls back to the latest and records a warning
rather than silently pretending the requested version existed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

DELTA_MISSING_MESSAGE = (
    "deltalake is required for Delta sources; install the [delta] extra"
)


def _delta_table(uri: str, storage_options: dict[str, str] | None, version: int | None):
    try:
        from deltalake import DeltaTable
    except ImportError as error:  # pragma: no cover - exercised without extra
        raise ImportError(DELTA_MISSING_MESSAGE) from error
    table = DeltaTable(uri, storage_options=storage_options or None)
    if version is not None:
        # delta-rs mutates in place and returns None.
        table.load_as_version(int(version))
    return table


def describe_delta_table(
    uri: str, storage_options: dict[str, str] | None = None
) -> dict[str, Any]:
    """Current version, commit history, schema and row count for assessment."""
    table = _delta_table(uri, storage_options, None)
    try:
        history = table.history()
    except Exception:  # noqa: BLE001  # pragma: no cover - tables without history metadata
        history = [{"version": table.version()}]
    return {
        "uri": uri,
        "current_version": int(table.version()),
        "history": sorted(
            [
                {
                    "version": int(entry.get("version", -1)),
                    "timestamp": entry.get("timestamp"),
                    "operation": entry.get("operation"),
                }
                for entry in history
            ],
            key=lambda entry: entry["version"],
        ),
        "schema": [
            {"name": field.name, "type": str(field.type)}
            for field in table.schema().fields
        ],
        "partition_columns": list(table.metadata().partition_columns),
        "rows_at_current": int(table.to_pandas().shape[0]),
    }


@dataclass
class DeltaSource:
    uri: str
    storage_options: dict[str, str] | None = None
    stage: str = "warehouse"
    stage_tables: dict[str, str] | None = None
    dim_tables: dict[str, str] | None = None
    version_map: dict[str, dict[str, int]] | None = None
    warnings: list[str] = field(default_factory=list, init=False)
    _histories: dict[str, list[int]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _table_uri(self, stage_name: str) -> str:
        if self.stage_tables:
            if stage_name not in self.stage_tables:
                raise KeyError(
                    f"stage {stage_name!r} is not configured; "
                    f"available: {sorted(self.stage_tables)}"
                )
            return self.stage_tables[stage_name]
        if stage_name != self.stage:
            raise KeyError(
                f"single-table source exposes only stage {self.stage!r}"
            )
        return self.uri

    def _stage_names(self) -> list[str]:
        if self.stage_tables:
            return list(self.stage_tables)
        return [self.stage]

    def _versions(self, stage_name: str) -> list[int]:
        uri = self._table_uri(stage_name)
        if uri not in self._histories:
            table = _delta_table(uri, self.storage_options, None)
            try:
                versions = sorted(
                    int(entry.get("version", -1)) for entry in table.history()
                )
                versions = [version for version in versions if version >= 0]
            except Exception:  # noqa: BLE001  # pragma: no cover
                versions = [int(table.version())]
            self._histories[uri] = versions or [int(table.version())]
        return self._histories[uri]

    def _primary_stage(self) -> str:
        if self.stage_tables and self.stage not in self.stage_tables:
            primary = self._stage_names()[0]
        else:
            primary = self.stage
        return primary

    def list_versions(self) -> list[str]:
        return [str(version) for version in self._versions(self._primary_stage())]

    def available_stages(self, version: str) -> list[str]:
        if self.stage_tables:
            if not self.version_map or version not in self.version_map:
                raise ValueError("multi-table snapshots require explicit version_map")
            primary = self._primary_stage()
            if self.version_map[version].get(primary) != int(version):
                raise ValueError("primary stage mapping must match the selected fact version")
            cutoff = self.snapshot_metadata(version)["committed_at_ms"]
            available = []
            for name in self._stage_names():
                selected = self.version_map[version].get(name)
                if selected is None or selected not in self._versions(name):
                    continue
                table = _delta_table(self._table_uri(name), self.storage_options, selected)
                commit = next((item.get("timestamp") for item in table.history() if int(item["version"]) == selected), None)
                if cutoff is None or commit is None or commit > cutoff:
                    self.warnings.append(f"stage {name!r} was not available at selected fact commit")
                    continue
                available.append(name)
            return available
        number = int(version)
        return [
            stage_name
            for stage_name in self._stage_names()
            if number in self._versions(stage_name)
        ]

    def resolve_stage(self, version: str, preferred: Sequence[str]) -> str:
        available = self.available_stages(version)
        for stage_name in preferred:
            if stage_name in available:
                return stage_name
        if not available:
            raise FileNotFoundError(
                f"version {version!r} is not present in any configured stage"
            )
        return available[-1]

    def snapshot_metadata(self, version: str) -> dict[str, Any]:
        table = _delta_table(self._table_uri(self._primary_stage()), self.storage_options, int(version))
        entry: dict[str, Any] = next((item for item in table.history() if int(item["version"]) == int(version)), {})
        mapping = (self.version_map or {}).get(version, {})
        inputs = {}
        for name, uri in {**(self.stage_tables or {self.stage: self.uri}),
                          **{f"dim:{key}": value for key, value in (self.dim_tables or {}).items()}}.items():
            selected = mapping.get(name) if self.stage_tables or name.startswith("dim:") else int(version)
            detail: dict[str, Any] = {"uri": uri, "version": selected, "available": False}
            if selected is not None:
                try:
                    selected_table = _delta_table(uri, self.storage_options, selected)
                    commit = next((item.get("timestamp") for item in selected_table.history() if int(item["version"]) == selected), None)
                    detail.update(table_id=selected_table.metadata().id, committed_at_ms=commit, available=True)
                except Exception as error:  # noqa: BLE001 - retain unavailable input identity
                    detail["error"] = type(error).__name__
            inputs[name] = detail
        return {"table_id": table.metadata().id, "version": version,
                "committed_at_ms": entry.get("timestamp"),
                "version_map": mapping, "inputs": inputs}

    def read_fact(self, version: str, stage: str) -> pd.DataFrame:
        table = _delta_table(
            self._table_uri(stage), self.storage_options,
            self.version_map[version][stage] if self.stage_tables and self.version_map else int(version)
        )
        return table.to_pandas()

    def read_fact_columns(self, version: str, stage: str, columns: list[str]) -> pd.DataFrame:
        selected = self.version_map[version][stage] if self.stage_tables and self.version_map else int(version)
        table = _delta_table(self._table_uri(stage), self.storage_options, selected)
        dataset = table.to_pyarrow_dataset()
        return dataset.to_table(columns=[c for c in columns if c in dataset.schema.names]).to_pandas()

    def read_dim(self, version: str, name: str) -> pd.DataFrame | None:
        if not self.dim_tables or name not in self.dim_tables:
            return None
        uri = self.dim_tables[name]
        selected = (self.version_map or {}).get(version, {}).get(f"dim:{name}")
        if selected is None:
            self.warnings.append(f"dimension {name!r} has no explicit mapping at {version}")
            return None
        try:
            table = _delta_table(uri, self.storage_options, selected)
            entry: dict[str, Any] = next((item for item in table.history() if int(item["version"]) == selected), {})
            cutoff = self.snapshot_metadata(version).get("committed_at_ms")
            if cutoff is None or entry.get("timestamp") is None or entry["timestamp"] > cutoff:
                self.warnings.append(f"dimension {name!r} was not available at the selected fact commit")
                return None
            return table.to_pandas()
        except Exception as error:  # noqa: BLE001 - unavailable historical input
            self.warnings.append(f"dimension {name!r} unavailable: {type(error).__name__}")
            return None
