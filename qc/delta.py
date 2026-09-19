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

from dataclasses import dataclass, field
from typing import Any, Sequence

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
    except Exception:  # pragma: no cover - tables without history metadata
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
            except Exception:  # pragma: no cover
                versions = [int(table.version())]
            self._histories[uri] = versions or [int(table.version())]
        return self._histories[uri]

    def list_versions(self) -> list[str]:
        if self.stage_tables and self.stage not in self.stage_tables:
            primary = self._stage_names()[0]
        else:
            primary = self.stage
        return [str(version) for version in self._versions(primary)]

    def available_stages(self, version: str) -> list[str]:
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

    def read_fact(self, version: str, stage: str) -> pd.DataFrame:
        table = _delta_table(
            self._table_uri(stage), self.storage_options, int(version)
        )
        return table.to_pandas()

    def read_dim(self, version: str, name: str) -> pd.DataFrame | None:
        if not self.dim_tables or name not in self.dim_tables:
            return None
        uri = self.dim_tables[name]
        try:
            return _delta_table(uri, self.storage_options, int(version)).to_pandas()
        except Exception as error:
            latest = _delta_table(uri, self.storage_options, None).to_pandas()
            self.warnings.append(
                f"dimension {name!r} has no version {version}; used latest "
                f"({error.__class__.__name__})"
            )
            return latest
