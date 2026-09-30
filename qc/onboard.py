"""Data onboarding: profile a Delta table and assess readiness.

The first real refresh pair will not match the synthetic schema. This module
inspects a table, proposes a ``DatasetConfig``, and reports concrete blockers
(missing week column, non-numeric metrics, duplicate keys, gaps, contract
failures, version shape) so onboarding is a checklist rather than a debugging
session. It reads schema, a bounded sample, and optionally an exact row count;
it never writes to the table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

WEEK_NAMES = (
    "week",
    "week_id",
    "week_no",
    "week_number",
    "week_ending",
    "week_end",
    "period",
    "fiscal_week",
    "wk",
)
IDENTIFIER_SUFFIX = "_id"
MAX_WEEK_VALUE = 100000


def _is_numeric(dtype: str) -> bool:
    dtype = dtype.lower()
    return any(
        token in dtype
        for token in ("int", "long", "float", "double", "decimal", "short", "byte")
    )


def _base_name(column: str) -> str:
    return column.strip().lower()


def _infer_week_column(
    columns: Sequence[dict[str, str]],
    alias_map: dict[str, str] | None = None,
) -> str | None:
    alias_map = alias_map or {}
    by_name = {_base_name(column["name"]): column["name"] for column in columns}
    for production, canonical in alias_map.items():
        if canonical == "week" and production in by_name:
            return by_name[production]
    for candidate in WEEK_NAMES:
        if candidate in by_name:
            return by_name[candidate]
    numeric_columns = [
        column for column in columns if _is_numeric(column["type"])
    ]
    if len(numeric_columns) == 1:
        return numeric_columns[0]["name"]
    return None


def _split_columns(
    columns: Sequence[dict[str, str]],
    week_column: str | None,
    alias_map: dict[str, str] | None = None,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return (entity_columns, metric_columns, canonical -> production).

    Aliased columns are reported by canonical name; unaliased columns keep
    their production names (which are already canonical by assumption).
    """
    alias_map = alias_map or {}
    entity_columns: list[str] = []
    metric_columns: list[str] = []
    production_by_canonical: dict[str, str] = {}
    for column in columns:
        name = column["name"]
        if name == week_column:
            if alias_map.get(_base_name(name)) == "week":
                production_by_canonical["week"] = name
            continue
        canonical = alias_map.get(_base_name(name))
        if canonical:
            production_by_canonical[canonical] = name
            if _base_name(canonical).endswith(IDENTIFIER_SUFFIX):
                entity_columns.append(canonical)
            else:
                metric_columns.append(canonical)
            continue
        if _base_name(name).endswith(IDENTIFIER_SUFFIX):
            entity_columns.append(name)
        elif _is_numeric(column["type"]):
            metric_columns.append(name)
    return entity_columns, metric_columns, production_by_canonical


def _finding(severity: str, code: str, message: str, detail: Any = None) -> dict:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "detail": detail,
    }


def profile_table(
    uri: str,
    storage_options: dict[str, str] | None = None,
    sample_rows: int = 50000,
    count_rows: bool = False,
    version: int | None = None,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    from .delta import _delta_table

    alias_map = {
        str(production).strip().lower(): str(canonical)
        for production, canonical in (aliases or {}).items()
    }
    table = _delta_table(uri, storage_options, version)
    columns = [
        {"name": field.name, "type": str(field.type)}
        for field in table.schema().fields
    ]
    dataset = table.to_pyarrow_dataset()
    sample = dataset.head(sample_rows).to_pandas() if sample_rows > 0 else None
    if sample is None or len(sample) == 0:
        return {
            "uri": uri,
            "current_version": int(table.version()),
            "sampled_version": int(version) if version is not None else None,
            "columns": columns,
            "rows_sampled": 0,
            "count_rows": 0,
            "null_fraction": {},
            "duplicate_fraction": None,
            "week_column": None,
            "entity_columns": [],
            "metric_columns": [],
            "proposed": {},
            "findings": [
                _finding("BLOCKER", "EMPTY_SAMPLE", "table version has no rows")
            ],
        }

    week_production = _infer_week_column(columns, alias_map)
    week_column = (
        "week"
        if week_production is not None
        and alias_map.get(_base_name(week_production)) == "week"
        else week_production
    )
    entity_columns, metric_columns, production_by_canonical = _split_columns(
        columns, week_production, alias_map
    )
    entity_production = [
        production_by_canonical.get(column, column) for column in entity_columns
    ]
    metric_production = {
        production_by_canonical.get(column, column) for column in metric_columns
    }
    null_fraction = {
        column: float(sample[column].isna().mean())
        for column in sample.columns
    }

    duplicate_fraction = None
    if week_production is not None and entity_production:
        keys = [week_production] + entity_production
        duplicate_fraction = float(sample.duplicated(subset=keys).mean())

    findings: list[dict[str, Any]] = []
    if week_production is None:
        findings.append(
            _finding(
                "BLOCKER",
                "MISSING_WEEK_COLUMN",
                "no week-like column found; pass --alias production=week, "
                "for example --alias wk=week",
                {"candidates": [column["name"] for column in columns]},
            )
        )
    else:
        week_dtype = next(
            column["type"]
            for column in columns
            if column["name"] == week_production
        )
        if not _is_numeric(week_dtype):
            findings.append(
                _finding(
                    "BLOCKER",
                    "WEEK_NOT_INTEGER",
                    f"week column {week_production!r} is {week_dtype}; the engine "
                    "needs an integer week (derive one from the date)",
                )
            )
        else:
            weeks = sample[week_production].dropna()
            if len(weeks):
                unique = sorted(int(value) for value in weeks.unique())
                contiguous = all(
                    right - left == 1
                    for left, right in zip(unique, unique[1:], strict=False)
                )
                if not contiguous:
                    findings.append(
                        _finding(
                            "BLOCKER",
                            "WEEK_NOT_CONTIGUOUS",
                            "sampled weeks are not contiguous",
                            {"min": unique[0], "max": unique[-1]},
                        )
                    )
                if unique and (unique[0] < 0 or unique[-1] > MAX_WEEK_VALUE):
                    findings.append(
                        _finding(
                            "WARNING",
                            "WEEK_RANGE_UNUSUAL",
                            "week values look like dates or mis-scaled integers",
                            {"min": unique[0], "max": unique[-1]},
                        )
                    )
    if not metric_columns:
        findings.append(
            _finding(
                "BLOCKER",
                "MISSING_METRIC_COLUMNS",
                "no numeric metric columns found",
            )
        )
    if not entity_columns:
        findings.append(
            _finding(
                "WARNING",
                "MISSING_ENTITY_COLUMNS",
                "no *_id columns found; lifecycle and attribution need entity keys",
            )
        )
    if duplicate_fraction is not None and duplicate_fraction > 0:
        findings.append(
            _finding(
                "BLOCKER",
                "DUPLICATE_KEYS",
                "sampled week + entity keys contain duplicates",
                {"fraction": duplicate_fraction},
            )
        )
    for column, fraction in null_fraction.items():
        if column in metric_production and fraction > 0.01:
            findings.append(
                _finding(
                    "WARNING",
                    "NULL_METRICS",
                    f"{column!r} is {fraction:.1%} null in the sample",
                )
            )

    proposal = propose_config(
        week_column=week_column,
        entity_columns=entity_columns,
        metric_columns=metric_columns,
        columns=columns,
        uri=uri,
        production_by_canonical=production_by_canonical,
    )
    return {
        "uri": uri,
        "current_version": int(table.version()),
        "sampled_version": int(version) if version is not None else None,
        "columns": columns,
        "rows_sampled": int(len(sample)),
        "count_rows": int(dataset.count_rows()) if count_rows else None,
        "null_fraction": null_fraction,
        "duplicate_fraction": duplicate_fraction,
        "week_column": week_column,
        "entity_columns": entity_columns,
        "metric_columns": metric_columns,
        "proposed": proposal,
        "findings": findings,
    }


def propose_config(
    week_column: str | None,
    entity_columns: Sequence[str],
    metric_columns: Sequence[str],
    columns: Sequence[dict[str, str]],
    uri: str,
    production_by_canonical: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    canonical_names = set(entity_columns) | set(metric_columns)
    if week_column:
        canonical_names.add(week_column)
    primary_metric = (
        "dollar" if "dollar" in metric_columns else (metric_columns[0] if metric_columns else None)
    )
    key_columns = tuple(
        column for column in ("store_id", "product_id") if column in entity_columns
    )
    if not key_columns and entity_columns:
        key_columns = (entity_columns[0],)
    report_grain = tuple(
        column for column in ("banner_id", "state_id") if column in entity_columns
    )
    if not report_grain:
        report_grain = tuple(
            column for column in entity_columns if column not in key_columns
        )[:2]
    count_entity_map = tuple(
        (count, entity)
        for count, entity in (
            ("store_count", "store_id"),
            ("product_count", "product_id"),
        )
        if entity in entity_columns
    )
    required = tuple(
        column
        for column in ("week", primary_metric, "units")
        if column is not None and column in canonical_names
    )
    payload: dict[str, Any] = {
        "name": Path(uri.rstrip("/")).name or "dataset",
        "week_column": week_column,
        "metric_columns": list(metric_columns),
        "primary_metric": primary_metric,
        "entity_columns": list(entity_columns),
        "entity_key_columns": list(key_columns),
        "report_grain": list(report_grain),
        "count_entity_map": [list(pair) for pair in count_entity_map],
        "required_columns": list(required),
    }
    if production_by_canonical:
        column_map = {
            canonical: production
            for canonical, production in production_by_canonical.items()
            if canonical != production
        }
        if column_map:
            payload["column_map"] = column_map
    return payload


def config_from_proposal(proposal: dict[str, Any]):
    from .config import DatasetConfig

    overrides: dict[str, Any] = {}
    for key, value in proposal.items():
        if value is None:
            continue
        if key == "column_map":
            overrides[key] = tuple(
                (str(canonical), str(production))
                for canonical, production in value.items()
            )
        elif key in ("metric_columns", "entity_columns", "entity_key_columns", "report_grain", "required_columns"):
            overrides[key] = tuple(value)
        elif key == "count_entity_map":
            overrides[key] = tuple(tuple(pair) for pair in value)
        else:
            overrides[key] = value
    return replace(DatasetConfig(), **overrides)


def assess_versions(
    uri: str,
    config,
    previous_version: int,
    current_version: int,
    storage_options: dict[str, str] | None = None,
    stage: str = "warehouse",
) -> dict[str, Any]:
    from .contracts import validate_contracts
    from .delta import DeltaSource
    from .source import mapped
    from .versions import build_version_pair

    findings: list[dict[str, Any]] = []
    source = mapped(
        DeltaSource(uri=uri, storage_options=storage_options, stage=stage),
        config.column_map_dict(),
    )
    try:
        previous = source.read_fact(str(previous_version), stage)
        current = source.read_fact(str(current_version), stage)
    except Exception as error:  # noqa: BLE001 - report the failure
        findings.append(
            _finding(
                "BLOCKER",
                "VERSION_READ_FAILED",
                f"could not read versions {previous_version}/{current_version}: "
                f"{type(error).__name__}: {error}",
            )
        )
        return {"findings": findings, "version_pair": None, "contracts": None}

    contracts = validate_contracts(current, previous, config)
    for check in contracts.failed:
        findings.append(
            _finding(
                "BLOCKER",
                "CONTRACT_FAILED",
                f"{check.name}: {check.detail}",
            )
        )

    pair_payload: dict[str, Any] | None = None
    try:
        pair = build_version_pair(
            previous,
            current,
            str(previous_version),
            str(current_version),
            stage,
            stage,
            config,
        )
        pair_payload = pair.to_dict()
        if pair.shape != "NORMAL":
            findings.append(
                _finding(
                    "WARNING",
                    f"PAIR_{pair.shape}",
                    f"version pair shape is {pair.shape}; review new periods "
                    f"{list(pair.new_periods)} and overlap "
                    f"{pair.overlap_start}..{pair.overlap_end}",
                )
            )
    except Exception as error:  # noqa: BLE001
        findings.append(
            _finding(
                "BLOCKER",
                "PAIR_BUILD_FAILED",
                f"{type(error).__name__}: {error}",
            )
        )

    return {
        "findings": findings,
        "version_pair": pair_payload,
        "contracts": contracts.to_dict(),
    }
