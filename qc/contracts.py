"""Data contract checks.

Contract failures are not ordinary anomalies: a missing required column, a
non-numeric measure, a broken week progression, or a null/duplicate explosion
means the table is not fit for revision QC and the run stops with
``DATA_CONTRACT_FAILURE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DatasetConfig


@dataclass(frozen=True)
class ContractCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ContractResult:
    status: str
    checks: list[ContractCheck] = field(default_factory=list)

    @property
    def failed(self) -> list[ContractCheck]:
        return [check for check in self.checks if not check.passed]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }


def _null_fraction(frame: pd.DataFrame, columns: tuple[str, ...]) -> float:
    present = [column for column in columns if column in frame.columns]
    if not present or len(frame) == 0:
        return 0.0
    total = len(frame) * len(present)
    nulls = int(frame[present].isna().sum().sum())
    return nulls / total


def _duplicate_fraction(frame: pd.DataFrame, keys: list[str]) -> float:
    if len(frame) == 0:
        return 0.0
    if any(key not in frame.columns for key in keys):
        return 0.0
    if not keys:
        return 0.0
    duplicates = int(frame.duplicated(subset=keys).sum())
    return duplicates / len(frame)


def _week_progression_ok(frame: pd.DataFrame, week_column: str) -> tuple[bool, str]:
    if week_column not in frame.columns or len(frame) == 0:
        return False, "week column missing or empty"
    weeks = frame[week_column]
    if weeks.isna().any():
        return False, "null weeks present"
    if not pd.api.types.is_numeric_dtype(weeks):
        return False, "sequential weeks must be numeric integers"
    values = weeks.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or not (values == np.floor(values)).all():
        return False, "weeks must be finite integers"
    unique = sorted(int(value) for value in weeks.unique())
    contiguous = all(
        right - left == 1 for left, right in zip(unique, unique[1:])
    )
    if not contiguous:
        return False, f"week gaps in {unique[:3]}...{unique[-3:]}"
    return True, f"weeks {unique[0]}..{unique[-1]}"


def validate_contracts(
    current: pd.DataFrame,
    previous: pd.DataFrame | None,
    config: DatasetConfig,
    stage: str | None = None,
) -> ContractResult:
    checks: list[ContractCheck] = []
    if not current.columns.is_unique:
        return ContractResult("DATA_CONTRACT_FAILURE", [ContractCheck("unique_columns", False)])
    grain = dict(config.stage_keys).get(stage or "", config.report_grain if stage == config.contract_stage else config.entity_key_columns)
    keys = [config.week_column, *grain]
    missing_keys = [key for key in keys if key not in current.columns]
    checks.append(ContractCheck("business_keys", not missing_keys, str(missing_keys)))
    if not missing_keys:
        checks.append(ContractCheck("key_nulls", not current[keys].isna().any().any()))
        checks.append(ContractCheck("duplicate_keys", not current.duplicated(keys).any()))
    checks.append(ContractCheck("unique_columns", current.columns.is_unique))

    missing = [column for column in config.required_columns if column not in current.columns]
    checks.append(ContractCheck("required_columns", not missing, ", ".join(missing)))

    metrics = [column for column in config.metric_columns if column in current.columns]
    non_numeric = [
        column
        for column in metrics
        if not pd.api.types.is_numeric_dtype(current[column])
    ]
    checks.append(ContractCheck("metric_dtypes", not non_numeric, ", ".join(non_numeric)))

    nonfinite = [column for column in metrics if column not in non_numeric
                 and not np.isfinite(current[column].to_numpy(dtype=float, na_value=np.nan)).all()]
    checks.append(ContractCheck("finite_measures", not nonfinite, ", ".join(nonfinite)))

    progression_ok, progression_detail = _week_progression_ok(current, config.week_column)
    checks.append(ContractCheck("week_progression", progression_ok, progression_detail))

    latest_ok = (
        config.week_column in current.columns
        and len(current) > 0
        and current[config.week_column].notna().all()
    )
    checks.append(ContractCheck("latest_week_present", latest_ok))

    null_fraction = _null_fraction(current, tuple(metrics))
    checks.append(
        ContractCheck(
            "null_fraction",
            null_fraction <= config.null_fraction_limit,
            f"{null_fraction:.4f} (limit {config.null_fraction_limit})",
        )
    )

    duplicate_fraction = _duplicate_fraction(current, keys)
    checks.append(
        ContractCheck(
            "duplicate_fraction",
            duplicate_fraction <= config.duplicate_fraction_limit,
            f"{duplicate_fraction:.4f} (limit {config.duplicate_fraction_limit})",
        )
    )

    if previous is not None:
        prior = validate_contracts(previous, None, config, stage)
        checks.extend(ContractCheck(f"previous:{c.name}", c.passed, c.detail) for c in prior.checks)
        checks.append(ContractCheck("compatible_schema", set(current.columns) == set(previous.columns)))
        compatible_keys = all(pd.api.types.is_numeric_dtype(current[key]) == pd.api.types.is_numeric_dtype(previous[key])
                              for key in keys if key in current and key in previous)
        checks.append(ContractCheck("compatible_key_types", compatible_keys))

    status = (
        "DATA_CONTRACT_FAILURE"
        if any(not check.passed for check in checks)
        else "PASS"
    )
    return ContractResult(status=status, checks=checks)
