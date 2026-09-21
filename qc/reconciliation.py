"""Reconciliation and structural integrity checks.

Mass balance compares the report table against the analysis table *per week
and per report key*, so compensating errors (one duplicated key offset by one
missing key) cannot pass. Aggregate-marker scans catch parent rows mixed into
detail rows. Checks report ``PASS``, ``FAIL`` or ``SKIPPED``; a run whose
checks were all skipped is ``NOT_EVALUATED``, never ``PASS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import DatasetConfig

PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class ReconciliationCheck:
    name: str
    status: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == PASS


@dataclass
class ReconciliationResult:
    status: str
    checks: list[ReconciliationCheck] = field(default_factory=list)
    ratio_flags: list[dict] = field(default_factory=list)

    @property
    def failed(self) -> list[ReconciliationCheck]:
        return [check for check in self.checks if check.status == FAIL]

    @property
    def skipped(self) -> list[ReconciliationCheck]:
        return [check for check in self.checks if check.status == SKIPPED]

    @property
    def evaluated(self) -> list[ReconciliationCheck]:
        return [check for check in self.checks if check.status != SKIPPED]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail}
                for c in self.checks
            ],
            "ratio_flags": list(self.ratio_flags),
        }


def _check(name: str, ok: bool, detail: str) -> ReconciliationCheck:
    return ReconciliationCheck(name, PASS if ok else FAIL, detail)


def _key_columns(base: pd.DataFrame, report: pd.DataFrame, config: DatasetConfig) -> list[str]:
    week = config.week_column
    keys = [week]
    for column in config.report_grain:
        if column in base.columns and column in report.columns:
            keys.append(column)
    return keys


def _mass_balance(
    base: pd.DataFrame, report: pd.DataFrame, config: DatasetConfig
) -> list[ReconciliationCheck]:
    """Per-key, per-week comparison of report against analysis totals."""
    if base is report:
        return [
            ReconciliationCheck(
                "mass_balance",
                SKIPPED,
                "report and analysis frames are identical; nothing to reconcile",
            )
        ]
    missing = [key for key in (config.week_column, *config.report_grain)
               if key not in base.columns or key not in report.columns]
    if missing:
        return [ReconciliationCheck("mass_balance", SKIPPED, f"missing grain: {missing}")]
    keys = _key_columns(base, report, config)
    checks: list[ReconciliationCheck] = []
    for metric in config.metric_columns:
        if metric not in base.columns and metric not in report.columns:
            continue
        if metric not in base.columns or metric not in report.columns:
            checks.append(ReconciliationCheck(f"mass_balance:{metric}", SKIPPED, "metric unavailable on one stage"))
            continue
        base_grouped = base.groupby(keys, dropna=False)[metric].sum()
        report_grouped = report.groupby(keys, dropna=False)[metric].sum()
        aligned = pd.concat(
            [base_grouped.rename("base"), report_grouped.rename("report")], axis=1
        )
        if config.absent_entity_policy == "missing" and aligned.isna().any().any():
            checks.append(ReconciliationCheck(f"mass_balance:{metric}", SKIPPED, "grain coverage incomplete"))
            continue
        aligned = aligned.fillna(0.0)
        residual = (aligned["report"] - aligned["base"]).abs()
        scale = aligned["base"].abs().where(aligned["base"].abs() > 1e-9, 1.0)
        relative = residual / scale
        worst = float(relative.max()) if len(relative) else 0.0
        worst_key = relative.idxmax() if len(relative) else None
        checks.append(
            _check(
                f"mass_balance:{metric}",
                worst <= config.reconciliation_tolerance,
                f"keys={len(aligned)} worst_rel={worst:.2e} at {worst_key}",
            )
        )
    if not checks:
        checks.append(
            ReconciliationCheck(
                "mass_balance", SKIPPED, "no metric column present in both frames"
            )
        )
    return checks


def _count_consistency(
    base: pd.DataFrame, report: pd.DataFrame, config: DatasetConfig
) -> list[ReconciliationCheck]:
    """Compare report counts against the base table group by group.

    Summing per-group distinct counts would double count entities that appear
    in several groups, so the comparison is done on the report grain.
    """
    checks: list[ReconciliationCheck] = []
    week = config.week_column
    group_keys = [week, *config.report_grain]
    for count, entity in config.count_entity_map:
        if count not in report.columns or entity not in base.columns:
            continue
        if all(key in base.columns and key in report.columns for key in group_keys):
            expected = (
                base.groupby(group_keys, dropna=False)[entity]
                .nunique()
                .reset_index(name="expected")
            )
            merged = report[group_keys + [count]].merge(
                expected, on=group_keys, how="left"
            )
            mismatches = int((merged[count] != merged["expected"]).sum())
            checks.append(
                _check(
                    f"count_consistency:{count}",
                    mismatches == 0,
                    f"mismatched_groups={mismatches}",
                )
            )
        else:
            checks.append(
                ReconciliationCheck(
                    f"count_consistency:{count}",
                    SKIPPED,
                    "report grain unavailable in base",
                )
            )
    return checks


def _aggregate_marker_checks(
    base: pd.DataFrame, config: DatasetConfig
) -> list[ReconciliationCheck]:
    """Flag aggregate marker values (``TOTAL``, ``ALL``, ...) in detail rows.

    Parent/child reconciliation requires two frames; within one frame the only
    honest structural check is that aggregate markers are not mixed into
    detail rows.
    """
    checks: list[ReconciliationCheck] = []
    markers = {marker.lower() for marker in config.hierarchy_total_markers}
    for columns in config.hierarchy_columns:
        present = [column for column in columns if column in base.columns]
        if not present:
            continue
        for column in present:
            values = base[column].dropna().astype(str)
            hits = values[values.str.lower().isin(markers)]
            checks.append(
                _check(
                    f"aggregate_markers:{column}",
                    hits.empty,
                    f"aggregate_marker_rows={len(hits)}",
                )
            )
    return checks


def _ratio_flags(base: pd.DataFrame, config: DatasetConfig) -> list[dict]:
    week = config.week_column
    if "dollar" not in base.columns or "units" not in base.columns:
        return []
    grouped = base.groupby(week)[["dollar", "units"]].sum()
    grouped = grouped[grouped["units"] > 0]
    if len(grouped) < 5:
        return []
    ratios = grouped["dollar"] / grouped["units"]
    median = float(ratios.median())
    mad = float((ratios - median).abs().median())
    # Floating-point dust around a constant ratio is not a price outlier.
    if mad <= 1e-9 * max(abs(median), 1.0):
        return []
    flags = []
    for week_id, ratio in ratios.items():
        robust_z = 0.6745 * (float(ratio) - median) / mad
        if abs(robust_z) > config.price_outlier_k:
            flags.append(
                {
                    "week": int(week_id),
                    "dollar_per_unit": float(ratio),
                    "median": median,
                    "robust_z": robust_z,
                }
            )
    return flags


def run_reconciliation(
    base_current: pd.DataFrame | None,
    report_current: pd.DataFrame | None,
    config: DatasetConfig,
) -> ReconciliationResult:
    if base_current is None or report_current is None:
        return ReconciliationResult(
            status="NOT_EVALUATED",
            checks=[
                ReconciliationCheck(
                    "mass_balance",
                    SKIPPED,
                    "report or analysis frame unavailable",
                )
            ],
        )
    checks = _mass_balance(base_current, report_current, config)
    checks.extend(_count_consistency(base_current, report_current, config))
    checks.extend(_aggregate_marker_checks(base_current, config))
    if any(check.status == FAIL for check in checks):
        status = "RECONCILIATION_FAILURE"
    elif any(check.status == PASS for check in checks):
        status = "PASS"
    else:
        status = "NOT_EVALUATED"
    return ReconciliationResult(
        status=status,
        checks=checks,
        ratio_flags=_ratio_flags(base_current, config),
    )
