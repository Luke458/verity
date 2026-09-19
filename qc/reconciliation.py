"""Reconciliation and structural integrity checks.

Mass balance across grains (the report must sum back to the analysis table),
count consistency, and robust cross-metric ratio evidence. Invariant failures
are reconciliation failures; ratio outliers are evidence, not failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import DatasetConfig


@dataclass(frozen=True)
class ReconciliationCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ReconciliationResult:
    status: str
    checks: list[ReconciliationCheck] = field(default_factory=list)
    ratio_flags: list[dict] = field(default_factory=list)

    @property
    def failed(self) -> list[ReconciliationCheck]:
        return [check for check in self.checks if not check.passed]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
            "ratio_flags": list(self.ratio_flags),
        }


def _mass_balance(
    base: pd.DataFrame, report: pd.DataFrame, config: DatasetConfig
) -> list[ReconciliationCheck]:
    checks: list[ReconciliationCheck] = []
    for metric in config.metric_columns:
        if metric not in base.columns or metric not in report.columns:
            continue
        base_total = float(base[metric].sum())
        report_total = float(report[metric].sum())
        denominator = max(abs(base_total), 1e-9)
        relative = abs(report_total - base_total) / denominator
        checks.append(
            ReconciliationCheck(
                f"mass_balance:{metric}",
                relative <= config.reconciliation_tolerance,
                f"base={base_total:.6f} report={report_total:.6f} rel={relative:.2e}",
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
    group_keys = [week] + [
        column
        for column in config.report_grain
        if column in report.columns and column in base.columns
    ]
    for count, entity in config.count_entity_map:
        if count not in report.columns or entity not in base.columns:
            continue
        if len(group_keys) > 1 and all(key in base.columns for key in group_keys):
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
                ReconciliationCheck(
                    f"count_consistency:{count}",
                    mismatches == 0,
                    f"mismatched_groups={mismatches}",
                )
            )
        else:
            checks.append(
                ReconciliationCheck(
                    f"count_consistency:{count}",
                    True,
                    "skipped: report grain unavailable in base",
                )
            )
    return checks


def _hierarchy_checks(
    base: pd.DataFrame, config: DatasetConfig
) -> list[ReconciliationCheck]:
    """Parent/child consistency and aggregate-mixing detection.

    For each configured hierarchy the per-week grand total is compared with the
    total of its parts, and each part column is checked for marker values
    (``TOTAL``, ``ALL``, ...) that would mix aggregates into detail rows and
    double count. Unlike a parent-table comparison this needs only one frame.
    """
    checks: list[ReconciliationCheck] = []
    week = config.week_column
    if week not in base.columns:
        return checks
    markers = {marker.lower() for marker in config.hierarchy_total_markers}
    for columns in config.hierarchy_columns:
        present = [column for column in columns if column in base.columns]
        if not present:
            continue
        for column in present:
            values = base[column].dropna().astype(str)
            hits = values[values.str.lower().isin(markers)]
            checks.append(
                ReconciliationCheck(
                    f"hierarchy_markers:{column}",
                    hits.empty,
                    f"aggregate_marker_rows={len(hits)}",
                )
            )
        for metric in config.metric_columns:
            if metric not in base.columns:
                continue
            grand = base.groupby(week)[metric].sum()
            parts = (
                base.groupby([week] + present, dropna=False)[metric]
                .sum()
                .groupby(level=0)
                .sum()
            )
            aligned = pd.concat(
                [grand.rename("grand"), parts.rename("parts")], axis=1
            ).fillna(0.0)
            residual = float(
                (aligned["parts"] - aligned["grand"]).abs().sum()
            )
            denominator = float(aligned["grand"].abs().sum()) or 1.0
            relative = residual / denominator
            checks.append(
                ReconciliationCheck(
                    f"hierarchy:{'+'.join(present)}:{metric}",
                    relative <= config.hierarchy_tolerance,
                    f"residual={residual:.6f} rel={relative:.2e}",
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
    if mad <= 0:
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
    checks: list[ReconciliationCheck] = []
    if base_current is None or report_current is None:
        return ReconciliationResult(
            status="PASS",
            checks=[
                ReconciliationCheck(
                    "mass_balance",
                    True,
                    "report stage unavailable; skipped",
                )
            ],
        )
    checks.extend(_mass_balance(base_current, report_current, config))
    checks.extend(_count_consistency(base_current, report_current, config))
    checks.extend(_hierarchy_checks(base_current, config))
    status = "RECONCILIATION_FAILURE" if any(not c.passed for c in checks) else "PASS"
    return ReconciliationResult(
        status=status,
        checks=checks,
        ratio_flags=_ratio_flags(base_current, config),
    )
