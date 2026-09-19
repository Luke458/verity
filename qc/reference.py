"""Independent reference controls (ported from Verity's spine).

A pinned control snapshot answers one question: do this version's totals match
an independently produced reference? Status is explicit - ``MATCH``,
``MISMATCH`` or ``INCOMPLETE`` - and a mismatch is evidence, never a silent
clear. Missing metrics or an empty frame are ``INCOMPLETE``, not zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .config import DatasetConfig


@dataclass(frozen=True)
class ReferenceSpec:
    reference_id: str
    dataset: str
    metrics: tuple[str, ...]
    tolerance: float = 1e-6
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceSpec:
        return cls(
            reference_id=str(data["reference_id"]),
            dataset=str(data["dataset"]),
            metrics=tuple(str(metric) for metric in data["metrics"]),
            tolerance=float(data.get("tolerance", 1e-6)),
            note=str(data.get("note", "")),
        )


def compare_reference(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    spec: ReferenceSpec,
    config: DatasetConfig | None = None,
) -> dict[str, Any]:
    config = config or DatasetConfig()
    checks: list[dict[str, Any]] = []
    incomplete = False
    mismatch = False

    for metric in spec.metrics:
        if metric not in current.columns or metric not in reference.columns:
            checks.append(
                {
                    "metric": metric,
                    "status": "INCOMPLETE",
                    "detail": "metric missing from current or reference",
                }
            )
            incomplete = True
            continue
        current_total = float(current[metric].sum())
        reference_total = float(reference[metric].sum())
        denominator = max(abs(reference_total), 1e-9)
        relative = abs(current_total - reference_total) / denominator
        matched = relative <= spec.tolerance
        mismatch = mismatch or not matched
        checks.append(
            {
                "metric": metric,
                "status": "MATCH" if matched else "MISMATCH",
                "current_total": current_total,
                "reference_total": reference_total,
                "relative_delta": relative,
            }
        )

    if incomplete:
        status = "INCOMPLETE"
    elif mismatch:
        status = "MISMATCH"
    else:
        status = "MATCH"
    return {
        "reference_id": spec.reference_id,
        "dataset": spec.dataset,
        "status": status,
        "checks": checks,
        "method": "independent_reference_totals",
        "limitations": [
            "Reference totals are an independent control, not a root cause.",
            "Grain-level reference reconciliation is not implemented.",
        ],
    }


def reference_mismatches(report: dict[str, Any] | None) -> list[str]:
    if report is None or report.get("status") != "MISMATCH":
        return []
    return [
        check["metric"]
        for check in report.get("checks", [])
        if check.get("status") == "MISMATCH"
    ]
