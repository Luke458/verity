"""Independent reference controls.

A pinned control snapshot answers one question: do this version's totals match
an independently produced reference? Status is explicit - ``MATCH``,
``MISMATCH`` or ``INCOMPLETE`` - and a mismatch is evidence, never a silent
clear. Missing metrics or an empty frame are ``INCOMPLETE``, not zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import DatasetConfig


@dataclass(frozen=True)
class ReferenceSpec:
    reference_id: str
    dataset: str
    metrics: tuple[str, ...]
    tolerance: float = 1e-6
    note: str = ""
    grain: tuple[str, ...] = ()

    def __post_init__(self):
        import math
        if not self.reference_id or not self.dataset or not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("reference requires an id, dataset, and unique metrics")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("reference tolerance must be finite and nonnegative")
        if len(set(self.grain)) != len(self.grain):
            raise ValueError("reference grain must be unique")

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
            grain=tuple(data.get("grain", ())),
        )


def compare_reference(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    spec: ReferenceSpec,
    config: DatasetConfig | None = None,
) -> dict[str, Any]:
    explicit_config = config is not None
    config = config or DatasetConfig(name=spec.dataset, report_grain=())
    keys = list(dict.fromkeys((config.week_column, *(spec.grain or config.report_grain))))
    checks: list[dict[str, Any]] = []
    incomplete = (current.empty or reference.empty or not spec.metrics
                  or (explicit_config and spec.dataset != config.name)
                  or any(key not in current.columns or key not in reference.columns for key in keys))
    invalid = incomplete
    mismatch = False

    for metric in spec.metrics:
        if (invalid or metric not in current.columns or metric not in reference.columns
            or not pd.api.types.is_numeric_dtype(current[metric])
            or not pd.api.types.is_numeric_dtype(reference[metric])
            or not np.isfinite(current[metric].to_numpy(dtype=float)).all()
            or not np.isfinite(reference[metric].to_numpy(dtype=float)).all()):
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
        if keys:
            left = current.groupby(keys, dropna=False)[metric].sum(min_count=1)
            right = reference.groupby(keys, dropna=False)[metric].sum(min_count=1)
            aligned = pd.concat([left.rename("current"), right.rename("reference")], axis=1)
            if aligned.isna().any().any():
                incomplete = True
                checks.append({"metric": metric, "status": "INCOMPLETE", "detail": "reference grain coverage differs"})
                continue
            relative = float(((aligned.current - aligned.reference).abs() /
                              aligned.reference.abs().clip(lower=1e-9)).max())
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

    if mismatch:
        status = "MISMATCH"
    elif incomplete:
        status = "INCOMPLETE"
    else:
        status = "MATCH"
    return {
        "reference_id": spec.reference_id,
        "dataset": spec.dataset,
        "status": status,
        "checks": checks,
        "method": "independent_reference_by_grain" if keys else "independent_reference_totals",
        "grain": keys,
        "limitations": [
            "Reference totals are an independent control, not a root cause.",

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
