"""Scoped ratio expectations (ported from Verity's approval model).

An expectation explains exactly the alert it names - one metric, one week, one
ratio band, one validity window - and clears nothing else. Unapproved,
expired, wrong-dataset or out-of-band expectations are ignored with an audit
reason. This is the difference between "this ratio movement was approved
beforehand" and "an AI model decided it was fine".
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RatioExpectation:
    expectation_id: str
    dataset: str
    metric: str
    min_ratio: float | None = None
    max_ratio: float | None = None
    week: int | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    approved_by: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.min_ratio is None and self.max_ratio is None:
            raise ValueError(f"{self.expectation_id}: a ratio band is required")
        if (
            self.min_ratio is not None
            and self.max_ratio is not None
            and self.min_ratio > self.max_ratio
        ):
            raise ValueError(f"{self.expectation_id}: min_ratio exceeds max_ratio")
        if (
            self.effective_from
            and self.effective_to
            and self.effective_from > self.effective_to
        ):
            raise ValueError(f"{self.expectation_id}: invalid validity window")

    @property
    def approved(self) -> bool:
        return bool(self.approved_by and self.approved_by.strip())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RatioExpectation:
        return cls(
            expectation_id=str(data["expectation_id"]),
            dataset=str(data["dataset"]),
            metric=str(data["metric"]),
            min_ratio=(
                float(data["min_ratio"]) if data.get("min_ratio") is not None else None
            ),
            max_ratio=(
                float(data["max_ratio"]) if data.get("max_ratio") is not None else None
            ),
            week=int(data["week"]) if data.get("week") is not None else None,
            effective_from=data.get("effective_from"),
            effective_to=data.get("effective_to"),
            approved_by=data.get("approved_by"),
            note=str(data.get("note", "")),
        )


def load_expectations(path: str | Path) -> list[RatioExpectation]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("expectations", [])
    return [RatioExpectation.from_dict(item) for item in data]


def save_expectations(
    expectations: Sequence[RatioExpectation], path: str | Path
) -> None:
    Path(path).write_text(
        json.dumps(
            {"expectations": [item.to_dict() for item in expectations]},
            indent=2,
            sort_keys=True,
        )
    )


def _in_scope(expectation: RatioExpectation, flag: dict[str, Any], context: dict) -> tuple[bool, str]:
    if not expectation.approved:
        return False, "unapproved"
    if expectation.dataset != str(context.get("dataset", "")):
        return False, "wrong dataset"
    if expectation.metric != str(flag.get("metric", "")):
        return False, "wrong metric"
    if expectation.week is not None and expectation.week != int(flag.get("week", -1)):
        return False, "wrong week"
    as_of = context.get("as_of")
    if as_of:
        if expectation.effective_from and str(as_of) < expectation.effective_from:
            return False, "not yet effective"
        if expectation.effective_to and str(as_of) > expectation.effective_to:
            return False, "expired"
    ratio = flag.get("ratio")
    if ratio is not None and expectation.min_ratio is not None and ratio < expectation.min_ratio:
        return False, "outside approved band"
    if ratio is not None and expectation.max_ratio is not None and ratio > expectation.max_ratio:
        return False, "outside approved band"
    return True, "matched"


def apply_expectations(
    flags: Sequence[dict[str, Any]],
    expectations: Sequence[RatioExpectation],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explain only the flags that an approved expectation matches.

    Each flag is annotated with the matching expectation id or stays
    unexplained; nothing else is cleared.
    """
    context = context or {}
    explained: list[dict[str, Any]] = []
    unexplained: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for flag in flags:
        matched: RatioExpectation | None = None
        reasons: list[str] = []
        for expectation in expectations:
            in_scope, reason = _in_scope(expectation, flag, context)
            if in_scope:
                matched = expectation
                break
            reasons.append(f"{expectation.expectation_id}: {reason}")
        if matched is not None:
            explained.append(
                {
                    **flag,
                    "explained_by": matched.expectation_id,
                    "approved_by": matched.approved_by,
                    "note": matched.note,
                }
            )
        else:
            unexplained.append(dict(flag))
            if reasons:
                audit.append({"week": flag.get("week"), "reasons": reasons})
    return {
        "expected": explained,
        "unexpected": unexplained,
        "audit": audit,
        "method": "scoped_expectations",
    }
