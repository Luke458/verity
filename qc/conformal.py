"""Finite-sample conformal intervals (architecture section 28).

The interval uses the ``ceil((n + 1) * (1 - alpha))``-th smallest absolute
residual, which gives finite-sample marginal coverage when residuals are
exchangeable with future errors. When that rank exceeds ``n`` the interval is
reported as ``INSUFFICIENT_CALIBRATION`` instead of being widened by
assumption. Temporal drift breaks the exchangeability assumption and must be
re-checked with prequential (across-load) calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ConformalInterval:
    alpha: float
    lower: float | None
    upper: float | None
    n: int
    rank: int | None
    status: str  # OK | INSUFFICIENT_CALIBRATION
    detail: str = ""

    @property
    def width(self) -> float | None:
        if self.lower is None or self.upper is None:
            return None
        return self.upper - self.lower

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "lower": self.lower,
            "upper": self.upper,
            "n": self.n,
            "rank": self.rank,
            "status": self.status,
            "detail": self.detail,
            "width": self.width,
        }


def conformal_interval(
    residuals: Sequence[float],
    center: float,
    alpha: float = 0.05,
) -> ConformalInterval:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    absolute = sorted(abs(float(value)) for value in residuals)
    n = len(absolute)
    if n == 0:
        return ConformalInterval(
            alpha, None, None, n, None, "INSUFFICIENT_CALIBRATION", "no residuals"
        )
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        required = math.ceil(1.0 / alpha) - 1
        return ConformalInterval(
            alpha,
            None,
            None,
            n,
            None,
            "INSUFFICIENT_CALIBRATION",
            f"need at least {required} residuals for alpha={alpha}",
        )
    radius = absolute[rank - 1]
    return ConformalInterval(
        alpha, center - radius, center + radius, n, rank, "OK"
    )


def leave_one_out_coverage(
    residuals: Sequence[float], alpha: float = 0.05
) -> dict[str, float]:
    """Fraction of residuals inside the interval built without them.

    This is the honest small-sample check: in-sample coverage is trivially
    at least ``1 - alpha`` when the interval exists at all.
    """
    values = [float(value) for value in residuals]
    covered = 0
    total = 0
    for index, value in enumerate(values):
        others = values[:index] + values[index + 1 :]
        interval = conformal_interval(others, center=0.0, alpha=alpha)
        if interval.status != "OK":
            continue
        total += 1
        covered += int(
            interval.lower is not None
            and interval.upper is not None
            and interval.lower <= value <= interval.upper
        )
    return {
        "alpha": alpha,
        "n": total,
        "coverage": covered / total if total else 0.0,
    }
