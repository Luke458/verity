"""Finite-sample conformal intervals (architecture section 28).

The interval uses the ``ceil((n + 1) * (1 - alpha))``-th smallest absolute
residual, which gives finite-sample marginal coverage when residuals are
exchangeable with future errors. When that rank exceeds ``n`` the interval is
reported as ``INSUFFICIENT_CALIBRATION`` instead of being widened by
assumption. Temporal drift breaks the exchangeability assumption and must be
re-checked with prequential (across-load) calibration.

Residuals must be finite; a non-finite value is rejected rather than sorted
into the interval, because it silently destroys the coverage guarantee.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def minimum_samples(alpha: float) -> int:
    """Smallest n for which an interval at ``alpha`` exists.

    ``ceil((n + 1) * (1 - alpha)) <= n`` rearranges to ``n >= 1/alpha - 1``;
    the epsilon keeps float representation from rounding 0.2 up to 5.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return max(0, math.ceil(1.0 / alpha - 1.0 - 1e-9))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    phat = successes / n
    denominator = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denominator
    margin = (
        z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _require_finite(residuals: Sequence[float]) -> list[float]:
    values = [float(value) for value in residuals]
    for value in values:
        if not math.isfinite(value):
            raise ValueError("residuals must be finite")
    return values


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
    absolute = sorted(abs(value) for value in _require_finite(residuals))
    n = len(absolute)
    if n == 0:
        return ConformalInterval(
            alpha, None, None, n, None, "INSUFFICIENT_CALIBRATION", "no residuals"
        )
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        required = minimum_samples(alpha)
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
) -> dict[str, float | int | bool | None]:
    """Fraction of residuals inside the interval built without them.

    This is the honest small-sample check: in-sample coverage is trivially at
    least ``1 - alpha`` when the interval exists at all. Returns
    ``coverage=None`` when no leave-one-out interval could be built, so
    "not evaluable" is never reported as zero coverage.
    """
    values = _require_finite(residuals)
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
        "evaluated": total > 0,
        "coverage": covered / total if total else None,
    }
