"""Calibration challengers beyond a shared temperature.

A single softmax temperature cannot correct a wrong-direction bias: it is one
positive scalar, so it rescales confidence but preserves every ranking. When a
head is systematically inverted - we have measured exactly that, `severity`
reaching Matthews correlation -1.0 - temperature scaling can make calibration
*worse*, which is also what the nearest public benchmark reported
(docs/next-phase-plan.md, 2.4).

Two challengers, and the difference between them is the point:

* **Platt** (per-class one-vs-rest logistic scaling) fits a free-sign slope and
  an intercept per class. A negative slope *inverts* that class's ranking, so
  Platt is the one that can rescue an anti-correlated head.
* **Isotonic** (per-class monotone step map) is non-decreasing by construction.
  It repairs the shape of a probability curve - genuinely useful against
  overconfidence - but it cannot invert anything.

What isotonic does to a wrong-direction head is worth stating precisely,
because the intuitive claim is wrong in both directions. It is *not* true that
a monotone calibrator preserves argmax: these maps are fitted per class, and
monotone maps with different shapes can and do reorder across classes (only a
*shared* monotone transform preserves argmax exactly). Nor does isotonic rescue
the head. On a perfectly anti-correlated target the one-vs-rest labels are
decreasing in the score, so the non-decreasing least-squares fit collapses to
the constant class prior - it becomes uninformative rather than correct. On our
fixture that is exactly what happens: every fitted map is flat at 1/3 and
prediction degenerates to a single class.

Calibrators are fit on the calibration split only and selected there only. A
fitted calibrator of any kind is still not a calibration certificate: no
artifact may be promoted without a pinned real calibration set
(`docs/claims.md`).

Dependencies stay at numpy; there is no scikit-learn here and there will not
be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

CALIBRATOR_VERSION = 1

TEMPERATURE = "temperature"
PLATT = "platt"
ISOTONIC = "isotonic"


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def negative_log_likelihood(
    probabilities: np.ndarray, targets: np.ndarray
) -> float:
    """Mean NLL of the probability assigned to the true class."""
    if not len(targets):
        return float("inf")
    picked = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def reliability_bins(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bins: int = 10,
) -> dict[str, Any]:
    """Equal-width reliability-diagram data, enough to plot or diff.

    Returned rather than drawn so the report stays machine-readable and a
    change in calibration shape is visible in review as numbers.
    """
    if not len(targets):
        return {"bins": [], "ece": None, "n": 0}
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == targets).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    total = len(targets)
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        if not count:
            rows.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                }
            )
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        ece += (count / total) * abs(accuracy - mean_confidence)
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {"bins": rows, "ece": float(ece), "n": total}


@dataclass
class TemperatureCalibrator:
    """One positive scalar. Rescales confidence, preserves every ranking."""

    temperature: float = 1.0
    kind: str = TEMPERATURE

    def apply(self, logits: np.ndarray) -> np.ndarray:
        return _softmax(logits / max(self.temperature, 1e-6))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "temperature": float(self.temperature)}


@dataclass
class PlattCalibrator:
    """Per-class one-vs-rest logistic scaling with a free-sign slope.

    The free sign is what matters: a fitted negative slope inverts that class,
    so this is the only calibrator here that can rescue a wrong-direction head.
    Probabilities are renormalized across classes afterwards.
    """

    slopes: np.ndarray
    intercepts: np.ndarray
    kind: str = PLATT

    def apply(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits * self.slopes + self.intercepts
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(scaled, -60, 60)))
        totals = probabilities.sum(axis=-1, keepdims=True)
        return np.divide(
            probabilities,
            totals,
            out=np.full_like(probabilities, 1.0 / probabilities.shape[-1]),
            where=totals > 0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "slopes": self.slopes.tolist(),
            "intercepts": self.intercepts.tolist(),
        }


@dataclass
class IsotonicCalibrator:
    """Per-class monotone step map from score to probability.

    Non-decreasing by construction, so it can repair calibration shape but can
    never change a ranking or an argmax. Present to demonstrate that limit.
    """

    scores: list[np.ndarray]
    fitted: list[np.ndarray]
    kind: str = ISOTONIC

    def apply(self, logits: np.ndarray) -> np.ndarray:
        columns = []
        for index in range(logits.shape[-1]):
            columns.append(
                np.interp(
                    logits[:, index], self.scores[index], self.fitted[index]
                )
            )
        probabilities = np.stack(columns, axis=-1)
        totals = probabilities.sum(axis=-1, keepdims=True)
        return np.divide(
            probabilities,
            totals,
            out=np.full_like(probabilities, 1.0 / probabilities.shape[-1]),
            where=totals > 0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "scores": [values.tolist() for values in self.scores],
            "fitted": [values.tolist() for values in self.fitted],
        }


def _pava(values: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: least-squares non-decreasing fit."""
    blocks: list[list[float]] = []  # [mean, weight]
    for value in values:
        blocks.append([float(value), 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left[1] + right[1]
            mean = (left[0] * left[1] + right[0] * right[1]) / weight
            blocks.append([mean, weight])
    fitted: list[float] = []
    for mean, weight in blocks:
        fitted.extend([mean] * int(weight))
    return np.asarray(fitted, dtype=float)


def fit_platt(
    logits: np.ndarray,
    targets: np.ndarray,
    steps: int = 400,
    learning_rate: float = 0.5,
    l2: float = 1e-3,
) -> PlattCalibrator:
    """Fit one free-sign logistic scale per class on its one-vs-rest target.

    Gradient descent on two parameters per class. The slope starts at 1 and is
    allowed to go negative, which is the entire reason this calibrator exists.
    """
    n_classes = logits.shape[-1]
    slopes = np.ones(n_classes)
    intercepts = np.zeros(n_classes)
    one_hot = np.zeros_like(logits)
    one_hot[np.arange(len(targets)), targets] = 1.0
    for _ in range(steps):
        scaled = logits * slopes + intercepts
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(scaled, -60, 60)))
        error = probabilities - one_hot
        slope_gradient = (error * logits).mean(axis=0) + l2 * slopes
        intercept_gradient = error.mean(axis=0)
        slopes -= learning_rate * slope_gradient
        intercepts -= learning_rate * intercept_gradient
    return PlattCalibrator(slopes=slopes, intercepts=intercepts)


def fit_isotonic(logits: np.ndarray, targets: np.ndarray) -> IsotonicCalibrator:
    """Per-class monotone map from raw score to empirical probability."""
    n_classes = logits.shape[-1]
    scores: list[np.ndarray] = []
    fitted: list[np.ndarray] = []
    for index in range(n_classes):
        column = logits[:, index]
        truth = (targets == index).astype(float)
        order = np.argsort(column, kind="stable")
        sorted_scores = column[order]
        sorted_truth = truth[order]
        monotone = _pava(sorted_truth)
        # Collapse duplicate scores so the step map is well defined.
        unique_scores, inverse = np.unique(sorted_scores, return_inverse=True)
        collapsed = np.zeros(len(unique_scores))
        counts = np.zeros(len(unique_scores))
        np.add.at(collapsed, inverse, monotone)
        np.add.at(counts, inverse, 1.0)
        collapsed = np.divide(
            collapsed, counts, out=np.zeros_like(collapsed), where=counts > 0
        )
        scores.append(unique_scores)
        fitted.append(np.maximum.accumulate(collapsed))
    return IsotonicCalibrator(scores=scores, fitted=fitted)


CALIBRATORS: dict[str, Any] = {
    TEMPERATURE: lambda logits, targets: TemperatureCalibrator(
        temperature=_fit_temperature(logits, targets)
    ),
    PLATT: fit_platt,
    ISOTONIC: fit_isotonic,
}


def _fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    from .training import fit_temperature

    return fit_temperature(logits, targets)


def select_calibrator(
    logits: np.ndarray,
    targets: np.ndarray,
    names: tuple[str, ...] = (TEMPERATURE, PLATT, ISOTONIC),
) -> tuple[Any, dict[str, float]]:
    """Fit every challenger on the same split and pick the lowest NLL.

    Selection and scoring must happen on the same split here by design: this
    returns a winner plus the candidate NLLs so a caller can report the
    comparison. Callers must never fit on the split they report on.
    """
    if not len(targets):
        raise ValueError("calibration requires a non-empty split")
    candidates: dict[str, Any] = {}
    scores: dict[str, float] = {}
    for name in names:
        calibrator = CALIBRATORS[name](logits, targets)
        candidates[name] = calibrator
        scores[name] = negative_log_likelihood(calibrator.apply(logits), targets)
    best = min(scores, key=lambda key: scores[key])
    return candidates[best], scores


def calibrator_report(calibrator: Any) -> dict[str, Any]:
    """Identity and fitted parameters, for the artifact's provenance."""
    payload = dict(calibrator.to_dict())
    payload["calibrator_version"] = CALIBRATOR_VERSION
    payload["warning"] = (
        "A fitted calibrator is not a calibration certificate; no artifact may "
        "be promoted without a pinned real calibration set."
    )
    return payload
