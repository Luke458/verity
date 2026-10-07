"""Cause labellers behind one interface: ``fit(train, dev)`` and ``predict(rows)``.

A labeller returns a *set* of causes per refresh: empty for "nothing wrong", one
for a single fault, more for simultaneous faults. ``RuleLabeller`` wraps the
engine's rule-based cause set (``likely_causes``) and is the baseline. Any
other model, including a remote Jev-style decision service, plugs in by
implementing the same two methods over the same rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from qc.cohort import ORACLE_CAUSE

CAUSES: tuple[str, ...] = tuple(sorted(set(ORACLE_CAUSE.values())))


def tune_threshold(probabilities: Sequence[float] | np.ndarray, truth: Sequence[bool] | np.ndarray) -> float:
    """The decision threshold (0.1-0.9 in steps of 0.05) with the best F1 on dev."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(truth, dtype=bool)
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.1, 0.9, 17):
        predicted = p >= threshold
        denominator = int(predicted.sum()) + int(y.sum())
        f1 = 2 * int(np.sum(predicted & y)) / denominator if denominator else 1.0
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), f1
    return best_threshold


class Labeller(Protocol):
    name: str

    def fit(self, train: Sequence[dict[str, Any]], dev: Sequence[dict[str, Any]]) -> None: ...

    def predict(self, rows: Sequence[dict[str, Any]]) -> list[set[str]]: ...


class RuleLabeller:
    """The engine's rule cause set (``likely_causes``)."""

    name = "rules"

    def fit(self, train: Sequence[dict[str, Any]], dev: Sequence[dict[str, Any]]) -> None:
        return None

    def predict(self, rows: Sequence[dict[str, Any]]) -> list[set[str]]:
        return [set(row["rule_causes"]) for row in rows]


class TreeLabeller:
    """One gradient-boosted tree classifier per cause, thresholds tuned on dev.

    Each cause's decision threshold maximizes that cause's F1 on the dev split;
    the test split is never seen during fitting or tuning.
    """

    name = "trees"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.feature_names: list[str] = []
        self.models: dict[str, Any] = {}
        self.thresholds: dict[str, float] = {}

    def _matrix(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [[float(row["features"].get(name, 0.0)) for name in self.feature_names] for row in rows],
            dtype=float,
        )

    @staticmethod
    def _targets(rows: Sequence[dict[str, Any]], cause: str) -> np.ndarray:
        return np.asarray([cause in row["causes"] for row in rows], dtype=int)

    def fit(self, train: Sequence[dict[str, Any]], dev: Sequence[dict[str, Any]]) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.feature_names = sorted({name for row in train for name in row["features"]})
        x_train, x_dev = self._matrix(train), self._matrix(dev)
        for cause in CAUSES:
            y_train = self._targets(train, cause)
            if y_train.min() == y_train.max():
                continue  # never (or always) present in training: no model
            model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=self.seed
            )
            model.fit(x_train, y_train)
            self.models[cause] = model
            self.thresholds[cause] = tune_threshold(
                model.predict_proba(x_dev)[:, 1], self._targets(dev, cause)
            )

    def predict(self, rows: Sequence[dict[str, Any]]) -> list[set[str]]:
        x = self._matrix(rows)
        predictions: list[set[str]] = [set() for _ in rows]
        for cause, model in self.models.items():
            hits = model.predict_proba(x)[:, 1] >= self.thresholds[cause]
            for position in np.flatnonzero(hits):
                predictions[int(position)].add(cause)
        return predictions
