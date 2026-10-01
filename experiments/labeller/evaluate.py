"""Score cause-set predictions against the oracle, and models against each other."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from qc.conformal import wilson_interval

from .models import CAUSES


def _rate(hits: int, n: int) -> dict[str, Any]:
    low, high = wilson_interval(hits, n)
    return {"hits": hits, "n": n, "rate": hits / n if n else None,
            "ci_low": low if n else None, "ci_high": high if n else None}


def score(rows: Sequence[dict[str, Any]], predictions: Sequence[set[str]]) -> dict[str, Any]:
    truths = [set(row["causes"]) for row in rows]
    exact = [prediction == truth for prediction, truth in zip(predictions, truths, strict=True)]
    report: dict[str, Any] = {"exact": _rate(sum(exact), len(exact)), "by_kind": {}}
    for kind in ("single", "pair", "clean"):
        positions = [i for i, row in enumerate(rows) if row["kind"] == kind]
        entry = {"exact": _rate(sum(exact[i] for i in positions), len(positions))}
        if kind == "pair":
            entry["names_both"] = _rate(sum(truths[i] <= predictions[i] for i in positions), len(positions))
            entry["names_one_or_more"] = _rate(
                sum(bool(truths[i] & predictions[i]) for i in positions), len(positions)
            )
        report["by_kind"][kind] = entry
    per_cause: dict[str, dict[str, float | int]] = {}
    total_tp = total_fp = total_fn = 0
    for cause in CAUSES:
        tp = sum(cause in p and cause in t for p, t in zip(predictions, truths, strict=True))
        fp = sum(cause in p and cause not in t for p, t in zip(predictions, truths, strict=True))
        fn = sum(cause not in p and cause in t for p, t in zip(predictions, truths, strict=True))
        total_tp, total_fp, total_fn = total_tp + tp, total_fp + fp, total_fn + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_cause[cause] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    report["per_cause"] = per_cause
    micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    report["micro_f1"] = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    report["exact_by_case"] = exact
    return report


def mcnemar(first: Sequence[bool], second: Sequence[bool]) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired correctness (same test cases)."""
    only_first = sum(a and not b for a, b in zip(first, second, strict=True))
    only_second = sum(b and not a for a, b in zip(first, second, strict=True))
    n = only_first + only_second
    if n == 0:
        return {"only_first": 0, "only_second": 0, "p_value": 1.0}
    k = min(only_first, only_second)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return {"only_first": only_first, "only_second": only_second, "p_value": min(1.0, 2 * tail)}
