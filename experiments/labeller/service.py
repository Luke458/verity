"""Cause labelling by a decision service (Jev/System One style): state + yes/no questions.

The service sees the same evidence as the tree labeller: the 64 engine features
of a row, rendered as readable JSON (``render_state``). It is asked one yes/no
question per cause (``QUESTIONS``). Inference runs elsewhere (``lux.py`` on a
GPU, or any remote service) and writes one probability per cause per row;
``ServiceLabeller`` reads that file, so evaluation never needs the model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .features import CHECK_PREFIXES, EVENT_CLASSES, LEVELS, METRICS, STAGES, STATUSES
from .models import CAUSES, tune_threshold

PROMPT_VERSION = 1
CONTEXT = (
    "Automated quality control of a weekly retail sales table (store x product x week). "
    "A new version of the table was compared with the previous version; below is the "
    "engine's evidence about what changed."
)
QUESTIONS: dict[str, str] = {
    "MISSING_STORES": "Did stores that should have reported in the latest week fail to appear?",
    "MISSING_PRODUCTS": "Did products that should have sold in the latest week fail to appear?",
    "ENTITY_MERGE": "Were two stores or products merged, or one id replaced by another?",
    "BACKFILL": "Was a new store or product added together with its past history (a backfill)?",
    "HISTORICAL_CORRECTION": (
        "Did the source restate or correct already-published history "
        "(a recalculation, a truncation or a restatement of past weeks)?"
    ),
    "RECLASSIFICATION": "Were products moved from one category to another?",
    "CODING": "Was the change introduced by an error in the product-coding stage of the pipeline?",
    "WAREHOUSE": "Was the change introduced by an error in the warehouse transformation stage?",
    "SCHEMA_FAILURE": "Did the new data fail its contract (schema, types, nulls or duplicate rows)?",
    "MARKET_MOVEMENT": (
        "Did sales genuinely move in the latest week (a real market change, "
        "not a data or pipeline problem)?"
    ),
}
assert set(QUESTIONS) == set(CAUSES)


def _round(value: float) -> float:
    return float(f"{value:.4g}")


def render_state(features: dict[str, float]) -> dict[str, Any]:
    """The row's features as readable JSON; zero counts are omitted."""
    f = features
    status = next((s for s in STATUSES if f.get(f"status:{s}")), "UNKNOWN")
    state: dict[str, Any] = {"context": CONTEXT, "refresh status": status}
    if f.get("contracts_failed"):
        state["failed data contracts"] = int(f["contracts_failed"])
    events = {
        name.lower().replace("_", " "): int(f[f"events:{name}"])
        for name in EVENT_CLASSES
        if f.get(f"events:{name}")
    }
    if events:
        state["entity lifecycle events"] = events
    for entity in ("store", "product"):
        share = f.get(f"missing_share:{entity}", 0.0)
        if share:
            state[f"share of {entity}s missing in latest week"] = _round(share)
            state[f"missing {entity}s material"] = bool(f.get(f"missing_material:{entity}"))
    if "revision_relative" in f:
        state["revision of overlapping history"] = {
            "total change, relative": _round(f["revision_relative"]),
            "unexplained change, relative": _round(f["unexplained_relative"]),
            "fraction explained by entity events": _round(f["explained_fraction"]),
            "material": bool(f["revision_material"]),
            "breadth (share of keys revised)": _round(f["revision_breadth"]),
            "dollars changed without units": bool(f.get("dollar_without_units")),
        }
    if f.get("revision_weeks_material"):
        state["individual weeks materially revised"] = int(f["revision_weeks_material"])
        state["largest single-week revision, relative"] = _round(f["revision_week_max_relative"])
    if f.get("reconstruction_score", -1.0) >= 0:
        state["counterfactual reconstruction score"] = _round(f["reconstruction_score"])
    if f.get("reconciliation_failures"):
        state["reconciliation failures"] = int(f["reconciliation_failures"])
    if f.get("price_ratio_flags"):
        state["price ratio outliers"] = int(f["price_ratio_flags"])
    origin = next((stage for stage in STAGES if f.get(f"origin:{stage}")), None)
    increments = {
        stage: _round(10 ** f[f"increment_log:{stage}"])
        for stage in STAGES
        if f.get(f"increment_log:{stage}", -12.0) > -12.0
    }
    if origin or increments:
        state["pipeline lineage"] = {
            "stage where the change first appears": origin or "none",
            "revision added at each stage, relative": increments,
        }
    if f.get("relationships"):
        state["candidate id merges or replacements"] = int(f["relationships"])
    anomalies = {
        f"{level.removesuffix('_id')} {metric}": int(f[f"anomalies:{level}:{metric}"])
        for level in LEVELS
        for metric in METRICS
        if f.get(f"anomalies:{level}:{metric}")
    }
    if anomalies:
        state["latest-week anomalies"] = anomalies
    if f.get("min_share_p_log", 0.0) < 0:
        state["strongest category share shift"] = {
            "p-value": float(f"{10 ** f['min_share_p_log']:.2g}"),
            "t statistic": _round(f["max_abs_share_t"]),
        }
    state["national latest-week deviation from forecast, relative"] = _round(
        f.get("national_relative_residual", 0.0)
    )
    failing = {prefix.replace("_", " "): int(f[f"fails:{prefix}"]) for prefix in CHECK_PREFIXES if f.get(f"fails:{prefix}")}
    if failing:
        state["failing checks"] = failing
    return state


def questions() -> dict[str, dict[str, Any]]:
    return {
        cause: {"type": "noul", "instructions": text, "criteria": {"false": "No", "true": "Yes"}}
        for cause, text in QUESTIONS.items()
    }


def state_sha256(state: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def row_key(row: dict[str, Any]) -> str:
    return f"{row['split']}/{row['seed']}/{row['index']}"


def load_probabilities(path: str | Path) -> dict[str, dict[str, float]]:
    """``row_key -> {cause: P(yes)}`` from a service output file (JSONL)."""
    out: dict[str, dict[str, float]] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("prompt_version") != PROMPT_VERSION:
                raise ValueError(f"{path}: prompt version {record.get('prompt_version')} != {PROMPT_VERSION}")
            out[record["key"]] = record["probabilities"]
    return out


class ServiceLabeller:
    """Causes whose P(yes) clears a threshold: 0.5 (zero-shot) or tuned per cause on dev."""

    def __init__(self, name: str, probabilities: dict[str, dict[str, float]], tuned: bool):
        self.name = name
        self.probabilities = probabilities
        self.tuned = tuned
        self.thresholds = {cause: 0.5 for cause in CAUSES}

    def _p(self, row: dict[str, Any], cause: str) -> float:
        return float(self.probabilities[row_key(row)][cause])

    def fit(self, train: Sequence[dict[str, Any]], dev: Sequence[dict[str, Any]]) -> None:
        if not self.tuned:
            return
        for cause in CAUSES:
            self.thresholds[cause] = tune_threshold(
                [self._p(row, cause) for row in dev], [cause in row["causes"] for row in dev]
            )

    def predict(self, rows: Sequence[dict[str, Any]]) -> list[set[str]]:
        return [
            {cause for cause in CAUSES if self._p(row, cause) >= self.thresholds[cause]}
            for row in rows
        ]
