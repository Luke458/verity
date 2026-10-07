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

PROMPT_VERSIONS = (1, 2, 3)
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

# Version 2: the labelling policy an annotator would be given, plus the feed's
# declared configuration, go into the state; each question cites its definition.
POLICY = (
    "Label the causes of this refresh's changes. Most refreshes have one cause or none; "
    "answer yes only when the evidence below supports that cause under these definitions.\n"
    "- Missing stores / missing products: entities expected in the latest week are absent, "
    "and the absence is material. Products that vanish with their missing stores are not a "
    "separate missing-products cause.\n"
    "- Entity merge: a candidate id merge or replacement was found. The removal of the old id "
    "and the history of the new id are part of the merge.\n"
    "- Backfill: a new store or product arrived together with past history.\n"
    "- Reclassification: products moved between categories. A category emptied by the move, "
    "and the revisions the move causes, are part of the reclassification.\n"
    "- Historical correction: the source changed already-published history in a way lifecycle "
    "events do not explain: history truncated, a material unexplained revision first appearing "
    "at the source stage, or individual weeks materially revised. Revisions inside the "
    "declared late-arrival window are expected and are not a correction.\n"
    "- Coding error / warehouse error: an unexplained revision first appears at the coding / "
    "warehouse stage of the pipeline, even if it is small.\n"
    "- Schema failure: the new data failed its data contracts.\n"
    "- Market movement: latest-week sales moved (latest-week anomalies or a category share "
    "shift) and no other cause explains it."
)
DEFINITIONS: dict[str, str] = {
    "MISSING_STORES": "Under the policy, is there a missing-stores cause?",
    "MISSING_PRODUCTS": "Under the policy, is there a missing-products cause?",
    "ENTITY_MERGE": "Under the policy, is there an entity-merge cause?",
    "BACKFILL": "Under the policy, is there a backfill cause?",
    "HISTORICAL_CORRECTION": "Under the policy, is there a historical-correction cause?",
    "RECLASSIFICATION": "Under the policy, is there a reclassification cause?",
    "CODING": "Under the policy, is there a coding-error cause?",
    "WAREHOUSE": "Under the policy, is there a warehouse-error cause?",
    "SCHEMA_FAILURE": "Under the policy, is there a schema-failure cause?",
    "MARKET_MOVEMENT": "Under the policy, is there a market-movement cause?",
}
assert set(DEFINITIONS) == set(CAUSES)


def _round(value: float) -> float:
    return float(f"{value:.4g}")


POLICY_V3 = (
    "Label the causes of this refresh's changes. Most refreshes have one cause or none; "
    "answer yes only when the evidence below supports that cause under these definitions.\n"
    "- Missing stores / missing products: entities expected in the latest week are absent, "
    "and the absence is material. Products that vanish with their missing stores are not a "
    "separate missing-products cause.\n"
    "- Entity merge: a candidate id merge or replacement was found. The removal of the old id "
    "and the history of the new id are part of the merge.\n"
    "- Backfill: a new store or product arrived together with past history.\n"
    "- Reclassification: products moved between categories. A category emptied by the move, "
    "and the revisions the move causes, are part of the reclassification.\n"
    "- Historical correction: history was truncated, or the unexplained part of the revision "
    "is material and first appears at the source stage, or individual weeks were materially "
    "revised while no merge, backfill or reclassification explains them. Revisions inside the "
    "declared late-arrival window are expected and are not a correction.\n"
    "- Coding error / warehouse error: the unexplained change first appears at the coding / "
    "warehouse stage (the 'first appears at' field), even if it is small. A change explained "
    "by a reclassification, merge or backfill is not a coding or warehouse error.\n"
    "- Schema failure: the new data failed its data contracts.\n"
    "- Market movement: the engine flagged significant latest-week anomalies and no other cause "
    "explains them. Small deviations that were not flagged are normal noise."
)


def _render_v3(f: dict[str, float], late_arrival_weeks: int) -> dict[str, Any]:
    """Version 3: absence stated explicitly, judgments the engine already made spelled out."""
    from qc.config import DatasetConfig

    status = next((s for s in STATUSES if f.get(f"status:{s}")), "UNKNOWN")
    state: dict[str, Any] = {
        "context": CONTEXT,
        "labelling policy": POLICY_V3,
        "feed configuration": {"declared late-arrival window, weeks": late_arrival_weeks},
        "refresh status": status,
        "data contracts": "failed" if f.get("contracts_failed") else "passed",
    }
    events = {
        name.lower().replace("_", " "): int(f[f"events:{name}"])
        for name in EVENT_CLASSES
        if f.get(f"events:{name}")
    }
    state["entity lifecycle events"] = events or "none"
    state["candidate id merges or replacements"] = int(f.get("relationships", 0))
    for entity in ("store", "product"):
        share = f.get(f"missing_share:{entity}", 0.0)
        state[f"{entity}s missing in latest week"] = (
            {"share": _round(share), "material": bool(f.get(f"missing_material:{entity}"))} if share else "none"
        )
    if "revision_relative" in f:
        unexplained_material = bool(f["revision_material"]) and (
            f["explained_fraction"] < DatasetConfig().explained_fraction_threshold
        )
        state["revision of overlapping history"] = {
            "total change, relative": _round(f["revision_relative"]),
            "share explained by lifecycle events": _round(f["explained_fraction"]),
            "unexplained part material": unexplained_material,
        }
    else:
        state["revision of overlapping history"] = "none"
    state["individual weeks materially revised"] = int(f.get("revision_weeks_material", 0))
    origin = next((stage for stage in STAGES if f.get(f"origin:{stage}")), None)
    state["unexplained change first appears at"] = origin or "no stage"
    flagged = {
        f"{level.removesuffix('_id')} {metric}": int(f[f"anomalies:{level}:{metric}"])
        for level in LEVELS
        for metric in METRICS
        if f.get(f"anomalies:{level}:{metric}")
    }
    state["significant latest-week anomalies"] = flagged or "none"
    failing = {prefix.replace("_", " "): int(f[f"fails:{prefix}"]) for prefix in CHECK_PREFIXES if f.get(f"fails:{prefix}")}
    state["failing checks"] = failing or "none"
    return state


def render_state(
    features: dict[str, float], version: int = 1, late_arrival_weeks: int = 0
) -> dict[str, Any]:
    """The row's features as readable JSON; zero counts are omitted.

    Version 2 adds the labelling policy and the feed's declared late-arrival
    window (operator configuration, known before the refresh arrives).
    """
    f = features
    if version >= 3:
        return _render_v3(f, late_arrival_weeks)
    status = next((s for s in STATUSES if f.get(f"status:{s}")), "UNKNOWN")
    state: dict[str, Any] = {"context": CONTEXT}
    if version >= 2:
        state["labelling policy"] = POLICY
        state["feed configuration"] = {"declared late-arrival window, weeks": late_arrival_weeks}
    state["refresh status"] = status
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


def questions(version: int = 1) -> dict[str, dict[str, Any]]:
    texts = QUESTIONS if version == 1 else DEFINITIONS
    return {
        cause: {"type": "noul", "instructions": text, "criteria": {"false": "No", "true": "Yes"}}
        for cause, text in texts.items()
    }


def row_state(row: dict[str, Any], version: int) -> dict[str, Any]:
    """A dataset row's state for a prompt version (the window comes from its profile's config)."""
    from .data import PROFILE_CONFIG

    window = int(PROFILE_CONFIG[row["profile"]].get("restatement_weeks", 0))
    return render_state(row["features"], version, window)


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
            if record.get("prompt_version") not in PROMPT_VERSIONS:
                raise ValueError(f"{path}: unknown prompt version {record.get('prompt_version')}")
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
