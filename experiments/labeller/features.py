"""Feature vector for cause labelling, built only from the engine's output.

Nothing here reads the oracle: every value comes from a ``QCRunResult`` that a
production run would also have. ``FEATURE_VERSION`` changes whenever the
vector's meaning changes, so cached datasets are never silently mixed.
"""

from __future__ import annotations

import math
from typing import Any

from qc.lifecycle import missing_entity_impact

FEATURE_VERSION = 1
STATUSES = ("PASS", "PASS_WITH_EXPLANATION", "INVESTIGATE", "DATA_CONTRACT_FAILURE", "INCOMPLETE")
EVENT_CLASSES = (
    "NEW_ENTITY_HISTORICAL_BACKFILL",
    "NEW_ENTITY_RECENT",
    "ENTITY_REMOVED",
    "ENTITY_HISTORY_EXTENDED",
    "ENTITY_HISTORY_TRUNCATED",
    "LATEST_WEEK_MISSING",
    "POSSIBLE_RECLASSIFICATION",
)
STAGES = ("source", "coded", "warehouse", "report")
LEVELS = ("national", "banner_id", "commodity_id")
METRICS = ("dollar", "units", "scripts", "stock")
CHECK_PREFIXES = (
    "contracts",
    "reconciliation",
    "historical_revision",
    "revision_week",
    "temporal",
    "ratio",
    "counterfactual_residual",
    "recurrence",
)


def _log10(value: float, floor: float = 1e-12) -> float:
    return math.log10(max(float(value), floor))


def extract(result: Any, config: Any) -> dict[str, float]:
    """Named numeric features for one assessed refresh."""
    f: dict[str, float] = {}
    for status in STATUSES:
        f[f"status:{status}"] = float(result.status == status)
    f["contracts_failed"] = float(len(result.contracts.failed))

    events = list(result.events or ())
    for name in EVENT_CLASSES:
        f[f"events:{name}"] = float(sum(1 for event in events if event.classification == name))
    impact = missing_entity_impact(events, config) if events else {}
    for entity_type in ("store", "product"):
        entry = impact.get(entity_type) or {}
        f[f"missing_share:{entity_type}"] = float(entry.get("share") or 0.0)
        f[f"missing_material:{entity_type}"] = float(bool(entry.get("material")))

    attribution = result.attribution
    if attribution is not None:
        total = abs(float(attribution.previous_total)) or 1.0
        f["revision_relative"] = float(attribution.raw_delta) / total
        f["unexplained_relative"] = float(attribution.unexplained_delta) / total
        f["explained_fraction"] = float(attribution.explained_fraction)
        f["revision_material"] = float(attribution.material)
        f["revision_breadth"] = float(attribution.breadth)
        f["over_explained"] = float(attribution.over_explained)
        f["offsetting"] = float(attribution.offsetting)
        f["dollar_without_units"] = float(
            "dollar_change_without_units" in attribution.cross_metric_flags
        )
        f["structural_events"] = float(len(attribution.structural_events))
    counterfactual = result.counterfactual
    f["reconstruction_score"] = (
        float(counterfactual.reconciliation_score)
        if counterfactual is not None and counterfactual.reconciliation_score is not None
        else -1.0
    )
    reconciliation = result.reconciliation
    f["reconciliation_failures"] = float(len(reconciliation.failed)) if reconciliation else 0.0
    f["price_ratio_flags"] = float(len(reconciliation.ratio_flags)) if reconciliation else 0.0

    lineage = result.lineage
    origin = lineage.first_divergence if lineage is not None else None
    for stage in STAGES:
        f[f"origin:{stage}"] = float(origin == stage)
    increments = {
        item["stage"]: item.get("increment")
        for item in (lineage.divergences if lineage is not None else [])
    }
    for stage in STAGES:
        value = increments.get(stage)
        f[f"increment_log:{stage}"] = _log10(value) if value else -12.0

    f["relationships"] = float(len(result.relationships or ()))
    material_weeks = [item for item in (result.week_revisions or ()) if item.material]
    f["revision_weeks_material"] = float(len(material_weeks))
    f["revision_week_max_relative"] = max(
        (min(float(item.relative), 10.0) for item in (result.week_revisions or ())),
        default=0.0,
    )

    temporal = result.temporal
    series = list(temporal.series) if temporal is not None else []
    for level in LEVELS:
        for metric in METRICS:
            f[f"anomalies:{level}:{metric}"] = float(
                sum(1 for item in series if item.anomaly and item.level == level and item.metric == metric)
            )
    share_p = [item.share_p for item in series if item.share_p is not None]
    f["min_share_p_log"] = _log10(min(share_p)) if share_p else 0.0
    f["max_abs_share_t"] = max((abs(float(item.share_t)) for item in series if item.share_t is not None), default=0.0)
    national = [item for item in series if item.level == "national" and item.metric == "dollar"]
    f["national_relative_residual"] = float(national[0].relative_residual) if national else 0.0
    f["max_leaf_relative_residual"] = max(
        (abs(float(item.relative_residual)) for item in series if item.level != "national"),
        default=0.0,
    )

    findings = result.machine.get("findings", [])
    for prefix in CHECK_PREFIXES:
        f[f"fails:{prefix}"] = float(
            sum(
                1
                for item in findings
                if item["outcome"] in ("FAIL", "CONTRACT_FAILURE")
                and not item.get("approval_ids")
                and str(item["check"]).startswith(prefix)
            )
        )
    return f
