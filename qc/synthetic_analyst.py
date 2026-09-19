"""Synthetic analyst outcomes: exercise the feedback loop without real analysts.

Simulates what analysts do after a run: confirm or leave a draft, get the cause
right or wrong, sometimes answer UNKNOWN, sometimes correct themselves later,
and occasionally disagree with the engine about whether investigation was
needed. Outcomes are tagged with provenance ``synthetic``, so
``records_from_store`` and ``qc champion`` can never mistake them for validated
analyst labels: ``production_eligible`` stays False while the whole loop
(drafts, corrections, store, labels, training, gates, drift) runs end to end.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .decisions import CAUSE_VALUES, ORIGIN_VALUES, SEVERITY_VALUES
from .labels import oracle_labels_for_result
from .store import SqliteStore

EPOCH = _dt.date(2024, 1, 6)  # synthetic week 1 ends here

NOTE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "MISSING_STORES": (
        "Supplier extract omitted stores; re-delivered the source file and the stores returned.",
        "Source delivery dropped a store block; no transformation was involved.",
    ),
    "MISSING_PRODUCTS": (
        "Assortment extract missed products for the affected stores; corrected upstream.",
        "Product feed omitted lines; re-export fixed it.",
    ),
    "SOURCE_INGESTION": (
        "Source job completed with a partial file; re-run restored the missing volume.",
        "Upstream delivery was incomplete; the provider re-sent it.",
    ),
    "CODING": (
        "A mapping change was applied ahead of approval; reverted and re-coded.",
        "Product coding table update caused the shift; corrected in the mapping.",
    ),
    "WAREHOUSE": (
        "Warehouse transformation was redeployed with a bad config; rolled back.",
        "Aggregation logic changed without notice; restored the previous version.",
    ),
    "MARKET_MOVEMENT": (
        "Movement aligns with promotions and public holidays; no defect found.",
        "Category demand shift confirmed against external indicators.",
    ),
    "HISTORICAL_CORRECTION": (
        "Registered correction; the expected history window matches.",
        "Approved recalculation; no further action needed.",
    ),
    "RECLASSIFICATION": (
        "Product-to-commodity remap was intentional; mapping table updated.",
        "Reclassification under a new code; totals reconciled.",
    ),
    "ENTITY_MERGE": (
        "Store was merged into a successor; the successor carries its history.",
        "Identifier replacement confirmed by master data.",
    ),
    "BACKFILL": (
        "New entity onboarding with historical backfill; registered and expected.",
        "Historical load for a new entity; approved by the business.",
    ),
    "SCHEMA_FAILURE": (
        "Upstream schema change broke the contract; fixed and redeployed.",
        "Required column was dropped by a producer change; restored.",
    ),
    "UNKNOWN": (
        "Cause not established; escalated to the data engineering team.",
        "No single explanation found; monitoring next refresh.",
    ),
}


@dataclass(frozen=True)
class AnalystProfile:
    name: str
    cause_accuracy: float
    origin_accuracy: float
    severity_accuracy: float
    confirm_rate: float
    unknown_rate: float
    correction_rate: float
    investigation_false_positive: float
    investigation_false_negative: float
    max_delay_weeks: int = 2

    def __post_init__(self) -> None:
        for field_name in (
            "cause_accuracy",
            "origin_accuracy",
            "severity_accuracy",
            "confirm_rate",
            "unknown_rate",
            "correction_rate",
            "investigation_false_positive",
            "investigation_false_negative",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.max_delay_weeks < 0:
            raise ValueError("max_delay_weeks must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES: dict[str, AnalystProfile] = {
    "perfect": AnalystProfile("perfect", 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0),
    "careful": AnalystProfile("careful", 0.92, 0.95, 0.9, 1.0, 0.02, 0.05, 0.02, 0.02, 1),
    "typical": AnalystProfile("typical", 0.8, 0.85, 0.75, 0.85, 0.1, 0.1, 0.05, 0.08, 2),
    "sloppy": AnalystProfile("sloppy", 0.55, 0.7, 0.55, 0.6, 0.2, 0.15, 0.1, 0.15, 3),
}


def _perturb(
    rng: np.random.Generator, value: str, options: tuple[str, ...], accuracy: float
) -> str:
    if rng.random() < accuracy:
        return value
    others = [option for option in options if option != value]
    if not others:
        return value
    return others[int(rng.integers(len(others)))]


def _created_date(week: int, delay: int) -> str:
    return (EPOCH + _dt.timedelta(days=7 * max(0, week - 1) + 7 * delay)).isoformat()


def simulate_analyst(
    suite_dir: str | Path,
    store_path: str | Path,
    profile: str | AnalystProfile = "typical",
    seed: int = 0,
    oracle_dir: str | Path | None = None,
) -> dict[str, Any]:
    from qcgen.oracle_vault import OracleVault, default_oracle_root
    from qcgen.sources import ScenarioSource

    from .run import run_qc

    profile = PROFILES[profile] if isinstance(profile, str) else profile
    rng = np.random.default_rng(seed)
    suite_dir = Path(suite_dir)
    suite = json.loads((suite_dir / "suite.json").read_text())
    vault = OracleVault(oracle_dir or default_oracle_root(suite_dir))
    summary: dict[str, Any] = {
        "profile": profile.name,
        "store": str(store_path),
        "runs": 0,
        "confirmed": 0,
        "drafts": 0,
        "corrections": 0,
        "unknown": 0,
        "investigation_flips": 0,
    }

    with SqliteStore(store_path) as store:
        for entry in suite["scenarios"]:
            scenario_id = entry["scenario_id"]
            oracle_payload = vault.require(scenario_id)
            scenario_dir = suite_dir / scenario_id
            manifest = json.loads((scenario_dir / "manifest.json").read_text())
            run_id = f"{suite.get('suite_id')}:{scenario_id}"
            result = run_qc(
                ScenarioSource(scenario_dir),
                manifest["current_version"],
                manifest["previous_version"],
                run_id=run_id,
            )
            store.record_result(result)
            oracle = oracle_labels_for_result(result, oracle_payload).labels
            week = int(manifest["n_current_weeks"])

            cause = _perturb(
                rng, oracle["likely_cause"], CAUSE_VALUES, profile.cause_accuracy
            )
            if rng.random() < profile.unknown_rate:
                cause = "UNKNOWN"
            origin = _perturb(
                rng, oracle["likely_origin"], ORIGIN_VALUES, profile.origin_accuracy
            )
            severity = _perturb(
                rng, oracle["severity"], SEVERITY_VALUES, profile.severity_accuracy
            )

            requires = oracle["requires_investigation"] == "True"
            if requires and rng.random() < profile.investigation_false_negative:
                requires = False
                summary["investigation_flips"] += 1
            elif not requires and rng.random() < profile.investigation_false_positive:
                requires = True
                summary["investigation_flips"] += 1

            confirmed = bool(rng.random() < profile.confirm_rate)
            delay = int(rng.integers(0, profile.max_delay_weeks + 1))
            templates = NOTE_TEMPLATES.get(cause, NOTE_TEMPLATES["UNKNOWN"])
            note = templates[int(rng.integers(len(templates)))]
            store.record_outcome(
                run_id,
                root_cause=cause,
                confirmed=confirmed,
                likely_origin=origin,
                severity=severity,
                resolution=note,
                summary=(
                    f"Confirmed {cause.lower().replace('_', ' ')}."
                    if confirmed
                    else ""
                ),
                analyst=f"synthetic-{profile.name}",
                # Tags describe the simulated analyst's own conclusion, not
                # the engine's decisions, so stored tags cannot contradict the
                # recorded root cause.
                symptom_tags=sorted({cause, origin, f"synthetic-{profile.name}"}),
                requires_investigation=requires,
                provenance="synthetic",
                created=_created_date(week, delay),
            )
            summary["runs"] += 1
            summary["confirmed"] += int(confirmed)
            summary["drafts"] += int(not confirmed)
            summary["unknown"] += int(cause == "UNKNOWN")

            if rng.random() < profile.correction_rate:
                corrected_note = NOTE_TEMPLATES.get(
                    oracle["likely_cause"], NOTE_TEMPLATES["UNKNOWN"]
                )[0]
                store.record_outcome(
                    run_id,
                    root_cause=oracle["likely_cause"],
                    confirmed=True,
                    likely_origin=oracle["likely_origin"],
                    severity=oracle["severity"],
                    resolution=f"{corrected_note} (corrected after review)",
                    summary="Corrected after review.",
                    analyst=f"synthetic-{profile.name}",
                    requires_investigation=oracle["requires_investigation"] == "True",
                    provenance="synthetic",
                    created=_created_date(week, delay + 1),
                )
                summary["corrections"] += 1

    return summary
