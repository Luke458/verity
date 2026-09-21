"""Single source of truth for fault-family expectations.

Engine semantics win: ``expected_status``, ``kind``, ``expected_class`` and
``expected_origin`` are declared here and consumed by the injectors, the
generator verifier, label generation and the family tests. When engine
behaviour changes, change this table; never patch only one consumer.

The two historically contradictory cases are resolved here:

* ``null_duplicate_storm`` is a contract failure (the report table is not fit
  for revision QC), not an ordinary INVESTIGATE.
* ``market_movement`` is a control whose latest week is a genuine movement, so
  the engine status is INVESTIGATE (latest-week anomaly) even though the
  historical revision is clean.
"""

from __future__ import annotations

from dataclasses import dataclass

INVESTIGATE_STATUSES = frozenset({"INVESTIGATE", "DATA_CONTRACT_FAILURE"})


@dataclass(frozen=True)
class FamilySpec:
    family: str
    stage: str
    kind: str
    expected_status: str
    expected_class: str
    expected_origin: str | None
    registry_required: bool = False
    description: str = ""

    @property
    def requires_investigation(self) -> bool:
        return self.expected_status in INVESTIGATE_STATUSES


FAMILY_SPECS: dict[str, FamilySpec] = {
    "missing_stores": FamilySpec(
        "missing_stores", "source", "fault", "INVESTIGATE", "missing_stores", "source"
    ),
    "new_store_backfill": FamilySpec(
        "new_store_backfill", "source", "fault", "INVESTIGATE", "backfill", "source"
    ),
    # The registered event explains the historical revision, but every
    # configured measure is now assessed: residual secondary-measure
    # coordinated drift on this onboarding therefore requires review.
    "expected_event": FamilySpec(
        "expected_event",
        "source",
        "expected_event",
        "INVESTIGATE",
        "backfill",
        "source",
        registry_required=True,
    ),
    "history_truncation": FamilySpec(
        "history_truncation", "source", "fault", "INVESTIGATE", "truncation", "source"
    ),
    "commodity_remap": FamilySpec(
        "commodity_remap",
        "source",
        "fault",
        "INVESTIGATE",
        "reclassification",
        "source",
    ),
    "coding_error": FamilySpec(
        "coding_error", "coded", "fault", "INVESTIGATE", "coding", "coded"
    ),
    "warehouse_transform_error": FamilySpec(
        "warehouse_transform_error",
        "warehouse",
        "fault",
        "INVESTIGATE",
        "warehouse",
        "warehouse",
    ),
    "recalculation": FamilySpec(
        "recalculation",
        "source",
        "fault",
        "INVESTIGATE",
        "historical_correction",
        "source",
    ),
    "schema_failure": FamilySpec(
        "schema_failure",
        "report",
        "fault",
        "DATA_CONTRACT_FAILURE",
        "schema_failure",
        "report",
    ),
    "null_duplicate_storm": FamilySpec(
        "null_duplicate_storm",
        "warehouse",
        "fault",
        "DATA_CONTRACT_FAILURE",
        "null_duplicate_storm",
        "warehouse",
    ),
    "market_movement": FamilySpec(
        "market_movement",
        "warehouse",
        "control",
        "INVESTIGATE",
        "market_movement",
        None,
    ),
    "missing_products": FamilySpec(
        "missing_products", "source", "fault", "INVESTIGATE", "missing_products", "source"
    ),
    "entity_merge": FamilySpec(
        "entity_merge", "source", "fault", "INVESTIGATE", "entity_merge", "source"
    ),
}


def spec_for(family: str) -> FamilySpec:
    try:
        return FAMILY_SPECS[family]
    except KeyError as error:
        raise ValueError(f"unknown fault family: {family!r}") from error


def case_expectations(family: str, expected: bool = False) -> dict[str, str | None]:
    """Kind/status/class/origin for an injected case.

    ``expected=True`` applies only to the shared backfill injector and selects
    the registered ``expected_event`` variant of the case.
    """
    if expected:
        spec = spec_for("expected_event")
    else:
        spec = spec_for(family)
    return {
        "kind": spec.kind,
        "expected_status": spec.expected_status,
        "expected_class": spec.expected_class,
        "expected_origin": spec.expected_origin,
    }
