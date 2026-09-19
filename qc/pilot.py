"""Real-data pilot readiness gate.

The pilot cannot be declared ready by configuration alone: it needs confirmed
analyst outcomes in the durable store, a pinned cohort plan, and no synthetic
provenance in the pilot database. This module checks those prerequisites so
"production eligible" is a measured state, not an intention.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PilotReadiness:
    status: str
    blockers: list[str] = field(default_factory=list)
    analyst_labels: int = 0
    synthetic_labels: int = 0
    unknown_labels: int = 0
    plan_path: str | None = None
    plan_hash_verified: bool = False
    min_analyst_labels: int = 10
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.status == "READY"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pilot_readiness(
    store_path: str | Path,
    plan_path: str | Path | None = None,
    min_analyst_labels: int = 10,
) -> PilotReadiness:
    from .cohort import CohortPlan
    from .store import SqliteStore

    blockers: list[str] = []
    analyst = synthetic = unknown = 0
    with SqliteStore(store_path) as store:
        for row in store.confirmed_runs():
            provenance = str(row.get("provenance") or "unknown")
            if provenance == "analyst":
                analyst += 1
            elif provenance == "synthetic":
                synthetic += 1
            else:
                unknown += 1

    if analyst < min_analyst_labels:
        blockers.append(
            f"need at least {min_analyst_labels} confirmed analyst outcomes; "
            f"have {analyst}"
        )
    if synthetic:
        blockers.append(
            f"{synthetic} synthetic outcomes are present; use a separate "
            "pilot database with real analyst provenance only"
        )
    if unknown:
        blockers.append(
            f"{unknown} confirmed outcomes have unknown provenance; set "
            "provenance explicitly at import"
        )

    plan_hash_verified = False
    if plan_path is None:
        blockers.append("no pinned cohort plan supplied (--plan)")
    else:
        plan_file = Path(plan_path)
        if not plan_file.exists():
            blockers.append(f"cohort plan {plan_file} does not exist")
        else:
            plan = CohortPlan.from_json(plan_file)
            plan_sha = hashlib.sha256(
                json.dumps(plan.to_dict(), sort_keys=True).encode()
            ).hexdigest()
            digest_path = plan_file.with_suffix(plan_file.suffix + ".sha256")
            if not digest_path.exists():
                blockers.append(
                    f"cohort plan is not pinned: {digest_path} is missing"
                )
            else:
                expected = digest_path.read_text().strip().split()[0]
                if expected != plan_sha:
                    blockers.append(
                        "cohort plan hash does not match the committed digest; "
                        "the plan changed after pre-registration"
                    )
                else:
                    plan_hash_verified = True

    status = "READY" if not blockers else "NOT_READY"
    return PilotReadiness(
        status=status,
        blockers=blockers,
        analyst_labels=analyst,
        synthetic_labels=synthetic,
        unknown_labels=unknown,
        plan_path=str(plan_path) if plan_path is not None else None,
        plan_hash_verified=plan_hash_verified,
        min_analyst_labels=min_analyst_labels,
        detail=(
            "pilot prerequisites satisfied"
            if not blockers
            else "; ".join(blockers)
        ),
    )
