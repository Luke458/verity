"""Labels for the semantic layer.

Oracle labels come from synthetic fault manifests; analyst labels come from
shadow-mode feedback. Both are stored as versioned feature vectors plus
canonical field values so a decision provider can be retrained without
re-reading snapshots. Synthetic labels validate plumbing only: they are not
evidence of semantic accuracy on real refresh behaviour.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .decisions import FeatureEncoder
from .evidence_text import evidence_text
from .fingerprints import manifest_data_fingerprint

CAUSE_BY_ORACLE: dict[str, str] = {
    "missing_stores": "MISSING_STORES",
    "missing_products": "MISSING_PRODUCTS",
    "entity_merge": "ENTITY_MERGE",
    "schema_failure": "SCHEMA_FAILURE",
    "null_duplicate_storm": "SCHEMA_FAILURE",
    "coding": "CODING",
    "warehouse": "WAREHOUSE",
    "market_movement": "MARKET_MOVEMENT",
    "historical_correction": "HISTORICAL_CORRECTION",
    "reclassification": "RECLASSIFICATION",
    "backfill": "BACKFILL",
    "truncation": "HISTORICAL_CORRECTION",
}

ORIGIN_BY_ORACLE: dict[str, str] = {
    "source": "SOURCE",
    "preprocessing": "PREPROCESSING",
    "coded": "CODING",
    "warehouse": "WAREHOUSE",
    "report": "REPORT",
    "none": "UNKNOWN",
    "": "UNKNOWN",
}

# Oracle severity policy for synthetic labels. Real severity is an analyst
# decision recorded through shadow feedback.
SEVERITY_BY_FAMILY: dict[str, str] = {
    "missing_stores": "HIGH",
    "missing_products": "HIGH",
    "entity_merge": "MEDIUM",
    "coding_error": "HIGH",
    "warehouse_transform_error": "HIGH",
    "schema_failure": "HIGH",
    "null_duplicate_storm": "HIGH",
    "new_store_backfill": "MEDIUM",
    "history_truncation": "MEDIUM",
    "commodity_remap": "MEDIUM",
    "expected_event": "LOW",
    "recalculation": "LOW",
    "market_movement": "LOW",
}


@dataclass
class LabelRecord:
    run_id: str
    source: str
    family: str | None
    labels: dict[str, str]
    features: list[float]
    feature_version: int
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabelRecord:
        return cls(
            run_id=str(data["run_id"]),
            source=str(data["source"]),
            family=data.get("family"),
            labels={str(k): str(v) for k, v in data["labels"].items()},
            features=[float(value) for value in data["features"]],
            feature_version=int(data["feature_version"]),
            metadata=dict(data.get("metadata", {})),
            text=data.get("text"),
        )


class LabelStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, records: Sequence[LabelRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def load(self) -> list[LabelRecord]:
        if not self.path.exists():
            return []
        return [
            LabelRecord.from_dict(json.loads(line))
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.load())


def oracle_labels_for_result(result: Any, oracle: dict) -> LabelRecord:
    """Oracle labels from a vault payload (never from the scenario manifest)."""
    case: dict = oracle.get("cases", [{}])[0] if oracle.get("cases") else {}
    family = oracle.get("family")
    expected_class = str(case.get("expected_class", "unknown"))
    expected_origin = str(case.get("expected_origin") or "unknown").lower()
    expected_status = str(case.get("expected_status", "PASS"))
    requires = expected_status in ("INVESTIGATE", "DATA_CONTRACT_FAILURE")
    labels = {
        "likely_cause": CAUSE_BY_ORACLE.get(expected_class, "UNKNOWN"),
        "likely_origin": ORIGIN_BY_ORACLE.get(expected_origin, "UNKNOWN"),
        "severity": SEVERITY_BY_FAMILY.get(str(family), "MEDIUM"),
        "requires_investigation": "True" if requires else "False",
    }
    encoder = FeatureEncoder()
    return LabelRecord(
        run_id=result.run_id,
        source="oracle",
        family=family,
        labels=labels,
        features=encoder.encode(result).tolist(),
        feature_version=encoder.feature_version,
        metadata={
            "scenario_id": oracle.get("scenario_id"),
            "expected_status": expected_status,
            "expected_class": expected_class,
        },
        text=evidence_text(result),
    )


def build_oracle_labels(
    suite_dir: str | Path,
    config: DatasetConfig | None = None,
    scenario_ids: Sequence[str] | None = None,
    oracle_dir: str | Path | None = None,
) -> list[LabelRecord]:
    from qcgen.oracle_vault import OracleVault, default_oracle_root
    from qcgen.sources import ScenarioSource

    from .run import run_qc

    suite_dir = Path(suite_dir)
    config = config or DatasetConfig()
    suite = json.loads((suite_dir / "suite.json").read_text())
    vault = OracleVault(oracle_dir or default_oracle_root(suite_dir))
    selected = set(scenario_ids) if scenario_ids else None
    records: list[LabelRecord] = []
    for entry in suite["scenarios"]:
        scenario_id = entry["scenario_id"]
        if selected is not None and scenario_id not in selected:
            continue
        oracle = vault.require(scenario_id)
        scenario_dir = suite_dir / scenario_id
        manifest = json.loads((scenario_dir / "manifest.json").read_text())
        result = run_qc(
            ScenarioSource(scenario_dir),
            manifest["current_version"],
            manifest["previous_version"],
            config,
        )
        record = oracle_labels_for_result(result, oracle)
        record.metadata["suite_id"] = suite.get("suite_id")
        record.metadata["data_fingerprint"] = manifest_data_fingerprint(manifest)
        records.append(record)
    return records


def records_from_store(
    path: str | Path,
    dataset: str | None = None,
) -> list[LabelRecord]:
    """Confirmed-only analyst labels from the durable store.

    Only runs whose latest outcome is confirmed are returned, so drafts and
    model guesses never become training data. The provenance recorded with the
    outcome is authoritative and is never overridden by the caller: a
    ``synthetic`` or unknown outcome can never be relabelled ``analyst``. The
    `requires_investigation` label uses the explicit outcome field when set,
    otherwise it is derived (a confirmed non-unknown root cause implies an
    investigation happened).
    """
    from .store import SqliteStore

    records: list[LabelRecord] = []
    with SqliteStore(path) as store:
        for row in store.confirmed_runs():
            if dataset is not None and row["dataset"] != dataset:
                continue
            if not row.get("features"):
                continue
            record_source = str(row.get("provenance") or "unknown")
            requires = row.get("requires_investigation")
            if requires is None:
                requires_label = (
                    "True"
                    if str(row.get("root_cause") or "UNKNOWN") != "UNKNOWN"
                    else "False"
                )
            else:
                requires_label = "True" if bool(requires) else "False"
            records.append(
                LabelRecord(
                    run_id=str(row["run_id"]),
                    source=record_source,
                    family=None,
                    labels={
                        "likely_cause": str(row.get("root_cause") or "UNKNOWN"),
                        "likely_origin": str(row.get("likely_origin") or "UNKNOWN"),
                        "severity": str(row.get("severity") or "MEDIUM"),
                        "requires_investigation": requires_label,
                    },
                    features=[float(value) for value in json.loads(row["features"])],
                    feature_version=int(row.get("feature_version") or 1),
                    metadata={"dataset": row["dataset"]},
                    text=row.get("evidence_text"),
                )
            )
    return records
