"""Pinned forecast qualification for automatic statistical clearance (schema 2).

A qualification artifact names the model/metric/level/horizon combinations that
demonstrated independent held-out forecast coverage and bounded interval width
on development incident groups and were then validated on untouched test
groups, with the common detection, false-positive and false-clearance gates
already passed. Every entry carries its source assessment, incident group,
observation cutoff, split, model artifact identity and calendar/configuration
hashes; duplicate observations, overlapping splits, unsupported provenance and
missing evaluation references are rejected.

Coverage that fitted the interval is a calibration diagnostic and can never
serve as qualification evidence. Missing or unqualified artifacts prevent
automatic clearance. Synthetic qualification may only authorize synthetic
assessments: real qualification additionally requires linked analyst-labelled
evaluation evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from .conformal import wilson_interval

QUALIFICATION_SCHEMA = 2
DEFAULT_REQUIRED_COVERAGE = 0.8
DEFAULT_MAX_WIDTH_RATIO = 0.25
# Registered confidence-bound requirement: at this group count the Wilson
# lower bound of an all-covered split is at or above 0.8. Sufficiency is
# never a development row count plus one test observation.
DEFAULT_MINIMUM_GROUPS = 9
PROVENANCE_VALUES = ("synthetic", "real")
INDEPENDENT_COVERAGE_BASES = ("held_out_target",)
REQUIRED_GATES = ("detection_rate", "false_positive_rate", "false_clearance")


@dataclass(frozen=True)
class QualifiedCombination:
    model: str
    metric: str
    level: str
    horizon: int
    development_groups: int
    test_groups: int
    development_coverage: float
    test_coverage: float
    development_lower_bound: float
    test_lower_bound: float
    width_ratio: float
    qualified: bool = False
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "metric": self.metric,
            "level": self.level,
            "horizon": self.horizon,
            "development_groups": self.development_groups,
            "test_groups": self.test_groups,
            "development_coverage": self.development_coverage,
            "test_coverage": self.test_coverage,
            "development_lower_bound": self.development_lower_bound,
            "test_lower_bound": self.test_lower_bound,
            "width_ratio": self.width_ratio,
            "qualified": self.qualified,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QualifiedCombination:
        return cls(
            model=str(data.get("model", "")),
            metric=str(data.get("metric", "")),
            level=str(data.get("level", "")),
            horizon=int(data.get("horizon", 0)),
            development_groups=int(data.get("development_groups", 0)),
            test_groups=int(data.get("test_groups", 0)),
            development_coverage=float(data.get("development_coverage", 0.0)),
            test_coverage=float(data.get("test_coverage", 0.0)),
            development_lower_bound=float(
                data.get("development_lower_bound", 0.0)
            ),
            test_lower_bound=float(data.get("test_lower_bound", 0.0)),
            width_ratio=float(data.get("width_ratio", 1.0)),
            qualified=bool(data.get("qualified", False)),
            reasons=tuple(str(value) for value in data.get("reasons", [])),
        )


@dataclass(frozen=True)
class QualificationArtifact:
    provenance: str
    required_coverage: float
    max_width_ratio: float
    minimum_groups: int
    combinations: tuple[QualifiedCombination, ...]
    development_digest: str
    test_digest: str
    status: str = "UNQUALIFIED"
    policy_version: str = "3"
    schema_version: int = QUALIFICATION_SCHEMA
    gates: dict[str, Any] = field(default_factory=dict)
    engine: str = ""
    config_hash: str = ""
    calendar_hash: str = ""
    evidence_policy_version: str = ""
    label_source: str = ""
    label_digest: str = ""
    evaluation_reference: str = ""
    created: str = ""
    calibration_diagnostics: dict[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "status": self.status,
            "provenance": self.provenance,
            "required_coverage": self.required_coverage,
            "max_width_ratio": self.max_width_ratio,
            "minimum_groups": self.minimum_groups,
            "combinations": [item.to_dict() for item in self.combinations],
            "development_digest": self.development_digest,
            "test_digest": self.test_digest,
            "policy_version": self.policy_version,
            "gates": dict(self.gates),
            "engine": self.engine,
            "config_hash": self.config_hash,
            "calendar_hash": self.calendar_hash,
            "evidence_policy_version": self.evidence_policy_version,
            "label_source": self.label_source,
            "label_digest": self.label_digest,
            "evaluation_reference": self.evaluation_reference,
            "created": self.created,
            "calibration_diagnostics": dict(self.calibration_diagnostics),
            "reasons": list(self.reasons),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @property
    def digest(self) -> str:
        body = json.dumps(
            self.to_dict(include_digest=False),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(body.encode()).hexdigest()

    def supports(
        self, model: str, metric: str, level: str, horizon: int
    ) -> bool:
        if self.status != "QUALIFIED":
            return False
        return any(
            item.qualified
            and item.model == model
            and item.metric == metric
            and item.level == level
            and item.horizon == int(horizon)
            for item in self.combinations
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QualificationArtifact:
        return cls(
            provenance=str(data.get("provenance", "")),
            required_coverage=float(
                data.get("required_coverage", DEFAULT_REQUIRED_COVERAGE)
            ),
            max_width_ratio=float(
                data.get("max_width_ratio", DEFAULT_MAX_WIDTH_RATIO)
            ),
            minimum_groups=int(
                data.get("minimum_groups", DEFAULT_MINIMUM_GROUPS)
            ),
            combinations=tuple(
                QualifiedCombination.from_dict(item)
                for item in data.get("combinations", [])
            ),
            development_digest=str(data.get("development_digest", "")),
            test_digest=str(data.get("test_digest", "")),
            status=str(data.get("status", "UNQUALIFIED")),
            policy_version=str(data.get("policy_version", "")),
            schema_version=int(data.get("schema_version", 0)),
            gates=dict(data.get("gates", {})),
            engine=str(data.get("engine", "")),
            config_hash=str(data.get("config_hash", "")),
            calendar_hash=str(data.get("calendar_hash", "")),
            evidence_policy_version=str(
                data.get("evidence_policy_version", "")
            ),
            label_source=str(data.get("label_source", "")),
            label_digest=str(data.get("label_digest", "")),
            evaluation_reference=str(data.get("evaluation_reference", "")),
            created=str(data.get("created", "")),
            calibration_diagnostics=dict(
                data.get("calibration_diagnostics", {})
            ),
            reasons=tuple(str(value) for value in data.get("reasons", [])),
        )


def _digest_entries(entries: Iterable[Mapping[str, Any]]) -> str:
    body = json.dumps(
        [dict(item) for item in entries],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(body.encode()).hexdigest()


def _coverage(entries: Sequence[Mapping[str, Any]]) -> float:
    if not entries:
        return 0.0
    return sum(1 for item in entries if item.get("covered")) / len(entries)


def _lower_bound(entries: Sequence[Mapping[str, Any]]) -> float:
    if not entries:
        return 0.0
    covered = sum(1 for item in entries if item.get("covered"))
    lower, _ = wilson_interval(covered, len(entries))
    return float(lower)


def _width_ratio(entries: Sequence[Mapping[str, Any]]) -> float:
    ratios = [float(item["width_ratio"]) for item in entries]
    return float(median(ratios)) if ratios else 1.0


def _entry_key(entry: Mapping[str, Any]) -> tuple[str, str, str, str, str, int]:
    return (
        str(entry.get("model", "")),
        str(entry.get("metric", "")),
        str(entry.get("level", "")),
        str(entry.get("group", "")),
        str(entry.get("assessment_id", "")),
        int(entry.get("horizon", 0)),
    )


def _validate_entries(
    label: str,
    entries: Sequence[Mapping[str, Any]],
    *,
    require_split: str | None = None,
) -> None:
    seen: set[tuple[str, str, str, str, str, int]] = set()
    for entry in entries:
        for field_name in (
            "assessment_id",
            "group",
            "model",
            "metric",
            "level",
            "cutoff",
            "split",
            "basis",
        ):
            if not str(entry.get(field_name, "")).strip():
                raise ValueError(
                    f"{label} qualification entry is missing {field_name!r}"
                )
        basis = str(entry["basis"])
        if basis not in INDEPENDENT_COVERAGE_BASES:
            raise ValueError(
                f"{label} entry coverage basis {basis!r} is not independent"
            )
        if require_split is not None and str(entry["split"]) != require_split:
            raise ValueError(
                f"{label} entry declares split {entry['split']!r}"
            )
        if "width_ratio" not in entry or "covered" not in entry:
            raise ValueError(
                f"{label} entry is missing independent coverage or width"
            )
        width = float(entry["width_ratio"])
        if not 0.0 <= width < float("inf"):
            raise ValueError(f"{label} entry has an invalid width ratio")
        key = _entry_key(entry)
        if key in seen:
            raise ValueError(
                f"duplicate {label} qualification observation {key!r}"
            )
        seen.add(key)


def _gate_reasons(gates: Mapping[str, Any] | None) -> list[str]:
    if not gates:
        return ["registered evaluation gates are missing"]
    reasons: list[str] = []
    if str(gates.get("status", "")) != "PASS":
        reasons.append("registered evaluation gates did not pass")
    nested = gates.get("gates") or {}
    if not isinstance(nested, Mapping):
        return ["registered evaluation gates are malformed"]
    for name in REQUIRED_GATES:
        gate = nested.get(name)
        if not isinstance(gate, Mapping) or str(gate.get("status", "")) != "PASS":
            reasons.append(f"required gate {name!r} did not pass")
    return reasons


def build_qualification(
    development: Iterable[Mapping[str, Any]],
    test: Iterable[Mapping[str, Any]],
    *,
    provenance: str,
    required_coverage: float = DEFAULT_REQUIRED_COVERAGE,
    max_width_ratio: float = DEFAULT_MAX_WIDTH_RATIO,
    minimum_groups: int = DEFAULT_MINIMUM_GROUPS,
    gates: Mapping[str, Any] | None = None,
    label_source: str = "",
    label_digest: str = "",
    evaluation_reference: str = "",
    engine: str = "",
    config_hash: str = "",
    calendar_hash: str = "",
    evidence_policy_version: str = "",
    calibration_diagnostics: Mapping[str, Any] | None = None,
    created: str = "",
) -> QualificationArtifact:
    """Select combinations on development groups and validate on test groups."""
    if provenance not in PROVENANCE_VALUES:
        raise ValueError(f"unsupported provenance {provenance!r}")
    for name, value in (
        ("required_coverage", required_coverage),
        ("max_width_ratio", max_width_ratio),
    ):
        if not 0.0 < float(value) < float("inf"):
            raise ValueError(f"{name} must be registered and positive")
    if int(minimum_groups) < 1:
        raise ValueError("minimum_groups must be registered and positive")
    development = [dict(item) for item in development]
    test = [dict(item) for item in test]
    _validate_entries("development", development, require_split="development")
    _validate_entries("test", test, require_split="test")
    development_groups = {str(item["group"]) for item in development}
    test_groups = {str(item["group"]) for item in test}
    if development_groups & test_groups:
        raise ValueError(
            "development and test qualification groups must be disjoint"
        )
    if provenance == "real":
        if not label_source or not label_digest or not evaluation_reference:
            raise ValueError(
                "real qualification requires linked analyst-labelled evidence"
            )
    gate_reasons = _gate_reasons(gates)
    groups: dict[tuple[str, str, str, int], dict[str, list[dict[str, Any]]]] = {}
    for label, entries in (("development", development), ("test", test)):
        for entry in entries:
            key = (
                str(entry.get("model", "")),
                str(entry.get("metric", "")),
                str(entry.get("level", "")),
                int(entry.get("horizon", 0)),
            )
            groups.setdefault(key, {"development": [], "test": []})[label].append(
                dict(entry)
            )
    combinations: list[QualifiedCombination] = []
    for key, grouped in sorted(groups.items()):
        dev_entries = grouped["development"]
        test_entries = grouped["test"]
        model, metric, level, horizon = key
        reasons: list[str] = []
        if len(dev_entries) < minimum_groups:
            reasons.append(
                f"development groups {len(dev_entries)} < {minimum_groups}"
            )
        if len(test_entries) < minimum_groups:
            reasons.append(f"test groups {len(test_entries)} < {minimum_groups}")
        dev_coverage = _coverage(dev_entries)
        test_coverage = _coverage(test_entries)
        dev_lower = _lower_bound(dev_entries)
        test_lower = _lower_bound(test_entries)
        if dev_lower < required_coverage:
            reasons.append(
                f"development lower bound {dev_lower:.3f} < {required_coverage}"
            )
        if test_lower < required_coverage:
            reasons.append(
                f"test lower bound {test_lower:.3f} < {required_coverage}"
            )
        width_ratio = max(_width_ratio(dev_entries), _width_ratio(test_entries))
        if width_ratio > max_width_ratio:
            reasons.append(
                f"width ratio {width_ratio:.3f} > {max_width_ratio}"
            )
        combinations.append(
            QualifiedCombination(
                model=model,
                metric=metric,
                level=level,
                horizon=horizon,
                development_groups=len(dev_entries),
                test_groups=len(test_entries),
                development_coverage=dev_coverage,
                test_coverage=test_coverage,
                development_lower_bound=dev_lower,
                test_lower_bound=test_lower,
                width_ratio=width_ratio,
                qualified=not reasons and not gate_reasons,
                reasons=tuple(reasons),
            )
        )
    qualified = any(item.qualified for item in combinations) and not gate_reasons
    status = "QUALIFIED" if qualified else "UNQUALIFIED"
    all_reasons = tuple(gate_reasons)
    return QualificationArtifact(
        provenance=provenance,
        required_coverage=float(required_coverage),
        max_width_ratio=float(max_width_ratio),
        minimum_groups=int(minimum_groups),
        combinations=tuple(combinations),
        development_digest=_digest_entries(development),
        test_digest=_digest_entries(test),
        status=status,
        policy_version="3",
        gates=dict(gates or {}),
        engine=str(engine),
        config_hash=str(config_hash),
        calendar_hash=str(calendar_hash),
        evidence_policy_version=str(evidence_policy_version),
        label_source=str(label_source),
        label_digest=str(label_digest),
        evaluation_reference=str(evaluation_reference),
        created=str(created),
        calibration_diagnostics=dict(calibration_diagnostics or {}),
        reasons=all_reasons,
    )


def pin_qualification(
    artifact: QualificationArtifact,
    path: str | Path,
    *,
    created: str | None = None,
) -> str:
    """Write the artifact and its digest sidecar; return the artifact digest.

    Pinning stamps the creation time (unless supplied) so historical
    assessments can refuse artifacts published after their observation cutoff.
    """
    from dataclasses import replace
    from datetime import UTC, datetime

    if not artifact.created:
        artifact = replace(
            artifact,
            created=created or datetime.now(UTC).isoformat(),
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact.to_dict(), indent=2, sort_keys=True)
    target.write_text(payload + "\n")
    digest = artifact.digest
    target.with_name(target.name + ".sha256").write_text(digest + "\n")
    return digest


def load_qualification(path: str | Path) -> QualificationArtifact:
    """Load a pinned qualification, rejecting a missing or altered digest."""
    target = Path(path)
    artifact = QualificationArtifact.from_dict(json.loads(target.read_text()))
    if artifact.schema_version != QUALIFICATION_SCHEMA:
        raise ValueError(
            f"unsupported qualification schema {artifact.schema_version}"
        )
    sidecar = target.with_name(target.name + ".sha256")
    if not sidecar.is_file():
        raise ValueError("qualification digest sidecar is missing")
    expected = sidecar.read_text().strip()
    if not expected or expected != artifact.digest:
        raise ValueError("qualification digest mismatch")
    return artifact
