"""Historical replay with leakage guards (ported from Verity's spine).

A replay plan pins, per case, the previous and current versions and an
``as_of`` assertion supplied by the caller. Version pairs must increase per
source and ``as_of`` must not move backwards, so a replay cannot accidentally
read a later state than the one an analyst saw. Live registries and incident
retrieval are disabled by construction: only the plan's own expected events are
passed to the engine, the store is never queried, and every case records the
machine output it produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import DatasetConfig


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    previous_version: str
    current_version: str
    as_of: str
    uri: str | None = None
    expected_events: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.previous_version == self.current_version:
            raise ValueError(f"{self.case_id}: previous and current are equal")
        if not self.as_of:
            raise ValueError(f"{self.case_id}: as_of is required")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expected_events"] = list(self.expected_events)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayCase":
        return cls(
            case_id=str(data["case_id"]),
            previous_version=str(data["previous_version"]),
            current_version=str(data["current_version"]),
            as_of=str(data["as_of"]),
            uri=data.get("uri"),
            expected_events=tuple(data.get("expected_events", ())),
        )


def _version_key(value: str) -> tuple[int, int] | tuple[int, str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


@dataclass(frozen=True)
class ReplayPlan:
    cases: tuple[ReplayCase, ...]
    caller_assertion: str = (
        "as_of is a caller assertion, not an authenticated time boundary"
    )

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("replay plan has no cases")
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("replay case ids must be unique")
        last_as_of: str | None = None
        last_current: dict[str, str] = {}
        for case in self.cases:
            if last_as_of is not None and case.as_of < last_as_of:
                raise ValueError("as_of must not move backwards")
            last_as_of = case.as_of
            key = case.uri or ""
            previous_current = last_current.get(key)
            if previous_current is not None and _version_key(
                case.previous_version
            ) < _version_key(previous_current):
                raise ValueError(
                    f"versions must increase for {key or 'default source'}: "
                    f"after {previous_current} came {case.previous_version}"
                )
            last_current[key] = case.current_version

    def to_dict(self) -> dict[str, Any]:
        return {
            "caller_assertion": self.caller_assertion,
            "cases": [case.to_dict() for case in self.cases],
            "plan_sha256": hashlib.sha256(
                json.dumps(
                    [case.to_dict() for case in self.cases], sort_keys=True
                ).encode()
            ).hexdigest(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayPlan":
        return cls(cases=tuple(ReplayCase.from_dict(item) for item in data["cases"]))

    @classmethod
    def from_json(cls, path: str | Path) -> "ReplayPlan":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        payload = {
            "cases": [case.to_dict() for case in self.cases],
            "caller_assertion": self.caller_assertion,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def default_source_factory(
    storage_options: dict[str, str] | None = None,
) -> Callable[[str], Any]:
    def build(uri: str) -> Any:
        if (Path(uri) / "versions").exists():
            from qcgen.sources import ScenarioSource

            return ScenarioSource(uri)
        from .delta import DeltaSource

        return DeltaSource(uri=uri, storage_options=storage_options)

    return build


@dataclass
class ReplayReport:
    plan_sha256: str
    cases: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def replay(
    plan: ReplayPlan,
    out_dir: str | Path,
    default_uri: str | None = None,
    config: DatasetConfig | None = None,
    source_factory: Callable[[str], Any] | None = None,
    storage_options: dict[str, str] | None = None,
) -> ReplayReport:
    from .run import run_qc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "cases").mkdir()
    config = config or DatasetConfig()
    factory = source_factory or default_source_factory(storage_options)
    plan_sha = hashlib.sha256(
        json.dumps(
            [case.to_dict() for case in plan.cases], sort_keys=True
        ).encode()
    ).hexdigest()
    report = ReplayReport(plan_sha256=plan_sha)

    for case in plan.cases:
        uri = case.uri or default_uri
        if uri is None:
            report.failed.append(
                {"case_id": case.case_id, "error": "no uri for case"}
            )
            continue
        try:
            source = factory(uri)
            versions = set(source.list_versions())
            for version in (case.previous_version, case.current_version):
                if versions and version not in versions:
                    raise ValueError(f"version {version!r} not present in {uri}")
            result = run_qc(
                source,
                case.current_version,
                case.previous_version,
                config,
                expected_events=list(case.expected_events),
                run_id=case.case_id,
            )
        except Exception as error:  # noqa: BLE001 - recorded, not hidden
            report.failed.append(
                {
                    "case_id": case.case_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        machine_path = out / "cases" / f"{case.case_id}.json"
        machine_path.write_text(
            json.dumps(result.machine, indent=2, sort_keys=True, default=str)
        )
        report.cases.append(
            {
                "case_id": case.case_id,
                "uri": uri,
                "previous_version": case.previous_version,
                "current_version": case.current_version,
                "as_of": case.as_of,
                "status": result.status,
                "machine_path": str(machine_path),
            }
        )

    (out / "replay.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str)
    )
    return report
