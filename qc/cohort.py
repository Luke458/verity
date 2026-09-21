"""Frozen cohort evaluation with pre-registered gates (architecture §65-68).

Dev and held-out seeds are disjoint and declared before the run; gates are
evaluated on the held-out split only. Every case is written to JSONL, the plan
and the engine source are hashed, and ``production_eligible`` is always False
for synthetic cohorts: this harness exists so real-label cohorts can be run and
judged honestly, not to certify synthetic accuracy.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .conformal import wilson_interval
from .jsonutil import dumps as json_dumps
from .run import run_qc

DEFAULT_FAMILIES: tuple[str, ...] = (
    "missing_stores",
    "missing_products",
    "entity_merge",
    "new_store_backfill",
    "history_truncation",
    "commodity_remap",
    "coding_error",
    "warehouse_transform_error",
    "recalculation",
    "schema_failure",
    "null_duplicate_storm",
)
DEFAULT_CONTROLS: tuple[str, ...] = ("market_movement",)
DEFAULT_GATES: dict[str, float] = {
    "min_detection_rate": 0.9,
    "max_false_positive_rate": 0.1,
    "min_lineage_first_divergence_accuracy": 0.9,
}
KNOWN_GATES: tuple[str, ...] = (
    "min_detection_rate",
    "max_false_positive_rate",
    "min_reconstruction_score",
    "min_lineage_first_divergence_accuracy",
)
LINEAGE_FAMILIES: dict[str, str] = {
    "new_store_backfill": "source",
    "expected_event": "source",
    "history_truncation": "source",
    "entity_merge": "source",
    "coding_error": "coded",
    "warehouse_transform_error": "warehouse",
    "recalculation": "source",
}


@dataclass(frozen=True)
class CohortPlan:
    profile: str = "tiny"
    families: tuple[str, ...] = DEFAULT_FAMILIES
    controls: tuple[str, ...] = DEFAULT_CONTROLS
    scenarios_per_family: int = 1
    dev_seeds: tuple[int, ...] = (1101,)
    heldout_seeds: tuple[int, ...] = (7101, 7102)
    gates: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_GATES)
    )

    def __post_init__(self) -> None:
        if set(self.dev_seeds) & set(self.heldout_seeds):
            raise ValueError("dev and held-out seeds must be disjoint")
        if self.scenarios_per_family < 1:
            raise ValueError("scenarios_per_family must be >= 1")
        unknown = set(self.gates) - set(KNOWN_GATES)
        if unknown:
            raise ValueError(f"unknown gates: {sorted(unknown)}")
        if not self.gates:
            raise ValueError(
                "gates must be declared; an empty gate set passes vacuously"
            )
        missing = set(DEFAULT_GATES) - set(self.gates)
        if missing:
            raise ValueError(
                f"gates must be complete; missing {sorted(missing)}"
            )
        for name, threshold in self.gates.items():
            if name.startswith("min_") and not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            if name.startswith("max_") and not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CohortPlan:
        return cls(
            profile=str(data.get("profile", "tiny")),
            families=tuple(data.get("families", DEFAULT_FAMILIES)),
            controls=tuple(data.get("controls", DEFAULT_CONTROLS)),
            scenarios_per_family=int(data.get("scenarios_per_family", 1)),
            dev_seeds=tuple(int(value) for value in data.get("dev_seeds", (1101,))),
            heldout_seeds=tuple(
                int(value) for value in data.get("heldout_seeds", (7101, 7102))
            ),
            gates=dict(data.get("gates", DEFAULT_GATES)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> CohortPlan:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json_dumps(self.to_dict(), indent=2, sort_keys=True)
        )


@dataclass
class CohortCase:
    case_id: str
    split: str
    seed: int
    family: str
    is_control: bool
    engine_status: str
    historical_status: str | None
    expected_status: str | None
    expected_class: str | None
    injection_stage: str | None
    first_divergence: str | None
    reconstruction_score: float | None
    explained_fraction: float
    detected: bool
    expected_match: bool
    false_positive: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CohortResult:
    plan: dict[str, Any]
    metrics: dict[str, Any]
    gate_results: list[dict[str, Any]]
    gates_passed: bool
    code_sha256: str
    plan_sha256: str
    plan_path: str | None = None
    plan_hash_verified: bool = False
    git_dirty: bool | None = None
    production_eligible: bool = False
    limitations: list[str] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def code_sha256(root: str | Path | None = None) -> str:
    """Hash the engine, generator and dataset configs recursively."""
    base = Path(root or Path(__file__).resolve().parents[1])
    digest = hashlib.sha256()
    for folder in ("qc", "qcgen", "config", "optional"):
        directory = base / folder
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(base)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def git_dirty(root: str | Path | None = None) -> bool | None:
    """True when the working tree has uncommitted changes; None if unknown."""
    base = Path(root or Path(__file__).resolve().parents[1])
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=base,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _metrics(cases: list[CohortCase]) -> dict[str, Any]:
    faults = [case for case in cases if not case.is_control]
    controls = [case for case in cases if case.is_control]
    comparable = [
        case
        for case in faults
        if case.family in LINEAGE_FAMILIES and case.injection_stage is not None
    ]
    reconstruction = [
        case.reconstruction_score
        for case in faults
        if case.reconstruction_score is not None
    ]
    expected_contracts = [
        case for case in faults if case.family in ("schema_failure", "null_duplicate_storm")
    ]
    detected = [case.detected for case in faults]
    false_positives = [case.detected for case in controls]
    detection_ci = wilson_interval(sum(detected), len(detected))
    fpr_ci = wilson_interval(sum(false_positives), len(false_positives))
    control_status_counts: dict[str, int] = {}
    for case in controls:
        control_status_counts[case.engine_status] = (
            control_status_counts.get(case.engine_status, 0) + 1
        )
    return {
        "cases": len(cases),
        "fault_cases": len(faults),
        "control_cases": len(controls),
        "undefined_reason": "No observations in the corresponding subset" if not controls or not faults or not reconstruction else None,
        "detection_rate": _rate(detected),
        "detection_rate_ci_low": detection_ci[0],
        "detection_rate_ci_high": detection_ci[1],
        "false_positive_rate": _rate(false_positives),
        "false_positive_rate_ci_low": fpr_ci[0],
        "false_positive_rate_ci_high": fpr_ci[1],
        "control_status_counts": control_status_counts,
        "expected_status_match_rate": _rate(
            [
                case.engine_status == case.expected_status
                for case in cases
                if case.expected_status is not None
            ]
        ),
        "contract_failure_rate": _rate(
            [case.engine_status == "DATA_CONTRACT_FAILURE" for case in expected_contracts]
        ),
        "mean_reconstruction_score": (
            sum(reconstruction) / len(reconstruction) if reconstruction else None
        ),
        "lineage_first_divergence_accuracy": _rate(
            [
                case.first_divergence == LINEAGE_FAMILIES[case.family]
                for case in comparable
            ]
        ),
        "lineage_comparable": len(comparable),
    }


def _evaluation_cases(cases: Sequence[CohortCase]) -> list[Any]:
    """Map cohort cases onto the one common evaluation-case contract."""
    from .evaluation import EvaluationCase

    return [
        EvaluationCase(
            case_id=case.case_id,
            family=case.family,
            group=case.case_id,
            actionable=not case.is_control,
            verifier_status=case.engine_status,
            challenger_status=case.engine_status,
            effective_status=case.engine_status,
            verifier_review=case.detected,
            challenger_review=case.detected,
            effective_review=case.detected,
            expected_review=not case.is_control,
        )
        for case in cases
    ]


def _common_gate_results(
    cases: Sequence[CohortCase], gates: dict[str, float]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The one gate implementation shared by every evaluation path."""
    from .evaluation import evaluation_gates

    common = evaluation_gates(
        _evaluation_cases(cases),
        minimum_detection_rate=gates["min_detection_rate"],
        maximum_false_positive_rate=gates["max_false_positive_rate"],
    )
    nested = common["gates"]
    checks = [
        {
            "gate": "min_detection_rate",
            "threshold": gates["min_detection_rate"],
            "actual": nested["detection_rate"]["value"],
            "passed": nested["detection_rate"]["status"] == "PASS",
        },
        {
            "gate": "max_false_positive_rate",
            "threshold": gates["max_false_positive_rate"],
            "actual": nested["false_positive_rate"]["value"],
            "passed": nested["false_positive_rate"]["status"] == "PASS",
        },
        {
            "gate": "false_clearance_upper_bound",
            "threshold": 0.01,
            "actual": nested["false_clearance"]["bound"].get("upper_bound"),
            "passed": nested["false_clearance"]["status"] == "PASS",
        },
        {
            "gate": "independent_label_errors",
            "threshold": 0,
            "actual": len(nested["deterministic_fixture_errors"].get("errors", [])),
            "passed": nested["deterministic_fixture_errors"]["status"] == "PASS",
        },
    ]
    return checks, common


def _gate_results(metrics: dict[str, Any], gates: dict[str, float]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if "min_reconstruction_score" in gates:
        threshold = gates["min_reconstruction_score"]
        checks.append(
            {
                "gate": "min_reconstruction_score",
                "threshold": threshold,
                "actual": metrics["mean_reconstruction_score"],
                "passed": metrics["mean_reconstruction_score"] is not None and metrics["mean_reconstruction_score"] >= threshold,
            }
        )
    if "min_lineage_first_divergence_accuracy" in gates:
        threshold = gates["min_lineage_first_divergence_accuracy"]
        actual = metrics["lineage_first_divergence_accuracy"]
        checks.append(
            {
                "gate": "min_lineage_first_divergence_accuracy",
                "threshold": threshold,
                "actual": actual,
                "passed": (
                    actual >= threshold if metrics["lineage_comparable"] else False
                ),
            }
        )
    return checks


def _verify_plan_hash(plan_path: Path, plan_sha: str) -> bool:
    """Check a committed ``<plan>.sha256`` against the loaded plan."""
    digest_path = plan_path.with_suffix(plan_path.suffix + ".sha256")
    if not digest_path.exists():
        return False
    expected = digest_path.read_text().strip().split()[0]
    if expected != plan_sha:
        raise ValueError(
            f"cohort plan {plan_path} does not match {digest_path}: "
            f"expected {expected}, got {plan_sha}. Changing the plan after "
            "seeing results invalidates the evaluation."
        )
    return True


def run_cohort(
    plan: CohortPlan | None = None,
    workdir: str | Path = "reports/cohort",
    config: DatasetConfig | None = None,
    out_dir: str | Path | None = None,
    plan_path: str | Path | None = None,
) -> CohortResult:
    from qcgen.config import dataset_calendar, suite_config
    from qcgen.scenarios import build_scenario
    from qcgen.sources import ScenarioSource

    plan = plan or CohortPlan()
    config = config or DatasetConfig()
    if not config.calendar_anchor_date:
        config = replace(config, **dataset_calendar(plan.profile))
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    plan_sha = hashlib.sha256(
        json_dumps(plan.to_dict(), sort_keys=True).encode()
    ).hexdigest()
    plan_hash_verified = False
    if plan_path is not None:
        plan_hash_verified = _verify_plan_hash(Path(plan_path), plan_sha)

    cases: list[CohortCase] = []
    splits = (("dev", plan.dev_seeds), ("heldout", plan.heldout_seeds))
    for split, seeds in splits:
        for seed in seeds:
            for family in plan.families + plan.controls:
                is_control = family in plan.controls
                for index in range(plan.scenarios_per_family):
                    case_id = f"{split}-{seed}-{family}-{index}"
                    root = workdir / case_id
                    built = build_scenario(
                        suite_config(plan.profile, seed=seed),
                        index,
                        root,
                        family,
                        ("source", "coded", "warehouse", "report"),
                        oracle_root=workdir / "_oracle",
                    )
                    manifest = built.manifest
                    case = (
                        built.oracle["cases"][0]
                        if built.oracle.get("cases")
                        else {}
                    )
                    run_result = run_qc(
                        ScenarioSource(built.directory),
                        manifest["current_version"],
                        manifest["previous_version"],
                        config,
                    )
                    historical = (
                        run_result.machine["historical_revision"]["status"]
                        if run_result.machine.get("historical_revision")
                        else None
                    )
                    cases.append(
                        CohortCase(
                            case_id=case_id,
                            split=split,
                            seed=seed,
                            family=family,
                            is_control=is_control,
                            engine_status=run_result.status,
                            historical_status=historical,
                            expected_status=case.get("expected_status"),
                            expected_class=case.get("expected_class"),
                            injection_stage=case.get("injection_stage"),
                            first_divergence=(
                                run_result.lineage.first_divergence
                                if run_result.lineage is not None
                                else None
                            ),
                            reconstruction_score=(
                                run_result.counterfactual.reconciliation_score
                                if run_result.counterfactual is not None
                                else None
                            ),
                            explained_fraction=(
                                run_result.attribution.explained_fraction
                                if run_result.attribution is not None
                                else 0.0
                            ),
                            detected=run_result.status not in ("PASS", "PASS_WITH_EXPLANATION"),
                            expected_match=(
                                case.get("expected_status") is not None
                                and run_result.status == case.get("expected_status")
                            ),
                            false_positive=(
                                is_control and run_result.status not in ("PASS", "PASS_WITH_EXPLANATION")
                            ),
                        )
                    )

    heldout = [case for case in cases if case.split == "heldout"]
    dev = [case for case in cases if case.split == "dev"]
    metrics = _metrics(heldout)
    metrics["dev"] = _metrics(dev)
    common_checks, common = _common_gate_results(heldout, plan.gates)
    metrics["common_gates"] = common
    gates = common_checks + _gate_results(metrics, plan.gates)
    cohort_result = CohortResult(
        plan=plan.to_dict(),
        metrics=metrics,
        gate_results=gates,
        gates_passed=all(check["passed"] for check in gates)
        and common["status"] == "PASS",
        code_sha256=code_sha256(),
        plan_sha256=plan_sha,
        plan_path=str(plan_path) if plan_path is not None else None,
        plan_hash_verified=plan_hash_verified,
        git_dirty=git_dirty(),
        limitations=[
            "Synthetic oracle labels validate the harness and the engine's "
            "deterministic semantics, not real refresh accuracy.",
            "Held-out seeds are disjoint from dev seeds but share the same "
            "generator; generator bias is not measured.",
            "Gates are evaluated on the held-out split only; changing the plan "
            "after seeing results invalidates the evaluation.",
        ],
        cases=[case.to_dict() for case in cases],
    )
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "cohort.json").write_text(
            json_dumps(cohort_result.to_dict(), indent=2, sort_keys=True, default=str)
        )
        (out / "cases.jsonl").write_text(
            "\n".join(json_dumps(case.to_dict(), sort_keys=True) for case in cases)
            + "\n"
        )
    return cohort_result
