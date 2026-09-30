"""Run the pre-registered distribution-drift gate.

Implements the gate frozen in ``config/drift-gate.json`` *before* the result is
known. For every scenario in a suite the harness builds two arms:

* **control** - the scenario exactly as generated;
* **fault** - the same scenario with an invariant-preserving value
  redistribution injected into the appended period. Row count, entity set, null
  count and every column sum are preserved by construction, so the
  deterministic evidence layer (structural fingerprint, attribution,
  reconciliation, lineage) has nothing to key on and the engine-only arm is
  expected to be blind.

The gate passes only if the drift check fires on the fault arm more often than
the registered minimum, stays quiet on the control arm at or below the
registered ceiling, and separates the two arms in every scenario.

Usage::

    .venv/bin/python -m optional.drift_benchmark --suite data/suites/demo \\
        --out reports/benchmarks/drift.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from qc.config import DatasetConfig
from qc.distribution_drift import (
    DEFAULT_ABS_FLOOR,
    DEFAULT_BINS,
    DEFAULT_MIN_REFERENCE_WEEKS,
    DEFAULT_REFERENCE_WEEKS,
    DEFAULT_THRESHOLD_QUANTILE,
    assess_distribution_drift,
)

GATE_PATH = Path("config/drift-gate.json")
STAGE = "source"


@dataclass
class ScenarioArm:
    """Drift outcome for one arm of one scenario."""

    scenario_id: str
    family: str
    arm: str
    evaluated: bool
    drifted_scopes: list[str] = field(default_factory=list)
    psi: float | None = None
    threshold: float | None = None
    invariants_preserved: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_gate() -> dict[str, Any]:
    return json.loads(GATE_PATH.read_text())


def inject_redistribution(
    frame: pd.DataFrame,
    config: DatasetConfig,
    period: int,
    fraction: float,
) -> tuple[pd.DataFrame, bool]:
    """Move ``fraction`` of the target half's value to an equal-sized other half.

    The total is preserved exactly, so this is invisible to any check that
    reads sums, counts or entity sets. Returns the new frame and whether the
    invariant actually held.
    """
    week = config.week_column
    metric = config.primary_metric
    out = frame.copy()
    period_rows = out.index[out[week] == period]
    if len(period_rows) < 10:
        return out, True
    period_frame = out.loc[period_rows]
    entity = config.entity_key_columns[0] if config.entity_key_columns else None
    if entity and entity in period_frame.columns:
        ranked = sorted(period_frame[entity].dropna().unique().tolist())
        give = set(ranked[: max(1, len(ranked) // 2)])
    else:
        give = set(period_frame.index[: max(1, len(period_frame) // 2)])
    donate = [row for row in period_rows if out.at[row, entity] in give] if entity else list(period_rows[: len(period_rows) // 2])
    receive = [row for row in period_rows if row not in set(donate)]
    if not donate or not receive:
        return out, True
    moved = float(out.loc[donate, metric].sum()) * fraction
    per_out = moved / len(donate)
    per_in = moved / len(receive)
    out.loc[donate, metric] = out.loc[donate, metric].to_numpy() - per_out
    out.loc[receive, metric] = out.loc[receive, metric].to_numpy() + per_in
    before = float(frame.loc[period_rows, metric].sum())
    after = float(out.loc[period_rows, metric].sum())
    scale = max(1.0, abs(before))
    preserved = abs(after - before) / scale < 1e-6
    return out, preserved


def _arm(
    scenario_dir: Path,
    scenario_id: str,
    family: str,
    arm: str,
    config: DatasetConfig,
    fraction: float,
) -> ScenarioArm:
    path = scenario_dir / "versions" / "V0002" / STAGE / "fact.parquet"
    if not path.exists():
        return ScenarioArm(scenario_id, family, arm, False, note="missing fact parquet")
    frame = pd.read_parquet(path)
    period = int(frame[config.week_column].max())
    preserved = True
    if arm == "fault":
        frame, preserved = inject_redistribution(frame, config, period, fraction)
    result = assess_distribution_drift(
        frame,
        config,
        period,
        bins=DEFAULT_BINS,
        reference_weeks=DEFAULT_REFERENCE_WEEKS,
        abs_floor=DEFAULT_ABS_FLOOR,
        threshold_quantile=DEFAULT_THRESHOLD_QUANTILE,
        min_reference_weeks=DEFAULT_MIN_REFERENCE_WEEKS,
    )
    national = next(
        (scope for scope in result.scopes if scope.scope_type == "national"), None
    )
    return ScenarioArm(
        scenario_id=scenario_id,
        family=family,
        arm=arm,
        evaluated=result.evaluated,
        drifted_scopes=[scope.scope for scope in result.drifted],
        psi=national.psi if national else None,
        threshold=national.threshold if national else None,
        invariants_preserved=preserved,
        note=national.detail if national else "",
    )


def _family(suite_dir: Path, oracle_root: Path, scenario_id: str) -> str:
    for candidate in (
        oracle_root / f"{scenario_id}.json",
        oracle_root / suite_dir.name / f"{scenario_id}.json",
    ):
        if candidate.exists():
            payload = json.loads(candidate.read_text())
            cases = payload.get("cases") or [{}]
            return str(
                payload.get("fault_family")
                or payload.get("family")
                or cases[0].get("family")
                or "unknown"
            )
    return "unknown"


def run_benchmark(
    suite_dir: Path,
    oracle_root: Path | None = None,
    config: DatasetConfig | None = None,
    fraction: float = 0.25,
) -> dict[str, Any]:
    """Execute the pre-registered gate over a suite."""
    gate = load_gate()
    config = config or DatasetConfig()
    oracle_root = oracle_root or suite_dir.parent / "oracle"
    scenarios = sorted(
        path for path in suite_dir.iterdir() if path.is_dir() and path.name.startswith("scenario-")
    )
    arms: list[ScenarioArm] = []
    for scenario_dir in scenarios:
        family = _family(suite_dir, oracle_root, scenario_dir.name)
        arms.append(_arm(scenario_dir, scenario_dir.name, family, "control", config, fraction))
        arms.append(_arm(scenario_dir, scenario_dir.name, family, "fault", config, fraction))

    controls = [arm for arm in arms if arm.arm == "control"]
    faults = [arm for arm in arms if arm.arm == "fault"]
    control_alarms = [arm for arm in controls if arm.drifted_scopes]
    fault_detections = [arm for arm in faults if arm.drifted_scopes]
    pairs = [
        (control, fault)
        for control, fault in zip(controls, faults, strict=True)
        if control.scenario_id == fault.scenario_id
    ]
    separated = [
        pair
        for pair in pairs
        if pair[0].psi is not None
        and pair[1].psi is not None
        and pair[0].psi < pair[1].psi
    ]

    detection_rate = len(fault_detections) / len(faults) if faults else 0.0
    control_rate = len(control_alarms) / len(controls) if controls else 0.0
    separation = len(separated) / len(pairs) if pairs else 0.0
    invariant_failures = [arm.scenario_id for arm in arms if not arm.invariants_preserved]

    passed = (
        detection_rate >= float(gate["min_value"])
        and control_rate <= float(gate["max_control_alarm_rate"])
        and separation == 1.0
        and not invariant_failures
    )
    return {
        "gate": gate,
        "suite": str(suite_dir),
        "fraction": fraction,
        "scenarios": len(pairs),
        "blind_fault_detection_rate": detection_rate,
        "control_alarm_rate": control_rate,
        "paired_separation_rate": separation,
        "invariant_failures": invariant_failures,
        "control_alarms": [arm.scenario_id for arm in control_alarms],
        "fault_misses": [arm.scenario_id for arm in faults if not arm.drifted_scopes],
        "arms": [arm.to_dict() for arm in arms],
        "passed": passed,
        "claims_limit": gate.get("claims_limit", ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--oracle-root", default=None)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    result = run_benchmark(
        Path(args.suite),
        oracle_root=Path(args.oracle_root) if args.oracle_root else None,
        fraction=args.fraction,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    verdict = "PASS" if result["passed"] else "FAIL"
    print(
        f"drift gate verdict={verdict} "
        f"blind_fault_detection_rate={result['blind_fault_detection_rate']:.3f} "
        f"control_alarm_rate={result['control_alarm_rate']:.3f} "
        f"paired_separation={result['paired_separation_rate']:.3f}"
    )
    if result["control_alarms"]:
        print(f"  control alarms: {result['control_alarms']}")
    if result["fault_misses"]:
        print(f"  fault misses: {result['fault_misses']}")
    if result["invariant_failures"]:
        print(f"  INVARIANT FAILURES: {result['invariant_failures']}")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
