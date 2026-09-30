"""Run the pre-registered scope-coverage gate.

Implements the gate frozen in ``config/coverage-gate.json``. Unlike every other
gate in this repository this one is measured on a **real** corpus, because the
defect it addresses was only visible on real data.

Two arms over the real refresh sequence:

* **control** - unmodified real periods. A control alarm is a scope reported as
  coverage-regressed *while its coverage is still inside its own observed
  trailing range*. That is the registered definition: such a flag is
  indistinguishable from noise and would be a false escalation.
* **injected** - real periods with a fraction of a scope's entities removed from
  the appended period. A drop at or above the registered size must be flagged.

Usage::

    .venv/bin/python -m optional.coverage_benchmark --fact <shaped.parquet> \\
        --out reports/benchmarks/coverage.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from qc.config import DatasetConfig
from qc.coverage import (
    DEFAULT_MIN_REFERENCE_PERIODS,
    DEFAULT_QUANTILE,
    DEFAULT_REFERENCE_PERIODS,
    assess_scope_coverage,
)

GATE_PATH = Path("config/coverage-gate.json")


@dataclass
class CoverageArm:
    """Outcome for one arm of one period."""

    period: int
    arm: str
    evaluated_scopes: int = 0
    flags: int = 0
    false_alarms: int = 0
    genuine: int = 0
    dropped_entities: int = 0
    scopes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_gate() -> dict[str, Any]:
    return json.loads(GATE_PATH.read_text())


def _config(entity_columns: list[str]) -> DatasetConfig:
    return DatasetConfig(
        name="coverage-benchmark",
        entity_key_columns=tuple(entity_columns),
        entity_columns=tuple(entity_columns),
    )


def _classify(result: Any) -> tuple[int, int, int, int]:
    """Split flags into false alarms and genuine excursions.

    A flag is a *false alarm* when the scope's covered count still lies inside
    the range the scope has already demonstrated. That is the registered
    definition and it is deliberately strict.
    """
    evaluated = len([s for s in result.scopes if s.threshold is not None])
    false_alarms = 0
    genuine = 0
    for scope in result.regressed:
        if scope.baseline_min is None or scope.baseline_max is None:
            continue
        if scope.baseline_min <= scope.covered <= scope.baseline_max:
            false_alarms += 1
        else:
            genuine += 1
    return evaluated, false_alarms + genuine, false_alarms, genuine


def run_benchmark(
    fact: pd.DataFrame,
    entity_columns: list[str] | None = None,
    first_period: int | None = None,
) -> dict[str, Any]:
    """Execute the pre-registered gate over a real refresh sequence."""
    gate = load_gate()
    entity_columns = entity_columns or ["store_id", "product_id"]
    config = _config(entity_columns)
    week = config.week_column
    max_period = int(fact[week].max())
    start = first_period or (
        DEFAULT_REFERENCE_PERIODS + DEFAULT_MIN_REFERENCE_PERIODS + 1
    )
    periods = list(range(start, max_period + 1))

    control_arms: list[CoverageArm] = []
    for period in periods:
        result = assess_scope_coverage(
            fact,
            config,
            period,
            entity_columns=entity_columns,
            reference_periods=DEFAULT_REFERENCE_PERIODS,
            quantile=DEFAULT_QUANTILE,
            min_reference_periods=DEFAULT_MIN_REFERENCE_PERIODS,
        )
        evaluated, flags, false_alarms, genuine = _classify(result)
        control_arms.append(
            CoverageArm(
                period=period,
                arm="control",
                evaluated_scopes=evaluated,
                flags=flags,
                false_alarms=false_alarms,
                genuine=genuine,
            )
        )

    injected_arms: list[CoverageArm] = []
    fractions = [0.3, 0.5, 0.7]
    week_column = config.week_column
    for period in periods[-min(8, len(periods)) :]:
        for fraction in fractions:
            # Remove a fraction of the period's own entity pairs: a genuine
            # coverage drop, not a relabelling of an existing profile.
            period_rows = fact[week_column] == period
            period_frame = fact.loc[period_rows]
            pairs = period_frame[entity_columns].drop_duplicates()
            keep = pairs.sample(frac=1.0 - fraction, random_state=17)
            key_of = pairs.set_index(entity_columns).index
            keep_set = set(map(tuple, keep.to_numpy().tolist()))
            keep_mask = pd.Series(
                [tuple(row) in keep_set for row in key_of.to_numpy().tolist()],
                index=pairs.index,
            )
            surviving = pairs.index[keep_mask]
            dropped = period_frame.index.difference(surviving)
            mutated = fact.drop(index=dropped)
            result = assess_scope_coverage(
                mutated, config, period, entity_columns=entity_columns
            )
            ev, flags, false_alarms, genuine = _classify(result)
            injected_arms.append(
                CoverageArm(
                    period=period,
                    arm=f"drop_{fraction:.0%}",
                    evaluated_scopes=ev,
                    flags=flags,
                    false_alarms=false_alarms,
                    genuine=genuine,
                    dropped_entities=int(len(dropped)),
                )
            )

    total_flags = sum(arm.flags for arm in control_arms)
    total_false = sum(arm.false_alarms for arm in control_arms)
    control_alarm_rate = total_false / total_flags if total_flags else 0.0

    detected = 0
    trials = 0
    for arm in injected_arms:
        if arm.dropped_entities <= 0:
            continue
        trials += 1
        if arm.genuine > 0:
            detected += 1
    detection_rate = detected / trials if trials else 0.0

    passed = (
        control_alarm_rate <= float(gate["max_control_alarm_rate"])
        and detection_rate >= float(gate["min_detection_rate"])
    )
    return {
        "gate": gate,
        "entity_columns": entity_columns,
        "periods": len(periods),
        "control_alarm_rate": control_alarm_rate,
        "control_flags": total_flags,
        "control_false_alarms": total_false,
        "control_genuine": total_flags - total_false,
        "injected_detection_rate": detection_rate,
        "injected_trials": trials,
        "injected_detected": detected,
        "control_arms": [arm.to_dict() for arm in control_arms],
        "injected_arms": [arm.to_dict() for arm in injected_arms],
        "passed": passed,
        "claims_limit": gate.get("claims_limit", ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fact", required=True, help="shaped real fact parquet")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    fact = pd.read_parquet(args.fact)
    result = run_benchmark(fact)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    verdict = "PASS" if result["passed"] else "FAIL"
    print(
        f"coverage gate verdict={verdict} "
        f"control_alarm_rate={result['control_alarm_rate']:.3f} "
        f"(gate <= {result['gate']['max_control_alarm_rate']}) "
        f"injected_detection_rate={result['injected_detection_rate']:.3f} "
        f"(gate >= {result['gate']['min_detection_rate']})"
    )
    print(
        f"  control flags {result['control_flags']} "
        f"({result['control_genuine']} genuine, {result['control_false_alarms']} false)"
    )
    print(
        f"  injected {result['injected_detected']}/{result['injected_trials']} detected"
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
