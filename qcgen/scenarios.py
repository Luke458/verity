"""Scenario construction: clean pipeline, faulty pipeline, oracle manifest.

One scenario is one version pair (V0001 -> V0002) with exactly one scheduled
fault family. The clean pipeline is kept alongside the faulty pipeline so every
case's effect can be measured exactly at every stage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONTROL_FAMILIES, STAGES, SuiteConfig
from .dgp import generate_history, week_end_dates
from .faults import INJECTORS, FaultContext
from .oracle import GroundTruthCase, effect_delta, json_default
from .oracle_vault import OracleVault, default_oracle_root, default_registry_root
from .snapshots import SnapshotStore
from .stages import ADVANCE, STAGE_ORDER, State, build_source
from .universe import generate_universe


@dataclass(frozen=True)
class FaultSpec:
    family: str
    stage: str
    params: dict


@dataclass
class ScenarioResult:
    scenario_id: str
    directory: Path
    manifest: dict
    oracle: dict


def fault_spec(
    family: str,
    rng: np.random.Generator,
    n_stores: int,
    n_products: int,
    current_week: int,
) -> FaultSpec:
    """Default parameters per family, scaled to the universe size."""
    if family == "missing_stores":
        return FaultSpec(family, "source", {"n_stores": max(1, n_stores // 8), "week": current_week})
    if family == "missing_products":
        return FaultSpec(
            family,
            "source",
            {"n_products": max(1, n_products // 20), "week": current_week},
        )
    if family == "entity_merge":
        return FaultSpec(family, "source", {})
    if family in ("new_store_backfill", "expected_event"):
        return FaultSpec(
            family,
            "source",
            {
                "n_stores": 1,
                "backfill_weeks": min(20, max(6, current_week // 4)),
                "expected": family == "expected_event",
            },
        )
    if family == "history_truncation":
        return FaultSpec(
            family,
            "source",
            {"n_stores": max(1, n_stores // 10), "n_weeks": min(6, max(2, current_week // 10))},
        )
    if family == "commodity_remap":
        return FaultSpec(family, "source", {"n_products": max(1, n_products // 20)})
    if family == "coding_error":
        return FaultSpec(
            family,
            "coded",
            {"n_products": max(1, n_products // 15), "factor": float(rng.uniform(0.5, 0.8))},
        )
    if family == "warehouse_transform_error":
        return FaultSpec(family, "warehouse", {"factor": float(rng.uniform(0.4, 0.7))})
    if family == "recalculation":
        return FaultSpec(family, "source", {"sigma": 0.004})
    if family == "schema_failure":
        return FaultSpec(family, "report", {"column": "dollar"})
    if family == "null_duplicate_storm":
        return FaultSpec(
            family, "warehouse", {"null_fraction": 0.05, "duplicate_fraction": 0.02}
        )
    if family == "market_movement":
        return FaultSpec(family, "warehouse", {"factor": float(rng.uniform(0.7, 0.88))})
    raise ValueError(f"unknown fault family: {family!r}")


def schedule(config: SuiteConfig, n_scenarios: int) -> list[str]:
    """Cycle families and interleave controls after every third fault."""
    scheduled: list[str] = []
    control_index = 0
    for position, family in enumerate(config.families):
        scheduled.append(family)
        if config.controls and (position + 1) % 3 == 0:
            scheduled.append(config.controls[control_index % len(config.controls)])
            control_index += 1
    if not scheduled:
        scheduled = list(config.controls or CONTROL_FAMILIES)
    return [scheduled[i % len(scheduled)] for i in range(n_scenarios)]


def _split_effects(
    clean_fact: pd.DataFrame,
    faulty_fact: pd.DataFrame,
    column: str,
    values: list[str],
) -> dict[str, float]:
    if column not in clean_fact.columns or column not in faulty_fact.columns:
        return {}
    result: dict[str, float] = {}
    for value in values:
        before = float(clean_fact.loc[clean_fact[column] == value, "dollar"].sum())
        after = float(faulty_fact.loc[faulty_fact[column] == value, "dollar"].sum())
        result[str(value)] = round(after - before, 6)
    return result


def build_scenario(
    config: SuiteConfig,
    index: int,
    suite_dir: Path,
    family: str,
    stages: tuple[str, ...],
    oracle_root: str | Path | None = None,
) -> ScenarioResult:
    scenario_id = f"scenario-{index:04d}"
    seed = config.seed * 1000 + index
    rng = np.random.default_rng(seed)
    current_week = config.history.n_weeks

    universe = generate_universe(
        config.universe, rng, horizon_weeks=config.history.n_weeks
    )
    truth = generate_history(universe, config.history, rng)
    dates = week_end_dates(config.history.start_week, config.history.n_weeks)
    calendar = pd.DataFrame(
        {"week": np.arange(1, current_week + 1, dtype=np.int32), "week_end": dates}
    )
    source_state = build_source(truth, universe, calendar)

    spec = fault_spec(
        family, rng, len(universe.stores), len(universe.products), current_week
    )

    clean_states: dict[str, State] = {"source": source_state}
    for stage in STAGE_ORDER[1:]:
        clean_states[stage] = ADVANCE[stage](clean_states[STAGE_ORDER[STAGE_ORDER.index(stage) - 1]])

    context = FaultContext(
        scenario_id=scenario_id,
        family=family,
        stage=spec.stage,
        current_week=current_week,
        rng=rng,
    )

    faulty_states: dict[str, State] = {}
    state = source_state
    case: GroundTruthCase | None = None
    for stage in STAGE_ORDER:
        if stage != "source":
            state = ADVANCE[stage](state)
        if stage == spec.stage:
            state, case = INJECTORS[family](state, context, spec.params)
        faulty_states[stage] = state
    if case is None:
        raise RuntimeError(f"fault {family!r} was never injected")

    for stage in STAGE_ORDER:
        if STAGE_ORDER.index(stage) < STAGE_ORDER.index(spec.stage):
            case.effects[stage] = {}
        else:
            case.effects[stage] = effect_delta(
                clean_states[stage]["fact"],
                faulty_states[stage]["fact"],
                case.affected,
                case.weeks or None,
            )
    effect_stage = spec.stage
    for stage in STAGE_ORDER[STAGE_ORDER.index(spec.stage):]:
        if any(abs(value) > 1e-9 for value in case.effects[stage].values()):
            effect_stage = stage
            break
    case.details.setdefault("effect_stage", effect_stage)
    case.injected_effect = case.effects[effect_stage]

    split_by = case.details.get("split_by")
    if split_by:
        split_stage = max(effect_stage, "coded", key=STAGE_ORDER.index)
        split = _split_effects(
            clean_states[split_stage]["fact"],
            faulty_states[split_stage]["fact"],
            str(split_by),
            [str(v) for v in case.details.get("split_values", [])],
        )
        case.details["per_value"] = split
        case.details["split_stage"] = split_stage
        if split:
            case.details["effect_stage"] = split_stage
            values = list(split.values())
            case.injected_effect = {
                "dollar_net": round(sum(values), 6),
                "dollar_moved": round(max(abs(value) for value in values), 6),
            }

    previous_truth = truth.loc[truth["week"] < current_week].copy()
    previous_calendar = calendar.loc[calendar["week"] < current_week]
    previous_source = build_source(previous_truth, universe, previous_calendar)
    previous_states: dict[str, State] = {"source": previous_source}
    for stage in STAGE_ORDER[1:]:
        previous_states[stage] = ADVANCE[stage](
            previous_states[STAGE_ORDER[STAGE_ORDER.index(stage) - 1]]
        )

    store = SnapshotStore(suite_dir)
    scenario_dir = store.scenario_dir(scenario_id)
    dim_names = list(universe.dimensions())
    current_dims = {name: faulty_states["report"][name] for name in dim_names}
    previous_dims = {name: previous_states["report"][name] for name in dim_names}
    current_manifest = store.write_version(
        scenario_dir, "V0002", faulty_states, tuple(stages), current_dims
    )
    previous_manifest = store.write_version(
        scenario_dir, "V0001", previous_states, tuple(stages), previous_dims
    )

    manifest = {
        "scenario_id": scenario_id,
        "seed": seed,
        "profile": config.profile,
        "previous_version": "V0001",
        "current_version": "V0002",
        "n_previous_weeks": current_week - 1,
        "n_current_weeks": current_week,
        "stages_written": list(stages),
        "stages_computed": list(STAGE_ORDER),
        "versions": {"V0001": previous_manifest, "V0002": current_manifest},
    }
    oracle = {
        "scenario_id": scenario_id,
        "seed": seed,
        "profile": config.profile,
        "family": family,
        "fault": asdict(spec),
        "cases": [case.to_dict()],
        "expected_events": [case.event_registry] if case.event_registry else [],
    }
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=json_default)
    )
    if oracle_root is not None:
        OracleVault(oracle_root).write(scenario_id, oracle)
    return ScenarioResult(
        scenario_id=scenario_id,
        directory=scenario_dir,
        manifest=manifest,
        oracle=oracle,
    )


def generate_suite(
    config: SuiteConfig,
    out_root: str | Path,
    n_scenarios: int,
    suite_id: str,
    stages: tuple[str, ...] | None = None,
    oracle_root: str | Path | None = None,
    registry_root: str | Path | None = None,
) -> Path:
    stages = tuple(stages or config.stages)
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")
    suite_dir = Path(out_root) / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    oracle_root = Path(oracle_root) if oracle_root else default_oracle_root(suite_dir)
    registry_root = (
        Path(registry_root) if registry_root else default_registry_root(suite_dir)
    )

    planned = schedule(config, n_scenarios)
    scenarios = []
    for index, family in enumerate(planned):
        result = build_scenario(
            config, index, suite_dir, family, stages, oracle_root=oracle_root
        )
        expected_events = result.oracle.get("expected_events", [])
        if expected_events:
            registry_root.mkdir(parents=True, exist_ok=True)
            (registry_root / f"{result.scenario_id}.json").write_text(
                json.dumps(
                    {"events": expected_events},
                    indent=2,
                    sort_keys=True,
                    default=json_default,
                )
            )
        scenarios.append(
            {
                "scenario_id": result.scenario_id,
                "row_counts": result.manifest["versions"]["V0002"]["row_counts"],
            }
        )

    suite = {
        "suite_id": suite_id,
        "profile": config.profile,
        "seed": config.seed,
        "n_scenarios": n_scenarios,
        "stages": list(stages),
        "scenarios": scenarios,
    }
    (suite_dir / "suite.json").write_text(
        json.dumps(suite, indent=2, sort_keys=True, default=json_default)
    )
    return suite_dir
