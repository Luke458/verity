"""CLI for the QC engine.

    qc run --scenario-dir data/suites/demo/scenario-0000
    qc decide --scenario-dir ... --json
    qc train --suite-dir data/suites/demo --out artifacts/decision-v1
    qc investigate --scenario-dir ... --agent-cmd "my-llm-agent"
    qc shadow --suite-dir data/suites/demo --labels-out data/labels.jsonl
    qc incidents list --store data/incidents.jsonl

The scenario adapter lives in ``qcgen`` and is imported lazily so the engine
package itself has no dependency on the generator.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .agent import CommandAgent, NullAgent, build_investigation_brief
from .champion import (
    EvalItem,
    eval_cases_from_suite,
    run_champion,
    write_champion_report,
)
from .cohort import CohortPlan, run_cohort
from .config import DatasetConfig, load_dataset_config
from .decisions import (
    DecisionSet,
    FeatureEncoder,
    RuleDecisionProvider,
)
from .drift import monitor_drift
from .evidence_query import ALLOWED_QUERIES, query_evidence
from .expectations import apply_expectations, load_expectations
from .incidents import IncidentStore, build_incident_record, symptom_tags_for
from .labels import LabelStore, build_oracle_labels, records_from_store
from .prequential import PrequentialStore, prequential_sequence
from .rca import investigate
from .reconciliation import run_reconciliation  # noqa: F401  (public surface)
from .relationships import RelationshipStore
from .replay import ReplayPlan, replay
from .reporting import write_report
from .run import QCRunResult, run_qc
from .shadow import run_shadow
from .source import mapped
from .store import SqliteStore, import_outcomes
from .synthetic_analyst import PROFILES as ANALYST_PROFILES
from .synthetic_analyst import simulate_analyst
from .systemone import FallbackDecisionProvider, SystemOneDecisionProvider
from .text_provider import (
    ModernBertEmbedder,
    load_decision_provider,
    train_text_provider,
)
from .training import save_training_run, train_decision_provider
from .tspulse_benchmark import run_tspulse_benchmark
from .weekly import run_weekly


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _load_provider(path: str | None):
    if path in (None, "", "rule"):
        return None
    if path.startswith("systemone:"):
        return SystemOneDecisionProvider(url=path.split(":", 1)[1])
    if path.startswith("hybrid:"):
        url = path.split(":", 1)[1]
        return FallbackDecisionProvider(
            local=RuleDecisionProvider(),
            remote=SystemOneDecisionProvider(url=url),
        )
    return load_decision_provider(path)


def _print_decisions(decisions: DecisionSet | None) -> None:
    if decisions is None:
        return
    print(f"DECISION {decisions.provider}")
    for name in ("likely_cause", "likely_origin", "severity", "requires_investigation"):
        value = decisions.get(name)
        if value is None:
            continue
        probability = (
            value.probabilities.get(
                "True" if value.value is True else "False"
                if isinstance(value.value, bool)
                else str(value.value)
            )
            if value.probabilities
            else None
        )
        probability_text = f"{probability:.3f}" if probability is not None else "n/a"
        index_text = (
            f" idx={value.index:.2f}" if value.index is not None else ""
        )
        print(
            f"  {name:<24} {str(value.value):<24} {probability_text}{index_text}"
        )


def _print_human(result: QCRunResult) -> None:
    pair = result.version_pair
    print(f"QC RUN {result.run_id}")
    print(f"STATUS {result.status}")
    print(f"CONTRACTS {result.contracts.status} ({len(result.contracts.failed)} failed)")
    if pair is not None:
        print(
            f"VERSIONS {pair.previous_id} -> {pair.current_id} "
            f"[{pair.shape}] overlap {pair.overlap_start}..{pair.overlap_end} "
            f"new {list(pair.new_periods)}"
        )
    attribution = result.attribution
    if attribution is not None:
        print("HISTORICAL REVISION")
        print(f"  raw_delta          {attribution.raw_delta:,.2f}")
        print(f"  explained_delta    {attribution.explained_delta:,.2f}")
        print(f"  unexplained_delta  {attribution.unexplained_delta:,.2f}")
        print(f"  explained_fraction {attribution.explained_fraction:.4f}")
        print(f"  breadth            {attribution.breadth:.4f}")
        if attribution.contributors:
            print("  top contributors")
            for contributor in attribution.contributors[:5]:
                print(
                    f"    {contributor.entity_type}:{contributor.entity_id} "
                    f"{contributor.delta:+,.2f}"
                )
    if result.counterfactual is not None:
        print("COUNTERFACTUAL")
        print(
            f"  reconstructed_delta  "
            f"{result.counterfactual.reconstructed_delta:,.2f}"
        )
        print(
            "  reconciliation_score "
            + (
                f"{result.counterfactual.reconciliation_score:.4f}"
                if result.counterfactual.reconciliation_score is not None
                else "n/a"
            )
        )
    if result.reconciliation is not None:
        print(f"RECONCILIATION {result.reconciliation.status}")
    if result.lineage is not None:
        print(
            f"LINEAGE {result.lineage.status} "
            f"first_divergence={result.lineage.first_divergence}"
        )
    if result.temporal is not None:
        print(
            f"LATEST WEEK {result.temporal.target_week} "
            f"anomaly={result.temporal.anomaly} flags={result.temporal.flags}"
        )
        for item in result.temporal.series:
            if not item.anomaly:
                continue
            percentile = item.calibrated_percentile
            percentile_text = f"{percentile:.3f}" if percentile is not None else "n/a"
            print(
                f"  {item.series_id} actual={item.actual:,.0f} "
                f"adjusted={item.adjusted_actual:,.0f} "
                f"median={item.forecast_median:,.0f} "
                f"pct={percentile_text} flags={item.flags}"
            )
    if result.evidence is not None:
        print(
            f"EVIDENCE nodes={len(result.evidence.nodes)} "
            f"edges={len(result.evidence.edges)}"
        )
    _print_decisions(result.decisions)
    if result.reasons:
        print("REASONS")
        for reason in result.reasons:
            print(f"  {reason}")


def _scenario_versions(source):
    versions = source.list_versions()
    current = versions[-1] if versions else None
    previous = versions[-2] if len(versions) > 1 else None
    return current, previous


def _run_scenario(args, config, provider, registry=None):
    from qcgen.sources import ScenarioSource

    scenario_dir = Path(args.scenario_dir)
    if registry is None and getattr(args, "registry", None):
        from .registry import FileRegistry

        registry = FileRegistry(args.registry)

    raw_source = ScenarioSource(scenario_dir)
    source = mapped(raw_source, config.column_map_dict())
    current = args.current if getattr(args, "current", None) else None
    previous = args.previous if getattr(args, "previous", None) else None
    default_current, default_previous = _scenario_versions(source)
    current = current or default_current
    previous = previous or default_previous
    if current is None or previous is None:
        raise SystemExit("need at least two versions under the scenario directory")
    return run_qc(
        source,
        current,
        previous,
        config,
        registry=registry,
        run_id=getattr(args, "run_id", None),
        decision_provider=provider,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    if args.no_decisions:
        config = replace(config, decision_enabled=False)
    result = _run_scenario(args, config, _load_provider(args.provider))
    if args.json:
        print(json.dumps(result.machine, indent=2, default=_json_default))
    else:
        _print_human(result)
    return 0


def _cmd_decide(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    result = _run_scenario(args, config, _load_provider(args.provider))
    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "decision": (
                        result.decisions.to_dict() if result.decisions else None
                    ),
                },
                indent=2,
                default=_json_default,
            )
        )
    else:
        print(f"QC RUN {result.run_id}")
        print(f"STATUS {result.status}")
        _print_decisions(result.decisions)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    if args.labels:
        records = LabelStore(args.labels).load()
        if not records:
            print(f"no label records in {args.labels}", file=sys.stderr)
            return 2
    else:
        records = build_oracle_labels(
            args.suite_dir, config, oracle_dir=getattr(args, "oracle_dir", None)
        )
    if args.text_embedder:
        embedder = ModernBertEmbedder(
            model_id=args.text_embedder,
            revision=args.text_revision,
            max_length=args.text_max_length,
        )
        text_provider, metrics = train_text_provider(
            records,
            embedder,
            config,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        text_provider.save(args.out)
        (Path(args.out) / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True)
        )
        trained: Any = text_provider
    else:
        feature_provider, metrics = train_decision_provider(
            records,
            config,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        save_training_run(feature_provider, metrics, args.out)
        trained = feature_provider
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    print(
        f"trained decision provider on {len(records)} records -> {args.out}"
    )
    for split in ("train", "validation"):
        split_metrics = metrics[split]
        print(
            f"  {split:<11} n={split_metrics['n']:<4} "
            f"overall_accuracy={split_metrics['overall_accuracy']:.3f}"
        )
        for field, field_metrics in split_metrics["fields"].items():
            print(
                f"    {field:<24} acc={field_metrics['accuracy']:.3f} "
                f"brier={field_metrics['brier']:.3f} ece={field_metrics['ece']:.3f}"
            )
    print("  warning:", trained.metadata.get("warning", ""))
    return 0


def _cmd_investigate(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    result = _run_scenario(args, config, _load_provider(args.provider))
    decisions = result.decisions
    if decisions is None:
        print("decisions are disabled for this run", file=sys.stderr)
        return 2

    encoder = FeatureEncoder(config)
    incidents: list = []
    store = IncidentStore(args.incident_store) if args.incident_store else None
    if store is not None:
        incidents = store.retrieve(
            encoder.encode(result),
            k=args.incidents,
            tags=symptom_tags_for(result, decisions),
        )
    brief = build_investigation_brief(result, decisions, incidents)
    agent = CommandAgent(args.agent_cmd) if args.agent_cmd else NullAgent()
    investigation = agent.investigate(brief)

    if store is not None and args.save_draft:
        store.add(
            build_incident_record(
                result,
                decisions,
                feedback={
                    "root_cause": investigation.root_cause,
                    "resolution": "",
                    "analyst_summary": investigation.summary,
                    "confirmed": False,
                },
                encoder=encoder,
            )
        )

    if args.json:
        print(
            json.dumps(
                {
                    "brief": brief.to_dict(),
                    "investigation": investigation.to_dict(),
                },
                indent=2,
                default=_json_default,
            )
        )
        return 0

    print(f"INVESTIGATION {investigation.run_id} agent={investigation.agent}")
    print(f"  root_cause  {investigation.root_cause}")
    print(f"  confidence  {investigation.confidence:.3f}")
    print(f"  summary     {investigation.summary}")
    if investigation.recommended_actions:
        print("  recommended actions")
        for action in investigation.recommended_actions:
            print(f"    {action}")
    if investigation.follow_up_questions:
        print("  follow-up questions")
        for question in investigation.follow_up_questions:
            print(f"    {question}")
    return 0


def _cmd_tspulse_bench(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    result = run_tspulse_benchmark(
        args.suite_dir,
        config,
        allow_resample=not args.strict_length,
        max_scenarios=args.max_scenarios,
        oracle_dir=getattr(args, "oracle_dir", None),
    )
    payload = result.to_dict()
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
        )
    if args.json:
        print(json.dumps(payload, indent=2, default=_json_default))
        return 0
    print(
        f"tspulse benchmark suite={result.suite_id} "
        f"scenarios={result.n_scenarios} series={result.n_series} "
        f"embedded={result.embedded_series}"
    )
    print(
        f"  nearest_centroid_accuracy={result.nearest_centroid_accuracy:.3f}"
    )
    print(
        f"  control_mean_norm={result.control_mean_embedding_norm:.3f} "
        f"fault_mean_norm={result.fault_mean_embedding_norm:.3f}"
    )
    print(
        f"  intra_family_similarity={result.mean_intra_family_similarity:.3f} "
        f"inter_family_similarity={result.mean_inter_family_similarity:.3f}"
    )
    print(
        f"  transformations={result.transformations} "
        f"production_eligible={result.production_eligible}"
    )
    for limitation in result.limitations:
        print(f"  limitation: {limitation}")
    if args.out:
        print(f"  output: {args.out}")
    return 0


def _cmd_incidents(args: argparse.Namespace) -> int:
    store = IncidentStore(args.store)
    records = store.load()
    if args.json:
        print(
            json.dumps(
                [record.to_dict() for record in records],
                indent=2,
                default=_json_default,
            )
        )
        return 0
    print(f"incidents={len(records)} store={args.store}")
    for record in records:
        state = "confirmed" if record.confirmed else "draft"
        print(
            f"  {record.incident_id} [{state}] {record.root_cause} "
            f"({record.likely_origin}) {record.analyst_summary}"
        )
    return 0


def _cmd_shadow(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    summary = run_shadow(
        args.suite_dir,
        config=config,
        registry_path=args.registry,
        out_dir=args.out,
        labels_out=args.labels_out,
        prequential_store=args.prequential_store,
        oracle_dir=getattr(args, "oracle_dir", None),
        with_registry=getattr(args, "with_registry", False),
    )
    print(f"shadow suite={summary['suite_id']} "
          f"registry_mode={summary.get('registry_mode', 'blind')} "
          f"scenarios={summary['scenarios']}")
    print(
        f"  detection_rate={summary['detection_rate']:.3f} "
        f"false_positive_rate={summary['false_positive_rate']:.3f}"
    )
    print(
        f"  latest_week_detection_rate={summary['latest_week_detection_rate']:.3f} "
        f"latest_week_control_rate={summary['latest_week_control_rate']:.3f}"
    )
    print(
        f"  expected_event_pass_rate={summary['expected_event_pass_rate']:.3f} "
        f"reconciliation_failures={summary['reconciliation_failures']}"
    )
    print(
        f"  lineage_first_divergence_accuracy="
        f"{summary['lineage_first_divergence_accuracy']:.3f} "
        f"(n={summary['lineage_comparable']})"
    )
    print(
        f"  mean_reconstruction_score={summary['mean_reconstruction_score']:.3f} "
        f"status_counts={summary['engine_status_counts']}"
    )
    if args.out:
        print(f"  output: {args.out}")
    if args.labels_out:
        print(f"  labels: {args.labels_out}")
    return 0


def _add_scenario_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--current", default=None)
    parser.add_argument("--previous", default=None)
    parser.add_argument("--config", default=None, help="dataset YAML config")
    parser.add_argument("--registry", default=None, help="expected-event registry JSON")


def _cmd_relationships(args: argparse.Namespace) -> int:
    store = RelationshipStore(args.store)
    records = store.load()
    if args.json:
        print(
            json.dumps(
                [record.to_dict() for record in records],
                indent=2,
                default=_json_default,
            )
        )
        return 0
    print(f"relationships={len(records)} store={args.store}")
    for record in records:
        state = "confirmed" if record.confirmed else "candidate"
        print(
            f"  {record.source_id} -{record.relationship}-> {record.target_id} "
            f"({record.entity_type}, {state})"
        )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    result = _run_scenario(args, config, _load_provider(args.provider))

    brief = None
    if not args.no_brief and result.decisions is not None:
        incidents: list = []
        if args.incident_store:
            encoder = FeatureEncoder(config)
            store = IncidentStore(args.incident_store)
            incidents = store.retrieve(
                encoder.encode(result),
                k=3,
                tags=symptom_tags_for(result, result.decisions),
            )
        brief = build_investigation_brief(result, result.decisions, incidents)

    paths = write_report(result, args.out, brief)
    print(f"report written: {paths['markdown']}")
    print(f"machine output: {paths['machine']}")
    return 0


def _cmd_delta_info(args: argparse.Namespace) -> int:
    from .delta import describe_delta_table

    options = json.loads(args.storage_options) if args.storage_options else None
    info = describe_delta_table(args.uri, options)
    if args.json:
        print(json.dumps(info, indent=2, default=_json_default))
        return 0
    print(f"delta table {info['uri']}")
    print(
        f"  current_version={info['current_version']} "
        f"rows={info['rows_at_current']} "
        f"partitions={info['partition_columns']}"
    )
    print("  history")
    for entry in info["history"][-20:]:
        print(
            f"    v{entry['version']:<6} {entry.get('timestamp')} "
            f"{entry.get('operation')}"
        )
    print("  schema")
    for field in info["schema"]:
        print(f"    {field['name']:<24} {field['type']}")
    return 0


def _cmd_delta_run(args: argparse.Namespace) -> int:
    from .delta import DeltaSource

    options = json.loads(args.storage_options) if args.storage_options else None
    stage_tables = json.loads(args.stage_tables) if args.stage_tables else None
    delta_source = DeltaSource(
        uri=args.uri,
        storage_options=options,
        stage=args.stage,
        stage_tables=stage_tables,
    )
    versions = delta_source.list_versions()
    current = args.current or (versions[-1] if versions else None)
    previous = args.previous or (versions[-2] if len(versions) > 1 else None)
    if current is None or previous is None:
        print("need at least two versions in the table", file=sys.stderr)
        return 2
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    source = mapped(delta_source, config.column_map_dict())
    result = run_qc(
        source,
        current,
        previous,
        config,
        run_id=args.run_id,
        decision_provider=_load_provider(args.provider),
    )
    if args.json:
        print(json.dumps(result.machine, indent=2, default=_json_default))
    else:
        _print_human(result)
    for warning in getattr(source, "warnings", []):
        print(f"WARNING {warning}")
    return 0


def _cmd_evidence(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    result = _run_scenario(args, config, _load_provider(args.provider))
    filters: dict[str, str] = {}
    for item in args.filter or []:
        key, separator, value = item.partition("=")
        if not key or not separator:
            print(f"invalid --filter {item!r}; expected KEY=VALUE", file=sys.stderr)
            return 2
        filters[key] = value
    outcome = query_evidence(result, args.query, limit=args.limit, **filters)
    if args.json:
        print(json.dumps(outcome.to_dict(), indent=2, default=_json_default))
        return 0
    print(
        f"query={outcome.query} rows={len(outcome.rows)}/{outcome.total_rows} "
        f"truncated={outcome.truncated}"
    )
    for row in outcome.rows:
        print("  " + json.dumps(row, default=_json_default, sort_keys=True))
    return 0


def _cmd_prequential(args: argparse.Namespace) -> int:
    pool = PrequentialStore(args.store).pool()
    usable = (
        pool.usable(args.as_of, args.target_week)
        if args.as_of is not None and args.target_week is not None
        else []
    )
    print(
        f"prequential store={args.store} records={len(pool.records)} "
        f"usable={len(usable)}"
    )
    if (
        args.z is not None
        and args.as_of is not None
        and args.target_week is not None
    ):
        percentile = pool.percentile(args.z, args.as_of, args.target_week)
        interval = pool.interval(
            0.0, args.as_of, args.target_week, alpha=args.alpha
        )
        print(f"  percentile={percentile}")
        print(f"  interval={json.dumps(interval.to_dict(), default=_json_default)}")
    return 0


def _cmd_cohort(args: argparse.Namespace) -> int:
    if args.plan:
        plan = CohortPlan.from_json(args.plan)
    else:
        overrides: dict = {}
        if args.profile:
            overrides["profile"] = args.profile
        if args.scenarios_per_family:
            overrides["scenarios_per_family"] = args.scenarios_per_family
        if args.dev_seed:
            overrides["dev_seeds"] = tuple(args.dev_seed)
        if args.heldout_seed:
            overrides["heldout_seeds"] = tuple(args.heldout_seed)
        plan = CohortPlan(**overrides)
    result = run_cohort(plan, workdir=args.workdir, out_dir=args.out)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=_json_default))
        return 0
    print(
        f"cohort cases={result.metrics['cases']} "
        f"detection_rate={result.metrics['detection_rate']:.3f} "
        f"false_positive_rate={result.metrics['false_positive_rate']:.3f} "
        f"reconstruction={result.metrics['mean_reconstruction_score']:.3f}"
    )
    for check in result.gate_results:
        state = "PASS" if check["passed"] else "FAIL"
        print(
            f"  {state} {check['gate']} actual={check['actual']:.3f} "
            f"threshold={check['threshold']}"
        )
    print(
        f"  gates_passed={result.gates_passed} "
        f"production_eligible={result.production_eligible} "
        f"code_sha256={result.code_sha256[:12]}"
    )
    if args.out:
        print(f"  output: {args.out}")
    return 0


def _cmd_store_add_run(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    result = _run_scenario(args, config, _load_provider(args.provider))
    with SqliteStore(args.store) as store:
        try:
            store.record_result(result, created=args.created)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        if args.root_cause:
            origin = args.origin
            severity = args.severity
            if result.decisions is not None:
                origin_decision = result.decisions.get("likely_origin")
                severity_decision = result.decisions.get("severity")
                if origin is None and origin_decision is not None:
                    origin = str(origin_decision.value)
                if severity is None and severity_decision is not None:
                    severity = str(severity_decision.value)
            store.record_outcome(
                result.run_id,
                root_cause=args.root_cause,
                confirmed=args.confirmed,
                likely_origin=origin or "UNKNOWN",
                severity=severity or "MEDIUM",
                resolution=args.resolution or "",
                summary=args.summary or "",
                analyst=args.analyst,
                symptom_tags=(
                    symptom_tags_for(result, result.decisions)
                    if result.decisions is not None
                    else []
                ),
                created=args.created,
            )
    print(f"recorded run {result.run_id} in {args.store}")
    return 0


def _cmd_store_add_outcome(args: argparse.Namespace) -> int:
    with SqliteStore(args.store) as store:
        try:
            outcome_id = store.record_outcome(
                args.run_id,
                root_cause=args.root_cause,
                confirmed=args.confirmed,
                likely_origin=args.origin or "UNKNOWN",
                severity=args.severity or "MEDIUM",
                resolution=args.resolution or "",
                summary=args.summary or "",
                analyst=args.analyst,
                created=args.created,
            )
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
    print(f"recorded outcome {outcome_id} for {args.run_id}")
    return 0


def _cmd_store_list(args: argparse.Namespace) -> int:
    with SqliteStore(args.store) as store:
        rows = store.list_runs()
    if args.confirmed_only:
        rows = [row for row in rows if row.get("confirmed")]
    if args.json:
        print(json.dumps(rows, indent=2, default=_json_default))
        return 0
    print(f"runs={len(rows)} store={args.store}")
    for row in rows:
        state = (
            "confirmed"
            if row.get("confirmed")
            else "candidate"
            if row.get("root_cause")
            else "unconfirmed"
        )
        print(
            f"  {row['run_id']} {row['status']:<24} "
            f"{row.get('root_cause') or '-':<24} {state}"
        )
    return 0


def _cmd_store_incidents(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    result = _run_scenario(args, config, _load_provider(args.provider))
    encoder = FeatureEncoder()
    with SqliteStore(args.store) as store:
        hits = store.retrieve(
            encoder.encode(result),
            k=args.k,
            tags=(
                symptom_tags_for(result, result.decisions)
                if result.decisions is not None
                else []
            ),
        )
    if args.json:
        print(json.dumps(hits, indent=2, default=_json_default))
        return 0
    print(f"confirmed incidents matching {result.run_id}: {len(hits)}")
    for hit in hits:
        print(
            f"  {hit['run_id']} {hit['root_cause']} "
            f"({hit['likely_origin']}) similarity={hit['similarity']:.3f}"
        )
        if hit["resolution"]:
            print(f"    resolution: {hit['resolution']}")
    return 0


def _cmd_drift(args: argparse.Namespace) -> int:
    pool = PrequentialStore(args.store).pool(max_records=args.max_records)
    report = monitor_drift(
        pool,
        as_of=args.as_of,
        target_week=args.target_week,
        alpha=args.alpha,
        ratio_threshold=args.ratio_threshold,
        coverage_margin=args.coverage_margin,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=_json_default))
        return 0
    print(
        f"drift status={report.status} usable={report.usable} "
        f"baseline={report.baseline_n} recent={report.recent_n}"
    )
    print(
        f"  scale_ratio={report.scale_ratio} "
        f"coverage={report.recent_coverage} "
        f"expected={report.expected_coverage}"
    )
    print(f"  flags={report.flags} detail={report.detail}")
    return 0


def _cmd_champion(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    if args.train_labels:
        train_records = LabelStore(args.train_labels).load()
    else:
        train_records = records_from_store(args.store)
    if not train_records:
        print("no training records found", file=sys.stderr)
        return 2

    eval_items = eval_cases_from_suite(
        args.suite_dir, config, oracle_dir=getattr(args, "oracle_dir", None)
    )
    if args.eval_store:
        for record in records_from_store(args.eval_store):
            eval_items.append(
                EvalItem(
                    case_id=record.run_id,
                    family=record.family,
                    labels=record.labels,
                    record=record,
                )
            )

    embedder = (
        ModernBertEmbedder(
            model_id=args.text_embedder, revision=args.text_revision
        )
        if args.text_embedder
        else None
    )
    remote = (
        SystemOneDecisionProvider(url=args.systemone_url)
        if args.systemone_url
        else None
    )
    gates: dict[str, float] = {}
    if args.min_overall_accuracy is not None:
        gates["min_overall_accuracy"] = args.min_overall_accuracy
    if args.min_cause_accuracy is not None:
        gates["min_cause_accuracy"] = args.min_cause_accuracy

    result = run_champion(
        train_records,
        eval_items,
        config,
        text_embedder=embedder,
        remote_provider=remote,
        gates=gates or None,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    paths = write_champion_report(result, args.out) if args.out else {}

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=_json_default))
        return 0
    print(
        f"champion={result.champion or 'none'} "
        f"production_eligible={result.production_eligible} "
        f"labels={result.provenance['labels']} "
        f"train={result.provenance['train_records']} "
        f"eval={result.provenance['eval_cases']}"
    )
    for score in result.scores:
        state = "available" if score.available else "unavailable"
        print(
            f"  {score.provider:<14} {state:<11} "
            f"overall={score.overall_accuracy:.3f} "
            f"cause={score.cause_accuracy:.3f} "
            f"cases={score.cases} {score.note}"
        )
    print(
        "  gates: "
        + ", ".join(f"{name}>={value}" for name, value in sorted(result.gates.items()))
    )
    for path in paths.values():
        print(f"  output: {path}")
    return 0


def _cmd_store_import(args: argparse.Namespace) -> int:
    if args.csv:
        rows = pd.read_csv(args.csv, dtype=str).fillna("").to_dict("records")
    else:
        from .delta import _delta_table

        options = (
            json.loads(args.storage_options) if args.storage_options else None
        )
        rows = _delta_table(args.delta, options, None).to_pandas().to_dict("records")
    with SqliteStore(args.store) as store:
        report = import_outcomes(store, rows, dry_run=args.dry_run)
    if args.json:
        print(json.dumps(report, indent=2, default=_json_default))
    else:
        print(
            f"imported={report['imported']} errors={len(report['errors'])} "
            f"dry_run={report['dry_run']}"
        )
        for error in report["errors"][:10]:
            print(f"  row {error['row']}: {error['error']}")
    return 0 if not report["errors"] else 2


def _cmd_onboard(args: argparse.Namespace) -> int:
    from .onboard import assess_versions, config_from_proposal, profile_table

    aliases: dict[str, str] = {}
    for item in args.alias or []:
        production, separator, canonical = item.partition("=")
        if not separator or not production or not canonical:
            print(
                f"invalid --alias {item!r}; expected production=canonical",
                file=sys.stderr,
            )
            return 2
        aliases[production.strip()] = canonical.strip()
    options = json.loads(args.storage_options) if args.storage_options else None
    profile = profile_table(
        args.uri,
        storage_options=options,
        sample_rows=args.sample_rows,
        count_rows=args.count,
        version=args.version,
        aliases=aliases,
    )
    proposal = profile["proposed"]
    assessment = None
    if args.previous is not None and args.current is not None:
        config = config_from_proposal(proposal)
        assessment = assess_versions(
            args.uri,
            config,
            args.previous,
            args.current,
            storage_options=options,
            stage=args.stage,
        )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(proposal, sort_keys=False))
    if args.json:
        print(
            json.dumps(
                {"profile": profile, "assessment": assessment},
                indent=2,
                default=_json_default,
            )
        )
        return 0
    print(
        f"onboard uri={profile['uri']} version={profile['current_version']} "
        f"rows_sampled={profile['rows_sampled']}"
    )
    print(
        f"  week_column={profile['week_column']} "
        f"metrics={profile['metric_columns']} "
        f"entities={profile['entity_columns']}"
    )
    column_map = profile["proposed"].get("column_map")
    if column_map:
        print(
            "  column_map="
            + ", ".join(
                f"{canonical}<-{production}"
                for canonical, production in sorted(column_map.items())
            )
        )
    for finding in profile["findings"]:
        print(
            f"  {finding['severity']:<7} {finding['code']}: {finding['message']}"
        )
    if assessment is not None:
        pair = assessment.get("version_pair")
        contracts = assessment.get("contracts") or {}
        print(
            f"  pair={'yes' if pair else 'no'} "
            f"shape={pair['shape'] if pair else 'n/a'} "
            f"contracts={contracts.get('status', 'n/a')}"
        )
        for finding in assessment["findings"]:
            print(
                f"  {finding['severity']:<7} {finding['code']}: {finding['message']}"
            )
    if args.out:
        print(f"  config written: {args.out}")
    return 0


def _cmd_simulate_analyst(args: argparse.Namespace) -> int:
    summary = simulate_analyst(
        args.suite_dir,
        args.store,
        profile=args.profile,
        seed=args.seed,
        oracle_dir=getattr(args, "oracle_dir", None),
    )
    if args.json:
        print(json.dumps(summary, indent=2, default=_json_default))
        return 0
    print(
        f"simulated analyst profile={summary['profile']} runs={summary['runs']} "
        f"confirmed={summary['confirmed']} drafts={summary['drafts']} "
        f"corrections={summary['corrections']} unknown={summary['unknown']} "
        f"investigation_flips={summary['investigation_flips']}"
    )
    print(f"  store: {summary['store']}")
    print("  provenance: synthetic (champion stays production-ineligible)")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    plan = ReplayPlan.from_json(args.plan)
    options = json.loads(args.storage_options) if args.storage_options else None
    report = replay(
        plan,
        args.out,
        default_uri=args.uri,
        config=config,
        storage_options=options,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=_json_default))
        return 0 if report.passed else 2
    print(
        f"replay cases={len(report.cases)} failed={len(report.failed)} "
        f"plan={report.plan_sha256[:12]}"
    )
    for case in report.cases:
        print(
            f"  {case['case_id']} {case['status']} "
            f"{case['previous_version']}->{case['current_version']} "
            f"as_of={case['as_of']}"
        )
    for failure in report.failed:
        print(f"  FAILED {failure['case_id']}: {failure['error']}")
    return 0 if report.passed else 2


def _cmd_rca(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    result = _run_scenario(args, config, _load_provider(args.provider))
    investigation = investigate(result, max_steps=args.steps, limit=args.limit)
    if args.json:
        print(json.dumps(investigation.to_dict(), indent=2, default=_json_default))
        return 0
    print(
        f"RCA {investigation.run_id} steps={len(investigation.steps)} "
        f"stop={investigation.stop_reason} confirmed={investigation.confirmed}"
    )
    for step in investigation.steps:
        print(
            f"  {step.step}. {step.tool} args={step.args} "
            f"rows={len(step.rows)}/{step.total_rows} truncated={step.truncated}"
        )
    return 0


def _cmd_expectations(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    result = _run_scenario(args, config, _load_provider(args.provider))
    expectations = load_expectations(args.expectations)
    flags = [
        {
            "metric": "dollar_per_unit",
            "week": flag["week"],
            "ratio": flag["dollar_per_unit"],
            "median": flag["median"],
            "robust_z": flag["robust_z"],
        }
        for flag in (result.reconciliation.ratio_flags if result.reconciliation else [])
    ]
    report = apply_expectations(
        flags,
        expectations,
        context={"dataset": result.dataset, "as_of": args.as_of or ""},
    )
    if args.json:
        print(json.dumps(report, indent=2, default=_json_default))
        return 0
    print(
        f"expectations expected={len(report['expected'])} "
        f"unexpected={len(report['unexpected'])}"
    )
    for item in report["expected"]:
        print(
            f"  explained week={item['week']} by {item['explained_by']} "
            f"({item['approved_by']})"
        )
    for item in report["unexpected"]:
        print(f"  unexplained week={item['week']} ratio={item['ratio']}")
    return 0


def _cmd_prequential_forecast(args: argparse.Namespace) -> int:
    from .delta import DeltaSource

    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    options = json.loads(args.storage_options) if args.storage_options else None
    versions = [value.strip() for value in args.versions.split(",") if value.strip()]
    if not versions:
        print("--versions is empty", file=sys.stderr)
        return 2
    delta_source = DeltaSource(
        uri=args.uri, storage_options=options, stage=args.stage
    )
    source = mapped(delta_source, config.column_map_dict())
    result = prequential_sequence(
        source,
        versions,
        config,
        scope=args.scope,
        min_samples=args.min_samples,
    )
    if args.out:
        PrequentialStore(args.out).add(result.pool.records)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=_json_default))
        return 0
    print(
        f"prequential scope={result.scope} loads={result.loads} "
        f"records={result.records_committed} pool={len(result.pool.records)} "
        f"flagged={len(result.flagged)}"
    )
    for item in result.evidence:
        if item.status == "INSUFFICIENT_HISTORY":
            continue
        flag = f" {item.flag}" if item.flag else ""
        percentile = (
            f"{item.pool_percentile:.3f}"
            if item.pool_percentile is not None
            else "n/a"
        )
        print(
            f"  {item.load_id} week={item.target_week} {item.series_id} "
            f"actual={item.actual:,.1f} forecast={item.forecast_median:,.1f} "
            f"z={item.standardized_residual:.2f} pct={percentile}{flag}"
        )
    if args.out:
        print(f"  pool written: {args.out}")
    return 0


def _cmd_weekly(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    options = json.loads(args.storage_options) if args.storage_options else None
    try:
        result = run_weekly(
            args.uri,
            config=config,
            store_path=args.store,
            calibration_path=args.calibration_store,
            out_root=args.out,
            stage=args.stage,
            current=args.current,
            previous=args.previous,
            expectations_path=args.expectations,
            reference_uri=args.reference_uri,
            reference_spec_path=args.reference_spec,
            storage_options=options,
            min_samples=args.min_samples,
            decision_provider=_load_provider(args.provider),
            force=args.force,
        )
    except Exception as error:  # noqa: BLE001 - the scheduler sees a failure
        print(
            f"weekly run failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=_json_default))
    else:
        print(
            f"weekly dataset={result.dataset} "
            f"{result.previous_version}->{result.current_version} "
            f"status={result.status}"
        )
        if result.decision is not None:
            print(
                f"  decision {result.decision['likely_cause']} "
                f"({result.decision['likely_origin']}) "
                f"requires_investigation="
                f"{result.decision['requires_investigation']}"
            )
        if result.drift is not None:
            print(f"  drift {result.drift['status']}")
        print(
            f"  records={result.prequential_records} "
            f"store_recorded={result.recorded} skipped={result.skipped}"
        )
        for note in result.notes:
            print(f"  note: {note}")
        if result.report_dir:
            print(f"  report: {result.report_dir}")

    if args.allow_investigate and result.status == "INVESTIGATE":
        return 0
    return {
        "ALREADY_PROCESSED": 0,
        "PASS": 0,
        "PASS_WITH_EXPLANATION": 0,
        "INVESTIGATE": 2,
        "DATA_CONTRACT_FAILURE": 3,
    }.get(result.status, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qc", description="Run deterministic QC over a version pair."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run QC over a scenario directory")
    _add_scenario_arguments(run)
    run.add_argument("--provider", default=None, help="rule or trained artifact dir")
    run.add_argument("--no-decisions", action="store_true")
    run.add_argument("--run-id", default=None)
    run.add_argument("--json", action="store_true")
    run.set_defaults(func=_cmd_run)

    decide = subparsers.add_parser("decide", help="run QC and print typed decisions")
    _add_scenario_arguments(decide)
    decide.add_argument("--provider", default=None)
    decide.add_argument("--run-id", default=None)
    decide.add_argument("--json", action="store_true")
    decide.set_defaults(func=_cmd_decide)

    train = subparsers.add_parser(
        "train", help="fit a learned decision provider from labels"
    )
    train.add_argument("--suite-dir", default=None)
    train.add_argument("--oracle-dir", default=None, help="ground-truth vault root")
    train.add_argument("--labels", default=None, help="label store JSONL")
    train.add_argument("--out", required=True)
    train.add_argument("--config", default=None)
    train.add_argument("--validation-fraction", type=float, default=0.3)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument(
        "--text-embedder",
        default=None,
        help="HF encoder id for a frozen-encoder text probe (e.g. ModernBERT)",
    )
    train.add_argument("--text-revision", default=None)
    train.add_argument("--text-max-length", type=int, default=4096)
    train.add_argument("--json", action="store_true")
    train.set_defaults(func=_cmd_train)

    investigate = subparsers.add_parser(
        "investigate", help="hand a case off to an investigation agent"
    )
    _add_scenario_arguments(investigate)
    investigate.add_argument("--provider", default=None)
    investigate.add_argument("--run-id", default=None)
    investigate.add_argument("--incident-store", default=None)
    investigate.add_argument("--incidents", type=int, default=3)
    investigate.add_argument(
        "--agent-cmd",
        default=None,
        help="command that reads the brief JSON on stdin and returns result JSON",
    )
    investigate.add_argument("--save-draft", action="store_true")
    investigate.add_argument("--json", action="store_true")
    investigate.set_defaults(func=_cmd_investigate)

    incidents = subparsers.add_parser("incidents", help="manage incident memory")
    incidents_sub = incidents.add_subparsers(dest="incident_command", required=True)
    incidents_list = incidents_sub.add_parser("list", help="list stored incidents")
    incidents_list.add_argument("--store", required=True)
    incidents_list.add_argument("--json", action="store_true")
    incidents_list.set_defaults(func=_cmd_incidents)

    shadow = subparsers.add_parser(
        "shadow", help="run the engine over a suite and score outcomes"
    )
    shadow.add_argument("--suite-dir", required=True)
    shadow.add_argument("--config", default=None)
    shadow.add_argument("--registry", default=None)
    shadow.add_argument("--oracle-dir", default=None, help="ground-truth vault root")
    shadow.add_argument(
        "--with-registry",
        action="store_true",
        help="load generator-declared events from the vault (plumbing only)",
    )
    shadow.add_argument("--out", default=None)
    shadow.add_argument("--labels-out", default=None, help="append oracle labels")
    shadow.add_argument(
        "--prequential-store",
        default=None,
        help="append across-load calibration residuals",
    )
    shadow.set_defaults(func=_cmd_shadow)

    evidence = subparsers.add_parser(
        "evidence", help="run an allowlisted evidence query over a case"
    )
    _add_scenario_arguments(evidence)
    evidence.add_argument("--provider", default=None)
    evidence.add_argument("--run-id", default=None)
    evidence.add_argument("--query", required=True, choices=ALLOWED_QUERIES)
    evidence.add_argument("--limit", type=int, default=50)
    evidence.add_argument(
        "--filter", action="append", default=None, help="KEY=VALUE, repeatable"
    )
    evidence.add_argument("--json", action="store_true")
    evidence.set_defaults(func=_cmd_evidence)

    prequential = subparsers.add_parser(
        "prequential", help="inspect the across-load calibration pool"
    )
    prequential.add_argument("--store", required=True)
    prequential.add_argument("--as-of", type=int, default=None)
    prequential.add_argument("--target-week", type=int, default=None)
    prequential.add_argument("--z", type=float, default=None)
    prequential.add_argument("--alpha", type=float, default=0.05)
    prequential.set_defaults(func=_cmd_prequential)

    cohort = subparsers.add_parser(
        "cohort", help="run a frozen dev/held-out cohort evaluation"
    )
    cohort.add_argument("--plan", default=None, help="cohort plan JSON")
    cohort.add_argument("--profile", default=None)
    cohort.add_argument("--scenarios-per-family", type=int, default=None)
    cohort.add_argument("--dev-seed", action="append", type=int, default=None)
    cohort.add_argument("--heldout-seed", action="append", type=int, default=None)
    cohort.add_argument("--workdir", default="reports/cohort/work")
    cohort.add_argument("--out", default=None)
    cohort.add_argument("--json", action="store_true")
    cohort.set_defaults(func=_cmd_cohort)

    tspulse = subparsers.add_parser(
        "tspulse-bench",
        help="research benchmark: TSPulse embeddings of revision series",
    )
    tspulse.add_argument("--suite-dir", required=True)
    tspulse.add_argument("--oracle-dir", default=None, help="ground-truth vault root")
    tspulse.add_argument("--config", default=None)
    tspulse.add_argument("--out", default=None)
    tspulse.add_argument("--max-scenarios", type=int, default=None)
    tspulse.add_argument(
        "--strict-length",
        action="store_true",
        help="disable the 512-point linear resample (weekly series will be gated)",
    )
    tspulse.add_argument("--json", action="store_true")
    tspulse.set_defaults(func=_cmd_tspulse_bench)

    report = subparsers.add_parser(
        "report", help="write a Markdown and machine report for a run"
    )
    _add_scenario_arguments(report)
    report.add_argument("--provider", default=None)
    report.add_argument("--run-id", default=None)
    report.add_argument("--out", required=True)
    report.add_argument("--incident-store", default=None)
    report.add_argument("--no-brief", action="store_true")
    report.set_defaults(func=_cmd_report)

    delta_info = subparsers.add_parser(
        "delta-info", help="inspect a Delta table and its version history"
    )
    delta_info.add_argument("--uri", required=True)
    delta_info.add_argument("--storage-options", default=None, help="JSON object")
    delta_info.add_argument("--json", action="store_true")
    delta_info.set_defaults(func=_cmd_delta_info)

    delta_run = subparsers.add_parser(
        "delta-run", help="run QC directly over Delta table versions"
    )
    delta_run.add_argument("--uri", required=True)
    delta_run.add_argument("--current", default=None)
    delta_run.add_argument("--previous", default=None)
    delta_run.add_argument("--stage", default="warehouse")
    delta_run.add_argument(
        "--stage-tables", default=None, help="JSON object of stage -> table uri"
    )
    delta_run.add_argument("--storage-options", default=None, help="JSON object")
    delta_run.add_argument("--config", default=None)
    delta_run.add_argument("--provider", default=None)
    delta_run.add_argument("--run-id", default=None)
    delta_run.add_argument("--json", action="store_true")
    delta_run.set_defaults(func=_cmd_delta_run)

    relationships = subparsers.add_parser(
        "relationships", help="manage the entity relationship store"
    )
    relationships_sub = relationships.add_subparsers(
        dest="relationship_command", required=True
    )
    relationships_list = relationships_sub.add_parser(
        "list", help="list stored relationships"
    )
    relationships_list.add_argument("--store", required=True)
    relationships_list.add_argument("--json", action="store_true")
    relationships_list.set_defaults(func=_cmd_relationships)

    def _add_outcome_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--root-cause", default=None)
        parser.add_argument("--origin", default=None)
        parser.add_argument("--severity", default=None)
        parser.add_argument("--resolution", default=None)
        parser.add_argument("--summary", default=None)
        parser.add_argument("--analyst", default=None)
        parser.add_argument("--confirmed", action="store_true")
        parser.add_argument("--created", default=None)

    store = subparsers.add_parser(
        "store", help="confirmed-only run, outcome and registry store"
    )
    store_sub = store.add_subparsers(dest="store_command", required=True)

    store_add_run = store_sub.add_parser(
        "add-run", help="run a scenario and record it (optional outcome)"
    )
    _add_scenario_arguments(store_add_run)
    store_add_run.add_argument("--store", required=True)
    store_add_run.add_argument("--provider", default=None)
    store_add_run.add_argument("--run-id", default=None)
    _add_outcome_arguments(store_add_run)
    store_add_run.set_defaults(func=_cmd_store_add_run)

    store_add_outcome = store_sub.add_parser(
        "add-outcome", help="append an analyst outcome to a recorded run"
    )
    store_add_outcome.add_argument("--store", required=True)
    store_add_outcome.add_argument("--run-id", required=True)
    _add_outcome_arguments(store_add_outcome)
    store_add_outcome.set_defaults(func=_cmd_store_add_outcome)

    store_list = store_sub.add_parser("list", help="list recorded runs")
    store_list.add_argument("--store", required=True)
    store_list.add_argument("--confirmed-only", action="store_true")
    store_list.add_argument("--json", action="store_true")
    store_list.set_defaults(func=_cmd_store_list)

    store_incidents = store_sub.add_parser(
        "incidents", help="confirmed-only similar incident retrieval"
    )
    _add_scenario_arguments(store_incidents)
    store_incidents.add_argument("--store", required=True)
    store_incidents.add_argument("--provider", default=None)
    store_incidents.add_argument("--k", type=int, default=3)
    store_incidents.add_argument("--json", action="store_true")
    store_incidents.set_defaults(func=_cmd_store_incidents)

    store_import = store_sub.add_parser(
        "import", help="import analyst outcomes from CSV or a Delta table"
    )
    store_import.add_argument("--store", required=True)
    source = store_import.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", default=None)
    source.add_argument("--delta", default=None)
    store_import.add_argument("--storage-options", default=None, help="JSON object")
    store_import.add_argument("--dry-run", action="store_true")
    store_import.add_argument("--json", action="store_true")
    store_import.set_defaults(func=_cmd_store_import)

    onboard = subparsers.add_parser(
        "onboard", help="profile a Delta table and assess QC readiness"
    )
    onboard.add_argument("--uri", required=True)
    onboard.add_argument("--version", type=int, default=None)
    onboard.add_argument("--previous", type=int, default=None)
    onboard.add_argument("--current", type=int, default=None)
    onboard.add_argument("--stage", default="warehouse")
    onboard.add_argument("--storage-options", default=None, help="JSON object")
    onboard.add_argument("--sample-rows", type=int, default=50000)
    onboard.add_argument("--count", action="store_true", help="exact row count")
    onboard.add_argument(
        "--alias",
        action="append",
        default=None,
        help="production=canonical, repeatable (e.g. --alias pfc=product_id)",
    )
    onboard.add_argument("--out", default=None, help="write proposed config YAML")
    onboard.add_argument("--json", action="store_true")
    onboard.set_defaults(func=_cmd_onboard)

    drift = subparsers.add_parser(
        "drift", help="monitor prequential calibration for drift"
    )
    drift.add_argument("--store", required=True)
    drift.add_argument("--as-of", type=int, required=True)
    drift.add_argument("--target-week", type=int, required=True)
    drift.add_argument("--alpha", type=float, default=0.05)
    drift.add_argument("--ratio-threshold", type=float, default=1.5)
    drift.add_argument("--coverage-margin", type=float, default=0.1)
    drift.add_argument("--max-records", type=int, default=500)
    drift.add_argument("--json", action="store_true")
    drift.set_defaults(func=_cmd_drift)

    champion = subparsers.add_parser(
        "champion", help="bake off decision providers on a held-out cohort"
    )
    champion_source = champion.add_mutually_exclusive_group(required=True)
    champion_source.add_argument(
        "--train-labels", default=None, help="label store JSONL (analyst or oracle)"
    )
    champion_source.add_argument(
        "--store", default=None, help="SQLite store with confirmed outcomes"
    )
    champion.add_argument("--suite-dir", required=True, help="held-out eval suite")
    champion.add_argument("--oracle-dir", default=None, help="ground-truth vault root")
    champion.add_argument(
        "--eval-store", default=None, help="SQLite store with held-out outcomes"
    )
    champion.add_argument("--config", default=None)
    champion.add_argument("--text-embedder", default=None)
    champion.add_argument("--text-revision", default=None)
    champion.add_argument("--systemone-url", default=None)
    champion.add_argument("--min-overall-accuracy", type=float, default=None)
    champion.add_argument("--min-cause-accuracy", type=float, default=None)
    champion.add_argument("--validation-fraction", type=float, default=0.3)
    champion.add_argument("--seed", type=int, default=0)
    champion.add_argument("--out", default=None)
    champion.add_argument("--json", action="store_true")
    champion.set_defaults(func=_cmd_champion)

    simulate = subparsers.add_parser(
        "simulate-analyst",
        help="write synthetic analyst outcomes to a store (provenance: synthetic)",
    )
    simulate.add_argument("--suite-dir", required=True)
    simulate.add_argument("--oracle-dir", default=None, help="ground-truth vault root")
    simulate.add_argument("--store", required=True)
    simulate.add_argument(
        "--profile", choices=sorted(ANALYST_PROFILES), default="typical"
    )
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--json", action="store_true")
    simulate.set_defaults(func=_cmd_simulate_analyst)

    replay_parser = subparsers.add_parser(
        "replay", help="replay pinned version pairs with leakage guards"
    )
    replay_parser.add_argument("--plan", required=True, help="replay plan JSON")
    replay_parser.add_argument("--out", required=True)
    replay_parser.add_argument("--uri", default=None, help="default source uri")
    replay_parser.add_argument("--config", default=None)
    replay_parser.add_argument("--storage-options", default=None, help="JSON object")
    replay_parser.add_argument("--json", action="store_true")
    replay_parser.set_defaults(func=_cmd_replay)

    rca = subparsers.add_parser(
        "rca", help="bounded RCA investigation over a completed run"
    )
    _add_scenario_arguments(rca)
    rca.add_argument("--provider", default=None)
    rca.add_argument("--steps", type=int, default=4)
    rca.add_argument("--limit", type=int, default=50)
    rca.add_argument("--json", action="store_true")
    rca.set_defaults(func=_cmd_rca)

    expectations = subparsers.add_parser(
        "expectations", help="explain ratio alerts with scoped approvals"
    )
    _add_scenario_arguments(expectations)
    expectations.add_argument("--provider", default=None)
    expectations.add_argument("--expectations", required=True)
    expectations.add_argument("--as-of", default=None)
    expectations.add_argument("--json", action="store_true")
    expectations.set_defaults(func=_cmd_expectations)

    prequential_forecast = subparsers.add_parser(
        "prequential-forecast",
        help="forecast consecutive Delta versions with point-in-time calibration",
    )
    prequential_forecast.add_argument("--uri", required=True)
    prequential_forecast.add_argument(
        "--versions", required=True, help="comma-separated version ids in load order"
    )
    prequential_forecast.add_argument("--stage", default="warehouse")
    prequential_forecast.add_argument("--storage-options", default=None, help="JSON object")
    prequential_forecast.add_argument("--scope", default=None)
    prequential_forecast.add_argument("--min-samples", type=int, default=9)
    prequential_forecast.add_argument("--config", default=None)
    prequential_forecast.add_argument("--out", default=None, help="append pool JSONL")
    prequential_forecast.add_argument("--json", action="store_true")
    prequential_forecast.set_defaults(func=_cmd_prequential_forecast)

    weekly = subparsers.add_parser(
        "weekly",
        help="run the full weekly chain after a refresh (scheduler entry point)",
    )
    weekly.add_argument("--uri", required=True)
    weekly.add_argument("--stage", default="warehouse")
    weekly.add_argument("--config", default=None)
    weekly.add_argument("--store", default=None, help="SQLite store for runs/outcomes")
    weekly.add_argument(
        "--calibration-store", default=None, help="prequential pool JSONL"
    )
    weekly.add_argument("--out", default="reports/weekly")
    weekly.add_argument("--current", default=None)
    weekly.add_argument("--previous", default=None)
    weekly.add_argument("--expectations", default=None)
    weekly.add_argument("--reference-uri", default=None)
    weekly.add_argument("--reference-spec", default=None)
    weekly.add_argument("--storage-options", default=None, help="JSON object")
    weekly.add_argument("--min-samples", type=int, default=9)
    weekly.add_argument("--provider", default=None)
    weekly.add_argument("--force", action="store_true")
    weekly.add_argument(
        "--allow-investigate",
        action="store_true",
        help="return 0 even when the run requires investigation",
    )
    weekly.add_argument("--json", action="store_true")
    weekly.set_defaults(func=_cmd_weekly)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
