"""CLI for the QC engine.

    qc run --scenario-dir data/suites/demo/scenario-0000
    qc shadow --suite-dir data/suites/demo --out reports/shadow/demo
    qc cohort --plan config/cohort.json --out reports/cohort/v1
    qc weekly --uri ./lake/fact --store data/qc.db --out reports/weekly

The scenario adapter lives in ``qcgen`` and is imported lazily so the engine
package itself has no dependency on the generator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .cohort import CohortPlan, run_cohort
from .config import DatasetConfig, load_dataset_config
from .decisions import DecisionSet
from .expectations import apply_expectations, load_expectations
from .jsonutil import dumps as _json_dumps
from .reporting import write_report
from .run import QCRunResult, run_qc
from .shadow import run_shadow
from .source import mapped
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


def _print_decisions(decisions: DecisionSet | None) -> None:
    if decisions is None:
        return
    print(f"DECISION {decisions.provider}")
    for name in ("likely_cause", "likely_causes", "likely_origin", "severity", "requires_investigation"):
        value = decisions.get(name)
        if value is None:
            continue
        if value.kind == "set":
            print(f"  {name:<24} {', '.join(value.value) or '-'}")
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
        sensitivity = result.temporal.sensitivity()
        if sensitivity:
            print("SENSITIVITY (median drop a lone leaf needs to be flagged, 50% power; conservative)")
            for key, entry in sensitivity.items():
                print(
                    f"  {key:<24} {entry['median_detectable_change']:.0%} "
                    f"(80% power {entry['median_detectable_change_80']:.0%}; "
                    f"least sensitive {entry['least_sensitive']})"
                )
    _print_decisions(result.decisions)
    if result.reasons:
        print("REASONS")
        for reason in result.reasons:
            print(f"  {reason}")


def _run_scenario(args, config, registry=None):
    from qcgen.sources import ScenarioSource

    scenario_dir = Path(args.scenario_dir)
    if registry is None and getattr(args, "registry", None):
        from .registry import FileRegistry

        registry = FileRegistry(args.registry)

    raw_source = ScenarioSource(scenario_dir)
    source = mapped(raw_source, config.column_map_dict())
    current = args.current if getattr(args, "current", None) else None
    previous = args.previous if getattr(args, "previous", None) else None
    from .versions import select_versions
    current, previous = select_versions(source.list_versions(), current, previous)
    if current is None or previous is None:
        raise SystemExit("need at least two versions under the scenario directory")
    return run_qc(
        source,
        current,
        previous,
        config,
        registry=registry,
        run_id=getattr(args, "run_id", None),
    )


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    if args.no_decisions:
        config = replace(config, decision_enabled=False)
    result = _run_scenario(args, config)
    if args.json:
        print(_json_dumps(result.machine, indent=2, default=_json_default))
    else:
        _print_human(result)
    return 0


def _scenario_names(scenario_dir: Path, version: str) -> dict[str, str]:
    """Store and category display names from a scenario's dimension tables, if present."""
    import pandas as pd

    names: dict[str, str] = {}
    dims = scenario_dir / "versions" / version / "dims"
    for table, key, label in (("stores", "store_id", "store_name"), ("commodities", "commodity_id", "commodity_name")):
        path = dims / f"{table}.parquet"
        if path.exists():
            frame = pd.read_parquet(path, columns=[key, label])
            names.update(zip(frame[key].astype(str), frame[label].astype(str), strict=True))
    return names


def _cmd_notices(args: argparse.Namespace) -> int:
    from .events import save_registry
    from .notices import PatternMatcher, SystemOneMatcher, draft, load_notices

    if args.matcher == "systemone" and not args.endpoint:
        raise SystemExit("--matcher systemone needs --endpoint")
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    result = _run_scenario(args, config)
    matcher: Any = (
        SystemOneMatcher(args.endpoint, args.model, threshold=args.threshold)
        if args.matcher == "systemone"
        else PatternMatcher()
    )
    current = result.version_pair.current_id if result.version_pair is not None else ""
    report = draft(result, load_notices(args.notices), matcher, _scenario_names(Path(args.scenario_dir), current))
    if args.out:
        save_registry(report["registry_drafts"], args.out)
    if args.json:
        print(_json_dumps(report, indent=2, default=_json_default))
        return 0
    print(f"NOTICES dataset={report['dataset']} status={report['status']} matcher={report['matcher']}")
    for item in report["matches"]:
        notice, cand = item["notice"], item["candidate"]
        target = cand["description"] if cand else "no observed change"
        checks = item["details"].get("checks")
        weakest = f"  (weakest check {min(checks, key=checks.get)} {min(checks.values()):.2f})" if checks else ""
        print(f"  {notice['id']:<12} {item['action']:<15} {target}{weakest}")
        print(f"  {'':<12} {notice['text']}")
    drafts = report["registry_drafts"]
    if drafts:
        where = f" in {args.out}" if args.out else " (pass --out to save them)"
        print(f"{len(drafts)} draft registry entr{'y' if len(drafts) == 1 else 'ies'}{where}: unapproved; "
              "set approved_by, approved_at and confirmed before passing them as --registry.")
    return 0


def _cmd_shadow(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    summary = run_shadow(
        args.suite_dir,
        config=config,
        registry_path=args.registry,
        out_dir=args.out,
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
    return 0


def _add_scenario_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--current", default=None)
    parser.add_argument("--previous", default=None)
    parser.add_argument("--config", default=None, help="dataset YAML config")
    parser.add_argument("--registry", default=None, help="expected-event registry JSON")


def _cmd_report(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    result = _run_scenario(args, config)
    paths = write_report(result, args.out)
    print(f"report written: {paths['markdown']}")
    print(f"machine output: {paths['machine']}")
    return 0


def _cmd_delta_info(args: argparse.Namespace) -> int:
    from .delta import describe_delta_table

    options = json.loads(args.storage_options) if args.storage_options else None
    info = describe_delta_table(args.uri, options)
    if args.json:
        print(_json_dumps(info, indent=2, default=_json_default))
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
        version_map=json.loads(args.version_map) if args.version_map else None,
    )
    versions = delta_source.list_versions()
    from .versions import select_versions
    current, previous = select_versions(versions, args.current, args.previous)
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
    )
    if args.json:
        print(_json_dumps(result.machine, indent=2, default=_json_default))
    else:
        _print_human(result)
    for warning in getattr(source, "warnings", []):
        print(f"WARNING {warning}")
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
    result = run_cohort(
        plan,
        workdir=args.workdir,
        config=load_dataset_config(args.config) if args.config else None,
        out_dir=args.out,
        plan_path=args.plan,
    )
    # Exit 3 when a registered gate fails so a scheduler or CI job notices.
    code = 0 if result.gates_passed else 3
    if args.json:
        print(_json_dumps(result.to_dict(), indent=2, default=_json_default))
        return code
    print(
        f"cohort cases={result.metrics['cases']} "
        f"detection_rate={result.metrics['detection_rate']} "
        f"ci=[{result.metrics['detection_rate_ci_low']:.3f}, "
        f"{result.metrics['detection_rate_ci_high']:.3f}] "
        f"false_positive_rate={result.metrics['false_positive_rate']} "
        f"expected_match={result.metrics['expected_status_match_rate']} "
        f"reconstruction={result.metrics['mean_reconstruction_score']}"
    )
    for field_name, scores in result.metrics.get("labels", {}).items():
        if scores["n"]:
            print(
                f"  label {field_name}: accuracy={scores['accuracy']:.3f} "
                f"ci=[{scores['accuracy_ci_low']:.3f}, {scores['accuracy_ci_high']:.3f}] "
                f"n={scores['n']} (reported, not gated)"
            )
    for check in result.gate_results:
        state = check.get("status") or ("PASS" if check["passed"] else "FAIL")
        actual = check["actual"]
        shown = f"{actual:.3f}" if isinstance(actual, (int, float)) else "n/a"
        print(
            f"  {state} {check['gate']} actual={shown} "
            f"threshold={check['threshold']}"
        )
    print(
        f"  gates_passed={result.gates_passed} "
        f"plan_hash_verified={result.plan_hash_verified} "
        f"git_dirty={result.git_dirty} "
        f"code_sha256={result.code_sha256[:12]}"
    )
    if args.out:
        print(f"  output: {args.out}")
    return code


def _parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            low, high = (int(value) for value in part.split("-", 1))
            seeds.extend(range(low, high + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def _parse_pairs(
    text: str, default: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    if text == "default":
        return default
    if text == "none":
        return ()
    pairs: list[tuple[str, str]] = []
    for item in text.split(","):
        primary, separator, secondary = item.strip().partition("+")
        if not separator or not primary or not secondary:
            raise ValueError(f"invalid pair {item!r}; expected family+family")
        pairs.append((primary, secondary))
    return tuple(pairs)


def _cmd_sweep(args: argparse.Namespace) -> int:
    from .sweep import (
        DEFAULT_FAMILIES,
        DEFAULT_MAGNITUDES,
        DEFAULT_PAIRS,
        render_table,
        run_sweep,
        write_sweep,
    )

    overrides: dict[str, Any] = {}
    if args.config:
        load_dataset_config(args.config)  # validate before running anything
        overrides = dict(yaml.safe_load(Path(args.config).read_text()) or {})
    report = run_sweep(
        args.profile,
        _parse_seeds(args.seeds),
        families=(
            tuple(value.strip() for value in args.families.split(","))
            if args.families
            else DEFAULT_FAMILIES
        ),
        magnitudes=(
            tuple(float(value) for value in args.magnitudes.split(","))
            if args.magnitudes
            else DEFAULT_MAGNITUDES
        ),
        controls_per_seed=args.controls_per_seed,
        workdir=args.workdir,
        config_overrides=overrides,
        jobs=args.jobs,
        pairs=_parse_pairs(args.pairs, DEFAULT_PAIRS),
        keep_scenarios=args.keep_scenarios,
    )
    if args.out:
        write_sweep(report, args.out)
    if args.json:
        print(_json_dumps(report["summary"], indent=2, default=_json_default))
    else:
        print(f"sweep profile={report['profile']} seeds={len(report['seeds'])}")
        print(render_table(report))
        if args.out:
            print(f"output: {args.out}")
    return 0


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
            _json_dumps(
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


def _cmd_expectations(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config) if args.config else DatasetConfig()
    result = _run_scenario(args, config)
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
        print(_json_dumps(report, indent=2, default=_json_default))
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


def _explain_payload(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.report:
        directory = Path(args.report)
        machine_path = directory / "report.json"
        if not machine_path.is_file():
            raise ValueError(f"report.json not found under {directory}")
        return {
            "source": str(directory),
            "machine": json.loads(machine_path.read_text()),
        }
    from .store import SqliteStore

    if not args.store:
        raise ValueError("explain requires --report or --store")
    with SqliteStore(args.store) as store:
        assessment = (
            store.assessment(args.assessment)
            if args.assessment
            else store.latest_assessment(args.dataset)
        )
    if assessment is None:
        return None
    return {
        "source": f"{args.store}:{assessment['assessment_id']}",
        "assessment_id": assessment["assessment_id"],
        "machine": json.loads(assessment["artifacts"]["report.json"]),
    }


def _cmd_explain(args: argparse.Namespace) -> int:
    try:
        payload = _explain_payload(args)
    except (ValueError, OSError) as error:
        print(f"explain failed: {error}", file=sys.stderr)
        return 2
    if payload is None:
        print("no matching assessment found", file=sys.stderr)
        return 2
    if args.json:
        print(_json_dumps(payload, indent=2, default=_json_default))
        return 0
    machine = payload["machine"]
    print(f"status: {machine.get('status')}  run: {machine.get('run_id')}")
    findings = machine.get("findings", [])
    for item in findings:
        if item.get("outcome") == "PASS":
            continue
        print(
            f"  {item.get('check')} [{item.get('scope')}] "
            f"{item.get('outcome')} -> {item.get('disposition')}"
            + (
                f" explained_by={','.join(item['approval_ids'])}"
                if item.get("approval_ids")
                else ""
            )
        )
    return 0


def _dispatch_notifications(
    args: argparse.Namespace, result: Any
) -> list[Any]:
    """Fire the configured sinks. Never raises; returns one outcome per sink."""
    from .notify import (
        DEFAULT_MAX_PAYLOAD_BYTES,
        DEFAULT_NOTIFY_STATUSES,
        dispatch_all,
        load_notification_specs,
    )

    statuses = (
        tuple(part.strip() for part in args.notify_status.split(",") if part.strip())
        if args.notify_status
        else DEFAULT_NOTIFY_STATUSES
    )
    max_bytes = args.notify_max_bytes or DEFAULT_MAX_PAYLOAD_BYTES
    try:
        specs = load_notification_specs(args.notify)
    except Exception as error:  # noqa: BLE001 - never fail the assessment
        print(
            f"weekly: notification spec unusable: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return []
    return dispatch_all(specs, result, statuses=statuses, max_bytes=max_bytes)


def _cmd_weekly(args: argparse.Namespace) -> int:
    # Storage credentials may arrive via the environment so they never appear
    # in the process argument list.
    options_raw = args.storage_options or os.environ.get("QC_STORAGE_OPTIONS")
    try:
        config = load_dataset_config(args.config) if args.config else DatasetConfig()
        options = json.loads(options_raw) if options_raw else None
        result = run_weekly(
            args.uri,
            config=config,
            store_path=args.store,
            out_root=args.out,
            stage=args.stage,
            current=args.current,
            previous=args.previous,
            expectations_path=args.expectations,
            reference_uri=args.reference_uri,
            reference_spec_path=args.reference_spec,
            reference_version=args.reference_version,
            reference_stage=args.reference_stage,
            storage_options=options,
            force=args.force,
            stage_tables=json.loads(args.stage_tables) if args.stage_tables else None,
            dim_tables=json.loads(args.dim_tables) if args.dim_tables else None,
            version_map=json.loads(args.version_map) if args.version_map else None,
        )
    except Exception as error:  # noqa: BLE001 - the scheduler sees a failure
        print(
            f"weekly run failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(_json_dumps(result.to_dict(), indent=2, default=_json_default))
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
        print(f"  store_recorded={result.recorded} skipped={result.skipped}")
        for note in result.notes:
            print(f"  note: {note}")
        if result.report_dir:
            print(f"  report: {result.report_dir}")

    if args.notify:
        for outcome in _dispatch_notifications(args, result):
            if outcome.suppressed:
                continue
            if not outcome.delivered:
                # A sink failure is recorded, never propagated: the assessment
                # status and exit code above are already decided.
                print(
                    f"  note: notification to {outcome.sink} not delivered: "
                    f"{outcome.detail}",
                    file=sys.stderr,
                )
            elif not args.json:
                print(f"  notified {outcome.sink} ({outcome.bytes_sent} bytes)")

    return {
        "ALREADY_PROCESSED": 0,
        "LOCKED": 75,
        "INCOMPLETE": 4,
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
    run.add_argument("--no-decisions", action="store_true")
    run.add_argument("--run-id", default=None)
    run.add_argument("--json", action="store_true")
    run.set_defaults(func=_cmd_run)

    notices = subparsers.add_parser(
        "notices", help="match free-text change notices to observed changes; draft registry entries"
    )
    _add_scenario_arguments(notices)
    notices.add_argument("--notices", required=True, help="notices: JSON list, JSONL or one per line")
    notices.add_argument("--matcher", choices=("patterns", "systemone"), default="patterns")
    notices.add_argument("--endpoint", default=None, help="System One service base URL (https, or http to loopback)")
    notices.add_argument("--model", default="decision", help="model name sent to the service")
    notices.add_argument("--threshold", type=float, default=0.5, help="minimum probability for every check")
    notices.add_argument("--out", default=None, help="write the unapproved drafts as a registry file")
    notices.add_argument("--run-id", default=None)
    notices.add_argument("--json", action="store_true")
    notices.set_defaults(func=_cmd_notices)

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
    shadow.set_defaults(func=_cmd_shadow)

    cohort = subparsers.add_parser(
        "cohort", help="run a frozen dev/held-out cohort evaluation"
    )
    cohort.add_argument("--plan", default=None, help="cohort plan JSON")
    cohort.add_argument(
        "--config", default=None,
        help="dataset YAML config to evaluate (e.g. with an opt-in detector enabled)",
    )
    cohort.add_argument("--profile", default=None)
    cohort.add_argument("--scenarios-per-family", type=int, default=None)
    cohort.add_argument("--dev-seed", action="append", type=int, default=None)
    cohort.add_argument("--heldout-seed", action="append", type=int, default=None)
    cohort.add_argument("--workdir", default="reports/cohort/work")
    cohort.add_argument("--out", default=None)
    cohort.add_argument("--json", action="store_true")
    cohort.set_defaults(func=_cmd_cohort)

    sweep = subparsers.add_parser(
        "sweep", help="detection curves over fault size plus clean false alarms"
    )
    sweep.add_argument("--profile", default="small", help="generator profile (small, realistic, ...)")
    sweep.add_argument("--seeds", required=True, help="e.g. 5001-5010 or 5001,5003")
    sweep.add_argument("--families", default=None, help="comma-separated; default: all sizable families")
    sweep.add_argument("--magnitudes", default=None, help="comma-separated fractions in (0, 1]")
    sweep.add_argument("--controls-per-seed", type=int, default=4)
    sweep.add_argument(
        "--pairs", default="default",
        help="two-fault refreshes: 'default', 'none', or a+b,c+d",
    )
    sweep.add_argument("--config", default=None, help="dataset YAML overrides to evaluate")
    sweep.add_argument("--workdir", default="reports/sweep/work")
    sweep.add_argument("--jobs", type=int, default=1)
    sweep.add_argument(
        "--keep-scenarios", action="store_true",
        help="keep each generated scenario on disk (default: delete once scored)",
    )
    sweep.add_argument("--out", default=None)
    sweep.add_argument("--json", action="store_true")
    sweep.set_defaults(func=_cmd_sweep)

    report = subparsers.add_parser(
        "report", help="write a Markdown and machine report for a run"
    )
    _add_scenario_arguments(report)
    report.add_argument("--run-id", default=None)
    report.add_argument("--out", required=True)
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
    delta_run.add_argument("--version-map", default=None, help="JSON mapping snapshot -> stage/dim:name -> table version")
    delta_run.add_argument("--previous", default=None)
    delta_run.add_argument("--stage", default="warehouse")
    delta_run.add_argument(
        "--stage-tables", default=None, help="JSON object of stage -> table uri"
    )
    delta_run.add_argument("--storage-options", default=None, help="JSON object")
    delta_run.add_argument("--config", default=None)
    delta_run.add_argument("--run-id", default=None)
    delta_run.add_argument("--json", action="store_true")
    delta_run.set_defaults(func=_cmd_delta_run)

    onboard = subparsers.add_parser(
        "onboard", help="profile a Delta table and propose a dataset config"
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

    expectations = subparsers.add_parser(
        "expectations", help="explain ratio alerts with scoped approvals"
    )
    _add_scenario_arguments(expectations)
    expectations.add_argument("--expectations", required=True)
    expectations.add_argument("--as-of", default=None)
    expectations.add_argument("--json", action="store_true")
    expectations.set_defaults(func=_cmd_expectations)

    weekly = subparsers.add_parser(
        "weekly",
        help="run the full weekly chain after a refresh (scheduler entry point)",
    )
    weekly.add_argument("--uri", required=True)
    weekly.add_argument("--stage", default="warehouse")
    weekly.add_argument("--stage-tables", default=None, help="JSON stage to Delta URI map")
    weekly.add_argument("--dim-tables", default=None, help="JSON dimension to Delta URI map")
    weekly.add_argument("--version-map", default=None, help="JSON snapshot to stage/dim:name version map")
    weekly.add_argument("--config", default=None)
    weekly.add_argument("--store", default=None, help="SQLite assessment journal")
    weekly.add_argument("--out", default="reports/weekly")
    weekly.add_argument("--current", default=None)
    weekly.add_argument("--previous", default=None)
    weekly.add_argument("--expectations", default=None)
    weekly.add_argument("--reference-uri", default=None)
    weekly.add_argument("--reference-spec", default=None)
    weekly.add_argument("--reference-version", default=None)
    weekly.add_argument("--reference-stage", default=None)
    weekly.add_argument("--storage-options", default=None, help="JSON object")
    weekly.add_argument("--force", action="store_true")
    weekly.add_argument(
        "--notify",
        default=None,
        help="JSON file of notification sinks to fire when the status is actionable",
    )
    weekly.add_argument(
        "--notify-status",
        default=None,
        help="comma-separated statuses that trigger notification "
        "(default: INVESTIGATE,DATA_CONTRACT_FAILURE,CONTRACT_FAILURE,INCOMPLETE)",
    )
    weekly.add_argument(
        "--notify-max-bytes",
        type=int,
        default=16384,
        help="payload byte budget; optional sections are dropped, not truncated",
    )
    weekly.add_argument("--json", action="store_true")
    weekly.set_defaults(func=_cmd_weekly)

    explain = subparsers.add_parser(
        "explain", help="show an assessment's status and non-passing findings"
    )
    explain.add_argument("--report", default=None, help="report directory")
    explain.add_argument("--store", default=None, help="SQLite journal")
    explain.add_argument("--assessment", default=None, help="assessment id")
    explain.add_argument("--dataset", default=None, help="latest assessment for dataset")
    explain.add_argument("--json", action="store_true")
    explain.set_defaults(func=_cmd_explain)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
