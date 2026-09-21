"""Freeze analyst evidence before training or selection; fail closed on sparse labels."""

from __future__ import annotations

import json
from pathlib import Path

from .assessment import digest
from .jsonutil import dumps
from .store import SqliteStore

SPLITS = ("train", "calibration", "development", "test")


def freeze_store_cohort(path: str, cutoff: str, out: str | None = None) -> dict:
    if not cutoff:
        raise ValueError("store cohorts require an explicit observation cutoff")
    from .store import observation_time

    cutoff = observation_time(cutoff)
    with SqliteStore(path) as store:
        rows = store.connection.execute(
            """
            SELECT r.*, o.outcome_id, o.created AS label_observed_at, o.provenance,
                   o.analyst, o.root_cause, o.likely_origin, o.severity,
                   o.requires_investigation, o.confirmed, o.incident_group
            FROM runs r JOIN outcomes o ON o.outcome_id = (
                SELECT outcome_id FROM outcomes
                WHERE run_id = r.run_id AND created != '' AND created <= ?
                ORDER BY created DESC, outcome_id DESC LIMIT 1
            ) WHERE r.created != '' AND r.created <= ?
            ORDER BY r.created, r.run_id
        """,
            (cutoff, cutoff),
        ).fetchall()
    cases = []
    for row in rows:
        if not row["confirmed"] or row["provenance"] != "analyst" or not row["analyst"]:
            continue
        payload = json.loads(row["payload"])
        # No invented independence: missing incident grouping keeps a dataset together.
        group = row["incident_group"] or payload.get("incident_group") or row["dataset"]
        cases.append(
            {
                "run_id": row["run_id"],
                "group": group,
                "observed_at": row["created"],
                "outcome_revision": row["outcome_id"],
                "label_observed_at": row["label_observed_at"],
                "provenance": row["provenance"],
                "analyst": row["analyst"],
                "snapshot_manifest": payload.get("snapshot_manifest"),
                "evidence_hash": digest(payload),
                "evidence": payload,
                "features": json.loads(row["features"]) if row["features"] else None,
                "text": row["evidence_text"],
                "labels": {
                    key: row[key]
                    for key in (
                        "root_cause",
                        "likely_origin",
                        "severity",
                        "requires_investigation",
                    )
                },
            }
        )
    # Union repeated snapshots and related incidents before assigning chronology.
    parents = list(range(len(cases)))

    def root(index):
        while parents[index] != index:
            index = parents[index]
        return index

    seen: dict[tuple, int] = {}
    for index, case in enumerate(cases):
        manifest = case["snapshot_manifest"] or {}
        tokens: list[tuple] = [("incident", case["group"])]
        tokens += [
            ("snapshot", manifest.get("source"), snapshot.get("version"))
            for snapshot in manifest.get("snapshots", [])[-1:]
        ]
        for token in tokens:
            if token in seen:
                parents[root(index)] = root(seen[token])
            else:
                seen[token] = index
    for index, case in enumerate(cases):
        case["group"] = cases[root(index)]["group"]
    groups = sorted(
        {c["group"] for c in cases},
        key=lambda group: max(c["observed_at"] for c in cases if c["group"] == group),
    )
    enough = len(groups) >= 4 and all(c["snapshot_manifest"] for c in cases)
    for case in cases:
        case["split"] = (
            SPLITS[min(3, groups.index(case["group"]) * 4 // len(groups))]
            if enough
            else None
        )
    if enough:
        for left, right in zip(SPLITS, SPLITS[1:]):
            earlier = [c["observed_at"] for c in cases if c["split"] == left]
            later = [c["observed_at"] for c in cases if c["split"] == right]
            if not earlier or not later or max(earlier) >= min(later):
                enough = False
        if not enough:
            for case in cases:
                case["split"] = None
    report = {
        "schema_version": 1,
        "source": "store",
        "cutoff": cutoff,
        "cases": cases,
        "independent_groups": len(groups),
        "status": "FROZEN" if enough else "INSUFFICIENT_EVIDENCE",
        "evaluation_complete": False,
        "production_eligible": False,
        "limitations": [
            "A frozen cohort is not an evaluation certificate.",
            "Unknown incident grouping is conservatively grouped by dataset.",
        ],
    }
    report["manifest_hash"] = digest(report)
    if out:
        target = Path(out)
        target.mkdir(parents=True, exist_ok=True)
        file = target / "store-cohort.json"
        content = dumps(report, indent=2, sort_keys=True)
        if file.exists() and file.read_text() != content:
            raise ValueError("frozen cohort exists; choose a new output directory")
        file.write_text(content)
    return report


def training_records_from_cohort(manifest: dict):
    from .decisions import FEATURE_VERSION
    from .labels import LabelRecord

    if manifest.get("status") != "FROZEN" or digest(
        {k: v for k, v in manifest.items() if k != "manifest_hash"}
    ) != manifest.get("manifest_hash"):
        raise ValueError("training requires a valid frozen cohort")
    records = []
    for case in manifest["cases"]:
        labels = dict(case["labels"])
        labels["likely_cause"] = labels.pop("root_cause")
        if (
            any(value is None for value in labels.values())
            or case["features"] is None
            or not case["text"]
        ):
            raise ValueError(
                "INSUFFICIENT_EVIDENCE: complete labels, features and text required"
            )
        labels["requires_investigation"] = str(bool(labels["requires_investigation"]))
        records.append(
            LabelRecord(
                case["run_id"],
                "analyst",
                labels["likely_cause"],
                labels,
                case["features"],
                FEATURE_VERSION,
                {
                    "incident_group": case["group"],
                    "observed_at": case["observed_at"],
                    "frozen_split": case["split"],
                    "cohort_hash": manifest["manifest_hash"],
                },
                case["text"],
            )
        )
    return records


def production_eligibility(
    evaluation: dict, operational_checks: dict[str, bool]
) -> dict:
    reasons = []
    expected = evaluation.get("pinned_artifact_hash")
    if (
        not expected
        or digest({k: v for k, v in evaluation.items() if k != "pinned_artifact_hash"})
        != expected
    ):
        reasons.append("evaluation artifact is not pinned")
    if evaluation.get("source") != "store" or evaluation.get("provenance") != "analyst":
        reasons.append("real analyst provenance required")
    if not evaluation.get("evaluation_complete") or not evaluation.get("gates_passed"):
        reasons.append("independent evaluation and confidence-bound gates required")
    required = {
        "contracts",
        "point_in_time",
        "migration",
        "crash_recovery",
        "read_only",
        "provider_boundary",
    }
    if not required.issubset(operational_checks) or not all(
        operational_checks.values()
    ):
        reasons.append("operational checks incomplete")
    if evaluation.get("split") != "test":
        reasons.append("untouched test confirmation required")
    return {"production_eligible": not reasons, "reasons": reasons}


def evaluate_frozen_cohort(
    manifest: dict, provider, split: str = "development", selection: dict | None = None
) -> dict:
    """Score one pinned challenger against recorded incumbent on the same cases."""
    import math
    import resource
    import time
    from types import SimpleNamespace

    import numpy as np

    from .conformal import wilson_interval
    from .decisions import default_fields

    expected_hash = manifest.get("manifest_hash")
    if (
        digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
        != expected_hash
    ):
        raise ValueError("frozen cohort hash mismatch")
    if split not in ("development", "test"):
        raise ValueError("evaluation uses development or untouched test only")
    serialize = getattr(provider, "to_dict", None)
    identity = (
        serialize() if serialize else getattr(provider, "artifact_identity", None)
    )
    if identity is None:
        raise ValueError("challenger requires immutable artifact identity")
    identity = {
        "artifact": identity,
        "endpoint": getattr(provider, "url", None),
        "model": getattr(provider, "model", None),
        "timeout": getattr(provider, "timeout", None),
        "state_limit": getattr(provider, "state_limit", None),
    }
    provider_hash = digest(identity)
    if split == "test" and (
        not selection
        or selection.get("provider_hash") != provider_hash
        or selection.get("manifest_hash") != expected_hash
        or selection.get("split") != "development"
        or selection.get("status") != "EVALUATED"
        or not selection.get("gates_passed")
        or digest({k: v for k, v in selection.items() if k != "pinned_artifact_hash"})
        != selection.get("pinned_artifact_hash")
    ):
        raise ValueError("test requires the frozen development selection")
    metadata = getattr(provider, "metadata", {})
    if hasattr(provider, "heads"):
        if metadata.get("cohort_hash") != expected_hash:
            raise ValueError("learned challenger must train from this frozen cohort")
        used = set(metadata.get("fit_groups", []))
        heldout = {
            case["group"]
            for case in manifest["cases"]
            if case["split"] in ("development", "test")
        }
        if used & heldout or not used:
            raise ValueError(
                "training/calibration overlap or missing fit-group provenance"
            )
    cases = [case for case in manifest["cases"] if case["split"] == split]
    rows = []
    specs = default_fields()
    started = time.perf_counter()
    for case in cases:
        evidence = case["evidence"]
        result = SimpleNamespace(
            run_id=case["run_id"],
            dataset=evidence.get("dataset"),
            status=evidence["status"],
            machine=dict(evidence),
            recorded_features=case["features"],
            recorded_text=case["text"],
        )
        tick = time.perf_counter()
        try:
            decisions = provider.decide(result)
            prediction = decisions.to_dict()
            error = None
        except Exception as exc:  # noqa: BLE001 - measured abstention
            prediction = {"values": {}, "requires_investigation": True}
            error = str(exc)
        labels = dict(case["labels"])
        labels["likely_cause"] = labels.pop("root_cause")
        labels["requires_investigation"] = (
            str(bool(labels["requires_investigation"]))
            if labels["requires_investigation"] is not None
            else None
        )
        policy_review = evidence["status"] not in ("PASS", "PASS_WITH_EXPLANATION")
        effective_review = policy_review or bool(prediction["requires_investigation"])
        incumbent = evidence.get("decision") or {
            "requires_investigation": policy_review
        }
        rows.append(
            {
                "run_id": case["run_id"],
                "group": case["group"],
                "labels": labels,
                "prediction": prediction,
                "abstained": error is not None,
                "error": error,
                "latency_seconds": time.perf_counter() - tick,
                "detection": policy_review,
                "actionable_review": effective_review,
                "incumbent_review": bool(incumbent["requires_investigation"]),
            }
        )
    fields = {}
    for spec in specs:
        valid = [r for r in rows if r["labels"].get(spec.name) in spec.values]
        missing_classes = sorted(
            set(spec.values) - {r["labels"][spec.name] for r in valid}
        )
        brier, log_loss, correct = [], [], []
        calibration = []
        for row in valid:
            predicted = row["prediction"]["values"].get(spec.name, {})
            probabilities = predicted.get("probabilities", {})
            truth = row["labels"][spec.name]
            correct.append(str(predicted.get("value")) == truth)
            if probabilities and set(probabilities) == set(spec.values):
                brier.append(
                    sum((probabilities[k] - int(k == truth)) ** 2 for k in spec.values)
                )
                log_loss.append(-math.log(max(float(probabilities[truth]), 1e-15)))
                winner = max(probabilities, key=probabilities.get)
                calibration.append((float(probabilities[winner]), int(winner == truth)))
        bins = []
        for i in range(10):
            members = [(p, y) for p, y in calibration if min(9, int(p * 10)) == i]
            bins.append(
                {
                    "lower": i / 10,
                    "upper": (i + 1) / 10,
                    "n": len(members),
                    "confidence": float(np.mean([p for p, _ in members]))
                    if members
                    else None,
                    "accuracy": float(np.mean([y for _, y in members]))
                    if members
                    else None,
                }
            )
        ece = (
            (
                sum(
                    b["n"] * abs(b["confidence"] - b["accuracy"])
                    for b in bins
                    if b["n"]
                    and b["confidence"] is not None
                    and b["accuracy"] is not None
                )
                / len(calibration)
            )
            if calibration
            else None
        )
        fields[spec.name] = {
            "calibration_bins": bins,
            "expected_calibration_error": ece,
            "n": len(valid),
            "missing_classes": missing_classes,
            "accuracy": float(np.mean(correct)) if correct else None,
            "brier": float(np.mean(brier)) if brier else None,
            "log_loss": float(np.mean(log_loss)) if log_loss else None,
            "undefined_reason": None
            if brier
            else "no complete probability distributions",
            "by_class": {
                label: {
                    "n": sum(r["labels"][spec.name] == label for r in valid),
                    "correct": sum(
                        str(r["prediction"]["values"].get(spec.name, {}).get("value"))
                        == label
                        for r in valid
                        if r["labels"][spec.name] == label
                    ),
                }
                for label in spec.values
            },
        }
    faults = [r for r in rows if r["labels"].get("requires_investigation") == "True"]
    controls = [r for r in rows if r["labels"].get("requires_investigation") == "False"]
    fault_groups = {r["group"] for r in faults}
    control_groups = {r["group"] for r in controls}
    # Repeated snapshots are not independent trials. A fault group succeeds
    # only if every actionable case is reviewed; any false alert fails a control group.
    detection_ci = wilson_interval(
        sum(
            all(r["actionable_review"] for r in faults if r["group"] == group)
            for group in fault_groups
        ),
        len(fault_groups),
    )
    fpr_ci = wilson_interval(
        sum(
            any(r["actionable_review"] for r in controls if r["group"] == group)
            for group in control_groups
        ),
        len(control_groups),
    )
    groups = sorted({r["group"] for r in rows})
    paired = []
    for group in groups:
        group_rows = [
            r
            for r in rows
            if r["group"] == group
            and r["labels"].get("requires_investigation") is not None
        ]
        differences = [
            int(
                r["actionable_review"]
                == (r["labels"]["requires_investigation"] == "True")
            )
            - int(
                r["incumbent_review"]
                == (r["labels"]["requires_investigation"] == "True")
            )
            for r in group_rows
        ]
        if differences:
            paired.append(float(np.mean(differences)))
    paired_ci = None
    if len(paired) >= 2:
        means = (
            np.random.default_rng(0)
            .choice(paired, (2000, len(paired)), replace=True)
            .mean(axis=1)
        )
        paired_ci = np.quantile(means, [0.025, 0.975]).tolist()
    enough = bool(
        faults
        and controls
        and len(groups) >= 2
        and all(not field["missing_classes"] for field in fields.values())
    )
    passed = (
        enough
        and detection_ci[0] >= 0.9
        and fpr_ci[1] <= 0.1
        and paired_ci is not None
        and paired_ci[0] >= 0
        and not any(r["abstained"] for r in rows)
    )
    report = {
        "schema_version": 1,
        "source": "store",
        "provenance": "analyst",
        "manifest_hash": expected_hash,
        "provider_hash": provider_hash,
        "split": split,
        "status": "EVALUATED" if enough else "INSUFFICIENT_EVIDENCE",
        "fields": fields,
        "detection_95ci": detection_ci,
        "confidence_unit": "independent_incident",
        "fault_groups": len(fault_groups),
        "control_groups": len(control_groups),
        "false_positive_95ci": fpr_ci,
        "paired_incident_accuracy_delta_95ci": paired_ci,
        "review_volume": sum(r["actionable_review"] for r in rows),
        "abstentions": sum(r["abstained"] for r in rows),
        "unknown_causes": sum(
            r["prediction"]["values"].get("likely_cause", {}).get("value") == "UNKNOWN"
            for r in rows
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gates_passed": passed,
        "evaluation_complete": enough and split == "test",
        "production_eligible": False,
        "cases": rows,
    }
    report["pinned_artifact_hash"] = digest(report)
    return report
