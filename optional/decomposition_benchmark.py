"""Decomposition benchmark: narrow binary questions vs the flat multi-class head.

Runs the pre-registered comparison in ``config/decomposition-gate.json``. That
file was written before any result existed and is not edited afterwards; if the
hypothesis fails, this reports failure rather than moving the bar.

Held constant across both arms: the feature encoder and FEATURE_VERSION, the
frozen train/calibration/development/test split and its seed, and the optimizer
budget. Only the target formulation differs, so the measured delta isolates the
effect of decomposition.

Everything here is measured on synthetic oracle labels. A pass can never confer
``production_eligible`` (docs/claims.md).

Usage::

    .venv/bin/python optional/decomposition_benchmark.py
    .venv/bin/python optional/decomposition_benchmark.py --scenarios 40 --keep
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qc.config import DatasetConfig  # noqa: E402
from qc.decomposition import (  # noqa: E402
    evaluate_decomposed,
    train_decomposed_provider,
)
from qc.labels import LabelRecord, build_oracle_labels  # noqa: E402
from qc.training import (  # noqa: E402
    evaluate_decision_provider,
    label_metrics,
    split_records_four,
    train_decision_provider,
)
from qcgen.config import suite_config  # noqa: E402
from qcgen.scenarios import generate_suite  # noqa: E402

GATE_PATH = ROOT / "config" / "decomposition-gate.json"


def load_gate() -> dict[str, Any]:
    return json.loads(GATE_PATH.read_text())


def _group_by_scenario(records: list[LabelRecord]) -> list[LabelRecord]:
    """Scenario-level grouping, as declared in the gate.

    ``_group_key`` prefers ``metadata['incident_group']``, so setting it to the
    scenario id keeps sibling runs of one scenario in one split while leaving
    families represented on both sides - the realistic task, which is a new
    scenario from a family we have seen before.
    """
    return [
        LabelRecord(
            run_id=record.run_id,
            source=record.source,
            family=record.family,
            labels=dict(record.labels),
            features=list(record.features),
            feature_version=record.feature_version,
            metadata={**record.metadata, "incident_group": record.metadata.get("scenario_id")},
            text=record.text,
        )
        for record in records
    ]


def _per_record_predictions(
    provider: Any, records: list[LabelRecord], decomposed: bool
) -> dict[str, list[tuple[str, str]]]:
    """(expected, predicted) per field, in record order, for both arms."""
    from qc.decisions import field_index

    features = provider._standardize(
        np.asarray([record.features for record in records], dtype=float)
    )
    fields = (
        provider.field_distributions(features)
        if decomposed
        else {
            field_name: head.probabilities(features)
            for field_name, head in provider.heads.items()
        }
    )
    specs = field_index()
    result: dict[str, list[tuple[str, str]]] = {}
    for field_name, matrix in fields.items():
        if decomposed:
            values = list(specs[field_name].values)
        else:
            values = list(provider.heads[field_name].classes)
        guesses = [values[int(index)] for index in matrix.argmax(axis=1)]
        result[field_name] = [
            (record.labels.get(field_name, ""), guess)
            for record, guess in zip(records, guesses, strict=True)
        ]
    return result


def paired_bootstrap(
    expected: list[str],
    left: list[str],
    right: list[str],
    field: str,
    draws: int = 2000,
    seed: int = 0,
) -> dict[str, float | None]:
    """Paired bootstrap interval for ``right - left`` on macro-F1.

    Paired means both arms are resampled on the same case indices, so the
    interval reflects the difference and not the sample draw.
    """
    n = len(expected)
    if not n:
        return {"delta": None, "ci_low": None, "ci_high": None}
    base = label_metrics(field, expected, right)["macro_f1"] - label_metrics(
        field, expected, left
    )["macro_f1"]
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(draws):
        picks = rng.integers(0, n, size=n)
        sample_expected = [expected[int(i)] for i in picks]
        left_score = label_metrics(
            field, sample_expected, [left[int(i)] for i in picks]
        )["macro_f1"]
        right_score = label_metrics(
            field, sample_expected, [right[int(i)] for i in picks]
        )["macro_f1"]
        if left_score is None or right_score is None:
            continue
        deltas.append(float(right_score - left_score))
    if not deltas:
        return {"delta": base, "ci_low": None, "ci_high": None}
    ordered = sorted(deltas)
    return {
        "delta": base,
        "ci_low": float(ordered[int(0.025 * len(ordered))]),
        "ci_high": float(ordered[min(int(0.975 * len(ordered)), len(ordered) - 1)]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="reports/benchmarks/decomposition.json")
    parser.add_argument("--keep", action="store_true", help="keep the generated suite")
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()

    gate = load_gate()
    corpus = gate["corpus"]
    n_scenarios = args.scenarios or int(corpus["scenarios"])
    seed = args.seed if args.seed is not None else int(corpus["seed"])
    splits = gate["splits"]

    workdir = Path(args.workdir) if args.workdir else ROOT / "data" / "benchmarks" / "decomposition"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"gate: {gate['name']} (registered {gate['registered']})")
    print(f"corpus: profile={corpus['profile']} scenarios={n_scenarios} seed={seed}")
    suite_dir = generate_suite(
        suite_config(str(corpus["profile"])),
        workdir,
        n_scenarios,
        "decomposition",
        ("source", "coded", "warehouse", "report"),
    )
    records = _group_by_scenario(build_oracle_labels(suite_dir, DatasetConfig()))
    print(f"labels: {len(records)} oracle records in {time.time() - started:.1f}s")

    counts: dict[str, int] = {}
    for record in records:
        counts[record.labels["likely_cause"]] = counts.get(record.labels["likely_cause"], 0) + 1
    print(f"cause spread: {json.dumps(counts, sort_keys=True)}")

    train, validation, development, test = split_records_four(
        records,
        validation_fraction=float(splits["validation_fraction"]),
        test_fraction=float(splits["test_fraction"]),
        seed=int(splits["seed"]),
    )
    print(f"split: train={len(train)} cal={len(validation)} dev={len(development)} test={len(test)}")
    if not test:
        raise SystemExit("INSUFFICIENT_EVIDENCE: empty test split")

    print("\nfitting flat arm ...")
    flat, flat_metrics = train_decision_provider(
        records,
        DatasetConfig(),
        validation_fraction=float(splits["validation_fraction"]),
        test_fraction=float(splits["test_fraction"]),
        seed=int(splits["seed"]),
    )
    print("fitting decomposed arm ...")
    decomposed, decomposed_metrics = train_decomposed_provider(
        records,
        DatasetConfig(),
        validation_fraction=float(splits["validation_fraction"]),
        test_fraction=float(splits["test_fraction"]),
        seed=int(splits["seed"]),
    )

    flat_test = evaluate_decision_provider(flat, test)
    decomposed_test = evaluate_decomposed(decomposed, test)
    flat_preds = _per_record_predictions(flat, test, decomposed=False)
    decomposed_preds = _per_record_predictions(decomposed, test, decomposed=True)

    comparison: dict[str, Any] = {}
    for field_name in gate["fields_scored"]:
        expected = [truth for truth, _ in flat_preds[field_name]]
        left = [guess for _, guess in flat_preds[field_name]]
        right = [guess for _, guess in decomposed_preds[field_name]]
        boot = paired_bootstrap(
            expected, left, right, field_name, draws=2000, seed=0
        )
        flat_field = flat_test["fields"][field_name]
        decomposed_field = decomposed_test["fields"][field_name]
        comparison[field_name] = {
            "flat": {
                key: flat_field[key]
                for key in ("accuracy", "macro_f1", "mcc", "g_mean", "ece", "brier", "mean_cost", "cost_score")
            },
            "decomposed": {
                key: decomposed_field[key]
                for key in ("accuracy", "macro_f1", "mcc", "g_mean", "ece", "brier", "mean_cost", "cost_score")
            },
            "macro_f1_delta": boot["delta"],
            "macro_f1_delta_ci": [boot["ci_low"], boot["ci_high"]],
            # Distinguishes "the two arms genuinely agree on every case" from
            # "the aggregate metrics happen to tie". Without it a delta of 0
            # with a zero-width interval is ambiguous.
            "records_disagreeing": sum(
                1 for a, b in zip(left, right, strict=True) if a != b
            ),
            "records_scored": len(expected),
            "mcc_delta": (
                None
                if decomposed_field["mcc"] is None or flat_field["mcc"] is None
                else decomposed_field["mcc"] - flat_field["mcc"]
            ),
            "ece_delta": (
                None
                if decomposed_field["ece"] is None or flat_field["ece"] is None
                else decomposed_field["ece"] - flat_field["ece"]
            ),
        }
        print(
            f"  {field_name:<24} macro_f1 {flat_field['macro_f1']} -> "
            f"{decomposed_field['macro_f1']}  delta={boot['delta']} "
            f"ci=[{boot['ci_low']}, {boot['ci_high']}] "
            f"disagreeing={comparison[field_name]['records_disagreeing']}/{len(expected)}"
        )

    deltas = [
        item["macro_f1_delta"]
        for item in comparison.values()
        if item["macro_f1_delta"] is not None
    ]
    ci_pairs = [
        item["macro_f1_delta_ci"]
        for item in comparison.values()
        if item["macro_f1_delta_ci"][0] is not None
    ]
    overall_delta = float(np.mean(deltas)) if deltas else None
    ci_excludes_zero = bool(ci_pairs) and all(low > 0 for low, _ in ci_pairs)

    failures: list[str] = []
    minimum = float(gate["min_value"])
    if overall_delta is None or overall_delta < minimum:
        failures.append(f"macro_f1_delta {overall_delta} < {minimum}")
    if gate.get("require_paired_ci_excludes_zero") and not ci_excludes_zero:
        failures.append("paired 95% CI does not exclude zero on every field")

    report = {
        "gate": gate["name"],
        "gate_path": str(GATE_PATH.relative_to(ROOT)),
        "registered": gate["registered"],
        "passed": not failures,
        "failures": failures,
        "primary_metric": gate["primary_metric"],
        "overall_macro_f1_delta": overall_delta,
        "paired_ci_excludes_zero": ci_excludes_zero,
        "comparison": comparison,
        "split": {
            "seed": splits["seed"],
            "train": len(train),
            "calibration": len(validation),
            "development": len(development),
            "test": len(test),
        },
        "corpus": {**corpus, "scenarios": n_scenarios, "seed": seed},
        "labels_provenance": "oracle",
        "production_eligible": False,
        "note": "Synthetic oracle labels only. A pass on this gate can never confer production_eligible.",
        "flat_train_metrics": flat_metrics,
        "decomposed_train_metrics": decomposed_metrics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"\nresult: {'PASS' if report['passed'] else 'FAIL'}")
    for failure in failures:
        print(f"  gate failure: {failure}")
    print(f"report: {out}")

    if not args.keep and not args.workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
