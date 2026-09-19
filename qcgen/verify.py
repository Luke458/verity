"""Oracle verification.

Checks that written snapshots match their manifest fingerprints and that each
injected fault satisfies its family invariant. This is the self-test of the
synthetic generator; it does not use the QC engine (which does not exist yet).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .snapshots import frame_fingerprint

KNOWN_STATUSES = frozenset(
    {"INVESTIGATE", "PASS", "PASS_WITH_EXPLANATION", "DATA_CONTRACT_FAILURE"}
)
KNOWN_CLASSES = frozenset(
    {
        "missing_stores",
        "missing_products",
        "entity_merge",
        "backfill",
        "truncation",
        "reclassification",
        "coding",
        "warehouse",
        "historical_correction",
        "schema_failure",
        "null_duplicate_storm",
        "market_movement",
    }
)


def _check(checks: list[dict], name: str, ok: bool, detail: object = "") -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})


def _nonzero(effect: dict[str, float]) -> bool:
    return any(abs(float(value)) > 1e-9 for value in effect.values())


def _family_checks(
    checks: list[dict],
    scenario_dir: Path,
    manifest: dict,
    case: dict,
) -> None:
    family = case["family"]
    case_id = case["case_id"]
    effect = case.get("injected_effect") or {}

    if family == "schema_failure":
        column = case["details"]["column"]
        report_path = (
            scenario_dir
            / "versions"
            / manifest["current_version"]
            / "report"
            / "fact.parquet"
        )
        frame = pd.read_parquet(report_path)
        _check(checks, f"{case_id}: column {column!r} absent", column not in frame.columns)
        return

    if family == "commodity_remap":
        per_value = case["details"].get("per_value", {})
        values = list(per_value.values())
        _check(checks, f"{case_id}: two commodities measured", len(values) == 2, per_value)
        if len(values) == 2:
            first, second = values
            _check(checks, f"{case_id}: opposite signs", first * second < 0, per_value)
            scale = max(abs(first), abs(second), 1e-9)
            _check(
                checks,
                f"{case_id}: totals conserved",
                abs(first + second) <= 0.01 * scale,
                per_value,
            )
            _check(
                checks,
                f"{case_id}: volume redistributed",
                case["injected_effect"].get("dollar_moved", 0.0) > 0,
                case["injected_effect"],
            )
        return

    _check(checks, f"{case_id}: effect nonzero", _nonzero(effect), effect)

    if family == "missing_stores":
        _check(checks, f"{case_id}: dollar negative", effect.get("dollar", 0.0) < 0, effect.get("dollar"))
        _check(checks, f"{case_id}: units negative", effect.get("units", 0.0) < 0, effect.get("units"))
    elif family == "missing_products":
        _check(checks, f"{case_id}: dollar negative", effect.get("dollar", 0.0) < 0, effect.get("dollar"))
        _check(checks, f"{case_id}: units negative", effect.get("units", 0.0) < 0, effect.get("units"))
        _check(
            checks,
            f"{case_id}: products affected",
            bool(case["affected"].get("products")),
            case["affected"],
        )
    elif family == "entity_merge":
        source_store = str(case["details"]["source_store"])
        target_store = str(case["details"]["target_store"])
        version_info = manifest["versions"][manifest["current_version"]]
        stages = list(version_info["fingerprints"])
        stage = "source" if "source" in stages else stages[0]
        frame = pd.read_parquet(
            scenario_dir / "versions" / manifest["current_version"] / stage / "fact.parquet"
        )
        stores = set(frame["store_id"].astype(str))
        _check(checks, f"{case_id}: source store removed", source_store not in stores, source_store)
        _check(checks, f"{case_id}: target store present", target_store in stores, target_store)
        _check(
            checks,
            f"{case_id}: two stores affected",
            len(case["affected"].get("stores", [])) == 2,
            case["affected"],
        )
    elif family in ("new_store_backfill", "expected_event"):
        _check(checks, f"{case_id}: dollar positive", effect.get("dollar", 0.0) > 0, effect.get("dollar"))
        _check(checks, f"{case_id}: units positive", effect.get("units", 0.0) > 0, effect.get("units"))
    elif family == "history_truncation":
        _check(checks, f"{case_id}: dollar negative", effect.get("dollar", 0.0) < 0, effect.get("dollar"))
    elif family == "coding_error":
        _check(checks, f"{case_id}: units unchanged", abs(effect.get("units", 0.0)) <= 1e-9, effect)
        _check(checks, f"{case_id}: dollar changed", abs(effect.get("dollar", 0.0)) > 1e-9, effect.get("dollar"))
    elif family == "warehouse_transform_error":
        _check(checks, f"{case_id}: units unchanged", abs(effect.get("units", 0.0)) <= 1e-9, effect)
        _check(checks, f"{case_id}: dollar changed", abs(effect.get("dollar", 0.0)) > 1e-9, effect.get("dollar"))
    elif family == "recalculation":
        _check(checks, f"{case_id}: dollar changed", abs(effect.get("dollar", 0.0)) > 1e-9, effect.get("dollar"))
    elif family == "null_duplicate_storm":
        _check(checks, f"{case_id}: duplicate units added", effect.get("units", 0.0) > 0, effect.get("units"))
    elif family == "market_movement":
        _check(checks, f"{case_id}: dollar negative", effect.get("dollar", 0.0) < 0, effect.get("dollar"))
        _check(checks, f"{case_id}: units negative", effect.get("units", 0.0) < 0, effect.get("units"))
    else:
        _check(checks, f"{case_id}: known family", False, family)


def verify_suite(
    suite_dir: str | Path, oracle_dir: str | Path | None = None
) -> tuple[bool, dict]:
    from .oracle_vault import OracleVault, default_oracle_root

    suite_dir = Path(suite_dir)
    suite_path = suite_dir / "suite.json"
    if not suite_path.exists():
        raise FileNotFoundError(f"no suite.json under {suite_dir}")
    suite = json.loads(suite_path.read_text())
    vault = OracleVault(oracle_dir or default_oracle_root(suite_dir))

    checks: list[dict] = []
    for entry in suite["scenarios"]:
        scenario_id = entry["scenario_id"]
        scenario_dir = suite_dir / scenario_id
        manifest = json.loads((scenario_dir / "manifest.json").read_text())
        oracle = vault.require(scenario_id)

        for version, version_info in manifest["versions"].items():
            for stage, fingerprint in version_info["fingerprints"].items():
                path = scenario_dir / "versions" / version / stage / "fact.parquet"
                if not path.exists():
                    _check(checks, f"{scenario_id}/{version}/{stage}: file exists", False, path)
                    continue
                actual = frame_fingerprint(pd.read_parquet(path))
                _check(
                    checks,
                    f"{scenario_id}/{version}/{stage}: fingerprint",
                    actual == fingerprint,
                    actual,
                )

        for case in oracle["cases"]:
            _check(
                checks,
                f"{case['case_id']}: known status",
                case["expected_status"] in KNOWN_STATUSES,
                case["expected_status"],
            )
            _check(
                checks,
                f"{case['case_id']}: known class",
                case["expected_class"] in KNOWN_CLASSES,
                case["expected_class"],
            )
            _family_checks(checks, scenario_dir, manifest, case)

    failed = [check for check in checks if not check["ok"]]
    report = {
        "suite_id": suite["suite_id"],
        "checks": len(checks),
        "failed_count": len(failed),
        "failed": failed[:20],
        "passed": not failed,
    }
    return not failed, report
