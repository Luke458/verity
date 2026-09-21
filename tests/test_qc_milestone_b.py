from __future__ import annotations

import pandas as pd
import pytest

from qc.config import DatasetConfig
from qc.counterfactual import reconstruct_counterfactual
from qc.events import load_registry, propose_expected_events, save_registry
from qc.fingerprints import fingerprint_deltas, structural_fingerprint
from qc.lifecycle import NEW_BACKFILL, LifecycleEvent
from qc.lineage import analyze_lineage
from qc.reconciliation import _aggregate_marker_checks, run_reconciliation
from qc.versions import build_version_pair

CONFIG = DatasetConfig()


def _fact(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def test_structural_fingerprint_and_deltas():
    before = _fact(
        [{"week": 1, "store_id": "S1", "product_id": "P1", "dollar": 10.0, "units": 1}]
    )
    after = _fact(
        [
            {"week": 1, "store_id": "S1", "product_id": "P1", "dollar": 10.0, "units": 1},
            {"week": 2, "store_id": "S2", "product_id": "P1", "dollar": 5.0, "units": 1},
        ]
    )
    fingerprint_before = structural_fingerprint(before, CONFIG)
    fingerprint_after = structural_fingerprint(after, CONFIG)

    assert fingerprint_after["rows"] == 2
    assert fingerprint_after["distinct_counts"]["store_id"] == 2
    assert fingerprint_after["metric_sums"]["dollar"] == pytest.approx(15.0)

    deltas = fingerprint_deltas(fingerprint_before, fingerprint_after)
    assert deltas["rows"] == 1
    assert deltas["metric:dollar"] == pytest.approx(5.0)
    assert deltas["distinct:store_id"] == 1
    assert deltas["entity_set_changed"] == 1.0


# ---------------------------------------------------------------------------
# Counterfactual
# ---------------------------------------------------------------------------


def _counterfactual_frames():
    previous = _fact(
        [
            {"week": 1, "store_id": "S1", "product_id": "P1", "dollar": 10.0, "units": 1},
            {"week": 2, "store_id": "S1", "product_id": "P1", "dollar": 10.0, "units": 1},
        ]
    )
    current = pd.concat(
        [
            previous,
            _fact(
                [
                    {"week": 1, "store_id": "S9", "product_id": "P1", "dollar": 5.0, "units": 1},
                    {"week": 2, "store_id": "S9", "product_id": "P1", "dollar": 5.0, "units": 1},
                ]
            ),
        ],
        ignore_index=True,
    )
    return previous, current


def test_counterfactual_fully_explained():
    previous, current = _counterfactual_frames()
    event = LifecycleEvent(
        "store",
        "S9",
        NEW_BACKFILL,
        historical_weeks_added=(1, 2),
        historical_value_added=10.0,
        value_added_by_week={1: 5.0, 2: 5.0},
    )
    result = reconstruct_counterfactual(previous, current, [event], CONFIG)
    assert result.evaluated is True
    assert result.raw_delta == pytest.approx(10.0)
    assert result.explained_delta == pytest.approx(10.0)
    assert result.reconstructed_delta == pytest.approx(0.0)
    assert result.reconciliation_score == pytest.approx(1.0)
    assert [week["week"] for week in result.by_week] == [1, 2]


def test_counterfactual_partial_and_unexplained():
    previous, current = _counterfactual_frames()
    # Only week 1 is classified as added; week 2's excess stays unexplained.
    partial = LifecycleEvent(
        "store",
        "S9",
        NEW_BACKFILL,
        historical_weeks_added=(1,),
        historical_value_added=5.0,
        value_added_by_week={1: 5.0},
    )
    result = reconstruct_counterfactual(previous, current, [partial], CONFIG)
    assert result.reconstructed_delta == pytest.approx(5.0)
    assert result.reconciliation_score == pytest.approx(0.75)

    # No reconstructable events: no score at all, rather than a vacuous 1.0.
    unexplained = reconstruct_counterfactual(previous, current, [], CONFIG)
    assert unexplained.evaluated is False
    assert unexplained.reconciliation_score is None


def test_counterfactual_wrong_entity_scores_low():
    previous, current = _counterfactual_frames()
    wrong = LifecycleEvent(
        "store",
        "S1",
        NEW_BACKFILL,
        historical_weeks_added=(1, 2),
        historical_value_added=10.0,
    )
    result = reconstruct_counterfactual(previous, current, [wrong], CONFIG)
    # Values are recomputed from the frames, so pointing at the wrong entity
    # cannot produce a perfect reconstruction.
    assert result.reconciliation_score == pytest.approx(0.0)  # per-key errors cannot offset


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def _reconciliation_frames():
    base = _fact(
        [
            {"week": 1, "store_id": "S1", "banner_id": "B1", "state_id": "NSW", "product_id": "P1", "dollar": 10.0, "units": 1},
            {"week": 2, "store_id": "S1", "banner_id": "B1", "state_id": "NSW", "product_id": "P1", "dollar": 10.0, "units": 1},
        ]
    )
    report = _fact(
        [
            {"week": 1, "banner_id": "B1", "state_id": "NSW", "dollar": 10.0, "units": 1, "store_count": 1, "product_count": 1},
            {"week": 2, "banner_id": "B1", "state_id": "NSW", "dollar": 10.0, "units": 1, "store_count": 1, "product_count": 1},
        ]
    )
    return base, report


def test_reconciliation_pass_and_mass_balance_failure():
    base, report = _reconciliation_frames()
    result = run_reconciliation(base, report, CONFIG)
    assert result.status == "PASS"

    tampered = report.copy()
    tampered.loc[0, "dollar"] = 9.0
    broken = run_reconciliation(base, tampered, CONFIG)
    assert broken.status == "RECONCILIATION_FAILURE"
    assert any(
        check.name == "mass_balance:dollar" for check in broken.failed
    )


def test_reconciliation_count_mismatch():
    base, report = _reconciliation_frames()
    tampered = report.copy()
    tampered.loc[0, "product_count"] = 5
    result = run_reconciliation(base, tampered, CONFIG)
    assert any(
        check.name == "count_consistency:product_count" for check in result.failed
    )


def test_reconciliation_ratio_flags():
    rows = []
    ratios = [10.0, 10.5, 10.0, 10.5, 10.0, 100.0]
    for week, ratio in enumerate(ratios, start=1):
        rows.append(
            {
                "week": week,
                "store_id": "S1",
                "banner_id": "B1",
                "state_id": "NSW",
                "product_id": "P1",
                "dollar": ratio,
                "units": 1,
            }
        )
    base = _fact(rows)
    report = base.drop(columns=["store_id", "product_id"]).groupby(
        ["week", "banner_id", "state_id"], as_index=False
    )[["dollar", "units"]].sum()
    report["store_count"] = 1
    report["product_count"] = 1
    result = run_reconciliation(base, report, CONFIG)
    assert result.status == "PASS"
    flagged_weeks = {flag["week"] for flag in result.ratio_flags}
    assert 6 in flagged_weeks
    assert not {1, 2, 3, 4, 5} & flagged_weeks


def test_marker_scan_does_not_claim_parent_child_consistency():
    # A single frame cannot prove parent/child consistency: summing detail
    # rows always reproduces the total. The old tautological check is gone;
    # only aggregate-marker detection remains.
    base = _fact(
        [
            {"week": 1, "banner_id": "B1", "state_id": "NSW", "dollar": 10.0, "units": 1},
            {"week": 1, "banner_id": "B2", "state_id": "VIC", "dollar": 5.0, "units": 1},
            {"week": 2, "banner_id": "B1", "state_id": "NSW", "dollar": 7.0, "units": 1},
        ]
    )
    checks = _aggregate_marker_checks(base, CONFIG)
    assert checks
    assert all(check.passed for check in checks)
    names = {check.name for check in checks}
    assert "aggregate_markers:state_id" in names
    assert not any(name.startswith("hierarchy:") for name in names)


def test_hierarchy_markers_fail_reconciliation():
    base = _fact(
        [
            {"week": 1, "banner_id": "B1", "state_id": "NSW", "dollar": 10.0, "units": 1},
            {"week": 1, "banner_id": "TOTAL", "state_id": "NSW", "dollar": 10.0, "units": 1},
        ]
    )
    report = (
        base.drop(columns=["banner_id"])
        .groupby(["week", "state_id"], as_index=False)[["dollar", "units"]]
        .sum()
    )
    result = run_reconciliation(base, report, CONFIG)
    assert result.status == "RECONCILIATION_FAILURE"
    assert any(
        check.name == "aggregate_markers:banner_id" for check in result.failed
    )


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


class DictSource:
    def __init__(self, frames: dict):
        self.frames = frames

    def read_fact(self, version: str, stage: str) -> pd.DataFrame:
        return self.frames[(version, stage)]


def _lineage_rows(dollar_by_week):
    return [
        {
            "week": week,
            "store_id": "S1",
            "product_id": "P1",
            "dollar": value,
            "units": 1,
        }
        for week, value in dollar_by_week.items()
    ]


def test_lineage_first_divergence_stage():
    stages = ("source", "coded", "warehouse", "report")
    previous = {("V1", stage): _fact(_lineage_rows({1: 10.0, 2: 10.0, 3: 10.0})) for stage in stages}
    current = {}
    for stage in stages:
        values = {1: 10.0, 2: 10.0, 3: 5.0} if stage in ("coded", "warehouse", "report") else {1: 10.0, 2: 10.0, 3: 10.0}
        current[("V2", stage)] = _fact(_lineage_rows(values))
    frames = {**previous, **current}
    pair = build_version_pair(
        previous[("V1", "report")],
        current[("V2", "report")],
        "V1",
        "V2",
        "report",
        "report",
        CONFIG,
    )
    result = analyze_lineage(
        DictSource(frames), "V1", "V2", list(stages), CONFIG, pair
    )
    assert result.status == "FIRST_DIVERGENCE"
    assert result.first_divergence == "coded"
    divergences = {entry["stage"]: entry for entry in result.divergences}
    assert divergences["source"]["relative_divergence"] == 0.0
    assert divergences["coded"]["relative_divergence"] > CONFIG.lineage_materiality_ratio


# ---------------------------------------------------------------------------
# Expected-event registry
# ---------------------------------------------------------------------------


def test_registry_propose_and_roundtrip(tmp_path):
    event = LifecycleEvent(
        "store",
        "S9",
        NEW_BACKFILL,
        historical_weeks_added=(3, 4),
        historical_value_added=10.0,
    )
    proposals = propose_expected_events([event])
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["entity_ids"] == ["S9"]
    assert proposal["expected_history_start"] == 3
    assert proposal["expected_history_end"] == 4
    assert proposal["confirmed"] is False

    path = tmp_path / "registry.json"
    save_registry(proposals, path)
    assert load_registry(path) == proposals
