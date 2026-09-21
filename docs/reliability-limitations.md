# Reliability acceptance and remaining limitations

This implementation is local, read-only analyst shadow. Engineering regression
coverage does not establish production accuracy or eliminate every defect.

| Release | Implemented evidence |
|---|---|
| Correctness | Versioned findings/status policy; both-snapshot contracts; explicit grain/calendar; per-key reference, lineage and reconstruction controls; historical input guards. `tests/test_reliability.py`, `tests/test_negative_controls.py`, contract/reference/replay tests. |
| Persistence | SQLite schema 4 with backup migration, immutable identities, separate attempts and calibration revisions, checksum recovery and atomic report publication. Fault injection covers all five journal/publication seams in `tests/test_reliability.py`. |
| Evaluation | Frozen store cohorts and outcome revisions; chronological incident/snapshot groups; disjoint train/calibration/development/test; confidence-bound gates, calibration diagnostics and pinned selection; explicit synthetic ineligibility. `qc/store_cohort.py`, training/cohort tests and the local walkthrough below. |
| Evidence decisions | Declared business calendar, target-isolated rolling-origin selection, metric/level/horizon/revision residual pools, hierarchy checks, sparse parent-share support, quantified ledger with conservation, certificate-only statistical clearance and recurrence escalation. `tests/test_qc_calendar.py`, `tests/test_qc_hierarchy.py`, `tests/test_qc_ledger.py`, `tests/test_qc_policy.py`, `tests/test_qc_acceptance.py`; `qc evidence-bench` repeats the frozen threshold/ablation workflow. |
| Optional Laya | Pinned isolated CPU runtime; typed HTTP boundary, token budgeting, hard worker timeout and deterministic fallback. `tests/test_laya_adapter.py`, `optional/laya_smoke.py`, `optional/laya_compare.py`. |

## Reproduce local integration evidence

```sh
.venv/bin/python -m pytest
.venv/bin/ruff check qc qcgen tests optional
.venv/bin/mypy
.venv/bin/python -m optional.shadow_walkthrough
.venv/bin/python -m optional.scale_benchmark --stores 500 > reports/scale-benchmark.json
```

The [feedback walkthrough](../reports/onboarding-feedback.json) creates local
Delta snapshots, onboards, persists a weekly assessment, dry-runs and imports
synthetic feedback, then freezes a store cohort. It must finish with
INSUFFICIENT_EVIDENCE: synthetic feedback is deliberately excluded from real
analyst evaluation. No labels are relabelled as analyst evidence to pass a gate.

The [scale measurement](../reports/scale-benchmark.json) reports current/previous
rows, wall time, process peak RSS and snapshot read counts. It is a local CPU
Parquet measurement with temporal checks disabled, not a distributed benchmark
or a capacity promise. Cached snapshots are read once per version/stage;
projection is supported where the source provides column reads.

## Remaining limitations

- No real refreshes, real analyst labels or real-world effectiveness validation
  exist. Detection, actionable escalation, cause attribution, severity and
  calibration remain unqualified on actual retail data. All current model
  artifacts are production-ineligible.
- Statistical qualification requires enough independent incidents, all required
  classes and controls, and untouched test confirmation. Sparse cohorts cannot
  pass. Governance of labels, approval authority, thresholds and operational
  sign-off remains an operator responsibility.
- Legacy JSONL prequential APIs retain numeric availability periods for backward
  compatibility. Use the SQLite weekly observation-time journal for historical
  weekly assessment; business-week-only legacy artifacts are not equivalent.
- Point-in-time safety depends on truthful source commit/manifest times and
  explicitly supplied cross-table mappings. Parquet manifests are caller
  assertions, not an authenticated external audit log.
- Laya's English checkpoint and small evidence budget are not retail-qualified.
  CPU integration works; broad evidence can abstain, probabilities can be
  overconfident, and inference timeout requires restarting the service.
- Crash tests inject exceptions at persistence seams. They do not simulate every
  filesystem, disk-loss, machine-power-loss or SQLite corruption failure. Keep
  backups and test recovery on the deployment filesystem.
- Local pandas/Parquet/delta-rs processing retains selected snapshots in memory.
  The measured fixture is the demonstrated operating envelope; larger loads,
  remote object-store failures and concurrent production workloads need testing.
- Core CI requires Delta and lightweight Laya boundary tests. The heavyweight
  pinned CPU model job is separate and manual because it downloads model assets.
  Spark/Databricks execution, automatic publication blocking and GPU-specific
  serving remain out of scope.
