# Reliability acceptance and remaining limitations

This implementation is local, read-only analyst shadow. Engineering regression
coverage does not establish production accuracy or eliminate every defect.

| Release | Implemented evidence |
|---|---|
| Correctness | Versioned findings/status policy; both-snapshot contracts; explicit grain/calendar; per-key reference, lineage and reconstruction controls; historical input guards; every appended period assessed with its own finding, forecast, ledger and certificate; stable recurrence keys with scoped impact attribution. `tests/test_reliability.py`, `tests/test_negative_controls.py`, `tests/test_qc_remediation_acceptance.py`, contract/reference/replay tests. |
| Persistence | SQLite schema 6 with backup migration that preserves feature/evidence/text versions (incompatible evidence is rejected, never relabelled), immutable identities that freeze recurrence predecessors, their evidence hashes and the resolved qualification artifact/status, separate attempts and calibration revisions, checksum recovery and atomic report publication. Fault injection covers all five journal/publication seams in `tests/test_reliability.py`. |
| Evaluation | Frozen store cohorts and outcome revisions; chronological incident/snapshot groups; disjoint train/calibration/development/test; one shared gate implementation for synthetic benchmarks, store cohorts, qualification and production eligibility with group confidence bounds and the 1% false-clearance bound; development-only selection followed by one final test configuration; actual provider-adapter ablations with a separately labelled rule baseline and pinned selection; explicit synthetic ineligibility. `qc/store_cohort.py`, training/cohort tests, `tests/test_qc_second_remediation.py` and the local walkthrough below. |
| Evidence decisions | Declared business calendar, disjoint selection/calibration partitions, target-scale intervals from pooled standardized residuals, metric/level/horizon/revision residual pools, hierarchy checks, sparse parent-share support, per-measure quantified ledgers with conservation, strict certificate-only statistical clearance (identity, evidence digest, finding coverage, schema, policy, qualification digest and exact scope/metric/level/period) with all applicable integrity failures evaluated first, schema-2 qualification with independent groups and registered gates, and recurrence escalation on a serialized stable key with a registered cumulative budget. `tests/test_qc_calendar.py`, `tests/test_qc_hierarchy.py`, `tests/test_qc_ledger.py`, `tests/test_qc_policy.py`, `tests/test_qc_acceptance.py`, `tests/test_qc_remediation_acceptance.py`, `tests/test_qc_second_remediation.py`; `qc evidence-bench` repeats the frozen threshold/ablation workflow and pins the qualification artifact. |
| Notification | Delivery is fail-soft and unbuffered: a sink error is recorded on stderr and never changes the assessment status, exit code or published report. Payloads are byte-bounded by dropping optional sections, never truncating, and a cached retry self-suppresses so an idempotent re-run cannot re-page. No delivery guarantee, retry queue or rate limiting is claimed, and no sink has been pointed at a production endpoint. `qc/notify.py`, `tests/test_qc_notify.py`. |
| Distribution drift | Opt-in and off by default. A drift finding is evidence that a distribution moved, never why, and it cannot clear a finding or override a deterministic one. The per-scope threshold is derived from that scope's own consecutive-week PSI history, so a scope with too little history reports `NOT_EVALUATED` rather than guessing, and a degenerate (constant) reference is `NOT_EVALUATED` rather than zero. Synthetic-only: the gate in `config/drift-gate.json` passed on injected redistributions, which is not a production sensitivity or false-positive rate. `qc/distribution_drift.py`, `optional/drift_benchmark.py`. |
| Scope coverage | Two checks, both opt-in and neither active. `assess_scope_coverage` (per scope) passed its registered gate on real data at control alarm 0.010 and detection 1.000, but produces roughly 43 scope flags per real period, so it is **not wired** into the escalation path. `assess_aggregate_coverage` (per entity key) passed its control arm at 0.171 but **failed** its detection arm at 0.500 against a required 1.0, so it is **not enabled**; wiring it would suppress escalation for about half of genuine aggregate coverage drops. Both registrations are preserved unedited. A consequence recorded rather than hidden: on real sparse data the missing-entity escalation still fires on essentially every refresh, and the measured 100% alarm rate is **unresolved**. `qc/coverage.py`, `config/coverage-gate.json`, `config/coverage-gate-aggregate.json`. |
| Optional Laya | Pinned isolated CPU runtime; typed HTTP boundary, token budgeting, hard worker timeout and deterministic fallback. `tests/test_laya_adapter.py`, `optional/laya_smoke.py`, `optional/laya_compare.py`. |

## Reproduce local integration evidence

```sh
.venv/bin/python -m pytest
.venv/bin/ruff check qc qcgen tests optional
.venv/bin/mypy
.venv/bin/python -m optional.shadow_walkthrough
.venv/bin/python -m optional.scale_benchmark --profile one-million \
  --out reports/benchmarks/scale-1m.json
.venv/bin/python -m optional.scale_benchmark --profile five-million \
  --out reports/benchmarks/scale-5m.json
```

The [feedback walkthrough](../reports/onboarding-feedback.json) creates local
Delta snapshots, onboards, persists a weekly assessment, dry-runs and imports
synthetic feedback, then freezes a store cohort. It must finish with
INSUFFICIENT_EVIDENCE: synthetic feedback is deliberately excluded from real
analyst evaluation. No labels are relabelled as analyst evidence to pass a gate.

The 320-week scale measurements are [one million](../reports/benchmarks/scale-1m.json)
(8.2 s, ~1.07 GiB peak RSS) and [five million](../reports/benchmarks/scale-5m.json)
(53.3 s, ~4.58 GiB peak RSS) rows with the corrected multi-metric pipeline.
Each records requested/actual rows, wall time, process peak RSS, snapshot read
counts, assessed/absent metrics, periods, hierarchy counts, certificate counts,
status and a `complete` flag; incomplete or failed runs list explicit
limitations instead of qualifying. They are single-machine
local CPU Parquet measurements, not distributed benchmarks, capacity promises
or production data. Cached snapshots are read once per version/stage;
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
- Automatic statistical clearance is disabled unless a pinned qualification
  artifact covers the exact model/metric/level/horizon and the assessment
  provenance. Synthetic qualification cannot clear real Delta assessments, so
  real data remains review-only until a real qualification is pinned with
  analyst-labelled evidence.
- The 1M/5M-row measurements are single-machine, single-process and serial;
  RSS above ~5 GiB for the 5M fixture means concurrent production workloads
  and larger snapshots need capacity testing before deployment.
- Every configured measure present in a refresh is now assessed per period with
  its own findings, ledger and certificates; absent optional measures are
  reported and absent required measures make the assessment incomplete. On the
  current synthetic fixtures only `dollar` and `units` are populated, so the
  1M/5M benchmarks and generated scenarios exercise those two measures; real
  availability of `scripts`/`stock` and snapshot aggregation semantics still
  need production validation. Secondary-measure movements that the previous
  primary-only engine passed (for example a 99% units drop with unchanged
  dollars) now require review, which intentionally raises review volume on
  multi-measure refreshes.
- The evidence ablation reruns the deterministic rule provider adapter, because
  remote/learned providers are not affordable per ablation repetition on the
  synthetic benchmark. The bespoke rule is reported separately as
  `rule_baseline`; provider-specific ablation on real labels remains pending.
- Legacy JSONL prequential APIs retain numeric availability periods for backward
  compatibility. Use the SQLite weekly observation-time journal for historical
  weekly assessment; business-week-only legacy artifacts are not equivalent.
  Across-load prequential pools still mix scopes and are drift diagnostics, not
  per-series interval calibration.
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
