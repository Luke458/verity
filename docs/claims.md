# Claims matrix

Every capability claim in the README or docs must appear here with the evidence
that supports it. CI fails if a README milestone is missing from this table
(`tests/test_claims_matrix.py`).

Statuses:

| Status | Meaning |
|---|---|
| `validated-real` | Measured on real production data with analyst-confirmed outcomes and a frozen held-out cohort. |
| `validated-synthetic` | Measured against the generator's own oracle on held-out seeds. The oracle shares the generator's assumptions, so this validates plumbing and deterministic semantics, not real-world accuracy. |
| `plumbing-only` | Exercised end to end on synthetic data, but the check or metric is known to be weak, vacuous, or statistically unsound until the referenced remediation lands. |
| `research` | Exploratory; no production claim. Promotion requires a pre-registered evaluation with real labels. |

| Milestone | Status | Evidence / caveat |
|---|---|---|
| A0 Synthetic world, fault oracle, suite harness | `validated-synthetic` | `tests/test_dgp.py`, `tests/test_faults.py`, `tests/test_scenarios.py`; family expectations come from one spec table (`qcgen/spec.py`). |
| A Data contracts | `validated-synthetic` | Negative controls cover required columns, dtypes, week progression, nulls, duplicates. |
| A Version pair / revision cube | `plumbing-only` | Declared per-stage grain, strict period contracts and predecessor-relative selection; adversarial fixtures in `tests/test_reliability.py`. |
| A Lifecycle / attribution | `validated-synthetic` | Multi-class events, signed ledger with over-explanation/offsetting flags, and product-level reclassification conservation (M2). |
| B Reconciliation | `validated-synthetic` | Per-key/per-week mass balance, aggregate-marker scans, explicit SKIPPED/NOT_EVALUATED; negative controls prove failures (M2). |
| B Counterfactual | `validated-synthetic` | Reconstructed from frames; wrong-entity controls score low; no score when nothing is reconstructable (M2). |
| B Lineage first divergence | `validated-synthetic` | Both-version fingerprints per stage, unmapped stages report UNKNOWN (M2). |
| B Expected events / shadow mode | `plumbing-only` | Blind oracle separation landed: ground truth lives in a vault outside the scenario data, `qc/` cannot import `qcgen`, and `qc shadow` is blind by default (`--with-registry` is plumbing). PASS_WITH_EXPLANATION is excluded from actionable detection; explicit scoped approvals record covered finding IDs. |
| C Temporal intelligence / forecast calibration | `plumbing-only` | Synthetic-only; the target observation no longer enters model selection, fitting or calibration, missing weeks are preserved, and pools separate metric/level/horizon/revision. Declared-calendar candidates and intervals are covered by `tests/test_qc_temporal.py` and `tests/test_qc_calendar.py`; real accuracy is still unmeasured. |
| Evidence engine | `validated-synthetic` | Declared business calendar (retail year/week, Easter/Christmas/EOFY windows, 53-week years), frozen CPU model selection, hierarchy coverage checks, sparse parent-share support, quantified explanation ledger with symmetric decomposition and conservation, `AssessmentEvidence`/`ExplanationCertificate` schema 1 and certificate-only statistical clearance; `tests/test_qc_calendar.py`, `tests/test_qc_hierarchy.py`, `tests/test_qc_ledger.py`, `tests/test_qc_policy.py`, `tests/test_qc_acceptance.py`. Real-data accuracy remains unmeasured. |
| 11 TSPulse adapter + benchmark | `research` | Pre-registered scenario-holdout gate (`config/tspulse-gate.json`); synthetic benchmark gates cannot confer production eligibility. Leave-one-series-out accuracy is reported as informational only. |
| D Typed decisions / labels / training | `plumbing-only` | Trained on synthetic oracle labels; real-label accuracy pending. |
| D Incident memory / agent handoff | `validated-synthetic` | Confirmed-only retrieval, no-shell agent execution, env allowlist, output caps, strict response validation (M4); real-agent accuracy unmeasured. |
| C+ Relationships / reports / Delta source | `validated-synthetic` | Benjamini-Hochberg correction, active-week requirement, stable entity-set hashing (M2); reports/Delta still plumbing. |
| Evaluation cohorts / conformal / prequential | `validated-synthetic` | Conformal p-values, finite guards, scope partitioning, paired champion tests, complete gate sets and plan-hash pinning (M3); `qc evidence-bench` reports raw challenger, verifier and effective decisions separately, the 95% upper bound on false clearance (insufficient incidents stays insufficient), and an evidence ablation over materiality candidates; real-label cohort still pending. |
| Operations store / drift monitoring | `validated-synthetic` | WAL/busy-timeout, schema versioning, provenance CHECK with no analyst default, historical outcome revisions, preserved relationships, SQLite assessment/attempt journal, checksum recovery and locking; real load still pending. |
| Substrates ModernBERT-class text probe | `research` | Trainers always record `production_eligible=false`; central eligibility requires a pinned real test artifact and operational checks; real-label semantic benchmark still pending. |
| Champion provider bake-off | `validated-synthetic` | Complete gates, homogeneous cohorts, fail-closed identity, provenance without overrides, paired selection (M3); real-label selection pending. |
| Onboarding | `plumbing-only` | Never executed against a real production table. |
| Feedback simulation (synthetic analyst) | `research` | Feedback-loop plumbing; provenance `synthetic`, tags come from the simulated cause (M5). |
| Verity spine (replay/expectations/reference/RCA) | `plumbing-only` | Observation cutoffs, explicit version maps, approved-at filtering, revision-preserving calibration and per-grain controls have regression coverage; real data pending. |
| Weekly run orchestrator | `validated-synthetic` | flock, complete-input assessment identity, separate attempts, SQLite schema 4 migration and five crash-seam tests; reports publish by atomic rename without claiming a cross-resource transaction. |
| Integration Jev/systemone remote provider | `validated-synthetic` | Probability validation, response byte caps, evidence-state structural truncation, fallback keeps local fields (M4). |

## Not claimed

- Production eligibility from pilot label readiness or provenance alone. Central eligibility requires a pinned real analyst test evaluation, confidence-bound gates and operational checks.

- Azure Databricks or Spark execution. The supported backend is local
  versioned snapshots (Parquet scenarios, delta-rs `DeltaSource`).
- Any real-world detection, precision, or calibration number.
- Production eligibility of any learned artifact. `production_eligible` remains
  false until a real analyst-labelled frozen cohort exists.

## Reliability evidence additions

- Independent hand-authored contracts, doubled totals, offsetting references,
  historical selection, model veto prevention, migration and publication recovery:
  `tests/test_reliability.py` and `tests/test_negative_controls.py`.
- Store cohort freezing and four-way group splits: `qc/store_cohort.py`,
  `qc/training.py`, cohort/training tests. No real analyst cohort exists yet.
- Optional Laya: [pinned CPU integration](laya.md), protocol/budget/timeout/fallback
  tests and reproducible smoke/comparison artifacts; research only.
- [Local feedback flow and measured operating envelope](reliability-limitations.md).
  Delta integration is required in core CI; heavyweight model integration is a
  separate manual pinned job.
