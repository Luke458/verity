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
| C Temporal intelligence / forecast calibration | `plumbing-only` | Synthetic-only; every configured measure and appended period is assessed with period-specific findings, training always ends at the previous snapshot, the latest origins are reserved exclusively for calibration, pooled standardized errors are converted back through the target's own scale, residual pools separate metric/level/horizon/revision, snapshot measures are compared within a period and never summed across time, and absent measures are reported (required absence makes the assessment incomplete). `tests/test_qc_temporal.py`, `tests/test_qc_calendar.py`, `tests/test_qc_remediation_acceptance.py`, `tests/test_qc_second_remediation.py`; real accuracy is still unmeasured. |
| Evidence engine | `validated-synthetic` | Declared business calendar, disjoint selection/calibration, hierarchy coverage checks, sparse parent-share support, per-metric quantified explanation ledgers with conservation, per-period `AssessmentEvidence`/`ExplanationCertificate` schema 3 and certificate-only statistical clearance bound to exact finding IDs, assessment identity, evidence digest, policy version and a pinned qualification artifact. Certificate binding is strict (missing identity/digest/schema/policy/qualification or a wrong scope/metric/level/period is ignored), and all applicable contract, reconciliation, reference, lineage, hierarchy and required-input failures are evaluated before clearance. Qualification is schema 2 with incident groups, splits, source identities, independent held-out coverage lower bounds, registered gates and analyst-label linkage for real provenance; it is frozen into assessment identity before the cache lookup. `tests/test_qc_calendar.py`, `tests/test_qc_hierarchy.py`, `tests/test_qc_ledger.py`, `tests/test_qc_policy.py`, `tests/test_qc_acceptance.py`, `tests/test_qc_remediation_acceptance.py`, `tests/test_qc_second_remediation.py`. Real-data clearance stays disabled until a real qualification is pinned. |
| 11 TSPulse adapter + benchmark | `research` | Pre-registered scenario-holdout gate (`config/tspulse-gate.json`); synthetic benchmark gates cannot confer production eligibility. Leave-one-series-out accuracy is reported as informational only. |
| D Typed decisions / labels / training | `plumbing-only` | Trained on synthetic oracle labels; real-label accuracy pending. |
| D Incident memory / agent handoff | `validated-synthetic` | Confirmed-only retrieval, no-shell agent execution, env allowlist, output caps, strict response validation (M4); real-agent accuracy unmeasured. |
| C+ Relationships / reports / Delta source | `validated-synthetic` | Benjamini-Hochberg correction, active-week requirement, stable entity-set hashing (M2); reports/Delta still plumbing. |
| Evaluation cohorts / conformal / prequential | `validated-synthetic` | Conformal p-values, finite guards, scope partitioning, paired champion tests, complete gate sets and plan-hash pinning (M3). Synthetic benchmarks, store cohorts, qualification and production eligibility share one gate implementation: grouped confidence bounds and the 1% false-clearance upper bound everywhere, with insufficient groups staying insufficient. Evidence-bench selects the materiality candidate on development scenarios only, runs exactly one final configuration on untouched test scenarios, keeps a failed selection/qualification visible (never a fallback candidate) and reruns the actual provider adapter per evidence level with the bespoke rule baseline reported separately. Real-label cohort still pending. |
| Operations store / drift monitoring | `validated-synthetic` | WAL/busy-timeout, schema versioning (6) with backup before migration, provenance CHECK with no analyst default, historical outcome revisions, feature/text versions preserved through migration (incompatible evidence is rejected, never relabelled), preserved relationships, SQLite assessment/attempt journal, frozen recurrence-input manifests, checksum recovery and locking; real load still pending. |
| Substrates ModernBERT-class text probe | `research` | Trainers always record `production_eligible=false`; central eligibility requires a pinned real test artifact and operational checks; real-label semantic benchmark still pending. |
| Champion provider bake-off | `validated-synthetic` | Complete gates, homogeneous cohorts, fail-closed identity, provenance without overrides, paired selection (M3); real-label selection pending. |
| Onboarding | `plumbing-only` | Never executed against a real production table. |
| Feedback simulation (synthetic analyst) | `research` | Feedback-loop plumbing; provenance `synthetic`, tags come from the simulated cause (M5). |
| Verity spine (replay/expectations/reference/RCA) | `plumbing-only` | Observation cutoffs, explicit version maps, approved-at filtering, revision-preserving calibration and per-grain controls have regression coverage; real data pending. |
| Weekly run orchestrator | `validated-synthetic` | flock, complete-input assessment identity that freezes distinct recurrence predecessors and their evidence hashes, and the resolved qualification artifact (status, provenance, digest, availability) before the cache lookup, so replacement/revocation/loss changes identity and retries reuse frozen inputs; separate attempts, SQLite schema 6 migration and crash-seam tests; reports publish by atomic rename without claiming a cross-resource transaction. |
| Integration Jev/systemone remote provider | `validated-synthetic` | Providers consume the recorded evidence package (every assessed measure/period, failed checks compressed to references, missing required inputs, forecast uncertainty, per-metric ledgers, provenance; never final status, findings, certificates or decisions), omissions are recorded after packing, mandatory evidence is never dropped, and operational requests abstain rather than falling back to a thinner evidence level. Payload digests recorded, probability validation, response byte caps and fallback keeping local fields. |

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
