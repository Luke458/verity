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
| A Version pair / revision cube | `plumbing-only` | Grain semantics still loose (see M2 plan). |
| A Lifecycle / attribution | `validated-synthetic` | Multi-class events, signed ledger with over-explanation/offsetting flags, and product-level reclassification conservation (M2). |
| B Reconciliation | `validated-synthetic` | Per-key/per-week mass balance, aggregate-marker scans, explicit SKIPPED/NOT_EVALUATED; negative controls prove failures (M2). |
| B Counterfactual | `validated-synthetic` | Reconstructed from frames; wrong-entity controls score low; no score when nothing is reconstructable (M2). |
| B Lineage first divergence | `validated-synthetic` | Both-version fingerprints per stage, unmapped stages report UNKNOWN (M2). |
| B Expected events / shadow mode | `plumbing-only` | Blind oracle separation landed: ground truth lives in a vault outside the scenario data, `qc/` cannot import `qcgen`, and `qc shadow` is blind by default (`--with-registry` is plumbing). Detection semantics still weak until M2. |
| C Temporal intelligence / forecast calibration | `plumbing-only` | Synthetic-only; conformal/prequential fixes land in M3. |
| 11 TSPulse adapter + benchmark | `research` | Pre-registered scenario-holdout gate (`config/tspulse-gate.json`); production_eligible requires the gate to pass. Leave-one-series-out accuracy is reported as informational only. |
| D Typed decisions / labels / training | `plumbing-only` | Trained on synthetic oracle labels; real-label accuracy pending. |
| D Incident memory / agent handoff | `validated-synthetic` | Confirmed-only retrieval, no-shell agent execution, env allowlist, output caps, strict response validation (M4); real-agent accuracy unmeasured. |
| C+ Relationships / reports / Delta source | `validated-synthetic` | Benjamini-Hochberg correction, active-week requirement, stable entity-set hashing (M2); reports/Delta still plumbing. |
| Evaluation cohorts / conformal / prequential | `validated-synthetic` | Conformal p-values, finite guards, scope partitioning, paired champion tests, complete gate sets and plan-hash pinning (M3); real-label cohort still pending. |
| Operations store / drift monitoring | `validated-synthetic` | WAL/busy-timeout, schema versioning, provenance CHECK with no analyst default, latest-confirmed view, UPSERT relationships, atomic weekly writes and locking (M4); real load still pending. |
| Substrates ModernBERT-class text probe | `research` | Metadata records `production_eligible=false` for any non-analyst labels; real-label semantic benchmark still pending. |
| Champion provider bake-off | `validated-synthetic` | Complete gates, homogeneous cohorts, fail-closed identity, provenance without overrides, paired selection (M3); real-label selection pending. |
| Onboarding | `plumbing-only` | Never executed against a real production table. |
| Feedback simulation (synthetic analyst) | `research` | Feedback-loop plumbing; provenance `synthetic`, tags come from the simulated cause (M5). |
| Verity spine (replay/expectations/reference/RCA) | `plumbing-only` | Leakage guards are advisory until M3/M4. |
| Weekly run orchestrator | `validated-synthetic` | flock, completed-marker idempotency, atomic temp-dir swap, config-hash run_id, reference never reads ahead (M4). |
| Integration Jev/systemone remote provider | `validated-synthetic` | Probability validation, response byte caps, evidence-state structural truncation, fallback keeps local fields (M4). |

## Not claimed

- Production eligibility without a passing `qc pilot-check` and a frozen real held-out cohort (`docs/real-pilot.md`).

- Azure Databricks or Spark execution. The supported backend is local
  versioned snapshots (Parquet scenarios, delta-rs `DeltaSource`).
- Any real-world detection, precision, or calibration number.
- Production eligibility of any learned artifact. `production_eligible` remains
  false until a real analyst-labelled frozen cohort exists.
