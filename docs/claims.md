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
| A0 Synthetic world, fault oracle, suite harness | `validated-synthetic` | `tests/test_dgp.py`, `tests/test_faults.py`, `tests/test_scenarios.py` |
| A Data contracts | `plumbing-only` | Negative controls cover required columns, dtypes, week progression, nulls, duplicates. |
| A Version pair / revision cube | `plumbing-only` | Grain semantics still loose (see M2 plan). |
| A Lifecycle / attribution | `plumbing-only` | Multi-class events, offsetting explanations and reclassification accounting are unsound until M2. |
| B Reconciliation | `plumbing-only` | Hierarchy check is a tautology; mass balance compares global totals only. M2 replaces it. |
| B Counterfactual | `plumbing-only` | Currently arithmetic subtraction, not reconstruction; score can be 1.0 vacuously. M2 replaces it. |
| B Lineage first divergence | `plumbing-only` | Stage-local; both-version fingerprints land in M2. |
| B Expected events / shadow mode | `plumbing-only` | Registry currently auto-loaded from the generator manifest; blind separation lands in M1. |
| C Temporal intelligence / forecast calibration | `plumbing-only` | Synthetic-only; conformal/prequential fixes land in M3. |
| 11 TSPulse adapter + benchmark | `research` | Weekly-length suitability gate unresolved; never production-eligible. |
| D Typed decisions / labels / training | `plumbing-only` | Trained on synthetic oracle labels; real-label accuracy pending. |
| D Incident memory / agent handoff | `plumbing-only` | Draft contamination and untrusted-output handling fixes land in M4. |
| C+ Relationships / reports / Delta source | `plumbing-only` | Relationship correlation lacks multiple-comparison control until M2. |
| Evaluation cohorts / conformal / prequential | `plumbing-only` | Gate enforcement and statistical corrections land in M3. |
| Operations store / drift monitoring | `plumbing-only` | Provenance defaults and non-atomic weekly writes land in M4. |
| Substrates ModernBERT-class text probe | `research` | Token-oracle tests only; real-label semantic benchmark pending. |
| Champion provider bake-off | `plumbing-only` | Gate completeness, disjoint evaluation and provenance fixes land in M3. |
| Onboarding | `plumbing-only` | Never executed against a real production table. |
| Feedback simulation (synthetic analyst) | `research` | Feedback-loop plumbing; provenance must remain `synthetic`. |
| Verity spine (replay/expectations/reference/RCA) | `plumbing-only` | Leakage guards are advisory until M3/M4. |
| Weekly run orchestrator | `plumbing-only` | Idempotency is directory-existence based until M4. |
| Integration Jev/systemone remote provider | `plumbing-only` | Protocol validation fixes land in M4. |

## Not claimed

- Azure Databricks or Spark execution. The supported backend is local
  versioned snapshots (Parquet scenarios, delta-rs `DeltaSource`).
- Any real-world detection, precision, or calibration number.
- Production eligibility of any learned artifact. `production_eligible` remains
  false until a real analyst-labelled frozen cohort exists.
