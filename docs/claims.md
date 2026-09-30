# Claims matrix

Every capability in the README appears here with the evidence behind it
(`tests/test_claims_matrix.py` enforces this). All evidence is synthetic: the
project does not depend on, and makes no claim about, real refreshes or
analyst judgement.

| Status | Meaning |
|---|---|
| `validated-synthetic` | Measured through the full engine against the generator's oracle on held-out seeds, or pinned by exact unit/negative-control tests. Validates the engine against its own model of retail faults, not real-world accuracy. |
| `plumbing-only` | Implemented and tested for correctness of mechanics; no detection or quality metric is claimed. |
| `research` | Exploratory; not wired into the status. |

## Registered system result

`qc cohort --plan config/cohort.json` (small profile, held-out seeds
7101-7103, full engine, final status), engine code `46de38631c7a`:

| Measure | Result | Gate |
|---|---|---|
| Detection over 12 fault/movement families | 72/72 (Wilson lower 0.949) | >= 0.9, **PASS** |
| False positives on clean refreshes | 0/60 (Wilson upper 0.060) | <= 0.1, **PASS** |
| Lineage first-divergence accuracy | 1.000 | >= 0.9, **PASS** |

Before the review remediation the same plan could not measure a false-positive
rate (its only control was a genuine movement), and clean refreshes alarmed on
10/10 runs.

## Capabilities

| Capability | Status | Evidence / caveat |
|---|---|---|
| Synthetic world and fault oracle | `validated-synthetic` | `tests/test_dgp.py`, `tests/test_faults.py`, `tests/test_scenarios.py`; family expectations come from one table (`qcgen/spec.py`) mirrored in `docs/synthetic-data.md` (`tests/test_docs_sync.py`). Includes a `clean` negative control. |
| Data contracts | `validated-synthetic` | Negative controls for required columns, dtypes, week progression, nulls and duplicates; both snapshots and every stage are checked. |
| Version pair and revision cube | `plumbing-only` | Declared grain, strict period contracts and predecessor-relative selection; metadata-only Delta commits are skipped. `tests/test_reliability.py`, `tests/test_qc_delta.py`. |
| Lifecycle and attribution | `validated-synthetic` | Multi-class events, signed attribution, reclassification conservation. Missing entities escalate only when material (`lifecycle.missing_entity_impact`); negative control in `tests/test_negative_controls.py`. |
| Counterfactual reconstruction | `validated-synthetic` | Reconstructed from frames; no score when nothing is reconstructable. |
| Reconciliation | `validated-synthetic` | Per-key mass balance, count consistency, aggregate markers; price-ratio outliers only in appended periods. |
| Lineage first divergence | `validated-synthetic` | 1.000 on the registered cohort. |
| Expected events, ratio expectations, reference controls | `plumbing-only` | Scoped approvals must precede the observation cutoff and cover exact finding IDs; `expected_event` reaches PASS_WITH_EXPLANATION with its registry. |
| Temporal QC | `validated-synthetic` | Share-of-parent tests for leaves, forecast p-values for national series, one BH family per refresh at `temporal_fdr_q = 0.01`. `market_movement` detected on every held-out case. `tests/test_qc_temporal.py`, `tests/test_qc_policy.py`. |
| Findings and final status | `validated-synthetic` | One status path (`policy.apply_policy`); nothing is cleared automatically. |
| Recurrence | `plumbing-only` | Stable keys survive serialization and storage; frozen, hash-checked predecessors. `tests/test_qc_second_remediation.py`. |
| Rule cause labels | `validated-synthetic` | Cannot change status or review. Scored by `qc cohort` (reported, not gated): held-out cause accuracy 60/72 = 0.833 (Wilson 0.731-0.902), origin 66/66. All misses are two whole families: `market_movement` is labelled UNKNOWN by design (temporal-only anomalies get no cause) and `recalculation` is labelled SOURCE_INGESTION where the oracle says HISTORICAL_CORRECTION. Severity has no ground truth and is not scored. This is the baseline any decision model must beat. |
| Weekly orchestrator and journal | `validated-synthetic` | Content-addressed identity, locking, fault injection at all five journal/publication seams (`tests/test_reliability.py`), atomic publication. |
| Notification | `plumbing-only` | Suppressed statuses and cached retries reach no sink; accumulating size bound; https-only webhooks without redirects. `tests/test_qc_notify.py`. |
| Delta source and onboarding | `plumbing-only` | delta-rs reads, config proposal, field mapping. `tests/test_qc_delta.py`, `tests/test_qc_onboard.py`, `tests/test_qc_mapping.py`. |
| Cohort evaluation | `validated-synthetic` | Registered, hash-pinned plan with independently sized clean controls; exits 3 on gate failure. |

## Not claimed

- Any detection rate, false-positive rate or alarm volume on real refreshes.
  Real data typically has more entity churn, messier seasonality and faults the
  generator does not model.
- Accuracy of the cause labels on anything but the generator's families.
- Delivery guarantees for notifications (no retry queue).
- Capacity beyond a single machine. `optional/scale_benchmark.py` last measured
  8.2 s / 1.07 GiB peak RSS for 1M rows and 53.3 s / 4.58 GiB for 5M rows (320
  weeks, serial in-memory pandas) before the review remediation; it has not
  been re-measured since.
- Spark or Databricks execution.

## Known limitations

- Detector power depends on history length and noise: on the 30-week `tiny`
  profile a 12-30% single-commodity movement is missed about 40% of the time.
- Temporal thresholds (`temporal_fdr_q`, `temporal_min_relative_residual`) and
  `materiality_ratio` were chosen on synthetic data; any new dataset should be
  evaluated with `qc cohort --config` or a comparable held-out set first.
- Point-in-time safety depends on truthful commit times and explicitly supplied
  cross-table version mappings; Parquet manifests are caller assertions.
- Crash tests inject exceptions at persistence seams; they do not simulate disk
  loss or power failure.
