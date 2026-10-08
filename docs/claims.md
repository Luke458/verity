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
| Lifecycle and attribution | `validated-synthetic` | Multi-class events, signed attribution, reclassification conservation. Missing entities escalate only when material (`lifecycle.missing_entity_impact`); negative control in `tests/test_negative_controls.py`. Each overlap week's unexplained revision is also judged on its own (`attribution.week_revisions`, 0.5% of the week; 10% inside a declared `restatement_weeks` window): a single-week restatement of 1%-40% is detected 60/60 on `small` and on `realistic` with its window declared, where whole-history materiality needed ~10% of the week. Undeclared late arrival alarms on every clean `realistic` refresh (60/60). |
| Counterfactual reconstruction | `validated-synthetic` | Reconstructed from frames; no score when nothing is reconstructable. |
| Reconciliation | `validated-synthetic` | Per-key mass balance, count consistency, aggregate markers; price-ratio outliers only in appended periods. |
| Lineage first divergence | `validated-synthetic` | The origin is the stage adding the largest material weekly revision increment over its upstream stage, so a legitimate source restatement is not blamed for a downstream fault. 1.000 on the registered cohort; on the `realistic` profile (late-arriving data, 2-week window declared) coding and warehouse errors 60/60 on both seed sets; 55/60 for coding errors without the declaration. `tests/test_benchmark_realism.py`. |
| Expected events, ratio expectations, reference controls | `plumbing-only` | Scoped approvals must precede the observation cutoff and cover exact finding IDs; `expected_event` reaches PASS_WITH_EXPLANATION with its registry. |
| Temporal QC | `validated-synthetic` | Leaves are tested on their share of the national parent against the better of a trailing-mean and a year-over-year baseline (chosen on training history), national series on forecast p-values, one BH family per refresh at `temporal_fdr_q = 0.01`. `market_movement` detected on every held-out cohort case. On the `realistic` profile (per-commodity seasonality) a single-commodity drop of 40% is detected 20/20 across two seed sets, 20% 12/20 and 10% 2/20 (`qc sweep`). Each share-tested leaf reports `detectable_change`, the drop it would flag as a lone anomaly. Over two registered calibration runs this is a conservative bound: never optimistic, calibrated below 0.2 and above 0.8 predicted, under-stating detection in between ([evaluation.md](evaluation.md#sensitivity-what-a-refresh-could-not-have-seen)). `tests/test_qc_temporal.py`, `tests/test_qc_policy.py`, `tests/test_benchmark_realism.py`. |
| Findings and final status | `validated-synthetic` | One status path (`policy.apply_policy`); nothing is cleared automatically. |
| Recurrence | `plumbing-only` | Stable keys survive serialization and storage; frozen, hash-checked predecessors. `tests/test_qc_second_remediation.py`. |
| Rule cause labels | `validated-synthetic` | Cannot change status or review. Scored by `qc cohort` (reported, not gated): held-out cause accuracy 66/72 = 0.917 (Wilson 0.830-0.961), origin 66/66; on the `realistic` profile coding/warehouse errors are labelled correctly 54/54 and 60/60 (was 0/54, 0/60 before the lineage fix). Source-stage historical changes are labelled HISTORICAL_CORRECTION (the generator-less SOURCE_INGESTION label was removed): `recalculation` and `week_restatement` 60/60 on both profiles (`recalculation` was 0/60). `market_movement` is labelled UNKNOWN by design. `likely_causes` names every supported cause: on the labeller test split, single faults 236/260 exactly right and both causes of a two-fault refresh 51/80; on the sweeps, both causes of the four non-market pairs 38-40/40 per profile. A per-cause tree model is no longer significantly better on single faults and does not transfer across profiles; a zero-shot decision model (Decision-2.0-Lux-9B) ranks most causes well but gets 5/260 single faults exactly right, 168/260 at best with a written labelling policy and dev-tuned thresholds ([labeller-experiment.md](labeller-experiment.md)). Severity has no ground truth and is not scored. |
| Weekly orchestrator and journal | `validated-synthetic` | Content-addressed identity, locking, fault injection at all five journal/publication seams (`tests/test_reliability.py`), atomic publication. |
| Notification | `plumbing-only` | Suppressed statuses and cached retries reach no sink; accumulating size bound; https-only webhooks without redirects. `tests/test_qc_notify.py`. |
| Delta source and onboarding | `plumbing-only` | delta-rs reads, config proposal, field mapping. `tests/test_qc_delta.py`, `tests/test_qc_onboard.py`, `tests/test_qc_mapping.py`. |
| Fault-size sweep and realistic profile | `validated-synthetic` | `qc sweep`: detection curves over fault size (paired seeds), two-fault refreshes and clean false alarms per profile; results for engine `7d122859c6bd` in `docs/results/` and [evaluation.md](evaluation.md), including unseen seeds 7001-7010. Clean false alarms 3/60 (`small`), 3/60 and 4/60 (`realistic`). `tests/test_benchmark_realism.py`. |
| Cohort evaluation | `validated-synthetic` | Registered, hash-pinned plan with independently sized clean controls; exits 3 on gate failure. |
| Notice drafting (`qc notices`) | `validated-synthetic` | Drafts are unapproved and explain nothing until a person approves them; an approved draft clears only the findings its change raises (`tests/test_qc_notices.py`). Matching measured on synthetic notices ([notice-matching.md](notice-matching.md)): on LLM-reworded notices the patterns find 52% of true matches, and a System One service (Decision-2.0-Lux-9B) finds 89%, but with 11% false explanations, so about four in ten suggestions are wrong. |

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

- On the `realistic` profile a single-commodity movement of 10% or less is
  within noise at the 1% false-alarm budget (2/20 detected at 10%). The engine
  reports this limit per category (`latest_week.sensitivity`) rather than
  claiming more.
- A feed with late-arriving data must declare `restatement_weeks`; within one
  version pair the engine cannot tell late arrival from a restatement fault.
- The rule labeller never names a market movement, and credits a source-stage
  historical correction that coincides with a merge, backfill or truncation to
  that event; such pairs are named only in part.

- Detector power depends on history length and noise: on the 30-week `tiny`
  profile a 12-30% single-commodity movement is missed about 40% of the time.
- Temporal thresholds (`temporal_fdr_q`, `temporal_min_relative_residual`) and
  `materiality_ratio` were chosen on synthetic data; any new dataset should be
  evaluated with `qc cohort --config` or a comparable held-out set first.
- Point-in-time safety depends on truthful commit times and explicitly supplied
  cross-table version mappings; Parquet manifests are caller assertions.
- Crash tests inject exceptions at persistence seams; they do not simulate disk
  loss or power failure.
