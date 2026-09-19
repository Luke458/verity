# Evaluation, calibration and evidence queries

This document covers the evaluation rigor layer: frozen cohorts with
pre-registered gates, finite-sample conformal intervals, prequential
(across-load) calibration, and the bounded evidence-query tool used by the
investigation agent.

## Frozen cohorts

```sh
qc cohort --plan config/cohort.json --out reports/cohort/v1
qc cohort --heldout-seed 7101 --heldout-seed 7102 --dev-seed 1101 --out reports/cohort/v1
```

A `CohortPlan` declares development seeds, held-out seeds (disjoint, enforced),
families, controls and gates *before* running. Every case is built with the
fault oracle and scored by the full engine. The result records:

- held-out metrics: detection rate, historical false-positive rate, contract
  failure rate, mean reconstruction score, lineage first-divergence accuracy
  (controls are excluded from detection);
- development metrics for reference, never used for gates;
- per-gate actual vs threshold with pass/fail;
- `code_sha256` over the engine sources and `plan_sha256` over the plan, so a
  result can be tied to an exact revision and plan;
- `cases.jsonl`, one record per case.

`production_eligible` is always `False` for synthetic cohorts. The harness
exists so a real-label cohort can be run and judged honestly: changing the plan
after seeing results invalidates the evaluation, and generator bias is not
measured by disjoint seeds alone.

## Conformal intervals

`qc.conformal.conformal_interval` implements the finite-sample interval using
the `ceil((n + 1) * (1 - alpha))`-th smallest absolute residual. With too few
residuals the result is `INSUFFICIENT_CALIBRATION`, never a widened interval.
`leave_one_out_coverage` reports honest small-sample coverage: in-sample
coverage is trivially at least `1 - alpha` once the interval exists, so
leave-one-out is the meaningful check.

## Prequential (across-load) calibration

```sh
qc shadow --suite-dir data/suites/demo --prequential-store data/calibration.jsonl
qc prequential --store data/calibration.jsonl --as-of 104 --target-week 105 --z 2.4
```

`CalibrationPool` only admits residuals where `available_on < as_of` and
`target_week < target_week`; records are deduplicated per `(series, target)` so
a repeated or partially recomputed load cannot stack evidence for itself, and
the pool is bounded. `percentile` and `interval` return `None` /
`INSUFFICIENT_CALIBRATION` below `min_samples`. Shadow runs append one record
per temporal series using the standardized residual that was computed before
the target was scored.

## Bounded evidence queries

```sh
qc evidence --scenario-dir data/suites/demo/scenario-0000 --query lifecycle_changes
qc evidence --scenario-dir ... --query contributors --limit 5 --json
```

`query_evidence` is an allowlisted, read-only view over a completed run:
`lifecycle_changes`, `historical_residuals`, `temporal_anomalies`,
`first_divergence`, `contract_failures`, `contributors`, `relationships`.
Unknown names raise. Every result reports `total_rows` and `truncated`, so a
consumer can tell when it is seeing a limited answer. The investigation brief
includes `tool_calls` pairing the likely cause with the first queries to run.

## What remains

The harness is ready; the data is not. The next step is to run a cohort over
real refresh pairs with analyst outcomes, register the plan and gates, and then
retrain and recalibrate. Once confirmed outcomes exist, `qc champion` decides
which substrate runs in production; see [docs/champion.md](champion.md). Until
that happens, all semantic and threshold claims remain plumbing-validated only.
