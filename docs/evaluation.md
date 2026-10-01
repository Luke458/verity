# Evaluation

Everything here is measured on the synthetic generator (`qcgen`) against its
own oracle, through the full engine, on the final status. Nothing is measured
on real refreshes.

## What counts as what

| Kind | Families | Scored as |
|---|---|---|
| fault | missing stores/products, entity merge, backfill, truncation, remap, coding, warehouse transform, recalculation, schema failure, null/duplicate storm | detection: final status is not PASS / PASS_WITH_EXPLANATION |
| movement | `market_movement` (a genuine 12-30% single-commodity drop) | detection: it must surface as a latest-week anomaly |
| expected_event | registered backfill | explained: PASS_WITH_EXPLANATION when the registry is supplied |
| control | `clean` (no injection at all) | false positive when the final status is not PASS |

Before the review remediation the only control was `market_movement`, whose
spec expects INVESTIGATE, so no false-positive rate had ever been measured on
a clean refresh.

## Registered cohort gate

```sh
qc cohort --plan config/cohort.json --out reports/cohort/v1          # exit 3 = gate failed
qc cohort --plan config/cohort.json --config my.yaml --out ...       # evaluate another config
```

`config/cohort.json` lists its families, controls, seeds and gates explicitly
and is pinned by `config/cohort.json.sha256`; editing the plan after seeing
results fails the hash check. Dev seeds (1101, 1102) and held-out seeds (7101,
7102, 7103) are disjoint, the `small` profile (104 weeks) is used, and gates are
evaluated on the held-out split only:

| Gate | Rule |
|---|---|
| `min_detection_rate` | 95% Wilson lower bound over held-out fault/movement cases >= 0.9 |
| `max_false_positive_rate` | 95% Wilson upper bound over held-out clean controls <= 0.1 |
| `min_lineage_first_divergence_accuracy` | first divergent stage equals the injection stage >= 0.9 |

Controls are sized separately (`scenarios_per_control: 20`, so 60 held-out
clean refreshes): the FPR bound cannot reach 0.1 with fewer than ~35 controls.
The 1% false-clearance bound and oracle disagreements are reported, not gated.
So is label quality: the cause and origin labels attached to each fault case
are scored against the oracle (accuracy with a Wilson interval, per family,
with the predicted-label distribution) under `metrics.labels`. Severity has no
ground truth in the generator and is not scored.
Each result records the plan hash, the engine code hash, whether the tree was
dirty, and the evaluated configuration; a result is evidence only for exactly
that configuration. The current numbers are in [claims.md](claims.md).

## Shadow over a suite

```sh
qcgen generate --suite demo --scenarios 13 --profile small --stages all
qc shadow --suite-dir data/suites/demo --out reports/shadow/demo
```

`qc shadow` is blind by default: the engine gets no expected-event registry and
never reads the oracle, which lives in a vault outside the scenario data.
`--with-registry` supplies the generator-declared events and is plumbing, not
detection. The summary reports detection and false-positive rates (both on the
final status), the historical-layer false-positive rate, latest-week detection
and control rates, expected-event pass rate, lineage first-divergence accuracy
and mean reconstruction score.

## Choosing thresholds

Detector parameters are chosen on scenarios disjoint from the cohort plan and
recorded where they are set. `temporal_fdr_q = 0.01` was chosen on 40 clean +
45 fault `small` refreshes (seeds 3000-3039): 2/40 clean false alarms versus
3/40 at q = 0.05, with 45/45 detections either way. Any new detector must be
evaluated with `qc cohort --config` before it is enabled; a component-level
benchmark is not evidence for the system.

## Fault-size sweep

```sh
qc sweep --profile realistic --seeds 5001-5010 --controls-per-seed 6 --jobs 11 \
  --out reports/sweep/realistic.json
python -m optional.plot_sweep reports/sweep/small.json reports/sweep/realistic.json \
  --out docs/img/detection-curves
```

Each sizable family is injected at magnitudes 1%-40% on the same seeded worlds
(a paired design), six two-fault refreshes are scored on detection and on
whether the single cause label names either true cause, and clean refreshes of
the same profile measure false alarms. Results for engine `a6d485794644`,
seeds 5001-5010 (disjoint from the cohort and threshold-tuning seeds), are in
[`docs/results/`](results/):

| | `small` | `realistic` |
|---|---|---|
| Structural/revision faults (missing stores, truncation, remap, recalculation) | detected at every size down to 1% | same |
| Coding error, 1% / 2% / >=5% | 1/10, 7/10, 10/10 | 5/10, 9/10, 10/10 |
| Market movement, 5% / 10% / 20% / 40% | 4/10, 10/10, 10/10, 10/10 | 1/10, 1/10, 2/10, 6/10 |
| Clean false alarms | 2/60 (upper 0.114) | 3/60 (upper 0.137) |
| Cause label correct among detected coding / warehouse errors | 48/48, 54/54 | **0/54, 0/60** |

What this shows:

- **Category seasonality defeats the leaf share test.** Shares that swing
  through the year sit far from their trailing mean, so a 20% single-commodity
  drop is caught 2/10 times.
- **Lineage, and with it the cause label, collapses under late arrival.** The
  restated weeks diverge at the source stage, so "first divergent stage" is
  always source and every coding or warehouse error is labelled
  SOURCE_INGESTION. The cohort's lineage accuracy of 1.000 holds only on
  generators that never restate history.
- **Detection gains at 1% on `realistic` are not real gains.** Late arrival
  adds ~0.04% of total history in restatement, just under the 0.1% revision
  materiality, so even a small fault on top crosses it.
- **Revision materiality is relative to the whole overlap.** That is why
  late-arrival restatements pass, and also why a restatement confined to one
  recent week must exceed roughly `materiality_ratio x overlap weeks` (~10% of
  a week at defaults) to escalate on its own.
- **Pairs are detected, but one label cannot carry two causes.** An "either
  cause" rule is satisfied by naming the structural fault; on `realistic`
  coding+market pairs are all labelled SOURCE_INGESTION (0/10).

## Limits

- The generator shares the engine authors' assumptions; generator bias is not
  measured, and synthetic detection rates are not real-world rates.
- Power depends on history length and noise: on the 30-week `tiny` profile a
  12-30% single-commodity movement is only 1.2-3 standard deviations of share
  noise and is missed about 40% of the time.
