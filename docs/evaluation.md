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
echo "restatement_weeks: 2" > realistic.yaml   # the realistic feed's late arrival
qc sweep --profile realistic --seeds 5001-5010 --config realistic.yaml --jobs 6 \
  --out reports/sweep/realistic.json
python -m optional.plot_sweep reports/sweep/small.json reports/sweep/realistic.json \
  --out docs/img/detection-curves
```

Each sizable family is injected at magnitudes 1%-40% on the same seeded worlds
(a paired design), six two-fault refreshes are scored on detection, on whether
`likely_cause` names either true cause and on whether `likely_causes` names
both, and clean refreshes of the same profile measure false alarms. Summaries
are in [`docs/results/`](results/) (engine `7d122859c6bd`); the `realistic` runs
declare the profile's 2-week late-arrival window, as an operator of that feed
would (`sweep-realistic-undeclared.json` shows the cost of not declaring it).

### What the sweeps exposed, and the fixes

The first `realistic` sweep (engine `a6d485794644`) found two engine flaws, and
a closer look at revision materiality a third:

- **Category seasonality defeated the leaf share test.** Shares that swing
  through the year sit far from their trailing mean, so a 20%
  single-commodity drop was caught 2/10 times.
- **Late-arriving data broke lineage and the cause labels.** Restated recent
  weeks diverge at the source, so "first divergent stage" was always source
  and every coding or warehouse error was labelled SOURCE_INGESTION (0/54,
  0/60).
- **Single-week restatements were invisible.** Revision materiality is relative
  to the whole overlap history, so a restatement confined to one week had to
  exceed roughly `materiality_ratio x overlap weeks` (~10% of that week).

Fixes, each developed on seeds 6001-6005 only:

- **Lineage** names the stage that adds the largest material weekly revision
  increment over its upstream stage, so an upstream restatement is not blamed
  for a downstream change; `restatement_weeks` excludes a declared late-arrival
  window.
- **Share test** picks, per leaf and on training history only, between the
  trailing-mean baseline and a year-over-year baseline (log share change
  against recent year-over-year changes, last year's share averaged over the
  three weeks around t-52).
- **Per-week revision check**: each overlap week's revision minus what its
  structural events explain, against that week's previous value
  (`week_revision_ratio`, 0.5%; `restatement_tolerance`, 10%, inside the
  declared window). A new `week_restatement` family measures it.

### Results

| | `small` | `realistic` before fixes | `realistic` after | `realistic` after, unseen seeds 7001-7010 |
|---|---|---|---|---|
| Structural/revision faults | detected down to 1% | same | same | same |
| One week restated, 1%-40% | 60/60 | not measured (invisible below ~10%) | 60/60 | 60/60 |
| Market movement 10% / 20% / 40% | 10/10, 10/10, 10/10 | 1/10, 2/10, 6/10 | 2/10, 4/10, 10/10 | 0/10, 8/10, 10/10 |
| Cause correct, coding / warehouse errors | 48/48, 54/54 | 0/54, 0/60 | 54/54, 60/60 | 55/55, 59/59 |
| Lineage correct, coding errors | 60/60 | n/a | 60/60 | 60/60 |
| Clean false alarms | 3/60 | 3/60 | 3/60 | 4/60 |

Without the declared window the `realistic` profile alarms on 60/60 clean
refreshes: every refresh restates its last two weeks by up to ~3%, above the
0.5% per-week tolerance. A feed with late-arriving data must declare it.

The registered cohort still passes (detection 72/72, clean false positives
1/60 with Wilson upper 0.089, lineage 1.000); clean false alarms ticked up by
one on both the cohort and the `small` sweep after the share-test change.

### What remains

- **Small category movements on seasonal data.** With ~52 series tested per
  refresh at a 1% false-alarm budget, a drop of 10% or less in one category is
  within noise on `realistic`. Raising `temporal_fdr_q` to 0.05 added a false
  alarm without adding detection on the development seeds.
- **Pairs involving a market movement.** `market_movement` is UNKNOWN by
  design, so `likely_causes` names only the other fault of such a pair (0/10
  for both market-movement pairs; every other pair 38-40/40 per profile). A
  tree model names more of them but does not transfer across profiles
  ([labeller-experiment.md](labeller-experiment.md)).
- **Late arrival must be declared.** The engine cannot tell late arrival from
  a restatement fault within one version pair.

## Sensitivity: what a refresh could not have seen

A 10% single-category drop on `realistic` is caught 2/20 times. A diagnosis on
the development seeds (6001-6005) showed this is the noise floor, not a broken
detector:

- **The baseline is calibrated.** On clean refreshes the share t-statistics
  have SD 1.01.
- **A 10% drop sits at about the category noise.** It gives |t| ≈ 2.4-3.4,
  roughly what this world's 3% weekly category shocks allow.
- **The false-alarm budget sets the bar.** About 47 tests share it at
  q = 0.01, so a lone anomaly needs |t| ≈ 4.3.

A better seasonal model would buy about 15% more t. Raising q buys false
alarms.

So the engine reports its own limit instead. After the BH decision, every
share-tested leaf gets `detectable_change` (and `detectable_change_80`): the
smallest share drop it would flag with 50% (80%) power if it were the
refresh's only anomaly. That is the critical t for the refresh's family size
and q, times the leaf's own predictive spread, floored at materiality. The
machine record's `latest_week.sensitivity` gives the median per level and
measure, and the least sensitive leaf.

**Registered before the calibration run** (`experiments/sensitivity/calibration.py`):
market movements of 5-40% on seeds 5001-5010, both profiles, scoring the
affected category's dollar and units leaves (240 series). Each predicts its
own chance of being flagged:

1. The mean predicted probability is within 0.08 of the observed rate.
2. In each predicted-probability bin ([0, 0.2), [0.2, 0.5), [0.5, 0.8),
   [0.8, 1]), the bin's mean prediction lies inside the Wilson 95% interval
   of its observed rate.
3. The Brier score is lower than that of a constant prediction at the
   observed rate.
4. The prediction is not optimistic: the observed rate is at least the mean
   prediction minus 0.02. The lone-anomaly threshold is conservative when
   other series in the refresh are significant too.

## Limits

- The generator shares the engine authors' assumptions; generator bias is not
  measured, and synthetic detection rates are not real-world rates.
- Power depends on history length and noise: on the 30-week `tiny` profile a
  12-30% single-commodity movement is only 1.2-3 standard deviations of share
  noise and is missed about 40% of the time.
