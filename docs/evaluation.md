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

**Run 1 (seeds 5001-5010): two of four failed.**
[`results/sensitivity-run1.json`](results/sensitivity-run1.json)

| | Result |
|---|---|
| 1. Mean predicted within 0.08 of observed | 0.654 vs 0.629: **held** |
| 2. Each bin's prediction inside its observed interval | top bin 0.99 vs 133/136 (interval tops out at 0.99): **failed** |
| 3. Brier below a constant | 0.058 vs 0.233: **held** |
| 4. Not optimistic (observed ≥ predicted - 0.02) | 0.629 vs 0.634: **failed** (optimistic, mostly on `realistic`) |

The cause was a modelling error, not noise. A category that drops by m does
not lose m of its share, because the national total drops too. With share c,
the share moves by (1 - m)/(1 - mc): a 20% drop in a category holding 15% of
sales moves its share 17.5%. The prediction treated c as 0.

The fix changes what is reported. `detectable_change` is now the drop in the
**leaf's own value** (others unchanged), converted from the share drop with
the share the baseline expects, which is what an analyst means by "this
category could fall 20% unnoticed". The calibration converts the same way.

**Run 2 registered (unseen seeds 7001-7010), same four predictions and
thresholds.**

**Run 2: three of four held.**
[`results/sensitivity-run2.json`](results/sensitivity-run2.json)

| | Result |
|---|---|
| 1. Mean predicted within 0.08 of observed | 0.624 vs 0.679: **held** |
| 2. Each bin's prediction inside its observed interval | [0, 0.2): 0.04 vs 0.07 held; [0.8, 1]: 0.99 vs 0.99 held; [0.2, 0.5): 0.29 vs 0.61 (n = 23) and [0.5, 0.8): 0.65 vs 1.00 (n = 10): **failed** |
| 3. Brier below a constant | 0.060 vs 0.218: **held** |
| 4. Not optimistic | 0.679 vs 0.604: **held** |

What is left is the conservatism prediction 4 anticipated, larger than
assumed. A category movement moves dollars and units together, and two
simultaneous discoveries face a BH threshold twice as lenient as a lone one.
So movements are caught earlier than the lone-anomaly bound says. Run 1's
share-conversion error was optimistic and hid this.

**Reading `detectable_change`:** it is a **conservative bound**, not a
calibrated probability. Below about 0.2 predicted, leaves were flagged 7% of
the time; above 0.8, 99%. In between it under-states detection (0.29
predicted, 0.61 observed). It never over-states it. No further tuning was
done on these seeds; a third model change would be fitting the evaluation.

**Combining a category's measures does not help.** On clean development
refreshes the share t-statistics of one category's measures are highly
correlated: dollars and units ρ = 0.90, other pairs 0.58-0.85. They are
largely one measurement of one category shock. Averaging dollar and units
would gain about 3% in t, and grouping them per category before BH gains
nothing over BH's own rank-2 threshold for two simultaneous discoveries.
Testing one measure per category would cut the family from about 47 to
about 13, lowering the critical |t| from about 4.3 to about 4.0: not enough
for a 10% drop (|t| ≈ 2.8). None of these is implemented. At this
false-alarm budget, detection on this world is at its limit; the report of
that limit is the improvement.

## Clean false alarms on `realistic`

The sweeps flag 3/60 and 4/60 clean `realistic` refreshes. A rerun of the 60
clean refreshes of seeds 5001-5010 with every finding recorded flags 3:

- **One latest-week share flag** (a banner's units and stock). This is the
  false-alarm budget the BH family spends.
- **Two single-product absences** escalated by `missing_entity_impact`. Each
  product carried just over `materiality_ratio` (0.1%) of the period: 0.11%
  and 0.17%. One had sold in 97 of 103 previous weeks; the other in all 103.

Neither is late-arriving data, which the declared window absorbs. To stop the
absences, the engine would have to tell a natural product exit from a lost
product. The `missing_products` fault removes products that look the same:
of 70 injected absences on the same seeds, 65 had sold in every previous
week, and 11 carried no more than 0.2% of the period. A long-run
reliability cut-off would remove one of the two alarms (the 97/103 product),
a cut-off fitted to one case. Raising the absence materiality above 0.17%
would trade 2 false alarms in 60 for hiding 11 of 70 injected absences.
Neither change is made;
an operator who expects product churn should raise `materiality_ratio` for
that dataset knowingly.

## Approving closures and category moves

The registry can explain store closures (`LATEST_WEEK_MISSING`) and category
moves (`POSSIBLE_RECLASSIFICATION`), scoped to the weeks they are in effect
([engine.md](engine.md#expected-event-registry)). An approval restates the
refresh like-for-like before the latest-week test: closed stores are left out
of every week, and moved products' history is assigned to their new category.
The risk is that an approval hides a fault it did not cause, so the
measurement pairs each change with a second fault.

`python -m experiments.approvals.run` builds, per profile and seed:

- each change alone, assessed blind and with the entry `qc notices` drafts
  for it, approved;
- each change with a second fault (coding error, warehouse transform error,
  market movement, one-week restatement), approved;
- each second fault alone, and a clean refresh.

The pair injects the second fault first, so it shares the world and the
fault's parameters with the solo run. A market movement's category is drawn
at injection, though, so it can differ. A missed market movement is
therefore judged against the engine's own reported 80%-power
`detectable_change` for that category in the run that missed it.

Developed on seeds 6101-6105, which found two problems and fixed them before
registration:

- A product sold only in a closed store also goes missing. Absences are now
  re-checked like-for-like.
- A move explains none of a revision, so on `realistic` the immaterial
  late-arrival residual escalated every approved move. A refresh whose only
  structural changes are approved moves now judges its residual as if it had
  none.

### Registered predictions (seeds 6201-6220, both profiles)

1. **Blind, every change alone is flagged** (20/20 per change and profile).
2. **Approved, a change alone is explained:** PASS_WITH_EXPLANATION in at least
   17/20 per change and profile. No approved run leaves the change's own
   findings unexplained (`historical_revision`, `absence_event` or
   `historical_event`).
3. **Approvals do not hide large faults:** for coding errors, warehouse transform
   errors and one-week restatements, the approved pair is flagged in at least
   as many seeds as the second fault alone, minus one.
4. **Approvals hide nothing the engine claims it can see:** no missed market
   movement in an approved pair is at or above that run's reported 80%-power
   `detectable_change`.

### Test results (seeds 6201-6220)

| | `small` | `realistic` |
|---|---|---|
| Closure / move alone, flagged blind | 20/20, 20/20 | 20/20, 20/20 |
| Closure / move alone, approved: PASS_WITH_EXPLANATION | 19/20, 20/20 | 17/20, **16/20** |
| Coding, warehouse, one-week restatement + approved change: flagged | 20/20 in every cell (alone 20/20) | 20/20 in every cell (alone 20/20) |
| Market movement + approved closure / move: flagged | 20/20, 20/20 (alone 20/20) | 11/20, 9/20 (alone 10/20) |
| Missed market movements at or above the 80% limit | 0 | **1 and 1** of 11 and 13 missed (alone 0 of 12) |
| Clean refreshes flagged | 0/20 | 3/20 |

1. **Held.**
2. **Failed** on `realistic`: moves 16/20 against the 17 required, and four
   approved runs left a change's own finding (`historical_revision`)
   unexplained.
   - **Three of the four are the world.** Seed 6206 (move) and seed 6211
     (closure and move) carry a natural single-product absence
     ([above](#clean-false-alarms-on-realistic)), and the same seed's clean
     refresh is flagged for it too.
   - **The fourth (seed 6220) was a defect.** The move took every product out
     of C07, and the engine also saw that as "C07 removed", which no approval
     covered. Every overlap week then failed its revision check, because the
     removal was subtracted from weeks whose total had not changed.
   - **The other unexplained runs are latest-week flags of their world.**
     Seed 6216 shows the same C07 scripts flag as its clean refresh. Seed
     6206's closure leaves a C05 units/stock flag that its clean refresh does
     not show, so it may be noise or the restatement.
3. **Held.**
4. **Failed, twice.** The prediction was too strict for an 80%-power limit, at
   which about one movement in five is missed. The two misses were drops of
   0.257 and 0.269 against limits of 0.227 and 0.261. In the larger one (seed
   6215) the hit category had a seasonal up-week, so a 25.7% cut showed as an
   18% share drop, under that run's 50% limit. Detection with an approved
   change (11/20, 9/20) matches detection without one (10/20). These are
   different categories on the same worlds, so this is not a paired test.

**Fix after the test (not registered):** a category that approved moves
empty or create, with the value those moves carried, is now the moves'
consequence. It is covered by their approval and left out of the revision
sums and the counterfactual (`attribution._move_consequences`). Rerun on the
same seeds: realistic moves are explained in 17/20, and seed 6220 now passes
with its explanation. Nothing else changed; the two remaining move runs with an
unexplained `historical_revision` are the natural product absences above.

Results: [`results/approvals-test.json`](results/approvals-test.json)
(registered run) and
[`results/approvals-test-after-fix.json`](results/approvals-test-after-fix.json).

## Recurrence and slow movements

Every other measurement here scores one version pair. Recurrence is meant to
act across refreshes: a finding that repeats within `recurrence_window`
refreshes, with material cumulative impact, adds a `recurrence` finding.
`python -m experiments.sequences.run` publishes one seeded world as 11 weekly
snapshots and runs `qc weekly` with a journal over the 10 consecutive pairs,
with recurrence on and off. On `realistic`, every snapshot under-counts its
last two weeks. There are three kinds of sequence:

- **clean**: nothing changes;
- **drift**: one category falls a further 3% each week (−26% after ten);
- **step**: one category drops 10% and stays down.

Reading the code before measuring (development seed 6201) found that
recurrence cannot do its job, for two separate reasons:

- **It cannot change a status.** It considers only current findings that
  failed without approval, and any such finding already makes the refresh
  INVESTIGATE.
- **It never sees a predecessor when refreshes are replayed.** The journal
  records each run at wall-clock time but looks for predecessors observed
  before the refresh's own observation time. When past refreshes are
  assessed after the fact (as in this harness, or a backtest), no
  predecessor ever qualifies and every assessment is a cold start.

### Registered predictions (seeds 6301-6310, current engine)

1. **Recurrence changes nothing:** every refresh has the same status with
   recurrence on and off, and no refresh has a `recurrence` finding.
2. **Clean sequences:** at most 10% of refreshes flagged per profile.
3. **A sustained 10% drop is seen once at most.** On `realistic` its category is
   flagged in its first week in at most 3/10 sequences. In at least 8/10 it
   is never flagged after its first week, because the baseline absorbs the
   new level.
4. **A slow decline goes unseen on `realistic`:** in at most 5/10 sequences is
   the declining category flagged within ten weeks (−26%).

### Test results (seeds 6301-6310)

All four held.

1. **Recurrence changed nothing.** Status was identical with recurrence on and off
   in 600/600 refreshes, and no refresh had a `recurrence` finding.
2. **Clean sequences:** 7/100 refreshes flagged on `small`, 8/100 on
   `realistic`, in line with the single-pair false-alarm rates.
3. **The 10% step on `realistic`** was flagged in its first week 0/10 times,
   and never after its first week in 9/10 sequences.
4. **The 3%-a-week decline on `realistic`** was flagged at some point in 3/10
   sequences.

Week by week, a flagged category looks like this (`X` = flagged, first
assessed week on the left):

| | `small` | `realistic` |
|---|---|---|
| Step, −10% held | `X.........` or `XX........` in all 10 | `..........` in 9, `...X......` in 1 |
| Decline, −3% a week | first flag in week 1-4 in all 10, then mostly silent (e.g. `.XX.X.....`) | `..X.......`, `...X......`, `.....X....`, and 7 never |

The single-week test compares the latest week with a baseline trained on the
previous snapshot. Once a shift is in that snapshot, it is in the baseline.
A sustained change is therefore visible at most in its first week or two,
and a gradual one is absorbed as it happens. On `realistic` a category can
lose a quarter of its sales over ten weeks without a single flag.

**What changed:** recurrence was removed: the module, its journal inputs, its
config keys (`recurrence_enabled`, `recurrence_window`,
`recurrence_minimum`, `recurrence_budget_ratio`; a config that still sets
them now fails with "unknown dataset config keys") and its tests. The registered
cohort is unchanged. Sustained and gradual movements remain undetected on
`realistic`. Catching them needs a test that compares several recent weeks
with the weeks before them, and has not been built.

Results: [`results/sequences-test.json`](results/sequences-test.json).

## Limits

- The generator shares the engine authors' assumptions; generator bias is not
  measured, and synthetic detection rates are not real-world rates.
- Power depends on history length and noise: on the 30-week `tiny` profile a
  12-30% single-commodity movement is only 1.2-3 standard deviations of share
  noise and is missed about 40% of the time.
