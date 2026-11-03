# Pretrained share baseline: Timestral (experimental)

`share_baseline: timestral` replaces the leaf share test's trailing and
seasonal baselines with a pretrained time-series forecast of each leaf's share
of its parent. The default stays `engine`; nothing here changes a default
until the registered test below has run.

## Why the share test, not the forecaster

Every clean false alarm on the real test splits that comes from temporal QC is
a leaf (state or department) decided by its share-of-parent p-value
(`decide_anomalies`); the national series, the only one the `Forecaster`
decides, raised none. Swapping the forecaster cannot move those alarms.

Clean test refreshes that were not PASS, by the checks that fired (engines
`4513c0f6a164` for M5, `8c67828f00d8` for Favorita):

| | temporal only | temporal + price | price only | history only | history + temporal |
|---|---|---|---|---|---|
| M5 (14/100) | 4 | 4 | 5 | 1 | 0 |
| Favorita (31/100) | 26 | 0 | 0 | 4 | 1 |

A better share baseline can remove at most the temporal-only alarms: about 4
on M5 (the price check still fires on the other 4) and 26 on Favorita.

## The change

- `qc/share_model.py`: all leaves of one measure and period are forecast in
  one batch, from their shares on the parent's dense week grid (a missing week
  is NaN, not compressed), `target - last observed week` steps ahead.
- The observed share is scored against the forecast quantiles: the 10-50 or
  50-90 quantile gap on the side it falls, read as a normal sigma; the p-value
  is two-sided normal. The relative spread (80% half-width as a sigma over the
  median) is reported on the `trailing` baseline's scale, so
  `detectable_change` needs no change.
- Unchanged: the BH family, materiality, the sustained window (still the
  engine baselines when `temporal_sustained_weeks` > 1), and the fallback for
  a leaf that cannot be scored (fewer than 8 observed shares, a degenerate
  forecast: its forecast p-value, as under the engine baselines).
- Pins: code `mistralai/timestral@a534bdbd244d`, weights
  `Timestral-28M@49b26409ac87` and `Timestral-209M@9874d252ab0a`
  (`SHARE_MODEL_REVISIONS` in `experiments/real/m5.py`). Weights are under
  Mistral's Modified MIT licence; the code is Apache 2.0.

Run it:

```sh
pip install -e '.[timestral]'
python -m experiments.real.m5 run --weekly data/real/m5/weekly.parquet --split dev --jobs 4 \
    --share-baseline timestral --share-model mistralai/Timestral-28M --out reports/real/m5-dev-ts28m.json
```

## Dev evidence (spent weeks only)

**Single test, before BH** (M5 weeks 106-177, 10 leaves x 2 measures, 1,440
clean tests; the test weeks were not read). Rate of clean tests at p < 0.01
(nominal 1%), and the share of tests detecting a synthetic drop of the leaf's
value with the threshold set so both baselines have a 1% clean rate:

| | clean p < 0.01 | 80% interval coverage | 10% drop detected | CPU per refresh |
|---|---|---|---|---|
| engine (trailing / seasonal) | 3.0% | - | 10% | - |
| Timestral-28M | 1.1% | 84% | 21% | 70 ms |
| Timestral-209M | 1.3% | 84% | 27% | 395 ms |

Neither baseline has useful power at 2-5% drops.

**Full engine, M5 dev/confirm and Favorita dev/reserve splits** (2026-11-02,
engine `20db19243faa`, after the hardware fix below; outputs in
`reports/real/timestral/`, rerun with its `runall.sh`). The engine arm's
records match the published `favorita-dev.json` and `m5-confirm.json` record
for record, so the default path is unchanged:

| | engine | Timestral-28M | Timestral-209M |
|---|---|---|---|
| M5 dev clean false alarms | 0/20 | 0/20 | 0/20 |
| M5 dev market movement 20% / 40% | 1 / 4 of 5 | 3 / 4 of 5 | 3 / 4 of 5 |
| M5 confirm clean false alarms | 5/52 | 3/52 | 2/52 |
| M5 confirm market movement 10% / 20% / 40% | 2 / 5 / 12 of 13 | 3 / 6 / 11 of 13 | 4 / 9 / 11 of 13 |
| Favorita dev clean false alarms | 5/20 | 6/20 | - |
| Favorita dev market movement 20% / 40% | 1 / 0 of 5 | 2 / 3 of 5 | - |
| Favorita reserve clean false alarms | 7/16 | 7/16 | - |
| Favorita reserve market movement 40% | 0/4 | 2/4 | - |

All other fault families were identical across arms. Wall time per split was
within 3% of the engine arm for 28M and about 8% over for 209M (Favorita dev
about 18 minutes, M5 confirm about 11, at 3 jobs).

**At the engine's q, Favorita gets no fewer clean alarms.** On Favorita's
spent splits the motivating ceiling (26 of 31 test clean alarms decided by
the share test) does not move at q = 0.01. The clean alarms both arms raise
are mostly large real share moves that any baseline would flag: HOME AND
KITCHEN I at +243%, SCHOOL AND OFFICE at +363% (back-to-school), BABY CARE at
+165-265%. Timestral swaps some alarms rather than removing them, and it
reports more series per alarm (week 111: 7 series against 2) with much
smaller p-values (often below 1e-6, some 0). Week 126 is the source's known
perishables gap ending: PRODUCE's share sits near 0.05% for weeks 106-125 and
returns to 16%. Timestral flags it; the engine does not. That is a source
event, but the labels count it as a clean false alarm. The comparisons below
show this is a question of operating point, not of a worse test.

On M5 the two baselines disagree case by case: on confirm, Timestral removed
three clean alarms and added one (week 159, FOODS_1 units), and of the market
movements it caught five the engine missed and missed three the engine
caught. Two of those three are HOBBIES_2, a 0.5% department with a strong
annual cycle, where the seasonal year-over-year baseline is tighter (relative
spread 11.5% vs 15.5% at week 129) and the 40% drop's adjusted p was 0.013
against q = 0.01. With about two and a half years of weekly history the
pretrained model resolves the annual cycle less well than an explicit
year-over-year comparison.

**Share-rule curves** (2026-11-03; `experiments/real/share_rules.py`, the
harness behind the 2026-10-15 comparison in `docs/real-data-favorita.md`,
with a `timestral` rule added). The harness runs each latest-week share test
on every spent week (M5 106-277, Favorita 106-125 and 142-241), clean and with
one family cut by 40%, and approximates the refresh decision: one BH family,
2% materiality, judged on each rule's own impact. Its engine tables
regenerate identically to the 2026-10-15 ones. Alarmed clean refreshes and
the share of 40% cuts caught, as q falls from 0.01 to 1e-6:

| | q 0.01 | 0.001 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---|---|---|---|
| M5 engine (`current`) | 12/172, 0.82 | 7, 0.72 | 1, 0.61 | 0, 0.48 | 0, 0.35 |
| M5 Timestral-28M | 4/172, 0.92 | 2, 0.89 | 1, 0.85 | 0, 0.80 | 0, 0.76 |
| M5 Timestral-209M | 2/172, 0.95 | 1, 0.92 | 0, 0.88 | 0, 0.85 | 0, 0.81 |
| Favorita engine (`current`) | 28/120, 0.43 | 17, 0.32 | 11, 0.23 | 10, 0.17 | 8, 0.13 |
| Favorita Timestral-28M | 37/120, 0.64 | 29, 0.56 | 23, 0.48 | 18, 0.42 | 12, 0.37 |
| Favorita Timestral-209M | 39/120, 0.65 | 32, 0.57 | 21, 0.51 | 19, 0.46 | 17, 0.42 |

At matched alarm counts Timestral catches more on both retailers (Favorita:
29 alarms and 0.56 against the engine's 28 and 0.43; 12 and 0.37 against 11
and 0.23). None of the 2026-10-15 rules did that; the best of them,
per-leaf calibration, is below Timestral too (Favorita 15, 0.34). Its leaf
p < 0.01 rate is nominal on M5 (36 of 3,440 clean tests, 34 expected) and
3x nominal on Favorita (165 against 56; the engine's is 4x, 215). A
Student-t tail on Timestral's z (3, 5 or 10 df) only moves the operating
point along a lower curve (Favorita 10 df: 9 alarms, 0.22), so the normal
tail stays. 209M is a little better on M5 and level on Favorita.

**Seasonal hybrid** (harness only, `hybrid` and `hybrid3` in
`share_rules.py`): Timestral or the seasonal baseline, or Timestral or
either engine baseline, whichever has the smaller relative spread. The
hybrid picks an engine baseline for 11-15% of cut rows. The curve barely
moves (Favorita q 0.001: 31/120, 0.57 against Timestral's 29, 0.56; M5 7/172
against 4 at q 0.01, same detection), and it raises more clean alarms at
the same q (`hybrid3` most). On HOBBIES_2 over all spent weeks Timestral
alone already beats the seasonal baseline (40% cuts at p < 1e-3: 0.62
against 0.47; median relative spread 0.114 against 0.141), and the hybrid
adds a little (0.67). The full-engine HOBBIES_2 misses on M5 confirm were
those weeks, not a seasonal weakness in general. The simpler rule stays.

**A feed Timestral cannot have seen** (harness, `share_rules.py
build-synthetic`): qcgen's `realistic` history assumptions (category
seasonality, 4% market and 3% category weekly shocks, a third of products
intermittent), run over 278 weeks (M5's length) on the `full` universe, 5
states x 15 categories, seeds 101 and 202; spent weeks 106-278, 173 clean
refreshes each. A 40% cut is caught by every rule (0.94-1.00), so the
curves use 10% and 20% cuts (`tests --drop`):

| seed, cut | engine q 0.01 | Timestral q 0.01 | engine q 0.001 | Timestral q 0.001 |
|---|---|---|---|---|
| 101, 20% | 1/173, 0.75 | 0/173, 0.88 | 0, 0.50 | 0, 0.77 |
| 202, 20% | 4/173, 0.79 | 2/173, 0.90 | 0, 0.55 | 0, 0.78 |
| 101, 10% | 1/173, 0.07 | 0/173, 0.13 | 0, 0.02 | 0, 0.05 |
| 202, 10% | 4/173, 0.07 | 2/173, 0.12 | 0, 0.02 | 0, 0.04 |

Leaf p < 0.01 among 6,920 clean tests (69 expected): engine 83 and 92,
Timestral 73 and 71. Timestral is no worse on false alarms, catches more and
is closer to nominal on a feed it cannot have memorised, and this generator
was written alongside the engine, so if anything it suits the engine's
baselines. Contamination does not explain the real-data gain. This does not
rule out that real feeds differ from qcgen in ways that matter.

**Full engine at q = 0.001** (2026-11-03, `runall-q.sh`; `--fdr-q` sets
`temporal_fdr_q`, which also tightens the national tests):

| | engine q 0.01 | engine q 0.001 | Timestral-28M q 0.01 | Timestral-28M q 0.001 |
|---|---|---|---|---|
| M5 dev clean | 0/20 | 0/20 | 0/20 | 0/20 |
| M5 dev movement 20% / 40% | 1 / 4 of 5 | 0 / 3 | 3 / 4 | 3 / 3 |
| M5 confirm clean | 5/52 | 4/52 | 3/52 | 2/52 |
| M5 confirm movement 10 / 20 / 40% | 2 / 5 / 12 of 13 | 0 / 0 / 9 | 3 / 6 / 11 | 1 / 5 / 9 |
| Favorita dev clean | 5/20 | 3/20 | 6/20 | 4/20 |
| Favorita dev movement 20% / 40% | 1 / 0 of 5 | 0 / 0 | 2 / 3 | 0 / 2 |
| Favorita reserve clean | 7/16 | 5/16 | 7/16 | 6/16 |
| Favorita reserve movement 40% | 0/4 | 0/4 | 2/4 | 2/4 |

Paired on the same 108 clean refreshes and 108 market movements (5-40%,
all four splits), at the same q:

| | clean alarms, engine / Timestral | movements caught, engine / Timestral | discordant movements (exact McNemar) |
|---|---|---|---|
| q 0.01 | 17 / 16 | 25 / 34 | 5 engine-only, 14 Timestral-only (p 0.064) |
| q 0.001 | 12 / 12 | 12 / 25 | 1 engine-only, 14 Timestral-only (p 0.001) |

In the full engine Timestral matches the engine's clean alarms at the same q
and catches more. That is a smaller alarm gap than the harness's, because
the alarms neither baseline decides (national tests, history checks) are
common to both arms. The harness agrees in direction on the same weeks (M5
106-177 at q 0.01: engine 4/72, 0.77; Timestral 2/72, 0.88. Favorita
106-125: engine 2/20, 0.01; Timestral 4/20, 0.52). On Favorita's test weeks
142-241 it predicts more alarms with Timestral at the same q (33 against 26
at q 0.01, 26 against 17 at q 0.001) and more detection (0.66 against 0.50;
0.58 against 0.37). Every number here is on spent weeks and two q values
were tried, so read the p-values as descriptive.

**Hardware fault, 2026-10-28.** The Timestral-209M confirm run segfaulted
twice, and all four Favorita dev/reserve runs (both arms) failed with indices
and allocation sizes of a valid value plus exactly 2^52: a single bit flipping
in RAM. The memory overclock was turned off (3000 MHz, default timings);
`memtester` then passed on 40 GB in 4 parallel instances, the rebuilt Favorita
weekly table matched the existing one exactly, and every run above completed
without error. The 2026-10-28 M5 results reproduced exactly. Nothing from the
faulty period is used.

## Design questions before registering

1. **Operating point.** Timestral's curve is above the engine's on both
   retailers; where to sit on it is the choice. `temporal_fdr_q` is global
   (share and national tests share one BH family), so a lower q for
   Timestral also tightens the national tests. The alternative is a
   share-only weight (weighted BH). The harness predicts that at q = 0.01
   Favorita's test split gets more alarmed refreshes with Timestral than
   with the engine.
2. **Model size.** 209M is a little better on M5 and level on Favorita, at
   about 6x the CPU (395 against 70 ms per refresh, single test).
3. **Settled: seasonal leaves.** A min-spread hybrid with the engine's
   baselines does not improve the curve; Timestral alone stays.
4. **One rule for both datasets.** A per-dataset choice made after seeing
   these splits would not be a registered test.
5. **Settled: tails.** Heavier tails move the operating point along a lower
   curve; the normal tail stays.

## Caveats

- **Possible contamination.** Mistral does not publish Timestral's training
  data. M5 and Favorita are common forecasting benchmarks, so the model may
  have seen these series. The share series are aggregations the benchmarks do
  not use, and on synthetic qcgen feeds it cannot have seen Timestral keeps
  its advantage (above). That makes contamination an unlikely explanation,
  though a held-out real retailer would still be the stronger check.
- **Spent weeks.** Every split of both datasets had already been run with the
  engine baselines, except Favorita's reserve weeks 126-141. Those were held
  unrun to confirm a share-test change (`docs/real-data-favorita.md`). The
  2026-11-02 comparison above ran them as exploratory dev, unregistered,
  with both arms. That spent the reserve, which was a mistake, and no unrun
  weeks remain on either retailer. Any registered test of this change is
  therefore like-for-like (both arms on the same engine commit and cases),
  not a first look at unseen data.
- **The harness has read the test weeks.** The share-rule curves above
  include Timestral's latest-week statistics on M5 weeks 178-277 and
  Favorita 142-241 (2026-11-03), before any registration. The full engine
  has not run Timestral there. A registered test on those splits checks
  whether the full engine reproduces the harness, not how Timestral behaves
  on weeks nobody has looked at.
- **Normal tails.** The p-value beyond the outermost quantiles (1% / 99%) is
  extrapolated from a normal shape. It is checked by the clean rates above,
  not assumed.

## Registered predictions (before the test run)

Written 2026-11-03 by Claude at the author's direction. The author chose the
model (209M) and delegated the operating point and these predictions. They
are committed before either arm runs on the M5 test (weeks 178-277) or the
Favorita test (weeks 142-241), and both arms run on the registering commit.

**What this test can and cannot show.** The engine has run these splits
before. The share-rule harness has computed Timestral's latest-week
statistics on them (2026-11-03). On those weeks at q = 0.01 it gave M5:
engine 8/100 alarmed clean refreshes, 209M 1/100, 40% cut detection 0.86
against 0.96; Favorita: engine 26/100, 209M 33/100, 0.50 against 0.68. The
predictions below are built from those numbers and from the spent splits,
where the full engine roughly halved the harness's alarm gap. So the test
checks whether the full engine reproduces the harness at 100 refreshes per
retailer. It is not a first look at Timestral on unseen weeks.

**Arms.** `share_baseline: engine` and `share_baseline: timestral` with
`mistralai/Timestral-209M@9874d252ab0a`. Both use engine defaults otherwise,
including `temporal_fdr_q` = 0.01. The operating point stays the global q:
at the same q the full engine already gave equal clean alarms on the spent
splits, and q remains a per-feed choice, as decided on 2026-10-15
(`docs/real-data-favorita.md`). No share-only weight is added.

1. **Model:** Timestral-209M.
2. **M5 clean false alarms:** the Timestral arm raises no more than the
   engine arm (harness: 1 against 8).
3. **Favorita clean false alarms:** the Timestral arm raises at least as many
   as the engine arm and at most 7 more (harness: +7; spent splits: +1).
4. **Market movement:** pooled over 10%, 20% and 40% (60 cases per
   retailer), the Timestral arm catches at least as many as the engine arm on
   each retailer. Over both retailers it has more Timestral-only than
   engine-only detections.
5. **Unchanged:** missing stores, missing products (10 and 100), week
   restatement and recalculation detect the same number of cases in both
   arms. These checks do not read the share test.
6. **Failed registration:** any of 2-5 fails. Whatever the outcome, the
   default stays `engine`, because Timestral needs torch and a model
   download. If all hold, the doc recommends Timestral-209M as the share
   baseline for feeds that can carry that dependency. If any fail, the
   result is recorded as it is and Timestral stays experimental.

## Results

Pending the registered run.
