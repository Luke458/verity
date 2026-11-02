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

**Favorita: no fewer clean alarms.** The motivating ceiling (26 of 31 test
clean alarms decided by the share test) does not move on the spent splits.
The clean alarms both arms raise are mostly large real share moves that any
baseline would flag: HOME AND KITCHEN I at +243%, SCHOOL AND OFFICE at +363%
(back-to-school), BABY CARE at +165-265%. Timestral swaps some alarms rather
than removing them, and it reports more series per alarm (week 111: 7 series
against 2) with much smaller p-values (often below 1e-6, some 0). Its 10-90
quantile gap read as a normal sigma is too narrow for Favorita's volatile
leaves, so a real move scores as near impossible. Week 126 is the source's
known perishables gap ending: PRODUCE's share sits near 0.05% for weeks
106-125 and returns to 16%. Timestral flags it; the engine does not. That is
a source event, but the labels count it as a clean false alarm.

**Where Timestral helps:** detection of 40% market movements on Favorita
(0/9 to 5/9 across dev and reserve) and, with the 209M model, both clean
alarms and detection on M5 confirm.

On M5 the two baselines
disagree case by case rather than one dominating: on confirm, Timestral
removed three clean alarms and added one (week 159, FOODS_1 units), and of the
market movements it caught five the engine missed and missed three the engine
caught. Two of those three are HOBBIES_2, a 0.5% department with a strong
annual cycle, where the seasonal year-over-year baseline is tighter (relative
spread 11.5% vs 15.5% at week 129) and the 40% drop's adjusted p was 0.013
against q = 0.01. With about two and a half years of weekly history the
pretrained model resolves the annual cycle less well than an explicit
year-over-year comparison.

**Hardware fault, 2026-10-28.** The Timestral-209M confirm run segfaulted
twice, and all four Favorita dev/reserve runs (both arms) failed with indices
and allocation sizes of a valid value plus exactly 2^52: a single bit flipping
in RAM. The memory overclock was turned off (3000 MHz, default timings);
`memtester` then passed on 40 GB in 4 parallel instances, the rebuilt Favorita
weekly table matched the existing one exactly, and every run above completed
without error. The 2026-10-28 M5 results reproduced exactly. Nothing from the
faulty period is used.

## Design questions before registering

1. **Tail calibration.** The normal reading of the quantile gap is
   overconfident on Favorita. A heavier tail (Student-t on the same gap), or
   a scale calibrated on spent weeks, would temper the p-values. This is
   the likeliest route to fewer Favorita alarms, and it should be settled
   on the spent splits before registering.
2. **Model size.** 209M beats 28M on M5 confirm (clean 2 vs 3, 20% movement
   9 vs 6) at about 6x the CPU. It has not run on Favorita.
3. **Seasonal leaves.** HOBBIES_2 still loses to the year-over-year baseline.
   A hybrid (Timestral as a third candidate in the engine's min-spread
   choice) would keep it, at the cost of a more complex rule.
4. **One rule for both datasets.** A per-dataset choice made after seeing
   these splits would not be a registered test.

## Caveats

- **Possible contamination.** Mistral does not publish Timestral's training
  data. M5 and Favorita are common forecasting benchmarks, so the model may
  have seen these series. The share series are aggregations the benchmarks do
  not use, which lowers the risk without removing it. A held-out retailer the
  model cannot have seen would settle it.
- **Spent weeks.** Every split of both datasets has been run before with the
  engine baselines. The comparison below is like-for-like (both arms on the
  same engine commit and cases), not a first look at unseen data.
- **Normal tails.** The p-value beyond the outermost quantiles (1% / 99%) is
  extrapolated from a normal shape. It is checked by the clean rates above,
  not assumed.

## Registered predictions (before the test run)

*To be written by the author before the M5 test (weeks 178-277) and Favorita
test (weeks 142-241) splits run with `share_baseline: timestral`. Both arms
run on the registering commit.*

1. Model size registered for the test:
2. M5 clean false alarms (engine arm, Timestral arm):
3. Favorita clean false alarms (engine arm, Timestral arm):
4. Market movement detection at 10% / 20% / 40%, per dataset and arm:
5. Missing stores / products, restatement, recalculation: unchanged (these
   checks do not read the share test).
6. What counts as a failed registration:

## Results

Pending the registered run.
