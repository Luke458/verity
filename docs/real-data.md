# Real sales: Walmart's M5 data

Every other result in this repository is on synthetic data. This is the
first test on real retail sales. The faults are still injected, because no
public dataset comes with labelled refresh faults, but the noise, seasonality,
stock-outs and product churn they sit in are real.

```sh
pip install -e ".[test]"
mkdir -p data/real/m5 && cd data/real/m5
for f in calendar.csv sales_train_validation.csv sales_train_evaluation.csv sell_prices.csv; do
  curl -sSLO "https://huggingface.co/datasets/kashif/M5/resolve/main/$f"; done
cd ../../..
python -m experiments.real.m5 build --raw data/real/m5 --out data/real/m5/weekly.parquet
python -m experiments.real.m5 run --weekly data/real/m5/weekly.parquet --split test --jobs 4 \
  --out reports/real/m5-test.json
python -m experiments.real.m5 release --raw data/real/m5 --out reports/real/m5-release.json
```

## Data

[M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
([mirror](https://huggingface.co/datasets/kashif/M5); research use under the
competition's terms, never committed here) holds daily unit sales of 3,049
products in 10 Walmart stores across 3 states (CA, TX, WI), 2011-01-29 to
2016-05-22, with weekly shelf prices. `experiments/real/m5.py build`:

- aggregates days to Walmart weeks (Saturday to Friday) and drops the trailing
  partial week: 277 full weeks;
- keeps store x product x week rows that sold, because a sales feed has no row
  for a product that did not sell: 5,089,796 rows;
- prices units at that week's shelf price to give dollars (every sold row has
  a price).

The engine sees `store_id`, `state_id`, `commodity_id` (M5's 7 departments),
`product_id`, `dollar` and `units`, at one stage. `CONFIG` in the module only
maps these columns and the single stage. Every threshold is the engine default,
none was tuned on M5, and no business calendar is declared.

**Refreshes.** Refresh *k* compares the table through week *k*-1 with the table
through week *k*. M5's two public releases (`sales_train_validation`, through
2016-04-24, and `sales_train_evaluation`, through 2016-05-22) agree on all
58,327,370 overlapping cells, so the source does not restate history and
these refreshes are clean by construction. Any alarm on one is a false alarm.

**Faults** are the generator's single-stage injectors applied to the current
version: a one-week market movement in one department (5%, 10%, 20%, 40%), one
store missing, 10 or 100 random products missing, one overlap week restated by
10%, and a broad recalculation (row noise, sigma 0.4%).

**Splits.** Annual baselines need 104 completed weeks, so the first assessable
refresh appends week 106. Dev: weeks 106-125 (20 clean refreshes; each fault at
every 4th week). Test: weeks 178-277 (100 clean refreshes, 2014-06 to
2016-05; each fault at every 5th week, so 20 per fault, each fault size on its
own weeks).

## What dev found: two engine defects

The first dev run raised an alarm on **all 20 clean refreshes**. Two causes:

1. **A single-stage feed could never pass reconciliation.** When the report
   stage is the analysis stage, the engine passed no report frame. Mass balance
   then became *unavailable required evidence*, and the marker and price checks
   did not run at all. Now mass balance is `NOT_APPLICABLE` (informational)
   and the other checks run on the one frame. The synthetic suites always
   have separate stages, so they never exercised this path.
2. **Real product churn is larger than the materiality floor.** The
   missing-entity rule flags products that sold last week and in at least half
   of the 8 weeks before, but sold nothing in the new week. On M5 such products
   carry a median 0.5% of a clean week (90th percentile 1%, 99th 1.7%). Even
   products that sold in all of the last 8 weeks go missing, about 15 per week:
   stock-outs and delistings. Against a 0.1% materiality floor, the rule fired
   on 99% of clean weeks. The synthetic generator's products sell
   every week, so this never showed. Now an absence escalates only if its
   missing share also exceeds every share the same rule gives at each of the
   previous 52 weeks of the feed (`entity_absence_baseline_weeks`, 0 disables).
   Under exchangeability that is a false-alarm rate of about 1 in 53 per entity
   type. On synthetic data it changes nothing: store absences have a zero
   baseline, and on both profiles the missing-products and missing-stores
   sweeps (seeds 5001-5010, 1%-40%) match the registered sweeps cell for cell.
   The cohort is unchanged (72/72, 1/60).

**Disclosure.** The churn quantiles above were computed over all M5 weeks,
including the test weeks, before the baseline was designed. They describe the
missing-share statistic only; the engine was not run on any test week. The
baseline's two settings (52 weeks; exceed the maximum) were chosen on
principle, not fitted to those quantiles.

| Dev (weeks 106-125) | First run | After the fixes |
|---|---|---|
| Clean refreshes alarmed | 20/20 | **0/20** |
| Market movement 5% / 10% / 20% / 40% (target department flagged) | 0/5, 0/5, 1/5, 4/5 | 0/5, 0/5, 1/5, 4/5 |
| One store missing | 5/5 | 5/5 |
| 10 / 100 products missing | 5/5 / not run | 1/5, 5/5 |
| One week restated by 10% | 5/5 | 5/5 |
| Broad recalculation (sigma 0.4%) | 5/5 | 5/5 |

On the first run, every faulted refresh was flagged anyway, by the absence
false alarm. Only the target-specific rules in the table count.

## Registered predictions (before the test run)

Detection is counted only when the check that should see the fault flags it:
the target department's latest-week series for a market movement, a material
store or product absence for missing entities, and a history or week-revision
finding for restatements. Engine code is that of the registering commit.

1. **Clean false alarms: at most 6 of 100** test refreshes are not PASS.
   Product absences are expected to give about 2 by construction. Temporal
   QC is expected to give about 1 at q = 0.01 (0/20 on dev).
2. **Market movement:** 40%: at least 12/20; 20%: 2 to 12 of 20; 10%: at most
   4/20; 5%: at most 2/20. Real weekly department noise is larger than the
   synthetic `realistic` profile's (20/20, 12/20, 2/20 at 40/20/10%).
3. **One store missing** (about a tenth of the week): at least 19/20.
4. **100 products missing:** at least 18/20. **10 products missing:** at most
   6/20; such an absence is within this feed's own weekly churn.
5. **One week restated by 10%:** at least 19/20. **Broad recalculation:** at
   least 19/20.
6. **M5's own releases** (validation, 273 full weeks, to evaluation, 277): PASS,
   with no history finding. This is a single real refresh, an illustration rather
   than a rate.

## Results (test weeks 178-277, engine `4513c0f6a164`)

Five of six predictions held. **Prediction 1 failed: 14 of 100 clean real
refreshes raised an alarm, against at most 6 predicted** (Wilson 95%
0.085-0.221). Results: [`results/m5-test.json`](results/m5-test.json).

| | Result | Prediction | |
|---|---|---|---|
| Clean refreshes alarmed | **14/100** | at most 6 | **failed** |
| Market movement 40% | 19/20 | at least 12 | held |
| Market movement 20% | 9/20 | 2 to 12 | held |
| Market movement 10% | 1/20 | at most 4 | held |
| Market movement 5% | 0/20 | at most 2 | held |
| One store missing | 20/20 | at least 19 | held |
| 100 products missing | 20/20 | at least 18 | held |
| 10 products missing | 0/20 | at most 6 | held |
| One week restated by 10% | 20/20 | at least 19 | held |
| Broad recalculation | 20/20 | at least 19 | held |
| M5's own releases (273 to 277 weeks) | PASS, no history finding | PASS | held |

**Where the 14 false alarms came from** (some refreshes have more than one):

- **Price per unit, 9.** M5's dollars per unit rise from 2.70 (first year) to
  3.15 (last). The dollar-per-unit check compares the new week with the
  median and MAD of *all* earlier weeks, so inflation alone puts later weeks
  near its k = 4 robust-z cut (z = 4.0-4.3 at flagged weeks 194, 203, 204,
  212 and 221; 1.9 and 3.4 at unflagged weeks 230 and 250). The check assumes stationary prices. The synthetic generator's
  prices are stationary, so this never showed. A trailing reference window
  is the obvious fix; it is not applied here, because this run is the test.
- **Latest-week share tests, 8** (at q = 0.01, about 1 was expected). They
  are real department share movements in Walmart's data, not faults. Some are
  large against expectation (HOBBIES_1 dollars +19% and +38% at weeks 205 and
  212, FOODS_1 units -11% at 197), others small moves in a share that rarely moves (HOBBIES_2,
  weeks 247-248). The engine is built to surface genuine market movements, and
  a reviewer may want some of these. Still, the share test's spread, taken from
  each department's own trailing or year-over-year history, under-states how
  much real department shares move, so the nominal 1% per-refresh rate does not
  hold on this feed.
- **Product absence, 1** (week 266, 1.9% of the week against a 52-week
  maximum of 1.5%), about the 2 expected by construction.

**What held up on real data.** Store, product-outage, restatement and
recalculation faults were caught every time. A department-wide drop of 40% was
caught 19/20, and 20% 9/20, about as on the synthetic `realistic`
profile (20/20 and 12/20). The absence baseline separated normal churn (10
products, 0/20) from an outage (100 products, 20/20). M5's own release-to-release
refresh passed, with its 7 missing products recognised as normal churn.

**Not shown.** M5 has no real pipeline faults and no real restatements, so
these are injected faults in real noise, not real incidents. The refreshes
append exactly one clean week; late-arriving data, real revisions and real
outages remain untested. A fix to either false-alarm source needs fresh weeks:
weeks 126-177 have not been run, and a second dataset (Favorita) would test
transfer.
