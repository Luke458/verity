# Real sales, second retailer: Corporación Favorita

The M5 test ([real-data.md](real-data.md)) ran the engine on one retailer's
real sales. This is the transfer test: a second retailer, a different
country, a different feed shape, the same injected faults and scoring, and
predictions registered before the held-out weeks are run.

```sh
pip install -e ".[test]"
mkdir -p data/real/favorita && cd data/real/favorita
for f in train items stores; do
  curl -sSLO "https://huggingface.co/datasets/hsnalmasri/favorita/resolve/main/$f.parquet"; done
cd ../../..
python -m experiments.real.favorita build --raw data/real/favorita --out data/real/favorita/weekly.parquet
python -m experiments.real.favorita run --weekly data/real/favorita/weekly.parquet --split test --jobs 2 \
  --out reports/real/favorita-test.json
```

Each worker holds the 23-million-row table, about 7 GB; `--jobs 2` is enough.
A run is resumable: each case is kept under `<out>.work/<engine>-<adapter>/`,
so a crashed worker costs one case.

## Data

[Favorita](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)
([mirror](https://huggingface.co/datasets/hsnalmasri/favorita), no dataset
card; research use under the competition's terms, never committed here) holds
daily unit sales of 4,100 items in 54 stores across 16 Ecuadorian provinces,
2013-01-01 to 2017-08-15: 125,497,040 rows, matching the Kaggle release.
`train.parquet` sha256 `c35a8d957ecc...` (the full digests are in the
`weekly.json` that `build` writes). `experiments/real/favorita.py build`:

- aggregates days to Wednesday-to-Tuesday weeks, so the last day
  (2017-08-15, a Tuesday) closes a full week: 241 weeks from 2013-01-02;
- keeps store x item x week rows with nonzero net sales: 23,494,434 rows,
  4,036 items;
- joins each item's family (33, the engine's `commodity_id`) and each
  store's province (`state_id`).

It differs from M5 in ways that matter to the engine: **no prices** (the
engine runs on units alone, so the price-per-unit check does not apply);
**fractional units** (perishables sold by weight) and **returns** (negative
daily values); 54 stores against 10; and **real structural events** (store
closures, openings and a coverage gap, below), where M5 had none. The engine
runs with its defaults; `CONFIG` only maps the columns, the single stage and
units as the only measure.

**Refreshes, faults, splits.** As for M5: refresh *k* compares the table
through week *k*-1 with the table through week *k*; the same nine injected
faults (market movement 5/10/20/40% in one family, one store missing, 10 or
100 products missing, one overlap week restated by 10%, broad recalculation).
The injectors' integer rounding of units would destroy fractional sales, so
the adapter carries units through the injectors' unrounded dollar path and
takes them back. Splits: dev weeks 106-125 (each fault every 4th week);
reserve 126-141, unrun; **test 142-241** (100 clean refreshes, 2015-09 to
2017-08; each fault every 5th week, 20 per fault).

**Clean is not quite clean here.** Favorita's own history has real events,
so an alarm on a clean refresh is not always false. A clean alarm counts as a
**real-event alarm**, not a false alarm, only when its one finding is a
material absence and every materially absent entity type is a store, province
or family of which some entity really disappeared that week (traded the week
before, no rows in the new week); `real_event_alarm` in the module. Anything
else counts as false, even when a real event caused it: a closed store that
moves its province's latest-week series is a false alarm under this rule. A
product absence is always false: product churn is the noise the absence
baseline is meant to absorb.

## What dev found

The first dev run (weeks 106-125) raised an alarm on **all 20 clean
refreshes**, and every refresh was INCOMPLETE. The data hold a coverage gap
that M5 never had: in week 106 (2015-01-07 to 13) eight families (BABY CARE,
CELEBRATION, HOME CARE, LADIESWEAR, MAGAZINES, PET SUPPLIES, PLAYERS AND
ELECTRONICS, SCHOOL AND OFFICE SUPPLIES) and 605 items drop out of the feed,
and national units fall from about 5M to 3.4M a week. They come back 12 to
20 weeks later (all families by week 126, all but two items by week 141).
The same families had shorter gaps in 2014 (weeks 58-60, 66-77, 84-86). Real
stores also close and open: store 24 is dark for weeks 68-81, store 14 for
84-88, store 12 closes for good at week 118; eight stores open, two of them
in the test weeks (store 22 at week 145, store 52 at 225). Fixing what this exposed took four engine changes, one
generator fix and one choice about the simulated operator.

1. **A leaf with no current sales was required evidence.** A family that
   sold nothing in the new week or the week before had no testable share, so
   its latest-week test was unavailable and, being required, made every
   refresh INCOMPLETE. Such a leaf is now a *dormant series*, and a leaf
   first seen fewer than `temporal_min_history` weeks ago a *new series*;
   both are reported and not required (`qc/temporal.py`).
2. **An approved closure stopped counting after the week it was approved.**
   Approving the week-106 absences restated week 106 like-for-like, but the
   next refresh read the closed families' history back into every baseline,
   so the gap dominated every share test after it (market movement 0/20 at
   every size). An approved closure now *stands*: while its entry is in
   effect and the entity has no rows, later refreshes keep excluding it
   (`standing_closures`). An entity that trades again is no longer excluded.
3. **The absence baseline counted approved absences as churn.** The
   missing-entity rule escalates only above the largest share the same rule
   gave at each of the previous 52 weeks. That baseline was measured on the
   raw history, so the approved week-106 gap (and store 24's closure) set it,
   and 100 products missing was caught 0/5. It is now measured on the
   like-for-like history, and leaves out every entity an approved closure
   covered at some week of its window, even after the entity returns
   (`explained_closures`): an explained absence is no evidence of how much
   the feed churns on its own. Before that last part, the baseline jumped
   between 1.6% and 13% from week to week as returning items dropped their
   standing exclusion.
4. **The market-movement injector could pick a family that no longer
   traded.** It drew from every family in the table's history, so 4 of the
   first 14 dev injections hit a family in the gap and changed nothing. It
   now draws from the families that trade in the target week. On M5 and on
   the synthetic profiles, every family trades every week, so the pick is
   unchanged.
5. **Speed.** Restating 600 closures filtered the 10-million-row history
   once per entity; it is now one filter per entity column (same result).

**The simulated operator.** The engine flags the week-106 absences; the
adapter plays the operator and approves exactly the families and items the
engine lists as missing (`coverage_change_registry`), effective from week 106,
and ends each approval at the week before that entity trades again. Ending it
there uses the entity's return week from the data; an operator would close
the approval when coverage resumes. An open-ended approval would instead
silently explain any later disappearance of the same entity, injected faults
included. Nothing else is approved: no store closure or opening, no later
event.

| Dev (weeks 106-125) | First run | Registered engine |
|---|---|---|
| Clean refreshes alarmed | 20/20 | 5/20: 4 latest-week share tests, and week 106 itself (PASS_WITH_EXPLANATION, the approved gap) |
| Market movement 5% / 10% / 20% / 40% | 0/5 each | 0/5, 0/5, 1/5, 0/5 |
| One store missing | 2/5 | 2/5 |
| 10 / 100 products missing | 0/5, 0/5 | 0/5, 5/5 |
| One week restated by 10% | 5/5 | 5/5 |
| Broad recalculation | 5/5 | 5/5 |

**Why market movement is so much weaker than on M5** (M5 dev 40%: 4/5,
test 19/20). Favorita has 33 families and 16 provinces, 49 latest-week leaves
against M5's 10, so the multiplicity correction asks for more: a FROZEN FOODS
drop of 40% at week 109 gives share t = -3.87 (p = 0.0007), adjusted p 0.028,
not below q = 0.01. And the families' share histories hold the coverage gaps:
the engine's own per-leaf detectable change (a conservative bound) at week
115 lets 13 of 24 trading families show a 40% move, at week 125 only 4 of 27;
PRODUCE, HARDWARE and both HOME AND KITCHEN families sit at the 100% cap.
The gaps leave the trailing window after about week 178.

**Why one store missing is weak on dev.** Store 24 went dark at week 68,
2.2% of a week. Until week 120 that absence is in the store baseline's 52
weeks, so a store smaller than 2.2% passes as normal churn; the three misses
are weeks 106, 110 and 114. This is the price of the absence baseline (an
unapproved real incident blinds the rule for a year, by design) and is
registered, not tuned.

**Checks on the registered engine (`8c67828f00d8`).** Synthetic sweeps
(seeds 5001-5010, `small` and `realistic`, 640 cases each) are identical,
case for case, to the sweeps before these changes (compared in two steps) in status, detection, named
causes and lineage; the registered cohort is unchanged (132 cases, 72/72,
1/60, lineage 1.000, cause 0.917). M5's spent confirm weeks (169 cases) rerun
identically, with one caveat: twice, a case came back different in a
parallel batch (a clean week as DATA_CONTRACT_FAILURE; another as
INCOMPLETE) and identical when rerun alone. This workstation corrupts memory
under load (two 4.5 PB allocation failures and a segfault during this work);
see the rerun rule below.

**Disclosures.**
- The structural facts above (when stores, provinces, families and the
  week-106 items first and last trade) were read from all 241 weeks,
  including the test weeks. They describe entity existence only, and are
  what the real-event rule and the simulated operator need; no sales
  statistic of a test week was looked at before registration.
- The alternatives to the absence baseline that were tried and rejected
  (single-entity materiality, a Tukey fence, sequentially excluding escalated
  weeks) were evaluated on M5 weeks 106-277 (spent) and Favorita weeks up to
  125.
- Dev was run several times while fixing; the table is the registered
  engine's run.

## Registered predictions (test weeks 142-241)

100 clean refreshes; each fault at every 5th week, 20 per fault and size, on
its own weeks. Engine code `8c67828f00d8`, the registering commit's. Detection
rules as for M5. A clean alarm is false unless `real_event_alarm` holds.

**Rerun rule.** A case whose status is not PASS, PASS_WITH_EXPLANATION or
INVESTIGATE (none has occurred in any published real-data run, 579 cases) is rerun
once on its own and the rerun is counted; both results are reported. No other
case is rerun.

1. **Clean false alarms: at most 25 of 100** (dev 4/20 share tests in a
   window full of the coverage gap; M5's share tests ran about 3x nominal).
   Product absences are expected to give about 2 by construction. The
   2016-04-16 Ecuador earthquake falls in week 172; its week and the weeks
   after are likely to move provincial and family shares and are counted as
   false alarms under the rule (it is not a disappearance).
2. **Real-event alarms: at most 2.** No store or province disappears in the
   test weeks (store 12 is already gone); BABY CARE's two-week gap at weeks
   151-152 is far below materiality.
3. **Market movement:** 40%: 3 to 14 of 20; 20%: at most 6; 10%: at most 3;
   5%: at most 2.
4. **One store missing: at least 15 of 20.** Through week 170 the store
   baseline holds store 12's closure (1.1% of a week, larger than 10 of 50
   stores); after it, no real store absence remains in the window.
5. **100 products missing: at least 16 of 20. 10 products missing: at most
   4 of 20.**
6. **One week restated by 10%: at least 19 of 20. Broad recalculation: at
   least 19 of 20.**
