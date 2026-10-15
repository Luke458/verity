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

## Results (test weeks 142-241, engine `8c67828f00d8`)

Five of six predictions held. **Prediction 1 failed: 30 of 100 clean real
refreshes raised a false alarm, against at most 25 predicted** (Wilson 95%
0.219-0.396). Results: [`results/favorita-test.json`](results/favorita-test.json).
The run needed four restarts after worker crashes (it resumes per case).

| | Result | Prediction | |
|---|---|---|---|
| Clean false alarms | **30/100** | at most 25 | **failed** |
| Real-event alarms | 1 (store 18, week 190) | at most 2 | held |
| Market movement 40% | 8/20 | 3 to 14 | held |
| Market movement 20% / 10% / 5% | 3/20, 0/20, 0/20 | at most 6, 3, 2 | held |
| One store missing | 18/20 | at least 15 | held |
| 100 / 10 products missing | 20/20, 2/20 | at least 16; at most 4 | held |
| One week restated / broad recalculation | 20/20, 20/20 | at least 19 each | held |

**The rerun rule applied once.** Clean week 151 came back INCOMPLETE; rerun
alone it is INCOMPLETE again, so it is deterministic and counts as a false
alarm. It is an engine defect, found here and not fixed: BABY CARE sold the
week before and nothing in week 151, and a leaf that vanishes in the new week
is treated as a missing required observation. The dormant-series fix only
covered leaves already absent the week before. A vanished leaf is the
lifecycle checks' evidence, as a dormant one is (BABY CARE's absence is far
below materiality).

**Where the 30 false alarms came from:**

- **Latest-week share tests, 25 refreshes.** 7 are the Christmas weeks
  (155-156 in 2015, 204-208 in 2016), with up to seven families flagged at
  once; the year-ago weeks are aligned (Christmas falls in weeks 104, 156 and
  208), so the holiday mix simply moves more than the share spread allows.
  8 are families with feed gaps of their own: MAGAZINES at +2,909% and
  +773% in weeks 145-146 as it rebuilds after the gap, HOME AND KITCHEN II
  and I, and SCHOOL AND OFFICE SUPPLIES at +307% to +849% in weeks 172-174
  and 221-222 (consistent with the coastal school year starting in April).
  In 2016 its year-ago weeks fell in the 2015 gap; in 2017 the season came
  two to three weeks earlier than in 2016. Weeks 172-174 also carry
  the 2016-04-16 earthquake: GROCERY I +65% at 172 and Manabi, the worst-hit
  province, +176% at 173. 10 are other share movements (GROCERY II -54% and
  -61% at 185-186, Loja -28% and +31%, among others).
- **Product absences, 3** (weeks 161, 180, 223: 0.47-0.59% of the week
  against 52-week maxima of 0.26-0.55%), about the 2 expected.
- **The vanished leaf, 1** (week 151, above).
- **A real closure counted as false, 1.** At week 191 store 25, the only
  store in Santa Elena, goes dark; the alarm also carries a latest-week
  finding, so the registered rule does not credit it.

**A registration statement was wrong.** The registration said no store or
province disappears in the test weeks. Store 18 is dark for weeks 190-204 and
store 25 (with its province) for 191-199; the existence check behind that
statement missed temporary gaps. It changed the count by at most one: week
190 is the real-event alarm, week 191 a false alarm under the rule.

**What held up.** Restatement, recalculation and 100-product outages were
caught every time on a second retailer with fractional, units-only sales.
One store missing was caught 18/20; both misses are the registered blindness
of the absence baseline (store 15 at 1.03% of the week under store 12's 1.1%
closure; store 14 at 0.92% under store 18's 1.23% at week 190). Normal
product churn stayed separate from an outage (10 products 2/20, 100 products
20/20).

**What did not.** Department-level market movement transfers poorly: 40%
drops were caught 8/20 (M5: 19/20), 2 of 7 before week 178 and 6 of 13 after,
when the coverage gap has left the baselines. The misses are mostly small or
erratic families (AUTOMOTIVE, PET SUPPLIES, HARDWARE, BABY CARE, MAGAZINES,
HOME AND KITCHEN I). And the share tests' over-alarm, about 3x nominal on M5,
is far worse here: 25 of 100 refreshes with a share flag, against about 1 at
the nominal rate. On a feed with real coverage gaps and a moving holiday, a
quarter of clean refreshes would need a person to look.

**Not shown.** Favorita has no labelled pipeline faults, so these are
injected faults in real noise. Weeks 126-141 (reserve) are unrun.

## After the test: the vanished-leaf fix (engine `df6b7cc9db6a`)

**The change.** A leaf with no row in the new week is now reported and not
required whether it was already absent the week before (dormant) or sold
then (vanished); only the national series stays required. Its absence is
the missing-entity check's evidence, which judges it on materiality, as it
already did for dormant leaves (`qc/temporal.py`). The unit test that pinned
the old behaviour (a vanished leaf required) was changed with it.

**It cannot change a detection.** INVESTIGATE outranks INCOMPLETE, so the
change alters a status only when nothing else fails, and no fault is credited
through INCOMPLETE: store and product outages are credited only through a
material absence, market movement through its family's share flag,
restatement and recalculation through the revision checks.

**Effect on the spent test weeks (post hoc).** In weeks 107-241 a province
or family vanishes from the clean feed twice: BABY CARE at week 151 and
Santa Elena (store 25's closure) at week 191. Rerun on the fixed engine,
week 151 is PASS (it was the INCOMPLETE false alarm), and week 191's only
remaining finding is the material province absence, so the registered rule
now credits it as a real-event alarm. Apart from the status and the dropped
temporal finding, both records are identical to the published ones. Clean false alarms would be 28/100 and real-event alarms 2,
against the registered at most 25 and at most 2. This is post hoc: the
registered result stays 30/100, failed.

**Why the reserve weeks were not spent on it.** The plan was to confirm the
fix on the unrun weeks 126-141, as the M5 price fix was confirmed. Nothing
vanishes in those weeks' clean feed (an existence check, as before), so the
run could not exercise the change; the weeks stay unrun for a change they
can test, such as one to the share tests.

**Synthetic check.** Synthetic sweeps (seeds 5001-5010, `small` and
`realistic`, 640 cases each) are identical case for case to the registered
engine's, and the registered cohort is unchanged (132 cases, 72/72, 1/60,
lineage 1.000, cause 0.917).
