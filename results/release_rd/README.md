# The 30-day release rule as a natural experiment (22_release_rd.py)

Steam bars discounts in the 30 days after release, and seasonal sales are not exempt
(Steamworks "Discounting" docs). Running variable: age at the sale start - 30 days.
Local linear, triangular kernel, bootstrap SE over games (B = 300). Outcomes from
03h_pull_rd_reviews.py: review counts for all 8,043 game-events of paid games aged 4-55
days at a sale (7,739 with complete windows; 6,033 within 21 days of the cutoff). The
counts match the study's own review data for overlapping games (correlation 0.9999).

**First stage: strong.** The share discounting jumps by 0.54 / 0.57 / 0.66 at bandwidths
10 / 14 / 21 days (SE 0.02-0.03); about 10-19% discount at ages 15-29 (launch discounts
ending), 80-88% from age 30.

**Fuzzy RD: effect of discounting for games just made eligible** (ages ~30 days)

| Outcome (per week, log, relative to the week before the sale) | bw 10 | bw 14 | bw 21 |
|---|---|---|---|
| Reviews during the sale | +0.42 (0.11) | +0.45 (0.08) | +0.35 (0.06) |
| Reviews weeks 1-2 after | -0.00 (0.10) | +0.02 (0.07) | -0.00 (0.06) |
| Reviews weeks 3-4 after | -0.02 (0.10) | -0.04 (0.08) | -0.12 (0.06) |
| Revenue proxy during | +0.22 (0.10) | +0.25 (0.08) | +0.16 (0.06) |
| Revenue proxy, sale to week 4 | +0.06 (0.08) | +0.08 (0.07) | +0.01 (0.05) |

Reading: discounting causes a large spike during the sale, nothing after it (a slight dip
in weeks 3-4 at the widest bandwidth), and net revenue over the sale and four weeks after
that is close to zero, with intervals of roughly +/-0.1-0.15. This matches the
selection-on-observables estimates where overlap is good (19_sensitivity: net revenue
-0.005 in the propensity 0.1-0.9 sample), from variation nobody chose.

**Levels instead of changes** (log reviews in each window, no baseline): estimates swing
with the bandwidth (reviews during: -0.06 / +0.15 / +0.32, SE 0.15-0.24), because young
games' review levels vary hugely and decay fast after launch. The change from the week
before removes most of that.

**Caveats**
- *The week before the sale is not entirely clean.* Eligible games have somewhat fewer
  reviews that week (balance: -0.22 (0.13) / -0.14 (0.10) / -0.01 (0.08)). That could be
  buyers waiting for a discount they expect, the anticipation seen in the main results; it
  would inflate the "during" effect and the net figures above by up to that amount at
  narrow bandwidths. Other balance checks (first-week reviews, price, publisher size)
  show no jump.
- *Placebo cutoffs* (reduced form, bw 10, change during): ages 23 -0.05 (0.06), 26 -0.05
  (0.07), 44 +0.02 (0.06), against +0.23 (0.06) at 30. Ages 34-40 pick up the real jump
  inside their window (their first stages move too), and age 20 sits in the launch period
  where launch discounts end (-0.21, 0.07), so neither is a clean placebo.
- *Density:* releases are 25% more common in the two weeks after the cutoff than the two
  before (jump +0.23 to +0.35 log points), but the same test finds similar jumps where
  nothing happens; inconclusive.
- The estimate applies to month-old games, not the main sample's established games.
