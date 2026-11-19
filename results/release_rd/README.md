# The 30-day release rule as a natural experiment (22_release_rd.py)

Steam bars discounts in the 30 days after release, and seasonal sales are not exempt
(Steamworks "Discounting" docs). Running variable: age at the sale start - 30 days.
Local linear, triangular kernel, bootstrap SE over games.

**First stage: strong.** Across 11,816 game-events of paid games near the cutoff (all
games with price histories), the share discounting jumps by 0.54 / 0.57 / 0.67 at
bandwidths 10 / 14 / 21 days (SE 0.02-0.03); 0.61-0.73 dropping ages 29-30. In 3-day bins:
about 9-19% discount at ages 15-29 (mostly launch discounts ending), 80-88% from age 30.

**Outcomes: too few games.** Reviews exist only for the 3,850 study games, which are mostly
older, leaving 1,020 game-events (913 games) near the cutoff. First stage there 0.30-0.52.
Fuzzy RD standard errors are 0.5-1+ log points, so the design cannot yet say anything
about the effect. Point estimates are near zero to slightly negative for reviews after the
sale; nothing is close to significant.

**Balance:** reviews in the first 7 days, price and publisher catalog size show no jump.

**Density:** releases are 25% more common in the two weeks after the cutoff than the two
before (2,367 vs 1,870 game-events), and the local linear jump is +0.23 to +0.35 log
points. But the same test finds jumps of -0.19 and +0.23 at ages 44 and 51, where nothing
happens, so the weekly lumpiness of release dates makes its SE too small. No clear
evidence that developers time launches around the cutoff yet.

**What it needs:** reviews for the ~10,000 games in the universe near a cutoff (window
counts only, via the existing review pull), which would multiply the outcome sample about
tenfold. Placebo cutoffs at ages 20 and 40 show first-stage changes of +0.09 and -0.10
(launch-discount endings and noise), far smaller than at 30.
