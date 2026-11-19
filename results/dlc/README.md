# DLC (32_dlc.py)

Each study game's most-reviewed DLC: reviews around each sale since 2024 (03k) and price
history (03l, 984 DLCs). Quick settings (LEARNER=boost, CF_REPEATS=1).

**When the base game is discounted, how often is its DLC?** Base discounted: DLC also
discounted in 4,342 game-events, DLC at full price in 1,630. Base at full price: DLC at
full price 501, discounted 58.

**A. Spillover to DLC** (log DLC reviews vs the window 21-8 days before; ATT of
discounting the base game)

| Sample | DLC reviews, during | DLC reviews, weeks 1-2 after | Base reviews, during |
|---|---|---|---|
| DLC at full price (clean; 2,131) | +0.31 (0.11) | +0.07 (0.13) | +0.41 (0.11) |
| Both discounted vs neither (4,843) | +0.26 (0.06) | +0.13 (0.07) | +0.36 (0.06) |

A base-game discount does sell DLC at full price: DLC reviews rise about as much as the
base game's own during the sale. But the top DLC's revenue is small next to the base game:
about 0.8% of base-game revenue in the pre-sale window (reviews x regular price, assuming
similar review rates), so the extra DLC revenue is roughly 0.2-0.3% of base revenue, an
order of magnitude short of offsetting the -0.06 net revenue proxy even if a game's other
DLCs add several times as much.

**B. Main results by monetization** (ATT)

| Games | n | Reviews, during | Revenue proxy, net |
|---|---|---|---|
| No DLC | 19,996 | +0.237 (0.026) | -0.070 (0.018) |
| Has DLC | 11,806 | +0.276 (0.032) | -0.043 (0.025) |
| No DLC and no in-app purchases | 19,830 | +0.231 (0.026) | -0.076 (0.018) |
| In-app purchases | 1,004 | +0.49 (0.26) | +0.11 (0.21) |

The net loss is at least as large for games with no add-on revenue at all, so missing DLC
or in-app revenue cannot explain it. In-app games are too few to read.
