# Placebo sale dates (provisional)

Quick settings (LEARNER=boost, CF_REPEATS=1). Full-settings run pending.

Each sale moved 42 days earlier; controls recomputed at the fake date; game-events with a
discount in force during the fake windows dropped (12,121 game-events kept, 3,204 games
with full review histories). ATT on log reviews, SE clustered by game.

| Sample | Date | Week before | During | Weeks 1-2 after |
|---|---|---|---|---|
| All games | placebo | +0.025 (0.036) | +0.013 (0.030) | +0.002 (0.031) |
| All games | real | -0.109 (0.030) | +0.300 (0.032) | +0.138 (0.030) |
| Busier games | placebo | -0.049 (0.067) | -0.098 (0.094) | -0.081 (0.081) |
| Busier games | real | -0.159 (0.090) | +0.253 (0.119) | +0.226 (0.123) |

Reading: no effect at fake dates in any window, so the main estimates are not a standing
difference between discounting and full-price games. The week-before dip appears only at
the real date, so it is tied to the upcoming sale rather than to who discounts. It is not
proof the dip is buyers waiting: 16_review_flags shows reviews from non-Steam buyers dip
too, so part of it may be a general pre-sale lull.
