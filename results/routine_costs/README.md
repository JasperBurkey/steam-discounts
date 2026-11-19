# What do discount routines cost, and who pays? (28_routine_costs.py)

Out-of-fold gains of the policy rule that passed the criterion (2-level tree learned on the
net revenue proxy; full settings, rerun Nov 19 with the same totals as the committed run:
+0.048 proxy, +0.038 lag-adjusted, 14.7% of choices changed), per game-event, broken down by
signs of routine. Mean gain in log points vs the actual choice; SE clustered by game.

| Group | Share of game-events | Share changed | Mean gain, proxy | Share of total gain (proxy / lag-adj.) |
|---|---|---|---|---|
| Habitual discounters (propensity > 0.9) | 75% | 12% | +0.059 (0.016) | 91% / 105% |
| Could go either way | 25% | 22% | +0.016 (0.031) | 9% / -5% |
| Same depth as previous sale | 55% | 11% | +0.009 (0.003) | 11% / 3% |
| Different depth from previous sale | 34% | 20% | +0.116 (0.041) | 83% / 90% |
| Publisher gave all games the same depth | 8% | 17% | +0.114 (0.071) | 20% / 23% |
| Publisher's depths differ | 56% | 10% | +0.032 (0.016) | 38% / 44% |
| Catalog 1 / 2-4 / 5-19 / 20+ | 24 / 23 / 22 / 32% | | +0.026 / +0.076 / +0.044 / +0.047 | |
| Outside publisher / self-published | 45 / 55% | | +0.061 (0.020) / +0.037 (0.021) | 57% / 42% |

Reading:
- **Habitual discounters are where the money is.** Essentially all of the rule's gain comes
  from games that discount almost every sale, matching the finding that net losses are
  concentrated there (19_sensitivity).
- **Repeating last sale's depth is not itself costly.** Games that repeat their depth gain
  almost nothing from the rule (+0.009); the gains come from games that changed depth,
  often in the wrong direction by the rule's lights. So the cost is not "sticking" but
  which depth habitual discounters land on.
- **Uniform publishers lose somewhat more** per game (+0.114 vs +0.032), but the difference
  is imprecise. The coarse rule rarely splits a uniform publisher's games into different
  depths (3%), so the cost there is the level of the common depth, not its uniformity.
- No clear gradient by catalog size.
