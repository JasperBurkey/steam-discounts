# Buy-or-wait model (pipeline/15_buy_or_wait.py)

A calibrated dynamic model built from three QuantEcon lecture models: Finite Markov Chains
(discount beliefs, estimated from the price histories of 1,927 busier games), Job Search
(buy now or wait for a sale) and Discrete State Dynamic Programming (the solver; checked
against `quantecon.markov.DiscreteDP`). Run Nov 17, 2026, 8 minutes.

## Calibration
Targets are doubly robust ATEs for busier games: the week before the sale, during it, weeks
1-2, 3-4 and 5-6 after, and the sale-week effect by depth. Patience is chosen by fit on the
grid 0.97-0.998. Best fit: beta 0.995, delta 0.083 (weekly chance of losing interest),
pi 0.50 (share who know the sale price a week ahead), a mean review lag of 2.5 weeks, and
widely spread valuations (sigma 2.9). Fit loss 24.8 over 8 moments. At the other patience
values the fit is worse (34-41), with buyers who drop out within a week or two.

Held-out checks: the revenue proxy computed in the model is -0.048 net (data -0.072,
SE 0.021) and -0.627 during the sale (data -0.535, SE 0.028).

## Findings
1. **Waiting is needed.** Without buyers who wait, the fit loss rises from 24.8 to 244: the
   model can't produce the pre-sale dip and understates the sale-week jump by a third.
2. **The elevated reviews after a sale are review lag**, not lasting demand.
3. **The revenue proxy.** Over the six-week window it understates the true revenue cost of
   one sale's discount (-0.048 vs -0.124 for the mix of discounted games). Over a whole year
   the true cost is much smaller (-0.017), because waiting buyers who miss a sale mostly buy
   in the next one at a discount anyway.
4. **Schedules** (commitment; beliefs match the schedule): every schedule earns less than
   never discounting; 61%+ every sale earns least (-17% at the best fit, -6% to -7% at the
   other patience values). The ordering holds at 2x and 3x sale-week traffic.

## Limits
Pure pricing: discounts change only the price. Visibility for discounted games, follow-on
spending (DLC, friends) and reviews attracting later buyers are left out, and the causal
estimates include them. One representative game type (rare and habitual discounters differ
only in beliefs). Calibrated, not estimated: no standard errors. The calibration surface has
more than one local optimum, so the patience grid is reported in `sensitivity.csv`.
