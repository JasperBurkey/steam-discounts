# Sensitivity to unobserved confounding (19_sensitivity.py)

Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2022) bounds for the ATT, worst case
(rho = 1). Benchmarks: a hidden confounder as strong as leaving out each group of observed
controls.

Settings: the all-game-events rows for reviews come from the full ensemble run, recovered
from its log (3-decimal precision; the run was stopped by the time limit). The
overlap-trimmed rows and the revenue-proxy rows (all game-events) are quick settings
(LEARNER=boost, CF_REPEATS=1).

| Sample | Outcome | ATT | RV | Bounds if as strong as discount history | ...review history | ...game & publisher |
|---|---|---|---|---|---|---|
| All | Reviews, during | +0.255 (0.022) | 0.071 | [-0.06, +0.57] | [+0.18, +0.33] | [+0.16, +0.35] |
| All | Reviews, net | +0.083 (0.019) | 0.032 | [-0.17, +0.33] | [-0.15, +0.31] | [+0.08, +0.08] |
| All | Revenue proxy, net | -0.067 (0.019) | 0.026 | [-0.29, +0.16] | [-0.06, -0.06] | [-0.07, -0.07] |
| Trimmed 0.1-0.9 | Reviews, during | +0.299 (0.028) | 0.133 | [+0.20, +0.40] | [+0.30, +0.30] | [+0.30, +0.30] |
| Trimmed 0.1-0.9 | Reviews, net | +0.117 (0.021) | 0.070 | [+0.06, +0.18] | [+0.12, +0.12] | [+0.12, +0.12] |
| Trimmed 0.1-0.9 | Revenue proxy, net | -0.005 (0.022) | 0.003 | [-0.02, +0.01] | | |

Reading:
- In the full sample, poor overlap inflates the bounds (games certain to discount get huge
  weights), so only a confounder weaker than the discount-history controls is ruled out.
- Where discounting and full-price games are comparable (propensity 0.1-0.9), the review
  effects survive a confounder as strong as any group of observed controls.
- **The negative net revenue effect is not robust to trimming.** In the comparable sample
  it is -0.005 (0.022): about zero. The full-sample -0.06 comes from games almost certain
  to discount, where the estimate leans on few comparable full-price games. The headline
  should say "sales roughly break even on net revenue where the comparison is credible,"
  not "sales lose money."
