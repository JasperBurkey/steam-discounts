# Spillovers (batch 8): do rivals' discounts affect a game?

Run on Nov 14, 2026: full ensemble, 3 cross-fitting repetitions, 40 placebo draws, 16 minutes.
Exposure is the share of a game's rivals that discounted in the sale (any cut in force during
the sale window). Controls are the usual pre-sale features plus the same exposure in the
previous sale, so the variation used is rivals departing from their own habits.
Coefficients compare all rivals discounting with none, on the log review scale. A realistic
swing is one within-sale SD, about 0.13 for the nearest-rival definitions, so multiply
coefficients by about 0.13.

Sample: 33,124 game-events with every exposure and its lag (4,901 holdouts, 28,223 discounted).

## Main estimates (top 20 rivals, or the whole cluster)
| Definition | Holdouts | Discounted games |
|---|---|---|
| audience | +0.175 (0.091) | -0.026 (0.050) |
| tags | +0.030 (0.097) | -0.025 (0.044) |
| tag k-means cluster | +0.557 (0.890) | +0.547 (0.543) |
| tag topic | +1.291 (1.051) | +0.505 (0.561) |

The cluster definitions barely vary within a sale (SD 0.04-0.05), so they have no power.

## Pre-specified criteria (holdouts)
1. **Placebo** (40 random 20-game rival sets): audience p = 0.075 (3 of 40 placebos as large); tags p = 0.78. Audience borderline, tags fail.
2. **Gradient** (ranks 1-5, 6-20, 21-50 in one model): audience +0.036 (0.047), +0.180 (0.085), -0.081 (0.119). Does not fade with distance: the nearest five matter least. Fails.
3. **Head-to-head** (all four in one model): audience +0.141 (0.098), tags -0.040 (0.102). Audience points the right way but is not significant.
4. **Out-of-sample**: audience +0.315 (0.134) in 2023-24 and +0.113 (0.128) in 2025-26. The sign holds, but the later period does not confirm on its own.

## Reading
- No robust spillover. The one candidate (audience rivals, holdouts) is positive, not negative, and fails the gradient test.
- Policy-relevant bound: holdouts' 95% interval for audience rivals is about [0.00, +0.35]. That rules out holdouts losing reviews when their rivals discount, so there is no evidence for "defensive" discounting.
- Discounted games: rivals discounting too changes their gain by -0.03 (SE 0.05). At a realistic 0.13 swing in exposure, that is under about 1.5% either way.
- Audience rivals beat tag rivals on every criterion they can be compared on, which modestly supports the segmentation. None of the definitions passes all four criteria.
