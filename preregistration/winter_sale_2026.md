# Predictions for the Steam Winter Sale 2026, recorded before the sale

Recorded November 17, 2026. The sale runs December 17, 2026 to January 4, 2027 (10:00
Pacific). Everything below is fixed now, before any Winter Sale 2026 data exists. The git
commit that adds this file timestamps it. Any later change to this file is a deviation, and
the evaluation must report it.

The purpose is an honest out-of-sample test. Every estimate in the thesis so far uses sales
that had already happened when the methods were chosen. The Winter Sale 2026 is the first
sale that no model, rule or specification has seen.

## Data and timing
- **Sample:** the 3,850 study games (`collector/study_apps.csv`), with game-events built by
  the same pipeline (`pipeline/04b_build_events.py`) and the same exclusions.
- **Reviews and prices:** pulled after the follow-up window ends (six weeks after the sale,
  mid-February 2027), with the same scripts and settings.
- **Player counts:** the collector's study panel (`tier = study` in the `ccu` table),
  every 4 hours since Nov 16, 2026, and every 2 hours from Dec 14, 2026 to Jan 7, 2027.
- **Models:** all nuisance models, forests and policy trees are fit on the 15 sales from
  Spring 2023 to Autumn 2026 only, then applied unchanged to Winter Sale 2026 game-events.
  Nothing is refit on Winter Sale data, except where a test below says so.

## Predictions

### P1. Average effects
Doubly robust ATT, Winter Sale 2026 game-events only, same outcomes as the thesis. The
point predictions are the pooled estimates from the 15 past sales
(`results/main/main_estimates.csv`). A prediction **passes** if the Winter Sale 95% interval
contains the pooled estimate. Signs are also predicted: a sign failure counts as a failure
even when the interval is wide.

| Outcome | Predicted ATT (pooled, SE) | Predicted sign |
|---|---|---|
| Reviews during the sale | +0.250 (0.017) | positive |
| Reviews, net (7 days before to 4 weeks after) | +0.083 (0.014) | positive |
| Revenue proxy during the sale | -0.337 (0.018) | negative |
| Revenue proxy, net (headline) | -0.060 (0.014) | negative or zero |

A single sale has about 1/15 of the data, so its intervals will be roughly 3-4 times wider.
Winter Sales run 18-19 days rather than 7, so effects during the sale may be larger. The
direction of that difference is not predicted.

### P2. Who gains most
The honest forest of section 7 (busier games, revenue proxy, net) is trained on the 15 past
sales and predicts each Winter Sale game-event's effect from its pre-sale features. Game-events
are sorted into fifths of predicted effect.
- **Prediction:** the realized doubly robust effect in the top fifth exceeds the bottom fifth
  (one-sided test at 5%). The realized effects use scores whose nuisance models are cross-fit
  on the Winter Sale game-events alone.
- **Direction of characteristics:** the top fifth is younger, pricier and discounts less
  often and less deeply than the bottom fifth.

### P3. The policy rule
The rule learned in `pipeline/13_policy.py` (all game-events, net revenue proxy), frozen as:

    if reviews per day before the sale <= 0.4643:
        if days since last discount <= 31: discount 1-33%
        else:                               discount 61%+
    else:
        if days since last discount <= 16: discount 1-33%
        else:                               discount 61%+

The overlap constraint is the same: where the assigned depth has propensity below 0.05,
the game keeps its actual choice. The propensity model is fit on the 15 past sales.
- **Value:** the mean doubly robust score under the rule minus the mean under the status
  quo, on Winter Sale game-events. The scores' outcome models are fit on the past sales only.
- **Prediction:** the gain is positive under both the net revenue proxy and its lag-adjusted
  version. Learned-on-past estimates are +0.048 and +0.038. Because the rule was picked
  from 18 candidates, the realized gain is expected to be smaller: between 0 and +0.05.
- **Failure condition:** the gain is negative under either measure, beyond one standard
  error.

### P4. Discount depth
ATT by depth for the game-events that chose each depth (`results/main/depth.csv`):
- **Reviews during the sale:** positive at every depth, and similar across depths. The
  61%+ ATT exceeds the 1-33% ATT by less than 0.15 log points (pooled: +30% vs +31%).
- **Revenue proxy, net:** the 61%+ ATT is below the 1-33% ATT (pooled: -11% vs +3%), and
  the 61%+ ATT is negative.

### P5. Spillovers
Audience-rival exposure (share of the 20 nearest audience rivals discounting) has no
significant effect on holdouts' reviews during the sale (|t| < 1.96), and no significant
negative effect on discounted games.

### P6. Player counts (new outcome; descriptive, no prior estimate)
For study-panel games with at least 50 players online on average in the four weeks before
the sale:
- The ATT of discounting on log(1 + mean players online) during the sale is positive.
- Its ranking agrees with reviews: across fifths of predicted effect (P2), the rank
  correlation between player-count effects and review effects is positive.
- Player counts rise within the first day. The jump at 10:00 Pacific on Dec 17 is visible in
  the 2-hour samples for discounted games and absent for holdouts.

## Reporting
All predictions are reported whatever the outcome, in this order, with the same estimators
and settings as the thesis (ensemble learner, 3 cross-fitting repetitions, SEs clustered by
game). Results go in `results/winter_sale_2026/`.
