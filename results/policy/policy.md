# Policy learning (batch 9): which games should discount, and how deep?

Run on Nov 16, 2026: full ensemble, 3 cross-fitting repetitions, 20 minutes (`pipeline/13_policy.py`).
Policies are trees (1 split or 2 levels) over eight features known before the sale. They
are learned on 4 of 5 game folds and valued on the held-out games, using doubly robust
scores for each depth (none, 1-33%, 34-60%, 61%+). Gains are versus the status quo (what
publishers actually chose), in log points, so +0.05 is roughly +5%.

**Overlap constraint.** A policy may assign only a depth that similar games sometimes choose
(propensity >= 0.05); otherwise the game keeps its actual choice. That binds for 45-70% of
game-events, because most games almost always discount at about their habitual depth.
Without it, the trees recommend choices almost no comparable game made.

**Criterion (fixed in advance).** A rule is robust only if it beats the status quo under both
the net revenue proxy and its lag-adjusted version. The chart-calibrated objective
(reviews + 0.135 x log price, during the sale) is a sensitivity check.

## Result
One rule passes, in the full sample (all game-events): the tree learned on the net revenue proxy.

| Learned rule | Proxy, net | Lag-adjusted | Chart-calibrated | Reviews |
|---|---|---|---|---|
| 2 levels, learned on proxy | +0.048 (0.015) | +0.038 (0.015) | +0.053 (0.020) | +0.06 |
| 1 split, learned on proxy | +0.048 (0.016) | +0.033 (0.016) | +0.050 (0.020) | +0.06 |

Fitted on all data, the rule is:

    if the game discounted recently (within ~16-31 days; the cutoff is longer for quieter games):
        discount shallowly (1-33%)
    else:
        discount deeply (61%+)

It never recommends skipping the sale. It changes 14% of choices: mostly 34-60% to 61%+
(6%), no discount to 61%+ (4%), and 34-60% to 1-33% (3%).

## What fails
- **Busier games:** no rule passes. The proxy-learned 2-level tree gains +0.009 (0.013) and
  +0.000 (0.014). The tree learned on the lag-adjusted measure passes both revenue measures
  (+0.022, +0.040) but loses under the chart-calibrated one.
- **Fixed rules:** "never discount" and each single depth swing with the revenue measure.
  For example, never discounting is +0.032 on lag-adjusted but -0.064 on chart-calibrated.
  The advice for any one-size-fits-all rule depends on how revenue is measured.

## Caveats
- **Many rules were compared.** 18 learned rules across two samples and three objectives, and
  the passing one was identified after seeing results. Each value is honestly held-out, but
  picking the best of several overstates it somewhat. The real test is to fix this rule
  before the Winter Sale 2026 and evaluate it on the collector's data.
- **Revenue is a proxy** (reviews x price). The rank validation suggests it overstates what
  discounting costs for big games.
- **One game deviating at a time.** Effects hold rivals' choices fixed. Batch 8 found no
  spillovers, but an industry-wide change could still shift Steam's traffic.
