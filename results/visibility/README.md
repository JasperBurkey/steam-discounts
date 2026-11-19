# Visibility benefits the four-week window could miss (23_visibility.py)

Quick settings (LEARNER=boost, CF_REPEATS=1). ATT on log reviews vs the pre-sale baseline.

**A. Longer horizon: no lasting gain.**

| Sample | During | Weeks 3-4 | Weeks 5-6 | Weeks 7-12 |
|---|---|---|---|---|
| All sales (31,818) | +0.252 (0.018) | +0.013 (0.018) | +0.031 (0.018) | |
| Spring + Summer, full histories (15,183) | +0.279 (0.025) | | +0.043 (0.024) | -0.053 (0.025) |

The effect fades by week 3 and turns slightly negative in weeks 7-12, consistent with
purchases pulled forward rather than a lasting boost.

**B. The next sale: no effect.** Among game-events that discount in the next sale, having
discounted in this one changes the next sale's reviews by -0.013 (0.021): no visibility
carry-over, and no sign of a depleted pool either.

**C. Review labels (labels.csv): too few crossings to tell.** Crossing 50 purchaser
reviews at 80%+ (Positive -> Very Positive) vs 70-79% (no change), 60+ days after release:
+0.17 (0.14), 169 games. Crossing 500 at 95%+ (-> Overwhelmingly Positive) vs 80-94%:
+0.21 (0.13), 99 games. Both point the way a label effect would, neither is significant.
(Rows for test C in an earlier version of visibility.csv used crossings in the launch
burst and are superseded by labels.csv.)

Reading: in reviews, nothing suggests a hidden payoff from joining a sale beyond the
first two weeks. Benefits the review data cannot see (wishlists, DLC) remain possible.
