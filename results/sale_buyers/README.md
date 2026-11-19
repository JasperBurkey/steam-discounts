# Are sale buyers different? (25_sale_buyers.py)

Quick settings (LEARNER=boost, CF_REPEATS=1). Steam-purchaser reviews in the sale and two
weeks after vs the 60 days before; ATT of discounting (comparison: full-price games over
the same weeks). 15,456 game-events, 2,020 games with complete histories and 5+ reviews in
both periods.

| | Share positive | Median log playtime at review | Share with <2h played |
|---|---|---|---|
| All discounts | +0.003 (0.005) | -0.149 (0.021) | +0.026 (0.006) |
| Deep (50%+) | -0.005 (0.007) | -0.175 (0.029) | +0.036 (0.007) |
| Shallow (<50%) | +0.013 (0.013) | -0.132 (0.049) | +0.016 (0.012) |

**Ratings don't change.** The share positive moves less than a point either way, and the
interval rules out a drop of more than about 1.3 points. Marginal sale buyers like the game
as much as full-price buyers do.

**Engagement at review is lower.** Reviewers after a discount have played about 14% less
(exp(-0.149)), and 2.6 points more review within two hours; both stronger for deep cuts.
Sale buyers review sooner and after less play: consistent with impulse purchases, and with
the review lag found in 16_review_flags. Lower engagement could mean less DLC and word of
mouth later, which the visibility tests (23) also fail to find.

For the proxy: since ratings don't shift, the review proxy is not distorted by sale buyers
being unhappier; the open question is only how often they review, not how they review.
