# Two checks from review flags (pipeline/16_review_flags.py)

Run Nov 18, 2026: full ensemble, 3 cross-fitting repetitions, 70 minutes. 3,204 games with
complete review histories (6.0M reviews); the 645 largest games are left out because the
export has only a recent sample of their reviews. Every series is compared with paid Steam
reviews randomly thinned to the same share, game by game, so sparsity on the log(1 + x)
scale can't masquerade as a smaller effect. ATT, log points (SE in effects.csv).

## 1. Placebo: reviews from people who did not pay Steam's price (17% of reviews)
| Window | Placebo | Thinned paid benchmark | Difference (SE) |
|---|---|---|---|
| During the sale | +0.024 | +0.142 | -0.118 (0.025) |
| Weeks 1-2 after | -0.028 | +0.068 | -0.096 (0.026) |
| Net window | -0.019 | +0.048 | -0.068 (0.021) |

Reviews from key activations elsewhere and free copies barely respond to a Steam discount,
about a sixth of the paid response at equal sparsity. **Passes**: the main effect runs
through Steam's price, not through a shock that moves all reviews. They dip slightly after
the sale, consistent with buyers who would have bought a key elsewhere buying on Steam
during the sale instead.

**But the pre-sale dip is not specific to paid buyers.** The week before the sale, placebo
reviews fall as much as paid ones (-0.043 vs -0.046). Buyers who don't pay Steam's price
have no reason to wait for a Steam sale, so the dip is weaker evidence of waiting than
assumed. Candidates: key stores discounting in step with Steam's sales, or games that
discount being on a slight downswing just before. The buy-or-wait model leaned on this dip
to identify waiting; that part of its calibration should be treated with caution.

## 2. A faster purchase signal: reviews written under 2 hours of play (13% of reviews)
| Window | Under 2 hours | Thinned paid benchmark | Difference (SE) |
|---|---|---|---|
| During the sale | +0.268 | +0.159 | +0.109 (0.030) |
| Weeks 1-2 after | +0.057 | +0.083 | -0.025 (0.029) |
| Net window | +0.066 | +0.054 | +0.013 (0.024) |

Reviews written soon after buying jump more during the sale and less afterwards, with the
same total. The response is concentrated in the sale itself: weeks 1-2 after are a fifth
of the sale-week effect, against half for paid reviews generally. **Confirms review lag**:
much of the elevated review count after a sale is sale buyers reviewing late, not extra
demand after the sale.

Busier games show the same pattern with wider intervals (tests.csv).
