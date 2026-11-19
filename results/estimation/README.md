# Estimating the buy-or-wait model (31_estimate_buy_or_wait.py)

Simulated method of moments with logit taste shocks (Sargent and Stachurski, Dynamic
Programming Vol. I, sec. 5.3.1-5.3.2): the same model and 8 moments as 15_buy_or_wait
(doubly robust effects for busier games), now with a smooth criterion, standard errors and
an overidentification test. Taste-shock scale lam = 0.05 of the regular price (fixed);
patience profiled over 0.97-0.998.

| Parameter | Estimate (SE) |
|---|---|
| pi: share who know the sale price a week ahead | 0.41 (0.15) |
| phi: review lag carry-over (mean lag 1.9 weeks) | 0.65 (0.14) |
| delta: weekly chance of losing interest | 0.15 (0.36) |
| mu, sigma: valuation distribution | -0.32 (3.34), 3.49 (2.91): not identified |
| beta: patience | profile flat, J 24.46 -> 24.14 from 0.97 to 0.998: not identified |

**Fit and test.** J = 24.1 with 3 df (p < 0.001): the model is rejected. It overshoots the
sale-week jump (0.37 vs 0.31) and misses the depth gradient (61%+: 0.54 vs 0.61) and weeks
5-6 (0.01 vs 0.05). The calibrated version had the same misfit (24.8) without a test.
Untargeted: revenue proxy net -0.051 (data -0.072), during -0.60 (data -0.535).
lam = 0.02 / 0.10 give J 24.3 / 26.1 and similar pi and phi (lam_sensitivity.csv).

**Counterfactuals** (counterfactuals.csv). Committed schedules, steady state per year,
change vs never discounting in units of never-discount revenue. Ranges: min-max over the
90% confidence set (253 of 1,000 draws from the normal approximation whose criterion is
within the chi-squared(5) 90% critical value of the minimum), because the valuation
parameters are weakly identified.

| No extra sale traffic | Revenue | Publisher (70%) | Valve (30%) | Consumer surplus | Total welfare |
|---|---|---|---|---|---|
| 61%+ every sale | -0.12 [-0.19, -0.05] | -0.085 [-0.13, -0.04] | -0.036 [-0.06, -0.02] | +0.14 [+0.10, +0.19] | +0.02 [-0.01, +0.05] |
| 34-60% every sale | -0.06 [-0.10, -0.03] | -0.041 | -0.018 | +0.07 [+0.05, +0.10] | +0.01 [-0.01, +0.03] |
| 61%+ once a year | -0.03 [-0.08, -0.01] | -0.022 | -0.010 | +0.04 [+0.02, +0.08] | +0.005 [-0.01, +0.01] |

With double sale-week traffic the losses are larger (61%+ every sale: revenue -0.16).

Reading: in a pure pricing model with buyers who wait, sales transfer surplus from
publishers and Valve to buyers, with a small, insignificant gain in total. Every schedule
lowers revenue across the whole confidence set. Limits: one buyer type, rejected by the J
test; no visibility or follow-on spending; valuations not identified from these moments,
so levels of surplus are model-dependent (changes are reported instead).
