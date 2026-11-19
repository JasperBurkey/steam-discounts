# Results produced outside the notebook

Outputs of pipeline scripts run on the full data, kept here so they can be read without
rerunning. Each folder names the script that produced it; rerun the script to reproduce.

| Folder | Script | What it holds |
| --- | --- | --- |
| `rank_validation/` | `pipeline/10_rank_validation.py` (full ensemble settings) | Discounting and Steam's weekly revenue charts, large and very large games; the revenue-proxy weighting test |
| `spillovers/` | `pipeline/12_spillovers.py` | Rival discounts and a game's reviews: four rival definitions, placebo, gradient, head-to-head, out-of-sample |
| `policy/` | `pipeline/13_policy.py` | Policy learning: overlap-constrained policy trees, cross-fitted values under three revenue measures |
| `buy_or_wait/` | `pipeline/15_buy_or_wait.py` | Calibrated buy-or-wait model (QuantEcon): discount Markov chains, the buyer's problem with DiscreteDP, fit to the busier-game profile, proxy vs true revenue, discount schedules over a year |
| `main/` | `pipeline/14_notebook_tables.py` | The notebook's tables at full thesis settings |
| `segments/` | `pipeline/11_segments.py` | Audience factors (K by held-out prediction), segments, tag mapping, robustness, rival lists, tag baselines. Includes the big-game sample (03f): all large and very large games are audience-placed |
| `review_flags/` | `pipeline/16_review_flags.py` | Placebo outcome (reviews from non-Steam buyers) and fast reviews, against thinned benchmarks |
| `update_controls/` | `pipeline/17_update_controls.py` | Main effects with developer-update controls from the news data (provisional quick settings) |
| `placebo_dates/` | `pipeline/18_placebo_dates.py` | Sales moved six weeks earlier, controls recomputed: no effect at fake dates |
| `sensitivity/` | `pipeline/19_sensitivity.py` | Omitted-variable bounds (Chernozhukov et al. 2022): robustness values and benchmarks |
| `rejoin/` | `pipeline/20_rejoin.py` | Does rejoining the next sale, or its depth, follow the last sale's spike or net? |
| `inertia/` | `pipeline/21_inertia.py` | Stickiness of participation and depth by publisher size, experience, self-publishing |
| `release_rd/` | `pipeline/22_release_rd.py` | The 30-day release rule as a fuzzy RD: first stage, outcomes, density, balance, placebo cutoffs; `universe.csv` feeds `03h_pull_rd_reviews.py` |
| `visibility/` | `pipeline/23_visibility.py` | Longer-horizon effects, next-sale effects, review-label thresholds |
| `milestones/` | `pipeline/24_milestones.py` | Copies sold per review from developers' milestone posts |
| `sale_buyers/` | `pipeline/25_sale_buyers.py` | Ratings, playtime and fast reviews of buyers after a discount |

Game positions (`game_positions.parquet`, ~57 MB) are not committed; the script regenerates them.
