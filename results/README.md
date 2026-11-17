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

Game positions (`game_positions.parquet`, ~57 MB) are not committed; the script regenerates them.
