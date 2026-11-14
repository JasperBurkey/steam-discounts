# Steam discounts: full project (scaled-up MVP)

The MVP's code and analysis, scaled from 500 mid-sized games to a stratified sample of
about 4,000 games across all sizes, using Steam sale events from 2023 to 2026. Lives in its
own folder so the MVP (C:\steam_mvp) stays unchanged and reproducible.

## Setup

```
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy C:\steam_mvp\.env .env          # brings your IsThereAnyDeal key
```

Keep the same STEAM_SALT as the MVP (it is a user-wide setting, so it already is). Reviewer
IDs are hashed with it, and the same salt lets games reused from the MVP be linked with new ones.

## Batch 4: sample and data

| Step | Command | Time |
| --- | --- | --- |
| 1b | `python 01b_build_strata_sample.py` | seconds |
| verify | `python 03_pull_reviews.py --verify-dir C:\steam_mvp\data\raw\reviews` | ~1 min |
| plan | `python 03_pull_reviews.py --plan` | ~30-60 min |
| pull | `python 03_pull_reviews.py --reuse-from C:\steam_mvp\data\raw\reviews` | ~1.5-3 days |
| prices | `python 03b_pull_prices.py --scope all` | several hours |
| top sellers | `python 03d_test_topsellers.py` | seconds |

**Sample (1b).** Every released, paid game with at least 50 reviews in the collector's
latest store snapshot, split into bands by total reviews (`STRATA` in config.py) and sampled
at random within each band. Big games are over-sampled, so each game has a `weight`: use it
for estimates about the typical Steam game; leave it out for estimates by band.

**Reviews (03).** Games with up to `PAGE_MAX_REVIEWS` reviews since 2023 are paged in full
(as in the MVP). Bigger games are counted instead: one request per window around each sale
event, using the API's date filter and `total_matching`. Run `--verify-dir` first; it checks
that Steam's window counts match reviews you already downloaded, and the count method should
only be used if it says PASS. `REVIEW_WORKERS` (default 2) sets requests in flight; if the
pull reports many rate-limited requests, set it to 1. Stop and rerun any time.

**Prices (03b).** `--scope all` downloads Steam price histories for every eligible game
(needed later for spillovers between rivals), including the sample. Runs alongside the review
pull (different service).

**Top sellers (03d).** Tests whether past weekly top-seller charts are available. Uses the
collector's Steam key.

The analysis scripts (04 onward) are updated for counted games in batch 5.

## Tests

`python -m unittest tests.test_full_pull -v` runs batch 4 against a fake Steam API in a
temporary folder (never touches data/).

## Batch 5b: estimation following Athey and Imbens (2019)

After the downloads finish, rebuild the events and run the analysis:

```
.venv\Scripts\python 04_build_panel.py
.venv\Scripts\python 04b_build_events.py
.venv\Scripts\python 05b_did_event_study.py
.venv\Scripts\python 07_dml.py
```

`07_dml.py` with the default settings (ensemble learners, 3 cross-fitting repetitions) can take
one to a few hours on the full sample. For a quick check first, run it with the fast settings:
`$env:LEARNER="boost"; $env:CF_REPEATS="1"; .venv\Scripts\python 07_dml.py`, then
`Remove-Item Env:LEARNER, Env:CF_REPEATS` before the full run.

**Decided in advance (analysis plan):** the headline is the **ATT on the net revenue proxy**.

| Piece | What it does | Paper |
| --- | --- | --- |
| Revenue proxy (`r_during`, `r_net`, `r_after`) | Each window's reviews valued at the average price actually paid in that window (from the price histories), assuming reviews move with units sold | — |
| Richer pre-event controls | Discount history over the past year, behavior and review response in earlier sales, the publisher's discounting in the previous sale | 6.1 |
| Ensemble learners | LASSO + random forest + boosting, weights chosen on out-of-fold predictions (`output/nuisance_fit.csv` shows weights and out-of-sample fit) | 2.7 |
| Repeated cross-fitting | Folds split by game, repeated with different splits; median estimate, SE includes the split-to-split spread | 6.2 |
| ATT, ATE, ATU and PLR | Doubly robust estimates for each main outcome | 6.1 |
| Residual balancing (L2 variant) | ATT with balancing weights on comparison games; reports remaining imbalance and effective comparison sample | 6.1 |
| Depth bins (`depth_bin`) | None, 1-33%, 34-60%, 61%+, for the dose-response batch | 6.3 |

Revenue-proxy caveats: discount buyers may review less (understates units), reviews lag
purchases (post-sale weeks valued at full price), and DLC or later sales are not included.
Interpret size-band differences in the revenue proxy with care for very low-volume games.

## Batch 5c: ratio estimates (after the first full-sample run)

The first full-sample run showed effects shrinking toward zero for low-volume games: the
log(1 + x) outcomes compress changes when a game has only a few reviews in a window, and
61% of game-days have none. `07_dml.py` now also reports, for discounted games:

| Rows | What | Why |
| --- | --- | --- |
| `RATIO: ...` | Doubly robust ratio of mean counts (actual vs. estimated without the discount), no "+1"; Poisson count model | Not compressed for small counts; a ratio of means, so busier games weigh more |
| `RATIO, size band ...` | The same within each size band | The fair comparison across sizes |
| `Busier games only ...` | The log version for games with >= 1 review a day before the sale | Where the "+1" barely matters |
| `RATIO, overlap-trimmed ...` | Ratio estimate where propensity is 0.1-0.9 | Overlap sensitivity (with 85-90% discounting, estimates move with the propensity cap) |
| `Without Winter 2023 and 2024 ...` | Main estimates without the two Winter Sales that followed Autumn Sales within ~3 weeks | Their baseline overlaps the end of the Autumn Sale |

Log-scale heterogeneity by size band and pre-event review rate is partly driven by this
compression: compare the RATIO rows by size band before interpreting it.
`tests/test_ratio_lowcount.py` checks the ratio estimator on low-count data with a known effect.

## Weekly top-seller history (03e)

`python 03e_pull_weekly_topsellers.py` downloads Steam's weekly top-seller charts (revenue
ranks) for every completed week since January 2023: global plus US, DE, GB, BR, IN and JP, up
to 500 places each, into `data/weekly_topsellers.parquet`. About 7,000 Web API requests
(roughly an hour); resumable. Chart weeks start on Tuesdays (UTC), so seasonal sales
(Thursday starts) begin mid-week: align sales to Steam's own `week_start`. Ranks cover the
top-selling games only, so this is mainly a revenue-ranked check for large games.

## Batch 7 and 7b: heterogeneity and discount depth

| Script | Question | Method | Outputs |
| --- | --- | --- | --- |
| `08_heterogeneity.py` | Which games gain or lose more from discounting? | Honest forest on doubly robust scores (`hetero.py`), summarized with generic ML inference: BLP (is there heterogeneity?), GATES (effects by predicted-effect quintile), CLAN (who is most/least affected) | `het_blp.csv`, `het_gates.csv`, `het_clan.csv`, `het_importance.csv`, `het_gates.png` |
| `09_discount_depth.py` | How deep should a discount be? | Doubly robust multi-arm effects for none / 1-33% / 34-60% / 61%+; versus not discounting, and among discounters versus shallow | `depth_effects.csv`, `depth_effects.png` |

Main heterogeneity sample: busier games (>= 1 review/day before the sale), where the log scale
is not compressed; all games as a secondary check. Heterogeneity p-values are adjusted for
the repeated splits (2 x median p). Quick check first:
`$env:LEARNER="boost"; $env:CF_REPEATS="1"; $env:HET_SPLITS="5"; $env:HET_TREES="50"`, then
`Remove-Item Env:LEARNER, Env:CF_REPEATS, Env:HET_SPLITS, Env:HET_TREES` before the full runs.
`tests/test_hetero_depth.py` checks both on synthetic data with known answers.

### Checks added after the quick runs

- `r_net_lag`: net revenue proxy with the 14 days after the sale valued at the sale price.
  Sale buyers often review after the sale; the standard proxy values those reviews at full
  price. The two versions bracket the effect of review lag (largest for deep discounts).
- `09_discount_depth.py` reports each depth's effect for all game-events (ATE) and for the
  game-events that chose that depth (ATT): a deep discount can help the typical game while
  doing little for the habitual deep discounters.
- `08_heterogeneity.py` shows each group's average discount propensity in the CLAN table, and
  repeats the busier-games analysis within the overlap region (propensity 0.1-0.9).
