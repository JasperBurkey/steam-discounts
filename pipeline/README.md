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
