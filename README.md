[![Run Notebook](https://github.com/YOUR-GITHUB-USERNAME/YOUR-REPO-NAME/actions/workflows/run-notebook.yml/badge.svg)](https://github.com/YOUR-GITHUB-USERNAME/YOUR-REPO-NAME/actions/workflows/run-notebook.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Should Games Go on Sale? Measuring the Impact of Steam Discounts

Project by Jasper Burkey for *Impact-Driven Business Decisions* (University of Washington, Autumn 2026), built on the [course repository template](https://github.com/eisenhauerIO/projects-student-template).

Four times a year, Valve runs a Steam-wide sale and every game publisher decides whether to discount and how deep. The notebook [project.ipynb](project.ipynb) measures what that decision buys: how much a discount raises demand during the sale, whether the gain lasts or is borrowed from later weeks, and for which games it pays off. It compares discounted with non-discounted games within the same seasonal sale, adjusts for each game's pre-sale history with double machine learning, and reports anticipation, net effects, robustness checks and heterogeneity tests.

### What the project finds

With the thesis settings: discounting raises review activity by about 28% during the sale for games that discount, but on a review-based revenue proxy costs about 6% of net revenue (95% CI −8% to −3%). Effects are similar across game sizes; games that discount rarely gain the most; and for the games that use them, deep discounts bring no more activity than shallow ones but cost 11–30% of net revenue, while shallow discounts roughly break even. See [project.ipynb](project.ipynb) for the analyses and their limitations.

### Repository structure

| Folder | What it holds | Runs on GitHub? |
| --- | --- | --- |
| `project.ipynb` | The analysis and its reasoning, step by step | Yes, on every push (quick settings) |
| `auxiliary/` | Notebook helpers; `estimates.py` calls the pipeline's own estimators | Yes |
| `data/` | `events.parquet` and `events_daily.parquet`: the derived event data | — |
| `pipeline/` | Builds the data (sample, reviews, prices, events) and runs the full analyses (`07_dml.py`, `08_heterogeneity.py`, `09_discount_depth.py`); estimation code in `dml_core.py` and `hetero.py` | No: needs API keys and days of downloading |
| `collector/` | The daily Steam store collector (prices, player counts, top-seller rankings, front page) | No: runs on a schedule |
| `tests/` | Smoke tests for the notebook helpers | Yes |

### Reproducing the results

```
conda env create -f environment.yml
conda activate steam-discounts
pytest tests/ -v
pytest --nbmake --nbmake-timeout=3600 project.ipynb
```

The notebook needs only `data/`. GitHub Actions runs it with quick settings (`LEARNER=boost`, `CF_REPEATS=1`, `HET_SPLITS=5`, `HET_TREES=50`), whose estimates are close to the thesis results. To reproduce the thesis results exactly, run it with the defaults (`LEARNER=ensemble`, `CF_REPEATS=3`, `HET_SPLITS=25`, `HET_TREES=200`), which takes roughly an hour. To rebuild `data/` from scratch, follow [pipeline/README.md](pipeline/README.md).

### Data and privacy

All inputs are public: Steam's store and review APIs and IsThereAnyDeal's price histories. Raw reviews are not included; reviewer IDs are one-way hashed in the pipeline. API keys live in `.env` files, excluded by `.gitignore`. Third-party data are not redistributed beyond the derived event-level variables in `data/`.

### Project history

- **MVP (tag [`mvp`](https://github.com/JasperBurkey/steam-discounts/tree/mvp)):** a feasibility version on 500 randomly sampled mid-sized games, with a first double machine learning analysis of reviews.
- **Full sample (current):** 3,850 games stratified by size, a revenue proxy, heterogeneity and discount-depth analyses.

### Data and privacy

All inputs are public: Steam's store and review APIs and IsThereAnyDeal's price histories. Raw reviews are not included; reviewer IDs are one-way hashed in the pipeline and never stored in plain form. API keys live in `.env` files, which are excluded by `.gitignore`. Third-party data are not redistributed beyond the derived event-level variables in `data/`.
