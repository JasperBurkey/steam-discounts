[![Run Notebook](https://github.com/YOUR-GITHUB-USERNAME/YOUR-REPO-NAME/actions/workflows/run-notebook.yml/badge.svg)](https://github.com/YOUR-GITHUB-USERNAME/YOUR-REPO-NAME/actions/workflows/run-notebook.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Should Games Go on Sale? Measuring the Impact of Steam Discounts

Project by Jasper Burkey for *Impact-Driven Business Decisions* (University of Washington, Autumn 2026), built on the [course repository template](https://github.com/eisenhauerIO/projects-student-template).

Four times a year, Valve runs a Steam-wide sale and every game publisher decides whether to discount and how deep. The notebook [project.ipynb](project.ipynb) measures what that decision buys: how much a discount raises demand during the sale, whether the gain lasts or is borrowed from later weeks, and for which games it pays off. It compares discounted with non-discounted games within the same seasonal sale, adjusts for each game's pre-sale history with double machine learning, and reports anticipation, net effects, robustness checks and heterogeneity tests.

### Repository structure

| Folder | What it holds | Runs on GitHub? |
| --- | --- | --- |
| `project.ipynb` | The analysis and its reasoning, step by step | Yes, on every push |
| `auxiliary/` | Helper code the notebook uses (data, event study, double ML, plots) | Yes |
| `data/` | `events.parquet` and `events_daily.parquet`: the derived event data the notebook analyzes | — |
| `pipeline/` | Scripts that build the event data from Steam and IsThereAnyDeal (sample, reviews, prices, events) | No: needs API keys and days of downloading |
| `collector/` | The daily Steam store collector (prices, player counts, rankings, front page) | No: runs on a schedule |
| `tests/` | Smoke tests for the helper code | Yes |

### Reproducing the results

```
conda env create -f environment.yml
conda activate steam-discounts
pytest tests/ -v
pytest --nbmake --nbmake-timeout=3600 project.ipynb
```

The notebook needs only `data/`. To rebuild `data/` from scratch, follow [pipeline/README.md](pipeline/README.md): it draws the sample, downloads reviews and prices, and runs `04b_build_events.py`, whose outputs are copied into `data/`.

### Data and privacy

All inputs are public: Steam's store and review APIs and IsThereAnyDeal's price histories. Raw reviews are not included; reviewer IDs are one-way hashed in the pipeline and never stored in plain form. API keys live in `.env` files, which are excluded by `.gitignore`. Third-party data are not redistributed beyond the derived event-level variables in `data/`.
