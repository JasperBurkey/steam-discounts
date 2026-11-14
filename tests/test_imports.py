"""Smoke tests: the helpers import, and the pipeline's estimator recovers a known effect."""

import os

import numpy as np
import pandas as pd

os.environ.setdefault("LEARNER", "boost")
os.environ.setdefault("CF_REPEATS", "1")


def test_imports():
    from auxiliary import data, estimates, event_study, plots

    assert hasattr(data, "load_events")
    assert hasattr(estimates, "main_estimates")
    assert hasattr(event_study, "event_study")
    assert hasattr(plots, "plot_gates")


def _fake_events(n_games=300, n_events=6, effect=0.4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        rate = np.exp(rng.normal(0, 0.8))
        for k in range(n_events):
            d = rng.random() < 1 / (1 + np.exp(-(0.8 * np.log(rate))))
            rows.append(
                {
                    "app_id": g,
                    "sale": f"S{k}",
                    "discounted": d,
                    "pre_rate_28": rate,
                    "pre_rate_60": rate,
                    "age_days": 500 + g,
                    "regular_usd": 19.99,
                    "y_during": 0.2 * np.log(rate) + effect * d + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


def test_plr_recovers_known_effect():
    import dml_core as M

    from auxiliary import estimates  # noqa: F401  (puts pipeline/ on the path)

    est, se = M.plr(_fake_events(), "y_during", "discounted", repeats=1)
    assert abs(est - 0.4) < 4 * se + 0.05
