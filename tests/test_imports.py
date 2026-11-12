"""Smoke tests: the helper modules import and the estimator recovers a known effect."""

import numpy as np
import pandas as pd


def test_imports():
    from auxiliary import data, dml, event_study, plots

    assert hasattr(data, "load_events")
    assert hasattr(dml, "plr")
    assert hasattr(event_study, "event_study")
    assert hasattr(plots, "plot_event_study")


def _fake_events(n_games=300, n_events=6, effect=0.4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        rate = np.exp(rng.normal(0, 0.8))
        for k in range(n_events):
            p = 1 / (1 + np.exp(-(0.8 * np.log(rate))))
            d = rng.random() < p
            rows.append(
                {
                    "app_id": g,
                    "sale": f"S{k}",
                    "discounted": d,
                    "pre_rate_28": rate,
                    "pre_rate_60": rate,
                    "pre_trend": rng.normal(0, 0.1),
                    "pct_pos_60": 0.8,
                    "age_days": 500 + g,
                    "regular_usd": 19.99,
                    "days_since_discount": 90,
                    "never_discounted_before": False,
                    "y_during": 0.2 * np.log(rate) + effect * d + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


def test_plr_recovers_known_effect():
    from auxiliary import dml

    est, se = dml.plr(_fake_events(), "y_during", "discounted")
    assert abs(est - 0.4) < 4 * se + 0.05
