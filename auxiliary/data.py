"""Loading the event data and describing the sample."""

import pandas as pd

from auxiliary import settings


def load_events():
    """All game x sale-event rows built by the pipeline (pipeline/04b_build_events.py)."""
    if not settings.EVENTS.exists():
        raise FileNotFoundError(
            f"{settings.EVENTS} is missing. Copy events.parquet and events_daily.parquet "
            "from the pipeline's data folder into data/ (see README)."
        )
    return pd.read_parquet(settings.EVENTS)


def load_daily():
    """Daily reviews from day -21 to +56 around each event (paged games only)."""
    return pd.read_parquet(settings.EVENTS_DAILY)


def analysis_sample(events):
    """Rows used for the main estimates, with the outcomes they need present."""
    keep = events["analysis_sample"]
    cols = ["y_during", "y_after", "y_pre_week", "pre_trend"]
    return events[keep].dropna(subset=cols).reset_index(drop=True)


def exclusion_table(events):
    """How many game-events each exclusion rule removes (a row can fail several)."""
    rules = {
        "Too little review history around the event": ~events["coverage_ok"],
        "Price unknown": ~events["price_known"],
        "Released under 60 days before the event": events["too_new"],
        "Free or no regular price": events["free"],
        "Discounted in the 7 days before the event": events["pre_discount"],
    }
    out = pd.DataFrame(
        {
            "rule": list(rules),
            "game_events_removed": [int(m.sum()) for m in rules.values()],
        }
    )
    total = pd.DataFrame(
        {
            "rule": ["Analysis sample (kept)"],
            "game_events_removed": [int(events["analysis_sample"].sum())],
        }
    )
    return pd.concat([out, total], ignore_index=True)


def event_summary(events):
    """Per sale event: games in the analysis sample, share discounted, median discount."""
    a = analysis_sample(events)
    s = a.groupby("sale", sort=False).agg(
        games=("app_id", "size"),
        share_discounted=("discounted", "mean"),
        median_cut_pct=("cut", lambda c: c[c > 0].median()),
    )
    s["share_discounted"] = (100 * s["share_discounted"]).round(0)
    return s.rename(columns={"share_discounted": "discounted_pct"})
