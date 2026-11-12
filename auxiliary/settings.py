"""Analysis settings used by the notebook. They match the pipeline that built the data."""

from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
EVENTS = DATA / "events.parquet"  # one row per game x sale event
EVENTS_DAILY = DATA / "events_daily.parquet"  # daily reviews around each event

WINDOW = (-21, 56)  # event-study days relative to sale start
BASELINE = (-21, -8)  # pre-event days used as each game's baseline
FOLDS = 5  # cross-fitting folds, split by game
CLIP = 0.02  # propensity scores kept within [CLIP, 1 - CLIP]
OVERLAP_TRIM = (0.1, 0.9)  # propensity range for the overlap-trimmed estimate
BOOTSTRAP_REPS = 300
SEED = 0
