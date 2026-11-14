"""Settings shared by every script. Edit here, not in the scripts."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv(path):
    """Read KEY=value lines from .env into the environment (no extra package needed)."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
RAW = DATA / "raw"  # untouched API responses (never edited)
REVIEWS_DIR = RAW / "reviews"  # one Parquet file per app
TAGS_DIR = RAW / "steamspy_tags"  # one JSON file per app
OUTPUT = ROOT / "output"  # figures and tables

SAMPLE_CSV = DATA / "sample.csv"
SNAPSHOT_CSV = DATA / "daily_snapshots.csv"
PANEL_PARQUET = DATA / "review_panel.parquet"
SALES_CSV = ROOT / "sale_calendar.csv"

# --- Sample definition (01_build_sample.py) ---
STEAMSPY_PAGES = 5  # ~1,000 apps per page, sorted by popularity
MIN_REVIEWS = 200  # thesis sample rule
PAID_ONLY = True

# --- Review history pull (03_pull_reviews.py) ---
# Mid-sized games: enough reviews for a daily signal, few enough to pull in full.
REVIEW_MIN, REVIEW_MAX = 1_000, 20_000
MAX_REVIEW_APPS = 500  # study sample size: a random draw of mid-sized games
SAMPLE_SEED = 0  # fixes the random draw, so every script uses the same games
REVIEW_HISTORY_START = "2023-01-01"  # stop paging once reviews are older than this

# --- Past prices from IsThereAnyDeal (03b_pull_prices.py) ---
ITAD_API_KEY = os.environ.get("ITAD_API_KEY", "")
ITAD_DIR = RAW / "itad"  # one JSON file per game, exactly as received
ITAD_IDS_CSV = DATA / "itad_ids.csv"  # Steam app ID -> IsThereAnyDeal game ID
PRICE_HISTORY_PARQUET = DATA / "price_history.parquet"
SLEEP_ITAD = 0.4  # IsThereAnyDeal allows 1,000 requests per 5 minutes
# Without `since`, IsThereAnyDeal returns only about the last 3 months. Start well before the
# first sale event so the price in force at the start of the window is known too.
PRICE_HISTORY_SINCE = "2022-06-01T00:00:00Z"

# --- Full project: stratified sample (01b) ---
STUDY_SAMPLE_CSV = (
    DATA / "study_sample.csv"
)  # when present, every script uses this sample
SAMPLE_SEED_FULL = 1
MIN_REVIEWS_FULL = 50  # eligible games: paid, released, >= 50 reviews
# band: (minimum total reviews, maximum total reviews, games to sample)
STRATA = {
    "small": (50, 1_000, 1_500),
    "mid": (1_000, 20_000, 1_500),
    "large": (20_000, 100_000, 800),
    "very_large": (100_000, 10**12, 200),
}

# --- Full project: review pull v2 (03) ---
REVIEW_PLAN_CSV = DATA / "review_plan.csv"
COUNTS_DIR = RAW / "review_counts"  # windowed counts for games too big to page
PAGE_MAX_REVIEWS = (
    15_000  # games with more reviews since REVIEW_HISTORY_START are counted
)
# by window instead of paged (cheaper above ~13,500)
REVIEW_WORKERS = 2  # requests in flight at once; set 1 if Steam returns many 429s
REVIEW_SLEEP = 1.0  # pause per worker between requests (seconds)
SECONDS_PER_REQUEST = 1.1  # measured on the MVP pull; used for time estimates

# --- Small-scale study (04b, 05b, 07) ---
# Release dates come from the collector's daily store snapshot, if available.
COLLECTOR_DATA = Path(os.environ.get("COLLECTOR_DATA", r"C:\steam_collector\data"))
EVENTS_PARQUET = DATA / "events.parquet"  # one row per game x sale event
EVENTS_DAILY_PARQUET = DATA / "events_daily.parquet"  # daily reviews around each event
LOOKBACK_DAYS = 60  # pre-event history needed for controls
PRE_DISCOUNT_DAYS = (
    7  # main spec: a discount this close before an event excludes the game
)
PRE_DISCOUNT_DAYS_STRICT = (
    30  # robustness: the stricter exclusion, reported alongside the main result
)
POST_DAYS = 14  # outcome window after the sale ends
LONG_POST_DAYS = 42  # longer follow-up (6 weeks after the sale ends) for review lag
# and pulled-forward demand; events without it are left out
# of those outcomes only
CUM_PRE_DAYS = 7  # the net-effect window starts this many days before the sale
CUM_POST_DAYS = 28  # ...and ends this many days after it ends
OVERLAP_TRIM = (
    0.1,
    0.9,
)  # propensity range for the overlap-trimmed estimate (Crump et al.)

# --- Estimation (dml_core.py, 07_dml.py) ---
LEARNER = os.environ.get(
    "LEARNER", "ensemble"
)  # "ensemble": LASSO + random forest + boosting (Athey-Imbens 2.7)
# "boost": boosting only (much faster; for quick checks)
CF_REPEATS = int(
    os.environ.get("CF_REPEATS", 3)
)  # cross-fitting repetitions for the main estimates (median reported)
FOLDS = 5  # cross-fitting folds, split by game
CLIP = 0.02  # propensity scores kept within [CLIP, 1 - CLIP]
SEED = 0
# Main outcomes: reviews and the revenue proxy, during the sale and net of the weeks around it.
# Headline: ATT on the net revenue proxy (decided in advance; see the analysis plan).
MAIN_OUTCOMES = {
    "y_during": "Reviews, during the sale",
    "r_during": "Revenue proxy, during the sale",
    "y_net": "Reviews, net (7 days before to 4 weeks after)",
    "r_net": "Revenue proxy, net (7 days before to 4 weeks after)",
}
HEADLINE_OUTCOME = "r_net"

# --- Heterogeneity (08) and discount depth (09) ---
HET_SPLITS = int(
    os.environ.get("HET_SPLITS", 25)
)  # game-level sample splits for BLP/GATES/CLAN
HET_TREES = int(os.environ.get("HET_TREES", 200))  # trees in each honest forest
BUSY_MIN = 1.0  # main heterogeneity sample: >= 1 review/day before the sale (little compression)
DEPTH_ARMS = ["none", "1-33%", "34-60%", "61%+"]
MIN_AGE_DAYS = 60  # skip events in a game's first two months (launch dynamics)

# --- Event study (05_event_study.py) ---
WINDOW = (-21, 28)  # days relative to sale start
WINDOW_LONG = (-21, 56)  # event-study window in 05b (needs LONG_POST_DAYS of follow-up)
BASELINE = (-21, -8)  # pre-period used to normalize each game
MIN_BASELINE_RATE = 0.5  # skip game-sales averaging under 0.5 reviews/day before
PLACEBO_SHIFT_DAYS = 56  # fake sale 8 weeks earlier, as a sanity check

# --- Clustering (06_cluster_tags.py) ---
K_GRID = [8, 12, 16, 20]
K_FINAL = 12
MIN_COREVIEW = 5  # pairs with fewer shared reviewers are ignored

# --- HTTP behaviour ---
SLEEP_STORE = 1.5  # store.steampowered.com (strict rate limits)
SLEEP_API = 0.3  # api.steampowered.com
SLEEP_STEAMSPY = 1.1  # SteamSpy: 1 request/second
SLEEP_STEAMSPY_ALL = 61  # SteamSpy "all" pages: 1 request/minute
USER_AGENT = "UW-econ-thesis (academic research)"

# Reviewer IDs are hashed with this before storage. Keep it private and constant.
SALT = os.environ.get("STEAM_SALT", "set-STEAM_SALT-env-var")

TZ = "America/Los_Angeles"  # Steam sales start 10:00 AM Pacific

for d in (RAW, REVIEWS_DIR, TAGS_DIR, OUTPUT, ITAD_DIR, RAW / "review_counts"):
    d.mkdir(parents=True, exist_ok=True)
