"""All settings. Secrets come from environment variables or a .env file next to this one."""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_dotenv(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(HERE / ".env")

# ---------- Secrets ----------
STEAM_API_KEY = os.environ.get("STEAM_API_KEY", "")
TWITCH_CLIENT_ID = os.environ.get("TWITCH_CLIENT_ID", "")  # optional
TWITCH_CLIENT_SECRET = os.environ.get("TWITCH_CLIENT_SECRET", "")
ALERT_WEBHOOK_URL = os.environ.get(
    "ALERT_WEBHOOK_URL", ""
)  # optional, e.g. https://ntfy.sh/<topic>
HEARTBEAT_URL = os.environ.get(
    "HEARTBEAT_URL", ""
)  # optional dead man's switch, e.g. healthchecks.io

# ---------- Where data goes ----------
DATA = Path(os.environ.get("STEAM_DATA_DIR", HERE / "data"))
RAW, TABLES, STATE, LOGS, ASSETS = (
    DATA / d for d in ("raw", "tables", "state", "logs", "assets")
)

# ---------- Universe ----------
INCLUDE_DLC = True  # DLC prices matter for monetization questions
KEEP_DELISTED_DAYS = (
    30  # keep querying apps that vanish from the list, to record delisting
)

# ---------- Store snapshot ----------
BASE_COUNTRY = "US"
REGIONAL_COUNTRIES = ["DE", "GB", "BR", "IN", "JP"]  # tier A games only
ITEMS_BATCH = 100  # apps per GetItems call (reduce if Steam rejects large batches)
WEEKLY_BATCH = 25  # heavy requests (full descriptions) are much larger
TAG_COUNT = 20  # Steam's maximum
DOWNLOAD_CAPSULES = True  # header image for tier A games, only when it changes
SAVE_RAW_ITEMS = (
    True  # full daily GetItems responses (largest raw files; see README, storage)
)

# ---------- Player-count tiers ----------
# A: twice daily. B: daily. C: everything else, 1/7 of it each day (weekly per game).
TIER_A_MIN_REVIEWS = 1_000
TIER_B_MIN_REVIEWS = 50
NEW_RELEASE_DAYS = 90  # recent releases with >= NEW_MIN_REVIEWS go to tier A
NEW_MIN_REVIEWS = 10
CCU_C_ROTATION = 7

# ---------- Budget and politeness ----------
DAILY_STEAM_API_BUDGET = 90_000  # Steam's terms allow 100,000 calls/day; keep headroom
# --- Top-seller ranks (job "ranks"): Steam store search, Global Top Sellers ---
RANKS_GLOBAL_DEPTH = (
    10_000  # places in the overall ranking (games only; Steam's list is ~7,300)
)
RANKS_TAG_DEPTH = 500  # places within each tag's ranking (0 = skip tag rankings)
RANKS_PAGE = 100  # results per request (Steam's maximum)
RANKS_CATEGORY = "998"  # 998 = games; DLC, software and soundtracks excluded
SAVE_RAW_RANKS = False  # raw pages are large HTML; the parsed table keeps what matters
# Seasonal sales (from 3 days before to 3 days after), for the extra ranking snapshots.
# Add new sales as Valve announces them (YYYY-MM-DD, inclusive).
RANKS_SALE_PERIODS = [
    ("2026-12-14", "2027-01-07"),  # Winter Sale 2026: Dec 17 - Jan 4
    ("2027-03-15", "2027-03-28"),  # Spring Sale 2027: Mar 18 - 25
    ("2027-06-21", "2027-07-11"),  # Summer Sale 2027: Jun 24 - Jul 8
]

REQS_PER_SEC = {  # per process
    "api.steampowered.com": 2.0,
    "store.steampowered.com": 0.5,
    "api.twitch.tv": 5.0,
    "id.twitch.tv": 1.0,
    "shared.fastly.steamstatic.com": 2.0,
}
USER_AGENT = "UW-econ-research-collector/1.0 (academic, non-commercial)"
TWITCH_MAX_PAGES = 100  # 100 streams per page -> top 10,000 live streams

# ---------- Schedule (UTC hours) ----------
# Steam sales and price changes go live at 10:00 Pacific (17:00 or 18:00 UTC), so the
# daily price snapshot runs at 20:00 UTC, safely after any same-day change.
SLOTS = {
    "ranks": {"hours": [17]},  # global + tag lists: ~110 min; global only: ~6 min
    # Extra global-only ranking snapshots every 6 hours during seasonal sales (with 17:00 above)
    "ranks_sale": {"hours": [5, 11, 23], "periods": RANKS_SALE_PERIODS},
    # fast group
    "featured": {"hours": [0, 6, 12, 18]},
    "twitch": {"hours": [0, 6, 12, 18]},
    "charts": {"hours": [4, 16]},
    "topsellers": {"hours": [20]},
    # daily group
    "universe": {"hours": [19]},
    "items": {"hours": [20]},
    "regional": {"hours": [20]},
    "weekly": {"hours": [22], "weekday": 2},  # Wednesday
    # ccu group
    "ccu_a": {"hours": [4, 16]},
    "ccu_b": {"hours": [6]},
    "ccu_c": {"hours": [6]},
}
# A job whose slot passed while the machine was off runs at the next hourly check, and its
# log records how late it ran. Every row also carries its exact captured_at time.
GROUPS = {
    "fast": ["featured", "twitch", "charts", "topsellers", "ranks_sale"],
    "daily": ["ranks", "universe", "items", "regional", "weekly"],
    "ccu": ["ccu_a", "ccu_b", "ccu_c"],
}
STALE_LOCK_HOURS = 20
