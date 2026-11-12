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
    "fast": ["featured", "twitch", "charts", "topsellers"],
    "daily": ["universe", "items", "regional", "weekly"],
    "ccu": ["ccu_a", "ccu_b", "ccu_c"],
}
STALE_LOCK_HOURS = 20
