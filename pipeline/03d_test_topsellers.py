"""Step 3d: can Steam's weekly top-seller charts be retrieved for PAST weeks?

If yes, weekly top-seller ranks for 2023-2026 become a second outcome (a revenue-ranked
signal, unlike reviews). This makes a handful of requests and prints what comes back.

Needs a Steam Web API key: STEAM_API_KEY in this folder's .env, or it is read from the
collector's .env (C:\\steam_collector\\.env).

  python 03d_test_topsellers.py
"""

import json
import os

import config
import pandas as pd
import requests

URL = "https://api.steampowered.com/IStoreTopSellersService/GetWeeklyTopSellers/v1/"


def steam_key():
    key = os.environ.get("STEAM_API_KEY", "")
    env = config.COLLECTOR_DATA.parent / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("STEAM_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def week(start=None):
    payload = {
        "country_code": "US",
        "context": {"language": "english", "country_code": "US", "steam_realm": 1},
        "page_start": 0,
        "page_count": 20,
    }
    if start is not None:
        payload["start_date"] = int(start)
    r = requests.get(
        URL, params={"key": KEY, "input_json": json.dumps(payload)}, timeout=30
    )
    body = r.json() if r.status_code == 200 else {}
    resp = body.get("response") or {}
    ranks = [
        x.get("appid") or (x.get("item") or {}).get("appid")
        for x in resp.get("ranks") or []
    ]
    return r.status_code, resp.get("start_date"), ranks


KEY = steam_key()
if not KEY:
    raise SystemExit(
        "No STEAM_API_KEY found (this folder's .env or the collector's .env)."
    )

code, cur_start, cur = week()
print(
    f"Current week: HTTP {code}, week starting {pd.to_datetime(cur_start, unit='s') if cur_start else '?'}, "
    f"top 5 app IDs {cur[:5]}"
)
results = []
for label, date in [
    ("Summer Sale 2024", "2024-06-27"),
    ("Winter Sale 2023", "2023-12-26"),
    ("Spring 2025, normal week", "2025-04-15"),
]:
    t = int(pd.Timestamp(date, tz="UTC").timestamp())
    code, start, ranks = week(t)
    same = ranks[:20] == cur[:20]
    results.append(bool(ranks) and not same and start not in (None, cur_start))
    print(
        f"{label:<26} HTTP {code}, week starting {pd.to_datetime(start, unit='s') if start else '?'}, "
        f"{len(ranks)} ranks, top 5 {ranks[:5]}{'  (same as current week!)' if same else ''}"
    )

print()
if all(results):
    print(
        "PASS: past weeks are available. Tell Claude, and the full 2023-2026 rank history can be added."
    )
else:
    print(
        "NOT AVAILABLE: Steam returned the current week (or nothing) for past dates. "
        "Reviews stay the only historical outcome; the collector records ranks from now on."
    )
