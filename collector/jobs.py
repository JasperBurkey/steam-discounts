"""Collection jobs. Each takes (http, run_id) and returns a dict of stats for the run log."""

import gzip
import json
import re

import pandas as pd
import parse as P
import settings as S
from core import (
    calls_today,
    chunks,
    iso,
    latest_run,
    read_json,
    read_table,
    utcnow,
    write_json,
    write_table,
)

API = "https://api.steampowered.com"
STORE = "https://store.steampowered.com"
CDN = "https://shared.fastly.steamstatic.com/store_item_assets/"

DAILY_REQUEST = {
    "include_assets": True,
    "include_release": True,
    "include_platforms": True,
    "include_tag_count": S.TAG_COUNT,
    "include_reviews": True,
    "include_basic_info": True,
    "include_best_purchase_option": True,
}
PRICE_REQUEST = {"include_best_purchase_option": True}
WEEKLY_REQUEST = {
    "include_basic_info": True,
    "include_full_description": True,
    "include_screenshots": True,
    "include_trailers": True,
    "include_supported_languages": True,
    "include_ratings": True,
    "include_all_purchase_options": True,
}


def _key(params):
    if S.STEAM_API_KEY:
        params["key"] = S.STEAM_API_KEY
    return params


def _need_key():
    if not S.STEAM_API_KEY:
        raise RuntimeError(
            "STEAM_API_KEY is not set (see README: get one at steamcommunity.com/dev/apikey)"
        )


def _ctx(country):
    return {"language": "english", "country_code": country, "steam_realm": 1}


# ---------------------------------------------------------------- GetItems with adaptive batching
def get_items(http, appids, country, data_request, save_raw=True):
    """Returns (items, failed_appids). Splits a batch in half if Steam rejects its size."""
    payload = {
        "ids": [{"appid": int(a)} for a in appids],
        "context": _ctx(country),
        "data_request": data_request,
    }
    body, status = http.get(
        f"{API}/IStoreBrowseService/GetItems/v1/",
        _key({"input_json": json.dumps(payload, separators=(",", ":"))}),
        save_raw=save_raw,
    )
    if status == 200:
        items = ((body or {}).get("response") or {}).get("store_items") or []
        got = {i.get("appid") or i.get("id") for i in items}
        return items, [a for a in appids if int(a) not in got]
    if status in (400, 413, 414) and len(appids) > 1:
        half = len(appids) // 2
        a_items, a_fail = get_items(
            http, appids[:half], country, data_request, save_raw
        )
        b_items, b_fail = get_items(
            http, appids[half:], country, data_request, save_raw
        )
        return a_items + b_items, a_fail + b_fail
    return [], list(appids)


# ---------------------------------------------------------------- universe
def job_universe(http, run_id):
    _need_key()
    now = utcnow()
    kinds = [("game", {"include_games": "true", "include_dlc": "false"})]
    if S.INCLUDE_DLC:
        kinds.append(("dlc", {"include_games": "false", "include_dlc": "true"}))
    current = {}
    for kind, flags in kinds:
        last = 0
        while True:
            body, status = http.get(
                f"{API}/IStoreService/GetAppList/v1/",
                _key(
                    {
                        "max_results": 50000,
                        "last_appid": last,
                        "include_software": "false",
                        "include_videos": "false",
                        "include_hardware": "false",
                        **flags,
                    }
                ),
            )
            if status != 200:
                raise RuntimeError(
                    f"GetAppList failed with HTTP {status}; universe not updated"
                )
            r = (body or {}).get("response") or {}
            apps = r.get("apps") or []
            for a in apps:
                current[a["appid"]] = (kind, a)
            if not r.get("have_more_results") or not apps:
                break
            last = r.get("last_appid") or apps[-1]["appid"]

    known_path = S.STATE / "known_apps.parquet"
    today = pd.Timestamp(now)
    cur = pd.DataFrame(
        [
            {
                "appid": int(k),
                "kind": v[0],
                "name": v[1].get("name"),
                "last_modified": P._ts(v[1].get("last_modified")),
                "price_change_number": v[1].get("price_change_number"),
            }
            for k, v in current.items()
        ]
    )
    if known_path.exists():
        known = pd.read_parquet(known_path)
    else:
        known = pd.DataFrame(
            {
                "appid": pd.Series(dtype="int64"),
                "kind": pd.Series(dtype="object"),
                "name": pd.Series(dtype="object"),
                "first_seen": pd.Series(dtype="datetime64[ns, UTC]"),
                "last_seen": pd.Series(dtype="datetime64[ns, UTC]"),
            }
        )
    listed = cur[["appid", "kind", "name"]].merge(
        known[["appid", "first_seen"]], on="appid", how="left"
    )
    listed["first_seen"] = listed["first_seen"].fillna(today)
    listed["last_seen"] = today
    gone = known[~known["appid"].isin(cur["appid"])]
    pd.concat([listed, gone], ignore_index=True).to_parquet(known_path, index=False)

    cutoff = today - pd.Timedelta(days=S.KEEP_DELISTED_DAYS)
    recent_gone = gone[pd.to_datetime(gone["last_seen"], utc=True) >= cutoff]
    rows = cur.assign(captured_at=now, in_current_list=True).to_dict("records")
    rows += (
        recent_gone[["appid", "kind", "name"]]
        .assign(captured_at=now, in_current_list=False)
        .to_dict("records")
    )
    n = write_table("universe", rows, P.UNIVERSE, run_id, now)
    return {"rows": n, "listed": len(current), "recently_delisted": n - len(current)}


# ---------------------------------------------------------------- daily store snapshot + tiers
def job_items(http, run_id):
    uni = latest_run("universe")
    if uni.empty:
        raise RuntimeError("no universe yet; run the universe job first")
    appids = uni["appid"].astype(int).tolist()
    rows = []
    n_batches = -(-len(appids) // S.ITEMS_BATCH)
    print(
        f"  store snapshot: {len(appids):,} apps in {n_batches:,} requests", flush=True
    )
    for b, batch in enumerate(chunks(appids, S.ITEMS_BATCH), 1):
        if b % 100 == 0:
            print(
                f"  store snapshot {b:,}/{n_batches:,} requests ({utcnow():%H:%M} UTC)",
                flush=True,
            )
        now = utcnow()
        items, failed = get_items(
            http, batch, S.BASE_COUNTRY, DAILY_REQUEST, save_raw=S.SAVE_RAW_ITEMS
        )
        for it in items:
            row = P.store_item_row(it, now, S.BASE_COUNTRY)
            row["http_status"] = "200"
            rows.append(row)
        rows += [
            {
                "captured_at": now,
                "country": S.BASE_COUNTRY,
                "appid": a,
                "http_status": "failed",
            }
            for a in failed
        ]
    prev = latest_run("store_items")
    n = write_table("store_items", rows, P.STORE_ITEMS, run_id)
    tiers = _assign_tiers(pd.DataFrame(rows), uni)
    write_table("tiers", tiers, P.TIERS, run_id)
    print(f"  saved {n:,} store rows and {len(tiers):,} tier assignments", flush=True)
    capsules = {"downloaded": 0, "errors": 0}
    if S.DOWNLOAD_CAPSULES:
        try:
            capsules = _download_capsules(http, pd.DataFrame(rows), prev, tiers)
        except Exception as e:  # noqa: BLE001
            capsules["errors"] = f"{type(e).__name__}: {e}"
    counts = pd.Series([t["tier"] for t in tiers]).value_counts().to_dict()
    return {
        "rows": n,
        "failed": sum(r["http_status"] != "200" for r in rows),
        "tier_counts": counts,
        "capsules_downloaded": capsules["downloaded"],
        "capsule_errors": capsules["errors"],
    }


def _assign_tiers(items, universe):
    now = utcnow()
    games = items[(items.get("http_status") == "200")].copy()
    games = games.merge(universe[["appid", "kind"]], on="appid", how="left")
    games = games[(games["kind"] == "game") & (games["is_coming_soon"] != True)]  # noqa: E712
    reviews = pd.to_numeric(games["review_count"], errors="coerce").fillna(0)
    released = pd.to_datetime(games["steam_release_date"], utc=True, errors="coerce")
    recent = released > (pd.Timestamp(now) - pd.Timedelta(days=S.NEW_RELEASE_DAYS))
    charted = set()
    for t in ("charts_most_played", "charts_concurrent", "topsellers"):
        df = latest_run(t)
        if not df.empty:
            charted |= set(df["appid"].dropna().astype(int))
    out = []
    for appid, rc, rec in zip(
        games["appid"].astype(int), reviews, recent.fillna(False)
    ):
        if rc >= S.TIER_A_MIN_REVIEWS:
            tier, why = "A", "reviews"
        elif appid in charted:
            tier, why = "A", "charts"
        elif rec and rc >= S.NEW_MIN_REVIEWS:
            tier, why = "A", "new_release"
        elif rc >= S.TIER_B_MIN_REVIEWS:
            tier, why = "B", "reviews"
        else:
            tier, why = "C", "other"
        out.append(
            {
                "captured_at": now,
                "appid": appid,
                "tier": tier,
                "review_count": int(rc),
                "reason": why,
            }
        )
    return out


def _download_capsules(http, today, prev, tiers):
    """Save new or changed header images for tier A games. Never fails the snapshot:
    a bad or missing image is counted and skipped."""
    tier_a = {t["appid"] for t in tiers if t["tier"] == "A"}
    old = (
        {}
        if prev.empty
        else {
            int(a): h
            for a, h in zip(prev["appid"], prev["asset_header"])
            if isinstance(h, str)
        }
    )
    todo = []
    for appid, header, fmt in zip(
        today["appid"], today.get("asset_header", []), today.get("asset_url_format", [])
    ):
        if (
            pd.notna(appid)
            and int(appid) in tier_a
            and isinstance(header, str)
            and isinstance(fmt, str)
            and header
            and fmt
            and old.get(int(appid)) != header
        ):
            todo.append((int(appid), header, fmt))
    print(f"  capsule images to download: {len(todo):,}", flush=True)
    n = errors = 0
    for i, (appid, header, fmt) in enumerate(todo, 1):
        try:
            body, status = http.get(
                CDN + fmt.replace("${FILENAME}", header), binary=True, save_raw=False
            )
            if status == 200 and body:
                d = S.ASSETS / "capsules" / str(appid)
                d.mkdir(parents=True, exist_ok=True)
                safe = re.sub(r"[^A-Za-z0-9._-]", "_", header)
                (d / f"{utcnow():%Y%m%d}_{safe}").write_bytes(body)
                n += 1
            else:
                errors += 1
        except Exception:  # noqa: BLE001 - images are optional; never lose the snapshot over one
            errors += 1
        if i % 500 == 0:
            print(f"  capsules {i:,}/{len(todo):,}", flush=True)
    return {"downloaded": n, "errors": errors}


def _tier(letter):
    t = latest_run("tiers")
    return t[t["tier"] == letter] if not t.empty else t


# ---------------------------------------------------------------- regional prices
def job_regional(http, run_id):
    a = _tier("A")
    if a.empty:
        return {"rows": 0, "note": "no tiers yet"}
    rows = []
    for country in S.REGIONAL_COUNTRIES:
        print(f"  regional prices: {country}", flush=True)
        for batch in chunks(a["appid"].astype(int).tolist(), S.ITEMS_BATCH):
            now = utcnow()
            items, failed = get_items(http, batch, country, PRICE_REQUEST)
            for it in items:
                row = P.regional_row(it, now, country)
                row["http_status"] = "200"
                rows.append(row)
            rows += [
                {
                    "captured_at": now,
                    "country": country,
                    "appid": x,
                    "http_status": "failed",
                }
                for x in failed
            ]
    return {"rows": write_table("regional_prices", rows, P.REGIONAL, run_id)}


# ---------------------------------------------------------------- price-change watcher
def _app_list(http):
    """Every game in the store catalog: appid -> (last_modified, price_change_number)."""
    out, last = {}, 0
    while True:
        body, status = http.get(
            f"{API}/IStoreService/GetAppList/v1/",
            _key(
                {
                    "max_results": 50000,
                    "last_appid": last,
                    "include_games": "true",
                    "include_dlc": "false",
                    "include_software": "false",
                    "include_videos": "false",
                    "include_hardware": "false",
                }
            ),
        )
        if status != 200:
            raise RuntimeError(f"GetAppList failed with HTTP {status}")
        r = (body or {}).get("response") or {}
        apps = r.get("apps") or []
        for a in apps:
            out[int(a["appid"])] = a.get("price_change_number")
        if not r.get("have_more_results") or not apps:
            return out
        last = r.get("last_appid") or apps[-1]["appid"]


def job_prices_watch(http, run_id):
    """Hourly: find games whose store price changed since the last check (Steam's
    price-change counter) and record the study games' new prices right away, so every
    discount's start and end is timed to the hour. Once a day (and on the first run) it
    also records all study games' prices as a full snapshot."""
    _need_key()
    now = utcnow()
    state_path = S.STATE / "price_change_numbers.parquet"
    current = _app_list(http)
    cur = pd.DataFrame({"appid": list(current), "pcn": list(current.values())})
    cur["pcn"] = pd.to_numeric(cur["pcn"], errors="coerce").fillna(-1).astype("int64")
    study = set(_study_ids())
    if state_path.exists():
        prev = pd.read_parquet(state_path)
        m = cur.merge(prev, on="appid", how="inner", suffixes=("", "_prev"))
        changed = m[m["pcn"] != m["pcn_prev"]]
    else:
        changed = cur.iloc[0:0].assign(pcn_prev=None)
    full = (not state_path.exists()) or now.hour == S.PRICE_SNAPSHOT_HOUR
    events = [
        {
            "captured_at": now,
            "appid": int(r.appid),
            "old_number": r.pcn_prev,
            "new_number": r.pcn,
            "in_study": int(r.appid) in study,
        }
        for r in changed.itertuples(index=False)
    ]
    n_events = write_table("price_changes", events, P.PRICE_CHANGES, run_id, now)
    targets = sorted(study) if full else sorted(set(changed["appid"]) & study)
    rows = []
    for batch in chunks(targets, S.ITEMS_BATCH):
        t = utcnow()
        items, failed = get_items(http, batch, S.PRICE_WATCH_COUNTRY, PRICE_REQUEST)
        for it in items:
            row = P.regional_row(it, t, S.PRICE_WATCH_COUNTRY)
            row["http_status"] = "200"
            row["reason"] = "snapshot" if full else "changed"
            rows.append(row)
        rows += [
            {
                "captured_at": t,
                "country": S.PRICE_WATCH_COUNTRY,
                "appid": x,
                "http_status": "failed",
                "reason": "snapshot" if full else "changed",
            }
            for x in failed
        ]
    n_prices = write_table("study_prices", rows, P.STUDY_PRICES, run_id, now)
    cur.to_parquet(state_path, index=False)  # only after both tables are written
    stats = {
        "rows": n_events + n_prices,
        "changed_games": len(changed),
        "changed_study_games": int(changed["appid"].isin(study).sum()),
        "prices_recorded": n_prices,
        "full_snapshot": full,
    }
    if not stats["rows"]:
        stats["note"] = "no price changes this hour"
    return stats


# ---------------------------------------------------------------- player counts
def _ccu(http, run_id, appids, tier):
    rows = []
    print(f"  player counts, tier {tier}: {len(appids):,} games", flush=True)
    for i, appid in enumerate(appids, 1):
        if i % 1000 == 0:
            print(
                f"  player counts {i:,}/{len(appids):,} ({utcnow():%H:%M} UTC)",
                flush=True,
            )
        body, status = http.get(
            f"{API}/ISteamUserStats/GetNumberOfCurrentPlayers/v1/",
            {"appid": int(appid)},
        )
        r = (body or {}).get("response") or {}
        rows.append(
            {
                "captured_at": utcnow(),
                "appid": int(appid),
                "tier": tier,
                "player_count": r.get("player_count"),
                "result": r.get("result"),
                "http_status": str(status),
            }
        )
    return {
        "rows": write_table("ccu", rows, P.CCU, run_id),
        "requested": len(appids),
        "ok": sum(r["player_count"] is not None for r in rows),
    }


def _budget_left(reserve=0):
    return max(0, S.DAILY_STEAM_API_BUDGET - calls_today() - reserve)


def job_ccu_a(http, run_id):
    a = _tier("A")
    if a.empty:
        charted = set()
        for t in ("charts_most_played", "charts_concurrent", "topsellers"):
            df = latest_run(t)
            if not df.empty:
                charted |= set(df["appid"].dropna().astype(int))
        if not charted:
            return {"rows": 0, "note": "no tiers or charts yet"}
        return _ccu(http, run_id, sorted(charted), "A_bootstrap")
    return _ccu(http, run_id, a["appid"].astype(int).tolist(), "A")


def _study_ids():
    if not S.STUDY_APPS.exists():
        return []
    return pd.read_csv(S.STUDY_APPS)["app_id"].astype(int).tolist()


def _study_reserve(now=None):
    """Calls still needed today for the study panel's remaining slots, so tiers B and C
    never use up the budget the thesis panel needs."""
    now = now or utcnow()
    n = len(_study_ids())
    runs = 0
    for job in ("ccu_study", "ccu_study_sale"):
        cfg = S.SLOTS[job]
        periods = cfg.get("periods")
        d = now.date().isoformat()
        if periods and not any(a <= d <= b for a, b in periods):
            continue
        runs += sum(h > now.hour for h in cfg["hours"])
    return n * runs


def job_ccu_study(http, run_id):
    ids = _study_ids()
    if not ids:
        return {"rows": 0, "note": f"no {S.STUDY_APPS.name}"}
    left = _budget_left()
    stats = _ccu(http, run_id, ids[:left], "study")
    stats["trimmed_for_budget"] = max(0, len(ids) - left)
    return stats


def job_ccu_b(http, run_id):
    b = _tier("B")
    if b.empty:
        return {"rows": 0, "note": "no tiers yet"}
    b = b.sort_values("review_count", ascending=False)
    # keep room for the next tier-A run, the store jobs and the study panel
    reserve = len(_tier("A")) + 2000 + _study_reserve()
    left = _budget_left(reserve)
    ids = b["appid"].astype(int).tolist()
    stats = _ccu(http, run_id, ids[:left], "B")
    stats["trimmed_for_budget"] = max(0, len(ids) - left)
    return stats


def job_ccu_c(http, run_id):
    c = _tier("C")
    if c.empty:
        return {"rows": 0, "note": "no tiers yet"}
    day = (utcnow().date() - pd.Timestamp("2026-01-01").date()).days % S.CCU_C_ROTATION
    ids = [a for a in c["appid"].astype(int) if a % S.CCU_C_ROTATION == day]
    left = _budget_left(len(_tier("A")) + 2000 + _study_reserve())
    stats = _ccu(http, run_id, ids[:left], "C")
    stats["trimmed_for_budget"] = max(0, len(ids) - left)
    return stats


# ---------------------------------------------------------------- charts and top sellers
def job_charts(http, run_id):
    _need_key()
    now = utcnow()
    ctx = json.dumps({"context": _ctx(S.BASE_COUNTRY)})
    mp, s1 = http.get(
        f"{API}/ISteamChartsService/GetMostPlayedGames/v1/", _key({"input_json": ctx})
    )
    cc, s2 = http.get(
        f"{API}/ISteamChartsService/GetGamesByConcurrentPlayers/v1/",
        _key({"input_json": ctx}),
    )
    n1 = write_table(
        "charts_most_played", P.most_played_rows(mp, now), P.MOST_PLAYED, run_id
    )
    n2 = write_table(
        "charts_concurrent", P.concurrent_rows(cc, now), P.CONCURRENT, run_id
    )
    if s1 != 200 and s2 != 200:
        raise RuntimeError(f"both chart endpoints failed ({s1}, {s2})")
    return {"most_played_rows": n1, "concurrent_rows": n2}


def job_topsellers(http, run_id):
    _need_key()
    rows = []
    for country in ["", S.BASE_COUNTRY] + S.REGIONAL_COUNTRIES:
        for start in range(0, 500, 100):
            now = utcnow()
            payload = {
                "country_code": country,
                "context": _ctx(country or S.BASE_COUNTRY),
                "page_start": start,
                "page_count": 100,
            }
            body, status = http.get(
                f"{API}/IStoreTopSellersService/GetWeeklyTopSellers/v1/",
                _key({"input_json": json.dumps(payload)}),
            )
            got = P.topseller_rows(body, now, country)
            rows += got
            if status != 200 or len(got) < 100:
                break
    return {"rows": write_table("topsellers", rows, P.TOPSELLERS, run_id)}


# ---------------------------------------------------------------- front page
def job_featured(http, run_id):
    rows = []
    for ep in ("featuredcategories", "featured"):
        now = utcnow()
        body, status = http.get(
            f"{STORE}/api/{ep}/", {"cc": S.BASE_COUNTRY, "l": "english"}
        )
        rows += P.featured_rows(body, now, ep)
    return {"rows": write_table("featured", rows, P.FEATURED, run_id)}


# ---------------------------------------------------------------- top-seller ranks
def _search_page(http, start, count, tagid=None):
    """One page of Steam's store search, sorted as Global Top Sellers (games only)."""
    params = {
        "filter": "globaltopsellers",
        "category1": S.RANKS_CATEGORY,
        "ignore_preferences": 1,
        "infinite": 1,
        "start": start,
        "count": count,
        "cc": S.BASE_COUNTRY,
        "l": "english",
    }
    if tagid is not None:
        params["tags"] = tagid
    return http.get(f"{STORE}/search/results/", params, save_raw=S.SAVE_RAW_RANKS)


def _rank_list(http, now, list_name, depth, tagid=None):
    """Page through one ranking until `depth` places or the end of the list. Duplicates
    (an item that moved between pages while paging) keep their first, higher rank."""
    rows, seen, start, failed = [], set(), 0, 0
    while start < depth:
        body, status = _search_page(http, start, S.RANKS_PAGE, tagid)
        if status != 200:
            failed += 1
            break
        page = P.search_rank_rows(body, now, list_name, tagid, start)
        if not page:
            break
        for r in page:
            if r["itemkey"] and r["itemkey"] not in seen:
                seen.add(r["itemkey"])
                rows.append(r)
        total = page[0]["total_count"] or 0
        start += S.RANKS_PAGE
        if total and start >= total:
            break
    return rows, failed


def _tag_ids():
    tags = read_table("tag_names", days=21)
    if tags is None or tags.empty:
        return []
    return sorted(int(t) for t in tags["tagid"].dropna().unique())


def _ranks(http, run_id, with_tags):
    now = utcnow()
    print("ranks: global list...", flush=True)
    rows, failed = _rank_list(http, now, "global", S.RANKS_GLOBAL_DEPTH)
    n_global = len(rows)
    print(f"ranks: global list done, {n_global:,} games", flush=True)
    tags = _tag_ids() if with_tags else []
    for i, tagid in enumerate(tags, 1):
        r, f = _rank_list(http, now, "tag", S.RANKS_TAG_DEPTH, tagid)
        rows += r
        failed += f
        if i % 25 == 0 or i == len(tags):
            print(
                f"ranks: {i}/{len(tags)} tag lists, {len(rows):,} rows so far",
                flush=True,
            )
    written = write_table("search_ranks", rows, P.SEARCH_RANKS, run_id, when=now)
    return {
        "rows": written,
        "global_ranked": n_global,
        "tags": len(tags),
        "failed_lists": failed,
        "warning": None if n_global >= 1000 else "global list shorter than expected",
    }


def job_ranks(http, run_id):
    """Daily Global Top Sellers ranking: the whole ranking (up to RANKS_GLOBAL_DEPTH games)
    and the top RANKS_TAG_DEPTH within each Steam tag. Ranks are Steam's revenue ordering;
    amounts are not published."""
    return _ranks(http, run_id, with_tags=S.RANKS_TAG_DEPTH > 0)


def job_ranks_sale(http, run_id):
    """Extra global-only ranking snapshot during seasonal sales (~6 minutes)."""
    return _ranks(http, run_id, with_tags=False)


# ---------------------------------------------------------------- twitch
def _twitch_token(http):
    tok = read_json(S.STATE / "twitch_token.json", {})
    if (
        tok.get("access_token")
        and tok.get("expires_at", 0) > utcnow().timestamp() + 3600
    ):
        return tok["access_token"]
    body, status = http.request(
        "POST",
        "https://id.twitch.tv/oauth2/token",
        params={
            "client_id": S.TWITCH_CLIENT_ID,
            "client_secret": S.TWITCH_CLIENT_SECRET,
            "grant_type": "client_credentials",
        },
        save_raw=False,
    )
    if status != 200 or not body or "access_token" not in body:
        raise RuntimeError(f"Twitch token request failed ({status})")
    write_json(
        S.STATE / "twitch_token.json",
        {
            "access_token": body["access_token"],
            "expires_at": utcnow().timestamp() + body.get("expires_in", 0),
        },
    )
    return body["access_token"]


def job_twitch(http, run_id):
    if not (S.TWITCH_CLIENT_ID and S.TWITCH_CLIENT_SECRET):
        return {"rows": 0, "note": "Twitch keys not set; skipped"}
    headers = {
        "Client-Id": S.TWITCH_CLIENT_ID,
        "Authorization": f"Bearer {_twitch_token(http)}",
    }
    streams, cursor, now = [], None, utcnow()
    for _ in range(S.TWITCH_MAX_PAGES):
        params = {"first": 100, **({"after": cursor} if cursor else {})}
        body, status = http.get(
            "https://api.twitch.tv/helix/streams",
            params,
            headers=headers,
            save_raw=False,
        )
        if status != 200:
            break
        streams += body.get("data") or []
        cursor = (body.get("pagination") or {}).get("cursor")
        if not cursor:
            break
    n1 = write_table(
        "twitch_streams", P.twitch_stream_rows(streams, now), P.TWITCH_STREAMS, run_id
    )
    n2 = write_table(
        "twitch_games", P.twitch_game_rows(streams, now), P.TWITCH_GAMES, run_id
    )
    return {"streams": n1, "game_rows": n2}


# ---------------------------------------------------------------- weekly heavy snapshot
def job_weekly(http, run_id):
    _need_key()
    now = utcnow()
    uni = latest_run("universe")
    games = (
        uni[uni["kind"] == "game"]["appid"].astype(int).tolist()
        if not uni.empty
        else []
    )
    hash_path = S.STATE / "text_hashes.parquet"
    hashes = (
        pd.read_parquet(hash_path)
        if hash_path.exists()
        else pd.DataFrame(columns=["appid", "field", "sha1"])
    )
    known = (
        set(zip(hashes["appid"].astype(int), hashes["field"], hashes["sha1"]))
        if len(hashes)
        else set()
    )
    texts, media, options, new_hashes = [], [], [], []
    raw_path = S.RAW / "weekly_lite" / f"{now:%Y-%m-%d}" / f"{run_id}.jsonl.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_path, "at", encoding="utf-8") as raw:
        n_batches = -(-len(games) // S.WEEKLY_BATCH)
        for b, batch in enumerate(chunks(games, S.WEEKLY_BATCH), 1):
            if b % 200 == 0:
                print(
                    f"  weekly {b:,}/{n_batches:,} requests ({utcnow():%H:%M} UTC)",
                    flush=True,
                )
            items, _ = get_items(
                http, batch, S.BASE_COUNTRY, WEEKLY_REQUEST, save_raw=False
            )
            for it in items:
                appid, t, m, o = P.weekly_rows(it, now)
                for field, text in t:
                    h = P.sha1(text)
                    if (int(appid), field, h) not in known:
                        texts.append(
                            {
                                "first_seen": now,
                                "appid": appid,
                                "field": field,
                                "sha1": h,
                                "text": text,
                            }
                        )
                        new_hashes.append(
                            {"appid": int(appid), "field": field, "sha1": h}
                        )
                media.append(m)
                options += o
                lite = {
                    k: v for k, v in it.items() if k != "full_description"
                }  # text kept in store_text
                raw.write(
                    json.dumps(
                        {"captured_at": iso(now), "item": lite}, separators=(",", ":")
                    )
                    + "\n"
                )
    if new_hashes:
        pd.concat([hashes, pd.DataFrame(new_hashes)], ignore_index=True).to_parquet(
            hash_path, index=False
        )

    tags, s1 = http.get(
        f"{API}/IStoreService/GetTagList/v1/", _key({"language": "english"})
    )
    tag_rows = [
        {"captured_at": now, "tagid": t.get("tagid"), "name": t.get("name")}
        for t in ((tags or {}).get("response") or {}).get("tags") or []
    ]
    cats, s2 = http.get(
        f"{API}/IStoreBrowseService/GetStoreCategories/v1/",
        _key({"input_json": json.dumps({"language": "english"})}),
    )
    cat_rows = [
        {
            "captured_at": now,
            "categoryid": c.get("categoryid"),
            "type": P._str(c.get("type")),
            "internal_name": c.get("internal_name"),
            "display_name": c.get("display_name"),
        }
        for c in ((cats or {}).get("response") or {}).get("categories") or []
    ]
    return {
        "text_versions_new": write_table("store_text", texts, P.TEXT, run_id),
        "media_rows": write_table("store_media", media, P.MEDIA, run_id),
        "purchase_option_rows": write_table(
            "purchase_options", options, P.PURCHASE_OPTIONS, run_id
        ),
        "tag_names": write_table("tag_names", tag_rows, P.TAG_NAMES, run_id),
        "categories": write_table("store_categories", cat_rows, P.CATEGORIES, run_id),
    }


JOBS = {
    "universe": job_universe,
    "items": job_items,
    "regional": job_regional,
    "ccu_a": job_ccu_a,
    "ccu_b": job_ccu_b,
    "ccu_c": job_ccu_c,
    "ccu_study": job_ccu_study,
    "prices_watch": job_prices_watch,
    "ccu_study_sale": job_ccu_study,
    "charts": job_charts,
    "topsellers": job_topsellers,
    "featured": job_featured,
    "twitch": job_twitch,
    "ranks": job_ranks,
    "ranks_sale": job_ranks_sale,
    "weekly": job_weekly,
}
