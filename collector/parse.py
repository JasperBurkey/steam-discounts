"""Turn API responses into flat rows with stable schemas.

Nested lists (tags, developers, categories) are stored as JSON strings so that every
daily file has identical column types. Parse them with json.loads when building features.
Enum fields are stored as strings exactly as Steam sends them (usually integers).
"""

import hashlib
import json
from datetime import datetime, timezone

S_, I_, F_, B_, T_ = "string", "Int64", "Float64", "boolean", "datetime"


def _ts(v):
    try:
        v = int(v)
        return datetime.fromtimestamp(v, timezone.utc) if v > 0 else None
    except (TypeError, ValueError):
        return None


def _int(v):
    """Steam sends 64-bit integers (prices) as JSON strings; accept both."""
    if v is None or isinstance(v, bool):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None


def _j(v):
    return (
        json.dumps(v, separators=(",", ":"), ensure_ascii=False)
        if v not in (None, [], {})
        else None
    )


def _str(v):
    return None if v is None else str(v)


# ---------------------------------------------------------------- store items (daily)
STORE_ITEMS = {
    "captured_at": T_,
    "country": S_,
    "appid": I_,
    "http_status": S_,
    "success": I_,
    "visible": B_,
    "unlisted": B_,
    "unavailable_for_country": B_,
    "name": S_,
    "type": S_,
    "is_free": B_,
    "is_early_access": B_,
    "parent_appid": I_,
    "steam_release_date": T_,
    "original_release_date": T_,
    "original_steam_release_date": T_,
    "is_coming_soon": B_,
    "is_preload": B_,
    "custom_release_date_message": S_,
    "coming_soon_display": S_,
    "platform_windows": B_,
    "platform_mac": B_,
    "platform_linux": B_,
    "vr_supported": B_,
    "vr_only": B_,
    "deck_compat": S_,
    "review_count": I_,
    "review_pct_positive": I_,
    "review_score": S_,
    "review_count_unfiltered": I_,
    "review_pct_positive_unfiltered": I_,
    "price_final_cents": I_,
    "price_original_cents": I_,
    "discount_pct": I_,
    "discount_end_date": T_,
    "discount_descriptions": S_,
    "lowest_recent_price_cents": I_,
    "hide_discount_pct_for_compliance": B_,
    "purchase_packageid": I_,
    "purchase_bundleid": I_,
    "purchase_option_name": S_,
    "formatted_final_price": S_,
    "developers": S_,
    "publishers": S_,
    "franchises": S_,
    "capsule_headline": S_,
    "tags": S_,
    "tagids": S_,
    "categories": S_,
    "content_descriptorids": S_,
    "asset_url_format": S_,
    "asset_header": S_,
    "asset_main_capsule": S_,
    "asset_small_capsule": S_,
    "free_weekend_start": T_,
    "free_weekend_end": T_,
}


def _purchase(item):
    po = item.get("best_purchase_option")
    if not po:
        opts = item.get("purchase_options") or []
        po = opts[0] if opts else {}
    return po or {}


def store_item_row(item, captured_at, country):
    rel = item.get("release") or {}
    plat = item.get("platforms") or {}
    vr = plat.get("vr_support") or {}
    rev = item.get("reviews") or {}
    rf, ru = rev.get("summary_filtered") or {}, rev.get("summary_unfiltered") or {}
    po = _purchase(item)
    discounts = po.get("active_discounts") or []
    ends = [
        _int(d.get("discount_end_date"))
        for d in discounts
        if _int(d.get("discount_end_date"))
    ]
    basic = item.get("basic_info") or {}
    assets = item.get("assets") or {}
    fw = item.get("free_weekend") or {}

    def names(key):
        return [c.get("name") for c in basic.get(key) or [] if c.get("name")]

    return {
        "captured_at": captured_at,
        "country": country,
        "appid": _int(item.get("appid") or item.get("id")),
        "success": _int(item.get("success")),
        "visible": item.get("visible"),
        "unlisted": item.get("unlisted"),
        "unavailable_for_country": item.get(
            "unvailable_for_country_restriction"
        ),  # Steam's spelling
        "name": item.get("name"),
        "type": _str(item.get("type")),
        "is_free": item.get("is_free"),
        "is_early_access": item.get("is_early_access") or rel.get("is_early_access"),
        "parent_appid": _int((item.get("related_items") or {}).get("parent_appid")),
        "steam_release_date": _ts(rel.get("steam_release_date")),
        "original_release_date": _ts(rel.get("original_release_date")),
        "original_steam_release_date": _ts(rel.get("original_steam_release_date")),
        "is_coming_soon": rel.get("is_coming_soon"),
        "is_preload": rel.get("is_preload"),
        "custom_release_date_message": rel.get("custom_release_date_message"),
        "coming_soon_display": rel.get("coming_soon_display"),
        "platform_windows": plat.get("windows"),
        "platform_mac": plat.get("mac"),
        "platform_linux": plat.get("steamos_linux"),
        "vr_supported": vr.get("vrhmd"),
        "vr_only": vr.get("vrhmd_only"),
        "deck_compat": _str(plat.get("steam_deck_compat_category")),
        "review_count": _int(rf.get("review_count")),
        "review_pct_positive": _int(rf.get("percent_positive")),
        "review_score": _str(rf.get("review_score")),
        "review_count_unfiltered": _int(ru.get("review_count")),
        "review_pct_positive_unfiltered": _int(ru.get("percent_positive")),
        "price_final_cents": _int(po.get("final_price_in_cents")),
        "price_original_cents": _int(po.get("original_price_in_cents"))
        or _int(po.get("final_price_in_cents")),
        "discount_pct": _int(po.get("discount_pct")) or (0 if po else None),
        "discount_end_date": _ts(max(ends)) if ends else None,
        "discount_descriptions": _j(
            [
                d.get("discount_description")
                for d in discounts
                if d.get("discount_description")
            ]
        ),
        "lowest_recent_price_cents": _int(po.get("lowest_recent_price_in_cents")),
        "hide_discount_pct_for_compliance": po.get("hide_discount_pct_for_compliance"),
        "purchase_packageid": _int(po.get("packageid")),
        "purchase_bundleid": _int(po.get("bundleid")),
        "purchase_option_name": po.get("purchase_option_name"),
        "formatted_final_price": po.get("formatted_final_price"),
        "developers": _j(names("developers")),
        "publishers": _j(names("publishers")),
        "franchises": _j(names("franchises")),
        "capsule_headline": basic.get("capsule_headline"),
        "tags": _j(
            [
                [_int(t.get("tagid")), _int(t.get("weight"))]
                for t in item.get("tags") or []
            ]
        ),
        "tagids": _j(item.get("tagids")),
        "categories": _j(item.get("categories")),
        "content_descriptorids": _j(item.get("content_descriptorids")),
        "asset_url_format": assets.get("asset_url_format"),
        "asset_header": assets.get("header"),
        "asset_main_capsule": assets.get("main_capsule"),
        "asset_small_capsule": assets.get("small_capsule"),
        "free_weekend_start": _ts(fw.get("start_time")),
        "free_weekend_end": _ts(fw.get("end_time")),
    }


REGIONAL = {
    "captured_at": T_,
    "country": S_,
    "appid": I_,
    "http_status": S_,
    "success": I_,
    "unavailable_for_country": B_,
    "price_final_cents": I_,
    "price_original_cents": I_,
    "discount_pct": I_,
    "lowest_recent_price_cents": I_,
    "formatted_final_price": S_,
    "formatted_original_price": S_,
}


def regional_row(item, captured_at, country):
    po = _purchase(item)
    return {
        "captured_at": captured_at,
        "country": country,
        "appid": item.get("appid") or item.get("id"),
        "success": item.get("success"),
        "unavailable_for_country": item.get("unvailable_for_country_restriction"),
        "price_final_cents": _int(po.get("final_price_in_cents")),
        "price_original_cents": _int(po.get("original_price_in_cents"))
        or _int(po.get("final_price_in_cents")),
        "discount_pct": _int(po.get("discount_pct")) or (0 if po else None),
        "lowest_recent_price_cents": _int(po.get("lowest_recent_price_in_cents")),
        "formatted_final_price": po.get("formatted_final_price"),
        "formatted_original_price": po.get("formatted_original_price"),
    }


# ---------------------------------------------------------------- weekly heavy
TEXT = {"first_seen": T_, "appid": I_, "field": S_, "sha1": S_, "text": S_}
MEDIA = {
    "captured_at": T_,
    "appid": I_,
    "screenshots": S_,
    "screenshot_count": I_,
    "trailers": S_,
    "trailer_count": I_,
    "languages": S_,
    "rating": S_,
}
PURCHASE_OPTIONS = {
    "captured_at": T_,
    "appid": I_,
    "packageid": I_,
    "bundleid": I_,
    "name": S_,
    "price_final_cents": I_,
    "price_original_cents": I_,
    "discount_pct": I_,
    "bundle_discount_pct": I_,
    "included_game_count": I_,
    "is_commercial_license": B_,
}


def sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def weekly_rows(item, captured_at):
    appid = item.get("appid") or item.get("id")
    basic = item.get("basic_info") or {}
    texts = [
        (f, t)
        for f, t in (
            ("short_description", basic.get("short_description")),
            ("full_description", item.get("full_description")),
        )
        if t
    ]
    shots = item.get("screenshots") or {}
    files = [
        s.get("filename")
        for s in (shots.get("all_ages_screenshots") or [])
        + (shots.get("mature_content_screenshots") or [])
        if s.get("filename")
    ]
    tr = item.get("trailers") or {}
    trailers = [
        {"name": t.get("trailer_name"), "base_id": t.get("trailer_base_id")}
        for t in (tr.get("highlights") or []) + (tr.get("other_trailers") or [])
    ]
    media = {
        "captured_at": captured_at,
        "appid": appid,
        "screenshots": _j(files),
        "screenshot_count": len(files),
        "trailers": _j(trailers),
        "trailer_count": len(trailers),
        "languages": _j(item.get("supported_languages")),
        "rating": _j(item.get("game_rating")),
    }
    options = [
        {
            "captured_at": captured_at,
            "appid": appid,
            "packageid": _int(o.get("packageid")),
            "bundleid": _int(o.get("bundleid")),
            "name": o.get("purchase_option_name"),
            "price_final_cents": _int(o.get("final_price_in_cents")),
            "price_original_cents": _int(o.get("original_price_in_cents"))
            or _int(o.get("final_price_in_cents")),
            "discount_pct": _int(o.get("discount_pct")),
            "bundle_discount_pct": _int(o.get("bundle_discount_pct")),
            "included_game_count": _int(o.get("included_game_count")),
            "is_commercial_license": o.get("is_commercial_license"),
        }
        for o in item.get("purchase_options") or []
    ]
    return appid, texts, media, options


# ---------------------------------------------------------------- charts
MOST_PLAYED = {
    "captured_at": T_,
    "rollup_date": T_,
    "rank": I_,
    "appid": I_,
    "last_week_rank": I_,
    "peak_in_game": I_,
}
CONCURRENT = {
    "captured_at": T_,
    "last_update": T_,
    "rank": I_,
    "appid": I_,
    "concurrent_in_game": I_,
    "peak_in_game": I_,
}
TOPSELLERS = {
    "captured_at": T_,
    "country": S_,
    "week_start": T_,
    "rank": I_,
    "appid": I_,
    "last_week_rank": I_,
    "consecutive_weeks": I_,
}


def most_played_rows(body, captured_at):
    r = (body or {}).get("response") or {}
    return [
        {
            "captured_at": captured_at,
            "rollup_date": _ts(r.get("rollup_date")),
            "rank": x.get("rank"),
            "appid": x.get("appid"),
            "last_week_rank": x.get("last_week_rank"),
            "peak_in_game": x.get("peak_in_game"),
        }
        for x in r.get("ranks") or []
    ]


def concurrent_rows(body, captured_at):
    r = (body or {}).get("response") or {}
    return [
        {
            "captured_at": captured_at,
            "last_update": _ts(r.get("last_update")),
            "rank": x.get("rank"),
            "appid": x.get("appid"),
            "concurrent_in_game": x.get("concurrent_in_game"),
            "peak_in_game": x.get("peak_in_game"),
        }
        for x in r.get("ranks") or []
    ]


def topseller_rows(body, captured_at, country):
    r = (body or {}).get("response") or {}
    return [
        {
            "captured_at": captured_at,
            "country": country or "GLOBAL",
            "week_start": _ts(r.get("start_date")),
            "rank": x.get("rank"),
            "appid": x.get("appid") or (x.get("item") or {}).get("appid"),
            "last_week_rank": x.get("last_week_rank"),
            "consecutive_weeks": x.get("consecutive_weeks"),
        }
        for x in r.get("ranks") or []
    ]


# ---------------------------------------------------------------- featured
FEATURED = {
    "captured_at": T_,
    "endpoint": S_,
    "section": S_,
    "position": I_,
    "item_id": I_,
    "item_type": S_,
    "name": S_,
    "discounted": B_,
    "discount_pct": I_,
    "original_price_cents": I_,
    "final_price_cents": I_,
    "discount_expiration": T_,
}


def featured_rows(body, captured_at, endpoint):
    rows = []
    if not isinstance(body, dict):
        return rows
    for key, val in body.items():
        if isinstance(val, dict) and isinstance(val.get("items"), list):
            section, items = val.get("id") or val.get("name") or key, val["items"]
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            section, items = key, val
        else:
            continue
        for pos, it in enumerate(items):
            if not isinstance(it, dict) or it.get("id") is None:
                continue
            rows.append(
                {
                    "captured_at": captured_at,
                    "endpoint": endpoint,
                    "section": str(section),
                    "position": pos,
                    "item_id": it.get("id"),
                    "item_type": _str(it.get("type")),
                    "name": it.get("name"),
                    "discounted": it.get("discounted"),
                    "discount_pct": it.get("discount_percent"),
                    "original_price_cents": it.get("original_price"),
                    "final_price_cents": it.get("final_price"),
                    "discount_expiration": _ts(it.get("discount_expiration")),
                }
            )
    return rows


# ---------------------------------------------------------------- player counts
CCU = {
    "captured_at": T_,
    "appid": I_,
    "tier": S_,
    "player_count": I_,
    "result": I_,
    "http_status": S_,
}

# ---------------------------------------------------------------- twitch
TWITCH_STREAMS = {
    "captured_at": T_,
    "rank": I_,
    "stream_id": S_,
    "user_id": S_,
    "user_login": S_,
    "game_id": S_,
    "game_name": S_,
    "viewer_count": I_,
    "language": S_,
    "started_at": T_,
    "is_mature": B_,
}
TWITCH_GAMES = {
    "captured_at": T_,
    "game_id": S_,
    "game_name": S_,
    "language": S_,
    "viewers": I_,
    "streams": I_,
    "max_stream_viewers": I_,
}


def twitch_stream_rows(streams, captured_at):
    return [
        {
            "captured_at": captured_at,
            "rank": i + 1,
            "stream_id": s.get("id"),
            "user_id": s.get("user_id"),
            "user_login": s.get("user_login"),
            "game_id": s.get("game_id"),
            "game_name": s.get("game_name"),
            "viewer_count": s.get("viewer_count"),
            "language": s.get("language"),
            "started_at": s.get("started_at"),
            "is_mature": s.get("is_mature"),
        }
        for i, s in enumerate(streams)
    ]


def twitch_game_rows(streams, captured_at):
    agg = {}
    for s in streams:
        for lang in ("ALL", s.get("language") or "unknown"):
            k = (s.get("game_id"), lang)
            a = agg.setdefault(
                k,
                {
                    "captured_at": captured_at,
                    "game_id": s.get("game_id"),
                    "game_name": s.get("game_name"),
                    "language": lang,
                    "viewers": 0,
                    "streams": 0,
                    "max_stream_viewers": 0,
                },
            )
            v = int(s.get("viewer_count") or 0)
            a["viewers"] += v
            a["streams"] += 1
            a["max_stream_viewers"] = max(a["max_stream_viewers"], v)
    return list(agg.values())


# ---------------------------------------------------------------- universe and reference
UNIVERSE = {
    "captured_at": T_,
    "appid": I_,
    "name": S_,
    "kind": S_,
    "last_modified": T_,
    "price_change_number": I_,
    "in_current_list": B_,
}
TAG_NAMES = {"captured_at": T_, "tagid": I_, "name": S_}
CATEGORIES = {
    "captured_at": T_,
    "categoryid": I_,
    "type": S_,
    "internal_name": S_,
    "display_name": S_,
}
TIERS = {"captured_at": T_, "appid": I_, "tier": S_, "review_count": I_, "reason": S_}
