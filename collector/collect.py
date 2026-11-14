"""Entry point.

python collect.py selftest          one small call to every endpoint; run this first
python collect.py run --group fast  run whatever is due in a group (cron calls this hourly)
python collect.py run --job items --force   run one job now, ignoring the schedule
python collect.py status            last run of every job and today's call count
python collect.py health            check for stale or failing jobs; alert if configured
"""

import argparse
import json
import sys
import time
import traceback
from datetime import timedelta

import parse as P
import requests
import settings as S
from core import (
    Http,
    Lock,
    calls_today,
    iso,
    log_run,
    new_run_id,
    read_json,
    utcnow,
    write_json,
)
from jobs import CDN, DAILY_REQUEST, JOBS, PRICE_REQUEST, get_items

SCHED = S.STATE / "schedule.json"
PERIOD_H = {
    job: (24 * 7 if "weekday" in cfg else 24 / len(cfg["hours"]))
    for job, cfg in S.SLOTS.items()
}


# ---------------------------------------------------------------- scheduling
def in_periods(cfg, t):
    """True if the job has no date limits, or t falls inside one of its periods (inclusive)."""
    if "periods" not in cfg:
        return True
    d = t.date().isoformat()
    return any(start <= d <= end for start, end in cfg["periods"])


def latest_slot(job, now):
    cfg = S.SLOTS[job]
    t = now.replace(minute=0, second=0, microsecond=0)
    for _ in range(24 * 8):
        if (
            t.hour in cfg["hours"]
            and ("weekday" not in cfg or t.weekday() == cfg["weekday"])
            and in_periods(cfg, t)
        ):
            return t
        t -= timedelta(hours=1)
    return None


def due(job, now, sched):
    slot = latest_slot(job, now)
    return slot is not None and sched.get(job) != iso(slot), slot


def run_job(job, slot=None):
    run_id, started = new_run_id(), utcnow()
    http = Http(job, run_id)
    rec = {
        "run_id": run_id,
        "job": job,
        "slot": iso(slot) if slot else None,
        "started": iso(started),
    }
    try:
        stats = JOBS[job](http, run_id)
        rec.update(status="ok", stats=stats)
    except Exception as e:  # noqa: BLE001 - every failure must be logged, never crash the scheduler
        rec.update(
            status="error",
            error=f"{type(e).__name__}: {e}",
            trace=traceback.format_exc()[-2000:],
        )
    finally:
        http.close()
    rec.update(
        finished=iso(utcnow()),
        seconds=round((utcnow() - started).total_seconds()),
        calls=http.calls,
        failed_calls=http.failures,
    )
    if slot:
        rec["lag_hours"] = round((started - slot).total_seconds() / 3600, 2)
    log_run(rec)
    print(
        json.dumps(
            {k: rec[k] for k in ("job", "status", "seconds", "calls", "failed_calls")}
            | (
                {"stats": rec["stats"]}
                if "stats" in rec
                else {"error": rec.get("error")}
            )
        )
    )
    return rec


def cmd_run(args):
    if args.job:
        return run_job(
            args.job, None if args.force else latest_slot(args.job, utcnow())
        )
    failed = False
    with Lock(f"group_{args.group}"):
        for job in S.GROUPS[args.group]:
            sched = read_json(SCHED, {})
            is_due, slot = due(job, utcnow(), sched)
            if not is_due:
                continue
            rec = run_job(job, slot)
            if rec["status"] == "ok":
                sched = read_json(SCHED, {})  # re-read: another group may have written
                sched[job] = iso(slot)
                write_json(SCHED, sched)
            else:
                failed = True
                alert(
                    f"steam collector: {job} failed: {rec.get('error')}",
                    key=f"fail_{job}",
                )
    heartbeat(failed)
    cmd_health(args, quiet=True)


def heartbeat(failed=False):
    """Ping an outside monitor after every scheduled run. If the pings stop (machine off,
    scheduler broken), the monitor alerts you; this is the failure the collector cannot report."""
    if not S.HEARTBEAT_URL:
        return
    try:
        requests.get(
            S.HEARTBEAT_URL.rstrip("/") + ("/fail" if failed else ""), timeout=10
        )
    except requests.RequestException:
        pass


# ---------------------------------------------------------------- monitoring
def last_runs():
    out = {}
    path = S.LOGS / "runs.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out.setdefault(r["job"], {})[r["status"]] = r
    return out


def cmd_status(args):
    runs = last_runs()
    print(
        f"Steam Web API calls today: {calls_today():,} / budget {S.DAILY_STEAM_API_BUDGET:,}"
    )
    for job in S.SLOTS:
        ok, err = runs.get(job, {}).get("ok"), runs.get(job, {}).get("error")
        print(
            f"  {job:<11} last ok: {ok['finished'] if ok else 'never':<21} "
            f"{('last error: ' + err['finished'] + ' ' + err['error'][:60]) if err else ''}"
        )


def cmd_health(args, quiet=False):
    runs, now, problems = last_runs(), utcnow(), []
    for job, period in PERIOD_H.items():
        if job == "twitch" and not (S.TWITCH_CLIENT_ID and S.TWITCH_CLIENT_SECRET):
            continue
        if not in_periods(S.SLOTS[job], now):
            continue  # date-limited job (e.g. sale periods only) outside its periods
        ok = runs.get(job, {}).get("ok")
        if not ok:
            continue  # never run yet: not an alert until the first success
        age = (now - _parse(ok["finished"])).total_seconds() / 3600
        if age > period * 1.5 + 2:
            problems.append(f"{job} last succeeded {age:.0f}h ago")
        stats = ok.get("stats") or {}
        if ok.get("failed_calls", 0) > 0.2 * max(
            1, sum((ok.get("calls") or {}).values())
        ):
            problems.append(f"{job}: {ok['failed_calls']} failed calls in last run")
        if stats.get("rows") == 0 and "note" not in stats:
            problems.append(f"{job}: last run wrote 0 rows")
    if problems:
        alert("steam collector health: " + "; ".join(problems), key="health")
    if not quiet:
        print("\n".join(problems) or "healthy")


def _parse(s):
    from datetime import datetime, timezone

    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def alert(message, key):
    print("ALERT:", message, file=sys.stderr)
    if not S.ALERT_WEBHOOK_URL:
        return
    sent = read_json(S.STATE / "alerts.json", {})
    if (
        time.time() - sent.get(key, 0) < 12 * 3600
    ):  # at most one alert per issue per 12h
        return
    try:
        if "discord" in S.ALERT_WEBHOOK_URL:
            requests.post(
                S.ALERT_WEBHOOK_URL, json={"content": message[:1900]}, timeout=15
            )
        else:
            requests.post(S.ALERT_WEBHOOK_URL, data=message.encode(), timeout=15)
        sent[key] = time.time()
        write_json(S.STATE / "alerts.json", sent)
    except requests.RequestException:
        pass


# ---------------------------------------------------------------- selftest
def cmd_selftest(args):
    """One small request per endpoint. Prints what came back so you can see it works."""
    http, now, results = Http("selftest", new_run_id()), utcnow(), []
    sample = [730, 413150, 1145360, 2379780]

    def check(name, ok, detail):
        results.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    def attempt(name, fn):
        """Run one check; an unexpected response shape fails that line instead of the whole test."""
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            check(name, False, f"{type(e).__name__}: {e}")

    if not S.STEAM_API_KEY:
        print(
            "[WARN] STEAM_API_KEY not set: universe, charts, top sellers and weekly will fail."
        )
    k = {"key": S.STEAM_API_KEY} if S.STEAM_API_KEY else {}
    ctx = json.dumps({"context": {"language": "english", "country_code": "US"}})
    rows = []

    def t_items():
        items, failed = get_items(http, sample, "US", DAILY_REQUEST)
        rows.extend(P.store_item_row(i, now, "US") for i in items)
        check(
            "GetItems (daily store snapshot)",
            len(rows) == len(sample) and rows[0]["review_count"] is not None,
            "; ".join(
                f"{r['name']}: ${(r['price_final_cents'] or 0) / 100:.2f}, {r['discount_pct']}% off, "
                f"{r['review_count']} reviews, {len(json.loads(r['tags'] or '[]'))} tags"
                for r in rows
            )
            or f"failed {failed}",
        )

    def t_regional():
        items, _ = get_items(http, sample[1:3], "DE", PRICE_REQUEST)
        check(
            "GetItems (regional price, DE)",
            bool(items),
            ", ".join(
                str(P.regional_row(i, now, "DE")["formatted_final_price"])
                for i in items
            ),
        )

    def t_ccu():
        body, st = http.get(
            "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/",
            {"appid": 730},
        )
        check(
            "GetNumberOfCurrentPlayers",
            st == 200,
            f"CS2 players now: {(body or {}).get('response', {}).get('player_count')}",
        )

    def t_applist():
        body, st = http.get(
            "https://api.steampowered.com/IStoreService/GetAppList/v1/",
            {**k, "max_results": 5},
        )
        apps = ((body or {}).get("response") or {}).get("apps") or []
        check(
            "GetAppList",
            st == 200 and bool(apps),
            f"HTTP {st}; {len(apps)} apps, first: {apps[0] if apps else None}",
        )

    def t_most_played():
        body, st = http.get(
            "https://api.steampowered.com/ISteamChartsService/GetMostPlayedGames/v1/",
            {**k, "input_json": ctx},
        )
        mp = P.most_played_rows(body, now)
        check(
            "GetMostPlayedGames",
            bool(mp),
            f"{len(mp)} ranks; #1 appid {mp[0]['appid']}, daily peak {mp[0]['peak_in_game']}"
            if mp
            else f"HTTP {st}, no ranks",
        )

    def t_ranks():
        from jobs import _search_page

        body, st = _search_page(http, 0, 50)
        rr = P.search_rank_rows(body, now, "global", None, 0) if st == 200 else []
        check(
            "Store search: Global Top Sellers",
            len(rr) >= 25 and rr[0]["appid"] is not None,
            f"{len(rr)} ranks; #1 {rr[0]['name']} (app {rr[0]['appid']})"
            if rr
            else f"HTTP {st}, nothing parsed",
        )

    def t_concurrent():
        body, st = http.get(
            "https://api.steampowered.com/ISteamChartsService/GetGamesByConcurrentPlayers/v1/",
            {**k, "input_json": ctx},
        )
        cc = P.concurrent_rows(body, now)
        check(
            "GetGamesByConcurrentPlayers",
            bool(cc),
            f"{len(cc)} ranks; #1 appid {cc[0]['appid']}, {cc[0]['concurrent_in_game']} playing"
            if cc
            else f"HTTP {st}, no ranks",
        )

    def t_topsellers():
        payload = json.dumps(
            {
                "country_code": "",
                "context": {"language": "english", "country_code": "US"},
                "page_start": 0,
                "page_count": 5,
            }
        )
        body, st = http.get(
            "https://api.steampowered.com/IStoreTopSellersService/GetWeeklyTopSellers/v1/",
            {**k, "input_json": payload},
        )
        ts = P.topseller_rows(body, now, "")
        check(
            "GetWeeklyTopSellers",
            bool(ts),
            f"{len(ts)} ranks; #1 appid {ts[0]['appid']}"
            if ts
            else f"HTTP {st}, no ranks",
        )

    def t_tags():
        body, st = http.get(
            "https://api.steampowered.com/IStoreService/GetTagList/v1/",
            {**k, "language": "english"},
        )
        n = len(((body or {}).get("response") or {}).get("tags") or [])
        check("GetTagList", n > 0, f"HTTP {st}; {n} tags")

    def t_featured():
        body, st = http.get(
            "https://store.steampowered.com/api/featuredcategories/",
            {"cc": "US", "l": "english"},
        )
        fr = P.featured_rows(body, now, "featuredcategories")
        check(
            "featuredcategories",
            bool(fr),
            f"{len(fr)} featured items in {len({r['section'] for r in fr})} sections",
        )

    def t_capsule():
        r = next(
            (r for r in rows if r.get("asset_header") and r.get("asset_url_format")),
            None,
        )
        if not r:
            check("capsule image download", False, "no asset info in store snapshot")
            return
        img, st = http.get(
            CDN + r["asset_url_format"].replace("${FILENAME}", r["asset_header"]),
            binary=True,
            save_raw=False,
        )
        check(
            "capsule image download",
            st == 200 and bool(img),
            f"HTTP {st}; {len(img or b'')} bytes",
        )

    def t_twitch():
        from jobs import _twitch_token

        h = {
            "Client-Id": S.TWITCH_CLIENT_ID,
            "Authorization": f"Bearer {_twitch_token(http)}",
        }
        body, st = http.get(
            "https://api.twitch.tv/helix/streams",
            {"first": 5},
            headers=h,
            save_raw=False,
        )
        d = (body or {}).get("data") or []
        check(
            "Twitch streams",
            st == 200 and bool(d),
            ", ".join(f"{x.get('game_name')}: {x.get('viewer_count')}" for x in d)
            or f"HTTP {st}",
        )

    tests = [
        ("GetItems (daily store snapshot)", t_items),
        ("GetItems (regional price, DE)", t_regional),
        ("GetNumberOfCurrentPlayers", t_ccu),
    ]
    if S.STEAM_API_KEY:
        tests += [
            ("GetAppList", t_applist),
            ("GetMostPlayedGames", t_most_played),
            ("GetGamesByConcurrentPlayers", t_concurrent),
            ("GetWeeklyTopSellers", t_topsellers),
            ("GetTagList", t_tags),
        ]
    tests += [
        ("featuredcategories", t_featured),
        ("capsule image download", t_capsule),
        ("Store search: Global Top Sellers", t_ranks),
    ]
    if S.TWITCH_CLIENT_ID and S.TWITCH_CLIENT_SECRET:
        tests.append(("Twitch streams", t_twitch))
    else:
        print("[SKIP] Twitch: keys not set (optional)")
    for name, fn in tests:
        attempt(name, fn)
    http.close()
    failed = [n for n, ok in results if not ok]
    print(
        f"\n{len(results) - len(failed)}/{len(results)} passed. Raw responses: {http.raw_path}"
    )
    if failed:
        print(
            "Fix failures before scheduling. Send me the output above and I can adjust the parsers."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--group", choices=list(S.GROUPS))
    r.add_argument("--job", choices=list(JOBS))
    r.add_argument("--force", action="store_true")
    for name in ("selftest", "status", "health"):
        sub.add_parser(name)
    a = ap.parse_args()
    if a.cmd == "run" and not (a.group or a.job):
        ap.error("run needs --group or --job")
    cmds = {
        "run": cmd_run,
        "selftest": cmd_selftest,
        "status": cmd_status,
        "health": cmd_health,
    }
    rc = cmds[a.cmd](a)
    sys.exit(rc if isinstance(rc, int) else 0)
