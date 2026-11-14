"""Step 4b: build the analysis dataset, one row per game x seasonal sale event.

For each study game and each finished sale event in sale_calendar.csv it records:

  Treatment   discounted during the event (yes/no) and the deepest cut, from IsThereAnyDeal
  Flags       discounted shortly before the event (7 days for the main analysis, 30 days
              for the stricter robustness check), price unknown,
              too little review history, too new, free game
  Controls    measured only from information available BEFORE the event:
              review rate (last 28 and 60 days), trend, % positive, game age,
              regular price, days since the game's last discount, event and season
  Outcomes    log change in reviews (as 14-day-equivalent totals, log(1 + x)) versus the
              pre-event baseline (days -21 to -8): during the sale, in the 14 days after
              it, a pre-period check (days -7 to -1), and, where 6 weeks of follow-up
              exist, the net effect (7 days before through 4 weeks after) and weeks 3-4
              and 5-6 after the sale ends

Two kinds of games feed it (see 03_pull_reviews.py):
  paged    every review is downloaded; window totals are summed from the daily panel
  counted  Steam's exact counts per window; no daily detail
Both go through the SAME window totals and the SAME formulas, so their outcomes are
directly comparable (tests/test_counted_paged.py checks this).

Outputs:
  data/events.parquet        the table used by 07_dml.py (with band, weight and method)
  data/events_daily.parquet  daily reviews from day -21 to +56 (paged games only), used by 05b

  python 04b_build_events.py
"""

from importlib import import_module

import config
import numpy as np
import pandas as pd
from prices import load_history, price_arrays, sale_times, window_state

_pull = import_module("03_pull_reviews")
WINDOWS = _pull.WINDOWS  # the windows Steam was asked to count
day_bounds = _pull.day_bounds  # Pacific calendar days -> unix bounds
DEPTH_BINS = [(0, 0, "none"), (1, 33, "1-33%"), (34, 60, "34-60%"), (61, 100, "61%+")]

DAY = pd.Timedelta(days=1)


def release_dates():
    """Release date per app from the collector's latest store snapshot, if available."""
    base = config.COLLECTOR_DATA / "tables" / "store_items"
    files = (
        sorted(base.glob("date=*/*.parquet"), key=lambda f: f.stem)
        if base.exists()
        else []
    )
    if not files:
        print(
            f"Note: no collector data at {base}; game age will be approximated "
            f"from the first review in the data."
        )
        return {}
    items = pd.read_parquet(files[-1], columns=["appid", "steam_release_date"])
    items = items.dropna(subset=["steam_release_date"])
    rel = pd.to_datetime(items["steam_release_date"], utc=True)
    print(
        f"Release dates from collector snapshot {files[-1].parent.name}: {len(items):,} apps"
    )
    return dict(zip(items["appid"].astype(int), rel))


def last_discount_end(h, before):
    """When the game's most recent discount ended before `before` (None if never)."""
    h = h[h["ts"] < before]
    ends = h[(h["cut"] == 0) & (h["cut"].shift(1) > 0)]
    return ends["ts"].iloc[-1] if len(ends) else None


def treatment(h, start_utc, end_utc):
    """Discount status around one sale event from a game's price history (None = unknown)."""
    two_h = pd.Timedelta(hours=2)
    st = window_state(h, start_utc, end_utc) if h is not None else None
    pre = (
        window_state(h, start_utc - config.PRE_DISCOUNT_DAYS * DAY, start_utc - two_h)
        if h is not None
        else None
    )
    pre_strict = (
        window_state(
            h, start_utc - config.PRE_DISCOUNT_DAYS_STRICT * DAY, start_utc - two_h
        )
        if h is not None
        else None
    )
    last_end = last_discount_end(h, start_utc - two_h) if h is not None else None
    return {
        "price_known": st is not None,
        "discounted": bool(st and st["max_cut"] > 0),
        "cut": st["max_cut"] if st else np.nan,
        "regular_usd": st["regular_cents"] / 100 if st else np.nan,
        "pre_discount": bool(pre and pre["max_cut"] > 0),
        "pre_discount_strict": bool(pre_strict and pre_strict["max_cut"] > 0),
        "days_since_discount": (start_utc - last_end).days
        if last_end is not None
        else np.nan,
        "never_discounted_before": last_end is None,
    }


def window_lengths(n):
    return {w: f(n)[1] - f(n)[0] + 1 for w, f in WINDOWS.items()}


def outcomes(sums, n, ratios=None):
    """Controls and outcomes from window totals (reviews per window; POS = positive in A-D).

    Outcomes compare 14-day-equivalent totals on a log(1 + x) scale; using totals rather
    than daily averages keeps the "+1" from swamping the change for low-volume games."""
    L = window_lengths(n)

    def rate(*ws):
        return sum(sums[w] for w in ws) / sum(L[w] for w in ws)

    def t(daily_mean):
        return np.log1p(14 * daily_mean)

    out = {"coverage_ok": all(w in sums for w in "ABCDEF") and "POS" in sums}
    out["coverage_long"] = out["coverage_ok"] and all(w in sums for w in "GH")
    if not out["coverage_ok"]:
        return out
    pre28, pre60, old, base = (
        rate("B", "C", "D"),
        rate("A", "B", "C", "D"),
        rate("A"),
        rate("C"),
    )
    n60 = sums["A"] + sums["B"] + sums["C"] + sums["D"]
    out.update(
        pre_rate_28=pre28,
        pre_rate_60=pre60,
        pre_trend=np.log1p(28 * pre28) - np.log1p(28 * old),
        pct_pos_60=sums["POS"] / n60 if n60 > 0 else np.nan,
        base_rate=base,
        y_during=t(rate("E")) - t(base),
        y_after=t(rate("F")) - t(base),
        y_pre_week=t(rate("D")) - t(base),
    )
    # Level outcomes (14-day-equivalent counts, no "+1") for the ratio estimator, which
    # measures multiplicative effects without the log(1 + x) compression of small counts.
    out.update(lvl_base=14 * base, lvl_during=14 * rate("E"))
    if out["coverage_long"]:
        out.update(
            y_net=t(rate("D", "E", "F", "G")) - t(base),
            y_wk3_4=t(rate("G")) - t(base),
            y_wk5_6=t(rate("H")) - t(base),
            lvl_net=14 * rate("D", "E", "F", "G"),
        )

    # Revenue proxy: each window's reviews valued at that window's average price ratio
    # (price paid / regular price), assuming reviews move in proportion to units sold.
    # The ratio is applied inside log(1 + x), so the formula stays exact for small games.
    if ratios is not None and all(np.isfinite(ratios.get(w, np.nan)) for w in "CDEF"):

        def rev_rate(*ws):
            return sum(sums[w] * ratios[w] for w in ws) / sum(L[w] for w in ws)

        rbase = rev_rate("C")
        out.update(
            r_during=t(rev_rate("E")) - t(rbase),
            r_after=t(rev_rate("F")) - t(rbase),
            price_ratio_sale=ratios["E"],
            lvl_r_base=14 * rbase,
            lvl_r_during=14 * rev_rate("E"),
        )
        if out["coverage_long"] and np.isfinite(ratios.get("G", np.nan)):
            out["r_net"] = t(rev_rate("D", "E", "F", "G")) - t(rbase)
            out["lvl_r_net"] = 14 * rev_rate("D", "E", "F", "G")
            # Review-lag bound: reviews in the 14 days after the sale valued at the SALE price
            # (as if all came from sale purchases). Over-corrects, so r_net and r_net_lag
            # bracket the truth for discounted games.
            lag = dict(ratios, F=ratios["E"])
            lag_rate = sum(sums[w] * lag[w] for w in "DEFG") / sum(L[w] for w in "DEFG")
            out["r_net_lag"] = t(lag_rate) - t(rbase)
    return out


def paged_sums(r, pos_r, d0, n):
    """Window totals from a paged game's daily series (NaN = day not covered)."""
    rel = (r.index - d0).days
    sums = {}
    for w, f in WINDOWS.items():
        a, b = f(n)
        m = (rel >= a) & (rel <= b)
        vals = (pos_r if w == "POS" else r).to_numpy(dtype=float)[m]
        if m.sum() == b - a + 1 and not np.isnan(vals).any():
            sums[w] = float(vals.sum())
    return sums


def add_depth_bins(ev):
    """Discount depth in four bins, for dose-response and multi-arm policy learning."""
    cut = ev["cut"].fillna(-1)
    ev["depth_bin"] = None
    for lo, hi, label in DEPTH_BINS:
        ev.loc[(cut >= lo) & (cut <= hi), "depth_bin"] = label
    return ev


def add_lagged_features(ev):
    """The game's behavior and response in EARLIER sale events (all realized before this one):
    whether it discounted last time, how deep, its review response during that sale, and the
    share of all earlier sales it joined."""
    ev = ev.sort_values(["app_id", "event_start"]).copy()
    g = ev.groupby("app_id", sort=False)
    ev["prev_discounted"] = g["discounted"].shift(1).astype(float)
    ev["prev_cut"] = g["cut"].shift(1)
    ev["prev_y_during"] = g["y_during"].shift(1) if "y_during" in ev else np.nan
    ev["prev_share_joined"] = g["discounted"].transform(
        lambda x: x.astype(float).shift(1).expanding().mean()
    )
    return ev.sort_values(["event_start", "app_id"]).reset_index(drop=True)


def publishers():
    """First listed publisher per app, from the collector's latest store snapshot."""
    import json

    base = config.COLLECTOR_DATA / "tables" / "store_items"
    files = (
        sorted(base.glob("date=*/*.parquet"), key=lambda f: f.stem)
        if base.exists()
        else []
    )
    if not files:
        return {}
    items = pd.read_parquet(files[-1], columns=["appid", "publishers"]).dropna()

    def first(js):
        try:
            v = json.loads(js)
            return v[0] if v else None
        except (TypeError, ValueError):
            return None

    return {
        int(a): p for a, p in zip(items["appid"], items["publishers"].map(first)) if p
    }


def add_publisher_habits(ev, arrays, sales):
    """How the game's publisher behaved in the PREVIOUS sale event: the share of its OTHER
    games (any eligible game with a price history, not just the sample) that were discounted,
    and how many games it has. Previous event only, so it is decided before this one."""
    pub = publishers()
    if not pub:
        print(
            "Note: no publisher data in the collector snapshot; publisher habits skipped."
        )
        ev["pub_prev_share"], ev["pub_n_games"] = np.nan, np.nan
        return ev
    by_pub = {}
    for app, p in pub.items():
        if app in arrays:
            by_pub.setdefault(p, []).append(app)
    order = sales.assign(_s=pd.to_datetime(sales["start"])).sort_values("_s")
    status = {}  # sale -> {app: discounted (0/1)}
    for _, s in order.iterrows():
        a, b = (int(x.timestamp()) for x in sale_times(s))
        status[s["sale"]] = {
            app: float(arrays[app].max_cut(a, b) > 0)
            for apps in by_pub.values()
            for app in apps
        }
    prev_sale = dict(zip(order["sale"].iloc[1:], order["sale"].iloc[:-1]))
    share, n_games = [], []
    for app, sale in zip(ev["app_id"], ev["sale"]):
        p = pub.get(int(app))
        others = [o for o in by_pub.get(p, []) if o != int(app)]
        n_games.append(len(others) + 1 if p else np.nan)
        prev = prev_sale.get(sale)
        if not p or not others or prev is None:
            share.append(np.nan)
            continue
        vals = [status[prev][o] for o in others if np.isfinite(status[prev][o])]
        share.append(float(np.mean(vals)) if vals else np.nan)
    ev["pub_prev_share"], ev["pub_n_games"] = share, n_games
    return ev


def main():
    sample = (
        pd.read_csv(config.STUDY_SAMPLE_CSV)
        if config.STUDY_SAMPLE_CSV.exists()
        else None
    )

    panel = (
        pd.read_parquet(config.PANEL_PARQUET)
        if config.PANEL_PARQUET.exists()
        else pd.DataFrame(columns=["app_id", "day", "reviews", "positive"])
    )
    panel["day"] = pd.to_datetime(panel["day"])
    counted = {
        int(p.stem): pd.read_parquet(p)
        for p in sorted(config.COUNTS_DIR.glob("*.parquet"))
    }
    if sample is not None:
        keep = set(sample["app_id"].astype(int))
        panel = panel[panel["app_id"].isin(keep)]
        counted = {a: c for a, c in counted.items() if a in keep}
    print(
        f"Paged games: {panel['app_id'].nunique():,}; counted games: {len(counted):,}"
    )

    hist = load_history()
    hist_by_app = {a: g for a, g in hist.groupby("app_id")}
    arrays = price_arrays(hist)  # every game with a price history
    released = release_dates()
    first_review_day = panel.groupby("app_id")["day"].min()

    sales = pd.read_csv(config.SALES_CSV)
    sales = sales[sales["verified"].astype(str).str.lower() == "yes"].dropna(
        subset=["start", "end"]
    )
    last_day = (
        panel["day"].max() if len(panel) else pd.Timestamp.now().normalize() - DAY
    )
    sales = sales[
        pd.to_datetime(sales["end"]) + pd.Timedelta(days=config.POST_DAYS) <= last_day
    ]
    print(
        f"{len(sales)} finished sale events with {config.POST_DAYS} days of follow-up"
    )

    wide = panel.pivot(
        index="day", columns="app_id", values="reviews"
    )  # NaN = no coverage
    pos = panel.pivot(index="day", columns="app_id", values="positive")
    lo, hi = config.WINDOW_LONG

    rows, daily = [], []
    for _, s in sales.iterrows():
        start_utc, end_utc = sale_times(s)
        d0 = pd.Timestamp(s["start"])  # day 0 = first sale day (Pacific)
        n = (pd.Timestamp(s["end"]) - d0).days
        span = pd.date_range(
            d0 - config.LOOKBACK_DAYS * DAY, d0 + (n + config.LONG_POST_DAYS - 1) * DAY
        )
        print(f"  {s['sale']}", flush=True)

        win_bounds = {
            w: day_bounds(d0, *f(n)) for w, f in WINDOWS.items() if w != "POS"
        }
        t0 = int(start_utc.timestamp())

        def base_row(app_id):
            """Treatment, discount history and age: everything that does not depend on reviews."""
            row = {
                "app_id": int(app_id),
                "sale": s["sale"],
                "season": s["season"],
                "event_start": d0,
                "sale_days": n,
            }
            row.update(treatment(hist_by_app.get(app_id), start_utc, end_utc))
            pa = arrays.get(int(app_id))
            row.update(pa.history_features(t0) if pa is not None else {})
            rel_date = released.get(int(app_id))
            if rel_date is not None:
                row["age_days"], row["age_source"] = (
                    (start_utc - rel_date).days,
                    "collector",
                )
            elif app_id in first_review_day.index:
                row["age_days"] = (d0 - first_review_day[app_id]).days
                row["age_source"] = "first_review (lower bound)"
            else:
                row["age_days"], row["age_source"] = np.nan, "unknown"
            return row

        def ratios_for(app_id):
            """Average price ratio (paid / regular) in each review window."""
            pa = arrays.get(int(app_id))
            return (
                {w: pa.mean_ratio(a, b) for w, (a, b) in win_bounds.items()}
                if pa is not None
                else None
            )

        for app_id in wide.columns:
            r = wide[app_id].reindex(span)
            row = base_row(app_id)
            row["method"] = "page"
            row.update(
                outcomes(
                    paged_sums(r, pos[app_id].reindex(span), d0, n),
                    n,
                    ratios_for(app_id),
                )
            )
            if row["coverage_ok"]:
                rel = (r.index - d0).days
                for k, v in zip(rel, r.to_numpy(dtype=float)):
                    if lo <= k <= hi and not np.isnan(v):
                        daily.append((int(app_id), s["sale"], int(k), v))
            rows.append(row)

        for app_id, c in counted.items():
            c = c[c["sale"] == s["sale"]]
            row = base_row(app_id)
            row["method"] = "count"
            row.update(
                outcomes(
                    dict(zip(c["window"], c["reviews"].astype(float))),
                    n,
                    ratios_for(app_id),
                )
            )
            rows.append(row)

    ev = pd.DataFrame(rows)
    ev = add_depth_bins(ev)
    ev = add_lagged_features(ev)
    ev = add_publisher_habits(ev, arrays, sales)
    if sample is not None:
        ev = ev.merge(sample[["app_id", "band", "weight"]], on="app_id", how="left")
    ev["too_new"] = ev["age_days"] < config.MIN_AGE_DAYS
    ev["free"] = ev["regular_usd"].fillna(0) <= 0
    ev["analysis_sample"] = (
        ev["coverage_ok"]
        & ev["price_known"]
        & ~ev["too_new"]
        & ~ev["free"]
        & ~ev["pre_discount"]
    )
    ev.to_parquet(config.EVENTS_PARQUET, index=False)
    pd.DataFrame(daily, columns=["app_id", "sale", "rel_day", "reviews"]).to_parquet(
        config.EVENTS_DAILY_PARQUET, index=False
    )

    # ---- summary ----
    print(
        f"\n{len(ev):,} game x event rows ({ev['app_id'].nunique()} games x {ev['sale'].nunique()} events)"
    )
    print("Excluded from the main analysis (a row can have several reasons):")
    for col, label in [
        ("coverage_ok", "too little review history around the event"),
        ("price_known", "price unknown (no IsThereAnyDeal record yet)"),
    ]:
        print(f"  {label:<48} {(~ev[col]).sum():>6,}")
    for col, label in [
        ("too_new", f"released under {config.MIN_AGE_DAYS} days before"),
        ("free", "free or no regular price"),
        ("pre_discount", f"discounted in the {config.PRE_DISCOUNT_DAYS} days before"),
    ]:
        print(f"  {label:<48} {ev[col].sum():>6,}")
    a = ev[ev["analysis_sample"]]
    print(
        f"Analysis sample: {len(a):,} game-events, {a['app_id'].nunique()} games "
        f"({a['coverage_long'].sum():,} with {config.LONG_POST_DAYS} days of follow-up)\n"
    )
    if "band" in a:
        by_band = a.groupby("band", sort=False).agg(
            game_events=("app_id", "size"),
            games=("app_id", "nunique"),
            discounted=("discounted", "mean"),
        )
        by_band["discounted"] = (by_band["discounted"] * 100).round(0).astype(
            int
        ).astype(str) + "%"
        print(by_band.to_string(), "\n")
    summ = a.groupby("sale", sort=False).agg(
        games=("app_id", "size"),
        discounted=("discounted", "mean"),
        median_cut=("cut", lambda c: c[c > 0].median()),
    )
    summ["discounted"] = (summ["discounted"] * 100).round(0).astype(int).astype(
        str
    ) + "%"
    print(summ.to_string())
    print(f"\nWrote {config.EVENTS_PARQUET} and {config.EVENTS_DAILY_PARQUET}")


if __name__ == "__main__":
    main()
