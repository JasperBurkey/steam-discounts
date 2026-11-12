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

Outputs:
  data/events.parquet        the table used by 07_dml.py
  data/events_daily.parquet  daily reviews from day -21 to +28, used by 05b

  python 04b_build_events.py
"""

import config
import numpy as np
import pandas as pd
from prices import load_history, sale_times, window_state

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


def main():
    panel = pd.read_parquet(config.PANEL_PARQUET)
    panel["day"] = pd.to_datetime(panel["day"])
    hist = load_history()
    hist_by_app = {a: g for a, g in hist.groupby("app_id")}
    released = release_dates()
    first_review_day = panel.groupby("app_id")["day"].min()

    sales = pd.read_csv(config.SALES_CSV)
    sales = sales[sales["verified"].astype(str).str.lower() == "yes"].dropna(
        subset=["start", "end"]
    )
    last_day = panel["day"].max()
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
        n_sale_days = (pd.Timestamp(s["end"]) - d0).days
        need = pd.date_range(
            d0 - config.LOOKBACK_DAYS * DAY,
            d0 + (n_sale_days + config.POST_DAYS - 1) * DAY,
        )
        need_long = pd.date_range(
            d0 - config.LOOKBACK_DAYS * DAY,
            d0 + (n_sale_days + config.LONG_POST_DAYS - 1) * DAY,
        )
        for app_id in wide.columns:
            r_long = wide[app_id].reindex(need_long)
            r = r_long.reindex(need)
            row = {
                "app_id": int(app_id),
                "sale": s["sale"],
                "season": s["season"],
                "event_start": d0,
                "sale_days": n_sale_days,
            }

            # ---- treatment from IsThereAnyDeal ----
            h = hist_by_app.get(app_id)
            st = window_state(h, start_utc, end_utc) if h is not None else None
            pre = (
                window_state(
                    h,
                    start_utc - config.PRE_DISCOUNT_DAYS * DAY,
                    start_utc - pd.Timedelta(hours=2),
                )
                if h is not None
                else None
            )
            row["price_known"] = st is not None
            row["discounted"] = bool(st and st["max_cut"] > 0)
            row["cut"] = st["max_cut"] if st else np.nan
            row["regular_usd"] = st["regular_cents"] / 100 if st else np.nan
            row["pre_discount"] = bool(pre and pre["max_cut"] > 0)
            pre_strict = (
                window_state(
                    h,
                    start_utc - config.PRE_DISCOUNT_DAYS_STRICT * DAY,
                    start_utc - pd.Timedelta(hours=2),
                )
                if h is not None
                else None
            )
            row["pre_discount_strict"] = bool(pre_strict and pre_strict["max_cut"] > 0)
            last_end = (
                last_discount_end(h, start_utc - pd.Timedelta(hours=2))
                if h is not None
                else None
            )
            row["days_since_discount"] = (
                (start_utc - last_end).days if last_end is not None else np.nan
            )
            row["never_discounted_before"] = last_end is None

            # ---- coverage and controls (pre-event only) ----
            row["coverage_ok"] = bool(r.notna().all())
            if row["coverage_ok"]:
                rel = (r.index - d0).days
                rv = r.to_numpy(dtype=float)
                pv = pos[app_id].reindex(need).to_numpy(dtype=float)
                pre28 = rv[(rel >= -28) & (rel <= -1)].mean()
                pre60 = rv[(rel >= -60) & (rel <= -1)].mean()
                old = rv[(rel >= -60) & (rel <= -29)].mean()
                n60 = rv[(rel >= -60) & (rel <= -1)].sum()
                base = rv[(rel >= -21) & (rel <= -8)].mean()
                during = rv[(rel >= 0) & (rel < n_sale_days)].mean()
                after = rv[
                    (rel >= n_sale_days) & (rel < n_sale_days + config.POST_DAYS)
                ].mean()
                pre_week = rv[(rel >= -7) & (rel <= -1)].mean()

                # Outcomes compare 14-day-equivalent totals on a log(1 + x) scale. Using
                # totals rather than daily averages keeps the "+1" from swamping the change
                # for games with only one or two reviews a day.
                def t(daily_mean):
                    return np.log1p(14 * daily_mean)

                row.update(
                    pre_rate_28=pre28,
                    pre_rate_60=pre60,
                    pre_trend=np.log1p(28 * pre28) - np.log1p(28 * old),
                    pct_pos_60=(pv[(rel >= -60) & (rel <= -1)].sum() / n60)
                    if n60 > 0
                    else np.nan,
                    base_rate=base,
                    y_during=t(during) - t(base),
                    y_after=t(after) - t(base),
                    y_pre_week=t(pre_week) - t(base),
                )

            # ---- longer follow-up: net effect and later weeks (only if covered) ----
            row["coverage_long"] = bool(row["coverage_ok"] and r_long.notna().all())
            if row["coverage_long"]:
                rel_l = (r_long.index - d0).days
                rl = r_long.to_numpy(dtype=float)
                n = n_sale_days
                cum = rl[
                    (rel_l >= -config.CUM_PRE_DAYS) & (rel_l < n + config.CUM_POST_DAYS)
                ].mean()
                wk34 = rl[(rel_l >= n + 14) & (rel_l < n + 28)].mean()
                wk56 = rl[(rel_l >= n + 28) & (rel_l < n + 42)].mean()
                row.update(
                    y_net=t(cum) - t(base),
                    y_wk3_4=t(wk34) - t(base),
                    y_wk5_6=t(wk56) - t(base),
                )
            if row["coverage_ok"]:
                src = r_long if row["coverage_long"] else r
                for k, v in zip((src.index - d0).days, src.to_numpy(dtype=float)):
                    if lo <= k <= hi and not np.isnan(v):
                        daily.append((int(app_id), s["sale"], int(k), v))

            # ---- game age ----
            rel_date = released.get(int(app_id))
            if rel_date is not None:
                row["age_days"] = (start_utc - rel_date).days
                row["age_source"] = "collector"
            else:
                row["age_days"] = (d0 - first_review_day[app_id]).days
                row["age_source"] = "first_review (lower bound)"
            rows.append(row)

    ev = pd.DataFrame(rows)
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
    strict_n = (ev["analysis_sample"] & ev["pre_discount_strict"]).sum()
    print(
        f"  (robustness: a {config.PRE_DISCOUNT_DAYS_STRICT}-day window would drop another {strict_n:,} "
        f"from the analysis sample)"
    )
    a = ev[ev["analysis_sample"]]
    print(
        f"Analysis sample: {len(a):,} game-events, {a['app_id'].nunique()} games "
        f"({a['coverage_long'].sum():,} with {config.LONG_POST_DAYS} days of follow-up for the "
        f"long-window outcomes)\n"
    )
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
