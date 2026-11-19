"""Step 23: visibility benefits the four-week net window could miss.

If a sale pays off later (wishlists, chart momentum, followers, better review labels),
always joining could be rational even with a negative four-week net. Three tests:

A. Longer horizon. ATT on log reviews in weeks 5-6 after the sale (from 04b) and in weeks
   7-12 after (from reviews_compact, games with full histories). Weeks 7-12 use only the
   Spring and Summer sales, whose next seasonal sale starts more than 12 weeks later;
   other sales would mix in the next sale.

B. The next sale. Among game-events that discount in the next seasonal sale, does having
   discounted in this one raise or lower the next sale's reviews (relative to this sale's
   baseline)? Visibility (wishlists, followers, reviews) says raise; a depleted pool of
   waiting buyers says lower. Conditioning on next-sale discounting is descriptive (it is
   a later choice), so read it as "for games that rejoin."

C. Review labels. Steam's summary label depends on the share positive AND the count of
   reviews from Steam purchasers: with 80%+ positive, a game is "Positive" with 10-49
   reviews and "Very Positive" from 50; with 70-79% it stays "Mostly Positive" either way.
   For games released since review history starts (full cumulative counts), compare the
   review rate in the 28 days after vs before crossing 50 purchaser reviews, for games at
   80%+ (label improves) vs 70-79% (no change): a difference-in-differences on the label.
   The same at 500 reviews: 95%+ moves from "Very Positive" to "Overwhelmingly Positive",
   80-94% stays "Very Positive". Crossings within 60 days of release are dropped (launch
   burst). If a better label raises demand, more reviews from a sale have a lasting payoff.

  python 23_visibility.py      (LEARNER / CF_REPEATS as in config)
Outputs: results/visibility/
"""

import importlib
import time

import config
import dml_core as M
import numpy as np
import pandas as pd

F16 = importlib.import_module("16_review_flags")
OUT = config.ROOT.parent / "results" / "visibility"
COMPACT = config.DATA / "for_claude"
LONG_SALES = ("Spring", "Summer")


def att(df, ycols):
    res = M.aipw(df.reset_index(drop=True), ycols)[0]
    return {y: res[y]["ATT"] for y in ycols}


def part_a(ev, days, day0, last_day, t0):
    rows = []
    base = ev[ev["analysis_sample"]].dropna(subset=["y_during", "pre_trend"])
    d = base.dropna(subset=["y_wk3_4", "y_wk5_6"])
    r = att(d, ["y_during", "y_wk3_4", "y_wk5_6"])
    for y, lab in [
        ("y_during", "during"),
        ("y_wk3_4", "weeks 3-4 after"),
        ("y_wk5_6", "weeks 5-6 after"),
    ]:
        rows.append(
            dict(
                test="A",
                sample="all sales",
                outcome=lab,
                est=r[y][0],
                se=r[y][1],
                n=len(d),
            )
        )
    print(f"  A: weeks 5-6 done  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    # weeks 7-12 from raw reviews, Spring and Summer sales, full histories
    lo = base[base["sale"].str.startswith(LONG_SALES) & base["app_id"].isin(days)]
    vals = []
    for r_ in lo.itertuples(index=False):
        n = int(r_.sale_days)
        d0 = (pd.Timestamp(r_.event_start).normalize() - day0).days
        if d0 + n + 83 > last_day or d0 - 21 < 0:
            vals.append(np.nan)
            continue
        dd = days[r_.app_id]
        c = lambda a, b, dd=dd: (  # noqa: E731
            np.searchsorted(dd, b, "right") - np.searchsorted(dd, a, "left")
        )
        late = c(d0 + n + 42, d0 + n + 83) / 42
        basec = c(d0 - 21, d0 - 8) / 14
        vals.append(np.log1p(14 * late) - np.log1p(14 * basec))
    lo = lo.assign(y_wk7_12=vals).dropna(subset=["y_wk7_12", "y_wk5_6"])
    r = att(lo, ["y_during", "y_wk5_6", "y_wk7_12"])
    for y, lab in [
        ("y_during", "during"),
        ("y_wk5_6", "weeks 5-6 after"),
        ("y_wk7_12", "weeks 7-12 after"),
    ]:
        rows.append(
            dict(
                test="A",
                sample="Spring + Summer sales, full histories",
                outcome=lab,
                est=r[y][0],
                se=r[y][1],
                n=len(lo),
            )
        )
    print(f"  A: weeks 7-12 done  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    return rows


def part_b(ev, t0):
    ev = ev.sort_values(["app_id", "event_start"]).copy()
    g = ev.groupby("app_id")
    ev["next_disc"] = g["discounted"].shift(-1)
    ev["next_lvl_during"] = g["lvl_during"].shift(-1)
    ev["next_cut"] = g["cut"].shift(-1)
    ev["y_next_sale"] = np.log1p(ev["next_lvl_during"]) - np.log1p(ev["lvl_base"])
    d = ev[ev["analysis_sample"] & (ev["next_disc"] == 1)].dropna(
        subset=["y_next_sale", "y_during", "pre_trend"]
    )
    r = att(d, ["y_during", "y_next_sale"])
    rows = [
        dict(
            test="B",
            sample="games discounting in the next sale",
            outcome=lab,
            est=r[y][0],
            se=r[y][1],
            n=len(d),
        )
        for y, lab in [
            ("y_during", "this sale, during"),
            ("y_next_sale", "next sale, during (vs this sale's baseline)"),
        ]
    ]
    print(f"  B done  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    return rows


THRESHOLDS = [  # (count, upgrade range, comparison range, label change)
    (50, (0.80, 1.01), (0.70, 0.80), "Positive -> Very Positive"),
    (500, (0.95, 1.01), (0.80, 0.95), "Very Positive -> Overwhelmingly Positive"),
]
MIN_DAYS_AFTER_RELEASE = 60  # both 28-day windows after the launch burst


def part_c(t0):
    rv = pd.read_parquet(
        COMPACT / "reviews_compact.parquet",
        columns=["app_id", "day", "voted_up", "steam_purchase", "sampled"],
    )
    gi = pd.read_parquet(
        COMPACT / "game_info.parquet", columns=["appid", "steam_release_date"]
    )
    rel = pd.to_datetime(
        gi.set_index("appid")["steam_release_date"], utc=True, errors="coerce"
    )
    day0 = pd.Timestamp(config.REVIEW_HISTORY_START, tz="UTC")
    young = rel[rel >= day0 + pd.Timedelta(days=7)].index
    samp = rv.groupby("app_id")["sampled"].any()
    rv = rv[rv["app_id"].isin(young) & rv["app_id"].isin(samp[~samp].index)]
    last_day = int(rv["day"].max()) - 2
    all_days = {a: np.sort(g["day"].to_numpy()) for a, g in rv.groupby("app_id")}
    pur = rv[rv["steam_purchase"].fillna(False).astype(bool)].sort_values(
        ["app_id", "day"]
    )
    rows = []
    for a, g in pur.groupby("app_id"):
        rel_day = (rel[a] - day0).days
        dd = all_days[a]
        for k, up, ctl, _ in THRESHOLDS:
            if len(g) < k:
                continue
            cross = int(g["day"].iloc[k - 1])  # day of the k-th purchaser review
            pos = g["voted_up"].iloc[:k].fillna(False).astype(bool).mean()
            since = cross - rel_day
            if since < MIN_DAYS_AFTER_RELEASE or cross + 28 > last_day:
                continue
            grp = (
                "upgrade"
                if up[0] <= pos < up[1]
                else "comparison"
                if ctl[0] <= pos < ctl[1]
                else None
            )
            if grp is None:
                continue
            before = np.searchsorted(dd, cross - 1, "right") - np.searchsorted(
                dd, cross - 28, "left"
            )
            after = np.searchsorted(dd, cross + 28, "right") - np.searchsorted(
                dd, cross + 1, "left"
            )
            rows.append(
                dict(
                    app_id=a,
                    threshold=k,
                    group=grp,
                    pct_pos=pos,
                    days_since_release=since,
                    before=before,
                    after=after,
                    y=np.log1p(after) - np.log1p(before),
                )
            )
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "label_crossings.csv", index=False)
    out = []
    for k, up, ctl, lab in THRESHOLDS:
        m = c[c["threshold"] == k].copy()
        for grp, d in m.groupby("group"):
            out.append(
                dict(
                    threshold=k,
                    row=f"{grp} ({'label: ' + lab if grp == 'upgrade' else 'no label change'})",
                    est=d["y"].mean(),
                    se=d["y"].std() / np.sqrt(len(d)),
                    n=len(d),
                )
            )
        if m["group"].nunique() < 2:
            continue
        m["T"] = (m["group"] == "upgrade").astype(float)
        X = np.column_stack(
            [
                np.ones(len(m)),
                m["T"],
                np.log(m["days_since_release"]),
                np.log1p(m["before"]),
            ]
        )
        y = m["y"].to_numpy()
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        u = y - X @ b
        XtX = np.linalg.inv(X.T @ X)
        V = XtX @ (X.T * u**2) @ X @ XtX * len(y) / (len(y) - X.shape[1])
        out.append(
            dict(
                threshold=k,
                row="difference (upgrade - comparison), controls: log days since release, log reviews before",
                est=b[1],
                se=float(np.sqrt(V[1, 1])),
                n=len(m),
            )
        )
    out = pd.DataFrame(out)
    out.to_csv(OUT / "labels.csv", index=False)
    print(f"  C done  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    part_c(t0)
    rows = []
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    rv = pd.read_parquet(
        COMPACT / "reviews_compact.parquet", columns=["app_id", "day", "sampled"]
    )
    samp = rv.groupby("app_id")["sampled"].any()
    rv = rv[rv["app_id"].isin(samp[~samp].index) & (rv["day"] >= 0)]
    last_day = int(rv["day"].max()) - 2
    days = {a: np.sort(g["day"].to_numpy()) for a, g in rv.groupby("app_id")}
    day0 = pd.Timestamp(config.REVIEW_HISTORY_START)
    rows += part_b(ev, t0)
    pd.DataFrame(rows).to_csv(OUT / "visibility.csv", index=False)
    rows += part_a(ev, days, day0, last_day, t0)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "visibility.csv", index=False)
    res["est (se)"] = res.apply(lambda x: f"{x['est']:+.3f} ({x['se']:.3f})", axis=1)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(res[["test", "sample", "outcome", "est (se)", "n"]].to_string(index=False))
    print(f"Done in {(time.time() - t0) / 60:.0f} minutes")


if __name__ == "__main__":
    main()
