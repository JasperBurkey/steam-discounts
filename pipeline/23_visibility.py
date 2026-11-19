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
   If a better label raises demand, more reviews from a sale have a lasting payoff.

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
        if len(g) < 50:
            continue
        cross = int(g["day"].iloc[49])  # day of the 50th purchaser review
        pos = g["voted_up"].iloc[:50].fillna(False).astype(bool).mean()
        if cross - 28 < 0 or cross + 28 > last_day:
            continue
        dd = all_days[a]
        c = lambda lo, hi, dd=dd: (  # noqa: E731
            np.searchsorted(dd, hi, "right") - np.searchsorted(dd, lo, "left")
        )
        before = c(cross - 28, cross - 1)
        after = c(cross + 1, cross + 28)
        rows.append(
            dict(
                app_id=a,
                pct_pos=pos,
                cross_day=cross,
                days_since_release=cross - (rel[a] - day0).days,
                y=np.log1p(after) - np.log1p(before),
            )
        )
    c = pd.DataFrame(rows)
    c["group"] = pd.cut(
        c["pct_pos"],
        [0, 0.6999, 0.7999, 1.0001],
        labels=[
            "under 70%",
            "70-79% (no label change)",
            "80%+ (Positive -> Very Positive)",
        ],
    )
    c.to_csv(OUT / "label_crossings.csv", index=False)
    out = []
    summ = c.groupby("group", observed=True)["y"].agg(["mean", "std", "size"])
    for k, s in summ.iterrows():
        out.append(
            dict(
                test="C",
                sample=str(k),
                outcome="log reviews 28d after - 28d before crossing 50",
                est=s["mean"],
                se=s["std"] / np.sqrt(s["size"]),
                n=int(s["size"]),
            )
        )
    # DiD, with a local comparison (75-79% vs 80-84%) and a control for time since release
    for lab, lo_, hi_ in [
        ("all", (0.70, 0.80), (0.80, 1.01)),
        ("near 80%", (0.75, 0.80), (0.80, 0.85)),
    ]:
        m = c[(c["pct_pos"] >= lo_[0]) & (c["pct_pos"] < hi_[1])].copy()
        m["T"] = (m["pct_pos"] >= 0.80).astype(float)
        X = np.column_stack(
            [np.ones(len(m)), m["T"], np.log1p(m["days_since_release"].clip(lower=0))]
        )
        y = m["y"].to_numpy()
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        u = y - X @ b
        XtX = np.linalg.inv(X.T @ X)
        V = XtX @ (X.T * u**2) @ X @ XtX
        out.append(
            dict(
                test="C",
                sample=f"label upgrade vs none ({lab})",
                outcome="difference, controlling for time since release",
                est=b[1],
                se=float(np.sqrt(V[1, 1])),
                n=len(m),
            )
        )
    print(f"  C done  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = part_c(t0)
    pd.DataFrame(rows).to_csv(OUT / "visibility.csv", index=False)
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
