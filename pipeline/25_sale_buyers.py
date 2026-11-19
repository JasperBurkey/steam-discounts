"""Step 25: are sale buyers different? Ratings and playtime of reviews after a discount.

For each game-event, among reviews from Steam purchasers:
  pos     share positive in the sale and the two weeks after (windows E+F) minus the share
          in the 60 days before (A-D)
  play    median log(1 + playtime at review, minutes) in E+F minus the same before
  fast    share of reviews written with under 2 hours played, E+F minus before
ATT of discounting on each (same doubly robust estimator and controls as the main
results), so the comparison is with full-price games over the same weeks. Reported for all
discounts and separately for deep (50%+) and shallow (<50%) cuts.

A negative `pos` means the marginal sale buyer likes the game less (adverse selection on
taste); positive means a deal makes people happier or the extra buyers are fans who waited.
Lower `play` means sale buyers review after less play, consistent with impulse purchases
(and with the fast reviews in 16_review_flags). Requires 5+ purchaser reviews in both
periods; games with complete review histories only.

  python 25_sale_buyers.py
Outputs: results/sale_buyers/
"""

import importlib
import time

import config
import dml_core as M
import numpy as np
import pandas as pd

F16 = importlib.import_module("16_review_flags")
OUT = config.ROOT.parent / "results" / "sale_buyers"
COMPACT = config.DATA / "for_claude"
MIN_N = 5
OUTCOMES = {
    "pos": "share positive",
    "play": "median log playtime at review",
    "fast": "share reviewing with <2h played",
}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rv = pd.read_parquet(
        COMPACT / "reviews_compact.parquet",
        columns=[
            "app_id",
            "day",
            "voted_up",
            "steam_purchase",
            "playtime_min",
            "sampled",
        ],
    )
    samp = rv.groupby("app_id")["sampled"].any()
    rv = rv[
        rv["app_id"].isin(samp[~samp].index)
        & rv["steam_purchase"].fillna(False).astype(bool)
        & (rv["day"] >= 0)
    ].sort_values(["app_id", "day"])
    rv["pos"] = rv["voted_up"].fillna(False).astype(float)
    rv["lp"] = np.log1p(rv["playtime_min"].astype(float))
    rv["fast"] = (rv["playtime_min"].astype(float) < 120).astype(float)
    last_day = int(rv["day"].max()) - 2
    by = {
        a: (g["day"].to_numpy(), g[["pos", "lp", "fast"]].to_numpy())
        for a, g in rv.groupby("app_id")
    }
    day0 = pd.Timestamp(config.REVIEW_HISTORY_START)
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    ev = ev[ev["analysis_sample"] & ev["app_id"].isin(by)].dropna(
        subset=["y_during", "pre_trend"]
    )
    rows = []
    for r in ev.itertuples(index=False):
        n = int(r.sale_days)
        d0 = (pd.Timestamp(r.event_start).normalize() - day0).days
        if d0 - 60 < 0 or d0 + n + 13 > last_day:
            continue
        days, v = by[r.app_id]
        pre = v[
            np.searchsorted(days, d0 - 60, "left") : np.searchsorted(
                days, d0 - 1, "right"
            )
        ]
        post = v[
            np.searchsorted(days, d0, "left") : np.searchsorted(
                days, d0 + n + 13, "right"
            )
        ]
        pre_lp, post_lp = pre[:, 1], post[:, 1]
        pre_lp, post_lp = pre_lp[np.isfinite(pre_lp)], post_lp[np.isfinite(post_lp)]
        if len(pre_lp) < MIN_N or len(post_lp) < MIN_N:
            continue
        rows.append(
            {
                "app_id": r.app_id,
                "sale": r.sale,
                "pos": post[:, 0].mean() - pre[:, 0].mean(),
                "play": np.median(post_lp) - np.median(pre_lp),
                "fast": post[:, 2].mean() - pre[:, 2].mean(),
                "n_post": len(post),
            }
        )
    o = ev.merge(pd.DataFrame(rows), on=["app_id", "sale"]).reset_index(drop=True)
    print(
        f"{len(o):,} game-events ({o['discounted'].mean():.0%} discounted), "
        f"{o['app_id'].nunique():,} games  [{time.time() - t0:.0f}s]",
        flush=True,
    )
    res = []
    samples = {
        "all discounts": o,
        "deep (50%+) vs full price": o[(o["discounted"] == 0) | (o["cut"] >= 50)],
        "shallow (<50%) vs full price": o[(o["discounted"] == 0) | (o["cut"] < 50)],
    }
    for sname, d in samples.items():
        r = M.aipw(d.reset_index(drop=True), list(OUTCOMES))[0]
        for y, lab in OUTCOMES.items():
            est, se = r[y]["ATT"]
            raw = d.groupby(d["discounted"].astype(int))[y].mean()
            res.append(
                {
                    "sample": sname,
                    "outcome": lab,
                    "ATT": est,
                    "se": se,
                    "mean change, full price": raw.get(0, np.nan),
                    "mean change, discounted": raw.get(1, np.nan),
                    "n": len(d),
                }
            )
        print(f"  {sname} done  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    res = pd.DataFrame(res)
    res.to_csv(OUT / "sale_buyers.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
