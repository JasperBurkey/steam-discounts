"""Step 22: the 30-day release rule as a natural experiment (fuzzy regression discontinuity).

Steam does not let a game be discounted for 30 days after release, and seasonal sales are
not exempt (Steamworks documentation, "Discounting"). So a game released 31 days before a
seasonal sale can join it and one released 29 days before cannot. Whether a game may
discount jumps at age 30 days, a cutoff nobody chose for demand reasons.

Running variable  a = age at the sale start in days - 30 (eligible when a >= 0)
First stage       jump in the share discounting at a = 0
Reduced form      jump in outcomes at a = 0
Fuzzy RD          reduced form / first stage: the effect of discounting for games that
                  discount because they are eligible (compliers), at about one month old
Estimation        local linear on each side, triangular kernel, bandwidths 10 / 14 / 21
                  days; a donut version drops a in {-1, 0} (timing of release vs the 10:00
                  PT sale start). SE by bootstrap over games (B = 300).

Samples
  universe   every paid game in game_info with a price history: first stage and density
  outcomes   study games with full (unsampled) review histories: reviews from
             reviews_compact in the sale windows of 04b (D week before, E sale, F weeks
             1-2 after, G weeks 3-4 after)
Outcomes: log(1 + reviews) in E, F, G, and D-G; revenue proxy during and over D-G
(reviews in E weighted by the price paid / regular price, as in 04b).

Checks
  density    are releases bunched just past the cutoff (developers timing launches to be
             eligible)? Local linear on daily counts with day-of-week effects.
  balance    covariates fixed before the cutoff could matter: reviews in the first 7 days
             after release, log regular price, publisher catalog size.
  placebo    the same RD at fake cutoffs (ages 20 and 40), where nothing changes.

  python 22_release_rd.py
Outputs: results/release_rd/
"""

import importlib
import json
import time

import config
import numpy as np
import pandas as pd
from prices import price_arrays, sale_times

F16 = importlib.import_module("16_review_flags")
OUT = config.ROOT.parent / "results" / "release_rd"
COMPACT = config.DATA / "for_claude"
CUT = 30
HMAX = 21
BWS = [10, 14, 21]
B = 300
rng = np.random.default_rng(config.SEED)


def first(s):
    try:
        v = json.loads(s) if isinstance(s, str) else None
    except ValueError:
        return None
    return v[0] if v else None


def llr_jump(a, y, h, donut=False, c=0):
    """Local linear jump at a = c with a triangular kernel; returns (jump, n)."""
    x = a - c
    m = np.abs(x) < h
    if donut:
        m &= ~np.isin(x, [-1, 0])
    x, yy = x[m], y[m]
    ok = np.isfinite(yy)
    x, yy = x[ok], yy[ok]
    t = (x >= 0).astype(float)
    w = 1 - np.abs(x) / h
    X = np.column_stack([np.ones_like(x), t, x, t * x])
    W = np.sqrt(w)
    b, *_ = np.linalg.lstsq(X * W[:, None], yy * W, rcond=None)
    return b[1], len(yy)


def rd(df, ycol, h, donut=False, c=0, fuzzy=True):
    """Jump (sharp) or fuzzy RD estimate with bootstrap SE over games."""
    a = df["a"].to_numpy(float)
    y = df[ycol].to_numpy(float)
    d = df["discounted"].to_numpy(float)
    games = df["app_id"].to_numpy()

    def est(idx):
        ry, n = llr_jump(a[idx], y[idx], h, donut, c)
        if not fuzzy:
            return ry, n
        fs, _ = llr_jump(a[idx], d[idx], h, donut, c)
        return ry / fs, n

    full, n = est(np.arange(len(df)))
    ug, inv = np.unique(games, return_inverse=True)
    members = pd.Series(np.arange(len(df))).groupby(inv).apply(np.asarray).to_list()
    boots = []
    for _ in range(B):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([members[i] for i in pick])
        boots.append(est(idx)[0])
    boots = np.array(boots)
    boots = boots[np.isfinite(boots)]
    return full, float(np.std(boots)), n


def density(cnt, h):
    """Jump in log daily counts at a = 0, local linear, day-of-week dummies, HC1 SE."""
    c = cnt[np.abs(cnt["a"]) < h]
    x = c["a"].to_numpy(float)
    t = (x >= 0).astype(float)
    dow = pd.get_dummies(c["dow"], drop_first=True, dtype=float).to_numpy()
    X = np.column_stack([np.ones_like(x), t, x, t * x, dow])
    y = np.log(c["n"].to_numpy(float))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ b
    XtX = np.linalg.inv(X.T @ X)
    V = XtX @ (X.T * u**2) @ X @ XtX * len(y) / (len(y) - X.shape[1])
    return b[1], float(np.sqrt(V[1, 1]))


def study_outcomes(u, rv):
    """Outcomes for study games with full (unsampled) review histories."""
    samp = rv.groupby("app_id")["sampled"].any()
    rv = rv[rv["app_id"].isin(samp[~samp].index) & (rv["day"] >= 0)]
    last_day = int(rv["day"].max()) - 2
    days = {a: np.sort(g["day"].to_numpy()) for a, g in rv.groupby("app_id")}
    day0 = pd.Timestamp(config.REVIEW_HISTORY_START, tz="UTC")
    out = []
    for r in u[u["app_id"].isin(days)].itertuples(index=False):
        d0 = (r.sale_start.normalize() - day0).days
        n = int(r.sale_days)
        rel0 = (r.rel - day0).days
        if rel0 < 0 or d0 + n + 27 > last_day:
            continue
        dd = days[r.app_id]
        cnt = lambda lo, hi: int(  # noqa: E731
            np.searchsorted(dd, hi, "right") - np.searchsorted(dd, lo, "left")
        )
        D = cnt(d0 - 7, d0 - 1)
        E = cnt(d0, d0 + n - 1)
        F = cnt(d0 + n, d0 + n + 13)
        G = cnt(d0 + n + 14, d0 + n + 27)
        ratio = r.ratio_sale if np.isfinite(r.ratio_sale) else 1.0
        out.append(
            {
                "app_id": r.app_id,
                "sale": r.sale,
                "y_E": np.log1p(E),
                "y_F": np.log1p(F),
                "y_G": np.log1p(G),
                "y_DG": np.log1p(D + E + F + G),
                "r_E": np.log1p(E * ratio),
                "r_DG": np.log1p(D + E * ratio + F + G),
                "early7": np.log1p(cnt(rel0, rel0 + 6)),
            }
        )
    return u.merge(pd.DataFrame(out), on=["app_id", "sale"])


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    gi = pd.read_parquet(COMPACT / "game_info.parquet")
    gi = gi[(gi["type"].astype(str) == "0") & (gi["price_original_cents"] > 0)].copy()
    gi["rel"] = pd.to_datetime(gi["steam_release_date"], utc=True, errors="coerce")
    gi["pub"] = gi["publishers"].map(first)
    gi["catalog"] = gi["pub"].map(gi.groupby("pub").size())
    gi = gi.dropna(subset=["rel"])
    hist = (
        pd.read_parquet(COMPACT / "price_history.parquet")
        .assign(ts=lambda h: pd.to_datetime(h["ts"], utc=True))
        .sort_values(["app_id", "ts"])
    )
    hist = hist[hist["app_id"].isin(gi["appid"])]
    arrays = price_arrays(hist)
    cal = pd.read_csv(config.ROOT / "sale_calendar.csv")
    print(f"{len(arrays):,} paid games with price histories  [{time.time() - t0:.0f}s]")

    rows = []
    rel = gi.set_index("appid")["rel"]
    for _, s in cal.iterrows():
        start, end = sale_times(s)
        age = (start - rel).dt.days
        near = age[(age >= CUT - HMAX - 25) & (age < CUT + HMAX + 25)]
        for app, ag in near.items():
            pa = arrays.get(int(app))
            if pa is None:
                continue
            a0, a1 = int(start.timestamp()), int(end.timestamp())
            cut = pa.max_cut(a0, a1)
            if not np.isfinite(cut):
                continue
            rows.append(
                {
                    "app_id": int(app),
                    "sale": s["sale"],
                    "sale_start": start,
                    "sale_days": (end - start).days,
                    "age": int(ag),
                    "a": int(ag) - CUT,
                    "cut": cut,
                    "discounted": float(cut > 0),
                    "ratio_sale": pa.mean_ratio(a0, a1),
                    "release_dow": rel[app].tz_convert(config.TZ).dayofweek,
                }
            )
    u = pd.DataFrame(rows).merge(
        gi[["appid", "price_original_cents", "catalog", "rel"]],
        left_on="app_id",
        right_on="appid",
    )
    u["log_price"] = np.log(u["price_original_cents"] / 100)
    u["log_catalog"] = np.log(u["catalog"])
    print(
        f"universe: {len(u):,} game-events, {u['app_id'].nunique():,} games  [{time.time() - t0:.0f}s]"
    )

    # list of near-cutoff game-events for 03h_pull_rd_reviews.py
    near = u[np.abs(u["a"]) < HMAX + 5]
    pd.DataFrame(
        {
            "app_id": near["app_id"],
            "sale": near["sale"],
            "sale_date": near["sale_start"].dt.tz_convert(config.TZ).dt.date,
            "sale_days": near["sale_days"],
            "release_date": near["rel"].dt.tz_convert(config.TZ).dt.date,
            "a": near["a"],
        }
    ).to_csv(OUT / "universe.csv", index=False)

    def from_counts(c):
        """Outcomes from 03h window counts (the whole universe)."""
        c = u.merge(c, on=["app_id", "sale"]).dropna(subset=["D", "E", "F", "G"])
        ratio = c["ratio_sale"].fillna(1.0)
        return pd.DataFrame(
            {
                "app_id": c["app_id"],
                "sale": c["sale"],
                "y_E": np.log1p(c["E"]),
                "y_F": np.log1p(c["F"]),
                "y_G": np.log1p(c["G"]),
                "y_DG": np.log1p(c["D"] + c["E"] + c["F"] + c["G"]),
                "r_E": np.log1p(c["E"] * ratio),
                "r_DG": np.log1p(c["D"] + c["E"] * ratio + c["F"] + c["G"]),
                "early7": np.log1p(c["rel7"]),
            }
        )

    counts_file = COMPACT / "rd_review_counts.parquet"
    if counts_file.exists():
        o = u.merge(from_counts(pd.read_parquet(counts_file)), on=["app_id", "sale"])
        print(f"outcomes from {counts_file.name} (whole universe)")
    else:
        cols = ["app_id", "day", "sampled"]
        rv = pd.read_parquet(COMPACT / "reviews_compact.parquet", columns=cols)
        o = study_outcomes(u, rv)
    print(
        f"outcome sample: {len(o):,} game-events, {o['app_id'].nunique():,} games  [{time.time() - t0:.0f}s]"
    )
    res = []
    # first stage (universe and outcome sample)
    for sname, d in [("universe", u), ("outcome sample", o)]:
        for h in BWS:
            for donut in (False, True):
                j, se, n = rd(d, "discounted", h, donut, fuzzy=False)
                res.append(
                    dict(
                        part="first stage",
                        sample=sname,
                        outcome="discounted",
                        bw=h,
                        donut=donut,
                        est=j,
                        se=se,
                        n=n,
                    )
                )
    print(f"first stage done  [{time.time() - t0:.0f}s]", flush=True)
    labels = {
        "y_E": "reviews during sale",
        "y_F": "reviews weeks 1-2 after",
        "y_G": "reviews weeks 3-4 after",
        "y_DG": "reviews week before to week 4",
        "r_E": "revenue proxy during",
        "r_DG": "revenue proxy week before to week 4",
    }
    for y, lab in labels.items():
        for h in BWS:
            for donut in (False, True):
                for part, fz in [("reduced form", False), ("fuzzy RD", True)]:
                    j, se, n = rd(o, y, h, donut, fuzzy=fz)
                    res.append(
                        dict(
                            part=part,
                            sample="outcome sample",
                            outcome=lab,
                            bw=h,
                            donut=donut,
                            est=j,
                            se=se,
                            n=n,
                        )
                    )
        print(f"  {lab} done  [{time.time() - t0:.0f}s]", flush=True)
    # balance
    for y, lab, d in [
        ("early7", "reviews in first 7 days", o),
        ("log_price", "log regular price", u),
        ("log_catalog", "log publisher catalog", u),
    ]:
        for h in BWS:
            j, se, n = rd(d, y, h, fuzzy=False)
            res.append(
                dict(
                    part="balance",
                    sample="universe" if d is u else "outcome sample",
                    outcome=lab,
                    bw=h,
                    donut=False,
                    est=j,
                    se=se,
                    n=n,
                )
            )
    # placebo cutoffs (first stage and reduced form at ages 20 and 40)
    for c in (-10, 10):
        for y, d in [("discounted", u), ("y_E", o), ("r_DG", o)]:
            j, se, n = rd(d, y, 10, c=c, fuzzy=False)
            res.append(
                dict(
                    part=f"placebo cutoff (age {CUT + c})",
                    sample="universe" if d is u else "outcome sample",
                    outcome=labels.get(y, y),
                    bw=10,
                    donut=False,
                    est=j,
                    se=se,
                    n=n,
                )
            )
    res = pd.DataFrame(res)

    # density of releases by age at the sale
    cnt = (
        u.groupby("a")
        .agg(n=("app_id", "size"))
        .reset_index()
        .assign(dow=lambda c: c["a"] % 7)
    )
    dens = []
    for h in (14, 21, 28):
        j, se = density(cnt, h)
        dens.append({"bandwidth": h, "jump in log count": j, "se": se})
    dens = pd.DataFrame(dens)
    cnt.to_csv(OUT / "daily_counts.csv", index=False)
    bins = (
        u.assign(bin=lambda x: (x["a"] // 3) * 3)
        .groupby("bin")
        .agg(
            n=("app_id", "size"),
            share_discounted=("discounted", "mean"),
            mean_cut_if_disc=("cut", lambda c: c[c > 0].mean()),
        )
        .reset_index()
    )
    res.to_csv(OUT / "rd.csv", index=False)
    dens.to_csv(OUT / "density.csv", index=False)
    bins.to_csv(OUT / "bins.csv", index=False)
    o.to_parquet(OUT / "rd_sample.parquet")

    pd.set_option("display.width", 220)
    res["est (se)"] = res.apply(lambda x: f"{x['est']:+.3f} ({x['se']:.3f})", axis=1)
    print(
        res.pivot_table(
            index=["part", "sample", "outcome", "donut"],
            columns="bw",
            values="est (se)",
            aggfunc="first",
        ).to_string()
    )
    print(dens.round(3).to_string(index=False))
    print(
        bins[(bins["bin"] >= -21) & (bins["bin"] < 21)].round(3).to_string(index=False)
    )
    print(f"Done in {(time.time() - t0) / 60:.0f} minutes")


if __name__ == "__main__":
    main()
