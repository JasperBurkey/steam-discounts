"""Step 16: two checks built from review flags.

1. Placebo outcome. Reviews from people who did NOT pay Steam's price (activated a key
   bought elsewhere, or received the game free) should not respond to a Steam discount
   through price. If they rise as much as paid reviews, something other than the price
   (an update, visibility, a common shock) is moving reviews, and the main effect is
   suspect. Not a perfect placebo: sale visibility can also bring back owners who got the
   game elsewhere, so a small positive effect is expected.
2. A faster purchase signal. A review written under FAST_MIN minutes of play (Steam's
   refund window is 2 hours) was almost certainly written soon after buying, so these
   reviews track purchase timing with little lag. The window profile shows whether the
   elevated weeks after a sale are review lag (they should vanish here) or extra demand.

Sparsity: both series are a small share of reviews, and on the log(1 + x) scale a sparse
series shows smaller effects mechanically. Each is therefore compared with a THINNED
BENCHMARK: paid Steam reviews randomly thinned, game by game, to the same share. The
question is whether the series responds differently from paid reviews at equal sparsity.

Reviews come from the compact export (UTC days since REVIEW_HISTORY_START); only games
with complete review histories (not the large games' recent samples) are used. Outcomes
use the same windows and transform as 04b_build_events.py; the all-review outcome is
recomputed the same way, so every comparison uses identical days.

  python 16_review_flags.py
Outputs: results/review_flags/
"""

import time

import config
import dml_core as M
import numpy as np
import pandas as pd

OUT = config.ROOT.parent / "results" / "review_flags"
COMPACT = config.DATA / "for_claude" / "reviews_compact.parquet"
FAST_MIN = 120  # minutes of play at review
SEED = 0
WINDOWS = {  # same as 03_pull_reviews.WINDOWS (n = sale length in days)
    "A": lambda n: (-60, -29),
    "B": lambda n: (-28, -22),
    "C": lambda n: (-21, -8),
    "D": lambda n: (-7, -1),
    "E": lambda n: (0, n - 1),
    "F": lambda n: (n, n + 13),
    "G": lambda n: (n + 14, n + 27),
    "H": lambda n: (n + 28, n + 41),
}
OUTCOMES = {  # name: windows averaged (vs. baseline window C)
    "pre_week": "D",
    "during": "E",
    "after": "F",
    "wk3_4": "G",
    "wk5_6": "H",
    "net": "DEFG",
}
SERIES = {
    "all": "All reviews",
    "paid": "Paid Steam purchases",
    "placebo": "Not paid on Steam (placebo)",
    "paid_thin_placebo": "Paid, thinned to the placebo share (benchmark)",
    "fast": f"Written under {FAST_MIN // 60} hours of play",
    "paid_thin_fast": "Paid, thinned to the fast share (benchmark)",
}


def series_days(rv, rng):
    """Per game and series: sorted review days."""
    sp = rv["steam_purchase"].fillna(False).to_numpy(bool)
    free = rv["received_for_free"].fillna(False).to_numpy(bool)
    paid = sp & ~free
    placebo = ~paid
    fast = rv["playtime_min"].fillna(10**9).to_numpy() < FAST_MIN
    # thinning: keep each paid review with probability (target share / paid share), per game
    g = rv["app_id"].to_numpy()
    df = pd.DataFrame({"g": g, "paid": paid, "placebo": placebo, "fast": fast})
    shares = df.groupby("g")[["paid", "placebo", "fast"]].mean()
    p_pl = (
        (shares["placebo"] / shares["paid"].where(shares["paid"] > 0))
        .clip(upper=1)
        .fillna(0)
    )
    p_fa = (
        (shares["fast"] / shares["paid"].where(shares["paid"] > 0))
        .clip(upper=1)
        .fillna(0)
    )
    u = rng.random(len(rv))
    masks = {
        "all": np.ones(len(rv), bool),
        "paid": paid,
        "placebo": placebo,
        "paid_thin_placebo": paid & (u < p_pl.reindex(g).to_numpy()),
        "fast": fast,
        "paid_thin_fast": paid & (u < p_fa.reindex(g).to_numpy()),
    }
    day = rv["day"].to_numpy()
    out = {}
    for name, m in masks.items():
        s = pd.Series(day[m]).groupby(g[m]).apply(lambda x: np.sort(x.to_numpy()))
        out[name] = s.to_dict()
    return out, shares


def window_sums(days, d0, n):
    """Reviews in each window for one game-event (days relative to the sale start d0)."""
    out = {}
    for w, f in WINDOWS.items():
        a, b = f(n)
        out[w] = int(
            np.searchsorted(days, d0 + b, "right")
            - np.searchsorted(days, d0 + a, "left")
        )
    return out


def outcomes(sums, n):
    """log(1 + 14 x daily rate) in each outcome window minus the same in window C."""
    L = {w: f(n)[1] - f(n)[0] + 1 for w, f in WINDOWS.items()}

    def t(ws):
        return np.log1p(14 * sum(sums[w] for w in ws) / sum(L[w] for w in ws))

    base = t("C")
    return {k: t(ws) - base for k, ws in OUTCOMES.items()}


def build(ev, days, day0, last_day):
    rows = []
    for r in ev.itertuples(index=False):
        d0 = (pd.Timestamp(r.event_start) - day0).days
        n = int(r.sale_days)
        if d0 + WINDOWS["A"](n)[0] < 0:
            continue
        long_ok = d0 + WINDOWS["H"](n)[1] <= last_day
        if d0 + WINDOWS["F"](n)[1] > last_day:
            continue
        row = {"app_id": r.app_id, "sale": r.sale}
        for s in SERIES:
            dd = days[s].get(r.app_id, np.array([], dtype=int))
            o = outcomes(window_sums(dd, d0, n), n)
            for k, v in o.items():
                if k in ("wk3_4", "wk5_6", "net") and not long_ok:
                    v = np.nan
                row[f"{s}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rv = pd.read_parquet(
        COMPACT,
        columns=[
            "app_id",
            "day",
            "steam_purchase",
            "received_for_free",
            "playtime_min",
            "sampled",
        ],
    )
    sampled = rv.groupby("app_id")["sampled"].any()
    rv = rv[rv["app_id"].isin(sampled[~sampled].index) & (rv["day"] >= 0)].reset_index(
        drop=True
    )
    last_day = int(rv["day"].max()) - 2  # the last days of the export may be incomplete
    day0 = pd.Timestamp(config.REVIEW_HISTORY_START)
    days, shares = series_days(rv, np.random.default_rng(SEED))
    print(
        f"{rv['app_id'].nunique():,} games with full histories, {len(rv):,} reviews  [{time.time() - t0:.0f}s]",
        flush=True,
    )

    ev = pd.read_parquet(config.EVENTS_PARQUET)
    ev = ev[ev["analysis_sample"] & ev["app_id"].isin(days["all"].keys())].dropna(
        subset=["y_during", "pre_trend"]
    )
    flags = build(
        ev[["app_id", "sale", "event_start", "sale_days"]], days, day0, last_day
    )
    df = ev.merge(flags, on=["app_id", "sale"]).reset_index(drop=True)
    check = df[["y_during", "all_during"]].corr().iloc[0, 1]
    print(
        f"{len(df):,} game-events; recomputed all-review outcome vs. original: correlation {check:.3f}  [{time.time() - t0:.0f}s]",
        flush=True,
    )

    samples = {
        "all games (full histories)": df,
        "busier games": df[df["base_rate"] >= config.BUSY_MIN].reset_index(drop=True),
    }
    rows = []
    for sname, d in samples.items():
        for s, label in SERIES.items():
            cols = [f"{s}_{k}" for k in OUTCOMES]
            dd = d.dropna(subset=[f"{s}_during"]).reset_index(drop=True)
            res = M.aipw(dd, [c for c in cols if dd[c].notna().all()])[0]
            for k in OUTCOMES:
                c = f"{s}_{k}"
                if c not in res:
                    dn = dd.dropna(subset=[c]).reset_index(drop=True)
                    res.update(M.aipw(dn, [c])[0])
                est, se = res[c]["ATT"]
                rows.append(
                    {
                        "sample": sname,
                        "series": label,
                        "outcome": k,
                        "ATT": est,
                        "se": se,
                        "n": int(dd[c].notna().sum()),
                    }
                )
            print(
                f"  {sname}: {label} done  [{(time.time() - t0) / 60:.0f} min]",
                flush=True,
            )
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "effects.csv", index=False)

    # the two tests: series minus its thinned benchmark (difference of ATTs; SEs combined
    # conservatively as if independent, since both come from the same games)
    tests = []
    for sname in samples:
        r = res[res["sample"] == sname].set_index(["series", "outcome"])
        for s, bench in [("placebo", "paid_thin_placebo"), ("fast", "paid_thin_fast")]:
            for k in OUTCOMES:
                a, b = r.loc[(SERIES[s], k)], r.loc[(SERIES[bench], k)]
                tests.append(
                    {
                        "sample": sname,
                        "test": s,
                        "outcome": k,
                        "series ATT": a["ATT"],
                        "benchmark ATT": b["ATT"],
                        "difference": a["ATT"] - b["ATT"],
                        "se (conservative)": float(np.hypot(a["se"], b["se"])),
                    }
                )
    pd.DataFrame(tests).to_csv(OUT / "tests.csv", index=False)
    shares.describe().to_csv(OUT / "series_shares.csv")

    pd.set_option("display.width", 220)
    show = res.copy()
    show["ATT (SE)"] = show.apply(lambda x: f"{x['ATT']:+.3f} ({x['se']:.3f})", axis=1)
    print(
        show.pivot_table(
            index=["sample", "series"],
            columns="outcome",
            values="ATT (SE)",
            aggfunc="first",
        )[list(OUTCOMES)].to_string()
    )
    print(pd.DataFrame(tests).round(3).to_string(index=False))
    print(f"Done in {(time.time() - t0) / 60:.0f} minutes")


if __name__ == "__main__":
    main()
