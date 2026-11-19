"""Step 18: placebo sale dates.

Each sale is moved SHIFT_DAYS earlier, to a time with no sale, and the estimator is rerun
with the real treatment (whether the game discounted in the real sale that follows). If
the estimator is sound, games that are about to discount show no "effect" at the fake
date. A clear effect would mean the main estimates partly reflect differences between
discounting and full-price games rather than the discount.

To keep the placebo honest:
  - every control is recomputed as of the fake date: review rates, trend and % positive
    from the review data, discount history from the price histories, and age;
  - game-events with any discount in force during the fake windows are dropped (a habitual
    discounter's off-sale discount would otherwise show up as a real effect);
  - the real-date effect is recomputed on the same games with the same review data, so the
    two are directly comparable.
Only games with complete review histories are used (as in 16_review_flags.py).

  python 18_placebo_dates.py
Outputs: results/placebo_dates/
"""

import importlib
import time

import config
import dml_core as M
import numpy as np
import pandas as pd
from prices import load_history, price_arrays, sale_times

F16 = importlib.import_module("16_review_flags")
OUT = config.ROOT.parent / "results" / "placebo_dates"
COMPACT = config.DATA / "for_claude"
SHIFT_DAYS = 42
OUTCOMES = {"pre_week": "Week before", "during": "During", "after": "Weeks 1-2 after"}
HIST = [
    "disc_starts_365",
    "mean_cut_365",
    "share_days_disc_365",
    "min_ratio_365",
    "max_cut_ever",
]


def review_features(all_days, pos_days, d0, n):
    """Pre-period controls as in 04b_build_events.outcomes, from sorted review days."""
    s = F16.window_sums(all_days, d0, n)
    p = F16.window_sums(pos_days, d0, n)
    L = {w: f(n)[1] - f(n)[0] + 1 for w, f in F16.WINDOWS.items()}

    def rate(*ws):
        return sum(s[w] for w in ws) / sum(L[w] for w in ws)

    pre28, pre60, old = rate("B", "C", "D"), rate("A", "B", "C", "D"), rate("A")
    n60 = s["A"] + s["B"] + s["C"] + s["D"]
    return {
        "pre_rate_28": pre28,
        "pre_rate_60": pre60,
        "pre_trend": np.log1p(28 * pre28) - np.log1p(28 * old),
        "pct_pos_60": (p["A"] + p["B"] + p["C"] + p["D"]) / n60 if n60 > 0 else np.nan,
        "base_rate": rate("C"),
    }


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rv = pd.read_parquet(
        COMPACT / "reviews_compact.parquet",
        columns=["app_id", "day", "voted_up", "sampled"],
    )
    sampled = rv.groupby("app_id")["sampled"].any()
    rv = rv[rv["app_id"].isin(sampled[~sampled].index) & (rv["day"] >= 0)]
    last_day = int(rv["day"].max()) - 2
    days = {a: np.sort(g["day"].to_numpy()) for a, g in rv.groupby("app_id")}
    pos = {
        a: np.sort(g.loc[g["voted_up"].fillna(False).astype(bool), "day"].to_numpy())
        for a, g in rv.groupby("app_id")
    }
    hist = (
        load_history()
        if config.PRICE_HISTORY_PARQUET.exists()
        else pd.read_parquet(COMPACT / "price_history.parquet")
        .assign(ts=lambda h: pd.to_datetime(h["ts"], utc=True))
        .sort_values(["app_id", "ts"])
    )
    arrays = price_arrays(hist)
    cal = pd.read_csv(config.ROOT / "sale_calendar.csv")
    starts = {r["sale"]: sale_times(r)[0] for _, r in cal.iterrows()}
    day0 = pd.Timestamp(config.REVIEW_HISTORY_START)
    print(
        f"{len(days):,} games with full review histories  [{time.time() - t0:.0f}s]",
        flush=True,
    )

    ev = pd.read_parquet(config.EVENTS_PARQUET)
    ev = (
        ev[ev["analysis_sample"] & ev["app_id"].isin(days)]
        .dropna(subset=["y_during", "pre_trend"])
        .reset_index(drop=True)
    )
    rows = []
    dropped_disc = 0
    for r in ev.itertuples(index=False):
        n = int(r.sale_days)
        real0 = (pd.Timestamp(r.event_start) - day0).days
        fake0 = real0 - SHIFT_DAYS
        if (
            fake0 + F16.WINDOWS["A"](n)[0] < 0
            or real0 + F16.WINDOWS["F"](n)[1] > last_day
        ):
            continue
        pa = arrays.get(int(r.app_id))
        t_fake = int((starts[r.sale] - pd.Timedelta(days=SHIFT_DAYS)).timestamp())
        if pa is None:
            continue
        cut = pa.max_cut(t_fake - 7 * 86400, t_fake + (n + 14) * 86400)
        if not np.isfinite(cut) or cut > 0:
            dropped_disc += 1
            continue
        row = {"app_id": r.app_id, "sale": r.sale}
        for tag, d0 in [("real", real0), ("fake", fake0)]:
            o = F16.outcomes(F16.window_sums(days[r.app_id], d0, n), n)
            for k in OUTCOMES:
                row[f"{tag}_{k}"] = o[k]
            for k, v in review_features(days[r.app_id], pos[r.app_id], d0, n).items():
                row[f"{tag}__{k}"] = v
        h = pa.history_features(t_fake)
        row.update({f"fake__{k}": v for k, v in h.items()})
        # end of the last discount before the fake date (as 04b_build_events does)
        last_end = None
        disc = np.where((pa.cut > 0) & (pa.ts < t_fake))[0]
        if len(disc):
            k = disc[-1]
            last_end = min(pa.ts[k + 1], t_fake) if k + 1 < len(pa.ts) else t_fake
        row["fake__days_since_discount"] = (
            (t_fake - last_end) / 86400 if last_end is not None else np.nan
        )
        row["fake__never_discounted_before"] = last_end is None
        rows.append(row)
    pl = pd.DataFrame(rows)
    df = ev.merge(pl, on=["app_id", "sale"]).reset_index(drop=True)
    print(
        f"{len(df):,} game-events ({dropped_disc:,} dropped: discount in force at the fake date)  [{time.time() - t0:.0f}s]",
        flush=True,
    )

    results = []
    for tag in ["fake", "real"]:
        d = df.copy()
        # controls as of that date
        for k in ["pre_rate_28", "pre_rate_60", "pre_trend", "pct_pos_60", "base_rate"]:
            d[k] = d[f"{tag}__{k}"]
        if tag == "fake":
            for k in HIST + ["days_since_discount", "never_discounted_before"]:
                d[k] = d[f"fake__{k}"]
            d["age_days"] = d["age_days"] - SHIFT_DAYS
        ycols = [f"{tag}_{k}" for k in OUTCOMES]
        for sname, dd in [
            ("all games (full histories)", d),
            ("busier games", d[d["base_rate"] >= config.BUSY_MIN]),
        ]:
            dd = dd.dropna(subset=ycols).reset_index(drop=True)
            res = M.aipw(dd, ycols)[0]
            for k in OUTCOMES:
                est, se = res[f"{tag}_{k}"]["ATT"]
                results.append(
                    {
                        "date": "placebo (6 weeks earlier)"
                        if tag == "fake"
                        else "real sale",
                        "sample": sname,
                        "outcome": OUTCOMES[k],
                        "ATT": est,
                        "se": se,
                        "n": len(dd),
                    }
                )
        print(f"  {tag} dates done  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    res = pd.DataFrame(results)
    res.to_csv(OUT / "effects.csv", index=False)
    res["ATT (SE)"] = res.apply(lambda x: f"{x['ATT']:+.3f} ({x['se']:.3f})", axis=1)
    pd.set_option("display.width", 200)
    print(
        res.pivot_table(
            index=["sample", "date"],
            columns="outcome",
            values="ATT (SE)",
            aggfunc="first",
        )[list(OUTCOMES.values())].to_string()
    )
    print(f"Done in {(time.time() - t0) / 60:.0f} minutes")


if __name__ == "__main__":
    main()
