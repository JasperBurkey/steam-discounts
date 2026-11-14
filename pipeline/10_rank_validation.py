"""Step 10: does Steam's own revenue ranking agree with the revenue proxy?

The revenue proxy values reviews at the price paid. Steam's weekly top-seller charts rank
games by actual revenue (game, DLC and in-game purchases), but only the top 100 per list.
For the large and very large games that appear in them, this asks: during a sale, do
discounted games move up the revenue ranking relative to similar games that held out?

Outcomes per game-event, from the global chart (and, as a check, the best position across
the global and country lists):
  points     101 - rank when charted, 0 otherwise (1st place = 100 points)
  d_points_during   mean points in sale weeks minus baseline weeks
  d_points_net      mean points from 1 week before to 4 weeks after the sale, minus baseline
  charted_during    in the chart in any sale week (0/1)
Chart weeks start Tuesdays 00:00 UTC; a sale week is a chart week at least half inside the
sale. Baseline: the 3 latest chart weeks ending at least 7 days before the sale, skipping
weeks that overlap another sale. Controls: the usual pre-event controls plus chart
position in the baseline weeks.

Caveats: ranks are relative (when most games discount, holdouts can lose places without
losing revenue); the top 100 censors everything below.

  python 10_rank_validation.py
Weighting test: for game-events in the chart both before and during the sale, the change
in log rank is regressed on the review change and log(price paid / regular price). The
proxy weights them equally; a smaller price coefficient means it overstates the revenue
cost of discounting.

Outputs (output/): rank_validation.csv, rank_correlations.csv, rank_weighting_test.csv,
rank_event_study.png
"""

import matplotlib

matplotlib.use("Agg")
import config
import dml_core as M
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WEEKLY = config.DATA / "weekly_topsellers.parquet"
CALENDAR = config.ROOT / "sale_calendar.csv"
BIG = ["large", "very_large"]
H = pd.Timedelta(hours=18)  # sales start 10:00 Pacific (18:00 UTC in winter)
WEEK = pd.Timedelta(days=7)


def sale_windows():
    cal = pd.read_csv(CALENDAR)
    cal["s"] = pd.to_datetime(cal["start"]).dt.tz_localize("UTC") + H
    cal["e"] = pd.to_datetime(cal["end"]).dt.tz_localize("UTC") + H
    return cal


def overlap(a0, a1, b0, b1):
    return max(pd.Timedelta(0), min(a1, b1) - max(a0, b0))


def week_sets(cal, weeks):
    """For each sale: sale weeks, baseline weeks, net-window weeks, and relative week index."""
    out = {}
    for _, r in cal.iterrows():
        sale_w = [w for w in weeks if overlap(w, w + WEEK, r.s, r.e) >= WEEK / 2]
        if not sale_w:
            continue
        busy = [
            w
            for w in weeks
            if any(
                overlap(w, w + WEEK, s, e) >= pd.Timedelta(days=1)
                for s, e in zip(cal.s, cal.e)
            )
        ]
        base = [w for w in weeks if w + WEEK <= r.s - WEEK and w not in busy][-3:]
        net = [w for w in weeks if w + WEEK > r.s - WEEK and w < r.e + 4 * WEEK]
        rel = {w: int(np.floor((w - sale_w[0]) / WEEK)) for w in weeks}
        out[r.sale] = {"sale": sale_w, "base": base, "net": net, "rel": rel}
    return out


def points_table(w, lists):
    """Points per (app, week): 101 - rank, best across the given lists."""
    x = w[w["list"].isin(lists)].copy()
    x["points"] = 101 - x["rank"].clip(upper=100)
    return x.groupby(["appid", "week_start"])["points"].max()


def outcomes(ev, sets, pts, tag):
    rows = []
    pts = pts.to_dict()
    for r in ev.itertuples():
        s = sets.get(r.sale)
        if s is None or not s["base"]:
            rows.append({})
            continue

        def mean(ws, app=r.app_id):
            return (
                float(np.mean([pts.get((app, w), 0.0) for w in ws])) if ws else np.nan
            )

        base = mean(s["base"])
        rows.append(
            {
                f"base_points_{tag}": base,
                f"base_charted_{tag}": float(
                    np.mean([(r.app_id, w) in pts for w in s["base"]])
                ),
                f"d_points_during_{tag}": mean(s["sale"]) - base,
                f"d_points_net_{tag}": mean(s["net"]) - base,
                f"charted_during_{tag}": float(
                    any((r.app_id, w) in pts for w in s["sale"])
                ),
            }
        )
    return pd.DataFrame(rows, index=ev.index)


def log_rank_change(ev, sets, w):
    """For game-events in the global chart in at least one baseline AND one sale week:
    change in mean log rank (negative = climbed). Zipf-type revenue curves make log rank
    roughly linear in log revenue, so this is the outcome for the weighting test."""
    g = w[w["list"] == "global"].set_index(["appid", "week_start"])["rank"].to_dict()
    out = []
    for r in ev.itertuples():
        s = sets.get(r.sale)
        b = (
            [np.log(g[(r.app_id, x)]) for x in s["base"] if (r.app_id, x) in g]
            if s
            else []
        )
        d = (
            [np.log(g[(r.app_id, x)]) for x in s["sale"] if (r.app_id, x) in g]
            if s
            else []
        )
        out.append(np.mean(d) - np.mean(b) if b and d else np.nan)
    return pd.Series(out, index=ev.index)


def weighting_test(df):
    """Proxy check: -dlog(rank) ~ reviews change + log(price paid / regular) + sale effects.
    The proxy weights the two equally (r = y + log price ratio); equal coefficients support
    it, a smaller price coefficient means the proxy overstates the revenue cost of price."""
    from hetero import _cluster_ols

    x = df.dropna(subset=["dlogrank", "y_during", "price_ratio_sale"]).copy()
    x = x[x["price_ratio_sale"] > 0]
    y = -x["dlogrank"].to_numpy()
    lp = np.log(x["price_ratio_sale"].to_numpy())
    fe = pd.get_dummies(x["sale"], dtype=float).to_numpy()
    X = np.column_stack([x["y_during"].to_numpy(), lp, fe])
    b, se = _cluster_ols(y, X, x["app_id"].to_numpy())
    # b_y*y + b_p*lp = b_y*(y + lp) + (b_p - b_y)*lp: the lp coefficient is the difference
    Xd = np.column_stack([x["y_during"].to_numpy() + lp, lp, fe])
    bd, sed = _cluster_ols(y, Xd, x["app_id"].to_numpy())
    return {
        "n": len(x),
        "games": x["app_id"].nunique(),
        "discounted_pct": round(100 * x["discounted"].mean()),
        "b_reviews": b[0],
        "se_reviews": se[0],
        "b_log_price": b[1],
        "se_log_price": se[1],
        "diff_price_minus_reviews": bd[1],
        "se_diff": sed[1],
    }


def within_sale_corr(df, a, b):
    """Correlation of a and b after removing each sale's mean (ranks are relative)."""
    x = df[[a, b, "sale"]].dropna()
    x[a] = x[a] - x.groupby("sale")[a].transform("mean")
    x[b] = x[b] - x.groupby("sale")[b].transform("mean")
    return x[a].corr(x[b]), len(x)


def main():
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    w = pd.read_parquet(WEEKLY)
    w = w[w["week_start"].notna()]
    weeks = sorted(w["week_start"].unique())
    sets = week_sets(sale_windows(), weeks)

    main_df = ev[ev["analysis_sample"]].dropna(subset=["y_during", "r_during"])
    big = main_df[main_df["band"].isin(BIG)].reset_index(drop=True)
    lists_all = sorted(w["list"].unique())
    big = big.join(outcomes(big, sets, points_table(w, ["global"]), "global"))
    big = big.join(outcomes(big, sets, points_table(w, lists_all), "any"))
    big = big.dropna(subset=["d_points_during_global"]).reset_index(drop=True)

    print(
        f"Large and very large games: {len(big):,} game-events, {big['app_id'].nunique()} "
        f"games, {big['discounted'].mean():.0%} discounted"
    )
    print(
        f"In the global top 100 before the sale: {(big['base_charted_global'] > 0).mean():.0%}"
        f" of game-events; during the sale: {big['charted_during_global'].mean():.0%}"
    )
    print(f"Lists used for 'any': {', '.join(lists_all)}\n")

    rows = []

    def add(label, df, ycols, extra):
        res, _, _ = M.aipw(df, ycols, extra=extra)
        for yc in ycols:
            for est in ["ATT", "ATE"]:
                e, se = res[yc][est]
                rows.append(
                    {
                        "sample": label,
                        "outcome": yc,
                        "estimand": est,
                        "estimate": e,
                        "se": se,
                        "ci_low": e - 1.96 * se,
                        "ci_high": e + 1.96 * se,
                        "n": len(df),
                        "games": df["app_id"].nunique(),
                        "discounted_pct": round(100 * df["discounted"].mean()),
                    }
                )

    chart_ctrl = ["base_points_global", "base_charted_global", "base_points_any"]
    add(
        "large + very large",
        big,
        [
            "d_points_during_global",
            "d_points_net_global",
            "charted_during_global",
            "d_points_during_any",
            "r_during",
            "y_during",
        ],
        chart_ctrl,
    )
    charted = big[big["base_charted_global"] > 0].reset_index(drop=True)
    if charted["discounted"].sum() >= 30 and (~charted["discounted"]).sum() >= 30:
        add(
            "charted before the sale",
            charted,
            ["d_points_during_global", "r_during", "y_during"],
            chart_ctrl,
        )
    else:
        print("Too few non-discounted games charting before sales for that subsample.")
    res = pd.DataFrame(rows)
    res.to_csv(config.OUTPUT / "rank_validation.csv", index=False)

    # game-level agreement: do events where the proxy says revenue rose also climb the chart?
    corr = []
    for samp, df in [("large + very large", big), ("charted before the sale", charted)]:
        d = df[df["discounted"]]
        for proxy in ["r_during", "y_during"]:
            c, n = within_sale_corr(d, "d_points_during_global", proxy)
            corr.append(
                {
                    "sample": samp,
                    "discounted only": True,
                    "proxy": proxy,
                    "corr": c,
                    "n": n,
                }
            )
    corr = pd.DataFrame(corr)
    corr.to_csv(config.OUTPUT / "rank_correlations.csv", index=False)

    big["dlogrank"] = log_rank_change(big, sets, w)
    wt = pd.DataFrame([weighting_test(big)])
    wt.to_csv(config.OUTPUT / "rank_weighting_test.csv", index=False)

    # event-study figure: mean global points relative to baseline, by week
    pts = points_table(w, ["global"]).to_dict()
    recs = []
    for r in big.itertuples():
        s = sets[r.sale]
        for wk, k in s["rel"].items():
            if -4 <= k <= 6:
                recs.append(
                    {
                        "k": k,
                        "discounted": r.discounted,
                        "pts": pts.get((r.app_id, wk), 0.0) - r.base_points_global,
                    }
                )
    es = pd.DataFrame(recs).groupby(["discounted", "k"])["pts"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, lab in [(True, "discounted"), (False, "not discounted")]:
        if col in es:
            ax.plot(es.index, es[col], marker="o", label=lab)
    ax.axvspan(-0.5, 0.5, color="gold", alpha=0.2, label="first sale week")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Chart week relative to the first sale week")
    ax.set_ylabel("Global chart points vs. baseline\n(101 - rank; 0 if not in top 100)")
    ax.set_title("Steam weekly revenue chart, large and very large games")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "rank_event_study.png", dpi=150)

    pd.set_option("display.width", 200)
    show = res.copy()
    show["estimate (se)"] = show.apply(
        lambda r: f"{r.estimate:+.3f} ({r.se:.3f})", axis=1
    )
    print(
        show[
            [
                "sample",
                "outcome",
                "estimand",
                "estimate (se)",
                "n",
                "games",
                "discounted_pct",
            ]
        ].to_string(index=False)
    )
    print(
        "\nWithin-sale correlation, discounted game-events: chart-point change vs. proxy"
    )
    print(corr.round(3).to_string(index=False))
    print(
        "\nWeighting test (game-events in the global chart before AND during the sale):"
        "\n  -dlog(rank) on reviews change and log(price paid / regular), sale effects"
    )
    print(wt.round(3).to_string(index=False))
    print(
        "\nReading: d_points are rank places (out of 100); r_during and y_during are log "
        "changes. Positive d_points for discounted games = they climbed relative to holdouts."
    )
    print(
        f"Wrote rank_validation.csv, rank_correlations.csv, rank_event_study.png in {config.OUTPUT}"
    )


if __name__ == "__main__":
    main()
