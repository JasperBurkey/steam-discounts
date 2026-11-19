"""Step 26: do publishers give every game the same discount? (uniform discounting)

DellaVigna and Gentzkow (2019) found retail chains charge nearly the same price in every
store although demand differs: a sign of pricing by rule. The analog here: within a
seasonal sale, does a publisher give all its games the same discount, although the games
differ?

Data: every paid game with a price history (the 29,000-game universe of 22_release_rd),
its discount in each seasonal sale (largest cut during the sale), its publisher from
game_info. Publisher-sales with 2+ discounted games.

Measures per publisher-sale
  identical    all discounted games at the same depth (within 1 point)
  modal share  share of the publisher's discounted games at its most common depth
  depths       number of distinct depths
Benchmark: the same measures after shuffling depths at random among all discounted games
in the same sale and the same cell of regular price (terciles) x age (terciles) x the
game's previous-sale depth (none, 1-33, 34-60, 61+), 200 times. Games that look alike get
alike discounts by chance in the benchmark too, so the excess over it is the part due to
the publisher.

By catalog size (the publisher's games on Steam): 2-4, 5-19, 20+.

Link to policy learning (study games, needs results/policy/event_gains.parquet from
13_policy.py): in publisher-sales where all games got the same depth, does the learned
rule recommend different depths for those games, and how large are the gains it finds
there compared with elsewhere?

  python 26_uniformity.py
Outputs: results/uniformity/
"""

import json
import time

import config
import numpy as np
import pandas as pd
from prices import price_arrays, sale_times

OUT = config.ROOT.parent / "results" / "uniformity"
COMPACT = config.DATA / "for_claude"
POLICY_GAINS = config.ROOT.parent / "results" / "policy" / "event_gains.parquet"
R = 200
rng = np.random.default_rng(config.SEED)


def first(s):
    try:
        v = json.loads(s) if isinstance(s, str) else None
    except ValueError:
        return None
    return v[0] if v else None


def group_stats(df):
    """Per publisher-sale: identical, modal share, number of depths (depth rounded)."""
    d = df.assign(c=df["cut"].round())
    g = d.groupby(["pub", "sale"])["c"]
    n = g.size()
    modal = g.agg(lambda s: s.value_counts().iloc[0]) / n
    nd = g.nunique()
    return pd.DataFrame(
        {"n": n, "modal_share": modal, "depths": nd, "identical": nd == 1}
    )


def size_band(c):
    return pd.cut(c, [1, 4, 19, 1e9], labels=["2-4 games", "5-19 games", "20+ games"])


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    gi = pd.read_parquet(COMPACT / "game_info.parquet")
    gi = gi[(gi["type"].astype(str) == "0") & (gi["price_original_cents"] > 0)].copy()
    gi["pub"] = gi["publishers"].map(first)
    gi["rel"] = pd.to_datetime(gi["steam_release_date"], utc=True, errors="coerce")
    gi["catalog"] = gi["pub"].map(gi.groupby("pub").size())
    hist = (
        pd.read_parquet(COMPACT / "price_history.parquet")
        .assign(ts=lambda h: pd.to_datetime(h["ts"], utc=True))
        .sort_values(["app_id", "ts"])
    )
    hist = hist[hist["app_id"].isin(gi["appid"])]
    arrays = price_arrays(hist)
    cal = pd.read_csv(config.ROOT / "sale_calendar.csv")
    info = gi.set_index("appid")
    rows = []
    for _, s in cal.iterrows():
        start, end = sale_times(s)
        a0, a1 = int(start.timestamp()), int(end.timestamp())
        for app, pa in arrays.items():
            if app not in info.index or info.at[app, "rel"] > start - pd.Timedelta(
                days=30
            ):
                continue  # not yet eligible
            cut = pa.max_cut(a0, a1)
            if np.isfinite(cut):
                rows.append(
                    {"app_id": app, "sale": s["sale"], "start": start, "cut": cut}
                )
    ev = pd.DataFrame(rows).sort_values(["app_id", "start"])
    ev["prev_cut"] = ev.groupby("app_id")["cut"].shift(1)
    ev = ev.merge(
        gi[["appid", "pub", "catalog", "rel", "price_original_cents"]],
        left_on="app_id",
        right_on="appid",
    ).dropna(subset=["pub"])
    ev["age"] = (ev["start"] - ev["rel"]).dt.days
    print(
        f"{len(ev):,} game-sales, {ev['app_id'].nunique():,} games, "
        f"{(ev['cut'] > 0).mean():.0%} discounted  [{time.time() - t0:.0f}s]",
        flush=True,
    )

    d = ev[ev["cut"] > 0].copy()
    d["n_disc"] = d.groupby(["pub", "sale"])["app_id"].transform("size")
    d = d[d["n_disc"] >= 2].copy()
    d["price_t"] = d.groupby("sale")["price_original_cents"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=False)
    )
    d["age_t"] = d.groupby("sale")["age"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=False)
    )
    d["prev_bin"] = pd.cut(
        d["prev_cut"].fillna(0), [-1, 0, 33, 60, 100], labels=False
    ).astype(int)
    d["cell"] = (
        d["sale"]
        + "|"
        + d["price_t"].astype(str)
        + d["age_t"].astype(str)
        + d["prev_bin"].astype(str)
    )
    obs = group_stats(d)
    obs["band"] = size_band(
        obs.index.get_level_values("pub").map(gi.groupby("pub")["catalog"].first())
    )
    print(
        f"{len(obs):,} publisher-sales with 2+ discounted games, "
        f"{obs.index.get_level_values('pub').nunique():,} publishers  [{time.time() - t0:.0f}s]",
        flush=True,
    )

    # benchmark: shuffle depths within cell (all discounted games in those publisher-sales)
    cells = d.groupby("cell").indices
    cut = d["cut"].to_numpy()
    sims = []
    for r in range(R):
        perm = cut.copy()
        for idx in cells.values():
            perm[idx] = cut[rng.permutation(idx)]
        st = group_stats(d.assign(cut=perm))
        st["band"] = obs["band"]
        sims.append(
            st.groupby("band", observed=True)[["identical", "modal_share", "depths"]]
            .mean()
            .assign(r=r)
        )
        sims.append(
            st[["identical", "modal_share", "depths"]]
            .mean()
            .to_frame()
            .T.assign(band="all", r=r)
            .set_index("band")
        )
    sims = pd.concat(sims)
    out = []
    groups = [("all", obs)] + [(b, g) for b, g in obs.groupby("band", observed=True)]
    for b, g in groups:
        sb = sims.loc[[b]] if b in sims.index else None
        for m in ["identical", "modal_share", "depths"]:
            out.append(
                {
                    "publisher size": b,
                    "measure": m,
                    "observed": g[m].mean(),
                    "random within similar games": sb[m].mean(),
                    "benchmark 95% range": f"[{sb[m].quantile(0.025):.3f}, {sb[m].quantile(0.975):.3f}]",
                    "publisher-sales": len(g),
                    "mean games per publisher-sale": g["n"].mean(),
                }
            )
    res = pd.DataFrame(out)
    res.to_csv(OUT / "uniformity.csv", index=False)
    # focal depths
    dd = d["cut"].round()
    focal = pd.DataFrame(
        {
            "share at multiples of 5": [(dd % 5 == 0).mean()],
            "share at multiples of 10": [(dd % 10 == 0).mean()],
            "share at 50 or 75": [dd.isin([50, 75]).mean()],
            "top depths": [
                ", ".join(
                    f"{int(k)}% ({v:.0%})"
                    for k, v in dd.value_counts(normalize=True).head(6).items()
                )
            ],
        }
    )
    focal.to_csv(OUT / "focal_depths.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print(focal.to_string(index=False))

    # link to the learned rule (study games)
    if POLICY_GAINS.exists():
        pg = pd.read_parquet(POLICY_GAINS)
        pg = pg[
            (pg["sample"] == "all game-events")
            & (pg["policy"] == "tree, 2 levels [learned on proxy]")
        ]
        lk = []
        for obj, p in pg.groupby("objective"):
            m = p.merge(
                d[["app_id", "sale", "pub"]], on=["app_id", "sale"], how="inner"
            )
            st = obs.reset_index()[["pub", "sale", "identical", "band"]]
            m = m.merge(st, on=["pub", "sale"])
            for ident, q in m.groupby("identical"):
                rec = q.groupby(["pub", "sale"])["assigned"].nunique()
                lk.append(
                    {
                        "objective": obj,
                        "publisher-sales": "all games same depth"
                        if ident
                        else "depths differ",
                        "game-events": len(q),
                        "rule recommends different depths within the publisher-sale": (
                            rec > 1
                        ).mean(),
                        "rule changes the choice": (
                            q["assigned"] != q["chosen"]
                        ).mean(),
                        "mean gain": q["gain"].mean(),
                    }
                )
        lk = pd.DataFrame(lk)
        lk.to_csv(OUT / "policy_link.csv", index=False)
        print(lk.round(3).to_string(index=False))
    print(f"Done in {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
