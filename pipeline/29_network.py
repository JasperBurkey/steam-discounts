"""Step 29: network effects. Do discounts pay off later in online multiplayer games?

In an online multiplayer game each new player makes the game more valuable to others, so
a discount that grows the player base could raise demand after the sale. In a
single-player game it should not. Local-only multiplayer (couch co-op, split screen) is a
genre control: multiplayer design, but new buyers don't play with existing ones.

Groups (game_info tags, the store's top tags for each game):
  online     Multiplayer, Online Co-Op, PvP, Massively Multiplayer, MMORPG, Social
             Deduction or Asynchronous Multiplayer among the tags
  local only Local Co-Op, Local Multiplayer, Split Screen or 4 Player Local, no online tag
  single     none of these
  (online, strong: an online tag among the game's top 5 tags)

A. Release cutoff (22_release_rd.py sample): fuzzy RD effects by group on reviews during
   the sale, weeks 1-2 and 3-4 after, and net revenue, relative to the week before. A
   bigger lasting effect for online games, from an essentially random player-base shock,
   is evidence of network effects in demand. Weeks 5-12 need the long windows of
   03h_pull_rd_reviews.py --long (rd_review_counts_long.parquet); used when present, for
   Spring and Summer sales only (otherwise the next seasonal sale falls in the window).
B. Main sample: doubly robust ATT by group on reviews during, weeks 3-4, 5-6, the net
   revenue proxy, and weeks 7-12 for Spring and Summer sales (as in 23_visibility).
   Observational heterogeneity: online games also differ in other ways.

  python 29_network.py        (LEARNER / CF_REPEATS as in config)
Outputs: results/network/
"""

import importlib
import json
import time

import config
import numpy as np
import pandas as pd

R = importlib.import_module("22_release_rd")
V = importlib.import_module("23_visibility")
OUT = config.ROOT.parent / "results" / "network"
COMPACT = config.DATA / "for_claude"
ONLINE = {128, 1754, 1775, 3843, 3859, 745697, 17770}
LOCAL = {3841, 4840, 7368, 10816}


def tag_groups():
    gi = pd.read_parquet(COMPACT / "game_info.parquet", columns=["appid", "tagids"])

    def parse(s):
        try:
            return json.loads(s) if isinstance(s, str) else []
        except ValueError:
            return []

    tags = gi["tagids"].map(parse)
    online = tags.map(lambda t: bool(ONLINE & set(t)))
    strong = tags.map(lambda t: bool(ONLINE & set(t[:5])))
    local = tags.map(lambda t: bool(LOCAL & set(t)))
    grp = np.where(online, "online", np.where(local, "local only", "single"))
    return pd.DataFrame(
        {"app_id": gi["appid"].astype(int), "group": grp, "online_strong": strong}
    )


def part_a(groups):
    o = pd.read_parquet(
        config.ROOT.parent / "results" / "release_rd" / "rd_sample.parquet"
    )
    o = o.merge(groups, on="app_id", how="left")
    ys = {
        "dE": "during the sale",
        "dF": "weeks 1-2 after",
        "dG": "weeks 3-4 after",
        "drEFG": "net revenue, sale to week 4",
    }
    long_file = COMPACT / "rd_review_counts_long.parquet"
    if long_file.exists():
        lg = pd.read_parquet(long_file)
        o = o.merge(lg[["app_id", "sale", "H", "I"]], on=["app_id", "sale"], how="left")
        base = np.expm1(o["y_D"])  # reviews in the week before
        o["dH"] = np.log1p(o["H"] / 2) - np.log1p(base)
        o["dI"] = np.log1p(o["I"] / 6) - np.log1p(base)
        # weeks 5-12 overlap the next seasonal sale except after Spring and Summer
        far = o["sale"].str.startswith(("Spring", "Summer"))
        o.loc[~far, ["dH", "dI"]] = np.nan
        ys.update({"dH": "weeks 5-6 after", "dI": "weeks 7-12 after"})
    sets = {
        "online": o[o["group"] == "online"],
        "online, strong": o[o["online_strong"] == True],  # noqa: E712
        "local only": o[o["group"] == "local only"],
        "single": o[o["group"] == "single"],
    }
    rows = []
    for y, lab in ys.items():
        for h in (14, 21):
            est = {}
            for g, d in sets.items():
                d = d.dropna(subset=[y]).reset_index(drop=True)
                e, se, n = R.rd(d, y, h)
                fs, _, _ = R.rd(d, "discounted", h, fuzzy=False)
                est[g] = (e, se)
                rows.append(
                    dict(
                        part="A: release cutoff (fuzzy RD)",
                        outcome=lab,
                        bw=h,
                        group=g,
                        est=e,
                        se=se,
                        n=n,
                        first_stage=fs,
                    )
                )
            for g in ("online", "online, strong"):
                d_ = est[g][0] - est["single"][0]
                s_ = np.hypot(est[g][1], est["single"][1])
                rows.append(
                    dict(
                        part="A: release cutoff (fuzzy RD)",
                        outcome=lab,
                        bw=h,
                        group=f"{g} - single",
                        est=d_,
                        se=s_,
                        n=np.nan,
                    )
                )
    return rows


def part_b(groups, t0):
    ev = pd.read_parquet(config.EVENTS_PARQUET).merge(groups, on="app_id", how="left")
    rv = pd.read_parquet(
        COMPACT / "reviews_compact.parquet", columns=["app_id", "day", "sampled"]
    )
    samp = rv.groupby("app_id")["sampled"].any()
    rv = rv[rv["app_id"].isin(samp[~samp].index) & (rv["day"] >= 0)]
    last_day = int(rv["day"].max()) - 2
    days = {a: np.sort(g["day"].to_numpy()) for a, g in rv.groupby("app_id")}
    day0 = pd.Timestamp(config.REVIEW_HISTORY_START)
    rows = []
    for g in ("online", "local only", "single"):
        d = ev[ev["group"] == g]
        base = d[d["analysis_sample"]].dropna(subset=["y_during", "pre_trend"])
        b2 = base.dropna(subset=["y_wk3_4", "y_wk5_6", "r_net"]).reset_index(drop=True)
        r = V.att(b2, ["y_during", "y_wk3_4", "y_wk5_6", "r_net"])
        for y, lab in [
            ("y_during", "during"),
            ("y_wk3_4", "weeks 3-4 after"),
            ("y_wk5_6", "weeks 5-6 after"),
            ("r_net", "net revenue proxy"),
        ]:
            rows.append(
                dict(
                    part="B: main sample (ATT)",
                    outcome=lab,
                    bw=np.nan,
                    group=g,
                    est=r[y][0],
                    se=r[y][1],
                    n=len(b2),
                )
            )
        long = V.part_a(d, days, day0, last_day, t0)
        for x in long:
            if x["outcome"] == "weeks 7-12 after":
                rows.append(
                    dict(
                        part="B: main sample (ATT)",
                        outcome="weeks 7-12 after (Spring/Summer)",
                        bw=np.nan,
                        group=g,
                        est=x["est"],
                        se=x["se"],
                        n=x["n"],
                    )
                )
        print(f"  B: {g} done  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    return rows


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    groups = tag_groups()
    print(groups["group"].value_counts().to_string())
    rows = part_a(groups)
    pd.DataFrame(rows).to_csv(OUT / "network.csv", index=False)
    print(f"  A done  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    rows += part_b(groups, t0)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "network.csv", index=False)
    res["est (se)"] = res.apply(lambda x: f"{x['est']:+.3f} ({x['se']:.3f})", axis=1)
    pd.set_option("display.width", 220)
    print(
        res.pivot_table(
            index=["part", "outcome"],
            columns=["group", "bw"],
            values="est (se)",
            aggfunc="first",
        ).T.to_string()
    )


if __name__ == "__main__":
    main()
