"""Step 28: what do discount routines cost, and who pays?

Policy learning (13_policy.py) found a rule that changes 14% of choices and raises the net
revenue proxy. Here its out-of-fold, game-event level gains (doubly robust score of the
rule's depth minus the score of the depth actually chosen; output/policy/event_gains.parquet)
are broken down by the signs of routine found in 20, 21 and 26:

  habitual        propensity to discount above 0.9 (cross-fitted, the main controls)
  same depth      same depth as the game's previous sale
  uniform         the publisher gave all its discounted games the same depth in this sale
                  (26_uniformity.py, output/uniform_groups.parquet)
  catalog size    1 / 2-4 / 5-19 / 20+ games on Steam; self-published

For each group: share of game-events, share the rule changes, mean gain (SE clustered by
game) and the group's share of the total gain. Rule: the 2-level tree learned on the
net revenue proxy, the one that passed the criterion; evaluated on the proxy and on the
lag-adjusted proxy.

Gains are estimated per game-event from doubly robust scores, which are noisy one by one;
group means are what is reliable.

  python 28_routine_costs.py
Outputs: results/routine_costs/
"""

import json

import config
import dml_core as M
import numpy as np
import pandas as pd

OUT = config.ROOT.parent / "results" / "routine_costs"
COMPACT = config.DATA / "for_claude"
RULE = "tree, 2 levels [learned on proxy]"
OBJ = {
    "Revenue proxy, net": "proxy",
    "Revenue proxy, net, lag-adjusted": "lag-adjusted",
}


def first(s):
    try:
        v = json.loads(s) if isinstance(s, str) else None
    except ValueError:
        return None
    return v[0] if v else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pg = pd.read_parquet(config.OUTPUT / "policy" / "event_gains.parquet")
    pg = pg[
        (pg["sample"] == "all game-events")
        & (pg["policy"] == RULE)
        & pg["objective"].isin(OBJ)
    ]
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    base = ev[ev["analysis_sample"]].dropna(subset=["y_during", "pre_trend"])
    base = base.reset_index(drop=True)
    e = M.crossfit(
        M.features(base),
        base["discounted"].astype(int).to_numpy(),
        M.game_folds(base["app_id"].to_numpy(), config.SEED),
        binary=True,
    )
    base = base.assign(e=e).sort_values(["app_id", "event_start"])
    base["prev_cut_ev"] = base.groupby("app_id")["cut"].shift(1)
    gi = pd.read_parquet(COMPACT / "game_info.parquet")
    gi = gi[gi["type"].astype(str) == "0"].copy()
    gi["pub"] = gi["publishers"].map(first)
    gi["dev"] = gi["developers"].map(first)
    gi["catalog"] = gi["pub"].map(gi.groupby("pub").size())
    base = base.merge(
        gi[["appid", "pub", "dev", "catalog"]],
        left_on="app_id",
        right_on="appid",
        how="left",
    )
    uni = config.OUTPUT / "uniform_groups.parquet"
    if uni.exists():
        u = pd.read_parquet(uni)[["app_id", "sale", "identical"]]
        base = base.merge(u, on=["app_id", "sale"], how="left")
    else:
        base["identical"] = np.nan

    groups = {
        "habitual (propensity > 0.9)": lambda d: np.where(
            d["e"] > 0.9, "habitual", "could go either way (<= 0.9)"
        ),
        "depth vs previous sale": lambda d: np.select(
            [
                d["prev_cut_ev"].isna(),
                (d["cut"].round() == d["prev_cut_ev"].round()),
            ],
            ["no previous sale", "same depth"],
            "different depth",
        ),
        "publisher's depths this sale": lambda d: np.select(
            [d["identical"] == True, d["identical"] == False],  # noqa: E712
            ["all its games the same", "differ"],
            "not applicable (one game or none discounted)",
        ),
        "publisher catalog": lambda d: pd.cut(
            d["catalog"],
            [0, 1, 4, 19, 1e9],
            labels=["1 game", "2-4", "5-19", "20+"],
        ).astype(str),
        "self-published": lambda d: np.where(
            d["pub"].isna(),
            "unknown",
            np.where(d["pub"] == d["dev"], "self-published", "outside publisher"),
        ),
    }
    rows = []
    for obj, short in OBJ.items():
        p = pg[pg["objective"] == obj].merge(
            base.drop(columns=["sale_days"], errors="ignore"),
            on=["app_id", "sale"],
            how="inner",
        )
        p["changed"] = p["assigned"] != p["chosen"]
        total = p["gain"].sum()
        rows.append(
            {
                "objective": short,
                "split": "all",
                "group": "all",
                "share of game-events": 1.0,
                "share changed": p["changed"].mean(),
                "mean gain": p["gain"].mean(),
                "se": M.cluster_se(
                    p["gain"].to_numpy() - p["gain"].mean(),
                    p["app_id"].to_numpy(),
                    len(p),
                ),
                "share of total gain": 1.0,
                "n": len(p),
            }
        )
        for split, f in groups.items():
            p["_g"] = f(p)
            for k, q in p.groupby("_g"):
                gv = q["gain"].to_numpy()
                rows.append(
                    {
                        "objective": short,
                        "split": split,
                        "group": k,
                        "share of game-events": len(q) / len(p),
                        "share changed": q["changed"].mean(),
                        "mean gain": gv.mean(),
                        "se": M.cluster_se(
                            gv - gv.mean(), q["app_id"].to_numpy(), len(q)
                        ),
                        "share of total gain": gv.sum() / total if total else np.nan,
                        "n": len(q),
                    }
                )
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "routine_costs.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
