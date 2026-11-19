"""Step 21: does sale inertia vary by publisher?

20_rejoin.py found that publishers rejoin almost every seasonal sale at the same depth,
barely responding to how the last sale went. Here the same behavior is split by publisher:
  catalog size   the publisher's number of games on Steam (game_info universe)
  experience     years since the publisher's first Steam release, at the sale
  self-published developer and publisher are the same
For discounted game-events with a following sale:
  stickiness     rejoin rate, share using exactly the same cut, share changing the cut by
                 10+ points, mean absolute change
  responsiveness change in cut (next - this) on the last sale's spike (r_during) and net
                 (r_net), standardized, with sale FE, by group; SE clustered by publisher.
If inertia is a rule of thumb, smaller and newer publishers should be stickier and less
responsive; if bigger publishers are just as sticky, a missing benefit (e.g. visibility)
or a stable optimum is the likelier reason.

  python 21_inertia.py
Outputs: results/inertia/
"""

import importlib
import json

import config
import numpy as np
import pandas as pd

R = importlib.import_module("20_rejoin")
OUT = config.ROOT.parent / "results" / "inertia"
INFO = config.DATA / "for_claude" / "game_info.parquet"


def first(s):
    try:
        v = json.loads(s) if isinstance(s, str) else None
    except ValueError:
        return None
    return v[0] if v else None


def adjusted(j):
    """Same-cut and 10+ point change rates by publisher group relative to the first group,
    holding game age, price, this cut, review rate and the sale fixed (linear probability,
    SE clustered by publisher)."""
    d = j.dropna(subset=["age_days", "regular_usd", "pre_rate_60"]).reset_index(
        drop=True
    )
    d["log_age_z"] = np.log1p(d["age_days"].clip(lower=0))
    d["log_price_z"] = np.log(d["regular_usd"].clip(lower=0.5))
    d["cut_bin"] = (d["cut"] // 10).astype(int)
    out = []
    for col in ["size_group", "exp_group", "self_group"]:
        dums = pd.get_dummies(d[col], drop_first=True, dtype=float)
        names = [f"g_{i}" for i in range(dums.shape[1])]
        dd = pd.concat([d, dums.set_axis(names, axis=1)], axis=1)
        for y in ["same", "big"]:
            r = R.ols(
                dd,
                y,
                names + ["log_age_z", "log_price_z", "pre_rate_60_z"],
                ["sale", "cut_bin"],
                cluster="pub",
            )
            for n, lab in zip(names, dums.columns):
                out.append(
                    {
                        "split": col.removesuffix("_group"),
                        "vs first group": lab,
                        "outcome": {"same": "same cut", "big": "change 10+ pts"}[y],
                        "diff": r.loc[n, "coef"],
                        "se": r.loc[n, "se"],
                        "n": len(dd),
                    }
                )
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gi = pd.read_parquet(INFO)
    gi = gi[gi["type"].astype(str) == "0"].copy()
    gi["pub"] = gi["publishers"].map(first)
    gi["dev"] = gi["developers"].map(first)
    gi["rel"] = pd.to_datetime(gi["steam_release_date"], utc=True, errors="coerce")
    cat = gi.groupby("pub").size()
    pub_first = gi.groupby("pub")["rel"].min()

    ev = pd.read_parquet(config.EVENTS_PARQUET).sort_values(["app_id", "event_start"])
    g = ev.groupby("app_id")
    ev["next_disc"] = g["discounted"].shift(-1)
    ev["next_cut"] = g["cut"].shift(-1)
    ev["next_known"] = g["price_known"].shift(-1)
    ev = ev.merge(
        gi[["appid", "pub", "dev"]], left_on="app_id", right_on="appid", how="left"
    )
    ev["catalog"] = ev["pub"].map(cat)
    ev["pub_years"] = (
        pd.to_datetime(ev["event_start"], utc=True) - ev["pub"].map(pub_first)
    ).dt.days / 365.25
    ev["self_pub"] = ev["pub"] == ev["dev"]
    s = ev[
        (ev["discounted"] == 1)
        & ev["price_known"]
        & (ev["next_known"] == True)  # noqa: E712
        & ~ev["free"]
        & ~ev["too_new"]
        & ev["pub"].notna()
    ].copy()
    s["next_disc"] = s["next_disc"].astype(float)
    s["size_group"] = pd.cut(
        s["catalog"],
        [0, 1, 4, 19, 1e9],
        labels=["1 game", "2-4 games", "5-19 games", "20+ games"],
    )
    s["exp_group"] = pd.cut(
        s["pub_years"],
        [-1, 3, 8, 1e9],
        labels=["under 3 years", "3-8 years", "8+ years"],
    )
    s["self_group"] = np.where(s["self_pub"], "self-published", "outside publisher")

    stick = []
    for col in ["size_group", "exp_group", "self_group"]:
        for k, d in s.groupby(col, observed=True):
            j = d[d["next_disc"] == 1].dropna(subset=["next_cut"])
            ch = (j["next_cut"] - j["cut"]).abs()
            stick.append(
                {
                    "split": col.removesuffix("_group"),
                    "group": k,
                    "game-events": len(d),
                    "games": d["app_id"].nunique(),
                    "publishers": d["pub"].nunique(),
                    "rejoin": d["next_disc"].mean(),
                    "same cut": (ch < 0.5).mean(),
                    "change 10+ pts": (ch >= 10).mean(),
                    "mean abs change": ch.mean(),
                }
            )
    stick = pd.DataFrame(stick)

    j = s[s["next_disc"] == 1].dropna(subset=["next_cut", "r_during", "r_net"]).copy()
    j["dcut"] = j["next_cut"] - j["cut"]
    for c in ["r_during", "r_net", "pre_rate_60", "pre_trend", "cut"]:
        j[c + "_z"] = (j[c] - j[c].mean()) / j[c].std()
    resp = []
    for col in ["size_group", "exp_group", "self_group"]:
        for k, d in j.groupby(col, observed=True):
            d = d.dropna(subset=["pre_rate_60", "pre_trend"]).reset_index(drop=True)
            xs = ["r_during_z", "r_net_z", "pre_rate_60_z", "pre_trend_z", "cut_z"]
            r = R.ols(d, "dcut", xs, ["sale"], cluster="pub")
            for v in ["r_during_z", "r_net_z"]:
                resp.append(
                    {
                        "split": col.removesuffix("_group"),
                        "group": k,
                        "term": v.removesuffix("_z"),
                        "coef": r.loc[v, "coef"],
                        "se": r.loc[v, "se"],
                        "n": len(d),
                    }
                )
    resp = pd.DataFrame(resp)
    adj = adjusted(
        j.assign(
            same=(j["dcut"].abs() < 0.5).astype(float),
            big=(j["dcut"].abs() >= 10).astype(float),
        )
    )
    adj.to_csv(OUT / "stickiness_adjusted.csv", index=False)
    stick.to_csv(OUT / "stickiness.csv", index=False)
    resp.to_csv(OUT / "responsiveness.csv", index=False)
    pd.set_option("display.width", 200)
    print(stick.round(3).to_string(index=False))
    print(adj.round(3).to_string(index=False))
    resp["coef (se)"] = resp.apply(
        lambda x: f"{x['coef']:+.3f} ({x['se']:.3f})", axis=1
    )
    print(
        resp.pivot_table(
            index=["split", "group"],
            columns="term",
            values="coef (se)",
            aggfunc="first",
        ).to_string()
    )


if __name__ == "__main__":
    main()
