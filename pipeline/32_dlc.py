"""Step 32: DLC. Does a base-game discount sell DLC, and do the main results hold without it?

Revenue the review proxy misses: DLC bought by players a discount brings in. Data from
03k_pull_dlc.py / 03l_pull_dlc_prices.py: each study game's most-reviewed DLC, its review
counts around each seasonal sale since 2024 (base: 21 to 8 days before; E: the sale; F:
weeks 1-2 after), and its price history (IsThereAnyDeal).

A. DLC spillover. Outcome: DLC reviews per two weeks in the sale (E) and in weeks 1-2
   after (F), log, relative to the base window. Doubly robust ATT of discounting the BASE
   game (main controls):
     clean   DLC at full price during the sale (so any lift comes through the base game)
     both    base and DLC both discounted vs neither
B. Main results by monetization: ATT on reviews during and the net revenue proxy, for
   games with no DLC, with DLC, and with in-app purchases (Steam's store category).

  python 32_dlc.py     (LEARNER / CF_REPEATS as in config)
Outputs: results/dlc/
"""

import config
import dml_core as M
import numpy as np
import pandas as pd
from prices import price_arrays, sale_times

OUT = config.ROOT.parent / "results" / "dlc"
COMPACT = config.DATA / "for_claude"


def att(df, ys):
    r = M.aipw(df.reset_index(drop=True), ys)[0]
    return {y: r[y]["ATT"] for y in ys}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    base = ev[ev["analysis_sample"]].dropna(subset=["y_during", "pre_trend"])
    info = pd.read_parquet(COMPACT / "dlc_info.parquet")
    rc = pd.read_parquet(COMPACT / "dlc_review_counts.parquet")
    rc = rc[rc["dlc"] != rc["app_id"]]  # a few lookups returned the game itself
    h = pd.read_parquet(COMPACT / "dlc_price_history.parquet")
    h["ts"] = pd.to_datetime(h["ts"], utc=True)
    arrays = price_arrays(h.sort_values(["app_id", "ts"]))
    cal = pd.read_csv(config.ROOT / "sale_calendar.csv")
    span = {r["sale"]: sale_times(r) for _, r in cal.iterrows()}

    def dlc_cut(r):
        pa = arrays.get(int(r.dlc))
        if pa is None:
            return np.nan
        a, b = span[r.sale]
        return pa.max_cut(int(a.timestamp()), int(b.timestamp()))

    rc = rc.assign(dlc_cut=[dlc_cut(r) for r in rc.itertuples()]).dropna(
        subset=["dlc_cut"]
    )
    d = base.merge(rc, on=["app_id", "sale"])
    n = d["sale_days"].astype(float)
    d["dlc_E"] = np.log1p(d["E"] * 14 / n) - np.log1p(d["base"])
    d["dlc_F"] = np.log1p(d["F"]) - np.log1p(d["base"])
    d["dlc_disc"] = d["dlc_cut"] > 0
    disc = d["discounted"].astype(bool)
    rows = []
    samples = {
        "clean: DLC at full price": d[~d["dlc_disc"]],
        "both discounted vs neither": d[
            (disc & d["dlc_disc"]) | (~disc & ~d["dlc_disc"])
        ],
    }
    for s, dd in samples.items():
        r = att(dd, ["dlc_E", "dlc_F", "y_during"])
        for y, lab in [
            ("dlc_E", "DLC reviews, during the sale"),
            ("dlc_F", "DLC reviews, weeks 1-2 after"),
            ("y_during", "base-game reviews, during (same sample)"),
        ]:
            rows.append(
                {
                    "part": "A: DLC spillover",
                    "sample": s,
                    "outcome": lab,
                    "ATT": r[y][0],
                    "se": r[y][1],
                    "n": len(dd),
                    "treated": int(dd["discounted"].astype(bool).sum()),
                }
            )
        print(f"  A: {s} done", flush=True)

    g = base.merge(info[["app_id", "n_dlc", "in_app"]], on="app_id", how="left")
    groups = {
        "no DLC": g[g["n_dlc"].fillna(0) == 0],
        "has DLC": g[g["n_dlc"].fillna(0) > 0],
        "in-app purchases": g[g["in_app"] == True],  # noqa: E712
        "no DLC and no in-app purchases": g[
            (g["n_dlc"].fillna(0) == 0) & (g["in_app"] != True)  # noqa: E712
        ],
    }
    for s, dd in groups.items():
        dd = dd.dropna(subset=["r_net"])
        r = att(dd, ["y_during", "r_net"])
        for y, lab in [
            ("y_during", "reviews, during"),
            ("r_net", "revenue proxy, net"),
        ]:
            rows.append(
                {
                    "part": "B: main results by monetization",
                    "sample": s,
                    "outcome": lab,
                    "ATT": r[y][0],
                    "se": r[y][1],
                    "n": len(dd),
                    "treated": int(dd["discounted"].astype(bool).sum()),
                }
            )
        print(f"  B: {s} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "dlc.csv", index=False)
    ct = pd.crosstab(d["discounted"].astype(bool), d["dlc_disc"])
    ct.to_csv(OUT / "dlc_discount_overlap.csv")
    pd.set_option("display.width", 200)
    print(ct)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
