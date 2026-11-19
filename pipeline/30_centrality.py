"""Step 30: do discounts by CENTRAL games spill over, when the average game's do not?

12_spillovers.py found no robust spillover from rivals' discounts on average. Network
theory says the structure matters: shocks to a few highly connected nodes spread much
further than shocks to the rest (Sargent and Stachurski, Economic Networks, ch. 1-2). Here
the network is the audience-rival graph of 11_segments.py: an edge i -> j when game j is
among study game i's 20 nearest games by audience position.

Centrality of each game j (computed on that graph):
  in-degree   number of study games that count j among their 20 nearest rivals
  Katz        authority-based Katz centrality, kappa = (I - alpha A')^-1 1 with alpha at
              half of 1 / spectral radius: j is central if central games point to it
For each study game i and sale, the share of its top-20 rivals that discounted is split
into central rivals (Katz above the median rival slot) and peripheral ones, and both enter
one partially linear double ML model (same controls and lagged exposures as 12):
holdouts (did not discount) and discounted games, outcome reviews during the sale.

If network structure matters, central rivals' discounts should move a game's reviews more
than peripheral rivals' (negative for holdouts if they steal buyers, positive if they bring
traffic). Also reported: a Katz-weighted exposure alongside the plain one.

  python 30_centrality.py
Outputs: results/centrality/
"""

import importlib
import time

import config
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, spsolve

S12 = importlib.import_module("12_spillovers")
OUT = config.ROOT.parent / "results" / "centrality"
TOP = 20


def centrality(edges):
    """In-degree and authority-based Katz centrality for every node in the edge list."""
    nodes = np.unique(np.concatenate([edges["app_id"], edges["rival"]]))
    idx = {n: k for k, n in enumerate(nodes)}
    r = edges["app_id"].map(idx).to_numpy()
    c = edges["rival"].map(idx).to_numpy()
    n = len(nodes)
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    try:
        lam = abs(
            eigs(A.T.astype(float), k=1, which="LM", return_eigenvectors=False)[0]
        )
    except Exception:  # noqa: BLE001
        lam = float(A.sum(axis=0).max())
    alpha = 0.5 / lam if lam > 0 else 0.5
    kappa = spsolve((sp.identity(n, format="csc") - alpha * A.T).tocsc(), np.ones(n))
    out = pd.DataFrame(
        {
            "app_id": nodes,
            "in_degree": np.asarray(A.sum(axis=0)).ravel(),
            "katz": kappa,
        }
    )
    return out, lam, alpha


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    riv = S12.audience_rivals()
    riv = riv[riv["rank"] <= TOP]
    cen, lam, alpha = centrality(riv)
    cen.to_csv(OUT / "centrality.csv", index=False)
    print(
        f"graph: {riv['app_id'].nunique():,} study games -> {riv['rival'].nunique():,} rivals, "
        f"{len(riv):,} edges; spectral radius {lam:.2f}, alpha {alpha:.4f}  [{time.time() - t0:.0f}s]",
        flush=True,
    )
    riv = riv.merge(cen.rename(columns={"app_id": "rival"}), on="rival")
    cut = riv["katz"].median()
    riv["central"] = riv["katz"] > cut

    cal = pd.read_csv(config.ROOT / "sale_calendar.csv")
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    cal = cal[cal["sale"].isin(ev["sale"].unique())].reset_index(drop=True)
    order = {s: k for k, s in enumerate(cal["sale"])}
    hist = pd.read_parquet(S12.COMPACT / "price_history.parquet")
    hist["ts"] = pd.to_datetime(hist["ts"], utc=True)
    hist = hist.sort_values(["app_id", "ts"])
    status = S12.discount_status(hist, cal)
    print(f"discount status built  [{time.time() - t0:.0f}s]", flush=True)

    expo = ev[["app_id", "sale"]].copy()
    parts = {
        "x_all": riv,
        "x_central": riv[riv["central"]],
        "x_peripheral": riv[~riv["central"]],
    }
    for col, rr in parts.items():
        e = S12.exposure(rr[["app_id", "rival", "rank"]], status).rename(
            columns={"x": col}
        )
        expo = expo.merge(e, on=["app_id", "sale"], how="left")
    # Katz-weighted share of rivals discounted
    rr = riv[riv["rival"].isin(status.index)]
    s = status.loc[rr["rival"].to_numpy()].to_numpy()
    w = rr["katz"].to_numpy()[:, None] * np.isfinite(s)
    num = pd.DataFrame(np.nan_to_num(s) * w, columns=status.columns).assign(
        app_id=rr["app_id"].to_numpy()
    )
    den = pd.DataFrame(w, columns=status.columns).assign(app_id=rr["app_id"].to_numpy())
    kw = (num.groupby("app_id").sum() / den.groupby("app_id").sum()).stack(
        future_stack=True
    )
    kw = kw.rename("x_katz_weighted").reset_index().rename(columns={"level_1": "sale"})
    expo = expo.merge(kw, on=["app_id", "sale"], how="left")
    cols = ["x_all", "x_central", "x_peripheral", "x_katz_weighted"]
    expo["k"] = expo["sale"].map(order)
    for col in cols:
        prev = expo[["app_id", "k", col]].copy()
        prev["k"] += 1
        expo = expo.merge(
            prev.rename(columns={col: f"lag_{col}"}), on=["app_id", "k"], how="left"
        )
    df = (
        ev[ev["analysis_sample"]]
        .dropna(subset=["y_during", "pre_trend"])
        .merge(expo, on=["app_id", "sale"])
        .dropna(subset=cols + [f"lag_{c}" for c in cols])
        .reset_index(drop=True)
    )
    holdouts = df[~df["discounted"].astype(bool)].reset_index(drop=True)
    disc = df[df["discounted"].astype(bool)].reset_index(drop=True)
    print(
        f"{len(df):,} game-events ({len(holdouts):,} holdouts); within-sale SD: "
        + ", ".join(
            f"{c} {(df[c] - df.groupby('sale')[c].transform('mean')).std():.3f}"
            for c in cols
        )
        + f"  [{time.time() - t0:.0f}s]",
        flush=True,
    )
    rows = []
    specs = [
        ("central vs peripheral", ["x_central", "x_peripheral"]),
        ("plain vs Katz-weighted", ["x_all", "x_katz_weighted"]),
        ("plain only", ["x_all"]),
    ]
    for samp, d in [("holdouts", holdouts), ("discounted", disc)]:
        for name, xs in specs:
            res = S12.plr_multi(d, "y_during", xs, extra=[f"lag_{c}" for c in xs])
            for x in xs:
                rows.append(S12.row(name, samp, "y_during", x[2:], *res[x], d))
        print(f"  {samp} done  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "centrality_spillovers.csv", index=False)
    pd.set_option("display.width", 200)
    res["est (se)"] = res.apply(
        lambda x: f"{x['estimate']:+.3f} ({x['se']:.3f})", axis=1
    )
    print(res[["test", "sample", "term", "est (se)", "n"]].to_string(index=False))
    print(cen.describe().round(3).to_string())


if __name__ == "__main__":
    main()
