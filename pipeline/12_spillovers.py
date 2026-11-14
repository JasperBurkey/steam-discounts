"""Step 12 (batch 8): spillovers. Do rivals' discounts affect a game?

Exposure for game i in sale s = share of i's rivals that discounted in s (any price cut in
force during the sale window, the same rule as for the sampled games). Rival definitions:
  audience   nearest eligible games by audience position (11_segments.py rivals)
  tags       nearest eligible games by tag similarity (TF-IDF cosine)
  tag_kmeans same k-means tag cluster (the MVP approach; whole cluster)
  tag_topic  same tag topic (LDA; whole topic)
Main question (defensive discounting): do HOLDOUTS (games that did not discount) lose
reviews when their rivals discount? Also: does a discounted game gain less when its
rivals discount too?

Estimation: partially linear double ML with continuous exposure(s), cross-fitted by game,
SEs clustered by game. Controls: the usual pre-event controls plus the same exposure in
the PREVIOUS sale (rivals' habitual discounting), so the variation used is rivals
deviating from their own habits.

Criteria for comparing rival definitions, fixed before any spillover result:
  1. placebo:     the real estimate vs. random rival sets of the same size
  2. gradient:    ranks 1-5, 6-20, 21-50 together; effects should fade with distance
  3. head-to-head: definitions in one model; which still matters
  4. out-of-sample: chosen on 2023-2024 sales, confirmed on 2025-2026
All definitions are reported.

  python 12_spillovers.py            (PLACEBO_DRAWS=40 by default)
Outputs (output/spillovers/): exposures.parquet, main.csv, placebo.csv, gradient.csv,
  head_to_head.csv, out_of_sample.csv, exposure_variation.csv
"""

import importlib
import os
import time

import config
import dml_core as M
import numpy as np
import pandas as pd
from hetero import _cluster_ols
from prices import load_history, price_arrays, sale_times

OUT = config.OUTPUT / "spillovers"
COMPACT = config.DATA / "for_claude"
SEG_DIR = next(
    (
        p
        for p in [
            config.OUTPUT / "segments",
            config.ROOT.parent / "results" / "segments",
        ]
        if (p / "rivals.parquet").exists()
    ),
    config.OUTPUT / "segments",
)
BANDS = {"1-5": (1, 5), "6-20": (6, 20), "21-50": (21, 50)}
MAIN_BAND = (1, 20)
PLACEBO_DRAWS = int(os.environ.get("PLACEBO_DRAWS", 40))
SEED = 0
EARLY = ("2023", "2024")


# ---------------------------------------------------------------- rival discount status
def discount_status(hist, cal):
    """Matrix: eligible game x sale -> 1 discounted, 0 not, NaN unknown."""
    arrays = price_arrays(hist)
    apps = np.array(sorted(arrays))
    out = np.full((len(apps), len(cal)), np.nan)
    for k, (_, row) in enumerate(cal.iterrows()):
        a, b = (int(x.timestamp()) for x in sale_times(row))
        for i, app in enumerate(apps):
            c = arrays[app].max_cut(a, b)
            if np.isfinite(c):
                out[i, k] = float(c > 0)
    return pd.DataFrame(out, index=apps, columns=cal["sale"].tolist())


# ---------------------------------------------------------------- rival sets
def audience_rivals():
    r = pd.read_parquet(SEG_DIR / "rivals.parquet")
    return r[["app_id", "rival", "rank"]]


def tag_rivals(focal, pool, info, n=50):
    seg = importlib.import_module("11_segments")
    games = np.array(sorted(set(focal) | set(pool)))
    t, _, ok = seg.tag_vectors(games, info)
    t = t.astype(np.float32)
    idx = {g: i for i, g in enumerate(games)}
    pool_idx = np.array([idx[g] for g in pool if ok[idx[g]]])
    recs = []
    for a in focal:
        i = idx[a]
        if not ok[i]:
            continue
        sim = t[pool_idx] @ t[i]
        sim[games[pool_idx] == a] = -np.inf
        top = np.argpartition(-sim, n)[:n]
        top = top[np.argsort(-sim[top])]
        recs += [(a, int(games[pool_idx[j]]), r + 1) for r, j in enumerate(top)]
    return pd.DataFrame(recs, columns=["app_id", "rival", "rank"])


def cluster_rivals(focal, labels):
    """Whole-cluster rivals: every other game with the focal game's label (rank = 1)."""
    lab = labels.set_index("app_id")
    recs = []
    members = lab.groupby(lab.columns[0]).groups
    for a in focal:
        if a not in lab.index:
            continue
        for g in members[lab.loc[a].iloc[0]]:
            if g != a:
                recs.append((a, int(g), 1))
    return pd.DataFrame(recs, columns=["app_id", "rival", "rank"])


def exposure(rivals, status, lo=1, hi=10**9):
    """Share of rivals (rank lo..hi) discounted in each sale; NaN if none known."""
    r = rivals[(rivals["rank"] >= lo) & (rivals["rank"] <= hi)]
    r = r[r["rival"].isin(status.index)]
    s = status.loc[r["rival"].to_numpy()].to_numpy()
    df = pd.DataFrame(s, columns=status.columns)
    df["app_id"] = r["app_id"].to_numpy()
    mean = df.groupby(
        "app_id"
    ).mean()  # NaN-aware: share among rivals with known status
    return (
        mean.stack(future_stack=True)
        .rename("x")
        .reset_index()
        .rename(columns={"level_1": "sale"})
    )


# ---------------------------------------------------------------- estimation
def plr_multi(df, y, ds, extra=(), repeats=None):
    """Partially linear model with several continuous treatments ds (residual on residual,
    jointly). Returns {d: (estimate, se)}."""
    x = M.features(df, extra)
    g = df["app_id"].to_numpy()
    yv = df[y].to_numpy(float)
    results = {d: [] for d in ds}
    for r in range(repeats or config.CF_REPEATS):
        f = M.game_folds(g, config.SEED + r)
        yr = yv - M.crossfit(x, yv, f)
        dr = np.column_stack(
            [
                df[d].to_numpy(float) - M.crossfit(x, df[d].to_numpy(float), f)
                for d in ds
            ]
        )
        b, se = _cluster_ols(yr, dr, g)
        for k, d in enumerate(ds):
            results[d].append((b[k], se[k]))
    return {d: M.median_combine(*zip(*v)) for d, v in results.items()}


def row(test, sample, outcome, term, est, se, df):
    return {
        "test": test,
        "sample": sample,
        "outcome": outcome,
        "term": term,
        "estimate": est,
        "se": se,
        "ci_low": est - 1.96 * se,
        "ci_high": est + 1.96 * se,
        "n": len(df),
        "games": df["app_id"].nunique(),
    }


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cal = pd.read_csv(config.ROOT / "sale_calendar.csv")
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    cal = cal[cal["sale"].isin(ev["sale"].unique())].reset_index(drop=True)
    order = {s: k for k, s in enumerate(cal["sale"])}
    if config.PRICE_HISTORY_PARQUET.exists():
        hist = load_history()
    else:  # uploaded export
        hist = pd.read_parquet(COMPACT / "price_history.parquet")
        hist["ts"] = pd.to_datetime(hist["ts"], utc=True)
        hist = hist.sort_values(["app_id", "ts"])
    status = discount_status(hist, cal)
    print(
        f"Rival discount status: {len(status):,} eligible games x {status.shape[1]} sales; "
        f"discounted share per sale {np.nanmean(status.to_numpy()):.0%}  [{time.time() - t0:.0f}s]"
    )
    info = (
        pd.read_parquet(COMPACT / "game_info.parquet")
        if (COMPACT / "game_info.parquet").exists()
        else importlib.import_module("export_for_claude").game_info(set(status.index))
    )
    focal = sorted(ev["app_id"].unique())
    pool = sorted(status.index)

    rival_sets = {"audience": audience_rivals(), "tags": tag_rivals(focal, pool, info)}
    tb = pd.read_csv(SEG_DIR / "tag_baselines.csv")
    rival_sets["tag_kmeans"] = cluster_rivals(focal, tb[["app_id", "tag_kmeans"]])
    rival_sets["tag_topic"] = cluster_rivals(focal, tb[["app_id", "tag_topic"]])
    print(f"Rival sets built  [{time.time() - t0:.0f}s]")

    # exposures: main band, gradient bands, and previous-sale (lagged) controls
    expo = ev[["app_id", "sale"]].copy()
    cols = {}
    for name, rv in rival_sets.items():
        if name in ("audience", "tags"):
            specs = {"": MAIN_BAND, **{f"_{b}": v for b, v in BANDS.items()}}
        else:
            specs = {"": (1, 1)}
        for suffix, (lo, hi) in specs.items():
            col = f"x_{name}{suffix}"
            e = exposure(rv, status, lo, hi).rename(columns={"x": col})
            expo = expo.merge(e, on=["app_id", "sale"], how="left")
            cols[col] = (name, suffix)
    expo["k"] = expo["sale"].map(order)
    for col in list(cols):
        prev = expo[["app_id", "k", col]].copy()
        prev["k"] = prev["k"] + 1
        expo = expo.merge(
            prev.rename(columns={col: f"lag_{col}"}), on=["app_id", "k"], how="left"
        )
    expo.to_parquet(OUT / "exposures.parquet", index=False)

    df = (
        ev[ev["analysis_sample"]]
        .dropna(subset=["y_during", "pre_trend"])
        .merge(expo, on=["app_id", "sale"])
    )
    main_cols = ["x_audience", "x_tags", "x_tag_kmeans", "x_tag_topic"]
    df = df.dropna(subset=main_cols + [f"lag_{c}" for c in main_cols]).reset_index(
        drop=True
    )
    holdouts = df[~df["discounted"]].reset_index(drop=True)
    discounted = df[df["discounted"]].reset_index(drop=True)
    print(
        f"Analysis: {len(df):,} game-events with every exposure and its lag "
        f"({len(holdouts):,} holdouts, {len(discounted):,} discounted)  [{time.time() - t0:.0f}s]"
    )

    # how much exposure varies within a sale (no variation = no power)
    var = []
    for c in main_cols:
        resid = df[c] - df.groupby("sale")[c].transform("mean")
        var.append(
            {
                "definition": c[2:],
                "mean": df[c].mean(),
                "sd_within_sale": resid.std(),
                "corr_with_lag": df[c].corr(df[f"lag_{c}"]),
            }
        )
    pd.DataFrame(var).to_csv(OUT / "exposure_variation.csv", index=False)

    rows = []
    # main: each definition alone (top 20 rivals, or whole cluster)
    for samp, d in [("holdouts", holdouts), ("discounted", discounted)]:
        for c in main_cols:
            res = plr_multi(d, "y_during", [c], extra=[f"lag_{c}"])
            rows.append(row("main", samp, "y_during", c[2:], *res[c], d))
        print(
            f"  main estimates for {samp} done  [{time.time() - t0:.0f}s]", flush=True
        )
    main = pd.DataFrame(rows)
    main.to_csv(OUT / "main.csv", index=False)

    # 1. placebo: random rival sets of the same size (20), fixed per game across sales
    rng = np.random.default_rng(SEED)
    learner, config.LEARNER = (
        config.LEARNER,
        "boost",
    )  # cheaper learner for the many draws
    # real estimates with the same learner and one repetition, so the comparison is like for like
    real = {
        c[2:]: plr_multi(holdouts, "y_during", [c], extra=[f"lag_{c}"], repeats=1)[c][0]
        for c in ["x_audience", "x_tags"]
    }
    pool_arr = np.array(pool)
    placebo = []
    for b in range(PLACEBO_DRAWS):
        recs = [
            (a, int(g), r + 1)
            for a in focal
            for r, g in enumerate(rng.choice(pool_arr, 20, replace=False))
        ]
        rv = pd.DataFrame(recs, columns=["app_id", "rival", "rank"])
        e = exposure(rv, status).rename(columns={"x": "x_placebo"})
        e["k"] = e["sale"].map(order)
        lag = (
            e[["app_id", "k", "x_placebo"]]
            .assign(k=lambda t: t["k"] + 1)
            .rename(columns={"x_placebo": "lag_x_placebo"})
        )
        d = holdouts.drop(
            columns=["x_placebo", "lag_x_placebo"], errors="ignore"
        ).merge(
            e[["app_id", "sale", "x_placebo", "k"]].drop(columns="k"),
            on=["app_id", "sale"],
            how="left",
        )
        d = (
            d.merge(lag, on=["app_id", "k"], how="left")
            .dropna(subset=["x_placebo", "lag_x_placebo"])
            .reset_index(drop=True)
        )
        est, se = plr_multi(
            d, "y_during", ["x_placebo"], extra=["lag_x_placebo"], repeats=1
        )["x_placebo"]
        placebo.append({"draw": b, "estimate": est, "se": se})
        if (b + 1) % 10 == 0:
            print(
                f"  placebo draws {b + 1}/{PLACEBO_DRAWS}  [{time.time() - t0:.0f}s]",
                flush=True,
            )
    config.LEARNER = learner
    placebo = pd.DataFrame(placebo)
    placebo.to_csv(OUT / "placebo.csv", index=False)
    plac_summary = pd.DataFrame(
        [
            {
                "definition": t,
                "estimate": real[t],
                "placebo_mean": placebo["estimate"].mean(),
                "placebo_sd": placebo["estimate"].std(),
                # two-sided placebo p-value: share of random rival sets with an effect at least as large
                "placebo_p": float((placebo["estimate"].abs() >= abs(real[t])).mean()),
            }
            for t in ["audience", "tags"]
        ]
    )
    plac_summary.to_csv(OUT / "placebo_summary.csv", index=False)

    # 2. gradient: rank bands together, per definition
    grad = []
    for name in ["audience", "tags"]:
        ds = [f"x_{name}_{b}" for b in BANDS]
        d = holdouts.dropna(subset=ds + [f"lag_{c}" for c in ds]).reset_index(drop=True)
        res = plr_multi(d, "y_during", ds, extra=[f"lag_{c}" for c in ds])
        grad += [row("gradient", "holdouts", "y_during", c[2:], *res[c], d) for c in ds]
    pd.DataFrame(grad).to_csv(OUT / "gradient.csv", index=False)

    # 3. head-to-head: all definitions in one model
    res = plr_multi(
        holdouts, "y_during", main_cols, extra=[f"lag_{c}" for c in main_cols]
    )
    h2h = pd.DataFrame(
        [
            row("head_to_head", "holdouts", "y_during", c[2:], *res[c], holdouts)
            for c in main_cols
        ]
    )
    h2h.to_csv(OUT / "head_to_head.csv", index=False)

    # 4. out-of-sample: 2023-2024 sales vs 2025-2026 sales
    oos = []
    for label, m in [
        ("2023-2024", holdouts["sale"].str.contains("|".join(EARLY))),
        ("2025-2026", ~holdouts["sale"].str.contains("|".join(EARLY))),
    ]:
        d = holdouts[m].reset_index(drop=True)
        for c in main_cols:
            r = plr_multi(d, "y_during", [c], extra=[f"lag_{c}"])
            oos.append(row(f"period {label}", "holdouts", "y_during", c[2:], *r[c], d))
    pd.DataFrame(oos).to_csv(OUT / "out_of_sample.csv", index=False)

    pd.set_option("display.width", 200)

    def show(t):
        t = t.copy()
        t["estimate (se)"] = t.apply(
            lambda r: f"{r['estimate']:+.3f} ({r['se']:.3f})", axis=1
        )
        return t[["test", "sample", "term", "estimate (se)", "n", "games"]].to_string(
            index=False
        )

    print("\nExposure variation (share of rivals discounting):")
    print(pd.DataFrame(var).round(3).to_string(index=False))
    print(
        "\nMain: effect on reviews during the sale of ALL rivals discounting vs. none (log scale)"
    )
    print(show(main))
    print("\n1. Placebo (holdouts): real estimate vs. random 20-game rival sets")
    print(plac_summary.round(3).to_string(index=False))
    print("\n2. Gradient (holdouts): rank bands in one model")
    print(show(pd.DataFrame(grad)))
    print("\n3. Head-to-head (holdouts): all definitions in one model")
    print(show(h2h))
    print("\n4. Out-of-sample (holdouts): by period")
    print(show(pd.DataFrame(oos)))
    print(f"\nDone in {(time.time() - t0) / 60:.0f} minutes. Outputs in {OUT}")


if __name__ == "__main__":
    main()
