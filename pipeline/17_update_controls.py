"""Step 17: do updates timed with sales explain the discount effects?

Developers often release patches and announcements during a sale. If games that discount
are also the ones updating, part of the "discount effect" could be the update. This uses
each game's news feed (03g_pull_news.py) to measure that.

Only the developer's own Steam announcements count (feed_type 1; outside press is left
out). Each is classified by its title and tags:
  sale     a post about the sale or discount itself ("50% off", "Summer Sale"). Caused by
           the discount, so never a control.
  update   patch notes (Steam's tag) or titles naming a patch, hotfix, update or version.
  other    every other announcement (new content, DLC, events, dev diaries).

Specifications (doubly robust ATT, same estimator and controls as the main results):
  main       the usual controls
  pre        + updates and other posts in the 14 days before the sale, and the
             developer's posting rate over the prior 90 days (valid pre-treatment controls)
  co         + updates and other posts during the sale. These can be chosen together with
             the discount, so this is a sensitivity check (it may absorb part of the effect)
and the effect separately for game-events with and without an update during the sale.

  python 17_update_controls.py
Outputs: results/update_controls/
"""

import re
import time

import config
import dml_core as M
import numpy as np
import pandas as pd
from prices import sale_times

OUT = config.ROOT.parent / "results" / "update_controls"
NEWS = config.DATA / "for_claude" / "news.parquet"
SALE = re.compile(r"\bsales?\b|% ?off|\bdiscount|\bdeals?\b|\bsave \d|-\d+ ?%", re.I)
UPDATE = re.compile(
    r"\bpatch|\bhotfix|\bupdate|\bversion\b|\bv\d+(\.\d+)+|\b\d+\.\d+\.\d+|\bbuild\b|\bbug ?fix",
    re.I,
)
OUTCOMES = {
    "y_during": "Reviews, during the sale",
    "y_net": "Reviews, net",
    "r_during": "Revenue proxy, during",
    "r_net": "Revenue proxy, net",
}
PRE = ["pre_update_14", "pre_other_14", "posts_per_month_90"]
CO = ["co_update", "co_other"]


def classify(news):
    d = news[news["feed_type"] == 1].copy()
    title = d["title"].fillna("")
    tags = d["tags"].fillna("")
    d["kind"] = np.where(
        title.str.contains(SALE),
        "sale",
        np.where(
            tags.str.contains("patchnotes") | title.str.contains(UPDATE),
            "update",
            "other",
        ),
    )
    d["date"] = pd.to_datetime(d["date"], utc=True)
    return d[["app_id", "date", "kind"]]


def event_features(ev, posts, cal):
    """Counts of each kind of post around each game-event (days relative to the sale's
    start time; the sale lasts sale_days)."""
    starts = {r["sale"]: sale_times(r)[0] for _, r in cal.iterrows()}
    by_app = {a: g for a, g in posts.groupby("app_id")}
    rows = []
    for r in ev[["app_id", "sale", "sale_days"]].itertuples(index=False):
        g = by_app.get(r.app_id)
        f = {"app_id": r.app_id, "sale": r.sale}
        if g is None:
            f.update({k: 0.0 for k in PRE + CO + ["co_sale", "pre_sale_14"]})
            rows.append(f)
            continue
        rel = (g["date"] - starts[r.sale]).dt.total_seconds().to_numpy() / 86400
        k = g["kind"].to_numpy()
        pre14, pre90 = (rel >= -14) & (rel < 0), (rel >= -90) & (rel < 0)
        dur = (rel >= 0) & (rel < r.sale_days)
        f.update(
            pre_update_14=float(((k == "update") & pre14).sum()),
            pre_other_14=float(((k == "other") & pre14).sum()),
            pre_sale_14=float(((k == "sale") & pre14).sum()),
            posts_per_month_90=float(((k != "sale") & pre90).sum()) / 3,
            co_update=float(((k == "update") & dur).sum()),
            co_other=float(((k == "other") & dur).sum()),
            co_sale=float(((k == "sale") & dur).sum()),
        )
        rows.append(f)
    return pd.DataFrame(rows)


def att(df, ycols, extra=()):
    """ATT for each outcome on the game-events where it is defined."""
    out = {}
    for y in ycols:
        d = df.dropna(subset=[y]).reset_index(drop=True)
        out[y] = M.aipw(d, [y], extra=list(extra))[0][y]["ATT"]
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    posts = classify(pd.read_parquet(NEWS))
    cal = pd.read_csv(config.ROOT / "sale_calendar.csv")
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    ev = (
        ev[ev["analysis_sample"]]
        .dropna(subset=["y_during", "pre_trend"])
        .reset_index(drop=True)
    )
    feats = event_features(ev, posts, cal)
    df = ev.merge(feats, on=["app_id", "sale"])
    for c in PRE + CO + ["co_sale", "pre_sale_14"]:
        df[c] = np.log1p(df[c])  # counts on a log scale; a few games post constantly
    print(
        f"{len(posts):,} developer posts ({posts['kind'].value_counts().to_dict()}); {len(df):,} game-events  [{time.time() - t0:.0f}s]",
        flush=True,
    )

    # how often updates coincide with sales, discounted vs. full-price game-events
    desc = (
        df.assign(
            update_during=df["co_update"] > 0,
            update_before=df["pre_update_14"] > 0,
            sale_post=(df["co_sale"] > 0) | (df["pre_sale_14"] > 0),
        )
        .groupby("discounted")[["update_during", "update_before", "sale_post"]]
        .mean()
        .rename(index={True: "discounted", False: "full price"})
    )
    desc.to_csv(OUT / "update_rates.csv")
    print(desc.round(3).to_string(), flush=True)

    rows = []
    for spec, extra in [("main", []), ("pre", PRE), ("co", PRE + CO)]:
        for y, (est, se) in att(df, [c for c in OUTCOMES if c in df], extra).items():
            rows.append(
                {
                    "spec": spec,
                    "group": "all",
                    "outcome": y,
                    "ATT": est,
                    "se": se,
                    "n": int(df[y].notna().sum()),
                }
            )
        print(f"  spec {spec} done  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    for name, m in [
        ("updated during the sale", df["co_update"] > 0),
        ("no update during the sale", df["co_update"] == 0),
    ]:
        d = df[m].reset_index(drop=True)
        for y, (est, se) in att(d, list(OUTCOMES), PRE).items():
            rows.append(
                {
                    "spec": "pre",
                    "group": name,
                    "outcome": y,
                    "ATT": est,
                    "se": se,
                    "n": int(d[y].notna().sum()),
                }
            )
        print(f"  group '{name}' done  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "effects.csv", index=False)
    res["ATT (SE)"] = res.apply(lambda r: f"{r['ATT']:+.3f} ({r['se']:.3f})", axis=1)
    pd.set_option("display.width", 200)
    print(
        res.pivot_table(
            index=["group", "spec"],
            columns="outcome",
            values="ATT (SE)",
            aggfunc="first",
        )[list(OUTCOMES)].to_string()
    )
    print(f"Done in {(time.time() - t0) / 60:.0f} minutes")


if __name__ == "__main__":
    main()
