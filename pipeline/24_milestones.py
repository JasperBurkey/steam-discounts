"""Step 24: copies sold per review, from developers' own "X copies sold" announcements.

The review proxy assumes reviews scale with buyers. Developers often announce sales
milestones in their Steam news ("1 million copies sold - thank you!"). Matching each claim
to the game's cumulative count of Steam-purchaser reviews on that date gives copies per
review, the one public link between the proxy and units sold.

Steps
  1. Extract claims from developer posts (feed_type 1) in news.parquet: a number followed
     by copies / units / sales, with sale-claim wording nearby, dropping physical editions,
     giveaways, keys, wishlists, DLC and soundtracks. One claim per post (title first).
     Flags: "steam" (says on Steam), "multi" (mentions consoles, all platforms or
     worldwide), "other title" (mentions a series, sequel, predecessor or another game).
  2. Cumulative purchaser reviews at each claim date:
       exact      from milestone_counts.parquet (03i_pull_milestone_counts.py), if present
       otherwise  games with complete review histories: Steam's current total (game_info
                  review_count, which matches purchaser reviews) minus purchaser reviews
                  after the date.
  3. Ratios. Level: copies / cumulative reviews. Interval: between two claims for the same
     game, new copies / new reviews, and whether the ratio differs when the interval
     contains more seasonal-sale days (sale buyers reviewing at a different rate).
Claims are rounded lower bounds ("over 1 million"), so ratios are slightly understated;
claims including consoles overstate them. Both are reported separately.

  python 24_milestones.py
Outputs: results/milestones/
"""

import re

import config
import numpy as np
import pandas as pd
from prices import sale_times

OUT = config.ROOT.parent / "results" / "milestones"
COMPACT = config.DATA / "for_claude"
NUM = r"(\d{1,3}(?:[,\.]\d{3})+|\d+(?:\.\d+)?)\s*(k|thousand|million|mil|m)?\+?"
PAT = re.compile(NUM + r"\s*(?:copies|units|sales)\b", re.I)
BAD = re.compile(
    r"limited|giveaway|only \d|print run|physical|edition of|available|win |keys?\b|"
    r"wishlist|dlc|soundtrack|bundle",
    re.I,
)
SOLD = re.compile(
    r"sold|sales|thank|celebrat|milestone|reached|crossed|surpass|passed|over|more than",
    re.I,
)
MULTI = re.compile(
    r"all platforms|console|switch|playstation|xbox|ps4|ps5|worldwide|across|mobile|ios|android",
    re.I,
)
OTHER = re.compile(
    r"series|franchise|sequel|predecessor|previous game|first game|original game|our games|"
    r"studio's|combined|trilogy|collection",
    re.I,
)
MULT = {"k": 1e3, "thousand": 1e3, "million": 1e6, "mil": 1e6, "m": 1e6}


def parse(x, unit):
    if re.fullmatch(r"\d{1,3}(?:[,\.]\d{3})+", x):
        v = float(re.sub(r"[,\.]", "", x))
    else:
        v = float(x.replace(",", "."))
    return v * MULT.get((unit or "").lower(), 1)


def extract(news):
    rows = []
    dev = news[news["feed_type"] == 1]
    for r in dev.itertuples(index=False):
        for field in ("title", "contents"):
            s = getattr(r, field) or ""
            hit = None
            for m in PAT.finditer(s[:4000]):
                ctx = s[max(0, m.start() - 80) : m.end() + 60]
                if BAD.search(ctx) or not SOLD.search(ctx):
                    continue
                v = parse(m.group(1), m.group(2))
                if v >= 500:
                    hit = (v, ctx.replace("\n", " "))
                    break
            if hit:
                v, ctx = hit
                rows.append(
                    {
                        "app_id": r.app_id,
                        "date": r.date,
                        "copies": v,
                        "field": field,
                        "steam": bool(re.search(r"on steam", ctx, re.I)),
                        "multi": bool(MULTI.search(ctx)),
                        "other_title": bool(OTHER.search(ctx)),
                        "context": ctx,
                    }
                )
                break
    c = pd.DataFrame(rows)
    c["date"] = pd.to_datetime(c["date"], utc=True)
    return c.drop_duplicates(["app_id", "copies"], keep="first").sort_values(
        ["app_id", "date"]
    )


def sale_days_between(a, b, sales):
    """Seasonal-sale days in [a, b)."""
    tot = 0.0
    for s, e in sales:
        lo, hi = max(a, s), min(b, e)
        if hi > lo:
            tot += (hi - lo).total_seconds() / 86400
    return tot


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    news = pd.read_parquet(COMPACT / "news.parquet")
    news["date"] = pd.to_datetime(
        news["date"], utc=True, unit="s" if news["date"].dtype.kind in "iuf" else None
    )
    c = extract(news)
    c.to_csv(OUT / "claims.csv", index=False)
    print(f"{len(c)} claims from {c['app_id'].nunique()} games")

    gi = pd.read_parquet(COMPACT / "game_info.parquet")
    gi = gi.set_index("appid")
    exact = COMPACT / "milestone_counts.parquet"
    if exact.exists():
        mc = pd.read_parquet(exact)
        c = c.merge(
            mc[["app_id", "date", "reviews_to_date"]], on=["app_id", "date"], how="left"
        )
        c["source"] = np.where(c["reviews_to_date"].notna(), "exact count", None)
    else:
        c["reviews_to_date"] = np.nan
        c["source"] = None
    rv = pd.read_parquet(
        COMPACT / "reviews_compact.parquet",
        columns=["app_id", "day", "steam_purchase", "sampled"],
    )
    samp = rv.groupby("app_id")["sampled"].any()
    full = rv[
        rv["app_id"].isin(samp[~samp].index)
        & rv["steam_purchase"].fillna(False).astype(bool)
    ]
    day0 = pd.Timestamp(config.REVIEW_HISTORY_START, tz="UTC")
    pdays = {a: np.sort(g["day"].to_numpy()) for a, g in full.groupby("app_id")}
    for i, r in c[c["reviews_to_date"].isna()].iterrows():
        if r["app_id"] not in pdays or r["date"] < day0:
            continue
        d = (r["date"] - day0).days
        after = len(pdays[r["app_id"]]) - np.searchsorted(
            pdays[r["app_id"]], d, "right"
        )
        total = gi["review_count"].get(r["app_id"], np.nan)
        if np.isfinite(total):
            c.loc[i, "reviews_to_date"] = total - after
            c.loc[i, "source"] = "total minus later reviews"
    c = c[c["reviews_to_date"] > 0].copy()
    c["copies_per_review"] = c["copies"] / c["reviews_to_date"]
    rel = pd.to_datetime(gi["steam_release_date"], utc=True, errors="coerce")
    c["years_since_release"] = (c["date"] - c["app_id"].map(rel)).dt.days / 365.25
    c["price"] = c["app_id"].map(gi["price_original_cents"]) / 100
    c["class"] = np.select(
        [c["other_title"], c["multi"], c["steam"]],
        ["mentions another title", "multi-platform", "says on Steam"],
        "platform not stated",
    )
    c.to_csv(OUT / "ratios.csv", index=False)

    lvl = (
        c.groupby("class")["copies_per_review"]
        .agg(
            claims="size",
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    clean = c[c["class"].isin(["says on Steam", "platform not stated"])]
    lvl = pd.concat(
        [
            lvl,
            pd.DataFrame(
                [
                    {
                        "class": "Steam or not stated (main)",
                        "claims": len(clean),
                        "median": clean["copies_per_review"].median(),
                        "p25": clean["copies_per_review"].quantile(0.25),
                        "p75": clean["copies_per_review"].quantile(0.75),
                    }
                ]
            ),
        ]
    )
    # what predicts the ratio (main claims): log ratio on log price, years since release, year
    m = clean.dropna(subset=["price", "years_since_release"])
    m = m[(m["price"] > 0) & (m["years_since_release"] >= 0)]
    X = np.column_stack(
        [
            np.ones(len(m)),
            np.log(m["price"]),
            np.log1p(m["years_since_release"]),
            np.log(m["copies"]),
        ]
    )
    y = np.log(m["copies_per_review"].to_numpy())
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ b
    XtX = np.linalg.inv(X.T @ X)
    V = XtX @ (X.T * u**2) @ X @ XtX * len(y) / (len(y) - X.shape[1])
    pred = pd.DataFrame(
        {
            "term": [
                "const",
                "log price",
                "log(1 + years since release)",
                "log copies",
            ],
            "coef": b,
            "se": np.sqrt(np.diag(V)),
        }
    ).assign(n=len(m), r2=1 - u.var() / y.var())

    # intervals between consecutive claims for the same game
    cal = pd.read_csv(config.ROOT / "sale_calendar.csv")
    sales = [sale_times(r) for _, r in cal.iterrows()]
    iv = []
    for a, g in clean.sort_values("date").groupby("app_id"):
        g = g.reset_index(drop=True)
        for k in range(1, len(g)):
            p, n = g.loc[k - 1], g.loc[k]
            dc, dr = (
                n["copies"] - p["copies"],
                n["reviews_to_date"] - p["reviews_to_date"],
            )
            if dc <= 0 or dr <= 0:
                continue
            days = (n["date"] - p["date"]).days
            iv.append(
                {
                    "app_id": a,
                    "from": p["date"],
                    "to": n["date"],
                    "days": days,
                    "new_copies": dc,
                    "new_reviews": dr,
                    "copies_per_review": dc / dr,
                    "sale_day_share": sale_days_between(p["date"], n["date"], sales)
                    / max(days, 1),
                    "level_ratio_before": p["copies_per_review"],
                }
            )
    iv = pd.DataFrame(iv)
    iv.to_csv(OUT / "intervals.csv", index=False)
    rows = []
    if len(iv) >= 8:
        X = np.column_stack(
            [
                np.ones(len(iv)),
                iv["sale_day_share"],
                np.log(iv["level_ratio_before"]),
                np.log(iv["days"].clip(lower=1)),
            ]
        )
        y = np.log(iv["copies_per_review"].to_numpy())
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        u = y - X @ b
        XtX = np.linalg.inv(X.T @ X)
        V = XtX @ (X.T * u**2) @ X @ XtX * len(y) / (len(y) - X.shape[1])
        for t, bb, ss in zip(
            [
                "const",
                "share of interval in seasonal sales",
                "log level ratio at start",
                "log days",
            ],
            b,
            np.sqrt(np.diag(V)),
        ):
            rows.append({"term": t, "coef": bb, "se": ss, "n": len(iv)})
    inter = pd.DataFrame(rows)
    lvl.to_csv(OUT / "level_ratios.csv", index=False)
    pred.to_csv(OUT / "ratio_predictors.csv", index=False)
    inter.to_csv(OUT / "interval_test.csv", index=False)
    pd.set_option("display.width", 200)
    print(c["source"].value_counts().to_string())
    print(lvl.round(2).to_string(index=False))
    print(pred.round(3).to_string(index=False))
    if len(iv):
        print(
            f"intervals: {len(iv)} from {iv['app_id'].nunique()} games; median copies per new "
            f"review {iv['copies_per_review'].median():.1f}; corr(log interval ratio, log level ratio) "
            f"{np.corrcoef(np.log(iv['copies_per_review']), np.log(iv['level_ratio_before']))[0, 1]:.2f}"
        )
    print(inter.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
