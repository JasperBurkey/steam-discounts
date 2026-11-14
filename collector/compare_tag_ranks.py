"""Are the per-tag rankings just the global ranking filtered by tag?

If yes (same games, same order, and the store snapshot's tags reproduce which games belong
to each tag), the tag lists add nothing and can be turned off (RANKS_TAG_DEPTH = 0 in
settings.py), cutting the daily ranking job from ~110 minutes to ~6.

  .venv\\Scripts\\python compare_tag_ranks.py
"""

import json

import numpy as np
import pandas as pd
from core import read_table
from scipy import stats

r = read_table("search_ranks", days=7)
if r is None or r.empty:
    raise SystemExit(
        "No search_ranks data yet. Run: collect.py run --job ranks --force"
    )
r = r[r["captured_at"] == r["captured_at"].max()]
glob = r[r["list"] == "global"].dropna(subset=["appid"]).drop_duplicates("appid")
grank = dict(zip(glob["appid"].astype(int), glob["rank"].astype(int)))
tags = r[r["list"] == "tag"].dropna(subset=["appid"])
if tags.empty:
    raise SystemExit(
        "The latest ranking run has no tag lists (RANKS_TAG_DEPTH = 0?). Nothing to compare."
    )
print(
    f"Ranking captured {r['captured_at'].max()}: {len(glob):,} games in the global list, "
    f"{tags['tagid'].nunique()} tag lists, {len(tags):,} tag-list rows\n"
)

# each game's tags in the latest store snapshot (Steam's top 20 tags)
items = read_table("store_items", days=3)
snap_tags = {}
if items is not None and not items.empty:
    items = items[items["captured_at"] == items["captured_at"].max()]
    tagids = items["tagids"] if "tagids" in items else pd.Series([None] * len(items))
    for appid, tj, tid in zip(items["appid"], items["tags"], tagids):
        s = set()
        for src in (tj, tid):
            try:
                v = json.loads(src) if isinstance(src, str) else []
            except ValueError:
                v = []
            s |= {
                int(x[0]) if isinstance(x, list) else int(x)
                for x in (v or [])
                if x is not None
            }
        snap_tags[int(appid)] = s

ordered_global = [a for a, _ in sorted(grank.items(), key=lambda kv: kv[1])]
rows = []
for tagid, t in tags.groupby("tagid"):
    ids = t.sort_values("rank")["appid"].astype(int).tolist()
    in_g = [a for a in ids if a in grank]
    g_order = [grank[a] for a in in_g]
    same_order = all(x < y for x, y in zip(g_order, g_order[1:]))
    rho = (
        stats.spearmanr(range(len(g_order)), g_order).statistic
        if len(g_order) > 2
        else np.nan
    )
    has = [int(tagid) in snap_tags.get(a, set()) for a in ids if a in snap_tags]
    rebuilt = [a for a in ordered_global if int(tagid) in snap_tags.get(a, set())][
        : len(ids)
    ]
    rows.append(
        {
            "tagid": int(tagid),
            "games": len(ids),
            "share_in_global": len(in_g) / len(ids),
            "same_order_as_global": same_order,
            "spearman": rho,
            "snapshot_tags_precision": float(np.mean(has)) if has else np.nan,
            "rebuild_recall": len(set(rebuilt) & set(ids)) / len(ids)
            if snap_tags
            else np.nan,
        }
    )
res = pd.DataFrame(rows)
res.to_csv("tag_rank_comparison.csv", index=False)

print("Across all tag lists:")
print(
    f"  share of tag-list games also in the global list: median {res['share_in_global'].median():.1%}, "
    f"min {res['share_in_global'].min():.1%}"
)
print(
    f"  tag lists in exactly the global order:          {res['same_order_as_global'].mean():.1%}"
)
print(
    f"  rank correlation with the global list:          median {res['spearman'].median():.3f}"
)
if snap_tags:
    print(
        f"  tag-list games carrying that tag in the snapshot: median {res['snapshot_tags_precision'].median():.1%}"
    )
    print(
        f"  tag list rebuilt from global + snapshot tags:     median {res['rebuild_recall'].median():.1%} of games recovered"
    )
print("\nLeast consistent tags:")
print(
    res.sort_values("rebuild_recall" if snap_tags else "spearman")
    .head(8)
    .round(3)
    .to_string(index=False)
)

redundant = (
    res["share_in_global"].median() > 0.99
    and res["same_order_as_global"].mean() > 0.9
    and (not snap_tags or res["rebuild_recall"].median() > 0.95)
)
print()
if redundant:
    print(
        "VERDICT: the tag lists are the global ranking filtered by tag, and can be rebuilt from data you"
    )
    print("already collect. Set RANKS_TAG_DEPTH = 0 in settings.py to turn them off.")
else:
    print(
        "VERDICT: the tag lists contain information the global list plus snapshot tags don't reproduce."
    )
    print("Keep them for now, and send this output to Claude.")
print("Details: tag_rank_comparison.csv")
