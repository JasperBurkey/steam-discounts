"""Tables for the notebook sections whose results were computed by pipeline scripts and
saved in results/ (rank validation, segmentation, spillovers, policy learning). Reading the
saved files keeps the notebook fast; rerun the named script to reproduce them."""

from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _fmt(est, se, digits=3):
    return f"{est:+.{digits}f} ({se:.{digits}f})"


# ---------------------------------------------------------------- rank validation
RANK_LABELS = {
    "d_points_during_global": "Chart points during the sale (global top 100)",
    "d_points_net_global": "Chart points, net of the weeks after",
    "charted_during_global": "Probability of charting during the sale",
    "d_points_during_any": "Chart points during the sale (any list)",
    "y_during": "Reviews during the sale (log)",
    "r_during": "Revenue proxy during the sale (log)",
}


def rank_validation():
    t = pd.read_csv(RESULTS / "rank_validation" / "rank_validation.csv")
    t = t[t["estimand"] == "ATT"].copy()
    t["outcome"] = t["outcome"].map(RANK_LABELS)
    t["ATT (SE)"] = [_fmt(e, s) for e, s in zip(t["estimate"], t["se"])]
    return t[["sample", "outcome", "ATT (SE)", "n", "games"]].reset_index(drop=True)


def weighting_test():
    w = pd.read_csv(RESULTS / "rank_validation" / "rank_weighting_test.csv").iloc[0]
    return pd.DataFrame(
        {
            "term": [
                "Reviews (log change)",
                "Price paid (log ratio to regular)",
                "Price minus reviews",
            ],
            "coefficient (SE)": [
                _fmt(w["b_reviews"], w["se_reviews"]),
                _fmt(w["b_log_price"], w["se_log_price"]),
                _fmt(w["diff_price_minus_reviews"], w["se_diff"]),
            ],
            "proxy assumes": ["1", "1", "0"],
        }
    ), int(w["n"])


# ---------------------------------------------------------------- segments
def segment_checks():
    k = pd.read_csv(RESULTS / "segments" / "k_selection.csv")
    k["K"] = k["K"].astype("Int64")
    tm = pd.read_csv(RESULTS / "segments" / "tag_mapping.csv").iloc[0]
    rob = pd.read_csv(RESULTS / "segments" / "robustness.csv")
    ch = pd.read_csv(RESULTS / "segments" / "segment_choice.csv")
    return k, tm, rob, ch


def segment_table(n=12):
    s = pd.read_csv(RESULTS / "segments" / "segment_summary.csv")
    s = s.sort_values("games", ascending=False).head(n)
    s["tags"] = s["top_tags"].str.split(", ").str[:3].str.join(", ")
    s["main language"] = s["languages"].str.split(", ").str[0]
    s["examples"] = s["most_reviewed"].str.split("; ").str[:3].str.join("; ")
    # titles of adult-only segments are left out of the notebook (they stay in results/)
    adult = s["top_tags"].str.contains("Sexual Content|Nudity|Hentai")
    s.loc[adult, "examples"] = "(adult content)"
    return s[["segment", "games", "tags", "main language", "examples"]].reset_index(
        drop=True
    )


# ---------------------------------------------------------------- spillovers
def spillover_main():
    t = pd.read_csv(RESULTS / "spillovers" / "main.csv")
    t["estimate (SE)"] = [_fmt(e, s) for e, s in zip(t["estimate"], t["se"])]
    out = t.pivot(index="term", columns="sample", values="estimate (SE)")
    return out.loc[["audience", "tags", "tag_kmeans", "tag_topic"]].rename_axis(
        "rival definition"
    )


def spillover_criteria():
    p = pd.read_csv(RESULTS / "spillovers" / "placebo_summary.csv").set_index(
        "definition"
    )
    g = pd.read_csv(RESULTS / "spillovers" / "gradient.csv")
    h = pd.read_csv(RESULTS / "spillovers" / "head_to_head.csv").set_index("term")
    o = pd.read_csv(RESULTS / "spillovers" / "out_of_sample.csv")
    rows = []
    for d in ["audience", "tags"]:
        gd = g[g["term"].str.startswith(d + "_")]
        od = o[o["term"] == d]
        rows.append(
            {
                "rival definition": d,
                "1. placebo p": round(p.loc[d, "placebo_p"], 3),
                "2. ranks 1-5 / 6-20 / 21-50": " / ".join(
                    f"{e:+.2f}" for e in gd["estimate"]
                ),
                "3. head-to-head": _fmt(h.loc[d, "estimate"], h.loc[d, "se"], 2),
                "4. 2023-24 / 2025-26": " / ".join(
                    _fmt(e, s, 2) for e, s in zip(od["estimate"], od["se"])
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- policy
OBJ_SHORT = {
    "Revenue proxy, net": "proxy, net",
    "Revenue proxy, net, lag-adjusted": "lag-adjusted",
    "Chart-calibrated revenue, during (sensitivity)": "chart-calibrated",
}


def policy_table(sample="all game-events"):
    v = pd.read_csv(RESULTS / "policy" / "policy_values.csv")
    v = v[v["sample"] == sample].copy()
    v["objective"] = v["objective"].map(OBJ_SHORT)
    v["gain (SE)"] = [_fmt(e, s) for e, s in zip(v["gain_vs_status_quo"], v["gain_se"])]
    t = v.pivot(index="policy", columns="objective", values="gain (SE)")
    order = [p for p in v["policy"].unique() if p != "status quo"]
    return t.loc[order, list(OBJ_SHORT.values())]


def policy_tree(sample="all game-events", objective="Revenue proxy, net"):
    text = (RESULTS / "policy" / "policy_trees.txt").read_text()
    for block in text.split("\n\n"):
        if block.startswith(f"[{sample}] objective: {objective}\n"):
            return block
    return ""
