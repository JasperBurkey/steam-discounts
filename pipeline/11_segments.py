"""Step 11 (batch 6): audience segments, from who reviews which games.

1. Reviewers x games matrix from the paged reviews (+ the big-game sample, 03f), cleaned:
   free copies dropped, single-game reviewers dropped (they carry no co-review signal),
   heavy reviewers downweighted (weight 1 / log2(1 + games reviewed)).
2. Audience factors: shifted positive PMI of the weighted co-review matrix, factorized by
   SVD (equivalent to learning game embeddings from co-choices; Levy and Goldberg 2014).
   The number of factors is chosen by HELD-OUT PREDICTION: hide one game per test
   reviewer, rank all games by similarity to the reviewer's other games; K = smallest
   value within 1 point of the best hit rate. Compared with tags alone and popularity.
3. Games with too few co-reviewers, and all eligible games outside the sample, get a
   position predicted from their tags (ridge regression, validated on held-out games).
4. Soft segments: a Gaussian mixture on the leading components of the factors (each game
   has a membership probability per segment); the number of segments is chosen by
   bootstrap stability.
5. Baselines from tags alone: k-means (the MVP approach) and a topic model (LDA).
6. Rivals: for every sampled game, its 50 nearest eligible games by audience position,
   for the spillover analysis (batch 8).

Caution for batch 8: tag-predicted positions are shrunk toward the average (ridge), so
tag-placed games look more similar to each other than audience-placed games do (in the
first run, 99% of a tag-placed game's rivals were tag-placed). Compare similarities within
the same source, or calibrate them, rather than pooling the two naively.

  python 11_segments.py               (reads data/for_claude/ if present, else raw data)
  REUSE_K=1 python 11_segments.py     (skip the ~15-minute factor selection on a rerun)
Outputs (output/segments/): k_selection.csv, tag_mapping.csv, robustness.csv,
  segment_choice.csv, game_positions.parquet, segments.csv, segments.md, rivals.parquet,
  tag_baselines.csv
"""

import importlib
import json
import os
import time

import config
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.utils.extmath import randomized_svd

OUT = config.OUTPUT / "segments"
COMPACT = config.DATA / "for_claude"
K_GRID = (16, 32, 64, 128, 192, 256, 384)
MIN_SUPPORT = 30  # weighted co-reviewer mass below this: position from tags instead
SEG_GRID = (15, 20, 25, 30, 40)
N_RIVALS = 50
SEED = 0


# ---------------------------------------------------------------- data
def load_reviews():
    path = COMPACT / "reviews_compact.parquet"
    if path.exists():
        return pd.read_parquet(path)
    ex = importlib.import_module("export_for_claude")
    return ex.reviews()


def load_info():
    path = COMPACT / "game_info.parquet"
    if path.exists():
        return pd.read_parquet(path)
    ex = importlib.import_module("export_for_claude")
    ids = set(pd.read_csv(config.STUDY_SAMPLE_CSV)["app_id"].astype(int))
    if config.PRICE_HISTORY_PARQUET.exists():
        ids |= set(pd.read_parquet(config.PRICE_HISTORY_PARQUET)["app_id"].astype(int))
    return ex.game_info(ids)


def clean(r, steam_only=False, drop_top=0.0):
    r = r[r["received_for_free"] != True]  # noqa: E712
    if steam_only:
        r = r[r["steam_purchase"] == True]  # noqa: E712
    r = r.drop_duplicates(["reviewer", "app_id"])
    r = r[r.groupby("reviewer")["app_id"].transform("size") >= 2]
    if drop_top:
        n = r.groupby("reviewer")["app_id"].transform("size")
        r = r[n <= n.quantile(1 - drop_top)]
    return r


def matrix(r, games=None):
    games = np.array(sorted(r["app_id"].unique())) if games is None else games
    gi = pd.Series(np.arange(len(games)), index=games)
    r = r[r["app_id"].isin(gi.index)]
    _, ri = np.unique(r["reviewer"].to_numpy(), return_inverse=True)
    x = sp.csr_matrix(
        (np.ones(len(r)), (ri, gi[r["app_id"]].to_numpy())),
        shape=(ri.max() + 1, len(games)),
    )
    return x, games


def cooc(x):
    n = np.asarray(x.sum(1)).ravel()
    xw = sp.diags(np.sqrt(1.0 / np.log2(1.0 + n))) @ x
    c = (xw.T @ xw).toarray()
    np.fill_diagonal(c, 0.0)
    return c


def sppmi(c, alpha=0.75):
    total = c.sum()
    row = c.sum(1)
    col = row**alpha
    col = col / col.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(c / (row[:, None] / total) / col[None, :] / total)
    pmi[~np.isfinite(pmi)] = 0.0
    return np.maximum(pmi, 0.0)


def embed(m, k):
    u, s, _ = randomized_svd(m, n_components=k, random_state=SEED)
    e = u * np.sqrt(s)
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)


def tag_vectors(games, info):
    """TF-IDF of Steam's tag weights (top 20 tags per game), L2-normalized."""
    tags = info.set_index("appid")["tags"]
    rows, cols, vals, vocab = [], [], [], {}
    for i, g in enumerate(games):
        t = tags.get(g)
        for pair in json.loads(t) if isinstance(t, str) else []:
            if not pair or pair[0] is None:
                continue
            j = vocab.setdefault(int(pair[0]), len(vocab))
            rows.append(i)
            cols.append(j)
            vals.append(float(pair[1] or 1))
    t = sp.csr_matrix((vals, (rows, cols)), shape=(len(games), max(len(vocab), 1)))
    t = t.toarray()
    t = t / np.maximum(t.sum(1, keepdims=True), 1e-9)
    idf = np.log(len(games) / (1 + (t > 0).sum(0)))
    t = t * idf
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.maximum(norm, 1e-12), vocab, norm.ravel() > 0


# ---------------------------------------------------------------- held-out prediction
def split_reviewers(r, frac=0.2):
    rng = np.random.default_rng(SEED)
    rv = r["reviewer"].unique()
    test = set(rng.choice(rv, int(len(rv) * frac), replace=False))
    m = r["reviewer"].isin(test)
    return r[~m], r[m]


def leave_one_out(sim, games, test, n_users=20000):
    rng = np.random.default_rng(SEED + 1)
    gi = pd.Series(np.arange(len(games)), index=games)
    t = test[test["app_id"].isin(gi.index)]
    size = t.groupby("reviewer").size()
    users = size[size >= 3].index.to_numpy()
    users = rng.choice(users, min(n_users, len(users)), replace=False)
    lists = t[t["reviewer"].isin(users)].groupby("reviewer")["app_id"].apply(list)
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    hits, pct = [], []
    for items in lists:
        idx = gi[items].to_numpy()
        h = rng.integers(len(idx))
        target, keep = idx[h], np.delete(idx, h)
        score = s[keep].mean(0)
        score[keep] = -np.inf
        rank = (score > score[target]).sum()
        hits.append(rank < 20)
        pct.append(rank / (len(games) - len(keep)))
    return float(np.mean(hits)), float(np.mean(pct)), len(lists)


def choose_k(r, info):
    train, test = split_reviewers(r)
    x, games = matrix(train)
    c = cooc(x)
    m = sppmi(c)
    rows = []
    pop = np.tile(c.sum(0), (len(games), 1))
    rows.append(("popularity only", None, *leave_one_out(pop, games, test)))
    t, _, _ = tag_vectors(games, info)
    rows.append(("tags only", None, *leave_one_out(t @ t.T, games, test)))
    for k in K_GRID:
        e = embed(m, k)
        rows.append(("audience factors", k, *leave_one_out(e @ e.T, games, test)))
        print(f"  K={k}: hit@20 {rows[-1][2]:.3f}", flush=True)
    res = pd.DataFrame(
        rows, columns=["method", "K", "hit_at_20", "mean_pct_rank", "test_reviewers"]
    )
    fac = res[res["method"] == "audience factors"]
    k = int(fac.loc[fac["hit_at_20"] >= fac["hit_at_20"].max() - 0.01, "K"].min())
    return res, k


# ---------------------------------------------------------------- tags -> factors
def fit_tag_map(e, t, ok, support):
    """Ridge from tag vectors to audience positions, trained on well-supported games,
    validated on held-out well-supported games: does the predicted position find the same
    audience neighbors as the real one, compared with plain tag similarity?"""
    rng = np.random.default_rng(SEED)
    good = np.where(ok & (support >= 3 * MIN_SUPPORT))[0]
    test = rng.choice(good, len(good) // 5, replace=False)
    train = np.setdiff1d(np.where(ok & (support >= MIN_SUPPORT))[0], test)
    best = None
    for a in (0.1, 0.3, 1.0, 3.0):
        model = Ridge(alpha=a).fit(t[train], e[train])
        p = model.predict(t[test])
        p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
        cos = float(np.mean(np.sum(p * e[test], axis=1)))
        if best is None or cos > best[1]:
            best = (a, cos, model)
    a, cos, model = best
    p = model.predict(t[test])
    p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
    pool = np.where(ok & (support >= MIN_SUPPORT))[0]

    def overlap(q, k=20):
        out = []
        for i, row in zip(test, q):
            true = set(pool[np.argsort(-(e[pool] @ e[i]))[1 : k + 1]])
            got = set(pool[np.argsort(-(row @ e[pool].T))[: k + 1]]) - {i}
            out.append(len(true & set(list(got)[:k])) / k)
        return float(np.mean(out))

    def overlap_tags(k=20):
        out = []
        for i in test:
            true = set(pool[np.argsort(-(e[pool] @ e[i]))[1 : k + 1]])
            got = [j for j in pool[np.argsort(-(t[pool] @ t[i]))] if j != i][:k]
            out.append(len(true & set(got)) / k)
        return float(np.mean(out))

    stats = {
        "ridge_alpha": a,
        "heldout_games": len(test),
        "cosine_predicted_vs_actual": cos,
        "audience_neighbors_found_top20_predicted": overlap(p),
        "audience_neighbors_found_top20_tags_only": overlap_tags(),
    }
    model = Ridge(alpha=a).fit(
        t[ok & (support >= MIN_SUPPORT)], e[ok & (support >= MIN_SUPPORT)]
    )
    return model, stats


# ---------------------------------------------------------------- segments
def choose_segments(e):
    z = PCA(n_components=min(30, e.shape[1]), random_state=SEED).fit_transform(e)
    rng = np.random.default_rng(SEED)
    rows = []
    for s in SEG_GRID:
        base = (
            GaussianMixture(s, covariance_type="diag", random_state=SEED)
            .fit(z)
            .predict(z)
        )
        aris = []
        for b in range(5):
            idx = rng.choice(len(z), len(z), replace=True)
            gm = GaussianMixture(s, covariance_type="diag", random_state=SEED + b + 1)
            aris.append(adjusted_rand_score(base, gm.fit(z[idx]).predict(z)))
        rows.append({"segments": s, "stability_ari": float(np.mean(aris))})
        print(f"  {s} segments: stability {rows[-1]['stability_ari']:.3f}", flush=True)
    res = pd.DataFrame(rows)
    best = res["stability_ari"].max()
    s = int(res.loc[res["stability_ari"] >= best - 0.02, "segments"].max())
    gm = GaussianMixture(s, covariance_type="diag", random_state=SEED).fit(z)
    return res, s, gm.predict_proba(z)


def describe(seg_probs, games, info, r, tag_names, t, vocab, k_top=8):
    names = info.set_index("appid")["name"]
    lab = seg_probs.argmax(1)
    inv = {j: tid for tid, j in vocab.items()}
    tn = tag_names.drop_duplicates("tagid").set_index("tagid")["name"]
    lang = r.groupby(["app_id", "language"], observed=True).size()
    lang = lang / lang.groupby(level=0).transform("sum")
    n_rev = r.groupby("app_id").size()
    play = r.groupby("app_id")["playtime_min"].median()
    lines, rows = ["# Audience segments", ""], []
    overall = t.mean(0)
    for s in range(seg_probs.shape[1]):
        m = lab == s
        if not m.any():
            continue
        g = games[m]
        top = n_rev.reindex(g).sort_values(ascending=False).head(k_top).index
        lift = t[m].mean(0) - overall
        tags = [str(tn.get(inv[j], inv[j])) for j in np.argsort(-lift)[:6]]
        have = [a for a in g if a in lang.index.get_level_values(0)]
        langs = (
            (lang.loc[have].groupby(level=1, observed=True).sum() / len(have))
            .sort_values(ascending=False)
            .head(3)
            if have
            else pd.Series(dtype=float)
        )
        row = {
            "segment": s,
            "games": int(m.sum()),
            "top_tags": ", ".join(tags),
            "languages": ", ".join(f"{k} {v:.0%}" for k, v in langs.items()),
            "median_playtime_h": round(float(play.reindex(g).median()) / 60, 1),
            "most_reviewed": "; ".join(str(names.get(a, a)) for a in top),
        }
        rows.append(row)
        lines += [
            f"## Segment {s} ({row['games']} games)",
            f"- Distinctive tags: {row['top_tags']}",
            f"- Review languages: {row['languages']}",
            f"- Median playtime at review: {row['median_playtime_h']} h",
            f"- Most reviewed: {row['most_reviewed']}",
            "",
        ]
    return pd.DataFrame(rows), "\n".join(lines)


def tag_baselines(all_games, info, n=20):
    t, _, ok = tag_vectors(all_games, info)
    km = KMeans(n, n_init=10, random_state=SEED).fit(t[ok])
    counts = np.round(t[ok] * 100).astype(int)
    lda = LatentDirichletAllocation(n, random_state=SEED, learning_method="online")
    top = lda.fit_transform(counts).argmax(1)
    return pd.DataFrame(
        {"app_id": all_games[ok], "tag_kmeans": km.labels_, "tag_topic": top}
    )


def robustness(r, games_main, e_main):
    """Do cleaning choices change who a game's audience neighbors are?"""
    rows = []
    for label, kw in [
        ("Steam purchases only", {"steam_only": True}),
        ("without the top 1% heaviest reviewers", {"drop_top": 0.01}),
    ]:
        x, games = matrix(clean(r, **kw), games_main)
        e = embed(sppmi(cooc(x)), e_main.shape[1])
        common = np.arange(len(games_main))
        ov = []
        for i in common[:: max(1, len(common) // 500)]:
            a = set(np.argsort(-(e_main @ e_main[i]))[1:21])
            b = set(np.argsort(-(e @ e[i]))[1:21])
            ov.append(len(a & b) / 20)
        rows.append(
            {"variant": label, "top20_neighbor_overlap_with_main": float(np.mean(ov))}
        )
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    r_all = load_reviews()
    info = load_info()
    tag_names = (
        pd.read_parquet(COMPACT / "tag_names.parquet")
        if (COMPACT / "tag_names.parquet").exists()
        else pd.DataFrame(columns=["tagid", "name"])
    )
    sample = pd.read_csv(
        COMPACT / "study_sample.csv"
        if (COMPACT / "study_sample.csv").exists()
        else config.STUDY_SAMPLE_CSV
    )
    r = clean(r_all)
    print(
        f"Reviews {len(r_all):,} -> {len(r):,} after cleaning; {r['reviewer'].nunique():,} "
        f"reviewers of 2+ games; {r['app_id'].nunique():,} games  [{time.time() - t0:.0f}s]"
    )

    if os.environ.get("REUSE_K") and (OUT / "k_selection.csv").exists():
        ksel = pd.read_csv(OUT / "k_selection.csv")
        fac = ksel[ksel["method"] == "audience factors"]
        k = int(fac.loc[fac["hit_at_20"] >= fac["hit_at_20"].max() - 0.01, "K"].min())
        print("Reusing the saved factor selection (REUSE_K=1).")
    else:
        print("Choosing the number of factors by held-out prediction:")
        ksel, k = choose_k(r, info)
        ksel.to_csv(OUT / "k_selection.csv", index=False)
    print(f"  -> K = {k}")

    x, games = matrix(r)
    c = cooc(x)
    e = embed(sppmi(c), k)
    support = c.sum(1)

    # one tag matrix (one vocabulary) for every game, so the tag model applies everywhere
    all_games = np.array(sorted(set(info["appid"].astype(int)) | set(games)))
    ta, vocab, ok_all = tag_vectors(all_games, info)
    row_of = {g: i for i, g in enumerate(all_games)}
    rows_m = np.array([row_of[g] for g in games])
    t, ok = ta[rows_m], ok_all[rows_m]
    model, stats = fit_tag_map(e, t, ok, support)
    pd.DataFrame([stats]).to_csv(OUT / "tag_mapping.csv", index=False)
    print(
        "Tags -> audience position, held-out games:",
        {k_: round(v, 3) if isinstance(v, float) else v for k_, v in stats.items()},
    )

    # positions for every eligible game: audience where well supported, else from tags
    pos = model.predict(ta)
    pos /= np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 1e-12)
    source = np.full(len(all_games), "tags", dtype=object)
    for j, g in enumerate(games):
        if support[j] >= MIN_SUPPORT:
            pos[row_of[g]] = e[j]
            source[row_of[g]] = "audience"
    keep = (source == "audience") | ok_all
    all_games, pos, source, ta = all_games[keep], pos[keep], source[keep], ta[keep]
    gp = pd.DataFrame(pos, columns=[f"f{i}" for i in range(k)])
    gp.insert(0, "source", source)
    gp.insert(0, "app_id", all_games)
    gp.to_parquet(OUT / "game_positions.parquet", index=False)
    print(
        f"Positions: {(source == 'audience').sum():,} from audiences, "
        f"{(source == 'tags').sum():,} from tags"
    )

    print("Robustness of audience neighbors to cleaning choices:")
    rob = robustness(r_all, games, e)
    rob.to_csv(OUT / "robustness.csv", index=False)
    print(rob.round(3).to_string(index=False))

    print(
        "Choosing the number of segments (bootstrap stability), on audience-placed games:"
    )
    aud = source == "audience"
    sch, s, probs = choose_segments(pos[aud])
    sch.to_csv(OUT / "segment_choice.csv", index=False)
    segtab, md = describe(probs, all_games[aud], info, r_all, tag_names, ta[aud], vocab)
    segtab.to_csv(OUT / "segment_summary.csv", index=False)
    seg = pd.DataFrame(probs, columns=[f"p{i}" for i in range(s)])
    seg.insert(0, "segment", probs.argmax(1))
    seg.insert(0, "app_id", all_games[aud])
    seg.to_csv(OUT / "segments.csv", index=False)
    (OUT / "segments.md").write_text(md, encoding="utf-8")
    print(f"  -> {s} segments")

    # rivals: for each sampled game, nearest eligible games by audience position
    sid = [a for a in sample["app_id"].astype(int) if a in set(all_games)]
    idx = {g: i for i, g in enumerate(all_games)}
    recs = []
    for a in sid:
        sim = pos @ pos[idx[a]]
        sim[idx[a]] = -np.inf
        top = np.argsort(-sim)[:N_RIVALS]
        for rank, j in enumerate(top, 1):
            recs.append(
                {
                    "app_id": a,
                    "rival": int(all_games[j]),
                    "rank": rank,
                    "similarity": float(sim[j]),
                    "rival_source": source[j],
                }
            )
    pd.DataFrame(recs).to_parquet(OUT / "rivals.parquet", index=False)
    tag_baselines(all_games, info).to_csv(OUT / "tag_baselines.csv", index=False)
    print(f"Done in {(time.time() - t0) / 60:.0f} minutes. Outputs in {OUT}")


if __name__ == "__main__":
    main()
