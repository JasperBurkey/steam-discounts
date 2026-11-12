"""Step 6: pilot market segmentation.

Part A clusters games on SteamSpy user tags (vote shares -> TF-IDF -> k-means) and
writes a table of segments with their top tags and best-known games. For each K in
K_GRID it prints silhouette (cluster separation) and stability (agreement between fits
on random 80% subsamples). Pick K_FINAL in config.py by reading both and the table.

Part B (needs step 3 data) compares each game's nearest neighbors by tags with its
nearest neighbors by shared reviewers (PMI). Disagreements are where Steam's tags may
not describe who competes with whom.

Outputs: output/clusters.csv, output/clusters.md, output/neighbors.md
"""

import argparse
import json

import config
import numpy as np
import pandas as pd
from common import get_json, study_games
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser()
parser.add_argument(
    "--limit",
    type=int,
    default=1000,
    help="most-reviewed games to tag, plus all study games",
)
args = parser.parse_args()
rng = np.random.default_rng(0)

full = pd.read_csv(config.SAMPLE_CSV)
# Tag the most-reviewed games (to define segments broadly) AND every study game (so the
# shared-reviewer comparison and later analyses cover the games with review data).
sample = (
    pd.concat([full.head(args.limit), study_games(full)])
    .drop_duplicates("app_id")
    .reset_index(drop=True)
)
names = sample.set_index("app_id")["name"]

# ---------- Part A: tags ----------
print(f"Fetching tags for {len(sample)} games (cached after the first run)")
tag_rows = []
for app_id in sample["app_id"]:
    path = config.TAGS_DIR / f"{app_id}.json"
    if path.exists():
        tags = json.loads(path.read_text())
    else:
        data = (
            get_json(
                "https://steamspy.com/api.php",
                {"request": "appdetails", "appid": app_id},
                sleep=config.SLEEP_STEAMSPY,
            )
            or {}
        )
        tags = data.get("tags") or {}
        tags = tags if isinstance(tags, dict) else {}  # SteamSpy returns [] for no tags
        path.write_text(json.dumps(tags))
    tag_rows += [(app_id, t, v) for t, v in tags.items()]

tags = pd.DataFrame(tag_rows, columns=["app_id", "tag", "votes"])
counts = tags.pivot_table(index="app_id", columns="tag", values="votes", fill_value=0)
counts = counts.loc[counts.sum(axis=1) > 0]
share = counts.div(counts.sum(axis=1), axis=0)  # tag vote shares within each game
X = normalize(TfidfTransformer().fit_transform(share.to_numpy()))
apps = share.index.to_numpy()
print(f"{len(apps)} games with tags, {share.shape[1]} distinct tags")

print("\n  K   silhouette   stability (ARI)")
for k in config.K_GRID:
    labels = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    sil = silhouette_score(
        X, labels, metric="cosine", sample_size=min(2000, len(apps)), random_state=0
    )
    fits = []
    for seed in range(5):
        idx = rng.choice(len(apps), int(0.8 * len(apps)), replace=False)
        fits.append(KMeans(k, n_init=10, random_state=seed).fit(X[idx]).predict(X))
    ari = np.mean(
        [
            adjusted_rand_score(fits[i], fits[j])
            for i in range(5)
            for j in range(i + 1, 5)
        ]
    )
    print(f"  {k:<3} {sil:10.3f}   {ari:10.3f}")

km = KMeans(config.K_FINAL, n_init=20, random_state=0).fit(X)
clusters = pd.DataFrame({"app_id": apps, "cluster": km.labels_})
clusters = clusters.merge(sample[["app_id", "name", "reviews_total"]], on="app_id")
clusters.to_csv(config.OUTPUT / "clusters.csv", index=False)

tag_names = share.columns.to_numpy()
lines = [
    f"# Tag-based segments (K = {config.K_FINAL})",
    "",
    "| Segment | Games | Most distinctive tags | Most-reviewed games |",
    "| --- | --- | --- | --- |",
]
for c in range(config.K_FINAL):
    members = clusters[clusters["cluster"] == c].sort_values(
        "reviews_total", ascending=False
    )
    top_tags = tag_names[np.argsort(km.cluster_centers_[c])[::-1][:6]]
    lines.append(
        f"| {c} | {len(members)} | {', '.join(top_tags)} | "
        f"{'; '.join(members['name'].head(6).astype(str))} |"
    )
(config.OUTPUT / "clusters.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(
    "\nWrote output/clusters.csv and output/clusters.md. Read the table and name each segment."
)

# ---------- Part B: co-review neighbors ----------
files = sorted(config.REVIEWS_DIR.glob("*.parquet"))
if not files:
    raise SystemExit("No review data yet; skipping the co-review comparison.")
rv = pd.concat((pd.read_parquet(f, columns=["app_id", "reviewer_hash"]) for f in files))
rv = rv[rv["app_id"].isin(apps)].drop_duplicates()
r_codes, r_ids = pd.factorize(rv["reviewer_hash"])
a_codes, a_ids = pd.factorize(rv["app_id"])
R = sparse.csr_matrix(
    (np.ones(len(rv)), (r_codes, a_codes)), shape=(len(r_ids), len(a_ids))
)
co = (R.T @ R).toarray()  # shared reviewers per pair of games
n_a = np.diag(co).astype(float)
with np.errstate(divide="ignore", invalid="ignore"):
    pmi = np.log(co * len(r_ids) / np.outer(n_a, n_a))
pmi[co < config.MIN_COREVIEW] = -np.inf
np.fill_diagonal(pmi, -np.inf)

pos = {a: i for i, a in enumerate(apps)}
Xr = X[[pos[a] for a in a_ids]]
tag_sim = (Xr @ Xr.T).toarray()
np.fill_diagonal(tag_sim, -np.inf)

multi = (np.asarray(R.sum(1)).ravel() > 1).sum()
print(
    f"\n{len(r_ids):,} reviewers across {len(a_ids)} games; {multi:,} reviewed 2+ of them"
)

order = np.argsort(n_a)[::-1]
lines = [
    "# Nearest neighbors: tags versus shared reviewers",
    "",
    f"Reviewer neighbors use PMI and need at least {config.MIN_COREVIEW} shared reviewers.",
    "",
]
overlap = []
for i in order[:10]:
    by_tag = np.argsort(tag_sim[i])[::-1][:5]
    valid = np.isfinite(pmi[i])
    by_rev = [
        j for j in np.argsort(np.where(valid, pmi[i], -1e9))[::-1][:5] if valid[j]
    ]
    overlap.append(len(set(by_tag) & set(by_rev)))
    lines += [
        f"## {names.get(a_ids[i], a_ids[i])}",
        "",
        f"- By tags: {'; '.join(str(names.get(a_ids[j], a_ids[j])) for j in by_tag)}",
        f"- By shared reviewers: {'; '.join(str(names.get(a_ids[j], a_ids[j])) for j in by_rev) or 'too few shared reviewers'}",
        "",
    ]
(config.OUTPUT / "neighbors.md").write_text("\n".join(lines), encoding="utf-8")
print(
    f"Top-5 neighbor lists overlap by {np.mean(overlap):.1f} games on average "
    f"(5 = tags and audiences agree). Wrote output/neighbors.md"
)
