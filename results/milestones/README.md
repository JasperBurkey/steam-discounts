# Copies sold per review, from developers' milestone posts (24_milestones.py)

251 "X copies sold" claims from developer posts for 155 study games (claims.csv, with the
text around each). Matched to cumulative Steam-purchaser reviews on the claim date for the
99 claims whose games have complete review histories; the rest (mostly the largest games)
need 03i_pull_milestone_counts.py (about 250 requests).

**Copies per review: median 61 (IQR 38-104)** for claims that say "on Steam" or don't name
a platform (89 claims). Claims mentioning consoles or all platforms: median 88 (8 claims).

**Bigger games have more copies per review.** log(copies per review) rises 0.30 (SE 0.06)
per log copy, so across games reviews grow roughly as copies^0.7. Older games have slightly
fewer copies per review (-0.25, SE 0.15); price doesn't matter. If the same held within a
game, a 10% rise in reviews would mean roughly a 14% rise in copies, so the review proxy
would understate proportional sales effects. This is cross-sectional and from a selected
group (games that announce milestones), so it is a caution, not a correction.

**Sale periods:** only 18 intervals between two claims (12 games). More seasonal-sale days
in an interval go with fewer copies per new review (-2.0 per unit share, SE 1.7), i.e. sale
buyers reviewing more, not less; far from significant. The interval ratio is only weakly
related to the game's earlier level (corr 0.27), so ratios are not very stable over time.

Caveats: claims are rounded lower bounds; developers announce when things go well; some
"platform not stated" totals likely include consoles.
