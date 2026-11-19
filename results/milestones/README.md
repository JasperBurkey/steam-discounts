# Copies sold per review, from developers' milestone posts (24_milestones.py)

251 "X copies sold" claims from developer posts for 155 study games (claims.csv, with the
text around each), matched to Steam's count of purchaser reviews up to the claim date
(03i_pull_milestone_counts.py; exact for 246 claims).

**Copies per review: median 57 (IQR 34-95)** for the 208 claims that say "on Steam" or
don't name a platform (main sample). Claims mentioning consoles or all platforms: median 81
(32 claims); saying "on Steam": 42 (10 claims).

**What predicts the ratio** (log copies per review; 195 claims, R2 0.17):
- bigger games: +0.19 (SE 0.03) per log copy, so across games reviews grow roughly as
  copies^0.84;
- pricier games: -0.27 (0.07) per log price (buyers of expensive games review more often);
- older games at the claim: -0.36 (0.12) per log(1 + years).
If the size relation held within a game, a 10% rise in reviews would mean about a 12% rise
in copies: the review proxy would slightly understate proportional sales changes. This is
cross-sectional and from games that announce milestones, so a caution, not a correction.

**Stability over time:** 76 intervals between two claims (41 games). Copies per new review
in an interval track the game's earlier level closely (coefficient 0.80, SE 0.11; corr
0.71; median 57.5), so a game's ratio is fairly stable.

**Sale periods:** intervals with more seasonal-sale days have somewhat fewer copies per new
review (-0.62 per unit share, SE 0.48), the direction of sale buyers reviewing slightly
more often; not significant. No evidence that sale buyers review less, which is the
direction that would make the proxy overstate sale effects.

Caveats: claims are rounded lower bounds; developers announce when things go well; some
"platform not stated" totals likely include consoles.
