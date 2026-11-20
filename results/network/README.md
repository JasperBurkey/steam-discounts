# Network effects: do discounts pay off later in online multiplayer games? (29_network.py)

Groups from store tags: online (Multiplayer, Online Co-Op, PvP, MMO, ...), local-only
multiplayer (genre control: no online network), single. Quick settings for part B
(LEARNER=boost, CF_REPEATS=1).

## A. Release cutoff (fuzzy RD; change vs the week before the sale)

| Outcome | Online (1,135) | Single (4,812) | Online - single |
|---|---|---|---|
| During the sale, bw 14 / 21 | +0.85 (0.27) / +0.65 (0.17) | +0.40 (0.10) / +0.31 (0.07) | +0.45 (0.29) / +0.34 (0.19) |
| Weeks 1-2 after | -0.20 (0.26) / -0.22 (0.16) | +0.07 (0.08) / +0.04 (0.06) | -0.26 (0.27) / -0.26 (0.17) |
| Weeks 3-4 after | -0.19 (0.25) / -0.25 (0.18) | +0.00 (0.09) / -0.08 (0.06) | -0.19 (0.26) / -0.17 (0.19) |
| Net revenue, sale to week 4 | +0.10 (0.20) / -0.01 (0.14) | +0.09 (0.07) / +0.02 (0.05) | +0.01 (0.21) / -0.03 (0.15) |

| Weeks 5-6 after, bw 14 / 21 | -0.01 (0.46) / -0.17 (0.31) | -0.06 (0.12) / -0.08 (0.09) | +0.05 (0.48) / -0.09 (0.32) |
| Weeks 7-12 after, bw 14 / 21 | -0.21 (0.42) / -0.36 (0.32) | -0.07 (0.13) / -0.07 (0.09) | -0.15 (0.44) / -0.30 (0.34) |

Weeks 5-12 come from 03h_pull_rd_reviews.py --long and use Spring and Summer sales only
(otherwise the next seasonal sale falls in the window): 519 online and 2,293 single-player
game-events within 21 days of the cutoff.

Online games get a bigger spike from an essentially random discount, but nothing lasts:
weeks 1-4 lean negative, and weeks 5-12 show no gain over single-player games (point
estimates negative). "Online, strong" (online tag in the top 5) has positive point
estimates at weeks 5-12 (+0.18 to +0.28 over single-player) with SEs near 0.4-0.5.
Local-only is too small to read (38-86 game-events). The long-horizon estimates are
imprecise (SEs 0.3-0.5), so they find no sign of network effects but cannot rule out
moderate ones.

## B. Main sample (ATT, observational)

| Outcome | Online (10,770) | Single (20,423) | Local only (609) |
|---|---|---|---|
| During | +0.361 (0.034) | +0.187 (0.026) | +0.28 (0.29) |
| Weeks 3-4 after | +0.044 (0.036) | -0.014 (0.027) | -0.55 (0.32) |
| Weeks 5-6 after | +0.078 (0.036) | -0.029 (0.025) | -0.23 (0.31) |
| Weeks 7-12 after (Spring/Summer) | -0.007 (0.067) | -0.066 (0.039) | -0.39 (0.37) |
| Net revenue proxy | -0.031 (0.028) | -0.088 (0.021) | -0.24 (0.20) |

In established games, online titles keep a little more after the sale (weeks 5-6: about
+0.11 above single-player, SE ~0.04) and lose less on net (+0.06, SE ~0.035), fading by
weeks 7-12. That pattern fits a modest network effect, but online games also get more
updates, DLC and live-service events, so it is suggestive only.

Reading: no causal evidence of network effects. The cutoff design shows a bigger spike for
online games and no persistence through week 12 (imprecise at the long horizon); the main
sample shows modest persistence to week 6 that could have other causes (updates, DLC, live
events). Network effects do not explain why habitual discounting loses money, though a
moderate long-run effect for the most online games cannot be excluded.
