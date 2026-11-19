# Does sale inertia vary by publisher? (21_inertia.py)

Discounted game-events with a following sale; publishers from game_info (catalog = the
publisher's games on Steam).

| Catalog | Game-events | Publishers | Rejoin | Same cut | Change 10+ pts |
|---|---|---|---|---|---|
| 1 game | 6,281 | 749 | 91.9% | 59.9% | 23.4% |
| 2-4 | 7,237 | 614 | 93.5% | 62.9% | 20.7% |
| 5-19 | 7,970 | 374 | 97.7% | 63.8% | 17.8% |
| 20+ | 13,077 | 163 | 97.6% | 72.3% | 12.8% |

Holding game age, price, the current cut (10-point bins), review rate and the sale fixed,
publishers with 20+ games are 8.1 points (SE 2.0) more likely to repeat the exact cut and
8.1 points (1.3) less likely to change it by 10+. Self-published games are 6.5 points (1.4)
less likely to repeat it. Publisher experience matters little once game age is held fixed
(8+ years: +3.6, SE 1.8).

Responsiveness to the last sale's spike and net is small in every group: a 1 SD bigger
spike goes with a 0.2-0.6 point shallower next cut; no group responds to the net.

Reading: the stickiest sellers are the largest, not the smallest. That fits catalog-wide
standing rules (the same calendar and depth across many titles, as in uniform pricing by
retail chains) better than inexperience. Small publishers vary more but do not respond to
results more.
