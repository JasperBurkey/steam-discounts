# Uniform discounting within publishers (26_uniformity.py)

Every paid game with a price history (29,597 games, 362,393 game-sales, 77% discounted).
Publisher-sales with 2+ discounted games: 34,341 (3,332 publishers, 6 games on average).
Benchmark: depths shuffled at random among discounted games in the same sale and the same
cell of price tercile x age tercile x previous-sale depth (200 draws), so similar games
get similar depths by chance in the benchmark too.

| Publisher's catalog | Publisher-sales | Games each | All same depth: observed | ...random among similar games | Share at the publisher's most common depth: observed | ...random |
|---|---|---|---|---|---|---|
| All | 34,341 | 6.0 | 29.3% | 5.2% | 64.5% | 45.0% |
| 2-4 games | 22,094 | 2.4 | 36.8% | 7.7% | 69.0% | 51.2% |
| 5-19 games | 9,613 | 6.8 | 18.3% | 0.8% | 59.6% | 36.0% |
| 20+ games | 2,634 | 33.5 | 6.3% | 0.1% | 44.7% | 25.6% |

Benchmark 95% ranges are within +/-0.3 points of each benchmark mean, so every gap is far
outside chance. Publishers also use fewer distinct depths than chance (2.6 vs 3.7 per
publisher-sale; 8.0 vs 12.3 for 20+ catalogs).

Depths are focal: 85% are multiples of 5, 61% multiples of 10; 50% (15%), 80% (11%) and
75% (10%) are the most common.

Reading: publishers discount their games far more alike than games that look alike are
discounted by others. It is the discount analog of uniform pricing across a chain's stores
(DellaVigna and Gentzkow 2019): a catalog-wide rule rather than game-by-game choice. Large
catalogs are rarely fully uniform (33 games is a lot to make identical) but still put 45%
of their games at a single depth.

policy_link.csv (when present): whether the learned rule would split those uniform
publisher-sales into different depths, and the gains it finds there.
