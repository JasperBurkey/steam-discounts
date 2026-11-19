# Do central games' discounts spill over? (30_centrality.py)

Network: the audience-rival graph (each of the 3,850 study games -> its 20 nearest games by
audience position; 17,621 rivals, 77,000 edges). Centrality of each game: in-degree (how
many study games count it as a top-20 rival; median 2, max 133) and authority-based Katz
centrality (alpha = 0.5 / spectral radius); the two correlate at 0.97 (centrality.csv).

Exposure split: share of a game's top-20 rivals that discounted, separately for central
rivals (Katz above the median rival slot) and peripheral ones, both in one partially linear
double ML model with the lagged exposures (as in 12_spillovers). Full settings. Outcome:
reviews during the sale. Coefficients compare all such rivals discounting with none; a
realistic swing is one within-sale SD (0.18-0.19).

| | Holdouts (3,749) | Discounted games (23,032) |
|---|---|---|
| Central rivals | +0.063 (0.067) | +0.005 (0.035) |
| Peripheral rivals | +0.109 (0.054) | -0.045 (0.030) |
| All rivals (alone) | +0.229 (0.110) | -0.007 (0.056) |

The Katz-weighted exposure is almost collinear with the plain one (SE 0.54 when both are
included), so that comparison is uninformative.

Reading: central rivals' discounts do not matter more than peripheral rivals'; if anything
the reverse, and neither is clearly different from zero. Network position does not reveal
spillovers that the average hides. Together with 12_spillovers: no evidence that a game is
hurt when rivals, central or not, discount, so no case for "defensive" discounting.
