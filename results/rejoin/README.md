# Do publishers rejoin because of the spike or the net gain? (20_rejoin.py)

Sample: 25,373 discounted game-events with a following sale (2,526 games). Linear
probability models, regressors standardized, SE clustered by game.

**Participation barely responds to outcomes.** 96.9% of discounters rejoin the next sale
(33% of full-price games join the next one). A 1 SD better sale outcome moves the chance of
rejoining by under 1 percentage point, for the spike and the net alike, with or without
game fixed effects. Neither the spike nor the net drives the decision.

**Depth is sticky too.** 66% of rejoiners use exactly the same cut; the mean absolute
change is 3.6 points (SD of cuts 22). Holding the net fixed, a 1 SD bigger spike goes with
a 0.4-point shallower next cut (SE 0.08-0.09); the net has no detectable effect.

Reading: seasonal-sale participation looks like a standing rule (join every sale, at the
usual depth) rather than a choice updated on results. The salience test can't separate
spike from net because almost nobody changes behavior. Caveat: reviews are a noisy stand-in
for what publishers see in their own sales reports.
