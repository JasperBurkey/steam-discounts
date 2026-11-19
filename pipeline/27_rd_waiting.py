"""Step 27: do buyers wait for a discount they expect? A test at the release cutoff.

At the 30-day release cutoff (22_release_rd.py), games just old enough can join the
seasonal sale and games just too young cannot. Buyers who expect a discount have a reason
to hold off in the week before the sale, but only for eligible games. So a drop in reviews
in that week, just above the cutoff, is evidence of waiting that does not rely on which
games publishers choose to discount.

Outcome: reviews in the week before the sale (ages ~23-29 at the cutoff) relative to the
game's launch week (rel7, fixed before either group could discount), log per week. Also
the raw level of the week-before reviews. Reduced form (jump at the cutoff) and fuzzy RD
(scaled by the jump in discounting), bandwidths 10 / 14 / 21, with and without the donut,
and placebo cutoffs at ages 23, 26 and 44 (clean of the launch period and the real jump).

Reading: a waiting dip shows as a negative jump. Anticipation needs buyers to expect the
discount; a month-old game's buyers may not, so a zero would not rule waiting out for
established games.

  python 27_rd_waiting.py   (needs results/release_rd/rd_sample.parquet from 22_release_rd)
Outputs: results/release_rd/waiting.csv
"""

import importlib

import config
import pandas as pd

R = importlib.import_module("22_release_rd")
OUT = config.ROOT.parent / "results" / "release_rd"


def main():
    o = pd.read_parquet(OUT / "rd_sample.parquet")
    o = o.assign(pre_vs_launch=o["y_D"] - o["early7"]).dropna(subset=["pre_vs_launch"])
    rows = []
    for y, lab in [
        ("pre_vs_launch", "week before the sale vs launch week"),
        ("y_D", "week before the sale (level)"),
    ]:
        for h in (10, 14, 21):
            for donut in (False, True):
                for part, fz in [("reduced form", False), ("fuzzy RD", True)]:
                    est, se, n = R.rd(o, y, h, donut, fuzzy=fz)
                    rows.append(
                        dict(
                            outcome=lab,
                            part=part,
                            bw=h,
                            donut=donut,
                            est=est,
                            se=se,
                            n=n,
                        )
                    )
        for age in (23, 26, 44):
            est, se, n = R.rd(o, y, 10, c=age - R.CUT, fuzzy=False)
            rows.append(
                dict(
                    outcome=lab,
                    part=f"placebo cutoff (age {age})",
                    bw=10,
                    donut=False,
                    est=est,
                    se=se,
                    n=n,
                )
            )
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "waiting.csv", index=False)
    res["est (se)"] = res.apply(lambda x: f"{x['est']:+.3f} ({x['se']:.3f})", axis=1)
    pd.set_option("display.width", 200)
    print(
        res.pivot_table(
            index=["outcome", "part", "donut"],
            columns="bw",
            values="est (se)",
            aggfunc="first",
        ).to_string()
    )
    print("n (bw 21):", int(res.loc[res["bw"] == 21, "n"].max()))


if __name__ == "__main__":
    main()
