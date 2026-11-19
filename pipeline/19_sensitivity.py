"""Step 19: sensitivity to unobserved confounding (Chernozhukov, Cinelli, Newey, Sharma and
Syrgkanis 2022, "Long Story Short: Omitted Variable Bias in Causal Machine Learning").

For the ATT estimated by double machine learning, a confounder A we don't observe can bias
the estimate by at most
    |bias| <= S * C_Y * C_D / sqrt(1 - C_D^2),   S^2 = E[(Y - g)^2] * E[alpha^2]
where g is the outcome regression, alpha the ATT's Riesz representer
    alpha = D / p - (1 - D) e(X) / ((1 - e(X)) p),
C_Y^2 is the share of the outcome's remaining variation A would explain, and C_D^2 how
much A would change the Riesz representer (roughly, how much it explains who discounts).
(The bound takes the worst case, rho = 1.)

Reported for each headline outcome:
  robustness value   C_Y^2 = C_D^2 needed to push the estimate to zero (RV), or to make it
                     insignificant at 5% (RV_a)
  samples            all game-events, and the overlap-trimmed sample (propensity 0.1-0.9,
                     as in the main robustness checks). With poor overlap, a few full-price
                     games that look certain to discount get huge weight in alpha, which
                     inflates S and makes any estimate look fragile.
  benchmarks         implied C_Y^2 and C_D^2 from leaving out each group of observed
                     controls (review history, discount history, earlier sales, game and
                     publisher), and the bias bound if a hidden confounder were as strong
                     as that group, or twice as strong.

  python 19_sensitivity.py
  SENS_SAMPLE=trimmed SENS_OUTCOMES=r_net SENS_GROUPS="discount history" python 19_sensitivity.py
      (run part of the grid; rows are appended to the output files as they finish)
Outputs: results/sensitivity/
"""

import os
import time

import config
import dml_core as M
import numpy as np
import pandas as pd
from scipy.optimize import brentq

OUT = config.ROOT.parent / "results" / "sensitivity"
OUTCOMES = {
    "y_during": "Reviews, during",
    "y_net": "Reviews, net",
    "r_net": "Revenue proxy, net (headline)",
}
GROUPS = {
    "review history": ["pre_rate_28", "pre_rate_60", "pre_trend", "pct_pos_60"],
    "discount history": [
        "days_since_discount",
        "never_discounted_before",
        "disc_starts_365",
        "mean_cut_365",
        "share_days_disc_365",
        "min_ratio_365",
        "max_cut_ever",
    ],
    "earlier sales": [
        "prev_discounted",
        "prev_cut",
        "prev_y_during",
        "prev_share_joined",
    ],
    "game and publisher": ["age_days", "regular_usd", "pub_prev_share", "pub_n_games"],
}
CLIP = 0.01


def nuisances(df, y, drop=()):
    """Cross-fitted ATT, residual variance of the outcome and E[alpha^2], leaving out the
    controls in `drop` (and their missing-value indicators)."""
    x, names = M.features(df, names=True)
    keep = [
        i for i, n in enumerate(names) if n.removesuffix("_missing") not in set(drop)
    ]
    x = x[:, keep]
    g = df["app_id"].to_numpy()
    d = df["discounted"].astype(int).to_numpy()
    yv = df[y].to_numpy(float)
    folds = M.game_folds(g, config.SEED)
    e = np.clip(M.crossfit(x, d, folds, binary=True), CLIP, 1 - CLIP)
    g0 = M.crossfit(x, yv, folds, mask=d == 0)
    g1 = M.crossfit(x, yv, folds, mask=d == 1)
    p = d.mean()
    alpha = d / p - (1 - d) * e / ((1 - e) * p)
    psi = alpha * (yv - g0)  # ATT moment: E[alpha (Y - g0)] = theta
    theta = psi.mean()
    se = M.cluster_se(psi - theta * d / p, g, len(psi))
    resid = yv - np.where(d == 1, g1, g0)
    return {
        "theta": theta,
        "se": se,
        "sigma2": float(np.mean(resid**2)),
        "nu2": float(np.mean(alpha**2)),
        "n": len(df),
    }


def bias_bound(S, cy2, cd2):
    cd2 = min(cd2, 0.999)
    return S * np.sqrt(cy2) * np.sqrt(cd2 / (1 - cd2))


def robustness_value(S, target):
    """C_Y^2 = C_D^2 = k at which the bias bound equals `target`."""
    if target <= 0:
        return 0.0
    f = lambda k: S * k / np.sqrt(1 - k) - target  # noqa: E731
    return brentq(f, 1e-9, 0.999999) if f(0.999999) > 0 else 1.0


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    base = ev[ev["analysis_sample"]].dropna(subset=["y_during", "pre_trend"])
    rows, bench = [], []
    # overlap trimming: propensity from the full control set, cross-fitted by game
    x = M.features(base.reset_index(drop=True))
    e = M.crossfit(
        x,
        base["discounted"].astype(int).to_numpy(),
        M.game_folds(base["app_id"].to_numpy(), config.SEED),
        binary=True,
    )
    base = base.reset_index(drop=True).assign(_e=e)
    samples = {
        "all game-events": base,
        "overlap-trimmed (0.1-0.9)": base[(base["_e"] > 0.1) & (base["_e"] < 0.9)],
    }
    pick = os.environ.get("SENS_SAMPLE")
    if pick:
        samples = {k: v for k, v in samples.items() if k.startswith(pick)}
    outs = os.environ.get("SENS_OUTCOMES")
    outcomes = {k: v for k, v in OUTCOMES.items() if not outs or k in outs.split(",")}
    gp = os.environ.get("SENS_GROUPS")
    groups = {k: v for k, v in GROUPS.items() if not gp or k in gp.split(",")}

    def save():
        for name, rr in (("robustness_values.csv", rows), ("benchmarks.csv", bench)):
            if rr:
                f = OUT / name
                new = pd.DataFrame(rr)
                if f.exists():
                    old = pd.read_csv(f)
                    keys = ["sample", "outcome"] + (
                        ["hidden confounder as strong as"]
                        if "bias bound" in new
                        else []
                    )
                    old = old[
                        ~old.set_index(keys).index.isin(new.set_index(keys).index)
                    ]
                    new = pd.concat([old, new], ignore_index=True)
                new.to_csv(f, index=False)

    for sname, sdf in samples.items():
        for y, label in outcomes.items():
            df = sdf.dropna(subset=[y]).reset_index(drop=True)
            full = nuisances(df, y)
            S = np.sqrt(full["sigma2"] * full["nu2"])
            th, se = full["theta"], full["se"]
            rv = robustness_value(S, abs(th))
            rva = robustness_value(S, abs(th) - 1.96 * se)
            rows.append(
                {
                    "sample": sname,
                    "outcome": label,
                    "ATT": th,
                    "se": se,
                    "n": full["n"],
                    "S": S,
                    "E[alpha^2]": full["nu2"],
                    "RV (to zero)": rv,
                    "RV (to insignificant)": rva,
                }
            )
            print(
                f"  {sname} | {label}: ATT {th:+.3f} ({se:.3f}), RV {rv:.3f}, RV_a {rva:.3f}  [{(time.time() - t0) / 60:.0f} min]",
                flush=True,
            )
            save()
            for gname, cols in groups.items():
                short = nuisances(df, y, drop=cols)
                cy2 = max(0.0, 1 - full["sigma2"] / short["sigma2"])
                cd2 = max(0.0, 1 - short["nu2"] / full["nu2"])
                for k in (1, 2):
                    b = bias_bound(S, min(k * cy2, 0.999), min(k * cd2, 0.999))
                    bench.append(
                        {
                            "sample": sname,
                            "outcome": label,
                            "hidden confounder as strong as": f"{k}x {gname}",
                            "C_Y^2": k * cy2,
                            "C_D^2": k * cd2,
                            "bias bound": b,
                            "ATT bounds": f"[{th - b:+.3f}, {th + b:+.3f}]",
                            "95% CI bounds": f"[{th - b - 1.96 * se:+.3f}, {th + b + 1.96 * se:+.3f}]",
                            "estimate when dropped": short["theta"],
                        }
                    )
                print(
                    f"    benchmark {gname}: C_Y^2 {cy2:.3f}, C_D^2 {cd2:.3f}  [{(time.time() - t0) / 60:.0f} min]",
                    flush=True,
                )
                save()
    save()
    rows, bench = (
        pd.read_csv(OUT / "robustness_values.csv"),
        pd.read_csv(OUT / "benchmarks.csv"),
    )
    pd.set_option("display.width", 220)
    print(rows.round(3).to_string(index=False))
    print(bench.round(3).to_string(index=False))
    print(f"Done in {(time.time() - t0) / 60:.0f} minutes")


if __name__ == "__main__":
    main()
