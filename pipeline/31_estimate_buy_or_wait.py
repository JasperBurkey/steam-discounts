"""Step 31: estimating the buy-or-wait model (simulated method of moments, with standard
errors) and welfare counterfactuals.

15_buy_or_wait.py CALIBRATED the model: buyers compare buying now with waiting, and the
buy decision is all-or-nothing, so the fit criterion jumps as parameters move and has
several local optima (results/buy_or_wait/calibration_grid.json). Here the model is
ESTIMATED, following the structural-estimation route in Sargent and Stachurski,
Dynamic Programming, Vol. I, sec. 5.3.1-5.3.2 (expected value functions and the Gumbel max
trick):

  Taste shocks. Each week a buyer's payoffs from buying and from waiting get independent
  Gumbel shocks with scale lam. The Bellman equation becomes smooth (log-sum-exp):
      V(v, s) = lam * log( exp((v - p(s)) / lam) + exp(rho (P V)(v, s) / lam) )
  with rho = beta (1 - delta), and the chance of buying is a logit:
      Pr(buy | v, s) = 1 / (1 + exp((rho (P V)(v, s) - (v - p(s))) / lam)).
  As lam -> 0 this is the calibrated model. With lam > 0, simulated moments are smooth in
  the parameters (common random numbers for price paths), so a gradient-based optimizer
  and finite-difference Jacobians work.

  Estimator. theta = (mu, sigma, delta, pi, phi) minimizes (m(theta) - m_hat)' W
  (m(theta) - m_hat), the same 8 doubly robust moments as 15 (busier games), W = diag(1 /
  se^2). Patience beta is profiled over the same grid; lam is fixed (0.05 of the regular
  price) with a sensitivity run at 0.02 and 0.10.
  Standard errors: Var(theta) = (G' W G)^-1, G the Jacobian of the moments (central
  differences), treating the 8 moment estimates as independent (their covariances were not
  saved; the moments come from overlapping windows of the same game-events, so this is an
  approximation). Overidentification: J = criterion at the optimum, chi-squared with
  8 - 5 = 3 degrees of freedom.

Counterfactuals (publisher commits to a schedule; buyers' beliefs match it; steady state
per year; 200 draws of theta from its estimated distribution for 90% intervals):
  schedules       never discount, 1-33% / 34-60% / 61%+ every sale, every other sale, once
                  a year
  welfare         publisher revenue (70% of sales), Valve's share (30%), consumer surplus
                  (valuation minus price at purchase, plus the expected taste shock), total
  platform        sale-week traffic multiplier kappa = 1 (no extra traffic) and 2
Changes are relative to never discounting, in units of never-discount revenue (so +0.05 is
a gain worth 5% of the publisher's revenue without sales); purchases as log change.

  python 31_estimate_buy_or_wait.py
Outputs: results/estimation/
"""

import importlib
import json
import time

import config
import numpy as np
import pandas as pd
from scipy.optimize import minimize

B = importlib.import_module("15_buy_or_wait")
OUT = config.ROOT.parent / "results" / "estimation"
LAM = 0.05
LAM_SENS = [0.02, 0.10]
N_SIM = 300  # price paths per evaluation (common random numbers)
N_DRAWS = 200
EULER = 0.5772156649
VALVE_SHARE = 0.30
NAMES = ["mu", "sigma", "delta", "pi", "phi"]
LABELS = {
    "mu": "mu (mean log valuation, share of regular price)",
    "sigma": "sigma (spread of log valuations)",
    "delta": "delta (weekly chance of losing interest)",
    "pi": "pi (share who know the sale price a week ahead)",
    "phi": "phi (review lag: weekly carry-over)",
}


# ---------------------------------------------------------------- smooth model
def solve_logit(P, price, v, beta, delta, lam):
    """Smooth Bellman fixed point with Gumbel taste shocks; returns V and Pr(buy)."""
    rho = beta * (1 - delta)
    now = v[:, None] - price[None, :]
    V = np.maximum(now, 0)
    for _ in range(50000):
        cont = rho * V @ P.T
        m = np.maximum(now, cont)
        Vn = m + lam * np.log(np.exp((now - m) / lam) + np.exp((cont - m) / lam))
        if np.abs(Vn - V).max() < 1e-10:
            V = Vn
            break
        V = Vn
    cont = rho * V @ P.T
    pb = 1 / (1 + np.exp(np.clip((cont - now) / lam, -700, 700)))
    return V, pb, cont


def step_p(mass, pb, delta, arrivals=1.0):
    """One week with buy probabilities: q purchases; the rest may lose interest."""
    mass = mass + arrivals / mass.shape[1]
    q = (mass * pb).sum(1)
    return q, mass * (1 - pb) * (1 - delta)


def flows_p(path, pb, delta, mass=None):
    n, T = path.shape
    mass = np.zeros((n, pb.shape[0])) if mass is None else mass.copy()
    q = np.zeros((n, T))
    for t in range(T):
        q[:, t], mass = step_p(mass, pb[:, path[:, t]].T, delta)
    return q, mass


def simulate_logit(P, price, theta, beta, lam, draws):
    """As 15_buy_or_wait.simulate, with logit purchase probabilities and fixed draws."""
    mu, sigma, delta, pi, phi = theta
    v = B.valuations(mu, sigma)
    V, pb, _ = solve_logit(P, price, v, beta, delta, lam)
    start, u_pre, u_post = draws
    T0 = B.W * B.PRE_CYCLES
    pre = B.simulate_paths(P, start, u_pre)
    q_pre, mass = flows_p(pre[:, :-1], pb, delta)
    wk1 = pre[:, -1]
    rho = beta * (1 - delta)
    arms = {}
    for k in range(B.D):
        # informed buyers know this week's sale depth k: logit of buying now vs waiting
        cont_inf = rho * V[:, k][None, :]
        now1 = v[None, :] - price[wk1][:, None]
        p_inf = 1 / (1 + np.exp(np.clip((cont_inf - now1) / lam, -700, 700)))
        p_un = pb[:, wk1].T
        q1i, m_inf = step_p(pi * mass, p_inf, delta, pi)
        q1u, m_un = step_p((1 - pi) * mass, p_un, delta, 1 - pi)
        cont = B.simulate_paths(P, np.full(len(start), k), u_post)
        q, _ = flows_p(cont, pb, delta, m_inf + m_un)
        qq = np.hstack([q_pre, (q1i + q1u)[:, None], q])
        pp = np.hstack([price[pre[:, :-1]], price[wk1][:, None], price[cont]])
        arms[k] = (qq, B.lagged(qq, phi), pp)
    return arms, T0


def make_draws(n, seed):
    rng = np.random.default_rng(seed)
    T0 = B.W * B.PRE_CYCLES
    return (
        rng.integers(0, B.D, n),
        rng.random((n, T0 - 1)),
        rng.random((n, 52 - 1)),
    )


def model_moments(z, P, price, mix, beta, lam, draws):
    th = B.unpack(z)
    m = B.moments(B.effects(*simulate_logit(P, price, th, beta, lam, draws), mix))
    return np.array([m[k] for k in B.TARGETS]), m


TGT = np.array([t for t, _ in B.TARGETS.values()])
SE = np.array([s for _, s in B.TARGETS.values()])


def criterion(z, *args):
    m, _ = model_moments(z, *args)
    return float((((m - TGT) / SE) ** 2).sum())


def estimate(P, price, mix, beta, lam, draws, starts):
    best = None
    args = (P, price, mix, beta, lam, draws)
    for s0 in starts:
        r = minimize(
            criterion,
            s0,
            args=args,
            method="Nelder-Mead",
            options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-4},
        )
        r2 = minimize(criterion, r.x, args=args, method="BFGS", options={"gtol": 1e-4})
        r = r2 if r2.fun <= r.fun else r
        if best is None or r.fun < best.fun:
            best = r
    return best


def jacobian(z, args, h=1e-3):
    G = np.zeros((len(TGT), len(z)))
    for j in range(len(z)):
        e = np.zeros(len(z))
        e[j] = h
        G[:, j] = (model_moments(z + e, *args)[0] - model_moments(z - e, *args)[0]) / (
            2 * h
        )
    return G


def theta_cov(z, args):
    """Var(z) = (G'WG)^-1 and the delta-method Var(theta)."""
    G = jacobian(z, args)
    Wm = np.diag(1 / SE**2)
    Vz = np.linalg.pinv(G.T @ Wm @ G)
    # d theta / d z (diagonal): mu: 1, sigma: exp, logits: p(1-p)
    th = np.array(B.unpack(z))
    d = np.array(
        [1.0, th[1], th[2] * (1 - th[2]), th[3] * (1 - th[3]), th[4] * (1 - th[4])]
    )
    Vt = Vz * np.outer(d, d)
    return Vz, Vt, G


# ---------------------------------------------------------------- counterfactuals
def schedule_welfare(depths, pg, theta, beta, lam, years=6, kappa=1.0):
    """Steady-state purchases, revenue and consumer surplus per year under a committed
    schedule, with logit purchases. Surplus of a purchase: v - p plus the expected Gumbel
    shock of the chosen option given purchase, lam (gamma - log Pr(buy))."""
    mu, sigma, delta, _, _ = theta
    P, d = B.schedule_chain(depths)
    p = pg[d]
    v = B.valuations(mu, sigma)
    _, pb, _ = solve_logit(P, p, v, beta, delta, lam)
    L = len(p)
    weeks = 52 * years
    mass = np.zeros((1, B.N_V))
    q = np.zeros(weeks)
    cs = np.zeros(weeks)
    for t in range(weeks):
        s = t % L
        arr = kappa if s % B.W == 0 else 1.0
        mass = mass + arr / B.N_V
        pr = pb[:, s][None, :]
        buy = mass * pr
        q[t] = buy.sum()
        gain = (v - p[s]) + lam * (EULER - np.log(np.clip(pr[0], 1e-300, 1)))
        cs[t] = (buy[0] * gain).sum()
        mass = mass * (1 - pr) * (1 - delta)
    last = slice(weeks - 52 * len(depths), weeks)
    path = np.arange(weeks) % L
    rev = (q[last] * p[path[last]]).sum() / len(depths)
    return {
        "purchases": q[last].sum() / len(depths),
        "revenue": rev,
        "publisher": (1 - VALVE_SHARE) * rev,
        "valve": VALVE_SHARE * rev,
        "consumer surplus": cs[last].sum() / len(depths),
    }


def counterfactuals(pg, theta, beta, lam, kappa):
    rows = {}
    for name, dep in B.SCHEDULES.items():
        rows[name] = schedule_welfare(dep, pg, theta, beta, lam, kappa=kappa)
    base = rows["never discount"]
    R0 = base["revenue"]
    out = {}
    for name, r in rows.items():
        # changes in units of never-discount revenue, so the (model-dependent) level of
        # inframarginal surplus does not matter
        out[name] = {
            "purchases vs never (log)": np.log(r["purchases"] / base["purchases"]),
            "revenue": (r["revenue"] - R0) / R0,
            "publisher (70%)": (r["publisher"] - base["publisher"]) / R0,
            "Valve (30%)": (r["valve"] - base["valve"]) / R0,
            "consumer surplus": (r["consumer surplus"] - base["consumer surplus"]) / R0,
            "total welfare": (
                r["revenue"] + r["consumer surplus"] - R0 - base["consumer surplus"]
            )
            / R0,
        }
    return out


# ---------------------------------------------------------------- main
def setup():
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    busy = ev[ev["analysis_sample"] & (ev["base_rate"] >= config.BUSY_MIN)]
    apps = sorted(busy["app_id"].unique())
    hist = (
        pd.read_parquet(config.PRICE_HISTORY_PARQUET)
        if config.PRICE_HISTORY_PARQUET.exists()
        else pd.read_parquet(B.COMPACT / "price_history.parquet")
    )
    hist["ts"] = pd.to_datetime(hist["ts"], utc=True)
    hist = hist.sort_values(["app_id", "ts"])
    s, e, _, w = B.week_grid()
    _, g, mean_cut = B.weekly_states(hist, apps, s, e)
    pg = 1 - mean_cut / 100
    price = np.tile(pg, B.W)
    P = B.transition(g, w)
    mix = busy.loc[busy["discounted"], "depth_bin"].value_counts(normalize=True)
    depth_mix = {k: float(mix.get(B.GROUPS[k], 0)) for k in range(1, B.D)}
    return P, price, pg, depth_mix


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    P, price, pg, mix = setup()
    draws = make_draws(N_SIM, config.SEED)
    print(f"setup done  [{time.time() - t0:.0f}s]", flush=True)
    calib = json.loads((B.OUT / "calibration_grid.json").read_text())
    starts_all = [B.pack(v[0]) for v in calib.values()] + [
        B.pack((0.3, 1.0, 0.1, 0.4, 0.6)),
        B.pack((0.8, 2.5, 0.08, 0.5, 0.7)),
    ]

    # profile over beta
    prof = {}
    for beta in B.BETAS:
        r = estimate(P, price, mix, beta, LAM, draws, starts_all)
        prof[beta] = r
        print(
            f"  beta {beta}: J {r.fun:.2f}, theta {np.round(B.unpack(r.x), 3)}  "
            f"[{(time.time() - t0) / 60:.0f} min]",
            flush=True,
        )
    beta = min(prof, key=lambda b: prof[b].fun)
    z = prof[beta].x
    th = np.array(B.unpack(z))
    args = (P, price, mix, beta, LAM, draws)
    Vz, Vt, G = theta_cov(z, args)
    se = np.sqrt(np.diag(Vt))
    J = prof[beta].fun
    from scipy.stats import chi2

    par = [
        {"parameter": LABELS[n], "estimate": th[i], "se": se[i]}
        for i, n in enumerate(NAMES)
    ]
    lag = th[4] / (1 - th[4])
    lag_se = se[4] / (1 - th[4]) ** 2
    par += [
        {"parameter": "mean review lag (weeks)", "estimate": lag, "se": lag_se},
        {
            "parameter": "beta (weekly patience, profiled)",
            "estimate": beta,
            "se": np.nan,
        },
        {"parameter": "lam (taste-shock scale, fixed)", "estimate": LAM, "se": np.nan},
        {
            "parameter": "J statistic (3 df)",
            "estimate": J,
            "se": np.nan,
            "p": 1 - chi2.cdf(J, 3),
        },
    ]
    pd.DataFrame(par).to_csv(OUT / "parameters.csv", index=False)
    pd.DataFrame(
        [
            {"beta": b, "J": r.fun, **dict(zip(NAMES, B.unpack(r.x)))}
            for b, r in prof.items()
        ]
    ).to_csv(OUT / "beta_profile.csv", index=False)
    pd.DataFrame(G, index=list(B.TARGETS), columns=NAMES).to_csv(OUT / "jacobian.csv")

    _, m_fit = model_moments(z, *args)
    fit = [
        {"moment": k, "targeted": True, "data": t, "data_se": s_, "model": m_fit[k]}
        for k, (t, s_) in B.TARGETS.items()
    ] + [
        {"moment": k, "targeted": False, "data": t, "data_se": s_, "model": m_fit[k]}
        for k, (t, s_) in B.UNTARGETED.items()
    ]
    pd.DataFrame(fit).to_csv(OUT / "fit.csv", index=False)
    print(f"estimated; SEs done  [{(time.time() - t0) / 60:.0f} min]", flush=True)

    # sensitivity to the taste-shock scale (same beta)
    sens = []
    for lam in LAM_SENS:
        r = estimate(P, price, mix, beta, lam, draws, [z])
        sens.append({"lam": lam, "J": r.fun, **dict(zip(NAMES, B.unpack(r.x)))})
    sens.append({"lam": LAM, "J": J, **dict(zip(NAMES, th))})
    pd.DataFrame(sens).sort_values("lam").to_csv(
        OUT / "lam_sensitivity.csv", index=False
    )
    print(f"lam sensitivity done  [{(time.time() - t0) / 60:.0f} min]", flush=True)

    # counterfactuals with intervals from draws of theta
    rng = np.random.default_rng(config.SEED + 7)
    zs = rng.multivariate_normal(z, Vz, N_DRAWS)
    rows = []
    for kappa in (1.0, 2.0):
        point = counterfactuals(pg, th, beta, LAM, kappa)
        sims = [counterfactuals(pg, B.unpack(zz), beta, LAM, kappa) for zz in zs]
        for name in B.SCHEDULES:
            for key, val in point[name].items():
                d = np.array([s_[name][key] for s_ in sims])
                d = d[np.isfinite(d)]
                rows.append(
                    {
                        "sale-week traffic": kappa,
                        "schedule": name,
                        "outcome": key,
                        "estimate": val,
                        "p05": np.quantile(d, 0.05),
                        "p95": np.quantile(d, 0.95),
                    }
                )
        print(
            f"  counterfactuals kappa {kappa} done  [{(time.time() - t0) / 60:.0f} min]",
            flush=True,
        )
    cf = pd.DataFrame(rows)
    cf.to_csv(OUT / "counterfactuals.csv", index=False)

    pd.set_option("display.width", 220)
    for f in ["parameters", "beta_profile", "fit", "lam_sensitivity"]:
        print(
            f"\n{f}:\n" + pd.read_csv(OUT / f"{f}.csv").round(4).to_string(index=False)
        )
    cf["est [90%]"] = cf.apply(
        lambda x: f"{x['estimate']:+.3f} [{x['p05']:+.3f}, {x['p95']:+.3f}]", axis=1
    )
    print(
        cf.pivot_table(
            index=["sale-week traffic", "schedule"],
            columns="outcome",
            values="est [90%]",
            aggfunc="first",
        ).to_string()
    )
    print(f"Done in {(time.time() - t0) / 60:.0f} minutes")


if __name__ == "__main__":
    main()
