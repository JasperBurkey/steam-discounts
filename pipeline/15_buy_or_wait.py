"""Step 15: a calibrated buy-or-wait model (QuantEcon lecture models).

What do buyers who wait for sales, and reviews written after the purchase, imply for the
estimated effects, for the review-based revenue proxy, and for discount schedules over a
whole year?

Three parts, each a QuantEcon lecture model:
  1. Discount beliefs (Finite Markov Chains). Each week a game is at full price or in one
     of three depth groups (1-33%, 34-60%, 61%+). The state is (weeks until the next Valve
     sale, discount group). Transitions are estimated from the IsThereAnyDeal price
     histories of the busier study games; rare and habitual discounters separately too.
  2. The buyer's problem (McCall job search; quantecon.markov.DiscreteDP). A potential
     buyer who values the game at v (share of its regular price) buys at this week's price
     or waits; waiting is discounted by beta and by the weekly chance delta of losing interest:
         V(s) = max( v - p(s),  beta (1 - delta) E[V(s') | s] )
     In the week before a sale, a share pi of buyers already knows this game's sale price
     (announcements, wishlist previews, deal sites) and decides on that instead of beliefs.
  3. Reviews and calibration. New buyers arrive every week with v ~ lognormal(mu, sigma).
     Reviews follow purchases with a lag: a share (1 - phi) phi^k of a week's buyers review
     k weeks later. The five parameters (mu, sigma, delta, pi, phi; beta fixed) are chosen so
     that a simulated discount reproduces the doubly robust profile for busier games: the
     week before, the sale, weeks 1-2, 3-4 and 5-6 after, and the sale-week effect by depth.

Uses: (a) does the model need waiting at all; (b) the review-based revenue proxy against
true revenue (purchases x price paid) inside the model, which shows how review lag biases
the proxy; (c) the publisher's side: discount schedules the publisher commits to, with
buyers' beliefs matching the schedule, compared by revenue over a whole year.

Calibrated, not estimated: no standard errors; an interpretation of the causal estimates.

  python 15_buy_or_wait.py
Outputs: results/buy_or_wait/
"""

import time

import config
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from prices import price_arrays, sale_times  # noqa: E402
from quantecon.markov import DiscreteDP, MarkovChain  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import norm  # noqa: E402

OUT = config.ROOT.parent / "results" / "buy_or_wait"
COMPACT = config.DATA / "for_claude"

W = 13  # weeks in a sale cycle: 0 = sale week, 12 = just after a sale
D = 4  # discount groups: full price, 1-33%, 34-60%, 61%+
GROUPS = ["full price", "1-33%", "34-60%", "61%+"]
EDGES = [0, 33, 60]  # cut > edge -> next group
N_V = 80  # valuation grid (equal-weight quantiles)
BETAS = [0.97, 0.98, 0.99, 0.995, 0.998]  # weekly patience grid; best fit is kept
SMOOTH = 0.5  # additive smoothing for transition counts
PRE_CYCLES = 20  # burn-in, in sale cycles
AFTER = 7  # simulated weeks from the sale week on (0..6)

# Doubly robust ATEs for busier games (>= 1 review/day), full ensemble settings.
# Windows: D = week before; E = sale; F = weeks 1-2 after; G = 3-4; H = 5-6.
TARGETS = {
    "week before the sale": (-0.060, 0.017),
    "during the sale": (0.311, 0.023),
    "weeks 1-2 after": (0.191, 0.024),
    "weeks 3-4 after": (0.052, 0.024),
    "weeks 5-6 after": (0.054, 0.024),
    "during, 1-33%": (0.206, 0.024),
    "during, 34-60%": (0.261, 0.026),
    "during, 61%+": (0.610, 0.031),
}
UNTARGETED = {
    "revenue proxy, during (ATT)": (-0.535, 0.028),
    "revenue proxy, net (ATT)": (-0.072, 0.021),
}
WINDOWS = {  # model weeks relative to the sale week
    "week before the sale": [-1],
    "during the sale": [0],
    "weeks 1-2 after": [1, 2],
    "weeks 3-4 after": [3, 4],
    "weeks 5-6 after": [5, 6],
    "net": [-1, 0, 1, 2, 3, 4],
}


# ---------------------------------------------------------------- 1. discount beliefs
def week_grid():
    cal = pd.read_csv(config.ROOT / "sale_calendar.csv")
    spans = [
        tuple(int(x.timestamp()) for x in sale_times(r)) for _, r in cal.iterrows()
    ]
    # weeks start Thursday 18:00 UTC, near Valve's 10:00 Pacific sale starts
    starts = pd.date_range("2023-01-05 18:00", "2026-11-05 18:00", freq="7D", tz="UTC")
    s = np.array([int(t.timestamp()) for t in starts])
    e = s + 7 * 86400
    sale = np.array(
        [
            any(min(b, e_) - max(a, s_) >= 4 * 86400 for a, b in spans)
            for s_, e_ in zip(s, e)
        ]
    )
    w = np.full(len(s), W - 1)  # weeks until the next sale week, capped at W - 1
    nxt = None
    for i in range(len(s) - 1, -1, -1):
        if sale[i]:
            nxt = i
        if nxt is not None:
            w[i] = min(nxt - i, W - 1)
    return s, e, sale, w


def group(cut):
    return int(np.searchsorted(EDGES, cut, side="left")) if cut > 0 else 0


def weekly_states(hist, apps, s, e):
    """games x weeks discount groups (-1 = unknown) and the mean cut in each group."""
    arrays = price_arrays(hist[hist["app_id"].isin(apps)])
    apps = [a for a in apps if a in arrays]
    g = np.full((len(apps), len(s)), -1, int)
    cuts = {k: [] for k in range(D)}
    for i, a in enumerate(apps):
        pa = arrays[a]
        for t in range(len(s)):
            # middle five days: sales start at 10:00 Pacific, which can fall an hour
            # either side of the week boundary depending on daylight saving time
            c = pa.max_cut(int(s[t]) + 86400, int(e[t]) - 86400)
            if np.isfinite(c):
                k = group(c)
                g[i, t] = k
                cuts[k].append(c)
    mean_cut = np.array([0.0] + [np.mean(cuts[k]) for k in range(1, D)])
    return np.array(apps), g, mean_cut


def transition(g, w, games=None):
    """P[(w, d) -> (w', d')]: w' follows the cycle; d' estimated given (d, w')."""
    rows = g if games is None else g[games]
    cnt = np.full((W, D, D), SMOOTH)  # [w', d, d']
    for t in range(rows.shape[1] - 1):
        a, b = rows[:, t], rows[:, t + 1]
        ok = (a >= 0) & (b >= 0)
        np.add.at(cnt, (np.full(ok.sum(), w[t + 1]), a[ok], b[ok]), 1)
    pdd = cnt / cnt.sum(2, keepdims=True)
    P = np.zeros((W * D, W * D))
    for wi in range(W):
        wn = W - 1 if wi == 0 else wi - 1
        for d in range(D):
            P[wi * D + d, wn * D : (wn + 1) * D] = pdd[wn, d]
    return P


def schedule_chain(depths):
    """Deterministic schedule a publisher commits to: depth group depths[k] in the k-th sale
    of each cycle of len(depths) sales, full price otherwise. States: week of the cycle."""
    L = W * len(depths)
    P = np.zeros((L, L))
    d = np.zeros(L, int)
    for t in range(L):
        P[t, (t + 1) % L] = 1
        if t % W == 0:
            d[t] = depths[t // W]
    return P, d


# ---------------------------------------------------------------- 2. the buyer's problem
def solve(P, price, v, beta, delta):
    """Values (valuations x states) and buy decisions, vectorized over valuations.
    Same Bellman equation as solve_discretedp (checked in the tests)."""
    rho = beta * (1 - delta)
    now = v[:, None] - price[None, :]
    V = np.maximum(now, 0)
    for _ in range(20000):
        Vn = np.maximum(now, rho * V @ P.T)
        if np.abs(Vn - V).max() < 1e-11:
            V = Vn
            break
        V = Vn
    return V, now >= rho * V @ P.T


def solve_discretedp(P, price, v, beta, delta):
    """The same problem for one valuation with quantecon's DiscreteDP: price states plus an
    absorbing 'done' state; actions 0 = wait, 1 = buy."""
    n = P.shape[0]
    R = np.zeros((n + 1, 2))
    R[:n, 1] = v - price
    R[n, 1] = -1e9  # nothing left to buy
    Q = np.zeros((n + 1, 2, n + 1))
    Q[:n, 0, :n] = (1 - delta) * P
    Q[:n, 0, n] = delta  # loses interest
    Q[:n, 1, n] = 1
    Q[n, :, n] = 1
    res = DiscreteDP(R, Q, beta).solve(method="policy_iteration")
    return res.v[:n], res.sigma[:n] == 1


# ---------------------------------------------------------------- 3. simulation
def valuations(mu, sigma):
    q = (np.arange(N_V) + 0.5) / N_V
    return np.exp(mu + sigma * norm.ppf(q))


def step(mass, b, delta, arrivals=1.0):
    """One week: arrivals join, buyers with b True buy, the rest may lose interest."""
    mass = mass + arrivals / mass.shape[1]
    q = (mass * b).sum(1)
    return q, np.where(b, 0, mass * (1 - delta))


def flows(path, buy, delta, mass=None):
    n, T = path.shape
    mass = np.zeros((n, buy.shape[0])) if mass is None else mass.copy()
    q = np.zeros((n, T))
    for t in range(T):
        q[:, t], mass = step(mass, buy[:, path[:, t]].T, delta)
    return q, mass


def lagged(q, phi):
    """Reviews: a share (1 - phi) phi^k of a week's buyers review k weeks later."""
    r = np.zeros_like(q)
    acc = np.zeros(q.shape[0])
    for t in range(q.shape[1]):
        acc = phi * acc + (1 - phi) * q[:, t]
        r[:, t] = acc
    return r


def simulate_paths(P, start, u):
    cum = np.cumsum(P, 1)
    path = [np.asarray(start)]
    for t in range(u.shape[1]):
        path.append((cum[path[-1]] < u[:, [t]]).sum(1))
    return np.column_stack(path)


def simulate(P, price, theta, beta, n=400, seed=0):
    """Fork price paths at a sale week: each depth vs full price. Returns per-arm weekly
    purchases, reviews and prices for weeks -1..AFTER-1 (plus a long tail for revenue)."""
    mu, sigma, delta, pi, phi = theta
    v = valuations(mu, sigma)
    V, buy = solve(P, price, v, beta, delta)
    rng = np.random.default_rng(seed)
    T0 = W * PRE_CYCLES  # path starts at a sale week, so step T0 is a sale week again
    pre = simulate_paths(P, rng.integers(0, D, n), rng.random((n, T0 - 1)))
    q_pre, mass = flows(pre[:, :-1], buy, delta)  # up to two weeks before the sale
    wk1 = pre[:, -1]  # the week before the sale (w = 1)
    rho = beta * (1 - delta)
    u = rng.random((n, 52 - 1))
    arms = {}
    for k in range(D):
        # week before: informed buyers compare buying now with waiting for the known price
        b_inf = (v[None, :] - price[wk1][:, None]) >= rho * V[:, k][None, :]
        b_uninf = buy[:, wk1].T
        m_inf, m_un = pi * mass, (1 - pi) * mass
        q1i, m_inf = step(m_inf, b_inf, delta, pi)
        q1u, m_un = step(m_un, b_uninf, delta, 1 - pi)
        cont = simulate_paths(P, np.full(n, k), u)  # sale week in group k, then a year
        q, _ = flows(cont, buy, delta, m_inf + m_un)
        qq = np.hstack([q_pre, (q1i + q1u)[:, None], q])
        pp = np.hstack([price[pre[:, :-1]], price[wk1][:, None], price[cont]])
        arms[k] = (qq, lagged(qq, phi), pp)
    return arms, T0  # column T0 = sale week


EPS = 1e-12


def _lr(a, b):
    return np.log((a + EPS) / (b + EPS))


def effects(arms, T0, depth_mix):
    """Window effects (log ratio to the no-discount arm), averaged over paths: reviews by
    window and depth; revenue proxy (reviews x price in that week) and true revenue."""
    qc, rc, pc = arms[0]
    out = {}
    for k in range(1, D):
        qt, rt, pt = arms[k]
        o = {}
        for name, weeks in WINDOWS.items():
            cols = [T0 + x for x in weeks]
            o[name] = np.mean(_lr(rt[:, cols].sum(1), rc[:, cols].sum(1)))
            o["proxy " + name] = np.mean(
                _lr(
                    (rt[:, cols] * pt[:, cols]).sum(1),
                    (rc[:, cols] * pc[:, cols]).sum(1),
                )
            )
            o["true " + name] = np.mean(
                _lr(
                    (qt[:, cols] * pt[:, cols]).sum(1),
                    (qc[:, cols] * pc[:, cols]).sum(1),
                )
            )
        year = slice(T0 - 1, T0 + 51)
        o["true revenue, year"] = np.mean(
            _lr((qt[:, year] * pt[:, year]).sum(1), (qc[:, year] * pc[:, year]).sum(1))
        )
        o["purchases, year"] = np.mean(_lr(qt[:, year].sum(1), qc[:, year].sum(1)))
        o["curve"] = rt[:, T0 - 3 : T0 + 7].mean(0)
        o["curve_q"] = qt[:, T0 - 3 : T0 + 7].mean(0)
        out[k] = o
    out["curve_c"] = rc[:, T0 - 3 : T0 + 7].mean(0)
    out["curve_q_c"] = qc[:, T0 - 3 : T0 + 7].mean(0)
    w = np.array([depth_mix[k] for k in range(1, D)])
    w = w / w.sum()
    out["mix"] = {
        key: float(np.dot(w, [out[k][key] for k in range(1, D)]))
        for key in out[1]
        if not key.startswith("curve")
    }
    return out


def moments(out):
    m = {name: out["mix"][name] for name in list(WINDOWS)[:5]}
    m.update({"during, " + GROUPS[k]: out[k]["during the sale"] for k in range(1, D)})
    m["revenue proxy, during (ATT)"] = out["mix"]["proxy during the sale"]
    m["revenue proxy, net (ATT)"] = out["mix"]["proxy net"]
    return m


def unpack(z):
    """Unconstrained z -> (mu, sigma, delta, pi, phi)."""
    lg = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    return (z[0], np.exp(z[1]), lg(z[2]), lg(z[3]), lg(z[4]))


def pack(theta):
    lgt = lambda p: np.log(p / (1 - p))  # noqa: E731
    mu, sigma, delta, pi, phi = theta
    return np.array([mu, np.log(sigma), lgt(delta), lgt(pi), lgt(phi)])


def calibrate(P, price, depth_mix, beta=0.99, theta0=(0.0, 0.9, 0.05, 0.3, 0.4), n=250):
    """Weighted least squares of model moments on TARGETS (fixed simulation seed)."""

    def loss(z):
        m = moments(effects(*simulate(P, price, unpack(z), beta, n=n), depth_mix))
        return sum(((m[k] - t) / se) ** 2 for k, (t, se) in TARGETS.items())

    best = None
    for start in [theta0, (0.3, 0.6, 0.02, 0.5, 0.6), (-0.3, 1.2, 0.15, 0.1, 0.2)]:
        r = minimize(
            loss,
            pack(start),
            method="Nelder-Mead",
            options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-4},
        )
        if best is None or r.fun < best.fun:
            best = r
    return unpack(best.x), best.fun


# ---------------------------------------------------------------- publisher's side
def schedule_revenue(depths, price_by_group, theta, beta, years=6, kappa=1.0):
    """Steady-state purchases and revenue per year under a committed schedule. kappa
    multiplies new arrivals in sale weeks (Steam's sale traffic), for every schedule."""
    mu, sigma, delta, _, _ = theta
    P, d = schedule_chain(depths)
    p = price_by_group[d]
    _, buy = solve(P, p, valuations(mu, sigma), beta, delta)
    L = len(p)
    weeks = 52 * years
    path = (np.arange(weeks) % L)[None, :]
    mass = np.zeros((1, N_V))
    q = np.zeros(weeks)
    for t in range(weeks):
        arr = kappa if t % W == 0 else 1.0
        q[t], mass = step(mass, buy[:, [path[0, t]]].T, delta, arr)
    last = slice(weeks - 52 * len(depths), weeks)  # whole cycles in steady state
    return q[last].sum() / len(depths), (q[last] * p[path[0, last]]).sum() / len(depths)


SCHEDULES = {
    "never discount": [0],
    "1-33% every sale": [1],
    "34-60% every sale": [2],
    "61%+ every sale": [3],
    "34-60% every other sale": [2, 0],
    "61%+ every other sale": [3, 0],
    "61%+ once a year": [3, 0, 0, 0],
}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    busy = ev[ev["analysis_sample"] & (ev["base_rate"] >= config.BUSY_MIN)]
    apps = sorted(busy["app_id"].unique())
    hist = (
        pd.read_parquet(config.PRICE_HISTORY_PARQUET)
        if config.PRICE_HISTORY_PARQUET.exists()
        else pd.read_parquet(COMPACT / "price_history.parquet")
    )
    hist["ts"] = pd.to_datetime(hist["ts"], utc=True)
    hist = hist.sort_values(["app_id", "ts"])
    s, e, sale, w = week_grid()
    games, g, mean_cut = weekly_states(hist, apps, s, e)
    pg = 1 - mean_cut / 100  # price by group, share of regular price
    price = np.tile(pg, W)
    print(
        f"Weekly discount states: {len(games):,} busier games x {len(s)} weeks; mean cuts {np.round(mean_cut, 1)}  [{time.time() - t0:.0f}s]"
    )

    known = g >= 0
    share = ((g > 0) & known).sum(1) / np.maximum(known.sum(1), 1)
    lo, hi = np.quantile(share, [1 / 3, 2 / 3])
    types = {
        "all busier games": None,
        "rare discounters": share <= lo,
        "habitual discounters": share >= hi,
    }
    chains = {k: transition(g, w, m) for k, m in types.items()}

    rows = []
    for k, Pk in chains.items():
        st = MarkovChain(Pk).stationary_distributions[0].reshape(W, D)
        st = st / st.sum(1, keepdims=True)
        m = types[k] if types[k] is not None else np.ones(len(games), bool)
        for wi, lab in [
            (0, "sale week"),
            (1, "week before a sale"),
            (6, "mid-cycle"),
            (12, "week after a sale"),
        ]:
            obs = g[m][:, w == wi]
            obs = obs[obs >= 0]
            rows.append(
                {
                    "type": k,
                    "week": lab,
                    "chain: share discounted": 1 - st[wi, 0],
                    "data: share discounted": float((obs > 0).mean()),
                    "chain: share 61%+": st[wi, 3],
                    "data: share 61%+": float((obs == 3).mean()),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "discount_chains.csv", index=False)

    mix = busy.loc[busy["discounted"], "depth_bin"].value_counts(normalize=True)
    depth_mix = {k: float(mix.get(GROUPS[k], 0)) for k in range(1, D)}
    P = chains["all busier games"]

    # the vectorized solver against quantecon's DiscreteDP
    v_test = valuations(0.0, 0.9)
    Vt, pol = solve(P, price, v_test, 0.99, 0.05)
    agree = np.mean(
        [
            np.array_equal(pol[i], solve_discretedp(P, price, v_test[i], 0.99, 0.05)[1])
            for i in range(0, N_V, 4)
        ]
    )
    print(f"DiscreteDP agreement on sampled valuations: {agree:.0%}")

    # patience is not pinned down a priori: calibrate at each beta, keep the best fit
    grid = {}
    theta0 = (0.0, 0.9, 0.05, 0.3, 0.4)
    for b in BETAS:
        grid[b] = calibrate(P, price, depth_mix, beta=b, theta0=theta0)
        print(
            f"  beta {b}: loss {grid[b][1]:.2f}  [{time.time() - t0:.0f}s]", flush=True
        )
    beta = min(grid, key=lambda b: grid[b][1])
    theta, fit = grid[beta]
    print(
        f"Best fit at beta {beta}: mu {theta[0]:.3f}, sigma {theta[1]:.3f}, delta {theta[2]:.4f}, pi {theta[3]:.3f}, phi {theta[4]:.3f}; loss {fit:.2f}",
        flush=True,
    )
    out = effects(*simulate(P, price, theta, beta, n=2000, seed=1), depth_mix)
    m = moments(out)
    fitrows = [
        {"moment": k, "targeted": True, "data": t, "data_se": se, "model": m[k]}
        for k, (t, se) in TARGETS.items()
    ]
    fitrows += [
        {"moment": k, "targeted": False, "data": t, "data_se": se, "model": m[k]}
        for k, (t, se) in UNTARGETED.items()
    ]
    pd.DataFrame(fitrows).to_csv(OUT / "calibration_fit.csv", index=False)
    names = [
        "mu (log valuation, share of regular price)",
        "sigma (spread of log valuations)",
        "delta (weekly chance of losing interest)",
        "pi (share who know the sale price a week ahead)",
        "phi (review lag: weekly carry-over)",
    ]
    par = [{"parameter": n_, "value": float(x)} for n_, x in zip(names, theta)]
    par += [
        {"parameter": "beta (weekly patience, best fit on the grid)", "value": beta},
        {"parameter": "mean review lag (weeks)", "value": theta[4] / (1 - theta[4])},
        {
            "parameter": "fit loss (sum of squared t-statistics, 8 moments)",
            "value": fit,
        },
    ]
    pd.DataFrame(par).to_csv(OUT / "parameters.csv", index=False)

    # (a) is waiting needed? refit with pi = 0 and delta = 1 (no waiting at all)
    def loss_fixed(z, fixed):
        th = list(unpack(z))
        th[2], th[3] = fixed
        mm = moments(effects(*simulate(P, price, th, beta, n=250), depth_mix))
        return sum(((mm[k] - t) / se) ** 2 for k, (t, se) in TARGETS.items())

    nw = minimize(
        lambda z: loss_fixed(z, (0.999, 0.0)),
        pack(theta),
        method="Nelder-Mead",
        options={"maxiter": 600},
    )
    th_nw = list(unpack(nw.x))
    th_nw[2], th_nw[3] = 0.999, 0.0
    m_nw = moments(effects(*simulate(P, price, th_nw, beta, n=2000, seed=1), depth_mix))
    pd.DataFrame(
        [
            {"moment": k, "data": t, "with waiting": m[k], "without waiting": m_nw[k]}
            for k, (t, _) in TARGETS.items()
        ]
        + [
            {
                "moment": "fit loss",
                "data": np.nan,
                "with waiting": fit,
                "without waiting": nw.fun,
            }
        ]
    ).to_csv(OUT / "waiting_test.csv", index=False)

    # (b) revenue proxy vs true revenue inside the model
    rv = []
    for k in range(1, D):
        rv.append(
            {
                "depth": GROUPS[k],
                "proxy, net window": out[k]["proxy net"],
                "true, net window": out[k]["true net"],
                "true, whole year": out[k]["true revenue, year"],
                "purchases, whole year": out[k]["purchases, year"],
            }
        )
    rv.append(
        {
            "depth": "mix of discounted games",
            "proxy, net window": out["mix"]["proxy net"],
            "true, net window": out["mix"]["true net"],
            "true, whole year": out["mix"]["true revenue, year"],
            "purchases, whole year": out["mix"]["purchases, year"],
        }
    )
    pd.DataFrame(rv).to_csv(OUT / "proxy_vs_true.csv", index=False)

    # figure: reviews and purchases around a sale, deep discount vs none (model)
    rel = np.arange(-3, 7)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (ct, cc, lab) in zip(
        axes,
        [
            (out[3]["curve"], out["curve_c"], "Reviews"),
            (out[3]["curve_q"], out["curve_q_c"], "Purchases"),
        ],
    ):
        ax.plot(rel, cc / cc[0], "o-", color="0.5", label="No discount in this sale")
        ax.plot(rel, ct / cc[0], "o-", color="C0", label="61%+ discount in this sale")
        ax.axvline(0, color="black", lw=0.8, ls=":")
        ax.set_title(f"{lab} (model)")
        ax.set_xlabel("Weeks relative to the sale week")
    axes[0].set_ylabel("Relative to week -3")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "model_event_study.png", dpi=150)
    plt.close(fig)

    # habit types: same buyers, different beliefs (untargeted)
    hab = []
    for k in ["rare discounters", "habitual discounters"]:
        o = effects(*simulate(chains[k], price, theta, beta, n=2000, seed=2), depth_mix)
        hab.append(
            {
                "type": k,
                "week before": o["mix"]["week before the sale"],
                "during": o["mix"]["during the sale"],
                "true revenue, net window": o["mix"]["true net"],
                "true revenue, whole year": o["mix"]["true revenue, year"],
            }
        )
    pd.DataFrame(hab).to_csv(OUT / "habit_types.csv", index=False)

    # (c) publisher's side: committed schedules, revenue per year
    sched = []
    for kappa in [1.0, 2.0, 3.0]:
        for name, dep in SCHEDULES.items():
            q, r = schedule_revenue(dep, pg, theta, beta, kappa=kappa)
            sched.append(
                {
                    "sale-week traffic": kappa,
                    "schedule": name,
                    "purchases per year": q,
                    "revenue per year": r,
                }
            )
    sched = pd.DataFrame(sched)
    g0 = sched.groupby("sale-week traffic")
    sched["purchases vs never (log)"] = np.log(
        sched["purchases per year"] / g0["purchases per year"].transform("first")
    )
    sched["revenue vs never (log)"] = np.log(
        sched["revenue per year"] / g0["revenue per year"].transform("first")
    )
    sched.to_csv(OUT / "schedules.csv", index=False)

    # sensitivity: every beta on the grid, with its own calibration
    sens = []
    for b, (th_b, fit_b) in grid.items():
        for name in [
            "never discount",
            "1-33% every sale",
            "61%+ every sale",
            "61%+ once a year",
        ]:
            _, r = schedule_revenue(SCHEDULES[name], pg, th_b, b)
            sens.append(
                {
                    "beta": b,
                    "fit loss": fit_b,
                    "delta": th_b[2],
                    "pi": th_b[3],
                    "phi": th_b[4],
                    "sigma": th_b[1],
                    "schedule": name,
                    "revenue per year": r,
                }
            )
    sens = pd.DataFrame(sens)
    sens["revenue vs never (log)"] = sens.groupby("beta")["revenue per year"].transform(
        lambda x: np.log(x / x.iloc[0])
    )
    sens.to_csv(OUT / "sensitivity.csv", index=False)

    pd.set_option("display.width", 220)
    for f in [
        "calibration_fit",
        "parameters",
        "waiting_test",
        "proxy_vs_true",
        "habit_types",
        "schedules",
        "sensitivity",
        "discount_chains",
    ]:
        print(
            f"\n{f}:\n" + pd.read_csv(OUT / f"{f}.csv").round(3).to_string(index=False)
        )
    print(f"\nDone in {(time.time() - t0) / 60:.0f} minutes")


if __name__ == "__main__":
    main()
