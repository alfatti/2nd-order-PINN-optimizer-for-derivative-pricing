"""
mc_1d.py
========
Monte Carlo pricer for the 1-D discrete autocallable with antithetic variates
and a control variate (the non-callable reverse-convertible payoff, whose
expectation is available in closed form).

Discrete monitoring is exact (the 12 call dates are simulated directly), so the
estimator is unbiased; only statistical error remains. The MC convergence wall
(error ~ 1/sqrt(N)) is the centerpiece of the accuracy plot: reaching 0.1 bp of
vol requires astronomically many paths without heavy variance reduction, which
is the honest reason MC is out of the running at the stringent tolerance.
"""
import numpy as np
from scipy.stats import norm
from config import PAR


def _cv_expectation(par):
    """E[f(S_T)] in closed form for the control variate (uncalled maturity payoff)."""
    mT = (par.r - par.q - 0.5 * par.sig**2) * par.T
    sT = par.sig * np.sqrt(par.T)
    dstar = (np.log(par.L / par.S0) - mT) / sT
    P_above = 1 - norm.cdf(dstar)
    d1 = (np.log(par.L / par.S0) - mT - sT * sT) / sT
    ES_below = par.S0 * np.exp((par.r - par.q) * par.T) * norm.cdf(d1)
    return par.I * P_above + ES_below


def price_mc(par=PAR, N=2_000_000, seed=0, use_cv=True, chunk=1_000_000):
    """Memory-safe chunked MC. Antithetic pairs within each chunk."""
    rng = np.random.default_rng(seed)
    dt = par.T / par.n_calls
    cv_mean = np.exp(-par.disc * par.T) * _cv_expectation(par)
    # accumulators for a pooled mean and (optionally) regression CV
    s_pv = s_pv2 = 0.0
    s_cv = s_cv2 = s_pvcv = 0.0
    n_tot = 0
    remaining = N
    while remaining > 0:
        m = min(chunk, remaining)
        half = m // 2
        Z = rng.standard_normal((half, par.n_calls))
        Z = np.vstack([Z, -Z])
        incr = (par.r - par.q - 0.5 * par.sig**2) * dt + par.sig * np.sqrt(dt) * Z
        S = par.S0 * np.exp(np.cumsum(incr, axis=1))
        called = np.zeros(2 * half, bool); cf = np.zeros(2 * half); ct = np.zeros(2 * half)
        for i in range(par.n_calls):
            hit = (~called) & (S[:, i] >= par.C)
            cf[hit] = par.called_payoff((i + 1) * dt); ct[hit] = (i + 1) * dt
            called[hit] = True
        surv = ~called; ST = S[:, -1]; f = par.maturity_payoff(ST)
        cf[surv] = f[surv]; ct[surv] = par.T
        pv = np.exp(-par.disc * ct) * cf
        cv = np.exp(-par.disc * par.T) * f
        s_pv += pv.sum(); s_pv2 += (pv**2).sum()
        s_cv += cv.sum(); s_cv2 += (cv**2).sum(); s_pvcv += (pv * cv).sum()
        n_tot += len(pv); remaining -= m
    mean_pv = s_pv / n_tot
    if use_cv:
        cov = s_pvcv / n_tot - (s_pv / n_tot) * (s_cv / n_tot)
        var_cv = s_cv2 / n_tot - (s_cv / n_tot) ** 2
        beta = cov / var_cv if var_cv > 0 else 0.0
        mean_pv = mean_pv - beta * (s_cv / n_tot - cv_mean)
        var_pv = s_pv2 / n_tot - (s_pv / n_tot) ** 2
        var_adj = max(var_pv - beta**2 * var_cv, 0.0)
        se = np.sqrt(var_adj / n_tot)
    else:
        var_pv = s_pv2 / n_tot - mean_pv**2
        se = np.sqrt(var_pv / n_tot)
    return float(mean_pv), float(se)


def convergence_curve(par=PAR, Ns=(1e4, 1e5, 1e6, 1e7), seed=0, use_cv=True):
    out = []
    for N in Ns:
        m, se = price_mc(par, N=int(N), seed=seed, use_cv=use_cv)
        out.append((int(N), m, se))
    return out


if __name__ == "__main__":
    for N in (2e5, 2e6):
        m, se = price_mc(N=int(N))
        print(f"N={int(N):>9d}  price={m:.4f}  se={se:.4f}  (ref 97.51)")
