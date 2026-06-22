"""
wof3_mc.py  (includes the shared worst-of-3 payoff spec)
=======================================================
Worst-of-3 autocallable: the call / threshold / maturity tests are driven by the
WORST-performing of three correlated underlyings (each normalized to its own
spot). This is the classic structure where FD dies (state dimension 3 -> grid
cost N^3) while MC and PINN scale mildly.

Driver:  W_t = min_k S^(k)_t / S0_k      (worst performer ratio)
  * call date i: if W_{t_i} >= C/S0  -> called, pay P_{t_i} = H e^{B t_i}
  * never called: maturity payoff f(W_T) = I if W_T > L/S0 else I * W_T
    (you absorb the worst performer's loss)

MC is the GROUND TRUTH in 3-D: discrete monitoring is exact (unbiased), and cost
is independent of dimension. Antithetic + correlated GBM via Cholesky.
"""
import numpy as np
from config import WOF3


def worst_of_payoffs(W_path, par):
    """Given worst-ratio path (N, n_calls), return cashflow and time."""
    N, nc = W_path.shape
    dt = par.T / nc
    c_level = par.C / par.S0
    L_level = par.L / par.S0
    called = np.zeros(N, bool)
    cf = np.zeros(N); ct = np.zeros(N)
    for i in range(nc):
        hit = (~called) & (W_path[:, i] >= c_level)
        cf[hit] = par.H * np.exp(par.B * (i + 1) * dt)
        ct[hit] = (i + 1) * dt
        called[hit] = True
    surv = ~called
    WT = W_path[surv, -1]
    cf[surv] = np.where(WT > L_level, par.H, par.H * WT)
    ct[surv] = par.T
    return cf, ct


def price_mc(par=WOF3, N=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    d = len(par.vols)
    nc = par.n_calls
    dt = par.T / nc
    chol = par.chol
    vols = np.array(par.vols)

    half = N // 2
    # correlated normals: (half, nc, d)
    Z = rng.standard_normal((half, nc, d))
    Z = np.concatenate([Z, -Z], axis=0)                  # antithetic
    Zc = Z @ chol.T                                      # correlate across assets
    drift = (par.r - par.q - 0.5 * vols**2) * dt
    incr = drift + vols * np.sqrt(dt) * Zc               # (N, nc, d)
    logratio = np.cumsum(incr, axis=1)                   # log(S^k_t / S0_k)
    W = np.exp(logratio).min(axis=2)                     # worst ratio (N, nc)

    cf, ct = worst_of_payoffs(W, par)
    pv = np.exp(-par.disc * ct) * cf
    return float(pv.mean()), float(pv.std() / np.sqrt(len(pv)))


def convergence_curve(par=WOF3, Ns=(1e4, 1e5, 1e6, 1e7), seed=0):
    return [(int(N), *price_mc(par, N=int(N), seed=seed)) for N in Ns]


if __name__ == "__main__":
    for N in (2e5, 2e6):
        m, se = price_mc(N=int(N))
        print(f"WoF3 MC  N={int(N):>9d}  price={m:.4f}  se={se:.4f}")
