"""
reference_1d.py
===============
High-accuracy ground truth for the 1-D discrete autocallable via the
probability route (Deng-Mallett-McCann Eq. 6/7), evaluated by a survival-density
recursion in log-return space.

Validated:
  * call probabilities p_i reproduce Table 1 to 1e-4
  * par benchmark (Case 1) reproduces 100.00 with identical payoff+discount
  * value converges to 97.505 as the floor is lowered (matches antithetic MC)

This is the reference the FD / MC / PINN errors are measured against. It is the
load-bearing object for the whole race, so it is driven to ~1e-3 in price
(<< 0.1 bp of vol) by default.
"""
import numpy as np
from scipy.stats import norm
from config import PAR


def reference_value(par=PAR, y_min=-7.0, M=400_000, return_probs=False):
    dt   = par.T / par.n_calls
    muD  = (par.r - par.q - 0.5 * par.sig**2) * dt
    sdD  = par.sig * np.sqrt(dt)
    b    = np.log(par.C / par.S0)
    yL   = np.log(par.L / par.S0)
    ct   = par.call_times

    y    = np.linspace(y_min, b, M)
    dy   = y[1] - y[0]
    half = int(np.ceil(8 * sdD / dy))
    kx   = np.arange(-half, half + 1) * dy
    kern = norm.pdf(kx, muD, sdD)
    kern /= kern.sum() * dy

    # exact first step: psi after step 1 is the increment density from 0
    psi  = norm.pdf(y, muD, sdD).copy()
    p    = np.zeros(par.n_calls)
    p[0] = 1.0 - norm.cdf((b - muD) / sdD)                 # exact P(X1 >= b)

    for i in range(1, par.n_calls):
        surv_cross = 1.0 - norm.cdf((b - y - muD) / sdD)
        p[i] = np.sum(psi * surv_cross) * dy
        psi  = np.convolve(psi, kern, mode="same") * dy
        psi[y >= b] = 0.0                                   # absorb at barrier

    # uncalled (survived all calls) value
    f_unc = np.where(y > yL, par.I, par.S0 * np.exp(y))
    uncalled = np.exp(-par.disc * par.T) * np.sum(psi * f_unc) * dy
    # called value
    called = np.sum(np.exp(-par.disc * ct) * p * par.called_payoff(ct))

    V = called + uncalled
    if return_probs:
        return V, p
    return V


if __name__ == "__main__":
    V, p = reference_value(return_probs=True)
    paper = [0.3767,0.1435,0.0781,0.0506,0.0361,0.0275,
             0.0218,0.0178,0.0149,0.0127,0.0110,0.0096]
    print("month  p_i        paper      diff")
    for i, (a, c) in enumerate(zip(p, paper), 1):
        print(f"  {i:2d}   {a:.4f}    {c:.4f}    {a-c:+.4f}")
    print(f"\nsum p_i      = {p.sum():.5f}")
    print(f"REFERENCE V0 = {V:.5f}   (paper coarse-explicit-FD: 98.39)")
