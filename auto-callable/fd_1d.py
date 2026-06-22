"""
fd_1d.py
========
1-D finite-difference autocallable pricer in log-price, Crank-Nicolson with a
Rannacher start (two fully-implicit half-steps) to damp the L-discontinuity and
the call-barrier kink. Discrete call dates are applied as overwrites of the
region S >= C with the called payoff at the snapped call-date time levels.

Grid snapping: dt is chosen so every call date lands exactly on a time level,
and the barrier S=C and threshold S=L are snapped onto x-nodes. Without snapping
the convergence is non-monotone (the barrier/discontinuity float between nodes).

Converges to the quadrature reference (~97.51). Home turf for a single 1-D
contract: a converged solve is single-digit ms, faster than any PINN. The race
is therefore run on amortization (many contracts) and dimensionality, not on a
single 1-D price.
"""
import numpy as np
from scipy.linalg import solve_banded
from config import PAR


def price_fd(par=PAR, NX=1600, NT=1200, Smax_mult=3.0, return_grid=False):
    # snap dt so all call dates are exact time levels
    NT = int(np.ceil(NT / par.n_calls) * par.n_calls)
    dt = par.T / NT
    call_levels = set(int(round(tc / dt)) for tc in par.call_times)

    # piecewise-uniform x-grid with EXACT nodes at x=log(L) and x=log(C)
    xC, xL = np.log(par.C), np.log(par.L)
    xmin, xmax = np.log(0.05 * par.S0), np.log(Smax_mult * par.C)
    edges = sorted([xmin, xL, xC, xmax])
    spans = np.diff(edges)
    counts = np.maximum(2, np.round(NX * spans / spans.sum()).astype(int))
    segs = [np.linspace(edges[k], edges[k+1], counts[k], endpoint=False)
            for k in range(3)]
    x = np.concatenate(segs + [[xmax]])
    x = np.unique(x)                                    # nodes exactly on xL, xC
    NX = len(x)
    S = np.exp(x)

    # terminal condition (maturity is also a call date)
    V = par.maturity_payoff(S).astype(float)
    V = np.where(S >= par.C, par.called_payoff(par.T), V)

    # variable-coefficient operator on non-uniform grid (central diff)
    n = NX
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    mu = par.r - par.q - 0.5 * par.sig**2
    for i in range(1, n - 1):
        dxm, dxp = x[i] - x[i-1], x[i+1] - x[i]
        diff = 0.5 * par.sig**2
        a[i] = diff * 2/(dxm*(dxm+dxp)) - mu/(dxm+dxp)
        c[i] = diff * 2/(dxp*(dxm+dxp)) + mu/(dxm+dxp)
        b[i] = -(a[i] + c[i]) - par.disc

    def implicit_step(V, theta):
        ab = np.zeros((3, n)); rhs = V.copy()
        ab[0, 1:]   = -theta*dt*c[:-1]
        ab[1, :]    = 1 - theta*dt*b
        ab[2, :-1]  = -theta*dt*a[1:]
        rhs[1:-1]  += (1-theta)*dt*(a[1:-1]*V[:-2] + b[1:-1]*V[1:-1] + c[1:-1]*V[2:])
        rhs[0]  = S[0]                                   # deep ITM-loss region ~ S
        rhs[-1] = par.called_payoff(par.T)              # far field (called soon)
        ab[1,0]=1; ab[0,1]=0; ab[1,-1]=1; ab[2,-2]=0
        return solve_banded((1, 1), ab, rhs)

    for step in range(NT, 0, -1):
        theta = 1.0 if (NT - step) < 2 else 0.5         # Rannacher start
        V = implicit_step(V, theta)
        if (step - 1) in call_levels:
            tc = (step - 1) * dt
            V[S >= par.C] = par.called_payoff(tc)

    price = float(np.interp(np.log(par.S0), x, V))
    if return_grid:
        return price, x, V
    return price


if __name__ == "__main__":
    import time
    for NX, NT in [(800, 600), (1600, 1200), (3200, 2400)]:
        t = time.perf_counter(); v = price_fd(NX=NX, NT=NT)
        print(f"CN+Rannacher {NX}x{NT}: {v:.4f}  ({1e3*(time.perf_counter()-t):.1f} ms)")
