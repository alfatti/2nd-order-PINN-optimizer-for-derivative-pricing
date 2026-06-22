"""
wof3_fd.py
==========
3-D finite-difference pricer for the worst-of-3 autocallable, included to EXHIBIT
the curse of dimensionality, not to win. State (x1,x2,x3)=log-ratios, forward
time tau. The PDE carries three second derivatives plus three cross terms:

  V_tau = sum_k 0.5 sig_k^2 V_{kk}
        + sum_{k<l} rho sig_k sig_l V_{kl}
        + sum_k (r - q - 0.5 sig_k^2) V_k
        - disc V.

Grid cost scales as N^3 in memory and N^3 * NT in work (explicit, with the
CFL tie NT ~ N^2). So total work ~ N^5. Reaching the per-axis resolution that
the 1-D solve needed for bp accuracy (N ~ 1500) is utterly infeasible in 3-D --
that is the entire point. We run small grids to MEASURE the scaling constant and
extrapolate the wall-clock for an accuracy-grade grid.
"""
import time
import numpy as np
from config import WOF3


def price_fd_3d(par=WOF3, N=40, NT=None, xmin=-1.6, xmax=0.05):
    """Explicit 3-D solve on N^3 grid. Returns (price_at_spot, seconds)."""
    vols = np.array(par.vols); d = 3
    if NT is None:
        # CFL: explicit diffusion stability dt <= dx^2 / (sum sig^2); add margin
        dx = (xmax - xmin) / (N - 1)
        dt_max = dx**2 / (np.sum(vols**2) + 1e-9) * 0.4
        NT = int(np.ceil(par.T / dt_max))
        NT = int(np.ceil(NT / par.n_calls) * par.n_calls)
    dt = par.T / NT
    x = np.linspace(xmin, xmax, N)
    dx = x[1] - x[0]
    call_levels = set(int(round(tc / dt)) for tc in par.call_times)
    c_level = np.log(par.C / par.S0)
    L_level = np.log(par.L / par.S0)

    X1, X2, X3 = np.meshgrid(x, x, x, indexing="ij")
    Wlog = np.minimum(np.minimum(X1, X2), X3)            # log worst-ratio
    # terminal (maturity also a call date)
    f = np.where(Wlog > L_level, par.H, par.H * np.exp(Wlog))
    V = np.where(Wlog >= c_level, par.H * np.exp(par.B * par.T), f).astype(np.float32)

    mu = (par.r - par.q - 0.5 * vols**2)
    t0 = time.perf_counter()
    for step in range(NT, 0, -1):
        Vn = V.copy()
        # second derivatives
        for k, ax in enumerate((0, 1, 2)):
            Vp = np.roll(V, -1, axis=ax); Vm = np.roll(V, 1, axis=ax)
            Vxx = (Vp - 2*V + Vm) / dx**2
            Vx  = (Vp - Vm) / (2*dx)
            Vn += dt * (0.5 * vols[k]**2 * Vxx + mu[k] * Vx)
        # cross derivatives
        for (k, l) in ((0,1),(0,2),(1,2)):
            Vpp = np.roll(np.roll(V, -1, k), -1, l)
            Vpm = np.roll(np.roll(V, -1, k),  1, l)
            Vmp = np.roll(np.roll(V,  1, k), -1, l)
            Vmm = np.roll(np.roll(V,  1, k),  1, l)
            Vkl = (Vpp - Vpm - Vmp + Vmm) / (4 * dx**2)
            Vn += dt * (par.rho * vols[k] * vols[l] * Vkl)
        Vn += dt * (-par.disc * V)
        V = Vn
        if (step - 1) in call_levels:
            tc = (step - 1) * dt
            V[Wlog >= c_level] = par.H * np.exp(par.B * tc)
    secs = time.perf_counter() - t0
    # spot is x=0 in all dims
    j0 = np.argmin(np.abs(x - 0.0))
    return float(V[j0, j0, j0]), secs


def scaling_study(par=WOF3, Ns=(16, 24, 32), accuracy_N=1500):
    """Measure (N, NT, seconds) and extrapolate cost to an accuracy-grade grid."""
    rows = []
    for N in Ns:
        price, secs = price_fd_3d(par, N=N)
        rows.append((N, price, secs))
    # fit secs ~ c * N^5 (N^3 grid * N^2 timesteps from CFL)
    Narr = np.array([r[0] for r in rows], float)
    sarr = np.array([r[2] for r in rows], float)
    c = np.median(sarr / Narr**5)
    extrap = c * accuracy_N**5
    return rows, c, extrap


if __name__ == "__main__":
    rows, c, extrap = scaling_study(Ns=(16, 24, 32))
    print("N     price      seconds")
    for N, p, s in rows:
        print(f"{N:3d}   {p:8.4f}   {s:8.2f}")
    print(f"\nfit: seconds ~ {c:.2e} * N^5")
    print(f"extrapolated wall-clock at accuracy-grade N=1500: "
          f"{extrap:.3e} s  (~ {extrap/3.15e7:.2e} years)")
