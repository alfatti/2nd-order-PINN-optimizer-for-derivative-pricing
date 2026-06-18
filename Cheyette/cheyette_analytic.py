# -*- coding: utf-8 -*-
"""
Analytical caplet price in the 3-Factor Exponential Cheyette model.
Beyna, Chiarella & Kang (2012), Section 4.2 / 4.2.1, Eq. (20).

Caplet price (Black-type formula on the forward bond price):
    C(t,TC,TB) = B(t,TC) N(d1) - B(t,TB)(1+K*Delta) N(d2)
with
    d1 = [ ln( B(t,TC) / (B(t,TB)(1+K*Delta)) ) + 0.5 v^2 ] / v
    d2 = d1 - v
    v^2(t,TC,TB) = sum_k  int_t^{TC} ( b_k(u,TB) - b_k(u,TC) )^2 du

Bond price volatilities (Section 4.2.1):
    b1(t,T) = (1/l1)[ a0_1 + a1_1 t - c l1 t
                      - exp(l1 (t-T))(a0_1 + a1_1 t) + c l1 T ]
    b2(t,T) = ((a0_2 + a1_2 t)/l2)[ exp(l2 (t-T)) - 1 ]
    b3(t,T) = ((a0_3 + a1_3 t)/l3)[ exp(l3 (t-T)) - 1 ]

Bonds use the analytical formula (Eq. 13) with a constant initial forward
f(0,T)=f0, so B(0,T)/B(0,t) = exp(-f0 (T-t)).  We verify against the paper's
Table 5 benchmark: TC=1, TB=2, K=5%, f0=5% -> C = 0.004183.

v^2 is computed by high-accuracy numerical quadrature of the (short, clean)
b_k expressions rather than transcribing the very long closed form in
Appendix A.2 -- the two are equal, and quadrature avoids transcription risk.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm

# ------------------------- calibrated parameters (Table 2) --------------------
# factor:        constant c     a1            a0           lambda
PARAMS = dict(
    c   = 0.0097,
    a1_1=-0.0005,  a0_1=-0.000165, lam1=-0.004,
    a1_2= 0.000021, a0_2=-0.000742, lam2=-0.430,
    a1_3= 0.0000193, a0_3=0.000701, lam3=-0.510,
)
# NOTE: lambda^(k) are reported negative in Table 2; the volatility uses
# exp(-lambda (T-t)) and the b_k formulas use exp(lambda (t-T)) consistently,
# so we plug Table 2 values in verbatim.

F0 = 0.05  # constant initial forward rate (ATM setup)


# ------------------------------ bond volatilities -----------------------------
def b1(u, T, p):
    l1 = p["lam1"]
    return (1.0 / l1) * (
        p["a0_1"] + p["a1_1"] * u - p["c"] * l1 * u
        - np.exp(l1 * (u - T)) * (p["a0_1"] + p["a1_1"] * u)
        + p["c"] * l1 * T
    )

def b2(u, T, p):
    l2 = p["lam2"]
    return ((p["a0_2"] + p["a1_2"] * u) / l2) * (np.exp(l2 * (u - T)) - 1.0)

def b3(u, T, p):
    l3 = p["lam3"]
    return ((p["a0_3"] + p["a1_3"] * u) / l3) * (np.exp(l3 * (u - T)) - 1.0)


# ------------------------------- v^2 by quadrature ----------------------------
def v_squared(t, TC, TB, p, n=200):
    """v^2(t,TC,TB) = sum_k int_t^{TC} (b_k(u,TB)-b_k(u,TC))^2 du via Gauss-Legendre."""
    nodes, weights = leggauss(n)
    # map [-1,1] -> [t,TC]
    um = 0.5 * (TC - t) * nodes + 0.5 * (TC + t)
    jac = 0.5 * (TC - t)
    total = 0.0
    for bk in (b1, b2, b3):
        integrand = (bk(um, TB, p) - bk(um, TC, p)) ** 2
        total += jac * np.sum(weights * integrand)
    return total


# ------------------------------- bond prices ----------------------------------
def bond_price_0(T, p, f0=F0):
    """B(0,T) with constant initial forward f0."""
    return np.exp(-f0 * T)


# ------------------------------ analytical caplet -----------------------------
def caplet_price(TC, TB, K, p=PARAMS, f0=F0, t=0.0):
    """Analytical caplet price C(t,TC,TB), Eq. (20), at t=0 by default."""
    Delta = TB - TC
    # Bonds at t=0 (state x=0): B(0,T) directly from the initial curve.
    B_tTC = bond_price_0(TC, p, f0)
    B_tTB = bond_price_0(TB, p, f0)
    v2 = v_squared(t, TC, TB, p)
    v = np.sqrt(v2)
    moneyness = B_tTC / (B_tTB * (1.0 + K * Delta))
    d1 = (np.log(moneyness) + 0.5 * v2) / v
    d2 = d1 - v
    return B_tTC * norm.cdf(d1) - B_tTB * (1.0 + K * Delta) * norm.cdf(d2)


if __name__ == "__main__":
    # Reproduce Table 5 (strike 5%, f0=5%): expected analytical prices.
    expected = {
        (1.0, 2.0): 0.004183,
        (2.0, 3.0): 0.005318,
        (3.0, 4.0): 0.006077,
        (4.0, 5.0): 0.006792,
        (5.0, 6.0): 0.007788,
    }
    K = 0.05
    print(f"{'TC':>4}{'TB':>4}{'analytic(ours)':>18}{'paper':>12}{'abs diff':>12}")
    for (TC, TB), ref in expected.items():
        c = caplet_price(TC, TB, K)
        print(f"{TC:>4.1f}{TB:>4.1f}{c:>18.6f}{ref:>12.6f}{abs(c-ref):>12.2e}")
    # detail for the primary contract
    v2 = v_squared(0.0, 1.0, 2.0, PARAMS)
    print(f"\nPrimary contract TC=1,TB=2: v^2={v2:.6e}, v={np.sqrt(v2):.6e}")
