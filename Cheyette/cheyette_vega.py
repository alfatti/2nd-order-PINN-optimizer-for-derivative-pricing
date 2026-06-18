# -*- coding: utf-8 -*-
"""
ATM Normal (Bachelier) Vega for the Cheyette caplet, and the price->vol error
conversion used as the benchmark accuracy metric.

A caplet is a call on the LIBOR rate R(TC,TB) with payoff
    Delta * (R - K)^+  paid at TB,
so its value is  Delta * B(0,TB) * E_TB[(R-K)^+].
In the Bachelier (normal-vol) convention the forward rate F is normal with
total stdev  s = sigma_N * sqrt(TC):
    C = Delta * B(0,TB) * [ (F-K) Phi(d) + s phi(d) ],   d = (F-K)/s.

Normal Vega:
    dC/dsigma_N = Delta * B(0,TB) * sqrt(TC) * phi(d).
ATM (F=K): d=0, phi(0)=1/sqrt(2pi), giving the maximal Vega
    Vega_ATM = Delta * B(0,TB) * sqrt(TC) / sqrt(2pi).

The reported error metric is:
    vol_error  =  |C_PINN - C_analytic| / Vega_ATM      [in absolute vol]
    vol_error_bps = vol_error * 1e4                      [in bps of vol]

Target threshold: vol_error_bps <= 0.1 bps.
"""

import numpy as np
from cheyette_analytic import caplet_price, bond_price_0, PARAMS, F0, v_squared

SQRT_2PI = np.sqrt(2.0 * np.pi)


def forward_libor(TC, TB, p=PARAMS, f0=F0):
    """Forward LIBOR R(0,TC,TB) from the initial curve: (B(0,TC)/B(0,TB)-1)/Delta."""
    Delta = TB - TC
    B_TC = bond_price_0(TC, p, f0)
    B_TB = bond_price_0(TB, p, f0)
    return (B_TC / B_TB - 1.0) / Delta


def atm_normal_vega(TC, TB, p=PARAMS, f0=F0):
    """ATM Normal Vega = Delta * B(0,TB) * sqrt(TC) / sqrt(2 pi)."""
    Delta = TB - TC
    B_TB = bond_price_0(TB, p, f0)
    return Delta * B_TB * np.sqrt(TC) / SQRT_2PI


def implied_normal_vol(TC, TB, K, price, p=PARAMS, f0=F0):
    """Invert the Bachelier caplet formula for sigma_N (for context/reporting)."""
    from scipy.optimize import brentq
    Delta = TB - TC
    B_TB = bond_price_0(TB, p, f0)
    F = forward_libor(TC, TB, p, f0)
    def bachelier(sigmaN):
        s = sigmaN * np.sqrt(TC)
        d = (F - K) / s
        from scipy.stats import norm
        return Delta * B_TB * ((F - K) * norm.cdf(d) + s * norm.pdf(d))
    target = price
    return brentq(lambda sN: bachelier(sN) - target, 1e-8, 1.0)


def price_err_to_vol_bps(price_err, TC, TB, p=PARAMS, f0=F0):
    """Convert an absolute price error to bps of normal vol via ATM Vega."""
    vega = atm_normal_vega(TC, TB, p, f0)
    return (price_err / vega) * 1e4


if __name__ == "__main__":
    TC, TB, K = 1.0, 2.0, 0.05
    C = caplet_price(TC, TB, K)
    F = forward_libor(TC, TB)
    vega = atm_normal_vega(TC, TB)
    sigmaN = implied_normal_vol(TC, TB, K, C)

    print(f"Primary caplet: TC={TC}, TB={TB}, K={K:.3%}, f0={F0:.3%}")
    print(f"  forward LIBOR F        = {F:.6%}")
    print(f"  analytic price C       = {C:.8f}")
    print(f"  implied normal vol     = {sigmaN*1e4:.4f} bps  ({sigmaN:.6e})")
    print(f"  ATM Normal Vega        = {vega:.8f}  (price per unit vol)")
    print()
    # What price accuracy must the PINN hit for 0.1 bps of vol?
    target_bps = 0.1
    max_price_err = vega * (target_bps * 1e-4)
    print(f"  >>> 0.1 bps vol  <=>  price error <= {max_price_err:.3e}")
    print(f"  >>> as fraction of price            = {max_price_err / C:.3e}")
    print(f"  >>> i.e. relative price error       <= {max_price_err / C * 100:.4f}%")
    print()
    # Sanity: reproduce the paper's "error in implied vol" for the QMC row.
    # QMC TC=1,TB=2 price error was 1.9722e-06 -> paper says 0.0110% (=1.10 bps).
    qmc_err = 1.9722e-06
    print(f"  cross-check: QMC price err {qmc_err:.3e} -> "
          f"{price_err_to_vol_bps(qmc_err, TC, TB):.4f} bps "
          f"(paper reports ~1.10 bps in BS-implied vol)")
