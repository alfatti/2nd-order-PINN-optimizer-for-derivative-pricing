# -*- coding: utf-8 -*-
"""
Grid of caplets in the 3-Factor Exponential Cheyette model.
One calibrated market (Table 2 params), many contracts.

Defines:
  * the contract grid (maturities x strikes),
  * a vectorized analytical price oracle over the grid,
  * the per-contract ATM Normal Vega and 0.1-bps price budgets.

The grid mirrors the paper's Section 8.2 test set: option lifetimes
T_C = 1..5 (with T_B = T_C + 1) and strikes {3% ITM, 5% ATM, 7% OTM}.
"""

import numpy as np
from cheyette_analytic import caplet_price, bond_price_0, PARAMS, F0
from cheyette_vega import atm_normal_vega, forward_libor

# ----------------------------- grid definition --------------------------------
MATURITIES = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0)]
STRIKES = [0.03, 0.05, 0.07]   # ITM / ATM / OTM
DELTA = 1.0                    # T_B - T_C, constant across the grid

# ---------------------- SINGLE reference ATM Normal Vega ----------------------
# The vol-error metric across the ENTIRE grid is normalized by ONE fixed scalar:
# the ATM Normal Vega of the reference contract (T_C=1, T_B=2 ATM caplet).
# This is NOT a per-contract / per-maturity Vega. Using one constant makes the
# 0.1-bps bar uniform across the grid and keeps the grid numbers directly
# comparable to the single-contract benchmark.
REF_TC, REF_TB = 1.0, 2.0
VEGA_REF = float(atm_normal_vega(REF_TC, REF_TB))   # ~= 0.36097790
BUDGET_01BPS = VEGA_REF * 0.1e-4                     # fixed price budget for 0.1 bps


def build_grid():
    """Return a list of contract dicts with analytic price (Vega is the single
    global VEGA_REF, identical for every contract)."""
    rows = []
    for (TC, TB) in MATURITIES:
        F = float(forward_libor(TC, TB))
        for K in STRIKES:
            price = float(caplet_price(TC, TB, K))
            rows.append(dict(
                TC=TC, TB=TB, K=K, Delta=TB - TC,
                price=price, fwd=F,
                vega_ref=VEGA_REF,                  # same scalar for all rows
                budget_01bps=BUDGET_01BPS,          # same budget for all rows
            ))
    return rows


def grid_arrays():
    """Return (contracts array, prices array) for vectorized use.
    contracts: (N,3) columns = [TC, TB, K].  prices: (N,)."""
    rows = build_grid()
    contracts = np.array([[r["TC"], r["TB"], r["K"]] for r in rows], dtype=np.float64)
    prices = np.array([r["price"] for r in rows], dtype=np.float64)
    return contracts, prices


def vol_error_bps(price_pred, price_analytic):
    """Vectorized Normal-Vega vol error in bps, normalized by the SINGLE
    reference ATM Vega (VEGA_REF) for every contract."""
    return np.abs(np.asarray(price_pred) - np.asarray(price_analytic)) / VEGA_REF * 1e4


if __name__ == "__main__":
    rows = build_grid()
    print(f"Grid: {len(MATURITIES)} maturities x {len(STRIKES)} strikes "
          f"= {len(rows)} contracts")
    print(f"SINGLE reference ATM Normal Vega (TC={REF_TC},TB={REF_TB}) = "
          f"{VEGA_REF:.8f}")
    print(f"Fixed price budget for 0.1 bps (all contracts) = {BUDGET_01BPS:.3e}\n")
    print(f"{'TC':>4}{'TB':>4}{'K':>6}{'money':>7}{'price':>12}"
          f"{'rel budget':>12}")
    for r in rows:
        moneyness = ("ITM" if r["K"] < r["fwd"] else
                     "ATM" if abs(r["K"] - r["fwd"]) < 0.005 else "OTM")
        print(f"{r['TC']:>4.1f}{r['TB']:>4.1f}{r['K']:>6.0%}{moneyness:>7}"
              f"{r['price']:>12.6f}{r['budget_01bps']/r['price']:>12.2%}")
    print(f"\nThe 0.1 bps bar is a single uniform price tolerance of "
          f"{BUDGET_01BPS:.3e} for EVERY contract (one fixed Vega), not a "
          f"per-contract Vega. The relative-budget column just shows that the "
          f"same absolute tolerance is tighter in relative terms for cheap OTM "
          f"contracts.")
