"""
vol_normalization.py
====================
Converts a price error into an error measured in basis points of volatility,
using a SINGLE fixed reference Vega applied identically to every contract --
the same methodology as the Cheyette caplet benchmark (one fixed ATM anchor,
not per-contract Vega).

For this EQUITY product the natural vol is the lognormal sigma, so "bp of vol"
means bp of sigma (1 bp == 1e-4 in sigma). The fixed anchor is the Black-Scholes
ATM vega of a reference vanilla at the product's spot and maturity:
    VEGA_REF = S0 * sqrt(T) * phi(d1_atm)   ~ 39.9   (price per unit sigma)

Rationale (carried over deliberately): normalizing each contract by its own
Vega would hand high-Vega contracts an artificially looser price tolerance for
the same vol tolerance. A single VEGA_REF keeps the bp target a fixed scalar
across the whole grid, so cross-contract comparisons are fair.

    bp_of_vol(price_err) = price_err / VEGA_REF / 1e-4
    BUDGET_01BPS         = VEGA_REF * 1e-5      # price error == 0.1 bp of vol
"""
import numpy as np
from scipy.stats import norm
from config import PAR


def bs_atm_vega(S0, T, sigma, r=0.0, q=0.0):
    """Black-Scholes ATM (K=S0 forward) lognormal vega: dPrice/dsigma."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S0 / S0) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * np.sqrt(T) * norm.pdf(d1)


def compute_vega_ref(par=PAR):
    """
    Fixed reference lognormal Vega for the autocallable example: the ATM
    BS vega of a vanilla at the product's spot/maturity. Single scalar reused
    everywhere (analogue of the TC=1,TB=2 ATM caplet anchor in Cheyette).
    """
    return float(bs_atm_vega(par.S0, par.T, par.sig, par.r, par.q))


VEGA_REF = compute_vega_ref()
BUDGET_01BPS = VEGA_REF * 1e-5              # price tolerance == 0.1 bp of vol


def price_err_to_bp(price_err):
    """Absolute price error -> basis points of (Normal) vol."""
    return np.abs(price_err) / VEGA_REF / 1e-4


def bp_to_price_err(bp):
    return bp * 1e-4 * VEGA_REF


if __name__ == "__main__":
    print(f"VEGA_REF        = {VEGA_REF:.6f}  (price per unit Normal vol)")
    print(f"BUDGET_01BPS    = {BUDGET_01BPS:.6e}  (price err == 0.1 bp of vol)")
    print(f"1 bp of vol     = {bp_to_price_err(1.0):.6e} in price")
    # sanity: the paper's 0.88 gap (98.39 vs 97.51) expressed in bp of vol
    print(f"paper gap 0.88  = {price_err_to_bp(0.88):.1f} bp of vol")
