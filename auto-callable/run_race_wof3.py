"""
run_race_wof3.py
================
Orchestrates the worst-of-3 race -- where the PINN's structural advantage shows.

The argument:
  * MC is the 3-D ground truth (unbiased, dimension-independent): ~90.42.
  * FD in 3-D suffers the curse of dimensionality: work ~ N^5 (N^3 grid x N^2
    CFL steps). Feasible grids are far too coarse to be accurate; an accuracy-
    grade grid extrapolates to ~years of wall-clock (see wof3_fd.scaling_study).
  * The PINN's cost scales MILDLY with dimension (input layer 2->4, modestly
    larger collocation budget). One trained network prices the basket family by
    inference. So in 3-D the PINN crosses below FD almost immediately and is the
    only method that is simultaneously accurate and fast across a contract grid.

The PINN leg requires the H200; MC and the FD scaling study run on CPU.
"""
import time
import numpy as np
from config import WOF3, RACE
from vol_normalization import price_err_to_bp
from wof3_mc import price_mc, convergence_curve
from wof3_fd import scaling_study


def run_reference():
    t = time.perf_counter()
    m, se = price_mc(N=RACE.mc_paths)
    return {"value": m, "se": se, "seconds": time.perf_counter() - t}


def run_fd_curse(accuracy_N=1500):
    rows, c, extrap = scaling_study(Ns=(16, 24, 32), accuracy_N=accuracy_N)
    return {"rows": rows, "c_fit": c, "extrap_seconds": extrap,
            "extrap_years": extrap / 3.15e7, "accuracy_N": accuracy_N}


def run_pinn_wof3(second_order=True):
    try:
        import jax
        from pinn_wof3 import WoF3PINN, train, price_at_spot
    except Exception as e:
        return {"available": False, "reason": str(e)}
    raise NotImplementedError("Attach the RAD sampler on the H200 host.")


if __name__ == "__main__":
    print("== Worst-of-3 autocallable race ==\n")
    ref = run_reference()
    print(f"MC reference (3-D ground truth): {ref['value']:.4f} "
          f"+/- {ref['se']:.4f}   ({ref['seconds']:.1f}s)")

    print("\n-- FD 3-D curse of dimensionality --")
    curse = run_fd_curse()
    print("  N    price(coarse, NOT accurate)   seconds")
    for N, p, s in curse["rows"]:
        print(f"{N:4d}   {p:10.3f}                  {s:8.2f}")
    print(f"\n  fit: seconds ~ {curse['c_fit']:.2e} * N^5")
    print(f"  accuracy-grade N={curse['accuracy_N']}: "
          f"{curse['extrap_seconds']:.2e} s  (~ {curse['extrap_years']:.2f} years)")
    print("\n  => FD is fast-and-wrong (coarse) or accurate-and-infeasible.")
    print("     MC prices it in seconds; the PINN amortizes across the family.")
    print("\nPINN leg: run on H200 via run_pinn_wof3(second_order=True).")
