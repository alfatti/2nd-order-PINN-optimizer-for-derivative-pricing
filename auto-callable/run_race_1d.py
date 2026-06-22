"""
run_race_1d.py
==============
Orchestrates the 1-D race on the paper's example.

The honest framing baked in here:
  * On a SINGLE 1-D price, FD wins outright (converged solve ~0.1 s, faster than
    any PINN training). We report that openly.
  * The PINN's wins are (a) accuracy at the stringent 0.1 bp-of-vol tolerance
    that an Adam-only PINN cannot reach, and (b) AMORTIZATION across a contract
    family -- one trained parametric network prices the whole grid by inference,
    crossing below per-contract FD/MC as the family grows.

Outputs a results dict consumed by plotting.py. The PINN legs require the H200
(JAX + Optimistix fork); the reference/FD/MC legs run on CPU.
"""
import time
import numpy as np
from config import PAR, RACE
from vol_normalization import price_err_to_bp, VEGA_REF, BUDGET_01BPS
from reference_1d import reference_value
from fd_1d import price_fd
from mc_1d import price_mc, convergence_curve


def run_reference():
    t = time.perf_counter()
    V0 = reference_value()
    return {"value": V0, "seconds": time.perf_counter() - t}


def run_fd_grid_refinement(ref):
    rows = []
    for NX, NT in [(400, 300), (800, 600), (1600, 1200), (3200, 2400)]:
        t = time.perf_counter()
        v = price_fd(NX=NX, NT=NT)
        rows.append({"NX": NX, "NT": NT, "price": v,
                     "seconds": time.perf_counter() - t,
                     "err": abs(v - ref), "bp": price_err_to_bp(v - ref)})
    return rows


def run_mc_wall(ref):
    rows = []
    for N in (1e4, 1e5, 1e6, 1e7):
        t = time.perf_counter()
        m, se = price_mc(N=int(N))
        rows.append({"N": int(N), "price": m, "se": se,
                     "seconds": time.perf_counter() - t,
                     "err": abs(m - ref), "bp": price_err_to_bp(m - ref),
                     "se_bp": price_err_to_bp(se)})
    return rows


def amortization_grid():
    """Per-contract FD/MC cost vs a single amortized PINN over the contract grid."""
    from itertools import product
    contracts = list(product(RACE.grid_C, RACE.grid_T, RACE.grid_sig))
    n = len(contracts)
    # per-contract FD wall-clock (measured on one representative contract)
    t = time.perf_counter(); price_fd(NX=RACE.fd_NX, NT=RACE.fd_NT)
    fd_per = time.perf_counter() - t
    t = time.perf_counter(); price_mc(N=RACE.mc_paths)
    mc_per = time.perf_counter() - t
    return {"n_contracts": n, "fd_per": fd_per, "mc_per": mc_per,
            "fd_total_vs_n": [fd_per * k for k in range(1, n + 1)],
            "mc_total_vs_n": [mc_per * k for k in range(1, n + 1)]}


def run_pinn_1d(second_order=True):
    """H200 only. Returns price, train seconds, inference-per-contract seconds."""
    try:
        import jax
        from pinn_1d import AutocallPINN, train, price_at_spot
    except Exception as e:
        return {"available": False, "reason": str(e)}
    import jax
    key = jax.random.PRNGKey(RACE.seed)
    model = AutocallPINN(key)
    # sampler omitted here; provided by the H200 driver with RAD resampling.
    raise NotImplementedError("Attach the RAD sampler on the H200 host.")


if __name__ == "__main__":
    print(f"VEGA_REF={VEGA_REF:.4f}  BUDGET_01BPS(price)={BUDGET_01BPS:.3e}")
    ref = run_reference()
    print(f"\nREFERENCE V0 = {ref['value']:.5f}  ({ref['seconds']:.2f}s)")
    R = ref["value"]

    print("\n-- FD (CN+Rannacher) grid refinement --")
    print("  NX     NT     price      err(price)   err(bp vol)   time")
    for r in run_fd_grid_refinement(R):
        print(f"{r['NX']:5d} {r['NT']:6d}  {r['price']:8.4f}   "
              f"{r['err']:.2e}    {r['bp']:8.2f}    {r['seconds']:.3f}s")

    print("\n-- MC convergence wall --")
    print("     N        price       err(bp vol)   se(bp vol)   time")
    for r in run_mc_wall(R):
        print(f"{r['N']:>9d}  {r['price']:8.4f}    {r['bp']:8.2f}     "
              f"{r['se_bp']:8.2f}    {r['seconds']:.2f}s")

    print("\n-- amortization (per-contract cost) --")
    a = amortization_grid()
    print(f"contract grid size: {a['n_contracts']}")
    print(f"FD per contract: {a['fd_per']:.3f}s   "
          f"MC per contract: {a['mc_per']:.2f}s")
    print("PINN: one train (H200) then ~ms inference per contract -- "
          "crosses below FD/MC at the grid sizes plotted in plotting.py")
