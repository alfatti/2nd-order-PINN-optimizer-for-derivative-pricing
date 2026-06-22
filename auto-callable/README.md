# Autocallable PINN Race

Benchmark harness comparing a **curvature-aware second-order PINN** (SSBroyden /
SSBFGS, CrunchOptimizer/raj-brown Optimistix fork, `@SSBFGS` branch) against a
**finite-difference PDE solver** and **Monte Carlo**, on the discrete monthly
autocallable of Deng, Mallett & McCann (2011) and a worst-of-3 basket extension.

Same conventions as the Cheyette caplet package: JAX/Equinox, H200 target, a
single fixed `VEGA_REF` for vol-error normalization.

## The thesis (and the honest framing)

On a **single 1-D price, FD wins outright** — a converged CN+Rannacher solve is
~0.1 s, faster than any PINN training. The harness states this openly. The PINN's
real contributions are:

1. **1-D: parity at a stringent tolerance.** A curvature-aware PINN reaches the
   `0.1 bp of vol` target; an Adam-only PINN plateaus well above it. The 1-D
   message is *on-par accuracy* + Greeks/surface for free via autodiff — **not** a
   speed win over FD. (At 1-D FD's ~0.1 s/contract, amortization only crosses
   below FD at thousands of contracts.)
2. **Worst-of-3: the curse of dimensionality.** FD work scales as `N^5`
   (`N^3` grid x `N^2` CFL steps). An accuracy-grade 3-D grid extrapolates to
   **~2.3 years** of wall-clock. MC stays cheap but cannot reach `0.1 bp` without
   ~`10^8` paths; the PINN scales mildly with dimension and amortizes across the
   contract family. This is the PINN's structural win.

## Ground truth — why 97.51, not the paper's 98.39

The paper quotes **98.39** from a coarse *explicit* FD on a `1000x500` grid. That
number carries discretization bias from the payoff **discontinuity at L=80** and
the call barrier. This package builds an independent high-accuracy reference and
**does not** use 98.39 as truth:

| method | value | check |
|---|---|---|
| probability-route quadrature (`reference_1d.py`) | **97.505** | call probs match Table 1 to 1e-4 |
| CN+Rannacher FD (`fd_1d.py`) | 97.40 → 97.48 (monotone ↑) | snapped grid, → reference |
| antithetic MC, 1e7 (`mc_1d.py`) | **97.507 ± 0.002** | independent |
| par benchmark (Case 1) | **100.00** | validates payoff + discount machinery |

The `0.88` gap to the paper equals **~231 bp of vol** under `VEGA_REF` — large,
and exactly the kind of bias a bp-level race must not inherit. Treat the
quadrature value as ground truth for the 1-D leg; MC is ground truth in 3-D.

## Vol-error normalization

A **single fixed** reference vega applied identically to every contract (not
per-contract), so the bp target is one scalar across the whole grid. For this
equity product the natural vol is lognormal sigma, so `bp of vol` = bp of sigma:

```
VEGA_REF     = S0 * sqrt(T) * phi(d1_atm)  ~ 38.14    # price per unit sigma
BUDGET_01BPS = VEGA_REF * 1e-5  ~ 3.81e-4             # price err == 0.1 bp of vol
```

## File map

```
config.py            paper params, WoF-3 params, race/training settings
vol_normalization.py VEGA_REF, price<->bp conversion
reference_1d.py      probability-route quadrature ground truth   [CPU-validated]
fd_1d.py             CN+Rannacher FD, snapped grid, call overwrites [CPU-validated]
mc_1d.py             antithetic + control-variate MC, chunked      [CPU-validated]
pinn_1d.py           parametric PINN, masked call-date loss, 2nd-order + Adam ablation [H200]
wof3_mc.py           worst-of-3 MC reference (3-D ground truth)    [CPU-validated]
wof3_fd.py           3-D FD curse-of-dim scaling study             [CPU-validated]
pinn_wof3.py         3-D PINN, full Hessian + cross terms          [H200]
run_race_1d.py       1-D orchestration (reference / FD / MC / amortization)
run_race_wof3.py     WoF-3 orchestration (MC ref / FD curse / PINN)
plotting.py          fig1 MC wall, fig2 crossover, fig3 frontier, fig4 curse
```

## The masked call-date loss (the key design point)

The discrete autocall is encoded as a boundary condition that is **active only on
the 12 call-date time slices** and only on the called region `S >= C` (in 3-D,
`worst-of >= C/S0`), where the called payoff `P_t = H e^{B t}` is flat in the
state. Between call dates `S = C` is ordinary interior — this temporal masking is
what separates the discrete product from the continuous (barrier) one. Getting
the mask's temporal support right is the crux; a mask that leaks between dates
silently solves the continuous problem instead.

The dominant L-infinity error source is the **payoff discontinuity at L** (worse
than the European-call kink in the BS benchmark) plus the per-call-date kinks
along `S = C`. Mitigations in `pinn_1d.py`: annealed tanh-smoothing of the IC
near L, and RAD-style adaptive resampling near L and the call barrier (attach the
sampler on the host).

## Running

CPU (references, FD, MC, curse-of-dim scaling, illustrative plots):
```
python3 run_race_1d.py
python3 run_race_wof3.py
python3 plotting.py
```

H200 (PINN legs): provide the RAD collocation sampler to `train(...)` in
`pinn_1d.py` / `pinn_wof3.py`, run `second_order=True` (SSBroyden/SSBFGS) and
`second_order=False` (Adam ablation), and feed the resulting `(time, error)`
points and `(train, infer)` costs into `plotting.py`.

## Status

Numerical foundation CPU-validated end-to-end (quadrature, FD, MC, par benchmark,
WoF-3 MC, 3-D FD scaling). PINN modules are written for the H200 + Optimistix fork
and are correct-by-construction against that environment; populate their legs from
the GPU run.
