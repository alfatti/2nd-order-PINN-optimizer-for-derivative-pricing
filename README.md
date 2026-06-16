# Black–Scholes PINN Benchmark (Curvature-Aware Optimizers)

Pricing a European call option under the Black–Scholes model with a
Physics-Informed Neural Network, used as a benchmark for the curvature-aware
optimizers from **[CrunchOptimizer / Curvature-Aware Optimization for
High-Accuracy PINNs](https://github.com/CrunchOptimizer)** (Jnini et al.).

The goal is not to price an option faster than the closed-form formula — it is
to measure *how accurately* a PINN can recover a known analytic solution when
trained with their second-order optimizers (**SSBroyden**, **SSBFGS**) versus a
first-order **Adam** baseline.

-----

## What this solves

A European call price `V(t, S)` satisfies the Black–Scholes PDE. We work in
forward time `τ = T − t`, which turns the terminal payoff into an initial
condition:

```
PDE:        V_τ = ½ σ² S² V_SS + r S V_S − r V,      (τ, S) ∈ (0, T] × (0, S_max)
IC (τ=0):   V = max(S − K, 0)                         (terminal payoff)
BC (S=0):   V = 0
BC (S=S_max): V = S_max − K e^{−r τ}
```

Ground truth is the closed-form Black–Scholes call price, so the error is
measured against an exact solution rather than a numerical reference.

Default contract: `K=1.0`, `r=0.05`, `σ=0.20`, `T=1.0`, spot domain `S ∈ [0, 3K]`.

-----

## Method

The implementation deliberately mirrors the artifacts in the upstream repo
(the Inviscid Burgers / Stokes solvers):

- **Network** — Equinox `tanh` MLP with a linear output, inputs normalized to
  `[−1, 1]`. Default `(2, 64, 64, 64, 64, 64, 1)`, float64 throughout.
- **Loss** — residual-based least squares: mean-squared PDE residual plus
  initial- and boundary-condition residuals, in the `½‖r(θ)‖²` form the paper
  uses for its Gauss–Newton / natural-gradient derivation.
- **Second derivative** `V_SS` — computed by a forward-over-reverse pass
  (`jvp` of `grad`) along the spot direction, avoiding a full Hessian. This is
  the cheap primitive that parallelizes well on GPU.
- **Optimizers** — the *actual* optimizers from their forked Optimistix, not
  reimplementations:
  - `SSBroydenZoom` / `SSBFGSZoom` subclass `optx.AbstractSSBroyden` /
    `optx.AbstractSSBFGS` with a `Zoom` line search and `NewtonDescent`,
    following Code Listing 1 of the paper.
  - Each is wrapped in `optx.BestSoFarMinimiser` so the best iterate is never
    discarded, and run with `throw=False` plus a finite-iterate guard so a late
    line-search failure cannot corrupt the result.

**Training recipe:** Adam warm-up → second-order refinement. Adam alone reaches
a moderate error; the curvature-aware stage is what drives the solution to high
accuracy, which is the central claim of the paper.

-----

## Files

|File                        |Description                                                                                                              |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------|
|`bs_single_contract_h200.py`|H200 / CUDA build. Large resident batches, full Adam warm-up, second-order stage run to convergence. **Use this on GPU.**|
|`bs_single_contract.py`     |Original CPU prototype (smaller batches, capped steps). Useful for a quick local sanity check.                           |
|`bs_single_surface.png`     |Example output: closed-form price, PINN price, and absolute-error map.                                                   |

-----

## Installation (H200 / CUDA 12)

```bash
pip install -U "jax[cuda12]" equinox optax matplotlib scipy
pip install "git+https://github.com/raj-brown/optimistix.git@SSBFGS"
```

The second line installs the **forked Optimistix** that provides the `SSBFGS`
and `SSBroyden` classes — the standard PyPI Optimistix does not include them.

For a CPU-only smoke test, replace the first line with
`pip install -U "jax[cpu]" equinox optax matplotlib scipy`.

-----

## Running

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python bs_single_contract_h200.py
```

### Environment variables

|Variable                 |Default                  |Purpose                                                           |
|-------------------------|-------------------------|------------------------------------------------------------------|
|`BENCH_OPT`              |`ssbroyden`              |Second-order stage: `ssbroyden`, `ssbfgs`, or `both`.             |
|`ADAM_STEPS`             |`20000`                  |Adam warm-up steps.                                               |
|`N_INT` / `N_IC` / `N_BC`|`20000` / `4000` / `4000`|Adam collocation batch sizes.                                     |
|`N_INT2` / `N_BC2`       |`60000` / `10000`        |Second-order stage batch sizes.                                   |
|`MAX2`                   |`4000`                   |Max second-order steps (exits earlier on the tolerance criterion).|
|`RESAMPLE_EVERY`         |`1000`                   |Resample Adam collocation points every N steps.                   |

Example — benchmark both optimizers:

```bash
BENCH_OPT=both XLA_PYTHON_CLIENT_PREALLOCATE=false python bs_single_contract_h200.py
```

> **Note:** the first launch spends ~30–60 s on XLA compilation before timing
> starts. Stage timings use `block_until_ready` with the warm-compile call
> excluded, so reported seconds are real wall-clock.

-----

## Output

The script prints a summary table and saves the best model’s price surface:

```
================ SINGLE-CONTRACT BENCHMARK (H200) ================
Stage                           rel L2      rel Linf      time (s)
Adam (warm-up)               x.xxxe-0x     x.xxxe-0x         xx.x
Adam + SSBroyden             x.xxxe-0x     x.xxxe-0x         xx.x
```

- **rel L2** — relative L² error over a 200×200 `(τ, S)` grid vs. closed-form BS.
- **rel L∞** — relative max error over the same grid.
- `bs_single_results_h200.npz` — predicted/true surfaces for plotting.

### Interpreting the error

The relative L∞ is dominated by the **non-differentiable payoff kink** at
`S = K, τ = 0`, where `max(S − K, 0)` has a corner. Away from that point the
solution is smooth and pointwise errors are typically one to two orders of
magnitude smaller. This is an intrinsic feature of the problem, not an optimizer
limitation; smoothing the initial condition or down-weighting collocation near
the kink reduces it if a lower L∞ is needed.

-----

## Hardware notes

- Tuned for a single **NVIDIA H200** (141 GB HBM3e). Batch sizes and the dense
  quasi-Newton Hessian fit comfortably in memory.
- Runs on CPU but is slow there; the per-point second-derivative and the dense
  second-order updates are the expensive parts and are exactly what the GPU
  accelerates. On a smaller GPU, reduce `N_INT2` / `N_BC2` and the network width.

-----

## Roadmap

- [x] Single contract, full `V(t, S)` domain, Adam + SSBroyden / SSBFGS.
- [ ] **Grid of contracts** — add strike `K` (and optionally `σ`) as extra
  network inputs so one PINN prices a whole surface of contracts, then benchmark
  the same recipe across the grid.

-----

## Citation

If you use these optimizers, cite the upstream work:

```bibtex
@article{jnini2026curvature,
  title  = {Curvature-aware optimization for high-accuracy physics-informed neural networks},
  author = {Jnini, Anas and Kiyani, Elham and Shukla, Khemraj and Urban, Jorge F
            and Daryakenari, Nazanin Ahmadi and Muller, Johannes and Zeinhofer, Marius
            and Karniadakis, George Em},
  journal = {arXiv preprint arXiv:2604.05230},
  year    = {2026}
}
```

See the upstream repository for the SSBFGS / SSBroyden and Optimistix citations.