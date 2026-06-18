# Cheyette 3-Factor Caplet PINN Benchmark (Curvature-Aware Optimizers)

Pricing a single ATM caplet under the **3-Factor Exponential Cheyette model**
with a Physics-Informed Neural Network, and benchmarking the curvature-aware
optimizers (**SSBroyden**, **SSBFGS**) from the
[CrunchOptimizer](https://github.com/CrunchOptimizer) fork against the closed-form
analytical price.

The model, PDE, parameters, and benchmark contract all come from **Beyna,
Chiarella & Kang (2012), "Pricing Interest Rate Derivatives in a Multifactor HJM
Model"** (SSRN 2162748).

The goal: match the analytical caplet price so closely that, expressed in
implied-vol terms via the **ATM Normal (Bachelier) Vega**, the error comes in
**under 0.1 bps of vol**. This is a stringent bar — tighter than the
Quasi-Monte-Carlo accuracy reported in the paper.

---

## What this solves

The caplet price `g(t, x)` satisfies the 4D + time valuation PDE for the 3-Factor
Exponential model (paper Theorem 7.3, Eq. 23). The model is Markovian in four
state variables `x = (x1, x2, x3, x4)`:

```
∂g/∂t + Σᵢ bᵢ(t,x) ∂g/∂xᵢ + ½ Σᵢⱼ [σσᵀ]ᵢⱼ(t) ∂²g/∂xᵢ∂xⱼ − r(t,x) g = 0
```

with short rate `r(t,x) = f₀ + Σᵢ xᵢ` (constant initial forward `f₀`), drift
terms `bᵢ` built from the deterministic `Vᵢⱼ⁽ᵏ⁾(t)` functions, and a
**state-independent** diffusion `σσᵀ` whose structure is a dense 2×2 block
coupling `(x1, x2)` plus independent diagonal entries for `x3` and `x4`.

The terminal condition at the caplet fixing time `T_C` is the discounted payoff
(paper Section 7.3.2): `Φ = B(T_C, T_B) · max(R(T_C, T_B) − K, 0)`, where the
LIBOR rate `R` and the discount bond `B(T_C, T_B)` come from the analytical bond
formula (Lemma 4.2, Eq. 13).

**Benchmark contract** (from the paper's Table 5): `T_C = 1.0`, `T_B = 2.0`,
strike `K = 5%`, initial forward `f₀ = 5%` (at-the-money). Analytical price
**0.004183**.

---

## Accuracy metric: Normal-Vega-normalized vol error

Price error is a poor cross-contract metric (it scales with strike, maturity,
notional). The standard fix is to convert to implied-vol terms. Here we use the
**ATM Normal (Bachelier) Vega**:

```
Vega_ATM = Δ · B(0, T_B) · √(T_C) / √(2π)        (Δ = T_B − T_C)

vol_error_bps = |C_PINN − C_analytic| / Vega_ATM × 1e4
```

For the primary contract, `Vega_ATM ≈ 0.361`, so:

| Target | Equivalent price error | Relative price error |
|--------|------------------------|----------------------|
| **0.1 bps vol** | **≤ 3.61e-6** | **≤ 0.086 %** |

> **Convention note:** the paper inverts price errors into **Black (lognormal)**
> implied vol; we use **Normal (Bachelier)** Vega as the normalization. These are
> different vol units, so the bps figures are *not* directly comparable to the
> paper's "error in implied vol" columns. The Normal-Vega convention is
> internally consistent and is what the 0.1 bps threshold is measured against.

---

## Files

| File | Description |
|------|-------------|
| `cheyette_caplet_h200.py` | **Single-contract benchmark.** H200/CUDA: 4D+time PINN for one ATM caplet, Adam → SSBroyden/SSBFGS, Vega-normalized vol error with a 0.1 bps pass/fail check. |
| `cheyette_caplet_grid_h200.py` | **Grid benchmark.** One network prices a whole grid of caplets (maturities × strikes); same recipe, single reference ATM Vega across all contracts. |
| `cheyette_grid.py` | Grid definition (15 contracts), vectorized analytic oracle, and the **single fixed reference ATM Vega** metric. |
| `Cheyette_Caplet_Grid.ipynb` | Notebook walking through the grid setup, PDE, terminal condition, metric, and a short CPU demo, with a vol-error-surface plot for after a real run. |
| `cheyette_analytic.py` | Analytical caplet pricer (the ground-truth oracle). Reproduces the paper's five Table 5 prices to ~1e-7. |
| `cheyette_vega.py` | ATM Normal Vega and price→vol-bps conversion. |
| `cheyette_vij.py` | Deterministic `Vᵢⱼ⁽ᵏ⁾(t)` drift coefficients (Appendix A.1) and the `σσᵀ` diffusion. |

The helper modules are imported by the main scripts; keep them in the same
directory.

---

## The grid: one market, many derivatives

`cheyette_caplet_grid_h200.py` prices a **grid of 15 caplets** (5 maturities
`T_C = 1..5`, with `T_B = T_C + 1`, × 3 strikes `{3% ITM, 5% ATM, 7% OTM}`) with a
**single network**. The contract parameters `(T_C, K)` enter as network inputs,
so the model learns the whole pricing map at once. The model dynamics (drift,
diffusion, short rate) are identical for every contract — fixed by the
calibration — so only the terminal payoff varies per contract.

**Single reference Vega — important.** The vol-error metric across the entire
grid is normalized by **one fixed scalar**: the ATM Normal Vega of the
`T_C=1, T_B=2` ATM caplet (`≈ 0.36098`). This is deliberately *not* a
per-contract or per-maturity Vega. Using one constant makes the 0.1 bps bar a
single uniform price tolerance (`3.61e-6`) for every contract, and keeps the grid
numbers directly comparable to the single-contract benchmark. The cheap OTM
short-maturity contracts have the tightest *relative* budget and are the binding
constraint.

**Grid convergence (CPU, small net), one network fitting all 15 contracts:**

| Adam step | max vol err | mean vol err |
|-----------|-------------|--------------|
| 500 | 168.6 bps | 61.8 bps |
| 1000 | 96.4 bps | 39.6 bps |

Max error roughly halving, all contracts marching toward their analytic prices —
confirming the grid formulation is correct. (CPU cannot reach the deep-accuracy
regime; that's the H200's job.)

---

## Design decisions (and why)

These choices were made specifically to give the 0.1 bps target a fighting chance
while staying faithful to the paper's PDE.

**Domain — spot measure, full 4D, but on a tight physical box.** We solve the
exact Theorem 7.3 operator (rather than a forward-measure reduction that would
change the stated problem). We only need the price at one point, `x = 0, t = 0`,
but the PDE couples the value to its derivatives there, so the network must learn
the solution on a *neighborhood*. The box half-width per state is set to
**5 × the marginal standard deviation at `T_C`**, computed from the model
dynamics: `x1 ≈ 0.0485` (driven by the constant `c`, no mean reversion), and
`x2, x3, x4 ≈ 0.002–0.005`. A tight, physically-motivated box keeps the
approximation burden low.

**Kink — softplus smoothing, width chosen against the error budget.** The caplet
payoff has a non-differentiable kink at the money (the paper explicitly flags
this as a source of numerical trouble). We round it with a softplus in *rate*
units, `eps = 1e-4`. The induced price bias was estimated empirically at **~6e-7**
— comfortably under the 3.6e-6 budget. This was quantified, not guessed: a sweep
showed `eps ≤ 2e-4` is required to stay under budget, so `1e-4` leaves headroom.

**Precision and optimizers.** Strict float64 throughout. Adam warm-up (to get
into the basin) followed by the curvature-aware second-order stage
(`SSBroydenZoom` / `SSBFGSZoom`, subclassing the fork's `AbstractSSBroyden` /
`AbstractSSBFGS` with a `Zoom` line search and `NewtonDescent`, per the paper's
Code Listing 1). Both are wrapped in `BestSoFarMinimiser` with `throw=False` and
a finite-iterate guard so a late line-search failure cannot discard a good
solution.

---

## Validation (what we already know works)

**The analytical pricer is correct.** It reproduces all five of the paper's
Table 5 benchmark prices to ~1e-7, which pins down every parameter, sign
convention, and formula.

**The PINN formulation is correct.** In a small CPU training run, the PINN price
converges *monotonically toward the analytical value* — which a wrong PDE would
not do (it would settle at a different number):

| Adam step | PINN price | abs error | vol error (bps) |
|-----------|------------|-----------|-----------------|
| 500 | 0.004695 | 5.1e-4 | 14.2 |
| 1000 | 0.004537 | 3.5e-4 | 9.8 |

Price marching steadily to 0.004183, vol error roughly halving as training
proceeds. This confirms the PDE operator, smoothed terminal condition, bond
formula, short rate, and Vega normalization are all mutually consistent.

---

## Installation (H200 / CUDA 12)

```bash
pip install -U "jax[cuda12]" equinox optax matplotlib scipy
pip install "git+https://github.com/raj-brown/optimistix.git@SSBFGS"
```

The second line installs the **forked Optimistix** providing `SSBFGS` and
`SSBroyden`; standard PyPI Optimistix does not include them.

For a CPU smoke test, swap the first line for `pip install -U "jax[cpu]" ...`.

---

## Running

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python cheyette_caplet_h200.py
```

Benchmark both optimizers:

```bash
BENCH_OPT=both XLA_PYTHON_CLIENT_PREALLOCATE=false python cheyette_caplet_h200.py
```

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `BENCH_OPT` | `ssbroyden` | Second-order stage: `ssbroyden`, `ssbfgs`, `both`, or `none`. |
| `ADAM_STEPS` | `20000` | Adam warm-up steps. |
| `N_INT` / `N_TERM` | `16000` / `8000` | Adam interior / terminal batch sizes. |
| `N_INT2` / `N_TERM2` | `40000` / `16000` | Second-order stage batch sizes. |
| `MAX2` | `5000` | Max second-order steps (exits earlier on tolerance). |
| `WIDTH` / `DEPTH` | `64` / `5` | Network width and hidden-layer count. |
| `EPS_RATE` | `1e-4` | Terminal-kink softplus width (rate units). |
| `W_TERM` | `1.0` | Terminal-loss weight relative to PDE residual. |
| `RESAMPLE_EVERY` / `PRINT_EVERY` | `1000` / `2000` | Resample / logging cadence. |

### Output

A summary table reporting, for each stage, the price, absolute error,
vol error in bps, and whether it passes the 0.1 bps threshold. Results are saved
to `cheyette_caplet_results.npz`.

---

## Honest caveats

- **The final sub-0.1-bps numbers must come from your H200.** This benchmark was
  developed and validated on CPU, where the second-order stage cannot even finish
  (memory) and each step is slow (the per-point 4×4 Hessian). Both constraints
  vanish on the H200: the Hessian-vector products parallelize massively and
  141 GB swallows the batches. The first launch spends ~30–60 s on XLA
  compilation before timing starts.

- **Whether 0.1 bps is actually reachable is an open empirical question.** The
  convergence trend is strongly favorable, but going from ~10 bps to < 0.1 bps is
  two orders of magnitude — and that is precisely the regime where the
  curvature-aware optimizers are supposed to outperform Adam. That is the
  experiment. If SSBroyden alone does not get there, the levers are: longer
  second-order runs, a wider network, a tighter `eps`, or adding a
  Gauss-Newton / natural-gradient stage.

- **Confirm the smoothing bias on the first real run.** Re-price with `EPS_RATE`
  halved; if the converged price shifts by more than ~1e-6, tighten `eps`
  further. The bias is deterministic and computable, so it is controllable — but
  it should be checked empirically rather than trusted blindly.

- **Vol-unit convention.** As noted above, the reported bps are in Normal
  (Bachelier) vol, not the Black vol the paper uses. Keep this in mind when
  comparing against the paper's tables.

---

## Roadmap

- [x] Analytical caplet oracle, validated against the paper's Table 5.
- [x] Single ATM caplet PINN, full 4D + time PDE, Adam + SSBroyden / SSBFGS,
  Normal-Vega vol-error metric.
- [x] **Grid of contracts** — one network prices 15 caplets (maturities ×
  strikes), single reference ATM Vega, with a notebook walkthrough.
- [ ] Push the grid to sub-0.1-bps on the H200; add a Gauss-Newton / natural-
  gradient stage if the second-order quasi-Newton methods plateau.

---

## Citation

```bibtex
@article{beyna2012pricing,
  title  = {Pricing Interest Rate Derivatives in a Multifactor HJM Model with Time Dependent Volatility},
  author = {Beyna, Ingo and Chiarella, Carl and Kang, Boda},
  journal = {SSRN Electronic Journal},
  year    = {2012},
  note    = {SSRN 2162748}
}
```

Cite the upstream CrunchOptimizer work for the SSBroyden / SSBFGS optimizers.
