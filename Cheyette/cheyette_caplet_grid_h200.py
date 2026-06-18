# -*- coding: utf-8 -*-
"""
Cheyette 3-Factor Caplet PINN -- GRID of contracts -- H200 / CUDA build
=======================================================================
One calibrated market, many caplets. A SINGLE network learns the pricing map
    (t, x1, x2, x3, x4 ; TC, K)  ->  caplet price
across a grid of maturities and strikes, by solving the 4D+time valuation PDE
(Beyna, Chiarella & Kang 2012, Theorem 7.3) with the terminal condition enforced
per-contract at each contract's own fixing time TC.

The model dynamics (drift b, diffusion sigma sigma^T, short rate) are FIXED by
the calibration (Table 2); only the terminal payoff varies per contract. So the
contract parameters enter purely through the terminal condition and the network
inputs -- the PDE operator is identical for every contract.

Accuracy metric (as requested): price error normalized by a SINGLE fixed
reference ATM Normal Vega -- the Vega of the (TC=1,TB=2) ATM caplet, ~0.36098 --
applied identically to every contract. NOT a per-contract Vega. Target <= 0.1 bps.

Requires:
  pip install -U "jax[cuda12]" equinox optax matplotlib scipy
  pip install "git+https://github.com/raj-brown/optimistix.git@SSBFGS"

Run:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python cheyette_caplet_grid_h200.py
  BENCH_OPT=ssbroyden|ssbfgs|both  selects the second-order stage.
"""

import os
import time
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax.flatten_util import ravel_pytree
import optimistix as optx

jax.config.update("jax_enable_x64", True)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from cheyette_analytic import PARAMS, F0
from cheyette_grid import (MATURITIES, STRIKES, DELTA, VEGA_REF, BUDGET_01BPS,
                           build_grid, grid_arrays)

_BACKEND = jax.default_backend()
print(f"[device] backend={_BACKEND}  devices={jax.devices()}")
if _BACKEND != "gpu":
    print("[warning] Not on GPU; tuned for an H200, slow on CPU.")

BENCH_OPT = os.environ.get("BENCH_OPT", "ssbroyden").lower()

# ----------------------------- model params -----------------------------------
P = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in PARAMS.items()}

# grid as arrays
CONTRACTS, PRICES = grid_arrays()      # (Ncon,3)=[TC,TB,K], (Ncon,)
CONTRACTS_J = jnp.asarray(CONTRACTS)
PRICES_J = jnp.asarray(PRICES)
N_CON = CONTRACTS.shape[0]
TC_MAX = float(max(tc for tc, _ in MATURITIES))   # 5.0
TC_MIN = float(min(tc for tc, _ in MATURITIES))   # 1.0
K_MIN, K_MAX = float(min(STRIKES)), float(max(STRIKES))

print(f"[grid] {N_CON} contracts; TC in [{TC_MIN},{TC_MAX}], "
      f"K in [{K_MIN:.0%},{K_MAX:.0%}]")
print(f"[target] single reference ATM Vega = {VEGA_REF:.8f}; "
      f"0.1bps price budget = {BUDGET_01BPS:.3e} (uniform for all contracts)")

# ----------------- domain: state box sized to the LARGEST maturity ------------
# marginal stdev scales ~ proportional to sqrt(TC) for the diffusion-dominated
# states; size the box to TC_MAX so it covers every contract's state spread.
# (half-widths from the single-contract analysis, scaled up to TC_MAX)
HALF_TC1 = jnp.array([0.0485, 0.0022, 0.00461, 0.00467])   # at TC=1
HALF = HALF_TC1 * jnp.sqrt(TC_MAX / 1.0)                    # ~ at TC_MAX

# ------------------------- deterministic coefficients -------------------------
def V11_1(t):  return P["c"]**2 * t
def V12_1(t):
    c, a1, a0, lam = P["c"], P["a1_1"], P["a0_1"], P["lam1"]
    return (c/lam**2)*(-a1 + a0*lam + jnp.exp(-lam*t)*(a1 - a0*lam) + a1*lam*t)
def V22_1(t):
    a1, a0, lam = P["a1_1"], P["a0_1"], P["lam1"]
    return (1.0/(4*lam**3))*(2*a0**2*lam**2 + 2*a0*a1*lam*(-1+2*lam*t)
            - jnp.exp(-2*lam*t)*(a1**2 - 2*a0*a1*lam + 2*a0**2*lam**2)
            + a1**2*(1 + 2*lam*t*(-1+lam*t)))
def V11_k(t, a1, a0, lam):
    return (1.0/(4*lam**3))*(2*a0**2*lam**2 + 2*a0*a1*lam*(-1+2*lam*t)
            - jnp.exp(-2*lam*t)*(a1**2 - 2*a0*a1*lam + 2*a0**2*lam**2)
            + a1**2*(1 + 2*lam*t*(-1+lam*t)))

def drift_b(t, x):
    x2, x3, x4 = x[..., 1], x[..., 2], x[..., 3]
    b1 = V11_1(t) + V12_1(t)
    b2 = -P["lam1"]*x2 + V12_1(t) + V22_1(t)
    b3 = -P["lam2"]*x3 + V11_k(t, P["a1_2"], P["a0_2"], P["lam2"])
    b4 = -P["lam3"]*x4 + V11_k(t, P["a1_3"], P["a0_3"], P["lam3"])
    return jnp.stack([b1*jnp.ones_like(x2), b2, b3, b4], axis=-1)

def sigma_sigmaT(t):
    c = P["c"]
    s2 = P["a1_2"]*t + P["a0_2"]
    s3 = P["a1_3"]*t + P["a0_3"]
    b  = P["a1_1"]*t + P["a0_1"]
    M = jnp.zeros((4, 4), dtype=jnp.float64)
    M = M.at[0, 0].set(c**2)
    M = M.at[0, 1].set(c*b); M = M.at[1, 0].set(c*b)
    M = M.at[1, 1].set(b**2)
    M = M.at[2, 2].set(s2**2)
    M = M.at[3, 3].set(s3**2)
    return M

def short_rate(t, x):
    return F0 + jnp.sum(x, axis=-1)

# ------------------------- bond price B(t,T | x) (Eq. 13) ---------------------
def G_vec(t, T):
    l1, l2, l3 = P["lam1"], P["lam2"], P["lam3"]
    dt = T - t
    G1_1 = dt
    G2_1 = (1.0/l1)*(1 - jnp.exp(-l1*dt))
    G1_2 = (1.0/l2)*(1 - jnp.exp(-l2*dt))
    G1_3 = (1.0/l3)*(1 - jnp.exp(-l3*dt))
    return jnp.stack([G1_1*jnp.ones_like(t), G2_1, G1_2, G1_3], axis=-1)

def H_term(t, T):
    l1, l2, l3 = P["lam1"], P["lam2"], P["lam3"]
    dt = T - t
    term1 = 0.5*dt**2 * V11_1(t)
    num2 = (dt*jnp.exp(-l1*t) - jnp.exp(-l1*T))
    term2 = (num2/(l1*jnp.exp(-l1*t)))*V12_1(t)
    e1 = (jnp.exp(-l1*t) - jnp.exp(-l1*T))
    term3 = (e1**2/(2*l1**2*jnp.exp(-2*l1*t)))*V22_1(t)
    e2 = (jnp.exp(-l2*t) - jnp.exp(-l2*T))
    term4 = (e2**2/(2*l2**2*jnp.exp(-2*l2*t)))*V11_k(t, P["a1_2"], P["a0_2"], l2)
    e3 = (jnp.exp(-l3*t) - jnp.exp(-l3*T))
    term5 = (e3**2/(2*l3**2*jnp.exp(-2*l3*t)))*V11_k(t, P["a1_3"], P["a0_3"], l3)
    return term1 + term2 + term3 + term4 + term5

def bond_price(t, T, x):
    """B(t,T) given state x, constant f0 => B(0,T)/B(0,t)=exp(-f0 (T-t))."""
    G = G_vec(t, T)
    H = H_term(t, T)
    base = jnp.exp(-F0*(T - t))
    return base*jnp.exp(-jnp.sum(G*x, axis=-1) - H)

# ------------------------- terminal payoff (smoothed) -------------------------
EPS_RATE = float(os.environ.get("EPS_RATE", 1e-4))
def softplus_eps(y, eps):
    z = y/eps
    return eps*(jnp.maximum(z, 0) + jnp.log1p(jnp.exp(-jnp.abs(z))))

def terminal_payoff(x, TC, TB, K):
    """Caplet terminal value at fixing TC, per contract (vectorized over points)."""
    B = bond_price(TC, TB, x)              # (N,)
    R = (1.0/(TB - TC))*(1.0/B - 1.0)      # LIBOR rate
    return B*(TB - TC)*softplus_eps(R - K, EPS_RATE)

# ------------------------------- network --------------------------------------
# Inputs: (t, x1..x4, TC_norm, K_norm).  Time normalized PER CONTRACT by TC so
# each contract's clock runs [0,1]; tau = t/TC in [0,1].
class GridCapletPINN(eqx.Module):
    linears: list
    def __init__(self, layer_dims, *, key):
        self.linears = []
        for i in range(len(layer_dims)-1):
            key, sub = jax.random.split(key)
            self.linears.append(eqx.nn.Linear(layer_dims[i], layer_dims[i+1], key=sub))
    def _features_one(self, t, x, TC, K):
        # all scalars except x (shape (4,)); returns (8,)
        tau = t/TC
        xn = x / HALF
        tcn = (2.0*(TC - TC_MIN)/(TC_MAX - TC_MIN) - 1.0) if TC_MAX > TC_MIN else 0.0*TC
        kn  = 2.0*(K - K_MIN)/(K_MAX - K_MIN) - 1.0
        return jnp.concatenate([jnp.array([2.0*tau - 1.0]), xn,
                                jnp.array([tcn]), jnp.array([kn])])
    def forward_one(self, t, x, TC, K):
        """Single-example forward. t,TC,K scalars; x shape (4,). Returns scalar."""
        Z = self._features_one(t, x, TC, K)
        for i, lin in enumerate(self.linears):
            Z = lin(Z)
            if i < len(self.linears)-1:
                Z = jax.nn.tanh(Z)
        return Z[0]
    def __call__(self, t, x, TC, K):
        """Batched. t,TC,K shape (N,1); x shape (N,4). Returns (N,1)."""
        out = jax.vmap(self.forward_one)(t[:, 0], x, TC[:, 0], K[:, 0])
        return out[:, None]

WIDTH = int(os.environ.get("WIDTH", 96))     # wider than single-contract: harder map
DEPTH = int(os.environ.get("DEPTH", 6))
# features: [2*tau-1, x1n..x4n, TC_norm, K_norm] = 7 inputs
layer_dims = (7,) + (WIDTH,)*DEPTH + (1,)
key = jax.random.PRNGKey(0)
model = GridCapletPINN(layer_dims, key=key)

# ------------------------- PDE residual (grid) --------------------------------
@eqx.filter_jit
def pde_residual(model, t, x, TC, K):
    """Theorem 7.3 residual. Derivatives wrt physical t and x (not tau)."""
    def per_point(tt, xx, tc, kk):
        f = lambda s, y: model.forward_one(s, y, tc, kk)
        g = f(tt, xx)
        g_t = jax.grad(lambda s: f(s, xx))(tt)
        g_x = jax.grad(lambda y: f(tt, y))(xx)                       # (4,)
        Hx = jax.jacfwd(lambda y: jax.grad(lambda z: f(tt, z))(y))(xx)  # (4,4)
        b = drift_b(tt, xx)
        sst = sigma_sigmaT(tt)
        drift_term = jnp.sum(b*g_x)
        diff_term = 0.5*jnp.sum(sst*Hx)
        r = short_rate(tt[None], xx[None])[0]
        return g_t + drift_term + diff_term - r*g
    return jax.vmap(per_point)(t[:, 0], x, TC[:, 0], K[:, 0])

# ------------------------------- sampling -------------------------------------
gen = jax.random.PRNGKey(1)
def sample_contracts(N):
    """Sample N contracts uniformly from the discrete grid (with replacement)."""
    global gen
    k, gen = jax.random.split(gen)
    idx = jax.random.randint(k, (N,), 0, N_CON)
    cc = CONTRACTS_J[idx]                          # (N,3) [TC,TB,K]
    return cc[:, 0:1], cc[:, 1:2], cc[:, 2:3]      # TC, TB, K columns

def sample_interior(N):
    """Interior collocation: random (t,x) with t in [0,TC] per sampled contract."""
    global gen
    k1, k2, gen = jax.random.split(gen, 3)
    TC, TB, K = sample_contracts(N)
    t = TC * jax.random.uniform(k1, (N, 1))        # 0..TC per contract
    x = (2*jax.random.uniform(k2, (N, 4)) - 1.0)*HALF[None, :]
    return t, x, TC, TB, K

def sample_terminal(N):
    """Terminal collocation at t=TC per sampled contract."""
    global gen
    k, gen = jax.random.split(gen)
    TC, TB, K = sample_contracts(N)
    x = (2*jax.random.uniform(k, (N, 4)) - 1.0)*HALF[None, :]
    u = terminal_payoff(x, TC[:, 0], TB[:, 0], K[:, 0])
    return TC, x, TC, TB, K, u   # (t=TC), x, TC, TB, K, payoff

# ------------------------------- loss -----------------------------------------
W_TERM = float(os.environ.get("W_TERM", 1.0))
@eqx.filter_jit
def loss_fn(model, ti, xi, TCi, Ki, tt, xt, TCt, TBt, Kt, ut):
    r = pde_residual(model, ti, xi, TCi, Ki)
    pred = model(tt, xt, TCt, Kt)[:, 0]
    rt = pred - ut
    return jnp.mean(r**2) + W_TERM*jnp.mean(rt**2)

# ------------------- evaluation: price every grid contract at x=0,t=0 ----------
EVAL_t = jnp.zeros((N_CON, 1))
EVAL_x = jnp.zeros((N_CON, 4))
EVAL_TC = CONTRACTS_J[:, 0:1]
EVAL_K = CONTRACTS_J[:, 2:3]

@eqx.filter_jit
def grid_prices(model):
    return model(EVAL_t, EVAL_x, EVAL_TC, EVAL_K)[:, 0]

def report_grid(model, tag):
    pred = np.array(grid_prices(model))
    err = np.abs(pred - PRICES)
    vol_bps = err / VEGA_REF * 1e4          # SINGLE reference Vega
    npass = int((vol_bps <= 0.1).sum())
    print(f"  [{tag}] max_vol_err={vol_bps.max():.4f} bps  "
          f"mean={vol_bps.mean():.4f} bps  pass {npass}/{N_CON}")
    return pred, err, vol_bps

# =============================== Adam warm-up ==================================
N_INT = int(os.environ.get("N_INT", 24000))
N_TERM = int(os.environ.get("N_TERM", 12000))
ADAM_STEPS = int(os.environ.get("ADAM_STEPS", 30000))
RESAMPLE_EVERY = int(os.environ.get("RESAMPLE_EVERY", 1000))
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", 2000))

ti, xi, TCi, TBi, Ki = sample_interior(N_INT)
tt, xt, TCt, TBt, Kt, ut = sample_terminal(N_TERM)

lr = optax.exponential_decay(2e-3, 3000, 0.9, end_value=1e-5)
opt = optax.adam(lr)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def adam_step(model, opt_state, ti, xi, TCi, Ki, tt, xt, TCt, TBt, Kt, ut):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(
        model, ti, xi, TCi, Ki, tt, xt, TCt, TBt, Kt, ut)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

print("=== Adam warm-up (grid) ===")
_m, _s, _l = adam_step(model, opt_state, ti, xi, TCi, Ki, tt, xt, TCt, TBt, Kt, ut)
_l.block_until_ready()
t0 = time.time()
for it in range(ADAM_STEPS):
    if it > 0 and it % RESAMPLE_EVERY == 0:
        ti, xi, TCi, TBi, Ki = sample_interior(N_INT)
        tt, xt, TCt, TBt, Kt, ut = sample_terminal(N_TERM)
    model, opt_state, loss = adam_step(
        model, opt_state, ti, xi, TCi, Ki, tt, xt, TCt, TBt, Kt, ut)
    if (it+1) % PRINT_EVERY == 0:
        loss.block_until_ready()
        report_grid(model, f"adam {it+1}")
        print(f"      loss={float(loss):.3e}  ({time.time()-t0:.1f}s)")
jax.block_until_ready(eqx.filter(model, eqx.is_array))
adam_time = time.time() - t0
print(f"Adam done in {adam_time:.1f}s")
adam_pred, adam_err, adam_vb = report_grid(model, "ADAM final")

# ===================== Second-order stage (their optimizers) ==================
class SSBroydenZoom(optx.AbstractSSBroyden):
    rtol: float; atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.Zoom()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset = frozenset()

class SSBFGSZoom(optx.AbstractSSBFGS):
    rtol: float; atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.Zoom()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset = frozenset()

N_INT2 = int(os.environ.get("N_INT2", 60000))
N_TERM2 = int(os.environ.get("N_TERM2", 30000))
ti2, xi2, TCi2, TBi2, Ki2 = sample_interior(N_INT2)
tt2, xt2, TCt2, TBt2, Kt2, ut2 = sample_terminal(N_TERM2)

flat0, unflatten = ravel_pytree(eqx.filter(model, eqx.is_array))
static = eqx.filter(model, eqx.is_array, inverse=True)
print("num params:", flat0.size)

def scalar_loss(flat, _):
    mdl = eqx.combine(unflatten(flat), static)
    return loss_fn(mdl, ti2, xi2, TCi2, Ki2, tt2, xt2, TCt2, TBt2, Kt2, ut2)

def run_second_order(make_solver, name, max_steps):
    solver = optx.BestSoFarMinimiser(make_solver())
    print(f"\n=== {name} (grid) ===")
    _ = scalar_loss(flat0, None); jax.block_until_ready(_)
    t1 = time.time()
    sol = optx.minimise(scalar_loss, solver, flat0, max_steps=max_steps, throw=False)
    jax.block_until_ready(sol.value)
    dt = time.time()-t1
    final = jnp.where(jnp.all(jnp.isfinite(sol.value)), sol.value, flat0)
    mdl = eqx.combine(unflatten(final), static)
    floss = float(scalar_loss(final, None))
    nsteps = int(sol.stats["num_steps"])
    print(f"  loss={floss:.3e}  steps={nsteps}  {dt:.1f}s")
    pred, err, vb = report_grid(mdl, name)
    return dict(name=name, model=mdl, pred=pred, err=err, vol_bps=vb,
                steps=nsteps, time=dt)

MAX2 = int(os.environ.get("MAX2", 6000))
runs = []
if BENCH_OPT in ("ssbroyden", "both"):
    runs.append(run_second_order(lambda: SSBroydenZoom(rtol=1e-12, atol=1e-14), "SSBroyden", MAX2))
if BENCH_OPT in ("ssbfgs", "both"):
    runs.append(run_second_order(lambda: SSBFGSZoom(rtol=1e-12, atol=1e-14), "SSBFGS", MAX2))

# ------------------------------- summary --------------------------------------
print("\n================ GRID CAPLET BENCHMARK (H200) ================")
print(f"Grid: {N_CON} contracts (TC={TC_MIN}..{TC_MAX}, K={K_MIN:.0%}..{K_MAX:.0%})")
print(f"SINGLE reference ATM Normal Vega = {VEGA_REF:.8f}")
print(f"0.1 bps  <=>  price error <= {BUDGET_01BPS:.3e} (uniform, all contracts)\n")

def summarize(name, vb):
    print(f"{name:<14}{vb.max():>14.4f}{vb.mean():>14.4f}"
          f"{int((vb<=0.1).sum()):>10}/{N_CON}")
print(f"{'Stage':<14}{'max bps':>14}{'mean bps':>14}{'pass':>13}")
summarize("Adam", adam_vb)
for rr in runs:
    summarize(rr["name"], rr["vol_bps"])

# per-contract table for the best run
best = min(runs, key=lambda d: d["vol_bps"].max()) if runs else None
if best is not None:
    print(f"\nPer-contract (best: {best['name']}):")
    print(f"{'TC':>4}{'K':>6}{'analytic':>12}{'PINN':>12}{'vol bps':>10}{'pass':>6}")
    rows = build_grid()
    for i, r in enumerate(rows):
        print(f"{r['TC']:>4.1f}{r['K']:>6.0%}{PRICES[i]:>12.6f}"
              f"{best['pred'][i]:>12.6f}{best['vol_bps'][i]:>10.4f}"
              f"{'OK' if best['vol_bps'][i]<=0.1 else 'X':>6}")
    np.savez("cheyette_caplet_grid_results.npz",
             contracts=CONTRACTS, analytic=PRICES, vega_ref=VEGA_REF,
             budget=BUDGET_01BPS, adam_pred=adam_pred, adam_vol_bps=adam_vb,
             best_name=best["name"], best_pred=best["pred"],
             best_vol_bps=best["vol_bps"])
    print(f"\nsaved cheyette_caplet_grid_results.npz "
          f"(best max {best['vol_bps'].max():.4f} bps, "
          f"{int((best['vol_bps']<=0.1).sum())}/{N_CON} pass)")
