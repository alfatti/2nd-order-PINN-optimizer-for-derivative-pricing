# -*- coding: utf-8 -*-
"""
Black-Scholes European Call PINN  --  H200 / CUDA build
=======================================================
Single contract, 2D domain V(t, S).  Stays close to the CrunchOptimizer repo
artifacts (Equinox PINN, float64, 1/sqrt(N) residual-LS loss, Adam warm-up ->
their REAL optimistix.SSBroyden / SSBFGS, paper Code Listing 1 subclasses).

H200-specific design choices (vs. the CPU prototype):
  * Large collocation batches held resident on-device (H200 has 141 GB HBM3e).
  * Full Adam warm-up, then second-order stage run to CONVERGENCE, not a step cap.
  * Strict float64 throughout: on Black-Scholes the accuracy ceiling is set by
    the optimizer, not by throughput, and the H200 has the memory to spare.
  * NaN-safe second-order wrapper (BestSoFarMinimiser + throw=False + a manual
    finite-loss guard) so a late line-search failure cannot discard a good iterate.
  * Honest GPU timing via block_until_ready around each stage.

------------------------------------------------------------------------------
INSTALL (run once on the H200 box; needs CUDA 12):

  pip install -U "jax[cuda12]" equinox optax matplotlib scipy
  pip install "git+https://github.com/raj-brown/optimistix.git@SSBFGS"

RUN:

  XLA_PYTHON_CLIENT_PREALLOCATE=false python bs_single_contract_h200.py

Set BENCH_OPT=ssbroyden (default) | ssbfgs | both  to choose the 2nd-order stage.
------------------------------------------------------------------------------
"""

import os
import time
from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax.flatten_util import ravel_pytree
from scipy.stats import norm as _norm
import optimistix as optx

# ----------------------------- Device / precision -----------------------------
jax.config.update("jax_enable_x64", True)          # strict float64 everywhere
# Helpful on a single big GPU; harmless if the env vars are already set.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_latency_hiding_scheduler=true")

_BACKEND = jax.default_backend()
_DEVICES = jax.devices()
print(f"[device] backend={_BACKEND}  devices={_DEVICES}")
if _BACKEND != "gpu":
    print("[warning] JAX is not on a GPU backend. This script is tuned for an "
          "H200; on CPU it will run but be very slow. Check your jax[cuda12] install.")

BENCH_OPT = os.environ.get("BENCH_OPT", "ssbroyden").lower()   # ssbroyden | ssbfgs | both

# --------------------------- Contract / market --------------------------------
K     = 1.0     # strike
r     = 0.05    # risk-free rate
sigma = 0.20    # volatility
T     = 1.0     # maturity
S_max = 3.0 * K # truncated spot domain [0, S_max]

# Forward time tau = T - t in [0, T]; terminal payoff becomes an initial condition.
#   PDE:  V_tau = 0.5 sigma^2 S^2 V_SS + r S V_S - r V
#   IC (tau=0):    V = max(S - K, 0)
#   BC (S=0):      V = 0
#   BC (S=S_max):  V = S_max - K e^{-r tau}

# ----------------------------- Closed-form BS ---------------------------------
def bs_call(S, tau):
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    out = np.maximum(S - K, 0.0)
    m = tau > 1e-14
    if np.any(m):
        Sm, tm = S[m], tau[m]
        d1 = (np.log(np.maximum(Sm, 1e-300) / K) + (r + 0.5 * sigma**2) * tm) / (sigma * np.sqrt(tm))
        d2 = d1 - sigma * np.sqrt(tm)
        out[m] = Sm * _norm.cdf(d1) - K * np.exp(-r * tm) * _norm.cdf(d2)
    return out

# ------------------------- Model architecture ---------------------------------
# Inputs normalized to ~[-1,1].  tanh MLP, linear output.  On the H200 we can
# afford a wider/deeper net than the CPU prototype without hurting iteration time.
class PINN(eqx.Module):
    linears: list
    def __init__(self, layer_dims, *, key):
        self.linears = []
        for i in range(len(layer_dims) - 1):
            key, sub = jax.random.split(key)
            self.linears.append(eqx.nn.Linear(layer_dims[i], layer_dims[i + 1], key=sub))
    def __call__(self, X):
        tau = X[:, 0:1] / T
        s   = X[:, 1:2] / S_max
        Z = jnp.concatenate([2.0 * tau - 1.0, 2.0 * s - 1.0], axis=1)
        for i, layer in enumerate(self.linears):
            Z = jax.vmap(layer)(Z)
            if i < len(self.linears) - 1:
                Z = jax.nn.tanh(Z)
        return Z

WIDTH = 64
DEPTH = 5
layer_dims = (2,) + (WIDTH,) * DEPTH + (1,)
key = jax.random.PRNGKey(0)
model = PINN(layer_dims, key=key)

# ------------------------------ Data makers -----------------------------------
gen_key = jax.random.PRNGKey(1)

def sample_interior(N):
    global gen_key
    k1, k2, gen_key = jax.random.split(gen_key, 3)
    tau = T     * jax.random.uniform(k1, (N, 1))
    S   = S_max * jax.random.uniform(k2, (N, 1))
    return jnp.concatenate([tau, S], axis=1)

def sample_ic(N):
    global gen_key
    k, gen_key = jax.random.split(gen_key)
    S = S_max * jax.random.uniform(k, (N, 1))
    X = jnp.concatenate([jnp.zeros((N, 1)), S], axis=1)
    u = jnp.maximum(S[:, 0] - K, 0.0)
    return X, u

def sample_bc(N):
    global gen_key
    k, gen_key = jax.random.split(gen_key)
    tau = T * jax.random.uniform(k, (N, 1))
    X0  = jnp.concatenate([tau, jnp.zeros((N, 1))], axis=1)
    u0  = jnp.zeros((N,))
    Xm  = jnp.concatenate([tau, jnp.full((N, 1), S_max)], axis=1)
    um  = S_max - K * jnp.exp(-r * tau[:, 0])
    Xb  = jnp.concatenate([X0, Xm], axis=0)
    ub  = jnp.concatenate([u0, um], axis=0)
    return Xb, ub

# ------------------------ PDE residual machinery ------------------------------
# V_SS via forward-over-reverse (jvp of grad) along e_S -- one extra pass, no
# full Hessian.  This is the cheap primitive that the GPU parallelizes well.
@eqx.filter_jit
def pde_residual(model, X):
    def u_single(p):
        return model(p[None, :])[0, 0]
    eS = jnp.array([0.0, 1.0])
    def per_point(p):
        u, g = jax.value_and_grad(u_single)(p)            # V, (V_tau, V_S)
        _, V_SS = jax.jvp(lambda q: jax.grad(u_single)(q)[1], (p,), (eS,))
        return u, g[0], g[1], V_SS
    V, V_tau, V_S, V_SS = jax.vmap(per_point)(X)
    S = X[:, 1]
    return V_tau - (0.5 * sigma**2 * S**2 * V_SS + r * S * V_S - r * V)

# ----------------- Validation grid (closed-form ground truth) -----------------
ng = 200
tt = np.linspace(0.0, T, ng)
ss = np.linspace(0.0, S_max, ng)
TT, SS = np.meshgrid(tt, ss, indexing="ij")
X_val = jnp.array(np.stack([TT.ravel(), SS.ravel()], axis=1))
u_val = jnp.array(bs_call(SS.ravel(), TT.ravel()))

@eqx.filter_jit
def val_errors(model):
    up = model(X_val)[:, 0]
    diff = up - u_val
    rel_l2  = jnp.linalg.norm(diff) / jnp.linalg.norm(u_val)
    rel_inf = jnp.max(jnp.abs(diff)) / jnp.max(jnp.abs(u_val))
    return rel_l2, rel_inf

# ------------------------------ Loss ------------------------------------------
LAM = 1.0

@eqx.filter_jit
def loss_fn(model, Xint, Xic, uic, Xbc, ubc):
    r_pde = pde_residual(model, Xint)
    r_ic  = model(Xic)[:, 0] - uic
    r_bc  = model(Xbc)[:, 0] - ubc
    return (jnp.mean(r_pde**2)
            + LAM * jnp.mean(r_ic**2)
            + LAM * jnp.mean(r_bc**2))

# =============================== Adam warm-up ==================================
# H200 budget: big resident batches + a long warm-up.  Resample periodically so
# the warm-up sees the whole domain (cheap on-device).
N_int, N_ic, N_bc = (int(os.environ.get("N_INT", 20000)),
                     int(os.environ.get("N_IC", 4000)),
                     int(os.environ.get("N_BC", 4000)))
RESAMPLE_EVERY    = int(os.environ.get("RESAMPLE_EVERY", 1000))
ADAM_STEPS        = int(os.environ.get("ADAM_STEPS", 20000))
PRINT_EVERY       = int(os.environ.get("PRINT_EVERY", 1000))

Xint = sample_interior(N_int)
Xic, uic = sample_ic(N_ic)
Xbc, ubc = sample_bc(N_bc)

lr = optax.exponential_decay(3e-3, 2000, 0.9, end_value=1e-5)
opt = optax.adam(lr)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def adam_step(model, opt_state, Xint, Xic, uic, Xbc, ubc):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, Xint, Xic, uic, Xbc, ubc)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

print("=== Adam warm-up (H200) ===")
# warm compile
_m, _s, _l = adam_step(model, opt_state, Xint, Xic, uic, Xbc, ubc)
_l.block_until_ready()
t0 = time.time()
for it in range(ADAM_STEPS):
    if it > 0 and it % RESAMPLE_EVERY == 0:
        Xint = sample_interior(N_int)
        Xic, uic = sample_ic(N_ic)
        Xbc, ubc = sample_bc(N_bc)
    model, opt_state, loss = adam_step(model, opt_state, Xint, Xic, uic, Xbc, ubc)
    if (it + 1) % PRINT_EVERY == 0:
        loss.block_until_ready()
        l2, li = val_errors(model)
        print(f"  adam {it+1:6d} | loss {float(loss):.3e} | relL2 {float(l2):.3e} "
              f"| relLinf {float(li):.3e} | {time.time()-t0:.1f}s")
jax.block_until_ready(eqx.filter(model, eqx.is_array))
adam_time = time.time() - t0
adam_l2, adam_inf = val_errors(model)
print(f"Adam done: relL2 {float(adam_l2):.3e} | relLinf {float(adam_inf):.3e} | {adam_time:.1f}s")

# ================= Their second-order optimizers (Code Listing 1) =============
class SSBroydenZoom(optx.AbstractSSBroyden):
    """SSBroyden + Zoom line search (paper Code Listing 1)."""
    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.Zoom()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset = frozenset()

class SSBFGSZoom(optx.AbstractSSBFGS):
    """SSBFGS + Zoom line search (paper Code Listing 1 analogue)."""
    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.Zoom()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset = frozenset()

# Large fixed batch for the second-order stage (resident on the H200).
N_int2, N_bc2 = int(os.environ.get("N_INT2", 60000)), int(os.environ.get("N_BC2", 10000))
Xint2 = sample_interior(N_int2)
Xic2, uic2 = sample_ic(N_bc2)
Xbc2, ubc2 = sample_bc(N_bc2)

params0 = eqx.filter(model, eqx.is_array)
flat0, unflatten = ravel_pytree(params0)
static = eqx.filter(model, eqx.is_array, inverse=True)
print("num params:", flat0.size)

def scalar_loss(flat, _):
    mdl = eqx.combine(unflatten(flat), static)
    return loss_fn(mdl, Xint2, Xic2, uic2, Xbc2, ubc2)

def run_second_order(make_solver, name, max_steps=4000):
    solver = make_solver()
    solver = optx.BestSoFarMinimiser(solver)   # never discard the best iterate
    print(f"\n=== {name} (Zoom line search) ===")
    # warm compile
    _ = scalar_loss(flat0, None); jax.block_until_ready(_)
    t1 = time.time()
    sol = optx.minimise(
        scalar_loss, solver, flat0,
        max_steps=max_steps,
        throw=False,                            # NaN-safe: return best-so-far
    )
    jax.block_until_ready(sol.value)
    dt = time.time() - t1
    # Guard against a non-finite final iterate (fall back to flat0 if needed).
    final_flat = jnp.where(jnp.all(jnp.isfinite(sol.value)), sol.value, flat0)
    mdl = eqx.combine(unflatten(final_flat), static)
    l2, li = val_errors(mdl)
    floss = float(scalar_loss(final_flat, None))
    nsteps = int(sol.stats["num_steps"])
    res = int(sol.result._value) if hasattr(sol.result, "_value") else -1
    print(f"{name} done: loss {floss:.3e} | relL2 {float(l2):.3e} | relLinf {float(li):.3e} "
          f"| steps {nsteps} | result {res} | {dt:.1f}s")
    return dict(name=name, model=mdl, l2=float(l2), inf=float(li),
                loss=floss, steps=nsteps, time=dt)

MAX2 = int(os.environ.get("MAX2", 4000))
runs = []
if BENCH_OPT in ("ssbroyden", "both"):
    runs.append(run_second_order(
        lambda: SSBroydenZoom(rtol=1e-12, atol=1e-14, norm=optx.max_norm),
        "Adam + SSBroyden", max_steps=MAX2))
if BENCH_OPT in ("ssbfgs", "both"):
    runs.append(run_second_order(
        lambda: SSBFGSZoom(rtol=1e-12, atol=1e-14, norm=optx.max_norm),
        "Adam + SSBFGS", max_steps=MAX2))

# ------------------------------- Summary --------------------------------------
print("\n================ SINGLE-CONTRACT BENCHMARK (H200) ================")
print(f"Contract: K={K}, r={r}, sigma={sigma}, T={T}, domain S in [0,{S_max}]")
print(f"Net: {layer_dims}, tanh, float64, #params={flat0.size}\n")
print(f"{'Stage':<24}{'rel L2':>14}{'rel Linf':>14}{'time (s)':>12}")
print(f"{'Adam (warm-up)':<24}{float(adam_l2):>14.3e}{float(adam_inf):>14.3e}{adam_time:>12.1f}")
for rr in runs:
    print(f"{rr['name']:<24}{rr['l2']:>14.3e}{rr['inf']:>14.3e}{adam_time+rr['time']:>12.1f}")

# Save best model's surface for plotting
best = min(runs, key=lambda d: d["l2"]) if runs else None
if best is not None:
    up = np.array(best["model"](X_val)[:, 0]).reshape(ng, ng)
    np.savez("bs_single_results_h200.npz",
             tt=tt, ss=ss, u_pred=up, u_true=np.array(u_val).reshape(ng, ng),
             adam_l2=float(adam_l2), adam_inf=float(adam_inf), adam_time=adam_time,
             best_name=best["name"], best_l2=best["l2"], best_inf=best["inf"],
             best_time=best["time"], best_steps=best["steps"])
    print(f"\nsaved bs_single_results_h200.npz  (best: {best['name']})")
