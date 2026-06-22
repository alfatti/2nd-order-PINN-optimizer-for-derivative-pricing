"""
pinn_1d.py
==========
Parametric PINN for the 1-D discrete autocallable, JAX / Equinox, targeting the
H200. Curvature-aware second-order optimization (SSBroyden / SSBFGS from the
CrunchOptimizer/raj-brown Optimistix fork, @SSBFGS branch) with an Adam warmup,
plus an Adam-only ablation switch.

Formulation
-----------
State x = log(S), forward time tau = T - t (tau=0 is maturity). The credit-
adjusted Black-Scholes PDE becomes, in forward time,

    V_tau = 0.5 sig^2 V_xx + (r - q - 0.5 sig^2) V_x - (r + CDS) V.

Loss terms
----------
  (1) PDE residual on interior collocation points.
  (2) Initial condition at tau=0 (maturity, which is also a call date):
          V(x,0) = P_T              for x >= log(C)
                 = f(S)             otherwise            (discontinuous at L)
  (3) MASKED call-date condition -- THE discrete-autocall encoder:
          V(x, tau_i) = P_{t_i}     for x >= log(C), ONLY at tau_i = T - t_i^c.
      Enforced on the discrete call-date time slices; SILENT elsewhere. This is
      what distinguishes the discrete product from the continuous one: between
      call dates x=log(C) is ordinary interior, not a Dirichlet edge.
  (4) Lower boundary x -> x_min: V = S e^{-disc * tau} (deep-loss region, f=S).

The payoff discontinuity at L and the per-call-date kinks along x=log(C) are the
dominant L-infinity error sources (the analogue of the European-call kink in the
BS benchmark). We use (a) tanh-smoothing of the IC near L with an annealed width,
and (b) RAD-style adaptive resampling concentrating collocation near L and the
call-date barrier lines.

Parametric inputs
-----------------
To amortize across a contract family the network also takes (C, T, sig) as
inputs (normalized). One trained network then prices the whole grid by
inference, which is where the PINN beats per-contract FD/MC -- see run_race_1d.

Note: this module is written for the H200 environment (JAX + the Optimistix
fork) and is not exercised on CPU. The numerical references it is scored against
(reference_1d, fd_1d, mc_1d) are CPU-validated.
"""
from __future__ import annotations
import functools
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# Second-order curvature-aware optimizers from the fork. The exact import path
# follows CrunchOptimizer/raj-brown Optimistix @SSBFGS branch.
try:
    import optimistix as optx
    from optimistix import SSBroyden, SSBFGS          # provided by the fork
    _HAVE_SECONDORDER = True
except Exception:                                      # pragma: no cover
    _HAVE_SECONDORDER = False

from config import PAR, RACE


# ----------------------------------------------------------------------------
# Network: a parametric MLP V_theta(x, tau ; C, T, sig)
# ----------------------------------------------------------------------------
class AutocallPINN(eqx.Module):
    mlp: eqx.nn.MLP
    in_scale: jax.Array
    in_shift: jax.Array

    def __init__(self, key, width=RACE.pinn_width, depth=RACE.pinn_depth,
                 in_size=5):
        self.mlp = eqx.nn.MLP(
            in_size=in_size, out_size=1, width_size=width, depth=depth,
            activation=jnp.tanh, key=key)
        # rough input normalization (x, tau, C, T, sig)
        self.in_shift = jnp.array([0.0, 0.0, 102.0, 1.0, 0.20])
        self.in_scale = jnp.array([1.0, 1.0, 5.0, 1.0, 0.10])

    def __call__(self, x, tau, C, T, sig):
        z = (jnp.stack([x, tau, C, T, sig]) - self.in_shift) / self.in_scale
        return self.mlp(z)[0]


def _smooth_payoff(S, par, eps):
    """f(S) with tanh-smoothed step at L (width eps), annealed during training."""
    step = 0.5 * (1.0 + jnp.tanh((S - par.L) / eps))      # ~0 below L, ~1 above
    return step * par.I + (1.0 - step) * S


# ----------------------------------------------------------------------------
# Residual and loss
# ----------------------------------------------------------------------------
def pde_residual(model, x, tau, C, T, sig, disc, mu_extra):
    """V_tau - [0.5 sig^2 V_xx + (r-q-0.5sig^2) V_x - disc V]."""
    f = lambda x_, tau_: model(x_, tau_, C, T, sig)
    V   = f(x, tau)
    V_x = jax.grad(f, argnums=0)(x, tau)
    V_xx = jax.grad(lambda a, b: jax.grad(f, 0)(a, b), 0)(x, tau)
    V_tau = jax.grad(f, argnums=1)(x, tau)
    drift = (mu_extra)                                    # r - q - 0.5 sig^2
    return V_tau - (0.5 * sig**2 * V_xx + drift * V_x - disc * V)


@functools.partial(jax.jit, static_argnums=(0,))
def loss_fn(model, batch, par_tuple, weights, eps):
    (r, q, CDS) = par_tuple
    disc = r + CDS
    xi, ti, Ci, Ti, si = batch["interior"].T
    mu = r - q - 0.5 * si**2
    res = jax.vmap(pde_residual, in_axes=(None,0,0,0,0,0,None,0))(
        model, xi, ti, Ci, Ti, si, disc, mu)
    L_pde = jnp.mean(res**2)

    # initial condition at tau=0
    xic, Cic, Tic, sic = batch["ic"].T
    S = jnp.exp(xic)
    P_T = par_smap_called(par_tuple, Tic)                 # P_T = H e^{B T}
    f_ic = _smooth_payoff(S, PAR, eps)
    target_ic = jnp.where(S >= Cic, P_T, f_ic)
    V_ic = jax.vmap(lambda x_,C_,T_,s_: model(x_, 0.0, C_, T_, s_))(
        xic, Cic, Tic, sic)
    L_ic = jnp.mean((V_ic - target_ic)**2)

    # masked call-date condition: x>=log(C), tau = T - t_i^c
    xcd, tcd, Ccd, Tcd, scd, Pcd, mask = batch["call"].T
    V_cd = jax.vmap(lambda x_,t_,C_,T_,s_: model(x_, t_, C_, T_, s_))(
        xcd, tcd, Ccd, Tcd, scd)
    L_call = jnp.mean(mask * (V_cd - Pcd)**2)

    # lower boundary x->xmin: V = S e^{-disc tau}
    xlb, tlb, Clb, Tlb, slb = batch["lb"].T
    V_lb = jax.vmap(lambda x_,t_,C_,T_,s_: model(x_, t_, C_, T_, s_))(
        xlb, tlb, Clb, Tlb, slb)
    target_lb = jnp.exp(xlb) * jnp.exp(-disc * tlb)
    L_lb = jnp.mean((V_lb - target_lb)**2)

    w_pde, w_ic, w_call, w_lb = weights
    return (w_pde*L_pde + w_ic*L_ic + w_call*L_call + w_lb*L_lb,
            {"pde": L_pde, "ic": L_ic, "call": L_call, "lb": L_lb})


def par_smap_called(par_tuple, T):
    # H e^{B T} with package constants
    return PAR.H * jnp.exp(PAR.B * T)


# ----------------------------------------------------------------------------
# Training: Adam warmup -> second-order refinement (or Adam-only ablation)
# ----------------------------------------------------------------------------
def train(model, sampler, par_tuple, *, second_order=True,
          adam_steps=RACE.adam_steps, so_steps=RACE.secondorder_steps,
          weights=(1.0, 50.0, 50.0, 10.0), key=jax.random.PRNGKey(0)):
    """
    Returns trained model. `sampler` yields a fresh batch dict each step
    (with RAD adaptive resampling near L and the call barrier).

    second_order=False  -> Adam-only ablation (the accuracy-claim control).
    second_order=True   -> Adam warmup then SSBroyden/SSBFGS to drive the loss
                           into the regime an Adam plateau cannot reach.
    """
    opt = optax.adam(3e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def adam_step(model, opt_state, batch, eps):
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, batch, par_tuple, weights, eps)
        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, aux

    # anneal the IC smoothing width eps from coarse to sharp
    for step in range(adam_steps):
        eps = 2.0 * (0.5 ** (4.0 * step / adam_steps)) + 0.05
        batch = sampler(step)
        model, opt_state, loss, aux = adam_step(model, opt_state, batch, eps)

    if second_order and _HAVE_SECONDORDER:
        eps = 0.05                                       # sharp payoff
        batch = sampler(adam_steps)                      # large fixed batch
        flat, unravel = jax.flatten_util.ravel_pytree(
            eqx.filter(model, eqx.is_inexact_array))

        def scalar_loss(flat_params):
            m = eqx.combine(unravel(flat_params), model)
            return loss_fn(m, batch, par_tuple, weights, eps)[0]

        # SSBroyden by default; SSBFGS available identically from the fork.
        solver = SSBroyden(rtol=1e-12, atol=1e-12)
        sol = optx.minimise(
            lambda p, _: scalar_loss(p), solver, flat,
            max_steps=so_steps, throw=False)
        model = eqx.combine(unravel(sol.value), model)

    return model


def price_at_spot(model, par=PAR):
    """V(log S0, tau=T) for one contract by inference."""
    x0 = jnp.log(par.S0)
    return float(model(x0, par.T, par.C, par.T, par.sig))


if __name__ == "__main__":
    print("pinn_1d: H200 module. Run via run_race_1d.py on the GPU host.")
    print("second-order optimizers available:", _HAVE_SECONDORDER)
