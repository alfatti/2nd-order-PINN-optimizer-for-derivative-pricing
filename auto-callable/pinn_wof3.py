"""
pinn_wof3.py
============
PINN for the worst-of-3 autocallable, JAX/Equinox, H200. Same machinery as
pinn_1d but state is (x1,x2,x3) = log-ratios plus forward time tau. The point of
this module in the race: the PINN's cost scales MILDLY with dimension. Going from
1-D to 3-D changes the input layer from 2 to 4 and modestly raises the
collocation budget; it does NOT incur the N^3 grid blow-up that kills FD
(wof3_fd extrapolates ~2.6 years for an accuracy-grade 3-D grid).

PDE (forward time):
  V_tau = sum_k 0.5 sig_k^2 V_{kk} + sum_{k<l} rho sig_k sig_l V_{kl}
        + sum_k (r - q - 0.5 sig_k^2) V_k - disc V.

Worst-of driver W = min_k exp(x_k); call/threshold/maturity conditions as in
wof3_mc. The masked call-date loss enforces V = P_{t_i} on the called region
{ W >= C/S0 } at the discrete slices tau_i = T - t_i^c (flat plateau above the
worst-of barrier, exactly as in 1-D).
"""
from __future__ import annotations
import functools
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

try:
    import optimistix as optx
    from optimistix import SSBroyden, SSBFGS
    _HAVE_SECONDORDER = True
except Exception:                                       # pragma: no cover
    _HAVE_SECONDORDER = False

from config import WOF3, RACE


class WoF3PINN(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key, width=RACE.pinn_width, depth=RACE.pinn_depth):
        # inputs: (x1, x2, x3, tau)
        self.mlp = eqx.nn.MLP(in_size=4, out_size=1, width_size=width,
                              depth=depth, activation=jnp.tanh, key=key)

    def __call__(self, x1, x2, x3, tau):
        return self.mlp(jnp.stack([x1, x2, x3, tau]))[0]


def _full_hessian_terms(f, x1, x2, x3, tau, vols, rho):
    """Second-order spatial operator with the three cross terms."""
    g = lambda v: f(v[0], v[1], v[2], tau)
    v = jnp.array([x1, x2, x3])
    H = jax.hessian(g)(v)                                # 3x3
    diff = 0.0
    for k in range(3):
        diff += 0.5 * vols[k]**2 * H[k, k]
    for (k, l) in ((0, 1), (0, 2), (1, 2)):
        diff += rho * vols[k] * vols[l] * H[k, l]
    grad = jax.grad(g)(v)
    return H, grad, diff


def pde_residual(model, x1, x2, x3, tau, par):
    vols = jnp.array(par.vols)
    f = lambda a, b, c, t: model(a, b, c, t)
    _, grad, diff = _full_hessian_terms(f, x1, x2, x3, tau, vols, par.rho)
    mu = jnp.array([par.r - par.q - 0.5 * v**2 for v in par.vols])
    V = f(x1, x2, x3, tau)
    V_tau = jax.grad(lambda t: f(x1, x2, x3, t))(tau)
    return V_tau - (diff + jnp.dot(mu, grad) - par.disc * V)


@functools.partial(jax.jit, static_argnums=(0, 2))
def loss_fn(model, batch, par, weights):
    xi = batch["interior"]
    res = jax.vmap(lambda p: pde_residual(model, p[0], p[1], p[2], p[3], par))(xi)
    L_pde = jnp.mean(res**2)

    # IC at tau=0 on worst-of driver
    xic = batch["ic"]
    W = jnp.exp(jnp.min(xic[:, :3], axis=1))
    cL, LL = par.C/par.S0, par.L/par.S0
    f_ic = jnp.where(W > LL, par.H, par.H * W)
    target_ic = jnp.where(W >= cL, par.H * jnp.exp(par.B * par.T), f_ic)
    V_ic = jax.vmap(lambda p: model(p[0], p[1], p[2], 0.0))(xic)
    L_ic = jnp.mean((V_ic - target_ic)**2)

    # masked call-date condition on called region W >= C/S0
    xc = batch["call"]                                   # cols: x1,x2,x3,tau,P,mask
    V_c = jax.vmap(lambda p: model(p[0], p[1], p[2], p[3]))(xc)
    L_call = jnp.mean(xc[:, 5] * (V_c - xc[:, 4])**2)

    w_pde, w_ic, w_call = weights
    return (w_pde*L_pde + w_ic*L_ic + w_call*L_call,
            {"pde": L_pde, "ic": L_ic, "call": L_call})


def train(model, sampler, par=WOF3, *, second_order=True,
          adam_steps=RACE.adam_steps, so_steps=RACE.secondorder_steps,
          weights=(1.0, 50.0, 50.0)):
    opt = optax.adam(3e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def adam_step(model, opt_state, batch):
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, batch, par, weights)
        updates, opt_state = opt.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), opt_state, loss, aux

    for step in range(adam_steps):
        model, opt_state, loss, aux = adam_step(model, opt_state, sampler(step))

    if second_order and _HAVE_SECONDORDER:
        batch = sampler(adam_steps)
        flat, unravel = jax.flatten_util.ravel_pytree(
            eqx.filter(model, eqx.is_inexact_array))
        scalar = lambda p, _: loss_fn(
            eqx.combine(unravel(p), model), batch, par, weights)[0]
        sol = optx.minimise(scalar, SSBroyden(rtol=1e-12, atol=1e-12),
                            flat, max_steps=so_steps, throw=False)
        model = eqx.combine(unravel(sol.value), model)
    return model


def price_at_spot(model, par=WOF3):
    return float(model(0.0, 0.0, 0.0, par.T))            # all ratios = 1 at spot


if __name__ == "__main__":
    print("pinn_wof3: H200 module. second-order available:", _HAVE_SECONDORDER)
