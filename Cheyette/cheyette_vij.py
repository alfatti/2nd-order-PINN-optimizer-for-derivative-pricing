# -*- coding: utf-8 -*-
"""
Deterministic V_ij^(k)(t) functions for the 3-Factor Exponential Cheyette model.

Source: Beyna, Chiarella & Kang (2012), "Pricing Interest Rate Derivatives in a
Multifactor HJM Model", Appendix A.1 (pages 40-41 of the PDF), transcribed from
the rasterized equations.

Model parameterization (linear polynomials, m = 1):
    factor 1 (N1 = 2):  beta_1^(1) = c ;            beta_2^(1) = a1^(1) t + a0^(1)
    factor 2 (N2 = 1):  beta_1^(2) = a1^(2) t + a0^(2)
    factor 3 (N3 = 1):  beta_1^(3) = a1^(3) t + a0^(3)
    alpha_1^(1) = 1, alpha_2^(1) = exp(-lam1 t), alpha_1^(2) = exp(-lam2 t),
    alpha_1^(3) = exp(-lam3 t)

These V's enter the valuation-PDE drift (Theorem 7.3, Eq. 23):
    b1 = V11^(1) + V12^(1)
    b2 = -lam1 x2 + V21^(1) + V22^(1)
    b3 = -lam2 x3 + V11^(2)
    b4 = -lam3 x4 + V11^(3)

All functions are deterministic in t (no state dependence).
Parameters per factor k: (a1, a0, lam) = (a1^(k), a0^(k), lambda^(k));
factor 1 additionally uses the constant c.
"""

import jax.numpy as jnp


# ----- factor 1 (two summands; constant c plus linear beta) -------------------
def V11_1(t, c):
    """V_11^(1)(t) = c^2 t."""
    return c**2 * t


def V12_1(t, c, a1, a0, lam):
    """V_12^(1) = V_21^(1).
        = (c / lam^2) [ -a1 + a0 lam + exp(-lam t)(a1 - a0 lam) + a1 lam t ]
    """
    return (c / lam**2) * (
        -a1 + a0 * lam
        + jnp.exp(-lam * t) * (a1 - a0 * lam)
        + a1 * lam * t
    )


# V_21^(1) is identical to V_12^(1)
V21_1 = V12_1


def V22_1(t, a1, a0, lam):
    """V_22^(1)(t)
        = 1/(4 lam^3) [ 2 a0^2 lam^2 + 2 a0 a1 lam (-1 + 2 lam t)
                        - exp(-2 lam t)( a1^2 - 2 a0 a1 lam + 2 a0^2 lam^2 )
                        + a1^2 ( 1 + 2 lam t (-1 + lam t) ) ]
    """
    return (1.0 / (4.0 * lam**3)) * (
        2 * a0**2 * lam**2
        + 2 * a0 * a1 * lam * (-1 + 2 * lam * t)
        - jnp.exp(-2 * lam * t) * (a1**2 - 2 * a0 * a1 * lam + 2 * a0**2 * lam**2)
        + a1**2 * (1 + 2 * lam * t * (-1 + lam * t))
    )


# ----- factors 2 and 3 (single summand; same functional form) -----------------
def V11_single(t, a1, a0, lam):
    """V_11^(k)(t) for k in {2,3} (single-summand factors):
        = 1/(4 lam^3) [ 2 a0^2 lam^2 + 2 a0 a1 lam (-1 + 2 lam t)
                        - exp(-2 lam t)( a1^2 - 2 a0 a1 lam + 2 a0^2 lam^2 )
                        + a1^2 ( 1 + 2 lam t (-1 + lam t) ) ]

    Note: identical algebraic form to V_22^(1); only the parameters differ.
    """
    return (1.0 / (4.0 * lam**3)) * (
        2 * a0**2 * lam**2
        + 2 * a0 * a1 * lam * (-1 + 2 * lam * t)
        - jnp.exp(-2 * lam * t) * (a1**2 - 2 * a0 * a1 * lam + 2 * a0**2 * lam**2)
        + a1**2 * (1 + 2 * lam * t * (-1 + lam * t))
    )


def drift_b(t, x, params):
    """Assemble the 4-vector drift b_i(t,x) of the valuation PDE (Eq. 23).

    x      : array-like of shape (..., 4) = (x1, x2, x3, x4)
    params : dict with keys
        c,
        a1_1, a0_1, lam1,   # factor 1 linear-beta coeffs + decay
        a1_2, a0_2, lam2,   # factor 2
        a1_3, a0_3, lam3,   # factor 3
    Returns array of shape (..., 4).
    """
    c = params["c"]
    a1_1, a0_1, lam1 = params["a1_1"], params["a0_1"], params["lam1"]
    a1_2, a0_2, lam2 = params["a1_2"], params["a0_2"], params["lam2"]
    a1_3, a0_3, lam3 = params["a1_3"], params["a0_3"], params["lam3"]

    x2 = x[..., 1]
    x3 = x[..., 2]
    x4 = x[..., 3]

    b1 = V11_1(t, c) + V12_1(t, c, a1_1, a0_1, lam1)
    b2 = -lam1 * x2 + V21_1(t, c, a1_1, a0_1, lam1) + V22_1(t, a1_1, a0_1, lam1)
    b3 = -lam2 * x3 + V11_single(t, a1_2, a0_2, lam2)
    b4 = -lam3 * x4 + V11_single(t, a1_3, a0_3, lam3)

    return jnp.stack([b1, b2, b3, b4], axis=-1)


def sigma_matrix(t, params):
    """Diffusion matrix sigma(t) in R^{4x3} (Eq. 22), state-independent."""
    c = params["c"]
    a1_1, a0_1 = params["a1_1"], params["a0_1"]
    a1_2, a0_2 = params["a1_2"], params["a0_2"]
    a1_3, a0_3 = params["a1_3"], params["a0_3"]
    z = jnp.zeros_like(jnp.asarray(t, dtype=jnp.float64))
    row1 = jnp.stack([c + z,                z,               z], axis=-1)
    row2 = jnp.stack([a1_1 * t + a0_1,      z,               z], axis=-1)
    row3 = jnp.stack([z,    a1_2 * t + a0_2,                 z], axis=-1)
    row4 = jnp.stack([z,                z,   a1_3 * t + a0_3], axis=-1)
    return jnp.stack([row1, row2, row3, row4], axis=-2)  # (...,4,3)


def sigma_sigmaT(t, params):
    """[sigma sigma^T](t) in R^{4x4} (Eq. 22), the PDE diffusion coefficient."""
    s = sigma_matrix(t, params)
    return jnp.einsum("...ik,...jk->...ij", s, s)


if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)
    p = dict(c=0.2,
             a1_1=0.1, a0_1=0.3, lam1=0.5,
             a1_2=0.05, a0_2=0.2, lam2=1.0,
             a1_3=0.02, a0_3=0.1, lam3=1.5)
    t = jnp.array(2.0)
    x = jnp.array([0.01, -0.02, 0.005, 0.0])
    print("b(t,x) =", drift_b(t, x, p))
    sst = sigma_sigmaT(t, p)
    print("sigma sigma^T(t) =\n", sst)
    # Structural checks
    assert sst.shape == (4, 4)
    # x3, x4 blocks must be decoupled from everything else
    assert jnp.allclose(sst[2, 0], 0.0) and jnp.allclose(sst[3, 0], 0.0)
    assert jnp.allclose(sst[2, 3], 0.0)
    # 2x2 (x1,x2) block is dense and symmetric PSD
    assert jnp.allclose(sst[0, 1], sst[1, 0])
    # V_22^(1) and V_11^(2) share functional form -> sanity: positive for these params
    print("V22_1 =", float(V22_1(t, p["a1_1"], p["a0_1"], p["lam1"])))
    print("V11_single(factor2) =", float(V11_single(t, p["a1_2"], p["a0_2"], p["lam2"])))
    print("V11_single(factor3) =", float(V11_single(t, p["a1_3"], p["a0_3"], p["lam3"])))
    print("OK: structural checks passed.")
