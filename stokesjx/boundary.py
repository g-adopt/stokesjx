"""Boundary bases and output rows at the CMB and at the surface.

Everything here is in the ``y`` coordinates of eq. 3.1 with the viscosity scale
``eta_s = eta_ref`` (the largest node viscosity) and the density scale ``rho_s``;
``propagate`` converts to the sweep coordinates ``z = S_l y``. The arrays are built
from static coefficient arrays (numpy, per degree) times a few traced scalars, so that
XLA compiles few operations.

Notation: ``c_s = 4 pi G R rho_s / g(R)`` and ``c_c = 4 pi G r_c rho_s / g(r_c)``,
dimensionless, the coefficients of the potential conditions at the two boundaries
(eq. 8.2, 8.5). With ``rho_s = 4500 kg/m^3`` they are about 2.4 (``c_s``) and 1.2 to
1.4 (``c_c``) on the hydrostatic test tables.

The functions read the record ``formulation.Terms``: ``self_gravity`` switches the
``rho_c`` and ``rho_w`` terms (eq. 5.2), and the boundary classes ``terms.surface`` and
``terms.cmb`` give the free tangential direction of each boundary
(``BoundaryCondition.tangential_unknown``). The side of a boundary (``+1`` at the
surface, ``-1`` at the CMB, the outward normal of the mantle) is fixed inside
``surface_basis`` and ``cmb_basis``, so no caller passes it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .formulation import DegreeOneFrame, Terms


def _unit(k: int) -> np.ndarray:
    """Unit vector ``e_(k+1)`` of length 6 (zero-based ``k``)."""
    e = np.zeros(6)
    e[k] = 1.0
    return e


def _tangential_column(cls, n_degrees: int, *, radius, eta_ref, side: int, parameter):
    """The free tangential direction ``a e2 + b e4`` of a boundary class, ``(L, 6)``.

    ``a`` and ``b`` come from ``cls.tangential_unknown``. ``FreeSlip`` and ``NoSlip``
    return Python floats, so the column is a numpy constant and no traced operation
    enters the program; a class that returns a traced value gives a traced column. The
    column is formed by a product with numpy unit vectors, never by ``float(a)`` or a
    type test, so that both cases take the same code path.
    """
    a, b = cls.tangential_unknown(radius, eta_ref, side, parameter)
    col = a * _unit(1) + b * _unit(3)
    return jnp.broadcast_to(jnp.asarray(col), (n_degrees, 6))


def cmb_basis(degrees, c_c, rho_c_rel, *, terms: Terms, radius, eta_ref, parameter=None) -> jax.Array:
    """Three states at the CMB that satisfy the CMB conditions, ``(L, 6, 3)``.

    Conditions (eq. 8.1, 8.5): ``u_r = 0`` (``y1 = 0``); the tangential condition of the
    class ``terms.cmb`` (free slip ``y4 = 0``, no slip ``y2 = 0``); the potential of the
    regular core solution ``~ r^l`` and the CMB sheet ``(sigma_rr + rho_c phi) / g``:

        y6 = l y5 - c_c (y3 + (rho_c / rho_s) y5)        (FULL)
        y6 = l y5 - c_c y3                                (NO_SELF_GRAVITY, eq. 5.2)

    Under NO_SELF_GRAVITY the core does not respond to the perturbed potential, so the
    sheet is ``sigma_rr / g`` alone. The free unknowns are ``y3``, the tangential
    unknown and ``y5``, so the columns are ``e3 - c_c e6``, ``a e2 + b e4`` (the class's
    direction, ``side = -1``) and ``e5 + (l - c_c rho_c / rho_s [FULL]) e6``.

    Parameters
    ----------
    degrees
        Static degrees, length ``L``.
    c_c
        Scalar ``4 pi G r_c rho_s / g(r_c)``.
    rho_c_rel
        Scalar ``rho_c / rho_s``.
    terms
        The term record (``self_gravity`` and the CMB class).
    radius, eta_ref
        Scalars: the CMB radius in m and the viscosity scale in Pa s, passed to
        ``tangential_unknown``.
    parameter
        ``None`` or a traced scalar, the boundary parameter of the class (``None`` for
        ``FreeSlip`` and ``NoSlip``).
    """
    l = np.asarray(degrees, dtype=np.float64)
    e3, e5, e6 = _unit(2), _unit(4), _unit(5)
    ones = np.ones_like(l)[:, None]
    col1 = ones * e3 - c_c * (ones * e6)
    col2 = _tangential_column(terms.cmb, len(l), radius=radius, eta_ref=eta_ref, side=-1, parameter=parameter)
    if terms.self_gravity:
        col3 = ones * e5 + (l[:, None] - c_c * rho_c_rel) * e6
    else:
        col3 = jnp.asarray(ones * e5 + l[:, None] * e6)
    return jnp.stack([col1, jnp.broadcast_to(col2, col1.shape), jnp.broadcast_to(col3, col1.shape)], axis=-1)


def surface_potential_coefficient(degrees, frame: DegreeOneFrame | None) -> np.ndarray:
    """Static coefficient ``a_l`` of the surface potential row
    ``y6 + a_l y5 + c_s y3 = 0`` per degree, ``(L,)`` float64.

    ``a_l = l + 1`` (the raw row, eq. 8.2: exterior potential ``~ r^-(l+1)``, no
    applied field). At ``l = 1`` in the centre-of-mass frame ``a_1 = -l = -1`` (the CM
    row, eq. 10.5: no exterior self dipole, a free uniform field). The CM row is the
    raw row minus ``(2l + 1) y5``, so the exchange is one static number (CONVENTIONS 5).
    The row holds under both gravity levels: the potential at the surface is that of
    the load and of the deflected boundaries in both (eq. 5.2).
    """
    l = np.asarray(degrees, dtype=np.float64)
    a = l + 1.0
    if frame is DegreeOneFrame.CENTRE_OF_MASS:
        a = np.where(l == 1, -l, a)
    return a


def surface_basis(degrees, c_s, a_l, *, terms: Terms, radius, eta_ref, parameter=None) -> jax.Array:
    """Three states at the surface that satisfy the surface conditions, ``(L, 6, 3)``.

    Conditions (eq. 8.1, 8.2, 10.5): ``y1 = 0``, the tangential condition of the class
    ``terms.surface`` (free slip ``y4 = 0``, no slip ``y2 = 0``) and
    ``y6 + a_l y5 + c_s y3 = 0``. The null space of these three rows has the basis
    ``a e2 + b e4`` (the class's direction, ``side = +1``), ``e3 - c_s e6`` and
    ``e5 - a_l e6``.

    Under ``NoSlip`` no column has a ``y2`` entry, so the second row of every
    orthonormal basis of the downward sweep at the surface is an exact zero, and so is
    the carried velocity row: the surface velocity kernel is an exact zero without a
    special case (``Kernels.surface_velocity``).

    Parameters
    ----------
    degrees
        Static degrees.
    c_s
        Scalar ``4 pi G R rho_s / g(R)``.
    a_l
        Static ``(L,)`` from ``surface_potential_coefficient``.
    terms
        The term record (the surface class).
    radius, eta_ref, parameter
        As in ``cmb_basis``, at the surface.
    """
    l = np.asarray(degrees, dtype=np.float64)
    e3, e5, e6 = _unit(2), _unit(4), _unit(5)
    ones = np.ones_like(l)[:, None]
    col2 = ones * e3 - c_s * (ones * e6)
    col1 = _tangential_column(terms.surface, len(l), radius=radius, eta_ref=eta_ref, side=+1,
                              parameter=parameter)
    col3 = jnp.asarray(ones * e5 - np.asarray(a_l)[:, None] * e6)
    return jnp.stack([jnp.broadcast_to(col1, col2.shape), col2, jnp.broadcast_to(col3, col2.shape)], axis=-1)


def surface_functionals(degrees, *, R, g_R, rho_top, rho_w, rho_s, eta_ref, c_s, geoid_mask,
                        terms: Terms) -> jax.Array:
    """Four linear functionals of the surface state ``y(R)``, ``(L, 4, 6)``.

    Row 0, geoid (eq. 9.1): ``N = eta_ref y5(R) / (rho_s R g(R))``, m per kg/m^2, times
    the static ``geoid_mask`` per degree (0 at ``l = 1`` in the CM frame, where the
    self-potential is zero by construction, eq. 10(d), and in CODE-A's frame,
    driver.f:282-284).

    Row 1, surface topography (eq. 9.2, 8.3):
    ``h_s = -(sigma_rr + rho_w phi) / ((rho_top - rho_w) g(R))`` with
    ``sigma_rr = eta_ref y3 / R`` and ``phi = eta_ref y5 / (rho_s R)``, m per kg/m^2,
    outward positive. The water surface follows the total equipotential, so the water
    term acts on the total ``y5(R)``, which carries no geoid mask. At ``l = 1`` in the
    CM frame of a split core the total potential includes the applied uniform field
    (equations.md 10(e)) while the geoid row is the masked zero, so there the water
    term is not ``rho_w`` times the geoid kernel. Under NO_SELF_GRAVITY the water does
    not respond to the perturbed potential (eq. 5.2): the ``rho_w phi`` term is absent
    and ``h_s = -sigma_rr / ((rho_top - rho_w) g(R))``, so ``(rho_top - rho_w) h_s`` does
    not depend on ``rho_w``.

    Row 2, surface velocity (eq. 9.4): ``u_t(R) sqrt(l (l + 1))``, m/s per kg/m^2, the
    coefficient of the RMS horizontal speed. The same row for every boundary class;
    under ``NoSlip`` it gives an exact zero (``surface_basis``).

    Row 3, the applied part of the potential at ``R`` (equations.md 10(e)):
    ``y5_a = ((l + 1) y5 + y6 + c_s y3) / (2l + 1)``, the amplitude of the exterior
    ``r^l`` part. It is used only at ``l = 1`` for ``degree_one_imbalance`` (eq. 10.6),
    and is zero in the raw frame up to rounding.

    Parameters
    ----------
    degrees
        Static degrees.
    R, g_R, rho_top, rho_w, rho_s, eta_ref, c_s
        Scalars: surface radius (m), surface gravity (m/s^2), topography density,
        water density, density scale (kg/m^3), viscosity scale (Pa s) and
        ``4 pi G R rho_s / g(R)``.
    geoid_mask
        Static ``(L,)`` of zeros and ones.
    terms
        The term record (``self_gravity``).
    """
    l = np.asarray(degrees, dtype=np.float64)
    e2, e3, e5, e6 = _unit(1), _unit(2), _unit(4), _unit(5)
    ones = np.ones_like(l)[:, None]
    mask = np.asarray(geoid_mask, dtype=np.float64)[:, None]
    geoid = (eta_ref / (rho_s * R * g_R)) * (mask * e5)
    den = (rho_top - rho_w) * g_R
    topo = (-(eta_ref / R) / den) * (ones * e3)
    if terms.self_gravity:
        topo = topo + (-(rho_w * eta_ref / (rho_s * R)) / den) * (ones * e5)
    vel = jnp.asarray(np.sqrt(l * (l + 1))[:, None] * e2)
    applied = (((l + 1)[:, None] * e5 + ones * e6) / (2 * l + 1)[:, None]) + c_s * (ones * e3 / (2 * l + 1)[:, None])
    return jnp.stack([jnp.broadcast_to(geoid, topo.shape), topo, jnp.broadcast_to(vel, topo.shape),
                      jnp.broadcast_to(applied, topo.shape)], axis=1)


def cmb_functional(*, r_c, g_c, rho_cmb, rho_c, rho_s, eta_ref, terms: Terms) -> jax.Array:
    """The CMB topography as a functional of the CMB state, ``(6,)`` (eq. 9.3, 8.4):

        h_c = -(sigma_rr + rho_c phi) / ((rho0(r_c) - rho_c) g(r_c))     (FULL)
        h_c = -sigma_rr / ((rho0(r_c) - rho_c) g(r_c))                   (NO_SELF_GRAVITY)

    with ``sigma_rr = eta_ref y3 / r_c`` and ``phi = eta_ref y5 / (rho_s r_c)``, m per
    kg/m^2, outward positive. The state is the one just above the CMB deflection
    (below a sheet at the CMB node). ``rho_c phi`` is the pressure of the core in the
    perturbed potential, a self-gravity term (eq. 5.2).
    """
    den = (rho_cmb - rho_c) * g_c
    row = (-(eta_ref / r_c) / den) * _unit(2)
    if terms.self_gravity:
        row = row + (-(rho_c * eta_ref / (rho_s * r_c)) / den) * _unit(4)
    return row
