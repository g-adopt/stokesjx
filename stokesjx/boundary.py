"""Boundary bases and output rows at the CMB and at the surface.

Everything here is in the ``y`` coordinates of eq. 3.1 with the viscosity scale
``eta_s = eta_ref`` (the largest node viscosity) and the density scale ``rho_s``;
``propagate`` converts to the sweep coordinates ``z = S_l y``. The arrays are built
from static coefficient arrays (numpy, per degree) times a few traced scalars, so that
XLA compiles few operations (risk R1 of the chunk 2 plan).

Notation: ``c_s = 4 pi G R rho_s / g(R)`` and ``c_c = 4 pi G r_c rho_s / g(r_c)``,
dimensionless (about 2.4 and 3.4 on PREM-like tables), the coefficients of the
potential conditions at the two boundaries (eq. 8.2, 8.5).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .formulation import Boundary, DegreeOneFrame, Formulation, GravityModel


def _unit(k: int) -> np.ndarray:
    """Unit vector ``e_(k+1)`` of length 6 (zero-based ``k``)."""
    e = np.zeros(6)
    e[k] = 1.0
    return e


def _check_chunk2(formulation: Formulation):
    """The bases and rows of this module are written for FULL with free slip at both
    boundaries; ``StokesConfig`` refuses the other values (chunk 3 adds the branches)."""
    if (formulation.gravity is not GravityModel.FULL or formulation.surface is not Boundary.FREE_SLIP
            or formulation.cmb is not Boundary.FREE_SLIP):
        raise NotImplementedError("NO_SELF_GRAVITY and no-slip boundaries are added in chunk c3-flags")


def cmb_basis(degrees, c_c, rho_c_rel, *, formulation: Formulation) -> jax.Array:
    """Three states at the CMB that satisfy the CMB conditions, ``(L, 6, 3)``.

    Conditions (eq. 8.1, 8.5): ``u_r = 0`` (``y1 = 0``); free slip ``sigma_rt = 0``
    (``y4 = 0``); the potential of the regular core solution ``~ r^l`` and the CMB sheet
    ``(sigma_rr + rho_c phi) / g``:

        y6 = l y5 - c_c (y3 + (rho_c / rho_s) y5)

    The free unknowns are ``y3``, ``y2``, ``y5``, so the columns are
    ``e3 - c_c e6``, ``e2``, ``e5 + (l - c_c rho_c / rho_s) e6``.

    Parameters
    ----------
    degrees
        Static degrees, length ``L``.
    c_c
        Scalar ``4 pi G r_c rho_s / g(r_c)``.
    rho_c_rel
        Scalar ``rho_c / rho_s``.
    formulation
        FULL with free slip (the ``rho_c`` term is the self-gravitation of the core).
    """
    _check_chunk2(formulation)
    l = np.asarray(degrees, dtype=np.float64)
    e2, e3, e5, e6 = _unit(1), _unit(2), _unit(4), _unit(5)
    ones = np.ones_like(l)[:, None]
    col1 = ones * e3 - c_c * (ones * e6)
    col2 = jnp.asarray(ones * e2)
    col3 = ones * e5 + (l[:, None] - c_c * rho_c_rel) * e6
    return jnp.stack([col1, jnp.broadcast_to(col2, col1.shape), col3], axis=-1)


def surface_potential_coefficient(degrees, frame: DegreeOneFrame | None) -> np.ndarray:
    """Static coefficient ``a_l`` of the surface potential row
    ``y6 + a_l y5 + c_s y3 = 0`` per degree, ``(L,)`` float64.

    ``a_l = l + 1`` (the raw row, eq. 8.2: exterior potential ``~ r^-(l+1)``, no
    applied field). At ``l = 1`` in the centre-of-mass frame ``a_1 = -l = -1`` (the CM
    row, eq. 10.5: no exterior self dipole, a free uniform field). The CM row is the
    raw row minus ``(2l + 1) y5``, so the exchange is one static number (CONVENTIONS 5).
    """
    l = np.asarray(degrees, dtype=np.float64)
    a = l + 1.0
    if frame is DegreeOneFrame.CENTRE_OF_MASS:
        a = np.where(l == 1, -l, a)
    return a


def surface_basis(degrees, c_s, a_l, *, formulation: Formulation) -> jax.Array:
    """Three states at the surface that satisfy the surface conditions, ``(L, 6, 3)``.

    Conditions (eq. 8.1, 8.2, 10.5): ``y1 = 0``, ``y4 = 0`` (free slip),
    ``y6 + a_l y5 + c_s y3 = 0``. The null space of these three rows has the basis
    ``e2``, ``e3 - c_s e6``, ``e5 - a_l e6``.

    Parameters
    ----------
    degrees
        Static degrees.
    c_s
        Scalar ``4 pi G R rho_s / g(R)``.
    a_l
        Static ``(L,)`` from ``surface_potential_coefficient``.
    formulation
        FULL with free slip.
    """
    _check_chunk2(formulation)
    l = np.asarray(degrees, dtype=np.float64)
    e2, e3, e5, e6 = _unit(1), _unit(2), _unit(4), _unit(5)
    ones = np.ones_like(l)[:, None]
    col2 = ones * e3 - c_s * (ones * e6)
    col1 = jnp.broadcast_to(jnp.asarray(ones * e2), col2.shape)
    col3 = jnp.broadcast_to(jnp.asarray(ones * e5 - np.asarray(a_l)[:, None] * e6), col2.shape)
    return jnp.stack([col1, col2, col3], axis=-1)


def surface_functionals(degrees, *, R, g_R, rho_top, rho_w, rho_s, eta_ref, c_s, geoid_mask,
                        formulation: Formulation) -> jax.Array:
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
    term is not ``rho_w`` times the geoid kernel.

    Row 2, surface velocity (eq. 9.4): ``u_t(R) sqrt(l (l + 1))``, m/s per kg/m^2, the
    coefficient of the RMS horizontal speed.

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
    formulation
        FULL with free slip.
    """
    _check_chunk2(formulation)
    l = np.asarray(degrees, dtype=np.float64)
    e2, e3, e5, e6 = _unit(1), _unit(2), _unit(4), _unit(5)
    ones = np.ones_like(l)[:, None]
    mask = np.asarray(geoid_mask, dtype=np.float64)[:, None]
    geoid = (eta_ref / (rho_s * R * g_R)) * (mask * e5)
    den = (rho_top - rho_w) * g_R
    topo = (-(eta_ref / R) / den) * (ones * e3) + (-(rho_w * eta_ref / (rho_s * R)) / den) * (ones * e5)
    vel = jnp.asarray(np.sqrt(l * (l + 1))[:, None] * e2)
    applied = (((l + 1)[:, None] * e5 + ones * e6) / (2 * l + 1)[:, None]) + c_s * (ones * e3 / (2 * l + 1)[:, None])
    return jnp.stack([geoid, topo, jnp.broadcast_to(vel, topo.shape), applied], axis=1)


def cmb_functional(*, r_c, g_c, rho_cmb, rho_c, rho_s, eta_ref, formulation: Formulation) -> jax.Array:
    """The CMB topography as a functional of the CMB state, ``(6,)`` (eq. 9.3, 8.4):

        h_c = -(sigma_rr + rho_c phi) / ((rho0(r_c) - rho_c) g(r_c))

    with ``sigma_rr = eta_ref y3 / r_c`` and ``phi = eta_ref y5 / (rho_s r_c)``, m per
    kg/m^2, outward positive. The state is the one just above the CMB deflection
    (below a sheet at the CMB node). ``rho_c phi`` is the pressure of the core in the
    perturbed potential (FULL).
    """
    _check_chunk2(formulation)
    den = (rho_cmb - rho_c) * g_c
    return (-(eta_ref / r_c) / den) * _unit(2) + (-(rho_c * eta_ref / (rho_s * r_c)) / den) * _unit(4)
