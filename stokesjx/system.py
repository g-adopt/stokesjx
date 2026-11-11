"""The 6x6 system in ``ln r``, its interval exponents, and the source jump.

State vector (NOTES/docs/equations.md, eq. 3.1), all components in m/s:

    y = (u_r, u_t, r sigma_rr / eta_s, r sigma_rt / eta_s,
         rho_s r phi / eta_s, rho_s r^2 dphi/dr / eta_s)

with the density scale ``rho_s`` (``Units.density_scale``) and the viscosity scale
``eta_s``. The solver uses ``eta_s = eta_ref``, the largest node viscosity
(``propagate.reference_viscosity``; NOTES/chunks/c2-forward-core/01-plan.md, D1), so every scaled interval
viscosity ``e_i = etabar_i / eta_ref`` is at most one.

``dy / d ln r = A y`` with A constant in each interval (interval-mean material, eq.
12.1), so the step across an interval of width ``h`` in ``ln r`` is exactly
``expm(A h)``; ``g`` does not enter A (F1). Two exact similarities remove the large
entries of A (eq. 4.5, 4.6):

- viscosity: ``A(e) = D A(1) D^-1``, ``D = diag(1, 1, e, e, e, e)``;
- degree: ``S_l = diag(1, l, 1/(2l), 1/2, 1/2, 1/(2l))`` brings every entry of
  ``S_l A(1) S_l^-1`` to at most about ``2 (l + 1)``, and the 1-norm below
  ``3 l + 10 + rho0 / rho_s``.

The sweeps run in ``z = S_l y``. Because ``S_l`` and ``D`` are diagonal they commute,
and the step of interval ``i`` in ``z`` is ``Z_i = D_i expm(B_i) D_i^-1`` with the
balanced exponent ``B_i = S_l A(1, d_i) S_l^-1 h_i`` (``d_i = rhobar_i / rho_s``). The
exponential has no viscosity in it, so the derivative with respect to a viscosity never
passes through it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .profile import NodeStructure
from .units import Units


def degree_scaling(degrees) -> np.ndarray:
    """Diagonal of ``S_l = diag(1, l, 1/(2l), 1/2, 1/2, 1/(2l))`` per degree (eq. 4.6).

    Parameters
    ----------
    degrees
        Static degrees (a tuple or array of positive integers).

    Returns
    -------
    float64 numpy array of shape ``(L, 6)``. The degrees are static, so the scaling
    is a compile-time constant.
    """
    l = np.asarray(degrees, dtype=np.float64)
    one = np.ones_like(l)
    return np.stack([one, l, 1 / (2 * l), one / 2, one / 2, 1 / (2 * l)], axis=-1)


def a_matrix(l, eta_rel, rho_rel, *, self_gravity: bool) -> jax.Array:
    """The incompressible matrix A0 in ``ln r`` for one interval (eq. 4.1, driver.f:369-398).

    Parameters
    ----------
    l
        Degree (broadcasts).
    eta_rel
        Interval viscosity over the viscosity scale, ``etabar / eta_s`` (broadcasts).
    rho_rel
        Interval density over the density scale, ``rhobar / rho_s`` (broadcasts).
    self_gravity
        FULL: the entries ``a(3,6) = a(4,5) = -rho0 / rho_s`` (eq. 5.1). False:
        NO_SELF_GRAVITY, both zero (eq. 5.2).

    Returns
    -------
    float64 array of shape ``broadcast(l, eta_rel, rho_rel).shape + (6, 6)``.

    Rows (``L = l (l + 1)``): continuity ``d y1 = -2 y1 + L y2``; definition of
    ``sigma_rt``: ``d y2 = -y1 + y2 + y4 / e``; radial momentum
    ``d y3 = 12 e y1 - 6 L e y2 + y3 + L y4 - d y6``; tangential momentum
    ``d y4 = -6 e y1 + 2 (2L - 1) e y2 - y3 - 2 y4 - d y5``; Laplace's equation
    ``d y5 = y5 + y6``, ``d y6 = L y5``. For tests and documentation; the solver
    builds the balanced exponent from static coefficient arrays
    (``balanced_exponent``).
    """
    l, e, d = jnp.broadcast_arrays(*(jnp.asarray(v, jnp.float64) for v in (l, eta_rel, rho_rel)))
    L = l * (l + 1)
    z, one = jnp.zeros_like(l), jnp.ones_like(l)
    g = d if self_gravity else z
    rows = [
        [-2 * one, L, z, z, z, z],
        [-one, one, z, 1 / e, z, z],
        [12 * e, -6 * L * e, one, L, z, -g],
        [-6 * e, 2 * (2 * L - 1) * e, -one, -2 * one, -g, z],
        [z, z, z, z, one, one],
        [z, z, z, z, L, z],
    ]
    return jnp.stack([jnp.stack(r, axis=-1) for r in rows], axis=-2)


def k_matrix(eta_rel) -> jax.Array:
    """``K = dA/dkappa`` of TALA (eq. 4.3): ``K(1,1) = -1``, ``K(3,1) = 4 e``,
    ``K(4,1) = -2 e``, all other entries zero. ``K = v e1^T`` with ``e1^T v = -1``, so
    ``K^2 = -K``. Shape ``eta_rel.shape + (6, 6)``. For tests and documentation; the solver
    uses the balanced static array of ``_static_exponent_parts``."""
    e = jnp.asarray(eta_rel, jnp.float64)
    z = jnp.zeros_like(e)
    col = jnp.stack([-jnp.ones_like(e), z, 4 * e, -2 * e, z, z], axis=-1)
    # Outer product of the column v with e1: only column 1 of K is nonzero.
    return col[..., :, None] * jnp.eye(6)[0]


def _static_exponent_parts(degrees, self_gravity: bool):
    """Static coefficient arrays of the balanced exponent, batch-last ``(6, 6, L)``.

    ``A(1, d)`` is affine in ``d = rho0 / rho_s``: ``d`` enters only ``a(3,6)`` and
    ``a(4,5)`` (eq. 4.1, 5.1). So ``S_l A(1, d) S_l^-1 = P_l + d Q_l`` with
    ``P_l = S_l A(1, 0) S_l^-1`` and ``Q_l = S_l (dA/dd) S_l^-1``. Built in numpy on the
    host; the degrees are static.
    """
    l = np.asarray(degrees, dtype=np.float64)
    L = l * (l + 1)
    A0 = np.zeros((len(l), 6, 6))
    A0[:, 0, 0], A0[:, 0, 1] = -2, L
    A0[:, 1, 0], A0[:, 1, 1], A0[:, 1, 3] = -1, 1, 1
    A0[:, 2, 0], A0[:, 2, 1], A0[:, 2, 2], A0[:, 2, 3] = 12, -6 * L, 1, L
    A0[:, 3, 0], A0[:, 3, 1], A0[:, 3, 2], A0[:, 3, 3] = -6, 2 * (2 * L - 1), -1, -2
    A0[:, 4, 4], A0[:, 4, 5] = 1, 1
    A0[:, 5, 4] = L
    dA = np.zeros((len(l), 6, 6))
    if self_gravity:
        dA[:, 2, 5] = -1.0
        dA[:, 3, 4] = -1.0
    K = np.zeros((len(l), 6, 6))
    K[:, 0, 0], K[:, 2, 0], K[:, 3, 0] = -1.0, 4.0, -2.0
    s = degree_scaling(degrees)
    sim = s[:, :, None] / s[:, None, :]
    to_batch_last = lambda M: np.moveaxis(M * sim, 0, -1)
    return to_batch_last(A0), to_batch_last(dA), to_batch_last(K)


def balanced_exponent(degrees, rho_rel, h, kappa_h=None, *, self_gravity: bool) -> jax.Array:
    """Balanced interval exponents ``B = S_l (A(1, d) h + K(1) kappa_h) S_l^-1``, for
    every interval and degree, batch-last (eq. 4.5, 4.6).

    Parameters
    ----------
    degrees
        Static degrees, length ``L``.
    rho_rel
        ``(n,)``: interval density over ``rho_s``, ``d_i = rhobar_i / rho_s``.
    h
        ``(n,)``: interval width in ``ln r`` (zero at a pair).
    kappa_h
        ``(n,)`` or ``None``: ``ln(rho0_i / rho0_(i-1))`` of TALA (eq. 12.2), zero at a
        jump pair (``kappa_widths``); ``None`` for the incompressible model (no K term).
    self_gravity
        FULL (True) or NO_SELF_GRAVITY (False): the density entries of eq. 5.1.

    Returns
    -------
    float64 array of shape ``(6, 6, n, L)``.

    The exponent is two broadcast products of the static arrays of
    ``_static_exponent_parts`` with the traced ``h`` and ``d``: one fused elementwise
    expression, no stack of 36 entries, which compiles faster. The
    derivative with respect to ``h`` and ``d`` is exact and cheap. The batch order
    ``(n, L)`` makes the batch-first view ``(n, L, 6, 6)`` of the scan one
    ``moveaxis``. This function is the one place where a Magnus exponent with
    material that varies inside an interval (ALA, PLAN section 5) would change.
    """
    P, Q, K = _static_exponent_parts(degrees, self_gravity)
    h = jnp.asarray(h, jnp.float64)[None, None, :, None]
    d = jnp.asarray(rho_rel, jnp.float64)[None, None, :, None]
    B = h * (P[:, :, None, :] + d * Q[:, :, None, :]) if self_gravity else h * P[:, :, None, :]
    if kappa_h is not None:
        B = B + jnp.asarray(kappa_h, jnp.float64)[None, None, :, None] * K[:, :, None, :]
    return B


def viscosity_similarity(E, e_rel) -> jax.Array:
    """``D E D^-1`` with ``D = diag(1, 1, e, e, e, e)`` per interval, elementwise
    (eq. 4.5).

    Parameters
    ----------
    E
        ``(6, 6, n, L)``, batch-last, the exponential of the balanced exponent.
    e_rel
        ``(n,)``: interval viscosity over the viscosity scale, ``etabar_i / eta_ref``.

    Returns
    -------
    ``(6, 6, n, L)``: entry ``(a, b)`` of ``E`` times ``D_a / D_b``, that is times 1,
    ``e`` or ``1 / e``. The viscosity enters the step only here, so the derivative with
    respect to a viscosity is one elementwise product.
    """
    e = jnp.asarray(e_rel, jnp.float64)
    one = jnp.ones_like(e)
    D = jnp.stack([one, one, e, e, e, e])                 # (6, n)
    return E * (D[:, None, :] / D[None, :, :])[..., None]


def interval_widths(radius, structure: NodeStructure) -> jax.Array:
    """Interval widths in ``ln r``, ``(N - 1,)``:
    ``h_i = log1p((r_i - r_(i-1)) / r_(i-1))``, times the static zero of the pairs.

    ``log1p`` of the relative gap in place of ``log(r_i / r_(i-1))``: the ratio is
    close to one, and its rounding gives ``h`` a relative error of ``2.2e-16 / h``
    (2.4e-14 on 129 nodes), which set the error floor of the steps at high degree
    (NOTES/chunks/c1-foundations/04-review-physics-r3.md, minor 3). The static mask makes the
    width of a pair exactly zero, so its step is exactly the identity.
    """
    r = jnp.asarray(radius, jnp.float64)
    return jnp.log1p((r[1:] - r[:-1]) / r[:-1]) * structure.interval_mask


def cut_intervals(radius, cut_radius, structure: NodeStructure) -> tuple[jax.Array, jax.Array, jax.Array]:
    """For every viscosity cut: the interval that holds it and the widths of its two parts
    in ``ln r`` (NOTES/docs/equations.md section 19).

    Parameters
    ----------
    radius
        ``(N,)`` node radii in m (can be traced).
    cut_radius
        ``(C,)`` cut radii ``c_q`` in m (can be traced).
    structure
        Static ``NodeStructure`` with ``cuts``: the range nodes ``(lo_q, hi_q)``.

    Returns
    -------
    ``(a, h_lo, h_hi)``, each ``(C,)``: the 0-based interval index ``a`` (the interval
    between the nodes ``a`` and ``a + 1``, ``r_a <= c < r_(a+1)``, ``lo <= a < hi``), and
    ``h_lo = ln(c / r_a)``, ``h_hi = ln(r_(a+1) / c)``, both as ``log1p`` of the relative
    gap (the ratios are close to one; ``interval_widths`` explains the rounding).

    One comparison of all cuts with all nodes under the static ``(C, N)`` mask of the range
    nodes, and a count: the number ``k`` of range nodes at or below ``c`` gives
    ``a = lo + k - 1``. The comparison ``r_n <= c`` selects, at a tie ``c = r_a``, the
    interval above the node, with a zero-width lower part (the identity step exactly), so
    the value and the derivative at a tie are the limits from above (the convention of the
    upper one-sided derivative of ``controls.Layers``). Both nodes of a jump pair have one
    radius and pass the comparison together, so a pair interval is never selected
    inside the range. The count is clipped to ``[1, hi - lo]``: for a cut radius outside
    ``[r_lo, r_hi)`` (a traced value that the caller did not keep inside its range) the
    index stays on the first or the last interval of the range, whose widths are
    nonzero (``NodeStructure.__post_init__``), and every value stays finite. No search,
    no loop, no data-dependent shape; the count has no derivative, and the derivative
    with respect to ``c`` and ``radius`` passes the two logarithms only.
    """
    r = jnp.asarray(radius, jnp.float64)
    c = jnp.asarray(cut_radius, jnp.float64)
    lo, hi = structure.cut_lo, structure.cut_hi
    count = jnp.sum(structure.cut_mask & (r[None, :] <= c[:, None]), axis=1)
    count = jnp.clip(count, 1, hi - lo)
    a = lo + count - 1
    ra, rb = r[a], r[a + 1]
    return a, jnp.log1p((c - ra) / ra), jnp.log1p((rb - c) / c)


def interval_material(eta, rho0, rule: str) -> tuple[jax.Array, jax.Array]:
    """Interval viscosity and density, ``(N - 1,)`` each.

    ``rule="interval_mean"`` (stokesjx, eq. 12.1): ``etabar_i = (eta_(i-1) + eta_i) / 2``
    and the same for ``rho0``. ``rule="upper_node"`` (CODE-A, driver.f:362-363, test
    only): the values of the upper node ``i``.
    """
    if rule == "interval_mean":
        return 0.5 * (eta[1:] + eta[:-1]), 0.5 * (rho0[1:] + rho0[:-1])
    if rule == "upper_node":
        return eta[1:], rho0[1:]
    raise ValueError(f"unknown material rule {rule!r}")


def kappa_widths(radius, rho0, structure: NodeStructure, rule: str) -> jax.Array:
    """The TALA term ``kappa_h_i`` of every interval, ``(N - 1,)``, zero at a jump pair.

    ``kappa = d ln rho0 / d ln r`` (eq. 4.3) enters the interval exponent as
    ``K(1) kappa_h`` with ``kappa_h = kappa h`` (eq. 12.2). The function never forms
    ``kappa`` itself, which is undefined at a pair (``h = 0``; CONVENTIONS 4).

    Parameters
    ----------
    radius, rho0
        ``(N,)``: node radii in m and reference densities in kg/m^3 (positive).
    structure
        Static ``NodeStructure``: its ``interval_mask`` is 0 at a pair, 1 elsewhere.
    rule
        ``"log_slope"`` (stokesjx, eq. 12.2): ``kappa_h_i = ln(rho0_i / rho0_(i-1))``,
        exact for a power-law ``rho0 ~ r^kappa`` inside the interval. ``"legacy_one_sided"``
        (CODE-A, driver.f:429-430, test only): the one-sided slope at the upper node
        times the width, ``(r_i / rho0_i) (rho0_i - rho0_(i-1)) / (r_i - r_(i-1)) h_i``.
        It divides by the radial gap, so ``propagate.static_plan`` refuses it on a
        structure with pairs.

    Returns
    -------
    float64 array ``(N - 1,)``, dimensionless, times the static interval mask. At a pair
    the step is the closed form of eq. 12.3 (``anelastic_pair_terms``), so the masked
    value is an exact zero and the Pade pair of that interval stays the identity. The
    logarithm of a ratio of two positive numbers is finite at a pair, so the product
    with the zero of the mask has a zero derivative in both modes.
    """
    r = jnp.asarray(radius, jnp.float64)
    rho = jnp.asarray(rho0, jnp.float64)
    mask = structure.interval_mask
    if rule == "log_slope":
        return jnp.log(rho[1:] / rho[:-1]) * mask
    if rule == "legacy_one_sided":
        h = jnp.log1p((r[1:] - r[:-1]) / r[:-1])
        return (r[1:] / rho[1:]) * (rho[1:] - rho[:-1]) / (r[1:] - r[:-1]) * h * mask
    raise ValueError(f"unknown kappa rule {rule!r}")


def anelastic_pair_terms(degrees, rho0, structure: NodeStructure) -> tuple[jax.Array, jax.Array]:
    """The TALA corrections of the step and the inverse step at the jump pairs, in the
    balanced coordinates, batch-last ``(6, 6, N - 1, L)`` each (eq. 12.3).

    At a pair ``h = 0``, and the step is the ``h -> 0`` limit of
    ``expm(B)`` with ``B = t Kb``, ``Kb = S_l K(1) S_l^-1`` and
    ``t = ln(rho0_+ / rho0_-)`` (the anelastic rule: ``rho0 u_r`` is continuous). ``K(1)``
    is the outer product ``v e1^T`` with ``e1^T v = -1``, so ``Kb^2 = -Kb`` and

        expm(t Kb) = I + Kb (1 - exp(-t)) = I + tau Kb,    tau = 1 - rho0_- / rho0_+,

    and the inverse step is ``I + tau' Kb`` with ``tau' = 1 - rho0_+ / rho0_-``. The Pade
    pair of a pair interval is the identity exactly (its exponent is zero), so the step
    is that identity plus the term returned here; on an ordinary interval the term is
    zero by the static mask. No division by a width, no ``where``, exact for any density
    ratio across the pair. The viscosity enters afterwards through the similarity
    ``K(e) = D K(1) D^-1`` (eq. 4.5) with the interval mean of the pair, as in the
    oracle.

    Parameters
    ----------
    degrees
        Static degrees, length ``L``.
    rho0
        ``(N,)``, kg/m^3, positive.
    structure
        Static ``NodeStructure``.

    Returns
    -------
    ``(dE, dE_inv)``, float64, ``(6, 6, N - 1, L)``: ``(1 - m_i) tau_i Kb_l`` and
    ``(1 - m_i) tau'_i Kb_l`` with the static interval mask ``m_i``. The caller adds them
    only when the structure has a pair (a static condition).
    """
    rho = jnp.asarray(rho0, jnp.float64)
    _, _, Kb = _static_exponent_parts(degrees, True)            # (6, 6, L), K has no density entry
    pair = (1.0 - structure.interval_mask)[None, None, :, None]
    tau = (1.0 - rho[:-1] / rho[1:])[None, None, :, None]
    tau_inv = (1.0 - rho[1:] / rho[:-1])[None, None, :, None]
    K = Kb[:, :, None, :]
    return pair * tau * K, pair * tau_inv * K


def jump_vectors(radius, gravity, eta_ref, units: Units) -> jax.Array:
    """Jump ``y(r_j+) - y(r_j-)`` of a unit sheet (1 kg/m^2) at every node, in the
    ``y`` coordinates of eq. 3.1 with ``eta_s = eta_ref``, shape ``(N, 6)`` (eq. 7.1).

    ``[y3] = r g / eta_ref``: the radial stress below the sheet carries its weight.
    ``[y6] = -4 pi G r^2 rho_s / eta_ref``: the potential gradient jumps by the sheet's
    attraction. All other components are continuous. Units m/s per kg/m^2.
    """
    r = jnp.asarray(radius, jnp.float64)
    z = jnp.zeros_like(r)
    y3 = r * gravity / eta_ref
    y6 = -4.0 * units.pi * units.G * r ** 2 * units.density_scale / eta_ref
    return jnp.stack([z, z, y3, z, z, y6], axis=-1)
