"""Fields from kernels: the radial quadrature, the four fields of a density anomaly,
and the free-air gravity anomaly and the admittance as post-processing (ruling 4).

A field coefficient is the radial integral of a kernel times the density anomaly
(eq. 7.1, 12.4):

    F_o[l, m] = sum_j K_o[l, j] w_j drho[j, l, m]

with the kernels ``K_o`` of ``stokes_kernels`` (``o`` over the geoid, the surface
topography, the CMB topography and the surface velocity), the trapezoid weights
``w_j`` of the node table (``quadrature_weights``) and the density coefficients
``drho[j, l, m]`` in kg/m^3. Units: a kernel in m (or m/s) per kg/m^2, times a weight in
m, times a density in kg/m^3, gives m (or m/s).

Every coefficient array follows the layout rule of ``stokesjx.sh`` (the degree and
order axes last, the ``C/S`` axis at ``-3`` for a real convention, batch axes before
them, the radial axis first for a density). The radial integral does not mix any
other axis, because a kernel is a response per coefficient, so ``apply`` does not read
the convention: a field has the convention of the density.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import sh
from .kernels import Kernels, StokesConfig
from .profile import RadialProfile

#: The four outputs of a field record, in the order of the stacked kernels.
FIELD_NAMES = ("geoid", "surface_topography", "cmb_topography", "surface_velocity")


def quadrature_weights(profile: RadialProfile) -> jax.Array:
    """Trapezoid weights of the node table, ``(N,)``, m (eq. 12.4; audit row 4).

    ``w_j = (Delta_j + Delta_(j+1)) / 2`` with the interval widths
    ``Delta_i = (r_i - r_(i-1)) m_i`` (static mask ``m_i``: 0 at a jump pair, 1
    elsewhere) and ``Delta_0 = Delta_N = 0``. The zero-width interval of a pair adds
    nothing, and each node of a pair gets the half interval on its own side, so the
    density just below and just above a jump each count on their side. The weights sum
    to ``R - r_c`` and integrate a linear function exactly. They follow a traced radius
    (a movable boundary), and the static mask keeps the width of a pair an exact zero
    under every transformation, as in ``system.interval_widths``.
    """
    r = jnp.asarray(profile.radius, jnp.float64)
    delta = (r[1:] - r[:-1]) * profile.structure.interval_mask
    zero = jnp.zeros(1, dtype=jnp.float64)
    return 0.5 * (jnp.concatenate([delta, zero]) + jnp.concatenate([zero, delta]))


@dataclass(frozen=True, eq=False)
class Fields:
    """The four fields of a density anomaly. A JAX pytree: the arrays are leaves,
    ``degrees`` is static. Equality is identity (``eq=False``), as for ``Kernels``.

    Attributes
    ----------
    degrees
        ``Kernels.degrees``: the degree rows that are not zero by construction. The rows
        ``l = 0`` and, where degree 1 is refused, ``l = 1`` are exact zeros.
    geoid
        ``(*batch, [2,] lmax+1, lmax+1)``, m: geoid height coefficients.
    surface_topography
        Same shape, m, outward positive.
    cmb_topography
        Same shape, m, outward positive.
    surface_velocity
        Same shape, m/s: the coefficients whose degree power is the mean square of the
        horizontal surface speed (eq. 9.4; zero under a no-slip surface).

    Each field has the convention and the batch axes of the density.
    """

    degrees: tuple[int, ...]
    geoid: jax.Array
    surface_topography: jax.Array
    cmb_topography: jax.Array
    surface_velocity: jax.Array


def _fields_flatten(f: Fields):
    return tuple(getattr(f, n) for n in FIELD_NAMES), f.degrees


def _fields_unflatten(degrees, children) -> Fields:
    # No validation: JAX unflattens with tracers and placeholders.
    f = object.__new__(Fields)
    object.__setattr__(f, "degrees", degrees)
    for n, v in zip(FIELD_NAMES, children):
        object.__setattr__(f, n, v)
    return f


jax.tree_util.register_pytree_node(Fields, _fields_flatten, _fields_unflatten)


def _check_density(kernels: Kernels, density) -> tuple[int, int]:
    """Shape checks of a density under the layout rule; returns ``(d0, lmax)``."""
    degrees = kernels.degrees
    d0, lmax = degrees[0], degrees[-1]
    if tuple(degrees) != tuple(range(d0, lmax + 1)):
        raise ValueError(f"the kernel degrees must be contiguous, got {degrees}")
    n_nodes = jnp.shape(kernels.geoid)[-1]
    shape = jnp.shape(density)
    if len(shape) < 3:
        raise ValueError(f"a density has the radial axis first and the degree and order axes last, "
                         f"(N, ..., Ld+1, Ld+1); got the shape {shape}")
    if shape[0] != n_nodes:
        raise ValueError(f"the density has {shape[0]} entries on its first (radial) axis, the kernels "
                         f"{n_nodes} nodes; expected the shape (N, ..., Ld+1, Ld+1) with N = {n_nodes}")
    if shape[-1] != shape[-2]:
        raise ValueError(f"the last two axes of a density are the degree and the order axis, of equal "
                         f"length Ld + 1; got the shape {shape} (the layout [j, l, m, C/S] is converted "
                         "with np.moveaxis(c, -1, -3))")
    if shape[-1] - 1 < lmax:
        raise ValueError(f"the density holds degrees up to {shape[-1] - 1}, the kernels up to {lmax}; "
                         f"expected Ld >= {lmax}")
    return d0, lmax


def radial_integral(kernels: Kernels, density, weights) -> Fields:
    """The four fields of a density with explicit radial weights.

    Parameters
    ----------
    kernels
        ``Kernels`` (each kernel ``(L, N)``, contiguous degrees ``d0 .. lmax``).
    density
        ``(N, *batch, Ld+1, Ld+1)``, kg/m^3, CMB first, ``Ld >= lmax``: complex
        ``complex_4pi`` coefficients, or real coefficients with the ``C/S`` axis at
        ``-3`` (``*batch`` ends in 2). The convention is not read: the radial integral
        does not mix the ``C/S``, degree or order axes.
    weights
        ``(N,)``, m: the radial quadrature weights (``quadrature_weights``, or CODE-A's
        equal weights in the tests).

    Returns
    -------
    ``Fields`` with each array ``(*batch, lmax+1, lmax+1)``.

    Method: the weights go into the stacked kernels first, ``Kw = K w`` ``(4, L, N)``, the
    density is cut statically to the kernel degrees and to ``lmax + 1`` orders,
    ``rows = density[..., d0:lmax+1, :lmax+1]``, and one einsum
    ``"olj,j...lm->o...lm"`` (one ``dot_general``) gives the fields; the rows ``l < d0``
    are a concatenation of exact zeros on axis ``-2``. A three-operand einsum with the
    weights as an operand lowers to two ``dot_general``, so the weights are multiplied in
    first. Entries with ``m > l`` are the kernel times the input entries there, zero for
    a valid coefficient array. Caveat: at ``Ld = 1`` a density in the layout
    ``[j, l, m, C/S]`` has the same shape as ``(N, 2, 2, 2)`` and cannot be told apart.
    """
    d0, lmax = _check_density(kernels, density)
    w = jnp.asarray(weights, jnp.float64)
    K = jnp.stack([jnp.asarray(getattr(kernels, n)) for n in FIELD_NAMES])        # (4, L, N)
    Kw = K * w
    rows = jnp.asarray(density)[..., d0: lmax + 1, : lmax + 1]
    F = jnp.einsum("olj,j...lm->o...lm", Kw, rows)
    if d0 > 0:
        F = jnp.concatenate([jnp.zeros(F.shape[:-2] + (d0, lmax + 1), F.dtype), F], axis=-2)
    return Fields(kernels.degrees, *(F[k] for k in range(4)))


def apply(kernels: Kernels, density, profile: RadialProfile) -> Fields:
    """The four fields of a density anomaly with the trapezoid weights of the profile.

    ``radial_integral(kernels, density, quadrature_weights(profile))``; see there for
    the layouts. ``profile`` is the profile of the kernels: its radius sets the weights,
    so a derivative with respect to a radius reaches the weights too. When both radii
    are concrete, ``kernels.radius`` must equal ``profile.radius`` (``ValueError``): a
    profile with the same node count and other radii would give wrong weights.

    The density rule (NOTES/chunks/c4-controls-derivatives/01-plan.md D11). The density
    is given on the node table of ``profile``. For a ``controls.Layers`` with movable
    boundaries the density of a field is given on the INPUT table of the ``Layers`` and
    does not move with a boundary; that field is ``Layers.fields(kernels, density)``, and
    this function refuses the profile of a movable ``Layers`` (its structure has
    ``limits``) with ``ValueError``, so that an anomaly cannot be attached to the moving
    nodes by mistake. A ``Layers`` without a movable boundary is accepted here: this is
    then the trapezoid rule of the construction table, for a density that the caller
    gives on that table (for example the output of ``Layers.sheet_mass``); a density on
    the input table goes through ``Layers.fields`` also then. ``radial_integral`` with
    explicit weights does not check: a caller who passes weights takes the
    responsibility.

    Example::

        K = stokes_kernels(profile, config)
        F = apply(K, density, profile)       # density (N, 2, L+1, L+1), real_4pi
        F.geoid                              # (2, lmax+1, lmax+1), m

    Compile time: pass the density to a jitted function as an argument. A density that
    the function closes over is embedded as a constant of the program (42 MB for 257
    nodes and ``L = 100``), and XLA then needs about 0.3 s more to compile the misfit and
    its gradient; as an argument the field map adds 0.01 to 0.03 s
    (NOTES/chunks/c3-flags-fields/impl-evidence/density_constant.out).
    """
    if profile.structure.limits is not None:
        raise ValueError("the density of a movable Layers profile is given on the input table and does not "
                         "move with a boundary; use Layers.fields(kernels, density)")
    k_r, p_r = kernels.radius, profile.radius
    if not isinstance(k_r, jax.core.Tracer) and not isinstance(p_r, jax.core.Tracer):
        if np.shape(k_r) != np.shape(p_r) or not np.array_equal(np.asarray(k_r), np.asarray(p_r)):
            raise ValueError("the kernels were computed on other radii than the profile's: kernels.radius "
                             "differs from profile.radius, so the trapezoid weights would not belong to the "
                             "kernels; pass the profile of the kernels")
    return radial_integral(kernels, density, quadrature_weights(profile))


def spectral_misfit(field, observed, weights, convention: str) -> jax.Array:
    """A weighted spectral misfit of a field, ``sum_batch sum_l w_l S_dd(l)`` with
    ``d = field - observed`` (NOTES/chunks/c4-controls-derivatives/01-plan.md D8).

    ``S_dd(l)`` is the degree power of ``sh.degree_power(d, d, convention)``: in
    ``real_4pi`` (and equally ``complex_4pi``) ``sum_m (C_lm^2 + S_lm^2)`` of the
    difference, the mean square over the sphere per degree; ``real_orthonormal`` gives
    ``4 pi`` times it. For data with independent errors of standard deviation
    ``sigma_l`` per coefficient, ``weights = 1 / (2 sigma_l^2)`` makes the misfit the
    negative Gaussian log-likelihood up to a constant; zeros leave degrees out (for
    example ``l < 2``). The sign and the prior are the caller's (ruling of 2026-11-10:
    the package does not choose a sampler).

    Parameters
    ----------
    field, observed
        Coefficient arrays of one shape under the layout rule of ``convention``
        (CONVENTIONS 15), batch axes allowed (summed).
    weights
        ``(L + 1,)`` float64, one weight per degree of the arrays; another length
        raises ``ValueError``.
    convention
        ``"real_4pi"``, ``"complex_4pi"`` or ``"real_orthonormal"``.

    Returns
    -------
    A scalar. No square root is taken, so the derivative is finite at ``d = 0``: with
    respect to ``field`` it is ``2 w_l d`` in ``real_4pi``.
    """
    d = jnp.asarray(field) - jnp.asarray(observed)
    S = sh.degree_power(d, d, convention)                                 # (..., L+1)
    w = jnp.asarray(weights, dtype=jnp.float64)
    if w.shape != (S.shape[-1],):
        raise ValueError(f"weights must hold one weight per degree, shape ({S.shape[-1]},), got {w.shape}")
    return jnp.sum(S * w)


def surface_gravity(profile: RadialProfile, config: StokesConfig) -> jax.Array:
    """The background gravity at the surface, m/s^2 (scalar): the hydrostatic ``g(R)``
    of the profile (eq. 6.1) or the uniform value (eq. 6.2). It is the ``g(R)`` that
    converts the surface potential into the geoid kernel (eq. 9.1), so with it the
    gravity anomaly is ``(l - 1) phi_l / R`` for every background (``gravity_anomaly``)."""
    from .propagate import background_gravity
    return background_gravity(profile, config.formulation, config.units)[-1]


def gravity_anomaly(geoid, g_surface, radius, degrees=None) -> jax.Array:
    """The free-air gravity anomaly from geoid coefficients or a geoid kernel, m/s^2:

        Delta g_l = f_l g(R) N_l / R,    f_l = l - 1 for l >= 1,  f_0 = 0

    (1 mGal = 1e-5 m/s^2). Because the same ``g(R)`` divides the potential in eq. 9.1,
    ``Delta g_l = f_l phi_l / R`` for every background. At ``l = 1`` the factor is zero:
    in the centre-of-mass frame the degree-1 geoid is zero, and a degree-1 geoid term
    (raw frame) is the sign of an origin away from the centre of mass, for which the
    factor ``l - 1 = 0`` gives no free-air anomaly. At ``l = 0`` the factor is set to zero: the package has no
    degree-0 response (a density anomaly of degree 0 changes the mass of the Earth), and
    a degree-0 entry from elsewhere (a mean geoid height, a reference offset) is not a
    mass anomaly of the model; ``(l - 1) = -1`` would turn it into a gravity value with
    no meaning. This replaces CODE-A's free-air kernel, which uses ``G Emass / R^2`` here
    and 10 m/s^2 everywhere else (audit row 2).

    Parameters
    ----------
    geoid
        With ``degrees=None``: a coefficient array under the layout rule (for example
        ``Fields.geoid``), any convention and batch axes; the degree axis is ``-2`` with
        the degrees ``0 .. shape[-2] - 1``. With ``degrees`` given: a kernel array
        ``(L, ...)`` (for example ``Kernels.geoid``, ``(L, N)``) with the degree axis first.
    g_surface
        ``g(R)`` in m/s^2 (``surface_gravity``), scalar, can be traced.
    radius
        ``R`` in m, scalar, can be traced.
    degrees
        ``None``, or the static degrees of the first axis (normally ``Kernels.degrees``);
        ``len(degrees)`` must equal ``geoid.shape[0]``.

    Returns
    -------
    An array of the shape (and convention) of ``geoid``.
    """
    geoid = jnp.asarray(geoid)
    if degrees is None:
        if geoid.ndim < 2 or geoid.shape[-1] != geoid.shape[-2]:
            raise ValueError(f"a coefficient array has the degree and order axes last, of equal length L + 1; got "
                             f"the shape {geoid.shape} (a kernel array (L, N) needs degrees=Kernels.degrees)")
        l = np.arange(geoid.shape[-2], dtype=np.float64)
        f = np.where(l >= 1, l - 1.0, 0.0)[:, None]                      # (L+1, 1): over the order axis
    else:
        degrees = tuple(int(d) for d in degrees)
        if len(degrees) != geoid.shape[0]:
            raise ValueError(f"{len(degrees)} degrees for a first axis of length {geoid.shape[0]}")
        l = np.asarray(degrees, dtype=np.float64)
        f = np.where(l >= 1, l - 1.0, 0.0).reshape((-1,) + (1,) * (geoid.ndim - 1))
    return f * (g_surface / radius) * geoid


def admittance(gravity, topography, convention: str) -> jax.Array:
    """The spectral admittance per degree, ``Z_l = S_gh(l) / S_hh(l)``, ``(..., L+1)``.

    ``S_xy`` is the degree cross power of ``sh.degree_power``. Both inputs follow the
    layout rule for ``convention`` and have the same shape (for example the gravity
    anomaly of ``gravity_anomaly`` and ``Fields.surface_topography``); the batch axes are
    kept and the last axis is the degree. The two real conventions differ by ``4 pi`` in
    both powers, and the complex power equals the ``real_4pi`` power, so ``Z_l`` does not
    depend on the convention. The convention is an explicit argument: at small ``L``
    the shape alone does not tell a real array with a ``C/S`` axis from a complex one.

    Units: ``(m/s^2) / m = s^-2``; 1 mGal/km = 1e-8 s^-2. Where ``S_hh(l) = 0`` (degree 0,
    a refused degree 1) the result is 0, through the double ``where`` of CONVENTIONS 2,
    so the gradient is finite there. This is ruling 4: the admittance is post-processing
    of field coefficients, not a kernel (audit row 10).
    """
    s_gh = sh.degree_power(gravity, topography, convention)
    s_hh = sh.degree_power(topography, topography, convention)
    ok = s_hh != 0
    return jnp.where(ok, s_gh / jnp.where(ok, s_hh, 1.0), 0.0)
