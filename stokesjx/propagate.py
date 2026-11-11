"""Low layer: the two-sweep propagator solver.

For every degree ``l`` and every source node ``j`` the solver needs the state of the
mantle under a unit sheet of density at ``r_j``. The state below the sheet satisfies
the three CMB conditions, the state above it the three surface conditions, and the two
differ by the jump of the sheet (eq. 7.1). The method (NOTES/docs/equations.md
section 16; PLAN 3.2):

1. Interval steps. All interval exponentials of all degrees, and their inverses, in
   one batched Pade call (``linalg.expm_pade_pair``) outside the sweeps, in the sweep
   coordinates ``z = S_l y`` (``system``). The step of interval ``i`` is
   ``Z_i = D_i expm(B_i) D_i^-1``.
2. Upward sweep. A basis of the three CMB solutions is carried from the CMB to the
   surface with ``lax.scan``; after every interval it is replaced by an orthonormal
   basis of its span (``linalg.gram_schmidt3``). Downward sweep: the same from the
   surface with the inverse steps (``lax.scan(..., reverse=True)``).
3. Output rows. The kernels need four linear functionals of the surface state and one
   of the CMB state. Each sweep carries them as rows in the coordinates of its current
   basis, through the triangular factors: ``W_(i-1) = W_i R_i^-1``
   (``linalg.right_solve_upper3``). A state ``Q_j b`` above node ``j`` has the surface
   outputs ``W_j b``; a state ``Q_j a`` below node ``j`` has the CMB output ``v_j a``.
   This is the cumulative product of the triangular factors of PLAN 3.2 step 4, applied
   one step at a time. The rows shrink where the basis grows, and ``W_j b`` has the size
   of the physical kernel, so no log scale is needed.
4. Node solve. At every node, ``[-Q_up, Q_dn] (a, b) = S_l J_j`` (the jump in ``z``), for
   all degrees and nodes in one batched 6x6 solve with row equilibration.

Viscosity scale (NOTES/chunks/c2-forward-core/01-plan.md, D1). The state uses ``eta_s = eta_ref``, the
largest node viscosity, through ``lax.stop_gradient``. The kernels do not depend on the
scale in exact arithmetic (equations.md section 11), so the derivative at a fixed scale
is the total derivative up to rounding, and the ``max`` needs no derivative where two
nodes share the largest value. With this scale every interval viscosity is at most
one, and the measured accuracy does not degrade with the viscosity contrast (lids up
to 1e31 Pa s on a 1e20 Pa s mantle); a smaller scale (the geometric mean) loses digits
in proportion to ``eta_max / eta_ref``.

The jitted core counts its traces in a module-level counter (``trace_count``), for the
test of CONVENTIONS 2 that a call with new values of the same shapes does not trace
again.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import boundary, linalg, system
from .formulation import DegreeOneFrame, Formulation, Hydrostatic, Terms, Uniform, resolve_degree_one, terms
from .kernels import Kernels, StokesConfig
from .profile import NodeStructure, RadialProfile, mantle_mass
from .units import Units


@dataclass(frozen=True)
class Discretisation:
    """Test-only legacy switches of the low layer (PLAN section 5, last paragraph).

    Attributes
    ----------
    material
        ``"interval_mean"`` (stokesjx, eq. 12.1) or ``"upper_node"``: the viscosity and
        the density of an interval are those of its upper node, as CODE-A's
        ``aconstruct`` (driver.f:362-363).
    degree_one_geoid_zero
        CODE-A's degree-1 frame: the raw solve with the degree-1 geoid set to zero
        (driver.f:282-284). Needs the raw frame to be offered (a uniform background).
    kappa_rule
        The TALA term of an interval (``system.kappa_widths``): ``"log_slope"``
        (stokesjx, eq. 12.2) or ``"legacy_one_sided"``, CODE-A's one-sided slope at the
        upper node times the width (``kappaindex = 1``, driver.f:429-430). The legacy
        rule divides by the radial gap of an interval, so ``static_plan`` refuses it on
        a structure with jump pairs.

    ``kernels.stokes_kernels`` passes no ``Discretisation``, so the high layer cannot
    reach any of these switches.
    """

    material: Literal["interval_mean", "upper_node"] = "interval_mean"
    degree_one_geoid_zero: bool = False
    kappa_rule: Literal["log_slope", "legacy_one_sided"] = "log_slope"

    def __post_init__(self):
        if self.material not in ("interval_mean", "upper_node"):
            raise ValueError(f"unknown material rule {self.material!r}")
        if self.kappa_rule not in ("log_slope", "legacy_one_sided"):
            raise ValueError(f"unknown kappa rule {self.kappa_rule!r}")


@dataclass(frozen=True)
class StaticPlan:
    """Everything static of one solve, built on the host. Frozen and hashable: it is a
    static argument of the jitted core.

    Attributes
    ----------
    degrees
        The degrees of the kernel rows.
    frame
        The resolved frame at ``l = 1``; ``None`` when degree 1 is absent.
    potential_coefficient
        ``a_l`` of the surface potential row per degree (``boundary.surface_potential_coefficient``).
    geoid_mask
        1.0 or 0.0 per degree (0 at ``l = 1`` in the CM frame and in CODE-A's frame).
    pade
        The Pade plan of the interval exponentials (``linalg.pade_plan``).
    pairs
        Interval indices of the jump pairs (from the structure).
    material
        Interval material rule (``Discretisation.material``).
    node_solver
        Method of the batched node solve (``Numerics.node_solver``).
    rho_max
        ``Numerics.rho0_max / rho_s``, the static density limit of the Pade bound;
        0 under NO_SELF_GRAVITY, where the exponent has no density entry.
    terms
        The term record of the formulation (``formulation.terms``): gravity level,
        TALA and the two boundary classes.
    kappa_rule
        The TALA rule of an interval (``Discretisation.kappa_rule``).
    kappa_h_max
        ``Numerics.kappa_h_max`` under TALA, else 0: the static limit of ``|kappa_h|`` on
        an ordinary interval in the Pade bound.
    """

    degrees: tuple[int, ...]
    frame: DegreeOneFrame | None
    potential_coefficient: tuple[float, ...]
    geoid_mask: tuple[float, ...]
    pade: linalg.PadePlan
    pairs: tuple[int, ...]
    material: str
    node_solver: str
    rho_max: float
    terms: Terms
    kappa_rule: str
    kappa_h_max: float


@functools.lru_cache(maxsize=256)
def static_plan(structure: NodeStructure, config: StokesConfig,
                discretisation: Discretisation = Discretisation()) -> StaticPlan:
    """Build the static plan of a solve (cached; every argument is hashable).

    1. Frame at ``l = 1``: ``formulation.resolve_degree_one`` of the requested frame;
       with ``degree_one_geoid_zero`` the request is CODE-A's frame, which needs raw
       to be offered. A refused frame removes degree 1 from the degrees.
    2. ``a_l = l + 1``, and ``-1`` at ``l = 1`` in the CM frame (eq. 8.2, 10.5).
    3. The geoid mask (0 at ``l = 1`` in the CM frame and in CODE-A's frame).
    4. The term record of the flags (``formulation.terms``).
    5. The Pade plan from the column-wise norm bound
       (``linalg.balanced_norm_bound_intervals``) of the construction widths, the
       density limit ``Numerics.rho0_max / rho_s`` (0 under NO_SELF_GRAVITY) and, under
       TALA, the limit ``Numerics.kappa_h_max`` of ``|kappa_h|`` on every ordinary
       interval (zero at a pair, whose step is the closed form of eq. 12.3;
       CONVENTIONS 2). With ``structure.limits`` (a table with movable boundaries) the
       widths are the widened limits (``NodeStructure.width_bounds``).

    Raises ``ValueError`` for a frame that the static rule does not offer, for
    CODE-A's one-sided TALA rule on a structure with jump pairs (it divides by the
    radial gap of an interval), and for ``structure.limits`` beyond the static limits
    of ``Numerics``: ``limits.rho0_max > rho0_max`` under FULL, or ``limits.kappa_h >
    kappa_h_max`` on an ordinary interval under TALA. These two checks replace, for the
    moved nodes, the value checks that ``check_profile`` cannot run on a traced radius
    and ``rho0``.
    """
    form = config.formulation
    request = config.degree_one
    if discretisation.degree_one_geoid_zero:
        if request not in (DegreeOneFrame.DEFAULT, DegreeOneFrame.RAW):
            raise ValueError("degree_one_geoid_zero is CODE-A's raw frame; it cannot be combined with "
                             f"the request {request.value!r}")
        request = DegreeOneFrame.LEGACY_GEOID_ZERO
    frame = resolve_degree_one(form, structure.consistent_core, request)
    degrees = tuple(range(1 if frame is not None else 2, config.lmax + 1))
    if not degrees:
        raise ValueError("no degree to solve: degree 1 is refused and lmax = 1")
    frame_at_one = frame if degrees[0] == 1 else None
    a_l = boundary.surface_potential_coefficient(degrees, frame_at_one)
    mask = np.ones(len(degrees))
    if frame_at_one in (DegreeOneFrame.CENTRE_OF_MASS, DegreeOneFrame.LEGACY_GEOID_ZERO):
        mask[0] = 0.0
    term_record = terms(form)
    rho_max = config.numerics.rho0_max / config.units.density_scale if term_record.self_gravity else 0.0
    kappa_h_max = config.numerics.kappa_h_max if term_record.anelastic else 0.0
    if term_record.anelastic and discretisation.kappa_rule == "legacy_one_sided" and structure.pairs:
        raise ValueError("the one-sided kappa rule of CODE-A divides by the radial gap of an interval and is "
                         f"undefined at the jump pairs {structure.pairs}; use kappa_rule='log_slope'")
    limits = structure.limits
    if limits is not None:
        # A table whose nodes move under a trace (controls.Layers): the value checks of
        # check_profile cannot run on a traced radius or rho0, so the static limits of
        # every admissible position replace them here, at every call (the plan is
        # cached, the exception is not).
        if term_record.self_gravity and limits.rho0_max > config.numerics.rho0_max:
            raise ValueError(f"the movable boundaries of this table give rho0 up to {limits.rho0_max} kg/m^3, "
                             f"above Numerics.rho0_max = {config.numerics.rho0_max} kg/m^3, the limit of the "
                             "static Pade plan; raise Numerics.rho0_max")
        if term_record.anelastic:
            k_lim = np.asarray(limits.kappa_h, dtype=np.float64) * structure.interval_mask
            if k_lim.size and np.max(k_lim) > config.numerics.kappa_h_max:
                i = int(np.argmax(k_lim))
                raise ValueError(f"|kappa_h| = |ln(rho0_i / rho0_(i-1))| on the interval {i + 1} (nodes {i} and "
                                 f"{i + 1}) reaches {np.max(k_lim):.4f} at an admissible position of a movable "
                                 f"boundary, above Numerics.kappa_h_max = {config.numerics.kappa_h_max}, the "
                                 "static limit of the Pade plan under TALA; narrow the movable range that "
                                 "holds this interval, or raise Numerics.kappa_h_max")
    bound = linalg.balanced_norm_bound_intervals(degrees[-1], structure.width_bounds,
                                                 kappa_h_max * structure.interval_mask, rho_max)
    return StaticPlan(degrees=degrees, frame=frame_at_one,
                      potential_coefficient=tuple(float(a) for a in a_l),
                      geoid_mask=tuple(float(m) for m in mask), pade=linalg.pade_plan(bound),
                      pairs=structure.pairs, material=discretisation.material,
                      node_solver=config.numerics.node_solver, rho_max=rho_max, terms=term_record,
                      kappa_rule=discretisation.kappa_rule, kappa_h_max=kappa_h_max)


def reference_viscosity(eta, rule) -> jax.Array:
    """The viscosity scale ``eta_ref`` of the state vector, Pa s
    (NOTES/chunks/c2-forward-core/01-plan.md, D1).

    ``rule``: ``"largest"`` (the solver's rule, ``stop_gradient(max eta)``),
    ``"geometric_mean"`` (``stop_gradient(exp(mean(log eta)))``, for the measurement
    of D1 only) or a number in Pa s (a fixed scale, for the comparison with the oracle
    in its scale and for the invariance tests). The kernels do not depend on the scale
    in exact arithmetic; ``stop_gradient`` keeps any derivative out of it.
    """
    if rule == "largest":
        return lax.stop_gradient(jnp.max(eta))
    if rule == "geometric_mean":
        return lax.stop_gradient(jnp.exp(jnp.mean(jnp.log(eta))))
    if isinstance(rule, (int, float)) and not isinstance(rule, bool) and rule > 0:
        return jnp.asarray(float(rule), dtype=jnp.float64)
    raise ValueError(f"unknown reference viscosity rule {rule!r}")


def background_gravity(profile: RadialProfile, formulation: Formulation, units: Units) -> jax.Array:
    """Background gravity at the nodes, m/s^2, ``(N,)``: hydrostatic from ``rho0`` and the
    core mass (eq. 6.1) or uniform (eq. 6.2). Evaluated inside the differentiated
    function, so it follows ``rho0``, the radius and the core under ``jax.grad``. The
    uniform value is the static ``Uniform.value``, or for ``Uniform(None)`` the profile
    leaf ``uniform_gravity``, which can be traced (a control ``g0``)."""
    if isinstance(formulation.background, Hydrostatic):
        return profile.hydrostatic_gravity(units)
    if isinstance(formulation.background, Uniform):
        # A number in the flag is static; Uniform(None) reads the traced profile leaf
        # (check_profile has checked on the host that exactly one of the two is set).
        value = formulation.background.value
        if value is None:
            value = profile.uniform_gravity
        return jnp.full(profile.radius.shape, value, dtype=jnp.float64)
    raise TypeError(formulation.background)


def interval_steps(profile: RadialProfile, eta_ref, plan: StaticPlan,
                   config: StokesConfig) -> tuple[jax.Array, jax.Array]:
    """The interval steps of both sweeps in ``z`` coordinates, ``(N - 1, L, 6, 6)`` each.

    ``Z_up[i - 1] = D_i E_i D_i^-1`` carries ``z(r_(i-1))`` to ``z(r_i)``, and
    ``Z_down[i - 1] = D_i E_i^-1 D_i^-1`` carries it back, with ``E_i`` and ``E_i^-1``
    the Pade pair of the balanced exponent (``system.balanced_exponent``). The steps do
    not depend on the gravity (F1): a derivative with respect to ``g`` never passes
    through the exponential. At a pair the width is a static zero and the exponent is
    zero, so the Pade pair is the identity exactly; incompressible, that is the pair
    step. Under TALA (``plan.terms.anelastic``) the exponent of an ordinary interval
    gains ``kappa_h K(1)`` (``system.kappa_widths``, eq. 12.2), and the step and the
    inverse step of a pair gain the closed-form terms of eq. 12.3
    (``system.anelastic_pair_terms``), added before the viscosity similarity, so the
    pair uses the interval-mean viscosity of its two nodes.
    """
    rho_s = config.units.density_scale
    structure = profile.structure
    h = system.interval_widths(profile.radius, structure)
    eta_bar, rho_bar = system.interval_material(profile.eta, profile.rho0, plan.material)
    kappa_h = None
    if plan.terms.anelastic:
        kappa_h = system.kappa_widths(profile.radius, profile.rho0, structure, plan.kappa_rule)
    B = system.balanced_exponent(plan.degrees, rho_bar / rho_s, h, kappa_h, self_gravity=plan.terms.self_gravity)
    E, E_inv = linalg.expm_pade_pair(B, plan.pade.order, plan.pade.squarings)
    if plan.terms.anelastic and structure.pairs:
        dE, dE_inv = system.anelastic_pair_terms(plan.degrees, profile.rho0, structure)
        E, E_inv = E + dE, E_inv + dE_inv
    e_rel = eta_bar / eta_ref
    to_scan = lambda Z: jnp.moveaxis(system.viscosity_similarity(Z, e_rel), (0, 1), (2, 3))
    return to_scan(E), to_scan(E_inv)


class SweepResult(NamedTuple):
    """Bases and carried rows of both sweeps at every node.

    Attributes
    ----------
    q_up
        ``(N, L, 6, 3)``: orthonormal basis (``z``) of the CMB solutions at each node.
    row_up
        ``(N, L, 1, 3)``: the CMB output row in the coordinates of ``q_up``.
    q_down
        ``(N, L, 6, 3)``: orthonormal basis of the surface solutions at each node.
    rows_down
        ``(N, L, 4, 3)``: the four surface output rows in the coordinates of ``q_down``.
    """

    q_up: jax.Array
    row_up: jax.Array
    q_down: jax.Array
    rows_down: jax.Array


def _sweep_body(carry, Z):
    """One interval of a sweep: carry the basis through the step, orthonormalise it,
    and carry the output rows through the triangular factor. Shapes: ``Q (L, 6, 3)``,
    ``W (L, k, 3)``, ``Z (L, 6, 6)``."""
    Q, W = carry
    Q, R = linalg.gram_schmidt3(Z @ Q)
    W = linalg.right_solve_upper3(W, R)
    return (Q, W), (Q, W)


def sweeps(Z_up, Z_down, start_up, start_down, row_up, rows_down) -> SweepResult:
    """The upward and the downward sweep (two ``lax.scan`` calls).

    Parameters
    ----------
    Z_up, Z_down
        ``(N - 1, L, 6, 6)``: the steps of ``interval_steps``.
    start_up, start_down
        ``(L, 6, 3)``: the CMB and the surface basis in ``z``.
    row_up, rows_down
        ``(L, 1, 6)`` and ``(L, 4, 6)``: the output rows on ``z`` at the CMB and at the
        surface.

    The upward scan runs over the intervals from the CMB; its output ``i`` is the basis
    at node ``i + 1``. The downward scan runs with ``reverse=True`` from the surface; its
    output ``i`` is the basis at node ``i``. So neither needs a flip.
    """
    Q0, _ = linalg.gram_schmidt3(start_up)
    QN, _ = linalg.gram_schmidt3(start_down)
    v0 = row_up @ Q0
    WN = rows_down @ QN
    _, (Qu, vu) = lax.scan(_sweep_body, (Q0, v0), Z_up)
    _, (Qd, Wd) = lax.scan(_sweep_body, (QN, WN), Z_down, reverse=True)
    return SweepResult(q_up=jnp.concatenate([Q0[None], Qu]), row_up=jnp.concatenate([v0[None], vu]),
                       q_down=jnp.concatenate([Qd, QN[None]]), rows_down=jnp.concatenate([Wd, WN[None]]))


def solve_nodes_and_outputs(sweep: SweepResult, jumps_z, plan: StaticPlan) -> tuple[jax.Array, jax.Array]:
    """The node systems and the outputs.

    At node ``j``: ``Q_dn b - Q_up a = S_l J_j`` (the state above the sheet minus the
    state below is the jump), so ``[-Q_up, Q_dn] (a, b) = S_l J_j``. The rows of every
    system and its right-hand side are scaled by ``1 / max_k |M_jk|`` (row
    equilibration, through ``stop_gradient``: the solution does not depend on it).

    Parameters
    ----------
    sweep
        ``SweepResult``.
    jumps_z
        ``(N, L, 6)``: the jumps in ``z``.
    plan
        The static plan (node solver).

    Returns
    -------
    ``(cmb, surface)``: ``(L, N)`` CMB topography and ``(4, L, N)`` surface outputs
    (geoid, surface topography, surface velocity, applied potential ``y5_a``).
    """
    M = jnp.concatenate([-sweep.q_up, sweep.q_down], axis=-1)
    w = lax.stop_gradient(1.0 / jnp.max(jnp.abs(M), axis=-1))
    x = linalg.solve_nodes(M * w[..., None], jumps_z * w, plan.node_solver)
    cmb = jnp.einsum("nlk,nlk->ln", sweep.row_up[:, :, 0, :], x[..., :3])
    surface = jnp.einsum("nlok,nlk->oln", sweep.rows_down, x[..., 3:])
    return cmb, surface


_TRACE_COUNT = 0


def trace_count() -> int:
    """Number of traces of the jitted core in this process (CONVENTIONS 2, test 7.7)."""
    return _TRACE_COUNT


def _core(profile: RadialProfile, plan: StaticPlan, config: StokesConfig, ref_rule) -> dict:
    """The forward map from the profile leaves to the kernel arrays (jitted below)."""
    global _TRACE_COUNT
    _TRACE_COUNT += 1
    units, form = config.units, config.formulation
    term_record = plan.terms
    rho_s, pi, G = units.density_scale, units.pi, units.G
    degrees = plan.degrees
    S = system.degree_scaling(degrees)                                   # (L, 6), static
    r, rho0 = profile.radius, profile.rho0
    R, rc = r[-1], r[0]

    # Scales and background.
    eta_ref = reference_viscosity(profile.eta, ref_rule)
    g = background_gravity(profile, form, units)
    g_R, g_c = g[-1], g[0]
    c_s = 4.0 * pi * G * R * rho_s / g_R
    c_c = 4.0 * pi * G * rc * rho_s / g_c

    # Interval steps of all degrees.
    Z_up, Z_down = interval_steps(profile, eta_ref, plan, config)

    # Boundary bases and output rows, converted to z = S y: a basis vector
    # maps by S, a row by S^-1. The boundary classes enter through
    # tangential_unknown with parameter=None: FreeSlip and NoSlip have no parameter. A
    # class with a traced parameter (a later partial slip) passes its profile leaf here
    # and nowhere else (formulation.BoundaryCondition).
    Yc = boundary.cmb_basis(degrees, c_c, profile.core_density / rho_s, terms=term_record, radius=rc,
                            eta_ref=eta_ref, parameter=None)
    Ys = boundary.surface_basis(degrees, c_s, np.asarray(plan.potential_coefficient), terms=term_record,
                                radius=R, eta_ref=eta_ref, parameter=None)
    ws = boundary.surface_functionals(degrees, R=R, g_R=g_R, rho_top=profile.rho_top,
                                      rho_w=profile.outer_density, rho_s=rho_s, eta_ref=eta_ref, c_s=c_s,
                                      geoid_mask=np.asarray(plan.geoid_mask), terms=term_record)
    wc = boundary.cmb_functional(r_c=rc, g_c=g_c, rho_cmb=rho0[0], rho_c=profile.core_density,
                                 rho_s=rho_s, eta_ref=eta_ref, terms=term_record)
    start_up = S[:, :, None] * Yc
    start_down = S[:, :, None] * Ys
    row_up = (wc[None, :] / S)[:, None, :]
    rows_down = ws / S[:, None, :]

    # The two sweeps.
    sweep = sweeps(Z_up, Z_down, start_up, start_down, row_up, rows_down)

    # Jumps in z and the node solve.
    J = system.jump_vectors(r, g, eta_ref, units)                        # (N, 6)
    cmb, surface = solve_nodes_and_outputs(sweep, J[:, None, :] * S[None], plan)

    out = dict(geoid=surface[0], surface_topography=surface[1], cmb_topography=cmb,
               surface_velocity=surface[2], degree_one_imbalance=None)
    if plan.frame is DegreeOneFrame.CENTRE_OF_MASS:
        # eq. 10.6: the uniform applied field A = phi_a(R) / R per unit sheet, with
        # phi_a = eta_ref y5_a / (rho_s R); the mass that it accelerates is the mantle
        # plus a core of density rho_c (the field acts on the core through rho_c phi).
        A = eta_ref * surface[3, 0] / (rho_s * R ** 2)
        m_eff = mantle_mass(r, rho0, units) + 4.0 * pi / 3.0 * rc ** 3 * profile.core_density
        out["degree_one_imbalance"] = m_eff * A / (4.0 * pi / 3.0 * g * r ** 2)
    elif plan.frame is not None:
        # The raw frame and CODE-A's frame have no applied field: an exact zero, not
        # the rounding residue of the fourth row.
        out["degree_one_imbalance"] = jnp.zeros_like(r)
    return out


_core_jit = jax.jit(_core, static_argnames=("plan", "config", "ref_rule"))


def _concrete_leaves(profile: RadialProfile) -> dict:
    """The leaves of a profile whose values the host can read, as numpy arrays.

    A leaf is left out when it is a JAX tracer (inside ``jit``, ``grad``, ``vmap``) or
    ``None``. Each value check of ``check_profile`` runs when the leaves that it reads
    are here, independent of the other leaves: in an HMC over the viscosity, ``eta`` is
    traced while ``rho0`` and the radius are concrete, and the checks of ``rho0`` and of
    the interval widths still run.
    """
    out = {}
    for k in ("radius", "rho0", "eta", "core_density", "core_mass", "outer_density", "surface_density",
              "uniform_gravity"):
        v = getattr(profile, k)
        if v is not None and not isinstance(v, jax.core.Tracer):
            out[k] = np.asarray(v)
    return out


def check_profile(profile: RadialProfile, plan: StaticPlan, config: StokesConfig) -> None:
    """Value checks of the concrete leaves of a profile; a traced leaf cannot be checked.

    Raises ``ValueError`` when a value is not finite, a viscosity is not positive,
    ``rho0`` is not positive (the logarithm of the TALA term, eq. 12.2, and the density
    contrasts) or exceeds ``Numerics.rho0_max`` (under FULL, where the density enters
    the Pade bound), ``rho0`` at the CMB equals ``rho_c`` or the top density equals
    ``rho_w`` (a zero density contrast of a boundary deflection), the radius does not
    have the structure, ``|kappa_h|`` on an ordinary interval exceeds
    ``Numerics.kappa_h_max`` under TALA, or the widths (and under TALA the actual
    ``kappa_h``) make the static Pade plan invalid (scaled norm above the accuracy
    threshold or a denominator that is not diagonally dominant). The leaf
    ``uniform_gravity`` must be set exactly under ``Uniform(None)`` (a static check of
    the pytree structure, which also runs under a trace), and a concrete value of it
    must be finite and positive.

    Each check runs when the leaves that it reads are concrete (``_concrete_leaves``).
    """
    # The uniform-gravity leaf and the flag go together. The test reads only whether
    # the leaf is None, which is pytree structure and so static: it runs under a trace.
    background = config.formulation.background
    wants_leaf = isinstance(background, Uniform) and background.value is None
    if wants_leaf and profile.uniform_gravity is None:
        raise ValueError("Uniform(None) reads the uniform gravity from the profile leaf uniform_gravity, "
                         "which is None; set it with profile.replace(uniform_gravity=g0)")
    if not wants_leaf and profile.uniform_gravity is not None:
        raise ValueError(f"the profile carries uniform_gravity, which only the background Uniform(None) "
                         f"reads; the formulation has {background!r}")
    leaves = _concrete_leaves(profile)
    for k, v in leaves.items():
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{k} has a value that is not finite")
    if "uniform_gravity" in leaves and not leaves["uniform_gravity"] > 0:
        raise ValueError(f"a uniform background gravity must be positive, got {leaves['uniform_gravity']}")
    if "eta" in leaves and np.any(leaves["eta"] <= 0):
        raise ValueError("every viscosity must be positive")
    rho0 = leaves.get("rho0")
    if rho0 is not None:
        if np.any(rho0 <= 0):
            raise ValueError("every reference density rho0 must be positive")
        if plan.rho_max > 0 and np.max(rho0) > config.numerics.rho0_max:
            raise ValueError(f"rho0 reaches {np.max(rho0)} kg/m^3, above Numerics.rho0_max = "
                             f"{config.numerics.rho0_max} kg/m^3, the limit of the static Pade plan")
        if "core_density" in leaves and rho0[0] == leaves["core_density"]:
            raise ValueError("rho0 at the CMB equals the core density: the CMB deflection has no density contrast")
    if "surface_density" in leaves:
        rho_top = leaves["surface_density"]
    elif profile.surface_density is None and rho0 is not None:
        rho_top = rho0[-1]
    else:
        rho_top = None
    if rho_top is not None and "outer_density" in leaves and rho_top == leaves["outer_density"]:
        raise ValueError("the topography density equals the outer density: the surface deflection has no "
                         "density contrast")
    r = leaves.get("radius")
    mask = profile.structure.interval_mask
    if r is not None:
        if np.any(np.diff(r) < 0):
            raise ValueError("the radius must be non-decreasing")
        pairs = tuple(int(i) for i in np.nonzero(r[1:] == r[:-1])[0] + 1)
        if pairs != profile.structure.pairs:
            raise ValueError(f"the radius has the pairs {pairs}, the structure {profile.structure.pairs}")
    # The TALA term of every ordinary interval, where the leaves that it reads are
    # concrete; otherwise its static limit enters the Pade check below.
    kappa_h = plan.kappa_h_max * mask
    if plan.terms.anelastic and rho0 is not None and (r is not None or plan.kappa_rule == "log_slope"):
        if plan.kappa_rule == "log_slope":
            kappa_h = np.log(rho0[1:] / rho0[:-1]) * mask
        else:
            kappa_h = (r[1:] / rho0[1:]) * (rho0[1:] - rho0[:-1]) / (r[1:] - r[:-1]) \
                * np.log1p(np.diff(r) / r[:-1]) * mask
        worst = float(np.max(np.abs(kappa_h))) if kappa_h.size else 0.0
        if worst > plan.kappa_h_max:
            i = int(np.argmax(np.abs(kappa_h)))
            raise ValueError(f"|kappa_h| = |ln(rho0_i / rho0_(i-1))| reaches {worst:.4f} on the interval {i + 1} "
                             f"(nodes {i} and {i + 1}), above Numerics.kappa_h_max = {plan.kappa_h_max}, the static "
                             "limit of the Pade plan under TALA. Put a density jump at a jump pair (two nodes "
                             "of equal radius), where its step is exact, or raise Numerics.kappa_h_max")
    if r is not None:
        h = np.log1p(np.diff(r) / r[:-1]) * mask
        x = linalg.balanced_norm_bound_intervals(plan.degrees[-1], h, kappa_h, plan.rho_max) / 2.0 ** plan.pade.squarings
        if x > linalg.PADE_THETA[plan.pade.order] or linalg.pade_denominator_margin(plan.pade.order, x) <= 0:
            raise ValueError(f"the interval widths of this radius give a scaled norm {x:.3f} outside the static "
                             f"Pade plan {plan.pade}; build the profile with a structure of wider bounds")


def solve(profile: RadialProfile, config: StokesConfig, discretisation: Discretisation = Discretisation(),
          *, _reference_viscosity: float | str = "largest") -> Kernels:
    """The kernels of a profile (low-layer entry; ``kernels.stokes_kernels`` calls it with
    the default discretisation).

    Parameters
    ----------
    profile
        ``RadialProfile``; every leaf can be traced.
    config
        ``StokesConfig``.
    discretisation
        Test-only legacy switches (``Discretisation``).
    _reference_viscosity
        Private hook of the tests: the viscosity scale rule of ``reference_viscosity``.
        It selects a static branch of the jitted core.

    Returns
    -------
    ``Kernels``.

    The static plan is built on the host (cached), the concrete leaves of the profile
    are checked (``check_profile``), and one jitted core with the plan and the config as
    static arguments computes the arrays.
    """
    if not isinstance(profile, RadialProfile):
        raise TypeError(f"profile must be a RadialProfile, got {type(profile).__name__}")
    plan = static_plan(profile.structure, config, discretisation)
    check_profile(profile, plan, config)
    out = _core_jit(profile, plan, config, _reference_viscosity)
    return Kernels(degrees=plan.degrees, degree_one_frame=plan.frame, radius=profile.radius, **out)
