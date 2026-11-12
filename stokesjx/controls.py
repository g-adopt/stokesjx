"""Control maps: pure functions from the controls of an inversion to a ``RadialProfile``,
the density map of a moved node table, and a flat-vector front for samplers.

The derivatives of stokesjx are for Hamiltonian Monte Carlo (HMC) over the radial
viscosity, including the radii of movable viscosity jumps (NOTES/PLAN.md section 3.3).
This module supplies the maps from such controls to the profile that
``stokes_kernels`` takes, and nothing that chooses a sampler or a prior (ruling of
2026-11-10). Every map is built once, eagerly and on the host, from a concrete profile;
the object it returns is a plain function of its traced arguments, valid under
``jax.jit``, ``jax.grad``, ``jax.vmap`` and ``jax.jvp``. The parameterisation follows
lovejx ``controls.py`` (ruling 3), ported with attribution and changed where the
instantaneous Stokes problem differs (NOTES/chunks/c4-controls-derivatives/01-plan.md D1):

``Layers``
    One natural-log viscosity per layer between given boundary radii, a duplicated node
    pair per fixed boundary, and movable boundaries whose radii are controls. A movable
    boundary moves by one of two methods (``motion``). Under ``"cut"``, the default, it is
    a viscosity cut: no node moves, and the solver splits the interval that holds the
    boundary into two exact sub-steps (NOTES/docs/equations.md section 19); the range can
    hold the density jumps of the input table. Under ``"affine"``, on request, it is a
    node pair, the nodes of its range follow it by two affine maps, ``rho0`` of a moved
    node is read from the input profile at its new radius, and the gravity is integrated
    again inside the solver. ``Layers`` also maps the kernels of the construction table to
    the fixed radii of the input table (``input_kernels``), so that a density anomaly
    given on the input table stays where it is when a boundary moves (``fields``,
    ``sheet_mass``).
``viscosity_by_node``
    One natural-log viscosity per node of a fixed table.
``ControlVector``
    A pytree of controls to one float64 vector and back, the value and the gradient of
    a user's scalar function of the controls as one jitted function of that vector, and
    a NumPy front for samplers and optimisers outside JAX.

No function of this module has a loop at trace time. The static index arrays of the
movable ranges (which node moves with which boundary, on which side, the knot table of
``rho0``, the brackets of the density map, the layer of every node and the node ranges of
the cuts) are built in ``Layers.__init__`` with numpy, and every traced operation is
elementwise work, one comparison of all moved nodes against all knots (or of all nodes
against all cuts), gathers and scatters.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

from .fields import Fields, apply as _apply_fields
from .kernels import Kernels, StokesConfig
from .profile import NodeLimits, NodeStructure, RadialProfile, _LEAF_NAMES, _is_traced

#: The clamp margin of a movable range as a fraction of its width (lovejx
#: ``_CLAMP_FRACTION``). A traced radius is clamped to ``[low + m, high - m]`` with
#: ``m = CLAMP_FRACTION (high - low)``, so that the nodes on one side of the boundary
#: never collapse onto a range end (a zero-width interval that is not a static pair).
CLAMP_FRACTION = 1.0e-3


def _km(r) -> str:
    """A radius in m as text in km, for error messages."""
    return f"{float(r) / 1e3:.6g} km"


def _interp_d4(x, x0, x1, f0, f1):
    """Linear interpolation ``f0 + (x - x0) ((f1 - f0) inv_w)`` with ``inv_w = 1/(x1 - x0)``
    (plan D4), numpy. The same expression as the traced map of ``Layers.__call__``, so
    that a call at the construction radius reproduces the inserted values to rounding."""
    return f0 + (x - x0) * ((f1 - f0) * (1.0 / (x1 - x0)))


@dataclass(frozen=True, eq=False)
class MovableRange:
    """Static description of one movable boundary of a ``Layers``. Host numpy only.

    ``Layers`` builds one record per movable boundary (``Layers.ranges``); the fields below
    are its constructor parameters and its attributes.

    Parameters
    ----------
    index
        The boundary index in ``Layers.boundaries``.
    low, high
        The range ends in m: node radii of the construction table, fixed. The boundary
        moves inside the open range; a traced radius is clamped to
        ``[low + margin, high - margin]``.
    start
        The construction radius ``r0`` of the boundary, m.
    margin
        ``CLAMP_FRACTION (high - low)``, m.
    below, above
        Construction indices of the nodes strictly between the inner range end and the
        lower pair member (``below``) and between the upper pair member and the other
        range end (``above``). They move with the boundary.
    members
        ``(lower, upper)``: the construction indices of the pair of the boundary.
    radii_below, radii_above
        Construction radii of ``below`` and ``above``, m.
    lo_node, hi_node
        Construction indices of the two range ends: the inner member where a range end
        is a pair (the upper member at ``low``, the lower member at ``high``).
    motion
        ``"cut"`` (the boundary is a viscosity cut of the fixed node table; ``below``,
        ``above``, ``radii_below``, ``radii_above`` are empty and ``members`` is
        ``(-1, -1)``) or ``"affine"`` (the nodes of the range follow the boundary).
    density_jumps
        The radii in m of the jump pairs of the input table with a ``rho0`` jump strictly
        inside the range, ascending (always empty under ``"affine"``, which refuses them).
    """

    index: int
    low: float
    high: float
    start: float
    margin: float
    below: np.ndarray
    above: np.ndarray
    members: tuple[int, int]
    radii_below: np.ndarray
    radii_above: np.ndarray
    lo_node: int
    hi_node: int
    motion: str
    density_jumps: tuple[float, ...]


class Layers:
    """Layered mantle viscosity with movable, differentiable boundary radii.

    ``Layers(profile, boundaries, movable={k: (low, high)})`` fixes the structure of the
    node table once, on the host. The instance is a pure function

        prof = layers(boundaries, log_eta, rho0=None)

    traceable in every argument, so that ``jax.grad`` of a misfit gives the derivative
    with respect to the layer viscosities, the radii of the movable boundaries and the
    reference density. It follows lovejx ``Layers`` (ruling 3), with differences that
    the physics of this package requires: the boundary vector runs from the CMB to the
    surface (there is no elastic lid), the call returns one ``RadialProfile`` (the
    profile carries ``eta``, so there is no separate rheology), there is no relaxed
    layer and no Kelvin element, ``rho0`` can be a control, a density anomaly stays on
    the input table, and a movable boundary is by default a viscosity cut that can
    cross the density jumps of the table (NOTES/chunks/c4-controls-derivatives/01-plan.md
    D1; rulings 14 and 17).

    **The layering.** ``boundaries`` are radii in m, strictly ascending, from the CMB to
    the surface: ``boundaries[0]`` must be the CMB radius of the profile and
    ``boundaries[-1]`` its surface radius, exactly. Layer ``j`` lies between
    ``boundaries[j]`` and ``boundaries[j + 1]`` and carries the viscosity
    ``exp(log_eta[j])`` Pa s on every node, so there are ``B - 1`` layers. lovejx ends its
    layering at the base of an elastic lid; the instantaneous Stokes problem has no
    elastic lid, every node flows, and a stiff lid is a layer with its own ``log_eta``.
    The CMB and the surface are never movable.

    **The pairs.** Each fixed interior boundary, and under ``motion="affine"`` each
    movable one, is a duplicated node pair: the lower member
    closes the layer below and the upper member opens the layer above. A boundary on an
    existing pair of the profile uses that pair; a boundary on a single node duplicates
    it with its own values on both members; a boundary between two nodes gets two copies
    of the linear interpolation of ``rho0`` and ``eta`` at its radius. An inserted pair
    changes the discretisation of the interval it splits by a term of second order in
    the width (the material of an interval is the mean of its two node values).

    **Movable boundaries and the method.** ``movable = {k: (low, high)}`` lets boundary
    ``k`` move inside the open range ``(low, high)``. Both ends must be node radii of the
    construction table (an input node, or a pair of another, fixed boundary); at a pair
    the range reads its inner member. The keyword ``motion`` selects the method for all
    movable boundaries of one ``Layers``: ``"cut"``, the default for every range (the
    paragraph "The viscosity cut" below), or ``"affine"`` on request (the next
    paragraph). ``Layers.motion`` holds the method (``None`` without a movable boundary).
    The method does not depend on where the density jumps of the table are. The cut is
    the default for three reasons. Under an incompressible formulation its derivative
    with respect to the radius has no kinks: on the tables T1 (137-node PREM table, range
    (5060.5, 5700) km, ``lmax = 50``) and T2 (257-node refined table, range (5466.9, 5970)
    km, ``lmax = 100``) of ``NOTES/benchmarks/scripts/measure_kink.py``, ``E_rms`` (below,
    "Measured kinks of the affine method") is at most 1.2e-11 nat for the smooth and the LEBEDEV
    anomaly and 7.6e-11 nat for the rough anomaly, against 0.103 and 0.137 nat for the
    affine method, and the relative jump of the derivative is at most 3.2e-11
    (NOTES/benchmarks/c6-kink-default.json). On the 137-node PREM table the cut is within
    1e-6 of a fixed boundary pair at the new radius, where the affine method is within
    5e-4 to 2e-3. It is as fast as the affine method. Under TALA the cut has small kinks at the
    input nodes also on a range without a density jump: ``E_rms`` 0.036 nat at a
    viscosity contrast of 10 and 0.061 nat at 1.1 (range (5700, 5970) km of the 137-node
    table, NOTES/benchmarks/c5-kink-t3te.json); a range with a density jump inside adds
    the value jump of the paragraph "TALA and a density jump inside a range".

    **The affine method.** Inside the range, with the construction radius ``r0`` and
    the new radius ``r``, the nodes below the pair map affinely from ``[low, r0]`` onto
    ``[low, r]`` and those above it from ``[r0, high]`` onto ``[r, high]``:

        x = low  + (r_c - low)  s_below,    s_below = (r - low) / (r0 - low)
        x = high - (high - r_c) s_above,    s_above = (high - r) / (high - r0)

    and both pair members sit at ``r``. The code evaluates both maps in the equivalent
    offset form ``x = r_c + Q (r - r0) / V`` (``Q = r_c - low``, ``V = r0 - low`` below;
    ``Q = high - r_c``, ``V = high - r0`` above). At ``r = r0`` the factor ``r - r0`` is an
    exact zero, so every moved node is its construction radius bit for bit, eager and
    under ``jit``, whatever order the compiler gives the constant factors. The node
    count, the pairs and every static decision of the solver stay those of the
    construction (``Layers.profile.structure``), and one jitted program serves every
    position.

    Range rules (each refused with ``ValueError``): a range must not hold another
    boundary strictly inside; its ends must not be the radius of another movable
    boundary; two ranges must not overlap (they can share an end node). Under
    ``"affine"`` a range must also not hold, strictly inside, a pair of the input profile
    at which ``rho0`` jumps: its nodes move with the boundary and would carry the jump
    along, so under ``"affine"`` a viscosity boundary cannot cross a density jump. A pair
    without a ``rho0`` jump (a viscosity-only pair of the input) is allowed inside and
    moves with the range, its two nodes at one radius. Under ``"cut"`` a range can hold
    density jumps (NOTES/PLAN.md ruling 14): on a PREM-like table a boundary near 670 km
    depth moves freely across the 670 km density jump.

    **The viscosity cut (motion="cut", the default).** A movable boundary inserts no node: the
    construction table is the input table with the pairs of the fixed boundaries, and it
    does not move. The boundary radius ``c`` is the profile leaf ``cut_radius`` (one per
    movable boundary, in the order of ``movable_index``), and the structure records the
    node range ``(lo, hi)`` of each cut (``NodeStructure.cuts``; the inner member at a
    pair end). The interval between the nodes ``a`` and ``a + 1`` with ``r_a <= c <
    r_(a+1)`` is split at ``c`` into two exact sub-steps,

        Z = D(e_hi) expm(f_hi B_a) D(e_hi)^-1 D(e_lo) expm(f_lo B_a) D(e_lo)^-1,

    with the balanced interval exponent ``B_a`` (the interval-mean ``rho0`` and, under
    TALA, ``kappa_h`` of the whole interval), the fractions ``f_lo = ln(c / r_a) / h_a``,
    ``f_hi = ln(r_(a+1) / c) / h_a`` of its width, and the viscosities ``e_lo`` of node ``a``
    (below the cut) and ``e_hi`` of node ``a + 1`` (above it). The call sets the viscosity
    of every node by side: a node with ``r_n > c`` is above the cut, a node on the cut
    (``r_n = c``) below it. ``rho0``, the radii and the gravity stay those of the input
    table at every position, so the density structure of the input is exact everywhere,
    and the construction radius ``r0`` does not enter the model (two ``Layers`` with
    different ``r0`` and the same range give the same kernels bit for bit). The density
    map is the static gather of the input nodes. ``fields.apply(K, d, layers.profile)``
    accepts a cut profile (no node moves); ``Layers.fields`` is the rule for a density on
    the input table, as for the affine method.

    Accuracy of the cut (NOTES/chunks/c4b-boundaries-across-jumps/03-implementation.md):
    the cut is exact for its discrete model, to at most 1e-13 (typically 2e-14) against
    the mpmath oracle and against the package on the concrete tables of
    ``tests/reference_cut.py`` (swapped pair, node split, pair, triple, geometric pair),
    on both sides of a density pair, at 1 m from it and on it, under every formulation.
    Against the same layering built with a fixed boundary pair at ``c`` (the reference of
    the affine method, rho0 of the pair interpolated) it differs by the rho0 of the two
    parts of the cut interval, which grows with the interval width: on the 137-node PREM
    table 1e-6 of the kernels (FULL, lmax 10 to 30; 7e-6 under TALA off a density pair;
    the fields of the chunk 4 layouts 4e-8 and 2e-7), on the 25-node reduced table 1e-4
    (2e-4 under TALA). Under NO_SELF_GRAVITY with a uniform ``g`` the two models are equal
    to rounding.

    Derivative of the cut: under an incompressible formulation the map from ``c`` to every
    output is continuously differentiable everywhere, also where the cut crosses an
    input node or a density pair. The derivative of the chain of steps with respect to
    ``c`` holds the factor ``(A(e_lo) - A(e_hi)) / c``. This factor does not depend on the
    interval density (the density enters only the entries (3,6) and (4,5) of ``A``, which
    the viscosity similarity leaves unchanged), so it is the same matrix on both sides of
    a node; it maps the jump vector of a source node, which has entries in rows 3 and 6
    only, to zero; and a density pair is a step equal to the identity
    (NOTES/docs/equations.md section 19.3). The map
    is not twice continuously differentiable: its second derivative jumps where the cut
    crosses any input node (the second derivative of the kernel of the crossed node
    holds ``(A(e_lo) - A(e_hi)) A J``, which is not zero; the interval density adds a
    smaller term where it changes). Measured jumps of the second derivative: 0.9 of the
    kernel of the crossed node, and 0.07 to 0.15 of the geoid field of a smooth density
    on the 137-node table (0.47 on the 25-node table); the estimated energy error of one
    such jump for a leapfrog step of ``sigma_r = 10`` km is about 2e-3 nat
    (NOTES/chunks/c4b-boundaries-across-jumps/04-review-physics-r1.md section 3.4).
    The numbers that follow are for the first derivative. Measured: the relative jump of the
    derivative of the geoid field at most 5e-10 at every crossing of four tables (the
    rounding of the extrapolation; the chunk 4 tables T1 and T2 and two ranges across
    the 670 km and 400 km density jumps, smooth, LEBEDEV and rough anomaly), and the
    energy estimate ``E_rms`` of the kinks at most 1e-9 nat, for every construction radius
    (NOTES/benchmarks/c4b-kink.json; the affine method on T1 and T2: 0.103 and 0.137 nat). The kinks
    of the affine method below do not exist for a cut range, and ``crossings`` lists only
    the input radii (kind 2) and the density-jump pairs (kind 4). At a tie (``c`` on a
    node or on a pair) the package returns the upper one-sided derivative and the value
    from above, the convention of the affine method. The cut has no static width limits
    (``profile.structure.limits`` is ``None``, ``width_factor`` is 1, ``pade_plan`` is the
    plan of the construction table) and is as fast as the affine method (0.95 and 1.00 of
    its gradient run time and 0.92 of its gradient compile time at S1 and S2 of
    ``NOTES/benchmarks/scripts/bench_controls.py``; NOTES/benchmarks.md).

    **TALA and a density jump inside a range.** Under TALA the kernels jump when a cut
    crosses a pair with a density jump: the TALA step of the pair (eq. 12.3) takes the
    viscosity of its two members, and both change side when the cut passes. The jump is
    a property of the TALA model with a sharp density jump, not of the cut (the oracle
    agrees with the cut to 2e-14 on both sides). ``stokes_kernels`` issues a
    ``RuntimeWarning`` on every call when a jump pair lies strictly inside a cut range
    under TALA (also for a pair with equal ``rho0``, where nothing jumps). Measured on the
    full PREM table, the 670 km boundary in (5466.875, 5970) km, lmax 50
    (NOTES/chunks/c4b-boundaries-across-jumps/01-plan.md sections 6 and 6.3,
    NOTES/benchmarks/c4b-kink.json):

    - the relative jump of the geoid field is 0.5 to 2.5 % at viscosity contrasts 3 to
      100 across the boundary;
    - as a displacement of the boundary (``L = |[N]| / |dN/dc|``) it is about 10 km at a
      contrast of 10 (1e22 | 1e21 Pa s), 16 to 27 km at contrasts 1.1 to 3 and 2 to 4 km
      at a contrast of 100;
    - for an HMC over the radius with a Gaussian geoid likelihood whose posterior
      standard deviation of the radius is ``sigma_r``, one crossing of a trajectory costs
      the energy error ``E_value = sqrt(lam^2 ([N] . (N_mid - N(r_true)))^2 + lam |[N]|^2)``
      (at least ``L / sigma_r``; NOTES/docs/equations.md section 19.5): 1.0 to 1.5 nat at
      ``sigma_r = 10`` km and 0.34 to 0.64 nat at 30 km for 1e22 | 1e21 Pa s, and 0.2 to
      4.5 nat at 10 km over the contrasts 1.1 to 100 (the error grows as the contrast
      falls). The leapfrog map stays reversible and volume-preserving and the Metropolis
      test sees the jump, so the chain keeps its target; the jump lowers the acceptance
      of the trajectories that cross the pair: under detailed balance the mean
      acceptance of an attempted crossing is ``2 exp(-u) / (1 + exp(-u))`` for
      ``u = |[U]|``, about 0.83 at 0.34 nat, 0.54 at 1.0 nat, 0.36 at 1.5 nat and 0.02 at
      4.4 nat. The derivative kinks of such a range (the small kinks at the input nodes
      and the derivative jump at the pair) add ``E_rms`` up to 0.04 nat at a contrast of
      10 and up to 0.2 nat at a contrast of 1.1.

    Usage rule: under TALA, when the radius is sampled with HMC, end the movable
    range at the density pair (a range end on a pair is allowed; the cut then never
    crosses it, nothing jumps, and no warning is issued). Values and derivatives are
    exact for the discrete model on both sides of the pair, so evaluating the kernels at
    given radii, or a sampler that does not integrate a trajectory across the pair, is
    not affected. Under FULL and NO_SELF_GRAVITY (incompressible) nothing jumps.

    Three TALA models at one pair radius ``J``. The pair step can take the viscosity
    above (``e_above``: the cut just below ``J``, and its limit from below), below
    (``e_below``: the cut at or above ``J``) or the mean of the two (a FIXED boundary on
    the pair, eq. 12.1). The tie rule: at ``c = J`` both members are below the cut, so the
    value is the limit from above (``e_below``). Consequence: under TALA a movable
    boundary that starts at a density pair (``r0 = J``) gives at its construction radius
    kernels that differ by 4.4e-2 from the same layering with that boundary fixed on the
    pair (and by 9e-2 from the limit from below). Under FULL the three models are one.
    Under TALA the step across a density pair is the anelastic rule of eq. 12.3 in all
    three models (``rho0 u_r`` continuous), and the models differ only in the viscosity
    of the step. A movable boundary is allowed to cross a density pair under TALA:
    ``stokes_kernels`` issues the warning of ``propagate.solve``, and the usage rule
    above applies.

    **Reference density of a moved node (affine method).** A moved node reads ``rho0`` from the
    piecewise-linear "knot table" at its new radius, so that the density structure stays
    where the input profile puts it and only the viscosity jump moves (lovejx rule). The
    knots are every input node at its static input radius (for a node that a boundary
    duplicates, the lower member) and both members of every pair that a FIXED boundary
    inserts between two input nodes. Not knots: the two members of a pair that a
    movable boundary inserts between two input nodes, and the upper member of a
    duplicated node. A pair of knots (an input pair, a fixed inserted pair) is two knots
    at one radius with a zero slope weight between them; the clamp keeps every moved
    node strictly inside its range, so that zero-width interval is never selected.

    The keyword ``rho0`` (shape ``(N,)`` of the construction profile) makes the
    reference density a control. Its entries at the knots are the knot values (read
    through ``table_nodes``), its entries at the fixed nodes are their values, and its
    entries at the moved nodes are overwritten by the interpolation. In particular the
    two members of a pair that a movable boundary inserts between nodes are never read,
    and their derivative is exactly zero. A ``rho0`` control must keep equal values on
    the two members of an input pair without a density jump inside a movable range or
    under a movable boundary, and on the two members of a duplicated node: otherwise the
    knot function has a jump that a moved node meets as a discontinuity in the boundary
    radius. A concrete ``rho0`` that breaks this raises ``ValueError``; a traced one is
    not checked. A traced ``rho0`` is also not checked against the static limits
    ``kappa_h`` and ``rho0_max`` below (no traced leaf is checked against its values); the
    caller's prior must keep it inside them. Under ``"cut"`` no node moves: every entry of
    ``rho0`` is data, and the equal-value rule applies to the members of a duplicated node
    only.

    **Gravity.** The profile carries no gravity array; ``stokes_kernels`` integrates the
    hydrostatic gravity from ``rho0`` and the radii of the profile inside the
    differentiated function (``propagate.background_gravity``). Under ``"cut"`` the
    radii and ``rho0`` are those of the construction table at every position, so the
    gravity does not depend on the boundary radius. Under ``"affine"`` it is integrated
    on the moved radii and follows the boundary.

    **The clamp.** A concrete boundary vector is checked: a fixed entry that differs
    from its construction radius, or a movable one outside its open range, raises. A
    traced vector cannot be checked; its fixed entries are not read (their derivative is
    exactly zero), and every movable radius is clamped with ``jnp.clip`` into the closed
    range ``Layers.radius_bounds = [low + m, high - m]``. Outside it the derivative with
    respect to that radius is exactly zero, strictly inside it is one, and at a clamp
    bound one half (the JAX rule for a tie). A concrete radius within ``m`` of an end is
    moved to the bound as well. A sampler's prior must have its support inside
    ``radius_bounds``: outside it the gradient carries no information.

    **One-sided slopes and kinks (affine method).** The map from a movable radius to every output is
    continuous and piecewise smooth. Its derivative jumps where a moved node crosses an
    input radius (the knot table of ``rho0`` and the bracket of the density map both
    change interval), where the pair crosses an input radius (an input node changes
    side of the viscosity jump), and at the construction radius ``r0``, where every
    moved node sits on its own input radius at once. At every such crossing the package
    returns the upper one-sided derivative, the slope for an increasing boundary radius
    (the comparisons ``x >= knot`` of the knot table and ``x_a < r_i`` of the bracket),
    in reverse and in forward mode. A finite-difference check at ``r0`` therefore
    agrees with the forward difference and not with the backward one. Crossings are
    dense: on the full PREM table (137 nodes) a range of 640 km holds about 400, one per
    1.5 km. Crossings of the nodes below the boundary group at rational stretch factors
    ``s = (r - low) / (r0 - low)`` with small denominators when the input radii below the
    boundary are equally spaced from the range end; ``r0`` is the group ``s = 1``, and
    the largest is ``s = 2``, which lies inside the range when ``r0`` is closer to that
    range end than to the other. The paragraph "Measured kinks of the affine method"
    below gives the measured kink sizes and the HMC energy estimate
    (NOTES/chunks/c4-controls-derivatives/03-implementation.md section 5).

    **Static limits and the Pade cost (affine method).** The Pade order and squaring count of the
    interval exponentials are static. A moved node widens the intervals on one side of
    its boundary, by up to ``(high - low) / (r0 - low)`` (or the mirror). ``Layers``
    stores in ``profile.structure.limits`` the largest width of every interval, the
    largest ``|kappa_h|`` (TALA) and the largest ``rho0`` over every admissible position
    (clamp bounds and every crossing; plan D3), and ``stokes_kernels`` builds its plan
    from them and refuses a configuration whose limits exceed ``Numerics``. A range with
    the boundary near one end costs squarings: put the construction radius near the
    middle of its range. ``width_factor`` reports the largest widening, ``pade_plan``
    the plan.

    **Density anomalies (fields).** A density anomaly is a function of the radius given
    on the INPUT table (the nodes of the profile given to the constructor), radial axis
    first, with the degree and order axes last under the layout rule of
    ``stokesjx.sh`` (``sh.check_layout``; CONVENTIONS 15). A boundary that moves does not
    move it. The
    field is the trapezoid rule of the input table,

        F_o[l, m] = sum_i K_o[l](r_i) w_i drho[i, l, m],

    with the input radii ``r_i``, the static trapezoid weights ``input_weights`` and the
    kernel of the current profile at the input radius (``input_kernels``). Under
    ``"cut"`` (the default) no node moves, and every input node takes the kernel of its
    own construction node (a static gather). Under ``"affine"`` an input node outside
    every movable range does the same, and an input node strictly inside a range takes
    the linear interpolation between the two moved nodes that bracket its radius. Use
    ``layers.fields(K, drho)``; ``fields.apply`` refuses the profile of a ``Layers``
    with a movable boundary under ``motion="affine"``. A caller whose density is given
    on the input table uses ``Layers.fields`` under either method and also when no
    boundary moves, so that the field does not change when a boundary is made movable:
    ``fields.apply(K, d, layers.profile)`` is the trapezoid rule of the construction
    table, a different quadrature (they differ by up to 9e-4 of the geoid for a density
    with an independent value per node). ``sheet_mass`` gives the same field in the transposed
    form (the anomaly as sheet masses on the construction nodes).

    **Accuracy of the density map (affine method).** The linear interpolation of the kernels between
    stretched nodes has an error of second order in the input node spacing times the
    stretch factor (halving every input interval divides the error by about 4). The
    reference is a fixed table built with the boundary pair at the new radius. Measured
    on the full PREM table (137 input nodes, spacing 22.6 km),
    NOTES/chunks/c4-controls-derivatives/04-review-physics-r1.md section 5:

    - deep range (the 670 km boundary, range 670 to 1310 km depth): the fields at
      ``lmax = 30`` to 5e-4 of the geoid; the error per degree falls with the degree
      (9e-4 of a geoid kernel row at ``l = 10``, 2e-7 at ``l = 100``), because the kernel
      at the nodes of the range decays with the degree;
    - shallow range (a 100 km boundary movable in (6150, 6370) km): the geoid field at
      ``lmax = 30`` to 3e-3 to 5.5e-3, and 1 to 3 % of a geoid kernel row from ``l = 30``
      to 100, because the kernel is large inside the range at every degree and changes
      over the thickness of the low-viscosity layer.

    For a shallow movable range, refine the input table inside the range. The
    derivative with respect to a movable radius is the exact derivative of this
    discrete map. Against the derivative of the field of a table built at the new
    radius it differs by a term of first order in the node spacing (1e-3 to 6e-3 per
    degree in the middle of a deep range, 2e-2 to 4e-2 where the nodes of one side are
    stretched by about 2). The cause and the remedy are those of the kinks below.

    **Measured kinks of the affine method** (``NOTES/benchmarks/scripts/measure_kink.py``,
    NOTES/benchmarks/c4-kink.json and c4-kink-t2s.json;
    NOTES/chunks/c4-controls-derivatives/03-implementation.md section 5).
    The size of a kink is ``rho``, the jump of the derivative of the geoid field over the
    derivative; crossings closer than 1 m are one group with one total jump. With a
    Gaussian geoid likelihood whose posterior standard deviation of the radius is
    ``sigma_r`` (10 or 30 km), ``E_rms`` estimates the leapfrog energy error that the
    groups within ``1.5 sigma_r`` of the true radius add to one step of ``sigma_r``; it
    uses the bound ``|dH| <= J sigma_r`` per group, about 2.5 times the root mean square
    of ``dH`` itself. Measured, the largest ``E_rms`` over a grid of true radii covering
    the range (smooth anomaly, LEBEDEV tomography, rough anomaly with an independent
    value per node):

    - full PREM table (137 input nodes), range (5060.5, 5700) km, ``r0 = 5370`` km at 48 %
      of the range, ``lmax = 50``: 398 crossings in 344 groups; the largest group is 13
      crossings at ``s = 2`` (``r0 + 309.5`` km), ``rho`` 9.3e-2, 4.2e-2, 1.3e-1; at
      ``r0`` 2.5e-2, 1.3e-2, 3.1e-2; ``E_rms`` 0.103, 0.049, 0.21 nat;
    - the same with ``r0 = 5395`` km (52 %): ``s = 2`` lies outside the range, the largest
      group is ``r0`` itself; ``E_rms`` 0.059, 0.032, 0.21 nat;
    - refined LEBEDEV table (257 input nodes), hager4 viscosities, range
      (5466.9, 5970) km, ``r0 = 5700`` km at 46 %, ``lmax = 100``: 981 crossings in 832
      groups; ``E_rms`` 0.103, 0.137, 0.33 nat (largest groups at ``s = 2`` and ``11/6``);
    - the same with the range (5466.9, 5895.9) km (``r0`` at 54 %): 0.050, 0.125, 0.13 nat.

    The rule of the plan (``E_rms <= 0.1`` nat for the smooth and the LEBEDEV anomaly on
    both tables) is not met. The linear interpolation is accepted with a usage rule
    (NOTES/PLAN.md ruling 15): place the construction radius at or above the middle of
    its range, measured from the range end whose input radii are equally spaced. This
    moves the largest group out of the range and halves the estimate on the PREM table.
    The usage rule applies to ``motion="affine"`` only. A C1 interpolation of the kernel
    rows in ``r`` would remove the group jumps for every ``r0``. The package does not
    implement it: the cut, the default method (NOTES/PLAN.md ruling 17), has no such
    kinks under an incompressible formulation. The kinks never bias a Metropolis-corrected
    sampler (leapfrog stays volume preserving and reversible); they lower the acceptance
    rate. Measured with the HMC example (``examples/hmc_layers.py``, runs B and C of
    NOTES/chunks/c5-integration/03-implementation.md): on a range that ends at the 670 km
    density jump (FULL, 137 nodes, ``lmax = 20``, 1000 transitions) the affine method and
    the cut give the same mean Metropolis probability within its standard error: 0.957 for
    both at step 0.5, 0.832 against 0.833 (standard error of the paired difference 0.002)
    at step 1.0.

    Parameters
    ----------
    profile
        A concrete ``RadialProfile`` without ``structure.limits`` and without viscosity
        cuts (a ``Layers`` profile is not the input of another ``Layers``).
    boundaries
        ``(B,)`` radii in m, ``B >= 2``, strictly ascending, from the CMB radius to the
        surface radius.
    movable
        ``None`` or ``{boundary index: (low, high)}`` in m, with indices in ``1 .. B - 2``.
    motion
        ``"cut"`` (default) or ``"affine"``: the method of the movable boundaries
        (above).

    Attributes
    ----------
    profile
        The construction profile: the input with the inserted pairs, ``eta`` of the
        input (interpolated at inserted nodes; every call overwrites it), the scalar
        leaves of the input, and a structure with ``cuts`` and the leaf ``cut_radius``
        (the construction radii) when a boundary is movable under ``"cut"``, or with
        ``limits`` under ``"affine"``.
    motion
        The method of the movable boundaries, the keyword ``motion``: ``"cut"`` or
        ``"affine"``, and ``None`` without a movable boundary.
    input_profile
        The profile given to the constructor.
    input_weights
        ``(N_in,)`` trapezoid weights of the input table, m, numpy.
    input_node
        ``(N_in,)`` construction index of each input node (the lower member for a node
        that a boundary duplicates).
    boundaries
        ``(B,)`` construction radii, m.
    ranges
        ``{boundary index: MovableRange}``.
    movable_index
        ``(R,)`` boundary indices of the movable boundaries, ascending.
    radius_bounds
        ``(R, 2)`` closed clamp range ``[low + m, high - m]`` per movable boundary, m.
    start_radii
        ``(R,)`` construction radius ``r0`` of each movable boundary, m, in the order of
        ``movable_index`` (``boundaries[movable_index]``): the start vector of a sampler
        or an optimiser over the movable radii (``full_boundaries`` expands it).
    pairs
        ``(B - 2,)`` lower member of each interior boundary pair; ``-1`` for a cut boundary.
    layer_of
        ``(N,)`` layer of every construction node at the construction radii.
    table_nodes
        Construction indices of the knots of the ``rho0`` table, ascending (every node
        under ``"cut"``, where no node moves).
    n_layers
        The number of layers, ``B - 1``.
    width_factor
        The largest ratio of the width limit of an interval to its construction width:
        1.0 without a movable boundary and under ``"cut"``; under ``"affine"`` a large
        factor costs squarings (put the construction radius near the middle of its
        range).

    Raises
    ------
    ValueError
        On a ``motion`` that is not one of the two strings; on a traced profile or a
        profile whose structure has limits or cuts; on boundaries
        that are not a finite vector of at least two radii, not strictly ascending,
        that do not start at the CMB or end at the surface; on a boundary on three or
        more nodes; on a ``movable`` key that is not an int in ``1 .. B - 2``; on a
        boundary not strictly inside its range, a range end that is not a node radius,
        a range end at another movable boundary, another boundary strictly inside a
        range, a ``rho0`` jump strictly inside a range under ``"affine"``, or two
        overlapping ranges.
    """

    def __init__(self, profile: RadialProfile, boundaries, *, movable: dict | None = None,
                 motion: Literal["cut", "affine"] = "cut"):
        if not isinstance(profile, RadialProfile):
            raise ValueError(f"Layers takes a RadialProfile, got {type(profile).__name__}")
        if not isinstance(motion, str) or motion not in ("cut", "affine"):
            raise ValueError(f"motion must be 'cut' or 'affine', got {motion!r}")
        for name in _LEAF_NAMES:
            if _is_traced(getattr(profile, name)):
                raise ValueError("Layers takes a concrete profile: it fixes the structure of the node table "
                                 f"on the host, once, before any trace (the leaf {name} is traced)")
        if profile.structure.limits is not None or profile.structure.cuts:
            raise ValueError("the profile of a Layers with movable boundaries cannot be the input of another "
                             "Layers; pass the profile that the first Layers was built from")
        self.input_profile = profile
        r_in = np.asarray(profile.radius, dtype=np.float64)
        rho_in = np.asarray(profile.rho0, dtype=np.float64)
        eta_in = np.asarray(profile.eta, dtype=np.float64)
        self._r_in, self._rho_in = r_in, rho_in
        n_in = r_in.size
        start = self._checked_boundaries(r_in, boundaries)
        self.boundaries = start
        B = start.size
        movable = dict(movable or {})
        for key in movable:
            if isinstance(key, bool) or not isinstance(key, (int, np.integer)) or not 1 <= key <= B - 2:
                raise ValueError(f"movable is keyed by the index of an interior boundary, 1 to {B - 2} (the CMB "
                                 f"and the surface are never movable); got {key!r}")
        movable = {int(k): (float(v[0]), float(v[1])) for k, v in movable.items()}
        # The density-jump pairs of the input table strictly inside each range: recorded
        # in MovableRange.density_jumps under "cut", refused under "affine"
        # (_movable_ranges). The method of the movable boundaries is the keyword itself:
        # "cut" (the default) or "affine", for every range of this Layers, whether or not a
        # range holds a density jump; None without a movable boundary.
        jumps_in = {k: self._density_jumps(r_in, rho_in, lo, hi) for k, (lo, hi) in movable.items()}
        self.motion = motion if movable else None
        cut = self.motion == "cut"
        # The boundaries that insert a node pair: every interior boundary under "affine",
        # the fixed ones only under "cut" (a cut is not a node of the table).
        paired = [k for k in range(1, B - 1) if not (cut and k in movable)]

        # ---- 1. Insert one pair per paired interior boundary (lovejx _insert_pairs).
        # origin: input index of each construction node, -1 for an inserted node;
        # kind: 0 input node, 1 upper copy of a duplicated node, 2 inserted between two
        # nodes; owner: the boundary index of an inserted or duplicated node.
        positions, ins_r, ins_rho, ins_eta, ins_kind, ins_owner = [], [], [], [], [], []
        for k in paired:
            b = start[k]
            on = np.flatnonzero(r_in == b)
            if on.size == 2:
                continue                       # an existing pair: used as it is
            if on.size > 2:
                raise ValueError(f"boundary {k} at {_km(b)} falls on {on.size} nodes of one radius; a boundary "
                                 "needs one duplicated pair")
            if on.size == 1:
                # On a single node: a copy after it with its own values; the input node
                # is the lower member, the copy the upper member.
                node = int(on[0])
                positions.append(node + 1)
                ins_r.append(b); ins_rho.append(rho_in[node]); ins_eta.append(eta_in[node])
                ins_kind.append(1); ins_owner.append(k)
                continue
            # Between nodes j and j + 1: two copies of the D4 interpolation at b.
            j = int(np.sum(r_in < b)) - 1             # the last node below b
            v_rho = _interp_d4(b, r_in[j], r_in[j + 1], rho_in[j], rho_in[j + 1])
            v_eta = _interp_d4(b, r_in[j], r_in[j + 1], eta_in[j], eta_in[j + 1])
            positions += [j + 1, j + 1]
            ins_r += [b, b]; ins_rho += [v_rho, v_rho]; ins_eta += [v_eta, v_eta]
            ins_kind += [2, 2]; ins_owner += [k, k]
        radius = np.insert(r_in, positions, ins_r)
        rho0 = np.insert(rho_in, positions, ins_rho)
        eta = np.insert(eta_in, positions, ins_eta)
        origin = np.insert(np.arange(n_in), positions, -1)
        kind = np.insert(np.zeros(n_in, dtype=int), positions, ins_kind)
        owner = np.insert(np.full(n_in, -1), positions, ins_owner)
        N = radius.size
        self.n_nodes = N
        self.input_node = np.flatnonzero(origin >= 0)
        assert self.input_node.size == n_in

        # ---- 2. Pairs and layers. A cut boundary has no pair (-1).
        lowers = []
        for k in range(1, B - 1):
            if k not in paired:
                lowers.append(-1)
                continue
            at = np.flatnonzero(radius == start[k])
            if at.size != 2:
                raise ValueError(f"boundary {k} at {_km(start[k])} does not give one pair ({at.size} nodes)")
            lowers.append(int(at[0]))
        self.pairs = np.asarray(lowers, dtype=int)
        # _fixed_layer[n] counts the paired interior boundaries whose lower member lies
        # below n: a node counts as above a paired boundary from the upper member of its
        # pair on. A cut boundary at c counts for every node with r_n > c (a node on the
        # cut is below it, the bracket r_a <= c of system.cut_intervals).
        fixed_lowers = self.pairs[self.pairs >= 0]
        self._fixed_layer = np.sum(np.arange(N)[:, None] > fixed_lowers[None, :], axis=1).astype(int)
        cut_radii = np.array([start[k] for k in sorted(movable)]) if cut else np.zeros(0)
        # layer_of[n]: the layer of every construction node at the construction radii.
        self.layer_of = (self._fixed_layer + np.sum(radius[:, None] > cut_radii[None, :], axis=1)).astype(int)
        self._construction_radius = radius
        self._construction_rho0 = rho0

        # ---- 3. Movable ranges.
        self.ranges = self._movable_ranges(movable, r_in, rho_in, radius, jumps_in)
        self.movable_index = np.asarray(sorted(self.ranges), dtype=int)
        R = self.movable_index.size
        if R:
            rs = [self.ranges[k] for k in self.movable_index]
            low = np.array([g.low for g in rs]); high = np.array([g.high for g in rs])
            margin = np.array([g.margin for g in rs])
            self.radius_bounds = np.stack([low + margin, high - margin], axis=1)
        else:
            self.radius_bounds = np.zeros((0, 2))
        # The construction radii of the movable boundaries, public (class docstring).
        self.start_radii = np.array([self.ranges[k].start for k in self.movable_index], dtype=np.float64)

        # ---- 4. The static arrays of the traced map (plan D2): every moved node of
        # every range in one set, with its range, side and construction radius. Under
        # "cut" no node moves, and the set is empty.
        moved, side, range_of = [], [], []
        for q, k in enumerate(self.movable_index if not cut else []):
            g = self.ranges[k]
            moved += [g.below, np.asarray(g.members), g.above]
            side += [np.ones(g.below.size), np.zeros(2), -np.ones(g.above.size)]
            range_of.append(np.full(g.below.size + 2 + g.above.size, q))
        self.moved_nodes = np.concatenate(moved).astype(int) if moved else np.zeros(0, dtype=int)
        self._side = np.concatenate(side) if side else np.zeros(0)
        self._range_of = np.concatenate(range_of).astype(int) if range_of else np.zeros(0, dtype=int)
        self._r_c = radius[self.moved_nodes]
        if R:
            self._low = np.array([self.ranges[k].low for k in self.movable_index])
            self._high = np.array([self.ranges[k].high for k in self.movable_index])
            self._start = np.array([self.ranges[k].start for k in self.movable_index])
            self._margin = np.array([self.ranges[k].margin for k in self.movable_index])
        # Per moved node: the numbers of its range, gathered on the host, and the static
        # coefficients of one affine form for both sides and the pair, written as an
        # offset from the construction radius,
        #     x = P + Q ((n - U) / V),     P = r_c,  U = r0,
        # below the pair  Q = r_c - low,   V = r0 - low;
        # above the pair  Q = high - r_c,  V = high - r0;
        # at the pair     Q = V = 1        (and r_c = r0).
        # Below the pair this is low + (r_c - low) (n - low) / (r0 - low) with n - low split
        # into (n - r0) + (r0 - low); above the pair it is high - (high - r_c) (high - n) /
        # (high - r0) with high - n split into (high - r0) - (n - r0). Q and V are positive
        # on both sides, and Q / V is the stretch factor of the side (dx / dn).
        # Why the offset form: at n == r0 the factor n - r0 is an exact zero, so x is r_c
        # bit for bit for ANY evaluation order of the constant factors. This matters under
        # jit: P, Q, U, V are constants of the compiled program, and XLA replaces a
        # division by a constant with a multiplication by its reciprocal and folds the
        # two constant factors into one, x = P + (n - U) c with c = fl(Q fl(1 / V)). A
        # form whose factor n - U is not zero at r0 (for example U = low) then returns
        # r_c only when the rounding of c allows it, which fails for a fraction of wide
        # ranges. With n - r0 = 0 the product is 0 and x = P = r_c exactly. At the pair
        # x = r0 + (n - r0) is n exactly: n - r0 is exact (Sterbenz, both radii lie in
        # [r_cmb, r_surface] and so within a factor 2), and the sum n is representable.
        # One expression and no masked branch: no where in the traced map.
        if R and not cut:
            self._low_n = self._low[self._range_of]
            self._start_n = self._start[self._range_of]
            self._high_n = self._high[self._range_of]
            below, above = self._side > 0, self._side < 0
            self._P = np.where(below | above, self._r_c, self._start_n)
            self._U = self._start_n.copy()
            self._Q = np.where(below, self._r_c - self._low_n, np.where(above, self._high_n - self._r_c, 1.0))
            self._V = np.where(below, self._start_n - self._low_n, np.where(above, self._high_n - self._start_n, 1.0))

        # ---- 5. The knot table of rho0 (plan D1 item 6, D4). Under "cut" no node moves
        # and every node is data.
        movable_set = set(self.ranges)
        knot = origin >= 0                                         # every input node
        knot |= (kind == 2) & np.isin(owner, [k for k in range(1, B - 1) if k not in movable_set])
        if cut:
            knot[:] = True
        self.table_nodes = np.flatnonzero(knot)
        self._knots = radius[self.table_nodes]
        d = np.diff(self._knots)
        # The slope weight 1 / width, and a static zero at a pair of knots (never
        # selected by a moved node, and finite in every mode).
        self._inv_width = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)

        # ---- 6. Equal-value rule of the rho0 keyword (class docstring): pairs of
        # knots inside or under a movable range, and the members of duplicated nodes.
        equal_pairs = []
        for i in range(N - 1):
            if radius[i] != radius[i + 1]:
                continue
            if kind[i + 1] == 1:
                equal_pairs.append(i)                              # a duplicated node
            elif not cut and origin[i] >= 0 and origin[i + 1] >= 0 and any(
                    g.low < radius[i] < g.high for g in self.ranges.values()):
                equal_pairs.append(i)                              # an input pair inside a range
        self._equal_pairs = np.asarray(equal_pairs, dtype=int)

        # ---- 7. The density map (plan D11): the input nodes strictly inside a range.
        self.input_weights = self._trapezoid(r_in, profile.structure.pairs)
        dyn, dyn_range = [], []
        for q, k in enumerate(self.movable_index if not cut else []):
            g = self.ranges[k]
            idx = np.flatnonzero((r_in > g.low) & (r_in < g.high))
            dyn.append(idx); dyn_range.append(np.full(idx.size, q))
        self._dyn = np.concatenate(dyn).astype(int) if dyn else np.zeros(0, dtype=int)
        self._dyn_range = np.concatenate(dyn_range).astype(int) if dyn_range else np.zeros(0, dtype=int)
        self._r_dyn = r_in[self._dyn]
        if self._dyn.size:
            lo_n = np.array([self.ranges[k].lo_node for k in self.movable_index])
            hi_n = np.array([self.ranges[k].hi_node for k in self.movable_index])
            self._range_nodes = np.unique(np.concatenate([np.arange(a, b + 1) for a, b in zip(lo_n, hi_n)]))
            self._same_range = ((self._range_nodes[None, :] >= lo_n[self._dyn_range][:, None])
                                & (self._range_nodes[None, :] <= hi_n[self._dyn_range][:, None]))
            self._first_node = lo_n[self._dyn_range]

        # ---- 8. The construction profile: with the static limits of plan D3 under
        # "affine", with the viscosity cuts (their range nodes, in the order of
        # movable_index, and the construction radii as cut_radius) under "cut".
        cuts = tuple((self.ranges[k].lo_node, self.ranges[k].hi_node) for k in self.movable_index) if cut else ()
        structure = NodeStructure.from_radius(radius, consistent_core=profile.core_mass is None, cuts=cuts)
        if R and not cut:
            structure = dataclasses.replace(structure, limits=self._limits(structure))
        self.profile = RadialProfile(
            radius, rho0, eta, core_density=profile.core_density, core_mass=profile.core_mass,
            outer_density=profile.outer_density, surface_density=profile.surface_density,
            uniform_gravity=profile.uniform_gravity, cut_radius=self.start_radii.copy() if cut else None,
            structure=structure)

    # ------------------------------------------------------------------
    # Construction helpers (host, numpy).
    # ------------------------------------------------------------------

    @staticmethod
    def _checked_boundaries(r_in, boundaries) -> np.ndarray:
        """The boundary radii as a float64 vector, after the eager checks of plan 3.1."""
        start = np.asarray(boundaries, dtype=np.float64)
        if start.ndim != 1 or start.size < 2:
            raise ValueError(f"boundaries must be a vector of at least two radii in m (the CMB and the surface), "
                             f"got shape {start.shape}")
        if not np.all(np.isfinite(start)):
            raise ValueError(f"boundaries must be finite radii in m, got {start}")
        for k in range(1, start.size):
            if start[k] <= start[k - 1]:
                raise ValueError(f"boundaries must ascend strictly from the CMB: boundary {k} at {_km(start[k])} "
                                 f"is not above boundary {k - 1} at {_km(start[k - 1])}")
        if start[0] != r_in[0]:
            raise ValueError(f"the first boundary must be the CMB radius of the profile, {_km(r_in[0])}; "
                             f"got {_km(start[0])}")
        if start[-1] != r_in[-1]:
            raise ValueError(f"the last boundary must be the surface radius of the profile, {_km(r_in[-1])}; "
                             f"got {_km(start[-1])}")
        return start

    @staticmethod
    def _density_jumps(r_in, rho_in, low, high) -> tuple[float, ...]:
        """The radii in m of the jump pairs of the input table with a ``rho0`` jump
        strictly inside ``(low, high)``, ascending (host, numpy)."""
        return tuple(float(r_in[i]) for i in range(r_in.size - 1)
                     if r_in[i] == r_in[i + 1] and low < r_in[i] < high and rho_in[i] != rho_in[i + 1])

    def _movable_ranges(self, movable, r_in, rho_in, radius, jumps_in) -> dict:
        """Check every movable range against the rules of plan 3.1 and record it.

        ``jumps_in``: ``{k: density_jumps}`` of every range (``_density_jumps``). Under
        ``"affine"`` a density jump inside a range is refused; under ``"cut"`` it is
        allowed (NOTES/PLAN.md ruling 14) and recorded in ``MovableRange.density_jumps``."""
        ranges = {}
        start = self.boundaries
        cut = self.motion == "cut"
        movable_radii = {k: start[k] for k in movable}
        for k, (low, high) in sorted(movable.items()):
            b = start[k]
            if not low < b < high:
                raise ValueError(f"boundary {k} at {_km(b)} must lie strictly inside its movable range "
                                 f"({_km(low)}, {_km(high)})")
            for end in (low, high):
                # First the end at another movable boundary: under "cut" that boundary is
                # not a node, so the node check below would hide the reason.
                for k2, b2 in movable_radii.items():
                    if k2 != k and end == b2:
                        raise ValueError(f"the end {_km(end)} of the movable range of boundary {k} is the radius "
                                         f"of the movable boundary {k2}; a range end must stay where it is")
                if not np.any(radius == end):
                    raise ValueError(f"the end {_km(end)} of the movable range of boundary {k} is not a node "
                                     "radius of the construction table; a range end must be a node (or the "
                                     "pair of a fixed boundary), so that it stays where it is when the "
                                     "boundary moves")
            # Under "affine" a pair of the input with a rho0 jump strictly inside the range
            # would move the density structure with the nodes (lovejx "elastic
            # interface"). Under "cut" no node moves, and the density jumps stay where
            # the input table puts them.
            jumps = jumps_in[k]
            if jumps and not cut:
                i = int(np.flatnonzero(r_in == jumps[0])[0])
                raise ValueError(f"the movable range ({_km(low)}, {_km(high)}) of boundary {k} holds the density "
                                 f"jump pair of the input at {_km(r_in[i])} (rho0 {rho_in[i]} to "
                                 f"{rho_in[i + 1]} kg/m^3); under motion='affine' the nodes of a range move with "
                                 "its boundary and would carry the jump along, so the boundary cannot cross a "
                                 "density jump: use motion='cut' (the default), or choose range ends that leave "
                                 "it outside")
            others = [k2 for k2 in range(start.size) if k2 != k and low < start[k2] < high]
            if others:
                k2 = others[0]
                raise ValueError(f"boundary {k2} at {_km(start[k2])} lies inside the movable range "
                                 f"({_km(low)}, {_km(high)}) of boundary {k}; the range would move its pair")
            lo_node = int(np.flatnonzero(radius == low)[-1])        # inner member at a pair
            hi_node = int(np.flatnonzero(radius == high)[0])
            if cut:
                empty = np.zeros(0, dtype=int)
                ranges[k] = MovableRange(index=k, low=low, high=high, start=float(b),
                                         margin=CLAMP_FRACTION * (high - low), below=empty, above=empty,
                                         members=(-1, -1), radii_below=np.zeros(0), radii_above=np.zeros(0),
                                         lo_node=lo_node, hi_node=hi_node, motion="cut", density_jumps=jumps)
                continue
            lower = int(np.flatnonzero(radius == b)[0])
            below = np.arange(lo_node + 1, lower)
            above = np.arange(lower + 2, hi_node)
            ranges[k] = MovableRange(index=k, low=low, high=high, start=float(b),
                                     margin=CLAMP_FRACTION * (high - low), below=below, above=above,
                                     members=(lower, lower + 1), radii_below=radius[below],
                                     radii_above=radius[above], lo_node=lo_node, hi_node=hi_node,
                                     motion="affine", density_jumps=())
        ordered = sorted(ranges.values(), key=lambda g: g.low)
        for first, second in zip(ordered[:-1], ordered[1:]):
            if second.low < first.high:
                raise ValueError(f"the movable ranges of boundaries {first.index} and {second.index} overlap "
                                 f"between {_km(second.low)} and {_km(first.high)}; a node can follow one "
                                 "boundary only")
        return ranges

    @staticmethod
    def _trapezoid(r, pairs) -> np.ndarray:
        """Trapezoid weights of a concrete node table, m, numpy (``fields.quadrature_weights``
        on the host: zero width at a pair, each node of a pair the half interval on its side)."""
        delta = np.diff(r)
        delta[np.asarray(pairs, dtype=int) - 1] = 0.0
        z = np.zeros(1)
        return 0.5 * (np.concatenate([delta, z]) + np.concatenate([z, delta]))

    def _numpy_map(self, new_radii, rho0=None) -> tuple[np.ndarray, np.ndarray]:
        """The map of ``__call__`` in numpy for the movable radii ``new_radii`` (R,), m:
        the radius and ``rho0`` of the construction table. Host only; used for the
        static limits (the clamp is applied as in the traced map)."""
        rho0 = self._construction_rho0 if rho0 is None else np.asarray(rho0, dtype=np.float64)
        new = np.clip(np.asarray(new_radii, dtype=np.float64), self._low + self._margin, self._high - self._margin)
        n = new[self._range_of]
        x = self._P + self._Q * ((n - self._U) / self._V)
        vals = rho0[self.table_nodes]
        k = np.sum(x[:, None] >= self._knots[None, 1:-1], axis=1)
        slope = (vals[1:] - vals[:-1]) * self._inv_width
        radius = self._construction_radius.copy()
        rho = rho0.copy()
        radius[self.moved_nodes] = x
        rho[self.moved_nodes] = vals[k] + (x - self._knots[k]) * slope[k]
        return radius, rho

    def crossings(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """The radii of boundary ``index`` at which the derivative of the map jumps,
        inside the closed clamp range, ascending, and their kinds (host, numpy).

        Under ``motion="cut"`` (the default) no node moves, and two kinds exist. Kind 2:
        the cut over an input radius inside the clamp range that is not a density-jump
        pair (the cut changes interval; the map is continuously differentiable there under
        an incompressible formulation, with a kink of about 2e-4 relative under TALA).
        Kind 4: the cut over a density-jump pair of the input
        (``MovableRange.density_jumps``; continuously differentiable under an
        incompressible formulation, a jump of the value under TALA). A pair is one radius.

        Under ``motion="affine"`` three kinds exist. Kind 1: a moved node of the range
        over an input radius strictly inside the range (the knot table of ``rho0`` and the
        bracket of the density map change interval). Kind 2: the pair over such an input
        radius (an input node changes side of the viscosity jump). Kind 3: the
        construction radius ``r0``, where every moved node sits on its own input radius. A
        node ``j`` below the boundary crosses the input radius ``r_k`` at
        ``r = low + (r_k - low) (r0 - low) / (r_j - low)``; a node above at
        ``r = high - (high - r_k) (high - r0) / (high - r_j)``.

        Parameters
        ----------
        index
            The boundary index of a movable boundary (a key of ``ranges``).

        Returns
        -------
        tuple of numpy arrays
            ``(radii, kinds)``: ``(C,)`` float64 in m and ``(C,)`` int.
        """
        g = self.ranges[int(index)]
        r_in = self._r_in
        if g.motion == "cut":
            lo, hi = g.low + g.margin, g.high - g.margin
            kn = np.unique(r_in[(r_in >= lo) & (r_in <= hi)])
            kd = np.where(np.isin(kn, np.asarray(g.density_jumps)), 4, 2).astype(int)
            return kn, kd
        kn = np.unique(r_in[(r_in > g.low) & (r_in < g.high)])
        c, kd = [], []
        for rc in g.radii_below:
            c.append(g.low + (kn - g.low) * (g.start - g.low) / (rc - g.low)); kd.append(np.ones(kn.size, int))
        for rc in g.radii_above:
            c.append(g.high - (g.high - kn) * (g.high - g.start) / (g.high - rc)); kd.append(np.ones(kn.size, int))
        c.append(kn); kd.append(np.full(kn.size, 2))
        c = np.concatenate(c) if c else np.zeros(0)
        kd = np.concatenate(kd) if kd else np.zeros(0, int)
        lo, hi = g.low + g.margin, g.high - g.margin
        ok = (c >= lo) & (c <= hi) & (c != g.start)
        c = np.concatenate([c[ok], [g.start]])
        kd = np.concatenate([kd[ok], [3]])
        o = np.argsort(c, kind="stable")
        return c[o], kd[o]

    def _limits(self, structure: NodeStructure) -> NodeLimits:
        """The static limits of plan D3 (host, numpy).

        width: the interval widths in ``ln r`` increase with the stretch of their side, so
        their largest value over a range is at a clamp bound; each range is evaluated at
        its two clamp bounds with the other boundaries at their construction radii (an
        interval belongs to at most one range), and the elementwise maximum with the
        construction widths is taken.

        kappa_h: on a segment of boundary positions between two neighbouring crossings
        both node densities of an interval are linear in the position ``n``, so
        ``d ln(a/b)/dn = (a1 b0 - b1 a0) / (a b)`` has one sign there and ``|ln(a/b)|`` takes
        its maximum at a segment end. The limit is the exact maximum over the clamp bounds
        and every crossing of each range. Zero at a pair.

        rho0_max: the largest construction ``rho0`` (an interpolation stays inside the
        range of its knot values).
        """
        mask = structure.interval_mask
        r0 = self._construction_radius

        def widths(r):
            return np.log1p(np.diff(r) / r[:-1]) * mask

        def kappa(rho):
            return np.abs(np.log(rho[1:] / rho[:-1])) * mask

        w_lim = widths(r0)
        k_lim = kappa(self._construction_rho0)
        for q, k in enumerate(self.movable_index):
            g = self.ranges[k]
            positions = np.concatenate([[g.low + g.margin, g.high - g.margin], self.crossings(k)[0]])
            for pos in positions:
                new = self._start.copy()
                new[q] = pos
                r, rho = self._numpy_map(new)
                k_lim = np.maximum(k_lim, kappa(rho))
                if pos in (g.low + g.margin, g.high - g.margin):
                    w_lim = np.maximum(w_lim, widths(r))
        return NodeLimits(width=tuple(float(v) for v in w_lim), kappa_h=tuple(float(v) for v in k_lim),
                          rho0_max=float(np.max(self._construction_rho0)))

    # ------------------------------------------------------------------
    # Static properties.
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        """The number of layers, ``B - 1``."""
        return int(self.boundaries.size) - 1

    @property
    def width_factor(self) -> float:
        """The largest ratio of the width limit of an interval to its construction width
        (1.0 without a movable boundary and under ``"cut"``, where no node moves and the
        structure has no limits). Under ``"affine"`` the Pade plan is built for the
        limits, so a large factor costs squarings and, with them, rounding: a factor 4 to
        17 in the error of the kernels for three more squarings, up to 1.6e-12 at a width
        factor of 7.7 (NOTES/chunks/c4-controls-derivatives/04-review-physics-r2.md, minor
        1). Under ``"affine"``, put the construction radius near the middle of its range
        (plan D3)."""
        lim = self.profile.structure.limits
        if lim is None:
            return 1.0
        r = self._construction_radius
        h = np.log1p(np.diff(r) / r[:-1]) * self.profile.structure.interval_mask
        w = np.asarray(lim.width)
        ok = h > 0
        return float(np.max(w[ok] / h[ok])) if np.any(ok) else 1.0

    def pade_plan(self, config: StokesConfig):
        """The static Pade plan of the interval exponentials for ``config``
        (``propagate.static_plan`` of the construction structure, with its limits).

        Parameters
        ----------
        config
            The ``StokesConfig`` of the kernels.
        """
        from . import propagate
        return propagate.static_plan(self.profile.structure, config).pade

    def full_boundaries(self, movable_radii) -> jax.Array:
        """The ``(B,)`` boundary vector from the movable radii alone (the fixed entries are
        the construction radii, bit for bit). A sampler's control vector then holds only
        radii that have a derivative.

        Parameters
        ----------
        movable_radii
            ``(R,)`` radii in m, one per movable boundary in the order of
            ``movable_index``; concrete or traced.
        """
        x = jnp.asarray(movable_radii, dtype=jnp.float64)
        if x.shape != (self.movable_index.size,):
            raise ValueError(f"movable_radii must hold one radius in m per movable boundary, shape "
                             f"({self.movable_index.size},), got {x.shape}")
        return jnp.asarray(self.boundaries).at[self.movable_index].set(x)

    # ------------------------------------------------------------------
    # The pure map (plan D2, D4).
    # ------------------------------------------------------------------

    def __call__(self, boundaries, log_eta, *, rho0=None) -> RadialProfile:
        """The profile of the controls.

        Parameters
        ----------
        boundaries
            ``(B,)`` radii in m, concrete or traced: a NumPy or JAX array, or a list or
            tuple of numbers (a list that holds a tracer is stacked into one traced
            array). Only the movable entries are read under a trace; see the class
            docstring for the checks and the clamp. ``full_boundaries`` builds the vector
            from the movable radii alone.
        log_eta
            ``(n_layers,)`` natural log of the layer viscosities in Pa s; when concrete it
            must be finite.
        rho0
            ``None`` (the construction ``rho0``) or ``(N,)`` kg/m^3 of the construction
            table: the knot values and the fixed nodes (class docstring).

        Returns
        -------
        ``RadialProfile`` with the construction structure and the scalar leaves of the
        input profile (set others with ``replace``).

        - Under ``"affine"`` (and without a movable boundary): the moved radii, ``rho0``
          from the knot table at the moved nodes (the ``rho0`` control or the
          construction ``rho0`` elsewhere), ``eta = exp(log_eta)[layer_of]``.
        - Under ``"cut"``: the radii of the construction table, ``rho0`` of the control
          (or of the construction) at every node, ``eta = exp(log_eta)[layer]`` with the
          layer of each node from its side of the clamped cut radii (a node on a cut is
          below it), and the leaf ``cut_radius`` holding the clamped radii, ``(R,)`` m.
        """
        values = jnp.asarray(log_eta, dtype=jnp.float64)
        cut = self.motion == "cut"
        if values.shape != (self.n_layers,):
            raise ValueError(f"log_eta must hold one natural-log viscosity in Pa s per layer: {self.n_layers} "
                             f"layers, shape ({self.n_layers},), got {values.shape}")
        # A list or tuple: stack it into one array first. With a tracer inside, the
        # host function np.shape cannot read it (TracerArrayConversionError), and the
        # concrete check below needs one array in either case.
        if isinstance(boundaries, (list, tuple)):
            boundaries = (jnp.asarray(boundaries, dtype=jnp.float64) if any(_is_traced(b) for b in boundaries)
                          else np.asarray(boundaries, dtype=np.float64))
        if np.shape(boundaries) != self.boundaries.shape:
            raise ValueError(f"boundaries must have shape {self.boundaries.shape}, one radius in m per boundary, "
                             f"got {np.shape(boundaries)}")
        if rho0 is not None and np.shape(rho0) != (self.n_nodes,):
            raise ValueError(f"rho0 must have the shape ({self.n_nodes},) of the construction table, got "
                             f"{np.shape(rho0)}")
        if not _is_traced(boundaries):
            self._check_concrete_boundaries(np.asarray(boundaries, dtype=np.float64))
        if not _is_traced(log_eta) and not np.all(np.isfinite(np.asarray(log_eta, dtype=np.float64))):
            raise ValueError(f"log_eta must be finite, got {np.asarray(log_eta)}")
        if rho0 is not None and not _is_traced(rho0) and self._equal_pairs.size:
            r = np.asarray(rho0, dtype=np.float64)
            bad = self._equal_pairs[r[self._equal_pairs] != r[self._equal_pairs + 1]]
            if bad.size:
                i = int(bad[0])
                raise ValueError(f"rho0 has different values {r[i]} and {r[i + 1]} kg/m^3 on the two nodes {i} and "
                                 f"{i + 1} at {_km(self._construction_radius[i])}, a duplicated node (or, under "
                                 "motion='affine', a pair without a density jump in or under a movable range); "
                                 "its two members must carry one value (under 'affine' the moved nodes would "
                                 "otherwise meet a jump of the knot table)")
        rho_ctrl = (jnp.asarray(self._construction_rho0) if rho0 is None
                    else jnp.asarray(rho0, dtype=jnp.float64))
        if cut:
            # The viscosity cuts: no node moves. The clamp as for the affine method
            # (derivative 1 inside, 0 outside, 1/2 at a bound). The layer of every node is
            # the static count of the paired boundaries below it plus the number of cuts
            # below it, one comparison of all nodes with all cuts (a node on a cut is
            # below it). For a node outside the range of a cut the comparison has the
            # same result at every admissible position, so it needs no mask. The
            # comparison has no derivative: the derivative with respect to log_eta passes
            # the gather (whose reverse-mode transpose sums the node derivatives of a
            # layer into its control), the one with respect to a cut radius passes
            # cut_radius into the solver (propagate.interval_steps).
            b = jnp.asarray(boundaries, dtype=jnp.float64)
            new = jnp.clip(b[self.movable_index], self._low + self._margin, self._high - self._margin)   # (R,)
            above = jnp.asarray(self._construction_radius)[:, None] > new[None, :]                     # (N, R)
            layer = jnp.asarray(self._fixed_layer) + jnp.sum(above, axis=1)
            return self.profile.replace(eta=jnp.exp(values)[layer], rho0=rho_ctrl, cut_radius=new)
        # One exponential per layer, then the gather; the reverse-mode transpose of the
        # gather sums the node derivatives of a layer into its control.
        eta = jnp.exp(values)[self.layer_of]
        if not self.movable_index.size:
            return self.profile.replace(eta=eta, rho0=rho_ctrl)
        b = jnp.asarray(boundaries, dtype=jnp.float64)
        # The clamp: derivative 1 inside, 0 outside, 1/2 at a bound (jnp.clip).
        new = jnp.clip(b[self.movable_index], self._low + self._margin, self._high - self._margin)   # (R,)
        n = new[self._range_of]                                                                       # (M,)
        # The two affine maps of the class docstring and the pair in one form with static
        # per-node coefficients (built in __init__), as an offset from the construction
        # radius: P = r_c, U = r0. At n == r0 the factor n - U is an exact zero, so every
        # moved node is its construction radius bit for bit there, also after XLA folds
        # the constant factors Q and 1 / V into one multiplication. The denominators are
        # static and nonzero, so every mode is finite.
        x = self._P + self._Q * ((n - self._U) / self._V)
        radius = jnp.asarray(self._construction_radius).at[self.moved_nodes].set(x)
        rho = rho_ctrl.at[self.moved_nodes].set(self._interp(x, rho_ctrl[self.table_nodes]))
        return self.profile.replace(radius=radius, rho0=rho, eta=eta)

    def _interp(self, x, values) -> jax.Array:
        """Piecewise-linear interpolation of the knot values at ``x`` (plan D4).

        The interpolation function of ``jax.numpy`` lowers its bisection search to a
        ``scan``; this map has no loop. Every moved node is compared with every interior
        knot (``M x K`` comparisons), and the
        interval index is the count of knots at or below ``x``: at a knot the interval
        above it is selected, so the derivative there is the upper slope, the slope for
        an increasing boundary radius. The slope weight ``1 / width`` is static, zero at a
        pair of knots, so the values (a ``rho0`` control) have a derivative and no
        division by a traced number occurs.
        """
        knots = jnp.asarray(self._knots)
        k = jnp.sum(x[:, None] >= knots[None, 1:-1], axis=1)
        slope = (values[1:] - values[:-1]) * jnp.asarray(self._inv_width)
        return values[k] + (x - knots[k]) * slope[k]

    def _check_concrete_boundaries(self, given) -> None:
        """Refuse a fixed boundary that moved or a movable one outside its open range."""
        for k, (value, b0) in enumerate(zip(given, self.boundaries)):
            if k in self.ranges:
                g = self.ranges[k]
                if not g.low < value < g.high:
                    raise ValueError(f"boundary {k} at {_km(value)} lies outside its movable range "
                                     f"({_km(g.low)}, {_km(g.high)}), which is open")
            elif value != b0:
                raise ValueError(f"boundary {k} is fixed at {_km(b0)}; got {_km(value)} (only the boundaries "
                                 f"{sorted(self.ranges)} are movable)")

    # ------------------------------------------------------------------
    # The density map (plan D11).
    # ------------------------------------------------------------------

    def _bracket(self, radius) -> tuple[jax.Array, jax.Array]:
        """For every dynamic input node, the construction index ``a`` with
        ``x_a < r_i <= x_(a+1)`` among the nodes of its range, and the weight
        ``t = (r_i - x_a) / (x_(a+1) - x_a)``.

        One comparison of the dynamic input radii with the union of the range nodes, a
        static mask that keeps the nodes of the own range, and a count. The range nodes of
        one range are contiguous construction indices from ``first_node`` (the inner
        member at a pair end) and ascending in radius, so the count of nodes below ``r_i``
        locates the bracket. The strict comparison gives ``t = 1`` at a tie, the interval
        below the tied node, which is valid when the node moves up: the upper one-sided
        derivative, the side of the knot table. ``x_(a+1) > x_a`` strictly, so no division
        by zero (a pair interval cannot be selected); the range ends are construction
        nodes with ``x = low < r_i`` and ``x = high > r_i``, so ``a`` stays in the range.
        """
        xs = radius[self._range_nodes]
        count = jnp.sum(self._same_range & (xs[None, :] < self._r_dyn[:, None]), axis=1)
        a = self._first_node + count - 1
        t = (self._r_dyn - radius[a]) / (radius[a + 1] - radius[a])
        return a, t

    def _to_input(self, K, a, t):
        """A ``(..., N)`` array on the construction nodes to ``(..., N_in)`` at the input
        radii: the static gather, and at the dynamic input nodes the interpolation with
        the bracket ``(a, t)``."""
        out = K[..., self.input_node]
        if not self._dyn.size:
            return out
        return out.at[..., self._dyn].set((1.0 - t) * K[..., a] + t * K[..., a + 1])

    def input_kernels(self, kernels: Kernels) -> Kernels:
        """The kernels at the radii of the input table (plan D11).

        Parameters
        ----------
        kernels
            ``Kernels`` of a profile of this ``Layers`` (``stokes_kernels(layers(...),
            config)``); the current radius is read from ``kernels.radius``.

        Returns
        -------
        Kernels
            Every kernel ``(L, N_in)``, ``degree_one_imbalance`` ``(N_in,)`` mapped with
            the same weights, ``radius`` the static input radius, and the degrees and the
            frame of the input.

        Notes
        -----
        Under ``"cut"`` (the default) and without a
        movable boundary every input node takes the kernel of its construction node.
        Under ``"affine"`` an input node outside every movable range (the range ends
        included) does the same, and an input node strictly inside a range takes the
        linear interpolation in ``r`` between the two moved nodes that bracket its radius
        (``_bracket``). At the construction radii
        every dynamic input node is at a tie with its own node, the weight is exactly 1,
        and the map equals the static gather bit for bit, with one exception: the upper
        member of an input viscosity-only pair inside a movable range (or under a
        movable boundary) takes the kernel of the lower member, because the two members
        sit at one radius and the bracket ties both with the lower one. The two kernels
        are equal in exact arithmetic (measured difference 1.6e-15 to 3.4e-15 in the
        relative row norm). The kernel of an input node is
        its sensitivity times its weight: ``d F / d drho[i, l, m] = K_in[l, i] w_i``.

        Raises ``ValueError`` for kernels whose node count is not ``N``.
        """
        n = int(np.shape(kernels.geoid)[-1])
        if n != self.n_nodes:
            raise ValueError(f"the kernels have {n} nodes, the construction table of this Layers {self.n_nodes}; "
                             "pass the kernels of a profile returned by this Layers")
        a = t = None
        if self._dyn.size:
            a, t = self._bracket(jnp.asarray(kernels.radius, dtype=jnp.float64))
        stacked = jnp.stack([jnp.asarray(kernels.geoid), jnp.asarray(kernels.surface_topography),
                             jnp.asarray(kernels.cmb_topography), jnp.asarray(kernels.surface_velocity)])
        mapped = self._to_input(stacked, a, t)
        imb = kernels.degree_one_imbalance
        imb = None if imb is None else self._to_input(jnp.asarray(imb), a, t)
        return Kernels(degrees=kernels.degrees, degree_one_frame=kernels.degree_one_frame,
                       radius=jnp.asarray(self._r_in), geoid=mapped[0], surface_topography=mapped[1],
                       cmb_topography=mapped[2], surface_velocity=mapped[3], degree_one_imbalance=imb)

    def fields(self, kernels: Kernels, density) -> Fields:
        """The four fields of a density anomaly given on the input table (plan D11):
        ``fields.apply(self.input_kernels(kernels), density, self.input_profile)``.

        A boundary that moves does not move the anomaly. Pass the density to a jitted
        function as an argument (``fields.apply``, compile time).

        Parameters
        ----------
        kernels
            ``Kernels`` of a profile of this ``Layers``.
        density
            ``(N_in, *batch, Ld+1, Ld+1)`` in kg/m^3 under the layout rule of
            ``stokesjx.sh`` (CONVENTIONS 15), with the radii of ``input_profile``; a first
            axis that is not ``N_in`` raises ``ValueError``.
        """
        return _apply_fields(self.input_kernels(kernels), density, self.input_profile)

    def sheet_mass(self, density, kernels: Kernels) -> jax.Array:
        """The transposed form of the density map: the sheet masses ``w_i drho[i]`` of the
        input nodes, kg/m^2, deposited onto the construction nodes, ``(N, *batch, Ld+1,
        Ld+1)``.

        Under ``"cut"`` (the default) every input node puts its sheet mass on its own
        construction node. Under ``"affine"`` an input node outside every movable range
        does the same, and an input node strictly inside a range splits it with the
        weights ``1 - t`` and ``t`` onto the two moved nodes that bracket its radius (the
        current radius is read from ``kernels.radius``). ``fields.radial_integral(kernels, sheet_mass, ones(N))``
        equals ``self.fields(kernels, density)`` to rounding. It costs ``O(N L^2)`` per
        batch entry against ``O(4 L N_in)`` of ``input_kernels``; use it for a density on
        the construction table and for the adjoint test of the map.

        Parameters
        ----------
        density
            ``(N_in, *batch, Ld+1, Ld+1)`` in kg/m^3 on the input table.
        kernels
            ``Kernels`` of a profile of this ``Layers``: the current radius is read from
            ``kernels.radius``.
        """
        d = jnp.asarray(density)
        if d.ndim < 3 or d.shape[0] != self._r_in.size:
            raise ValueError(f"the density has the radial axis first on the input table, (N_in, ..., Ld+1, Ld+1) "
                             f"with N_in = {self._r_in.size}; got the shape {d.shape}")
        n = int(np.shape(kernels.geoid)[-1])
        if n != self.n_nodes:
            raise ValueError(f"the kernels have {n} nodes, the construction table of this Layers {self.n_nodes}")
        w = jnp.asarray(self.input_weights).reshape((-1,) + (1,) * (d.ndim - 1))
        mass = w * d
        out = jnp.zeros((self.n_nodes,) + d.shape[1:], dtype=mass.dtype)
        static = np.setdiff1d(np.arange(self._r_in.size), self._dyn)
        out = out.at[self.input_node[static]].add(mass[static])
        if self._dyn.size:
            a, t = self._bracket(jnp.asarray(kernels.radius, dtype=jnp.float64))
            tt = t.reshape((-1,) + (1,) * (d.ndim - 1))
            md = mass[self._dyn]
            out = out.at[a].add((1.0 - tt) * md).at[a + 1].add(tt * md)
        return out


# ----------------------------------------------------------------------
# Per-node viscosity map (plan D6).
# ----------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class NodeViscosity:
    """A control map from one natural-log viscosity per node to a profile. Built by
    ``viscosity_by_node``. A callable, not a pytree; a jitted misfit closes over it.
    The call ``nv(log_eta)`` returns ``profile.replace(eta=exp(log_eta))``.

    Parameters
    ----------
    profile
        The concrete template profile; every other leaf is kept.
    size
        The number of controls ``N`` (every node, both members of a pair separately).
    initial
        ``log(profile.eta)``, ``(N,)``.
    """

    profile: RadialProfile
    size: int
    initial: jax.Array

    def __call__(self, log_eta) -> RadialProfile:
        """``profile.replace(eta=exp(log_eta))``; ``log_eta`` ``(N,)`` in ln(Pa s), concrete
        or traced. Raises ``ValueError`` for another shape (a scatter or a broadcast would
        otherwise hide it)."""
        values = jnp.asarray(log_eta, dtype=jnp.float64)
        if values.shape != (self.size,):
            raise ValueError(f"log_eta must hold one natural-log viscosity in Pa s per node, shape ({self.size},), "
                             f"got {values.shape}")
        return self.profile.replace(eta=jnp.exp(values))


def viscosity_by_node(profile: RadialProfile) -> NodeViscosity:
    """The per-node viscosity map of a concrete profile (plan D6).

    Every node is a control: the instantaneous Stokes problem has no elastic lid and no
    core node, and each node of a pair is its own control. The node table does not move,
    so a density on this table goes to ``fields.apply`` with the profile itself.

    Parameters
    ----------
    profile
        A concrete ``RadialProfile`` (a traced ``eta`` raises ``ValueError``).

    Returns
    -------
    NodeViscosity
        The map ``log_eta (N,) -> RadialProfile``.
    """
    if not isinstance(profile, RadialProfile):
        raise ValueError(f"viscosity_by_node takes a RadialProfile, got {type(profile).__name__}")
    if _is_traced(profile.eta):
        raise ValueError("viscosity_by_node takes a concrete profile")
    eta = np.asarray(profile.eta, dtype=np.float64)
    return NodeViscosity(profile=profile, size=int(eta.size), initial=jnp.asarray(np.log(eta)))


# ----------------------------------------------------------------------
# The flat vector of the controls (plan D7).
# ----------------------------------------------------------------------


def _path_name(path) -> str:
    """A readable name of a pytree path: dict keys and attribute names joined by dots,
    sequence indices in brackets."""
    parts = []
    for p in path:
        if isinstance(p, jax.tree_util.DictKey):
            parts.append(str(p.key))
        elif isinstance(p, jax.tree_util.GetAttrKey):
            parts.append(p.name)
        elif isinstance(p, jax.tree_util.SequenceKey):
            parts.append(f"<{p.idx}>")
        else:
            parts.append(str(p))
    return ".".join(parts) if parts else "x"


class _NumpyFunction:
    """A jitted function of the control vector with bound data, called with NumPy.

    ``f(x)`` converts ``x`` with ``np.asarray(x, dtype=np.float64)``, checks the shape
    ``(size,)``, calls the compiled function with the bound data as arguments, and returns
    host values (``float`` and a NumPy array). ``with_data(*data)`` binds new data of the
    same shapes to the same compiled function: no new trace or compile."""

    def __init__(self, compiled, size: int, data, gradient: bool):
        self._compiled = compiled
        self._size = size
        self._data = jax.tree_util.tree_map(jnp.asarray, tuple(data))
        self._gradient = gradient

    def with_data(self, *data) -> "_NumpyFunction":
        """The same compiled function with new data (same shapes and dtypes)."""
        return _NumpyFunction(self._compiled, self._size, data, self._gradient)

    def __call__(self, x):
        v = np.asarray(x, dtype=np.float64)
        if v.shape != (self._size,):
            raise ValueError(f"the control vector must have the shape ({self._size},), got {v.shape}")
        # A JAX array of the host vector, so that every call meets one entry of the jit
        # cache (a NumPy argument and a JAX argument are two entries of the fast path).
        out = self._compiled(jnp.asarray(v), *self._data)
        if self._gradient:
            value, grad = out
            return float(value), np.array(grad)
        return float(out)


class ControlVector:
    """A pytree of controls as one float64 vector, and the jitted value and gradient of a
    user's scalar function of it (plan D7).

    ``ControlVector(example)`` takes a pytree of float64 arrays (for example
    ``{"radii": (R,), "log_eta": (n_layers,)}``) and fixes its structure and shapes. The
    order of the vector is that of ``jax.flatten_util.ravel_pytree`` (the leaves in
    ``jax.tree_util.tree_leaves`` order, a dict by its sorted keys, each leaf raveled in C
    order); ``names`` gives one name per entry.

    ``fn(controls, *data)`` is the user's scalar function: a log density or a misfit; the
    sign is the caller's, and the prior is the caller's. ``data`` are arguments of the
    compiled function and are not differentiated (a density, observed fields, weights):
    pass them as arguments, never close over them, because a closed-over array is a
    constant of the program and costs compile time (0.3 s at 257 nodes, ``lmax = 100``,
    NOTES/chunks/c3-flags-fields/03-implementation.md).

    The jitted functions are cached per ``fn`` inside the ``ControlVector``: a second call
    of ``value_and_grad(fn)`` returns the same jitted function, and the NumPy fronts of
    one ``fn`` share it, so new data of the same shapes (``numpy_value_and_grad(fn,
    *new)`` or ``.with_data(*new)``) neither trace nor compile again. The key of the
    cache is the identity of ``fn``, and the cache holds a reference to it: keep one
    function object and pass it at every call. A new lambda at every call traces and
    compiles again, and each one adds an entry to the cache. ``value_and_grad(fn)``
    is a pure function of the vector, so ``jax.vmap`` over a batch of vectors and
    ``jax.jit`` around it work: this is the form of a BlackJAX log density, a NumPyro
    ``potential_fn`` and a hand-written leapfrog.

    For a sampler: the vector mixes radii in m with log viscosities, so a sampler needs a
    mass matrix (or the caller scales the radii, for example in km); outside
    ``Layers.radius_bounds`` the radius derivative is exactly zero, so the prior must
    have its support inside these bounds; the derivative with respect to a movable
    radius has jumps at the crossings documented in ``Layers``.

    Parameters
    ----------
    example
        A pytree of float64 arrays that fixes the structure and the shapes of the
        controls; another dtype raises ``TypeError``.

    Attributes
    ----------
    size
        The length of the control vector.
    names
        One name per entry of the vector (the pytree path and the index of the entry).
    """

    def __init__(self, example):
        leaves, _ = jax.tree_util.tree_flatten(example)
        for leaf in leaves:
            dt = np.asarray(leaf).dtype
            if dt != np.float64:
                raise TypeError(f"every control must be a float64 array; a leaf has dtype {dt}")
        flat, self._unravel = jax.flatten_util.ravel_pytree(
            jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=jnp.float64), example))
        self.size = int(flat.size)
        names = []
        for path, leaf in jax.tree_util.tree_flatten_with_path(example)[0]:
            base = _path_name(path)
            shape = np.shape(leaf)
            if shape == ():
                names.append(base)
            else:
                names += [f"{base}[{','.join(str(i) for i in idx)}]" for idx in np.ndindex(*shape)]
        self.names = tuple(names)
        self._cache: dict = {}

    def ravel(self, controls) -> jax.Array:
        """The ``(size,)`` vector of a controls pytree of the example's structure.

        Parameters
        ----------
        controls
            A pytree of the structure and the shapes of the example.
        """
        flat, _ = jax.flatten_util.ravel_pytree(controls)
        if flat.shape != (self.size,):
            raise ValueError(f"the controls have {flat.size} entries, the example {self.size}")
        return flat

    def unravel(self, vector):
        """The controls pytree of a ``(size,)`` vector.

        Parameters
        ----------
        vector
            ``(size,)`` float64.
        """
        v = jnp.asarray(vector, dtype=jnp.float64)
        if v.shape != (self.size,):
            raise ValueError(f"the control vector must have the shape ({self.size},), got {v.shape}")
        return self._unravel(v)

    def value_and_grad(self, fn: Callable) -> Callable:
        """Jitted ``(vector, *data) -> (scalar, (size,) gradient)`` of ``fn(controls, *data)``.

        Parameters
        ----------
        fn
            The user's scalar function ``fn(controls, *data)``; keep one function object
            (the cache key is its identity).
        """
        key = (fn, "value_and_grad")
        if key not in self._cache:
            unravel = self._unravel

            def vg(vector, *data):
                return jax.value_and_grad(lambda v: fn(unravel(v), *data))(vector)
            self._cache[key] = jax.jit(vg)
        return self._cache[key]

    def value(self, fn: Callable) -> Callable:
        """Jitted ``(vector, *data) -> scalar`` of ``fn(controls, *data)``.

        Parameters
        ----------
        fn
            The user's scalar function ``fn(controls, *data)``; keep one function object
            (the cache key is its identity).
        """
        key = (fn, "value")
        if key not in self._cache:
            unravel = self._unravel
            self._cache[key] = jax.jit(lambda vector, *data: fn(unravel(vector), *data))
        return self._cache[key]

    def numpy_value_and_grad(self, fn: Callable, *data) -> _NumpyFunction:
        """``np (size,) -> (float, np.ndarray)``: the value and the gradient of
        ``fn(controls, *data)`` for samplers and optimisers outside JAX (for example
        ``scipy.optimize.minimize(f, x0, jac=True)``). The data are bound as device arrays
        and passed to the cached compiled function as arguments.

        Parameters
        ----------
        fn
            The user's scalar function ``fn(controls, *data)``; keep one function object
            (the cache key is its identity).
        data
            The arguments after the controls, bound as device arrays.
        """
        return _NumpyFunction(self.value_and_grad(fn), self.size, data, gradient=True)

    def numpy_value(self, fn: Callable, *data) -> _NumpyFunction:
        """``np (size,) -> float``: the value of ``fn(controls, *data)``.

        Parameters
        ----------
        fn
            The user's scalar function ``fn(controls, *data)``; keep one function object
            (the cache key is its identity).
        data
            The arguments after the controls, bound as device arrays.
        """
        return _NumpyFunction(self.value(fn), self.size, data, gradient=False)
