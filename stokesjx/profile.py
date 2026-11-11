"""The radial profile: node table, reference density, viscosity, core and water.

A profile is a table of ``N`` nodes from the core-mantle boundary (CMB, index 0) to the
surface (index ``N - 1``). Two neighbouring nodes with the same radius are a "jump
pair": an interval of zero width across which the viscosity, the density or both jump
(NOTES/docs/equations.md section 12). The material of an interval is the mean of its
two node values (eq. 12.1).

The module holds two objects:

``NodeStructure``
    The static part: the construction radii, the positions of the pairs and the core
    model (consistent or split), for a table with movable boundaries the static
    limits of the moved nodes (``NodeLimits``), and for a table with viscosity cuts the
    node ranges of the cuts (``cuts``). It is frozen and hashable, it is read
    on the host, and it is part of the static key of every jitted solve. The Pade plan
    of the interval exponentials (``linalg.pade_plan``) is computed from its widths.
``RadialProfile``
    A JAX pytree of the nine arrays that a derivative can move (radius, ``rho0``,
    ``eta``, core density, core mass, water density, topography density, uniform
    gravity, the radii of the viscosity cuts), with the ``NodeStructure`` as static
    auxiliary data.

The profile carries no gravity array. Gravity is a function of the background flag
of the formulation, of ``rho0``, of the radius and of the core mass, and the solver
evaluates it inside the differentiated function (``propagate.background_gravity``).
A stored gravity leaf would not follow ``rho0`` under ``jax.grad``: a pytree
unflatten does not recompute it. ``RadialProfile.hydrostatic_gravity`` gives the
hydrostatic profile on request. The functions ``integrate_gravity`` and
``mantle_mass`` follow lovejx ``integrate_gravity`` (shells of constant density at the
interval mean), ported with attribution (PLAN section 6).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .units import Units


def _is_traced(x) -> bool:
    """True for a JAX tracer (inside ``jit``, ``grad``, ``vmap``), whose value the host
    cannot read."""
    return isinstance(x, jax.core.Tracer)


@dataclass(frozen=True)
class NodeLimits:
    """Static upper limits of a node table whose radii and ``rho0`` move under a trace
    (a ``controls.Layers`` with movable boundaries; CONVENTIONS 2). Frozen, hashable.

    The Pade order and squaring count of the interval exponentials are static, and
    under a trace no value check can run. A movable boundary widens the intervals on
    one side of its pair and changes ``rho0`` at the moved nodes, so the plan must be
    built from the largest values that any admissible position of the boundaries
    gives. ``Layers`` computes them on the host when it is built, from the clamp bounds
    and the crossing positions of every range (NOTES/chunks/c4-controls-derivatives/01-plan.md D3).

    Attributes
    ----------
    width
        ``(N - 1,)`` upper limit of each interval width in ``ln r`` over every
        admissible position (0 at a pair). ``NodeStructure.width_bounds`` returns the
        elementwise maximum of the construction widths and these.
    kappa_h
        ``(N - 1,)`` upper limit of ``|ln(rho0_i / rho0_(i-1))|`` per interval over every
        admissible position, 0 at a pair (whose TALA step is the closed form of eq.
        12.3). ``propagate.static_plan`` refuses a structure whose limit exceeds
        ``Numerics.kappa_h_max`` on an ordinary interval under TALA.
    rho0_max
        Largest ``rho0`` in kg/m^3 that any admissible position gives. ``static_plan``
        refuses it above ``Numerics.rho0_max`` under FULL self-gravitation.

    The limits cover the motion of the nodes only. A traced ``rho0`` control (the
    ``rho0`` keyword of ``Layers``) can leave them, as every traced leaf can leave
    the static limits of ``Numerics``; the caller's prior must keep it inside.
    """

    width: tuple[float, ...]
    kappa_h: tuple[float, ...]
    rho0_max: float


@dataclass(frozen=True)
class NodeStructure:
    """Static, hashable node structure of a profile, read on the host.

    Attributes
    ----------
    radius
        Construction radii in m, CMB first, as Python floats. They set the static
        upper limits of the interval widths (``width_bounds``), from which the Pade
        order and squaring count of the interval exponentials come (CONVENTIONS 2).
        The radius of a profile can move under a derivative; the structure does not.
    pairs
        Indices ``i`` (1 to ``N - 1``) of the intervals between the nodes ``i - 1``
        and ``i`` that have zero width (``r_i == r_(i-1)``): the jump pairs.
    consistent_core
        True when the core mass is the consistent value ``(4 pi / 3) r_c^3 rho_c``
        (``RadialProfile.core_mass is None``). It is a static switch of the degree-1
        frame rule (equations.md 10(f)), so it belongs to the structure.
    limits
        ``None`` (a fixed node table: every structure built by ``from_radius``), or the
        ``NodeLimits`` of a table whose nodes move with a boundary (set by
        ``controls.Layers`` only). With the default ``None``, two structures without
        limits compare and hash equal when their other fields are equal, so the jit key
        and the Pade plan of a fixed node table do not depend on this field.
    cuts
        ``()`` (no viscosity cut: every structure built without the keyword), or one
        ``(lo, hi)`` per viscosity cut: the node indices of the two ends of the range in
        which the cut can lie (NOTES/docs/equations.md section 19). A viscosity cut is a
        viscosity jump inside one interval of this fixed node table, at the traced
        radius ``RadialProfile.cut_radius[q]``; it is not a node pair. The solver
        brackets the cut among the nodes ``lo .. hi`` and splits the interval that holds
        it into two exact sub-steps (``propagate.interval_steps``). The last field, with
        the default ``()``: every structure without cuts compares and hashes as a
        structure of the same fields without this one, so the jit keys and the Pade
        plans of a table without cuts do not change. Checked in ``__post_init__``:
        ``0 <= lo < hi <= N - 1``, the cuts ascend and do not overlap (two cuts can share
        an end node), and the first and the last interval of a range are not jump pairs
        (the bracket of a cut outside its range is clipped to these two intervals, so
        they must have a nonzero width).
    """

    radius: tuple[float, ...]
    pairs: tuple[int, ...]
    consistent_core: bool
    limits: NodeLimits | None = None
    cuts: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        """Check the static description of the viscosity cuts (``cuts``)."""
        n = len(self.radius)
        previous_hi = 0
        for q, cut in enumerate(self.cuts):
            if len(cut) != 2:
                raise ValueError(f"cut {q} must be a pair (lo, hi) of node indices, got {cut!r}")
            lo, hi = int(cut[0]), int(cut[1])
            if not 0 <= lo < hi <= n - 1:
                raise ValueError(f"cut {q} has the range nodes ({lo}, {hi}); they must satisfy "
                                 f"0 <= lo < hi <= {n - 1} (the last node index)")
            if lo < previous_hi:
                raise ValueError(f"cut {q} with the range nodes ({lo}, {hi}) starts below the end node {previous_hi} "
                                 "of the cut before it: the cuts must ascend and must not overlap (a shared end "
                                 "node is allowed)")
            # The bracket of a traced cut radius is clipped to the intervals lo .. hi - 1
            # (system.cut_intervals); its two sub-step fractions divide by the width of
            # the selected interval, so the end intervals must not be zero-width pairs.
            # Every interior pair is never selected: both of its nodes have one radius and
            # pass the comparison r_n <= c together.
            if self.radius[lo] == self.radius[lo + 1] or self.radius[hi - 1] == self.radius[hi]:
                raise ValueError(f"cut {q} with the range nodes ({lo}, {hi}) starts or ends on the outer member of "
                                 "a jump pair; a range end at a pair must be its inner member")
            previous_hi = hi

    @property
    def cut_lo(self) -> np.ndarray:
        """``(C,)`` int numpy: the lower range node ``lo`` of every viscosity cut."""
        return np.asarray([c[0] for c in self.cuts], dtype=int)

    @property
    def cut_hi(self) -> np.ndarray:
        """``(C,)`` int numpy: the upper range node ``hi`` of every viscosity cut."""
        return np.asarray([c[1] for c in self.cuts], dtype=int)

    @property
    def cut_mask(self) -> np.ndarray:
        """``(C, N)`` bool numpy: ``True`` where node ``n`` lies in the range of cut ``q``,
        ``lo_q <= n <= hi_q``. It restricts the bracket comparison of a cut to its own
        range (``system.cut_intervals``)."""
        idx = np.arange(self.n_nodes)
        return (idx[None, :] >= self.cut_lo[:, None]) & (idx[None, :] <= self.cut_hi[:, None])

    @property
    def n_nodes(self) -> int:
        """Number of nodes ``N``, both nodes of every pair included."""
        return len(self.radius)

    @property
    def n_intervals(self) -> int:
        """Number of intervals ``N - 1``, the zero-width intervals of the pairs included."""
        return len(self.radius) - 1

    @property
    def width_bounds(self) -> np.ndarray:
        """Static upper limit of the interval width in ``ln r`` per interval, shape
        ``(N - 1,)``, float64 numpy.

        Here the limit is the construction width ``log1p((r_i - r_(i-1)) / r_(i-1))``,
        and zero at a pair. ``log1p`` of the relative gap is used in place of
        ``log(r_i / r_(i-1))``: the ratio is close to one, and its rounding gives the
        width a relative error of ``2.2e-16 / h`` (2.4e-14 on 129 nodes), which sets
        the error floor of the steps at high degree
        (NOTES/chunks/c1-foundations/04-review-physics-r3.md, minor 3). A movable
        boundary widens these limits to the largest width that its range allows
        (CONVENTIONS 2): with ``limits`` set, the result is the elementwise maximum of
        the construction widths and ``limits.width``."""
        r = np.asarray(self.radius, dtype=np.float64)
        h = np.log1p((r[1:] - r[:-1]) / r[:-1])
        h[list(np.asarray(self.pairs, dtype=int) - 1)] = 0.0
        if self.limits is not None:
            h = np.maximum(h, np.asarray(self.limits.width, dtype=np.float64))
        return h

    @property
    def interval_mask(self) -> np.ndarray:
        """Static mask per interval, shape ``(N - 1,)``: 0.0 at a pair, 1.0 elsewhere.
        It multiplies the traced widths, so that a pair has a width of exactly zero
        under every transformation (a traced radius of a pair node can differ from
        its partner by rounding)."""
        m = np.ones(self.n_intervals)
        m[list(np.asarray(self.pairs, dtype=int) - 1)] = 0.0
        return m

    @classmethod
    def from_radius(cls, radius, *, consistent_core: bool, cuts=()) -> "NodeStructure":
        """Structure of a concrete radius array (m, CMB first), with the viscosity cuts
        ``cuts`` (``((lo, hi), ...)`` node indices of the cut ranges; default none).

        Raises ``ValueError`` for fewer than two nodes, a radius that decreases, a
        non-positive or non-finite radius, three equal neighbouring radii (a pair
        is two nodes), or cuts that ``__post_init__`` refuses."""
        r = np.asarray(radius, dtype=np.float64)
        if r.ndim != 1 or r.size < 2:
            raise ValueError("a node table needs a one-dimensional radius with at least two nodes")
        if not np.all(np.isfinite(r)) or np.any(r <= 0):
            raise ValueError("every radius must be finite and positive")
        if np.any(np.diff(r) < 0):
            raise ValueError("the radius must be non-decreasing (CMB first)")
        pairs = tuple(int(i) for i in np.nonzero(r[1:] == r[:-1])[0] + 1)
        if any(b == a + 1 for a, b in zip(pairs[:-1], pairs[1:])):
            raise ValueError("three equal neighbouring radii: a jump pair has exactly two nodes")
        return cls(radius=tuple(float(x) for x in r), pairs=pairs, consistent_core=bool(consistent_core),
                   cuts=tuple((int(lo), int(hi)) for lo, hi in cuts))


def consistent_core_mass(r_cmb, core_density, units: Units) -> jax.Array:
    """Mass of a uniform core, ``(4 pi / 3) r_c^3 rho_c`` in kg (F16, eq. 6.1).

    Parameters
    ----------
    r_cmb
        CMB radius in m (scalar, float64, can be traced).
    core_density
        ``rho_c`` in kg/m^3 (scalar, can be traced).
    units
        Supplies ``pi``.
    """
    return 4.0 * units.pi / 3.0 * r_cmb ** 3 * core_density


def _shell_masses(radius, rho0, units: Units) -> jax.Array:
    """Mass of each interval in kg, shape ``(N - 1,)``: a shell of constant density at
    the interval mean, ``(4 pi / 3) rhobar_k (r_k^3 - r_(k-1)^3)`` (eq. 6.1). A pair has
    ``r_k = r_(k-1)`` and adds no mass without a guard."""
    rbar = 0.5 * (rho0[1:] + rho0[:-1])
    return 4.0 * units.pi / 3.0 * rbar * (radius[1:] ** 3 - radius[:-1] ** 3)


def integrate_gravity(radius, rho0, core_mass, units: Units) -> jax.Array:
    """Hydrostatic gravity at the nodes, m/s^2, shape ``(N,)`` (eq. 6.1):

        g_i = G (M_c + sum_(k <= i) (4 pi / 3) rhobar_k (r_k^3 - r_(k-1)^3)) / r_i^2

    Each interval is a shell of constant density at the mean of its node values, the
    same material rule as the interval steps (eq. 12.1), so the shell mass is exact for
    the discrete model (lovejx ``integrate_gravity``). The sum is one ``cumsum``.

    Parameters
    ----------
    radius
        Node radii in m, ``(N,)``, CMB first, float64.
    rho0
        Reference density at the nodes, kg/m^3, ``(N,)``.
    core_mass
        Mass below the CMB in kg (scalar). For the consistent core use
        ``consistent_core_mass``.
    units
        Supplies ``G`` and ``pi``.

    Numerical caveat: the shell volume ``r_k^3 - r_(k-1)^3`` is a difference of two
    numbers of about ``R^3``; its relative rounding is about ``1e-16 r / (3 dr)``,
    1e-15 on a 129-node table, and the cumulative sum keeps the relative error of
    ``g`` below 1e-15 (measured 3.6e-16 to 5.7e-16 against the oracle at 60 digits).
    """
    radius = jnp.asarray(radius, dtype=jnp.float64)
    rho0 = jnp.asarray(rho0, dtype=jnp.float64)
    shells = _shell_masses(radius, rho0, units)
    enclosed = core_mass + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(shells)])
    return units.G * enclosed / radius ** 2


def mantle_mass(radius, rho0, units: Units) -> jax.Array:
    """Mass of the mantle in kg (scalar): the sum of the interval shells of
    ``integrate_gravity``. It enters ``M_eff`` of the degree-one imbalance (eq. 10.6).

    Parameters
    ----------
    radius
        ``(N,)`` node radii in m, CMB first.
    rho0
        ``(N,)`` reference density in kg/m^3.
    units
        ``Units`` (``pi`` of the shell volumes).
    """
    radius = jnp.asarray(radius, dtype=jnp.float64)
    rho0 = jnp.asarray(rho0, dtype=jnp.float64)
    return jnp.sum(_shell_masses(radius, rho0, units))


def _as_f64(x):
    """``jnp.asarray(x, float64)``; ``None`` stays ``None``. The explicit dtype makes a
    NumPy input and a JAX input give the same strong float64 type, so they share one
    trace (NOTES/chunks/c1-foundations/04-review-jax-r3.md, minor 2)."""
    return None if x is None else jnp.asarray(x, dtype=jnp.float64)


class RadialProfile:
    """The radial model of the mantle, CMB first. A JAX pytree.

    Parameters
    ----------
    radius
        ``(N,)``, m, non-decreasing, CMB first. Two equal neighbouring radii are a
        jump pair (zero-width interval). Meaning of a radius derivative at a pair:
        the width of a pair interval is a static zero (``NodeStructure.interval_mask``),
        while the gravity integral (eq. 6.1) reads both radii. So the derivative with
        respect to the radius of one node of a pair alone is not the derivative of a
        model in which the pair opens into an interval of finite width: it moves the
        shell mass of the gravity integral and leaves the propagator at the identity.
        A control that moves a pair must move both of its nodes together (the Taylor
        test T4 does, and a control map with movable boundaries must).
    rho0
        ``(N,)``, kg/m^3: the reference density at the nodes. The interval density is
        the mean of the two node values (eq. 12.1).
    eta
        ``(N,)``, Pa s, positive: the viscosity at the nodes. The interval viscosity is
        the mean of the two node values (eq. 12.1).
    core_density
        Scalar, kg/m^3: ``rho_c`` of the inviscid uniform core in the CMB conditions
        (the core pressure ``rho_c phi`` and the density contrast of the CMB
        deflection, eq. 8.4, 8.5).
    core_mass
        Scalar, kg, or ``None``. The mass below the CMB in the hydrostatic gravity
        (eq. 6.1). ``None`` is the consistent core ``(4 pi / 3) r_c^3 rho_c`` (F16),
        computed inside the differentiated function, so it follows ``core_density``
        and the CMB radius. A number selects the "split core" rules of the degree-1
        frame (equations.md 10(f)): the centre-of-mass frame with a reported
        ``degree_one_imbalance`` by default, the raw frame on request. Caveat
        (CONVENTIONS 5): a number equal to the consistent value is still a split core,
        and an explicit raw request then meets a singular system (raw ratio 2.5e-13
        at a relative offset of 1e-10 from the consistent value), which float64
        cannot solve; ``DegreeOneFrame.RAW`` states this.
    outer_density
        Scalar, kg/m^3, default 0: ``rho_w``, the density of the layer above the
        surface (water). The water surface follows the total equipotential, so
        ``rho_w`` enters only the conversion from the surface stress to the surface
        topography (eq. 8.3, F17).
    surface_density
        Scalar, kg/m^3, or ``None``: ``rho_top``, the density of the material that the
        surface topography displaces (eq. 8.3). ``None`` means ``rho0`` of the top
        node, read inside the differentiated function, so it follows ``rho0`` under
        ``jax.grad`` and after ``replace(rho0=...)``. A number is an input of its own.
        Physical reason for the field (CONVENTIONS 14): PREM has 1020 or 2600 kg/m^3
        at the top node depending on its ocean layer, and CODE-A uses
        ``bgrho = 4500 kg/m^3`` (F3).
    uniform_gravity
        Scalar, m/s^2, or ``None`` (default). The value of a uniform background gravity
        as a traced leaf, so that ``g0`` can be a control (NOTES/chunks/c4-controls-derivatives/01-plan.md
        D5). It is read only under ``Uniform(None)`` (``formulation.Uniform``), and the
        two go together: ``propagate.check_profile`` refuses ``Uniform(None)`` without
        this leaf and this leaf with any other background. ``None`` is an empty subtree,
        so a profile without the leaf keeps its leaves.
    cut_radius
        ``(C,)``, m, or ``None`` (default): the radii of the viscosity cuts of
        ``structure.cuts`` (NOTES/docs/equations.md section 19), a traced leaf. For cut
        ``q`` with the range nodes ``(lo, hi)``, the interval between the nodes ``a`` and
        ``a + 1`` with ``r_a <= c_q < r_(a+1)`` and ``lo <= a < hi`` has the viscosity
        ``eta[a]`` between ``r_a`` and ``c_q`` and ``eta[a + 1]`` between ``c_q`` and
        ``r_(a+1)``; its ``rho0`` (the interval mean) and, under TALA, its ``kappa_h``
        are those of the whole interval, shared by the two parts in proportion to their
        widths in ``ln r``. Every other interval keeps the interval-mean rule. A cut on
        a node (``c = r_a``) gives a zero-width lower part, whose step is the identity
        exactly. ``controls.Layers(..., motion="cut")`` sets the leaf and the node
        viscosities on both sides. A structure with cuts needs this leaf and a structure
        without cuts refuses it (``ValueError``, here and in ``propagate.check_profile``,
        also under a trace). A concrete value must be finite and lie in
        ``[r_lo, r_hi)`` of its cut (``propagate.check_profile``); a traced value outside
        is the caller's responsibility (the solver clips the bracket to the range, so
        every value stays finite, but the step is then not the model above). ``None`` is
        an empty subtree, so a profile without cuts keeps its leaves and its tree
        definition.
    structure
        Static ``NodeStructure``. Without it, the radius must be concrete and the
        structure is built from it; with it, the radius can be traced (for a control
        map that moves radii), and the shapes and the core model must agree with it.

    Every array is converted with ``jnp.asarray(x, dtype=jnp.float64)``. The
    constructor validates the inputs (shapes, non-decreasing radius, at least two
    nodes); the pytree unflatten does not, because JAX unflattens with tracers and
    placeholders. ``propagate.solve`` checks the values of the concrete leaves.

    The structure is the static auxiliary data of the pytree, and so part of the key of
    every jitted solve: a profile built from a new radius array compiles again, even
    with the same node count and pairs. Build a profile once and change its leaves with
    ``replace``, which keeps the structure.
    """

    __slots__ = ("radius", "rho0", "eta", "core_density", "core_mass", "outer_density",
                 "surface_density", "uniform_gravity", "cut_radius", "structure")

    def __init__(self, radius, rho0, eta, *, core_density, core_mass=None, outer_density=0.0,
                 surface_density=None, uniform_gravity=None, cut_radius=None,
                 structure: NodeStructure | None = None):
        radius, rho0, eta = _as_f64(radius), _as_f64(rho0), _as_f64(eta)
        core_density, core_mass = _as_f64(core_density), _as_f64(core_mass)
        outer_density, surface_density = _as_f64(outer_density), _as_f64(surface_density)
        uniform_gravity, cut_radius = _as_f64(uniform_gravity), _as_f64(cut_radius)
        if radius.ndim != 1 or rho0.shape != radius.shape or eta.shape != radius.shape:
            raise ValueError(f"radius, rho0 and eta must be one-dimensional arrays of one shape; "
                             f"got {radius.shape}, {rho0.shape}, {eta.shape}")
        for name, v in (("core_density", core_density), ("core_mass", core_mass),
                        ("outer_density", outer_density), ("surface_density", surface_density),
                        ("uniform_gravity", uniform_gravity)):
            if v is not None and v.shape != ():
                raise ValueError(f"{name} must be a scalar, got shape {v.shape}")
        if structure is None:
            if _is_traced(radius):
                raise ValueError("a traced radius needs an explicit NodeStructure (structure=...)")
            if cut_radius is not None:
                raise ValueError("a cut_radius needs the node ranges of its cuts: pass a structure with cuts "
                                 "(NodeStructure.from_radius(..., cuts=...))")
            structure = NodeStructure.from_radius(np.asarray(radius), consistent_core=core_mass is None)
        else:
            if structure.n_nodes != radius.shape[0]:
                raise ValueError(f"the structure has {structure.n_nodes} nodes, the radius {radius.shape[0]}")
            if structure.consistent_core != (core_mass is None):
                raise ValueError("core_mass is None exactly when the structure has a consistent core")
        check_cut_leaf(structure, cut_radius)
        for name, v in (("radius", radius), ("rho0", rho0), ("eta", eta), ("core_density", core_density),
                        ("core_mass", core_mass), ("outer_density", outer_density),
                        ("surface_density", surface_density), ("uniform_gravity", uniform_gravity),
                        ("cut_radius", cut_radius), ("structure", structure)):
            object.__setattr__(self, name, v)

    def __setattr__(self, name, value):
        raise AttributeError("RadialProfile is immutable; use replace()")

    def replace(self, **arrays) -> "RadialProfile":
        """A profile with the same structure and some leaves replaced, for example
        ``p.replace(eta=jnp.exp(log_eta))``. The new leaves can be traced. A change of
        ``core_mass`` between ``None`` and a number changes the core model, which the
        structure fixes, and raises ``ValueError``.

        Parameters
        ----------
        **arrays
            New values by leaf name (``radius``, ``rho0``, ``eta``, ``core_density``,
            ``core_mass``, ``outer_density``, ``surface_density``, ``uniform_gravity``,
            ``cut_radius``); another name raises ``TypeError``.
        """
        kw = {k: getattr(self, k) for k in _LEAF_NAMES}
        unknown = set(arrays) - set(kw)
        if unknown:
            raise TypeError(f"unknown profile fields: {sorted(unknown)}")
        kw.update(arrays)
        radius, rho0, eta = kw.pop("radius"), kw.pop("rho0"), kw.pop("eta")
        return RadialProfile(radius, rho0, eta, structure=self.structure, **kw)

    @property
    def rho_top(self) -> jax.Array:
        """Density of the surface topography in use, kg/m^3: ``surface_density`` or
        ``rho0[-1]`` (eq. 8.3)."""
        return self.rho0[-1] if self.surface_density is None else self.surface_density

    def effective_core_mass(self, units: Units) -> jax.Array:
        """Core mass in kg in the gravity integral: ``core_mass``, or the consistent
        value ``(4 pi / 3) r_c^3 rho_c`` for ``core_mass=None``; ``units`` gives ``pi``."""
        if self.core_mass is None:
            return consistent_core_mass(self.radius[0], self.core_density, units)
        return self.core_mass

    def hydrostatic_gravity(self, units: Units) -> jax.Array:
        """Hydrostatic gravity at the nodes, m/s^2, ``(N,)`` (eq. 6.1, ``integrate_gravity``),
        with ``G`` and ``pi`` of ``units``."""
        return integrate_gravity(self.radius, self.rho0, self.effective_core_mass(units), units)

    def __repr__(self):
        return (f"RadialProfile(n_nodes={self.structure.n_nodes}, pairs={self.structure.pairs}, "
                f"consistent_core={self.structure.consistent_core})")


#: The leaves of a profile, in pytree order. ``uniform_gravity`` is the eighth child and
#: ``cut_radius`` the ninth: ``None`` is an empty subtree, so a profile without them has
#: seven array leaves, and a profile without a cut has the tree definition of a profile
#: whose structure has no ``cuts`` field.
_LEAF_NAMES = ("radius", "rho0", "eta", "core_density", "core_mass", "outer_density", "surface_density",
               "uniform_gravity", "cut_radius")


def check_cut_leaf(structure: NodeStructure, cut_radius) -> None:
    """The static consistency of the cut leaf with the structure (pytree structure and
    shapes only, so the check also runs under a trace).

    ``structure.cuts`` is non-empty exactly when ``cut_radius`` is not ``None``, and then
    ``cut_radius`` has the shape ``(len(structure.cuts),)``; otherwise ``ValueError``."""
    if structure.cuts and cut_radius is None:
        raise ValueError(f"the structure has {len(structure.cuts)} viscosity cut(s) and the profile no "
                         "cut_radius; set it with replace(cut_radius=...)")
    if not structure.cuts and cut_radius is not None:
        raise ValueError("the profile carries a cut_radius and its structure has no viscosity cut; pass a "
                         "structure with cuts (NodeStructure.from_radius(..., cuts=...))")
    if cut_radius is not None and np.shape(cut_radius) != (len(structure.cuts),):
        raise ValueError(f"cut_radius must have one radius in m per cut, shape ({len(structure.cuts)},), "
                         f"got {np.shape(cut_radius)}")


def _flatten(p: RadialProfile):
    """Nine children (``None`` children are empty subtrees) and the static structure."""
    return tuple(getattr(p, k) for k in _LEAF_NAMES), p.structure


def _unflatten(structure: NodeStructure, children) -> RadialProfile:
    """Rebuild without validation: JAX unflattens with tracers, ``None`` and other
    placeholders (``object()`` in some transformations), and a validation would read a
    tracer or fail on a placeholder."""
    p = object.__new__(RadialProfile)
    for name, v in zip(_LEAF_NAMES, children):
        object.__setattr__(p, name, v)
    object.__setattr__(p, "structure", structure)
    return p


jax.tree_util.register_pytree_node(RadialProfile, _flatten, _unflatten)
