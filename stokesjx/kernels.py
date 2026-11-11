"""High layer: the configuration, the kernel record and the pure function
``stokes_kernels(profile, config)``.

``StokesConfig`` is frozen and hashable: it is the static part of a solve (the
maximum degree, the physics flags, the constants, the numerical options). The
``RadialProfile`` is a pytree of the arrays that a derivative can move. So
``stokes_kernels`` is a pure function of the profile for a fixed config, valid under
``jax.jit``, ``jax.grad``, ``jax.vmap`` and ``jax.jvp``:

    config = StokesConfig(lmax=50)

    def misfit(log_eta):
        K = stokes_kernels(profile.replace(eta=jnp.exp(log_eta)), config)
        return jnp.sum((K.geoid[K.row(2):] @ drho_dr - observed) ** 2)

    g = jax.jit(jax.grad(misfit))(log_eta)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax

from .formulation import DegreeOneFrame, Formulation
from .profile import RadialProfile
from .units import Units


@dataclass(frozen=True)
class Numerics:
    """Numerical options of the solver. Static.

    Attributes
    ----------
    rho0_max
        Upper limit of ``rho0`` in kg/m^3 for the static norm bound of the interval
        exponents, from which the Pade order and squaring count come (CONVENTIONS 2).
        The limit is a density, so it does not change with the density scale
        ``rho_s``. 13 500 kg/m^3 is above the density of every mantle and core model; a
        concrete ``rho0`` above it raises ``ValueError`` in ``stokes_kernels`` under
        FULL self-gravitation (under a trace no check is possible). Under
        NO_SELF_GRAVITY the exponent has no density entry and the limit is not used.
    node_solver
        The batched 6x6 node solve: ``"lapack"`` (``jnp.linalg.solve``, default) or
        ``"pivot_elimination"`` (Gauss-Jordan with partial pivoting on batch-last arrays,
        with a derivative rule). Both agree to rounding (4e-15 to 7e-15 on the benchmark
        tables). ``"lapack"`` is the default because the gradients that an HMC uses are
        as fast or faster with it (S1, 129 nodes and ``lmax = 50``: 15.9 against 16.4 ms
        for the gradient with respect to ``log eta``) and its gradient compiles in about
        0.74 s against 1.03 s; the elimination is 5 to 7 % faster only in the forward
        map (``benchmarks/bench_forward.py``, NOTES/benchmarks.md).
    kappa_h_max
        Static upper limit of ``|kappa_h| = |ln(rho0_i / rho0_(i-1))|`` on an ordinary
        interval under TALA (dimensionless), for the norm bound of the interval
        exponents (``linalg.balanced_norm_bound_intervals``, CONVENTIONS 2). A jump pair
        is exempt: its TALA step is the exact closed form of eq. 12.3. Default 0.3, a
        density factor of ``exp(0.3) = 1.35`` inside one interval: the test and benchmark
        tables reach at most 0.094 (0.068 on the 129-node LEBEDEV table, 0.0076 on the
        full PREM table), and a crust of 2600 kg/m^3 on a mantle of 3380 kg/m^3 in one
        interval (0.262) passes. A concrete ``rho0`` above the limit raises
        ``ValueError``, with the remedies: put the jump at a pair, or raise the limit.
        The limit sets the static Pade plan, so a larger value costs squarings, not
        accuracy: on the golden table under TALA an ordinary interval with ``|kappa_h|``
        of 0.15, 0.26, 0.69 and 0.94 (with the limit raised to cover it) keeps the error
        against the oracle at 3e-14 to 1.2e-13
        (NOTES/chunks/c3-flags-fields/plan-review-r1-evidence/kappa_silent.out). The
        default keeps the Pade plan of the incompressible model at the benchmark sizes.
    """

    rho0_max: float = 13500.0
    node_solver: str = "lapack"
    kappa_h_max: float = 0.3

    def __post_init__(self):
        from .linalg import NODE_SOLVERS
        if self.node_solver not in NODE_SOLVERS:
            raise ValueError(f"node_solver must be one of {NODE_SOLVERS}, got {self.node_solver!r}")
        if not self.rho0_max > 0:
            raise ValueError("rho0_max must be positive")
        if not self.kappa_h_max > 0:
            raise ValueError("kappa_h_max must be positive")
        object.__setattr__(self, "rho0_max", float(self.rho0_max))
        object.__setattr__(self, "kappa_h_max", float(self.kappa_h_max))


@dataclass(frozen=True)
class StokesConfig:
    """Static configuration of a kernel computation. Frozen and hashable.

    Attributes
    ----------
    lmax
        Largest spherical-harmonic degree; the kernels hold the degrees
        ``1 .. lmax``, or ``2 .. lmax`` where the frame rule refuses degree 1
        (``Kernels.degrees``). No precision limit sets ``lmax``: the sweeps are stable
        at every degree, and a kernel underflows only where the physical kernel is
        below 1e-308 of its scale (``l`` above about 1100 on an Earth-like shell,
        ``(r_c / R)^l``).
    formulation
        Physics flags (``Formulation``), every combination solved: the gravity level
        (``FULL``, ``NO_SELF_GRAVITY``), the background (``Hydrostatic()``,
        ``Uniform(value)``), ``compressible`` (TALA) and the boundary class at the
        surface and at the CMB (``FreeSlip()``, ``NoSlip()``).
    degree_one
        Degree-1 frame request (``DegreeOneFrame``). ``DEFAULT`` resolves through
        the static rule (``formulation.degree_one_options``): the centre-of-mass frame
        for FULL with hydrostatic gravity and free slip at both boundaries, the raw
        frame for NO_SELF_GRAVITY or a uniform background, and no degree 1 for FULL
        with hydrostatic gravity and a no-slip boundary. ``LEGACY_GEOID_ZERO`` is
        refused here (test-only, low layer).
    units
        Constants and the density scale (``Units``).
    numerics
        Numerical options (``Numerics``).
    """

    lmax: int
    formulation: Formulation = field(default_factory=Formulation)
    degree_one: DegreeOneFrame = DegreeOneFrame.DEFAULT
    units: Units = field(default_factory=Units)
    numerics: Numerics = field(default_factory=Numerics)

    def __post_init__(self):
        if isinstance(self.lmax, bool) or int(self.lmax) != self.lmax or self.lmax < 1:
            raise ValueError(f"lmax must be an integer >= 1, got {self.lmax!r}")
        object.__setattr__(self, "lmax", int(self.lmax))
        if not isinstance(self.degree_one, DegreeOneFrame):
            raise TypeError(f"degree_one must be a DegreeOneFrame, got {self.degree_one!r}")
        if self.degree_one is DegreeOneFrame.LEGACY_GEOID_ZERO:
            raise ValueError("DegreeOneFrame.LEGACY_GEOID_ZERO is CODE-A's frame and is offered only by the "
                             "test-only low layer (propagate.Discretisation)")
        if not isinstance(self.formulation, Formulation):
            raise TypeError(f"formulation must be a Formulation, got {self.formulation!r}")


@dataclass(frozen=True, eq=False)
class Kernels:
    """The four kernels per degree and source node, SI units. A JAX pytree: the arrays
    are leaves, ``degrees`` and ``degree_one_frame`` are static.

    A kernel ``K[row(l), j]`` is the response at degree ``l`` to a sheet of surface
    density 1 kg/m^2 in one spherical harmonic at node ``j`` (eq. 7.1). A field is the
    radial integral of a kernel times the density anomaly (``fields.apply``).

    Equality is identity (``eq=False``): an elementwise comparison of arrays has no
    single truth value, so two kernel records are compared array by array.

    Attributes
    ----------
    degrees
        The degrees of the rows, contiguous: ``(1, ..., lmax)``, or ``(2, ..., lmax)``
        where the frame rule refuses degree 1 (FULL with hydrostatic gravity and a
        no-slip boundary, CONVENTIONS 5), so that no row of a kernel array is NaN.
        ``row(l)`` is the one way to index a degree.
    degree_one_frame
        The resolved frame at ``l = 1`` (``CENTRE_OF_MASS``, ``RAW`` or
        ``LEGACY_GEOID_ZERO``), ``None`` without degree 1.
    radius
        ``(N,)``, m: the source radii (the profile radius).
    geoid
        ``(L, N)``, m per kg/m^2: the geoid height ``N = phi(R) / g(R)`` (eq. 9.1). At
        ``l = 1`` in the centre-of-mass frame an exact zero (eq. 10(d)).
    surface_topography
        ``(L, N)``, m per kg/m^2, outward positive (eq. 9.2, 8.3). The water term
        ``rho_w phi`` uses the total potential, which at ``l = 1`` in the CM frame of a
        split core includes the applied uniform field; there
        ``(rho_top - rho_w) h_s + rho_w N`` depends on ``rho_w``, and
        ``(rho_top - rho_w) h_s + rho_w (N + R A / g(R))`` with the applied field
        ``A`` of ``degree_one_imbalance`` does not.
    cmb_topography
        ``(L, N)``, m per kg/m^2, outward positive (eq. 9.3, 8.4).
    surface_velocity
        ``(L, N)``, m/s per kg/m^2: ``u_t(R) sqrt(l (l + 1))``, the coefficient of the
        RMS horizontal speed (eq. 9.4). The only kernel that changes when every
        viscosity is multiplied by one number (it scales with the inverse). Under a
        ``NoSlip`` surface every entry is an exact zero, and so is its derivative: the
        surface basis has no ``u_t`` component, so the carried velocity row is zero in
        exact arithmetic and in float64.
    degree_one_imbalance
        ``(N,)``, dimensionless, or ``None`` without degree 1 (eq. 10.6):
        ``M_eff A_j / ((4 pi / 3) g(r_j) r_j^2)``, the uniform force that keeps the
        centre of mass at the origin over the gravity force on the unit sheet, with
        ``A_j = phi_a(R) / R`` and ``M_eff = M_mantle + (4 pi / 3) r_c^3 rho_c``. It is
        the momentum error of a split core (up to 7e-3 for a PREM core mass with
        ``rho_c = 9900``) and zero up to rounding for a consistent core. An exact zero
        in the raw frame and in CODE-A's frame, which have no applied field.
    """

    degrees: tuple[int, ...]
    degree_one_frame: DegreeOneFrame | None
    radius: jax.Array
    geoid: jax.Array
    surface_topography: jax.Array
    cmb_topography: jax.Array
    surface_velocity: jax.Array
    degree_one_imbalance: jax.Array | None

    def row(self, l: int) -> int:
        """Index of degree ``l`` in the kernel arrays."""
        try:
            return self.degrees.index(int(l))
        except ValueError:
            raise KeyError(f"degree {l} is not in the kernels (degrees {self.degrees[0]}..{self.degrees[-1]})")


_KERNEL_LEAVES = ("radius", "geoid", "surface_topography", "cmb_topography", "surface_velocity",
                  "degree_one_imbalance")


def _kernels_flatten(k: Kernels):
    return tuple(getattr(k, f) for f in _KERNEL_LEAVES), (k.degrees, k.degree_one_frame)


def _kernels_unflatten(aux, children) -> Kernels:
    # No validation: JAX unflattens with tracers and placeholders.
    k = object.__new__(Kernels)
    object.__setattr__(k, "degrees", aux[0])
    object.__setattr__(k, "degree_one_frame", aux[1])
    for f, v in zip(_KERNEL_LEAVES, children):
        object.__setattr__(k, f, v)
    return k


jax.tree_util.register_pytree_node(Kernels, _kernels_flatten, _kernels_unflatten)


def stokes_kernels(profile: RadialProfile, config: StokesConfig) -> Kernels:
    """The geoid, surface topography, CMB topography and surface velocity kernels of a
    radial profile, for the degrees ``1 .. config.lmax`` (``2 .. lmax`` where the frame
    rule refuses degree 1) and every node as a source.

    Parameters
    ----------
    profile
        ``RadialProfile`` (SI units, float64). Every leaf can be traced.
    config
        ``StokesConfig`` (static): ``lmax``, the flags of ``Formulation`` (gravity level,
        background, TALA, the boundary classes), the degree-1 frame, the units and the
        numerical options.

    Returns
    -------
    ``Kernels`` with arrays of shape ``(L, N)`` over ``Kernels.degrees`` (``row(l)`` gives
    the row of degree ``l``) and ``degree_one_imbalance`` of shape ``(N,)`` or ``None``.

    The jit key. The structure of the profile (``NodeStructure``: the construction
    radii, the pairs, the core model) is static, so a profile built from a new radius
    array compiles again (about 0.3 s), even with the same node count. Build a profile
    once and change it with ``replace`` (which keeps the structure, also for a traced
    radius); a control map with movable boundaries keeps one structure for all samples.

    Method (NOTES/docs/equations.md section 16). Per degree, two sweeps of a
    three-vector basis, upward from the CMB and downward from the surface, with the
    exact interval exponentials of all degrees and intervals computed in one batched
    Pade call outside the sweeps; after each interval the basis is orthonormalised
    (Gram-Schmidt), and the output rows are carried through the triangular factors.
    At every node one 6x6 system joins the two bases across the sheet; all systems are
    one batched solve. The viscosity scale of the state is the largest node viscosity.

    Accuracy, measured with this package against the mpmath oracle at 60 to 250 digits
    (max-normalised per kernel and degree; NOTES/chunks/c2-forward-core/03-implementation.md
    and NOTES/chunks/c3-flags-fields/03-implementation.md):

    - values, default flags: at most 8.5e-14 on the stored moderate cases (hager4, wide5,
      PREM with density jumps, a smooth viscosity at every node, layered and uniform
      densities; degrees 1 to 200), 3.1e-13 in the raw frame at ``l = 1`` on a split
      core. The error of the raw frame grows as the raw system approaches the singular
      case (a consistent core, eq. 10a): on two four-node tables with a pair at the CMB
      or at the surface it reached 1.2e-11 and 2.5e-12 (geoid), so the stated number
      holds for the stored tables;
    - values under the flags (NO_SELF_GRAVITY, a uniform background, TALA, no slip at
      either boundary, all 32 combinations): at most 4.6e-13 on the stored flag cases
      (TALA on PREM with density jumps at pairs, at ``l = 1`` in the raw frame of a split
      core), 3.5e-13 over all 32 combinations on a reduced PREM table and the 8
      combinations of FULL with hydrostatic gravity on a consistent core (``l`` in {1,
      2, 10}), at most 5.9e-14 on the full PREM table and 2.3e-13 on the stiff lid and on
      ten decades of viscosity at ``l`` up to 100 under no slip, TALA, NO_SELF_GRAVITY
      and a uniform background; against the analytical solutions of Kramer, Davies and
      Wilson (2021; NO_SELF_GRAVITY, uniform ``g``, free slip and no slip) 1.2e-14; with
      CODE-A's legacy switches against the Fortran under TALA (``kappaindex = 1``) and no
      slip at most 3.0e-13;
    - values, default flags, with a
      1e27 Pa s layer on a 1e20 Pa s mantle (a lid of 70 km, or a layer from 670 to
      1000 km depth) and on ten decades of layered viscosity (1e16 to 1e26 Pa s) at
      most 9.7e-13 at ``l >= 2`` and 1.5e-12 at ``l = 1`` (CM frame), without growth
      with the degree up to ``l = 200``; on lids of 1e25, 1e27, 1e29 and 1e31 Pa s above
      70 km at most 2.8e-13 at ``l >= 2`` and 1.6e-12 at ``l = 1`` up to ``l = 100``,
      without growth with the viscosity of the lid; with an independent viscosity at
      every node over eight decades (neighbour ratios up to 1.0e7) 1.3e-14. A prior that
      allows larger neighbour ratios is outside the range measured with this package
      (the prototypes of NOTES/chunks/c2-forward-core/01-plan.md measured 7.4e-12 at twelve decades with neighbour
      ratios up to 8.9e10). At ``l = 1200`` (hager4) 7.9e-15; the rows of the highest
      degrees hold exact zeros (and next to them a few subnormal numbers) at the few
      sources farthest from the boundary of the
      output (4 to 7 of 135 at ``l = 1200``), where the physical kernel is below the
      float64 range relative to its maximum;
    - ``degree_one_imbalance``, absolute: at most 9.6e-15 on the moderate tables, up to
      4.9e-13 on the lids and 1.8e-12 with the stiff layer at 670 to 1000 km depth;
    - derivatives with respect to ``log eta``, against a central difference of the
      oracle along a random direction at ``l`` in {1 (CM), 2, 10}, max-normalised per
      kernel and degree: 3.5e-14 in forward mode and 9.8e-14 in reverse mode on
      hager4; 4.8e-14 and 1.2e-13 on the eight-decade random table; 8.2e-13 and 8.9e-13
      on ten decades; on lids of 1e25 to 1e31 Pa s at most 2.3e-11 in both modes,
      without growth with the viscosity of the lid. On the lids the number depends on
      the direction (2.2e-12 on the 1e25 Pa s lid along a second direction): the
      derivative of a kernel row along a direction can be much smaller than the
      sensitivity of the row, and the metric divides by it. Under the flags the Taylor
      tests of CONVENTIONS 8 (second order for both direction signs in both modes) pass
      for ``log eta``, ``rho0`` and the radius together in all 32 combinations, and the
      jvp/vjp dot products agree to 1e-14 relative (TALA with the closed-form pair step,
      NO_SELF_GRAVITY with a uniform background, no slip at both boundaries).

    Numerical caveats: the Pade plan of the interval exponentials is static, from the
    construction widths of ``profile.structure``, ``config.numerics.rho0_max`` (under
    FULL) and, under TALA, ``config.numerics.kappa_h_max``. For every concrete leaf of the
    profile this function checks the values (``rho0 > 0``, ``rho0 <= rho0_max``,
    ``eta > 0``, ``|kappa_h| <= kappa_h_max`` on every ordinary interval under TALA, and
    that the widths keep the plan valid) and raises ``ValueError`` otherwise; a traced
    leaf cannot be checked, so under ``jax.grad`` over ``log eta`` the checks of ``rho0``
    and of the radius still run. A profile of a ``controls.Layers`` with movable
    boundaries under ``motion="affine"`` carries static limits in its structure (``NodeStructure.limits``: the
    largest interval widths, ``|kappa_h|`` and ``rho0`` over every admissible position of
    the boundaries); the Pade plan is built from these limits, and a configuration whose
    limits exceed ``Numerics.rho0_max`` (FULL) or ``Numerics.kappa_h_max`` (TALA) raises
    ``ValueError`` also under a trace. A traced ``rho0`` control is not covered by these
    limits; the caller's prior must keep it inside them. A profile with viscosity cuts
    (``controls.Layers(..., motion="cut")``; ``NodeStructure.cuts`` and the traced leaf
    ``RadialProfile.cut_radius``) splits the interval that holds each cut into two exact
    sub-steps (NOTES/docs/equations.md section 19); its Pade plan is the plan of the
    construction table. A concrete ``cut_radius`` outside its range raises; a traced one
    is clipped to the range by the bracket and is the caller's responsibility. Under TALA
    with a jump pair inside a cut range, this function warns on every call
    (``RuntimeWarning``): the kernels jump when the cut crosses a pair with a density
    jump (``controls.Layers``).
    """
    # The low layer is imported here: propagate imports this module for the
    # configuration and the kernel record.
    from . import propagate
    # stacklevel 3: the TALA warning of solve names the line of the caller of this
    # function (solve -> stokes_kernels -> caller).
    return propagate.solve(profile, config, _warn_stacklevel=3)
