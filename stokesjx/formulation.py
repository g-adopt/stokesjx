"""The physics flags (PLAN section 5), the boundary classes, the term record of the
solver and the static degree-1 frame rule.

A ``Formulation`` is a plain frozen record of the flags: the gravity model, the
background gravity, the compressibility and the boundary class at the surface and at
the CMB. Every combination of the flags is solved (32 combinations of the five binary
choices). ``terms(formulation)`` resolves the flags once into the record ``Terms``
that the solver reads, as lovejx resolves its flags into assembly objects
(``lovenumbers/formulation.py``): ``system``, ``boundary`` and ``propagate._core`` read
``Terms`` and never the gravity enum.

The gravity model keeps the lovejx name ``GravityModel`` and the member values
``"full"`` and ``"no_self_gravity"`` of lovejx (ruling 6: every convention follows
lovejx). The instantaneous Stokes problem has two levels (PLAN section 5):

- ``FULL``: the perturbed potential pushes on ``rho0`` (entries ``a(3,6)`` and
  ``a(4,5)`` of the matrix, eq. 5.1), on the core (``rho_c phi`` at the CMB, eq. 8.4,
  8.5) and on the water (``rho_w phi`` in the surface topography, eq. 8.3);
- ``NO_SELF_GRAVITY``: these three terms are off; the geoid is still computed from the
  load and the deflected boundaries (eq. 5.2). This is the second level of the G-ADOPT
  benchmark.

Static switch points of the solver (each one a Python ``if`` on a field of ``Terms``,
never on a traced value):

- ``system.balanced_exponent``: the self-gravity entries (``self_gravity``) and the
  TALA ``K`` term (``anelastic``, with ``system.kappa_widths``);
- ``system.anelastic_pair_terms``: the closed-form TALA step across a jump pair
  (``anelastic``, eq. 12.3);
- ``boundary.cmb_basis``: the tangential unknown of the CMB class and the ``rho_c``
  term (``self_gravity``);
- ``boundary.surface_basis``: the tangential unknown of the surface class;
- ``boundary.surface_functionals``: the ``rho_w phi`` term (``self_gravity``);
- ``boundary.cmb_functional``: the ``rho_c phi`` term (``self_gravity``);
- ``propagate.background_gravity``: hydrostatic or uniform ``g`` (the background is
  not part of ``Terms``; it changes a traced array, not a term of the equations).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


class GravityModel(Enum):
    """How much of the gravitational problem is solved (PLAN section 5; lovejx name)."""

    FULL = "full"
    NO_SELF_GRAVITY = "no_self_gravity"


@dataclass(frozen=True)
class Hydrostatic:
    """Background gravity integrated from ``rho0`` and the core mass (eq. 6.1)."""


@dataclass(frozen=True)
class Uniform:
    """Background gravity of one value at every node (eq. 6.2).

    Attributes
    ----------
    value
        ``g0`` in m/s^2, positive, or ``None`` (default). CODE-A uses 10 (``grav()``,
        matrix.f:98); the G-ADOPT benchmark and ``assess`` use their own value. A number
        is static: it is a field of the hashable config, and a new number compiles
        again. ``None`` means that the value is the profile leaf
        ``RadialProfile.uniform_gravity``, which can be traced, so that ``g0`` can be a
        control of a derivative (CONVENTIONS 2: a traced parameter of a flag never lives
        in the config; NOTES/chunks/c4-controls-derivatives/01-plan.md D5).
        ``propagate.check_profile`` refuses ``Uniform(None)`` without the leaf and the
        leaf with a number here or with ``Hydrostatic()``.

    The interval steps do not depend on ``g`` (F1), so the value does not change the
    Pade plan. Under ``NO_SELF_GRAVITY`` the kernels are exact powers of ``g0``: the
    geoid kernel is proportional to ``1 / g0``, the velocity kernel to ``g0``, and both
    topographies do not depend on it. Under ``FULL`` the dependence is not a power.
    """

    value: float | None = None

    def __post_init__(self):
        if self.value is None:
            return
        if not self.value > 0:
            raise ValueError(f"a uniform background gravity must be positive, got {self.value!r}")
        object.__setattr__(self, "value", float(self.value))


class BoundaryCondition(abc.ABC):
    """Base of the tangential boundary classes at the surface and at the CMB.

    Every class has ``u_r = 0`` at its boundary (eq. 8.1, ``y1 = 0``) and one condition
    on the tangential part of the state. With ``u_r = 0`` and that condition, exactly
    one direction of the tangential pair ``(y2, y4)`` stays free at the boundary (eq.
    3.1: ``y2 = u_t``, ``y4 = r sigma_rt / eta_s``). ``tangential_unknown`` returns it as
    the coefficients ``(a, b)`` of the state direction ``a e2 + b e4``;
    ``boundary.cmb_basis`` and ``boundary.surface_basis`` put that direction into the
    three-vector basis of the boundary.

    Rules for a class:

    - It is a frozen dataclass, and every field is a hashable static Python value. A
      boundary class is a field of ``Formulation``, which is a field of
      ``StokesConfig``, and the config is a static argument of the jitted solver and
      the key of the cache of ``propagate.static_plan``. A JAX array or a tracer in a
      field raises ``ValueError: Non-hashable static arguments are not supported``
      under ``jax.jit``. So a traced boundary parameter never lives in the config: it
      reaches ``tangential_unknown`` through the argument ``parameter``.
    - ``traction_free`` (class attribute) is true when the class sets the tangential
      traction to zero identically. The degree-1 frame rule (``degree_one_options``)
      reads only this attribute, never a class name: a boundary without a tangential
      traction cannot hold the net force of a degree-1 load, an anchored one can.

    The arguments of ``tangential_unknown``:

    radius
        Traced scalar, the boundary radius in m.
    eta_ref
        Traced scalar, the viscosity scale ``eta_s`` of the state in Pa s
        (``propagate.reference_viscosity``).
    side
        Static Python int, ``+1`` at the surface and ``-1`` at the CMB: the sign of the
        outward normal of the mantle at the boundary, ``n = side e_r``. The traction
        that the outside exerts on the mantle has the tangential part
        ``side sigma_rt``. A law that sets a component to zero does not need it; a law
        that depends on the direction of the traction (a drag) does.
    parameter
        ``None`` or a traced scalar, the value of a boundary parameter of the class.
        The solver passes ``None`` for every class of this version.

    A later partial-slip class (a drag that opposes the motion, ``side sigma_rt =
    -beta u_t`` with the drag coefficient ``beta >= 0`` in Pa s/m) returns
    ``(1.0, -side * radius * parameter / eta_ref)``: in the state this is
    ``y4 = -side (r beta / eta_ref) y2``, so ``sigma_rt = -beta u_t`` at the surface and
    ``sigma_rt = +beta u_t`` at the CMB, and the limit of large ``beta`` is the no-slip
    direction ``e4``. That class changes three things and nothing else: the class
    here; one optional scalar leaf per boundary in the traced inputs (for example a
    leaf of ``RadialProfile`` with default ``None``, so that the pytree of every
    existing profile stays the same); and the two calls of ``cmb_basis`` and
    ``surface_basis`` in ``propagate._core``, which pass that leaf as ``parameter``.
    ``boundary.py``, ``system.py``, the sweeps, the node solve and the output rows do
    not change. Open question for that class: it carries a tangential traction, so it
    has ``traction_free = False``, and the frame rule then treats it as an anchored
    boundary (degree 1 refused by default under FULL with hydrostatic gravity). Whether
    a finite drag anchors the mantle in the same sense as no slip needs its own
    argument when the class is added.
    """

    traction_free: ClassVar[bool]

    @abc.abstractmethod
    def tangential_unknown(self, radius, eta_ref, side: int, parameter=None) -> tuple:
        """The free tangential direction ``a e2 + b e4`` of the state at the boundary,
        as ``(a, b)``.

        Parameters (class docstring): ``radius`` the boundary radius in m, ``eta_ref`` the
        viscosity scale ``eta_s`` in Pa s, ``side`` +1 at the surface and -1 at the CMB,
        ``parameter`` ``None`` or a traced boundary parameter."""


@dataclass(frozen=True)
class FreeSlip(BoundaryCondition):
    """Free slip: ``u_r = 0`` and ``sigma_rt = 0`` (``y1 = y4 = 0``, eq. 8.1).

    The free tangential unknown is the velocity ``u_t``, the direction ``e2``. The
    tangential traction is zero identically (``traction_free = True``)."""

    traction_free: ClassVar[bool] = True

    def tangential_unknown(self, radius, eta_ref, side: int, parameter=None) -> tuple:
        """``(1.0, 0.0)``: the direction ``e2``, as Python floats for every argument, so
        that the boundary basis stays a static coefficient array. The arguments
        ``radius``, ``eta_ref``, ``side`` and ``parameter`` (``BoundaryCondition``) are not
        read."""
        return 1.0, 0.0


@dataclass(frozen=True)
class NoSlip(BoundaryCondition):
    """No slip: ``u_r = 0`` and ``u_t = 0`` (``y1 = y2 = 0``, eq. 8.1).

    The free tangential unknown is the shear stress, the direction ``e4``. The boundary
    carries a tangential traction (``traction_free = False``). Under a no-slip surface
    the surface velocity kernel is an exact zero (``Kernels.surface_velocity``)."""

    traction_free: ClassVar[bool] = False

    def tangential_unknown(self, radius, eta_ref, side: int, parameter=None) -> tuple:
        """``(0.0, 1.0)``: the direction ``e4``, as Python floats for every argument. The
        arguments ``radius``, ``eta_ref``, ``side`` and ``parameter``
        (``BoundaryCondition``) are not read."""
        return 0.0, 1.0


class DegreeOneFrame(Enum):
    """Frame of the degree-1 solution (equations.md section 10).

    DEFAULT
        Resolve through the static rule (``degree_one_options``).
    CENTRE_OF_MASS
        The member of the solution family with no degree-1 self-potential, ``N_1 = 0``
        (10d, 10e): the surface potential row ``y6 - l y5 + c_s y3 = 0`` (eq. 10.5).
        With a split core the model does not conserve momentum, and the CM row adds
        the uniform field that keeps the centre of mass at the origin; its amplitude
        is ``Kernels.degree_one_imbalance`` (eq. 10.6).
    RAW
        The raw surface row ``y6 + (l + 1) y5 + c_s y3 = 0`` (eq. 8.2): no applied
        field. Singular for FULL with hydrostatic gravity, a consistent core and free
        slip at both boundaries (10a), so not offered there. Caveat (CONVENTIONS 5): a
        ``core_mass`` number close to the consistent value is a split core, for which
        raw is offered, and the raw system is then nearly singular (raw ratio 2.5e-13
        at a relative offset of 1e-10), so float64 cannot solve it. The default is
        never raw for such a configuration.
    LEGACY_GEOID_ZERO
        CODE-A's frame: the raw solve with the degree-1 geoid set to zero
        (driver.f:282-284). Low layer only (``propagate.Discretisation``);
        ``StokesConfig`` refuses it.
    """

    DEFAULT = "default"
    CENTRE_OF_MASS = "centre_of_mass"
    RAW = "raw"
    LEGACY_GEOID_ZERO = "legacy_geoid_zero"


@dataclass(frozen=True)
class Formulation:
    """The physics flags of one run (PLAN section 5). Frozen and hashable.

    Attributes
    ----------
    gravity
        ``GravityModel.FULL`` (default) or ``NO_SELF_GRAVITY`` (eq. 5.1, 5.2).
    background
        ``Hydrostatic()`` (default) or ``Uniform(value)`` (eq. 6.1, 6.2).
    compressible
        ``False`` (default): incompressible, ``div v = 0``. ``True``: TALA, the
        anelastic continuity ``div(rho0 v) = 0`` (eq. 1.2, 4.3), with the deviatoric
        stress ``2 eta (e - div(v) I / 3)``; the ``-2/3 eta div v`` term leaves the
        matrix A unchanged (F2 of equations.md section 4), so TALA adds only the term
        ``kappa K`` with ``kappa = d ln rho0 / d ln r``. The discrete step is exact for
        a power-law ``rho0`` inside an interval (eq. 12.2), and the step across a jump
        pair is the anelastic rule (eq. 12.3): ``rho0 u_r`` is continuous across a
        density jump, the ``h -> 0`` limit of the interval step. The other reading
        (``u_r`` continuous, the "material" rule) changes the PREM TALA geoid kernel
        by up to 18 % at ``l = 2`` and the topographies by 2 % (equations.md section
        12, test_pair_rule_size of the oracle). The package implements the anelastic rule
        only. The density jumps of ``rho0`` in the mantle are phase changes: the material
        crosses them, so the mass flux ``rho0 u_r`` is continuous and ``u_r`` jumps by the
        inverse ratio of the two densities (eq. 12.3). The "material" rule is the rule of a
        chemical boundary, which the material does not cross; the package does not
        implement it, and the mpmath oracle evaluates it only for ``test_pair_rule_size``.
    surface, cmb
        Boundary class at each boundary, ``FreeSlip()`` (default) or ``NoSlip()``
        (``BoundaryCondition``).
    """

    gravity: GravityModel = GravityModel.FULL
    background: Hydrostatic | Uniform = field(default_factory=Hydrostatic)
    compressible: bool = False
    surface: BoundaryCondition = field(default_factory=FreeSlip)
    cmb: BoundaryCondition = field(default_factory=FreeSlip)

    def __post_init__(self):
        # Type checks only: a wrong type would select a wrong static branch without
        # an error (an enum member or a string in place of a boundary class has no
        # tangential_unknown, and a string "no_slip" is truthy).
        if not isinstance(self.gravity, GravityModel):
            raise TypeError(f"gravity must be a GravityModel, got {self.gravity!r}")
        if not isinstance(self.background, (Hydrostatic, Uniform)):
            raise TypeError(f"background must be Hydrostatic() or Uniform(value), got {self.background!r}")
        for name in ("surface", "cmb"):
            value = getattr(self, name)
            if not isinstance(value, BoundaryCondition):
                raise TypeError(f"{name} must be a BoundaryCondition instance such as FreeSlip() or NoSlip(), "
                                f"got {value!r}")
        object.__setattr__(self, "compressible", bool(self.compressible))


@dataclass(frozen=True)
class Terms:
    """The terms of the equations that a formulation switches on, resolved once from
    the flags (``terms``). Frozen and hashable; part of ``propagate.StaticPlan``.

    Attributes
    ----------
    self_gravity
        FULL: the entries ``a(3,6) = a(4,5) = -rho0 / rho_s`` of A (eq. 5.1), the
        ``rho_c phi`` term at the CMB (eq. 8.4, 8.5) and the ``rho_w phi`` term of the
        surface topography (eq. 8.3). False under NO_SELF_GRAVITY (eq. 5.2).
    anelastic
        TALA: the term ``kappa_h K`` of the interval exponent (eq. 4.3, 12.2) and the
        closed-form step across a jump pair (eq. 12.3).
    surface, cmb
        The boundary classes (``BoundaryCondition``).
    """

    self_gravity: bool
    anelastic: bool
    surface: BoundaryCondition
    cmb: BoundaryCondition


def terms(formulation: Formulation) -> Terms:
    """Resolve the flags of a formulation into the term record of the solver."""
    return Terms(self_gravity=formulation.gravity is GravityModel.FULL,
                 anelastic=formulation.compressible,
                 surface=formulation.surface, cmb=formulation.cmb)


@dataclass(frozen=True)
class FrameOptions:
    """The static degree-1 frame table of one configuration (equations.md 10(f)).

    Attributes
    ----------
    default
        The frame of ``DegreeOneFrame.DEFAULT``, or ``None`` when degree 1 is refused
        by default.
    offered
        Frames available on request.
    explicit_only
        Frames available only on an explicit request (not a default), with the reason
        in ``note``.
    note
        The reason of the table row.
    """

    default: DegreeOneFrame | None
    offered: tuple[DegreeOneFrame, ...]
    explicit_only: tuple[DegreeOneFrame, ...]
    note: str


def degree_one_options(formulation: Formulation, consistent_core: bool) -> FrameOptions:
    """The static frame table of equations.md 10(f), from the flags only.

    No numerical value of the profile enters (CONVENTIONS 5): only the gravity model,
    the background, the attribute ``traction_free`` of the two boundary classes and
    whether the core is consistent (``core_mass is None``). The table is the port of
    ``reference_mpmath.frame_options``:

    - FULL, hydrostatic, both boundaries traction-free, consistent core: the raw
      system is singular for every mantle density (10a); CM only, CM default;
    - the same with a split core: CM default, raw offered;
    - FULL, hydrostatic, a boundary with a tangential traction (no slip): the anchor
      carries no net force (10a), so degree 1 is refused by default; raw on explicit
      request for a split core;
    - NO_SELF_GRAVITY or a uniform background: a background field takes the net
      force, raw only (lovejx anchor convention).

    Parameters
    ----------
    formulation
        The ``Formulation`` (static).
    consistent_core
        ``True`` for a consistent core (``core_mass is None``), ``False`` for a split core.

    Parameters
    ----------
    formulation
        The ``Formulation`` (static).
    consistent_core
        ``True`` for a consistent core (``core_mass is None``), ``False`` for a split core.
    """
    free_body = formulation.gravity is GravityModel.FULL and isinstance(formulation.background, Hydrostatic)
    both_free = formulation.surface.traction_free and formulation.cmb.traction_free
    CM, RAW = DegreeOneFrame.CENTRE_OF_MASS, DegreeOneFrame.RAW
    if free_body and both_free:
        if consistent_core:
            return FrameOptions(CM, (CM,), (), "free body, consistent core: raw system singular")
        return FrameOptions(CM, (CM, RAW), (),
                            "free body, split core: CM with the reported imbalance; raw also offered")
    if free_body:
        if consistent_core:
            return FrameOptions(None, (), (), "no-slip boundary, consistent core: raw singular for uniform density")
        return FrameOptions(None, (), (RAW,),
                            "no-slip boundary, split core: the raw value is set by the core split and "
                            "by the flow of the null vector")
    return FrameOptions(RAW, (RAW,), (), "no self-gravitation or uniform g: raw frame")


def resolve_degree_one(formulation: Formulation, consistent_core: bool,
                       request: DegreeOneFrame) -> DegreeOneFrame | None:
    """Resolve a requested degree-1 frame (port of ``reference_mpmath.degree_one_rule``).

    Returns the frame, or ``None`` when ``DEFAULT`` is refused for the configuration.
    ``LEGACY_GEOID_ZERO`` needs the raw frame to be available. A frame that the rule
    does not offer raises ``ValueError``: for example raw on a consistent core under
    FULL with hydrostatic gravity and free slip, where the float64 solve of the
    singular system would return finite wrong numbers without an error.

    Parameters
    ----------
    formulation
        The ``Formulation`` (static).
    consistent_core
        ``True`` for a consistent core (``core_mass is None``).
    request
        The requested ``DegreeOneFrame``.
    """
    opts = degree_one_options(formulation, consistent_core)
    if request is DegreeOneFrame.DEFAULT:
        return opts.default
    available = opts.offered + opts.explicit_only
    if request is DegreeOneFrame.LEGACY_GEOID_ZERO:
        if DegreeOneFrame.RAW in available:
            return request
        raise ValueError("the legacy degree-1 frame needs a raw solve, which is not offered here: "
                         + opts.note)
    if request in available:
        return request
    raise ValueError(f"degree-1 frame {request.value!r} is not offered: {opts.note}")
