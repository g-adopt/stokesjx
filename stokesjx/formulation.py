"""The physics flags (PLAN section 5) and the static degree-1 frame rule.

A ``Formulation`` is a plain frozen record of the flags: the gravity model, the
background gravity, the compressibility and the boundary class at the surface and at
the CMB. It accepts every flag value, so that the static frame table of
``degree_one_options`` can be built and tested for every configuration. The solver of
this chunk solves the default incompressible, self-gravitating, free-slip problem with
a hydrostatic or a uniform background; ``kernels.StokesConfig`` refuses the other
values with ``NotImplementedError``. Every solve goes through a ``StokesConfig``, so no
path reaches the solver with such a value.

The gravity model keeps the lovejx name ``GravityModel`` and the member values
``"full"`` and ``"no_self_gravity"`` of lovejx (ruling 6: every convention follows
lovejx). The instantaneous Stokes problem has two levels (PLAN section 5):

- ``FULL``: the perturbed potential pushes on ``rho0`` (entries ``a(3,6)`` and
  ``a(4,5)`` of the matrix, eq. 5.1), on the core (``rho_c phi`` at the CMB, eq. 8.4,
  8.5) and on the water (``rho_w phi`` in the surface topography, eq. 8.3);
- ``NO_SELF_GRAVITY``: these three terms are off; the geoid is still computed from the
  load and the deflected boundaries (eq. 5.2). This is the second level of the G-ADOPT
  benchmark.

Static switch points of the next chunk (each one a Python ``if`` on a flag, never on a
traced value): ``system.balanced_exponent`` (self-gravity entries, the TALA ``K``
term), ``system.pair_steps`` (TALA pair rule), ``boundary.cmb_basis`` (tangential
unknown, ``rho_c`` term), ``boundary.surface_basis`` (tangential row),
``boundary.surface_functionals`` (``rho_w phi`` term, velocity under no slip),
``boundary.cmb_functional`` (``rho_c phi`` term), ``propagate.background_gravity``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
        ``g0`` in m/s^2, positive. CODE-A uses 10 (``grav()``, matrix.f:98); the
        G-ADOPT benchmark and ``assess`` use their own value.
    """

    value: float

    def __post_init__(self):
        if not self.value > 0:
            raise ValueError(f"a uniform background gravity must be positive, got {self.value!r}")
        object.__setattr__(self, "value", float(self.value))


class Boundary(Enum):
    """Boundary class at the surface or the CMB (eq. 8.1). ``u_r = 0`` in both."""

    FREE_SLIP = "free_slip"   # sigma_rt = 0
    NO_SLIP = "no_slip"       # u_t = 0


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
        ``GravityModel.FULL`` (default) or ``NO_SELF_GRAVITY``.
    background
        ``Hydrostatic()`` (default) or ``Uniform(value)``.
    compressible
        ``False`` (default): incompressible. ``True``: TALA, ``div(rho0 v) = 0``
        (eq. 1.2, 4.3).
    surface, cmb
        Boundary class at each boundary, default free slip.

    The record accepts every value; ``kernels.StokesConfig`` refuses the values that
    this chunk does not solve (NO_SELF_GRAVITY, TALA, no slip).
    """

    gravity: GravityModel = GravityModel.FULL
    background: Hydrostatic | Uniform = field(default_factory=Hydrostatic)
    compressible: bool = False
    surface: Boundary = Boundary.FREE_SLIP
    cmb: Boundary = Boundary.FREE_SLIP

    def __post_init__(self):
        # Type checks only: a wrong type would select a wrong static branch without
        # an error.
        if not isinstance(self.gravity, GravityModel):
            raise TypeError(f"gravity must be a GravityModel, got {self.gravity!r}")
        if not isinstance(self.background, (Hydrostatic, Uniform)):
            raise TypeError(f"background must be Hydrostatic() or Uniform(value), got {self.background!r}")
        if not isinstance(self.surface, Boundary) or not isinstance(self.cmb, Boundary):
            raise TypeError("surface and cmb must be Boundary members")
        object.__setattr__(self, "compressible", bool(self.compressible))


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
    the background, the boundary classes and whether the core is consistent
    (``core_mass is None``). The table is the port of ``reference_mpmath.frame_options``:

    - FULL, hydrostatic, free slip at both, consistent core: the raw system is
      singular for every mantle density (10a); CM only, CM default;
    - the same with a split core: CM default, raw offered;
    - FULL, hydrostatic, a no-slip boundary: the anchor carries no net force (10a),
      so degree 1 is refused by default; raw on explicit request for a split core;
    - NO_SELF_GRAVITY or a uniform background: a background field takes the net
      force, raw only (lovejx anchor convention).
    """
    free_body = formulation.gravity is GravityModel.FULL and isinstance(formulation.background, Hydrostatic)
    both_free = formulation.surface is Boundary.FREE_SLIP and formulation.cmb is Boundary.FREE_SLIP
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
