"""stokesjx: the instantaneous Stokes response of a spherically symmetric mantle.

stokesjx computes, per spherical-harmonic degree ``l`` and per source radius, the
response kernels of a viscous, self-gravitating, spherically symmetric mantle to a
sheet of internal density anomaly: the geoid, the dynamic topography of the surface
and of the core-mantle boundary (CMB), and the surface velocity. A field is the
radial integral of a kernel times the density anomaly. The physics is that of CODE-A
(Corrieu, Thoraval and Ricard 1995; Hager and Clayton 1989), with the errors of CODE-A
fixed (NOTES/PLAN.md section 4).

The method is a propagator method with two orthonormalised sweeps per degree: the
exact interval exponentials of all degrees in one batched Pade call, an upward and a
downward ``lax.scan`` of three-vector bases with Gram-Schmidt after every interval,
the output rows carried through the triangular factors, and one batched 6x6 solve at
every source node. It is stable in float64 at every degree, so it needs none of the
128-bit reals of CODE-A.

Two layers, as in lovejx. The low layer (``stokesjx.propagate``) is the solver, with
test-only switches that reproduce CODE-A. The high layer is the pure function
``stokes_kernels(profile, config)`` of a ``RadialProfile`` (a pytree of the arrays that
a derivative can move) and a static ``StokesConfig``, valid under ``jax.jit``,
``jax.grad``, ``jax.vmap`` and ``jax.jvp``. The derivatives are for Hamiltonian Monte
Carlo over the radial viscosity; the package does not choose a sampler.

Flags (``Formulation``): the gravity level (``GravityModel.FULL`` or
``NO_SELF_GRAVITY``), the background gravity (``Hydrostatic()`` or ``Uniform(value)``),
``compressible`` (TALA, the anelastic continuity ``div(rho0 v) = 0``) and the boundary
class at the surface and at the CMB (``FreeSlip()`` or ``NoSlip()``); every combination is
solved. Degree 1 is in the centre-of-mass frame, raw where the static rule offers it, or
refused (the kernels then start at ``l = 2``).

Fields (``stokesjx.fields``): ``apply(kernels, density, profile)`` integrates the four
kernels against a density anomaly with the trapezoid weights of the node table in one
einsum; ``gravity_anomaly`` and ``admittance`` are post-processing of the coefficients.
Spherical harmonics (``stokesjx.sh``): three coefficient conventions (``complex_4pi`` of
lovejx, ``real_4pi`` of SHTOOLS and of CODE-A's density files, ``real_orthonormal`` of
CODE-A's output) with one layout rule for every coefficient array (the degree and order
axes last, the ``C/S`` axis before them for a real convention), their conversions, a grid
synthesis and the Gauss-Legendre transform of lovejx. CODE-A's file formats are read and
written by ``stokesjx.io.legacy`` (imported on use).

Controls (``stokesjx.controls``): ``Layers`` maps one log viscosity per layer and the
radii of movable viscosity boundaries to a profile (the nodes of a range follow the
boundary affinely, ``rho0`` stays where the input profile puts it, and a density anomaly
stays on the input table through ``Layers.fields``); ``viscosity_by_node`` maps one log
viscosity per node; ``ControlVector`` turns a pytree of controls into one vector with a
jitted value and gradient and a NumPy front, and ``fields.spectral_misfit`` is a weighted
degree-power misfit. The prior and the sampler are the caller's.
"""

import jax as _jax

# Double precision, decided once and at import.
#
# The orthonormalised sweeps keep each step well conditioned, but every step still
# loses a few digits, and single precision (7 digits) leaves nothing for the
# derivatives. JAX defaults to 32-bit, and a silent demotion would show up as noise
# in the kernels, so the flag is set here and not left to the caller.
_jax.config.update("jax_enable_x64", True)

from .formulation import (  # noqa: E402
    BoundaryCondition,
    DegreeOneFrame,
    Formulation,
    FreeSlip,
    GravityModel,
    Hydrostatic,
    NoSlip,
    Uniform,
)
from .kernels import Kernels, Numerics, StokesConfig, stokes_kernels  # noqa: E402
from . import fields, sh  # noqa: E402
from .fields import Fields, admittance, apply, gravity_anomaly, quadrature_weights, surface_gravity  # noqa: E402
from .profile import NodeLimits, NodeStructure, RadialProfile, integrate_gravity  # noqa: E402
from .fields import spectral_misfit  # noqa: E402
from . import controls  # noqa: E402
from .controls import ControlVector, Layers, MovableRange, NodeViscosity, viscosity_by_node  # noqa: E402
from .units import Units  # noqa: E402

__version__ = "0.0.1.dev0"

__all__ = [
    "BoundaryCondition", "DegreeOneFrame", "Formulation", "FreeSlip", "GravityModel", "Hydrostatic",
    "NoSlip", "Uniform",
    "Kernels", "Numerics", "StokesConfig", "stokes_kernels",
    "Fields", "admittance", "apply", "gravity_anomaly", "quadrature_weights", "surface_gravity", "fields", "sh",
    "NodeLimits", "NodeStructure", "RadialProfile", "integrate_gravity", "Units",
    "spectral_misfit", "controls", "ControlVector", "Layers", "MovableRange", "NodeViscosity",
    "viscosity_by_node",
]
