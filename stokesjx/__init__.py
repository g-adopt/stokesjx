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

This version solves FULL self-gravitation, incompressible flow and free slip at both
boundaries, with a hydrostatic or a uniform background gravity, and degree 1 in the
centre-of-mass frame (or raw where the static rule offers it). The other flags
(NO_SELF_GRAVITY, TALA, no slip), the fields, the controls with movable boundaries
and the I/O follow in later versions.
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
    Boundary,
    DegreeOneFrame,
    Formulation,
    GravityModel,
    Hydrostatic,
    Uniform,
)
from .kernels import Kernels, Numerics, StokesConfig, stokes_kernels  # noqa: E402
from .profile import NodeStructure, RadialProfile, integrate_gravity  # noqa: E402
from .units import Units  # noqa: E402

__version__ = "0.0.1.dev0"

__all__ = [
    "Boundary", "DegreeOneFrame", "Formulation", "GravityModel", "Hydrostatic", "Uniform",
    "Kernels", "Numerics", "StokesConfig", "stokes_kernels",
    "NodeStructure", "RadialProfile", "integrate_gravity", "Units",
]
