"""stokesjx: the instantaneous Stokes response of a spherically symmetric mantle.

stokesjx computes, per spherical-harmonic degree ``l`` and per source radius, the
response kernels of a viscous, self-gravitating, spherically symmetric mantle to a
sheet of internal density anomaly: the geoid, the dynamic topography of the surface
and of the core-mantle boundary (CMB), and the surface velocity. A field is the
radial integral of a kernel times the density anomaly. The method is the
propagator-matrix method of CODE-A (Corrieu, Thoraval and Ricard 1995; Hager and
Clayton 1989), with QR-stabilised sweeps in place of CODE-A's 128-bit reals.

The package will have two layers, as lovejx has. The low layer is the solver and
the reference for the tests. The high layer is a set of pure functions from the
controls (layer viscosities and the radii of movable layer boundaries) to the
kernels, the fields and a misfit, valid under ``jax.jit``, ``jax.grad``,
``jax.vmap`` and ``jax.jvp``. The derivatives are for Hamiltonian Monte Carlo over
the radial viscosity. The package does not choose a sampler.

At this stage the package holds no physics code. The reference solver lives in
``tests/reference_mpmath.py`` and the Fortran reference data in ``tests/golden/``.
"""

import jax as _jax

# Double precision, decided once and at import.
#
# A single product of propagators from the CMB to the surface holds the growing and
# the decaying solutions together. Their ratio over the mantle is about
# (R / r_c)^(2l + 3): 1e11 at l = 20 and 1e27 at l = 50. The QR-stabilised sweeps
# keep each step well conditioned, but every step still loses a few digits, and
# single precision (7 digits) leaves nothing for the derivatives. JAX defaults to
# 32-bit, and a silent demotion would show up as noise in the kernels, so the flag
# is set here and not left to the caller.
_jax.config.update("jax_enable_x64", True)

__version__ = "0.0.1.dev0"
