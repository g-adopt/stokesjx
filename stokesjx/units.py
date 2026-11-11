"""Physical constants and the density scale of the state vector.

Two things live here: the values of the physical constants that enter the
solver (the gravitational constant ``G``, ``pi`` and the mass of the Earth for the
free-air conversion of a later chunk), and the density scale ``rho_s`` of the state
vector (NOTES/docs/equations.md, eq. 3.1).

There is no length scale and no viscosity scale. The radius comes from the node table
of the profile in metres, and the viscosity scale of the state is internal to the
solver (the largest node viscosity, ``propagate.reference_viscosity``). No kernel
depends on either scale in exact arithmetic (equations.md section 11), so neither is
a user choice.

Two modes, as in lovejx (``lovejx/lovenumbers/units.py``):

``legacy_constants=False`` (the default) uses CODATA 2018 ``G = 6.67430e-11
m^3 kg^-1 s^-2``, the exact ``pi`` and ``M_E = 5.9722e24 kg``.

``legacy_constants=True`` reproduces the arithmetic of CODE-A. gfortran without
``-fdefault-real-8`` rounds an untyped real literal to float32 before it promotes it
to double (equations.md, F13). So CODE-A's ``G`` is the float32 neighbour of
``6.67e-11`` (constants.h:8), its ``pi`` is the float32 neighbour of ``pi``
(constants.h:9; the literal has 40 digits, but it is single precision), and its
``Emass`` is the float32 neighbour of ``5.974e24`` (constants.h:12). The legacy ``pi``
of CODE-A differs from the legacy ``pi`` of lovejx, which is the float32 of the
decimal ``3.14159``: each switch reproduces the code that it is compared with.
The switch exists only for the comparison with the Fortran output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# CODATA 2018 gravitational constant, m^3 kg^-1 s^-2.
_G_CODATA = 6.67430e-11
# Mass of the Earth, kg (IAU 2015 nominal value of GM divided by G, rounded).
_EARTH_MASS = 5.9722e24

# CODE-A's constants, rounded to float32 as the compiled Fortran holds them (F13).
# float(np.float32(x)) is the exact binary value, so these equal the mpmath values
# reference_mpmath.LEGACY_G, LEGACY_PI and LEGACY_EMASS bit for bit.
_G_LEGACY = float(np.float32(6.67e-11))        # 6.66999996901118663573e-11
_PI_LEGACY = float(np.float32(math.pi))        # 3.14159274101257324219
_EARTH_MASS_LEGACY = float(np.float32(5.974e24))


@dataclass(frozen=True)
class Units:
    """Constants and the density scale of one run. Frozen and hashable, so that it
    can be part of the static configuration of a jitted function.

    Parameters
    ----------
    legacy_constants
        Reproduce CODE-A's float32 literals for ``G``, ``pi`` and the mass of the
        Earth (see the module docstring). Only for the comparison with the Fortran.
    density_scale
        ``rho_s`` in kg/m^3: the density unit of the state vector (eq. 3.1) and of the
        matrix entries ``a(3,6) = a(4,5) = -rho0 / rho_s`` (eq. 5.1). CODE-A's
        ``bgrho = 4500 kg/m^3`` (constants.h:80) is the default. No SI kernel depends
        on it in exact arithmetic (equations.md section 11; NOTES/chunks/c1-foundations/03-implementation.md, gate G6); in
        float64 a different value changes the kernels at the rounding level only, and
        it can change the Pade plan, because the static norm bound of the interval
        exponent contains ``rho0 / rho_s``.
    """

    legacy_constants: bool = False
    density_scale: float = 4500.0

    def __post_init__(self):
        # A non-positive scale would flip or remove the self-gravitation entries.
        if not self.density_scale > 0:
            raise ValueError(f"density_scale must be positive, got {self.density_scale!r}")
        # A plain Python float keeps the dataclass hashable and the jit key stable.
        object.__setattr__(self, "density_scale", float(self.density_scale))

    @property
    def G(self) -> float:
        """Gravitational constant, m^3 kg^-1 s^-2: ``6.67430e-11`` or CODE-A's
        ``float32(6.67e-11) = 6.66999996901118663573e-11``."""
        return _G_LEGACY if self.legacy_constants else _G_CODATA

    @property
    def pi(self) -> float:
        """``pi``, or CODE-A's ``float32(pi) = 3.14159274101257324219``. It enters every
        ``4 pi G`` (the potential jump of a sheet, the boundary potential conditions,
        the gravity integral)."""
        return _PI_LEGACY if self.legacy_constants else math.pi

    @property
    def earth_mass(self) -> float:
        """Mass of the Earth, kg: ``5.9722e24`` or CODE-A's ``float32(5.974e24)``. Used
        by CODE-A's free-air kernel (``fields.gravity_anomaly`` in the tests), never by the kernels."""
        return _EARTH_MASS_LEGACY if self.legacy_constants else _EARTH_MASS
