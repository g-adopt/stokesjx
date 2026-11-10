"""High-precision reference solver (the "oracle") for the instantaneous Stokes problem.

This module is test-only code. The package never imports it. It computes, in mpmath
at 60 or more significant digits, the response of a spherically symmetric,
self-gravitating, viscous mantle to a sheet of density anomaly, per spherical-harmonic
degree ``l`` and per source node. It is the reference for every later chunk of
stokesjx, and its own correctness rests on three independent checks: the theory gates
of tests/test_reference_oracle.py (exact identities that a wrong equation breaks), the
CODE-A Fortran output in tests/golden/fortran (through the legacy switch values of
``legacy_setup``), and a line-by-line transcription of the Fortran 8x8 system
(``transcription_kernels``).

The equation numbers in the comments refer to NOTES/docs/equations.md.

Method ("single shooting"). The state vector in the variable ``ln r`` is (eq. 3.1)

    y = (u_r, u_t, r sigma_rr / eta_s, r sigma_rt / eta_s,
         rho_s r phi / eta_s, rho_s r^2 dphi/dr / eta_s),

all six components in m/s, with the scales ``rho_s`` (density) and ``eta_s``
(viscosity). It obeys ``dy/d ln r = A y`` with A constant in each interval of the node
table (interval-mean material, eq. 12.1), so the step across an interval is exactly
``expm(A h)``, ``h = ln(r_i / r_(i-1))``. The oracle forms the single product of the
steps from each node to the surface, applies the three CMB conditions as a basis of
three CMB states, and solves the three surface conditions as a 3x3 system per
source. The product holds growing and decaying solutions with a ratio of about
``(R / r_c)^(2l+3)`` (1e27 at l = 50, 1e54 at l = 100), so the working precision is
raised with ``l`` (``required_dps``). Every kernel is rounded to float64 only at the
end.

One code path computes every case. The physics flags of PLAN section 5 live in
``Physics``; the discrete-model choices live in ``Discretisation``, whose defaults are
the stokesjx model and whose other values reproduce CODE-A (``legacy_setup``).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import mpmath as mp
import numpy as np

# --------------------------------------------------------------------------
# Configuration records. All three are frozen and hold strings for every number,
# so that they are hashable (the model cache below is keyed by them) and so that
# mpf(str) converts each number exactly.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeTable:
    """Node table of one radial profile, CMB first.

    Equal neighbouring radii are a jump pair: an interval of zero width across which
    the viscosity, the density or both jump.

    Attributes
    ----------
    radius
        Node radii in m, non-decreasing, CMB first.
    rho0
        Reference density at the nodes, kg/m^3.
    eta
        Viscosity at the nodes, Pa s.
    core_density
        ``rho_c``, kg/m^3: the density of the inviscid core in the CMB condition
        (the core pressure ``rho_c phi`` and the density contrast of the CMB
        deflection, eq. 8.4).
    core_mass
        Mass below the CMB in kg, used only by the hydrostatic gravity (eq. 6.1).
        ``None`` means the consistent uniform core ``(4 pi / 3) r_c^3 rho_c``. A
        number, even one equal to the consistent value, selects the "split core"
        rules of the degree-1 frame (``degree_one_rule``), and with a number equal
        to the consistent value an explicit raw degree-1 request meets a singular
        system, which the solver refuses through its singular-value assertion.
    outer_density
        ``rho_w``, kg/m^3: the density of the layer above the surface (water). It
        enters only the conversion from the surface stress to the surface
        topography (eq. 8.3, F17).
    """

    radius: tuple[str, ...]
    rho0: tuple[str, ...]
    eta: tuple[str, ...]
    core_density: str
    core_mass: str | None = None
    outer_density: str = "0"

    def __post_init__(self):
        n = len(self.radius)
        if len(self.rho0) != n or len(self.eta) != n:
            raise ValueError("radius, rho0 and eta must have the same length")
        if n < 2:
            raise ValueError("a node table needs at least two nodes")

    @property
    def n_nodes(self) -> int:
        """Number of nodes, including both nodes of every jump pair."""
        return len(self.radius)


@dataclass(frozen=True)
class Physics:
    """Continuous model: the physics flags of PLAN section 5.

    Attributes
    ----------
    self_gravity
        ``"full"``: the perturbed potential pushes on ``rho0`` (entries a(3,6) and
        a(4,5) of A), on the core (``rho_c phi`` at the CMB) and on the water
        (``rho_w phi`` in the surface topography). ``"none"``: these three terms are
        off; the geoid is still computed from the load and the boundary sheets.
    background
        ``"hydrostatic"``: ``g(r)`` integrated from the interval-mean density and the
        core mass (eq. 6.1). ``"uniform"``: ``g = uniform_g`` everywhere.
    compressible
        TALA (anelastic continuity ``div(rho0 v) = 0``) when true, eq. 4.3.
    surface, cmb
        Boundary class at each boundary: ``"free_slip"`` (``sigma_rt = 0``) or
        ``"no_slip"`` (``u_t = 0``). ``u_r = 0`` at both boundaries in both classes.
    degree_one
        Frame of the ``l = 1`` solution (F7f). ``"default"`` resolves through
        ``degree_one_rule``; a frame that the rule does not offer for the
        configuration raises ``ValueError``.
    G
        Gravitational constant, m^3 kg^-1 s^-2 (overridden by the legacy constants
        switch of ``Discretisation``).
    density_scale, viscosity_scale
        ``rho_s`` (kg/m^3) and ``eta_s`` (Pa s) of the state vector. No output may
        depend on them, except the legacy surface topography, which uses ``rho_s``
        as the surface density (F3, gate G6).
    """

    self_gravity: Literal["full", "none"] = "full"
    background: Literal["hydrostatic", "uniform"] = "hydrostatic"
    uniform_g: str = "9.81"
    compressible: bool = False
    surface: Literal["free_slip", "no_slip"] = "free_slip"
    cmb: Literal["free_slip", "no_slip"] = "free_slip"
    degree_one: Literal["default", "raw", "centre_of_mass", "legacy_geoid_zero"] = "default"
    G: str = "6.67430e-11"
    density_scale: str = "4500"
    viscosity_scale: str = "1e19"


@dataclass(frozen=True)
class Discretisation:
    """Discrete-model switches. The defaults are the stokesjx model.

    Attributes
    ----------
    material
        ``"interval_mean"``: the viscosity and density of an interval are the mean of
        its two node values (stokesjx, PLAN row 3). ``"upper_node"``: the values of
        the upper node, as ``aconstruct`` (driver.f:362-363).
    kappa_rule
        Compressible step. ``"log_slope"``: ``kappa h = ln(rho0_i / rho0_(i-1))``
        (stokesjx, F8). ``"legacy_one_sided"``: CODE-A's one-sided slope at the upper
        node times ``h`` (driver.f:429-430).
    pair_rule
        Step across a jump pair under TALA. ``"anelastic"``: ``rho0 u_r`` continuous,
        the ``h -> 0`` limit of the interval step (F8, F18). ``"material"``: the
        identity (``u_r`` continuous); for measurement only.
    potential_form
        ``"local"``: the potential conditions at the surface and the CMB as local
        rows (F5). ``"global"``: CODE-A's rows 7 and 8 (matrix.f:141-148), the
        Green's-function sums of the sheet potentials.
    topography_density
        Density in the conversion from the surface stress to the surface
        topography: ``"top_node"`` (``rho0`` of the surface node) or
        ``"density_scale"`` (``rho_s``, CODE-A's ``bgrho``, F3).
    constants
        ``"codata"``: ``Physics.G`` and the exact ``pi``. ``"legacy_float32"``: the
        float32-rounded literals of CODE-A (F13) for ``G``, ``pi`` and ``Emass``.
    output
        ``"si"``: the four stokesjx kernels in SI units. ``"fortran"``: the five
        CODE-A kernels with CODE-A's names and formulas (driver.f:254-273).
    """

    material: Literal["interval_mean", "upper_node"] = "interval_mean"
    kappa_rule: Literal["log_slope", "legacy_one_sided"] = "log_slope"
    pair_rule: Literal["anelastic", "material"] = "anelastic"
    potential_form: Literal["local", "global"] = "local"
    topography_density: Literal["top_node", "density_scale"] = "top_node"
    constants: Literal["codata", "legacy_float32"] = "codata"
    output: Literal["si", "fortran"] = "si"


SI_OUTPUTS = ("geoid", "surface_topography", "cmb_topography", "surface_velocity")
FORTRAN_OUTPUTS = ("geoid", "surftopo", "cmbtopo", "surfvel", "gravity")

# --------------------------------------------------------------------------
# Constants of CODE-A. gfortran without -fdefault-real-8 rounds an untyped real
# literal to float32 before it is promoted (F13), so CODE-A's G, pi, visc0 and Emass
# are the float32 neighbours of their decimal values. mpf(float) is exact, so these
# are the exact binary values that the Fortran holds.
# --------------------------------------------------------------------------


def _f32(x: float) -> mp.mpf:
    """The float32 value nearest to ``x``, as an exact mpf."""
    return mp.mpf(float(np.float32(x)))


LEGACY_G = _f32(6.67e-11)          # constants.h:8, grv_uni_const
LEGACY_PI = _f32(math.pi)          # constants.h:9, pi (the literal is float32)
LEGACY_VISC0 = _f32(1e19)          # constants.h:82, visc0 = 9999999980506447872
LEGACY_EMASS = _f32(5.974e24)      # constants.h:12, Emass (free-air kernel only)
LEGACY_BGRHO = "4500"              # constants.h:80, exact in float32
LEGACY_RHO_C = "9900"              # constants.h:85, exact in float32
LEGACY_G_SURFACE = "10"            # matrix.f:98, grav() returns 10.0


def legacy_visc0_string() -> str:
    """The exact decimal value of CODE-A's ``visc0`` (float32 of 1e19), Pa s."""
    return str(int(np.float32(1e19)))


def legacy_setup(bdry: int, bdry_bot: int, kappaindex: int) -> tuple[Physics, Discretisation]:
    """Physics and discrete-model switches that reproduce a CODE-A build.

    Parameters
    ----------
    bdry, bdry_bot
        The compile-time boundary flags of constants.h: 1 is free slip, 0 is no slip,
        at the surface and the CMB respectively.
    kappaindex
        1 for the compressible matrix with CODE-A's one-sided kappa, 0 otherwise.

    Returns
    -------
    (Physics, Discretisation)
        Uniform ``g = 10`` (matrix.f:98), the degree-1 frame of CODE-A (raw solve,
        geoid set to zero, driver.f:282-284), ``rho_s = bgrho = 4500``,
        ``eta_s = float32(1e19)``, upper-node material, one-sided kappa, the local
        potential rows (F5: equal to rows 7 and 8, so the Fortran checks the local
        form directly), ``bgrho`` as surface density, the float32 constants and the
        Fortran output formulas.
    """
    surface = "free_slip" if bdry == 1 else "no_slip"
    cmb = "free_slip" if bdry_bot == 1 else "no_slip"
    physics = Physics(
        self_gravity="full", background="uniform", uniform_g=LEGACY_G_SURFACE,
        compressible=bool(kappaindex), surface=surface, cmb=cmb,
        degree_one="legacy_geoid_zero", density_scale=LEGACY_BGRHO,
        viscosity_scale=legacy_visc0_string())
    disc = Discretisation(
        material="upper_node", kappa_rule="legacy_one_sided", pair_rule="anelastic",
        potential_form="local", topography_density="density_scale",
        constants="legacy_float32", output="fortran")
    return physics, disc


# --------------------------------------------------------------------------
# Precision.
# --------------------------------------------------------------------------

DPS_FLOOR = 60  # every oracle computation runs at 60 digits or more (review r3, minor 1)


def required_dps(l: int, r_cmb="3480000", r_surface="6370000", digits: int = 30) -> int:
    """Working precision (decimal digits) for degree ``l`` (F9).

    The single product of propagators holds a dynamic range of about
    ``(R / r_c)^(2l+3)``, so ``(2l + 3) log10(R / r_c)`` digits cancel in the 3x3
    solve. ``digits`` is the number of correct digits wanted after that loss, and 10
    more are a margin for the matrix exponentials and the products. The result is
    never below ``DPS_FLOOR``.
    """
    ratio = float(mp.mpf(r_surface) / mp.mpf(r_cmb))
    loss = math.ceil((2 * l + 3) * math.log10(ratio))
    return max(DPS_FLOOR, digits + loss + 10)


# --------------------------------------------------------------------------
# The matrix A (eq. 4.1 to 4.4).
# --------------------------------------------------------------------------


def matrix_a_rel(l: int, eta_rel, rho_rel, self_gravity: bool) -> mp.matrix:
    """Matrix A0 in ``ln r`` for one interval, incompressible part (kappa = 0).

    ``eta_rel = eta / eta_s`` and ``rho_rel = rho0 / rho_s`` are the dimensionless
    interval material. Row by row (driver.f:369-398, eq. 4.1):

    1. continuity: ``d y1 = -2 y1 + L y2``;
    2. definition of ``sigma_rt``: ``d y2 = -y1 + y2 + y4 / eta``;
    3. radial momentum with the constitutive law: ``d y3 = 12 eta y1 - 6 L eta y2
       + y3 + L y4 - rho0/rho_s y6``;
    4. tangential momentum: ``d y4 = -6 eta y1 + 2 (2L - 1) eta y2 - y3 - 2 y4
       - rho0/rho_s y5``;
    5. and 6. Laplace's equation for the perturbed potential: ``d y5 = y5 + y6``,
       ``d y6 = L y5``.

    The entries ``a(3,6)`` and ``a(4,5)`` are the force of the perturbed potential
    on ``rho0``; they are zero without self-gravitation.
    """
    L = l * (l + 1)
    a = mp.zeros(6, 6)
    a[0, 0] = -2
    a[0, 1] = L
    a[1, 0] = -1
    a[1, 1] = 1
    a[1, 3] = 1 / eta_rel
    a[2, 0] = 12 * eta_rel
    a[2, 1] = -6 * L * eta_rel
    a[2, 2] = 1
    a[2, 3] = L
    a[3, 0] = -6 * eta_rel
    a[3, 1] = 2 * (2 * L - 1) * eta_rel
    a[3, 2] = -1
    a[3, 3] = -2
    if self_gravity:
        a[2, 5] = -rho_rel
        a[3, 4] = -rho_rel
    a[4, 4] = 1
    a[4, 5] = 1
    a[5, 4] = L
    return a


def matrix_k_rel(eta_rel) -> mp.matrix:
    """``K = dA/dkappa`` (eq. 4.3, F8): K(1,1) = -1, K(3,1) = 4 eta, K(4,1) = -2 eta.

    K is ``v e1^T`` with ``e1^T v = -1``, so ``K^2 = -K`` and
    ``expm(K t) = I + K (1 - exp(-t))``.
    """
    k = mp.zeros(6, 6)
    k[0, 0] = -1
    k[2, 0] = 4 * eta_rel
    k[3, 0] = -2 * eta_rel
    return k


def matrix_a(l: int, eta, rho0, physics: Physics) -> mp.matrix:
    """Matrix A0 for SI material ``eta`` (Pa s) and ``rho0`` (kg/m^3) (eq. 4.1)."""
    return matrix_a_rel(l, mp.mpf(eta) / mp.mpf(physics.viscosity_scale),
                        mp.mpf(rho0) / mp.mpf(physics.density_scale),
                        physics.self_gravity == "full")


def matrix_k(eta, physics: Physics | None = None) -> mp.matrix:
    """``K = dA/dkappa`` for SI viscosity ``eta`` (eq. 4.3)."""
    scale = mp.mpf(physics.viscosity_scale) if physics is not None else mp.mpf("1e19")
    return matrix_k_rel(mp.mpf(eta) / scale)


# --------------------------------------------------------------------------
# The model of one (table, physics, discretisation, degree) at one precision.
# --------------------------------------------------------------------------


@dataclass
class _Constants:
    G: mp.mpf
    pi: mp.mpf
    emass: mp.mpf


def _constants(physics: Physics, disc: Discretisation) -> _Constants:
    if disc.constants == "legacy_float32":
        return _Constants(G=LEGACY_G, pi=LEGACY_PI, emass=LEGACY_EMASS)
    return _Constants(G=mp.mpf(physics.G), pi=+mp.pi, emass=mp.mpf("5.9722e24"))


class _Model:
    """All degree-dependent data of one configuration, built at the current dps.

    Built only through ``_model`` (cached). Attributes are mp values. Index ``i`` of
    an interval runs from 1 to n-1 and denotes the interval between nodes ``i-1``
    and ``i``.
    """

    def __init__(self, table: NodeTable, physics: Physics, disc: Discretisation, l: int):
        self.table, self.physics, self.disc, self.l = table, physics, disc, l
        c = _constants(physics, disc)
        self.G, self.pi, self.emass = c.G, c.pi, c.emass
        self.full = physics.self_gravity == "full"
        n = table.n_nodes
        self.n = n
        r = [mp.mpf(s) for s in table.radius]
        rho = [mp.mpf(s) for s in table.rho0]
        eta = [mp.mpf(s) for s in table.eta]
        if any(r[i] < r[i - 1] for i in range(1, n)):
            raise ValueError("radii must be non-decreasing")
        self.r, self.rho, self.eta = r, rho, eta
        self.rho_s = mp.mpf(physics.density_scale)
        self.eta_s = mp.mpf(physics.viscosity_scale)
        self.rho_c = mp.mpf(table.core_density)
        self.rho_w = mp.mpf(table.outer_density)
        self.R, self.rc = r[-1], r[0]

        # Interval data (eq. 12.1). A pair (zero width) is detected from the radii,
        # which are exact decimal strings, so the test is exact.
        self.pair = [False] + [r[i] == r[i - 1] for i in range(1, n)]
        self.h = [mp.mpf(0)] + [mp.mpf(0) if self.pair[i] else mp.log(r[i] / r[i - 1])
                              for i in range(1, n)]
        if disc.material == "interval_mean":
            self.eta_int = [None] + [(eta[i - 1] + eta[i]) / 2 for i in range(1, n)]
            self.rho_int = [None] + [(rho[i - 1] + rho[i]) / 2 for i in range(1, n)]
        else:
            # aconstruct(a, i, l) for the interval (r(i-1), r(i)) uses rvsc(i) and
            # rho0(i), the upper node (driver.f:362-363).
            self.eta_int = [None] + [eta[i] for i in range(1, n)]
            self.rho_int = [None] + [rho[i] for i in range(1, n)]
        self.kappa_h = [mp.mpf(0)] * n
        if physics.compressible:
            for i in range(1, n):
                if disc.kappa_rule == "log_slope":
                    # kappa h = ln(rho0_i / rho0_(i-1)): exact for a power law
                    # rho0 ~ r^kappa inside the interval; finite at a pair (F8).
                    self.kappa_h[i] = mp.log(rho[i] / rho[i - 1])
                else:
                    if self.pair[i]:
                        raise ValueError("the legacy one-sided kappa divides by the interval width")
                    # kappa(ind) = (r(ind)/rho0(ind)) (rho0(ind)-rho0(ind-1))/(r(ind)-r(ind-1))
                    # (driver.f:429-430), times h for the step argument.
                    kappa = (r[i] / rho[i]) * (rho[i] - rho[i - 1]) / (r[i] - r[i - 1])
                    self.kappa_h[i] = kappa * self.h[i]

        self._gravity()
        self._steps()
        self._products()
        self.cs = 4 * self.pi * self.G * self.R * self.rho_s / self.g[-1]
        self.cc = 4 * self.pi * self.G * self.rc * self.rho_s / self.g[0]

    # ----- gravity (eq. 6.1) -----
    def _gravity(self):
        r, rho, n = self.r, self.rho, self.n
        G, pi = self.G, self.pi
        if self.table.core_mass is None:
            mc = 4 * pi / 3 * r[0] ** 3 * self.rho_c
        else:
            mc = mp.mpf(self.table.core_mass)
        mass = [mc]
        for i in range(1, n):
            # Shell of constant density at the interval mean (as lovejx
            # integrate_gravity): exact mass for the piecewise-constant model. A
            # pair contributes no mass.
            mean = (rho[i - 1] + rho[i]) / 2
            mass.append(mass[-1] + 4 * pi / 3 * mean * (r[i] ** 3 - r[i - 1] ** 3))
        self.core_mass = mc
        self.mass = mass
        self.mantle_mass = mass[-1] - mc
        if self.physics.background == "hydrostatic":
            self.g = [G * mass[i] / r[i] ** 2 for i in range(n)]
        else:
            g0 = mp.mpf(self.physics.uniform_g)
            self.g = [g0] * n

    # ----- interval steps (eq. 12.2) -----
    def _steps(self):
        l, n = self.l, self.n
        self.steps = [None] * n
        for i in range(1, n):
            e = self.eta_int[i] / self.eta_s
            if self.pair[i]:
                if self.physics.compressible and self.disc.pair_rule == "anelastic":
                    # expm(K t) with t = ln(rho0_+ / rho0_-): I + K (1 - rho0_- / rho0_+).
                    t = 1 - self.rho[i - 1] / self.rho[i]
                    self.steps[i] = mp.eye(6) + matrix_k_rel(e) * t
                else:
                    self.steps[i] = mp.eye(6)
                continue
            # Viscosity similarity (eq. 4.5): A(eta) = D A(1) D^-1 with
            # D = diag(1, 1, eta, eta, eta, eta), exactly, also for K. The
            # exponential is taken of the matrix without viscosity, whose norm is
            # of order L h, and scaled back. This is an exact identity, so it only
            # improves the conditioning of mp.expm.
            arg = matrix_a_rel(l, 1, self.rho_int[i] / self.rho_s, self.full) * self.h[i]
            if self.kappa_h[i] != 0:
                arg += matrix_k_rel(1) * self.kappa_h[i]
            E = mp.expm(arg, method="taylor")
            for a_ in range(6):
                for b_ in range(6):
                    da = 1 if a_ < 2 else e
                    db = 1 if b_ < 2 else e
                    if da is not db:
                        E[a_, b_] = E[a_, b_] * da / db
            self.steps[i] = E

    def _products(self):
        """``up[j] = P(R <- r_j)``, the propagator from node j to the surface."""
        n = self.n
        up = [None] * n
        up[n - 1] = mp.eye(6)
        for j in range(n - 2, -1, -1):
            up[j] = up[j + 1] * self.steps[j + 1]
        self.up = up

    # ----- boundary data -----
    def jump(self, j, sigma=1) -> mp.matrix:
        """Jump ``y(r_j+) - y(r_j-)`` of a sheet of ``sigma`` kg/m^2 at node j (eq. 7.1).

        ``[y3] = r g sigma / eta_s`` (the sheet's weight carried by the radial
        stress) and ``[y6] = -4 pi G r^2 rho_s sigma / eta_s`` (the jump of the
        potential gradient).
        """
        r, g = self.r[j], self.g[j]
        v = mp.zeros(6, 1)
        v[2] = r * g * sigma / self.eta_s
        v[5] = -4 * self.pi * self.G * r * r * self.rho_s * sigma / self.eta_s
        return v

    def cmb_vector(self, comp: str) -> mp.matrix:
        """State at the CMB with one unit unknown, ``y1 = 0`` and the local CMB
        potential condition (eq. 8.5, F5):

            y6 = l y5 - c_c (y3 + (rho_c / rho_s) y5)     (FULL)
            y6 = l y5 - c_c y3                            (NO_SELF_GRAVITY)

        with ``c_c = 4 pi G r_c rho_s / g(r_c)``: the interior potential of the core
        is regular (``~ r^l``), and the CMB carries the sheet
        ``(sigma_rr + rho_c phi) / g``.
        """
        y = mp.zeros(6, 1)
        idx = {"y2": 1, "y3": 2, "y4": 3, "y5": 4}[comp]
        y[idx] = 1
        rc_rel = self.rho_c / self.rho_s if self.full else 0
        y[5] = self.l * y[4] - self.cc * (y[2] + rc_rel * y[4])
        return y

    def cmb_basis(self, cmb_class: str | None = None) -> list[mp.matrix]:
        """Three CMB states: (y3, y2, y5) under free slip, (y3, y4, y5) under no slip."""
        cmb_class = cmb_class or self.physics.cmb
        tang = "y2" if cmb_class == "free_slip" else "y4"
        return [self.cmb_vector("y3"), self.cmb_vector(tang), self.cmb_vector("y5")]

    def surface_functional(self, name: str, y) -> mp.mpf:
        """Linear functionals of the state at the surface (eq. 8.1, 8.2, 10.5)."""
        l = self.l
        if name == "ur":
            return y[0]
        if name == "ut":
            return y[1]
        if name == "srt":
            return y[3]
        if name == "y5":
            return y[4]
        if name == "raw":
            # Exterior potential ~ r^-(l+1), surface sheet -sigma_rr / g:
            # y6 + (l+1) y5 + c_s y3 = 0.
            return y[5] + (l + 1) * y[4] + self.cs * y[2]
        if name == "cm":
            # Exterior potential ~ r^l (uniform applied field at l = 1), no self
            # dipole: y6 - l y5 + c_s y3 = 0 (F7e).
            return y[5] - l * y[4] + self.cs * y[2]
        raise ValueError(name)

    def surface_rows(self, potential_row: str = "raw", surface_class: str | None = None,
                     replace_row: str | None = None) -> list[str]:
        """Names of the three surface rows.

        ``replace_row`` in {"radial", "tangential", "potential"} replaces that row of
        the raw system by ``y5(R) = 0`` (diagnostics of G8b only).
        """
        surface_class = surface_class or self.physics.surface
        tang = "srt" if surface_class == "free_slip" else "ut"
        rows = ["ur", tang, potential_row]
        if replace_row is not None:
            rows[{"radial": 0, "tangential": 1, "potential": 2}[replace_row]] = "y5"
        return rows

    def particular(self, source) -> mp.matrix:
        """Surface value of the particular solution: the sum of ``P(R <- r_j) jump_j``."""
        if isinstance(source, dict):
            items = source.items()
        else:
            items = [(source, 1)]
        y = mp.zeros(6, 1)
        for j, s in items:
            y += self.up[j] * self.jump(j, mp.mpf(s))
        return y

    def system(self, rows, basis):
        """3x3 matrix of the surface rows applied to the propagated CMB basis."""
        top = [self.up[0] * b for b in basis]
        S = mp.matrix(3, 3)
        for a_, name in enumerate(rows):
            for b_, t in enumerate(top):
                S[a_, b_] = self.surface_functional(name, t)
        return S, top


@functools.lru_cache(maxsize=64)
def _model_cached(table, physics, disc, l, dps):
    with mp.workdps(dps):
        return _Model(table, physics, disc, l)


def _model(table, physics, disc, l, dps=None):
    dps = dps or required_dps(l, table.radius[0], table.radius[-1])
    return _model_cached(table, physics, disc, l, dps), dps


def scaled_ratio(S: mp.matrix) -> mp.mpf:
    """Smallest over largest singular value of ``S`` after each row is scaled to a
    largest entry of one. Scale-free measure of how singular a 3x3 system is."""
    S = S.copy()
    for a_ in range(S.rows):
        mx = max(abs(S[a_, b_]) for b_ in range(S.cols))
        if mx != 0:
            for b_ in range(S.cols):
                S[a_, b_] /= mx
    sv = mp.svd_r(S, compute_uv=False)
    return min(sv) / max(sv)


def _svd_scaled(S: mp.matrix):
    S = S.copy()
    for a_ in range(S.rows):
        mx = max(abs(S[a_, b_]) for b_ in range(S.cols))
        if mx != 0:
            for b_ in range(S.cols):
                S[a_, b_] /= mx
    return mp.svd_r(S)


# --------------------------------------------------------------------------
# The degree-1 frame rule (F7f, review r3 R2).
# --------------------------------------------------------------------------


def frame_options(table: NodeTable, physics: Physics) -> dict:
    """Static degree-1 frame table of F7(f) with review r3 R2.

    Depends only on the physics flags, the boundary classes and whether the core is
    consistent (``core_mass is None``). No numerical value of the table enters.

    Returns
    -------
    dict
        ``default``: the default frame (``"centre_of_mass"``, ``"raw"`` or
        ``"refused"``); ``offered``: frames available on request;
        ``explicit_only``: frames available only on an explicit request, with the
        reason in ``note``.
    """
    free_body_physics = physics.self_gravity == "full" and physics.background == "hydrostatic"
    both_free = physics.surface == "free_slip" and physics.cmb == "free_slip"
    consistent = table.core_mass is None
    if free_body_physics and both_free:
        if consistent:
            # The raw system is singular for every mantle density (F7a); the CM row
            # gives the member with N_1 = 0 (F7e item 1).
            return dict(default="centre_of_mass", offered=("centre_of_mass",), explicit_only=(),
                        note="free body, consistent core: raw system singular")
        return dict(default="centre_of_mass", offered=("centre_of_mass", "raw"), explicit_only=(),
                    note="free body, split core: CM with the reported imbalance; raw also offered")
    if free_body_physics:
        # A no-slip boundary on a free body: the anchor carries no net force, the
        # raw solution exists only through the flow of the null vector and answers
        # a changed load (review r3, 3.1). Refused by default for both core models.
        if consistent:
            return dict(default="refused", offered=(), explicit_only=(),
                        note="no-slip boundary, consistent core: raw singular for uniform density")
        return dict(default="refused", offered=(), explicit_only=("raw",),
                    note=("no-slip boundary, split core: the raw value is set by the core split "
                          "and by the flow of the null vector (review r3 R2)"))
    # NO_SELF_GRAVITY or a uniform g: a background field takes the net force, the
    # anchor convention of lovejx applies, raw only.
    return dict(default="raw", offered=("raw",), explicit_only=(),
                note="no self-gravitation or uniform g: raw frame")


def degree_one_rule(table: NodeTable, physics: Physics) -> str:
    """Resolve ``physics.degree_one`` for this configuration (F7f).

    Returns ``"centre_of_mass"``, ``"raw"``, ``"legacy_geoid_zero"`` or ``"refused"``
    (only for ``"default"``). A request for a frame that the rule does not offer
    raises ``ValueError``.
    """
    opts = frame_options(table, physics)
    req = physics.degree_one
    if req == "default":
        return opts["default"]
    available = opts["offered"] + opts["explicit_only"]
    if req == "legacy_geoid_zero":
        if "raw" in available:
            return req
        raise ValueError("legacy_geoid_zero needs a raw degree-1 solve, which is not offered here: "
                         + opts["note"])
    if req in available:
        return req
    raise ValueError(f"degree-1 frame {req!r} is not offered: {opts['note']}")


# --------------------------------------------------------------------------
# Solves.
# --------------------------------------------------------------------------


def _solve_local(m: _Model, source, frame: str, *, rows=None, cmb_class=None,
                 check_ratio=True):
    """Local 3x3 solve for one source (int node index or dict {index: strength}).

    Returns the CMB coefficients, the surface and CMB states, and the scaled
    singular-value ratio of the solved system.
    """
    l = m.l
    if rows is None:
        potential_row = "cm" if frame == "centre_of_mass" else "raw"
        rows = m.surface_rows(potential_row)
    basis = m.cmb_basis(cmb_class)
    S, top = m.system(rows, basis)
    ratio = scaled_ratio(S) if l == 1 else None
    if check_ratio and l == 1 and ratio < mp.mpf(10) ** (-(mp.mp.dps // 3)):
        raise ArithmeticError(f"degree-1 system is singular to working precision (ratio {mp.nstr(ratio, 3)})")
    yp = m.particular(source)
    rhs = mp.matrix([-m.surface_functional(name, yp) for name in rows])
    x = mp.lu_solve(S, rhs)
    yR = yp.copy()
    yc = mp.zeros(6, 1)
    for k in range(3):
        yR += x[k] * top[k]
        yc += x[k] * basis[k]
    return dict(x=x, yR=yR, yc=yc, ratio=ratio, S=S, rows=rows)


def _solve_global(m: _Model, source):
    """CODE-A's 8x8 system with rows 7 and 8 (matrix.f:126-165), generalised to the
    stokesjx gravity (``g(R)``, ``g(r_c)``) and to NO_SELF_GRAVITY (no ``rho_c``
    terms). Unknowns: surface (y3, y2|y4, y5, y6), CMB (y3, y2|y4, y5, y6)."""
    l = m.l
    R, rc = m.R, m.rc
    P = m.up[0]
    fac = 4 * m.pi * m.G / (2 * l + 1)
    gR, gc = m.g[-1], m.g[0]
    rs = m.rho_s
    rc_rel = m.rho_c / rs if m.full else 0
    C = mp.zeros(8, 8)
    surf_t = 1 if m.physics.surface == "free_slip" else 3     # y2 or y4 at the surface
    cmb_t = 1 if m.physics.cmb == "free_slip" else 3
    # rows 1-6: y(R) - P y(r_c) = sum_j P(R <- r_j) jump_j
    sidx = [2, surf_t, 4, 5]
    cidx = [2, cmb_t, 4, 5]
    for k in range(6):
        for u, comp in enumerate(sidx):
            C[k, u] = 1 if k == comp else 0
        for u, comp in enumerate(cidx):
            C[k, 4 + u] = -P[k, comp]
    # row 7: potential at R from the surface sheet, the CMB sheet and the load.
    c71 = fac * R / gR * rs
    c75 = -fac * rc / gc * rs * (rc / R) ** l
    C[6, 0] = c71
    C[6, 2] = 1
    C[6, 4] = c75
    C[6, 6] = c75 * rc_rel
    # row 8: potential at r_c.
    C[7, 0] = c71 * (rc / R) ** (l + 1)
    C[7, 4] = -fac * rc / gc * rs
    C[7, 6] = 1 - fac * rc / gc * rs * rc_rel
    b = mp.zeros(8, 1)
    yp = m.particular(source)
    for k in range(6):
        b[k] = yp[k]
    items = source.items() if isinstance(source, dict) else [(source, 1)]
    for j, s in items:
        rj = m.r[j]
        b[6] += mp.mpf(s) * rs / m.eta_s * fac * (rj / R) ** l * rj ** 2
        b[7] += mp.mpf(s) * rs / m.eta_s * fac * (rc / rj) ** (l - 1) * rc ** 2
    u = mp.lu_solve(C, b)
    yR = mp.zeros(6, 1)
    yc = mp.zeros(6, 1)
    for k, comp in enumerate(sidx):
        yR[comp] = u[k]
    for k, comp in enumerate(cidx):
        yc[comp] = u[4 + k]
    return dict(yR=yR, yc=yc, ratio=None)


def _outputs(m: _Model, sol: dict, frame: str, source) -> dict:
    """Kernels from the surface and CMB states (eq. 9.1 to 9.4), SI or Fortran."""
    l = m.l
    L = l * (l + 1)
    yR, yc = sol["yR"], sol["yc"]
    eta_s, rho_s = m.eta_s, m.rho_s
    R, rc, gR, gc = m.R, m.rc, m.g[-1], m.g[0]
    full = m.full
    # Self and applied parts of the potential at R (F7e). y6p is the value just
    # above the surface sheet.
    y6p = yR[5] + m.cs * yR[2]
    y5_self = (l * yR[4] - y6p) / (2 * l + 1)
    y5_app = ((l + 1) * yR[4] + y6p) / (2 * l + 1)
    if frame == "centre_of_mass" or (frame == "legacy_geoid_zero" and l == 1):
        # CM frame: the self part is zero by construction (stored as exact zero).
        # CODE-A's frame: the raw solve with the geoid set to zero (driver.f:282-284).
        y5_geoid = mp.mpf(0)
    else:
        y5_geoid = yR[4]
    geoid = eta_s * y5_geoid / (rho_s * R * gR)
    rho_top = m.rho[-1] if m.disc.topography_density == "top_node" else rho_s
    # Surface topography: the water surface follows the total equipotential, so the
    # total potential enters (eq. 8.3). Without self-gravitation the rho_w phi term
    # is off.
    phiR = eta_s * yR[4] / (rho_s * R)
    sigma_rr_R = eta_s * yR[2] / R
    rho_w_term = m.rho_w * phiR if full else 0
    hs = -(sigma_rr_R + rho_w_term) / ((rho_top - m.rho_w) * gR)
    phic = eta_s * yc[4] / (rho_s * rc)
    sigma_rr_c = eta_s * yc[2] / rc
    rho_c_term = m.rho_c * phic if full else 0
    hc = -(sigma_rr_c + rho_c_term) / ((m.rho[0] - m.rho_c) * gc)
    out = {}
    if m.disc.output == "si":
        out["geoid"] = geoid
        out["surface_topography"] = hs
        out["cmb_topography"] = hc
        # u_t(R) sqrt(L): the coefficient of the RMS horizontal speed (eq. 9.4).
        # Zero under a no-slip surface by construction.
        out["surface_velocity"] = yR[1] * mp.sqrt(L) if m.physics.surface == "free_slip" else mp.mpf(0)
    else:
        out["geoid"] = geoid
        # GrvKernel (driver.f:259-261): factor (l - 1), 1e5 for mGal.
        out["gravity"] = mp.mpf(100000) * (m.G * m.emass / R ** 3) * (l - 1) * geoid
        out["surftopo"] = hs
        out["cmbtopo"] = hc
        # VelKernel (driver.f:268-269): u(2) is y2(R) under free slip and y4(R),
        # the shear stress, under no slip (F4).
        u2 = yR[1] if m.physics.surface == "free_slip" else yR[3]
        out["surfvel"] = u2 * mp.sqrt(L)
    if l == 1:
        # Uniform applied field phi_a = A r Y_1 with A = phi_a(R) / R.
        A = eta_s * y5_app / (rho_s * R) / R
        # The field acts on the mantle (through rho0) and on the core through
        # rho_c phi, so the mass it accelerates is the mantle mass plus
        # (4 pi/3) r_c^3 rho_c (review r3, minor 2).
        m_eff = m.mantle_mass + 4 * m.pi / 3 * rc ** 3 * m.rho_c
        if isinstance(source, dict):
            out["degree_one_imbalance"] = None
        else:
            rs_, gs = m.r[source], m.g[source]
            out["degree_one_imbalance"] = m_eff * A / (4 * m.pi / 3 * gs * rs_ ** 2)
        out["applied_y5"] = y5_app
        out["self_y5"] = y5_self
    return out


def _frame_for(table, physics, l):
    if l != 1:
        return "raw"
    frame = degree_one_rule(table, physics)
    if frame == "refused":
        raise ValueError("degree 1 is refused for this configuration: "
                         + frame_options(table, physics)["note"])
    return frame


def solve_state(table: NodeTable, physics: Physics, disc: Discretisation, l: int, source,
                *, nodes=None, frame: str | None = None, rows=None, dps=None) -> dict:
    """Full solution for one source.

    Parameters
    ----------
    source
        Node index of a unit sheet, or a dict ``{index: strength in kg/m^2}``.
    nodes
        Node indices at which to return the state ``y`` (below a sheet at the same
        node, so ``y(r_j-)``). Default: none beyond the surface and CMB values.
    frame
        Override of the degree-1 frame (diagnostics); default from the rule.
    rows
        Override of the three surface rows (diagnostics of G8b, G8f).

    Returns
    -------
    dict
        ``yR`` (state just below the surface), ``yc`` (state just above the CMB,
        below a sheet at the CMB node), ``outputs`` (the kernels of this source),
        ``ratio`` (l = 1), ``y_nodes`` (dict index -> state) and ``dps``.
    """
    m, dps = _model(table, physics, disc, l, dps)
    with mp.workdps(dps):
        fr = frame or _frame_for(table, physics, l)
        if disc.potential_form == "global" and rows is None:
            if l == 1 and fr != "raw":
                raise ValueError("the global potential form exists for the raw frame only")
            sol = _solve_global(m, source)
        else:
            sol = _solve_local(m, source, fr, rows=rows)
        out = _outputs(m, sol, fr, source)
        res = dict(yR=sol["yR"], yc=sol["yc"], outputs=out, ratio=sol.get("ratio"), dps=dps,
                   frame=fr, S=sol.get("S"))
        if nodes is not None:
            res["y_nodes"] = _states_at_nodes(m, sol["yc"], source, nodes)
        return res


def _states_at_nodes(m: _Model, yc, source, nodes):
    """State at every requested node, propagated up from the CMB, with each sheet's
    jump added above its node (the value returned at node j is y(r_j-), below a
    sheet at j)."""
    if isinstance(source, dict):
        sheets = {j: mp.mpf(s) for j, s in source.items()}
    else:
        sheets = {source: mp.mpf(1)}
    want = set(nodes)
    y = yc.copy()
    out = {}
    for i in range(m.n):
        if i > 0:
            y = m.steps[i] * y
        if i in want:
            out[i] = y.copy()
        if i in sheets:
            y = y + m.jump(i, sheets[i])
    return out


def kernels(table: NodeTable, physics: Physics, disc: Discretisation, degrees,
            *, sources=None, dps=None, return_mp: bool = False) -> dict:
    """Kernels at the given degrees and source nodes.

    Returns
    -------
    dict
        One float64 array of shape ``(len(degrees), n_sources)`` per output
        (``SI_OUTPUTS`` or ``FORTRAN_OUTPUTS``), rounded to nearest from the mp
        value; ``dps`` (per degree); and, when 1 is among the degrees, ``degree_one``
        with ``frame``, ``imbalance`` (per source) and ``ratio``. In the CM frame the
        degree-1 geoid is stored as exact zero. With ``return_mp`` the output arrays
        are object arrays of the mp values at the working precision, for the gates
        whose tolerance is far below float64 (1e-25 to 1e-40).
    """
    degrees = tuple(int(l) for l in degrees)
    sources = tuple(range(table.n_nodes)) if sources is None else tuple(sources)
    names = SI_OUTPUTS if disc.output == "si" else FORTRAN_OUTPUTS
    res = {k: np.zeros((len(degrees), len(sources)), dtype=object if return_mp else np.float64)
           for k in names}
    res["dps"] = np.zeros(len(degrees), dtype=int)
    for a_, l in enumerate(degrees):
        m, d = _model(table, physics, disc, l, dps)
        res["dps"][a_] = d
        with mp.workdps(d):
            fr = _frame_for(table, physics, l)
            imb = []
            ratio = None
            for b_, j in enumerate(sources):
                if disc.potential_form == "global":
                    if l == 1 and fr != "raw":
                        raise ValueError("the global potential form exists for the raw frame only")
                    sol = _solve_global(m, j)
                else:
                    sol = _solve_local(m, j, fr)
                    ratio = sol["ratio"]
                out = _outputs(m, sol, fr, j)
                for k in names:
                    res[k][a_, b_] = out[k] if return_mp else float(out[k])
                if l == 1:
                    v = out["degree_one_imbalance"]
                    imb.append(v if return_mp else float(v))
            if l == 1:
                res["degree_one"] = dict(frame=fr, imbalance=np.array(imb, dtype=object if return_mp else np.float64),
                                         ratio=None if ratio is None else (ratio if return_mp else float(ratio)))
    return res


def gravity_at_nodes(table: NodeTable, physics: Physics, disc: Discretisation | None = None,
                     dps: int = 60) -> list:
    """Gravity at the nodes (eq. 6.1), m/s^2, as mp values at ``dps`` digits."""
    disc = disc or Discretisation()
    with mp.workdps(dps):
        m = _Model.__new__(_Model)
        m.table, m.physics, m.disc = table, physics, disc
        c = _constants(physics, disc)
        m.G, m.pi = c.G, c.pi
        m.r = [mp.mpf(s) for s in table.radius]
        m.rho = [mp.mpf(s) for s in table.rho0]
        m.rho_c = mp.mpf(table.core_density)
        m.n = table.n_nodes
        m._gravity()
        return list(m.g)


def interval_steps(table: NodeTable, physics: Physics, disc: Discretisation, l: int, dps=None):
    """The interval steps ``P_i`` (i = 1..n-1, index 0 is None) at the oracle precision."""
    m, _ = _model(table, physics, disc, l, dps)
    return m.steps


# --------------------------------------------------------------------------
# Degree-1 diagnostics (G8a to G8f).
# --------------------------------------------------------------------------


def _with_bc(physics: Physics, surface: str, cmb: str) -> Physics:
    return replace(physics, surface=surface, cmb=cmb)


BOUNDARY_CLASSES = (("free_slip", "free_slip"), ("no_slip", "free_slip"),
                    ("free_slip", "no_slip"), ("no_slip", "no_slip"))


def raw_null_vector(table, physics, disc, dps=None) -> dict:
    """Right and left null vectors of the raw l = 1 system and the image of the
    right one: ``h_s / N_1``, ``h_c / N_1`` and ``|y2(R)| / max_i |y_i(R)|``."""
    m, d = _model(table, physics, disc, 1, dps)
    with mp.workdps(d):
        rows = m.surface_rows("raw")
        basis = m.cmb_basis()
        S, top = m.system(rows, basis)
        U, sv, V = _svd_scaled(S)
        x = V[2, :].T
        yR = mp.zeros(6, 1)
        yc = mp.zeros(6, 1)
        for k in range(3):
            yR += x[k] * top[k]
            yc += x[k] * basis[k]
        sol = dict(yR=yR, yc=yc)
        out = _outputs(m, sol, "raw", 0)
        N = out["geoid"]
        return dict(ratio=sv[2] / sv[0], x=x, yR=yR, yc=yc, outputs=out,
                    hs_over_N=out["surface_topography"] / N, hc_over_N=out["cmb_topography"] / N,
                    velocity_measure=abs(yR[1]) / max(abs(v) for v in yR),
                    left=U[:, 2], dps=d)


def boundary_system(table, physics, disc, l, source, *, potential_row="raw",
                    replace_row=None, dps=None):
    """The 3x3 system of the local form for one source.

    Parameters
    ----------
    potential_row
        ``"raw"`` (exterior ``r^-(l+1)``) or ``"cm"`` (exterior ``r^l``, F7e).
    replace_row
        Replace one row of the raw system by ``y5(R) = 0``: ``"potential"``,
        ``"tangential"`` or ``"radial"`` (diagnostics of G8b).

    Returns
    -------
    (S, rhs, info)
        ``info`` holds the row names, the propagated CMB basis and the particular
        surface state.
    """
    m, d = _model(table, physics, disc, l, dps)
    with mp.workdps(d):
        rows = m.surface_rows(potential_row, replace_row=replace_row)
        basis = m.cmb_basis()
        S, top = m.system(rows, basis)
        yp = m.particular(source)
        rhs = mp.matrix([-m.surface_functional(nm, yp) for nm in rows])
        return S, rhs, dict(rows=rows, basis=basis, top=top, particular=yp, dps=d)


def force_identity(table, physics, disc, sheet_nodes, dps=None) -> list:
    """Singular values of the column-scaled matrix ``[y4(R), p(R), y4(r_c)]`` over the
    four homogeneous CMB states (y2, y3, y4, y5 free) and the particular solutions of
    unit sheets at ``sheet_nodes`` (review r3, 2.2 and R3). ``p = y6 + (l+1) y5 +
    c_s y3`` at the surface. The third singular value is zero exactly when the
    whole-body force balance is an identity of the equations."""
    m, d = _model(table, physics, disc, 1, dps)
    with mp.workdps(d):
        rows = []
        for comp in ("y2", "y3", "y4", "y5"):
            yc = m.cmb_vector(comp)
            yR = m.up[0] * yc
            rows.append([yR[3], m.surface_functional("raw", yR), yc[3]])
        for j in sheet_nodes:
            yR = m.particular(j)
            rows.append([yR[3], m.surface_functional("raw", yR), mp.mpf(0)])
        M = mp.matrix(rows)
        for c in range(3):
            mx = max(abs(M[r_, c]) for r_ in range(M.rows))
            for r_ in range(M.rows):
                M[r_, c] /= mx
        return list(mp.svd_r(M, compute_uv=False))


def degree_one_report(table: NodeTable, physics: Physics, disc: Discretisation, *,
                      sources=None, dps=60) -> dict:
    """Numbers of gates G8a to G8f for one table at ``l = 1`` (all mp values).

    Contents: the raw and CM ratios for the four boundary classes; for a consistent
    free body the raw null vector, its image and the left null vector; per source
    the CM outputs, the outputs with the potential and with the tangential row
    replaced by ``y5(R) = 0``, the radial-row replacement (negative control), the
    residual of the dropped potential row, the imbalance; the raw outputs and the
    shift rule of revision r1 where the raw system is solvable.
    """
    m, d = _model(table, physics, disc, 1, max(dps, required_dps(1)))
    rep = dict(dps=d, ratios={})
    with mp.workdps(d):
        for sc, cc in BOUNDARY_CLASSES:
            ph = _with_bc(physics, sc, cc)
            mm, _ = _model(table, ph, disc, 1, d)
            with mp.workdps(d):
                Sraw, _t = mm.system(mm.surface_rows("raw"), mm.cmb_basis())
                Scm, _t = mm.system(mm.surface_rows("cm"), mm.cmb_basis())
                rep["ratios"][(sc, cc)] = dict(raw=scaled_ratio(Sraw), cm=scaled_ratio(Scm))
        consistent_free = (table.core_mass is None and physics.self_gravity == "full"
                           and physics.background == "hydrostatic"
                           and physics.surface == "free_slip" and physics.cmb == "free_slip")
        if consistent_free:
            rep["null"] = raw_null_vector(table, physics, disc, d)
        sources = range(1, table.n_nodes - 1) if sources is None else sources
        per = {}
        raw_ok = rep["ratios"][(physics.surface, physics.cmb)]["raw"] > mp.mpf(10) ** (-(d // 3))
        for j in sources:
            e = {}
            cm = _solve_local(m, j, "centre_of_mass")
            e["cm"] = _outputs(m, cm, "centre_of_mass", j)
            e["cm"]["yR"] = cm["yR"]
            for rr in ("potential", "tangential", "radial"):
                rows = m.surface_rows("raw", replace_row=rr)
                S_rr, _t = m.system(rows, m.cmb_basis())
                e["ratio_replace_" + rr] = scaled_ratio(S_rr)
                if e["ratio_replace_" + rr] < mp.mpf(10) ** (-(d // 2)):
                    # The replaced row is not part of the dependency, so the
                    # system stays singular (the radial row under TALA, where the
                    # dependency is between the tangential and the potential rows).
                    e["replace_" + rr] = None
                    continue
                s = _solve_local(m, j, "raw", rows=rows, check_ratio=False)
                o = _outputs(m, s, "raw", j)
                o["yR"] = s["yR"]
                e["replace_" + rr] = o
            # Residual of the dropped potential row in the CM solution, relative to
            # the largest of its three terms.
            yR = cm["yR"]
            terms = [yR[5], 2 * yR[4], m.cs * yR[2]]
            e["cm_raw_row_residual"] = abs(sum(terms)) / max(abs(t) for t in terms)
            if raw_ok:
                raw = _solve_local(m, j, "raw")
                ro = _outputs(m, raw, "raw", j)
                e["raw"] = ro
                # Shift rule of revision r1 (negative control of G8c): the raw
                # outputs with h -> h - N_1.
                e["shift"] = dict(surface_topography=ro["surface_topography"] - ro["geoid"],
                                  cmb_topography=ro["cmb_topography"] - ro["geoid"],
                                  surface_velocity=ro["surface_velocity"])
            per[j] = e
        rep["sources"] = per
    return rep


# --------------------------------------------------------------------------
# Momentum integral at l = 1 (G8d), exact per interval.
# --------------------------------------------------------------------------


def momentum_integral(table, physics, disc, source, frame=None, dps=None) -> tuple:
    """``integral rho0_pl (u_r + 2 u_t) r^2 dr`` over the mantle at ``l = 1``, with the
    exact interval solution and the power-law density of the step (F7g, G8d).

    Inside interval i the state is ``y(s) = expm(B s) y_(i-1)``, ``s = ln(r/r_(i-1))``,
    ``B = A0 + kappa K``, and ``rho0_pl = rho0_(i-1) exp(kappa s)``. The integrand in
    ``s`` is ``rho0_(i-1) r_(i-1)^3 exp((kappa + 3) s) (e1 + 2 e2)^T expm(B s) y``, and
    ``integral_0^h expm((B + (kappa+3) I) s) ds`` is the upper-right block of the
    exponential of the augmented matrix ``[[C, I], [0, 0]] h`` (Van Loan 1978). The
    integral is therefore exact to working precision, with no quadrature error.

    Returns ``(integral, scale)``: the scale is a trapezoid estimate of
    ``integral rho0 (|u_r| + 2 |u_t|) r^2 dr`` over the nodes, for normalisation.
    """
    m, d = _model(table, physics, disc, 1, dps)
    with mp.workdps(d):
        st = solve_state(table, physics, disc, 1, source, nodes=range(table.n_nodes),
                         frame=frame, dps=d)
        yn = st["y_nodes"]
        # y_nodes[i] is below a sheet at i; the interval (i-1, i) starts from the
        # state above node i-1, which includes the sheet at i-1.
        total = mp.mpf(0)
        scale = mp.mpf(0)
        sheets = {source: 1} if not isinstance(source, dict) else source
        for i in range(1, m.n):
            y0 = yn[i - 1]
            if (i - 1) in sheets:
                y0 = y0 + m.jump(i - 1, mp.mpf(sheets[i - 1]))
            if m.pair[i]:
                continue
            h = m.h[i]
            kappa = m.kappa_h[i] / h
            e = m.eta_int[i] / m.eta_s
            B = matrix_a_rel(1, e, m.rho_int[i] / m.rho_s, m.full) + matrix_k_rel(e) * kappa
            if physics.compressible:
                # Power-law density of the step: rho0_(i-1) exp(kappa s).
                rho_start, growth = m.rho[i - 1], kappa + 3
            else:
                # Incompressible: the interval density is its mean (diagnostic only).
                rho_start, growth = m.rho_int[i], mp.mpf(3)
            C = B + growth * mp.eye(6)
            aug = mp.zeros(12, 12)
            for a_ in range(6):
                for b_ in range(6):
                    aug[a_, b_] = C[a_, b_] * h
                aug[a_, 6 + a_] = h
            E = mp.expm(aug, method="taylor")
            Mint = E[0:6, 6:12]
            v = Mint * y0
            total += rho_start * m.r[i - 1] ** 3 * (v[0] + 2 * v[1])
            y1 = yn[i]
            f0 = m.rho[i - 1] * m.r[i - 1] ** 2 * (abs(y0[0]) + 2 * abs(y0[1]))
            f1 = m.rho[i] * m.r[i] ** 2 * (abs(y1[0]) + 2 * abs(y1[1]))
            scale += (f0 + f1) / 2 * (m.r[i] - m.r[i - 1])
        return total, scale


# --------------------------------------------------------------------------
# Transcription of the Fortran (second check, G1t and G1b).
# --------------------------------------------------------------------------


def transcription_kernels(table: NodeTable, bdry: int, bdry_bot: int, kappaindex: int,
                          degrees, *, propagator: Literal["expm", "sylvester"] = "expm",
                          dps=None, return_mp: bool = False) -> dict:
    """Line-by-line transcription of ``prop_matrix`` and its subroutines (CODE-A).

    The 8x8 system with the global rows 7 and 8 (``cconstruct``, matrix.f:126-165),
    the right-hand side of ``bconstruct`` (matrix.f:102-123), the matrix of
    ``aconstruct`` (driver.f:351-399) with the one-sided ``kappa`` (driver.f:419-436),
    and the kernel lines of ``prop_matrix`` (driver.f:254-284). Constants as the
    Fortran holds them (F13). ``table.eta`` must hold ``eta_rel * float32(1e19)``.

    ``propagator="expm"`` uses ``mp.expm`` for ``propmat``. ``"sylvester"`` copies
    ``propmat`` (matrix.f:1-54): eigenvalues from LAPACK ``dgeev`` on the float64
    matrix (through ``numpy.linalg.eigvals``), their real parts, the merge of
    eigenvalues within ``10E-4`` (a float32 literal, 1e-3) in the same order, and the
    Lagrange-Sylvester sum in mp.

    Returns the five Fortran kernels, shape ``(len(degrees), n_nodes)``; mp values
    in object arrays with ``return_mp``.
    """
    degrees = tuple(degrees)
    n = table.n_nodes
    nr = n - 1
    names = FORTRAN_OUTPUTS
    res = {k: np.zeros((len(degrees), n), dtype=object if return_mp else np.float64) for k in names}
    for a_, l in enumerate(degrees):
        d = dps or required_dps(l, table.radius[0], table.radius[-1])
        with mp.workdps(d):
            G, pi, visc0, emass = LEGACY_G, LEGACY_PI, LEGACY_VISC0, LEGACY_EMASS
            bgrho, rho_c, rho_w = mp.mpf(4500), mp.mpf(9900), mp.mpf(0)
            # Fortran arrays are 1-based: r(1) is the CMB, r(nr+1) the surface.
            r = [None] + [mp.mpf(s) for s in table.radius]
            rho0 = [None] + [mp.mpf(s) for s in table.rho0]
            rvsc = [None] + [mp.mpf(s) / visc0 for s in table.eta]
            grav = mp.mpf(10)                                        # matrix.f:98
            lq = mp.mpf(l)

            def kappa(ind):                                          # driver.f:419-436
                if kappaindex == 1:
                    if ind == 1:
                        return mp.mpf(0)
                    return (r[ind] / rho0[ind]) * ((rho0[ind] - rho0[ind - 1]) / (r[ind] - r[ind - 1]))
                return mp.mpf(0)

            def aconstruct(ind):                                     # driver.f:351-399
                a = mp.zeros(6, 6)
                fv, fr = rvsc[ind], rho0[ind]
                kp = kappa(ind)
                a[0, 0] = -2 - kp
                a[0, 1] = lq * (lq + 1)
                a[1, 0] = -1
                a[1, 1] = 1
                a[1, 3] = 1 / fv
                a[2, 0] = 4 * (3 + kp) * fv
                a[2, 1] = -6 * lq * (lq + 1) * fv
                a[2, 2] = 1
                a[2, 3] = lq * (lq + 1)
                a[2, 5] = -fr / bgrho
                a[3, 0] = -2 * (3 + kp) * fv
                a[3, 1] = 2 * (2 * lq * (lq + 1) - 1) * fv
                a[3, 2] = -1
                a[3, 3] = -2
                a[3, 4] = -fr / bgrho
                a[4, 4] = 1
                a[4, 5] = 1
                a[5, 4] = lq * (lq + 1)
                return a

            def propmat(a, rad):                                     # matrix.f:2-54
                if propagator == "expm":
                    return mp.expm(a * mp.log(rad), method="taylor")
                af = np.array([[float(a[i, j]) for j in range(6)] for i in range(6)])
                ev = np.linalg.eigvals(af)                           # dgeev, driver.f:333
                u = [mp.mpf(float(np.real(x))) for x in ev]
                eig = [u[0]]
                tol = _f32(10e-4)
                for i in range(6):
                    if all(abs(u[i] - e_) >= tol for e_ in eig):
                        eig.append(u[i])
                b = mp.zeros(6, 6)
                for k, ek in enumerate(eig):
                    t = mp.eye(6)
                    for i, ei in enumerate(eig):
                        if i != k:
                            t = t * (a - ei * mp.eye(6)) / (ek - ei)
                    b += t * rad ** ek
                return b

            prop = [None] * (nr + 3)
            prop[nr + 2] = mp.eye(6)                                 # driver.f:234
            for i in range(nr + 1, 1, -1):                           # driver.f:235-240
                p = propmat(aconstruct(i), r[i] / r[i - 1])
                prop[i] = prop[i + 1] * p
            # cconstruct (matrix.f:126-165), with p = prop(:,:,2)
            P = prop[2]
            C = mp.zeros(8, 8)
            fac = 4 * pi * G / (2 * lq + 1)
            C[1, 1] = bdry
            C[2, 0] = 1
            C[3, 1] = 1 - bdry
            C[4, 2] = 1
            C[5, 3] = 1
            C[6, 0] = fac * r[nr + 1] / grav * bgrho
            C[6, 2] = 1
            C[6, 4] = -fac * r[1] / grav * bgrho * ((r[1] / r[nr + 1]) ** l)
            C[6, 6] = C[6, 4] * rho_c / bgrho
            C[7, 0] = C[6, 0] * ((r[1] / r[nr + 1]) ** (l + 1))
            C[7, 4] = -fac * r[1] / grav * bgrho
            C[7, 6] = 1 - fac * r[1] / grav * rho_c
            for k in range(6):
                for mm in range(4, 8):
                    C[k, mm] = -P[k, mm - 2]
            if bdry_bot == 1:
                for k in range(6):
                    C[k, 5] = -P[k, 1]
            out = {k: [] for k in names}
            for i in range(1, nr + 2):                               # driver.f:248
                p = prop[i + 1]
                const = 4 * pi * G / (2 * l + 1)                     # bconstruct
                ave_rad, ave_grav = r[i], grav
                bvec = mp.zeros(8, 1)
                for k in range(6):
                    bvec[k] = (p[k, 2] * ave_rad * ave_grav / visc0
                               - p[k, 5] * 4 * pi * ave_rad * ave_rad * G * bgrho / visc0)
                bvec[6] = bgrho / visc0 * const * ((ave_rad / r[nr + 1]) ** l) * (ave_rad ** 2)
                bvec[7] = bgrho / visc0 * const * ((r[1] / ave_rad) ** (l - 1)) * (r[1] ** 2)
                u = mp.lu_solve(C, bvec)
                geo = (visc0 / (bgrho * r[nr + 1] * grav)) * u[2]
                grv = mp.mpf(100000) * ((G * emass) / r[nr + 1] ** 3) * (l - 1) * geo
                sur = -(visc0 / ((bgrho - rho_w) * r[nr + 1] * grav)) * (u[0] + (rho_w / bgrho) * u[2])
                vel = u[1] * mp.sqrt(lq * (lq + 1))
                cmb = -(visc0 / ((rho0[1] - rho_c) * r[1] * grav)) * (u[4] + (rho_c / bgrho) * u[6])
                if l == 1:
                    geo = mp.mpf(0)                                    # driver.f:282-284
                out["geoid"].append(geo)
                out["gravity"].append(grv)
                out["surftopo"].append(sur)
                out["surfvel"].append(vel)
                out["cmbtopo"].append(cmb)
            for k in names:
                res[k][a_, :] = out[k] if return_mp else [float(v) for v in out[k]]
    return res


# --------------------------------------------------------------------------
# Writing the stored oracle values (tests/golden/oracle).
# --------------------------------------------------------------------------


def _unit_kernels(args):
    """Worker: kernels of one (variant, degree, frame) unit. Returns a flat dict.

    The last element of the unit is the working precision of the variant
    (``OracleVariant.dps``; None: ``required_dps``)."""
    key, table, physics, disc, l, frame, sources, dps = args
    t0 = time.time()
    ph = physics if frame is None else replace(physics, degree_one=frame)
    k = kernels(table, ph, disc, (l,), sources=sources, dps=dps)
    flat = {}
    for name, arr in k.items():
        if name == "degree_one":
            flat["degree_one_imbalance"] = arr["imbalance"]
            flat["degree_one_ratio"] = np.array(np.nan if arr["ratio"] is None else arr["ratio"])
            flat["degree_one_frame"] = np.array(arr["frame"])
        elif name == "dps":
            flat["dps"] = arr[0:1]
        else:
            flat[name] = arr[0]
    if sources is not None:
        flat["sources"] = np.array(sources)
    return key, flat, time.time() - t0


def _unit_null(args):
    key, table, physics, disc = args
    nv = raw_null_vector(table, physics, disc)
    with mp.workdps(nv["dps"]):
        flat = dict(raw_ratio=np.array(float(nv["ratio"])),
                    null_hs_over_N=np.array(float(nv["hs_over_N"])),
                    null_hc_over_N=np.array(float(nv["hc_over_N"])),
                    null_velocity_measure=np.array(float(nv["velocity_measure"])),
                    null_left=np.array([float(v) for v in nv["left"]]),
                    null_yR=np.array([float(v) for v in nv["yR"]]))
        rep_ratios = {}
        for sc, cc in BOUNDARY_CLASSES:
            mm, d = _model(table, _with_bc(physics, sc, cc), disc, 1, nv["dps"])
            with mp.workdps(d):
                Sraw, _ = mm.system(mm.surface_rows("raw"), mm.cmb_basis())
                Scm, _ = mm.system(mm.surface_rows("cm"), mm.cmb_basis())
                rep_ratios[f"ratio_raw_{sc}_{cc}"] = np.array(float(scaled_ratio(Sraw)))
                rep_ratios[f"ratio_cm_{sc}_{cc}"] = np.array(float(scaled_ratio(Scm)))
        flat.update(rep_ratios)
    return key, flat, 0.0


def _write_npz(path: Path, arrays: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez_compressed(f, **arrays)


def sha256(path: Path) -> str:
    """sha256 hex digest of a file."""
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def main(argv=None) -> None:
    """Write the stored oracle values.

    ``python tests/reference_mpmath.py --write tests/golden/oracle [--cases a,b]
    [--workers N] [--legacy] [--ishan PATH]``. Each oracle case of
    ``cases.ORACLE_CASES`` becomes ``<name>.npz`` with keys
    ``<variant>/l<l>/<frame>/<output>``; ``--legacy`` writes ``legacy_<run>.npz`` for
    every Fortran run of ``cases.GOLDEN_RUNS`` at ``cases.LEGACY_DEGREES``.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cases  # noqa: E402

    p = argparse.ArgumentParser()
    p.add_argument("--write", required=True)
    p.add_argument("--cases", default="")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--legacy", action="store_true")
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--ishan", default="")
    a = p.parse_args(argv)
    out = Path(a.write)
    units = []
    nulls = []
    owners = {}
    if a.oracle:
        selected = [c for c in cases.ORACLE_CASES if not a.cases or c.name in a.cases.split(",")]
        for case in selected:
            for v in case.variants:
                for l in case.degrees:
                    frames = [None]
                    if l == 1:
                        frames = list(v.l1_frames)
                    for fr in frames:
                        key = (case.name, f"{v.label}/l{l}/{fr or 'raw'}")
                        units.append((key, v.table, v.physics, v.disc, l, fr, v.sources, v.dps))
                if 1 in case.degrees and v.store_null:
                    nulls.append(((case.name, f"{v.label}/l1/null"), v.table, v.physics, v.disc))
            owners[case.name] = case
    if a.legacy:
        for run, spec in cases.GOLDEN_RUNS.items():
            if spec.get("fields_only"):
                continue
            table = cases.golden_table(run)
            physics, disc = legacy_setup(spec["bdry"], spec["bdry_bot"], spec["kappaindex"])
            for l in cases.LEGACY_DEGREES:
                units.append(((f"legacy_{run}", f"l{l}"), table, physics, disc, l, None, None, None))
    if a.ishan:
        table = cases.ishan_table(Path(a.ishan))
        physics, disc = legacy_setup(1, 1, 0)
        for l in cases.ISHAN_DEGREES:
            units.append((("legacy_ishan257", f"l{l}"), table, physics, disc, l, None, None, None))
    # Longest units first, so that the pool is not left waiting on one long unit.
    units.sort(key=lambda u: -(u[4] + 0.02 * u[1].n_nodes * (1 + u[4] / 30)))
    results = {}
    seconds = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_unit_kernels, u) for u in units] + [ex.submit(_unit_null, u) for u in nulls]
        for f in futs:
            (fname, key), flat, sec = f.result()
            seconds[fname] = seconds.get(fname, 0.0) + sec
            results.setdefault(fname, {})
            for k, v in flat.items():
                results[fname][f"{key}/{k}"] = v
    print(f"computed {len(units)} units and {len(nulls)} null vectors in {time.time() - t0:.1f} s")
    # The manifest section of the oracle files: sha256, CPU seconds, dps range.
    manifest_path = out.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    section = manifest.setdefault("oracle", {})
    for fname, arrays in results.items():
        _write_npz(out / f"{fname}.npz", dict(sorted(arrays.items())))
        dps = [int(v[0]) for k, v in arrays.items() if k.endswith("/dps")]
        section[fname] = dict(npz=f"oracle/{fname}.npz", npz_sha256=sha256(out / f"{fname}.npz"),
                              cpu_seconds=round(seconds[fname], 2), n_arrays=len(arrays),
                              dps_min=min(dps) if dps else None, dps_max=max(dps) if dps else None)
        print(fname, section[fname]["npz_sha256"][:12])
    manifest["cases_py_sha256_oracle"] = sha256(Path(__file__).resolve().parent / "cases.py")
    manifest["reference_mpmath_py_sha256"] = sha256(Path(__file__).resolve())
    manifest["mpmath_version"] = mp.__version__
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
