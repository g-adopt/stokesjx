"""Gate G9: the oracle against the analytical delta-forcing solutions of Kramer,
Davies and Wilson (2021, GMD 14, 1899-1919), package ``assess``.

The ``assess`` problem is an isoviscous, incompressible shell with uniform gravity
and no self-gravitation: ``-div(tau) + grad(p) = -g rho' r_hat`` with
``rho' = delta(r - r') Y_lm``. That is the oracle's NO_SELF_GRAVITY, uniform-g
problem with a unit sheet (sigma = +1 kg/m^2 per unit harmonic coefficient) at
``r'``, the viscosity ``nu`` and the gravity ``g``. One global convention, no fitted
factor (NOTES/docs/equations.md section 7): the assess forcing is a positive sheet,
and both codes write the velocity as ``u_r Y r_hat + u_t dY/dtheta theta_hat``.
Non-dimensional values of the G-ADOPT benchmark: Rp = 2.22, Rm = 1.22, r' = 1.72,
nu = 1, g = 1.
"""

import mpmath as mp
import pytest

assess = pytest.importorskip("assess")

import reference_mpmath as ro  # noqa: E402

# assess evaluates closed-form solutions whose coefficients come from a 4x4 float64
# solve; the oracle is exact to 30 digits. Measured agreement: 1.2e-13 at worst
# (cause: the float64 coefficients of assess).
ASSESS_RTOL = 1e-10
RP, RM, RPRIME = "2.22", "1.22", "1.72"
THETA = 0.6          # colatitude at which assess is evaluated; any value with Y != 0
N_NODES = 51         # spacing 0.02: r' = 1.72 is node 25 and r = 2.0 is node 39


def _table():
    rad = tuple(str(round(1.22 + 0.02 * k, 2)) for k in range(N_NODES))
    assert rad[25] == RPRIME and rad[39] == "2.0" and rad[-1] == RP
    return ro.NodeTable(radius=rad, rho0=("1",) * N_NODES, eta=("1",) * N_NODES, core_density="0")


@pytest.mark.parametrize("bc", ["free_slip", "no_slip"])
@pytest.mark.parametrize("l", [2, 4, 8])
def test_delta_solution(bc, l):
    from assess import (SphericalStokesSolutionDeltaFreeSlip, SphericalStokesSolutionDeltaZeroSlip,
                        Y, dYdtheta)
    cls = SphericalStokesSolutionDeltaFreeSlip if bc == "free_slip" else SphericalStokesSolutionDeltaZeroSlip
    ph = ro.Physics(self_gravity="none", background="uniform", uniform_g="1", surface=bc, cmb=bc,
                    density_scale="1", viscosity_scale="1")
    st = ro.solve_state(_table(), ph, ro.Discretisation(), l, 25, nodes=[39])
    upper = cls(l, 0, +1, Rp=2.22, Rm=1.22, rp=1.72, nu=1.0, g=1.0)
    lower = cls(l, 0, -1, Rp=2.22, Rm=1.22, rp=1.72, nu=1.0, g=1.0)
    y, dy = Y(l, 0, THETA, 0), dYdtheta(l, 0, THETA, 0)
    with mp.workdps(40):
        yR, yc, y2 = st["yR"], st["yc"], st["y_nodes"][39]
        R, rc = mp.mpf(RP), mp.mpf(RM)
        ours = {
            "sigma_rr(R)": float(yR[2] / R),          # y3 = r sigma_rr / eta_s, eta_s = 1
            "sigma_rr(r_c)": float(yc[2] / rc),
            "u_r(2.0)": float(y2[0]),
            "u_t(2.0)": float(y2[1]),
        }
        if bc == "free_slip":
            ours["u_t(R)"] = float(yR[1])
        else:
            ours["sigma_rt(R)"] = float(yR[3] / R)
    ref = {
        "sigma_rr(R)": upper.radial_stress(2.22, THETA, 0) / y,
        "sigma_rr(r_c)": lower.radial_stress(1.22, THETA, 0) / y,
        "u_r(2.0)": upper.u_r(2.0, THETA, 0) / y,
        "u_t(2.0)": upper.u_theta(2.0, THETA, 0) / dy,
    }
    if bc == "free_slip":
        ref["u_t(R)"] = upper.u_theta(2.22, THETA, 0) / dy
    else:
        ref["sigma_rt(R)"] = upper.tau_rtheta(2.22, THETA, 0) / dy
    for k in ref:
        assert abs(ours[k] - ref[k]) <= ASSESS_RTOL * abs(ref[k]), (k, ours[k], ref[k])


# ---------------------------------------------------------------------------
# The package against assess (gate G6 of NOTES/chunks/c3-flags-fields/01-plan.md,
# section 5.4): NO_SELF_GRAVITY, Uniform(1.0), incompressible, isoviscous shell of the
# G-ADOPT benchmark on 51 nodes of spacing 0.02.
# ---------------------------------------------------------------------------

import functools  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

jax = pytest.importorskip("jax")

import stokesjx as sj  # noqa: E402

# The package kernels against the assess delta-forcing solutions, e_l over the five
# sources. assess solves a 4x4 float64 system per solution (1.2e-13 against the oracle,
# chunk 1); the plan prototype measured 1.3e-14 at most
# (NOTES/chunks/c3-flags-fields/plan-evidence/proto_assess.out). Cause: the float64
# coefficients of assess and the rounding of the package sweeps.
ASSESS_TOL = 1e-12
# Two package solves that differ only in rho_c and rho_w under NO_SELF_GRAVITY: the
# mechanical rows do not read them, so the outputs scale exactly by the density
# contrasts up to the rounding of one division (the c2 constant SCALE_TOL).
SCALE_TOL = 1e-12
PKG_SOURCES = (1, 10, 25, 40, 49)
PKG_DEGREES = (2, 4, 8, 16)


def _shell(n_intervals=50, core_density=0.0, outer_density=0.0):
    """The assess shell: ``R = 2.22``, ``r_c = 1.22``, ``rho0 = 1``, ``eta = 1``."""
    r = np.linspace(1.22, 2.22, n_intervals + 1)
    return sj.RadialProfile(r, np.ones_like(r), np.ones_like(r), core_density=core_density,
                            outer_density=outer_density), r


def _form(bc):
    cls = sj.FreeSlip() if bc == "free_slip" else sj.NoSlip()
    return sj.Formulation(gravity=sj.GravityModel.NO_SELF_GRAVITY, background=sj.Uniform(1.0), surface=cls, cmb=cls)


@functools.lru_cache(maxsize=None)
def _package_delta(bc, core_density=0.0, outer_density=0.0):
    p, r = _shell(core_density=core_density, outer_density=outer_density)
    K = sj.stokes_kernels(p, sj.StokesConfig(lmax=max(PKG_DEGREES), formulation=_form(bc)))
    return {o: np.asarray(getattr(K, o)) for o in ("geoid", "surface_topography", "cmb_topography",
                                                    "surface_velocity")}, K.degrees, r


@pytest.mark.parametrize("bc", ["free_slip", "no_slip"])
def test_package_delta(bc):
    """The package against the assess delta-forcing solutions at the sources
    {1, 10, 25, 40, 49} and ``l`` in {2, 4, 8, 16}: the surface topography
    ``-sigma_rr(R) / (rho_top g)``, the CMB topography ``-sigma_rr(r_c) / ((rho0(r_c) - rho_c) g)``,
    the free-slip velocity ``u_theta(R) / dY/dtheta sqrt(L)``, and the geoid against the
    potential of the three sheets built from the assess stresses,
    ``g N = 4 pi G / (2l+1) [R (-sigma_rr(R) / g) + r' (r'/R)^(l+1) + r_c (r_c/R)^(l+1) sigma_rr(r_c) / g]``
    (eq. 5.2: the two boundary sheets have the masses ``-sigma_rr(R) / g`` and
    ``sigma_rr(r_c) / g``). Under no slip the velocity is an exact zero."""
    from assess import SphericalStokesSolutionDeltaFreeSlip, SphericalStokesSolutionDeltaZeroSlip, Y, dYdtheta
    cls = SphericalStokesSolutionDeltaFreeSlip if bc == "free_slip" else SphericalStokesSolutionDeltaZeroSlip
    K, degrees, r = _package_delta(bc)
    G = sj.Units().G
    worst = 0.0
    for l in PKG_DEGREES:
        y, dy = Y(l, 0, THETA, 0), dYdtheta(l, 0, THETA, 0)
        ref = {k: [] for k in ("surface_topography", "cmb_topography", "surface_velocity", "geoid")}
        for j in PKG_SOURCES:
            up = cls(l, 0, +1, Rp=2.22, Rm=1.22, rp=r[j], nu=1.0, g=1.0)
            lo = cls(l, 0, -1, Rp=2.22, Rm=1.22, rp=r[j], nu=1.0, g=1.0)
            srr_R = up.radial_stress(2.22, THETA, 0) / y
            srr_c = lo.radial_stress(1.22, THETA, 0) / y
            ref["surface_topography"].append(-srr_R)
            ref["cmb_topography"].append(-srr_c)
            ref["surface_velocity"].append(up.u_theta(2.22, THETA, 0) / dy * math.sqrt(l * (l + 1)))
            ref["geoid"].append(4 * math.pi * G / (2 * l + 1) * (2.22 * (-srr_R) + r[j] * (r[j] / 2.22) ** (l + 1)
                                                                  + 1.22 * (1.22 / 2.22) ** (l + 1) * srr_c))
        k = degrees.index(l)
        for o, vals in ref.items():
            got = K[o][k, list(PKG_SOURCES)]
            if o == "surface_velocity" and bc == "no_slip":
                assert np.all(K[o] == 0)
                continue
            e = np.max(np.abs(got - np.array(vals))) / np.max(np.abs(vals))
            worst = max(worst, e)
            assert e <= ASSESS_TOL, (bc, l, o, e)
    print(f"assess delta {bc}: worst e_l {worst:.2e}")


@pytest.mark.parametrize("bc", ["free_slip", "no_slip"])
def test_package_delta_dense_core_water(bc):
    """The assess shell has ``rho_c = rho_w = 0``, so a ``rho_c`` or ``rho_w`` term that
    stayed under NO_SELF_GRAVITY would pass ``test_package_delta``. With ``rho_c = 2.5``
    and ``rho_w = 0.3`` on ``rho0 = 1``: the geoid and the velocity equal the run with
    both zero, the surface topography scales by ``rho_top / (rho_top - rho_w)`` and the
    CMB topography by ``rho0(r_c) / (rho0(r_c) - rho_c)`` (eq. 5.2)."""
    a, _, _ = _package_delta(bc)
    b, _, _ = _package_delta(bc, core_density=2.5, outer_density=0.3)
    for o in ("geoid", "surface_velocity"):
        scale = np.max(np.abs(a[o]))
        if scale == 0:
            assert np.all(b[o] == 0)
        else:
            assert np.max(np.abs(b[o] - a[o])) <= SCALE_TOL * scale, o
    hs = a["surface_topography"] * 1.0 / (1.0 - 0.3)
    hc = a["cmb_topography"] * 1.0 / (1.0 - 2.5)
    assert np.max(np.abs(b["surface_topography"] - hs)) <= SCALE_TOL * np.max(np.abs(hs))
    assert np.max(np.abs(b["cmb_topography"] - hc)) <= SCALE_TOL * np.max(np.abs(hc))


# Convergence of the trapezoid fields against the assess smooth-forcing solutions: the
# ratio of successive errors for a doubling of the node count (second order: 4), and
# the Richardson extrapolation (4 F_256 - F_128) / 3 against assess, relative. Measured
# on the plan prototype: ratios 3.999 to 4.002, Richardson within 1.1e-9
# (NOTES/chunks/c3-flags-fields/plan-evidence/proto_assess.out). Cause of the residue:
# the fourth-order term of the trapezoid error.
QUAD_RATIO_BAND = (3.9, 4.1)
RICHARDSON_TOL = 1e-8
SMOOTH_CASES = ((2, 3), (8, 3))


@functools.lru_cache(maxsize=None)
def _smooth_fields(bc, n_intervals):
    """Surface and CMB topography of the densities ``(r / R)^k Y_l0`` for the cases of
    ``SMOOTH_CASES``, from ``fields.apply`` on a uniform table of ``n_intervals``."""
    p, r = _shell(n_intervals)
    lmax = max(l for l, _ in SMOOTH_CASES)
    K = sj.stokes_kernels(p, sj.StokesConfig(lmax=lmax, formulation=_form(bc)))
    out = {}
    for l, k in SMOOTH_CASES:
        d = np.zeros((len(r), 2, lmax + 1, lmax + 1))
        d[:, 0, l, 0] = (r / 2.22) ** k
        F = sj.apply(K, d, p)
        out[(l, k)] = np.array([float(F.surface_topography[0, l, 0]), float(F.cmb_topography[0, l, 0])])
    return out


@pytest.mark.parametrize("bc", ["free_slip", "no_slip"])
def test_package_smooth_quadrature(bc):
    """The fields of ``fields.apply`` (trapezoid weights) for the assess smooth forcing
    ``(r / R)^k Y`` with ``(l, k)`` in {(2, 3), (8, 3)} on uniform tables of 64, 128 and
    256 intervals: second-order convergence of the surface and CMB topography
    (``h = -sigma_rr / (rho g)`` with ``rho = g = 1``), and the Richardson extrapolation
    of 128 and 256 within ``RICHARDSON_TOL`` of assess."""
    from assess import SphericalStokesSolutionSmoothFreeSlip, SphericalStokesSolutionSmoothZeroSlip, Y
    cls = SphericalStokesSolutionSmoothFreeSlip if bc == "free_slip" else SphericalStokesSolutionSmoothZeroSlip
    for l, k in SMOOTH_CASES:
        sol = cls(l, 0, k, Rp=2.22, Rm=1.22, nu=1.0, g=1.0)
        y = Y(l, 0, THETA, 0)
        ref = np.array([-sol.radial_stress(2.22, THETA, 0) / y, -sol.radial_stress(1.22, THETA, 0) / y])
        F = {n: _smooth_fields(bc, n)[(l, k)] for n in (64, 128, 256)}
        e = {n: np.abs(F[n] - ref) / np.abs(ref) for n in F}
        r1, r2 = e[64] / e[128], e[128] / e[256]
        assert np.all((QUAD_RATIO_BAND[0] < r1) & (r1 < QUAD_RATIO_BAND[1])), (l, k, r1)
        assert np.all((QUAD_RATIO_BAND[0] < r2) & (r2 < QUAD_RATIO_BAND[1])), (l, k, r2)
        rich = np.abs((4 * F[256] - F[128]) / 3 - ref) / np.abs(ref)
        assert np.all(rich <= RICHARDSON_TOL), (l, k, rich)
        print(f"smooth {bc} l={l} k={k}: ratios {r1} {r2}; Richardson {rich}")
