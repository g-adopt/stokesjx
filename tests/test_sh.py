"""Spherical-harmonic conventions, conversions and transforms (``stokesjx.sh``; gate G8
of NOTES/chunks/c3-flags-fields/01-plan.md, section 5.6).

References: the lovejx module that ``sh`` ports (``lovejx.sealevel.sht``), three
independent evaluations of the conventions (scipy ``lpmv`` with the Condon-Shortley
phase removed, lovejx ``evaluate`` of the converted complex coefficients, CODE-A's
orthonormal formula transcribed from calc.f:94-115), and the Fortran ``GRD_*`` grids of
a golden run.

The tests that compare with lovejx take the fixture ``sht`` and skip when lovejx is not
installed (lovejx is not on PyPI, and CI does not install it):
``test_legendre_table_equals_lovejx``, ``test_gauss_legendre_equals_lovejx``,
``test_convention_definitions_lovejx``, ``test_synthesis_grid_against_evaluate`` and
``test_gl_port_equals_lovejx``. Every other test of this file runs without lovejx;
``test_synthesis_grid_against_fortran_grd`` needs ``legacy/`` and skips without it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from scipy.special import lpmv  # noqa: E402

import stokesjx as sj  # noqa: E402
from stokesjx import sh  # noqa: E402



@pytest.fixture(scope="module")
def sht():
    """``lovejx.sealevel.sht``, the module that ``stokesjx.sh`` ports. The tests that
    compare with it skip when lovejx is not installed (lovejx is not on PyPI and CI
    does not install it); the other tests of this file run without it."""
    return pytest.importorskip("lovejx.sealevel.sht")


# Round trips of the conversions, relative to the largest coefficient. Cause: one
# multiplication and one division by sqrt(2) or sqrt(4 pi) per entry.
SH_ROUNDTRIP_TOL = 1e-15
# Three independent evaluations of a convention at random points, relative to the
# largest value: measured 1.1e-15 at L = 12 and 4.8e-15 at L = 50
# (NOTES/chunks/c3-flags-fields/plan-evidence/sh_conventions.out, sh_conventions_L50.out).
# Cause: the different summation orders and the factorials of the CODE-A formula.
SH_EVAL_TOL = 1e-13
# analysis(synthesis(U)) on a Gauss-Legendre grid, relative to the largest
# coefficient: measured 1.4e-14 (L = 50) and 6.4e-14 (L = 100) with lovejx
# (plan-evidence/sh_cost.out). Cause: the quadrature rounding of the transform.
GL_ROUNDTRIP_TOL = 1e-12
# The ported transform against lovejx, relative to the largest value: the same
# operations in the same order.
GL_PORT_TOL = 1e-15
# The orthonormal synthesis of the golden SPHCO coefficients on CODE-A's 1-degree grid
# against its GRD_* files, relative to the largest value: measured 1.1e-6
# (plan-evidence/grd_check.out). Cause: CODE-A's single-precision sqrt(real(i)) in the
# Legendre normalisation (calc.f:182-190, audit row 26).
GRD_TOL = 1e-5
# The velocity grid holds seven decimals (f15.7) on values of about 6e-6 m/s, absolute
# in m/s: measured 5.0e-8.
GRD_VEL_ATOL = 1e-7
# jax.grad of a quadratic loss of the grid synthesis against a central difference at
# step 1e-3 (no truncation error for a quadratic), relative. Cause: the rounding of the
# two loss values, about 1e-16 / 1e-3 relative.
GRAD_FD_TOL = 1e-8
VMAP_TOL = 1e-15


def _random_real(L, batch=(), seed=0):
    """Random real coefficients ``(*batch, 2, L+1, L+1)`` with zeros for ``m > l`` and
    ``S_l0 = 0``."""
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(batch + (2, L + 1, L + 1))
    tri = np.tril(np.ones((L + 1, L + 1)))
    c = c * tri
    c[..., 1, :, 0] = 0.0
    return c


@pytest.mark.parametrize("L", [50, 256])
def test_legendre_table_equals_lovejx(L, sht):
    x = np.linspace(-0.999, 0.999, 37)
    assert np.array_equal(sh.legendre_table(L, x), sht.legendre_table(L, x))


@pytest.mark.parametrize("L", [50, 256])
def test_gauss_legendre_equals_lovejx(L, sht):
    a, b = sh.gauss_legendre(L + 1), sht.gauss_legendre(L + 1)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_convert_round_trip():
    """Every pair of conventions and back, with batch axes ``(N, 3, ...)``, ``L = 50``;
    the identity conversion; a complex array with a real convention raises."""
    L = 50
    real = _random_real(L, batch=(4, 3), seed=1)
    start = {"real_4pi": real, "real_orthonormal": real * math.sqrt(4 * math.pi),
             "complex_4pi": np.asarray(sh.convert(real, "real_4pi", "complex_4pi"))}
    assert start["complex_4pi"].shape == (4, 3, L + 1, L + 1)
    for a in sh.CONVENTIONS:
        for b in sh.CONVENTIONS:
            x = start[a]
            y = np.asarray(sh.convert(sh.convert(x, a, b), b, a))
            assert y.shape == x.shape
            assert np.max(np.abs(y - x)) <= SH_ROUNDTRIP_TOL * np.max(np.abs(x)), (a, b)
            assert np.max(np.abs(np.asarray(sh.convert(x, a, b)) - start[b])) <= \
                SH_ROUNDTRIP_TOL * np.max(np.abs(start[b])), (a, b)
    with pytest.raises(ValueError):
        sh.convert(start["complex_4pi"], "real_4pi", "complex_4pi")
    with pytest.raises(ValueError):
        sh.convert(real[..., 0, :, :], "real_4pi", "complex_4pi")      # no C/S axis
    # The legacy constants give CODE-A's sqt = sqrt(4 float32(pi)).
    on = np.asarray(sh.convert(real, "real_4pi", "real_orthonormal", units=sj.Units(legacy_constants=True)))
    sqt = math.sqrt(4.0 * float(np.float32(math.pi)))
    assert np.array_equal(on, real * sqt)


def _eval_real_4pi(C, S, x, lam):
    """scipy ``lpmv`` (the Condon-Shortley phase removed) with the geodesy normalisation."""
    L = C.shape[0] - 1
    f = np.zeros_like(x)
    for l in range(L + 1):
        for m in range(l + 1):
            P = lpmv(m, l, x) * (-1) ** m
            N = math.sqrt((2 - (m == 0)) * (2 * l + 1) * math.exp(math.lgamma(l - m + 1) - math.lgamma(l + m + 1)))
            f += N * P * (C[l, m] * np.cos(m * lam) + S[l, m] * np.sin(m * lam))
    return f


def _eval_codea_orthonormal(c_on, s_on, x, lam):
    """CODE-A's orthonormal synthesis (calc.f:94-115) with exact factorials, from
    ``l = 0`` (calc.f starts at ``l_min = 1``; the ``l = 0`` term is added here)."""
    L = c_on.shape[0] - 1
    sqpi = 1 / math.sqrt(math.pi)
    facto = lambda n: math.exp(0.5 * math.lgamma(n + 1))
    f = np.zeros_like(x)
    for l in range(L + 1):
        for m in range(-l, l + 1):
            leg = lpmv(abs(m), l, x) * (-1) ** abs(m)
            if m == 0:
                f += c_on[l, 0] * math.sqrt(2 * l + 1) * 0.5 * sqpi * leg
            elif m < 0:
                Y = math.sqrt(2 * l + 1) * sqpi / math.sqrt(2) * facto(l + m) / facto(l - m) * leg * np.cos(m * lam)
                f += c_on[l, -m] * Y
            else:
                Y = math.sqrt(2 * l + 1) * sqpi / math.sqrt(2) * facto(l - m) / facto(l + m) * leg * np.sin(m * lam)
                f += s_on[l, m] * Y
    return f


def _convention_case(L):
    """Random ``real_4pi`` coefficients of degree ``L`` (seed 2), 40 random points
    (seed 3), and the reference values of scipy ``lpmv`` at the points. Returns
    ``(c, colat, lon, x, lam, ref, scale)``."""
    c = _random_real(L, seed=2)
    rng = np.random.default_rng(3)
    colat = rng.uniform(1, 179, 40)
    lon = rng.uniform(-180, 180, 40)
    x, lam = np.cos(np.radians(colat)), np.radians(lon)
    ref = _eval_real_4pi(c[0], c[1], x, lam)
    return c, colat, lon, x, lam, ref, np.max(np.abs(ref))


@pytest.mark.parametrize("L", [12, 50])
def test_convention_definitions(L):
    """The three conventions of D9 at 40 random points: ``synthesis_grid`` of each
    convention (evaluated point by point on a one-point grid per latitude) against
    scipy ``lpmv``, and CODE-A's orthonormal formula against scipy ``lpmv``. The
    comparison with lovejx ``evaluate`` is ``test_convention_definitions_lovejx``."""
    c, colat, lon, x, lam, ref, scale = _convention_case(L)
    U = np.asarray(sh.convert(c, "real_4pi", "complex_4pi"))
    on = c * math.sqrt(4 * math.pi)
    codea = _eval_codea_orthonormal(on[0], on[1], x, lam)
    assert np.max(np.abs(codea - ref)) <= SH_EVAL_TOL * scale
    for conv, coef in (("real_4pi", c), ("real_orthonormal", on), ("complex_4pi", U)):
        grid = np.asarray(sh.synthesis_grid(coef, conv, 90.0 - colat, lon))     # (40, 40): all pairs
        mine = np.diagonal(grid)
        assert np.max(np.abs(mine - ref)) <= SH_EVAL_TOL * scale, conv


@pytest.mark.parametrize("L", [12, 50])
def test_convention_definitions_lovejx(L, sht):
    """lovejx ``evaluate`` of the converted complex coefficients against scipy ``lpmv``
    at the 40 random points of ``test_convention_definitions``."""
    c, colat, lon, x, lam, ref, scale = _convention_case(L)
    U = np.asarray(sh.convert(c, "real_4pi", "complex_4pi"))
    lj = np.asarray(sht.evaluate(U, colat, lon))
    assert np.max(np.abs(lj - ref)) <= SH_EVAL_TOL * scale


@pytest.mark.parametrize("L", [50, 100])
def test_synthesis_grid_against_evaluate(L, sht):
    """``synthesis_grid`` on a 1-degree grid against lovejx ``evaluate`` at the same
    points, each convention."""
    c = _random_real(L, seed=4)
    lat = np.arange(89.0, -90.0, -1.0)
    lon = np.arange(-179.0, 180.0, 1.0)
    U = np.asarray(sh.convert(c, "real_4pi", "complex_4pi"))
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    ref = np.asarray(sht.evaluate(U, 90.0 - LAT, LON))
    for conv, coef in (("real_4pi", c), ("real_orthonormal", c * math.sqrt(4 * math.pi)), ("complex_4pi", U)):
        g = np.asarray(sh.synthesis_grid(coef, conv, lat, lon))
        assert g.shape == (lat.size, lon.size)
        assert np.max(np.abs(g - ref)) <= SH_EVAL_TOL * np.max(np.abs(ref)), conv


def _gl_case(L):
    """The Gauss-Legendre grid ``GaussLegendreGrid.build(L+1, 2L+3, L)``, random complex
    coefficients ``U`` (seed 5), the synthesis ``f`` and the analysis ``U2`` of ``f``."""
    grid = sh.GaussLegendreGrid.build(L + 1, 2 * L + 3, L)
    U = np.asarray(sh.convert(_random_real(L, seed=5), "real_4pi", "complex_4pi"))
    f = np.asarray(jax.jit(sh.synthesis)(grid, U))
    U2 = np.asarray(jax.jit(sh.analysis)(grid, f))
    return grid, U, f, U2


@pytest.mark.parametrize("L", [50, 100])
def test_gl_round_trip(L):
    """``analysis(synthesis(U))`` on ``GaussLegendreGrid.build(L+1, 2L+3, L)`` is the
    identity."""
    _, U, _, U2 = _gl_case(L)
    assert np.max(np.abs(U2 - U)) <= GL_ROUNDTRIP_TOL * np.max(np.abs(U))


@pytest.mark.parametrize("L", [50, 100])
def test_gl_port_equals_lovejx(L, sht):
    """The synthesis and the analysis on ``GaussLegendreGrid.build(L+1, 2L+3, L)`` equal
    those of lovejx on its own grid of the same size."""
    _, U, f, U2 = _gl_case(L)
    ref_grid = sht.GaussLegendreGrid.build(L + 1, 2 * L + 3, L)
    f_ref = np.asarray(sht.synthesis(ref_grid, U))
    assert np.max(np.abs(f - f_ref)) <= GL_PORT_TOL * np.max(np.abs(f_ref))
    U_ref = np.asarray(sht.analysis(ref_grid, f))
    assert np.max(np.abs(U2 - U_ref)) <= GL_PORT_TOL * np.max(np.abs(U_ref))


def test_synthesis_grid_against_fortran_grd(legacy_work, load_fortran):
    """The golden SPHCO coefficients of ``hager4_fsfs`` (``real_orthonormal``) on CODE-A's
    grid (latitudes 89 to -89, longitudes -179 to 179) against its ``GRD_*`` files."""
    d = load_fortran("hager4_fsfs")
    for name in ("geoid", "surftopo", "cmbtopo", "surfvel"):
        grd = np.loadtxt(legacy_work / f"GRD_{name}")
        lon = np.unique(grd[:, 0])
        lat = np.unique(grd[:, 1])[::-1]
        G = grd[:, 2].reshape(len(lat), len(lon))
        coef = np.stack([d[f"sph_{name}_C"], d[f"sph_{name}_S"]])
        f = np.asarray(sh.synthesis_grid(coef, "real_orthonormal", lat, lon,
                                         units=sj.Units(legacy_constants=True)))
        if name == "surfvel":
            err = np.max(np.abs(f - G))
            assert err <= GRD_VEL_ATOL, err
        else:
            err = np.max(np.abs(f - G)) / np.max(np.abs(G))
            assert err <= GRD_TOL, (name, err)
        print(f"GRD_{name}: {err:.2e}")


def test_synthesis_transforms():
    """``synthesis_grid`` under ``jit``, ``grad`` with respect to the coefficients (of the
    quadratic loss ``sum f^2 / 2``, for which the central difference has no truncation
    error) and ``vmap`` over a batch."""
    L = 20
    lat = np.linspace(80, -80, 9)
    lon = np.linspace(-170, 170, 11)
    c = jnp.asarray(_random_real(L, seed=6))
    f = jax.jit(lambda q: sh.synthesis_grid(q, "real_4pi", lat, lon))
    loss = lambda q: 0.5 * jnp.sum(f(q) ** 2)
    g = np.asarray(jax.grad(loss)(c))
    assert np.all(np.isfinite(g))
    d = jnp.asarray(_random_real(L, seed=7))
    fd = (float(loss(c + 1e-3 * d)) - float(loss(c - 1e-3 * d))) / 2e-3
    assert abs(fd - float(np.sum(g * np.asarray(d)))) <= GRAD_FD_TOL * abs(fd)
    batch = jnp.asarray(_random_real(L, batch=(3,), seed=8))
    vm = np.asarray(jax.vmap(f)(batch))
    for b in range(3):
        one = np.asarray(f(batch[b]))
        assert np.max(np.abs(vm[b] - one)) <= VMAP_TOL * np.max(np.abs(one)) + 1e-300
    full = np.asarray(sh.synthesis_grid(batch, "real_4pi", lat, lon))        # batch axes kept
    assert full.shape == (3, 9, 11)


def test_degree_power():
    """``degree_power`` in the three conventions: the complex and the ``real_4pi`` power
    agree, the orthonormal power is ``4 pi`` times it, batch axes are kept."""
    L = 10
    x, y = _random_real(L, batch=(2,), seed=9), _random_real(L, batch=(2,), seed=10)
    p4 = np.asarray(sh.degree_power(x, y, "real_4pi"))
    assert p4.shape == (2, L + 1)
    pc = np.asarray(sh.degree_power(sh.convert(x, "real_4pi", "complex_4pi"),
                                    sh.convert(y, "real_4pi", "complex_4pi"), "complex_4pi"))
    po = np.asarray(sh.degree_power(x * math.sqrt(4 * math.pi), y * math.sqrt(4 * math.pi), "real_orthonormal"))
    assert np.max(np.abs(pc - p4)) <= SH_ROUNDTRIP_TOL * 10 * np.max(np.abs(p4))
    assert np.max(np.abs(po - 4 * math.pi * p4)) <= SH_ROUNDTRIP_TOL * 10 * np.max(np.abs(po))
    ref = np.sum(x * y, axis=(-3, -1))
    assert np.max(np.abs(p4 - ref)) <= SH_ROUNDTRIP_TOL * 10 * np.sum(np.abs(x * y))
