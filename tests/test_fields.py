"""Fields from kernels: the radial weights, ``apply``, the Fortran fields, the gravity
anomaly and the admittance (``stokesjx.fields``; gate G7 of
NOTES/chunks/c3-flags-fields/01-plan.md, section 5.5).

References: the trapezoid rule (exact for linear functions), a numpy loop over every
output, degree, order and batch entry, the golden Fortran ``SPH_*`` arrays and free-air
kernels (CODE-A in 128-bit reals, legacy switches and CODE-A's equal weights), and
closed forms for the gravity anomaly and the admittance.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import cases  # noqa: E402
from adapters import legacy_weights, profile_from_table, to_cilm  # noqa: E402
from test_kernels_legacy import FORTRAN_TOL, LEGACY_UNITS, _legacy_solve, _no_slip_surface, legacy_group  # noqa: E402

import stokesjx as sj  # noqa: E402
from stokesjx import fields, sh, system  # noqa: E402

# ---- tolerances (CONVENTIONS 9) ----
# Weights of the trapezoid rule: sum w = R - r_c and sum w r = (R^2 - r_c^2) / 2,
# relative. Cause: the rounding of the sums of N terms of size 1e5 m.
WEIGHTS_TOL = 1e-15
# The derivative of sum w r^2 with respect to the radius against the closed form,
# relative to the largest entry. Cause: the rounding of a few products.
WEIGHTS_GRAD_TOL = 1e-13
# apply against a numpy loop, e_l per output and degree over every order, C/S and batch
# entry: measured 1.0e-15 (plan-evidence/layout_apply.out). Cause: a different
# summation order of the radial sum of N terms.
APPLY_TOL = 1e-14
# vmap of apply over three densities against three single calls (the c2 constant of
# tests/test_kernels_ad.py).
VMAP_RTOL = 1e-12
# The package legacy kernels with CODE-A's equal weights and CODE-A's sqt against the
# Fortran SPH_* arrays, per degree: the largest difference over the orders and C/S over
# the largest coefficient of the degree. Measured 1.2e-13 on the plan prototype
# (plan-evidence/proto_fields.out). Cause: the rounding of the kernels (FORTRAN_TOL
# measures them at 1.8e-13 at most) and of the radial sum.
FIELDS_FORTRAN_TOL = 1e-12
# The admittance of a field with gravity = c_l topography, and the same admittance in the
# three conventions, relative. Cause: a few roundings of the degree powers.
ADMITTANCE_TOL = 1e-14
# The gravity anomaly against its closed form (l - 1) g N / R, relative per entry.
# Cause: the rounding of two products.
ANOMALY_TOL = 1e-15


# ---------------------------------------------------------------------------
# Weights.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["prem_q", "hager"])
def test_weights_sum_and_linear(name):
    """The trapezoid weights sum to ``R - r_c`` and integrate ``r`` exactly."""
    table = cases.t_prem(reduced=True) if name == "prem_q" else cases.t_hager()
    p = profile_from_table(table)
    w = np.asarray(fields.quadrature_weights(p))
    r = np.asarray(p.radius)
    R, rc = r[-1], r[0]
    assert w.shape == r.shape and np.all(w >= 0)
    assert abs(np.sum(w) - (R - rc)) <= WEIGHTS_TOL * (R - rc)
    assert abs(np.sum(w * r) - (R ** 2 - rc ** 2) / 2) <= WEIGHTS_TOL * (R ** 2 - rc ** 2) / 2


def test_weights_pairs():
    """Each node of a pair gets the half interval on its own side; a traced radius with
    the two nodes of a pair 1 mm apart (the structure given) gives the same pair weights
    and a zero pair width in ``system.interval_widths`` (the static mask)."""
    table = cases.t_prem(reduced=True)
    p = profile_from_table(table)
    r = np.asarray(p.radius)
    w = np.asarray(fields.quadrature_weights(p))
    for i in p.structure.pairs:
        assert w[i - 1] == 0.5 * (r[i - 1] - r[i - 2])
        assert w[i] == 0.5 * (r[i + 1] - r[i]) if i + 1 < len(r) else w[i] == 0.0
    i = p.structure.pairs[0]
    r2 = r.copy()
    r2[i] += 1e-3

    def both(radius):
        q = p.replace(radius=radius)
        return fields.quadrature_weights(q), system.interval_widths(radius, q.structure)
    wt, ht = (np.asarray(x) for x in jax.jit(both)(jnp.asarray(r2)))
    assert ht[i - 1] == 0.0
    assert wt[i - 1] == 0.5 * (r2[i - 1] - r2[i - 2]) and wt[i] == 0.5 * (r2[i + 1] - r2[i])


def test_weights_derivative():
    """``grad`` of ``sum_j w_j r_j^2`` with respect to the radius against the closed form
    of the trapezoid sum ``sum_i Delta_i (r_i^2 + r_(i-1)^2) / 2`` with the masked widths."""
    p = profile_from_table(cases.t_prem(reduced=True))
    r = np.asarray(p.radius)
    m = p.structure.interval_mask
    g = np.asarray(jax.grad(lambda x: jnp.sum(fields.quadrature_weights(p.replace(radius=x)) * x ** 2))(
        jnp.asarray(r)))
    ref = np.zeros_like(r)
    for i in range(1, len(r)):
        # d/dr_i and d/dr_(i-1) of m_i (r_i - r_(i-1)) (r_i^2 + r_(i-1)^2) / 2.
        d = r[i] - r[i - 1]
        s = (r[i] ** 2 + r[i - 1] ** 2) / 2
        ref[i] += m[i - 1] * (s + d * r[i])
        ref[i - 1] += m[i - 1] * (-s + d * r[i - 1])
    assert np.max(np.abs(g - ref)) <= WEIGHTS_GRAD_TOL * np.max(np.abs(ref))


# ---------------------------------------------------------------------------
# apply.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _kernels(refuse_degree_one: bool):
    """Kernels of the reduced hager4 table at lmax = 8: the default (degrees 1..8) or a
    no-slip surface under FULL with hydrostatic gravity (degree 1 refused, 2..8)."""
    p = profile_from_table(cases.t_hager(reduced=True))
    form = sj.Formulation(surface=sj.NoSlip()) if refuse_degree_one else sj.Formulation()
    return sj.stokes_kernels(p, sj.StokesConfig(lmax=8, formulation=form)), p


def _loop(K, w, d, degrees, lmax):
    """Reference: an explicit loop over output, degree, order and the batch axes."""
    Ks = [np.asarray(getattr(K, n)) for n in fields.FIELD_NAMES]
    d = np.asarray(d)
    batch = d.shape[1:-2]
    out = np.zeros((4,) + batch + (lmax + 1, lmax + 1), dtype=np.result_type(Ks[0], d))
    for o in range(4):
        for i, l in enumerate(degrees):
            for m in range(lmax + 1):
                for idx in np.ndindex(*batch):
                    out[(o,) + idx + (l, m)] = np.sum(Ks[o][i] * w * d[(slice(None),) + idx + (l, m)])
    return out


def _density(shape, complex_=False, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(shape)
    if complex_:
        d = d + 1j * rng.standard_normal(shape)
    return d


@pytest.mark.parametrize("refuse", [False, True], ids=["degrees_1", "degrees_2"])
def test_apply_against_loop(refuse):
    """``apply`` against a numpy loop for a complex density ``(N, Ld+1, Ld+1)``, a real
    density ``(N, 2, Ld+1, Ld+1)`` and a real density with a batch axis
    ``(N, 3, 2, Ld+1, Ld+1)``, at ``Ld = lmax`` and ``Ld = lmax + 10``; the output shapes
    of the table of plan 3.7; the rows below the first kernel degree exact zeros."""
    K, p = _kernels(refuse)
    lmax, d0 = K.degrees[-1], K.degrees[0]
    assert d0 == (2 if refuse else 1)
    N = p.structure.n_nodes
    w = np.asarray(fields.quadrature_weights(p))
    worst = 0.0
    for Ld in (lmax, lmax + 10):
        for shape, cplx in (((N, Ld + 1, Ld + 1), True), ((N, 2, Ld + 1, Ld + 1), False),
                            ((N, 3, 2, Ld + 1, Ld + 1), False)):
            d = _density(shape, cplx, seed=Ld)
            F = fields.apply(K, jnp.asarray(d), p)
            ref = _loop(K, w, d, K.degrees, lmax)
            assert F.degrees == K.degrees
            for o, n in enumerate(fields.FIELD_NAMES):
                arr = np.asarray(getattr(F, n))
                assert arr.shape == shape[1:-2] + (lmax + 1, lmax + 1)
                assert arr.dtype == (np.complex128 if cplx else np.float64)
                assert np.all(arr[..., :d0, :] == 0)
                for l in K.degrees:
                    scale = np.max(np.abs(ref[o][..., l, :]))
                    e = np.max(np.abs(arr[..., l, :] - ref[o][..., l, :])) / scale if scale > 0 else 0.0
                    worst = max(worst, e)
                    assert e <= APPLY_TOL, (n, l, e)
    print(f"apply against the loop: worst e_l {worst:.2e}")


def test_apply_layout_checks(tmp_path):
    """``ValueError`` for a first axis that is not ``N``, a non-square degree-order block
    (the ``[j, l, m, C/S]`` layout of ``tests/cases.py`` at ``Ld = 20``) and ``Ld < lmax``;
    ``apply`` takes the output of ``read_sph`` with no transpose and equals ``apply`` of the
    written coefficients."""
    from stokesjx.io.legacy import read_sph
    K, p = _kernels(False)
    N = p.structure.n_nodes
    with pytest.raises(ValueError, match="radial"):
        fields.apply(K, np.zeros((N + 1, 2, 9, 9)), p)
    with pytest.raises(ValueError, match="moveaxis"):
        fields.apply(K, np.zeros((N, 21, 21, 2)), p)
    with pytest.raises(ValueError, match="Ld >= 8"):
        fields.apply(K, np.zeros((N, 2, 8, 8)), p)
    with pytest.raises(ValueError):
        fields.apply(K, np.zeros((N, 9)), p)
    table = cases.t_hager(reduced=True)
    rng = np.random.default_rng(14)
    coef = rng.standard_normal((N, 9, 9, 2)) * np.tril(np.ones((9, 9)))[None, :, :, None]
    path = tmp_path / "small.sph"
    path.write_text(cases.density_file_text(table.radius, table.rho0, coef, nr=N - 1, lmax=8))
    sph = read_sph(path)
    a = fields.apply(K, sph.coefficients, p)
    b = fields.apply(K, to_cilm(coef), p)
    for n in fields.FIELD_NAMES:
        assert np.array_equal(np.asarray(getattr(a, n)), np.asarray(getattr(b, n)))


def test_apply_vmap():
    """``vmap`` of ``apply`` over three densities equals three single calls."""
    K, p = _kernels(False)
    N = p.structure.n_nodes
    D = jnp.asarray(_density((3, N, 2, 9, 9), seed=15))
    Fv = jax.jit(jax.vmap(lambda d: fields.apply(K, d, p)))(D)
    for b in range(3):
        F1 = fields.apply(K, D[b], p)
        for n in fields.FIELD_NAMES:
            one = np.asarray(getattr(F1, n))
            assert np.max(np.abs(np.asarray(getattr(Fv, n))[b] - one)) <= VMAP_RTOL * np.max(np.abs(one))


@pytest.mark.parametrize("cplx", [False, True], ids=["real", "complex"])
def test_apply_one_einsum(cplx):
    """The jaxpr of ``apply`` (kernels and density as inputs) has exactly one
    ``dot_general``."""
    K, p = _kernels(True)
    N = p.structure.n_nodes
    d = jnp.asarray(_density((N, 9, 9) if cplx else (N, 2, 9, 9), cplx))
    jaxpr = str(jax.make_jaxpr(lambda k, x: fields.apply(k, x, p))(K, d))
    assert jaxpr.count("dot_general") == 1


FORTRAN_FIELD_RUNS = ("iso_fsfs", "hager4_fsfs", "smooth_fsfs", "hager4_nsfs", "hager4_fsfs_tala")
FORTRAN_FIELDS = (("geoid", "geoid"), ("surftopo", "surface_topography"), ("cmbtopo", "cmb_topography"),
                  ("surfvel", "surface_velocity"))


@functools.lru_cache(maxsize=None)
def _synthetic_orthonormal():
    """The synthetic density of the golden runs, ``real_orthonormal`` with CODE-A's sqt,
    ``(129, 2, 51, 51)``."""
    c = to_cilm(cases.synthetic_coefficients())
    return np.asarray(sh.convert(c, "real_4pi", "real_orthonormal", units=LEGACY_UNITS))


def _fortran_field_check(run, density, F):
    """The fields of the package legacy kernels with CODE-A's equal weights against the
    Fortran SPH_* arrays; returns the worst error."""
    sol = _legacy_solve(run)
    K = sj.Kernels(degrees=tuple(range(1, 51)), degree_one_frame=sol["frame"], radius=sol["radius"],
                   geoid=sol["geoid"], surface_topography=sol["surface_topography"],
                   cmb_topography=sol["cmb_topography"], surface_velocity=sol["surface_velocity"],
                   degree_one_imbalance=sol["imbalance"])
    out = fields.radial_integral(K, density, legacy_weights(sol["radius"]))
    worst = 0.0
    for fk, n in FORTRAN_FIELDS:
        if fk == "surfvel" and _no_slip_surface("hager4_fsfs" if run == "hager4_fsfs_lebedev" else run):
            assert np.all(np.asarray(out.surface_velocity) == 0)
            continue
        got = np.asarray(getattr(out, n))                          # (2, 51, 51)
        ref = np.stack([F[f"sph_{fk}_C"], F[f"sph_{fk}_S"]])
        assert got.shape == ref.shape
        for l in range(51):
            scale = np.max(np.abs(ref[:, l, :]))
            if scale == 0:
                assert np.all(got[:, l, :] == 0), (run, fk, l)
                continue
            e = np.max(np.abs(got[:, l, :] - ref[:, l, :])) / scale
            worst = max(worst, e)
            assert e <= FIELDS_FORTRAN_TOL, (run, fk, l, e)
    return worst


# hager4_nsfs and hager4_fsfs_tala run in the slow set: the second move of the
# quick-set rule of NOTES/chunks/c3-flags-fields/01-plan.md (risk R1). Their kernels are
# compared with the Fortran in the quick set (tests/test_kernels_legacy.py), and the
# field code is the same for every run.
_SLOW_FIELD_RUNS = ("hager4_nsfs", "hager4_fsfs_tala")


@pytest.mark.parametrize("run", [pytest.param(r, marks=(pytest.mark.slow, pytest.mark.xdist_group(f"legacy_{r}")))
                                 if r in _SLOW_FIELD_RUNS else legacy_group([r])[0] for r in FORTRAN_FIELD_RUNS])
def test_fields_against_fortran(run, load_fortran):
    """G7: the four fields (three under a no-slip surface) of the synthetic density with
    the package legacy kernels, CODE-A's equal weights and sqt, against the Fortran
    ``SPH_*`` arrays at every degree."""
    worst = _fortran_field_check(run, _synthetic_orthonormal(), load_fortran(run))
    print(f"fields {run}: worst {worst:.2e}")


@pytest.mark.legacy
@pytest.mark.xdist_group("legacy_hager4_fsfs")
def test_fields_against_fortran_lebedev(load_fortran, legacy_dir):
    """G7: the LEBEDEV density file read by ``read_sph`` (``real_4pi``, converted with
    CODE-A's sqt) with the kernels of ``hager4_fsfs`` against ``hager4_fsfs_lebedev``."""
    from stokesjx.io.legacy import read_sph
    sph = read_sph(legacy_dir / "INPUT" / "DENSITY_INPUTS" / "LEBEDEV_S4RTS.sph")
    d = np.asarray(sh.convert(sph.coefficients, "real_4pi", "real_orthonormal", units=LEGACY_UNITS))
    worst = _fortran_field_check("hager4_fsfs_lebedev", d, load_fortran("hager4_fsfs_lebedev"))
    print(f"fields hager4_fsfs_lebedev: worst {worst:.2e}")


@pytest.mark.parametrize("run", legacy_group(("iso_fsfs", "hager4_fsfs", "smooth_fsfs")))
def test_gravity_anomaly_against_fortran(run, load_fortran):
    """CODE-A's free-air kernel (driver.f:259-261) is ``1e5 (l - 1) (G Emass / R^2) N / R``:
    ``gravity_anomaly`` of the package legacy geoid kernel with CODE-A's gravity
    ``G Emass / R^2`` (float32 constants), times 1e5, against the Fortran ``gravity``
    kernel at all 50 degrees."""
    from adapters import assert_rows_close
    sol = _legacy_solve(run)
    R = float(sol["radius"][-1])
    g = LEGACY_UNITS.G * LEGACY_UNITS.earth_mass / R ** 2
    got = 1e5 * np.asarray(fields.gravity_anomaly(sol["geoid"], g, R, degrees=tuple(range(1, 51))))
    assert_rows_close(got, load_fortran(run)["gravity"], FORTRAN_TOL, f"{run} gravity")


def test_gravity_anomaly_rows():
    """D10: on a coefficient array with a nonzero ``l = 0`` entry and a nonzero ``l = 1``
    row, in each convention and with a batch axis: the output has the input shape, rows
    ``l = 0`` and ``l = 1`` are exact zeros, row ``l`` is ``(l - 1) g N_l / R``; on a kernel
    array the factor runs along axis 0; a wrong number of degrees raises;
    ``surface_gravity`` is the hydrostatic ``g(R)`` or the uniform value."""
    L, g, R = 12, 9.81, 6.37e6
    rng = np.random.default_rng(16)
    real = rng.standard_normal((3, 2, L + 1, L + 1))
    for conv, x in (("real_4pi", real), ("real_orthonormal", real * 3.5),
                    ("complex_4pi", np.asarray(sh.convert(real, "real_4pi", "complex_4pi")))):
        out = np.asarray(fields.gravity_anomaly(x, g, R))
        assert out.shape == x.shape
        assert np.all(out[..., 0, :] == 0) and np.all(out[..., 1, :] == 0)
        for l in range(2, L + 1):
            ref = (l - 1) * g * x[..., l, :] / R
            assert np.max(np.abs(out[..., l, :] - ref)) <= ANOMALY_TOL * np.max(np.abs(ref)), (conv, l)
    K = rng.standard_normal((19, 30))
    out = np.asarray(fields.gravity_anomaly(K, g, R, degrees=tuple(range(2, 21))))
    for i, l in enumerate(range(2, 21)):
        assert np.max(np.abs(out[i] - (l - 1) * g * K[i] / R)) <= ANOMALY_TOL * np.max(np.abs(out[i]))
    with pytest.raises(ValueError):
        fields.gravity_anomaly(K, g, R, degrees=tuple(range(2, 20)))
    p = profile_from_table(cases.t_hager(reduced=True))
    units = sj.Units()
    g_h = float(fields.surface_gravity(p, sj.StokesConfig(lmax=2)))
    assert g_h == float(sj.integrate_gravity(p.radius, p.rho0, p.core_mass, units)[-1])
    uni = sj.StokesConfig(lmax=2, formulation=sj.Formulation(background=sj.Uniform(9.5)))
    assert float(fields.surface_gravity(p, uni)) == 9.5


def test_admittance():
    """D10: ``gravity = c_l topography`` per degree gives ``Z_l = c_l``; the same fields in
    the three conventions give the same ``Z_l``; the batch axes are kept, the output is
    ``(..., L+1)``; ``S_hh = 0`` gives 0 with a finite gradient; a complex array with a real
    convention raises."""
    L = 10
    rng = np.random.default_rng(17)
    topo = rng.standard_normal((4, 2, L + 1, L + 1)) * np.tril(np.ones((L + 1, L + 1)))
    topo[..., 1, :, 0] = 0.0
    topo[..., 0, :] = 0.0                                 # degree 0: S_hh = 0
    c = rng.uniform(1, 3, L + 1)
    grav = topo * c[:, None]
    Z = np.asarray(fields.admittance(grav, topo, "real_4pi"))
    assert Z.shape == (4, L + 1)
    assert np.all(Z[:, 0] == 0)
    assert np.max(np.abs(Z[:, 1:] - c[1:]) / c[1:]) <= ADMITTANCE_TOL
    grav2 = topo * c[:, None] + rng.standard_normal(topo.shape) * np.tril(np.ones((L + 1, L + 1)))
    grav2[..., 1, :, 0] = 0.0
    Z4 = np.asarray(fields.admittance(grav2, topo, "real_4pi"))
    Zo = np.asarray(fields.admittance(grav2 * 3.5, topo * 3.5, "real_orthonormal"))
    Zc = np.asarray(fields.admittance(sh.convert(grav2, "real_4pi", "complex_4pi"),
                                      sh.convert(topo, "real_4pi", "complex_4pi"), "complex_4pi"))
    scale = np.max(np.abs(Z4))
    assert np.max(np.abs(Zo - Z4)) <= ADMITTANCE_TOL * scale
    assert np.max(np.abs(Zc - Z4)) <= ADMITTANCE_TOL * scale
    g = jax.grad(lambda t: jnp.sum(fields.admittance(jnp.asarray(grav), t, "real_4pi")))(jnp.asarray(topo))
    assert np.all(np.isfinite(np.asarray(g)))
    with pytest.raises(ValueError):
        fields.admittance(sh.convert(grav, "real_4pi", "complex_4pi"),
                          sh.convert(topo, "real_4pi", "complex_4pi"), "real_4pi")
