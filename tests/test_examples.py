"""Tests of the examples in ``examples/`` (NOTES/chunks/c5-integration/01-plan.md section
6.1): the PREM-like table of ``earth_model.py`` against ``cases.t_prem``, the forward run
from a CODE-A density file, the gradient example, and the HMC example (the leapfrog, the
Metropolis step, the metric, the MAP search and gate, the effective sample size, and the
command lines).

The examples are imported by their path with ``importlib.util.spec_from_file_location``
under a module name of their own (``load_example``), never through a ``sys.path`` change,
so that no example module shadows a test helper in a pytest-xdist worker.

Every tolerance is a named constant with its reference and the cause of the error
(CONVENTIONS 9); the measured values are those of the plan (section 6.1) and of
NOTES/chunks/c5-integration/03-implementation.md.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import cases  # noqa: E402
import densities  # noqa: E402
from adapters import profile_from_table  # noqa: E402

import stokesjx as sj  # noqa: E402
from stokesjx import sh  # noqa: E402
from stokesjx.io import legacy  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
LEBEDEV_SPH = REPO / "legacy" / "CODE-A" / "INPUT" / "DENSITY_INPUTS" / "LEBEDEV_S4RTS.sph"
EPS64 = np.finfo(np.float64).eps
KM = 1e3

# ---------------------------------------------------------------------------
# Named tolerances.
# ---------------------------------------------------------------------------
# E1: the example table has linspace radii, cases.t_prem the radii as LEBEDEV_S4RTS.sph
# prints them (two decimals): measured offset at most 5.0 mm.
RADIUS_ATOL = 0.01
# E1: rho0 at radii up to 5 mm apart: 5 mm times the largest PREM slope in the mantle
# (1.26e-3 kg/m^3 per m) over 3800 kg/m^3 is 1.66e-9; measured 1.68e-9.
RHO_RTOL = 5e-9
# E1: the viscosity of a layer is the same float64 number in both tables (measured 0).
ETA_RTOL = 1e-15
# E1: core density and mass: float64 Horner and closed form against 40-digit decimals
# (measured 2.2e-16).
CORE_RTOL = 1e-12
# E1p, E1b: float64 Horner against the 40-digit decimal evaluation at the same radius
# (measured 3.8e-16).
RHO_POLY_RTOL = 1e-14
# E2: the example and the test call the same solver with the same arrays (e_l of
# CONVENTIONS 10 per degree; same arithmetic, so the measured value is 0).
FWD_TOL = 1e-13
# E2b: the SPHCO files hold e15.6, a mantissa 0.dddddd of six digits: the rounding is at
# most half a unit of the sixth digit, 5e-7 of 10^e for a value in [0.1, 1) 10^e, so at
# most 5e-6 relative to the value (at a mantissa of 0.1) and at most 5e-6 in e_l. Measured
# 1.18e-6 in e_l. The pointwise check of the format (SPHCO_HALF_UNIT) is exact.
SPHCO_TOL = 5e-6
# E2b: each entry against half a unit of the sixth digit of its own mantissa, with a
# relative margin of 1e-9 for the float64 rounding of the reference.
SPHCO_HALF_UNIT = 0.5e-6
# E3: the per-degree correlation of the example's geoid (SI physics) with the Fortran
# geoid (legacy physics: g = 10, upper-node material, CODE-A's weights), l = 2 .. 50.
# Measured 0.980 (l = 4), median 0.9989.
CORR_MIN = 0.9
# E4, E9: the gradient against central differences, per component. The difference is the
# truncation error of the central difference (measured 5.6e-10 to 2.0e-9; a fourth of it
# at half the steps).
FD_RTOL = 1e-6
# E4: the radius derivative times the radius step over the misfit (measured 2.6e-4; at
# the isoviscous point 2e-20).
RADIUS_SIGNAL_MIN = 1e-8
# E4b: the sum of the log-viscosity derivatives over their norm (geoid invariance under
# a uniform viscosity factor; measured 8e-16 to 3e-15).
RIDGE_TOL = 1e-8
# E4c: the NumPy front calls the same compiled function (measured 0).
NUMPY_RTOL = 1e-14
# E5: the energy error of a second-order leapfrog falls by 4 when the step halves
# (measured 3.94 to 4.05 over 200 momentum draws at a tenth of the momentum scale).
ORDER_RATIO_BAND = (3.0, 5.5)
# E5: the energy error at the smaller step must be above the rounding of U by this factor
# times eps |U| (measured |dH| 3.6e-8 to 5.8e-6 against 1e3 eps |U| = 2.5e-12).
DH_RESOLUTION = 1e3
# E5b: a gradient biased by 1 % gives an energy error of order zero: the ratio falls
# below 3, or the error at the smaller step is more than 10 times the unbiased one
# (measured: control ratio median 0.952, |dH_b| / |dH| at least 19.3).
BIAS_FACTOR = 10.0
# E6: the largest |dH| of a smoke chain (measured 14.6 over 400 transitions of 50 seeds at
# step 0.2; at step 0.5 1.5e5 and 5 non-finite values).
SMOKE_DH_MAX = 100.0
# E6, E6b: the potential that the chain carries against the compiled value at the same
# point (measured 0.0 with the same call, 5.3e-15 between a batched and an unbatched call).
STATE_RTOL = 1e-10
# E10, E10b: the Gauss-Newton metric against jax.hessian at the noise-free truth, and the
# linearity of the metric in the weights (Frobenius, relative; measured 9.7e-16).
METRIC_RTOL = 1e-8
# E11: the smoke MAP against the end of a search from the truth, relative in U (both
# searches stop at |g|_M < 1e-6, so U agrees to about 1e-12 relative).
MAP_U_RTOL = 1e-9
# E12: the effective sample size of AR(1) against n / tau (measured over 200 seeds: sd of
# the relative error 0.014 at a = 0 and 0.022 at a = 0.5, largest 0.041 and 0.065).
ESS_RTOL = 0.15

# E6b (6.1.1 of the plan): the Metropolis step on 512 proposals at step 0.5.
METRO_SEED = 5
METRO_N = 512
METRO_STEP = 0.5
# Condition 1: the returned dH against the dH of the test (measured 7.4e-15; a sign error
# gives 2.0), for proposals with dH <= DH_CLASS_C.
ENERGY_RTOL = 1e-10
# The class limit of dH: above it a correct rule accepts with probability below exp(-40).
DH_CLASS_C = 40.0
# Condition 2: the class counts (measured minimum over 300 key sets: A 28, B 281, C 129;
# accepted in B 182, rejected in B 68).
MIN_A, MIN_B, MIN_C, MIN_ACC, MIN_REJ = 10, 100, 40, 50, 20
# Condition 5: the t statistic of the accepted count in class B (correct rule over 300 key
# sets: sd 1.00, largest |t| 3.83; the defects 14.8 to 20.2).
METRO_T_MAX = 5.0
# Condition 8: the returned probability against min(1, exp(-dH)), absolute.
PROB_ATOL = 1e-14
# Condition 9: the radius range of a path against its two end points, m.
RANGE_ATOL = 1e-6
# Condition 11: the covariance of the whitened start momenta against the identity
# (correct draw 0.052 to 0.268 over 300 key sets; the three wrong matrices at least 1.41).
MOMENTUM_COV_MAX = 0.5

# G5 of the plan (E8, the default run A).
MEAN_PROB_MIN = 0.9
DH_MAX_RUN = 10.0
MEAN_SD_MAX = 3.0
SD_RATIO_BAND = (0.5, 2.0)
CROSSINGS_MIN = 500


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def load_example(name: str):
    """The module ``examples/<name>.py`` under the name ``stokesjx_examples_<name>``
    (registered in ``sys.modules``; the examples load ``earth_model`` under the same
    name)."""
    mod_name = f"stokesjx_examples_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def strict_json(text: str):
    """``json.loads`` that refuses ``NaN`` and ``Infinity``."""
    def refuse(token):
        raise ValueError(f"non-standard JSON constant {token}")
    return json.loads(text, parse_constant=refuse)


def field_rows_error(actual, reference) -> float:
    """``max_l e_l`` of two coefficient arrays ``(2, L+1, L+1)``: per degree the largest
    difference over the ``C/S`` and order axes over the largest reference entry; degrees
    with an all-zero reference must be zero in ``actual``."""
    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    assert a.shape == r.shape, (a.shape, r.shape)
    worst = 0.0
    for l in range(r.shape[-2]):
        scale = np.max(np.abs(r[..., l, :]))
        diff = np.max(np.abs(a[..., l, :] - r[..., l, :]))
        if scale == 0:
            assert diff == 0, f"degree {l}: reference zero, actual {diff}"
        else:
            worst = max(worst, diff / scale)
    return worst


def hager4_depth_rule(radius) -> np.ndarray:
    """The hager4 viscosity of single nodes by depth, written in the test (not taken from
    the example): 1e20 Pa s above 100 km, 1e19/30 to 400 km, 1e19 to 670 km, 1e20 below;
    a node on a layer bottom belongs to the layer below."""
    d = 6370e3 - np.asarray(radius, dtype=np.float64)
    return np.where(d < 100e3, 1e20, np.where(d < 400e3, 1e19 / 30, np.where(d < 670e3, 1e19, 1e20)))


# The 17-node reduced LEBEDEV table of E2 (every 8th node, nr = 16) and lmax 6.
E2_INDEX = list(range(0, cases.LEBEDEV_NODES, 8))
E2_LMAX = 6


@pytest.fixture(scope="module")
def e2_files(tmp_path_factory):
    """A ``.sph`` file on the 17-node reduced LEBEDEV table (``cases.density_file_text``)
    with the coefficients ``densities.smooth(seed 3)``, and a viscosity file with a
    relative viscosity ``10^((k mod 5) - 2)`` at node ``k`` (CMB first), different at every
    neighbour and not symmetric about the middle."""
    d = tmp_path_factory.mktemp("e2")
    radius = [cases.LEBEDEV_RADIUS[k] for k in E2_INDEX]
    rho0 = [cases.LEBEDEV_RHO0[k] for k in E2_INDEX]
    r = np.array([float(x) for x in radius])
    coef = densities.smooth(r, E2_LMAX, 3)                       # (N, 2, L+1, L+1)
    sph = d / "reduced.sph"
    sph.write_text(cases.density_file_text(radius, rho0, np.moveaxis(coef, 1, -1), nr=len(radius) - 1,
                                           lmax=E2_LMAX))
    eta_rel = [repr(10.0 ** ((k % 5) - 2)) for k in range(len(radius))]
    visc = d / "visc.dat"
    visc.write_text(cases.viscosity_file_text(eta_rel))
    return {"sph": sph, "visc": visc, "radius": r, "rho0": np.array([float(x) for x in rho0]),
            "coef": coef, "eta_rel": np.array([float(x) for x in eta_rel]), "dir": d}


def e2_reference(files, eta):
    """The kernels and fields of the E2 table built in the test with the viscosity
    ``eta`` (Pa s, CMB first)."""
    p = sj.RadialProfile(files["radius"], files["rho0"], eta, core_density=9900.0)
    K = sj.stokes_kernels(p, sj.StokesConfig(lmax=E2_LMAX))
    return K, sj.apply(K, files["coef"], p)


# ---------------------------------------------------------------------------
# E1, E1p, E1b: the table of earth_model.py.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reduced", [False, True], ids=["refine1", "reduced"])
def test_earth_model_matches_t_prem(reduced):
    """E1: ``prem_profile()`` and ``prem_profile(reduced=True)`` against
    ``cases.t_prem`` (the table of every c4 and c4b measurement)."""
    em = load_example("earth_model")
    p = em.prem_profile(reduced=reduced)
    q = profile_from_table(cases.t_prem(reduced=reduced))
    assert p.structure.pairs == q.structure.pairs
    assert p.structure.consistent_core == q.structure.consistent_core
    np.testing.assert_allclose(np.asarray(p.radius), np.asarray(q.radius), rtol=0, atol=RADIUS_ATOL)
    np.testing.assert_allclose(np.asarray(p.rho0), np.asarray(q.rho0), rtol=RHO_RTOL, atol=0)
    np.testing.assert_allclose(np.asarray(p.eta), np.asarray(q.eta), rtol=ETA_RTOL, atol=0)
    np.testing.assert_allclose(float(p.core_density), float(q.core_density), rtol=CORE_RTOL)
    np.testing.assert_allclose(float(p.core_mass), float(q.core_mass), rtol=CORE_RTOL)


@pytest.mark.parametrize("reduced", [False, True], ids=["refine1", "reduced"])
def test_prem_density_matches_cases(reduced):
    """E1p: ``prem_density`` at the radii of the reference table against its ``rho0``:
    the polynomials and the region rule without the radius offset."""
    em = load_example("earth_model")
    table = cases.t_prem(reduced=reduced)
    r = np.array([float(x) for x in table.radius])
    side = np.zeros(r.size, dtype=int)
    for k in range(r.size - 1):
        if r[k] == r[k + 1]:
            side[k], side[k + 1] = -1, 1
    rho = em.prem_density(r, side)
    ref = np.array([float(x) for x in table.rho0])
    np.testing.assert_allclose(rho, ref, rtol=RHO_POLY_RTOL, atol=0)


def test_earth_model_refined_counts():
    """E1b: the refined tables have 265 and 521 nodes, pairs at 100, 220, 400, 670 km depth,
    and ``rho0`` of a pair member equal to ``prem_density`` of its side."""
    em = load_example("earth_model")
    for refine, n in ((2, 265), (4, 521)):
        p = em.prem_profile(refine)
        r = np.asarray(p.radius)
        assert r.size == n
        pairs = p.structure.pairs
        assert len(pairs) == 4
        np.testing.assert_array_equal(np.sort(6370e3 - r[list(pairs)]), [100e3, 220e3, 400e3, 670e3])
        # The density jumps at 220, 400 and 670 km and not at 100 km (a viscosity pair).
        jumps = [6370e3 - r[k] for k in pairs if p.rho0[k] != p.rho0[k - 1]]
        assert sorted(jumps) == [220e3, 400e3, 670e3]
        for k in pairs:
            # NodeStructure.pairs holds the upper member k of each pair; k - 1 is the lower.
            assert r[k - 1] == r[k]
            np.testing.assert_allclose(float(p.rho0[k - 1]), em.prem_density(r[k - 1], -1), rtol=RHO_POLY_RTOL)
            np.testing.assert_allclose(float(p.rho0[k]), em.prem_density(r[k], 1), rtol=RHO_POLY_RTOL)
        assert np.all(np.diff(r) >= 0)


# ---------------------------------------------------------------------------
# E2, E2v, E2g, E2b, E3: the forward example.
# ---------------------------------------------------------------------------


def test_forward_example_fields(e2_files, tmp_path):
    """E2 and E2b: ``run()`` on the E2 file against the package called in the test with
    the hager4 viscosity of the test's own depth rule; the SPHCO files read back."""
    fwd = load_example("forward_legacy_density")
    out = fwd.run(fwd.parse_args(["--sph", str(e2_files["sph"]), "--out", str(tmp_path)]))
    K_ref, F_ref = e2_reference(e2_files, hager4_depth_rule(e2_files["radius"]))
    assert out["kernels"].degrees == K_ref.degrees
    for name in ("geoid", "surface_topography", "cmb_topography", "surface_velocity"):
        e = field_rows_error(getattr(out["fields"], name), getattr(F_ref, name))
        assert e <= FWD_TOL, (name, e)
        # E2b: the SPHCO file of the field against the field in real_orthonormal.
        coef = legacy.read_sphco(tmp_path / fwd.SPHCO_NAMES[name])
        ref = np.array(sh.convert(getattr(F_ref, name), "real_4pi", "real_orthonormal"))
        ref[..., 0, :] = 0.0                                     # the files start at l = 1
        ref[1, :, 0] = 0.0                                       # and write S of m = 0 as zero
        e = field_rows_error(coef, ref)
        assert e <= SPHCO_TOL, (name, e)
        nz = ref != 0
        unit = 10.0 ** np.floor(np.log10(np.abs(ref[nz])) + 1.0)    # 10^e of the mantissa 0.dddddd
        assert np.all(np.abs(coef[nz] - ref[nz]) <= SPHCO_HALF_UNIT * unit * (1 + 1e-9)), name
        assert np.all(coef[~nz] == 0.0), name
    with np.load(tmp_path / "kernels_fields.npz") as z:
        np.testing.assert_array_equal(z["kernel_geoid"], np.asarray(out["kernels"].geoid))
        np.testing.assert_array_equal(z["radius"], e2_files["radius"])


def test_forward_example_viscosity_file(e2_files, tmp_path):
    """E2v: ``--viscosity`` with a file whose relative viscosity differs at every node; the
    reference uses ``1e19 eta_rel`` CMB first, so a reversed or shifted read fails."""
    fwd = load_example("forward_legacy_density")
    out = fwd.run(fwd.parse_args(["--sph", str(e2_files["sph"]), "--viscosity", str(e2_files["visc"]),
                                  "--out", str(tmp_path)]))
    np.testing.assert_array_equal(np.asarray(out["profile"].eta), 1e19 * e2_files["eta_rel"])
    _, F_ref = e2_reference(e2_files, 1e19 * e2_files["eta_rel"])
    for name in ("geoid", "surface_topography", "cmb_topography", "surface_velocity"):
        assert field_rows_error(getattr(out["fields"], name), getattr(F_ref, name)) <= FWD_TOL, name


def test_forward_example_grid(e2_files, tmp_path):
    """E2g: ``--grid`` writes the four fields on a 1-degree grid of cell centres; each grid
    against ``sh.synthesis_grid`` of the reference fields at the stored latitudes and
    longitudes."""
    fwd = load_example("forward_legacy_density")
    fwd.run(fwd.parse_args(["--sph", str(e2_files["sph"]), "--out", str(tmp_path), "--grid"]))
    _, F_ref = e2_reference(e2_files, hager4_depth_rule(e2_files["radius"]))
    with np.load(tmp_path / "grids.npz") as z:
        lat, lon = z["latitude"], z["longitude"]
        assert lat.shape == (180,) and lon.shape == (360,)
        assert lat[0] == 89.5 and lat[-1] == -89.5 and lon[0] == 0.5 and lon[-1] == 359.5
        for name in ("geoid", "surface_topography", "cmb_topography", "surface_velocity"):
            g = z[f"grid_{name}"]
            ref = np.asarray(sh.synthesis_grid(getattr(F_ref, name), "real_4pi", lat, lon))
            assert g.shape == (180, 360)
            assert np.max(np.abs(g - ref)) <= FWD_TOL * np.max(np.abs(ref)), name


@pytest.mark.legacy
def test_forward_example_lebedev(load_fortran, tmp_path):
    """E3: the real LEBEDEV file at ``lmax = 50``: finite fields, and the per-degree
    correlation (the cosine of the angle between the coefficient vectors of one degree)
    of the geoid with the Fortran golden geoid of ``hager4_fsfs_lebedev`` for
    ``2 <= l <= 50``. The Fortran used CODE-A's legacy physics, so the fields differ; the
    correlation shows that the reader and the conventions of the example agree with
    CODE-A."""
    if not LEBEDEV_SPH.is_file():
        pytest.skip("legacy/CODE-A/INPUT/DENSITY_INPUTS/LEBEDEV_S4RTS.sph is not present")
    fwd = load_example("forward_legacy_density")
    out = fwd.run(fwd.parse_args(["--out", str(tmp_path)]))
    for name in ("geoid", "surface_topography", "cmb_topography", "surface_velocity"):
        assert np.all(np.isfinite(np.asarray(getattr(out["fields"], name))))
    a = np.asarray(sh.convert(out["fields"].geoid, "real_4pi", "real_orthonormal"))
    f = load_fortran("hager4_fsfs_lebedev")
    b = np.stack([f["sph_geoid_C"], f["sph_geoid_S"]])
    corr = []
    for l in range(2, 51):
        x = np.concatenate([a[0, l, : l + 1], a[1, l, 1: l + 1]])
        y = np.concatenate([b[0, l, : l + 1], b[1, l, 1: l + 1]])
        corr.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))
    print(f"E3: smallest correlation {min(corr):.4f} at l = {2 + int(np.argmin(corr))}, median "
          f"{np.median(corr):.4f}")
    assert min(corr) >= CORR_MIN


# ---------------------------------------------------------------------------
# E4, E4b, E4c, E9: the gradient example.
# ---------------------------------------------------------------------------


def _check_gradient(out, P, steps):
    """E4 and E9: per component against the central differences, the radius signal, the
    distance of the base radius to the input nodes, and the 1.01 control."""
    g, fd = np.array(out["gradient"]), np.array(out["fd"])
    rel = np.abs(g - fd) / np.abs(g)
    print(f"gradient: relative differences {rel}")
    assert np.all(rel <= FD_RTOL), rel
    k = out["names"].index("radius[0]")
    assert g[k] != 0.0
    assert abs(g[k]) * steps[1] >= RADIUS_SIGNAL_MIN * abs(out["value"])
    r_in = np.asarray(P.layers.input_profile.radius)
    assert np.min(np.abs(r_in - out["base_radius_m"])) > steps[1]
    # The 1.01 control: the radius component times 1.01 must fail the check.
    assert abs(1.01 * g[k] - fd[k]) > FD_RTOL * abs(1.01 * g[k])


@pytest.fixture(scope="module")
def gradient_reduced(tmp_path_factory):
    """``run()`` of the gradient example on the reduced table at ``lmax = 6``."""
    gl = load_example("gradient_layers")
    args = gl.parse_args(["--reduced", "--lmax", "6", "--out", str(tmp_path_factory.mktemp("g") / "g.json")])
    return gl.run(args), gl.build(args), tuple(args.fd_steps)


def test_gradient_example(gradient_reduced):
    """E4: the reduced gradient example at ``Y_REF``, 5750 km."""
    out, P, steps = gradient_reduced
    _check_gradient(out, P, steps)


def test_gradient_example_ridge(gradient_reduced):
    """E4b: the sum of the four log-viscosity derivatives is zero to rounding."""
    out = gradient_reduced[0]
    assert abs(out["ridge"]["sum"]) <= RIDGE_TOL * out["ridge"]["norm"]


def test_gradient_example_numpy(gradient_reduced):
    """E4c: the NumPy front gives the value and the gradient of the JAX call."""
    out = gradient_reduced[0]
    assert abs(out["numpy_value"] - out["value"]) <= NUMPY_RTOL * abs(out["value"])
    g, ng = np.array(out["gradient"]), np.array(out["numpy_gradient"])
    assert np.max(np.abs(ng - g)) <= NUMPY_RTOL * np.max(np.abs(g))


def test_gradient_example_full(tmp_path):
    """E9: the gradient example on the 137-node table at ``lmax = 20``."""
    gl = load_example("gradient_layers")
    args = gl.parse_args(["--out", str(tmp_path / "g.json")])
    out = gl.run(args)
    _check_gradient(out, gl.build(args), tuple(args.fd_steps))
    assert abs(out["ridge"]["sum"]) <= RIDGE_TOL * out["ridge"]["norm"]


# ---------------------------------------------------------------------------
# The HMC example: shared smoke problem.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke():
    """The smoke problem (``--smoke``), its MAP continuation (gate asserted) and the
    metric with the target weights at the MAP."""
    hmc = load_example("hmc_layers")
    P = hmc.build(hmc.parse_args(["--smoke"]))
    res = hmc.find_map_continuation(P)
    hmc.check_map(res, P)
    M = hmc.gauss_newton_metric(res.z, P.density, P.data, P.weights, P)
    return hmc, P, res, M


def _energy(U, p, Minv):
    """``H = U + p M^-1 p / 2`` in float64 numpy."""
    p = np.asarray(p)
    return float(U) + 0.5 * float(p @ Minv @ p)


def test_hmc_energy_error_order(smoke):
    """E5: at the smoke MAP the energy error of the example's trajectory falls by about 4
    when the step halves at the same total time (second-order leapfrog)."""
    hmc, P, res, M = smoke
    C = np.linalg.cholesky(M)
    Minv = np.linalg.inv(M)
    xi = np.asarray(jax.random.normal(jax.random.PRNGKey(0), (5,), dtype=jnp.float64))
    p = jnp.asarray(0.1 * C @ xi)
    z0 = jnp.asarray(res.z)
    U0, g0 = P.value_and_grad(z0, P.density, P.data, P.weights)
    H0 = _energy(U0, p, Minv)
    dH = {}
    for eps, n in ((0.05, 10), (0.025, 20)):
        tr = hmc.make_trajectory(P, M, n)
        z1, p1, U1, g1, _, _ = tr(z0, g0, p, eps, P.density, P.data, P.weights)
        dH[eps] = _energy(U1, p1, Minv) - H0
    ratio = abs(dH[0.05]) / abs(dH[0.025])
    print(f"E5: dH {dH}, ratio {ratio:.3f}")
    assert abs(dH[0.025]) > DH_RESOLUTION * EPS64 * abs(float(U0))
    assert ORDER_RATIO_BAND[0] <= ratio <= ORDER_RATIO_BAND[1]


def test_hmc_energy_error_order_control(smoke):
    """E5b: the same with the gradient times 1.01 (also the start gradient): the energy
    error no longer falls as the step squared."""
    hmc, P, res, M = smoke
    C = np.linalg.cholesky(M)
    Minv = np.linalg.inv(M)
    xi = np.asarray(jax.random.normal(jax.random.PRNGKey(0), (5,), dtype=jnp.float64))
    p = jnp.asarray(0.1 * C @ xi)
    z0 = jnp.asarray(res.z)
    U0, g0 = P.value_and_grad(z0, P.density, P.data, P.weights)
    H0 = _energy(U0, p, Minv)
    vg = jax.value_and_grad(lambda z, d, o, w: hmc.potential(z, d, o, w, P))

    def biased(z, d, o, w):
        U, g = vg(z, d, o, w)
        return U, 1.01 * g
    dH, dHb = {}, {}
    for eps, n in ((0.05, 10), (0.025, 20)):
        z1, p1, U1, _, _, _ = hmc.make_trajectory(P, M, n)(z0, g0, p, eps, P.density, P.data, P.weights)
        dH[eps] = _energy(U1, p1, Minv) - H0
        z1, p1, U1, _, _, _ = hmc.make_trajectory(P, M, n, value_and_grad=biased)(
            z0, 1.01 * g0, p, eps, P.density, P.data, P.weights)
        dHb[eps] = _energy(U1, p1, Minv) - H0
    ratio_b = abs(dHb[0.05]) / abs(dHb[0.025])
    print(f"E5b: biased dH {dHb}, ratio {ratio_b:.3f}, |dH_b| / |dH| at 0.025: {abs(dHb[0.025] / dH[0.025]):.1f}")
    assert ratio_b < ORDER_RATIO_BAND[0] or abs(dHb[0.025]) > BIAS_FACTOR * abs(dH[0.025])


def test_hmc_smoke_main(tmp_path):
    """E6: ``main(["--smoke"])``: the summary contract, a chain that moves, finite and
    bounded energy errors, and the state of the host loop (the carried ``U`` equals the
    potential at each sample, and the start momenta differ)."""
    hmc = load_example("hmc_layers")
    out = tmp_path / "s.json"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        summary = hmc.main(["--smoke", "--out", str(out)])
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)], [str(w.message) for w in rec]
    data = strict_json(out.read_text())
    z = np.array(data["samples_z"])
    assert z.shape == (8, 5) and np.all(np.isfinite(z))
    prob = np.array(data["prob"])
    assert np.all((prob >= 0) & (prob <= 1))
    P = hmc.build(hmc.parse_args(["--smoke"]))
    lo, hi = P.layers.radius_bounds[0]
    r = np.array(data["samples_radius_km"]) * KM
    assert np.all((r >= lo) & (r <= hi))
    s = data["settings"]
    assert s["motion"] == "cut" and s["range_km"] == [5466.875, 5970.0]
    m = data["map"]
    assert m["gate_passed"] and m["gnorm"] <= hmc.GTOL_GATE and m["status"] in ("converged", "stalled")
    assert m["U_map"] <= m["U_truth"] and m["cond_metric"] <= hmc.METRIC_COND_MAX
    assert len(m["iterations"]) == 5
    assert data["accepted"] >= 1
    assert np.any(np.any(z != np.array(m["z"]), axis=1))
    dH = data["dH"]
    assert all(v is not None for v in dH) and data["n_nonfinite_dH"] == 0
    assert max(abs(v) for v in dH) <= SMOKE_DH_MAX
    assert np.all(np.array(data["r_min_km"]) <= np.array(data["r_max_km"]))
    # The state of the host loop: the potential that the chain carries against the
    # compiled value at each sample (unbatched, the call of the chain).
    for zi, Ui in zip(z, data["U"]):
        U, _ = P.value_and_grad(jnp.asarray(zi), P.density, P.data, P.weights)
        assert abs(Ui - float(U)) <= STATE_RTOL * abs(float(U))
    p0 = np.array(data["p0"])
    assert p0.shape == (8, 5)
    for i in range(8):
        for j in range(i + 1, 8):
            assert not np.array_equal(p0[i], p0[j]), (i, j)
    assert summary["accepted"] == data["accepted"]


def test_metropolis_rule(smoke):
    """E6b (plan 6.1.1): the Metropolis step and the returned state on 512 proposals at
    step 0.5 from the smoke MAP, with the energy difference computed again in the test."""
    hmc, P, res, M = smoke
    w = P.weights
    tr = hmc.make_transition(P, M, n_leapfrog=3)
    zM = jnp.asarray(res.z)
    U0, g0 = P.value_and_grad(zM, P.density, P.data, w)
    U0f, g0n, zMn = float(U0), np.asarray(g0), np.asarray(zM)
    Minv = np.linalg.inv(M)
    C = np.linalg.cholesky(M)
    keys = jax.random.split(jax.random.PRNGKey(METRO_SEED), METRO_N)
    rows = []
    for key in keys:
        (z, U, g), info = tr(zM, U0, g0, key, METRO_STEP, P.density, P.data, w)
        info = {k: np.asarray(v) for k, v in info.items()}
        z1 = info["z1"]
        if np.all(np.isfinite(z1)):
            Up, gp = P.value_and_grad(jnp.asarray(z1), P.density, P.data, w)
            Up, gp = float(Up), np.asarray(gp)
        else:
            Up, gp = np.nan, np.full(5, np.nan)
        H0 = _energy(U0f, info["p0"], Minv)
        H1p = Up + 0.5 * float(info["p1"] @ Minv @ info["p1"])
        rows.append(dict(z=np.asarray(z), U=float(U), g=np.asarray(g), info=info, dHp=H1p - H0, H0=H0, H1p=H1p,
                         Up=Up, gp=gp))
    dH = np.array([r["info"]["dH"] for r in rows], dtype=float)
    dHp = np.array([r["dHp"] for r in rows], dtype=float)
    acc = np.array([bool(r["info"]["accept"]) for r in rows])
    # Condition 1: the returned dH against the dH of the test where one is at most 40.
    worst = 0.0
    for r, a, b in zip(rows, dH, dHp):
        if min(np.inf if not np.isfinite(a) else a, np.inf if not np.isfinite(b) else b) <= DH_CLASS_C:
            scale = max(1.0, abs(r["H0"]), abs(r["H1p"]))
            worst = max(worst, abs(a - b) / scale)
            assert abs(a - b) <= ENERGY_RTOL * scale, (a, b)
        else:
            assert (not np.isfinite(a) or a > DH_CLASS_C) and (not np.isfinite(b) or b > DH_CLASS_C), (a, b)
    # Condition 2: the classes by the dH of the test.
    A = np.isfinite(dHp) & (dHp <= 0)
    B = np.isfinite(dHp) & (dHp > 0) & (dHp <= DH_CLASS_C)
    Cc = ~(A | B)
    nA, nB, nC = int(A.sum()), int(B.sum()), int(Cc.sum())
    assert nA >= MIN_A and nB >= MIN_B and nC >= MIN_C, (nA, nB, nC)
    assert int((acc & B).sum()) >= MIN_ACC and int((~acc & B).sum()) >= MIN_REJ

    def rule_checks(accept):
        """Conditions 3 (a), 4 (b) and 5 (c) for a vector of accept flags."""
        a = not np.any(A & ~accept)
        b = not np.any(Cc & accept)
        p = np.exp(-dHp[B])
        t = (np.sum(accept[B]) - np.sum(p)) / np.sqrt(np.sum(p * (1 - p)))
        return a, b, float(t)
    a_ok, b_ok, t = rule_checks(acc)
    assert a_ok and b_ok and abs(t) <= METRO_T_MAX, (a_ok, b_ok, t)
    # Conditions 6 and 7: the returned state.
    for r, a in zip(rows, acc):
        if a:
            assert np.array_equal(r["z"], r["info"]["z1"])
            assert abs(r["U"] - r["Up"]) <= STATE_RTOL * abs(r["Up"])
            assert np.linalg.norm(r["g"] - r["gp"]) <= STATE_RTOL * np.linalg.norm(r["gp"])
        else:
            assert np.array_equal(r["z"], zMn) and r["U"] == U0f and np.array_equal(r["g"], g0n)
    # Condition 8: the returned probability.
    for r, d in zip(rows, dH):
        expect = min(1.0, float(np.exp(-d))) if np.isfinite(d) else 0.0
        assert abs(float(r["info"]["prob"]) - expect) <= PROB_ATOL
    # Condition 9: the radius range of every finite path holds its two end points.
    rM = float(hmc.radius_of(zM[4], P))
    for r in rows:
        if np.all(np.isfinite(r["info"]["z1"])):
            r1 = float(hmc.radius_of(r["info"]["z1"][4], P))
            assert r["info"]["r_min"] <= min(rM, r1) + RANGE_ATOL
            assert r["info"]["r_max"] >= max(rM, r1) - RANGE_ATOL
    # Condition 10: a rule that accepts every finite proposal fails (b) and (c).
    a_n, b_n, t_n = rule_checks(np.isfinite(dHp))
    assert not (b_n and abs(t_n) <= METRO_T_MAX), (b_n, t_n)
    # Condition 11: the whitened start momenta have the identity covariance.
    W = np.linalg.solve(C, np.stack([r["info"]["p0"] for r in rows]).T).T     # (512, 5)
    S = W.T @ W / METRO_N
    cov = float(np.max(np.abs(S - np.eye(5))))
    print(f"E6b: classes A {nA}, B {nB}, C {nC}; in B accepted {int((acc & B).sum())}, rejected "
          f"{int((~acc & B).sum())}; t {t:.2f}; condition 1 {worst:.2e}; max |S - I| {cov:.3f}; control t {t_n:.1f}")
    assert cov <= MOMENTUM_COV_MAX


@pytest.mark.slow
def test_hmc_default_run(tmp_path):
    """E8: the default run A (1000 samples, lmax 20, step 0.5): every gate of G5 of the
    plan except the wall time, which the report records."""
    hmc = load_example("hmc_layers")
    s = hmc.main(["--out", str(tmp_path / "a.json")])
    strict_json((tmp_path / "a.json").read_text())
    P = hmc.build(hmc.parse_args([]))
    lo, hi = P.layers.ranges[1].low, P.layers.ranges[1].high
    assert lo < hmc.DENSITY_JUMP < hi and P.layers.motion == "cut" and s["settings"]["motion"] == "cut"
    assert P.layers.start_radii[0] >= 0.5 * (lo + hi)
    m = s["map"]
    assert m["gnorm"] <= hmc.GTOL_GATE and m["status"] in ("converged", "stalled")
    assert m["U_map"] <= m["U_truth"] and m["cond_metric"] <= hmc.METRIC_COND_MAX
    assert s["mean_prob"] >= MEAN_PROB_MIN
    assert s["n_nonfinite_dH"] == 0 and s["max_dH"] <= DH_MAX_RUN
    mean, sd = np.array(s["posterior_mean"]), np.array(s["posterior_sd"])
    assert np.all(np.abs(mean - np.array(s["z_true"])) <= MEAN_SD_MAX * sd)
    ratio = np.array(s["sd_over_laplace"])
    assert np.all((ratio >= SD_RATIO_BAND[0]) & (ratio <= SD_RATIO_BAND[1])), ratio
    assert s["crossings_5700km"] >= CROSSINGS_MIN


def test_metric_is_spd_and_matches_hessian():
    """E10: at the noise-free truth of the smoke problem the Gauss-Newton metric is
    symmetric, positive definite and equal to ``jax.hessian(U)`` (the residual is zero)."""
    hmc = load_example("hmc_layers")
    P = hmc.build(hmc.parse_args(["--smoke"]), noise_free=True)
    z = jnp.asarray(P.z_true)
    M = hmc.gauss_newton_metric(P.z_true, P.density, P.data, P.weights, P)
    assert np.array_equal(M, M.T) or np.max(np.abs(M - M.T)) <= METRIC_RTOL * np.max(np.abs(M))
    np.linalg.cholesky(M)
    H = np.asarray(jax.hessian(lambda q: hmc.potential(q, P.density, P.data, P.weights, P))(z))
    rel = np.linalg.norm(M - H) / np.linalg.norm(H)
    print(f"E10: |M - H| / |H| = {rel:.2e}")
    assert rel <= METRIC_RTOL


def test_metric_uses_its_weights(smoke):
    """E10b: ``M(4 w) - M(0) = 4 (M(w) - M(0))`` at the smoke MAP (the misfit part is
    linear in the weights, the prior part does not depend on them)."""
    hmc, P, res, _ = smoke
    w = P.weights

    def M(weights):
        return hmc.gauss_newton_metric(res.z, P.density, P.data, weights, P)
    M0, M1, M4 = M(0.0 * w), M(w), M(4.0 * w)
    lhs, rhs = M4 - M0, 4.0 * (M1 - M0)
    assert np.linalg.norm(lhs - rhs) <= METRIC_RTOL * np.linalg.norm(rhs)
    assert np.linalg.norm(M1 - M0) > 0.1 * np.linalg.norm(M0)


def test_map_gate(smoke):
    """E11: ``check_map`` refuses each failed condition and passes the smoke
    continuation, whose MAP equals the end of a search from the truth."""
    import dataclasses
    hmc, P, res, _ = smoke
    hmc.check_map(res, P)
    for change in (dict(gnorm=2e-5), dict(status="max_iter"), dict(U=res.U_truth + 1e-6)):
        with pytest.raises(hmc.MapGateError):
            hmc.check_map(dataclasses.replace(res, **change), P)
    st = hmc.find_map(P.z_true, P, P.weights)
    assert st.status == "converged"
    assert abs(st.U - res.U) <= MAP_U_RTOL * abs(res.U), (st.U, res.U)


def test_map_gate_rejects_second_mode():
    """E11b: on the full problem one stage from the isoviscous start ends at the second
    mode (``U(MAP) > U(truth)``) and the gate refuses it; the continuation passes."""
    hmc = load_example("hmc_layers")
    P = hmc.build(hmc.parse_args([]))
    one = hmc.find_map_continuation(P, ladder=(), max_iter=200)
    r = float(hmc.radius_of(one.z[4], P)) / KM
    print(f"E11b: one stage: {one.iterations} iterations, U {one.U:.3f} (truth {one.U_truth:.3f}), radius {r:.1f} km")
    assert one.U > one.U_truth
    with pytest.raises(hmc.MapGateError):
        hmc.check_map(one, P)
    cont = hmc.find_map_continuation(P)
    print(f"E11b: continuation: {cont.iterations}, U {cont.U:.3f}")
    hmc.check_map(cont, P)


def test_effective_sample_size_ar1():
    """E12: Sokal's ESS on AR(1) series against ``n / tau``, ``tau = (1 + a) / (1 - a)``;
    a constant series gives 1.0."""
    hmc = load_example("hmc_layers")
    n = 100000
    rng = np.random.default_rng(0)
    e = rng.standard_normal(n)
    for a in (0.0, 0.5):
        x = np.empty(n)
        x[0] = e[0]
        for t in range(1, n):
            x[t] = a * x[t - 1] + np.sqrt(1 - a * a) * e[t]
        tau = (1 + a) / (1 - a)
        ess = hmc.effective_sample_size(x)
        print(f"E12: a = {a}: ESS {ess:.0f} against {n / tau:.0f}")
        assert abs(ess / (n / tau) - 1) <= ESS_RTOL
    assert hmc.effective_sample_size(np.full(50, 3.0)) == 1.0


# ---------------------------------------------------------------------------
# E7: the command lines.
# ---------------------------------------------------------------------------


def _run_script(args, cwd):
    """Run a Python script in a subprocess with the package of this test process first
    on the path; returns the completed process."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable] + args, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("which", ["forward", "gradient", "hmc"])
def test_examples_cli(which, e2_files, tmp_path):
    """E7: each script in a subprocess with its smoke settings: exit code 0 and the
    output of ``--out``."""
    if which == "forward":
        out = tmp_path / "fwd"
        proc = _run_script([str(EXAMPLES / "forward_legacy_density.py"), "--sph", str(e2_files["sph"]),
                            "--out", str(out)], tmp_path)
        assert proc.returncode == 0, proc.stderr
        for f in ("SPH_geoid", "SPH_surftopo", "SPH_cmbtopo", "SPH_surfvel", "kernels_fields.npz"):
            assert (out / f).is_file(), f
    else:
        out = tmp_path / "s.json"
        script, extra = {"gradient": ("gradient_layers.py", ["--reduced", "--lmax", "6"]),
                         "hmc": ("hmc_layers.py", ["--smoke"])}[which]
        proc = _run_script([str(EXAMPLES / script)] + extra + ["--out", str(out)], tmp_path)
        assert proc.returncode == 0, proc.stderr
        strict_json(out.read_text())


def test_forward_example_missing_file(tmp_path):
    """The forward script stops with a message that names ``--sph`` when the density file
    is absent (the default file is in the untracked ``legacy/``)."""
    fwd = load_example("forward_legacy_density")
    with pytest.raises(SystemExit) as err:
        fwd.main(["--sph", str(tmp_path / "absent.sph"), "--out", str(tmp_path)])
    assert "--sph" in str(err.value)
