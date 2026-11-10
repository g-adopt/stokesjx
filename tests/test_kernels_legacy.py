"""The legacy path of the low layer against the Fortran CODE-A and the stored legacy
oracle (plan 7.5, gate G3).

The legacy path is ``propagate.solve`` with the test-only switches of
``Discretisation`` (upper-node material, CODE-A's degree-1 geoid), the legacy
constants (``Units(legacy_constants=True)``: float32 ``G`` and ``pi``), a uniform
background ``g = 10 m/s^2`` and the topography density ``bgrho = 4500 kg/m^3`` as an
input of the profile (F3). References: the golden Fortran kernels
(tests/golden/fortran, CODE-A in 128-bit reals, exact to 1.6e-16 at every degree, chunk
1 G1) and the stored legacy-switch oracle (tests/golden/oracle/legacy_<run>.npz).
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import cases  # noqa: E402
import reference_mpmath as ro  # noqa: E402
from adapters import assert_rows_close, profile_from_table, row_errors  # noqa: E402

import stokesjx as sj  # noqa: E402
from stokesjx import propagate  # noqa: E402

# The legacy kernels against the Fortran and the stored legacy oracle, e_l. The Fortran
# equals the exact legacy kernels to 1.6e-16 (chunk 1, G1), so the error against the
# Fortran is the error against the exact answer; the requirement is 1e-10 and the test
# constant 1e-12 as in tests/test_kernels_oracle.py. PLAN stage 3 asked for rtol 1e-6;
# chunk 1 makes the tighter value possible. Measured with the package: at most 1.8e-13
# (smooth_fsfs, geoid at l = 50), at most 2.7e-14 on iso_fsfs and hager4_fsfs. Cause:
# the rounding of the sweeps.
FORTRAN_TOL = 1e-12

RUNS = ("iso_fsfs", "hager4_fsfs", "smooth_fsfs")
PAIRS = (("geoid", "geoid"), ("surftopo", "surface_topography"), ("cmbtopo", "cmb_topography"),
         ("surfvel", "surface_velocity"))
LEGACY_CONFIG = sj.StokesConfig(lmax=50, formulation=sj.Formulation(background=sj.Uniform(10.0)),
                                units=sj.Units(legacy_constants=True))
LEGACY_DISC = propagate.Discretisation(material="upper_node", degree_one_geoid_zero=True)


@functools.lru_cache(maxsize=None)
def _legacy_solve(table):
    p = profile_from_table(table, surface_density=4500.0)
    K = propagate.solve(p, LEGACY_CONFIG, LEGACY_DISC)
    out = {o: np.asarray(getattr(K, o)) for _, o in PAIRS}
    out["imbalance"] = np.asarray(K.degree_one_imbalance)
    out["frame"] = K.degree_one_frame
    return out


@pytest.mark.parametrize("run", RUNS)
def test_legacy_against_fortran(run, load_fortran):
    """All 50 degrees and four kernels against the golden Fortran."""
    sol = _legacy_solve(cases.golden_table(run))
    F = load_fortran(run)
    assert sol["frame"] is sj.DegreeOneFrame.LEGACY_GEOID_ZERO
    for fk, o in PAIRS:
        e = assert_rows_close(sol[o], F[fk], FORTRAN_TOL, f"{run} {fk}")
        worst_l = int(np.argmax(np.where(np.isfinite(row_errors(sol[o], F[fk])), row_errors(sol[o], F[fk]), 0))) + 1
        print(f"{run} {fk}: max e_l {e:.2e} at l = {worst_l}")
    # CODE-A's degree-1 frame: the geoid is an exact zero, the topographies are raw, and
    # there is no applied field (exact zero imbalance).
    assert np.all(sol["geoid"][0] == 0) and np.all(F["geoid"][0] == 0)
    assert np.all(sol["imbalance"] == 0)


@pytest.mark.parametrize("run", RUNS)
def test_legacy_against_stored_oracle(run, load_oracle):
    sol = _legacy_solve(cases.golden_table(run))
    O = load_oracle(f"legacy_{run}")
    for l in cases.LEGACY_DEGREES:
        for fk, o in PAIRS:
            assert_rows_close(sol[o][l - 1], O[f"l{l}/{fk}"], FORTRAN_TOL, f"{run} l={l} {fk}")


def test_legacy_ishan257(legacy_dir, load_oracle):
    """The historical test_4_ishan table (257 nodes) against its stored legacy oracle;
    skipped without legacy/CODE-A."""
    table = cases.ishan_table(legacy_dir)
    sol = _legacy_solve(table)
    O = load_oracle("legacy_ishan257")
    for l in cases.ISHAN_DEGREES:
        for fk, o in PAIRS:
            assert_rows_close(sol[o][l - 1], O[f"l{l}/{fk}"], FORTRAN_TOL, f"ishan l={l} {fk}")


def test_legacy_constants_reach_the_solver():
    """The legacy switch changes the kernels by the float32 rounding of G and pi (about
    3e-8 relative), so a solve with the default constants does not pass FORTRAN_TOL."""
    table = cases.golden_table("hager4_fsfs")
    p = profile_from_table(table, surface_density=4500.0)
    cfg = sj.StokesConfig(lmax=2, formulation=sj.Formulation(background=sj.Uniform(10.0)))
    K = propagate.solve(p, cfg, LEGACY_DISC)
    ref = _legacy_solve(table)["geoid"][1]
    assert row_errors(np.asarray(K.geoid)[1], ref)[0] > 1e-9
    assert ro.LEGACY_G != ro.mp.mpf(sj.Units().G)
