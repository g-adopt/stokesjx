"""The mpmath oracle with the legacy switch values against CODE-A (gates G1, G1b, G1t,
G3 historical, G16 on the golden data).

Reference for G1: the Fortran golden output (tests/golden/fortran). The oracle under
``legacy_setup`` uses the local potential rows (F5), so the Fortran checks the
matrix, the jumps, the boundary conditions, the local potential conditions and the
output formulas of NOTES/docs/equations.md at once.
"""

import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

import cases
import reference_mpmath as ro

# G1 tolerance: the Fortran solves one 8x8 system per degree in 128-bit reals
# (2^-113 = 9.6e-35), and the single product of propagators loses up to about
# (R / r_c)^(2l+3) of it, which bounds the loss by 100 2^-113 (R / r_c)^(2l+3)
# (1.1e-5 at l = 50). The measured max-normalised error e_l is at most 1.6e-16 at
# every degree 1..50 for all seven runs and G1b (one float64 rounding of the
# output; report 4.1), so the realised loss stays below float64 rounding. The gate
# is the flat 1e-13: 600 times the measured value, and small enough to fail for an
# error of the float32 literals of constants.h (1e-9 to 5e-8) at every degree.
G1_TOL = 1e-13


def g1_tolerance(l: int) -> float:
    """The G1 tolerance at degree ``l`` (flat, see ``G1_TOL``)."""
    return G1_TOL


# G1t: the transcription and the switched path are two mp computations of the same
# numbers at 60 digits or more; cause of the difference: rounding at that precision
# amplified by the conditioning of the 8x8 and 3x3 systems.
G1T_TOL = 1e-25
# The historical test_4_ishan kernels: Summary rounds the radius to 1 m and the
# density to 0.01 kg/m^3 (measured 8.6e-6, review r2 section 2.1).
HISTORICAL_TOL = 2e-5
# End-node values on the Fortran output: exact relations, float64 kernels.
END_NODE_RTOL = 1e-10

RUNS = [r for r, s in cases.GOLDEN_RUNS.items() if not s.get("fields_only")]
KERNELS = ro.FORTRAN_OUTPUTS


def e_l(test: np.ndarray, ref: np.ndarray) -> float:
    """Max-normalised error of one kernel at one degree (CONVENTIONS section 10)."""
    mx = np.max(np.abs(ref))
    return float(np.max(np.abs(test - ref)) / mx)


def nodewise(test, ref):
    """Node-wise relative error over the nodes with |K_ref| > 1e-8 max |K_ref|
    (diagnostic only)."""
    mx = np.max(np.abs(ref))
    m = np.abs(ref) > 1e-8 * mx
    return float(np.max(np.abs(test[m] - ref[m]) / np.abs(ref[m]))) if np.any(m) else 0.0


def _compare(run, l, oracle_row, fortran):
    """e_l for the five kernels; at l = 1 the geoid and the free-air kernel must be
    exact zeros in both arrays and are left out."""
    out = {}
    for k in KERNELS:
        test = oracle_row[k]
        ref = fortran[k][l - 1]
        if l == 1 and k in ("geoid", "gravity"):
            assert np.all(test == 0) and np.all(ref == 0), (run, k)
            continue
        out[k] = e_l(test, ref)
    return out


@pytest.mark.parametrize("run", RUNS)
def test_g1_stored(run, load_fortran, load_oracle):
    """G1 on the stored legacy-switch oracle at l in {1, 2, 3, 10, 20, 30, 40, 50}."""
    F = load_fortran(run)
    O = load_oracle(f"legacy_{run}")
    for l in cases.LEGACY_DEGREES:
        row = {k: O[f"l{l}/{k}"] for k in KERNELS}
        errs = _compare(run, l, row, F)
        for k, e in errs.items():
            assert e <= g1_tolerance(l), (run, l, k, e, g1_tolerance(l))


@pytest.mark.parametrize("run", RUNS)
def test_g16_end_nodes_fortran(run, load_fortran):
    """F19 on the Fortran output: a sheet at an end node is compensated statically."""
    F = load_fortran(run)
    rho_cmb = F["rho0"][0]
    for l in range(1, 51):
        assert abs(F["surftopo"][l - 1, -1] * 4500.0 + 1) <= END_NODE_RTOL
        assert abs(F["cmbtopo"][l - 1, 0] * (rho_cmb - 9900.0) - 1) <= END_NODE_RTOL
        for k in ("geoid", "surfvel"):
            scale = np.max(np.abs(F[k][l - 1])) or 1.0
            assert abs(F[k][l - 1, 0]) <= END_NODE_RTOL * scale
            assert abs(F[k][l - 1, -1]) <= END_NODE_RTOL * scale


def _mp_rel(a, b, dps=80):
    with mp.workdps(dps):
        num = max(abs(x - y) for x, y in zip(a, b))
        den = max(abs(y) for y in b)
        return num / den if den != 0 else num


def _g1t(run, degrees):
    spec = cases.GOLDEN_RUNS[run]
    table = cases.golden_table(run)
    physics, disc = ro.legacy_setup(spec["bdry"], spec["bdry_bot"], spec["kappaindex"])
    sw = ro.kernels(table, physics, disc, degrees, return_mp=True)
    tr = ro.transcription_kernels(table, spec["bdry"], spec["bdry_bot"], spec["kappaindex"], degrees,
                                  propagator="expm", return_mp=True)
    worst = 0
    for a_, l in enumerate(degrees):
        for k in KERNELS:
            a, b = list(tr[k][a_]), list(sw[k][a_])
            if all(v == 0 for v in b):
                assert all(v == 0 for v in a)
                continue
            worst = max(worst, _mp_rel(a, b))
    return worst


@pytest.mark.parametrize("run", ["iso_fsfs", "hager4_fsfs_tala"])
def test_g1t_transcription_quick(run):
    """G1t: the line-by-line transcription of the 8x8 system equals the switched
    path (local rows, the same legacy model) at l in {1, 2}."""
    assert _g1t(run, (1, 2)) <= G1T_TOL


@pytest.mark.slow
@pytest.mark.parametrize("run", RUNS)
def test_g1t_transcription_full(run):
    assert _g1t(run, (1, 2, 10, 50)) <= G1T_TOL


@pytest.mark.legacy
def test_historical_kernels_stored(load_oracle, legacy_dir):
    """The stored legacy-switch oracle on the inputs of test_4_ishan (257 nodes)
    against the historical kernel files, at l in {1, 2, 10, 30, 50}."""
    O = load_oracle("legacy_ishan257")
    for l in cases.ISHAN_DEGREES:
        for k in KERNELS:
            ref = np.loadtxt(legacy_dir / "test_4_ishan/KERNELS" / f"{k}{l:03d}")[:, 1]
            test = O[f"l{l}/{k}"]
            if l == 1 and k in ("geoid", "gravity"):
                assert np.all(test == 0) and np.all(ref == 0)
                continue
            assert e_l(test, ref) <= HISTORICAL_TOL, (l, k, e_l(test, ref))


@pytest.mark.slow
@pytest.mark.legacy
def test_historical_kernels_live(legacy_dir, load_oracle):
    """The same with a live recomputation, which must equal the stored values."""
    table = cases.ishan_table(legacy_dir.parent / "CODE-A")
    physics, disc = ro.legacy_setup(1, 1, 0)
    k = ro.kernels(table, physics, disc, cases.ISHAN_DEGREES)
    O = load_oracle("legacy_ishan257")
    for a_, l in enumerate(cases.ISHAN_DEGREES):
        for name in KERNELS:
            np.testing.assert_array_equal(k[name][a_], O[f"l{l}/{name}"])


def _live_g1(args):
    run, l = args
    spec = cases.GOLDEN_RUNS[run]
    physics, disc = ro.legacy_setup(spec["bdry"], spec["bdry_bot"], spec["kappaindex"])
    k = ro.kernels(cases.golden_table(run), physics, disc, (l,))
    return run, l, {n: k[n][0] for n in KERNELS}


@pytest.mark.slow
@pytest.mark.parametrize("run", RUNS)
def test_g1_all_degrees_live(run, load_fortran):
    """G1 at all 50 degrees with a live mpmath computation."""
    F = load_fortran(run)
    with ProcessPoolExecutor(max_workers=8) as ex:
        for run_, l, row in ex.map(_live_g1, [(run, l) for l in range(1, 51)]):
            for k, e in _compare(run_, l, row, F).items():
                assert e <= g1_tolerance(l), (run_, l, k, e)


@pytest.mark.slow
def test_g1b_sylvester(load_fortran):
    """G1b: the TALA run with the transcription in Sylvester mode (eigenvalues from
    dgeev, merged within 1e-3, as propmat) and with the switched path (expm). Both
    must meet G1; the errors are printed for the report."""
    run = "hager4_fsfs_tala"
    F = load_fortran(run)
    table = cases.golden_table(run)
    degrees = (1, 2, 10, 30, 50)
    syl = ro.transcription_kernels(table, 1, 1, 1, degrees, propagator="sylvester")
    physics, disc = ro.legacy_setup(1, 1, 1)
    sw = ro.kernels(table, physics, disc, degrees)
    for a_, l in enumerate(degrees):
        es = _compare(run, l, {k: syl[k][a_] for k in KERNELS}, F)
        ee = _compare(run, l, {k: sw[k][a_] for k in KERNELS}, F)
        dd = max(e_l(syl[k][a_], sw[k][a_]) for k in KERNELS if np.any(sw[k][a_] != 0))
        print(f"G1b l={l}: sylvester {max(es.values()):.2e}, expm {max(ee.values()):.2e}, "
              f"sylvester-expm {dd:.2e}")
        assert max(ee.values()) <= g1_tolerance(l)
