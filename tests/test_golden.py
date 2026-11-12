"""The tracked golden data: manifest, shapes, and the consistency of the Fortran fields
with the Fortran kernels.

Reference: the Fortran output itself (tests/golden/fortran), reduced by
legacy/harness/run_golden.py.
"""

import hashlib
import json
import math
import shutil
import subprocess
import sys

import numpy as np
import pytest

import cases

# The SPHCO consistency: the field coefficient is a sum of 129 products in float64
# (fldcalc, calc.f:79-93), compiled with -ffast-math, against the same sum in numpy
# in another order. Cause of the difference: summation order. Relative to the field
# maximum per degree.
SPHCO_RTOL = 1e-12
# Regeneration with a toolchain that differs from the manifest: the quad-precision
# arithmetic keeps the float64 kernels to about 1e-17 (measured with the strict
# build, manifest diagnostics); 1e-12 is the plan's bound (G3).
REGEN_RTOL = 1e-12
GOLDEN_SIZE_LIMIT = 3.5e6   # bytes; see NOTES/TOLERANCES.md (plan 3.4 asked for 3 MB)

KERNELS = ("geoid", "surftopo", "cmbtopo", "surfvel", "gravity")
FIELDS = ("geoid", "surftopo", "cmbtopo", "surfvel")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_lists_every_npz(golden_dir, manifest):
    listed = {}
    for section in ("fortran", "oracle"):
        for name, rec in manifest[section].items():
            listed[rec["npz"]] = rec["npz_sha256"]
    present = {str(p.relative_to(golden_dir)) for p in golden_dir.rglob("*.npz")}
    assert present == set(listed)
    for rel, digest in listed.items():
        assert _sha(golden_dir / rel) == digest, rel


def test_golden_size(golden_dir):
    total = sum(p.stat().st_size for p in golden_dir.rglob("*") if p.is_file())
    assert total < GOLDEN_SIZE_LIMIT, total


@pytest.mark.parametrize("run", list(cases.GOLDEN_RUNS))
def test_fortran_arrays(run, load_fortran):
    d = load_fortran(run)
    spec = cases.GOLDEN_RUNS[run]
    assert d["radius"].shape == (129,) and d["radius"][0] == 3480000.0 and d["radius"][-1] == 6370000.0
    assert d["degrees"].tolist() == list(range(1, 51))
    if not spec.get("fields_only"):
        for k in KERNELS:
            assert d[k].shape == (50, 129)
            assert np.all(np.isfinite(d[k]))
        # CODE-A sets the degree-1 geoid to zero, and the free-air kernel has the
        # factor (l - 1) (driver.f:259-261, 282-284).
        assert np.all(d["geoid"][0] == 0) and np.all(d["gravity"][0] == 0)
    if spec.get("fields", True):
        for f in FIELDS:
            for cs in ("C", "S"):
                a = d[f"sph_{f}_{cs}"]
                assert a.shape == (51, 51) and np.all(np.isfinite(a))
                assert np.all(np.triu(a, 1) == 0)          # m <= l only
            assert np.all(d[f"sph_{f}_S"][:, 0] == 0)


def test_fortran_inputs_reproduce_from_cases(manifest):
    """The density and viscosity files of every run are the text that cases.py
    writes now (no input drifted since the golden data were made)."""
    synth = None
    for run, rec in manifest["fortran"].items():
        spec = cases.GOLDEN_RUNS[run]
        visc = cases.viscosity_file_text(cases.golden_eta_rel(spec["visc"]))
        assert hashlib.sha256(visc.encode()).hexdigest() == rec["inputs"]["viscosity_sha256"], run
        if spec["density"] == "synthetic":
            if synth is None:
                synth = cases.density_file_text(cases.LEBEDEV_RADIUS, cases.LEBEDEV_RHO0,
                                                cases.synthetic_coefficients(), nr=128)
            assert hashlib.sha256(synth.encode()).hexdigest() == rec["inputs"]["density_sha256"], run


# Largest distance, in units in the last place of the formula value, between a stored
# table entry and the formula evaluated with the maths library of the running platform.
# The table holds the macOS values of the golden runs; macOS and glibc differ by at most
# one ulp on these arguments, so two ulp leave a margin of one without letting a wrong
# entry (a different argument or formula) pass.
SYNTH_BASIS_ULP = 2


def test_synthetic_basis_matches_formula():
    """The stored cosines and Gaussian of the synthetic anomaly (cases.SYNTH_COS,
    cases.SYNTH_GAUSS) are the formula of cases.synthetic_density to within
    SYNTH_BASIS_ULP on this platform, so the table is the anomaly the docstring states
    and only the last bit can depend on where it was computed."""
    L = cases.SYNTH_LMAX
    assert cases.SYNTH_COS.shape == (L + 1, L + 1, 3, 2)
    assert cases.SYNTH_GAUSS.shape == (cases.LEBEDEV_NODES,)
    assert np.all(cases.SYNTH_COS[0] == 0.0)
    for l in range(1, L + 1):
        for m in range(l + 1):
            for k in range(3):
                for side in (0, 1):
                    f = cases.synth_cos_formula(l, m, k, bool(side))
                    t = float(cases.SYNTH_COS[l, m, k, side])
                    assert abs(t - f) <= SYNTH_BASIS_ULP * np.spacing(abs(f)), (l, m, k, side, t, f)
    for j, r in enumerate(cases.LEBEDEV_RADIUS):
        f = cases.synth_gauss_formula(r)
        t = float(cases.SYNTH_GAUSS[j])
        assert abs(t - f) <= SYNTH_BASIS_ULP * np.spacing(abs(f)), (j, t, f)


# Largest distance, in ulp of the formula value, between a stored smooth-profile value
# and the formula on the running platform. The formula is 10 ** (a + 0.7 tanh(x)): one
# ulp in tanh moves the exponent by about 0.7 ulp of 1 and the power turns an exponent
# error e into a relative error ln(10) e, so a few ulp separate two maths libraries.
# 8 ulp is far below any change of the profile itself.
SMOOTH_PROFILE_ULP = 8


def test_smooth_profile_matches_formula():
    """The stored smooth golden profile (cases.SMOOTH_ETA_REL_LEBEDEV) is the formula
    of cases.smooth_eta_rel to within SMOOTH_PROFILE_ULP on this platform."""
    for j, r in enumerate(cases.LEBEDEV_RADIUS):
        t = float(cases.SMOOTH_ETA_REL_LEBEDEV[j])
        f = float(cases.smooth_eta_rel(r))
        assert abs(t - f) <= SMOOTH_PROFILE_ULP * np.spacing(abs(f)), (j, t, f)


def _sqt():
    """CODE-A's sqrt(4 pi) with the float32 pi, evaluated in double (constants.h:14)."""
    return float(np.sqrt(4.0 * np.float64(np.float32(math.pi))))


def _laysize():
    """(R - r_c) / nr, computed in quad and rounded to double (calc.f:51)."""
    return float((6370000.0 - 3480000.0) / 128)


def _check_fields(d, kernels_from, coeffs):
    """sph_X[l, m] = sqt * laysize * sum_j K_X[l, j] rho_lm(r_j) for l = 1..50."""
    sqt, lay = _sqt(), _laysize()
    worst = 0.0
    for f in FIELDS:
        K = kernels_from[f]                                   # (50, 129)
        for ci, cs in enumerate(("C", "S")):
            pred = np.einsum("lj,jlm->lm", K, coeffs[:, 1:, :, ci]) * sqt * lay
            got = d[f"sph_{f}_{cs}"][1:]
            mx = np.maximum(np.max(np.abs(d[f"sph_{f}_C"][1:]), axis=1), np.max(np.abs(d[f"sph_{f}_S"][1:]), axis=1))
            scale = np.where(mx > 0, mx, 1.0)
            err = np.max(np.abs(pred - got), axis=1) / scale
            worst = max(worst, float(np.max(err)))
    return worst


@pytest.mark.parametrize("run", [r for r, s in cases.GOLDEN_RUNS.items()
                                 if s["density"] == "synthetic" and s.get("fields", True)])
def test_sphco_consistency(run, load_fortran):
    d = load_fortran(run)
    coeffs = cases.synthetic_coefficients()
    worst = _check_fields(d, d, coeffs)
    assert worst <= SPHCO_RTOL, worst


def _read_lebedev_coefficients(path):
    """Coefficients of LEBEDEV_S4RTS.sph as [j, l, m, C/S], CMB first (the file lists
    the surface first)."""
    lines = path.read_text().splitlines()
    out = np.zeros((129, 51, 51, 2))
    k = 263       # first line after "# Spherical Harmonics"
    assert lines[262].startswith("# Spherical Harmonics")
    for block in range(129):
        assert lines[k].startswith("# Radial Index")
        k += 1
        j = 128 - block
        for l in range(51):
            for m in range(l + 1):
                c, s = lines[k].split()
                out[j, l, m] = float(c), float(s)
                k += 1
    return out


@pytest.mark.legacy
def test_sphco_consistency_lebedev(load_fortran, legacy_dir):
    """The real legacy density file: fields of hager4_fsfs_lebedev from the kernels of
    hager4_fsfs (same build) and the coefficients read from LEBEDEV_S4RTS.sph."""
    coeffs = _read_lebedev_coefficients(legacy_dir / "INPUT/DENSITY_INPUTS/LEBEDEV_S4RTS.sph")
    worst = _check_fields(load_fortran("hager4_fsfs_lebedev"), load_fortran("hager4_fsfs"), coeffs)
    assert worst <= SPHCO_RTOL, worst


@pytest.mark.slow
@pytest.mark.legacy
def test_regenerate_bitwise(tmp_path, legacy_toolchain, manifest, golden_dir):
    """G3: a second harness run gives bitwise-equal arrays with the same toolchain."""
    repo = golden_dir.parent.parent
    out = tmp_path / "golden"
    out.mkdir()
    cmd = [sys.executable, str(repo / "legacy/harness/run_golden.py"), "--source", str(legacy_toolchain),
           "--work", str(tmp_path / "work"), "--out", str(out), "--no-manifest"]
    subprocess.run(cmd, check=True, capture_output=True, timeout=1500)
    same_toolchain = True
    try:
        cur = subprocess.run(["/opt/homebrew/bin/mpif90", "--version"], capture_output=True, text=True).stdout
        same_toolchain = cur.splitlines()[0] == manifest["toolchain"]["mpif90_version"]
    except Exception:
        same_toolchain = False
    for run in cases.GOLDEN_RUNS:
        with np.load(golden_dir / "fortran" / f"{run}.npz") as a, np.load(out / "fortran" / f"{run}.npz") as b:
            assert set(a.files) == set(b.files)
            for k in a.files:
                if same_toolchain:
                    assert np.array_equal(a[k], b[k]), (run, k)
                else:
                    scale = np.max(np.abs(a[k])) or 1.0
                    assert np.max(np.abs(a[k] - b[k])) <= REGEN_RTOL * scale, (run, k)
    shutil.rmtree(tmp_path / "work", ignore_errors=True)
