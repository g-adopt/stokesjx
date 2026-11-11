"""Readers and writers of CODE-A's file formats (``stokesjx.io.legacy``; gate G8 of
NOTES/chunks/c3-flags-fields/01-plan.md, section 5.7).

References: files written by ``tests/cases.py`` (the writers of the golden harness), the
LEBEDEV and Zurich density files and the viscosity file of ``legacy/CODE-A``, the
Fortran ``SPHCO`` output of ``test_4_ishan`` and of the golden run ``hager4_fsfs``.
"""

from __future__ import annotations

import numpy as np
import pytest

import cases
from adapters import to_cilm

from stokesjx.io import legacy as io_legacy

# write_sphco with digits=6 then read_sphco, relative per entry: half a unit in the
# sixth digit of a mantissa in [0.1, 1) is at most 5e-6 relative. Cause: the e15.6
# format.
SPHCO6_RTOL = 5e-6


def _small_table(n=9):
    """A small node table and density coefficients ``[j, l, m, C/S]`` (``lmax = 4``)."""
    radius = tuple(str(3480000 + 361250 * k) for k in range(n))
    rho0 = tuple(f"{5000 - 200 * k:.2f}" for k in range(n))
    rng = np.random.default_rng(18)
    coef = rng.standard_normal((n, 5, 5, 2)) * np.tril(np.ones((5, 5)))[None, :, :, None]
    return radius, rho0, coef


def test_read_sph_synthetic(tmp_path):
    """A file of ``cases.density_file_text`` (9 nodes, ``lmax = 4``, ``%.17e``): the
    coefficients ``(9, 2, 5, 5)`` equal the written ones exactly, the radius and the
    averages CMB first."""
    radius, rho0, coef = _small_table()
    path = tmp_path / "x.sph"
    path.write_text(cases.density_file_text(radius, rho0, coef, nr=8, lmax=4))
    s = io_legacy.read_sph(path)
    assert s.lmax == 4 and s.coefficients.shape == (9, 2, 5, 5)
    assert np.array_equal(s.coefficients, to_cilm(coef))
    assert np.array_equal(s.radius, np.array([float(r) for r in radius]))
    assert np.array_equal(s.average, np.array([float(r) for r in rho0]))


@pytest.mark.legacy
def test_read_sph_lebedev(legacy_dir):
    """LEBEDEV_S4RTS.sph against the reader of the golden tests
    (``test_golden._read_lebedev_coefficients``) and the tables of ``tests/cases.py``."""
    from test_golden import _read_lebedev_coefficients
    path = legacy_dir / "INPUT" / "DENSITY_INPUTS" / "LEBEDEV_S4RTS.sph"
    s = io_legacy.read_sph(path)
    assert s.lmax == 50 and s.coefficients.shape == (129, 2, 51, 51)
    assert np.array_equal(s.coefficients, to_cilm(_read_lebedev_coefficients(path)))
    assert np.array_equal(s.radius, np.array([float(r) for r in cases.LEBEDEV_RADIUS]))
    assert np.array_equal(s.average, np.array([float(r) for r in cases.LEBEDEV_RHO0]))


@pytest.mark.legacy
def test_read_sph_other_header(legacy_dir):
    """RHOSPHZurich_b_s: ``LMAX = 20`` and a second line ``# Temperature: INPUT ...``."""
    s = io_legacy.read_sph(legacy_dir / "INPUT" / "DENSITY_INPUTS" / "RHOSPHZurich_b_s")
    assert s.lmax == 20 and s.coefficients.shape == (129, 2, 21, 21)
    assert s.radius[0] == 3480000.0 and s.radius[-1] == 6370000.0
    assert np.all(np.diff(s.radius) >= 0) and np.all(np.isfinite(s.coefficients))


def test_read_viscosity(tmp_path):
    """Files of ``cases.viscosity_file_text`` for the golden profiles: CMB first, times
    ``visc0``; a two-column line takes the first number; too few lines raise."""
    for v in ("iso", "hager4", "smooth"):
        eta_rel = cases.golden_eta_rel(v)
        path = tmp_path / f"{v}.visc"
        path.write_text(cases.viscosity_file_text(eta_rel))
        got = io_legacy.read_viscosity(path, len(eta_rel))
        assert np.array_equal(got, np.array([float(e) for e in eta_rel]) * 1e19)
        assert np.array_equal(io_legacy.read_viscosity(path, len(eta_rel), unit=1.0),
                              np.array([float(e) for e in eta_rel]))
    path = tmp_path / "two.visc"
    path.write_text("3.0 7.0\n2.0\n1.0\n")
    assert np.array_equal(io_legacy.read_viscosity(path, 3, unit=1.0), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        io_legacy.read_viscosity(path, 4)


@pytest.mark.legacy
def test_read_viscosity_flip(legacy_dir):
    """Hager_4Layer_Flip.visc (257 values, no final newline) against ``cases.ishan_eta_rel``."""
    path = legacy_dir / "Hager_4Layer_Flip.visc"
    got = io_legacy.read_viscosity(path, 257)
    assert np.array_equal(got, np.array([float(e) for e in cases.ishan_eta_rel(path)]) * 1e19)


def test_sphco_round_trip(tmp_path):
    """``write_sphco`` then ``read_sphco``: exact with ``digits=16``, within half a unit of
    the sixth digit with ``digits=6``; degree 0 is not written (``lmin = 1``)."""
    rng = np.random.default_rng(19)
    L = 12
    c = rng.standard_normal((2, L + 1, L + 1)) * 10.0 ** rng.integers(-12, 12, (2, L + 1, L + 1))
    c *= np.tril(np.ones((L + 1, L + 1)))
    c[1, :, 0] = 0.0
    c[:, 0, :] = 0.0
    for digits in (16, 6):
        path = tmp_path / f"s{digits}"
        io_legacy.write_sphco(path, c, digits=digits)
        back = io_legacy.read_sphco(path)
        assert back.shape == c.shape
        if digits == 16:
            assert np.array_equal(back, c)
        else:
            nz = c != 0
            assert np.all(back[~nz] == 0)
            assert np.max(np.abs(back[nz] - c[nz]) / np.abs(c[nz])) <= SPHCO6_RTOL
    text = (tmp_path / "s6").read_text().splitlines()
    assert text[:4] == list(io_legacy.DEFAULT_HEADER) and text[-1] == "... END OF FILE!"
    with pytest.raises(ValueError):
        io_legacy.write_sphco(tmp_path / "bad", c, digits=8)


@pytest.mark.legacy
def test_sphco_fortran_text(legacy_dir, legacy_work, load_fortran, tmp_path):
    """The Fortran output: ``test_4_ishan/SPH_geoid`` and ``SPH_surfvel`` read and written
    with ``digits=6`` and their header give the identical text; the ``SPH_*`` files of the
    golden run ``hager4_fsfs`` (16 digits) equal the npz arrays and, written with
    ``digits=16``, give the identical text."""
    for name in ("SPH_geoid", "SPH_surfvel"):
        src = legacy_dir / "test_4_ishan" / name
        lines = src.read_text().splitlines()
        out = tmp_path / name
        io_legacy.write_sphco(out, io_legacy.read_sphco(src), header=tuple(lines[:4]))
        assert out.read_text() == src.read_text(), name
    d = load_fortran("hager4_fsfs")
    for fk in ("geoid", "surftopo", "cmbtopo", "surfvel"):
        src = legacy_work / f"SPH_{fk}"
        c = io_legacy.read_sphco(src)
        assert np.array_equal(c[0], d[f"sph_{fk}_C"]) and np.array_equal(c[1], d[f"sph_{fk}_S"])
        out = tmp_path / f"g_{fk}"
        io_legacy.write_sphco(out, c, digits=16, header=tuple(src.read_text().splitlines()[:4]))
        assert out.read_text() == src.read_text(), fk


def test_sphco_data_order(tmp_path):
    """Audit row 11: a file whose header says ``Cnm(m>=0), Snm(m<0)`` holds ``C`` in the
    first column and ``S`` in the second; also a three-digit exponent without ``E``."""
    path = tmp_path / "order"
    path.write_text("... HEADER\n... INPUT : x\n... out/\nl ,m , Cnm(m>=0) , Snm(m<0)\n"
                    "SPHCO       1       0   0.100000E+01   0.000000E+00\n"
                    "SPHCO       1       1   0.200000E+01   0.300000E+01\n"
                    "SPHCO       2       2   0.400000-100   0.500000E+00\n... END OF FILE!\n")
    c = io_legacy.read_sphco(path)
    assert c[0, 1, 0] == 1.0 and c[0, 1, 1] == 2.0 and c[1, 1, 1] == 3.0
    assert c[0, 2, 2] == 0.4e-100 and c[1, 2, 2] == 0.5
    io_legacy.write_sphco(tmp_path / "back", c)
    assert "0.400000-100" in (tmp_path / "back").read_text()
