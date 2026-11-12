"""Shared definitions of the reference cases: node tables, viscosity profiles, the
synthetic density, the Fortran run list and the mpmath oracle case list.

This module is the single source of every input number of the reference data. The
Fortran harness (legacy/harness/run_golden.py), the mpmath oracle
(tests/reference_mpmath.py, its ``main``) and the tests all import it, so that a
value cannot differ between the place that writes a reference file and the place
that reads it. The manifest of tests/golden/ records the sha256 of this file at the
time the reference data were written.

All numbers that enter a high-precision computation are decimal strings. The
oracle converts them with ``mpmath.mpf(str)``, which is exact, and the Fortran
reads the same strings into 128-bit reals. A float64 detour would round them.

Node order: index 0 is the core-mantle boundary (CMB), the last index is the
surface. Two neighbouring nodes with the same radius are a "jump pair": an interval
of zero width across which the viscosity or the density (or both) jump.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# The node table of LEBEDEV_S4RTS.sph (CODE-A/INPUT/DENSITY_INPUTS), the density
# file of every Fortran golden run. 129 nodes, equally spaced by 22578.125 m (the
# file prints the radius with two decimals). The radii and the "Averages" (the
# spherically averaged density, kg/m^3) are copied as printed. The file lists the
# surface first; these tuples list the CMB first.
# --------------------------------------------------------------------------

LEBEDEV_RADIUS = (
    "3480000.00", "3502578.12", "3525156.25", "3547734.38", "3570312.50", "3592890.62",
    "3615468.75", "3638046.88", "3660625.00", "3683203.12", "3705781.25", "3728359.38",
    "3750937.50", "3773515.62", "3796093.75", "3818671.88", "3841250.00", "3863828.12",
    "3886406.25", "3908984.38", "3931562.50", "3954140.62", "3976718.75", "3999296.88",
    "4021875.00", "4044453.12", "4067031.25", "4089609.38", "4112187.50", "4134765.62",
    "4157343.75", "4179921.88", "4202500.00", "4225078.12", "4247656.25", "4270234.38",
    "4292812.50", "4315390.62", "4337968.75", "4360546.88", "4383125.00", "4405703.12",
    "4428281.25", "4450859.38", "4473437.50", "4496015.62", "4518593.75", "4541171.88",
    "4563750.00", "4586328.12", "4608906.25", "4631484.38", "4654062.50", "4676640.62",
    "4699218.75", "4721796.88", "4744375.00", "4766953.12", "4789531.25", "4812109.38",
    "4834687.50", "4857265.62", "4879843.75", "4902421.88", "4925000.00", "4947578.12",
    "4970156.25", "4992734.38", "5015312.50", "5037890.62", "5060468.75", "5083046.88",
    "5105625.00", "5128203.12", "5150781.25", "5173359.38", "5195937.50", "5218515.62",
    "5241093.75", "5263671.88", "5286250.00", "5308828.12", "5331406.25", "5353984.38",
    "5376562.50", "5399140.62", "5421718.75", "5444296.88", "5466875.00", "5489453.12",
    "5512031.25", "5534609.38", "5557187.50", "5579765.62", "5602343.75", "5624921.88",
    "5647500.00", "5670078.12", "5692656.25", "5715234.38", "5737812.50", "5760390.62",
    "5782968.75", "5805546.88", "5828125.00", "5850703.12", "5873281.25", "5895859.38",
    "5918437.50", "5941015.62", "5963593.75", "5986171.88", "6008750.00", "6031328.12",
    "6053906.25", "6076484.38", "6099062.50", "6121640.62", "6144218.75", "6166796.88",
    "6189375.00", "6211953.12", "6234531.25", "6257109.38", "6279687.50", "6302265.62",
    "6324843.75", "6347421.88", "6370000.00",
)
LEBEDEV_RHO0 = (
    "5523.76", "5513.86", "5514.98", "5503.26", "5505.08", "5501.50",
    "5496.47", "5493.75", "5483.60", "5479.82", "5451.78", "5452.26",
    "5419.05", "5422.07", "5400.61", "5402.61", "5398.97", "5390.53",
    "5383.10", "5357.53", "5355.59", "5331.71", "5333.20", "5306.05",
    "5310.74", "5299.53", "5285.52", "5277.11", "5259.16", "5254.89",
    "5233.73", "5232.99", "5208.45", "5211.00", "5183.14", "5188.87",
    "5177.69", "5162.28", "5155.22", "5135.68", "5132.49", "5109.21",
    "5109.62", "5083.20", "5086.96", "5057.04", "5062.64", "5052.45",
    "5035.06", "5029.17", "5007.65", "5005.66", "4980.43", "4982.02",
    "4953.57", "4958.67", "4946.62", "4931.51", "4922.69", "4903.54",
    "4898.95", "4875.83", "4875.08", "4848.25", "4851.09", "4820.71",
    "4826.99", "4814.85", "4798.09", "4790.69", "4769.83", "4766.42",
    "4740.89", "4741.40", "4712.02", "4716.37", "4682.30", "4688.56",
    "4677.21", "4657.60", "4651.02", "4626.65", "4624.47", "4596.04",
    "4597.87", "4566.29", "4571.92", "4558.76", "4542.20", "4532.82",
    "4512.13", "4507.16", "4477.94", "4476.61", "4420.87", "4428.47",
    "4351.80", "4365.22", "4299.39", "4016.51", "3977.86", "3950.28",
    "3922.32", "3918.04", "3889.45", "3853.45", "3830.54", "3806.36",
    "3779.79", "3743.22", "3640.52", "3614.56", "3589.55", "3583.95",
    "3560.39", "3536.28", "3510.98", "3483.44", "3450.43", "3411.35",
    "3391.93", "3373.51", "3359.04", "3355.16", "3347.04", "3347.03",
    "3348.25", "3198.72", "3198.72",
)

R_SURFACE = "6370000"   # m, CODE-A's Rearth and the top node of every table
R_CMB = "3480000"       # m, the bottom node of every table
SURFACE_KM = 6370       # the depth of a node is (6370 km - r)
LEBEDEV_NODES = len(LEBEDEV_RADIUS)
assert LEBEDEV_NODES == 129 and LEBEDEV_RADIUS[0] == "3480000.00" and LEBEDEV_RADIUS[-1] == "6370000.00"

# --------------------------------------------------------------------------
# Decimal arithmetic for the inputs. Node depths, inserted radii and interpolated
# densities are computed with Python's Decimal at 40 digits, so that every input
# string is reproducible on every host and exact to far more digits than a test
# resolves.
# --------------------------------------------------------------------------

from decimal import Decimal, getcontext  # noqa: E402

getcontext().prec = 40


def _depth_km(radius: str) -> Decimal:
    """Depth below the 6370 km surface, km, exact for a decimal radius string."""
    return (Decimal(SURFACE_KM) * 1000 - Decimal(radius)) / 1000


def _fmt(x: Decimal | float) -> str:
    """A decimal string for a node value; Decimal keeps its digits, a float its repr."""
    if isinstance(x, float):
        return repr(x)
    s = format(x.normalize(), "f") if abs(x.adjusted()) < 30 else str(x)
    return s


# --------------------------------------------------------------------------
# Viscosity profiles, defined by depth (PLAN section 7, stage 2; plan 3.1).
# The Fortran files hold eta_rel = eta / 1e19, the value that CODE-A reads into
# rvsc and uses in aconstruct as it is.
# --------------------------------------------------------------------------

#: Isoviscous: 1e21 Pa s (visc_isoviscous_mt256.dat value 100).
ISO_ETA_REL = "100"

#: The 4-layer profile of Hager_4Layer_Flip.visc, which is also the test_4_ishan
#: profile (plan F11). Layers from the top: 1e20 Pa s above 100 km depth, 1e19/30
#: (3.333e17 Pa s, the file holds 0.03333333333333333) from 100 to 400 km, 1e19 from
#: 400 to 670 km, 1e20 below. Each entry is (depth of the bottom of the layer in km,
#: eta_rel); the last layer has no bottom.
HAGER4_LAYERS = ((100, "10"), (400, "0.03333333333333333"), (670, "1"), (None, "10"))
#: The two names of the chunk scope are one profile.
TEST_4_ISHAN_LAYERS = HAGER4_LAYERS

#: Five decades (T_wide5): lid 1e23 (visc_lith of constants.h) above 70 km, 1e18
#: (min_visc_asth) to 250 km, 1e21 to 670 km, 1e22 below.
WIDE5_LAYERS = ((70, "1e4"), (250, "0.1"), (670, "100"), (None, "1000"))

#: Stiff lid (T_stiff_lid): 1e27 Pa s above 70 km on a 1e20 Pa s mantle, seven
#: decades. The pairs at 250 and 670 km carry no viscosity jump; they keep the node
#: structure of T_wide5 and T_wide10.
STIFF_LID_LAYERS = ((70, "1e8"), (250, "10"), (670, "10"), (None, "10"))

#: Ten decades (T_wide10): 1e26 Pa s above 70 km, 1e16 to 250 km, 1e21 to 670 km,
#: 1e23 below.
WIDE10_LAYERS = ((70, "1e7"), (250, "1e-3"), (670, "100"), (None, "1e4"))


def layered_eta_rel(radius: str, layers, side: int = 0) -> str:
    """eta_rel of a node in a layered profile.

    ``side`` resolves a node exactly at a layer boundary: +1 is the upper node of a
    jump pair (the shallower layer), -1 the lower node. An ordinary node at a
    boundary depth (``side = 0``) belongs to the deeper layer, as the rule
    ``d < 100 km -> layer 0`` reads.
    """
    d = _depth_km(radius)
    for k, (bottom, value) in enumerate(layers):
        if bottom is None:
            return value
        b = Decimal(bottom)
        if d < b or (d == b and side > 0):
            return value
    raise AssertionError("unreachable")


def smooth_eta_rel(radius: str) -> str:
    """The closed-form smooth profile (plan 3.1), as eta_rel = eta / 1e19.

    ``log10(eta / Pa s) = 19.5 + 2.5 d / 2890 + 0.7 tanh((d - 670) / 100)``, with the
    depth ``d`` in km. Range 1e18.8 to 1e22.7 Pa s. It changes at every node, so the
    upper-node rule of CODE-A is tested at every interval. The value is a float64
    written with repr, so the Fortran and the oracle read the same decimal.
    """
    d = float(_depth_km(radius))
    log10_eta = 19.5 + 2.5 * d / 2890.0 + 0.7 * math.tanh((d - 670.0) / 100.0)
    return repr(10.0 ** (log10_eta - 19.0))


# The smooth profile at the 129 LEBEDEV nodes, as written for the Fortran golden runs.
#
# smooth_eta_rel uses math.tanh and the power 10.0 ** x, whose last bit comes from the
# C maths library of the platform (macOS and glibc differ for some arguments). The
# golden runs read the repr of these values, and test_golden.py checks the sha256 of
# the viscosity file text against the manifest, so the golden runs take the stored
# strings (reunion, macOS arm64, 2026-11-11) and the text is the same on every
# platform. Other node tables (T_smooth(N)) still use the formula, because only
# tolerance tests read them. test_smooth_profile_matches_formula checks the stored
# values against the formula to a few ulp on the running platform.
SMOOTH_ETA_REL_PATH = Path(__file__).parent / "data" / "smooth_eta_rel_lebedev.txt"
SMOOTH_ETA_REL_LEBEDEV = tuple(
    line.strip() for line in SMOOTH_ETA_REL_PATH.read_text().splitlines()
    if line.strip() and not line.startswith("#"))
assert len(SMOOTH_ETA_REL_LEBEDEV) == 129


def golden_eta_rel(profile: str) -> tuple[str, ...]:
    """eta_rel at the 129 LEBEDEV nodes for a golden-run profile, CMB first. The
    smooth profile comes from the stored strings ``SMOOTH_ETA_REL_LEBEDEV`` (see the
    comment above them)."""
    if profile == "iso":
        return tuple(ISO_ETA_REL for _ in LEBEDEV_RADIUS)
    if profile == "hager4":
        return tuple(layered_eta_rel(r, HAGER4_LAYERS) for r in LEBEDEV_RADIUS)
    if profile == "smooth":
        return SMOOTH_ETA_REL_LEBEDEV
    raise ValueError(profile)


# --------------------------------------------------------------------------
# The synthetic density anomaly of the golden runs (plan 3.1).
# --------------------------------------------------------------------------

SYNTH_LMAX = 50
SYNTH_AMPLITUDE = 10.0          # kg/m^3
_SYNTH_PHASE = (0.3, 1.1, 2.0)  # phase_k of the cosine coefficients
_SYNTH_S_SHIFT = 0.9            # extra phase of the sine coefficients


# The cosines and the Gaussian of the synthetic anomaly, stored as float64 values.
#
# The Fortran golden runs read a density file whose text this module writes with 17
# significant digits, and test_golden.py checks the sha256 of that text against the
# manifest. The values of math.cos and math.exp come from the C maths library of the
# platform, and the libraries of macOS and of glibc differ in the last bit for some
# arguments (168 of the 3872 cosines here are not correctly rounded on macOS). The
# text, and so the hash, would then depend on the machine. The table holds the values
# that the golden runs were made with (reunion, macOS arm64, 2026-11-11), so the text
# is the same on every platform: the remaining arithmetic is IEEE addition,
# multiplication and division of Python floats, which every platform rounds the same
# way. test_synthetic_basis_matches_formula checks that every entry is within a few
# ulp of the formula on the running platform.
SYNTH_BASIS_PATH = Path(__file__).parent / "data" / "synthetic_basis.npz"
_SYNTH_BASIS = np.load(SYNTH_BASIS_PATH)
# cos(1.3 (k+1) l + 0.7 m + phase_k [+ 0.9]) for [l, m, k, 0 (C) or 1 (S)]; zero at l = 0.
SYNTH_COS = _SYNTH_BASIS["cos"]
# exp(-((d - 2000 km) / 300 km)^2) at the 129 LEBEDEV nodes, CMB first.
SYNTH_GAUSS = _SYNTH_BASIS["gauss"]


def synth_cos_formula(l: int, m: int, k: int, sine: bool) -> float:
    """The cosine of the synthetic anomaly from the formula, with the maths library of
    the running platform. Used only to check the stored table ``SYNTH_COS``."""
    a = 1.3 * (k + 1) * l + 0.7 * m + _SYNTH_PHASE[k]
    return math.cos(a + _SYNTH_S_SHIFT) if sine else math.cos(a)


def synth_gauss_formula(radius: str) -> float:
    """The Gaussian radial shape from the formula, with the maths library of the
    running platform. Used only to check the stored table ``SYNTH_GAUSS``."""
    d = float(_depth_km(radius))
    return math.exp(-((d - 2000.0) / 300.0) ** 2)


def _synth_shapes(j: int) -> tuple[float, float, float]:
    """The three radial shapes b_k at LEBEDEV node j (CMB first): a constant, a
    Gaussian of 300 km half-width centred at 2000 km depth (from the stored table),
    and a linear ramp in depth (0 at the surface, 1 at the CMB). The depth comes from
    the decimal radius string, so the ramp is the same on every platform."""
    d = float(_depth_km(LEBEDEV_RADIUS[j]))
    return (1.0, float(SYNTH_GAUSS[j]), d / 2890.0)


def synthetic_density(l: int, m: int, j: int) -> tuple[float, float]:
    """(C, S) coefficient of the synthetic anomaly at degree l, order m, LEBEDEV node j.

    ``C = A sum_k cos(1.3 k l + 0.7 m + phase_k) / (l + 1) b_k(r_j)`` and the same for
    S with the phase shifted by 0.9. Non-separable in (l, m, r). S is zero at m = 0
    and both are zero at l = 0. Units kg/m^3, in the coefficient convention of the
    density file (CODE-A multiplies by sqrt(4 pi) on reading, util.f90:52). The
    cosines and the Gaussian come from the stored tables ``SYNTH_COS`` and
    ``SYNTH_GAUSS`` (see the comment above them).
    """
    if l == 0:
        return 0.0, 0.0
    b = _synth_shapes(j)
    c = sum(float(SYNTH_COS[l, m, k, 0]) * b[k] for k in range(3))
    s = sum(float(SYNTH_COS[l, m, k, 1]) * b[k] for k in range(3))
    c = SYNTH_AMPLITUDE * c / (l + 1)
    s = SYNTH_AMPLITUDE * s / (l + 1) if m > 0 else 0.0
    return c, s


def synthetic_coefficients() -> np.ndarray:
    """All synthetic coefficients as an array ``[j, l, m, 0 (C) or 1 (S)]``, float64,
    shape (129, 51, 51, 2), CMB first, with zeros for m > l."""
    out = np.zeros((LEBEDEV_NODES, SYNTH_LMAX + 1, SYNTH_LMAX + 1, 2))
    for j in range(LEBEDEV_NODES):
        for l in range(SYNTH_LMAX + 1):
            for m in range(l + 1):
                out[j, l, m] = synthetic_density(l, m, j)
    return out


def density_file_text(radius, rho0, coefficients, nr: int, lmax: int = SYNTH_LMAX) -> str:
    """Text of a CODE-A density file (.sph) in the layout ``initialize`` reads.

    ``initialize`` (util.f90:7-54) reads ``LMAX`` from columns 8-11 and ``NR`` from
    columns 16-19 of the first line, two header lines, the ``nr + 1`` radii (surface
    first), one header line, the ``nr + 1`` densities, one header line, and then per
    node (surface first) one header line and ``(l, m)`` lines with C and S for
    ``l = 0..lmax``, ``m = 0..l``. ``coefficients[j, l, m, 0/1]`` is CMB first; None
    writes zero anomalies. Values are written with ``%.17e``, so that the Fortran
    reads the float64 value exactly.
    """
    n = len(radius)
    assert n == nr + 1
    lines = [f"# LMAX={lmax:4d}, NR{nr:4d} 1", "# Density", "# Radius"]
    lines += [f"{r:>15s}" for r in reversed(radius)]
    lines.append("# Averages")
    lines += [f"{d:>15s}" for d in reversed(rho0)]
    lines.append("# Spherical Harmonics")
    for k in range(n):
        j = n - 1 - k
        lines.append(f"# Radial Index:{k + 1:4d}")
        for l in range(lmax + 1):
            for m in range(l + 1):
                if coefficients is None:
                    c, s = 0.0, 0.0
                else:
                    c, s = coefficients[j, l, m]
                lines.append(f"{c:25.17e} {s:25.17e}")
    return "\n".join(lines) + "\n"


def viscosity_file_text(eta_rel) -> str:
    """Text of a CODE-A viscosity file: eta_rel per line, surface first (viscinit,
    util.f90:353-364)."""
    return "\n".join(reversed(eta_rel)) + "\n"


# --------------------------------------------------------------------------
# Fortran golden runs (plan 3.2). All at 129 nodes, l_max = l_max_calc = 50,
# CorrFlag = 0, rho_c = 9900, rho_w = 0, bgrho = 4500, visc0 = 1e19.
# --------------------------------------------------------------------------

GOLDEN_RUNS = {
    "iso_fsfs": dict(visc="iso", bdry=1, bdry_bot=1, kappaindex=0, density="synthetic"),
    "hager4_fsfs": dict(visc="hager4", bdry=1, bdry_bot=1, kappaindex=0, density="synthetic"),
    "smooth_fsfs": dict(visc="smooth", bdry=1, bdry_bot=1, kappaindex=0, density="synthetic"),
    # The fields of the two isoviscous no-slip runs are not stored (plan risk 7:
    # tests/golden/ stays under 3 MB). Their kernels are.
    "iso_nsfs": dict(visc="iso", bdry=0, bdry_bot=1, kappaindex=0, density="synthetic", fields=False),
    "hager4_nsfs": dict(visc="hager4", bdry=0, bdry_bot=1, kappaindex=0, density="synthetic"),
    "iso_fsns": dict(visc="iso", bdry=1, bdry_bot=0, kappaindex=0, density="synthetic", fields=False),
    "hager4_fsfs_tala": dict(visc="hager4", bdry=1, bdry_bot=1, kappaindex=1, density="synthetic"),
    "hager4_fsfs_lebedev": dict(visc="hager4", bdry=1, bdry_bot=1, kappaindex=0, density="lebedev",
                                fields_only=True),
}

#: Degrees of the stored legacy-switch oracle values that G1 reads.
LEGACY_DEGREES = (1, 2, 3, 10, 20, 30, 40, 50)
#: Degrees of the stored legacy-switch oracle of the historical test_4_ishan run.
ISHAN_DEGREES = (1, 2, 10, 30, 50)


def golden_table(run: str):
    """Node table of a Fortran golden run for the oracle, with
    ``eta = eta_rel * float32(1e19)`` exactly (plan 6.2), as CODE-A sees it."""
    from reference_mpmath import NodeTable, legacy_visc0_string, LEGACY_RHO_C
    spec = GOLDEN_RUNS[run]
    visc0 = Decimal(legacy_visc0_string())
    eta = tuple(_fmt(Decimal(e) * visc0) for e in golden_eta_rel(spec["visc"]))
    return NodeTable(radius=LEBEDEV_RADIUS, rho0=LEBEDEV_RHO0, eta=eta,
                     core_density=LEGACY_RHO_C, core_mass=None, outer_density="0")


def parse_ishan_summary(path: Path):
    """Radii (m) and densities (kg/m^3) of the historical test_4_ishan run, CMB first,
    as printed in its Summary file (radius to 1 m, density to 0.01 kg/m^3)."""
    rows = []
    for ln in Path(path).read_text().splitlines()[10:]:
        parts = ln.split()
        if len(parts) == 5 and parts[0][0].isdigit():
            rows.append(parts)
    radius = tuple(p[0].rstrip(".") for p in reversed(rows))
    rho0 = tuple(p[2] for p in reversed(rows))
    visc = tuple(p[1] for p in reversed(rows))
    return radius, rho0, visc


def ishan_eta_rel(visc_file: Path) -> tuple[str, ...]:
    """eta_rel of Hager_4Layer_Flip.visc (257 values, surface first in the file),
    CMB first. The file has no terminal newline (plan F12)."""
    vals = Path(visc_file).read_text().split()
    assert len(vals) == 257
    return tuple(reversed(vals))


def ishan_table(legacy_root: Path):
    """Node table of the historical test_4_ishan run (257 nodes). The radii and the
    densities come from its Summary, the viscosity from Hager_4Layer_Flip.visc: the
    Summary prints the viscosity with three digits (0.333E+18), while the run used
    1e19/30 (the file value 0.03333333333333333)."""
    from reference_mpmath import NodeTable, legacy_visc0_string, LEGACY_RHO_C
    legacy_root = Path(legacy_root)
    radius, rho0, visc_printed = parse_ishan_summary(legacy_root / "test_4_ishan" / "Summary")
    eta_rel = ishan_eta_rel(legacy_root / "Hager_4Layer_Flip.visc")
    visc0 = Decimal(legacy_visc0_string())
    for p, e in zip(visc_printed, eta_rel):
        assert abs(Decimal(p) / (Decimal(e) * Decimal("1e19")) - 1) < Decimal("2e-3")
    eta = tuple(_fmt(Decimal(e) * visc0) for e in eta_rel)
    return NodeTable(radius=radius, rho0=rho0, eta=eta, core_density=LEGACY_RHO_C,
                     core_mass=None, outer_density="0")


# --------------------------------------------------------------------------
# Node tables of the oracle cases (plan 6.5).
# --------------------------------------------------------------------------

ETA_S = Decimal("1e19")  # Pa s: the viscosity unit of the eta_rel values above

# PREM (Dziewonski and Anderson 1981) density polynomials in g/cm^3 of
# x = r / 6371 km. The stokesjx tables have their surface at 6370 km, so the PREM
# discontinuities at 220, 400 and 670 km depth sit at 6150, 5970 and 5700 km.
_PREM_X0 = Decimal(6371000)
_PREM = {
    "inner_core": (Decimal("13.0885"), Decimal(0), Decimal("-8.8381"), Decimal(0)),
    "outer_core": (Decimal("12.5815"), Decimal("-1.2638"), Decimal("-3.6426"), Decimal("-5.5281")),
    "lower_mantle": (Decimal("7.9565"), Decimal("-6.4761"), Decimal("5.5283"), Decimal("-3.0807")),
    "tz_lower": (Decimal("5.3197"), Decimal("-1.4836"), Decimal(0), Decimal(0)),
    "tz_upper": (Decimal("11.2494"), Decimal("-8.0298"), Decimal(0), Decimal(0)),
    "upper_mantle": (Decimal("7.1089"), Decimal("-3.8045"), Decimal(0), Decimal(0)),
    "lid": (Decimal("2.6910"), Decimal("0.6924"), Decimal(0), Decimal(0)),
}


def _prem_poly(name: str, radius: Decimal) -> Decimal:
    a, b, c, d = _PREM[name]
    x = radius / _PREM_X0
    return 1000 * (a + b * x + c * x * x + d * x * x * x)


def _prem_region(radius: Decimal, side: int) -> str:
    """PREM mantle region of a node; ``side`` selects the polynomial at a pair.

    Boundaries (6370 km surface): 670 km (5700 km, lower mantle below), 600 km
    (5770 km, the first-order boundary of the transition zone), 400 km (5970 km),
    220 km (6150 km). The LID polynomial is extended to the surface (no crust)."""
    km = radius / 1000
    bounds = ((Decimal(5700), "lower_mantle"), (Decimal(5770), "tz_lower"),
              (Decimal(5970), "tz_upper"), (Decimal(6150), "upper_mantle"))
    for b, name in bounds:
        if km < b or (km == b and side < 0):
            return name
    return "lid"


def prem_core_mass() -> Decimal:
    """Mass of the PREM core (inner and outer), kg, by exact integration of the
    polynomials: about 1.9355e24 kg."""
    def shell(name, r0, r1):
        a, b, c, d = _PREM[name]
        x0 = _PREM_X0

        def prim(r):
            return 1000 * (a * r ** 3 / 3 + b * r ** 4 / (4 * x0) + c * r ** 5 / (5 * x0 ** 2)
                           + d * r ** 6 / (6 * x0 ** 3))
        return prim(r1) - prim(r0)
    pi = Decimal("3.141592653589793238462643383279502884197")
    return 4 * pi * (shell("inner_core", Decimal(0), Decimal(1221500))
                     + shell("outer_core", Decimal(1221500), Decimal(3480000)))


PREM_CORE_MASS = _fmt(prem_core_mass())
PREM_RHO_CMB_CORE = _fmt(_prem_poly("outer_core", Decimal(3480000)))
RHO_C = "9900"


@dataclass(frozen=True)
class _Node:
    radius: str
    lebedev: int | None   # index into the LEBEDEV table, None for an inserted node
    side: int              # -1 lower node of a pair, +1 upper node, 0 otherwise


def _with_pairs(radii, pair_depths_km, lebedev=True) -> list[_Node]:
    """Nodes of ``radii`` plus a jump pair at each depth (km), CMB first."""
    nodes = [_Node(r, k if lebedev else None, 0) for k, r in enumerate(radii)]
    for d in pair_depths_km:
        r = _fmt(Decimal(SURFACE_KM - d) * 1000)
        assert all(Decimal(n.radius) != Decimal(r) for n in nodes), "pair on an existing node"
        nodes += [_Node(r, None, -1), _Node(r, None, +1)]
    nodes.sort(key=lambda n: (Decimal(n.radius), n.side))
    return nodes


def _interp_lebedev_rho(radius: str) -> str:
    """rho0 at an inserted radius, linear in r between the neighbouring LEBEDEV nodes."""
    r = Decimal(radius)
    rr = [Decimal(x) for x in LEBEDEV_RADIUS]
    for k in range(len(rr) - 1):
        if rr[k] <= r <= rr[k + 1]:
            t = (r - rr[k]) / (rr[k + 1] - rr[k])
            return _fmt(Decimal(LEBEDEV_RHO0[k]) + t * (Decimal(LEBEDEV_RHO0[k + 1]) - Decimal(LEBEDEV_RHO0[k])))
    raise ValueError(radius)


def _eta_si(eta_rel: str) -> str:
    return _fmt(Decimal(eta_rel) * ETA_S)


def _reduce(nodes: list[_Node], every: int = 8) -> list[_Node]:
    """Every ``every``-th LEBEDEV node (both end nodes included) and every pair node."""
    return [n for n in nodes if n.lebedev is None or n.lebedev % every == 0]


def _table(nodes, rho_rule, eta_layers, core_density, core_mass, outer_density="0"):
    from reference_mpmath import NodeTable
    return NodeTable(
        radius=tuple(n.radius for n in nodes),
        rho0=tuple(rho_rule(n) for n in nodes),
        eta=tuple(_eta_si(layered_eta_rel(n.radius, eta_layers, n.side)) for n in nodes),
        core_density=core_density, core_mass=core_mass, outer_density=outer_density)


def _rho_lebedev(n: _Node) -> str:
    return LEBEDEV_RHO0[n.lebedev] if n.lebedev is not None else _interp_lebedev_rho(n.radius)


def _rho_prem(n: _Node) -> str:
    r = Decimal(n.radius)
    return _fmt(_prem_poly(_prem_region(r, n.side), r))


_LAYER_DENSITY = ((400, "3380"), (670, "3850"), (2000, "4650"), (None, "5400"))


def _rho_layers(n: _Node) -> str:
    return layered_eta_rel(n.radius, _LAYER_DENSITY, n.side)


def t_hager(reduced=False, eta_layers=HAGER4_LAYERS, outer_density="0", core="split"):
    """T_hager: LEBEDEV radii and rho0, viscosity pairs at 100, 400 and 670 km depth
    (rho0 interpolated, no density jump), hager4 viscosities, rho_c = 9900. 135 nodes
    (23 reduced, 3 pairs). ``core``: "split" (PREM core mass) or "consistent"."""
    nodes = _with_pairs(LEBEDEV_RADIUS, (100, 400, 670))
    nodes = _reduce(nodes) if reduced else nodes
    return _table(nodes, _rho_lebedev, eta_layers, RHO_C,
                  PREM_CORE_MASS if core == "split" else None, outer_density)


def t_prem(reduced=False, narrow_width_m: str | None = None):
    """T_prem: LEBEDEV radii with pairs at 100, 220, 400 and 670 km, PREM mantle
    density (density jumps at the 220, 400 and 670 km pairs), hager4 viscosities,
    rho_c the PREM outer-core value at r_c, PREM core mass. 137 nodes (25 reduced,
    4 pairs).

    ``narrow_width_m`` replaces the 670 km pair by an interval of that width: the
    lower node moves down to ``r_670 - w`` and keeps the lower-mantle polynomial
    evaluated at its radius (case ``prem_tala_narrow``)."""
    nodes = _with_pairs(LEBEDEV_RADIUS, (100, 220, 400, 670))
    nodes = _reduce(nodes) if reduced else nodes
    if narrow_width_m is not None:
        r670 = _fmt(Decimal(SURFACE_KM - 670) * 1000)
        nodes = [(_Node(_fmt(Decimal(r670) - Decimal(narrow_width_m)), None, -1)
                  if (n.radius == r670 and n.side == -1) else n) for n in nodes]
    return _table(nodes, _rho_prem, HAGER4_LAYERS, PREM_RHO_CMB_CORE, PREM_CORE_MASS)


def t_uniform(reduced=False, outer_density="0"):
    """T_uniform: LEBEDEV radii, rho0 = 4400 everywhere, hager4 viscosities (no
    pairs), rho_c = 9900, consistent core. 129 nodes (17 reduced)."""
    nodes = [_Node(r, k, 0) for k, r in enumerate(LEBEDEV_RADIUS)]
    nodes = _reduce(nodes) if reduced else nodes
    return _table(nodes, lambda n: "4400", HAGER4_LAYERS, RHO_C, None, outer_density)


def t_wide5():
    """T_wide5: LEBEDEV radii and rho0, pairs at 70, 250 and 670 km, viscosity over
    five decades (1e23, 1e18, 1e21, 1e22 Pa s), rho_c = 9900, PREM core mass."""
    nodes = _with_pairs(LEBEDEV_RADIUS, (70, 250, 670))
    return _table(nodes, _rho_lebedev, WIDE5_LAYERS, RHO_C, PREM_CORE_MASS)


def t_stiff_lid():
    """T_stiff_lid: LEBEDEV radii and rho0, pairs at 70, 250 and 670 km depth (rho0
    interpolated), 1e27 Pa s above 70 km on a 1e20 Pa s mantle (``STIFF_LID_LAYERS``,
    seven decades), rho_c = 9900, PREM core mass. 135 nodes.

    Purpose: the only stored case with ``eta_max / eta_geo`` above 1e6. At ``l = 1``
    (CM) a viscosity scale of the solver below the largest viscosity loses digits here:
    with the geometric mean the velocity kernel had 1.7e-10 under XLA (2.0e-11 in
    numpy), with the largest viscosity the worst kernel 1.1e-12 (1.9e-12 in numpy, CMB
    topography); at ``l`` in {2, 10, 50, 100} the two rules gave 3.4e-13 and 3.0e-13, so
    the separation is at ``l = 1`` (D1 of the chunk 2 plan, measurement 4)."""
    nodes = _with_pairs(LEBEDEV_RADIUS, (70, 250, 670))
    return _table(nodes, _rho_lebedev, STIFF_LID_LAYERS, RHO_C, PREM_CORE_MASS)


def t_wide10():
    """T_wide10: as T_stiff_lid with ``WIDE10_LAYERS`` (1e26, 1e16, 1e21, 1e23 Pa s from
    the top, ten decades). 135 nodes."""
    nodes = _with_pairs(LEBEDEV_RADIUS, (70, 250, 670))
    return _table(nodes, _rho_lebedev, WIDE10_LAYERS, RHO_C, PREM_CORE_MASS)


#: Viscosity of T_rough8 in Pa s at the 129 LEBEDEV nodes, CMB first: an independent
#: log-uniform value at every node, ``log10 eta = 21 + 8 (u - 0.5)``, eight decades
#: (1.8e17 to 9.8e24 Pa s), largest neighbour ratio 1.0e7. Written once from the
#: generator of NOTES/chunks/c2-forward-core/plan-evidence/r2/rough8_case.py
#: (``numpy.random.default_rng(108)``, ``u = rng.random(129)``,
#: ``repr(float(10 ** x))``), so that the table does not depend on the random stream of
#: a future NumPy.
ROUGH8_ETA = (
    '7.046128167368804e+23', '1.143408041258189e+22', '8.77494606321988e+17', '6.3835386087444775e+19',
    '8.334338270239834e+18', '4.042817082582799e+17', '6.483121202827946e+20', '1.4138474039970399e+24',
    '3.3892798803373805e+20', '8.828048019322719e+21', '1.8269516699534565e+21', '2.742950147397344e+21',
    '1.9067686898740458e+20', '9.512345166545254e+17', '9.83081216756065e+24', '3.155660344892831e+18',
    '5.795540196612637e+19', '6.06746698749937e+18', '4.637157348490234e+22', '6.267251494645022e+23',
    '3.712725170017855e+17', '6.382183587569013e+21', '6.70108987580437e+17', '1.658481281289755e+24',
    '9.104260377460785e+19', '1.0585888558571932e+23', '1.3059857389162636e+22', '1.2884080134777956e+24',
    '1.784652481459431e+21', '2.3512671261049725e+19', '3.4644414140011555e+23', '6.50002260710214e+19',
    '8.532259135220589e+19', '1.4890081419849797e+18', '6.75094538888959e+18', '7.312144081725733e+19',
    '2.938101317541048e+24', '3.0763140030633766e+24', '7.391934554282557e+19', '7.866307686429294e+17',
    '1.1539538203662422e+19', '9.611300430151183e+21', '3.3768672034101485e+17', '2.1164689365471716e+21',
    '3.618704984491077e+20', '5.447169137051987e+21', '1.0333471208870038e+18', '3.1606250208986395e+24',
    '2.2324342013132619e+24', '9.213169645775416e+17', '2.63053843876754e+24', '2.6198590662598076e+20',
    '5.938992751093183e+17', '3.952655600188563e+23', '2.3059346386866634e+22', '2.000268590581955e+19',
    '4.961891887965839e+17', '9.17409693052809e+19', '8.849910784239556e+17', '3.131495307788968e+22',
    '1.1727877263117418e+22', '3.081278320937556e+21', '3.044584372471626e+19', '2.894878636574595e+19',
    '5.8407883350791225e+19', '1.2838856613380037e+19', '5.819737442516136e+17', '2.6542640527413933e+20',
    '1.7905315289939053e+17', '6.151169925885079e+20', '1.106316509496618e+20', '3.768051331667382e+23',
    '2.691809682686623e+24', '1.6128632877879052e+22', '1.4968658072287773e+21', '4.17357473540715e+17',
    '7.338625684185136e+20', '9.000277239693044e+18', '7.730386572421821e+17', '7.336777668592972e+21',
    '2.814647112991429e+19', '2.706415682488745e+24', '3.3579841018301126e+17', '3.8120399344625616e+21',
    '2.4338869667148966e+22', '8.58304605129855e+21', '3.1368091876417927e+24', '8.921296901889938e+24',
    '3.5047070670529688e+19', '2.0952698912120742e+18', '1.174819304606746e+22', '2.626553612935363e+21',
    '1.6077654895690632e+21', '6.3377472125351034e+22', '5.093278213034374e+18', '1.7909368229852398e+20',
    '1.9915669604770023e+21', '3.739173227188639e+21', '1.2230166873792925e+24', '9.728466534508388e+22',
    '1.152558992830967e+19', '1.7747609490632653e+21', '5.188978821580355e+22', '5.00710118348218e+24',
    '1.3606273316841237e+23', '1.9606166904239156e+22', '5.88661982177293e+23', '3.1515942471893325e+18',
    '1.0240844424227308e+21', '2.2998227036649987e+17', '1.2991982079607776e+22', '3.732150667448239e+20',
    '2.1357199749760696e+18', '2.7373847862939406e+21', '2.6121263805751034e+20', '3.8260747811453685e+20',
    '1.3061100040023257e+22', '1.6362450143192654e+19', '4.2188435195796605e+21', '9.593738837016893e+22',
    '3.774498240720974e+24', '2.3707042046937057e+18', '4.5484599440877307e+18', '1.7560971020708335e+21',
    '3.4941925898310543e+18', '7.091233681885769e+21', '1.121077721475007e+24', '4.4004292232968326e+20',
    '1.467804696202435e+21',
)
assert len(ROUGH8_ETA) == LEBEDEV_NODES

#: Oracle precision of T_rough8 in decimal digits. On this table the default precision
#: (``required_dps``: 60 to 94 digits) is wrong at every stored degree above 2 without
#: an exception: max-normalised against 250 digits 3.3e-36 (l = 2), 3.3e-16 (l = 10),
#: 2.8e+4 (l = 50), 9.7e-7 (l = 100), and at l = 1 the singularity check raises
#: (ratio 1.26e-23). At 250 digits the oracle agrees with 320 digits to 2.4e-232
#: (l = 1) to 9.9e-163 (l = 100) (NOTES/chunks/c2-forward-core/plan-review-r3-evidence/
#: r3_chunk1_tests.out, r3_g4_dps250.out). Cause: the oracle carries the three CMB
#: solutions to the surface in one product, and on a rough table they arrive nearly
#: parallel. Every live oracle call on T_rough8 passes ``dps=ROUGH8_DPS``.
ROUGH8_DPS = 250


def t_rough8():
    """T_rough8: LEBEDEV radii and rho0, no pairs, rho_c = 9900, PREM core mass, and an
    independent viscosity at every node (``ROUGH8_ETA``, eight decades around
    1e21 Pa s, neighbour ratios up to 1.0e7). 129 nodes.

    Purpose: the only stored case where neighbouring nodes have very different
    viscosities, which is what a per-node control of a sampler produces (review r2 of
    the chunk 2 plan, other issue 2). Precision rule: every live oracle call on this
    table passes ``dps=ROUGH8_DPS``, because at the default precision the oracle returns
    wrong numbers at ``l >= 2`` without an exception."""
    from reference_mpmath import NodeTable
    return NodeTable(radius=LEBEDEV_RADIUS, rho0=LEBEDEV_RHO0, eta=ROUGH8_ETA,
                     core_density=RHO_C, core_mass=PREM_CORE_MASS)


def t_layers(reduced=False):
    """T_layers: LEBEDEV radii with pairs at 400, 670 and 2000 km depth, four layers
    of constant density (3380, 3850, 4650, 5400 kg/m^3 from the top) with the jumps
    at the pairs only, hager4 viscosities, rho_c = 9900, consistent core. 135 nodes
    (23 reduced). The exact translation sheets of G8e are the three jumps."""
    nodes = _with_pairs(LEBEDEV_RADIUS, (400, 670, 2000))
    nodes = _reduce(nodes) if reduced else nodes
    return _table(nodes, _rho_layers, HAGER4_LAYERS, RHO_C, None)


def t_smooth(n_intervals: int):
    """T_smooth(N): N + 1 equally spaced radii, a cubic density from 5500 kg/m^3 at
    the CMB to 3300 kg/m^3 at the surface, the smooth viscosity of plan 3.1,
    rho_c = 9900, PREM core mass."""
    from reference_mpmath import NodeTable
    R, rc = Decimal(R_SURFACE), Decimal(R_CMB)
    radius, rho, eta = [], [], []
    for i in range(n_intervals + 1):
        r = rc + (R - rc) * i / n_intervals
        x = (R - r) / (R - rc)          # 0 at the surface, 1 at the CMB
        radius.append(_fmt(r))
        rho.append(_fmt(Decimal(3300) + Decimal(2200) * (x / 2 + 3 * x * x / 2 - x * x * x)))
        eta.append(_eta_si(smooth_eta_rel(_fmt(r))))
    return NodeTable(radius=tuple(radius), rho0=tuple(rho), eta=tuple(eta),
                     core_density=RHO_C, core_mass=PREM_CORE_MASS)


def with_core(table, core_mass=..., core_density=None):
    """Copy of a table with another core."""
    from dataclasses import replace
    kw = {}
    if core_mass is not ...:
        kw["core_mass"] = core_mass
    if core_density is not None:
        kw["core_density"] = core_density
    return replace(table, **kw)


def consistent_core_mass(table, delta: str = "0") -> str:
    """``(4 pi / 3) r_c^3 rho_c (1 + delta)`` as a decimal string of 80 digits."""
    import mpmath as mp
    with mp.workdps(90):
        v = 4 * mp.pi / 3 * mp.mpf(table.radius[0]) ** 3 * mp.mpf(table.core_density) * (1 + mp.mpf(delta))
        return mp.nstr(v, 80, strip_zeros=False, min_fixed=-1, max_fixed=-1)


def perturbed_core_density(table, delta: str) -> str:
    """``rho_c (1 + delta)`` as a decimal string."""
    return _fmt(Decimal(table.core_density) * (1 + Decimal(delta)))


# --------------------------------------------------------------------------
# The oracle case list (plan 6.5, with review r3 R2 for the no-slip degree 1).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleVariant:
    """One stored configuration of an oracle case."""

    label: str
    table: object
    physics: object
    disc: object
    #: Frames to store at l = 1 (each one a separate solve; None never appears).
    l1_frames: tuple = ()
    #: Store the raw null vector and the boundary-class ratios at l = 1.
    store_null: bool = False
    #: Source nodes to store (None: every node). smooth_conv keeps the 65 nodes
    #: common to all four refinements, which is what its convergence study needs.
    sources: tuple | None = None
    #: Oracle precision in decimal digits (None: ``required_dps`` per degree). Every
    #: live oracle call on this variant passes it (``ROUGH8_DPS`` for T_rough8).
    dps: int | None = None


@dataclass(frozen=True)
class OracleCase:
    name: str
    degrees: tuple
    variants: tuple
    purpose: str


def _l1_frames(table, physics, store_raw=True):
    from reference_mpmath import frame_options
    opts = frame_options(table, physics)
    frames = []
    if opts["default"] != "refused":
        frames.append(opts["default"])
    for f in opts["offered"] + opts["explicit_only"]:
        if f not in frames and (store_raw or f != "raw"):
            frames.append(f)
    return tuple(frames)


def _variant(label, table, physics=None, disc=None, l1=True, store_raw=True, store_null=False,
             sources=None, dps=None):
    from reference_mpmath import Physics, Discretisation
    physics = physics or Physics()
    disc = disc or Discretisation()
    return OracleVariant(label, table, physics, disc,
                         _l1_frames(table, physics, store_raw) if l1 else (), store_null, sources, dps)


def build_oracle_cases() -> tuple[OracleCase, ...]:
    """The case list of plan 6.5. Built on call (it builds about 40 node tables)."""
    from reference_mpmath import Physics, Discretisation
    full = Physics()
    tala = Physics(compressible=True)
    cases = []
    th = t_hager()
    cases.append(OracleCase("hager4_full_fsfs", (1, 2, 3, 10, 30, 50, 100),
                            (_variant("base", th),), "main reference; split-core CM and raw"))
    th_cc = t_hager(core="consistent")
    cases.append(OracleCase("hager4_full_fsfs_cc", (1, 2, 10),
                            (_variant("base", th_cc, store_null=True),), "core model; G8a, G8b"))
    near = []
    for delta in ("1e-6", "1e-10", "1e-20"):
        store_raw = delta != "1e-20"
        near.append(_variant(f"mc_{delta}", with_core(th_cc, core_mass=consistent_core_mass(th_cc, delta)),
                             store_raw=store_raw))
        near.append(_variant(f"rhoc_{delta}",
                             with_core(th_cc, core_mass=consistent_core_mass(th_cc),
                                       core_density=perturbed_core_density(th_cc, delta)),
                             store_raw=store_raw))
    cases.append(OracleCase("hager4_near_fsfs", (1,), tuple(near), "G8c"))
    cases.append(OracleCase("hager4_full_nsfs", (1, 2, 10, 50),
                            (_variant("base", th, Physics(surface="no_slip")),),
                            "no slip; l = 1 raw by explicit request (review r3 R2)"))
    cases.append(OracleCase("hager4_full_fsns", (1, 2, 10, 50),
                            (_variant("base", th, Physics(cmb="no_slip")),),
                            "no slip; l = 1 raw by explicit request (review r3 R2)"))
    cases.append(OracleCase("hager4_full_nsns", (2, 10),
                            (_variant("base", th, Physics(surface="no_slip", cmb="no_slip")),), "no slip"))
    cases.append(OracleCase("hager4_noself_uniform_fsfs", (1, 2, 10, 50),
                            (_variant("base", th, Physics(self_gravity="none", background="uniform",
                                                          uniform_g="9.81")),), "G-ADOPT levels"))
    cases.append(OracleCase("hager4_noself_hydro_fsfs", (1, 2, 10, 50),
                            (_variant("base", th, Physics(self_gravity="none")),), "gravity level alone"))
    cases.append(OracleCase("hager4_full_tala_fsfs", (1, 2, 10, 50),
                            (_variant("base", th, tala),), "TALA; G8d"))
    cases.append(OracleCase("iso_full_fsfs_w", (2, 10),
                            tuple(_variant(f"rho_w_{w}", t_hager(eta_layers=((None, ISO_ETA_REL),),
                                                                 outer_density=w))
                                  for w in ("0", "1030")), "G17"))
    cases.append(OracleCase("wide5_full_fsfs", (2, 10, 50, 100), (_variant("base", t_wide5()),),
                            "conditioning reference"))
    tp = t_prem()
    cases.append(OracleCase("prem_tala_fsfs", (1, 2, 10, 50), (_variant("base", tp, tala),),
                            "density jumps at pairs, anelastic rule; G8d"))
    cases.append(OracleCase("prem_tala_fsfs_material", (2, 10),
                            (_variant("base", tp, tala, Discretisation(pair_rule="material")),),
                            "size of the pair-rule choice"))
    cases.append(OracleCase("prem_incomp_fsfs", (1, 2, 10, 50), (_variant("base", tp),),
                            "density jumps at pairs, incompressible"))
    tu = t_uniform()
    cases.append(OracleCase("uniform_cc_full_fsfs", (1, 2),
                            tuple(_variant(f"rho_w_{w}", t_uniform(outer_density=w), store_null=True)
                                  for w in ("0", "1030")), "G8a, G8b"))
    tl = t_layers()
    cases.append(OracleCase("layers_cc_fsfs", (1, 2), (_variant("base", tl, store_null=True),), "G8e"))
    cases.append(OracleCase("layers_cc_tala_fsfs", (1, 2), (_variant("base", tl, tala, store_null=True),),
                            "G8e, TALA"))
    cases.append(OracleCase("prem_tala_narrow", (2, 10),
                            tuple(_variant(f"w_{w}", t_prem(narrow_width_m=w), tala)
                                  for w in ("1000", "10", "0.1")), "pair limit, G18a"))
    cases.append(OracleCase("smooth_conv", (2, 10, 30),
                            tuple(_variant(f"N_{n}", t_smooth(n), tala, sources=tuple(range(0, n + 1, n // 64)))
                                  for n in (64, 128, 256, 512)),
                            "discretisation error, diagnostic"))
    # Chunk 2 (c2-forward-core): degrees above 100, high viscosity contrast, a rough
    # viscosity, and the interval mean of a smooth viscosity in the default path.
    cases.append(OracleCase("hager4_full_fsfs_high", (150, 200), (_variant("base", th, l1=False),),
                            "degree table above l = 100 (G2 of chunk 2)"))
    cases.append(OracleCase("wide5_full_fsfs_high", (150, 200), (_variant("base", t_wide5(), l1=False),),
                            "degree table above l = 100 (G2 of chunk 2)"))
    cases.append(OracleCase("stifflid_full_fsfs", (1, 2, 10, 50, 100),
                            (_variant("base", t_stiff_lid(), store_raw=False),),
                            "seven decades in a stiff lid; separates the viscosity-scale rules at l = 1"))
    cases.append(OracleCase("wide10_full_fsfs", (1, 2, 10, 50, 100),
                            (_variant("base", t_wide10(), store_raw=False),), "ten decades of viscosity"))
    cases.append(OracleCase("rough8_full_fsfs", (1, 2, 10, 50, 100),
                            (_variant("base", t_rough8(), store_raw=False, dps=ROUGH8_DPS),),
                            "an independent viscosity at every node over eight decades"))
    cases.append(OracleCase("smooth_incomp_fsfs", (1, 2, 10, 30, 100), (_variant("base", t_smooth(64)),),
                            "interval mean of a smooth viscosity, incompressible, default flags"))
    del tu
    return tuple(cases)


class _LazyCases:
    """``ORACLE_CASES`` is built on first use, so that importing this module stays cheap."""

    _cases = None

    def __iter__(self):
        if _LazyCases._cases is None:
            _LazyCases._cases = build_oracle_cases()
        return iter(_LazyCases._cases)

    def by_name(self, name):
        for c in self:
            if c.name == name:
                return c
        raise KeyError(name)


ORACLE_CASES = _LazyCases()

#: Reduced tables of the quick set (review r2, minor 3).
REDUCED_TABLES = {
    "T_hager_q": lambda: t_hager(reduced=True),
    "T_hager_cc_q": lambda: t_hager(reduced=True, core="consistent"),
    "T_prem_q": lambda: t_prem(reduced=True),
    "T_uniform_q": lambda: t_uniform(reduced=True),
    "T_layers_q": lambda: t_layers(reduced=True),
}
