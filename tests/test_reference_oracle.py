"""Self-tests of the mpmath oracle: exact identities that a wrong equation breaks.

Rule of this file (review r2, minor 3): every identity that needs a live mpmath
computation runs on the reduced tables (``cases.REDUCED_TABLES``, 17 to 25 nodes) in
the quick set and on the full tables in the slow set, with the same tolerance. Tests
on stored arrays run in the quick set.

Precision: every live computation here runs at 60 digits or more
(``reference_mpmath.DPS_FLOOR``), so a tolerance of 1e-35 leaves 25 digits for the
loss in the 3x3 solve and in the products of propagators (review r3, minor 1).
"""

from __future__ import annotations

import functools
import math
from dataclasses import replace

import mpmath as mp
import numpy as np
import pytest

import cases
import reference_mpmath as ro

Physics, Disc = ro.Physics, ro.Discretisation

# ---- tolerances (each with its reference and the cause of the error) ----
# Self-consistency of two mp computations of the same quantity at >= 60 digits.
# Cause: rounding at working precision, amplified by the 3x3 solve.
EXACT_TOL = mp.mpf("1e-35")
# required_dps self-check: kernels at dps and dps + 25 (cause: the truncation of the
# working precision itself, which this test measures).
DPS_TOL = mp.mpf("1e-30")
# Degree-1 identities that hold exactly in the theory of F7 (singular values and
# residuals that are zero up to rounding at 60 digits).
DEG1_TOL = mp.mpf("1e-40")
# Comparison of two different solves of the same member (CM row against a row
# replacement): 1e-30 (cause: the replaced systems lose up to 10 digits more).
DEG1_MEMBER_TOL = mp.mpf("1e-30")
# Stored float64 arrays against a live recomputation: rounding to nearest of the
# same mp value; they are equal unless the recomputation changed.
STORED_TOL = 1e-14
# Exact relations checked on stored float64 arrays: a few ulp.
FLOAT_TOL = 1e-14
# Gravity against lovejx integrate_gravity in float64 (cause: float64 cumsum).
GRAVITY_TOL = 1e-13
# Imbalance of the raw degree-1 frame: the raw frame solves the full system at
# >= 60 digits, so the imbalance is the residual of that solve (measured at most
# 1.6e-60 over every stored raw l = 1 case).
RAW_IMBALANCE_TOL = 1e-50
# Lowest observed order of the interval-mean discretisation on smooth_conv (measured
# 2.0 to 2.3 for l in {2, 10, 30}). The test measures the error against the N = 512
# solution, not against the exact one, so the observed order of an exact power law
# is biased upwards: an exact first-order law gives 1.22 (N = 64 to 128) and 1.58
# (N = 128 to 256), an exact second-order law 2.07 and 2.32. The upper-node rule
# for the viscosity, a first-order rule, gives 1.18 to 1.60 on smooth_conv at
# l = 2 and 30 (NOTES/chunks/c1-foundations/impl-review-physics-r2-evidence/
# first_order.out). 1.8 separates the two rules with
# a margin of about 0.2.
SMOOTH_MIN_ORDER = 1.8

# ---- helpers ----


@functools.lru_cache(maxsize=None)
def rt(name):
    """A reduced table, cached."""
    return cases.REDUCED_TABLES[name]()


def t_iso_q(rho_w):
    return cases.t_hager(reduced=True, eta_layers=((None, cases.ISO_ETA_REL),), outer_density=rho_w)


def mpk(table, physics, disc=None, degrees=(2,), sources=None, dps=None):
    return ro.kernels(table, physics, disc or Disc(), degrees, sources=sources, dps=dps, return_mp=True)


def maxnorm_diff(a, b, scale=None):
    """max |a - b| / max |b| over a list of mp values (``scale`` overrides the
    denominator)."""
    with mp.workdps(120):
        num = max(abs(x - y) for x, y in zip(a, b))
        den = scale if scale is not None else max(abs(y) for y in b)
        if den == 0:
            return num
        return num / den


def compare_kernels(ka, kb, names=ro.SI_OUTPUTS, degrees_idx=None):
    """Worst max-normalised difference over outputs and degrees. An output that is
    zero in both (no-slip velocity, CM geoid) counts as equal."""
    worst = mp.mpf(0)
    n_deg = ka[names[0]].shape[0]
    for a_ in range(n_deg) if degrees_idx is None else degrees_idx:
        for k in names:
            A, B = list(ka[k][a_]), list(kb[k][a_])
            if all(v == 0 for v in B):
                with mp.workdps(120):
                    worst = max(worst, max(abs(v) for v in A))
                continue
            worst = max(worst, maxnorm_diff(A, B))
    return worst


# ---- configurations of the quick set ----

QUICK_CONFIGS = [
    ("T_hager_q", Physics()),
    ("T_hager_q", Physics(surface="no_slip")),
    ("T_hager_q", Physics(cmb="no_slip")),
    ("T_hager_q", Physics(surface="no_slip", cmb="no_slip")),
    ("T_hager_q", Physics(self_gravity="none", background="uniform")),
    ("T_hager_q", Physics(self_gravity="none")),
    ("T_hager_q", Physics(compressible=True)),
    ("T_hager_cc_q", Physics()),
    ("T_prem_q", Physics(compressible=True)),
    ("T_prem_q", Physics()),
    ("T_uniform_q", Physics()),
    ("T_layers_q", Physics(compressible=True)),
]


def _cfg_id(c):
    t, p = c
    flags = [p.self_gravity, p.background, p.surface[:2] + p.cmb[:2], "tala" if p.compressible else "inc"]
    return t + "-" + "-".join(flags)


def raw_offered(table, physics):
    return "raw" in ro.frame_options(table, physics)["offered"]


def default_l1(table, physics):
    return ro.frame_options(table, physics)["default"] != "refused"


# --------------------------------------------------------------------------
# G2: the working precision.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("l", [2, 10])
def test_required_dps_self_check_quick(l):
    t = rt("T_hager_q")
    d = ro.required_dps(l)
    assert compare_kernels(mpk(t, Physics(), degrees=(l,), dps=d),
                           mpk(t, Physics(), degrees=(l,), dps=d + 25)) <= DPS_TOL


@pytest.mark.slow
@pytest.mark.parametrize("l", [2, 50, 100])
def test_g2_full(l):
    """G2 on hager4_full_fsfs at l in {2, 50, 100}."""
    t = cases.t_hager()
    d = ro.required_dps(l)
    worst = compare_kernels(mpk(t, Physics(), degrees=(l,), dps=d), mpk(t, Physics(), degrees=(l,), dps=d + 25))
    print(f"G2 l={l} dps={d}: {mp.nstr(worst, 3)}")
    assert worst <= DPS_TOL


# --------------------------------------------------------------------------
# G4: local against global potential conditions (rows 7 and 8 of CODE-A).
# --------------------------------------------------------------------------


def _g4(table, physics, degrees):
    worst = mp.mpf(0)
    for l in degrees:
        ph = replace(physics, degree_one="raw") if l == 1 else physics
        a = mpk(table, ph, Disc(potential_form="local"), (l,))
        b = mpk(table, ph, Disc(potential_form="global"), (l,))
        worst = max(worst, compare_kernels(a, b))
    return worst


@pytest.mark.parametrize("cfg", QUICK_CONFIGS, ids=_cfg_id)
def test_g4_local_equals_global_quick(cfg):
    name, physics = cfg
    t = rt(name)
    degrees = [2, 10] + ([1] if raw_offered(t, physics) else [])
    assert _g4(t, physics, degrees) <= EXACT_TOL


@pytest.mark.slow
@pytest.mark.parametrize("case", [c.name for c in cases.ORACLE_CASES])
def test_g4_full(case):
    c = cases.ORACLE_CASES.by_name(case)
    for v in c.variants:
        degrees = [l for l in c.degrees if l in (2, 10)]
        # The raw frame at l = 1 where the case stores it: the near-consistent core at
        # delta = 1e-20 is formally a split core, but its raw ratio (2.4e-23) is below
        # the oracle's singularity threshold at 60 digits, so the case list leaves it
        # out (plan 6.5).
        if 1 in c.degrees and "raw" in v.l1_frames:
            degrees.append(1)
        if degrees:
            ph = v.physics
            assert _g4(v.table, ph, degrees) <= EXACT_TOL, v.label


# --------------------------------------------------------------------------
# G5, G6: invariances.
# --------------------------------------------------------------------------


def _scaled_eta(table, factor):
    """The table with every viscosity times ``factor``, exact in decimal."""
    from decimal import Decimal
    return replace(table, eta=tuple(str(Decimal(e) * factor) for e in table.eta))


@pytest.mark.parametrize("compressible", [False, True])
def test_g5_viscosity_scale(compressible):
    """eta -> 7 eta: geoid and topographies unchanged, velocity divided by 7."""
    t = rt("T_hager_q")
    ph = Physics(compressible=compressible)
    a = mpk(t, ph, degrees=(1, 2, 10))
    b = mpk(_scaled_eta(t, 7), ph, degrees=(1, 2, 10))
    assert compare_kernels(b, a, names=("geoid", "surface_topography", "cmb_topography")) <= EXACT_TOL
    with mp.workdps(120):
        for a_ in range(3):
            assert maxnorm_diff([7 * v for v in b["surface_velocity"][a_]], list(a["surface_velocity"][a_])) <= EXACT_TOL


SCALES = [("4500", "1e19"), ("1000", "1e21"), ("1e4", "1e17")]


@pytest.mark.parametrize("compressible", [False, True])
def test_g6_scales_cancel(compressible):
    t = rt("T_hager_q")
    ref = None
    for rs, es in SCALES:
        k = mpk(t, Physics(compressible=compressible, density_scale=rs, viscosity_scale=es), degrees=(1, 2, 10))
        if ref is None:
            ref = k
        else:
            assert compare_kernels(k, ref) <= EXACT_TOL


def test_g6_legacy_surface_topography_scales_with_bgrho():
    """With the legacy topography density (bgrho), the surface topography scales as
    1 / (rho_s - rho_w) and everything else is unchanged (F3, PLAN row 13)."""
    t = rt("T_hager_q")
    base_ph, disc = ro.legacy_setup(1, 1, 0)
    res = {}
    for rs, es in SCALES:
        res[rs] = mpk(t, replace(base_ph, density_scale=rs, viscosity_scale=es), disc, degrees=(1, 2, 10))
    ref = res["4500"]
    for rs, k in res.items():
        assert compare_kernels(k, ref, names=("geoid", "cmbtopo", "surfvel", "gravity")) <= EXACT_TOL
        with mp.workdps(120):
            for a_ in range(3):
                pred = [v * mp.mpf(4500) / mp.mpf(rs) for v in ref["surftopo"][a_]]
                assert maxnorm_diff(list(k["surftopo"][a_]), pred) <= EXACT_TOL


# --------------------------------------------------------------------------
# Degree 1: G8a to G8f, the dipole relation, the frame rule.
# --------------------------------------------------------------------------

G8_TABLES = ["T_uniform_q", "T_hager_cc_q"]


def _g8a(table, uniform):
    rep = ro.degree_one_report(table, Physics(), Disc(), sources=[])
    r = rep["ratios"]
    assert r[("free_slip", "free_slip")]["raw"] < DEG1_TOL
    assert r[("free_slip", "free_slip")]["cm"] > mp.mpf("1e-6")
    m2, d2 = ro._model(table, Physics(), Disc(), 2)
    with mp.workdps(d2):
        S2, _ = m2.system(m2.surface_rows("raw"), m2.cmb_basis())
        assert ro.scaled_ratio(S2) > mp.mpf("1e-6")
    n = rep["null"]
    with mp.workdps(rep["dps"]):
        if uniform:
            assert abs(n["hs_over_N"] - 1) < mp.mpf("1e-30")
            assert abs(n["hc_over_N"] - 1) < mp.mpf("1e-30")
            assert n["velocity_measure"] < mp.mpf("1e-30")
        else:
            assert abs(n["hs_over_N"] - 1) > mp.mpf("1e-3")
            assert abs(n["hc_over_N"] - 1) > mp.mpf("1e-3")
    return rep


@pytest.mark.parametrize("name", G8_TABLES + ["T_uniform_w_q"])
def test_g8a_quick(name):
    if name == "T_uniform_w_q":
        t = cases.t_uniform(reduced=True, outer_density="1030")
    else:
        t = rt(name)
    rep = _g8a(t, uniform=name.startswith("T_uniform"))
    with mp.workdps(rep["dps"]):
        print(name, {k: (mp.nstr(v["raw"], 3), mp.nstr(v["cm"], 3)) for k, v in rep["ratios"].items()})


@pytest.mark.slow
@pytest.mark.parametrize("which", ["uniform", "uniform_w", "hager_cc"])
def test_g8a_full(which):
    t = {"uniform": cases.t_uniform(), "uniform_w": cases.t_uniform(outer_density="1030"),
         "hager_cc": cases.t_hager(core="consistent")}[which]
    _g8a(t, uniform=which.startswith("uniform"))


def _g8b(table, physics=Physics(), sources=None):
    sources = sources or range(1, table.n_nodes - 1)
    rep = ro.degree_one_report(table, physics, Disc(), sources=list(sources))
    per = rep["sources"]
    with mp.workdps(rep["dps"]):
        hs_scale = max(abs(e["cm"]["surface_topography"]) for e in per.values())
        for j, e in per.items():
            assert abs(e["cm"]["degree_one_imbalance"]) < DEG1_TOL, j
            assert e["cm_raw_row_residual"] < DEG1_TOL, j
            # Negative control: the radial row is not part of the dependency.
            rad = e["replace_radial"]
            if rad is not None:
                assert abs(rad["surface_topography"] - e["cm"]["surface_topography"]) > mp.mpf("1e-3") * hs_scale
            else:
                assert e["ratio_replace_radial"] < DEG1_TOL
        for k in ro.SI_OUTPUTS:
            cm = [per[j]["cm"][k] for j in per]
            scale = max(abs(v) for v in cm)
            if scale == 0:
                scale = hs_scale
            for rr in ("replace_potential", "replace_tangential"):
                other = [per[j][rr][k] for j in per]
                assert maxnorm_diff(other, cm, scale) <= DEG1_MEMBER_TOL, (k, rr)
    return rep


@pytest.mark.parametrize("name", G8_TABLES)
@pytest.mark.parametrize("compressible", [False, True])
def test_g8b_quick(name, compressible):
    _g8b(rt(name), Physics(compressible=compressible))


@pytest.mark.slow
@pytest.mark.parametrize("which", ["uniform", "hager_cc"])
def test_g8b_full(which):
    t = {"uniform": cases.t_uniform(), "hager_cc": cases.t_hager(core="consistent")}[which]
    _g8b(t)


DELTAS = ("1e-6", "1e-10", "1e-20")


def _g8c(table_cc, sources):
    ref = mpk(table_cc, Physics(), degrees=(1,), sources=sources)
    errs = {}
    for pert in ("mc", "rhoc"):
        for delta in DELTAS:
            if pert == "mc":
                t = cases.with_core(table_cc, core_mass=cases.consistent_core_mass(table_cc, delta))
            else:
                t = cases.with_core(table_cc, core_mass=cases.consistent_core_mass(table_cc),
                                    core_density=cases.perturbed_core_density(table_cc, delta))
            k = mpk(t, Physics(), degrees=(1,), sources=sources)
            errs[(pert, delta)] = max(maxnorm_diff(list(k[o][0]), list(ref[o][0]))
                                      for o in ("surface_topography", "cmb_topography", "surface_velocity"))
            assert errs[(pert, delta)] <= 100 * mp.mpf(delta), (pert, delta, errs[(pert, delta)])
            if delta != "1e-20":
                # Negative control: the shift rule of revision r1 misses.
                raw = mpk(t, replace(Physics(), degree_one="raw"), degrees=(1,), sources=sources)
                shifted = {o: [h - n for h, n in zip(raw[o][0], raw["geoid"][0])]
                           for o in ("surface_topography", "cmb_topography")}
                miss = max(maxnorm_diff(shifted[o], list(ref[o][0])) for o in shifted)
                assert miss > mp.mpf("1e-3"), (pert, delta, miss)
        ratio = errs[(pert, "1e-10")] / errs[(pert, "1e-20")]
        assert mp.mpf("1e9") <= ratio <= mp.mpf("1e11"), (pert, ratio)
    return errs


def test_g8c_quick():
    t = rt("T_hager_cc_q")
    errs = _g8c(t, list(range(1, t.n_nodes - 1)))
    print({k: mp.nstr(v, 3) for k, v in errs.items()})


@pytest.mark.slow
def test_g8c_full():
    t = cases.t_hager(core="consistent")
    errs = _g8c(t, list(range(1, t.n_nodes - 1, 4)))
    print({k: mp.nstr(v, 3) for k, v in errs.items()})


def _g8d(table, physics, frames, sources):
    worst = mp.mpf(0)
    for fr in frames:
        for j in sources:
            total, scale = ro.momentum_integral(table, replace(physics, degree_one=fr), Disc(), j)
            with mp.workdps(100):
                worst = max(worst, abs(total) / scale)
    return worst


@pytest.mark.parametrize("name", ["T_hager_q", "T_prem_q"])
def test_g8d_quick(name):
    """TALA: the net momentum ``integral rho0 (u_r + 2 u_t) r^2 dr`` is zero in the
    CM and in the raw frame (F7g)."""
    t = rt(name)
    assert _g8d(t, Physics(compressible=True), ("centre_of_mass", "raw"), [3, 9, t.n_nodes - 4]) < mp.mpf("1e-30")


def test_g8d_incompressible_diagnostic():
    """Incompressible flow with a variable rho0 does not conserve mass, so its net
    momentum is not zero (F7g); recorded, not asserted."""
    t = rt("T_hager_q")
    w = _g8d(t, Physics(), ("centre_of_mass",), [3, 9])
    print("incompressible net momentum, relative:", mp.nstr(w, 3))
    assert w > mp.mpf("1e-10")


@pytest.mark.slow
@pytest.mark.parametrize("case", ["hager4_full_tala_fsfs", "prem_tala_fsfs"])
def test_g8d_full(case):
    v = cases.ORACLE_CASES.by_name(case).variants[0]
    n = v.table.n_nodes
    assert _g8d(v.table, v.physics, ("centre_of_mass", "raw"), list(range(1, n - 1, 9))) < mp.mpf("1e-30")


def _pairs(table):
    return [k for k in range(table.n_nodes - 1) if table.radius[k] == table.radius[k + 1]]


def _g8e(table, physics):
    pairs = _pairs(table)
    m, d = ro._model(table, physics, Disc(), 1)
    with mp.workdps(d):
        rho = [mp.mpf(x) for x in table.rho0]
        sheets_up = {k + 1: rho[k] - rho[k + 1] for k in pairs}
        sheets_lo = {k: rho[k] - rho[k + 1] for k in pairs}
        up = ro.solve_state(table, physics, Disc(), 1, sheets_up)
        lo = ro.solve_state(table, physics, Disc(), 1, sheets_lo)
        nv = ro.raw_null_vector(table, physics, Disc())
        N_n = nv["outputs"]["geoid"]
        o = up["outputs"]
        hs = o["surface_topography"] + nv["outputs"]["surface_topography"] / N_n
        hc = o["cmb_topography"] + nv["outputs"]["cmb_topography"] / N_n
        yR = up["yR"] + nv["yR"] / N_n
        assert abs(hs - 1) < DEG1_TOL and abs(hc - 1) < DEG1_TOL
        assert abs(o["geoid"]) == 0
        assert abs(yR[1]) / max(abs(v) for v in yR) < DEG1_TOL
        assert max(abs(nv["hs_over_N"] - 1), abs(nv["hc_over_N"] - 1)) > mp.mpf("1e-3")
        for k in ro.SI_OUTPUTS:
            scale = max(abs(o["surface_topography"]), mp.mpf(1e-300))
            assert abs(up["outputs"][k] - lo["outputs"][k]) <= DEG1_TOL * scale, k


@pytest.mark.parametrize("compressible", [False, True])
def test_g8e_quick(compressible):
    """The rigid translation is the CM solution for the translation sheets plus
    n / N_1(n) (F7b), on four constant-density layers."""
    _g8e(rt("T_layers_q"), Physics(compressible=compressible))


@pytest.mark.slow
@pytest.mark.parametrize("compressible", [False, True])
def test_g8e_full(compressible):
    _g8e(cases.t_layers(), Physics(compressible=compressible))


def _g8f(table, physics):
    pairs_free = (physics.surface, physics.cmb) == ("free_slip", "free_slip")
    assert pairs_free
    n = table.n_nodes
    sheet_nodes = [n // 4, n // 2, 3 * n // 4]
    sv = ro.force_identity(table, physics, Disc(), sheet_nodes)
    with mp.workdps(100):
        assert sv[2] / sv[0] < DEG1_TOL
    nv = ro.raw_null_vector(table, physics, Disc())
    N_n = nv["outputs"]["geoid"]
    for surface, cmb in (("no_slip", "free_slip"), ("free_slip", "no_slip"), ("no_slip", "no_slip")):
        ph = replace(physics, surface=surface, cmb=cmb)
        for j in sheet_nodes:
            raw = ro.solve_state(table, ph, Disc(), 1, j, frame="raw")
            cm = ro.solve_state(table, physics, Disc(), 1, j, frame="centre_of_mass")
            with mp.workdps(raw["dps"]):
                yR, yc = raw["yR"], raw["yc"]
                big = max(abs(v) for v in yR)
                if surface == "no_slip" and cmb == "free_slip":
                    assert abs(yR[3]) / big < DEG1_TOL
                elif cmb == "no_slip" and surface == "free_slip":
                    assert abs(yc[3]) / big < DEG1_TOL
                else:
                    R, rc = mp.mpf(table.radius[-1]), mp.mpf(table.radius[0])
                    a, b = R * yR[3], rc * yc[3]
                    assert abs(a - b) / (abs(a) + abs(b)) < DEG1_TOL
                if not (surface == "no_slip" and cmb == "no_slip"):
                    c = raw["outputs"]["geoid"] / N_n
                    for k in ("surface_topography", "cmb_topography"):
                        pred = cm["outputs"][k] + c * nv["outputs"][k]
                        assert abs(raw["outputs"][k] - pred) / abs(raw["outputs"][k]) < DEG1_TOL, (surface, cmb, k)


@pytest.mark.parametrize("name,compressible", [("T_hager_cc_q", False), ("T_layers_q", True)])
def test_g8f_quick(name, compressible):
    """Review r3 R3: the force identity on the four-dimensional CMB space; a no-slip
    anchor carries no force, and the raw no-slip solution is the free-slip CM
    solution plus c n."""
    _g8f(rt(name), Physics(compressible=compressible))


@pytest.mark.slow
@pytest.mark.parametrize("which", ["hager_cc", "layers_tala"])
def test_g8f_full(which):
    if which == "hager_cc":
        _g8f(cases.t_hager(core="consistent"), Physics())
    else:
        _g8f(cases.t_layers(), Physics(compressible=True))


def _g8f_uniform(table):
    """Uniform density, consistent core, no-slip classes: the raw system is singular
    and the right-hand side of a unit sheet is not compatible (no solution)."""
    j = table.n_nodes // 2
    out = {}
    for surface, cmb in (("no_slip", "free_slip"), ("free_slip", "no_slip"), ("no_slip", "no_slip")):
        ph = Physics(surface=surface, cmb=cmb)
        S, rhs, info = ro.boundary_system(table, ph, Disc(), 1, j)
        with mp.workdps(info["dps"]):
            s = [max(abs(S[a_, b_]) for b_ in range(3)) for a_ in range(3)]
            Sn = mp.matrix([[S[a_, b_] / s[a_] for b_ in range(3)] for a_ in range(3)])
            bn = [rhs[a_] / s[a_] for a_ in range(3)]
            U, sv, V = mp.svd_r(Sn)
            w = [U[a_, 2] for a_ in range(3)]
            ratio = sv[2] / sv[0]
            compat = abs(sum(x * y for x, y in zip(w, bn))) / (mp.norm(mp.matrix(w)) * mp.norm(mp.matrix(bn)))
            assert ratio < DEG1_TOL, (surface, cmb)
            assert compat > mp.mpf("1e-3"), (surface, cmb)
            out[(surface, cmb)] = (ratio, compat)
    return out


def test_g8f_uniform_quick():
    out = _g8f_uniform(rt("T_uniform_q"))
    print({k: (mp.nstr(a, 3), mp.nstr(b, 3)) for k, (a, b) in out.items()})


@pytest.mark.slow
def test_g8f_uniform_full():
    _g8f_uniform(cases.t_uniform())


def _dipole(table, physics, frame, sources):
    """N_1 M = D in the raw frame, D = 0 in the CM frame. M = g(R) R^2 / G, and D is
    the dipole moment of all sheets: the load, the surface sheet -sigma_rr/g (built
    from y3(R), review r3 minor 3) and the CMB sheet (sigma_rr + rho_c phi)/g."""
    ph = replace(physics, degree_one=frame)
    m, d = ro._model(table, ph, Disc(), 1)
    for j in sources:
        st = ro.solve_state(table, ph, Disc(), 1, j)
        with mp.workdps(d):
            yR, yc = st["yR"], st["yc"]
            R, rc, rs = m.R, m.rc, m.r[j]
            sig_s = -(m.eta_s * yR[2] / R) / m.g[-1]
            rho_c_term = m.rho_c * m.eta_s * yc[4] / (m.rho_s * rc) if m.full else 0
            sig_c = (m.eta_s * yc[2] / rc + rho_c_term) / m.g[0]
            D = 4 * m.pi / 3 * (rs ** 3 + R ** 3 * sig_s + rc ** 3 * sig_c)
            load = 4 * m.pi / 3 * rs ** 3
            M = m.g[-1] * R ** 2 / m.G
            N = st["outputs"]["geoid"]
            if frame == "centre_of_mass":
                assert abs(D) / load < EXACT_TOL
            else:
                assert abs(N * M - D) / load < EXACT_TOL


@pytest.mark.parametrize("name,physics,frame", [
    ("T_hager_q", Physics(), "raw"),
    ("T_hager_q", Physics(), "centre_of_mass"),
    ("T_hager_cc_q", Physics(), "centre_of_mass"),
    ("T_hager_q", Physics(self_gravity="none", background="uniform"), "raw"),
    ("T_hager_q", Physics(compressible=True), "centre_of_mass"),
])
def test_dipole_relation(name, physics, frame):
    t = rt(name)
    _dipole(t, physics, frame, [1, 7, t.n_nodes - 2])


# The static frame rule: every combination of the flags and the core model, every
# requested frame. A request must succeed exactly when F7(f) with review r3 R2
# offers it.
_ALL = [(sg, bg, s, c, core) for sg in ("full", "none") for bg in ("hydrostatic", "uniform")
        for s in ("free_slip", "no_slip") for c in ("free_slip", "no_slip") for core in ("consistent", "split")]


def _expected_offer(sg, bg, s, c, core):
    if sg == "full" and bg == "hydrostatic":
        if s == c == "free_slip":
            return ("centre_of_mass",) if core == "consistent" else ("centre_of_mass", "raw")
        return () if core == "consistent" else ("raw",)
    return ("raw",)


@pytest.mark.parametrize("combo", _ALL, ids=lambda c: "-".join(c))
@pytest.mark.parametrize("frame", ["raw", "centre_of_mass", "legacy_geoid_zero"])
def test_frame_rule(combo, frame):
    sg, bg, s, c, core = combo
    t = rt("T_hager_q") if core == "split" else rt("T_hager_cc_q")
    ph = Physics(self_gravity=sg, background=bg, surface=s, cmb=c, degree_one=frame)
    offered = _expected_offer(*combo)
    ok = frame in offered or (frame == "legacy_geoid_zero" and "raw" in offered)
    if ok:
        assert ro.degree_one_rule(t, ph) == frame
    else:
        with pytest.raises(ValueError):
            ro.degree_one_rule(t, ph)


@pytest.mark.parametrize("combo", [c for c in _ALL if c[0] == "full" and c[1] == "hydrostatic"
                                   and (c[2], c[3]) != ("free_slip", "free_slip")], ids=lambda c: "-".join(c))
def test_refused_default_degree_one(combo):
    """FULL with hydrostatic g and a no-slip boundary: degree 1 is refused by default
    for both core models (review r3 R2); the other degrees are computed."""
    sg, bg, s, c, core = combo
    t = rt("T_hager_q") if core == "split" else rt("T_hager_cc_q")
    ph = Physics(surface=s, cmb=c)
    assert ro.degree_one_rule(t, ph) == "refused"
    with pytest.raises(ValueError, match="refused"):
        ro.kernels(t, ph, Disc(), (1,), sources=[3])
    ro.kernels(t, ph, Disc(), (2,), sources=[3])


def test_explicit_raw_on_consistent_number_core_is_singular():
    """core_mass given as a number equal to the consistent value selects the split
    rule; an explicit raw request then meets a singular system and the solver
    refuses it (review r3, minor 5)."""
    t = rt("T_hager_cc_q")
    t2 = cases.with_core(t, core_mass=cases.consistent_core_mass(t))
    with pytest.raises(ArithmeticError, match="singular"):
        ro.kernels(t2, Physics(degree_one="raw"), Disc(), (1,), sources=[3])


# --------------------------------------------------------------------------
# Gravity, G16, G17, G18.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["hager", "prem", "layers"])
def test_g19_gravity_against_lovejx(which):
    profile = pytest.importorskip("lovejx.lovenumbers.profile")
    units_mod = pytest.importorskip("lovejx.lovenumbers.units")
    t = {"hager": cases.t_hager(), "prem": cases.t_prem(), "layers": cases.t_layers()}[which]
    g_mp = ro.gravity_at_nodes(t, Physics())
    r = np.array([float(x) for x in t.radius])
    rho = np.array([float(x) for x in t.rho0])
    mc = float(mp.mpf(t.core_mass)) if t.core_mass is not None else 4 * math.pi / 3 * r[0] ** 3 * float(t.core_density)
    rho_core = mc / (4 * math.pi / 3 * r[0] ** 3)
    # A filled core below the CMB: a node at r = 0 and a pair at r_c, so that the
    # core shell has the mean density M_c / (4 pi r_c^3 / 3) and the pair adds no mass.
    rr = np.concatenate([[0.0, r[0]], r])
    dd = np.concatenate([[rho_core, rho_core], rho])
    g = np.asarray(profile.integrate_gravity(rr, dd, units_mod.Units()))[2:]
    rel = np.max(np.abs(g - np.array([float(x) for x in g_mp])) / np.array([float(x) for x in g_mp]))
    assert rel <= GRAVITY_TOL, rel


G16_CONFIGS = QUICK_CONFIGS + [("T_hager_q_w", Physics())]


@pytest.mark.parametrize("cfg", G16_CONFIGS, ids=_cfg_id)
def test_g16_end_nodes_quick(cfg):
    """F19 to 1e-35: a sheet at an end node is compensated statically, for every
    configuration, degree and frame."""
    name, physics = cfg
    t = cases.t_hager(reduced=True, outer_density="1030") if name == "T_hager_q_w" else rt(name)
    frames = [None]
    if default_l1(t, physics):
        frames = ro.frame_options(t, physics)["offered"]
    n = t.n_nodes
    with mp.workdps(120):
        rho_w = mp.mpf(t.outer_density)
        expect_s = -1 / (mp.mpf(t.rho0[-1]) - rho_w)
        expect_c = 1 / (mp.mpf(t.rho0[0]) - mp.mpf(t.core_density))
    for l in (1, 2, 10):
        for fr in (frames if l == 1 else [None]):
            if l == 1 and fr is None:
                continue
            ph = physics if fr is None else replace(physics, degree_one=fr)
            k = mpk(t, ph, degrees=(l,), sources=[0, n - 1])
            with mp.workdps(120):
                assert abs(k["surface_topography"][0][1] / expect_s - 1) < EXACT_TOL
                assert abs(k["cmb_topography"][0][0] / expect_c - 1) < EXACT_TOL
                for o in ("geoid", "surface_velocity"):
                    assert abs(k[o][0][0]) < EXACT_TOL * abs(expect_s)
                    assert abs(k[o][0][1]) < EXACT_TOL * abs(expect_s)


def test_g16_end_nodes_stored(load_oracle):
    """F19 on every stored oracle array (float64: a few ulp)."""
    for case in cases.ORACLE_CASES:
        O = load_oracle(case.name)
        for v in case.variants:
            t = v.table
            rho_w = float(t.outer_density)
            es = -1 / (float(mp.mpf(t.rho0[-1])) - rho_w)
            ec = 1 / (float(mp.mpf(t.rho0[0])) - float(mp.mpf(t.core_density)))
            for key in [k for k in O if k.startswith(v.label + "/") and k.endswith("/surface_topography")]:
                pre = key[: -len("surface_topography")]
                src = O[pre + "sources"] if pre + "sources" in O else np.arange(t.n_nodes)
                st, ct = O[pre + "surface_topography"], O[pre + "cmb_topography"]
                last = int(np.where(src == t.n_nodes - 1)[0][0])
                first = int(np.where(src == 0)[0][0])
                assert abs(st[last] / es - 1) < FLOAT_TOL, key
                assert abs(ct[first] / ec - 1) < FLOAT_TOL, key
                for o in ("geoid", "surface_velocity"):
                    assert abs(O[pre + o][first]) < FLOAT_TOL * abs(es)
                    assert abs(O[pre + o][last]) < FLOAT_TOL * abs(es)


def _g17(make_table, l, physics):
    k0 = mpk(make_table("0"), physics, degrees=(l,))
    k1 = mpk(make_table("1030"), physics, degrees=(l,))
    assert compare_kernels(k1, k0, names=("geoid", "cmb_topography", "surface_velocity")) <= EXACT_TOL
    with mp.workdps(120):
        # mpf(str) rounds to the working precision, so it is made inside the context.
        rho_top = mp.mpf(make_table("0").rho0[-1])
        a = [rho_top * h for h in k0["surface_topography"][0]]
        b = [(rho_top - 1030) * h + 1030 * n for h, n in zip(k1["surface_topography"][0], k1["geoid"][0])]
        assert maxnorm_diff(b, a) <= EXACT_TOL


@pytest.mark.parametrize("l", [2, 10])
def test_g17_rho_w_quick(l):
    """F17: geoid, CMB topography and velocity do not depend on rho_w, and
    (rho0(R) - rho_w) h_s + rho_w N does not either."""
    _g17(t_iso_q, l, Physics())


def test_g17_rho_w_degree_one_cm():
    _g17(lambda w: cases.t_uniform(reduced=True, outer_density=w), 1, Physics())


def test_g17_stored(load_oracle):
    """The same on the stored float64 arrays of iso_full_fsfs_w and uniform_cc_full_fsfs."""
    O = load_oracle("iso_full_fsfs_w")
    rho_top = float(cases.t_hager().rho0[-1])
    for l in (2, 10):
        a, b = (f"rho_w_0/l{l}/raw/", f"rho_w_1030/l{l}/raw/")
        for o in ("geoid", "cmb_topography", "surface_velocity"):
            np.testing.assert_allclose(O[b + o], O[a + o], rtol=0, atol=FLOAT_TOL * np.max(np.abs(O[a + o])))
        lhs = (rho_top - 1030) * O[b + "surface_topography"] + 1030 * O[b + "geoid"]
        rhs = rho_top * O[a + "surface_topography"]
        assert np.max(np.abs(lhs - rhs)) <= 1e-13 * np.max(np.abs(rhs))


def _pair_equal(table, physics, degrees):
    pairs = _pairs(table)
    srcs = sorted({k for p in pairs for k in (p, p + 1)})
    for l in degrees:
        ph = physics if l != 1 else replace(physics, degree_one=ro.frame_options(table, physics)["default"])
        k = mpk(table, ph, degrees=(l,), sources=srcs)
        with mp.workdps(120):
            for o in ro.SI_OUTPUTS:
                vals = dict(zip(srcs, k[o][0]))
                scale = max(abs(v) for v in k["surface_topography"][0])
                for p in pairs:
                    assert abs(vals[p] - vals[p + 1]) <= EXACT_TOL * scale, (l, o, p)


def test_g18b_pairs_without_density_jump():
    """The 100 km pair of T_hager carries no density jump: equal kernels at its two
    nodes (and at the other viscosity-only pairs)."""
    _pair_equal(rt("T_hager_q"), Physics(compressible=True), (1, 2, 10))


def test_g18b_prem_pairs_tala():
    """Under TALA the pair step is I + K t and K jump = 0, so the kernels at the two
    nodes of every pair are equal, with or without a density jump."""
    _pair_equal(rt("T_prem_q"), Physics(compressible=True), (1, 2, 10))


def test_g18c_prem_pairs_incompressible():
    _pair_equal(rt("T_prem_q"), Physics(), (1, 2, 10))


def test_g18a_pair_is_limit_of_narrow_interval(load_oracle):
    """The difference to the pair decreases linearly with the width w of a narrow
    interval that replaces the 670 km pair (F18 reason 2)."""
    P = load_oracle("prem_tala_fsfs")
    Nw = load_oracle("prem_tala_narrow")
    for l in (2, 10):
        diffs = {}
        for w in ("1000", "10", "0.1"):
            worst = 0.0
            for o in ro.SI_OUTPUTS:
                ref = P[f"base/l{l}/raw/{o}"]
                got = Nw[f"w_{w}/l{l}/raw/{o}"]
                worst = max(worst, float(np.max(np.abs(got - ref)) / np.max(np.abs(ref))))
            diffs[w] = worst
        r1, r2 = diffs["1000"] / diffs["10"], diffs["10"] / diffs["0.1"]
        print(f"G18a l={l}: {diffs}, ratios {r1:.1f} {r2:.1f}")
        assert 80 <= r1 <= 120 and 80 <= r2 <= 120, (l, diffs)


def test_pair_rule_size(load_oracle):
    """Size of the choice between the anelastic and the material pair rule
    (diagnostic for Q6)."""
    A = load_oracle("prem_tala_fsfs")
    B = load_oracle("prem_tala_fsfs_material")
    for l in (2, 10):
        d = {o: float(np.max(np.abs(B[f"base/l{l}/raw/{o}"] - A[f"base/l{l}/raw/{o}"]))
                      / np.max(np.abs(A[f"base/l{l}/raw/{o}"]))) for o in ro.SI_OUTPUTS}
        print(f"pair rule l={l}:", {k: f"{v:.2e}" for k, v in d.items()})
        assert max(d.values()) > 0


def test_smooth_convergence_diagnostic(load_oracle):
    """Discretisation error of the interval-mean model at the 65 common nodes of
    T_smooth(N), N = 64..512, against N = 512 (equations.md 12, audit row 3).

    The viscosity and density of an interval are the means of the two end nodes, a
    second-order rule. The observed order between N = 64, 128 and 256 must be at
    least ``SMOOTH_MIN_ORDER``: a first-order rule would also give errors that
    decrease with N, so the decrease alone is not a test of the rule."""
    O = load_oracle("smooth_conv")
    for l in (2, 10, 30):
        e = {}
        for n in (64, 128, 256):
            e[n] = max(float(np.max(np.abs(O[f"N_{n}/l{l}/raw/{o}"] - O[f"N_512/l{l}/raw/{o}"]))
                             / np.max(np.abs(O[f"N_512/l{l}/raw/{o}"]))) for o in ro.SI_OUTPUTS)
        orders = (math.log2(e[64] / e[128]), math.log2(e[128] / e[256]))
        print(f"smooth_conv l={l}: errors {e}, observed orders {orders[0]:.2f} {orders[1]:.2f}")
        assert e[64] > e[128] > e[256]
        assert min(orders) >= SMOOTH_MIN_ORDER, (l, e, orders)


# --------------------------------------------------------------------------
# Stored values against a live recomputation.
# --------------------------------------------------------------------------


def _stored_vs_live(case_name, degrees, load_oracle, labels=None):
    """Recompute the stored kernels of ``case_name`` at ``degrees`` (every stored
    frame at l = 1) and compare them with the stored float64 arrays to
    ``STORED_TOL``. ``labels`` restricts the variants (None: all)."""
    c = cases.ORACLE_CASES.by_name(case_name)
    O = load_oracle(case_name)
    for v in c.variants:
        if labels is not None and v.label not in labels:
            continue
        for l in [d for d in c.degrees if d in degrees]:
            frames = v.l1_frames if l == 1 else (None,)
            for fr in frames:
                ph = v.physics if fr is None else replace(v.physics, degree_one=fr)
                k = ro.kernels(v.table, ph, v.disc, (l,), sources=v.sources)
                pre = f"{v.label}/l{l}/{fr or 'raw'}/"
                for o in ro.SI_OUTPUTS:
                    ref = O[pre + o]
                    scale = np.max(np.abs(ref)) or 1.0
                    assert np.max(np.abs(k[o][0] - ref)) <= STORED_TOL * scale, (case_name, v.label, l, fr, o)


@pytest.mark.parametrize("case", ["hager4_full_fsfs", "hager4_full_fsfs_cc"])
def test_stored_equals_live_quick(case, load_oracle):
    """Stale-data guard on two cases at l in {1, 2}."""
    _stored_vs_live(case, (1, 2), load_oracle)


def test_stored_equals_live_smooth_conv(load_oracle):
    """Stale-data guard on smooth_conv, N = 64, l = 2: the only stored case whose
    viscosity changes inside every interval, so the only quick test that fails when
    the interval rule of the viscosity changes (audit row 3)."""
    _stored_vs_live("smooth_conv", (2,), load_oracle, labels=("N_64",))


@pytest.mark.slow
@pytest.mark.parametrize("case", [c.name for c in cases.ORACLE_CASES])
def test_stored_equals_live_full(case, load_oracle):
    """Stale-data guard: every stored degree of every case against a live
    recomputation (a full regeneration takes about 3 CPU-minutes)."""
    _stored_vs_live(case, tuple(cases.ORACLE_CASES.by_name(case).degrees), load_oracle)


def test_degree_one_stored_records(load_oracle):
    """Every stored l = 1 case has its frame, the imbalance, and the ratio of the
    solved system; the consistent free bodies store the null vector."""
    for case in cases.ORACLE_CASES:
        if 1 not in case.degrees:
            continue
        O = load_oracle(case.name)
        for v in case.variants:
            for fr in v.l1_frames:
                pre = f"{v.label}/l1/{fr}/"
                assert str(O[pre + "degree_one_frame"]) == fr
                assert np.all(np.isfinite(O[pre + "degree_one_imbalance"]))
                assert O[pre + "degree_one_ratio"] > 1e-15
                if fr == "raw":
                    # The raw frame solves the full system: no imbalance beyond the
                    # residual of the 60-digit solve.
                    assert np.max(np.abs(O[pre + "degree_one_imbalance"])) <= RAW_IMBALANCE_TOL, (case.name, v.label)
            if v.store_null:
                assert O[f"{v.label}/l1/null/raw_ratio"] < 1e-40
