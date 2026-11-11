"""The forward solver against the mpmath oracle, and its invariants (plan 7.4; gates
G1, G2, G4).

Reference: the oracle (tests/reference_mpmath.py), stored in tests/golden/oracle at 60
to 250 digits and rounded to float64, or computed live for the configurations that are
not stored. A row whose reference is an exact zero (the centre-of-mass geoid at
``l = 1``) is compared by exact equality, every other row by the max-normalised error
``e_l`` of CONVENTIONS 10 (``adapters.assert_rows_close``).

Each (table, configuration, lmax) is solved once per worker process
(``functools.lru_cache``), and the tests compare rows of these solves.
"""

from __future__ import annotations

import functools

import mpmath as mp
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import cases  # noqa: E402
import reference_mpmath as ro  # noqa: E402
from adapters import SI_OUTPUTS, assert_rows_close, nodewise_error, profile_from_table, row_errors  # noqa: E402

import stokesjx as sj  # noqa: E402
from stokesjx import propagate  # noqa: E402
from stokesjx.profile import integrate_gravity, mantle_mass  # noqa: E402

# ---- tolerances (CONVENTIONS 9) ----
# The requirement of the task is 1e-10 against the exact answer; the tests assert 1e-12
# so that a loss of two digits fails a test (review r1 of the plan, issue 6).
#
# Kernels against the oracle, e_l, every case except the two high-contrast cases.
# Measured with the package and the largest-viscosity scale (impl-evidence/d1_table.out):
# at most 8.5e-14 on the stored moderate cases (hager4 geoid, l = 2), 3.1e-13 for the
# raw frame at l = 1 on a split core, 1.3e-14 on the eight-decade rough table. Cause:
# the rounding of the sweeps, which grows slowly with the viscosity contrast (D1).
ORACLE_TOL = 1e-12
# degree_one_imbalance, absolute, every case except the two high-contrast cases. The
# imbalance is a ratio of two forces of order one (eq. 10.6), computed from a small
# difference of terms of order one, so float64 gives it an absolute error that does
# not scale with its size: measured 1.9e-15 to 9.6e-15 on the stored CM keys (4.1e-15 on
# the rough table). A relative gate cannot pass where the reference is 8.6e-8, 8.6e-12
# or about 1e-60 (near-consistent and consistent cores; review r1, B1).
IMBALANCE_ATOL = 1e-12
# e_l on stifflid_full_fsfs and wide10_full_fsfs, every stored degree. A factor 10 below
# the requirement. Degree 1 in the CM frame is the least accurate degree on a thin stiff
# lid: measured with the package 1.2e-12 (CMB topography) at l = 1 on the stiff lid and
# at most 2.1e-13 at l in {2, 10, 50, 100}; on wide10 at most 3.6e-13 (geoid, l = 10).
# With the geometric mean as the viscosity scale the velocity kernel at l = 1 on the
# stiff lid has 1.7e-10, which this constant fails: the case detects the scale rule of
# D1. Cause: the rounding of the sweeps; it does not grow with the viscosity of the lid
# (D1, measurement 4). Logged in NOTES/TOLERANCES.md.
CONTRAST_TOL = 1e-11
# degree_one_imbalance, absolute, on the same two cases: measured 4.9e-13 (stiff lid)
# and 6.4e-14 (wide10) with the package; same cause as CONTRAST_TOL. Logged.
IMBALANCE_ATOL_CONTRAST = 1e-11
# The "no loss with degree" criterion at l in {50, 100, 150, 200} of the lmax = 200
# solves (measured at most 9.2e-15).
HIGH_DEGREE_TOL = 1e-12
# Rows of the lmax = 50, 100 and 200 solves (Pade order 9 with 0, 1, 2 squarings)
# against each other, e_l. Cause: the rounding of a different number of squarings.
LMAX_CONSISTENCY_TOL = 1e-12
# End-node values of eq. 7.2, relative to the largest entry of the row. Cause: the
# rounding of the node solve at a node where the sheet is compensated without flow.
END_NODE_TOL = 1e-12
# Invariances of the exact problem (eq. 11): eta -> 7 eta, rho_s, the viscosity scale.
# Two float64 solves that differ in rounding only (in the Pade plan for rho_s).
SCALE_TOL = 1e-12
# The water load (F17, eq. 8.3). The water sheet follows the total potential, so the
# geoid, the CMB topography and the velocity do not depend on rho_w, and neither does
# (rho_top - rho_w) h_s + rho_w phi(R) / g(R), with phi(R) the total potential at the
# surface. At l >= 2, and at l = 1 on a consistent core, phi(R) / g(R) is the geoid
# kernel N. At l = 1 in the CM frame of a split core it is N + R A / g(R), with the
# applied uniform field A of eq. 10.6: the geoid kernel is the masked self part (zero)
# there (equations.md 10(e)). Measured on the prototypes: geoid, h_c, velocity equal bit
# for bit; the combinations 2.2e-16 to 4.4e-16; the combination with N at l = 1 on a
# split core 1.2e-3, which is the applied field and not an error. Cause of the
# measured values: the rounding of one subtraction of two solves.
RHO_W_TOL = 1e-12
# The two nodes of a pair are sources at the same radius with the same gravity; their
# kernels are equal up to the rounding of two node solves (G18b of chunk 1).
PAIR_TOL = 1e-13
# Precision of the 250-digit oracle on T_rough8: ROUGH8_DPS against ROUGH8_DPS + 70
# digits, max-normalised. The same limit as DPS_TOL of tests/test_reference_oracle.py;
# measured 2.4e-232 to 9.9e-163 (plan-review-r3-evidence/r3_g4_dps250.out). Cause: the
# truncation of the working precision, which the test measures.
ROUGH8_DPS_TOL = 1e-30

HIGH_CONTRAST = ("stifflid_full_fsfs", "wide10_full_fsfs")
FRAMES = {"centre_of_mass": sj.DegreeOneFrame.CENTRE_OF_MASS, "raw": sj.DegreeOneFrame.RAW,
          "default": sj.DegreeOneFrame.DEFAULT}

# lmax of the solve of each stored default-flag case. Cases that share a table share
# the solve (hager4 and wide5 at lmax = 200 for the degree table, G2).
STORED_LMAX = {
    "hager4_full_fsfs": 200, "hager4_full_fsfs_cc": 10, "hager4_near_fsfs": 1, "wide5_full_fsfs": 200,
    "prem_incomp_fsfs": 50, "uniform_cc_full_fsfs": 2, "layers_cc_fsfs": 2, "iso_full_fsfs_w": 10,
    "hager4_full_fsfs_high": 200, "wide5_full_fsfs_high": 200, "stifflid_full_fsfs": 100,
    "wide10_full_fsfs": 100, "rough8_full_fsfs": 100, "smooth_incomp_fsfs": 100,
}
# The near-consistent cores store raw keys, which are not compared: the raw system is
# nearly singular there (raw ratio 2.5e-13 at delta = 1e-10), so float64 cannot solve it,
# and the rule never defaults to raw (plan 7.4; measured 1.6e-8 to 3.7e-4).
SKIP_RAW = ("hager4_near_fsfs",)


def _variant(case_name, label):
    c = cases.ORACLE_CASES.by_name(case_name)
    return next(v for v in c.variants if v.label == label)


@functools.lru_cache(maxsize=None)
def _solve_table(table, lmax: int, frame: str = "default", formulation=sj.Formulation(), units=sj.Units(),
                 rule="largest", surface_density=None):
    """The kernels of an oracle table as numpy arrays, cached per worker."""
    p = profile_from_table(table, surface_density=surface_density)
    cfg = sj.StokesConfig(lmax=lmax, degree_one=FRAMES[frame], formulation=formulation, units=units)
    K = propagate.solve(p, cfg, _reference_viscosity=rule)
    out = {o: np.asarray(getattr(K, o)) for o in SI_OUTPUTS}
    out["degree_one_imbalance"] = None if K.degree_one_imbalance is None else np.asarray(K.degree_one_imbalance)
    out["degrees"] = K.degrees
    out["frame"] = K.degree_one_frame
    out["radius"] = np.asarray(K.radius)
    return out


def _check_keys(case_name, label, l, frame, sol, O, tol, imb_tol):
    """Compare one stored (variant, degree, frame) key with a solve; returns the worst e_l."""
    key = f"{label}/l{l}/{frame}/"
    k = sol["degrees"].index(l)
    worst = 0.0
    for o in SI_OUTPUTS:
        ref = O[key + o]
        worst = max(worst, assert_rows_close(sol[o][k], ref, tol, f"{case_name} {key}{o}"))
        # The node-wise metric of CONVENTIONS 10 is recorded, not asserted (end nodes of
        # the geoid and the velocity masked: the oracle stores residues of 1e-60 there).
        nw = nodewise_error(sol[o][k][1:-1] if o in ("geoid", "surface_velocity") else sol[o][k],
                            ref[1:-1] if o in ("geoid", "surface_velocity") else ref)
        print(f"{case_name} {key}{o}: e_l {row_errors(sol[o][k], ref)[0]:.2e}, node-wise {nw:.2e}")
    if l == 1:
        if frame == "centre_of_mass":
            assert np.all(sol["geoid"][k] == 0) and np.all(O[key + "geoid"] == 0)
            err = np.max(np.abs(sol["degree_one_imbalance"] - O[key + "degree_one_imbalance"]))
            assert err <= imb_tol, (case_name, key, err)
        else:
            assert np.all(sol["degree_one_imbalance"] == 0)
    return worst


# Tests that solve the same (table, lmax) run on one xdist worker (``--dist
# loadgroup`` in pyproject.toml), so that each solve compiles once in the quick set
# (plan section 7, G10).
_GROUP = {"hager4_full_fsfs": "hager4_full", "hager4_full_fsfs_high": "hager4_full",
          "wide5_full_fsfs": "wide5_full", "wide5_full_fsfs_high": "wide5_full",
          "hager": "hager_q", "prem": "prem_q"}


def _grouped(names):
    """Parameters with the ``xdist_group`` mark of their table, where it has one."""
    return [pytest.param(n, marks=pytest.mark.xdist_group(_GROUP[n])) if n in _GROUP else n for n in names]


STORED_CASES = _grouped(list(STORED_LMAX))


@pytest.mark.parametrize("case_name", STORED_CASES)
def test_stored_oracle_case(case_name, load_oracle):
    """G1: every stored degree, frame and variant of the default-flag cases (plan 7.4)."""
    c = cases.ORACLE_CASES.by_name(case_name)
    O = load_oracle(case_name)
    tol = CONTRAST_TOL if case_name in HIGH_CONTRAST else ORACLE_TOL
    imb_tol = IMBALANCE_ATOL_CONTRAST if case_name in HIGH_CONTRAST else IMBALANCE_ATOL
    lmax = STORED_LMAX[case_name]
    for v in c.variants:
        sol = _solve_table(v.table, lmax)
        for l in c.degrees:
            if l > 1:
                _check_keys(case_name, v.label, l, "raw", sol, O, tol, imb_tol)
        if 1 in c.degrees:
            assert sol["frame"] is sj.DegreeOneFrame.CENTRE_OF_MASS
            _check_keys(case_name, v.label, 1, "centre_of_mass", sol, O, tol, imb_tol)
            if "raw" in v.l1_frames and case_name not in SKIP_RAW:
                raw = _solve_table(v.table, 1, "raw")
                assert raw["frame"] is sj.DegreeOneFrame.RAW
                _check_keys(case_name, v.label, 1, "raw", raw, O, tol, imb_tol)


@pytest.mark.parametrize("which", [pytest.param("hager4", marks=pytest.mark.xdist_group("hager4_full")),
                                   pytest.param("wide5", marks=pytest.mark.xdist_group("wide5_full"))])
def test_degree_table_and_lmax_consistency(which, load_oracle):
    """G2: one lmax = 200 solve against every stored degree (1 to 100, 150, 200), with
    the no-loss criterion at l in {50, 100, 150, 200}; the rows of the lmax = 50, 100
    and 200 solves agree (Pade order 9 with 0, 1 and 2 squarings: the plan of the
    benchmark size S3 has a value test)."""
    base, high = f"{which}_full_fsfs", f"{which}_full_fsfs_high"
    table = cases.t_hager() if which == "hager4" else cases.t_wide5()
    s200 = _solve_table(table, 200)
    for name in (base, high):
        O = load_oracle(name)
        c = cases.ORACLE_CASES.by_name(name)
        for l in c.degrees:
            frame = "centre_of_mass" if l == 1 else "raw"
            tol = HIGH_DEGREE_TOL if l in (50, 100, 150, 200) else ORACLE_TOL
            e = _check_keys(name, "base", l, frame, s200, O, tol, IMBALANCE_ATOL)
            print(f"degree table {which} l={l}: worst e_l {e:.2e}")
    s50, s100 = _solve_table(table, 50), _solve_table(table, 100)
    for o in SI_OUTPUTS:
        assert_rows_close(s50[o], s200[o][:50], LMAX_CONSISTENCY_TOL, f"{which} {o} 50/200")
        assert_rows_close(s100[o], s200[o][:100], LMAX_CONSISTENCY_TOL, f"{which} {o} 100/200")
        assert_rows_close(s50[o], s100[o][:50], LMAX_CONSISTENCY_TOL, f"{which} {o} 50/100")


def test_water_split_core_degree_one():
    """G1, the only value test of the water term at l = 1 with a nonzero applied field
    (review r4 of the plan, K1): the stored cases with rho_w = 1030 are a consistent
    core (uniform_cc_full_fsfs) or l in {2, 10} (iso_full_fsfs_w). On a split core the
    water follows the total potential, which at l = 1 in the CM frame includes the
    applied field; a water term with the masked potential gives 1.2e-3 at l = 1 and
    fails this test by nine decades. Live oracle at l in {1 (CM), 2}."""
    table = cases.t_hager(reduced=True, outer_density="1030")
    sol = _solve_table(table, 2)
    ref = ro.kernels(table, ro.Physics(), ro.Discretisation(), (1, 2))
    for k, l in enumerate((1, 2)):
        for o in SI_OUTPUTS:
            assert_rows_close(sol[o][k], ref[o][k], ORACLE_TOL, f"water l={l} {o}")
    assert np.all(sol["geoid"][0] == 0)
    assert np.max(np.abs(sol["degree_one_imbalance"] - ref["degree_one"]["imbalance"])) <= IMBALANCE_ATOL
    assert np.max(np.abs(ref["degree_one"]["imbalance"])) > 1e-3     # a nonzero applied field


def test_uniform_background_full_live():
    """G1: FULL with a uniform background and the default discretisation (review r1 of
    the plan, issue 7), against a live oracle at l in {1, 2, 10}. The frame at l = 1 is
    raw (equations.md 10(f)), so the geoid at l = 1 is not masked and is compared, and
    the imbalance is an exact zero."""
    table = cases.t_hager(reduced=True)
    form = sj.Formulation(background=sj.Uniform(9.81))
    sol = _solve_table(table, 10, formulation=form)
    assert sol["frame"] is sj.DegreeOneFrame.RAW
    ref = ro.kernels(table, ro.Physics(background="uniform", uniform_g="9.81"), ro.Discretisation(), (1, 2, 10))
    for k, l in enumerate((1, 2, 10)):
        for o in SI_OUTPUTS:
            assert_rows_close(sol[o][sol["degrees"].index(l)], ref[o][k], ORACLE_TOL, f"uniform l={l} {o}")
    assert np.max(np.abs(sol["geoid"][0])) > 0
    assert np.all(sol["degree_one_imbalance"] == 0)


# ---------------------------------------------------------------------------
# G4: invariants.
# ---------------------------------------------------------------------------

REDUCED = {"hager": lambda: cases.t_hager(reduced=True), "prem": lambda: cases.t_prem(reduced=True)}
LMAX_INV = 20


def _scaled_table(table, factor):
    """The table with every viscosity times ``factor`` (exact decimal arithmetic)."""
    from dataclasses import replace
    from decimal import Decimal
    return replace(table, eta=tuple(str(Decimal(e) * Decimal(factor)) for e in table.eta))


@pytest.mark.parametrize("name", _grouped(list(REDUCED)))
def test_invariant_end_nodes_shapes_pairs(name):
    """End-node values (eq. 7.2), float64 leaves, shapes, degrees, and equal kernels at
    the two nodes of each pair."""
    table = REDUCED[name]()
    sol = _solve_table(table, LMAX_INV)
    _end_nodes(table, sol)
    N = len(table.radius)
    assert sol["degrees"] == tuple(range(1, LMAX_INV + 1))
    for o in SI_OUTPUTS:
        assert sol[o].dtype == np.float64 and sol[o].shape == (LMAX_INV, N)
    assert sol["degree_one_imbalance"].dtype == np.float64 and sol["degree_one_imbalance"].shape == (N,)
    pairs = [i for i in range(1, N) if table.radius[i] == table.radius[i - 1]]
    assert pairs
    for o in SI_OUTPUTS:
        K = sol[o]
        scale = np.max(np.abs(K), axis=1)
        for i in pairs:
            d = np.abs(K[:, i] - K[:, i - 1])
            assert np.all(d <= PAIR_TOL * scale), (o, i, np.max(d / scale))


def _end_nodes(table, sol):
    rho0 = np.array([float(x) for x in table.rho0])
    rho_w, rho_c = float(table.outer_density), float(table.core_density)
    for o in SI_OUTPUTS:
        scale = np.max(np.abs(sol[o]), axis=1)
        K = sol[o]
        if o == "surface_topography":
            assert np.all(np.abs(K[:, -1] + 1 / (rho0[-1] - rho_w)) <= END_NODE_TOL * scale)
        elif o == "cmb_topography":
            assert np.all(np.abs(K[:, 0] - 1 / (rho0[0] - rho_c)) <= END_NODE_TOL * scale)
        else:
            assert np.all(np.abs(K[:, [0, -1]]) <= END_NODE_TOL * scale[:, None]), o


@pytest.mark.xdist_group("hager4_full")
def test_invariant_end_nodes_full_table():
    """End-node values (eq. 7.2) on the full hager4 table at lmax = 200."""
    table = cases.t_hager()
    _end_nodes(table, _solve_table(table, 200))


@pytest.mark.parametrize("name", _grouped(list(REDUCED)))
def test_invariant_viscosity_scale(name):
    """eta -> 7 eta: geoid and both topographies unchanged, velocity divided by 7 (eq. 11)."""
    table = REDUCED[name]()
    a, b = _solve_table(table, LMAX_INV), _solve_table(_scaled_table(table, 7), LMAX_INV)
    for o in ("geoid", "surface_topography", "cmb_topography"):
        assert_rows_close(b[o], a[o], SCALE_TOL, o)
    assert_rows_close(7 * b["surface_velocity"], a["surface_velocity"], SCALE_TOL, "velocity")
    # The imbalance has an absolute rounding error that does not scale with its size
    # (IMBALANCE_ATOL), so it is compared absolutely.
    assert np.max(np.abs(b["degree_one_imbalance"] - a["degree_one_imbalance"])) <= IMBALANCE_ATOL


@pytest.mark.parametrize("name", _grouped(list(REDUCED)))
def test_invariant_density_scale(name):
    """rho_s in {1000, 4500, 1e4}: all outputs unchanged (the limit of D5 is a density, so
    every rho_s accepts the table; the Pade plan can differ)."""
    table = REDUCED[name]()
    a = _solve_table(table, LMAX_INV)
    for rho_s in (1000.0, 1e4):
        b = _solve_table(table, LMAX_INV, units=sj.Units(density_scale=rho_s))
        for o in SI_OUTPUTS:
            assert_rows_close(b[o], a[o], SCALE_TOL, f"rho_s={rho_s} {o}")
        assert np.max(np.abs(b["degree_one_imbalance"] - a["degree_one_imbalance"])) <= IMBALANCE_ATOL


@pytest.mark.parametrize("name", _grouped(list(REDUCED)))
def test_invariant_reference_viscosity(name):
    """The viscosity scale of the state: the largest node viscosity (default) against
    the geometric mean and against 1000 times the largest (private hook, D1)."""
    table = REDUCED[name]()
    a = _solve_table(table, LMAX_INV)
    big = 1000 * max(float(e) for e in table.eta)
    for rule in ("geometric_mean", big):
        b = _solve_table(table, LMAX_INV, rule=rule)
        for o in SI_OUTPUTS:
            assert_rows_close(b[o], a[o], SCALE_TOL, f"{rule} {o}")
        assert np.max(np.abs(b["degree_one_imbalance"] - a["degree_one_imbalance"])) <= IMBALANCE_ATOL


def _total_potential_over_g(sol, table):
    """``N + R A_j / g(R)`` at l = 1 in the CM frame: the total potential at the surface
    over g(R), with the applied field ``A_j = imbalance_j (4 pi / 3) g_j r_j^2 / M_eff``
    of eq. 10.6."""
    units = sj.Units()
    r = np.array([float(x) for x in table.radius])
    rho0 = np.array([float(x) for x in table.rho0])
    rho_c = float(table.core_density)
    mc = float(table.core_mass)
    g = np.asarray(integrate_gravity(r, rho0, mc, units))
    m_eff = float(mantle_mass(r, rho0, units)) + 4 * np.pi / 3 * r[0] ** 3 * rho_c
    A = sol["degree_one_imbalance"] * (4 * np.pi / 3) * g * r ** 2 / m_eff
    return sol["geoid"][0] + r[-1] * A / g[-1]


@pytest.mark.parametrize("name", _grouped(list(REDUCED)))
def test_invariant_water_split_core(name):
    """rho_w in {0, 1030} on the split-core reduced tables (F17): geoid, CMB topography
    and velocity unchanged at every row; (rho_top - rho_w) h_s + rho_w N unchanged at
    l >= 2; at l = 1 (CM) the combination with the total potential N + R A / g(R)
    unchanged (the only test that ties degree_one_imbalance to the topography row)."""
    from dataclasses import replace
    t0 = REDUCED[name]()
    t1 = replace(t0, outer_density="1030")
    a, b = _solve_table(t0, LMAX_INV), _solve_table(t1, LMAX_INV)
    for o in ("geoid", "cmb_topography", "surface_velocity"):
        assert_rows_close(b[o], a[o], RHO_W_TOL, o)
    rho_top = float(t0.rho0[-1])
    comb = lambda s, w: (rho_top - w) * s["surface_topography"] + w * s["geoid"]
    assert_rows_close(comb(b, 1030.0)[1:], comb(a, 0.0)[1:], RHO_W_TOL, "water, l >= 2")
    tot_a, tot_b = _total_potential_over_g(a, t0), _total_potential_over_g(b, t1)
    lhs = (rho_top - 1030.0) * b["surface_topography"][0] + 1030.0 * tot_b
    rhs = rho_top * a["surface_topography"][0] + 0.0 * tot_a
    assert_rows_close(lhs, rhs, RHO_W_TOL, "water, l = 1, total potential")
    # The combination with the masked geoid kernel at l = 1 does depend on rho_w here
    # (the applied field); this is the physics, not an error.
    assert row_errors(comb(b, 1030.0)[:1], comb(a, 0.0)[:1])[0] > 1e-6


def test_invariant_water_consistent_core():
    """rho_w in {0, 1030} on the consistent core, l = 1 (CM) included: all four
    combinations unchanged (no applied field; chunk 1 test_g17_rho_w_degree_one_cm)."""
    from dataclasses import replace
    t0 = cases.t_hager(reduced=True, core="consistent")
    t1 = replace(t0, outer_density="1030")
    a, b = _solve_table(t0, LMAX_INV), _solve_table(t1, LMAX_INV)
    for o in ("geoid", "cmb_topography", "surface_velocity"):
        assert_rows_close(b[o], a[o], RHO_W_TOL, o)
    rho_top = float(t0.rho0[-1])
    comb = lambda s, w: (rho_top - w) * s["surface_topography"] + w * s["geoid"]
    assert_rows_close(comb(b, 1030.0), comb(a, 0.0), RHO_W_TOL, "water, every row")


@pytest.mark.slow
def test_rough8_oracle_precision():
    """The 250-digit oracle on T_rough8 against 70 more digits, at the five stored
    degrees (review r3 of the plan, other issue 5)."""
    table = cases.t_rough8()
    for l in (1, 2, 10, 50, 100):
        a = ro.kernels(table, ro.Physics(), ro.Discretisation(), (l,), dps=cases.ROUGH8_DPS, return_mp=True)
        b = ro.kernels(table, ro.Physics(), ro.Discretisation(), (l,), dps=cases.ROUGH8_DPS + 70, return_mp=True)
        with mp.workdps(cases.ROUGH8_DPS + 70):
            for o in SI_OUTPUTS:
                ra, rb = a[o][0], b[o][0]
                scale = max(abs(x) for x in rb)
                if scale == 0:
                    assert all(x == 0 for x in ra)
                    continue
                err = max(abs(x - y) for x, y in zip(ra, rb)) / scale
                assert err <= ROUGH8_DPS_TOL, (l, o, mp.nstr(err, 3))
