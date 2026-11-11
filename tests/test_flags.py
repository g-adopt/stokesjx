"""The physics flags in the solver: gravity level, background, TALA and the boundary
classes (gates G1 to G4 of NOTES/chunks/c3-flags-fields/01-plan.md, sections 5.1, 5.2).

References: the mpmath oracle (``reference_mpmath``), stored in tests/golden/oracle for
the ten flag cases of chunk 1 and computed live for the covering design and the full
sweep; the boundary conditions of NOTES/docs/equations.md (eq. 8.1, 8.2, 8.5, 10.5);
closed forms (the pair step of eq. 12.3, the column sums of the balanced exponent).
Comparison rule: ``adapters.assert_rows_close`` (``e_l`` of CONVENTIONS 10, exact
equality for an exact-zero reference row).

Each (table, configuration, lmax) is solved once per worker process
(``functools.lru_cache``), and the tests of one covering run share an ``xdist_group``.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import ClassVar

import mpmath as mp
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import cases  # noqa: E402
import reference_mpmath as ro  # noqa: E402
from adapters import (SI_OUTPUTS, assert_rows_close, formulation_from_physics, profile_from_table,  # noqa: E402
                      row_errors)

import stokesjx as sj  # noqa: E402
from stokesjx import boundary, linalg, propagate, system  # noqa: E402
from stokesjx.formulation import BoundaryCondition, Terms, terms  # noqa: E402

# ---- tolerances (CONVENTIONS 9) ----
# Kernels against the oracle, e_l: the constant of tests/test_kernels_oracle.py (the
# requirement is 1e-10; 1e-12 fails a loss of two digits). Measured with the package
# under the flags: see the printed output and NOTES/chunks/c3-flags-fields/
# 03-implementation.md; the largest is 4.4e-13 on prem_tala_fsfs at l = 1 in the raw
# frame of a split core (a factor 2.3 below the constant). Cause: the rounding of the
# sweeps and of the node solve.
ORACLE_TOL = 1e-12
# degree_one_imbalance, absolute (same reasoning as tests/test_kernels_oracle.py).
IMBALANCE_ATOL = 1e-12
# The high-contrast tables of the slow high-degree test (stiff lid, ten decades): the
# constant of tests/test_kernels_oracle.py (logged in NOTES/TOLERANCES.md for c2).
CONTRAST_TOL = 1e-11
# Steps and inverse steps against the oracle's steps, max-normalised in the balanced
# coordinates of the interval: the c2 constant of tests/test_system_boundary.py.
STEP_ORACLE_TOL = 1e-13
# Boundary rows applied to the basis vectors, residual over the largest absolute term
# of the row: the c2 constant of tests/test_system_boundary.py.
BOUNDARY_TOL = 1e-15
# The closed-form pair step I + tau Kb against mp.expm(t Kb) at 60 digits, relative to
# the largest entry. Cause: two roundings (tau = 1 - rho_- / rho_+ and the product
# with the entries of Kb, which are small integers over powers of two times l).
PAIR_STEP_TOL = 1e-15
# The traced drag parameter of the boundary interface: sigma_rt / u_t = -side beta and
# the derivative of y4, relative. Cause: the rounding of one product and one quotient.
INTERFACE_TOL = 1e-15
# End-node values (eq. 7.2), the eta -> 7 eta and rho_s invariances, the water term:
# the c2 constants of tests/test_kernels_oracle.py.
END_NODE_TOL = 1e-12
SCALE_TOL = 1e-12
RHO_W_TOL = 1e-12
# The two nodes of a jump pair are sources at one radius with one gravity, and the TALA
# pair step does not act on the jump of a sheet (its column 1 is u_r, which the jump
# does not change), so their kernels are equal up to the rounding of two node solves.
# Relative to the row maximum; measured 7.8e-16
# (NOTES/chunks/c3-flags-fields/plan-review-r1-evidence/pairs.out).
PAIR_CONTINUITY_TOL = 1e-13
# vmap over three viscosity profiles against three single calls: the c2 constant of
# tests/test_kernels_ad.py.
VMAP_RTOL = 1e-12

FRAMES = {"centre_of_mass": sj.DegreeOneFrame.CENTRE_OF_MASS, "raw": sj.DegreeOneFrame.RAW,
          "default": sj.DegreeOneFrame.DEFAULT}
GRAVITY = (sj.GravityModel.FULL, sj.GravityModel.NO_SELF_GRAVITY)
BACKGROUND = (sj.Hydrostatic(), sj.Uniform(9.81))
CLASSES = (sj.FreeSlip(), sj.NoSlip())
ALL_FORMULATIONS = [sj.Formulation(gravity=g, background=b, compressible=c, surface=s, cmb=m)
                    for g, b, c, s, m in itertools.product(GRAVITY, BACKGROUND, (False, True), CLASSES, CLASSES)]


def _physics(form: sj.Formulation, degree_one: str = "default") -> ro.Physics:
    """The oracle ``Physics`` of a package formulation."""
    bc = lambda c: "free_slip" if isinstance(c, sj.FreeSlip) else "no_slip"
    uniform = isinstance(form.background, sj.Uniform)
    return ro.Physics(self_gravity="full" if form.gravity is sj.GravityModel.FULL else "none",
                      background="uniform" if uniform else "hydrostatic",
                      uniform_g=repr(form.background.value) if uniform else "9.81",
                      compressible=form.compressible, surface=bc(form.surface), cmb=bc(form.cmb),
                      degree_one=degree_one)


@functools.lru_cache(maxsize=None)
def _solve(table, lmax: int, form: sj.Formulation, frame: str = "default", units=sj.Units()):
    """The kernels of an oracle table as numpy arrays, cached per worker."""
    p = profile_from_table(table)
    cfg = sj.StokesConfig(lmax=lmax, formulation=form, degree_one=FRAMES[frame], units=units)
    K = propagate.solve(p, cfg)
    out = {o: np.asarray(getattr(K, o)) for o in SI_OUTPUTS}
    out["degree_one_imbalance"] = None if K.degree_one_imbalance is None else np.asarray(K.degree_one_imbalance)
    out["degrees"] = K.degrees
    out["frame"] = K.degree_one_frame
    return out


def _compare_row(sol, l, ref_rows, k_ref, tol, what):
    """Compare the four kernels of degree ``l`` of a solve with row ``k_ref`` of a live
    oracle result (or a dict of stored arrays); returns the worst e_l."""
    k = sol["degrees"].index(l)
    worst = 0.0
    for o in SI_OUTPUTS:
        ref = ref_rows[o][k_ref] if k_ref is not None else ref_rows[o]
        worst = max(worst, assert_rows_close(sol[o][k], ref, tol, f"{what} l={l} {o}"))
    return worst


# ---------------------------------------------------------------------------
# 5.1: units of the flag code.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ALL_FORMULATIONS, ids=lambda f: str(ALL_FORMULATIONS.index(f)))
def test_terms_dispatch(form):
    """``terms()`` for the 32 flag combinations: the table of the term record."""
    t = terms(form)
    assert t.self_gravity is (form.gravity is sj.GravityModel.FULL)
    assert t.anelastic is form.compressible
    assert t.surface is form.surface and t.cmb is form.cmb
    assert hash(t) == hash(terms(form)) and t == terms(form)


def test_boundary_classes():
    """Frozen, hashable, equal by value, ``traction_free``; ``tangential_unknown``
    returns the Python floats ``(1, 0)`` and ``(0, 1)`` for both sides and with or
    without a JAX parameter; ``Formulation`` refuses a string, an enum member and the
    base class."""
    for cls, expected, free in ((sj.FreeSlip, (1.0, 0.0), True), (sj.NoSlip, (0.0, 1.0), False)):
        a, b = cls(), cls()
        assert a == b and hash(a) == hash(b) and a is not b
        assert cls.traction_free is free and a.traction_free is free
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.x = 1
        for side in (+1, -1):
            for parameter in (None, jnp.asarray(3.0)):
                out = a.tangential_unknown(jnp.asarray(6.37e6), jnp.asarray(1e22), side, parameter)
                assert out == expected and all(type(v) is float for v in out)
    assert sj.FreeSlip() != sj.NoSlip()
    assert sj.Formulation(surface=sj.NoSlip()) == sj.Formulation(surface=sj.NoSlip())
    for bad in ("no_slip", sj.GravityModel.FULL, None):
        with pytest.raises(TypeError):
            sj.Formulation(surface=bad)
        with pytest.raises(TypeError):
            sj.Formulation(cmb=bad)
    with pytest.raises(TypeError):
        BoundaryCondition()


@dataclass(frozen=True)
class _DragForTest(BoundaryCondition):
    """Test-only drag class (the shape of a later partial slip, formulation docstring):
    ``side sigma_rt = -beta u_t``, ``beta`` the traced ``parameter`` in Pa s/m."""

    traction_free: ClassVar[bool] = False

    def tangential_unknown(self, radius, eta_ref, side, parameter=None):
        return 1.0, -side * radius * parameter / eta_ref


def test_boundary_interface_traced_parameter():
    """D1: the class interface carries a traced boundary parameter. With the drag
    class in both boundaries under ``jax.jit`` and ``jax.grad``: the tangential column
    gives ``sigma_rt / u_t = -beta`` at the surface and ``+beta`` at the CMB, the
    derivative of its ``y4`` entry with respect to ``beta`` is ``-R / eta_ref`` and
    ``+r_c / eta_ref``, and a new ``beta`` gives no new trace. With ``FreeSlip`` and
    ``NoSlip`` the jaxpr of each basis is the same with and without a traced
    parameter (no traced operation enters)."""
    deg = (1, 2, 50)
    R, rc, eta_ref = 6.37e6, 3.48e6, 1e22
    t_drag = Terms(self_gravity=True, anelastic=False, surface=_DragForTest(), cmb=_DragForTest())
    traces = []

    def bases(beta):
        traces.append(1)
        Yc = boundary.cmb_basis(deg, 1.3, 2.2, terms=t_drag, radius=rc, eta_ref=eta_ref, parameter=beta)
        Ys = boundary.surface_basis(deg, 2.4, np.array([2.0, 3.0, 51.0]), terms=t_drag, radius=R,
                                    eta_ref=eta_ref, parameter=beta)
        return Yc[:, :, 1], Ys[:, :, 0]               # the tangential columns, (L, 6)
    f = jax.jit(bases)
    for beta in (1e15, 3e16):
        tc, ts = (np.asarray(x) for x in f(jnp.asarray(beta)))
        # sigma_rt / u_t = (eta_ref y4 / r) / y2 = -side beta.
        surf = eta_ref * ts[:, 3] / R / ts[:, 1]
        cmb = eta_ref * tc[:, 3] / rc / tc[:, 1]
        assert np.all(np.abs(surf + beta) <= INTERFACE_TOL * beta), surf
        assert np.all(np.abs(cmb - beta) <= INTERFACE_TOL * beta), cmb
    assert len(traces) == 1
    ds = jax.grad(lambda b: bases(b)[1][0, 3])(jnp.asarray(1e15))
    dc = jax.grad(lambda b: bases(b)[0][0, 3])(jnp.asarray(1e15))
    assert abs(float(ds) + R / eta_ref) <= INTERFACE_TOL * R / eta_ref
    assert abs(float(dc) - rc / eta_ref) <= INTERFACE_TOL * rc / eta_ref
    # The static classes: no traced operation from the parameter.
    for cls in CLASSES:
        t = Terms(self_gravity=True, anelastic=False, surface=cls, cmb=cls)
        mk = lambda c_c, p: (boundary.cmb_basis(deg, c_c, 2.2, terms=t, radius=rc, eta_ref=eta_ref, parameter=p),
                             boundary.surface_basis(deg, c_c, np.array([2.0, 3.0, 51.0]), terms=t, radius=R,
                                                    eta_ref=eta_ref, parameter=p))
        with_p = str(jax.make_jaxpr(lambda c, p: mk(c, p))(1.3, 7.0))
        without = str(jax.make_jaxpr(lambda c, p: mk(c, None))(1.3, 7.0))
        assert with_p == without


def _row_residual(row, vecs):
    """``row . v`` over the largest absolute term ``|row_k v_k|`` per vector."""
    terms_ = row * vecs
    return np.abs(np.sum(terms_, axis=-1)) / np.max(np.abs(terms_), axis=-1)


@pytest.mark.parametrize("gravity", GRAVITY, ids=lambda g: g.value)
def test_bases_satisfy_conditions(gravity):
    """Both classes at both boundaries, both gravity levels, both frames at ``l = 1``,
    ``l`` in {1, 2, 50, 200}: the CMB rows ``y1``, the tangential row and eq. 8.5 (with
    ``rho_c`` under FULL, without under NO_SELF_GRAVITY); the surface rows ``y1``, the
    tangential row and eq. 8.2 or 10.5; rank 3 (eq. 8.1)."""
    deg = (1, 2, 50, 200)
    l = np.asarray(deg, float)
    c_c, c_s, rc_rel = 1.37, 2.45, 9900.0 / 4500.0
    full = gravity is sj.GravityModel.FULL
    for surf, cmb in itertools.product(CLASSES, CLASSES):
        t = Terms(self_gravity=full, anelastic=False, surface=surf, cmb=cmb)
        Yc = np.asarray(boundary.cmb_basis(deg, c_c, rc_rel, terms=t, radius=3.48e6, eta_ref=1e22))
        assert np.all(Yc[:, 0, :] == 0)
        tang_c = 3 if isinstance(cmb, sj.FreeSlip) else 1
        assert np.all(Yc[:, tang_c, :] == 0)
        rows = np.zeros((len(l), 6))
        rows[:, 5], rows[:, 4], rows[:, 2] = 1.0, -(l - (c_c * rc_rel if full else 0.0)), c_c
        for k in range(3):
            v = Yc[:, :, k]
            nz = np.max(np.abs(rows * v), axis=-1) > 0
            assert np.all(_row_residual(rows[nz], v[nz]) <= BOUNDARY_TOL)
        assert np.all(np.linalg.matrix_rank(Yc) == 3)
        for frame in (sj.DegreeOneFrame.RAW, sj.DegreeOneFrame.CENTRE_OF_MASS):
            a = boundary.surface_potential_coefficient(deg, frame)
            Ys = np.asarray(boundary.surface_basis(deg, c_s, a, terms=t, radius=6.37e6, eta_ref=1e22))
            assert np.all(Ys[:, 0, :] == 0)
            tang_s = 3 if isinstance(surf, sj.FreeSlip) else 1
            assert np.all(Ys[:, tang_s, :] == 0)
            rows = np.zeros((len(l), 6))
            rows[:, 5], rows[:, 4], rows[:, 2] = 1.0, a, c_s
            for k in range(3):
                v = Ys[:, :, k]
                nz = np.max(np.abs(rows * v), axis=-1) > 0
                assert np.all(_row_residual(rows[nz], v[nz]) <= BOUNDARY_TOL)
            assert np.all(np.linalg.matrix_rank(Ys) == 3)


def test_functionals_no_self_gravity():
    """Eq. 5.2: under NO_SELF_GRAVITY the surface topography row has no ``y5`` entry
    (no ``rho_w phi``) and the CMB topography row has no ``y5`` entry (no ``rho_c phi``);
    under FULL both entries are nonzero for nonzero ``rho_w`` and ``rho_c``."""
    deg = (1, 2, 10)
    kw = dict(R=6.37e6, g_R=9.8, rho_top=3300.0, rho_w=1030.0, rho_s=4500.0, eta_ref=1e22, c_s=2.4,
              geoid_mask=np.ones(3))
    for full in (True, False):
        t = Terms(self_gravity=full, anelastic=False, surface=sj.FreeSlip(), cmb=sj.FreeSlip())
        ws = np.asarray(boundary.surface_functionals(deg, terms=t, **kw))
        wc = np.asarray(boundary.cmb_functional(r_c=3.48e6, g_c=10.6, rho_cmb=5500.0, rho_c=9900.0, rho_s=4500.0,
                                                eta_ref=1e22, terms=t))
        if full:
            assert np.all(ws[:, 1, 4] != 0) and wc[4] != 0
        else:
            assert np.all(ws[:, 1, 4] == 0) and wc[4] == 0
        assert np.all(ws[:, 1, [0, 1, 3, 5]] == 0) and np.all(ws[:, 1, 2] != 0)
        assert np.all(wc[[0, 1, 3, 5]] == 0) and wc[2] != 0


STEP_TABLES = {"prem_q": lambda: cases.t_prem(reduced=True), "hager_q": lambda: cases.t_hager(reduced=True)}
STEP_FLAGS = {
    "full_tala": sj.Formulation(compressible=True),
    "nsg": sj.Formulation(gravity=sj.GravityModel.NO_SELF_GRAVITY),
    "nsg_tala": sj.Formulation(gravity=sj.GravityModel.NO_SELF_GRAVITY, compressible=True),
}


@pytest.mark.parametrize("flags", list(STEP_FLAGS))
@pytest.mark.parametrize("table_name", list(STEP_TABLES))
def test_steps_flags_against_oracle(table_name, flags):
    """The steps and inverse steps of both sweeps against the oracle's steps at ``l`` in
    {1, 2, 100} under FULL TALA, NO_SELF_GRAVITY and NO_SELF_GRAVITY with TALA, as
    ``test_interval_steps_against_oracle`` of c2 (balanced coordinates of the interval,
    the oracle's ``eta_s = 1e19`` through the private hook). At a pair under TALA the
    step is the closed form ``I + K(e) tau`` (eq. 12.3)."""
    table = STEP_TABLES[table_name]()
    form = STEP_FLAGS[flags]
    p = profile_from_table(table)
    cfg = sj.StokesConfig(lmax=100, formulation=form)
    plan = propagate.static_plan(p.structure, cfg)
    eta_s = 1e19
    # Jitted: the eager op-by-op Pade pair takes about 2 s, the compiled one 0.2 s.
    steps = jax.jit(propagate.interval_steps, static_argnames=("plan", "config"))
    Zu, Zd = (np.asarray(a) for a in steps(p, jnp.asarray(eta_s), plan=plan, config=cfg))
    eta = np.asarray(p.eta)
    e = 0.5 * (eta[1:] + eta[:-1]) / eta_s
    D = np.stack([np.ones_like(e), np.ones_like(e), e, e, e, e], axis=-1)
    for l in (1, 2, 100):
        k = plan.degrees.index(l)
        S = system.degree_scaling((l,))[0]
        model, dps = ro._model(table, _physics(form), ro.Discretisation(), l)
        for i in range(len(e)):
            with mp.workdps(dps):
                ref = np.array(model.steps[i + 1].tolist(), dtype=float)
                ref_inv = np.array(mp.inverse(model.steps[i + 1]).tolist(), dtype=float)
            for Z, Rf in ((Zu, ref), (Zd, ref_inv)):
                ref_b = (S / D[i])[:, None] * Rf * (D[i] / S)[None, :]
                mine_b = Z[i, k] / D[i][:, None] * D[i][None, :]
                err = np.max(np.abs(mine_b - ref_b)) / np.max(np.abs(ref_b))
                assert err <= STEP_ORACLE_TOL, (l, i, err)
    pairs = np.asarray(p.structure.pairs) - 1
    if not form.compressible:
        assert np.all(Zu[pairs] == np.eye(6)) and np.all(Zd[pairs] == np.eye(6))


def test_pair_closed_form():
    """Eq. 12.3: ``Kb^2 = -Kb``; the pair step ``I + tau Kb`` (``system.anelastic_pair_terms``
    on a structure with one pair) against ``mp.expm(t Kb)`` at 60 digits for
    ``t = ln(rho_+ / rho_-)`` in {0.01, 0.19, 1.0}; the step times the inverse step is
    the identity."""
    deg = (1, 2, 50)
    r = np.array([3.48e6, 4.0e6, 4.0e6, 5.0e6])
    structure = sj.NodeStructure.from_radius(r, consistent_core=True)
    _, _, Kb = system._static_exponent_parts(deg, True)
    for k in range(len(deg)):
        K = Kb[:, :, k]
        assert np.array_equal(K @ K, -K)
    for t in (0.01, 0.19, 1.0):
        rho0 = np.array([5000.0, 4500.0, 4500.0 * np.exp(t), 4000.0])
        dE, dE_inv = (np.asarray(x) for x in system.anelastic_pair_terms(deg, rho0, structure))
        assert np.all(dE[:, :, [0, 2]] == 0) and np.all(dE_inv[:, :, [0, 2]] == 0)
        for k in range(len(deg)):
            step = np.eye(6) + dE[:, :, 1, k]
            inv = np.eye(6) + dE_inv[:, :, 1, k]
            with mp.workdps(60):
                tt = mp.log(mp.mpf(rho0[2]) / mp.mpf(rho0[1]))
                ref = np.array(mp.expm(tt * mp.matrix(Kb[:, :, k].tolist())).tolist(), dtype=float)
            assert np.max(np.abs(step - ref)) <= PAIR_STEP_TOL * np.max(np.abs(ref)), (t, deg[k])
            assert np.max(np.abs(step @ inv - np.eye(6))) <= PAIR_STEP_TOL * np.max(np.abs(step)) * np.max(np.abs(inv))


def _c2_bound(lmax, h, rho):
    """The incompressible bound of c2, ``max_i (3 lmax + 10 + rho) h_i + 4 * 0``."""
    return float(np.max((3 * lmax + 10 + np.asarray(rho, float)) * np.asarray(h, float) + 4 * np.zeros_like(h)))


def test_norm_bound_columnwise():
    """D5: the column-wise bound is above the float64 1-norm of the balanced exponent
    (``norm <= bound (1 + 4 eps)``: at ``l = 1`` the bound is attained, so one rounding
    can put the norm above it; review r2 minor 2) on 2000 random draws with the density
    entries and 2000 without (NO_SELF_GRAVITY with ``rho_max = 0``); it equals the c2
    formula bit for bit at ``kappa_h = 0``; the column sums of the balanced A and K at
    ``l`` in {1, 2, 10, 50, 200} are those of the docstring."""
    rng = np.random.default_rng(12)
    eps = np.finfo(float).eps
    for self_gravity in (True, False):
        for _ in range(2000):
            lmax = int(rng.integers(1, 301))
            l = int(rng.integers(1, lmax + 1))
            h = float(rng.uniform(0, 0.05))
            kh_bound = float(rng.uniform(0, 0.5))
            kh = float(rng.uniform(-kh_bound, kh_bound))
            d = float(rng.uniform(0, 3)) if self_gravity else 0.0
            P, Q, K = system._static_exponent_parts((l,), self_gravity)
            B = (P[:, :, 0] + d * Q[:, :, 0]) * h + K[:, :, 0] * kh
            norm = np.max(np.sum(np.abs(B), axis=0))
            bound = linalg.balanced_norm_bound_intervals(lmax, h, kh_bound, d if self_gravity else 0.0)
            assert norm <= bound * (1 + 4 * eps), (lmax, l, h, kh, d, norm, bound)
    for lmax in (1, 2, 50, 100, 1200):
        h = rng.uniform(0, 0.05, 40)
        h[[3, 17]] = 0.0
        rho = rng.uniform(0, 3, 40)
        assert linalg.balanced_norm_bound_intervals(lmax, h, 0.0, rho) == _c2_bound(lmax, h, rho)
        assert linalg.balanced_norm_bound_intervals(lmax, h, np.zeros(40), 3.0) == _c2_bound(lmax, h, 3.0)
    for l in (1, 2, 10, 50, 200):
        d = 1.7
        P, Q, K = system._static_exponent_parts((l,), True)
        cols = np.sum(np.abs(P[:, :, 0] + d * Q[:, :, 0]), axis=0)
        expected = [l + 5 + 6 / l, 3 * l + 7 + 2 / l, l + 1, 3 * l + 3, l + 2 + d, l + d]
        assert np.allclose(cols, expected, rtol=1e-15, atol=0)
        kcols = np.sum(np.abs(K[:, :, 0]), axis=0)
        assert np.allclose(kcols, [2 + 2 / l, 0, 0, 0, 0, 0], rtol=1e-15, atol=0)


def _kappa_profile(kappa_h_at, where):
    """The reduced PREM table with ``|kappa_h| = kappa_h_at`` on one interval: the
    density of every node above interval 5 (``where="interval"``) or above the pair at
    interval ``p`` (``where="pair"``) is multiplied by a factor, so that the logarithm
    of the ratio across that interval grows by ``kappa_h_at`` minus its old value."""
    table = cases.t_prem(reduced=True)
    p = profile_from_table(table)
    rho = np.asarray(p.rho0).copy()
    i = 5 if where == "interval" else p.structure.pairs[0]
    old = np.log(rho[i] / rho[i - 1])
    rho[i:] *= np.exp(kappa_h_at - old)
    return p.replace(rho0=rho), i


def test_kappa_limit():
    """D5: a concrete profile with ``|kappa_h| = 0.35`` on an ordinary interval raises
    ``ValueError`` naming ``kappa_h_max``; the same jump at a pair passes; 0.26 passes;
    under ``jax.jit`` the profile traces; with ``eta`` traced and ``rho0`` and the radius
    concrete (``jax.grad`` over ``log eta``) the check still runs."""
    cfg = sj.StokesConfig(lmax=4, formulation=sj.Formulation(compressible=True))
    bad, i = _kappa_profile(0.35, "interval")
    with pytest.raises(ValueError, match="kappa_h_max"):
        sj.stokes_kernels(bad, cfg)
    at_pair, _ = _kappa_profile(0.35, "pair")
    assert np.all(np.isfinite(np.asarray(sj.stokes_kernels(at_pair, cfg).geoid)))
    ok, _ = _kappa_profile(0.26, "interval")
    assert np.all(np.isfinite(np.asarray(sj.stokes_kernels(ok, cfg).geoid)))
    out = jax.jit(lambda q: sj.stokes_kernels(q, cfg).geoid)(bad)
    assert out.shape == (4, bad.structure.n_nodes)
    with pytest.raises(ValueError, match="kappa_h_max"):
        jax.grad(lambda le: jnp.sum(sj.stokes_kernels(bad.replace(eta=jnp.exp(le)), cfg).geoid))(
            jnp.log(bad.eta))
    # A larger limit accepts the same table (the remedy that the message names).
    cfg2 = sj.StokesConfig(lmax=4, formulation=sj.Formulation(compressible=True),
                           numerics=sj.Numerics(kappa_h_max=0.4))
    assert np.all(np.isfinite(np.asarray(sj.stokes_kernels(bad, cfg2).geoid)))
    # The incompressible model does not read kappa_h.
    assert np.all(np.isfinite(np.asarray(sj.stokes_kernels(bad, sj.StokesConfig(lmax=4)).geoid)))


def test_legacy_kappa_refuses_pairs():
    """D6: CODE-A's one-sided kappa divides by the radial gap, so the static plan refuses
    it on a structure with pairs under TALA; without pairs it is accepted."""
    p = profile_from_table(cases.t_prem(reduced=True))
    cfg = sj.StokesConfig(lmax=4, formulation=sj.Formulation(compressible=True))
    disc = propagate.Discretisation(kappa_rule="legacy_one_sided")
    with pytest.raises(ValueError, match="one-sided"):
        propagate.static_plan(p.structure, cfg, disc)
    golden = profile_from_table(cases.golden_table("hager4_fsfs"))
    assert propagate.static_plan(golden.structure, cfg, disc).kappa_rule == "legacy_one_sided"
    with pytest.raises(ValueError):
        propagate.Discretisation(kappa_rule="central")


def test_trace_count_flags():
    """CONVENTIONS 2: TALA with no slip at both boundaries; a jitted call with a profile
    from NumPy arrays, from JAX arrays and with new values traces the core once."""
    form = sj.Formulation(compressible=True, surface=sj.NoSlip(), cmb=sj.NoSlip())
    cfg = sj.StokesConfig(lmax=3, formulation=form)
    p_np = profile_from_table(cases.t_layers(reduced=True))
    p_jx = jax.tree_util.tree_map(jnp.asarray, p_np)
    p_new = p_np.replace(eta=np.asarray(p_np.eta) * 1.7, rho0=np.asarray(p_np.rho0) * 1.01)
    f = jax.jit(lambda p: sj.stokes_kernels(p, cfg).geoid)
    before = propagate.trace_count()
    for p in (p_np, p_jx, p_new):
        jax.block_until_ready(f(p))
    assert propagate.trace_count() - before == 1
    assert f._cache_size() == 1


def test_vmap_flags():
    """``vmap`` over three viscosity profiles equals three single calls under TALA with
    no slip at both boundaries, values and gradients (c2 ``test_vmap_equals_single_calls``)."""
    form = sj.Formulation(compressible=True, surface=sj.NoSlip(), cmb=sj.NoSlip())
    cfg = sj.StokesConfig(lmax=8, formulation=form)
    p0 = profile_from_table(cases.t_prem(reduced=True))
    x0 = np.log(np.asarray(p0.eta))
    rng = np.random.default_rng(31)
    X = x0[None] + 0.2 * rng.standard_normal((3,) + x0.shape)
    w = jnp.asarray(rng.standard_normal((cfg.lmax - 1, x0.size)))

    def G(x):
        K = sj.stokes_kernels(p0.replace(eta=jnp.exp(x)), cfg)
        return K.geoid, K.surface_topography, K.cmb_topography

    F = lambda x: sum(jnp.sum(w * k) / s for k, s in zip(G(x), (1e-9, 1e-4, 1e-4)))
    Gv = jax.jit(jax.vmap(G))(X)
    gv = jax.jit(jax.vmap(jax.grad(F)))(X)
    G1, g1 = jax.jit(G), jax.jit(jax.grad(F))           # compiled once for the three calls
    for b in range(3):
        single = G1(X[b])
        for o in range(3):
            assert_rows_close(np.asarray(Gv[o][b]), np.asarray(single[o]), VMAP_RTOL, f"vmap {o}")
        gs = np.asarray(g1(X[b]))
        assert np.max(np.abs(np.asarray(gv[b]) - gs)) <= VMAP_RTOL * np.max(np.abs(gs))


# ---------------------------------------------------------------------------
# 5.2 item 1: the stored flag cases.
# ---------------------------------------------------------------------------

# lmax of the solve of each stored flag case (plan 5.2 item 1).
STORED_FLAG_LMAX = {
    "hager4_full_nsfs": 50, "hager4_full_fsns": 50, "hager4_full_nsns": 10, "hager4_noself_uniform_fsfs": 50,
    "hager4_noself_hydro_fsfs": 50, "hager4_full_tala_fsfs": 50, "prem_tala_fsfs": 50,
    "layers_cc_tala_fsfs": 2, "prem_tala_narrow": 10,
}
# Cases whose default solve refuses degree 1 (FULL, hydrostatic, a no-slip boundary):
# the stored l = 1 key is the raw frame by explicit request.
REFUSED_L1 = ("hager4_full_nsfs", "hager4_full_fsns", "hager4_full_nsns")


def _stored_case(case_name, lmax, load_oracle, sources=None):
    """Every stored variant, degree and frame of one case; returns the worst e_l."""
    c = cases.ORACLE_CASES.by_name(case_name)
    O = load_oracle(case_name)
    worst = 0.0
    for v in c.variants:
        assert v.disc == ro.Discretisation()
        form = formulation_from_physics(v.physics)
        sol = _solve(v.table, lmax, form)
        if case_name in REFUSED_L1:
            assert sol["degrees"][0] == 2 and sol["degree_one_imbalance"] is None and sol["frame"] is None
        for l in c.degrees:
            if l == 1:
                continue
            key = f"{v.label}/l{l}/raw/"
            ref = {o: O[key + o] for o in SI_OUTPUTS}
            if sources is not None:
                sol_l = {o: sol[o][:, list(sources)] for o in SI_OUTPUTS}
                sol_l["degrees"] = sol["degrees"]
            else:
                sol_l = sol
            worst = max(worst, _compare_row(sol_l, l, ref, None, ORACLE_TOL, f"{case_name} {key}"))
        if 1 in c.degrees:
            for frame in v.l1_frames:
                if sol["degrees"][0] == 1 and sol["frame"].value == frame:
                    s1 = sol
                else:
                    s1 = _solve(v.table, 1, form, frame)
                assert s1["frame"].value == frame
                key = f"{v.label}/l1/{frame}/"
                ref = {o: O[key + o] for o in SI_OUTPUTS}
                worst = max(worst, _compare_row(s1, 1, ref, None, ORACLE_TOL, f"{case_name} {key}"))
                if frame == "centre_of_mass":
                    err = np.max(np.abs(s1["degree_one_imbalance"] - O[key + "degree_one_imbalance"]))
                    assert err <= IMBALANCE_ATOL, (case_name, key, err)
                else:
                    assert np.all(s1["degree_one_imbalance"] == 0)
    return worst


@pytest.mark.parametrize("case_name", list(STORED_FLAG_LMAX))
def test_stored_flag_case(case_name, load_oracle):
    """G1: every stored variant, degree and frame of the stored flag cases of chunk 1;
    the default solve starts at ``l = 2`` where the frame rule refuses degree 1."""
    worst = _stored_case(case_name, STORED_FLAG_LMAX[case_name], load_oracle)
    plan = propagate.static_plan(
        profile_from_table(cases.ORACLE_CASES.by_name(case_name).variants[0].table).structure,
        sj.StokesConfig(lmax=STORED_FLAG_LMAX[case_name],
                        formulation=formulation_from_physics(cases.ORACLE_CASES.by_name(case_name).variants[0].physics)))
    print(f"{case_name}: worst e_l {worst:.2e}; Pade plan ({plan.pade.order}, {plan.pade.squarings}), "
          f"scaled norm {plan.pade.scaled_norm:.3f}")


def test_smooth_conv_stored(load_oracle):
    """G1: ``smooth_conv``, four refinements of a smooth TALA table, the 65 stored sources
    common to all four, ``l`` in {2, 10, 30}: the stored case where the interval-mean
    viscosity changes the result (CONVENTIONS 14)."""
    c = cases.ORACLE_CASES.by_name("smooth_conv")
    O = load_oracle("smooth_conv")
    worst = 0.0
    for v in c.variants:
        sol = _solve(v.table, 30, formulation_from_physics(v.physics))
        for l in c.degrees:
            key = f"{v.label}/l{l}/raw/"
            src = list(O[key + "sources"]) if key + "sources" in O else list(v.sources)
            k = sol["degrees"].index(l)
            for o in SI_OUTPUTS:
                worst = max(worst, assert_rows_close(sol[o][k][src], O[key + o], ORACLE_TOL, f"smooth_conv {key}{o}"))
    print(f"smooth_conv: worst e_l {worst:.2e}")


# ---------------------------------------------------------------------------
# 5.2 items 2 to 4: the covering design against the live oracle, and invariants.
# ---------------------------------------------------------------------------


def _covering_runs():
    """The L8 orthogonal array on the five binary flags: gravity ``a``, background ``b``,
    compressible ``c``, surface ``a XOR b``, CMB ``a XOR c``, with ``(a, b, c)`` over
    ``{0, 1}^3``. Every pair of flags takes all four value pairs."""
    runs = {}
    for a, b, c in itertools.product((0, 1), repeat=3):
        form = sj.Formulation(gravity=GRAVITY[a], background=BACKGROUND[b], compressible=bool(c),
                              surface=CLASSES[a ^ b], cmb=CLASSES[a ^ c])
        runs[f"g{a}b{b}c{c}"] = form
    return runs


COVERING = _covering_runs()
COVERING_LMAX = 10


def _live_check(table, form, degrees=(2, 10), lmax=COVERING_LMAX, tol=ORACLE_TOL):
    """The package against the live oracle at ``degrees`` and at ``l = 1`` in every frame
    that the rule offers or allows on request; returns the worst e_l and the worst
    absolute imbalance error."""
    structure = profile_from_table(table).structure
    opts = sj.formulation.degree_one_options(form, structure.consistent_core)
    sol = _solve(table, lmax, form)
    ref = ro.kernels(table, _physics(form), ro.Discretisation(), degrees)
    worst = max(_compare_row(sol, l, ref, k, tol, f"{form} raw") for k, l in enumerate(degrees))
    worst_imb = 0.0
    frames = ([opts.default] if opts.default is not None else []) + [
        f for f in opts.offered + opts.explicit_only if f is not opts.default]
    if opts.default is None:
        assert sol["degrees"][0] == 2 and sol["degree_one_imbalance"] is None
    for frame in frames:
        s1 = sol if (frame is opts.default and sol["degrees"][0] == 1) else _solve(table, 1, form, frame.value)
        r1 = ro.kernels(table, _physics(form, frame.value), ro.Discretisation(), (1,))
        assert r1["degree_one"]["frame"] == frame.value
        worst = max(worst, _compare_row(s1, 1, r1, 0, tol, f"{form} l=1 {frame.value}"))
        if frame is sj.DegreeOneFrame.CENTRE_OF_MASS:
            err = float(np.max(np.abs(s1["degree_one_imbalance"] - r1["degree_one"]["imbalance"])))
            assert err <= IMBALANCE_ATOL, err
            worst_imb = max(worst_imb, err)
        else:
            assert np.all(s1["degree_one_imbalance"] == 0)
    return worst, worst_imb


@pytest.mark.parametrize("run", [pytest.param(k, marks=pytest.mark.xdist_group(f"cover_{k}")) for k in COVERING])
def test_flag_covering_design(run):
    """G2: the eight covering runs on ``t_prem(reduced=True)`` (25 nodes, four pairs, three
    density jumps at pairs, split core) with ``Uniform(9.81)`` where the background is
    uniform, against the live oracle at ``l`` in {1 (each frame), 2, 10}."""
    worst, imb = _live_check(cases.t_prem(reduced=True), COVERING[run])
    print(f"covering {run}: worst e_l {worst:.2e}, imbalance {imb:.2e}")


def _end_nodes(table, sol, form):
    """End-node values of eq. 7.2 under every flag: the surface topography at the top
    node ``-1 / (rho_top - rho_w)``, the CMB topography at the CMB node
    ``1 / (rho0(r_c) - rho_c)``, the geoid and the velocity zero."""
    rho0 = np.array([float(x) for x in table.rho0])
    rho_w, rho_c = float(table.outer_density), float(table.core_density)
    for o in SI_OUTPUTS:
        K = sol[o]
        scale = np.max(np.abs(K), axis=1)
        if o == "surface_topography":
            assert np.all(np.abs(K[:, -1] + 1 / (rho0[-1] - rho_w)) <= END_NODE_TOL * scale)
        elif o == "cmb_topography":
            assert np.all(np.abs(K[:, 0] - 1 / (rho0[0] - rho_c)) <= END_NODE_TOL * scale)
        elif o == "surface_velocity" and isinstance(form.surface, sj.NoSlip):
            assert np.all(K == 0)
        else:
            assert np.all(np.abs(K[:, [0, -1]]) <= END_NODE_TOL * scale[:, None]), o


def _scaled(table, factor):
    return replace(table, eta=tuple(str(Decimal(e) * Decimal(factor)) for e in table.eta))


@pytest.mark.parametrize("run", [pytest.param(k, marks=pytest.mark.xdist_group(f"cover_{k}")) for k in COVERING])
def test_flag_invariants(run):
    """G3 on the eight covering runs: end nodes (eq. 7.2); ``eta -> 7 eta`` (geoid and
    topographies unchanged, velocity divided by 7, imbalance unchanged); ``rho_s`` in
    {1000, 1e4} under TALA; under a ``NoSlip`` surface the velocity kernel and its
    gradient with respect to ``log eta`` are exact zeros; equal kernels at the two
    nodes of every pair."""
    form = COVERING[run]
    table = cases.t_prem(reduced=True)
    a = _solve(table, COVERING_LMAX, form)
    _end_nodes(table, a, form)
    b = _solve(_scaled(table, 7), COVERING_LMAX, form)
    for o in ("geoid", "surface_topography", "cmb_topography"):
        assert_rows_close(b[o], a[o], SCALE_TOL, f"7 eta {o}")
    assert_rows_close(7 * b["surface_velocity"], a["surface_velocity"], SCALE_TOL, "7 eta velocity")
    if a["degree_one_imbalance"] is not None:
        assert np.max(np.abs(b["degree_one_imbalance"] - a["degree_one_imbalance"])) <= IMBALANCE_ATOL
    if form.compressible:
        for rho_s in (1000.0, 1e4):
            c = _solve(table, COVERING_LMAX, form, units=sj.Units(density_scale=rho_s))
            for o in SI_OUTPUTS:
                assert_rows_close(c[o], a[o], SCALE_TOL, f"rho_s={rho_s} {o}")
    if isinstance(form.surface, sj.NoSlip):
        assert np.all(a["surface_velocity"] == 0)
        p0 = profile_from_table(table)
        cfg = sj.StokesConfig(lmax=COVERING_LMAX, formulation=form)
        g = jax.grad(lambda le: jnp.sum(sj.stokes_kernels(p0.replace(eta=jnp.exp(le)), cfg).surface_velocity))(
            jnp.log(p0.eta))
        assert np.all(np.asarray(g) == 0)
    N = len(table.radius)
    pairs = [i for i in range(1, N) if table.radius[i] == table.radius[i - 1]]
    worst = 0.0
    for o in SI_OUTPUTS:
        K = a[o]
        scale = np.max(np.abs(K), axis=1)
        scale = np.where(scale > 0, scale, 1.0)
        for i in pairs:
            d = np.max(np.abs(K[:, i] - K[:, i - 1]) / scale)
            worst = max(worst, d)
            assert d <= PAIR_CONTINUITY_TOL, (o, i, d)
    print(f"invariants {run}: pair continuity {worst:.2e}")


def test_water_no_self_gravity():
    """G3: under NO_SELF_GRAVITY with a uniform ``g`` the surface topography has no
    ``rho_w phi`` term (eq. 5.2), so ``(rho_top - rho_w) h_s`` does not depend on
    ``rho_w``, and geoid, CMB topography and velocity do not either; the solve with
    ``rho_w = 1030`` against the live oracle at ``l`` in {1 (raw), 2}."""
    form = sj.Formulation(gravity=sj.GravityModel.NO_SELF_GRAVITY, background=sj.Uniform(9.81))
    t0 = cases.t_hager(reduced=True)
    t1 = cases.t_hager(reduced=True, outer_density="1030")
    a, b = _solve(t0, 2, form), _solve(t1, 2, form)
    assert b["frame"] is sj.DegreeOneFrame.RAW
    rho_top = float(t0.rho0[-1])
    for o in ("geoid", "cmb_topography", "surface_velocity"):
        assert_rows_close(b[o], a[o], RHO_W_TOL, o)
    assert_rows_close((rho_top - 1030.0) * b["surface_topography"], rho_top * a["surface_topography"], RHO_W_TOL,
                      "(rho_top - rho_w) h_s")
    ref = ro.kernels(t1, _physics(form), ro.Discretisation(), (1, 2))
    for k, l in enumerate((1, 2)):
        _compare_row(b, l, ref, k, ORACLE_TOL, "water NSG")


# ---------------------------------------------------------------------------
# All 32 combinations and the consistent core (quick set); high degree and contrast
# (slow set, the full tier of CONVENTIONS 7).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ALL_FORMULATIONS, ids=lambda f: str(ALL_FORMULATIONS.index(f)))
def test_flag_all_combinations(form):
    """G2: every one of the 32 combinations on the split-core reduced PREM table, and the
    8 combinations of FULL with hydrostatic gravity on the consistent-core reduced
    hager4 table, against the live oracle at ``l`` in {1 (each frame), 2, 10}."""
    worst, imb = _live_check(cases.t_prem(reduced=True), form)
    msg = f"all combinations {ALL_FORMULATIONS.index(form)} prem_q: {worst:.2e}, imbalance {imb:.2e}"
    if form.gravity is sj.GravityModel.FULL and isinstance(form.background, sj.Hydrostatic):
        w2, i2 = _live_check(cases.t_hager(reduced=True, core="consistent"), form)
        msg += f"; hager_cc_q: {w2:.2e}, imbalance {i2:.2e}"
    print(msg)


HIGH_DEGREE_FLAGS = {
    "nsfs": sj.Formulation(surface=sj.NoSlip()),
    "nsns": sj.Formulation(surface=sj.NoSlip(), cmb=sj.NoSlip()),
    "tala": sj.Formulation(compressible=True),
    "nsg": sj.Formulation(gravity=sj.GravityModel.NO_SELF_GRAVITY),
    "uniform": sj.Formulation(background=sj.Uniform(9.81)),
}
HIGH_DEGREE_TABLES = {"prem": (cases.t_prem, ORACLE_TOL), "stiff_lid": (cases.t_stiff_lid, CONTRAST_TOL),
                      "wide10": (cases.t_wide10, CONTRAST_TOL)}


@pytest.mark.slow
@pytest.mark.parametrize("flags", list(HIGH_DEGREE_FLAGS))
@pytest.mark.parametrize("table_name", list(HIGH_DEGREE_TABLES))
def test_flags_high_degree(table_name, flags):
    """The five flag configurations at ``l`` in {2, 10, 50, 100} against the live oracle at
    about 25 sources per table: the full PREM table at ``ORACLE_TOL``, the stiff lid and
    ten decades of viscosity at ``CONTRAST_TOL`` (review r1 minor 3; measured by the
    review 5.3e-14, 5.8e-14, 5.5e-13)."""
    make, tol = HIGH_DEGREE_TABLES[table_name]
    table = make()
    form = HIGH_DEGREE_FLAGS[flags]
    degrees = (2, 10, 50, 100)
    N = len(table.radius)
    sources = tuple(sorted(set(np.linspace(0, N - 1, 25).round().astype(int).tolist())))
    sol = _solve(table, 100, form)
    ref = ro.kernels(table, _physics(form), ro.Discretisation(), degrees, sources=sources)
    worst = 0.0
    for k, l in enumerate(degrees):
        kk = sol["degrees"].index(l)
        for o in SI_OUTPUTS:
            worst = max(worst, assert_rows_close(sol[o][kk][list(sources)], ref[o][k], tol, f"{table_name} {flags} l={l} {o}"))
    print(f"high degree {table_name} {flags}: worst e_l {worst:.2e}")
