"""Derivatives of the kernels (plan 7.6 and 7.7; gates G5, G6, G7).

Taylor policy of CONVENTIONS 8 through tests/taylor.py: the fixed ladder and window,
both direction signs, reverse and forward mode, the rate bounds 1.9 and (0.8, 1.2),
"unresolved" is not a pass. Negative controls: the derivative times 1.01 must fail for
both signs and both modes (smooth class); the zero class has the two controls of plan
7.6. Dot-product test ``assert_jvp_vjp`` on the map ``G`` from the control to the four
kernels, each divided by its constant scale ``s_o``, so that the random cotangent
weights the four kernels equally (plan 7.6, review r2 C1).

Reference of the Taylor tests: the forward values of the same functional. Reference of
the derivative test of the slow set: a central difference of the mpmath oracle at 80
digits, the only derivative check that does not use JAX.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from decimal import Decimal

import mpmath as mp
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import cases  # noqa: E402
import reference_mpmath as ro  # noqa: E402
from adapters import SI_OUTPUTS, assert_rows_close, profile_from_table  # noqa: E402
from taylor import (EPS64, assert_jvp_vjp, assert_taylor, assess, random_like, remeasure_taylor,  # noqa: E402
                    taylor_remainders)

import stokesjx as sj  # noqa: E402
from stokesjx import propagate  # noqa: E402

# ---- tolerances (CONVENTIONS 9) ----
# Dot product <dy, J dx> = <J^T dy, dx> on the normalised map G, relative (CONVENTIONS
# 8). Measured over ten seeds on the review prototypes: at most 1.2e-14 (T1), 1.6e-14
# (T3), 1.5e-14 (T4), 3.9e-14 (T8), 4.9e-15 and 8.1e-15 over the whole profile pytree
# (plan-review-r2-evidence/r2_dot_seeds.out, plan-review-r3-evidence/r3_pytree.out,
# r3_consistent.out). Cause: rounding of the forward and the reverse pass.
DOT_RTOL = 1e-12
# The same on the stiff lid (T9). The derivative on a thin stiff lid is correct and
# less accurate than the values: measured with the normalised map over ten seeds at most
# 2.0e-12, median 4.5e-13; forward and reverse Jacobians differ by up to 3.8e-11 of the
# largest entry of a kernel and degree (hager4: 2.1e-14), and against a central
# difference of the oracle the forward mode is accurate to 8.9e-13 and the reverse mode
# to 4.9e-12 (plan-review-r2-evidence/r2_dot_seeds.out, r2_jac.out,
# r2_deriv_oracle.out). A factor 50 above the largest measured value. A loosening of
# CONVENTIONS 8 for one case, logged in NOTES/TOLERANCES.md.
DOT_RTOL_CONTRAST = 1e-10
# Derivative with respect to log eta against a central difference of the oracle at 80
# digits (step 1e-15, truncation error about 1e-30), max-normalised per kernel and
# degree. hager4: measured 3.3e-14 forward and 2.9e-14 reverse on the prototype
# (r2_deriv_oracle.out). Cause: rounding of the sweeps and of the node solve in the
# forward and the reverse pass.
DERIV_ORACLE_TOL = 1e-12
# The same on the stiff lid: measured 8.9e-13 forward and 4.9e-12 reverse on the
# prototype; same cause as DOT_RTOL_CONTRAST. Logged in NOTES/TOLERANCES.md.
DERIV_ORACLE_TOL_CONTRAST = 1e-10
# Zero shift identity: the geoid and topography part of F does not change when every
# viscosity is multiplied by one factor (eq. 11), so the sum of its derivatives with
# respect to log eta is zero up to rounding, relative to the sum of their absolute
# values. Cause: rounding of the reverse pass.
ZERO_SHIFT_TOL = 1e-10
# dF_vel/ds = -F_vel for the uniform shift s of every log eta (the velocity scales as
# 1/eta), relative. Cause: rounding of the derivative.
SHIFT_TOL = 1e-10
# vmap over three control sets against three single calls: e_l for the kernels and
# max |g_vmap - g_single| / max |g_single| for the gradient (an elementwise relative
# difference is not used: entries next to a zero of a kernel give 31 there). Measured
# 9.3e-15 and 2.5e-15 on the reduced hager4 table (plan-review-r2-evidence/r2_gates.out).
# Cause: a different order of the float64 operations in the batched program.
VMAP_RTOL = 1e-12
# Symmetry of the Hessian of F, relative to its largest entry (cause: rounding of
# the forward-over-reverse pass).
HESSIAN_SYM_TOL = 1e-10
# Negative control of the smooth class (CONVENTIONS 8).
NEG_BIAS = 1.01

WEIGHT_SEED = 4


# ---------------------------------------------------------------------------
# Functionals of a case.
# ---------------------------------------------------------------------------


def _kernels(profile, cfg):
    K = sj.stokes_kernels(profile, cfg)
    return K.geoid, K.surface_topography, K.cmb_topography, K.surface_velocity


class Case:
    """A control map ``x -> profile`` with the functional ``F`` (a weighted sum of the
    four kernels, each divided by its largest absolute value at the base point) and
    the normalised kernel map ``G`` of the dot-product test.

    ``formulation`` and ``numerics`` set the flags and the numerical options of the
    configuration (defaults: the default flags, so the c2 cases do not change). An
    output whose base value is zero everywhere (the velocity under a no-slip surface)
    has the scale 0 and is left out of ``F`` and ``G`` (``self.active``); the case
    records it in ``self.zero_outputs``, and the test asserts that it is exactly zero."""

    def __init__(self, p0, lmax, to_profile, x0, outputs=(0, 1, 2, 3), formulation=None, numerics=None):
        self.cfg = sj.StokesConfig(lmax=lmax, formulation=formulation or sj.Formulation(),
                                   numerics=numerics or sj.Numerics())
        self.to_profile = to_profile
        self.x0 = x0
        K0 = [np.asarray(k) for k in _kernels(to_profile(x0), self.cfg)]
        self.K0 = K0
        self.scales = tuple(float(np.max(np.abs(k))) for k in K0)
        self.active = tuple(o for o in range(4) if self.scales[o] > 0)
        self.zero_outputs = tuple(o for o in range(4) if self.scales[o] == 0)
        rng = np.random.default_rng(WEIGHT_SEED)
        self.weights = tuple(jnp.asarray(rng.standard_normal(k.shape)) for k in K0)
        self.outputs = tuple(o for o in outputs if o in self.active)

    def G(self, x):
        K = _kernels(self.to_profile(x), self.cfg)
        return tuple(K[o] / self.scales[o] for o in self.active)

    def F(self, x, outputs=None):
        K = _kernels(self.to_profile(x), self.cfg)
        outputs = self.outputs if outputs is None else outputs
        return sum(jnp.sum(self.weights[o] * K[o]) / self.scales[o] for o in outputs)

    def abs_terms(self, outputs):
        """``sum_o sum_(l, j) |w K / s|`` at the base point (the size of the terms of F)."""
        return sum(float(np.sum(np.abs(np.asarray(self.weights[o]) * self.K0[o]))) / self.scales[o]
                   for o in outputs)


def _smooth_case(case: Case, dx, dot_rtol=DOT_RTOL):
    """Taylor test (both signs, both modes), the 1.01 control for every measurement, and
    the dot-product test on G."""
    results = assert_taylor(case.F, case.x0, dx)
    for res in results:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative))[0] == "fail", res.table()
    rel = assert_jvp_vjp(case.G, case.x0, rtol=dot_rtol)
    print(f"dot product {rel:.2e}; allowances at rung 3: "
          f"{min(r.r1[3] / r.allowance for r in results):.1e}")
    return results


def _log_eta_case(table, lmax):
    p0 = profile_from_table(table)
    return Case(p0, lmax, lambda x: p0.replace(eta=jnp.exp(x)), np.log(np.asarray(p0.eta)))


def _pair_direction(table, size, seed):
    """``size N(0, 1)`` m per distinct radius; both nodes of a pair move together."""
    r = np.array([float(x) for x in table.radius])
    rng = np.random.default_rng(seed)
    d = np.empty_like(r)
    for i in range(len(r)):
        d[i] = d[i - 1] if i > 0 and r[i] == r[i - 1] else size * rng.standard_normal()
    return d


# ---------------------------------------------------------------------------
# Required Taylor cases (plan 7.6, section 8).
# ---------------------------------------------------------------------------


@pytest.mark.taylor_required
def test_t1_log_eta():
    """T1: log eta per node, reduced hager4 (23 nodes, 3 pairs), lmax = 20, 0.1 N(0, 1)."""
    case = _log_eta_case(cases.t_hager(reduced=True), 20)
    _smooth_case(case, 0.1 * np.random.default_rng(1).standard_normal(case.x0.shape))


@pytest.mark.taylor_required
def test_t2_misfit():
    """T2: log eta, the misfit of the HMC shape against the kernels of a second
    viscosity profile (l >= 2, all four kernels, normalised); the dot product on the
    scalar misfit covers the reverse pass through it."""
    table = cases.t_hager(reduced=True)
    p0 = profile_from_table(table)
    cfg = sj.StokesConfig(lmax=20)
    x0 = np.log(np.asarray(p0.eta))
    x1 = x0 + 0.3 * np.random.default_rng(2).standard_normal(x0.shape)
    K1 = [np.asarray(k)[1:] for k in _kernels(p0.replace(eta=jnp.exp(x1)), cfg)]
    scales = [float(np.max(np.abs(k))) for k in K1]

    def F_m(x):
        K = _kernels(p0.replace(eta=jnp.exp(x)), cfg)
        return 0.5 * sum(jnp.sum(((K[o][1:] - K1[o]) / scales[o]) ** 2) for o in range(4))
    # Direction 0.01 N(0, 1), not the 0.1 N(0, 1) of T1: along this direction the
    # misfit is strongly curved compared with its slope (slope 1.6e-2, second-order
    # coefficient 0.25 at 0.1 N(0, 1)), and at 0.1 the 1.01 control passes on the fixed
    # window because the bias term stays below the second-order term. Scaling the
    # direction by 0.1 makes the bias term ten times larger relative to the
    # second-order term; the window is unchanged (CONVENTIONS 8, NOTES/TOLERANCES.md).
    dx = 0.01 * np.random.default_rng(3).standard_normal(x0.shape)
    results = assert_taylor(F_m, x0, dx)
    for res in results:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative))[0] == "fail", res.table()
    assert_jvp_vjp(F_m, x0, rtol=DOT_RTOL)


@pytest.mark.taylor_required
def test_t3_rho0():
    """T3: rho0 per node, reduced PREM (25 nodes, 4 pairs, density jumps at three of
    them), lmax = 20, 50 N(0, 1) kg/m^3 (5 N(0, 1) kg/m^3 is unresolved: plan 7.6)."""
    table = cases.t_prem(reduced=True)
    p0 = profile_from_table(table)
    case = Case(p0, 20, lambda x: p0.replace(rho0=x), np.asarray(p0.rho0))
    _smooth_case(case, 50.0 * np.random.default_rng(5).standard_normal(case.x0.shape))


@pytest.mark.taylor_required
def test_t4_radius():
    """T4: radius, reduced hager4, lmax = 20, 5000 N(0, 1) m per distinct radius (both
    nodes of a pair move together; the largest step, 50 m times a normal deviate,
    keeps the order of the nodes: the smallest gap is 38.75 km)."""
    table = cases.t_hager(reduced=True)
    p0 = profile_from_table(table)
    case = Case(p0, 20, lambda x: p0.replace(radius=x), np.asarray(p0.radius))
    _smooth_case(case, _pair_direction(table, 5000.0, 6))


def _shift_case():
    table = cases.t_hager(reduced=True)
    p0 = profile_from_table(table)
    eta0 = jnp.asarray(p0.eta)
    return Case(p0, 20, lambda s: p0.replace(eta=eta0 * jnp.exp(s)), np.array(0.0))


@pytest.mark.taylor_required
def test_t5_shift_zero_class():
    """T5: the uniform shift of every log eta, geoid and topography part of F: zero
    class (the geoid and the topographies do not change, eq. 11).

    The allowance is ``64 eps sum |w K / s|`` over the terms of the functional
    (``ZERO_CLASS_ALLOWANCE``): F_gt is a sum of terms of both signs, so its rounding is
    about ``eps sum |terms|``, not ``eps |F_gt|`` (plan 7.6, logged in TOLERANCES.md).
    Controls: (a) the derivative of the velocity part in place of the zero derivative
    must fail the derivative check; (b) the zero class on the whole functional
    (velocity included) must fail the forward-response check."""
    case = _shift_case()
    gt = (0, 1, 2)
    allowance = 64 * EPS64 * case.abs_terms(gt)
    F_gt = lambda s: case.F(s, gt)
    results = assert_taylor(F_gt, case.x0, np.array(1.0), classification="zero", allowance=allowance)
    F_vel0 = float(case.F(case.x0, (3,)))
    for res in results:
        ctrl = remeasure_taylor(res, -F_vel0)
        assert assess(ctrl, classification="zero")[0] == "fail", ctrl.table()
    for sign in (1.0, -1.0):
        whole = taylor_remainders(case.F, case.x0, np.array(1.0), sign=sign, derivative=0.0, allowance=allowance)
        assert assess(whole, classification="zero")[0] == "fail", whole.table()


@pytest.mark.taylor_required
def test_t6_shift_velocity():
    """T6: the same shift, velocity part of F: smooth, and dF_vel/ds = -F_vel."""
    case = _shift_case()
    F_vel = lambda s: case.F(s, (3,))
    results = assert_taylor(F_vel, case.x0, np.array(1.0))
    for res in results:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative))[0] == "fail", res.table()
    f0 = float(F_vel(case.x0))
    for res in results:
        assert abs(res.derivative + f0) <= SHIFT_TOL * abs(f0), (res.derivative, f0)
    assert_jvp_vjp(lambda s: case.G(s), case.x0, rtol=DOT_RTOL)


@pytest.mark.slow
@pytest.mark.taylor_required
def test_t7_golden_table():
    """T7: log eta per node on the LEBEDEV table of the hager4_fsfs golden run (129
    nodes), lmax = 50, 0.1 N(0, 1). In the slow set (5.5 s; the quick-set budget of
    CONVENTIONS 7), as the c2 plan (risk R6) foresaw."""
    t = cases.golden_table("hager4_fsfs")
    table = replace(t, core_mass=cases.PREM_CORE_MASS)
    case = _log_eta_case(table, 50)
    _smooth_case(case, 0.1 * np.random.default_rng(7).standard_normal(case.x0.shape))


@pytest.mark.slow
@pytest.mark.taylor_required
def test_t8_wide5():
    """T8: log eta per node on T_wide5 (135 nodes, five decades), lmax = 100."""
    case = _log_eta_case(cases.t_wide5(), 100)
    _smooth_case(case, 0.1 * np.random.default_rng(8).standard_normal(case.x0.shape))


@pytest.mark.slow
@pytest.mark.taylor_required
def test_t9_stiff_lid():
    """T9: log eta per node on T_stiff_lid (135 nodes, seven decades), lmax = 50, with
    DOT_RTOL_CONTRAST for the dot product."""
    case = _log_eta_case(cases.t_stiff_lid(), 50)
    _smooth_case(case, 0.1 * np.random.default_rng(9).standard_normal(case.x0.shape), DOT_RTOL_CONTRAST)


@pytest.mark.taylor_required
def test_t10_scalar_leaves():
    """T10: the four scalar leaves (core density, core mass, water density, topography
    density) together, each 0.1 N(0, 1) times its size, on the reduced PREM table with
    the PREM core mass, rho_w = 1030 and rho_top = 2600, lmax = 20."""
    table = replace(cases.t_prem(reduced=True), outer_density="1030")
    p0 = profile_from_table(table, surface_density=2600.0)
    keys = ("core_density", "core_mass", "outer_density", "surface_density")
    x0 = {k: np.asarray(getattr(p0, k)) for k in keys}
    case = Case(p0, 20, lambda x: p0.replace(**x), x0)
    rng = np.random.default_rng(10)
    dx = {k: 0.1 * rng.standard_normal() * np.abs(v) for k, v in x0.items()}
    _smooth_case(case, dx)


@pytest.mark.taylor_required
def test_t11a_consistent_core_density():
    """T11a: the core density on the consistent core (the core mass follows rho_c inside
    the differentiated function), 0.05 x 9900 kg/m^3, reduced hager4, lmax = 20."""
    p0 = profile_from_table(cases.t_hager(reduced=True, core="consistent"))
    case = Case(p0, 20, lambda x: p0.replace(core_density=x), np.asarray(p0.core_density))
    _smooth_case(case, np.array(0.05 * 9900.0))


@pytest.mark.taylor_required
def test_t11b_consistent_core_radius():
    """T11b: the radius on the consistent core (moves r_c, so the core mass), as T4."""
    table = cases.t_hager(reduced=True, core="consistent")
    p0 = profile_from_table(table)
    case = Case(p0, 20, lambda x: p0.replace(radius=x), np.asarray(p0.radius))
    _smooth_case(case, _pair_direction(table, 5000.0, 11))


# ---------------------------------------------------------------------------
# Taylor cases of the flags and the fields (T12 to T18;
# NOTES/chunks/c3-flags-fields/01-plan.md, section 6).
# ---------------------------------------------------------------------------


def _all_controls(table, p0, seed=13):
    """The control map of ``log eta``, ``rho0`` and the radius together, and its
    direction: ``0.1 N(0, 1)`` in ``log eta``, ``50 N(0, 1)`` kg/m^3 in ``rho0``,
    ``5000 N(0, 1)`` m per distinct radius (both nodes of a pair together), from the
    seeds ``seed`` and ``seed + 1``."""
    x0 = dict(log_eta=np.log(np.asarray(p0.eta)), rho0=np.asarray(p0.rho0), radius=np.asarray(p0.radius))
    rng = np.random.default_rng(seed)
    dx = dict(log_eta=0.1 * rng.standard_normal(x0["log_eta"].shape),
              rho0=50.0 * rng.standard_normal(x0["rho0"].shape),
              radius=_pair_direction(table, 5000.0, seed + 1))
    to_profile = lambda x: p0.replace(eta=jnp.exp(x["log_eta"]), rho0=x["rho0"], radius=x["radius"])
    return to_profile, x0, dx


@pytest.mark.taylor_required
def test_t12_tala_rho0():
    """T12: ``rho0`` per node under TALA on the reduced PREM table (density jumps at three
    pairs), lmax = 20, 50 N(0, 1) kg/m^3: the new paths of the TALA term
    (``kappa_h = ln(rho0_i / rho0_(i-1))``) and of the closed-form pair step."""
    table = cases.t_prem(reduced=True)
    p0 = profile_from_table(table)
    case = Case(p0, 20, lambda x: p0.replace(rho0=x), np.asarray(p0.rho0),
                formulation=sj.Formulation(compressible=True))
    _smooth_case(case, 50.0 * np.random.default_rng(12).standard_normal(case.x0.shape))


@pytest.mark.taylor_required
def test_t13_no_self_gravity_uniform():
    """T13: NO_SELF_GRAVITY with ``Uniform(9.81)`` (the G-ADOPT level; the uniform
    background and the raw frame at ``l = 1``), ``log eta``, ``rho0`` and the radius
    together, reduced hager4, lmax = 20."""
    table = cases.t_hager(reduced=True)
    p0 = profile_from_table(table)
    to_profile, x0, dx = _all_controls(table, p0)
    form = sj.Formulation(gravity=sj.GravityModel.NO_SELF_GRAVITY, background=sj.Uniform(9.81))
    case = Case(p0, 20, to_profile, x0, formulation=form)
    assert sj.stokes_kernels(p0, case.cfg).degree_one_frame is sj.DegreeOneFrame.RAW
    _smooth_case(case, dx)


@pytest.mark.taylor_required
def test_t14_tala_no_slip():
    """T14: FULL, hydrostatic, TALA, no slip at both boundaries (degree 1 refused,
    degrees 2 to 20; the velocity is an exact zero and is left out of F and G), ``log
    eta``, ``rho0`` and the radius together, reduced PREM, lmax = 20."""
    table = cases.t_prem(reduced=True)
    p0 = profile_from_table(table)
    to_profile, x0, dx = _all_controls(table, p0)
    form = sj.Formulation(compressible=True, surface=sj.NoSlip(), cmb=sj.NoSlip())
    case = Case(p0, 20, to_profile, x0, formulation=form)
    assert case.zero_outputs == (3,) and np.all(case.K0[3] == 0)
    assert sj.stokes_kernels(p0, case.cfg).degrees == tuple(range(2, 21))
    _smooth_case(case, dx)
    g = jax.jit(jax.grad(lambda x: jnp.sum(_kernels(to_profile(x), case.cfg)[3])))(x0)
    for k, v in g.items():
        assert np.all(np.asarray(v) == 0), k


@pytest.mark.slow
@pytest.mark.taylor_required
def test_t15_field_misfit():
    """T15: ``log eta`` and the radius together through the field misfit
    ``0.5 sum_(o, l >= 2, m, C/S) ((F_o - F1_o) / s_o)^2`` of ``apply`` with a fixed random
    ``real_4pi`` density ``(N, 2, 21, 21)`` and the fields ``F1`` of a second viscosity
    profile; the radius moves the quadrature weights too. Reduced hager4, lmax = 20.

    Direction ``0.01 N(0, 1)`` in ``log eta`` and ``500 N(0, 1)`` m in the radius, a tenth
    of the amplitudes of the other cases: along the larger direction the misfit is
    strongly curved compared with its slope (uncorrected rate 1.25 on the first window
    pair, outside (0.8, 1.2); NOTES/chunks/c3-flags-fields/plan-review-r2-evidence/
    t15_detail.out), the same cause as T2 (CONVENTIONS 8, NOTES/TOLERANCES.md). In the
    slow set: the third move of the quick-set rule of
    NOTES/chunks/c3-flags-fields/01-plan.md (risk R1)."""
    table = cases.t_hager(reduced=True)
    p0 = profile_from_table(table)
    cfg = sj.StokesConfig(lmax=20)
    N = p0.structure.n_nodes
    rng = np.random.default_rng(15)
    density = jnp.asarray(rng.standard_normal((N, 2, 21, 21)) * np.tril(np.ones((21, 21))))
    x0 = dict(log_eta=np.log(np.asarray(p0.eta)), radius=np.asarray(p0.radius))
    to_profile = lambda x: p0.replace(eta=jnp.exp(x["log_eta"]), radius=x["radius"])

    def fields_of(x):
        p = to_profile(x)
        F = sj.apply(sj.stokes_kernels(p, cfg), density, p)
        return [getattr(F, n)[:, 2:, :] for n in sj.fields.FIELD_NAMES]
    x1 = dict(log_eta=x0["log_eta"] + 0.3 * rng.standard_normal(N), radius=x0["radius"])
    F1 = [np.asarray(f) for f in fields_of(x1)]
    scales = [float(np.max(np.abs(f))) for f in F1]

    def misfit(x):
        return 0.5 * sum(jnp.sum(((f - f1) / s) ** 2) for f, f1, s in zip(fields_of(x), F1, scales))
    dx = dict(log_eta=0.01 * rng.standard_normal(N), radius=_pair_direction(table, 500.0, 16))
    results = assert_taylor(misfit, x0, dx)
    for res in results:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative))[0] == "fail", res.table()
    rel = assert_jvp_vjp(misfit, x0, rtol=DOT_RTOL)
    print(f"T15 dot product {rel:.2e}")


_ALL_FORMS = [sj.Formulation(gravity=g, background=b, compressible=c, surface=s, cmb=m)
              for g in (sj.GravityModel.FULL, sj.GravityModel.NO_SELF_GRAVITY)
              for b in (sj.Hydrostatic(), sj.Uniform(9.81)) for c in (False, True)
              for s in (sj.FreeSlip(), sj.NoSlip()) for m in (sj.FreeSlip(), sj.NoSlip())]


# T16: the smallest admissible second-order term of the combined direction, as a
# fraction of the largest second-order term of its three control components (log eta,
# rho0, radius), measured by second differences of F (no derivative of the code under
# test). Below it the direction is degenerate for a rate test (see test_t16).
T16_CURVATURE_MIN = 0.05


def _second_difference(F, x0, dx, step=2.5e-3):
    """``|F(x + s dx) + F(x - s dx) - 2 F(x)|``: about ``|d^2F[dx, dx]| s^2``."""
    shift = lambda a: jax.tree_util.tree_map(lambda x, d: x + a * d, x0, dx)
    return abs(float(F(shift(step))) + float(F(shift(-step))) - 2.0 * float(F(x0)))


def _curved_direction(F, table, p0, k):
    """The direction of T16 for combination ``k``: the first of the seeds
    ``1000 + 2k + 100 j`` (``j = 0, 1, ...``) whose second-order term is at least
    ``T16_CURVATURE_MIN`` times the largest second-order term of its components."""
    for j in range(10):
        seed = 1000 + 2 * k + 100 * j
        _, x0, dx = _all_controls(table, p0, seed=seed)
        whole = _second_difference(F, x0, dx)
        parts = max(_second_difference(F, x0, {c: (v if c == name else 0.0 * v) for c, v in dx.items()})
                    for name in dx)
        if whole >= T16_CURVATURE_MIN * parts:
            return dx, seed, whole / parts
    raise AssertionError(f"no direction with a second-order term above {T16_CURVATURE_MIN} of its components")


@pytest.mark.slow
@pytest.mark.taylor_required
@pytest.mark.parametrize("k", range(32))
def test_t16_every_combination(k):
    """T16: each of the 32 flag combinations, ``log eta``, ``rho0`` and the radius
    together, reduced PREM, lmax = 10.

    Direction rule, fixed before the run (CONVENTIONS 8; NOTES/TOLERANCES.md). With one
    random direction for all 32, the second-order terms of the three controls can
    cancel along it: on combination 9 (FULL, uniform ``g``, free-slip surface, no-slip
    CMB) each control alone gives a corrected remainder of about 1e-6 at the first rung
    and the three together 7.6e-10, so the third-order term (and at smaller steps the
    rounding) dominates the window and the rates fail although the derivative is
    correct (the dot product passes; NOTES/chunks/c3-flags-fields/impl-evidence/
    t16_case9.out). Over ten seeds such a cancellation below 1e-2 happens for 1 of 70
    directions (t16_seeds.out), so it is an accident of one direction, not a property of
    a combination. The rule skips a direction whose second-order term is below
    ``T16_CURVATURE_MIN`` of the largest of its components (``_curved_direction``),
    judged by second differences of the forward values only; the Taylor test, the 1.01
    controls and the dot product then run unchanged on the chosen direction."""
    table = cases.t_prem(reduced=True)
    p0 = profile_from_table(table)
    to_profile, x0, _ = _all_controls(table, p0)
    case = Case(p0, 10, to_profile, x0, formulation=_ALL_FORMS[k])
    for o in case.zero_outputs:
        assert np.all(case.K0[o] == 0)
    dx, seed, ratio = _curved_direction(case.F, table, p0, k)
    print(f"T16 combination {k}: seed {seed}, curvature ratio {ratio:.2e}")
    _smooth_case(case, dx)


# The dot product of the degree-one imbalance, absolute: the imbalance is a ratio of
# two forces of order one computed from a small difference (tests/test_kernels_oracle.py,
# IMBALANCE_ATOL), so its directional derivative has an absolute rounding error. Measured
# 4e-15 absolute by the JAX review r1 of c2
# (NOTES/chunks/c2-forward-core/impl-review-jax-r1-evidence/chk_imbalance_dot.py).
IMBALANCE_DOT_ATOL = 1e-13


@pytest.mark.slow
@pytest.mark.taylor_required
def test_t17_degree_one_imbalance():
    """T17: ``degree_one_imbalance`` (CM frame, split core, default flags) along ``log
    eta``, ``rho0`` and the radius together, reduced PREM, lmax = 1: Taylor test of a
    weighted sum and the dot product with an absolute bound."""
    table = cases.t_prem(reduced=True)
    p0 = profile_from_table(table)
    to_profile, x0, dx = _all_controls(table, p0)
    cfg = sj.StokesConfig(lmax=1)
    imb = lambda x: sj.stokes_kernels(to_profile(x), cfg).degree_one_imbalance
    i0 = np.asarray(imb(x0))
    assert np.max(np.abs(i0)) > 1e-4                       # a nonzero applied field
    w = jnp.asarray(np.random.default_rng(17).standard_normal(i0.shape))
    F = lambda x: jnp.sum(w * imb(x)) / np.max(np.abs(i0))
    results = assert_taylor(F, x0, dx)
    for res in results:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative))[0] == "fail", res.table()
    dy = jnp.asarray(np.random.default_rng(18).standard_normal(i0.shape))
    a = float(jnp.vdot(dy, jax.jit(lambda x, d: jax.jvp(imb, (x,), (d,))[1])(x0, dx)))
    jt = jax.jit(lambda x, c: jax.vjp(imb, x)[1](c)[0])(x0, dy)
    b = sum(float(np.vdot(np.asarray(jt[k]), dx[k])) for k in dx)
    print(f"T17 dot product: {a:.6e} {b:.6e}, absolute {abs(a - b):.2e}")
    assert abs(a - b) <= IMBALANCE_DOT_ATOL, (a, b)


# The pivoted elimination against LAPACK through stokes_kernels, e_l per kernel and
# degree: both solve the same equilibrated node systems; measured 4.2e-15 and 6.5e-15 on
# the benchmark tables (NOTES/chunks/c2-forward-core/03-implementation.md). Cause: the
# rounding of two different eliminations.
NODE_SOLVER_TOL = 1e-13
# The gradient of T1 with both solvers, max-normalised. Cause: the same, through the
# reverse pass.
NODE_SOLVER_GRAD_TOL = 1e-12


@pytest.mark.slow
@pytest.mark.taylor_required
def test_t18_pivot_elimination():
    """T18: ``node_solver="pivot_elimination"`` through ``stokes_kernels`` (reduced hager4,
    lmax = 20): the values against ``"lapack"``, the Taylor test of T1, and the gradient
    of T1 against ``"lapack"``."""
    table = cases.t_hager(reduced=True)
    p0 = profile_from_table(table)
    to_profile = lambda x: p0.replace(eta=jnp.exp(x))
    x0 = np.log(np.asarray(p0.eta))
    piv = Case(p0, 20, to_profile, x0, numerics=sj.Numerics(node_solver="pivot_elimination"))
    lap = Case(p0, 20, to_profile, x0)
    for o in range(4):
        assert_rows_close(piv.K0[o], lap.K0[o], NODE_SOLVER_TOL, f"pivot vs lapack {o}")
    _smooth_case(piv, 0.1 * np.random.default_rng(1).standard_normal(x0.shape))
    gp = np.asarray(jax.jit(jax.grad(piv.F))(x0))
    gl = np.asarray(jax.jit(jax.grad(lap.F))(x0))
    assert np.max(np.abs(gp - gl)) <= NODE_SOLVER_GRAD_TOL * np.max(np.abs(gl))


# ---------------------------------------------------------------------------
# Dot products over the whole profile pytree.
# ---------------------------------------------------------------------------


def _pytree_dot(p, lmax, seed):
    case = Case(p, lmax, lambda q: q, p)
    leaves = jax.tree_util.tree_leaves(p)
    dx = jax.tree_util.tree_map(lambda r, a: r * np.abs(np.asarray(a)), random_like(p, seed), p)
    rel = assert_jvp_vjp(case.G, p, dx=dx, rtol=DOT_RTOL)
    grad = jax.jit(jax.grad(case.F))(p)
    return case, grad, rel, leaves


def test_pytree_dot_split_core():
    """The whole profile (all seven leaves) on the reduced PREM table with the PREM core
    mass, rho_w = 1030 and rho_top = 2600; the gradient of F with respect to every
    scalar leaf is finite and nonzero."""
    table = replace(cases.t_prem(reduced=True), outer_density="1030")
    p = profile_from_table(table, surface_density=2600.0)
    case, grad, rel, leaves = _pytree_dot(p, 20, 21)
    assert len(leaves) == 7
    for k in ("radius", "rho0", "eta", "core_density", "core_mass", "outer_density", "surface_density"):
        g = np.asarray(getattr(grad, k))
        assert np.all(np.isfinite(g)), k
        if g.ndim == 0:
            assert g != 0, k


def test_pytree_dot_consistent_core():
    """The same on the consistent core (five leaves; rho_w = 0, so the direction does not
    move it); every gradient finite, the core density gradient nonzero."""
    p = profile_from_table(cases.t_hager(reduced=True, core="consistent"))
    case, grad, rel, leaves = _pytree_dot(p, 20, 22)
    assert len(leaves) == 5
    for g in jax.tree_util.tree_leaves(grad):
        assert np.all(np.isfinite(np.asarray(g)))
    assert float(grad.core_density) != 0


# ---------------------------------------------------------------------------
# G5: finite gradients; the zero-shift identity; vmap; Hessian.
# ---------------------------------------------------------------------------

G5_TABLES = {
    "hager4": cases.t_hager, "hager4_cc": lambda: cases.t_hager(core="consistent"), "wide5": cases.t_wide5,
    "prem": cases.t_prem, "uniform_cc": cases.t_uniform, "layers_cc": cases.t_layers,
    "stiff_lid": cases.t_stiff_lid, "wide10": cases.t_wide10, "rough8": cases.t_rough8,
    "smooth64": lambda: cases.t_smooth(64),
}
G5_LMAX100 = ("hager4", "wide5", "stiff_lid")


@pytest.mark.parametrize("name", list(G5_TABLES))
def test_g5_gradient_finite(name):
    """G5: the gradient of F with respect to log eta, rho0 and the radius (one reverse
    pass) is finite on every table of plan 7.4 at lmax = 50; hager4, wide5 and the stiff
    lid also at lmax = 100. (The near-consistent cores and the water variants share the
    structure of hager4.)"""
    table = G5_TABLES[name]()
    p0 = profile_from_table(table)
    x0 = dict(log_eta=np.log(np.asarray(p0.eta)), rho0=np.asarray(p0.rho0), radius=np.asarray(p0.radius))
    to_profile = lambda x: p0.replace(eta=jnp.exp(x["log_eta"]), rho0=x["rho0"], radius=x["radius"])
    for lmax in (50, 100) if name in G5_LMAX100 else (50,):
        case = Case(p0, lmax, to_profile, x0)
        g = jax.jit(jax.grad(case.F))(x0)
        for k, v in g.items():
            assert np.all(np.isfinite(np.asarray(v))), (name, lmax, k)
            assert np.any(np.asarray(v) != 0), (name, lmax, k)


def test_zero_shift_identity():
    """The geoid and topography part of F does not depend on a common factor of the
    viscosities, so the derivatives with respect to log eta sum to zero (eq. 11); full
    hager4 table, lmax = 50."""
    case = _log_eta_case(cases.t_hager(), 50)
    g = np.asarray(jax.jit(jax.grad(lambda x: case.F(x, (0, 1, 2))))(case.x0))
    assert abs(np.sum(g)) <= ZERO_SHIFT_TOL * np.sum(np.abs(g)), (np.sum(g), np.sum(np.abs(g)))


def test_vmap_equals_single_calls():
    """vmap over three control sets equals three single calls, values and gradients."""
    case = _log_eta_case(cases.t_hager(reduced=True), 20)
    rng = np.random.default_rng(31)
    X = case.x0[None] + 0.2 * rng.standard_normal((3,) + case.x0.shape)
    Gv = jax.jit(jax.vmap(case.G))(X)
    gv = jax.jit(jax.vmap(jax.grad(case.F)))(X)
    G1 = jax.jit(case.G)
    g1 = jax.jit(jax.grad(case.F))
    for b in range(3):
        single = G1(X[b])
        for o in range(4):
            assert_rows_close(np.asarray(Gv[o][b]), np.asarray(single[o]), VMAP_RTOL, f"vmap {o}")
        gs = np.asarray(g1(X[b]))
        assert np.max(np.abs(np.asarray(gv[b]) - gs)) <= VMAP_RTOL * np.max(np.abs(gs))


def test_hessian_finite_and_symmetric():
    """jax.hessian of F with respect to log eta on a five-node table at lmax = 3 is
    finite and symmetric (HMC mass-matrix adaptation can use it)."""
    r = np.array([3480e3, 4200e3, 5000e3, 5800e3, 6370e3])
    p0 = sj.RadialProfile(r, np.array([5500.0, 5000.0, 4600.0, 3900.0, 3300.0]),
                          np.array([1e22, 3e21, 1e21, 1e20, 1e22]), core_density=9900.0)
    case = Case(p0, 3, lambda x: p0.replace(eta=jnp.exp(x)), np.log(np.asarray(p0.eta)))
    H = np.asarray(jax.jit(jax.hessian(case.F))(case.x0))
    assert np.all(np.isfinite(H))
    assert np.max(np.abs(H - H.T)) <= HESSIAN_SYM_TOL * np.max(np.abs(H))


# ---------------------------------------------------------------------------
# G7: trace count (CONVENTIONS 2, plan 7.7).
# ---------------------------------------------------------------------------


def test_trace_count():
    """A jitted call with a profile from NumPy arrays, then from JAX arrays, then with new
    values of the same shapes traces the core once; the outer jit has one cache entry.
    The same for the eager ``stokes_kernels`` (the inner jit) on a table that no other
    call of this test uses. A new concrete radius is a new static structure and a new
    trace by design, so the third profile keeps the structure (``replace``)."""
    cfg = sj.StokesConfig(lmax=3)
    p_np = profile_from_table(cases.t_layers(reduced=True))
    p_jx = jax.tree_util.tree_map(jnp.asarray, p_np)
    p_new = p_np.replace(eta=np.asarray(p_np.eta) * 1.7, rho0=np.asarray(p_np.rho0) * 1.01)
    f = jax.jit(lambda p: sj.stokes_kernels(p, cfg).geoid)
    before = propagate.trace_count()
    for p in (p_np, p_jx, p_new):
        jax.block_until_ready(f(p))
    assert propagate.trace_count() - before == 1
    assert f._cache_size() == 1
    # Eager calls through the inner jit, on a table of its own.
    q_np = profile_from_table(cases.t_uniform(reduced=True))
    q_jx = jax.tree_util.tree_map(jnp.asarray, q_np)
    q_new = q_np.replace(eta=np.asarray(q_np.eta) * 0.3)
    cfg2 = sj.StokesConfig(lmax=4)
    before = propagate.trace_count()
    for q in (q_np, q_jx, q_new):
        jax.block_until_ready(sj.stokes_kernels(q, cfg2).geoid)
    assert propagate.trace_count() - before == 1


# ---------------------------------------------------------------------------
# Derivative against the oracle (slow).
# ---------------------------------------------------------------------------


def _oracle_shifted(table, direction, step):
    """The table with ``log eta`` moved by ``step * direction``, decimal strings at 90
    digits."""
    with mp.workdps(90):
        eta = tuple(mp.nstr(mp.mpf(e) * mp.exp(mp.mpf(step) * mp.mpf(repr(float(d)))), 85,
                            strip_zeros=False, min_fixed=-1, max_fixed=-1) for e, d in zip(table.eta, direction))
    return replace(table, eta=eta)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["hager4", "stiff_lid"])
def test_derivative_against_oracle(name):
    """Directional derivative of the four kernels with respect to log eta at l in
    {1 (CM), 2, 10}, forward mode (jax.jvp) and reverse mode (jax.vjp per kernel row,
    contracted with the direction), against a central difference of the oracle at 80
    digits with a step of 1e-15 (truncation error about 1e-30), max-normalised per
    kernel and degree."""
    table = cases.t_hager() if name == "hager4" else cases.t_stiff_lid()
    tol = DERIV_ORACLE_TOL if name == "hager4" else DERIV_ORACLE_TOL_CONTRAST
    degrees = (1, 2, 10)
    p0 = profile_from_table(table)
    x0 = np.log(np.asarray(p0.eta))
    direction = np.random.default_rng(41).standard_normal(x0.shape)
    cfg = sj.StokesConfig(lmax=10)
    rows = [d - 1 for d in degrees]

    def K(x):
        k = _kernels(p0.replace(eta=jnp.exp(x)), cfg)
        return jnp.stack([kk[jnp.asarray(rows)] for kk in k])          # (4, 3, N)
    fwd = np.asarray(jax.jit(lambda x: jax.jvp(K, (x,), (direction,))[1])(x0))
    _, vjp = jax.vjp(K, x0)
    vjp = jax.jit(vjp)
    N = x0.size
    rev = np.zeros_like(fwd)
    eye = np.eye(N)
    for o in range(4):
        for a in range(3):
            # Rows of the Jacobian of one kernel row by reverse mode, contracted with
            # the direction.
            ct = np.zeros((N, 4, 3, N))
            ct[np.arange(N), o, a, :] = eye
            J = np.asarray(jax.vmap(lambda c: vjp(c)[0])(jnp.asarray(ct)))
            rev[o, a] = J @ direction
    step = mp.mpf("1e-15")
    plus = ro.kernels(_oracle_shifted(table, direction, step), ro.Physics(), ro.Discretisation(), degrees,
                      dps=80, return_mp=True)
    minus = ro.kernels(_oracle_shifted(table, direction, -step), ro.Physics(), ro.Discretisation(), degrees,
                       dps=80, return_mp=True)
    worst = {"forward": 0.0, "reverse": 0.0}
    with mp.workdps(80):
        for o, name_o in enumerate(SI_OUTPUTS):
            for a in range(3):
                ref = np.array([float((x - y) / (2 * step)) for x, y in zip(plus[name_o][a], minus[name_o][a])])
                scale = np.max(np.abs(ref))
                if scale == 0:
                    assert np.max(np.abs(fwd[o, a])) == 0 and np.max(np.abs(rev[o, a])) == 0
                    continue
                for mode, arr in (("forward", fwd), ("reverse", rev)):
                    err = np.max(np.abs(arr[o, a] - ref)) / scale
                    worst[mode] = max(worst[mode], err)
                    assert err <= tol, (name, mode, name_o, degrees[a], err)
    print(f"derivative against the oracle, {name}: forward {worst['forward']:.2e}, reverse {worst['reverse']:.2e}")
