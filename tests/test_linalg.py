"""The batched linear algebra of the package and the interval step that the solver
builds from it.

Values: the Pade pair (``expm_pade_pair``) against ``mp.expm`` and ``mp.inverse`` at
60 digits; the elimination solver and both node solvers against LAPACK; the
three-column Gram-Schmidt against its definition and against LAPACK QR; the right
triangular solve against LAPACK; the static conditions of the Pade plan.

Derivatives (Taylor test with both signs and both modes, the 1.01 negative control,
and the jvp/vjp dot-product test of CONVENTIONS 8): the interval step of the package
(``system.balanced_exponent``, ``expm_pade_pair`` with the order and the squaring
count of ``pade_plan``, ``system.viscosity_similarity``) with respect to the
viscosity, the interval width, ``kappa h`` and the density; the closed-form TALA pair
step (``system.anelastic_pair_terms``); the Pade pair through the derivative rule of
``solve_elimination``, against plain AD through the same elimination; the two sweeps
(``propagate.sweeps``); the LAPACK and the pivoted node solve.

The step tests run on the widest interval of the 129-node table in ``ln r`` (the one
at the CMB) at ``lmax = 100``, with the two forms of the Pade pair that the plan
offers there: order 7 with its own squaring count, and the order of ``pade_plan``.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from taylor import assert_jvp_vjp, assert_taylor, assess, random_like, remeasure_taylor  # noqa: E402

from stokesjx import linalg, propagate, system  # noqa: E402
from stokesjx.profile import NodeStructure  # noqa: E402

# Pade pair against mp.expm (step) and mp.inverse(mp.expm) (inverse step) at 60
# digits, max-normalised per interval. Cause: rounding of the float64 powers, of the
# elimination and of the squarings; chunk 1 measured at most 2.3e-15 for the step
# over the plans (CONVENTIONS 3) and 1.6e-14 for the inverse on the oracle tables
# (physics review r3, chk_steps.out).
PAIR_ORACLE_TOL = 1e-13
# E E_inv against the identity, and E_inv against the step of -B: two float64
# evaluations of exactly inverse rational functions (cause: rounding; JAX review r3
# of chunk 1, minor 1).
PAIR_IDENTITY_TOL = 1e-13
# solve_elimination against LAPACK on column diagonally dominant matrices (cause:
# rounding of two eliminations; no pivoting is needed for such matrices).
SOLVE_TOL = 1e-13
# Dot-product tests (CONVENTIONS 8).
DOT_RTOL = 1e-12
# Gram-Schmidt: Q R = Z relative to max |Z|, and Q^T Q = I, on random columns and on
# columns of condition 1e8 (cause: rounding of CGS2, which keeps orthogonality at the
# working precision for columns of numerical full rank, Giraud et al. 2005, Theorem 2).
GS_TOL = 1e-14
# Right triangular solve against LAPACK, relative to max |X| (cause: rounding of the
# forward substitution).
RSOLVE_TOL = 1e-14
# Node solve, both methods, against LAPACK on systems of condition up to 4e4,
# max-normalised per system. Cause: the rounding of both solvers, amplified by the
# condition; measured about 2e-17 times the condition (7.5e-13 at 3.9e4, 1.5e-12 at
# 1.1e5 on systems of the same construction).
NODE_SOLVE_TOL = 1e-12
# Negative control of the Taylor tests (CONVENTIONS 8).
NEG_BIAS = 1.01
# The closed-form TALA pair step ``I + D tau Kb D^-1`` against the matrix exponential
# ``expm(t D Kb D^-1)``, relative to the largest entry: two float64 evaluations of the
# same function (cause: rounding of the Pade approximant inside ``expm``).
PAIR_TOL = 1e-14
# The Pade pair with elimination without pivoting against the same pair with the
# LAPACK solve of the denominators, max-normalised per matrix: the same approximant,
# two float64 solvers of column diagonally dominant 6x6 systems (cause: rounding of the
# two eliminations; measured 1.3e-16, NOTES/chunks/c1-foundations/impl-review-jax-r2-evidence).
PAIR_LAPACK_TOL = 1e-13
# LAPACK QR against the Gram-Schmidt factors on random, well-conditioned columns: the
# projector ``Q Q^T`` (independent of the signs of the columns) and the log of the
# diagonal of ``R`` (cause: rounding of two orthogonalisations).
GS_QR_PROJ_TOL = 1e-14
GS_QR_LOGDIAG_TOL = 1e-13

# Widths of the 129-node LEBEDEV table (22578.125 m spacing at the surface) and four
# test intervals with one pair (h = 0).
H129 = float(np.log1p(22578.125 / 6347421.875))
STEP_H = np.array([H129, 0.5 * H129, 0.0, 0.2 * H129])
STEP_RHO = np.array([0.8, 1.0, 1.2, 0.9])

# The widest interval of the 129-node table in ln r: the one at the CMB,
# ln((r_c + (R - r_c) / 128) / r_c) with r_c = 3480 km and R = 6370 km (0.006467).
# The step tests use it because it gives the largest exponent norm of the table.
H_CMB = float(np.log1p(22578.125 / 3480000.0))
# Four intervals of the step tests: widths up to H_CMB, one zero width (a jump pair,
# whose exponent is zero and whose Pade pair is the identity exactly).
STEP_H_CMB = H_CMB * np.array([1.0, 0.5, 0.0, 0.2])
# Interval viscosities over four decades (relative to the viscosity scale), and
# kappa h = ln(rho0_i / rho0_(i-1)) of either sign (TALA).
STEP_ETA = np.array([0.03, 1.0, 10.0, 300.0])
STEP_KH = np.array([0.0, 0.05, -0.08, 0.0])
# The static norm bound of the balanced exponent at lmax = 100 in its simple form
# (3 lmax + 10 + rho_max) h_max + 4 |kappa h|_max, with rho_max = 1.3 and
# |kappa h|_max = 0.2 (2.813). It is looser than balanced_norm_bound_intervals
# (2.013 at the same inputs) and gives the plan form one squaring, so the tests below
# also differentiate through a squaring.
BOUND_100 = (3 * 100 + 10 + 1.3) * H_CMB + 4 * 0.2
PLAN_100 = linalg.pade_plan(BOUND_100)
# The two forms (order, squarings) of the Pade pair under test: order 7 with the
# squaring count of the static conditions at BOUND_100, and the plan of pade_plan.
FORMS = {"pade7": (7, linalg.pade_squarings(BOUND_100, 7)),
         "plan": (PLAN_100.order, PLAN_100.squarings)}
# The forms must keep their squarings: the derivative through the squarings of the
# order-9 form is what the "plan" tests check. A change of the constants fails here.
assert FORMS == {"pade7": (7, 2), "plan": (9, 1)}, FORMS


def _exponent(l, h=STEP_H, rho=STEP_RHO):
    """Balanced exponents of the four test intervals at degree ``l``, batch-last
    ``(6, 6, 4, 1)``."""
    return system.balanced_exponent((l,), jnp.asarray(rho), jnp.asarray(h), None, self_gravity=True)


@pytest.mark.parametrize("order", linalg.PADE_ORDERS)
@pytest.mark.parametrize("l", [1, 2, 100, 200])
def test_pade_pair_against_mpmath(l, order):
    """Both members of ``expm_pade_pair`` against ``mp.expm(B)`` and
    ``mp.inverse(mp.expm(B))`` at 60 digits, for every order that ``pade_plan`` offers,
    with the squaring count of that order at the bound of the intervals. The pair
    interval (``h = 0``) gives the identity exactly."""
    B = _exponent(l)
    bound = linalg.balanced_norm_bound_intervals(l, STEP_H, 0.0, STEP_RHO)
    s = linalg.pade_squarings(bound, order)
    E, Ei = (np.asarray(a)[..., 0] for a in linalg.expm_pade_pair(B, order, s))
    Bn = np.asarray(B)[..., 0]
    for i in range(len(STEP_H)):
        with mp.workdps(60):
            M = mp.matrix(Bn[:, :, i].tolist())
            ref = mp.expm(M)
            ref_inv = mp.inverse(ref)
            ref = np.array(ref.tolist(), dtype=float)
            ref_inv = np.array(ref_inv.tolist(), dtype=float)
        for mine, r in ((E[:, :, i], ref), (Ei[:, :, i], ref_inv)):
            assert np.max(np.abs(mine - r)) / np.max(np.abs(r)) <= PAIR_ORACLE_TOL, (i, order, s)
        if STEP_H[i] == 0:
            assert np.array_equal(E[:, :, i], np.eye(6)) and np.array_equal(Ei[:, :, i], np.eye(6))


@pytest.mark.parametrize("l", [1, 2, 100, 200])
def test_pade_pair_identity(l):
    """``E E_inv = I``, and ``E_inv`` equals the step of ``-B`` (an independent value
    test of the inverse output, JAX review r3 of chunk 1, minor 1)."""
    B = _exponent(l)
    plan = linalg.pade_plan(linalg.balanced_norm_bound_intervals(l, STEP_H, 0.0, STEP_RHO))
    E, Ei = linalg.expm_pade_pair(B, plan.order, plan.squarings)
    prod = np.asarray(linalg.matmul_batch_last(E, Ei))
    assert np.max(np.abs(prod - np.eye(6)[:, :, None, None])) <= PAIR_IDENTITY_TOL
    Em = np.asarray(linalg.expm_pade_pair(-B, plan.order, plan.squarings)[0])
    assert np.max(np.abs(np.asarray(Ei) - Em)) / np.max(np.abs(Em)) <= PAIR_IDENTITY_TOL


def _dominant_systems(seed, batch=(3, 5)):
    """Random column diagonally dominant 6x6 matrices (off-diagonal column sums at
    most 0.8 of the diagonal), batch-last ``(6, 6, *batch)``, and right-hand sides
    ``(6, 4, *batch)``."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((6, 6) + batch)
    off = np.abs(A).sum(axis=0) - np.abs(A[np.arange(6), np.arange(6)])
    sign = np.sign(A[np.arange(6), np.arange(6)])
    A[np.arange(6), np.arange(6)] = sign * off / 0.8
    return A, rng.standard_normal((6, 4) + batch)


def test_solve_elimination_value_and_derivative():
    """Value against LAPACK; Taylor test of the derivative rule (both signs, both
    modes) with the 1.01 control; dot product of the rule against plain AD and on its
    own (CONVENTIONS 6)."""
    A, B = _dominant_systems(5)
    X = np.asarray(linalg.solve_elimination(jnp.asarray(A), jnp.asarray(B)))
    ref = np.linalg.solve(np.moveaxis(A, (0, 1), (-2, -1)), np.moveaxis(B, (0, 1), (-2, -1)))
    ref = np.moveaxis(ref, (-2, -1), (0, 1))
    assert float(np.max(np.abs(X - ref)) / np.max(np.abs(ref))) <= SOLVE_TOL
    W = jnp.asarray(np.random.default_rng(6).standard_normal(B.shape))

    def F(x):
        return linalg.solve_elimination(x["A"], x["B"])

    def F_plain(x):
        return linalg._solve_elimination_plain(x["A"], x["B"])
    x = {"A": A, "B": B}
    # Direction of size 0.1 per entry: the solve is rational in A, and a unit
    # direction moves the matrices far from the point at the first rungs.
    dx = jax.tree_util.tree_map(lambda v: 0.1 * v, random_like(x, 7))
    results = assert_taylor(lambda x: jnp.sum(F(x) * W), x, dx)
    for res in results:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative))[0] == "fail", res.table()
    assert_jvp_vjp(F, x, rtol=DOT_RTOL, forward=F_plain)
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)


@pytest.mark.parametrize("bound", [0.05, 0.5, 0.912, 1.0, 1.054, 1.82, 1.863, 2.024, 2.813, 3.964, 4.753,
                                   12.0, 40.0])
def test_pade_plan_static_conditions(bound):
    """For every order: the scaled norm is inside the accuracy threshold, both
    denominators are column diagonally dominant (``N_m(x) < 2``, a positive margin),
    and the squaring count is the smallest such count. The plan takes the order with
    the fewest products. The bounds 1.054, 2.024 and 3.964 give order 9 with 0, 1 and
    2 squarings. The bounds 1.82 and 1.863 give order 7 with one squaring: 6 products,
    against 7 for order 9 with one squaring and 8 for order 6 with two squarings. These
    are the only bounds of the list where order 7 wins over order 9. The bounds 2.813
    and 4.753 are the bound ``BOUND_100`` of the step tests and a bound with two
    squarings of order 9."""
    for order in linalg.PADE_ORDERS:
        s = linalg.pade_squarings(bound, order)
        x = bound / 2.0 ** s
        assert x <= linalg.PADE_THETA[order] and linalg.pade_numerator(order, x) < 2.0
        assert linalg.pade_denominator_margin(order, x) > 0.0
        if s > 0:
            x1 = bound / 2.0 ** (s - 1)
            assert x1 > linalg.PADE_THETA[order] or linalg.pade_numerator(order, x1) >= 2.0
    plan = linalg.pade_plan(bound)
    assert plan.margin > 0 and plan.scaled_norm <= linalg.PADE_THETA[plan.order]
    costs = [linalg.pade_product_count(m) + 2 * linalg.pade_squarings(bound, m) for m in linalg.PADE_ORDERS]
    assert plan.products == min(costs)
    # (order, squarings, products); products is compared where it is given.
    expected = {1.054: (9, 0, None), 1.82: (7, 1, 6), 1.863: (7, 1, 6), 2.024: (9, 1, None),
                3.964: (9, 2, None)}
    if bound in expected:
        order, squarings, products = expected[bound]
        assert (plan.order, plan.squarings) == (order, squarings)
        if products is not None:
            assert plan.products == products
    with pytest.raises(ValueError):
        linalg.pade_product_count(13)


def _check_gs(Z):
    Q, R = (np.asarray(a) for a in linalg.gram_schmidt3(jnp.asarray(Z)))
    assert np.max(np.abs(Q @ R - Z)) / np.max(np.abs(Z)) <= GS_TOL
    QtQ = np.swapaxes(Q, -1, -2) @ Q
    assert np.max(np.abs(QtQ - np.eye(3))) <= GS_TOL
    assert np.all(np.tril(R, -1) == 0) and np.all(np.diagonal(R, axis1=-2, axis2=-1) > 0)


def test_gram_schmidt3_random_and_nearly_dependent():
    """``Q R = Z``, ``Q^T Q = I``, ``R`` upper triangular with a positive diagonal, on
    random columns and on columns of condition 1e8. On the random columns the
    factorisation also equals LAPACK QR (``jnp.linalg.qr``): the same projector
    ``Q Q^T`` and the same ``log |diag R|``. The columns of condition 1e8 are not
    compared with QR: there two factorisations differ by about the condition times the
    rounding (1e8 x 1e-16), so a comparison at 1e-14 is not valid, and ``_check_gs``
    checks them against the definition."""
    rng = np.random.default_rng(11)
    Zr = rng.standard_normal((7, 5, 6, 3))
    _check_gs(Zr)
    Q, R = (np.asarray(a) for a in linalg.gram_schmidt3(jnp.asarray(Zr)))
    Qq, Rq = (np.asarray(a) for a in jnp.linalg.qr(jnp.asarray(Zr)))
    proj = lambda A: A @ np.swapaxes(A, -1, -2)
    assert np.max(np.abs(proj(Q) - proj(Qq))) <= GS_QR_PROJ_TOL
    logdiag = lambda A: np.log(np.abs(np.diagonal(A, axis1=-2, axis2=-1)))
    assert np.max(np.abs(logdiag(R) - logdiag(Rq))) <= GS_QR_LOGDIAG_TOL
    U = np.linalg.qr(rng.standard_normal((20, 6, 3)))[0]
    V = np.linalg.qr(rng.standard_normal((20, 3, 3)))[0]
    Z = U @ (np.diag([1.0, 1e-4, 1e-8]) @ V)
    assert np.min(np.linalg.cond(Z)) > 1e7
    _check_gs(Z)


def test_right_solve_upper3():
    rng = np.random.default_rng(12)
    _, R = linalg.gram_schmidt3(jnp.asarray(rng.standard_normal((8, 6, 3))))
    W = rng.standard_normal((8, 4, 3))
    X = np.asarray(linalg.right_solve_upper3(jnp.asarray(W), R))
    R = np.asarray(R)
    ref = np.stack([np.linalg.solve(R[k].T, W[k].T).T for k in range(8)])
    assert np.max(np.abs(X - ref)) / np.max(np.abs(ref)) <= RSOLVE_TOL


def _node_systems(seed, shape=(4, 7), smallest_tilt=1e-3):
    """Systems ``[-Q_up, Q_dn]`` with orthonormal 6x3 blocks, as in the solver, made
    worse conditioned by tilting ``Q_dn`` towards ``Q_up`` (tilts from
    ``smallest_tilt`` to 1; 1e-3 gives conditions from 5 to 4e4)."""
    rng = np.random.default_rng(seed)
    Qu = np.linalg.qr(rng.standard_normal(shape + (6, 3)))[0]
    tilt = np.logspace(np.log10(smallest_tilt), 0, int(np.prod(shape))).reshape(shape)[..., None, None]
    Qd = np.linalg.qr(Qu + tilt * rng.standard_normal(shape + (6, 3)))[0]
    M = np.concatenate([-Qu, Qd], axis=-1)
    return M, rng.standard_normal(shape + (6,))


@pytest.mark.parametrize("method", linalg.NODE_SOLVERS)
def test_solve_nodes_value(method):
    M, b = _node_systems(13)
    cond = np.linalg.cond(M)
    assert np.max(cond) > 1e4 and np.max(cond) < 1e5
    x = np.asarray(linalg.solve_nodes(jnp.asarray(M), jnp.asarray(b), method))
    ref = np.linalg.solve(M, b[..., None])[..., 0]
    err = np.max(np.abs(x - ref), axis=-1) / np.max(np.abs(ref), axis=-1)
    assert np.max(err) <= NODE_SOLVE_TOL, np.max(err)


def test_solve_nodes_pivot_derivative():
    """The derivative rule of the pivoted elimination: Taylor test with the 1.01
    control, dot product against plain AD and on its own (CONVENTIONS 6). The pivot
    order is piecewise constant in the matrix and does not change along the small
    perturbations of the ladder."""
    # Well-conditioned systems (condition below 200): the Taylor ladder moves every
    # entry by up to 1e-4, which must not come near a singular system.
    M, b = _node_systems(14, shape=(3, 2), smallest_tilt=1.0)
    assert np.max(np.linalg.cond(M)) < 200
    W = jnp.asarray(np.random.default_rng(15).standard_normal(b.shape))

    def F(x):
        return linalg.solve_nodes(x["M"], x["b"], "pivot_elimination")

    def F_plain(x):
        A = jnp.moveaxis(x["M"], (-2, -1), (0, 1))
        B = jnp.moveaxis(x["b"][..., None], (-2, -1), (0, 1))
        return jnp.moveaxis(linalg._pivot_elimination_plain(A, B), (0, 1), (-2, -1))[..., 0]
    x = {"M": M, "b": b}
    dx = jax.tree_util.tree_map(lambda v: 0.01 * v, random_like(x, 16))
    results = assert_taylor(lambda x: jnp.sum(F(x) * W), x, dx)
    for res in results:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative))[0] == "fail", res.table()
    assert_jvp_vjp(F, x, rtol=DOT_RTOL, forward=F_plain)
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)


# --------------------------------------------------------------------------
# The interval step of the package and its derivatives.
# --------------------------------------------------------------------------


def _package_step(l, e_rel, rho_rel, h, kappa_h, order, squarings):
    """The interval step ``D E D^-1`` of the package at degree ``l``, batch-last
    ``(6, 6, n)``: the balanced exponent (``system.balanced_exponent``), the step member
    of the Pade pair, and the viscosity similarity (``system.viscosity_similarity``).
    This is the step that ``propagate.interval_steps`` builds for one degree, without
    the move to the scan layout.

    Parameters
    ----------
    l
        Degree (static).
    e_rel, rho_rel, h, kappa_h
        ``(n,)`` each: interval viscosity over the viscosity scale, interval density over
        ``rho_s``, width in ``ln r`` and ``kappa h`` of TALA.
    order, squarings
        The form of the Pade pair (``FORMS``).
    """
    B = system.balanced_exponent((l,), rho_rel, h, kappa_h, self_gravity=True)
    E = linalg.expm_pade_pair(B, order, squarings)[0]
    return system.viscosity_similarity(E, e_rel)[..., 0]


@pytest.mark.parametrize("l", [2, 100])
@pytest.mark.parametrize("form", list(FORMS))
def test_interval_step_log_eta(form, l):
    """The functional ``sum(step(log_eta) * W)`` of the package step at four intervals
    of width ``H_CMB``, with ``log_eta`` as the control: Taylor test (both signs, both
    modes) and the jvp/vjp dot-product test.

    The viscosity enters only the diagonal similarity ``D``, so this test checks the
    derivative of the similarity and never passes through the exponential. The
    derivative through the exponential is tested by
    ``test_interval_step_through_exponent`` (``h``, ``kappa h``) and
    ``test_interval_step_is_affine_in_density`` (``rho``)."""
    order, squarings = FORMS[form]
    rng = np.random.default_rng(l)
    rho = jnp.asarray(rng.uniform(0.7, 1.25, 4))
    kh = jnp.asarray(STEP_KH)
    h = jnp.full(4, H_CMB)
    W = jnp.asarray(rng.standard_normal((6, 6, 4)))

    def F(x):
        return _package_step(l, jnp.exp(x["log_eta"]), rho, h, kh, order, squarings)

    def f(x):
        return jnp.sum(F(x) * W)
    x = {"log_eta": np.log(STEP_ETA)}
    dx = random_like(x, 7)
    assert_taylor(f, x, dx)
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)


@pytest.mark.parametrize("l", [1, 2, 100])
@pytest.mark.parametrize("form", list(FORMS))
def test_interval_step_through_exponent(form, l):
    """Derivative through the exponential: the interval width ``h`` and ``kappa h`` as
    controls of the package step, ``l`` in {1, 2, 100}, one ``h = 0`` interval.

    ``h`` is the control of a movable boundary radius (a boundary moves the widths of
    the two intervals next to it), and ``kappa h`` follows the density of the
    compressible reference. Both enter the balanced exponent ``B``, so the derivative
    passes through the Pade powers, the elimination of the denominators with its
    derivative rule, and the squarings. The direction of ``h`` is scaled by ``H_CMB``,
    the size of ``h`` itself, so that the first rung moves every width by about 1 % of
    the largest interval; the ``h = 0`` entry then moves to a small width of either
    sign, which the step accepts (``exp(B)`` is entire in ``h``). Taylor test (both
    signs, both modes), a 1.01 biased derivative that must fail for both signs, the
    jvp/vjp dot-product test, and a finite gradient."""
    order, squarings = FORMS[form]
    rng = np.random.default_rng(100 + l)
    W = jnp.asarray(rng.standard_normal((6, 6, 4)))
    eta = jnp.asarray(STEP_ETA)
    rho = jnp.asarray(STEP_RHO)

    def F(x):
        return _package_step(l, eta, rho, x["h"], x["kh"], order, squarings)

    def f(x):
        return jnp.sum(F(x) * W)
    x = {"h": STEP_H_CMB.copy(), "kh": STEP_KH.copy()}
    dx = random_like(x, 7)
    dx["h"] = H_CMB * dx["h"]
    results = assert_taylor(f, x, dx)
    # Negative control on the same forward values: a 1 % error of the derivative
    # through the exponential must be detected for both signs (results[:2] are the two
    # signs of the first mode).
    for res in results[:2]:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative))[0] == "fail", res.table()
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)
    g = jax.grad(f)(x)
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in g.values())


@pytest.mark.parametrize("l", [1, 2, 100])
@pytest.mark.parametrize("form", list(FORMS))
def test_interval_step_is_affine_in_density(form, l):
    """The package step is exactly affine in the density ``rho``.

    In the order (u_r, u_t, sigma_rr, sigma_rt, phi, dphi) the exponent is block upper
    triangular: the 4x4 Stokes block, the 2x2 potential block, and the coupling block
    with ``-rho`` at (3, 6) and (4, 5) (eq. 4.1); K has entries in column 1 of the
    Stokes block only, and the balancing ``S_l`` and the similarity ``D`` are diagonal.
    Every power of a block upper triangular matrix with the coupling linear in ``rho``
    has its coupling block linear in ``rho`` and diagonal blocks free of ``rho``; so has
    the inverse of such a matrix and the product of two of them. The Pade approximant
    and its squarings are therefore affine in ``rho`` in exact arithmetic. A
    smooth-class Taylor test would be "unresolved" (no second-order remainder above
    rounding), so the class is "affine": the corrected remainder must stay at rounding
    level on every rung. Negative control: a 1.01 biased derivative must fail."""
    order, squarings = FORMS[form]
    W = jnp.asarray(np.random.default_rng(200 + l).standard_normal((6, 6, 4)))
    eta, h, kh = jnp.asarray(STEP_ETA), jnp.asarray(STEP_H_CMB), jnp.asarray(STEP_KH)

    def F(x):
        return _package_step(l, eta, x["rho"], h, kh, order, squarings)

    def f(x):
        return jnp.sum(F(x) * W)
    x = {"rho": STEP_RHO.copy()}
    results = assert_taylor(f, x, random_like(x, 2), classification="affine")
    for res in results[:2]:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative), classification="affine")[0] == "fail"
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)


def test_anelastic_pair_step_derivative():
    """The TALA step of a jump pair, ``I + D tau Kb D^-1`` (eq. 12.3, from
    ``system.anelastic_pair_terms`` and the viscosity similarity), at degree 2 on a
    structure with three pairs.

    The step equals ``expm(t D Kb D^-1)`` with ``t = ln(rho0_+ / rho0_-)`` (``Kb^2 =
    -Kb``, so ``expm(t Kb) = I + (1 - exp(-t)) Kb``, and the similarity commutes with the
    exponential). The control is ``t`` of each pair, through the density of the upper
    member ``rho0_+ = rho0_- exp(t)``, for ``t`` in {-0.2, 0.05, 0.3} and the pair
    viscosities {0.03, 1, 50}. Checks: the value against ``jax.scipy.linalg.expm``
    (``PAIR_TOL``), the Taylor test (both signs, both modes) and the jvp/vjp dot-product
    test."""
    r = np.array([3.48e6, 4.0e6, 4.0e6, 4.5e6, 4.5e6, 5.0e6, 5.0e6, 5.5e6])
    structure = NodeStructure.from_radius(r, consistent_core=True)
    assert structure.pairs == (2, 4, 6)
    pair_iv = np.array([1, 3, 5])                      # interval index of each pair
    rho_minus = np.array([4500.0, 4800.0, 5100.0])     # kg/m^3, lower member of each pair
    rho_base = np.array([5000.0, 4500.0, 4500.0, 4800.0, 4800.0, 5100.0, 5100.0, 4000.0])
    e_rel = np.ones(7)
    e_rel[pair_iv] = [0.03, 1.0, 50.0]
    _, _, Kb = system._static_exponent_parts((2,), True)
    Kb = jnp.asarray(Kb[:, :, 0])

    def F(x):
        rho0 = jnp.asarray(rho_base).at[pair_iv + 1].set(rho_minus * jnp.exp(x["t"]))
        dE = system.anelastic_pair_terms((2,), rho0, structure)[0]
        step = jnp.eye(6)[:, :, None] + system.viscosity_similarity(dE, e_rel)[..., 0]
        return step[:, :, pair_iv]                     # (6, 6, 3)

    x = {"t": np.array([-0.2, 0.05, 0.3])}
    step = np.asarray(F(x))
    for k, (t, e) in enumerate(zip(x["t"], e_rel[pair_iv])):
        D = jnp.diag(jnp.array([1.0, 1.0, e, e, e, e]))
        ref = np.asarray(jax.scipy.linalg.expm(t * D @ Kb @ jnp.linalg.inv(D)))
        assert np.max(np.abs(step[:, :, k] - ref)) <= PAIR_TOL * np.max(np.abs(ref)), (t, e)
    W = jnp.asarray(np.random.default_rng(4).standard_normal((6, 6, 3)))
    assert_taylor(lambda x: jnp.sum(F(x) * W), x, random_like(x, 9))
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)


def _lapack_batch_last(A, B):
    """``jnp.linalg.solve`` on batch-last arrays: ``A`` of shape ``(6, 6, *batch)`` and
    ``B`` of shape ``(6, k, *batch)``. The reference solver of the Pade denominators in
    ``test_pade_pair_through_elimination`` (LU with partial pivoting)."""
    X = jnp.linalg.solve(jnp.moveaxis(A, (0, 1), (-2, -1)), jnp.moveaxis(B, (0, 1), (-2, -1)))
    return jnp.moveaxis(X, (-2, -1), (0, 1))


@pytest.mark.parametrize("output", ["step", "inverse"])
@pytest.mark.parametrize("l", [1, 2, 100])
@pytest.mark.parametrize("form", list(FORMS))
def test_pade_pair_through_elimination(form, l, output, monkeypatch):
    """The step ``r(X)^(2^s)`` and the inverse step ``r(-X)^(2^s)`` of
    ``expm_pade_pair`` with the derivative rule of ``solve_elimination``
    (``dX = A^-1 (dB - dA X)``), each as its own output.

    Controls ``h`` and ``kappa h`` (both enter the exponent), four intervals with one
    ``h = 0``, ``l`` in {1, 2, 100}. Checks: the Taylor test with both signs and both
    modes; a 1.01 biased derivative that must fail for both signs; the jvp/vjp
    dot-product test of the rule against the forward mode of plain AD through the same
    elimination (CONVENTIONS 6); the dot-product test of the rule alone; and the values
    against the pair with the LAPACK solve of the denominators.

    The two references replace ``linalg.solve_elimination`` inside their own body only
    (``monkeypatch.context``). ``assert_jvp_vjp`` traces each function when it first
    calls it, so the replacement is active while ``F_plain`` and ``F_lapack`` are traced
    and not active while ``F`` is traced. A replacement for the whole test would make
    ``F`` and ``F_plain`` the same program, and the dot product of the rule against plain
    AD would then pass without a test of the rule. Two controls check this: after each
    reference call ``linalg.solve_elimination`` is the ``custom_jvp`` object again, and
    the jaxpr of the jvp of ``F`` holds a ``custom_jvp_call`` that the jaxpr of the jvp
    of ``F_plain`` does not hold."""
    rule = linalg.solve_elimination            # the custom_jvp object of the package
    order, squarings = FORMS[form]
    idx = {"step": 0, "inverse": 1}[output]
    rng = np.random.default_rng(300 + l)
    W = jnp.asarray(rng.standard_normal((6, 6, 4)))
    rho = jnp.asarray(STEP_RHO)

    def pair(x):
        B = system.balanced_exponent((l,), rho, x["h"], x["kh"], self_gravity=True)
        return linalg.expm_pade_pair(B, order, squarings)[idx][..., 0]

    def F(x):
        return pair(x)

    def F_plain(x):
        with monkeypatch.context() as m:
            m.setattr(linalg, "solve_elimination", linalg._solve_elimination_plain)
            return pair(x)

    def F_lapack(x):
        with monkeypatch.context() as m:
            m.setattr(linalg, "solve_elimination", _lapack_batch_last)
            return pair(x)

    def f(x):
        return jnp.sum(F(x) * W)
    x = {"h": STEP_H_CMB.copy(), "kh": STEP_KH.copy()}
    dx = random_like(x, 17)
    dx["h"] = H_CMB * dx["h"]
    results = assert_taylor(f, x, dx)
    for res in results[:2]:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative))[0] == "fail", res.table()
    assert_jvp_vjp(F, x, rtol=DOT_RTOL, forward=F_plain)
    assert linalg.solve_elimination is rule
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)
    E = np.asarray(F(x))
    E_lapack = np.asarray(F_lapack(x))
    assert linalg.solve_elimination is rule
    err = np.max(np.abs(E - E_lapack), axis=(0, 1)) / np.max(np.abs(E_lapack), axis=(0, 1))
    assert float(np.max(err)) <= PAIR_LAPACK_TOL, err
    # F and F_plain are different programs: only the rule leaves a custom_jvp_call.
    jvp_text = lambda G: str(jax.make_jaxpr(lambda x: jax.jvp(G, (x,), (dx,)))(x))
    assert "custom_jvp_call" in jvp_text(F)
    assert "custom_jvp_call" not in jvp_text(F_plain)
    assert linalg.solve_elimination is rule


def test_sweeps_derivative():
    """The two sweeps of the solver (``propagate.sweeps``: ``lax.scan`` over 32 steps
    with ``gram_schmidt3`` and ``right_solve_upper3`` after every step) at 4 degrees.
    Control: the steps of both sweeps, two independent draws ``I + 0.25 N(0, 1)``, so
    the derivative of every step of both sweeps is tested.

    Functional: ``sum((Q Q^T) * (Wq Wq^T))`` over the bases of both sweeps (the
    projector does not depend on the sign of a column), plus ``sum(sin(rows))`` over the
    carried output rows of both sweeps. Taylor test (both signs, both modes) and the
    jvp/vjp dot-product test on the map that returns the projectors and the rows."""
    rng = np.random.default_rng(11)
    Z1 = np.eye(6) + 0.25 * rng.standard_normal((32, 4, 6, 6))
    Z2 = np.eye(6) + 0.25 * rng.standard_normal((32, 4, 6, 6))
    start_up = jnp.asarray(rng.standard_normal((4, 6, 3)))
    start_down = jnp.asarray(rng.standard_normal((4, 6, 3)))
    row_up = jnp.asarray(rng.standard_normal((4, 1, 6)))
    rows_down = jnp.asarray(rng.standard_normal((4, 4, 6)))
    Wq = jnp.asarray(rng.standard_normal((33, 4, 6, 3)))
    proj = lambda Q: jnp.einsum("nlik,nljk->nlij", Q, Q)

    def F(x):
        res = propagate.sweeps(x["Z_up"], x["Z_down"], start_up, start_down, row_up, rows_down)
        return proj(res.q_up), proj(res.q_down), res.row_up, res.rows_down

    def f(x):
        p_up, p_down, r_up, r_down = F(x)
        return (jnp.sum(p_up * proj(Wq)) + jnp.sum(p_down * proj(Wq))
                + jnp.sum(jnp.sin(r_up)) + jnp.sum(jnp.sin(r_down)))
    x = {"Z_up": Z1, "Z_down": Z2}
    # A direction of unit size per entry moves 4608 entries of each sweep at once, and
    # the 32-step product is strongly nonlinear in them: the third-order term then
    # lowers the corrected rate at the first rungs. A direction of norm 0.05 per leaf
    # puts the default window in the asymptotic range (TOLERANCES.md, 2026-11-11).
    dx = {k: 0.05 * v / np.sqrt(v.size) for k, v in random_like(x, 12).items()}
    assert_taylor(f, x, dx)
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)


def test_solve_nodes_lapack_derivative():
    """The derivative of the default node solve (``solve_nodes(..., "lapack")``, that
    is ``jnp.linalg.solve``) on ``(4, 33, 6, 6)`` systems ``N(0, 1) + 4 I`` (batch-first,
    as the solver calls it). Control: the matrices. Taylor test (both signs, both
    modes) and the jvp/vjp dot-product test."""
    rng = np.random.default_rng(5)
    M0 = rng.standard_normal((4, 33, 6, 6)) + 4 * np.eye(6)
    b = jnp.asarray(rng.standard_normal((4, 33, 6)))
    W = jnp.asarray(rng.standard_normal((4, 33, 6)))

    def F(x):
        return linalg.solve_nodes(x["M"], b, "lapack")
    x = {"M": M0}
    assert_taylor(lambda x: jnp.sum(F(x) * W), x, random_like(x, 6))
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)
