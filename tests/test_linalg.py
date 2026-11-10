"""The batched linear algebra of the package (plan 7.2, gate G9): the Pade pair, the
elimination solver, the Pade plan, the three-column Gram-Schmidt, the right triangular
solve and the node solve. Ported from tests/test_primitives_*.py of chunk 1 to the
package functions, with the minor issues of the chunk 1 reviews applied (an
independent value test of the inverse step; the oracle inverse inside a precision
context of 60 digits).

References: ``mp.expm`` and ``mp.inverse`` at 60 digits for the pair; LAPACK
(``jnp.linalg.solve``, ``np.linalg``) for the solves; the definitions for the
Gram-Schmidt factors.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from taylor import assert_jvp_vjp, assert_taylor, assess, random_like, remeasure_taylor  # noqa: E402

from stokesjx import linalg, system  # noqa: E402

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
# working precision up to a condition of about 1 / sqrt(eps)).
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

# Widths of the 129-node LEBEDEV table (22578.125 m spacing at the surface) and four
# test intervals with one pair (h = 0).
H129 = float(np.log1p(22578.125 / 6347421.875))
STEP_H = np.array([H129, 0.5 * H129, 0.0, 0.2 * H129])
STEP_RHO = np.array([0.8, 1.0, 1.2, 0.9])


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


@pytest.mark.parametrize("bound", [0.05, 0.5, 0.912, 1.0, 1.054, 1.82, 2.024, 3.964, 12.0, 40.0])
def test_pade_plan_static_conditions(bound):
    """For every order: the scaled norm is inside the accuracy threshold, both
    denominators are column diagonally dominant (``N_m(x) < 2``), and the squaring
    count is the smallest such count. The plan takes the order with the fewest
    products; the bounds of D5 give order 9 with 0, 1 and 2 squarings."""
    for order in linalg.PADE_ORDERS:
        s = linalg.pade_squarings(bound, order)
        x = bound / 2.0 ** s
        assert x <= linalg.PADE_THETA[order] and linalg.pade_numerator(order, x) < 2.0
        if s > 0:
            x1 = bound / 2.0 ** (s - 1)
            assert x1 > linalg.PADE_THETA[order] or linalg.pade_numerator(order, x1) >= 2.0
    plan = linalg.pade_plan(bound)
    assert plan.margin > 0 and plan.scaled_norm <= linalg.PADE_THETA[plan.order]
    costs = [linalg.pade_product_count(m) + 2 * linalg.pade_squarings(bound, m) for m in linalg.PADE_ORDERS]
    assert plan.products == min(costs)
    expected = {1.054: (9, 0), 2.024: (9, 1), 3.964: (9, 2)}
    if bound in expected:
        assert (plan.order, plan.squarings) == expected[bound]
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
    random columns and on columns of condition 1e8."""
    rng = np.random.default_rng(11)
    _check_gs(rng.standard_normal((7, 5, 6, 3)))
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
