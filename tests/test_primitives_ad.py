"""Gate G11: the JAX primitives of the planned solver are differentiable to second order.

No package code is tested here. The candidates live in benchmarks/candidates.py and
the solver of chunk 3 will be built from them, so each one must pass the Taylor test
(both signs, both modes) and the jvp/vjp dot-product test before it is used.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import candidates as c  # noqa: E402
from taylor import (assert_jvp_vjp, assert_taylor, assess, random_like,  # noqa: E402
                    remeasure_taylor)

# Dot-product tolerance: the forward and reverse contractions of a few hundred
# float64 terms differ by rounding only (cause: summation order). The plan sets
# 1e-12 for every primitive.
DOT_RTOL = 1e-12
# Agreement of the closed-form pair step with the matrix exponential of K t: both
# are float64 evaluations of the same function (cause: rounding of expm's Pade).
PAIR_TOL = 1e-14
# Agreement of a vmapped batch with single calls of the stand-in: the same float64
# operations in a different order (cause: vectorisation of the batch axis).
# Measured 1.3e-16 on the values (NOTES/chunks/c1-foundations/
# impl-review-jax-r2-evidence).
VMAP_RTOL = 1e-12
# Agreement of the step-and-inverse pair with elimination without pivoting and the
# same pair with the LAPACK solve: the same approximant, two float64 solvers of
# diagonally dominant 6x6 systems (cause: rounding of the two eliminations;
# measured 1.3e-16 on the stand-in value, NOTES/chunks/c1-foundations/
# impl-review-jax-r2-evidence). Max-normalised per matrix.
PAIR_LAPACK_TOL = 1e-13
# Agreement of solve_elimination with jnp.linalg.solve on column diagonally
# dominant matrices (cause: rounding; no pivoting is needed for such matrices).
SOLVE_TOL = 1e-13

R, RC = 6370000.0, 3480000.0
H129 = math.log((RC + (R - RC) / 128) / RC)
# Static squaring counts of the eligible candidates at lmax = 100 and the largest
# interval of the 129-node table (benchmarks/bench_primitives.py records the same).
BOUND_100 = c.balanced_norm_bound(100, H129)
S_TAYLOR = c.taylor_squarings(BOUND_100)
S_PADE6 = c.pade_squarings(BOUND_100, 6, "lapack")
# The selected form: elimination without pivoting with its derivative rule. Order
# 7 with the count of the dominance condition, and the order of pade_plan at this
# bound (order 9 with one squaring, denominator margin 0.009: the case closest to
# the limit N_m(x) = 2 that the plan accepts).
S_PADE7_GE = c.pade_squarings(BOUND_100, 7, "elimination")
PLAN_100 = c.pade_plan(BOUND_100, "elimination")

# Every eligible exponential of the balanced exponent (bench_primitives.py part 1).
# "pade6" and "pade6_bl" are the same approximant with the matrix products on
# batch-first and on batch-last arrays; both are tested because their derivative
# code differs. "pade7_bl_ge" and "plan_bl_ge" solve the denominators by
# elimination without pivoting (solve_elimination, custom_jvp rule).
STEP_METHODS = {
    "taylor12": dict(method="taylor", squarings=S_TAYLOR),
    "pade6": dict(method="pade6", squarings=S_PADE6),
    "pade6_bl": dict(method="pade6_bl", squarings=S_PADE6),
    "pade13": dict(method="pade13", squarings=c.pade_squarings(BOUND_100, 13, "lapack")),
    "pade7_bl_ge": dict(method="pade7_bl_ge", squarings=S_PADE7_GE),
    "plan_bl_ge": dict(method=f"pade{PLAN_100['order']}_bl_ge", squarings=PLAN_100["squarings"]),
    "jax_expm": dict(method="jax"),
}

# Four intervals for the step tests: widths in ln r up to the largest interval of
# the 129-node table, one zero-width interval (the pair of a jump, CONVENTIONS 4),
# viscosities over four decades, densities of the mantle range relative to
# rho_s = 4500 kg/m^3, and kappa h = ln(rho0_i / rho0_(i-1)) of either sign.
STEP_H = np.array([H129, 0.5 * H129, 0.0, 0.2 * H129])
STEP_ETA = np.array([0.03, 1.0, 10.0, 300.0])
STEP_RHO = np.array([0.8, 1.0, 1.2, 0.9])
STEP_KH = np.array([0.0, 0.05, -0.08, 0.0])
# Bias of every negative control (CONVENTIONS 8).
NEG_BIAS = 1.01


@pytest.mark.parametrize("l", [2, 100])
@pytest.mark.parametrize("name", list(STEP_METHODS))
def test_interval_step_taylor_and_dot_product(l, name):
    """The functional ``sum(step(log_eta) * W)`` of the balanced interval step at
    four intervals, with ``log_eta`` as the control.

    The viscosity enters only the diagonal similarity ``T = D_eta S_l^-1``, so this
    test checks the derivative of the similarity and never passes through the
    exponential of ``B``. The derivative through the exponential is tested by
    ``test_interval_step_through_exponent`` (``h``, ``kappa h``) and
    ``test_interval_step_is_affine_in_density`` (``rho``)."""
    kw = STEP_METHODS[name]
    rng = np.random.default_rng(l)
    rho = jnp.asarray(rng.uniform(0.7, 1.25, 4))
    kh = jnp.asarray([0.0, 0.05, -0.08, 0.0])
    h = jnp.full(4, H129)
    W = jnp.asarray(rng.standard_normal((4, 6, 6)))

    def F(x):
        return c.step_balanced(float(l), jnp.exp(x["log_eta"]), rho, h, kh, **kw)

    def f(x):
        return jnp.sum(F(x) * W)
    x = {"log_eta": np.log(np.array([0.03, 1.0, 10.0, 300.0]))}
    dx = random_like(x, 7)
    assert_taylor(f, x, dx)
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)


@pytest.mark.parametrize("l", [1, 2, 100])
@pytest.mark.parametrize("name", list(STEP_METHODS))
def test_interval_step_through_exponent(l, name):
    """Derivative through the exponential: the interval width ``h`` and ``kappa h``
    as controls, every eligible method, ``l`` in {1, 2, 100}, one ``h = 0`` interval.

    ``h`` is the control of a movable boundary radius (a boundary moves the widths
    of the two intervals next to it), and ``kappa h`` follows the density of the
    compressible reference. Both enter the balanced exponent ``B``, so the
    derivative passes through the series or the Pade powers, the solve of the
    denominator (LAPACK, or elimination with its derivative rule) and the
    squarings. The direction of ``h`` is scaled by ``H129``, the size of ``h``
    itself, so that the first rung moves every width by about 1 % of the largest
    interval; the ``h = 0`` entry then moves to a small width of either sign, which
    the step accepts (``exp(B)`` is entire in ``h``). Taylor test (both signs, both
    modes), a 1.01 biased derivative that must fail for both signs, the jvp/vjp
    dot-product test, and a finite gradient."""
    kw = STEP_METHODS[name]
    rng = np.random.default_rng(100 + l)
    W = jnp.asarray(rng.standard_normal((4, 6, 6)))
    eta = jnp.asarray(STEP_ETA)
    rho = jnp.asarray(STEP_RHO)

    def F(x):
        return c.step_balanced(float(l), eta, rho, x["h"], x["kh"], **kw)

    def f(x):
        return jnp.sum(F(x) * W)
    x = {"h": STEP_H.copy(), "kh": STEP_KH.copy()}
    dx = random_like(x, 7)
    dx["h"] = H129 * dx["h"]
    results = assert_taylor(f, x, dx)
    # Negative control on the same forward values: a 1 % error of the derivative
    # through the exponential must be detected for both signs.
    for res in results[:2]:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative))[0] == "fail", res.table()
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)
    g = jax.grad(f)(x)
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in g.values())


@pytest.mark.parametrize("l", [1, 2, 100])
@pytest.mark.parametrize("name", ["taylor12", "pade6", "pade6_bl", "pade7_bl_ge"])
def test_interval_step_is_affine_in_density(l, name):
    """The step is exactly affine in the density ``rho``.

    In the order (u_r, u_t, sigma_rr, sigma_rt, phi, dphi) the exponent is block
    upper triangular: the 4x4 Stokes block, the 2x2 potential block, and the
    coupling block with ``-rho`` at (3, 6) and (4, 5) (eq. 4.1); K has entries in
    column 1 of the Stokes block only, and the balancing ``S_l`` is diagonal. Every
    power of a block upper triangular matrix with the coupling linear in ``rho``
    has its coupling block linear in ``rho`` and diagonal blocks free of ``rho``;
    so has the inverse of such a matrix and the product of two of them. The
    Taylor polynomial, the Pade approximant and its squarings are therefore affine
    in ``rho`` in exact arithmetic. A smooth-class Taylor test would be
    "unresolved" (no second-order remainder above rounding), so the class is
    "affine": the corrected remainder must stay at rounding level on every rung.
    Negative control: a 1.01 biased derivative must fail."""
    kw = STEP_METHODS[name]
    W = jnp.asarray(np.random.default_rng(200 + l).standard_normal((4, 6, 6)))
    eta, h, kh = jnp.asarray(STEP_ETA), jnp.asarray(STEP_H), jnp.asarray(STEP_KH)

    def F(x):
        return c.step_balanced(float(l), eta, x["rho"], h, kh, **kw)

    def f(x):
        return jnp.sum(F(x) * W)
    x = {"rho": STEP_RHO.copy()}
    results = assert_taylor(f, x, random_like(x, 2), classification="affine")
    for res in results[:2]:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative), classification="affine")[0] == "fail"
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)


def test_pair_step_equals_expm_and_is_smooth():
    """``I + K (1 - exp(-t)) = expm(K t)`` (F8: K^2 = -K), and its derivative in t."""
    eta = jnp.asarray([0.03, 1.0, 50.0])
    t = jnp.asarray([-0.2, 0.05, 0.3])
    closed = c.pair_step(eta, t)
    ref = jax.scipy.linalg.expm(c.k_matrix(eta) * t[:, None, None])
    scale = jnp.max(jnp.abs(ref), axis=(1, 2))
    assert float(jnp.max(jnp.max(jnp.abs(closed - ref), axis=(1, 2)) / scale)) <= PAIR_TOL
    W = jnp.asarray(np.random.default_rng(4).standard_normal((3, 6, 6)))

    def f(x):
        return jnp.sum(c.pair_step(eta, x["t"]) * W)
    x = {"t": np.asarray(t)}
    assert_taylor(f, x, random_like(x, 9))
    assert_jvp_vjp(lambda x: c.pair_step(eta, x["t"]), x, rtol=DOT_RTOL)


# Forms of the step-and-inverse pair under test: (order, squarings) at the bound
# of lmax = 100 and the largest interval of the 129-node table.
PAIR_FORMS = {"pade7": (7, S_PADE7_GE), "plan": (PLAN_100["order"], PLAN_100["squarings"])}


@pytest.mark.parametrize("output", ["step", "inverse"])
@pytest.mark.parametrize("l", [1, 2, 100])
@pytest.mark.parametrize("form", list(PAIR_FORMS))
def test_pade_pair_through_elimination(form, l, output):
    """The step ``r(X)^(2^s)`` and the inverse step ``r(-X)^(2^s)`` of
    ``expm_pade_pair`` with the selected solver (elimination without pivoting and
    the rule ``dX = A^-1 (dB - dA X)``), each as its own output.

    Controls ``h`` and ``kappa h`` (both enter the exponent), four intervals with
    one ``h = 0``, ``l`` in {1, 2, 100}. Checks: Taylor test with both signs and
    both modes, a 1.01 biased derivative that must fail for both signs, the
    jvp/vjp dot-product test of the rule against the forward mode of plain AD
    through the same elimination (CONVENTIONS 6), the dot-product test of the rule
    alone, and the values against the pair with the LAPACK solve."""
    order, sq = PAIR_FORMS[form]
    idx = {"step": 0, "inverse": 1}[output]
    rng = np.random.default_rng(300 + l)
    W = jnp.asarray(rng.standard_normal((4, 6, 6)))
    rho = jnp.asarray(STEP_RHO)

    def pair(x, solver):
        B = c.balanced_argument(float(l), rho, x["h"], x["kh"])
        return c.expm_pade_pair(B, order, sq, solver=solver)[idx]

    def F(x):
        return pair(x, "elimination")

    def F_plain(x):
        return pair(x, "elimination_plain")

    def f(x):
        return jnp.sum(F(x) * W)
    x = {"h": STEP_H.copy(), "kh": STEP_KH.copy()}
    dx = random_like(x, 17)
    dx["h"] = H129 * dx["h"]
    results = assert_taylor(f, x, dx)
    for res in results[:2]:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative))[0] == "fail", res.table()
    assert_jvp_vjp(F, x, rtol=DOT_RTOL, forward=F_plain)
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)
    E = np.asarray(F(x))
    E_lapack = np.asarray(pair(x, "lapack"))
    err = np.max(np.abs(E - E_lapack), axis=(1, 2)) / np.max(np.abs(E_lapack), axis=(1, 2))
    assert float(np.max(err)) <= PAIR_LAPACK_TOL, err


def _dominant_systems(seed, batch=(3, 5)):
    """Random 6x6 matrices on batch-last arrays ``(6, 6, *batch)`` that are strictly
    diagonally dominant by columns (off-diagonal column sums at most 0.8 of the
    diagonal), and right-hand sides ``(6, 4, *batch)``."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((6, 6) + batch)
    off = np.abs(A).sum(axis=0) - np.abs(A[np.arange(6), np.arange(6)])
    sign = np.sign(A[np.arange(6), np.arange(6)])
    A[np.arange(6), np.arange(6)] = sign * off / 0.8
    return A, rng.standard_normal((6, 4) + batch)


def test_solve_elimination_value_and_derivative():
    """``solve_elimination`` equals ``jnp.linalg.solve`` on column diagonally dominant
    matrices, and its derivative rule passes the Taylor test (both signs, both
    modes) and the dot-product test against plain AD, with the matrices and the
    right-hand sides as controls."""
    A, B = _dominant_systems(5)
    X = np.asarray(c.solve_elimination(jnp.asarray(A), jnp.asarray(B)))
    ref = np.asarray(c.solve_lapack_batch_last(jnp.asarray(A), jnp.asarray(B)))
    assert float(np.max(np.abs(X - ref)) / np.max(np.abs(ref))) <= SOLVE_TOL
    W = jnp.asarray(np.random.default_rng(6).standard_normal(B.shape))

    def F(x):
        return c.solve_elimination(x["A"], x["B"])

    def F_plain(x):
        return c.solve_elimination_plain(x["A"], x["B"])
    x = {"A": A, "B": B}
    # Direction of norm 0.1 per entry: the solve is rational in A, and a unit
    # direction moves the dominant matrices far from the point at the first rungs.
    dx = jax.tree_util.tree_map(lambda v: 0.1 * v, random_like(x, 7))
    results = assert_taylor(lambda x: jnp.sum(F(x) * W), x, dx)
    for res in results[:2]:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative))[0] == "fail", res.table()
    assert_jvp_vjp(F, x, rtol=DOT_RTOL, forward=F_plain)
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)


@pytest.mark.parametrize("bound", [0.05, 0.5, 0.912, 1.0, 1.82, 1.863, 2.813, 4.753, 12.0, 40.0])
def test_pade_plan_static_conditions(bound):
    """``pade_plan`` and ``pade_squarings`` keep the scaled norm inside the accuracy
    threshold of the order and both denominators strictly diagonally dominant by
    columns (``N_m(x) < 2``) for the elimination solver, for every order. The
    bounds 1.863 and 1.82 of the two benchmark sizes give order 7 with one
    squaring (6 products against 8 for order 6 with two squarings)."""
    for order in (6, 7, 9, 13):
        s = c.pade_squarings(bound, order, "elimination")
        x = bound / 2.0 ** s
        assert x <= c.PADE_THETA[order]
        assert c.pade_numerator(order, x) < 2.0
        assert c.pade_denominator_margin(order, x) > 0.0
        # The count is the smallest one: one squaring fewer breaks a condition.
        if s > 0:
            x1 = bound / 2.0 ** (s - 1)
            assert x1 > c.PADE_THETA[order] or c.pade_numerator(order, x1) >= 2.0
    plan = c.pade_plan(bound, "elimination")
    assert plan["margin"] > 0.0 and plan["scaled_norm"] <= c.PADE_THETA[plan["order"]]
    if bound in (1.82, 1.863):
        assert (plan["order"], plan["squarings"], plan["products"]) == (7, 1, 6)


@pytest.mark.parametrize("ortho", ["qr", "gs"])
def test_qr_sweep(ortho):
    """``lax.scan`` over 32 propagators with a (4, 6, 3) carry, orthonormalisation
    after every step, the log of the removed scales accumulated. Control: the
    propagator entries."""
    rng = np.random.default_rng(11)
    P0 = np.eye(6) + 0.25 * rng.standard_normal((32, 4, 6, 6))
    start = jnp.asarray(rng.standard_normal((4, 6, 3)))
    Wq = jnp.asarray(rng.standard_normal((32, 4, 6, 3)))

    def F(x):
        Qs, logs = c.sweep(x["P"], start, ortho)
        # Q is defined up to the sign of each column; a functional that is
        # invariant under those signs (Q Q^T) and the log scales.
        return jnp.einsum("nlik,nljk->nlij", Qs, Qs), logs

    def f(x):
        proj, logs = F(x)
        return jnp.sum(proj * jnp.einsum("nlik,nljk->nlij", Wq, Wq)) + jnp.sum(jnp.sin(logs))
    x = {"P": P0}
    # A direction of unit size per entry moves 4608 entries at once, and the
    # 32-step product is strongly nonlinear in them: with a direction of norm one
    # the third-order term still lowers the corrected rate to 1.4 at the first
    # rungs (measured; it reaches 1.99 at 8e-5). A direction of norm 0.05 puts the
    # fixed default window in the asymptotic range.
    dx = {"P": 0.05 * random_like(x, 12)["P"] / np.sqrt(P0.size)}
    assert_taylor(f, x, dx)
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)


def test_gs_equals_qr_projector():
    """Gram-Schmidt and LAPACK QR span the same space and remove the same scales."""
    V = jnp.asarray(np.random.default_rng(3).standard_normal((5, 6, 3)))
    Qa, la = c.orthonormalise_qr(V)
    Qb, lb = c.orthonormalise_gs(V)
    np.testing.assert_allclose(Qa @ jnp.swapaxes(Qa, -1, -2), Qb @ jnp.swapaxes(Qb, -1, -2), atol=1e-14)
    np.testing.assert_allclose(la, lb, atol=1e-13)


def test_batched_solve():
    """``jnp.linalg.solve`` of (4, 33, 6, 6) systems; control: the matrices."""
    rng = np.random.default_rng(5)
    M0 = rng.standard_normal((4, 33, 6, 6)) + 4 * np.eye(6)
    b = jnp.asarray(rng.standard_normal((4, 33, 6, 1)))
    W = jnp.asarray(rng.standard_normal((4, 33, 6, 1)))

    def F(x):
        return jnp.linalg.solve(x["M"], b)
    x = {"M": M0}
    assert_taylor(lambda x: jnp.sum(F(x) * W), x, random_like(x, 6))
    assert_jvp_vjp(F, x, rtol=DOT_RTOL)


def _standin_case():
    """The budget stand-in (Pade order and squaring count of ``pade_plan``,
    denominators by elimination without pivoting with its derivative rule,
    batch-last products, Gram-Schmidt) at 5 degrees and 16 intervals, unjitted so that every input can be a control, with
    one zero-width interval. Returns ``(f, x, dx, hmax)``: the scalar functional of
    the pytree ``x = {log_eta, rho, h, kh}``, the point and a direction scaled to
    the size of each input (``h`` by 0.3 of the interval width, ``kappa h`` by 0.01,
    ``rho`` by 0.05, ``log_eta`` by 0.01), so that the first rung of the ladder is a
    small relative change of every input."""
    hmax = math.log((RC + (R - RC) / 16) / RC)
    fwd, _, args, info = c.make_standin(5, 16, hmax, jit=False)
    assert (info["method"], info["layout"], info["solver"], info["ortho"]) == ("pade", "batch_last", "elimination", "gs")
    assert info["denominator_margin"] > 0
    le, rho, h, kh = (np.asarray(a) for a in args)
    h = h.copy()
    h[7] = 0.0

    def f(x):
        return fwd(x["log_eta"], x["rho"], x["h"], x["kh"])
    x = {"log_eta": le, "rho": rho, "h": h, "kh": kh}
    dx = random_like(x, 8)
    dx["h"] = 0.3 * hmax * dx["h"]
    dx["kh"] = 0.01 * dx["kh"]
    dx["rho"] = 0.05 * dx["rho"]
    # log_eta moves the viscosity ratios of neighbouring intervals, and along this
    # direction the functional is strongly curved compared with its slope
    # (|<grad f, dx>| = 0.2, second-order coefficient about 20, measured). With an
    # unscaled direction the uncorrected rate is 1.41, 1.26, 1.15 on the window
    # (the corrected rate is 2.00). With a factor 0.1 the test passes, but the
    # 1.01 biased derivative changes the corrected remainder by less than the
    # second-order term on the window and is not detected. The factor 0.01 makes
    # the bias term 0.01 e |<grad f, dx>| larger than the second-order term on every
    # rung, so the negative control fails (corrected rate 1.41 and -3.5 for the two
    # signs) while the correct derivative gives the rates 2.00.
    dx["log_eta"] = 0.01 * dx["log_eta"]
    return f, x, dx, hmax


@pytest.mark.parametrize("control", ["log_eta", "rho", "h", "kh", "all"])
def test_standin_taylor_every_input(control):
    """The composed stand-in, budget candidate: Taylor test in each of the four
    inputs and in all of them together (both signs, both modes), with a zero-width
    interval in the table, and a 1.01 biased derivative that must fail. ``h`` and
    ``kappa h`` reach the exponential, the sweeps and the solve; ``log_eta`` reaches
    the diagonal similarity and the coordinate change between intervals."""
    f, x, dx, _ = _standin_case()
    if control != "all":
        dx = {k: (v if k == control else np.zeros_like(v)) for k, v in dx.items()}
    results = assert_taylor(f, x, dx)
    for res in results[:2]:
        assert assess(remeasure_taylor(res, NEG_BIAS * res.derivative))[0] == "fail", res.table()


def test_standin_gradient_finite_and_vmap():
    """The stand-in has a finite gradient in every input (zero-width interval
    included), and ``vmap`` over a batch of three control sets gives the values and
    gradients of the single calls. A sampler evaluates many control sets at once,
    so the forward map and its gradient must batch without a Python loop."""
    f, x, dx, _ = _standin_case()
    g = jax.jit(jax.grad(f))(x)
    for k, v in g.items():
        assert bool(jnp.all(jnp.isfinite(v))), k
    # Three control sets: the point and two steps along the scaled direction.
    xb = jax.tree_util.tree_map(lambda a, d: jnp.stack([a, a + 0.5 * d, a - 0.3 * d]), x, dx)
    vb = jax.jit(jax.vmap(f))(xb)
    gb = jax.jit(jax.vmap(jax.grad(f)))(xb)
    for k in range(3):
        xk = jax.tree_util.tree_map(lambda a: a[k], xb)
        v1 = jax.jit(f)(xk)
        g1 = jax.jit(jax.grad(f))(xk)
        # Batched and single evaluations differ by the order of float64 operations
        # only (cause: XLA vectorises the batch differently).
        assert abs(float(vb[k] - v1)) <= VMAP_RTOL * abs(float(v1))
        for name in g1:
            scale = float(jnp.max(jnp.abs(g1[name])))
            assert float(jnp.max(jnp.abs(gb[name][k] - g1[name]))) <= VMAP_RTOL * scale, name


def test_standin_dot_product():
    """jvp/vjp dot-product test of the stand-in in all four inputs (zero-width
    interval included)."""
    f, x, _, _ = _standin_case()
    assert_jvp_vjp(f, x, rtol=DOT_RTOL)
