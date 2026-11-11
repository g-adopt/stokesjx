"""Gate G10: the Taylor helper passes correct derivatives and fails wrong ones.

Every negative control here must fail. A helper that passes a 1 % biased derivative
would let a wrong adjoint through every later gate.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import stokesjx  # noqa: E402,F401
from taylor import (assert_jvp_vjp, assert_taylor, directional_derivative,  # noqa: E402
                    random_like, taylor_remainders)

# Tolerance of the dot-product test on a small analytic map: the two contractions
# differ only by the summation order of a handful of terms (cause: float64 rounding).
DOT_RTOL = 1e-13


def _smooth(x):
    """A smooth nonlinear functional of a pytree {a: (3,), b: (2, 2)}."""
    a, b = x["a"], x["b"]
    return jnp.sum(jnp.sin(a) * jnp.exp(0.3 * a)) + jnp.sum(jnp.tanh(b) ** 2) + a[0] * b[1, 0]


@pytest.fixture
def point():
    x = {"a": np.array([0.3, -0.7, 1.1]), "b": np.array([[0.2, -0.4], [0.9, 0.05]])}
    dx = random_like(x, 3)
    return x, dx


def test_smooth_pytree_passes_both_signs_both_modes(point):
    x, dx = point
    results = assert_taylor(_smooth, x, dx)
    assert len(results) == 4
    assert {r.mode for r in results} == {"reverse", "forward"}
    assert {r.sign for r in results} == {1.0, -1.0}
    # Reverse and forward give the same directional derivative.
    assert abs(results[0].derivative - results[2].derivative) <= 1e-14 * abs(results[0].derivative)


@pytest.mark.parametrize("bias", [1.01, 0.99])
def test_one_percent_biased_derivative_fails(point, bias):
    x, dx = point
    d = directional_derivative(_smooth, x, dx, "reverse")
    for sign in (1.0, -1.0):
        with pytest.raises(AssertionError, match="Taylor test did not pass"):
            assert_taylor(_smooth, x, dx, derivative=bias * d, signs=(sign,), modes=("reverse",))


def test_wrong_sign_derivative_fails(point):
    x, dx = point
    d = directional_derivative(_smooth, x, dx, "reverse")
    with pytest.raises(AssertionError):
        assert_taylor(_smooth, x, dx, derivative=-d, modes=("reverse",))


def test_affine_passes_affine_and_is_unresolved_as_smooth():
    """An affine functional has no second-order remainder: it passes the affine
    classification and is "unresolved" (not a pass) as smooth."""
    A = np.array([[1.0, -2.0, 0.5], [0.3, 0.1, -0.7]])

    def f(x):
        return jnp.sum(A @ x["v"]) + 2.0
    x = {"v": np.array([0.1, 0.2, -0.3])}
    dx = random_like(x, 5)
    assert_taylor(f, x, dx, classification="affine")
    with pytest.raises(AssertionError, match="unresolved"):
        assert_taylor(f, x, dx, classification="smooth")


def test_zero_classification():
    def f(x):
        return 0.0 * jnp.sum(x["v"])
    x = {"v": np.array([0.1, 0.2])}
    assert_taylor(f, x, random_like(x, 1), classification="zero")


def test_jvp_vjp_dot_product():
    A = jnp.asarray(np.random.default_rng(2).standard_normal((4, 3)))

    def F(x):
        return (A @ jnp.tanh(x["u"]), x["u"] ** 3)
    x = {"u": np.array([0.4, -1.2, 0.8])}
    rel = assert_jvp_vjp(F, x, rtol=DOT_RTOL)
    assert rel <= DOT_RTOL


def test_broken_custom_vjp_is_detected():
    """A custom reverse rule with a factor 1.001 is caught twice: by the dot-product
    test against the forward mode of the plain function, and by the Taylor test in
    reverse mode."""
    def plain(x):
        return jnp.sin(x) * x

    @jax.custom_vjp
    def broken(x):
        return plain(x)

    def fwd(x):
        return plain(x), x

    def bwd(x, g):
        return (1.001 * g * (jnp.cos(x) * x + jnp.sin(x)),)

    broken.defvjp(fwd, bwd)
    x = np.array([0.3, 1.2, -0.5])
    with pytest.raises(AssertionError, match="dot products differ"):
        assert_jvp_vjp(broken, x, forward=plain, rtol=1e-12)
    # The same reverse rule without the factor passes.

    @jax.custom_vjp
    def good(x):
        return plain(x)
    good.defvjp(fwd, lambda x, g: (g * (jnp.cos(x) * x + jnp.sin(x)),))
    assert_jvp_vjp(good, x, forward=plain, rtol=1e-12)
    dx = np.array([0.7, -0.2, 0.4])
    with pytest.raises(AssertionError):
        assert_taylor(lambda v: jnp.sum(broken(v)), x, dx, modes=("reverse",),
                      epsilons=tuple(1e-1 * 2.0 ** -k for k in range(8)), window=(4, 5, 6, 7))


def test_float32_leaf_raises():
    x = {"a": np.array([0.1, 0.2], dtype=np.float32)}
    with pytest.raises(TypeError, match="float64"):
        taylor_remainders(lambda v: jnp.sum(v["a"]), x, {"a": np.array([1.0, 1.0], dtype=np.float32)})


def test_unbatched_equals_batched(point):
    x, dx = point
    a = taylor_remainders(_smooth, x, dx, batched=True)
    b = taylor_remainders(_smooth, x, dx, batched=False)
    np.testing.assert_allclose(a.r1, b.r1, rtol=1e-6, atol=1e-15)


def test_remeasure_reuses_values_and_matches_a_fresh_measurement(point):
    """``remeasure_taylor`` with a 1.01 biased derivative gives the same remainders
    as a fresh measurement with that derivative, and the verdict is a fail. The
    forward values, the base value ``f(x)`` included, come from the one jitted
    ``vmap`` call of ``assert_taylor``, so the remeasurement needs no evaluation of
    ``f``."""
    from taylor import assess, remeasure_taylor
    x, dx = point
    results = assert_taylor(_smooth, x, dx, modes=("reverse",))
    for res in results:
        biased = remeasure_taylor(res, 1.01 * res.derivative)
        fresh = taylor_remainders(_smooth, x, dx, sign=res.sign, derivative=1.01 * res.derivative)
        np.testing.assert_allclose(biased.r1, fresh.r1, rtol=1e-9, atol=1e-16)
        assert assess(biased)[0] == "fail"


# --------------------------------------------------------------------------
# TaylorProgram and DotProgram (plan c4 D9): shared compiled programs.
# --------------------------------------------------------------------------


def test_program_equals_plain_function(point):
    """A ``TaylorProgram`` gives the same ``TaylorResult`` as the plain function, to the
    last bit, on a small analytic functional (same compiled arithmetic: the vmap of the
    plain path and of the program both evaluate f at x + t dx under jit)."""
    from taylor import TaylorProgram
    x, dx = point
    plain = assert_taylor(_smooth, x, dx)
    prog = assert_taylor(TaylorProgram(_smooth), x, dx)
    for a, b in zip(plain, prog):
        assert (a.sign, a.mode) == (b.sign, b.mode)
        assert a.base_value == b.base_value
        np.testing.assert_array_equal(a.values, b.values)
        assert a.derivative == b.derivative
        np.testing.assert_array_equal(a.r1, b.r1)


def test_program_second_case_does_not_compile_again(point):
    """A second base point and direction of the same structure reuse the compiled
    programs: a Python counter inside f counts the traces."""
    from taylor import DotProgram, TaylorProgram
    x, dx = point
    count = {"n": 0}

    def f(y):
        count["n"] += 1
        return _smooth(y)

    prog = TaylorProgram(f)
    assert_taylor(prog, x, dx)
    after_first = count["n"]
    assert after_first > 0
    x2 = jax.tree_util.tree_map(lambda a: a + 0.1, x)
    assert_taylor(prog, x2, random_like(x, 8))
    # A one-sign measurement pads its steps to the same length: no new program.
    taylor_remainders(prog, x2, dx, sign=-1.0, mode="forward")
    assert count["n"] == after_first
    # The dot-product programs behave the same way.
    F = lambda y: {"u": jnp.sin(y["a"]) * y["b"][0, 0], "v": y["b"] ** 2}
    dot = DotProgram(F)
    assert_jvp_vjp(dot, x, rtol=DOT_RTOL)
    traced = dot.jvp_both._cache_size()
    assert_jvp_vjp(dot, x2, seed=5, rtol=DOT_RTOL)
    assert dot.jvp_both._cache_size() == traced == 1
