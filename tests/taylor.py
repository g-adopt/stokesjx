"""Taylor tests and the jvp/vjp dot-product check for stokesjx.

The policy follows lovejx (tests/taylor_support.py there): a fixed epsilon ladder and
a fixed four-point window declared before the run, both direction signs, an
arithmetic allowance for the rounding of one forward value, and rate intervals that
account for that allowance. A measurement whose remainders sit near the allowance is
"unresolved", and unresolved is not a pass.

For a smooth scalar functional ``f`` and a direction ``dx``, with
``x_e = x + s e dx`` (``s`` the sign, ``e`` a rung of the ladder):

    r0(e) = |f(x_e) - f(x)|                       ~ e     (first order)
    r1(e) = |f(x_e) - f(x) - s e <grad f, dx>|    ~ e^2   (second order)

A correct derivative gives a corrected rate ``log(r1(e_k)/r1(e_k+1)) / log 2`` of 2;
a derivative off by a relative 1 % gives a corrected remainder that turns first order
once ``e`` is small enough, and the rate drops towards 1.

The derivative comes from reverse mode (``jax.grad``) or from forward mode
(``jax.jvp``); both modes run by default, each under ``jax.jit``. The forward values
do not depend on the mode: ``assert_taylor`` evaluates ``f`` at ``x`` (``t = 0``) and
at every perturbed point of both signs in one jitted ``jax.vmap`` call, and every
(mode, sign) measurement reuses these values. The base value ``f(x)`` therefore comes
from the same compiled function as the perturbed values, so the differences
``f(x_e) - f(x)`` carry no difference between eager and compiled evaluation. A
negative control (a biased derivative) reuses the stored values through
``remeasure_taylor`` and costs no further evaluation of ``f``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

DEFAULT_EPSILONS = tuple(1e-2 * 2.0 ** -k for k in range(8))
EPS64 = np.finfo(np.float64).eps


@dataclass(frozen=True)
class TaylorResult:
    """One Taylor measurement on a fixed ladder, for one sign and one mode.

    Attributes
    ----------
    sign, mode
        Direction sign (+1 or -1) and the source of the derivative.
    base_value
        ``f(x)``.
    values
        ``f(x + sign e dx)`` per rung (the forward values that ``remeasure_taylor``
        reuses).
    derivative
        ``<grad f, dx>`` (the directional derivative along ``+dx``).
    epsilons
        The ladder, strictly decreasing.
    r0, r1
        Uncorrected and corrected remainders per rung.
    rates0, rates1
        Rates between neighbouring rungs (NaN where a remainder is zero).
    allowance
        Arithmetic floor of one forward value:
        ``64 eps max(|f(x)|, max_e |f(x_e)|)`` unless given.
    """

    sign: float
    mode: str
    base_value: float
    values: np.ndarray
    derivative: float
    epsilons: np.ndarray
    r0: np.ndarray
    r1: np.ndarray
    rates0: np.ndarray
    rates1: np.ndarray
    allowance: float

    def table(self) -> str:
        """The full ladder as text, for failure messages."""
        lines = [f"sign {self.sign:+.0f}, mode {self.mode}, f(x) = {self.base_value:.17e}, "
                 f"derivative = {self.derivative:.17e}, allowance = {self.allowance:.3e}",
                 f"{'epsilon':>12s} {'r0':>12s} {'r1':>12s} {'rate0':>8s} {'rate1':>8s}"]
        for k, e in enumerate(self.epsilons):
            ra0 = self.rates0[k - 1] if k > 0 else np.nan
            ra1 = self.rates1[k - 1] if k > 0 else np.nan
            lines.append(f"{e:12.4e} {self.r0[k]:12.4e} {self.r1[k]:12.4e} {ra0:8.3f} {ra1:8.3f}")
        return "\n".join(lines)


def _leaves(tree):
    """The leaves of a pytree, in the order of ``jax.tree_util.tree_leaves``."""
    import jax
    return jax.tree_util.tree_leaves(tree)


def _check_float64(tree, what: str):
    """Every leaf must be a float64 array: a float32 leaf would make the ladder
    meaningless (its rounding is 1e-7, far above the second-order remainder)."""
    for leaf in _leaves(tree):
        dt = np.asarray(leaf).dtype
        if dt != np.float64:
            raise TypeError(f"{what} leaf has dtype {dt}; the Taylor helper needs float64")


def random_like(tree, seed: int):
    """A pytree of standard-normal float64 arrays with the structure of ``tree``."""
    import jax
    rng = np.random.default_rng(seed)
    return jax.tree_util.tree_map(lambda a: rng.standard_normal(np.shape(a)), tree)


def _vdot_tree(a, b) -> float:
    """``<a, b>``: the sum of the elementwise products over all leaves of two
    pytrees of the same structure, in float64 numpy (outside JAX, so that the
    contraction order is fixed)."""
    return float(sum(np.vdot(np.asarray(x).ravel(), np.asarray(y).ravel())
                     for x, y in zip(_leaves(a), _leaves(b))))


def _rates(rem, eps):
    """Observed convergence rates ``log(rem_k / rem_(k+1)) / log(eps_k / eps_(k+1))``
    between neighbouring rungs; NaN where either remainder is zero (log of zero)."""
    out = np.full(len(rem) - 1, np.nan)
    for k in range(len(rem) - 1):
        if rem[k] > 0 and rem[k + 1] > 0:
            out[k] = np.log(rem[k] / rem[k + 1]) / np.log(eps[k] / eps[k + 1])
    return out


class TaylorProgram:
    """The compiled programs of one scalar functional ``f`` for many base points and
    directions of one pytree structure (plan c4 D9).

    A plain callable passed to ``assert_taylor`` is compiled again at every call: the
    batched forward values close over the base point and the direction, and the
    gradient and the jvp are new ``jax.jit`` objects each time. On a solver of a few
    hundred nodes one compile costs about 0.7 s, and a Taylor case needs three of them.
    A ``TaylorProgram`` compiles each program once, with the base point, the direction
    and the steps as arguments, so that every further case of the same pytree
    structure and shapes reuses them (four cases on the reduced PREM table at
    ``lmax = 20``: 12.0 s with separate programs, 2.84 s with one program;
    NOTES/chunks/c4-controls-derivatives/plan-evidence/taylor_cost_proto.out).

    The steps of ``values`` are padded with zeros to a fixed length (``pad_to``, by
    default the length of the two-sign default ladder plus the base point), so a
    one-sign measurement and a one-sided difference reuse the same compiled vmap.

    Parameters
    ----------
    f
        Scalar functional of a pytree of float64 arrays. It must be traceable as a
        whole (``jit``, ``vmap``, ``grad`` and ``jvp``).
    pad_to
        Length to which ``values`` pads its steps.
    """

    def __init__(self, f: Callable, pad_to: int = 1 + 2 * len(DEFAULT_EPSILONS)):
        import jax

        self.f = f
        self.pad_to = int(pad_to)

        def point(x, dx, t):
            # One point x + t dx of the line through x along dx.
            return f(jax.tree_util.tree_map(lambda a, d: a + t * d, x, dx))

        # The steps are the only batched argument; x and dx are broadcast.
        self._values = jax.jit(jax.vmap(point, in_axes=(None, None, 0)))
        self._grad = jax.jit(jax.grad(f))
        self._jvp = jax.jit(lambda x, dx: jax.jvp(f, (x,), (dx,))[1])

    def values(self, x, dx, steps) -> np.ndarray:
        """``f(x + t dx)`` for every ``t`` in ``steps``, float64 numpy, one compiled call.

        Steps shorter than ``pad_to`` are padded with zeros (and the padding is cut off
        again); a longer list compiles a second program of its own length."""
        import jax.numpy as jnp
        steps = np.asarray(steps, dtype=np.float64)
        n = steps.size
        padded = np.zeros(max(n, self.pad_to))
        padded[:n] = steps
        return np.asarray(self._values(x, dx, jnp.asarray(padded)), dtype=np.float64)[:n]

    def derivative(self, x, dx, mode: str) -> float:
        """``<grad f(x), dx>`` from the compiled reverse mode (``jax.grad``, then a
        float64 dot product on the host) or the compiled forward mode (``jax.jvp``)."""
        if mode == "reverse":
            return _vdot_tree(self._grad(x), dx)
        if mode == "forward":
            return float(self._jvp(x, dx))
        raise ValueError(mode)


class LinearTaylorProgram(TaylorProgram):
    """A ``TaylorProgram`` of a functional that is linear in the outputs of a map,
    ``f(x) = <c, G(x)>`` with fixed weights ``c`` (a pytree of the structure of ``G(x)``),
    whose derivatives come from the jvp and the vjp of ``G`` (a ``DotProgram``):

        forward:  <grad f, dx> = <c, J dx>          (dot.jvp)
        reverse:  <grad f, dx> = <J^T c, dx>        (dot.vjp)

    These are the forward and the reverse mode of ``f`` itself (the contraction with
    ``c`` is linear), so the Taylor test checks the same derivatives, and the programs
    of the dot-product test serve both: three compiles per program (the batched values,
    the jvp, the vjp) in place of six."""

    def __init__(self, f: Callable, dot: "DotProgram", weights, pad_to: int = 1 + 2 * len(DEFAULT_EPSILONS)):
        super().__init__(f, pad_to)
        self.dot = dot
        self.weights = weights

    def derivative(self, x, dx, mode: str) -> float:
        if mode == "reverse":
            return _vdot_tree(self.dot.vjp(x, self.weights), dx)
        if mode == "forward":
            return _vdot_tree(self.weights, self.dot.jvp(x, dx))
        raise ValueError(mode)


class DotProgram:
    """The compiled programs of the jvp/vjp dot-product test of one map ``F`` (plan c4
    D9): the value, the jvp and the vjp, each one ``jax.jit`` with the point and the
    tangent or cotangent as arguments (``jax.jit`` compiles on the first call only).
    ``assert_jvp_vjp`` accepts a ``DotProgram`` in place of ``F``, takes the output shapes
    from ``jax.eval_shape`` (the value is not compiled) and compiles nothing new for a
    further point or seed."""

    def __init__(self, F: Callable):
        import jax
        self.F = F
        self.value = jax.jit(F)
        # The jvp program returns the primal output too (XLA computes it anyway), so that a
        # caller can read the outputs at a point from the program it compiles for the
        # tangents (for example the scales of a normalised map).
        self.jvp_both = jax.jit(lambda x, dx: jax.jvp(F, (x,), (dx,)))
        self.vjp = jax.jit(lambda x, dy: jax.vjp(F, x)[1](dy)[0])

    def jvp(self, x, dx):
        """``J dx`` (the tangent of the compiled ``jvp_both``)."""
        return self.jvp_both(x, dx)[1]


def directional_derivative(f, x, dx, mode: str, jit: bool = True) -> float:
    """``<grad f(x), dx>`` from reverse mode (``jax.grad``) or forward mode
    (``jax.jvp``), compiled with ``jax.jit`` unless ``jit=False`` (for a function
    that cannot be traced as a whole). ``f`` can be a ``TaylorProgram``, whose compiled
    programs are reused."""
    import jax
    if isinstance(f, TaylorProgram):
        return f.derivative(x, dx, mode)
    if mode == "reverse":
        g = jax.grad(f)
        return _vdot_tree((jax.jit(g) if jit else g)(x), dx)
    if mode == "forward":
        def jvp(x, dx):
            return jax.jvp(f, (x,), (dx,))[1]
        return float((jax.jit(jvp) if jit else jvp)(x, dx))
    raise ValueError(mode)


def _check_ladder(epsilons) -> np.ndarray:
    """The ladder as a float64 array; raises ``ValueError`` unless it is strictly
    decreasing and positive."""
    eps = np.asarray(epsilons, dtype=np.float64)
    if eps.ndim != 1 or len(eps) < 2 or np.any(np.diff(eps) >= 0) or np.any(eps <= 0):
        raise ValueError("epsilons must be a strictly decreasing positive ladder")
    return eps


def _values_along(f, x, dx, steps, batched: bool) -> np.ndarray:
    """``f(x + t dx)`` for every ``t`` in ``steps``, as float64 numpy.

    With ``batched=True`` all points are one ``jax.jit(jax.vmap(...))`` call (one
    compile, one launch); otherwise ``f`` is called once per point, eagerly. A
    ``TaylorProgram`` uses its own compiled vmap (``batched`` is then ignored)."""
    import jax
    import jax.numpy as jnp

    if isinstance(f, TaylorProgram):
        return f.values(x, dx, steps)

    def point(t):
        return f(jax.tree_util.tree_map(lambda a, d: a + t * d, x, dx))
    if batched:
        return np.asarray(jax.jit(jax.vmap(point))(jnp.asarray(steps)), dtype=np.float64)
    return np.array([float(point(t)) for t in steps], dtype=np.float64)


def _result(sign, mode, f0, vals, derivative, eps, allowance) -> TaylorResult:
    """Build a ``TaylorResult`` from the base value, the forward values along
    ``sign * eps * dx`` and a directional derivative along ``+dx``."""
    steps = sign * eps
    diff = vals - f0
    r0 = np.abs(diff)
    r1 = np.abs(diff - steps * derivative)
    if allowance is None:
        allowance = 64 * EPS64 * max(abs(f0), float(np.max(np.abs(vals))))
    return TaylorResult(sign=float(sign), mode=mode, base_value=float(f0), values=np.asarray(vals),
                        derivative=float(derivative), epsilons=eps, r0=r0, r1=r1,
                        rates0=_rates(r0, eps), rates1=_rates(r1, eps), allowance=float(allowance))


def remeasure_taylor(res: TaylorResult, derivative: float, *, allowance: float | None = None) -> TaylorResult:
    """The measurement ``res`` with another directional derivative (along ``+dx``).

    The forward values are reused, so this costs no evaluation of ``f``. Use it for
    negative controls: ``assess(remeasure_taylor(res, 1.01 * res.derivative))``
    must not pass. The allowance is kept unless given."""
    return _result(res.sign, res.mode, res.base_value, res.values, derivative, res.epsilons,
                   res.allowance if allowance is None else allowance)


def taylor_remainders(f: Callable, x, dx, *, sign: float = 1.0,
                      mode: Literal["reverse", "forward"] = "reverse",
                      epsilons=DEFAULT_EPSILONS, derivative: float | None = None,
                      batched: bool = True, allowance: float | None = None) -> TaylorResult:
    """Measure the Taylor remainders of ``f`` at ``x`` along ``sign * dx``.

    Parameters
    ----------
    f
        Scalar functional of a pytree of float64 arrays, or a ``TaylorProgram`` (its
        compiled programs are reused).
    derivative
        Directional derivative along ``+dx`` to test. Default: from ``mode``. A
        negative control passes a biased value here.
    batched
        Evaluate all perturbed points in one ``jax.vmap`` call (one compile). Set
        False for a function that ``vmap`` cannot take.
    """
    _check_float64(x, "control")
    _check_float64(dx, "direction")
    eps = _check_ladder(epsilons)
    if derivative is None:
        derivative = directional_derivative(f, x, dx, mode, jit=batched)
    # t = 0 first, so that f(x) comes from the same evaluation as the ladder.
    vals = _values_along(f, x, dx, np.concatenate([[0.0], sign * eps]), batched)
    return _result(sign, mode, vals[0], vals[1:], derivative, eps, allowance)


def _interval(large, small, ratio, allowance):
    """Rate interval of two remainders that each carry an error up to ``allowance``
    (the lovejx ``_rate_interval``). None when either is not above the allowance."""
    if not (np.isfinite(large) and np.isfinite(small)) or large <= allowance or small <= allowance:
        return None
    den = np.log(ratio)
    return (np.log((large - allowance) / (small + allowance)) / den,
            np.log((large + allowance) / (small - allowance)) / den)


def assess(res: TaylorResult, *, window=(0, 1, 2, 3), classification="smooth",
           min_rate=1.9, first_order_band=(0.8, 1.2), resolution=10.0) -> tuple[str, str]:
    """Verdict ("pass", "fail" or "unresolved") and its reason for one measurement.

    Parameters
    ----------
    res
        One measurement (one sign, one mode).
    window
        Rung indices on which the rates are judged (smooth class: at least four
        consecutive rungs). Fixed before the run.
    classification
        "smooth": the corrected rate interval must lie above ``min_rate`` and the
        uncorrected one inside ``first_order_band`` for every neighbouring pair of
        the window; both remainders must be above ``resolution * allowance``,
        otherwise the verdict is "unresolved". "affine": the response is exactly
        affine in the control, so the corrected remainder must stay below the
        allowance at every rung while the first-order signal is resolved. "zero":
        forward response and derivative are both zero within the allowance.
    min_rate
        Lower bound of the corrected rate (2 for a correct derivative, 1 for a
        biased one).
    first_order_band
        Interval for the uncorrected rate (1 for any nonzero derivative).
    resolution
        Factor above the allowance that a remainder needs to count as measured.

    Returns
    -------
    ``(verdict, reason)``.
    """
    w = np.asarray(window)
    if classification == "smooth" and (len(w) < 4 or np.any(np.diff(w) != 1)):
        raise ValueError("a smooth window has at least four consecutive rungs")
    b = res.allowance
    lin = np.abs(res.epsilons * res.derivative)
    if not (np.isfinite(res.base_value) and np.all(np.isfinite(res.r0)) and np.all(np.isfinite(res.r1))):
        return "unresolved", "non-finite forward value"
    if classification == "zero":
        if np.all(res.r0 <= b) and np.all(lin <= b):
            return "pass", "forward response and derivative are zero within the allowance"
        return "fail", "zero-response case has a nonzero forward value or derivative"
    if classification == "affine":
        if np.any(res.r1 > b):
            return "fail", "corrected remainder of an affine response exceeds the allowance"
        if np.any(res.r0 <= resolution * b) or np.any(lin <= resolution * b):
            return "unresolved", "affine signal below the resolution limit"
        return "pass", "resolved affine response within the allowance"
    if classification != "smooth":
        raise ValueError(classification)
    if np.any(res.r1[w] <= resolution * b) or np.any(res.r0[w] <= resolution * b):
        return "unresolved", "remainder below the resolution limit in the window"
    for a_, c_ in zip(w[:-1], w[1:]):
        ratio = res.epsilons[a_] / res.epsilons[c_]
        corr = _interval(res.r1[a_], res.r1[c_], ratio, b)
        unc = _interval(res.r0[a_], res.r0[c_], ratio, b)
        if corr is None or unc is None:
            return "unresolved", "allowance prevents a rate interval"
        if corr[0] < min_rate:
            return "fail", f"corrected rate lower bound {corr[0]:.3f} below {min_rate}"
        if unc[0] < first_order_band[0] or unc[1] > first_order_band[1]:
            return "fail", f"uncorrected rate interval ({unc[0]:.3f}, {unc[1]:.3f}) outside {first_order_band}"
    return "pass", "second-order corrected and first-order uncorrected rates"


def assert_taylor(f: Callable, x, dx, *, epsilons=DEFAULT_EPSILONS, window=(0, 1, 2, 3),
                  classification: Literal["smooth", "affine", "zero"] = "smooth",
                  min_rate: float = 1.9, first_order_band=(0.8, 1.2), resolution: float = 10.0,
                  allowance: float | None = None, signs=(1.0, -1.0),
                  modes=("reverse", "forward"), derivative: float | None = None,
                  batched: bool = True) -> list[TaylorResult]:
    """Run the Taylor test for every sign and mode and assert a pass for each.

    ``f`` is evaluated once at ``x`` and at every rung of every sign, in one jitted
    ``vmap`` call (``f`` can be a ``TaylorProgram``, whose compiled programs are
    reused for every base point and direction of the same structure); each mode computes its derivative once (jitted) and all
    (mode, sign) measurements share the forward values. Raises ``AssertionError``
    with the full ladder table on a fail or an unresolved measurement.
    ``derivative`` replaces the derivative of every mode (negative controls; for a
    control after a passing test, ``remeasure_taylor`` on the returned results is
    cheaper). Returns the measurements, ordered by mode and then by sign.
    """
    _check_float64(x, "control")
    _check_float64(dx, "direction")
    eps = _check_ladder(epsilons)
    signs = tuple(float(s) for s in signs)
    steps = np.concatenate([[0.0]] + [s * eps for s in signs])
    vals = _values_along(f, x, dx, steps, batched)
    f0 = vals[0]
    per_sign = {s: vals[1 + k * len(eps): 1 + (k + 1) * len(eps)] for k, s in enumerate(signs)}
    out = []
    failures = []
    for mode in modes:
        d = derivative if derivative is not None else directional_derivative(f, x, dx, mode, jit=batched)
        for s in signs:
            res = _result(s, mode, f0, per_sign[s], d, eps, allowance)
            verdict, reason = assess(res, window=window, classification=classification,
                                     min_rate=min_rate, first_order_band=first_order_band,
                                     resolution=resolution)
            out.append(res)
            if verdict != "pass":
                failures.append(f"[{verdict}] {reason}\n{res.table()}")
    if failures:
        raise AssertionError("Taylor test did not pass:\n" + "\n\n".join(failures))
    return out


def assert_jvp_vjp(F: Callable, x, *, dx=None, dy=None, seed: int = 0, rtol: float = 1e-12,
                   forward: Callable | None = None, jit: bool = True) -> float:
    """Dot-product test ``<dy, J dx> = <J^T dy, dx>`` between forward and reverse mode.

    ``a = <dy, J dx>`` comes from ``jax.jvp`` of ``forward`` (default ``F``), and
    ``b = <J^T dy, dx>`` from ``jax.vjp`` of ``F``. Pass ``forward`` to check a
    custom reverse rule (a ``custom_vjp``, which has no forward mode) against the
    forward mode of the plain implementation. Random ``dx`` and ``dy`` come from a
    fixed seed. The value, the jvp and the vjp are each compiled with ``jax.jit``
    unless ``jit=False``; a ``DotProgram`` in place of ``F`` brings its own compiled
    programs, so a second point or seed compiles nothing. Returns the relative difference
    ``|a - b| / max(|a|, |b|)`` and raises ``AssertionError`` above ``rtol``.
    """
    import jax
    _check_float64(x, "point")
    if isinstance(F, DotProgram):
        # The compiled programs of the map: no new compile for a further point or seed.
        # The output is needed only for its shapes (the random cotangent), which
        # jax.eval_shape gives without compiling the value.
        if forward is not None:
            raise ValueError("a DotProgram carries its own forward map; pass forward=None")
        jvp_fn, vjp_fn = F.jvp, F.vjp

        def value_fn(x):
            return jax.tree_util.tree_map(lambda s: np.zeros(s.shape, s.dtype), jax.eval_shape(F.F, x))
    else:
        forward = forward or F

        def jvp(x, dx):
            return jax.jvp(forward, (x,), (dx,))[1]

        def vjp(x, dy):
            return jax.vjp(F, x)[1](dy)[0]
        wrap = jax.jit if jit else (lambda g: g)
        value_fn, jvp_fn, vjp_fn = wrap(F), wrap(jvp), wrap(vjp)
    y = value_fn(x)
    _check_float64(y, "output")
    dx = random_like(x, seed) if dx is None else dx
    dy = random_like(y, seed + 1) if dy is None else dy
    jdx = jvp_fn(x, dx)
    jtdy = vjp_fn(x, dy)
    a = _vdot_tree(dy, jdx)
    b = _vdot_tree(jtdy, dx)
    rel = abs(a - b) / max(abs(a), abs(b))
    if not rel <= rtol:
        raise AssertionError(f"jvp/vjp dot products differ: <dy, J dx> = {a:.17e}, "
                             f"<J^T dy, dx> = {b:.17e}, relative {rel:.3e} > {rtol:.1e}")
    return rel
