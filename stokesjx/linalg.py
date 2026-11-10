"""Small batched linear algebra for the propagator solver.

The primitives of chunk 1 (``benchmarks/candidates.py``), as package code: the Pade
pair of the interval exponentials with its static plan, the elimination solver of the
Pade denominators with its derivative rule, the three-column Gram-Schmidt of the
sweeps with its triangular factor, the right triangular solve of the carried output
rows, and the batched 6x6 node solve. ``system.py`` and ``propagate.py`` both use
them, and ``tests/test_linalg.py`` tests them on their own.

Layouts. The exponential stage works on batch-last arrays ``(6, 6, *batch)`` (matrix
axes first): a 6x6 product is then one broadcast multiply and one sum over the batch,
which XLA vectorises over the long batch axes (CONVENTIONS 3, measured 12 to 19 %
faster than ``@`` on batch-first arrays). The sweeps and the node solve work on
batch-first arrays ``(*batch, 6, k)``: the scan carry is small, and a transpose to the
other layout costs more than it saves there (CONVENTIONS 3).

No function here has a default for the Pade order, the squaring count or the solver
(JAX review r3 of chunk 1, minor 8): the order and the count must come from
``pade_plan`` with a static norm bound, because nothing can check them under ``jit``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

# --------------------------------------------------------------------------
# Batch-last matrix products.
# --------------------------------------------------------------------------


def matmul_batch_last(a, b):
    """Product of two stacks of matrices stored with the matrix axes first.

    Parameters
    ----------
    a
        ``(n, k, *batch)``, float64.
    b
        ``(k, m, *batch)``, float64 (the batch axes broadcast against those of ``a``).

    Returns
    -------
    ``C[i, j, ...] = sum_k a[i, k, ...] b[k, j, ...]``, shape ``(n, m, *batch)``.

    One broadcast multiply and one sum over ``k``. With 6x6 matrices and tens of
    thousands of batch entries the long batch axes are innermost, so XLA emits one
    vectorised elementwise loop in place of many small ``dot`` calls (CONVENTIONS 3).
    The derivative of a multiply and a sum is again a multiply and a sum, so the
    reverse pass keeps the layout.
    """
    return jnp.sum(a[:, :, None] * b[None, :, :], axis=1)


# --------------------------------------------------------------------------
# Elimination without pivoting for the Pade denominators.
# --------------------------------------------------------------------------


def _solve_elimination_plain(A, B):
    """``X = A^-1 B`` by Gauss-Jordan elimination without pivoting on batch-last
    arrays: ``A`` of shape ``(n, n, *batch)``, ``B`` of shape ``(n, m, *batch)``.

    Step ``k`` divides the pivot row by the pivot and removes the pivot column from
    every other row by one rank-one update of the whole batch. Columns that are
    already eliminated are dropped from ``A`` by static slices, so step ``k`` updates
    an ``n x (n - k - 1)`` block of ``A`` and the ``n x m`` block of ``B``.

    ``A`` and ``B`` stay separate arrays through all steps (no augmented matrix). The
    reason is reverse mode through the derivative rule of ``solve_elimination``: JAX
    transposes the tangent solve ``dB -> A^-1 dB``, and the transpose exists only when
    the operations on ``B`` are linear in ``B`` by their structure, which an augmented
    array that mixes ``A`` and ``B`` would hide (JAX review r3 of chunk 1, section 5).
    """
    n = A.shape[0]
    shape = (n,) + (1,) * (A.ndim - 2)
    for k in range(n):
        # Column 0 of the remaining block of A is column k of the original matrix.
        inv_pivot = 1.0 / A[k, 0]
        row_a = A[k, 1:] * inv_pivot                  # pivot row, (n - k - 1, *batch)
        row_b = B[k] * inv_pivot                      # (m, *batch)
        # Multipliers of the pivot column; row k itself is replaced, not updated.
        factor = A[:, 0] * (jnp.arange(n) != k).astype(A.dtype).reshape(shape)
        A = (A[:, 1:] - factor[:, None] * row_a[None, :]).at[k].set(row_a)
        B = (B - factor[:, None] * row_b[None, :]).at[k].set(row_b)
    return B


@jax.custom_jvp
def solve_elimination(A, B):
    """``X = A^-1 B`` for a stack of small systems, by Gauss-Jordan elimination
    without pivoting (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed.,
    2002, section 14.4), with the derivative rule of a linear solve.

    Parameters
    ----------
    A
        ``(n, n, *batch)``, float64, strictly diagonally dominant by columns.
    B
        ``(n, m, *batch)``, float64.

    Returns
    -------
    ``X``, shape ``(n, m, *batch)``.

    Numerical caveat: there is no pivoting. For a matrix that is strictly diagonally
    dominant by columns every pivot is nonzero, every Schur complement keeps the
    property, and the growth factor is at most 2 (Wilkinson; Higham 2002, chapter 9).
    The Pade denominators ``N(+-X)`` have this property when ``N_m(||X||_1) < 2``,
    which ``pade_plan`` enforces from the static norm bound; their 1-norm condition
    number is then at most ``N_m / (2 - N_m)``, measured at most 2.5 over the plans of
    this package, so a small margin ``2 - N_m`` (0.32 to 0.37 at the plans of the
    benchmark sizes) is still a well-conditioned solve. For other matrices use
    ``solve_nodes``.

    Derivative rule: from ``A X = B``, ``dX = A^-1 (dB - dA X)``, one more elimination
    with the same ``A``. In reverse mode JAX transposes this linear map. Plain reverse
    mode through the elimination would differentiate every step of the elimination of
    ``A`` and was measured slower (gradient 26.7 against 24.2 ms, compile 0.66 against
    0.48 s at the size (51, 128); CONVENTIONS 3).
    """
    return _solve_elimination_plain(A, B)


@solve_elimination.defjvp
def _solve_elimination_jvp(primals, tangents):
    """Tangent of ``X = A^-1 B``: ``dX = A^-1 (dB - dA X)``."""
    A, B = primals
    dA, dB = tangents
    X = solve_elimination(A, B)
    return X, solve_elimination(A, dB - matmul_batch_last(dA, X))


# --------------------------------------------------------------------------
# The Pade pair of exp(M) and exp(-M).
# --------------------------------------------------------------------------


def _pade_coefficients(p: int) -> list[float]:
    """Coefficients ``c_k`` (``k = 0..p``) of the numerator of the diagonal ``(p, p)``
    Pade approximant of ``exp(x)``: ``c_k = (2p - k)! p! / ((2p)! k! (p - k)!)``. The
    denominator has the coefficients ``(-1)^k c_k``, so the approximant is
    ``N(x) / N(-x)``."""
    return [math.factorial(2 * p - k) * math.factorial(p)
            / (math.factorial(2 * p) * math.factorial(k) * math.factorial(p - k)) for k in range(p + 1)]


#: Pade orders that ``pade_plan`` chooses from. Order 13 is not offered: with the
#: elimination solver the dominance condition ``N_m(x) < 2`` binds at a scaled norm
#: of 1.406 for order 13 against 1.416 for order 9, so order 9 always needs fewer
#: products and order 13 is never selected (JAX review r3 of chunk 1, minor 3).
PADE_ORDERS = (6, 7, 9)

#: Accuracy thresholds of the scaled 1-norm per order. The relative backward error of
#: the diagonal approximant ``r_m`` at ``x = ||X||_1`` is bounded by
#: ``|log(exp(-x) r_m(x))| / x`` (Higham 2005, SIAM J. Matrix Anal. Appl. 26, 1179).
#: Higham's thresholds make this bound equal to the unit roundoff: 0.950 (order 7),
#: 2.098 (order 9). For order 6 the threshold 0.783 is this package's own choice,
#: with a bound of 9.3e-15 (computed at 50 digits); ``pade_plan`` never selects order
#: 6, because order 7 costs the same four products and allows a larger norm.
PADE_THETA = {6: 0.783, 7: 0.950, 9: 2.098}

#: Limit of the scalar numerator for the elimination solver: both denominators
#: ``N(+-X)`` are strictly diagonally dominant by columns when ``N_m(x) < 2``.
PADE_DENOMINATOR_LIMIT = 2.0


def pade_product_count(order: int) -> int:
    """Number of 6x6 products for the even powers and the odd part of the Pade
    numerator: the even powers ``X^2 .. X^(2 floor(order / 2))`` (one product each)
    and one product for the odd part. 4 for orders 6 and 7, 5 for order 9."""
    if order not in PADE_THETA:
        raise ValueError(f"Pade order {order} is not offered; use one of {PADE_ORDERS}")
    return order // 2 + 1


def pade_numerator(order: int, x: float) -> float:
    """Scalar Pade numerator ``N_m(x) = sum_k c_k x^k``. All ``c_k`` are positive, so
    for ``||X||_1 <= x`` every column of ``N(+-X) - I`` has a 1-norm of at most
    ``N_m(x) - 1``."""
    return sum(ck * x ** k for k, ck in enumerate(_pade_coefficients(order)))


def pade_denominator_margin(order: int, x: float) -> float:
    """Lower bound ``2 - N_m(x)`` of the column diagonal dominance of both Pade
    denominators for ``||X||_1 <= x``: in every column ``j``,
    ``|D_jj| - sum_(i != j) |D_ij| >= 1 - ||(D - I) e_j||_1 >= 2 - N_m(x)``. Positive
    means that elimination without pivoting is stable (``solve_elimination``)."""
    return PADE_DENOMINATOR_LIMIT - pade_numerator(order, x)


def pade_squarings(norm_bound: float, order: int) -> int:
    """Smallest squaring count ``s >= 0`` with ``x = norm_bound / 2^s <= PADE_THETA[order]``
    and ``N_m(x) < 2`` (the dominance condition of the elimination solver).

    ``norm_bound`` must bound the 1-norm of the exponent of every degree and interval,
    including the upper limits of every traced input (CONVENTIONS 2)."""
    theta = PADE_THETA[order]
    s = max(0, math.ceil(math.log2(max(norm_bound, 1e-300) / theta)))
    # N_m increases with x, and each squaring halves x, so the margin grows with s
    # and the loop ends.
    while pade_denominator_margin(order, norm_bound / 2.0 ** s) <= 0.0:
        s += 1
    return s


@dataclass(frozen=True)
class PadePlan:
    """Static Pade order and squaring count of the step-and-inverse pair.

    Attributes
    ----------
    order, squarings
        The approximant ``(order, order)`` and the number of squarings.
    scaled_norm
        ``norm_bound / 2^squarings``: the bound of the 1-norm of the scaled exponent.
    margin
        ``2 - N_order(scaled_norm)``: the dominance margin of both denominators.
    products
        6x6 products of the pair: ``pade_product_count(order) + 2 squarings`` (both
        members of the pair are squared).
    """

    order: int
    squarings: int
    scaled_norm: float
    margin: float
    products: int


def pade_plan(norm_bound: float) -> PadePlan:
    """The cheapest Pade plan for an exponent with 1-norm at most ``norm_bound``.

    For each order of ``PADE_ORDERS`` the squaring count comes from
    ``pade_squarings``. The plan takes the order with the fewest 6x6 products for the
    pair and, between orders of equal cost, the higher order (smaller backward error).
    Measured plans (D5 of the chunk 2 plan): ``lmax = 50`` on 129 LEBEDEV nodes has the
    bound 1.054 and order 9 without squaring; ``lmax = 100`` has 2.024 and order 9
    with one squaring; ``lmax = 200`` has 3.964 and order 9 with two squarings."""
    if not (norm_bound >= 0 and math.isfinite(norm_bound)):
        raise ValueError(f"norm bound must be finite and non-negative, got {norm_bound!r}")
    best = None
    for m in PADE_ORDERS:
        s = pade_squarings(norm_bound, m)
        cost = pade_product_count(m) + 2 * s
        if best is None or cost < best[0] or (cost == best[0] and m > best[1]):
            best = (cost, m, s)
    cost, m, s = best
    x = norm_bound / 2.0 ** s
    return PadePlan(order=m, squarings=s, scaled_norm=x, margin=pade_denominator_margin(m, x), products=cost)


def balanced_norm_bound_intervals(lmax: int, h_bound, kappa_h_bound, rho_max) -> float:
    """Upper bound of the 1-norm of the balanced exponent ``B`` (eq. 4.6) over the
    degrees ``1..lmax``, interval by interval:
    ``max_i ((3 lmax + 10 + rho_max_i) h_i + 4 |kappa_h|_i)`` (CONVENTIONS 2).

    The largest column sum of ``S_l A(1) S_l^-1`` is below ``3 l + 10`` for every
    ``l >= 1``; the density entries add at most ``rho0 / rho_s``; the balanced K term
    (TALA, chunk 3) has the column sum ``2 + 2/l``, at most 4.

    Parameters
    ----------
    lmax
        Largest degree.
    h_bound, kappa_h_bound, rho_max
        Static upper limits per interval (arrays or scalars that broadcast): the width
        in ``ln r``, ``|kappa h|`` and ``rho0 / rho_s``. Plain numpy; computed on the
        host when the static plan is built, never under ``jit``.
    """
    h = np.asarray(h_bound, dtype=float)
    kh = np.abs(np.asarray(kappa_h_bound, dtype=float))
    rho = np.asarray(rho_max, dtype=float)
    return float(np.max((3 * lmax + 10 + rho) * h + 4 * kh))


def _pade_even_odd(X, order: int):
    """Even and odd parts ``V`` and ``U`` of the Pade numerator ``N(X) = V + U`` on
    batch-last ``X`` (``(n, n, *batch)``): only the even powers ``X^2, X^4, ...`` are
    formed (one product each), and ``U = X (c_1 I + c_3 X^2 + ...)`` costs one more
    product. ``N(-X) = V - U`` needs no further product."""
    c = _pade_coefficients(order)
    eye = jnp.eye(X.shape[0], dtype=X.dtype).reshape(X.shape[:2] + (1,) * (X.ndim - 2))
    X2 = matmul_batch_last(X, X)
    even = [eye, X2]                     # X^0, X^2, X^4, ...
    while 2 * len(even) <= order:
        even.append(matmul_batch_last(even[-1], X2))
    V = sum(c[2 * j] * P for j, P in enumerate(even) if 2 * j <= order)
    odd = sum(c[2 * j + 1] * P for j, P in enumerate(even) if 2 * j + 1 <= order)
    return V, matmul_batch_last(X, odd)


def expm_pade_pair(M, order: int, squarings: int):
    """The Pade approximants of ``exp(M)`` and ``exp(-M)`` from one set of powers.

    Parameters
    ----------
    M
        Exponents, batch-last ``(n, n, *batch)``, float64.
    order, squarings
        From ``pade_plan`` with a static bound of ``||M||_1``. No defaults: a scaled
        norm above the threshold loses accuracy without a warning, and one with
        ``N_m >= 2`` removes the guarantee that the elimination is stable.

    Returns
    -------
    ``(E, E_inv)``, both batch-last of the shape of ``M``: ``E ~ expm(M)`` and
    ``E_inv ~ expm(-M) = E^-1``.

    The diagonal approximant satisfies ``r(-X) = r(X)^-1`` exactly: with
    ``X = M / 2^s``, ``r(X) = (V - U)^-1 (V + U)`` and ``r(-X) = (V + U)^-1 (V - U)``,
    and both factors of each commute, because they are polynomials in ``X``. Squaring
    commutes with the inverse, so the inverse step of the downward sweep costs no
    second exponential: one set of powers, two denominators solved in one stacked
    elimination (``solve_elimination``) and two sets of squarings (CONVENTIONS 3).
    """
    X = M / (2.0 ** squarings)
    V, U = _pade_even_odd(X, order)
    # Stack the two systems on a batch axis right after the matrix axes.
    E = solve_elimination(jnp.stack([V - U, V + U], axis=2), jnp.stack([V + U, V - U], axis=2))
    for _ in range(squarings):
        E = matmul_batch_last(E, E)
    return E[:, :, 0], E[:, :, 1]


# --------------------------------------------------------------------------
# Three-column Gram-Schmidt and the right triangular solve.
# --------------------------------------------------------------------------


def _dot(a, b):
    """Inner product over the last axis, kept as a length-one axis."""
    return jnp.sum(a * b, axis=-1, keepdims=True)


def gram_schmidt3(Z):
    """Orthonormal basis of the span of three columns, with the full triangular factor.

    Parameters
    ----------
    Z
        ``(..., 6, 3)``, float64, three linearly independent columns.

    Returns
    -------
    ``(Q, R)``: ``Q`` of shape ``(..., 6, 3)`` with orthonormal columns and ``R`` of
    shape ``(..., 3, 3)``, upper triangular with a positive diagonal, ``Q R = Z``.

    Classical Gram-Schmidt with one re-orthogonalisation (CGS2), written out for three
    columns, so that XLA fuses it into the scan body. For column ``k``:
    ``s = Q^T v``, ``v' = v - Q s``, ``t = Q^T v'``, ``v'' = v' - Q t``,
    ``R[:k, k] = s + t``, ``R[k, k] = ||v''||``. One re-orthogonalisation gives
    orthogonality to working precision for a condition number of the columns up to
    about ``1 / sqrt(eps)`` (Giraud, Langou and Rozloznik 2005), which the sweeps never
    approach: they re-orthonormalise after every interval. CGS2 was measured 3 x
    faster than LAPACK QR inside the scan (CONVENTIONS 14), and, with the largest node
    viscosity as the viscosity scale, as accurate (D1 of the chunk 2 plan).
    """
    v1 = Z[..., :, 0]
    r11 = jnp.sqrt(_dot(v1, v1))
    q1 = v1 / r11
    v2 = Z[..., :, 1]
    s = _dot(q1, v2)
    v2 = v2 - s * q1
    t = _dot(q1, v2)
    v2 = v2 - t * q1
    r12 = s + t
    r22 = jnp.sqrt(_dot(v2, v2))
    q2 = v2 / r22
    v3 = Z[..., :, 2]
    s1, s2 = _dot(q1, v3), _dot(q2, v3)
    v3 = v3 - s1 * q1 - s2 * q2
    t1, t2 = _dot(q1, v3), _dot(q2, v3)
    v3 = v3 - t1 * q1 - t2 * q2
    r13, r23 = s1 + t1, s2 + t2
    r33 = jnp.sqrt(_dot(v3, v3))
    q3 = v3 / r33
    Q = jnp.stack([q1, q2, q3], axis=-1)
    zero = jnp.zeros_like(r11)
    R = jnp.stack([jnp.concatenate([r11, r12, r13], axis=-1),
                   jnp.concatenate([zero, r22, r23], axis=-1),
                   jnp.concatenate([zero, zero, r33], axis=-1)], axis=-2)
    return Q, R


def right_solve_upper3(W, R):
    """``W R^-1`` for an upper triangular 3x3 ``R``, by forward substitution over the
    three columns, on whole arrays.

    Parameters
    ----------
    W
        ``(..., k, 3)``: ``k`` row vectors.
    R
        ``(..., 3, 3)``, upper triangular, nonzero diagonal (``gram_schmidt3``).

    Returns
    -------
    ``X`` of shape ``(..., k, 3)`` with ``X R = W``:
    ``x1 = w1 / r11``, ``x2 = (w2 - x1 r12) / r22``, ``x3 = (w3 - x1 r13 - x2 r23) / r33``.
    """
    r = lambda i, j: R[..., None, i, j]
    x1 = W[..., 0] / r(0, 0)
    x2 = (W[..., 1] - x1 * r(0, 1)) / r(1, 1)
    x3 = (W[..., 2] - x1 * r(0, 2) - x2 * r(1, 2)) / r(2, 2)
    return jnp.stack([x1, x2, x3], axis=-1)


# --------------------------------------------------------------------------
# The batched 6x6 node solve.
# --------------------------------------------------------------------------


def _pivot_elimination_plain(A, B):
    """Gauss-Jordan elimination with partial pivoting on batch-last arrays: ``A`` of
    shape ``(n, n, *batch)``, ``B`` of shape ``(n, m, *batch)``. The pivot of step
    ``k`` is the largest ``|A[i, k]|`` over ``i >= k``; the row exchange is a one-hot
    selection on the whole batch (no data-dependent shape). ``A`` and ``B`` stay
    separate arrays, so that the map ``B -> X`` is linear in ``B`` by structure, as
    the transpose of the tangent solve in reverse mode needs (see
    ``_solve_elimination_plain``)."""
    n = A.shape[0]
    ones = (1,) * (A.ndim - 2)
    for k in range(n):
        p = jnp.argmax(jnp.abs(A[k:, k]), axis=0)
        onehot = jnp.arange(n - k).reshape((n - k,) + ones) == p[None]

        def exchange(M, k=k, onehot=onehot):
            # Row p of the remaining block goes to position k, and row k goes to
            # position p; every other row stays.
            row_p = jnp.sum(jnp.where(onehot[:, None], M[k:], 0.0), axis=0)
            sub = jnp.where(onehot[:, None], M[k][None], M[k:]).at[0].set(row_p)
            return jnp.concatenate([M[:k], sub], axis=0)

        A, B = exchange(A), exchange(B)
        inv = 1.0 / A[k, k]
        ra, rb = A[k] * inv, B[k] * inv
        factor = A[:, k] * (jnp.arange(n) != k).astype(A.dtype).reshape((n,) + ones)
        A = (A - factor[:, None] * ra[None]).at[k].set(ra)
        B = (B - factor[:, None] * rb[None]).at[k].set(rb)
    return B


@jax.custom_jvp
def _pivot_elimination(A, B):
    """``_pivot_elimination_plain`` with the derivative rule ``dX = A^-1 (dB - dA X)``."""
    return _pivot_elimination_plain(A, B)


@_pivot_elimination.defjvp
def _pivot_elimination_jvp(primals, tangents):
    A, B = primals
    dA, dB = tangents
    X = _pivot_elimination(A, B)
    return X, _pivot_elimination(A, dB - matmul_batch_last(dA, X))


#: Methods of ``solve_nodes``.
NODE_SOLVERS = ("lapack", "pivot_elimination")


def solve_nodes(M, b, method: str):
    """Solve ``M x = b`` for a stack of 6x6 systems.

    Parameters
    ----------
    M
        ``(..., 6, 6)``, batch-first, float64. The node systems ``[-Q_up, Q_dn]`` are
        not diagonally dominant, so they need pivoting.
    b
        ``(..., 6)``.
    method
        ``"lapack"``: ``jnp.linalg.solve`` (LU with partial pivoting, one LAPACK call
        per matrix). ``"pivot_elimination"``: Gauss-Jordan with partial pivoting on
        batch-last arrays with the derivative rule ``dX = A^-1 (dB - dA X)``
        (measured 1.24 against 1.93 ms forward at the size (51, 128) in the JAX review
        r3 of chunk 1).

    Returns
    -------
    ``x``, shape ``(..., 6)``.
    """
    if method == "lapack":
        return jnp.linalg.solve(M, b[..., None])[..., 0]
    if method == "pivot_elimination":
        A = jnp.moveaxis(M, (-2, -1), (0, 1))
        B = jnp.moveaxis(b[..., None], (-2, -1), (0, 1))
        X = _pivot_elimination(A, B)
        return jnp.moveaxis(X, (0, 1), (-2, -1))[..., 0]
    raise ValueError(f"unknown node solver {method!r}; use one of {NODE_SOLVERS}")
