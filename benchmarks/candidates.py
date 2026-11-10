"""Candidate JAX primitives for the stokesjx solver, and a composed stand-in.

Nothing here is package code. This module holds the operations that chunk 3 will
assemble into the solver (plan section 8), in the form that the benchmark
(benchmarks/bench_primitives.py) times and that tests/test_primitives_ad.py
differentiates:

1. the interval step ``expm(A h + K kappa_h)`` in five candidate forms, and the
   step together with its inverse from one set of Pade powers
   (``expm_pade_pair``, ``r(-X) = r(X)^-1``), with the 6x6 products on batch-first
   arrays (``@``) or on batch-last arrays (``matmul_batch_last``), the Pade
   denominators solved by elimination without pivoting on batch-last arrays
   (``solve_elimination``, with a derivative rule) or by LAPACK, and the Pade order
   and the squaring count chosen from a static norm bound (``pade_plan``);
2. the step across a jump pair under TALA, ``I + K (1 - exp(-t))``;
3. the QR-stabilised sweep (``lax.scan`` with a (L, 6, 3) carry);
4. the batched 6x6 solve at every node;
5. a composed stand-in of the solver that chains 1, 3 and 4 under one ``jit``.

Notation (NOTES/docs/equations.md, section 4). ``A(eta)`` is the 6x6 matrix in
``ln r`` for one interval with viscosity ``eta`` (in units of ``eta_s``) and density
``rho`` (in units of ``rho_s``); ``K(eta) = dA/dkappa``. Two exact similarities
remove the large entries of A:

* viscosity: ``A(eta) = D_eta A(1) D_eta^-1`` with ``D_eta = diag(1, 1, eta, eta,
  eta, eta)``, also for K (eq. 4.5);
* degree: with ``S_l = diag(1, l, 1/(2l), 1/2, 1/2, 1/(2l))``, every entry of
  ``S_l A(1) S_l^-1`` is at most about ``2 (l + 1)`` in size and its 1-norm is about
  ``3 l``, while A(1) itself has entries up to ``6 l (l + 1)``. The cycle products
  of A (``a12 a21 = -L``, ``a24 a42 = 2 (2L - 1)``, ``a34 a43 = -L``,
  ``a56 a65 = L``) are invariant under a diagonal similarity, so entries of order
  ``l`` are the smallest that a diagonal scaling can reach.

So ``expm(A(eta) h + K(eta) kappa_h) = T expm(B) T^-1`` with ``T = D_eta S_l^-1`` and
the balanced argument ``B = S_l (A(1) h + K(1) kappa_h) S_l^-1``, whose norm is of
order ``2 l h + |kappa_h|``. B contains no viscosity, so the derivative with respect
to ``log eta`` never passes through the exponential: it enters only through the
diagonal T.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import stokesjx  # noqa: F401  (sets jax_enable_x64)

# --------------------------------------------------------------------------
# Matrices.
# --------------------------------------------------------------------------


def a_matrix(l, eta, rho, self_gravity: bool = True):
    """A0 (kappa = 0) for degree ``l``, viscosity ``eta`` and density ``rho``, both
    relative to their scales. All arguments broadcast; the result has shape
    ``broadcast(l, eta, rho).shape + (6, 6)``. Rows as ``aconstruct``
    (driver.f:369-398), eq. 4.1."""
    l, eta, rho = jnp.broadcast_arrays(jnp.asarray(l, jnp.float64), jnp.asarray(eta, jnp.float64),
                                       jnp.asarray(rho, jnp.float64))
    L = l * (l + 1)
    z = jnp.zeros_like(l)
    one = jnp.ones_like(l)
    g = rho if self_gravity else z
    rows = [
        [-2 * one, L, z, z, z, z],
        [-one, one, z, 1 / eta, z, z],
        [12 * eta, -6 * L * eta, one, L, z, -g],
        [-6 * eta, 2 * (2 * L - 1) * eta, -one, -2 * one, -g, z],
        [z, z, z, z, one, one],
        [z, z, z, z, L, z],
    ]
    return jnp.stack([jnp.stack(r, axis=-1) for r in rows], axis=-2)


def k_matrix(eta):
    """``K = dA/dkappa`` (eq. 4.3): K(1,1) = -1, K(3,1) = 4 eta, K(4,1) = -2 eta."""
    eta = jnp.asarray(eta, jnp.float64)
    k = jnp.zeros(eta.shape + (6, 6))
    k = k.at[..., 0, 0].set(-1.0)
    k = k.at[..., 2, 0].set(4 * eta)
    k = k.at[..., 3, 0].set(-2 * eta)
    return k


def degree_scaling(l):
    """Diagonal of ``S_l = diag(1, l, 1/(2l), 1/2, 1/2, 1/(2l))`` (shape ``l.shape + (6,)``)."""
    l = jnp.asarray(l, jnp.float64)
    one = jnp.ones_like(l)
    return jnp.stack([one, l, 1 / (2 * l), one / 2, one / 2, 1 / (2 * l)], axis=-1)


def viscosity_scaling(eta):
    """Diagonal of ``D_eta = diag(1, 1, eta, eta, eta, eta)``."""
    eta = jnp.asarray(eta, jnp.float64)
    one = jnp.ones_like(eta)
    return jnp.stack([one, one, eta, eta, eta, eta], axis=-1)


def balanced_argument(l, rho, h, kappa_h, self_gravity: bool = True):
    """``B = S_l (A(1) h + K(1) kappa_h) S_l^-1``: the exponent without viscosity and
    with the degree balanced. Broadcasts over ``l``, ``rho``, ``h``, ``kappa_h``."""
    l, rho, h, kappa_h = jnp.broadcast_arrays(*(jnp.asarray(v, jnp.float64) for v in (l, rho, h, kappa_h)))
    M = a_matrix(l, 1.0, rho, self_gravity) * h[..., None, None] + k_matrix(jnp.ones_like(l)) * kappa_h[..., None, None]
    s = degree_scaling(l)
    return s[..., :, None] * M / s[..., None, :]


# --------------------------------------------------------------------------
# Matrix exponential candidates.
# --------------------------------------------------------------------------


def expm_jax(M):
    """(a) ``jax.scipy.linalg.expm`` (Pade 13 with up to 16 squarings, chosen at run time)."""
    return jax.scipy.linalg.expm(M)


def expm_taylor(M, terms: int = 12, squarings: int = 0):
    """(b) Truncated Taylor series of degree ``terms`` with a static number of
    squarings: ``(sum_k (M / 2^s)^k / k!)^(2^s)``. Horner form, ``terms`` matrix
    products plus ``squarings``. Exact derivatives of a polynomial, so reverse mode
    is as accurate as the forward value."""
    X = M / (2.0 ** squarings)
    eye = jnp.eye(M.shape[-1], dtype=M.dtype)
    E = eye
    for k in range(terms, 0, -1):
        E = eye + (X @ E) / k
    for _ in range(squarings):
        E = E @ E
    return E


def _pade_coefficients(p: int):
    """Coefficients ``c_k``, ``k = 0..p``, of the numerator of the diagonal Pade
    approximant ``(p, p)`` of ``exp(x)``:
    ``c_k = (2p - k)! p! / ((2p)! k! (p - k)!)``. The denominator has the
    coefficients ``(-1)^k c_k``, so the approximant is ``N(x) / N(-x)``."""
    return [math.factorial(2 * p - k) * math.factorial(p)
            / (math.factorial(2 * p) * math.factorial(k) * math.factorial(p - k)) for k in range(p + 1)]


def matmul_batch_last(a, b):
    """Matrix product of two stacks of square matrices stored with the matrix axes
    first and the batch axes last: ``a``, ``b`` of shape ``(n, n, *batch)``.

    The product ``C[i, j, ...] = sum_k a[i, k, ...] b[k, j, ...]`` is written as one
    broadcast multiply and one sum over ``k``. With 6x6 matrices and tens of
    thousands of batch entries, this layout puts the long batch axes in the
    innermost (contiguous) position, so XLA emits one vectorised elementwise loop
    in place of many tiny 6x6 ``dot`` calls. Measured on reunion: the stand-in with
    the Pade (6, 6) pair is 12 to 19 % faster (forward and gradient, at both
    benchmark sizes) than with ``a @ b`` on ``(*batch, 6, 6)`` (the rows
    ``pade6_gs`` and ``pade6_gs_batch_first`` of the run 2026-11-11T02:33:44 in
    NOTES/benchmarks.md). The derivative of a multiply and a sum is again a
    multiply and a sum, so the reverse pass keeps the same layout."""
    return jnp.sum(a[:, :, None] * b[None, :, :], axis=1)


def pade_product_count(order: int) -> int:
    """Number of 6x6 matrix products that ``_pade_even_odd`` spends on ``V`` and
    ``U`` for the given order (the squarings and the solve not included).

    Orders up to 12: the even powers ``X^2 .. X^(2 floor(order / 2))`` (one product
    each) and one product for ``U``, so 4 for orders 6 and 7 and 5 for orders 8 and
    9. Order 13: the scheme of Higham (2005), 6 products."""
    if order == 13:
        return 6
    if order > 12:
        raise ValueError(order)
    return order // 2 + 1


def _pade_even_odd(X, order: int, layout: str):
    """Even and odd parts ``V = sum_(k even) c_k X^k`` and
    ``U = sum_(k odd) c_k X^k`` of the Pade numerator ``N(X) = V + U``.

    Only the even powers ``X^2, X^4, ...`` are formed (one product each), and
    ``U = X (c_1 I + c_3 X^2 + c_5 X^4 + ...)`` costs one more product. For orders
    6 and 7 this is four products (the plain power sequence needs six), for order
    9 five. Order 13 uses the scheme of Higham (2005, eq. 2.3 ff.), which forms
    ``X^2, X^4, X^6`` only and evaluates the terms of degree 8 and higher as
    ``X^6 (c_12 X^6 + c_10 X^4 + c_8 X^2)`` and
    ``X^6 (c_13 X^6 + c_11 X^4 + c_9 X^2)``: six products. Since
    ``N(-X) = V - U``, the denominator needs no further product.
    ``pade_product_count`` gives the count for each order.

    ``layout`` is ``"batch_first"`` (``X`` of shape ``(*batch, n, n)``, products
    by ``@``) or ``"batch_last"`` (``X`` of shape ``(n, n, *batch)``, products by
    ``matmul_batch_last``)."""
    c = _pade_coefficients(order)
    if layout == "batch_first":
        mm = jnp.matmul
        eye = jnp.eye(X.shape[-1], dtype=X.dtype)
    elif layout == "batch_last":
        mm = matmul_batch_last
        eye = jnp.eye(X.shape[0], dtype=X.dtype).reshape(X.shape[:2] + (1,) * (X.ndim - 2))
    else:
        raise ValueError(layout)
    X2 = mm(X, X)
    if order == 13:
        # Higham's scheme: three even powers and two products with X^6.
        X4 = mm(X2, X2)
        X6 = mm(X4, X2)
        V = (mm(X6, c[12] * X6 + c[10] * X4 + c[8] * X2)
             + c[6] * X6 + c[4] * X4 + c[2] * X2 + c[0] * eye)
        odd_poly = (mm(X6, c[13] * X6 + c[11] * X4 + c[9] * X2)
                    + c[7] * X6 + c[5] * X4 + c[3] * X2 + c[1] * eye)
        return V, mm(X, odd_poly)
    even = [eye, X2]                     # X^0, X^2, X^4, ...
    while 2 * len(even) <= order:
        even.append(mm(even[-1], X2))
    V = sum(c[2 * j] * P for j, P in enumerate(even) if 2 * j <= order)
    odd_poly = sum(c[2 * j + 1] * P for j, P in enumerate(even) if 2 * j + 1 <= order)
    U = mm(X, odd_poly)
    return V, U


# --------------------------------------------------------------------------
# Solvers for the Pade denominators, on batch-last arrays.
# --------------------------------------------------------------------------


def solve_elimination_plain(A, B):
    """Solve ``A X = B`` for a stack of small systems by Gauss-Jordan elimination
    without pivoting, on batch-last arrays.

    Parameters
    ----------
    A
        Matrices of shape ``(n, n, *batch)`` (matrix axes first).
    B
        Right-hand sides of shape ``(n, m, *batch)``.

    Returns
    -------
    ``X = A^-1 B`` of shape ``(n, m, *batch)``.

    Each of the ``n`` steps divides the pivot row by the pivot and removes the pivot
    column from every other row by one rank-one update on the whole batch. Columns
    that are already eliminated are dropped from ``A`` (static slices), so step
    ``k`` updates an ``n x (n - k - 1)`` block of ``A`` and the ``n x m`` block of
    ``B``. The batch axes stay innermost, so every update is one vectorised
    elementwise loop over tens of thousands of systems. On reunion this costs 0.53
    ms for the 13 056 Pade denominators of the benchmark size (51, 128), against
    3.4 ms for ``jnp.linalg.solve``, which runs one small LAPACK call per matrix.

    Numerical caveat: there is no pivoting. The function is only for matrices that
    are strictly diagonally dominant by columns. Then every pivot is nonzero, the
    property is kept by every elimination step (the Schur complement of a column
    diagonally dominant matrix is column diagonally dominant), and elimination
    without pivoting has a growth factor of at most 2 (Wilkinson; Higham, Accuracy
    and Stability of Numerical Algorithms, 2nd ed., 2002, chapter 9). The Pade
    denominators have this property when ``pade_squarings`` chose the squaring
    count (``pade_denominator_margin``). For other matrices use
    ``solve_lapack_batch_last``.

    Reverse mode through this function stores every elimination step of ``A``.
    ``solve_elimination`` has the same value and a derivative rule that is cheaper.
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
    """``X = A^-1 B`` by ``solve_elimination_plain`` (same arguments, same caveat:
    ``A`` must be strictly diagonally dominant by columns), with the derivative rule
    of a linear solve.

    From ``A X = B``: ``dA X + A dX = dB``, so ``dX = A^-1 (dB - dA X)``. The tangent
    is one more solve with the same ``A``. In reverse mode JAX transposes this
    linear map, which is again an elimination with the steps of ``A`` that the
    forward pass already computed. Plain reverse mode through the elimination
    instead differentiates every step of the elimination of ``A``: at the benchmark
    size (51, 128) the gradient of the stand-in with all controls takes 26.7 ms and
    0.66 s compile with plain AD, against 24.2 ms and 0.48 s with this rule (rows
    ``pade_ge_plain_gs`` and ``pade_ge_gs`` of the run 2026-11-11T03:20:48 in
    NOTES/benchmarks.md). ``tests/test_primitives_ad.py`` checks the rule against
    plain AD by the jvp/vjp dot-product test (CONVENTIONS 6)."""
    return solve_elimination_plain(A, B)


@solve_elimination.defjvp
def _solve_elimination_jvp(primals, tangents):
    """Tangent of ``X = A^-1 B``: ``dX = A^-1 (dB - dA X)`` (see ``solve_elimination``)."""
    A, B = primals
    dA, dB = tangents
    X = solve_elimination(A, B)
    return X, solve_elimination(A, dB - matmul_batch_last(dA, X))


def solve_lapack_batch_last(A, B):
    """``X = A^-1 B`` on batch-last arrays by ``jnp.linalg.solve`` (LU with partial
    pivoting, one LAPACK call per matrix). ``A``: ``(n, n, *batch)``, ``B``:
    ``(n, m, *batch)``. Kept as the comparison for ``solve_elimination`` and for
    matrices that are not diagonally dominant."""
    return _to_batch_last(jnp.linalg.solve(_to_batch_first(A), _to_batch_first(B)))


# Solvers of the Pade denominators by name. "elimination" is the selected one.
PADE_SOLVERS = {"elimination": solve_elimination, "elimination_plain": solve_elimination_plain,
                "lapack": solve_lapack_batch_last}


def _to_batch_first(M):
    """``(n, n, *batch)`` to ``(*batch, n, n)``."""
    return jnp.moveaxis(M, (0, 1), (-2, -1))


def _to_batch_last(M):
    """``(*batch, n, n)`` to ``(n, n, *batch)``."""
    return jnp.moveaxis(M, (-2, -1), (0, 1))


def _solve_denominator(D, R, layout: str, solver: str):
    """``D^-1 R`` for the Pade denominator ``D`` and numerator ``R`` in the given
    layout (both arrays of that layout). ``solver="lapack"`` on batch-first arrays
    calls ``jnp.linalg.solve`` directly; every other combination runs on
    batch-last arrays (``PADE_SOLVERS``) and transposes where the layout needs it."""
    if solver not in PADE_SOLVERS:
        raise ValueError(solver)
    if layout == "batch_first":
        if solver == "lapack":
            return jnp.linalg.solve(D, R)
        return _to_batch_first(PADE_SOLVERS[solver](_to_batch_last(D), _to_batch_last(R)))
    if layout == "batch_last":
        return PADE_SOLVERS[solver](D, R)
    raise ValueError(layout)


def expm_pade(M, order: int = 6, squarings: int = 0, layout: str = "batch_first", solver: str = "elimination"):
    """(c) Diagonal Pade approximant ``(order, order)`` of ``exp(M)`` with a static
    number of squarings.

    With ``X = M / 2^s``, ``N(X) = V + U`` (``V`` even, ``U`` odd part, see
    ``_pade_even_odd``) and ``N(-X) = V - U``, the approximant is
    ``r(X) = (V - U)^-1 (V + U)``, and the result is ``r(X)^(2^s)``. ``M`` has shape
    ``(..., n, n)`` in both layouts; with ``layout="batch_last"`` the products run
    on a transposed copy (``matmul_batch_last``) and the result is transposed back.

    ``solver`` (``PADE_SOLVERS``): ``"elimination"`` (Gauss-Jordan without pivoting
    with the derivative rule, the default), ``"elimination_plain"`` (the same,
    plain AD) or ``"lapack"``. Elimination without pivoting needs the denominator
    ``N(-X)`` to be strictly diagonally dominant by columns, which holds when the
    squaring count comes from ``pade_squarings`` with the elimination solver (see
    ``pade_denominator_margin``). The function cannot check this under ``jit``.

    Accuracy: the relative backward error of ``r`` is below
    ``|log(exp(-x) r(x))| / x`` at ``x = ||X||_1`` (Higham 2005); see
    ``PADE_THETA`` for the thresholds."""
    X = M / (2.0 ** squarings)
    if layout == "batch_first":
        V, U = _pade_even_odd(X, order, "batch_first")
        E = _solve_denominator(V - U, V + U, "batch_first", solver)
        for _ in range(squarings):
            E = E @ E
        return E
    V, U = _pade_even_odd(_to_batch_last(X), order, "batch_last")
    E = _solve_denominator(V - U, V + U, "batch_last", solver)
    for _ in range(squarings):
        E = matmul_batch_last(E, E)
    return _to_batch_first(E)


def expm_pade_pair(M, order: int = 7, squarings: int = 0, layout: str = "batch_last",
                   solver: str = "elimination"):
    """The Pade approximant of ``exp(M)`` and of ``exp(-M)`` from one set of powers.

    The diagonal Pade approximant satisfies ``r(-X) = r(X)^-1`` exactly:
    ``r(X) = (V - U)^-1 (V + U)`` and ``r(-X) = (V + U)^-1 (V - U)``, and the two
    factors of each commute because they are polynomials in ``X``. Squaring
    commutes with the inverse, so ``r(-X)^(2^s) = (r(X)^(2^s))^-1``. The forward
    step and its inverse (the step of the downward sweep) therefore cost one set of
    powers, two solves (stacked into one batched call) and two sets of squarings,
    in place of a second exponential. The inverse agrees with
    ``expm_pade(-M)`` to 4e-15 in balanced coordinates (rounding only).

    Parameters
    ----------
    M
        Exponents of shape ``(*batch, n, n)``.
    order, squarings
        Static Pade order and squaring count. Take both from ``pade_plan`` (or the
        count from ``pade_squarings``) with the same ``solver``: the norm bound
        that sets them is the only guarantee that the scaled exponent is inside
        the accuracy threshold and, for the elimination solver, that both
        denominators are diagonally dominant.
    layout
        ``"batch_last"`` (default): the products by ``matmul_batch_last`` on
        ``(n, n, *batch)`` copies; ``"batch_first"``: by ``@``.
    solver
        ``"elimination"`` (default; ``solve_elimination``, Gauss-Jordan without
        pivoting with the derivative rule), ``"elimination_plain"`` or ``"lapack"``.

    Both denominators ``V - U = N(-X)`` and ``V + U = N(X)`` are strictly
    diagonally dominant by columns when ``N_m(||X||_1) < 2`` (``N_m`` the scalar
    numerator, positive coefficients; ``pade_denominator_margin``). The two
    systems of every batch entry are stacked into one solve.

    Returns ``(E, E_inv)``, both of the shape of ``M``."""
    X = M / (2.0 ** squarings)
    if layout == "batch_first":
        V, U = _pade_even_odd(X, order, "batch_first")
        E = _solve_denominator(jnp.stack([V - U, V + U]), jnp.stack([V + U, V - U]), "batch_first", solver)
        for _ in range(squarings):
            E = E @ E
        return E[0], E[1]
    if layout != "batch_last":
        raise ValueError(layout)
    V, U = _pade_even_odd(_to_batch_last(X), order, "batch_last")
    # Stack the two systems on a batch axis after the matrix axes: (n, n, 2, *batch).
    E = _solve_denominator(jnp.stack([V - U, V + U], axis=2), jnp.stack([V + U, V - U], axis=2),
                           "batch_last", solver)
    for _ in range(squarings):
        E = matmul_batch_last(E, E)
    # Split the pair and return batch-first arrays.
    return _to_batch_first(E[:, :, 0]), _to_batch_first(E[:, :, 1])


def expm_closed_incompressible(B, l, h):
    """(d) Closed form for the incompressible matrix. ``B`` is ``A h`` in any
    similar form (here the balanced one). Its eigenvalues are ``(l+1) h`` (double),
    ``(l-1) h``, ``-l h`` (double), ``(-l-2) h``, independent of ``eta`` and ``rho``,
    and A is diagonalisable, so ``expm(B) = sum_k exp(lam_k) prod_(i != k) (B - lam_i)
    / (lam_k - lam_i)`` over the four distinct values (Lagrange-Sylvester).

    Numerical caveat: the denominators ``lam_k - lam_i`` are proportional to ``h``.
    At ``h = 0`` (the zero-width interval of every jump pair) the formula is
    ``0 / 0`` and the value is NaN, in the forward pass and in reverse mode; for
    small ``h`` it cancels digits. This is one of the reasons why the candidate is
    not selected (benchmarks/bench_primitives.py records it as "kappa_h = 0 only,
    NaN at h = 0")."""
    l = jnp.asarray(l, jnp.float64)[..., None, None]
    h = jnp.asarray(h, jnp.float64)[..., None, None]
    lam = [(l + 1) * h, (l - 1) * h, -l * h, (-l - 2) * h]
    eye = jnp.eye(6)
    E = jnp.zeros_like(B)
    for k in range(4):
        P = jnp.broadcast_to(eye, B.shape)
        for i in range(4):
            if i != k:
                P = P @ (B - lam[i] * eye) / (lam[k] - lam[i])
        E = E + jnp.exp(lam[k]) * P
    return E


def taylor_squarings(norm_bound: float, terms: int = 12, tol: float = 1e-17) -> int:
    """Static squaring count for the Taylor candidate: the smallest ``s`` with
    ``(x / 2^s)^(terms+1) / (terms+1)! * exp(x / 2^s) < tol``, ``x = norm_bound``
    (the 1-norm bound of the exponent over the whole batch)."""
    # Start from the squaring count that brings the norm to one, so that exp(x)
    # never overflows, and add squarings until the remainder bound holds.
    s = max(0, math.ceil(math.log2(max(norm_bound, 1e-300))))
    while True:
        x = norm_bound / 2.0 ** s
        if x ** (terms + 1) / math.factorial(terms + 1) * math.exp(x) < tol:
            break
        s += 1
    # Remove squarings while the bound still holds.
    while s > 0:
        x = norm_bound / 2.0 ** (s - 1)
        if x ** (terms + 1) / math.factorial(terms + 1) * math.exp(x) < tol:
            s -= 1
        else:
            break
    return s


# Scaling thresholds of the Pade candidates: the squaring count brings the 1-norm of
# the scaled exponent below theta. The relative backward error of the diagonal
# approximant r_m at x = ||X||_1 is bounded by |log(exp(-x) r_m(x))| / x (Higham
# 2005, SIAM J. Matrix Anal. Appl. 26, 1179, section 2). Higham's theta values make
# that bound equal to the unit roundoff 2^-53: theta_6 = 0.541, theta_7 = 0.950,
# theta_9 = 2.098, theta_13 = 5.37 (the bound is 1.1e-16 at 0.950 and 2.098,
# computed at 50 digits). For order 6 this module uses its own larger threshold
# 0.783, where the bound is 9.3e-15 (computed at 50 digits). That is 100 times
# below the eligibility limit of 1e-13 of benchmarks/bench_primitives.py, and the
# error measured against mp.expm at the worst case of the threshold (scaled norm
# 0.999 x 0.783, lmax up to 200) is 3.3e-15. Order 7 costs the same four products
# as order 6 (``pade_product_count``) and has the larger threshold 0.950 at the
# unit roundoff, so ``pade_plan`` never selects order 6; order 6 stays for the
# comparison with the earlier budget stand-in.
PADE_THETA = {6: 0.783, 7: 0.950, 9: 2.098, 13: 5.37}

# Limit of the scalar numerator N_m(x) for the elimination solver: both Pade
# denominators N(+-X) are strictly diagonally dominant by columns when N_m(x) < 2
# at x = ||X||_1 (``pade_denominator_margin``). The scaled norm where N_m(x) = 2 is
# 1.433 (order 6), 1.425 (7), 1.416 (9) and 1.406 (13), so the condition is
# weaker than the threshold for orders 6 and 7 (N_6(0.783) = 1.469,
# N_7(0.950) = 1.594) and is the binding one for orders 9 and 13.
PADE_DENOMINATOR_LIMIT = 2.0


def pade_numerator(order: int, x: float) -> float:
    """Scalar Pade numerator ``N_m(x) = sum_k c_k x^k`` (``_pade_coefficients``).
    All ``c_k`` are positive, so for a matrix ``X`` with ``||X||_1 <= x`` every
    column of ``N(+-X) - I`` has a 1-norm of at most ``N_m(x) - 1``."""
    return sum(ck * x ** k for k, ck in enumerate(_pade_coefficients(order)))


def pade_denominator_margin(order: int, x: float) -> float:
    """Lower bound of the column diagonal dominance of both Pade denominators.

    For ``D = N(X)`` or ``D = N(-X)`` with ``||X||_1 <= x``: ``D = I + E`` with
    ``||E||_1 <= N_m(x) - 1``, so in every column ``j``
    ``|D_jj| - sum_(i != j) |D_ij| >= 1 - ||E e_j||_1 >= 2 - N_m(x)``. A positive
    return value means that ``D`` is strictly diagonally dominant by columns and
    that elimination without pivoting is stable for it (``solve_elimination``).
    It also bounds ``||D^-1||_1 <= 1 / (2 - N_m(x))``."""
    return PADE_DENOMINATOR_LIMIT - pade_numerator(order, x)


def pade_squarings(norm_bound: float, order: int = 6, solver: str = "elimination") -> int:
    """Static squaring count ``s`` for the Pade candidate of the given order: the
    smallest ``s >= 0`` with ``x = norm_bound / 2^s <= PADE_THETA[order]`` and, for
    the elimination solvers, ``N_m(x) < 2`` (``pade_denominator_margin`` positive,
    checked here in Python because nothing can check it under ``jit``).

    ``norm_bound`` must bound the 1-norm of the exponent for every degree and
    interval of the batch, including the upper limits of the traced inputs (the
    movable interval widths, ``rho0 / rho_s``, ``kappa_h``): the count is static and
    nothing can check it under ``jit``. A scaled norm of 1.6 in place of 0.783
    gives an error of about 1e-10 without a warning, and a scaled norm with
    ``N_m(x) >= 2`` removes the guarantee that elimination without pivoting is
    stable."""
    if solver not in PADE_SOLVERS:
        raise ValueError(solver)
    theta = PADE_THETA[order]
    s = max(0, math.ceil(math.log2(max(norm_bound, 1e-300) / theta)))
    if solver != "lapack":
        # Each squaring halves x; N_m decreases with x, so the loop ends.
        while pade_denominator_margin(order, norm_bound / 2.0 ** s) <= 0.0:
            s += 1
        if not pade_denominator_margin(order, norm_bound / 2.0 ** s) > 0.0:
            raise ValueError(f"Pade ({order}, {order}) denominators not diagonally dominant at "
                             f"x = {norm_bound / 2.0 ** s}")
    return s


def pade_plan(norm_bound: float, solver: str = "elimination", orders=(6, 7, 9, 13)) -> dict:
    """Static Pade order and squaring count of the step-and-inverse pair
    (``expm_pade_pair``) for an exponent with 1-norm at most ``norm_bound``.

    For every order of ``orders`` the squaring count comes from
    ``pade_squarings`` (accuracy threshold and, for the elimination solvers, the
    dominance condition ``N_m(x) < 2``). The cost of the pair in 6x6 products is
    ``pade_product_count(order) + 2 s`` (both matrices of the pair are squared);
    the solve costs the same for every order. The plan takes the order with the
    fewest products and, between orders of equal cost, the higher order (the
    smaller backward error). Examples: the bounds 1.86 and 1.82 of the two
    benchmark sizes give order 7 with one squaring (6 products; order 6 needs two
    squarings, 8 products; order 9 needs one squaring because ``N_9(1.86) > 2``,
    7 products).

    Returns ``dict(order, squarings, scaled_norm, numerator, margin, products)``;
    ``margin`` is ``pade_denominator_margin`` at the scaled norm."""
    best = None
    for m in orders:
        s = pade_squarings(norm_bound, m, solver)
        cost = pade_product_count(m) + 2 * s
        if best is None or cost < best[0] or (cost == best[0] and m > best[1]):
            best = (cost, m, s)
    cost, m, s = best
    x = norm_bound / 2.0 ** s
    return dict(order=m, squarings=s, scaled_norm=x, numerator=pade_numerator(m, x),
                margin=pade_denominator_margin(m, x), products=cost)


def balanced_norm_bound(lmax: int, h_max: float, kappa_h_max: float = 0.2, rho_max: float = 1.3) -> float:
    """Upper bound of the 1-norm of the balanced exponent ``B`` over the degrees
    ``1..lmax`` and intervals up to ``h_max``.

    The largest column sum of ``S_l A(1) S_l^-1`` is ``(2l+2) + l + 1 + 6/l``
    (column 2) or ``2l + l + 1 + 2`` (column 4), below ``3 l + 10`` for every
    ``l >= 1``; the density adds at most ``rho / rho_s`` (checked numerically for
    ``l = 1..300``, largest ratio of the column sum to the bound 0.997). The
    balanced K has the column ``(-1, 0, 2/l, -1, 0, 0)``, whose sum ``2 + 2/l`` is
    largest at ``l = 1``, so K adds at most ``4 |kappa_h|``.

    The defaults ``kappa_h_max = 0.2`` and ``rho_max = 1.3`` (``rho_s = 4500``) are
    for the benchmark only. The solver computes both from the profile and the
    movable ranges when its config is built (see ``pade_squarings``).

    This form adds the largest width term and the largest ``kappa h`` term, which
    never occur in the same interval (a jump pair has ``h = 0``). At the benchmark
    inputs the true largest norm is 1.03 against the bound 1.86.
    ``balanced_norm_bound_intervals`` takes the maximum over the intervals of the
    per-interval bound and is the form for the solver."""
    return (3 * lmax + 10 + rho_max) * h_max + 4 * kappa_h_max


def balanced_norm_bound_intervals(lmax: int, h_max, kappa_h_max, rho_max) -> float:
    """Upper bound of the 1-norm of the balanced exponent ``B`` over the degrees
    ``1..lmax``, taken interval by interval:
    ``max_i ((3 lmax + 10 + rho_max_i) h_max_i + 4 |kappa_h|_max_i)``.

    Parameters
    ----------
    lmax
        Largest degree (static).
    h_max, kappa_h_max, rho_max
        Arrays of one entry per interval: the static upper limit of the width in
        ``ln r`` (the largest width that any position of a movable boundary inside
        its range allows), of ``|kappa h|`` and of ``rho0 / rho_s`` in that
        interval. Plain numpy or Python numbers: the bound is computed when the
        config is built, never under ``jit``.

    The per-interval terms are those of ``balanced_norm_bound``. The bound is never
    larger than that of ``balanced_norm_bound`` with the largest value of each
    input, and it is smaller when the widest interval and the largest density
    step are in different intervals (for example a jump pair, ``h = 0``)."""
    h = np.asarray(h_max, dtype=float)
    kh = np.abs(np.asarray(kappa_h_max, dtype=float))
    rho = np.asarray(rho_max, dtype=float)
    return float(np.max((3 * lmax + 10 + rho) * h + 4 * kh))


def parse_pade_method(method: str):
    """Split a Pade method name ``"pade<m>[_bl][_ge]"`` (``step_balanced``) into
    ``(order, layout, solver)``."""
    parts = method.split("_")
    if not parts[0].startswith("pade") or not set(parts[1:]) <= {"bl", "ge"}:
        raise ValueError(method)
    order = int(parts[0][4:])
    layout = "batch_last" if "bl" in parts[1:] else "batch_first"
    solver = "elimination" if "ge" in parts[1:] else "lapack"
    return order, layout, solver


def step_balanced(l, eta, rho, h, kappa_h, *, method: str = "taylor", terms: int = 12,
                  squarings: int = 2, self_gravity: bool = True):
    """(e) The interval step through both similarities:
    ``T expm(B) T^-1``, ``T = D_eta S_l^-1``.

    Parameters
    ----------
    l, eta, rho, h, kappa_h
        Degree, viscosity and density of the interval (relative to ``eta_s`` and
        ``rho_s``), interval width in ``ln r`` and ``kappa h = ln(rho0_i /
        rho0_(i-1))``. All broadcast; ``h = 0`` is allowed (zero-width interval).
    method
        Exponential of the balanced exponent: "taylor" (Taylor-12), "jax"
        (``jax.scipy.linalg.expm``), "closed" (incompressible only, NaN at
        ``h = 0``), or a Pade name ``"pade<m>[_bl][_ge]"``: order ``m``, the
        products on batch-last arrays (``matmul_batch_last``) with ``_bl`` and on
        batch-first arrays (``@``) without it, and the denominator solved by
        ``solve_elimination`` with ``_ge`` and by LAPACK without it. So "pade6",
        "pade6_bl" and "pade13" are the LAPACK forms and "pade7_bl_ge" is the
        selected form of the stand-in.
    terms, squarings
        Static series length (Taylor) and squaring count (Taylor and Pade). For a
        ``_ge`` method the count must come from ``pade_squarings`` with the
        elimination solver.

    Returns
    -------
    The step, shape ``broadcast(l, eta, rho, h, kappa_h).shape + (6, 6)``. The
    viscosity enters only through the diagonal ``T``; ``rho``, ``h`` and ``kappa_h``
    enter through the exponential."""
    B = balanced_argument(l, rho, h, kappa_h, self_gravity)
    if method == "taylor":
        E = expm_taylor(B, terms, squarings)
    elif method.startswith("pade"):
        order, layout, solver = parse_pade_method(method)
        E = expm_pade(B, order, squarings, layout=layout, solver=solver)
    elif method == "jax":
        E = expm_jax(B)
    elif method == "closed":
        E = expm_closed_incompressible(B, l, h)
    else:
        raise ValueError(method)
    t = viscosity_scaling(eta) / degree_scaling(l)
    return t[..., :, None] * E / t[..., None, :]


def step_unbalanced(l, eta, rho, h, kappa_h, *, method: str = "jax", terms: int = 12, squarings: int = 2):
    """The interval step from the raw matrix ``A(eta) h + K(eta) kappa_h``."""
    h = jnp.asarray(h, jnp.float64)[..., None, None]
    kh = jnp.asarray(kappa_h, jnp.float64)[..., None, None]
    M = a_matrix(l, eta, rho) * h + k_matrix(jnp.broadcast_to(jnp.asarray(eta, jnp.float64),
                                                           jnp.shape(a_matrix(l, eta, rho))[:-2])) * kh
    if method == "jax":
        return expm_jax(M)
    if method == "taylor":
        return expm_taylor(M, terms, squarings)
    raise ValueError(method)


def pair_step(eta, t):
    """Step across a jump pair under TALA: ``expm(K t) = I + K (1 - exp(-t))``,
    ``t = ln(rho0_+ / rho0_-)`` (F8). Uses ``K^2 = -K``."""
    t = jnp.asarray(t, jnp.float64)
    return jnp.eye(6) + k_matrix(eta) * (-jnp.expm1(-t))[..., None, None]


# --------------------------------------------------------------------------
# Orthonormalisation of three vectors, and the sweep.
# --------------------------------------------------------------------------


def orthonormalise_qr(V):
    """Orthonormal basis of the span of the three columns of ``V`` (..., 6, 3) by
    LAPACK QR, and ``log |diag R|``: the log of the scale removed from each
    direction."""
    Q, R = jnp.linalg.qr(V)
    return Q, jnp.log(jnp.abs(jnp.diagonal(R, axis1=-2, axis2=-1)))


def orthonormalise_gs(V):
    """The same by classical Gram-Schmidt with one re-orthogonalisation (CGS2),
    written out for three columns, so that XLA fuses it into the scan body. The
    columns stay in their order and the signs of ``diag R`` are positive, so ``Q``
    equals the QR factor up to the sign of each column."""
    def col(k):
        return V[..., :, k]

    def proj(q, v):
        return jnp.sum(q * v, axis=-1, keepdims=True) * q

    v1 = col(0)
    n1 = jnp.linalg.norm(v1, axis=-1, keepdims=True)
    q1 = v1 / n1
    v2 = col(1) - proj(q1, col(1))
    v2 = v2 - proj(q1, v2)
    n2 = jnp.linalg.norm(v2, axis=-1, keepdims=True)
    q2 = v2 / n2
    v3 = col(2) - proj(q1, col(2)) - proj(q2, col(2))
    v3 = v3 - proj(q1, v3) - proj(q2, v3)
    n3 = jnp.linalg.norm(v3, axis=-1, keepdims=True)
    q3 = v3 / n3
    Q = jnp.stack([q1, q2, q3], axis=-1)
    logs = jnp.log(jnp.concatenate([n1, n2, n3], axis=-1))
    return Q, logs


ORTHO = {"qr": orthonormalise_qr, "gs": orthonormalise_gs}


def sweep(steps, start, ortho: str = "qr"):
    """Propagate a (L, 6, 3) basis through ``steps`` (N, L, 6, 6) with ``lax.scan``,
    re-orthonormalising after every step. Returns the basis at every node (N, L, 6, 3)
    and the accumulated log scales (L, 3). The body is one batched matrix product,
    one orthonormalisation and one add."""
    f = ORTHO[ortho]

    def body(carry, P):
        Y, logs = carry
        Q, lg = f(P @ Y)
        return (Q, logs + lg), Q

    Q0, lg0 = f(start)
    (Q, logs), Qs = lax.scan(body, (Q0, lg0), steps)
    return Qs, logs


# --------------------------------------------------------------------------
# Composed stand-in of the solver (plan section 8, item 4).
# --------------------------------------------------------------------------


def make_standin(lmax: int, n_intervals: int, h_max: float, *, method: str = "pade",
                 ortho: str = "gs", terms: int = 12, layout: str = "batch_last",
                 solver: str = "elimination", jit: bool = True):
    """A jitted stand-in of the full forward map and its gradient: a cost model.

    The stand-in has the array shapes and the sequence of operations that chunk 3
    plans, so that its run and compile times set the speed budgets of
    NOTES/CONVENTIONS.md section 7. It is not a template for the indices or the
    coordinates of the solver: the 6x6 system pairs the upward basis after interval
    ``i`` with the downward basis before interval ``i``, which live in the
    coordinates of different intervals, the boundary bases are fixed placeholders,
    and the functional is a random weighted sum. Parts of the solver that it does
    not have: the cumulative product of the triangular factors of the downward sweep
    (PLAN 3.2 step 4; ``orthonormalise_gs`` returns only the log of the scales),
    the gravity integration, the boundary rows, degree 1, the output formulas and
    the control map.

    Structure:

    1. The interval steps for all degrees and intervals, and their inverses for the
       downward sweep, from one batched exponential outside any scan. For the Pade
       methods the inverse comes from the same powers (``expm_pade_pair``,
       ``r(-X) = r(X)^-1``) and both denominators are solved in one stacked call
       (by default ``solve_elimination``: no LAPACK call in this stage); for
       Taylor-12, whose truncated series has no such identity, it is one batched
       ``jnp.linalg.inv`` of the steps. The inverse never costs a second
       exponential.
    2. One upward and one downward ``lax.scan`` with an ``(L, 6, 3)`` carry.
    3. One batched 6x6 solve for every degree and node.
    4. A scalar functional of all solutions and of the log scales.

    Parameters
    ----------
    lmax, n_intervals, h_max
        Degrees ``1..lmax``, ``N = n_intervals`` intervals, the largest interval
        width in ``ln r`` (sets the static squaring count with
        ``balanced_norm_bound``).
    method
        "pade" (default, the budget candidate: order and squaring count from
        ``pade_plan``, order 7 with one squaring at both benchmark sizes),
        "pade6" (Pade (6, 6) with the count of ``pade_squarings``; with
        ``solver="lapack"`` this is the budget stand-in of the run
        2026-11-11T02:33:44) or "taylor" (Taylor-12).
    ortho
        "gs" (Gram-Schmidt, default) or "qr" (LAPACK).
    layout
        "batch_last" (default) or "batch_first": the layout of the matrix products
        of the Pade stage (``matmul_batch_last``). Ignored for Taylor.
    solver
        Solver of the Pade denominators (``PADE_SOLVERS``): "elimination"
        (default), "elimination_plain" or "lapack". Ignored for Taylor.
    jit
        With ``jit=False`` the forward map and its ``log_eta`` gradient are returned
        unjitted, so that a caller can differentiate with respect to other inputs
        or ``vmap`` the forward map.

    Returns
    -------
    ``(forward, gradient, example_args, info)``. Both functions take
    ``(log_eta, rho, h, kappa_h)``, each of shape ``(N,)``; the gradient is with
    respect to ``log_eta``. ``info`` records the Pade order, the squaring count,
    the dominance margin of the denominators and the options.
    """
    ls = jnp.arange(1, lmax + 1, dtype=jnp.float64)
    bound = balanced_norm_bound(lmax, h_max)
    order, margin = None, None
    if method == "taylor":
        s = taylor_squarings(bound, terms)
    elif method == "pade":
        plan = pade_plan(bound, solver)
        order, s, margin = plan["order"], plan["squarings"], plan["margin"]
    elif method == "pade6":
        order, s = 6, pade_squarings(bound, 6, solver)
        margin = pade_denominator_margin(6, bound / 2.0 ** s)
    else:
        raise ValueError(method)
    L, N = lmax, n_intervals
    cmb_start = jnp.zeros((6, 3)).at[2, 0].set(1).at[1, 1].set(1).at[4, 2].set(1).at[5, 2].set(1.0)
    top_start = jnp.zeros((6, 3)).at[1, 0].set(1).at[2, 1].set(1).at[4, 2].set(1).at[5, 2].set(-2.0)
    rng = np.random.default_rng(1)
    weights = jnp.asarray(rng.standard_normal((L, N + 1, 6)))

    def forward(log_eta, rho, h, kappa_h):
        eta = jnp.exp(log_eta)
        # Balanced exponents of all degrees and intervals: (L, N, 6, 6). They hold
        # no viscosity, so the viscosity controls never reach the exponential.
        B = balanced_argument(ls[:, None], rho[None, :], h[None, :], kappa_h[None, :])
        # Steps and inverse steps from one batched exponential.
        if method == "taylor":
            Ef = expm_taylor(B, terms, s)
            Eb = jnp.linalg.inv(Ef)
        else:
            Ef, Eb = expm_pade_pair(B, order, s, layout=layout, solver=solver)
        # Change of coordinates between neighbouring intervals: diag(1, 1, r, r, r, r)
        # with r = eta_(i-1) / eta_i (the degree scaling cancels).
        ratio = jnp.concatenate([jnp.ones(1), eta[:-1] / eta[1:]])
        cvec = viscosity_scaling(ratio)                                    # (N, 6)
        up_steps = jnp.swapaxes(Ef * cvec[None, :, None, :], 0, 1)          # (N, L, 6, 6)
        inv_c = viscosity_scaling(1 / jnp.concatenate([ratio[1:], jnp.ones(1)]))
        dn_steps = jnp.swapaxes(Eb * inv_c[None, :, None, :], 0, 1)[::-1]
        start_up = jnp.broadcast_to(cmb_start, (L, 6, 3))
        start_dn = jnp.broadcast_to(top_start, (L, 6, 3))
        Qu, lu = sweep(up_steps, start_up, ortho)                           # (N, L, 6, 3)
        Qd, ld = sweep(dn_steps, start_dn, ortho)
        Qd = Qd[::-1]
        # One 6x6 system per degree and node, all in one batched solve (the index
        # pairing is a cost model only, see the docstring).
        M = jnp.concatenate([Qu, -Qd], axis=-1)                           # (N, L, 6, 6)
        jump = jnp.zeros((N, L, 6)).at[..., 2].set(1.0).at[..., 5].set(-0.3)
        c = jnp.linalg.solve(M, jump[..., None])[..., 0]                   # (N, L, 6)
        return jnp.sum(c * jnp.swapaxes(weights[:, :N, :], 0, 1)) + jnp.sum(lu) + jnp.sum(ld)

    grad = jax.grad(forward)
    rng = np.random.default_rng(0)
    args = (jnp.asarray(rng.uniform(-1, 1, N)), jnp.asarray(rng.uniform(0.7, 1.2, N)),
            jnp.full(N, h_max), jnp.asarray(rng.uniform(-0.01, 0.01, N)))
    info = dict(order=order, squarings=s, norm_bound=bound, denominator_margin=margin, method=method,
                ortho=ortho, terms=terms, layout=layout, solver=solver)
    if not jit:
        return forward, grad, args, info
    return jax.jit(forward), jax.jit(grad), args, info
