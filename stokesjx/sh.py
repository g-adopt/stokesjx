r"""Spherical-harmonic conventions, conversions and transforms.

The Legendre table, the Gauss-Legendre nodes, the grid and the transforms ``analysis``
and ``synthesis`` are ported from lovejx (``lovejx/sealevel/sht.py``, Apache-2.0, by
the same authors; PLAN section 6), without its ``legacy_sqrt2`` switch, which belongs to
the Fortran sea-level code of lovejx. Everything else here is new.

Layout rule
-----------

One rule for every coefficient array of the package (``fields``, ``sh``,
``io.legacy``):

1. The degree axis and the order axis are the last two axes, ``[..., l, m]``, both of
   length ``L + 1``; the entries with ``m <= l`` are used, the entries with ``m > l``
   are zero.
2. A real convention has the ``C/S`` axis directly before them,
   ``(..., 2, L+1, L+1)``, index 0 for ``C`` (cosine) and 1 for ``S`` (sine), the
   SHTOOLS ``cilm`` layout. A complex convention has no such axis, ``(..., L+1, L+1)``.
3. Every other axis is a batch axis and comes before these. A density has the radial
   axis first, ``(N, ..., L+1, L+1)``, CMB first, the node order of ``RadialProfile``.
4. A kernel array is not a coefficient array. It keeps the layout of ``Kernels``,
   ``(L, N)``: the degree axis first over ``Kernels.degrees``, the node axis second.

Conventions
-----------

With colatitude ``theta``, longitude ``lambda`` east of Greenwich,
``x = cos(theta)``, and the 4 pi-normalised associated Legendre functions without the
Condon-Shortley phase,

    Pbar_lm(x) = sqrt((2l+1) (l-m)! / (l+m)!) P_l^m(x),   P_l^m = (1-x^2)^(m/2) d^m P_l / dx^m,

the three named conventions are:

``"complex_4pi"``
    complex ``U[..., l, m]``, ``m <= l`` (lovejx ``sealevel/sht.py``):
    ``f = sum_l [ U_l0 Pbar_l0 + 2 sum_(m>0) Re(U_lm Pbar_lm e^(i m lambda)) ]``.
``"real_4pi"``
    real ``(..., 2, L+1, L+1)`` (SHTOOLS ``cilm``):
    ``f = sum_lm (C_lm cos(m lambda) + S_lm sin(m lambda)) sqrt(2 - delta_m0) Pbar_lm``,
    the geodesy normalisation, in which every harmonic has the mean square 1 over the
    sphere. The ``.sph`` density files of CODE-A hold this convention.
``"real_orthonormal"``
    ``real_4pi`` times ``sqrt(4 pi)``: the harmonics of CODE-A (calc.f:94-115), whose
    square integrates to 1 over the sphere, and of its ``SPHCO`` output files.

Conversions: ``U_l0 = C_l0`` and ``U_lm = (C_lm - i S_lm) / sqrt(2)`` for ``m > 0``;
``real_orthonormal = sqrt(4 pi) real_4pi`` with ``pi`` from ``Units``, so that
``Units(legacy_constants=True)`` gives CODE-A's ``sqt = sqrt(4 float32(pi))``
(constants.h:14; equations.md section 12). The three conventions were checked against
three independent evaluations (scipy ``lpmv`` with the phase removed, lovejx
``evaluate``, CODE-A's orthonormal formula): 1.1e-15 to 4.8e-15 relative to the largest
value at ``L`` of 12 and 50 (``tests/test_sh.py``).
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from .units import Units

#: The coefficient conventions of the package (module docstring).
CONVENTIONS = ("complex_4pi", "real_4pi", "real_orthonormal")
_REAL = ("real_4pi", "real_orthonormal")


# --------------------------------------------------------------------------
# Port of lovejx: Gauss-Legendre nodes, the Legendre table, the grid, the transforms.
# --------------------------------------------------------------------------


def gauss_legendre(nlat: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights, north to south (port of lovejx).

    Returns ``(x, w)`` with ``x = sin(latitude) = cos(colatitude)`` in decreasing order,
    so that index 0 is the ring nearest the north pole. The weights sum to 2 and
    integrate polynomials of degree up to ``2 nlat - 1`` exactly.

    The nodes are the zeros of ``P_nlat``, polished by two Newton steps on the
    three-term recursion from numpy's nodes, and the weights are
    ``2 / ((1 - x^2) P_nlat'(x)^2)`` evaluated at the polished node (the Numerical
    Recipes ``gauleg`` scheme). The weights are not taken from
    ``numpy.polynomial.legendre.leggauss``: its weights are consistent with its nodes
    only to about 5e-13 relative, which shows up as quadrature noise of 6e-14 in the
    coefficients of a constant field at degree 200; with node-consistent weights the
    noise is 4e-15.
    """
    n = nlat
    x = np.polynomial.legendre.leggauss(n)[0]
    for _ in range(2):
        p_prev, p = np.zeros_like(x), np.ones_like(x)
        for k in range(1, n + 1):
            # P_k = ((2k-1) x P_(k-1) - (k-1) P_(k-2)) / k
            p_prev, p = p, ((2.0 * k - 1.0) * x * p - (k - 1.0) * p_prev) / k
        # P_n'(x) = n (x P_n - P_(n-1)) / (x^2 - 1)
        dp = n * (x * p - p_prev) / (x * x - 1.0)
        x = x - p / dp
    # P_n' at the final node, for the weights.
    p_prev, p = np.zeros_like(x), np.ones_like(x)
    for k in range(1, n + 1):
        p_prev, p = p, ((2.0 * k - 1.0) * x * p - (k - 1.0) * p_prev) / k
    dp = n * (x * p - p_prev) / (x * x - 1.0)
    w = 2.0 / ((1.0 - x * x) * dp * dp)
    # leggauss returns increasing x (south to north); flip to north first.
    return x[::-1].copy(), w[::-1].copy()


def legendre_table(lmax: int, x) -> np.ndarray:
    """Dense table of the 4 pi-normalised associated Legendre functions (port of lovejx).

    Parameters
    ----------
    lmax
        Triangular truncation ``L``; all ``(l, m)`` with ``0 <= m <= l <= L`` are filled.
    x
        Points ``cos(colatitude) = sin(latitude)`` in ``[-1, 1]``, shape ``(n,)``.

    Returns
    -------
    float64 numpy array ``(L+1, L+1, n)``: ``plm[l, m, j] = Pbar_lm(x_j)``, exactly zero
    for ``m > l``.

    The recursion is Whitaker's, with ``eps_lm = sqrt((l^2 - m^2) / (4 l^2 - 1))``:

        Pbar_(m+1, m) = sqrt(2m+3) x Pbar_mm
        Pbar_(l, m)   = (x Pbar_(l-1, m) - eps_(l-1, m) Pbar_(l-2, m)) / eps_(l, m)

    stable in the forward ``l`` direction for every ``m``. The sectoral seed
    ``Pbar_mm = sqrt((2m+1)!! / (2m)!!) sin^m(theta)`` underflows as a plain product at
    high ``m`` near the poles, so it is accumulated as a logarithm and the recursion runs
    on ``Pbar_lm / Pbar_mm``, which starts at 1; the seed is multiplied back at the end,
    and where it is below the double range the column becomes zero, the correctly
    rounded value. This runs on the host (numpy) for static points.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    L = lmax
    # sin(theta) >= 0 on the sphere; the clip protects the logarithm at an exact pole.
    sinth = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    with np.errstate(divide="ignore"):
        log_sinth = np.log(sinth)
    plm = np.zeros((L + 1, L + 1, n), dtype=np.float64)
    log_pref = 0.0                          # log sqrt((2m+1)!! / (2m)!!), accumulated in m
    for m in range(L + 1):
        if m > 0:
            log_pref += 0.5 * math.log((2.0 * m + 1.0) / (2.0 * m))
        q_prev2 = np.ones(n)                # q_m = Pbar_mm / Pbar_mm = 1
        col = [q_prev2]
        if m + 1 <= L:
            q_prev1 = math.sqrt(2.0 * m + 3.0) * x
            col.append(q_prev1)
            eps_prev = math.sqrt(((m + 1) ** 2 - m * m) / (4.0 * (m + 1) ** 2 - 1.0))
            for l in range(m + 2, L + 1):
                eps = math.sqrt((l * l - m * m) / (4.0 * l * l - 1.0))
                q = (x * q_prev1 - eps_prev * q_prev2) / eps
                col.append(q)
                q_prev2, q_prev1, eps_prev = q_prev1, q, eps
        # At m = 0 the seed has no sin(theta) factor; 0 * log(0) would be NaN at a pole.
        with np.errstate(under="ignore"):
            seed = np.full(n, math.exp(log_pref)) if m == 0 else np.exp(log_pref + m * log_sinth)
        plm[m:, m, :] = np.stack(col) * seed[None, :]
    return plm


@dataclasses.dataclass(frozen=True)
class GaussLegendreGrid:
    """A Gauss-Legendre grid with its Legendre table, built once in numpy (port of the
    lovejx class without its ``legacy_sqrt2`` switch). A JAX pytree: the arrays are
    leaves, the sizes static.

    Attributes
    ----------
    nlat, nlon, ntrunc
        Grid size and triangular truncation ``L <= nlon // 2``.
    x
        ``sin(latitude)`` of the rings, north to south, ``(nlat,)``.
    weights
        Gauss weights, ``(nlat,)``, summing to 2.
    lon
        Longitudes in radians, ``2 pi i / nlon``, ``(nlon,)``, Greenwich first.
    plm
        ``Pbar_lm(x_j)``, ``(L+1, L+1, nlat)``.

    Fields on the grid are ``(nlat, nlon)``: latitude ring ``j`` (north to south) and
    longitude ``i``. On an even grid with ``L = nlon / 2`` the column ``m = nlon / 2`` is
    the Nyquist order, whose sine part the grid cannot carry; with ``nlon >= 2 L + 1``
    every order is exact.
    """

    nlat: int
    nlon: int
    ntrunc: int
    x: np.ndarray
    weights: np.ndarray
    lon: np.ndarray
    plm: np.ndarray

    @classmethod
    def build(cls, nlat: int, nlon: int, ntrunc: int) -> "GaussLegendreGrid":
        """Construct the grid and its Legendre table. Raises ``ValueError`` for
        ``ntrunc > nlon // 2`` (the largest order that a real FFT of ``nlon`` points
        carries)."""
        if ntrunc > nlon // 2:
            raise ValueError("ntrunc must not exceed nlon // 2")
        x, w = gauss_legendre(nlat)
        lon = 2.0 * np.pi * np.arange(nlon) / nlon
        return cls(nlat, nlon, ntrunc, x, w, lon, legendre_table(ntrunc, x))

    @property
    def latitude_deg(self) -> np.ndarray:
        """Latitude of each ring in degrees, north to south."""
        return np.degrees(np.arcsin(np.asarray(self.x)))

    @property
    def longitude_deg(self) -> np.ndarray:
        """Longitude of each column in degrees east of Greenwich."""
        return np.degrees(np.asarray(self.lon))


jax.tree_util.register_dataclass(GaussLegendreGrid, data_fields=["x", "weights", "lon", "plm"],
                                 meta_fields=["nlat", "nlon", "ntrunc"])


def analysis(grid: GaussLegendreGrid, field) -> jax.Array:
    """Spherical-harmonic analysis, ``field (nlat, nlon)`` to ``complex_4pi`` ``U (L+1, L+1)``
    (port of lovejx):

        U(l, m) = (1 / (2 nlon)) sum_j w_j Pbar_lm(x_j) sum_i f_ji e^(-i m lambda_i),

    Gauss-Legendre quadrature in latitude and the trapezoid rule (exact for the retained
    orders) in longitude; ``1 / (2 nlon)`` is ``1 / 4 pi`` times the longitude spacing.
    The real and imaginary parts are contracted separately against the real table, so
    that the table is never promoted to complex.

    Parameters
    ----------
    grid
        ``GaussLegendreGrid`` (latitude nodes and weights, the Legendre table, ``nlon`` and
        the truncation ``ntrunc = L``).
    field
        ``(nlat, nlon)`` real values on the grid.
    """
    L, nlon = grid.ntrunc, grid.nlon
    am = jnp.fft.rfft(field, axis=1) / nlon                      # (nlat, nlon // 2 + 1)
    wam = am[:, : L + 1].T * grid.weights[None, :]               # (L+1, nlat)
    ure = jnp.einsum("lmj,mj->lm", grid.plm, wam.real)
    uim = jnp.einsum("lmj,mj->lm", grid.plm, wam.imag)
    return 0.5 * (ure + 1j * uim)


def synthesis(grid: GaussLegendreGrid, U) -> jax.Array:
    """Spherical-harmonic synthesis, ``complex_4pi`` ``U (L+1, L+1)`` to ``field (nlat, nlon)``
    (port of lovejx):
    ``f = sum_l [ U_l0 Pbar_l0 + 2 sum_(m>0) Re(U_lm Pbar_lm e^(i m lambda)) ]``.

    The latitude sum gives the Fourier coefficients ``a_m(j) = sum_l U_lm Pbar_lm(x_j)``,
    and ``irfft`` supplies the factor 2 on ``m > 0`` and the real part. The imaginary part
    of ``U_l0`` and, on an even grid with ``L = nlon / 2``, of the Nyquist column are set to
    zero explicitly (the grid cannot carry them), so that the value and the gradient do
    not depend on the FFT backend.
    """
    L, nlat, nlon = grid.ntrunc, grid.nlat, grid.nlon
    U = jnp.asarray(U)
    are = jnp.einsum("lm,lmj->mj", U.real, grid.plm)
    aim = jnp.einsum("lm,lmj->mj", U.imag, grid.plm)
    aim = aim.at[0].set(0.0)
    if nlon % 2 == 0 and L == nlon // 2:
        aim = aim.at[L].set(0.0)
    am = are + 1j * aim
    spec = jnp.zeros((nlat, nlon // 2 + 1), dtype=am.dtype).at[:, : L + 1].set(am.T)
    # irfft divides by nlon; the coefficients carry the 1 / nlon of the analysis.
    return jnp.fft.irfft(spec, n=nlon, axis=1) * nlon


# --------------------------------------------------------------------------
# Conventions, conversions, grid synthesis and degree power.
# --------------------------------------------------------------------------


def check_layout(coefficients, convention: str) -> None:
    """Raise ``ValueError`` when an array does not follow the layout rule for a
    convention (module docstring). Shapes and dtypes are static, so the check also runs
    under ``jit``.

    A real convention needs a real dtype and the ``C/S`` axis of length 2 at ``-3``; the
    complex convention needs at least two axes. In both the last two axes are square.

    Parameters
    ----------
    coefficients
        The array to check (only its shape and dtype are read).
    convention
        A name from ``CONVENTIONS``.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}; use one of {CONVENTIONS}")
    shape = jnp.shape(coefficients)
    dtype = jnp.result_type(coefficients)
    if len(shape) < 2 or shape[-1] != shape[-2]:
        raise ValueError(f"the degree and order axes must be the last two axes, of equal length L + 1; "
                         f"got the shape {shape}")
    if convention in _REAL:
        if jnp.issubdtype(dtype, jnp.complexfloating):
            raise ValueError(f"the convention {convention!r} is real; got a complex array")
        if len(shape) < 3 or shape[-3] != 2:
            raise ValueError(f"the convention {convention!r} needs the C/S axis of length 2 at axis -3, "
                             f"(..., 2, L+1, L+1); got the shape {shape}")


def _order_factor(L: int) -> np.ndarray:
    """``1`` at ``m = 0`` and ``1 / sqrt(2)`` at ``m > 0``, ``(L+1,)``: the factor of
    ``U_lm = (C_lm - i S_lm) / sqrt(2)``."""
    return np.where(np.arange(L + 1) == 0, 1.0, 1.0 / math.sqrt(2.0))


def convert(coefficients, source: str, target: str, *, units: Units | None = None) -> jax.Array:
    """Convert coefficients between two conventions (module docstring), keeping the batch
    axes and adding or removing the ``C/S`` axis.

    Parameters
    ----------
    coefficients
        Array under the layout rule for ``source``: ``(..., 2, L+1, L+1)`` real or
        ``(..., L+1, L+1)`` complex.
    source, target
        Names from ``CONVENTIONS``.
    units
        Supplies ``pi`` of the factor ``sqrt(4 pi)`` between ``real_4pi`` and
        ``real_orthonormal``; default ``Units()``. ``Units(legacy_constants=True)`` gives
        CODE-A's float32 ``pi`` (``sqt``).

    Returns
    -------
    The coefficients in ``target``: ``U_l0 = C_l0``, ``U_lm = (C_lm - i S_lm) / sqrt(2)``
    (real to complex, complex128), ``C_l0 = Re U_l0``, ``S_l0 = 0``,
    ``C_lm = sqrt(2) Re U_lm``, ``S_lm = -sqrt(2) Im U_lm`` (complex to real, float64), and
    the factor ``sqrt(4 pi)`` between the two real conventions. A linear map, valid under
    ``jit``, ``grad`` and ``vmap``.
    """
    check_layout(coefficients, source)
    if target not in CONVENTIONS:
        raise ValueError(f"unknown convention {target!r}; use one of {CONVENTIONS}")
    units = Units() if units is None else units
    sqrt4pi = math.sqrt(4.0 * units.pi)
    c = jnp.asarray(coefficients)
    if source == target:
        return c
    L = c.shape[-1] - 1
    # Step 1: to real_4pi.
    if source == "real_orthonormal":
        real = c / sqrt4pi
    elif source == "real_4pi":
        real = c
    else:
        f = jnp.asarray(1.0 / _order_factor(L))               # 1 at m = 0, sqrt(2) at m > 0
        C = f * c.real
        S = -f * c.imag * (np.arange(L + 1) > 0)
        real = jnp.stack([C, S], axis=-3)
    # Step 2: from real_4pi.
    if target == "real_4pi":
        return real
    if target == "real_orthonormal":
        return real * sqrt4pi
    f = jnp.asarray(_order_factor(L))
    return f * (real[..., 0, :, :] - 1j * real[..., 1, :, :])


@functools.lru_cache(maxsize=32)
def _legendre_cached(lmax: int, latitudes: tuple[float, ...]) -> np.ndarray:
    """The host Legendre table at static latitudes (degrees), cached on
    ``(lmax, latitudes)`` so that a call outside ``jit`` does not build it again."""
    return legendre_table(lmax, np.sin(np.radians(np.asarray(latitudes, dtype=np.float64))))


def synthesis_grid(coefficients, convention: str, latitude_deg, longitude_deg, *,
                   units: Units | None = None) -> jax.Array:
    """Evaluate coefficients on a latitude-longitude grid.

    Parameters
    ----------
    coefficients
        ``(..., [2,] L+1, L+1)`` under the layout rule for ``convention``.
    convention
        A name from ``CONVENTIONS``.
    latitude_deg, longitude_deg
        Static one-dimensional arrays of latitudes (degrees north) and longitudes
        (degrees east), ``(nlat,)`` and ``(nlon,)``.
    units
        For the ``sqrt(4 pi)`` of ``real_orthonormal`` (``convert``).

    Returns
    -------
    ``(..., nlat, nlon)``: ``f = sum_lm (C_lm cos(m lambda) + S_lm sin(m lambda)) sqrt(2 - delta_m0) Pbar_lm``
    with the ``real_4pi`` coefficients.

    Method: the coefficients are converted to ``real_4pi`` and multiplied by
    ``sqrt(2 - delta_m0)``; the table ``Pbar_lm`` at the static latitudes comes from the
    host (cached); ``a = einsum("...lm,lmi->...im", C, P)``, ``b`` the same for ``S``, and
    ``f = a @ cos(m lambda) + b @ sin(m lambda)`` with the static ``(L+1, nlon)`` matrices.
    Two contractions and two matrix products, no loop over ``l``, ``m`` or a grid point;
    valid under ``jit``, ``grad`` (in the coefficients) and ``vmap``.
    """
    real = convert(coefficients, convention, "real_4pi", units=units)
    L = real.shape[-1] - 1
    lat = tuple(float(v) for v in np.asarray(latitude_deg, dtype=np.float64).reshape(-1))
    lon = np.radians(np.asarray(longitude_deg, dtype=np.float64).reshape(-1))
    P = jnp.asarray(_legendre_cached(L, lat))                          # (L+1, L+1, nlat)
    norm = np.where(np.arange(L + 1) == 0, 1.0, math.sqrt(2.0))        # sqrt(2 - delta_m0)
    m = np.arange(L + 1)[:, None]
    cos_m = jnp.asarray(np.cos(m * lon[None, :]))                      # (L+1, nlon)
    sin_m = jnp.asarray(np.sin(m * lon[None, :]))
    a = jnp.einsum("...lm,lmi->...im", real[..., 0, :, :] * norm, P)   # (..., nlat, L+1)
    b = jnp.einsum("...lm,lmi->...im", real[..., 1, :, :] * norm, P)
    return a @ cos_m + b @ sin_m


def degree_power(x, y, convention: str) -> jax.Array:
    """The cross power per degree, ``S_xy(l)``, ``(..., L+1)``.

    ``"complex_4pi"``: ``S_xy(l) = sum_m w_m Re(x_lm conj(y_lm))`` with ``w_0 = 1`` and
    ``w_(m>0) = 2`` (the unstored negative orders). A real convention:
    ``S_xy(l) = sum_m (C^x_lm C^y_lm + S^x_lm S^y_lm)``. In ``real_4pi`` and in
    ``complex_4pi`` this is the degree power of the mean over the sphere, so the two
    agree; ``real_orthonormal`` gives ``4 pi`` times it. Both inputs follow the layout
    rule for the convention and have the same shape; the batch axes are kept.
    """
    check_layout(x, convention)
    check_layout(y, convention)
    if jnp.shape(x) != jnp.shape(y):
        raise ValueError(f"the two arrays must have the same shape; got {jnp.shape(x)} and {jnp.shape(y)}")
    x, y = jnp.asarray(x), jnp.asarray(y)
    if convention == "complex_4pi":
        L = x.shape[-1] - 1
        w = jnp.asarray(np.where(np.arange(L + 1) == 0, 1.0, 2.0))
        return jnp.sum(w * jnp.real(x * jnp.conj(y)), axis=-1)
    return jnp.sum(x * y, axis=(-3, -1))
