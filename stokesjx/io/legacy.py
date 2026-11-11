"""Readers and writers of the file formats of CODE-A (numpy only, no JAX).

- ``read_viscosity``: a viscosity profile of ``VIS_PROF`` (``viscinit``,
  util.f90:353-364);
- ``read_sph``: a ``.sph`` density file (``initialize``, util.f90:7-54);
- ``read_sphco`` and ``write_sphco``: the ``SPHCO`` coefficient files of the fields
  (``sphcout``, util.f90:215-240).

Every coefficient array follows the layout rule of ``stokesjx.sh``: the degree and
order axes last, the ``C/S`` axis at ``-3`` (index 0 for ``C``, 1 for ``S``), the radial
axis first for a density, CMB first. The ``.sph`` files hold ``real_4pi`` coefficients
(CODE-A multiplies them by ``sqt = sqrt(4 pi)`` on reading, util.f90:52); the ``SPHCO``
files hold ``real_orthonormal`` coefficients.

Each data line of an ``SPHCO`` file holds ``C`` and then ``S``, whatever its header line
``l ,m , Cnm(m>=0) , Snm(m<0)`` says (audit row 11): ``sphcout`` writes ``coef(l, -m)``
first, and ``calc.f:96-101`` multiplies ``coef(l, -m)`` by ``cos(m phi)``. The readers
parse the data order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

#: CODE-A's ``visc0 = 1e19 Pa s`` (constants.h:82), the unit of most files of VIS_PROF.
VISC0 = 1e19
#: The four header lines of ``sphcout`` with a correct column line (C, then S).
DEFAULT_HEADER = ("... HEADER", "... INPUT : ", "... ", "l, m, Cnm, Snm")
_END = "... END OF FILE!"


@dataclass(frozen=True)
class SphDensity:
    """The content of a ``.sph`` density file.

    Attributes
    ----------
    lmax
        ``LMAX`` of the file (columns 8-11 of the first line).
    radius
        ``(NR + 1,)``, m, CMB first.
    average
        ``(NR + 1,)``, kg/m^3, CMB first: the averages of the file, which CODE-A reads
        as ``rho0``.
    coefficients
        ``(NR + 1, 2, LMAX + 1, LMAX + 1)``, kg/m^3, ``real_4pi`` (the file convention),
        CMB first: the density layout of ``fields.apply``.
    """

    lmax: int
    radius: np.ndarray
    average: np.ndarray
    coefficients: np.ndarray


def read_viscosity(path, n_nodes: int, *, unit: float = VISC0) -> np.ndarray:
    """A viscosity profile in the format of ``viscinit`` (util.f90:353-364).

    Parameters
    ----------
    path
        The file: one value per line, surface first.
    n_nodes
        The number of nodes ``nr + 1`` to read; CODE-A reads exactly this many lines.
    unit
        The factor to Pa s: ``1e19`` (CODE-A's ``visc0``) for the relative profiles,
        ``1.0`` for a file in Pa s.

    Returns
    -------
    ``(n_nodes,)`` float64, Pa s, CMB first. Each line gives its first number, as the
    list-directed ``read(183, *) rvsc(i)`` takes it; a missing final newline is fine.
    Fewer than ``n_nodes`` values raise ``ValueError``.
    """
    values = []
    with open(path) as fh:
        for line in fh:
            tokens = line.replace(",", " ").split()
            if not tokens:
                continue
            values.append(float(tokens[0].replace("D", "E").replace("d", "e")))
            if len(values) == n_nodes:
                break
    if len(values) < n_nodes:
        raise ValueError(f"{path}: {len(values)} viscosity values, {n_nodes} needed")
    return np.asarray(values[::-1], dtype=np.float64) * float(unit)


def read_sph(path) -> SphDensity:
    """A ``.sph`` density file in the order of CODE-A's ``initialize`` (util.f90:7-54).

    Line 1: ``LMAX`` from columns 8-11 and ``NR`` from columns 16-19. Lines 2 and 3 are
    not read (the second holds ``# Density`` or ``# Temperature: INPUT ...``). Then
    ``NR + 1`` radii (surface first), one line, ``NR + 1`` averages, one line, and per node
    (surface first) one header line and ``(LMAX + 1)(LMAX + 2) / 2`` lines of ``C S`` for
    ``l = 0 .. LMAX``, ``m = 0 .. l``. The header text is never parsed beyond the two
    integers, so the reader does not depend on it.

    Parameters
    ----------
    path
        The ``.sph`` file.

    Returns
    -------
    ``SphDensity`` with the radius in m and everything CMB first.
    """
    with open(path) as fh:
        lines = fh.read().splitlines()
    first = lines[0]
    lmax = int(first[7:11])
    nr = int(first[15:19])
    n = nr + 1
    k = 3
    radius = np.array([float(lines[k + i].split()[0]) for i in range(n)])
    k += n + 1
    average = np.array([float(lines[k + i].split()[0]) for i in range(n)])
    k += n + 1
    n_lm = (lmax + 1) * (lmax + 2) // 2
    l_idx = np.concatenate([np.full(l + 1, l) for l in range(lmax + 1)])
    m_idx = np.concatenate([np.arange(l + 1) for l in range(lmax + 1)])
    coefficients = np.zeros((n, 2, lmax + 1, lmax + 1))
    for block in range(n):
        k += 1                                               # the "# Radial Index" line
        data = np.array([[float(v) for v in lines[k + i].split()[:2]] for i in range(n_lm)])
        k += n_lm
        j = n - 1 - block                                    # surface first in the file
        coefficients[j, 0, l_idx, m_idx] = data[:, 0]
        coefficients[j, 1, l_idx, m_idx] = data[:, 1]
    return SphDensity(lmax=lmax, radius=radius[::-1].copy(), average=average[::-1].copy(),
                      coefficients=coefficients)


def read_sphco(path) -> np.ndarray:
    """An ``SPHCO`` file, ``(2, L+1, L+1)`` float64, ``real_orthonormal``.

    Every line that starts with ``SPHCO`` holds ``l``, ``m``, ``C``, ``S`` in this order
    (audit row 11). ``L`` is the largest ``l`` of the file; degrees below the first line
    (``l_min = 1`` in CODE-A) are zero. A Fortran exponent of three digits without the
    letter ``E`` (``0.123456-100``) is read too.

    Parameters
    ----------
    path
        The ``SPHCO`` file.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("SPHCO"):
                tok = line.split()
                rows.append((int(tok[1]), int(tok[2]), _fortran_float(tok[3]), _fortran_float(tok[4])))
    L = max(r[0] for r in rows)
    out = np.zeros((2, L + 1, L + 1))
    for l, m, c, s in rows:
        out[0, l, m] = c
        out[1, l, m] = s
    return out


def _fortran_float(token: str) -> float:
    """A Fortran real: ``1.5E+002``, ``1.5D+002`` or ``0.15-100`` (an ``Ew.d`` exponent
    of three digits drops the letter)."""
    t = token.replace("D", "E").replace("d", "e")
    if "E" not in t and "e" not in t:
        for i in range(len(t) - 1, 0, -1):
            if t[i] in "+-":
                t = t[:i] + "E" + t[i:]
                break
    return float(t)


def _format_e15_6(x: float) -> str:
    """The Fortran edit descriptor ``e15.6``: ``0.ddddddE+xx`` in 15 columns, and
    ``0.dddddd+xxx`` without the letter for an exponent of three digits."""
    x = float(x)
    if x == 0.0:
        mant, exp = "0.000000", 0
        sign = "-" if np.signbit(x) else ""
    else:
        s = f"{x:.5e}"                                       # d.ddddde+XX, correctly rounded
        sign = "-" if s.startswith("-") else ""
        s = s.lstrip("-")
        digits = s[0] + s[2:7]
        exp = int(s[8:]) + 1
        mant = "0." + digits
    if abs(exp) <= 99:
        e = f"E{'+' if exp >= 0 else '-'}{abs(exp):02d}"
    else:
        e = f"{'+' if exp >= 0 else '-'}{abs(exp):03d}"
    return f"{sign}{mant}{e}".rjust(15)


def _format_es25_16e3(x: float) -> str:
    """The Fortran edit descriptor ``es25.16e3``: ``d.ddddddddddddddddE+xxx`` in 25
    columns (the format of the golden harness)."""
    x = float(x)
    s = f"{x:.16e}"
    mant, exp = s.split("e")
    exp = int(exp)
    return f"{mant}E{'+' if exp >= 0 else '-'}{abs(exp):03d}".rjust(25)


def write_sphco(path, coefficients, *, lmin: int = 1, digits: int = 6, header: tuple[str, ...] | None = None) -> None:
    """Write coefficients in the format of CODE-A's ``sphcout`` (util.f90:215-240).

    Parameters
    ----------
    path
        Output file.
    coefficients
        ``(2, L+1, L+1)``, ``real_orthonormal`` (for example one field of ``Fields`` of a
        real density, converted with ``sh.convert``).
    lmin
        First degree written (CODE-A's ``l_min = 1``).
    digits
        ``6``: the Fortran ``e15.6`` of CODE-A, byte for byte. ``16``: ``es25.16e3``, the
        format of the golden harness patch.
    header
        The lines before the data; default ``DEFAULT_HEADER``, CODE-A's four lines with a
        correct column line (C, then S). Pass the header of a Fortran file to reproduce
        it byte for byte.

    Each data line is ``SPHCO``, ``l`` and ``m`` as ``I8``, then ``C`` and ``S``; the ``S``
    of ``m = 0`` is written as zero, as in CODE-A. The last line is
    ``... END OF FILE!``.
    """
    c = np.asarray(coefficients, dtype=np.float64)
    if c.ndim != 3 or c.shape[0] != 2 or c.shape[1] != c.shape[2]:
        raise ValueError(f"coefficients must be (2, L+1, L+1); got {c.shape}")
    fmt = {6: _format_e15_6, 16: _format_es25_16e3}.get(digits)
    if fmt is None:
        raise ValueError("digits must be 6 (e15.6) or 16 (es25.16e3)")
    L = c.shape[-1] - 1
    out = list(DEFAULT_HEADER if header is None else header)
    for l in range(lmin, L + 1):
        for m in range(l + 1):
            s = 0.0 if m == 0 else c[1, l, m]
            out.append(f"SPHCO{l:8d}{m:8d}{fmt(c[0, l, m])}{fmt(s)}")
    out.append(_END)
    with open(os.fspath(path), "w") as fh:
        fh.write("\n".join(out) + "\n")
