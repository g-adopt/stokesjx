"""Test density anomalies as functions of the radius (NOTES/chunks/c4-controls-derivatives/01-plan.md D11).

A density anomaly of a field is given on the input table of a ``Layers`` and does not
move with a boundary, so the tests define it as a function of the radius and evaluate it
at the input radii. Both anomalies are ``real_4pi`` coefficients in the layout of
CONVENTIONS 15, ``(N, 2, lmax+1, lmax+1)`` in kg/m^3, with the entries ``m > l`` zero.

``smooth``
    Four cosines of the normalised radius ``xi = (r - r_c) / (R - r_c)`` with
    independent standard-normal coefficients per harmonic and per cosine:
    ``sum_k cos((k + 1) pi xi) c_k[lm]``. Radial structure on the scale of the mantle,
    as a tomography model resolves it.
``rough``
    An independent standard-normal value per input node and coefficient (the
    piecewise-linear function of these values between the nodes). Radial structure at
    the node spacing; the stress case of the kink measurement.

The generators are the planner's (``plan-evidence/multi_standin.py``), so that the
seeds of the plan give the same arrays.
"""

from __future__ import annotations

import numpy as np


def _lower_triangle(lmax: int) -> np.ndarray:
    """``(lmax+1, lmax+1)`` mask of the valid entries ``m <= l``."""
    return np.tril(np.ones((lmax + 1, lmax + 1)))


def smooth(radius, lmax: int, seed: int) -> np.ndarray:
    """Four cosines of the normalised radius with random coefficients per harmonic,
    ``(N, 2, lmax+1, lmax+1)`` kg/m^3, ``real_4pi``."""
    r = np.asarray(radius, dtype=np.float64)
    rng = np.random.default_rng(seed)
    xi = (r - r[0]) / (r[-1] - r[0])
    coef = rng.standard_normal((4, 2, lmax + 1, lmax + 1)) * _lower_triangle(lmax)
    return sum(np.cos((k + 1) * np.pi * xi)[:, None, None, None] * coef[k] for k in range(4))


def rough(radius, lmax: int, seed: int) -> np.ndarray:
    """An independent random value per node and coefficient, ``(N, 2, lmax+1, lmax+1)``
    kg/m^3, ``real_4pi``."""
    r = np.asarray(radius, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((r.size, 2, lmax + 1, lmax + 1)) * _lower_triangle(lmax)
