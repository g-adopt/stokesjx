"""Adapters between the oracle's records and the package, and the comparison rule of
the chunk 2 tests.

``profile_from_table`` turns an oracle ``NodeTable`` (decimal strings) into a
``RadialProfile`` (float64 by ``float(s)``, which rounds each decimal to nearest).
``assert_rows_close`` is the one comparison of two kernel arrays: per degree row the
max-normalised error ``e_l`` of CONVENTIONS 10, except for a row whose reference is an
exact zero (the CM geoid at ``l = 1``), which must be an exact zero too (``e_l`` would
be 0/0 there).
"""

from __future__ import annotations

import numpy as np

SI_OUTPUTS = ("geoid", "surface_topography", "cmb_topography", "surface_velocity")


def profile_from_table(table, *, surface_density=None, core_mass=..., structure=None):
    """``RadialProfile`` of an oracle ``NodeTable``.

    ``core_mass``: by default the table's value (``None`` is the consistent core);
    ``surface_density``: ``None`` (the top node) or a number in kg/m^3.
    """
    from stokesjx import RadialProfile
    f = lambda xs: np.array([float(x) for x in xs], dtype=np.float64)
    cm = table.core_mass if core_mass is ... else core_mass
    return RadialProfile(f(table.radius), f(table.rho0), f(table.eta),
                         core_density=float(table.core_density),
                         core_mass=None if cm is None else float(cm),
                         outer_density=float(table.outer_density),
                         surface_density=None if surface_density is None else float(surface_density),
                         structure=structure)


def formulation_from_physics(physics):
    """The package ``Formulation`` of an oracle ``Physics`` record (the oracle's strings
    to the package flags). ``degree_one`` and the scales are not flags of the
    formulation; the caller passes them to ``StokesConfig`` and ``Units``."""
    import stokesjx as sj
    gravity = {"full": sj.GravityModel.FULL, "none": sj.GravityModel.NO_SELF_GRAVITY}[physics.self_gravity]
    background = sj.Hydrostatic() if physics.background == "hydrostatic" else sj.Uniform(float(physics.uniform_g))
    bc = {"free_slip": sj.FreeSlip(), "no_slip": sj.NoSlip()}
    return sj.Formulation(gravity=gravity, background=background, compressible=physics.compressible,
                          surface=bc[physics.surface], cmb=bc[physics.cmb])


def legacy_weights(radius) -> np.ndarray:
    """CODE-A's radial quadrature weights: ``(R - r_c) / nr`` at every one of the
    ``nr + 1`` nodes (calc.f:51, 84-92; audit row 4), in m. Test-only: the package uses
    the trapezoid rule (``fields.quadrature_weights``)."""
    r = np.asarray(radius, dtype=np.float64)
    return np.full_like(r, (r[-1] - r[0]) / (len(r) - 1))


def to_cilm(c) -> np.ndarray:
    """``[..., l, m, C/S]`` (the layout of ``tests/cases.py`` and of
    ``test_golden._read_lebedev_coefficients``) to the layout rule of the package,
    ``[..., C/S, l, m]`` (SHTOOLS ``cilm``)."""
    return np.moveaxis(np.asarray(c), -1, -3)


def row_errors(actual, reference) -> np.ndarray:
    """``e_l`` per row of two ``(L, N)`` arrays: ``max_j |a - r| / max_j |r|``; ``inf`` for a
    row whose reference is an exact zero and whose actual row is not, 0 when both are
    exact zeros."""
    a = np.atleast_2d(np.asarray(actual, dtype=np.float64))
    r = np.atleast_2d(np.asarray(reference, dtype=np.float64))
    assert a.shape == r.shape, (a.shape, r.shape)
    scale = np.max(np.abs(r), axis=1)
    diff = np.max(np.abs(a - r), axis=1)
    out = np.empty(len(scale))
    for k in range(len(scale)):
        if scale[k] == 0:
            out[k] = 0.0 if np.all(a[k] == 0) else np.inf
        else:
            out[k] = diff[k] / scale[k]
    return out


def assert_rows_close(actual, reference, tol, what=""):
    """Assert ``e_l <= tol`` for every row; a row whose reference is an exact zero must
    be an exact zero (exact equality). Returns the largest ``e_l``."""
    e = row_errors(actual, reference)
    bad = np.nonzero(~(e <= tol))[0]
    assert bad.size == 0, f"{what}: rows {bad.tolist()} have e_l {e[bad].tolist()} > {tol}"
    return float(np.max(e)) if e.size else 0.0


def nodewise_error(actual, reference, floor=1e-8) -> float:
    """The node-wise relative error of CONVENTIONS 10 (diagnostic, not a gate): over the
    entries with ``|ref| > floor max |ref|`` of each row."""
    a = np.atleast_2d(np.asarray(actual))
    r = np.atleast_2d(np.asarray(reference))
    worst = 0.0
    for k in range(r.shape[0]):
        m = np.abs(r[k]) > floor * np.max(np.abs(r[k])) if np.max(np.abs(r[k])) > 0 else np.zeros_like(r[k], bool)
        if np.any(m):
            worst = max(worst, float(np.max(np.abs(a[k][m] - r[k][m]) / np.abs(r[k][m]))))
    return worst


def oracle_key(label, l, frame):
    """Key prefix of a stored oracle array: ``<variant>/l<l>/<frame>/``."""
    return f"{label}/l{l}/{frame}/"
