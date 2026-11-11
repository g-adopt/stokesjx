"""A PREM-like node table, the hager4 viscosity and a smooth density anomaly for the
examples and the benchmark of stokesjx.

The examples do not import from ``tests/``. This module therefore builds, in float64,
the same table family as ``tests/cases.py::t_prem`` (the table of the chunk 4 and c4b
measurements): the 129 equally spaced radii of CODE-A's density file
``LEBEDEV_S4RTS.sph`` between the core-mantle boundary (CMB) at 3480 km and the surface
at 6370 km, with a duplicated node pair at 100, 220, 400 and 670 km depth. The reference
density ``rho0`` is the PREM mantle (Dziewonski and Anderson 1981), so ``rho0`` jumps at
the 220, 400 and 670 km pairs, and the 100 km pair is a viscosity-only pair. The core is
the PREM core: its density at the CMB and its mass from the closed-form integral of the
PREM polynomials (a "split core" in the sense of ``RadialProfile.core_mass``).

The radii here are ``np.linspace(3480 km, 6370 km, 128 refine + 1)``. ``tests/cases.py``
copies the radii as the density file prints them, with two decimals, so the two tables
differ by at most 5 mm per node (test ``tests/test_examples.py::
test_earth_model_matches_t_prem`` checks the tolerance).

Contents
--------
``prem_density(radius, side)``
    PREM mantle density at given radii, kg/m^3.
``prem_profile(refine=1, reduced=False, eta_layers=HAGER4_LAYERS)``
    The ``RadialProfile`` of the table: 137 nodes (``refine=1``), 265 (2), 521 (4), or
    25 (``reduced=True``: every 8th LEBEDEV node and the eight pair nodes).
``smooth_density(radius, lmax, seed, amplitude=20.0)``
    A smooth density anomaly as ``real_4pi`` coefficients on a radius array.
"""

from __future__ import annotations

import numpy as np

import stokesjx as sj

# ---------------------------------------------------------------------------
# Geometry of the table.
# ---------------------------------------------------------------------------

#: The surface radius of every table, m: CODE-A's ``Rearth`` (6370 km, not the PREM
#: value 6371 km). The depth of a node is ``R_SURFACE - r``.
R_SURFACE = 6370e3
#: The CMB radius, m.
R_CMB = 3480e3
#: The 129 radii of ``LEBEDEV_S4RTS.sph``, CMB first, m: 128 intervals of 22578.125 m.
LEBEDEV_RADII = np.linspace(R_CMB, R_SURFACE, 129)
#: Depths of the duplicated node pairs, km. ``rho0`` jumps at 220, 400 and 670 km (PREM
#: discontinuities); the 100 km pair carries a viscosity jump only.
PAIR_DEPTHS_KM = (100, 220, 400, 670)

#: The hager4 viscosity by depth (CODE-A's ``Hager_4Layer_Flip.visc``, the
#: ``test_4_ishan`` profile): ``(depth of the bottom of the layer in m, viscosity in
#: Pa s)``, from the surface down; the last layer has no bottom. 1e20 Pa s above 100 km,
#: 1e19/30 Pa s (3.33e17) from 100 to 400 km, 1e19 Pa s from 400 to 670 km, 1e20 Pa s
#: below.
HAGER4_LAYERS = ((100e3, 1e20), (400e3, 1e19 / 30), (670e3, 1e19), (None, 1e20))

# ---------------------------------------------------------------------------
# PREM density polynomials.
# ---------------------------------------------------------------------------

#: PREM normalising radius, m: the polynomials take ``x = r / 6371 km``.
_PREM_X0 = 6371e3

#: PREM density polynomials ``a + b x + c x^2 + d x^3`` in g/cm^3 of ``x = r / 6371 km``
#: (Dziewonski and Anderson 1981, table 1).
_PREM = {
    "inner_core": (13.0885, 0.0, -8.8381, 0.0),
    "outer_core": (12.5815, -1.2638, -3.6426, -5.5281),
    "lower_mantle": (7.9565, -6.4761, 5.5283, -3.0807),
    "tz_lower": (5.3197, -1.4836, 0.0, 0.0),
    "tz_upper": (11.2494, -8.0298, 0.0, 0.0),
    "upper_mantle": (7.1089, -3.8045, 0.0, 0.0),
    "lid": (2.6910, 0.6924, 0.0, 0.0),
}

#: The region boundaries of the mantle on the 6370 km surface, ``(radius in m, region
#: below it)``, from the CMB up: 670 km depth (5700 km), 600 km (5770 km, the boundary
#: inside the transition zone), 400 km (5970 km) and 220 km (6150 km). Above 6150 km the
#: LID polynomial is used up to the surface (no crust and no ocean, as ``t_prem``).
_PREM_REGIONS = ((5700e3, "lower_mantle"), (5770e3, "tz_lower"), (5970e3, "tz_upper"),
                 (6150e3, "upper_mantle"))


def _prem_poly(name: str, radius) -> np.ndarray:
    """One PREM polynomial at ``radius`` (m), in kg/m^3 (the table gives g/cm^3).

    Horner form, float64. The 40-digit decimal evaluation of ``tests/cases.py`` agrees to
    about 4e-16 relative."""
    a, b, c, d = _PREM[name]
    x = np.asarray(radius, dtype=np.float64) / _PREM_X0
    return 1000.0 * (a + x * (b + x * (c + x * d)))


def prem_density(radius, side) -> np.ndarray:
    """The PREM mantle density at ``radius``, kg/m^3.

    Parameters
    ----------
    radius
        Radii in m, any shape, inside the mantle (3480 to 6370 km).
    side
        The side of a node at a region boundary, broadcast against ``radius``: ``-1``
        for the lower member of a jump pair (the region below), ``+1`` for the upper
        member, ``0`` for an ordinary node. A node exactly on a boundary with
        ``side = 0`` takes the region above, the rule of ``tests/cases.py::_prem_region``.

    Returns
    -------
    float64 array of the broadcast shape, kg/m^3. A node strictly inside a region takes
    the polynomial of that region whatever its ``side``.
    """
    r = np.asarray(radius, dtype=np.float64)
    s = np.broadcast_to(np.asarray(side), r.shape)
    out = _prem_poly("lid", r)
    # Walk the boundaries from the top down, so that the deepest region that holds a
    # node is written last. A node belongs to the region below boundary b when it lies
    # strictly below b, or on b as the lower member of a pair.
    for b, name in reversed(_PREM_REGIONS):
        below = (r < b) | ((r == b) & (s < 0))
        out = np.where(below, _prem_poly(name, r), out)
    return out


def _prem_core_mass() -> float:
    """The mass of the PREM core (inner core to 1221.5 km, outer core to 3480 km), kg:
    ``4 pi int rho(r) r^2 dr`` of the cubic polynomials in closed form, about
    1.9355e24 kg."""
    def shell(name, r0, r1):
        a, b, c, d = _PREM[name]
        x0 = _PREM_X0

        def prim(r):
            # Antiderivative of 1000 (a + b r/x0 + c r^2/x0^2 + d r^3/x0^3) r^2 in r.
            return 1000.0 * (a * r ** 3 / 3.0 + b * r ** 4 / (4.0 * x0) + c * r ** 5 / (5.0 * x0 ** 2)
                             + d * r ** 6 / (6.0 * x0 ** 3))
        return prim(r1) - prim(r0)
    return float(4.0 * np.pi * (shell("inner_core", 0.0, 1221.5e3) + shell("outer_core", 1221.5e3, R_CMB)))


#: PREM outer-core density at the CMB (r = 3480 km), kg/m^3: ``rho_c`` of the uniform
#: inviscid core in the CMB conditions.
PREM_CORE_DENSITY = float(_prem_poly("outer_core", R_CMB))
#: The mass of the PREM core, kg (closed-form integral of the PREM polynomials). With
#: ``PREM_CORE_DENSITY`` it makes a split core: the hydrostatic gravity at the CMB comes
#: from the real core mass, the CMB conditions from the density at the CMB.
PREM_CORE_MASS = _prem_core_mass()


def _layer_viscosity(radius, side, eta_layers) -> np.ndarray:
    """The viscosity of each node from a layering by depth, Pa s.

    A node with depth ``d`` belongs to the first layer (from the surface down) whose
    bottom depth ``b`` satisfies ``d < b``, or ``d == b`` for the upper member of a pair
    (``side > 0``); a lower pair member and an ordinary node on a layer bottom belong to
    the layer below. This is the rule of ``tests/cases.py::layered_eta_rel``."""
    r = np.asarray(radius, dtype=np.float64)
    s = np.asarray(side)
    depth = R_SURFACE - r
    eta = np.full(r.shape, np.nan)
    done = np.zeros(r.shape, dtype=bool)
    for bottom, value in eta_layers:
        if bottom is None:
            hit = ~done
        else:
            hit = ~done & ((depth < bottom) | ((depth == bottom) & (s > 0)))
        eta = np.where(hit, float(value), eta)
        done |= hit
    if not np.all(done):
        raise ValueError("eta_layers must end with a layer without a bottom (None)")
    return eta


def prem_profile(refine: int = 1, *, reduced: bool = False, eta_layers=HAGER4_LAYERS) -> sj.RadialProfile:
    """The PREM-like ``RadialProfile`` of the examples and the benchmark.

    Parameters
    ----------
    refine
        ``refine - 1`` equally spaced nodes are put inside every LEBEDEV interval:
        ``128 refine + 1`` radii, then the eight pair nodes. Node counts: 137
        (``refine=1``), 265 (2), 521 (4).
    reduced
        Keep every 8th node of the equally spaced radii (the CMB and the surface
        included) and every pair node: 25 nodes for ``refine=1``, the layout of
        ``tests/cases.py::t_prem(reduced=True)``.
    eta_layers
        The viscosity layering by depth, ``((bottom depth in m or None, Pa s), ...)``
        from the surface down (default ``HAGER4_LAYERS``). A pair member takes the
        viscosity of its own side.

    Returns
    -------
    ``RadialProfile`` (CMB first) with the PREM mantle ``rho0``, the layered viscosity,
    ``core_density = PREM_CORE_DENSITY`` and ``core_mass = PREM_CORE_MASS`` (split
    core), no water layer.

    Raises
    ------
    ValueError
        If ``refine`` is not a positive integer, or a pair radius falls on an existing
        node (no pair radius is a node for ``refine`` in {1, 2, 4}).
    """
    if isinstance(refine, bool) or int(refine) != refine or refine < 1:
        raise ValueError(f"refine must be a positive integer, got {refine!r}")
    refine = int(refine)
    base = np.linspace(R_CMB, R_SURFACE, 128 * refine + 1)
    if reduced:
        base = base[:: 8 * refine]
    pair_r = np.array([R_SURFACE - 1e3 * d for d in PAIR_DEPTHS_KM])
    for r in pair_r:
        if np.any(base == r):
            raise ValueError(f"the pair radius {r / 1e3} km falls on a node of the table")
    # Each node is (radius, side); a pair is two nodes at one radius, the lower member
    # (side -1) first. Sorting by (radius, side) gives the CMB-first table.
    nodes = [(float(r), 0) for r in base] + [(float(r), s) for r in pair_r for s in (-1, 1)]
    nodes.sort()
    radius = np.array([n[0] for n in nodes])
    side = np.array([n[1] for n in nodes])
    rho0 = prem_density(radius, side)
    eta = _layer_viscosity(radius, side, eta_layers)
    return sj.RadialProfile(radius, rho0, eta, core_density=PREM_CORE_DENSITY, core_mass=PREM_CORE_MASS)


def smooth_density(radius, lmax: int, seed: int, amplitude: float = 20.0) -> np.ndarray:
    """A smooth density anomaly, ``(N, 2, lmax+1, lmax+1)`` kg/m^3, ``real_4pi``, CMB
    first.

    With the normalised radius ``xi = (r - r_0) / (r_(N-1) - r_0)`` (0 at the CMB, 1 at the
    surface), each coefficient is ``sum_k cos((k + 1) pi xi) c_k[lm]`` over four cosines,
    with independent standard-normal ``c_k`` from ``np.random.default_rng(seed)`` (the
    rule of ``tests/densities.py::smooth``), then multiplied by ``amplitude /
    sqrt(l + 1)`` so that the power falls with the degree as in a tomography model. The
    entries ``m > l`` and the sine coefficients of ``m = 0`` are zero. The radial
    structure has the scale of the mantle, so the field of the anomaly converges with the
    node spacing of every table of this module.

    Parameters
    ----------
    radius
        ``(N,)`` radii in m, ascending (normally the input radius of a profile).
    lmax
        Largest degree.
    seed
        Seed of the generator.
    amplitude
        Scale of the coefficients at ``l = 0``, kg/m^3.
    """
    r = np.asarray(radius, dtype=np.float64)
    rng = np.random.default_rng(seed)
    xi = (r - r[0]) / (r[-1] - r[0])
    tri = np.tril(np.ones((lmax + 1, lmax + 1)))
    coef = rng.standard_normal((4, 2, lmax + 1, lmax + 1)) * tri
    out = sum(np.cos((k + 1) * np.pi * xi)[:, None, None, None] * coef[k] for k in range(4))
    out = out * (amplitude / np.sqrt(np.arange(lmax + 1) + 1.0))[None, None, :, None]
    out[:, 1, :, 0] = 0.0
    return out
