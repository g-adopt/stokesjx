"""Movable viscosity boundaries across density jumps: the viscosity cut of chunk c4b
(NOTES/chunks/c4b-boundaries-across-jumps/01-plan.md sections 9.1 to 9.4, gates G1, G3 to
G9; NOTES/PLAN.md ruling 14; NOTES/docs/equations.md section 19).

Under ``Layers(..., motion="cut")`` (the default ``"auto"`` when a movable range holds a
density-jump pair) a movable viscosity boundary is a cut inside one interval of the
FIXED node table, at a traced radius ``c``. The interval that holds ``c`` is split into
two exact sub-steps; no node moves, and the density jumps of the input table stay where
they are.

Fixtures (reduced PREM table ``cases.t_prem(reduced=True)``, 25 input nodes, density
pairs at 5700, 5970 and 6150 km, a viscosity-only pair at 6270 km):

- FJ1: boundaries 3480, 5600, 5970, 6270, 6370 km; boundary 1 movable in
  (5466.875, 5970) km, holding the 5700 km jump; viscosities 1e22, 1e21, 3e20, 1e22 Pa s.
- FJ2: boundaries 3480, 5800, 6270, 6370 km; boundary 1 movable in (5466.875, 6150) km,
  holding the 5700 and the 5970 km jumps; viscosities 1e22, 3e20, 1e22 Pa s.
- FJ3: boundaries 3480, 5600, 5900, 6270, 6370 km; boundary 1 movable in
  (5466.875, 5828.125) km (holds 5700 km) and boundary 2 in (5828.125, 6150) km (holds
  5970 km); the two ranges share an end node. Viscosities as FJ1.
- FJ1x: FJ1 with a fixed boundary at 4000 km between two input nodes (an inserted pair).
- T3-full and T4-full (slow rows and V7): the layouts of FJ1 (with ``r0 = J`` = 5700 km)
  and FJ2 on the full ``t_prem()`` table (137 input nodes).

Every reference table is built with numpy by ``tests/reference_cut.py`` from the input
table, the boundary radii and the layer viscosities (never from a profile that
``Layers`` returns). Under TALA every solve of a fixture whose cut range holds a density
pair issues the ``RuntimeWarning`` of ``propagate.solve``; the module mark below ignores
it, and C9 and K2 assert it with ``pytest.warns`` (which records it regardless of this
filter).
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import cases  # noqa: E402
import densities  # noqa: E402
import reference_cut as rc  # noqa: E402
from adapters import SI_OUTPUTS, profile_from_table, row_errors  # noqa: E402
from taylor import (TaylorProgram, assert_jvp_vjp, assert_taylor, assess, remeasure_taylor,  # noqa: E402
                    taylor_remainders)
import test_controls_ad as tca  # noqa: E402

import stokesjx as sj  # noqa: E402
from stokesjx import fields, propagate  # noqa: E402
from stokesjx.profile import NodeStructure, _unflatten  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:under TALA the kernels jump:RuntimeWarning")

KM = 1e3
J = 5700e3                  # the 670 km density-jump pair of t_prem
J2 = 5970e3                 # the 400 km density-jump pair

# ---------------------------------------------------------------------------
# Named tolerances (plan 9.0; measured values in the plan and in
# NOTES/chunks/c4b-boundaries-across-jumps/impl-evidence/measure_constants.out).
# ---------------------------------------------------------------------------
# The cut against the package solver on a concrete reference table of 9.0 (swapped pair,
# node split, pair at J under FULL, geometric pair), e_l: two solves of the same
# discrete model, rounding only (measured 2.1e-14).
CUT_SWAP_TOL = 1e-13
# The same with a viscosity contrast of 1e6 across the cut (measured 5.2e-13 at lmax 50,
# 4.1e-13 at lmax 100): the rounding of the similarity grows with the contrast.
CUT_SWAP_TOL_CONTRAST = 2e-12
# V9: under TALA the cut at J (M-below) and the fixed boundary on the pair (M-mean) are
# different models (measured 4.4e-2); they must differ by at least this.
TALA_TIE_MIN = 1e-3
# V9: the change of the kernels over 1 mm on the continuous side of the tie (measured
# 8.9e-9 at lmax 10).
TIE_SIDE_TOL = 1e-7
# Against the mpmath oracle (CONVENTIONS 10, chunk 2): measured 2.5e-14.
ORACLE_TOL = 1e-12
# V4: the solve without the cut step must miss the reference by at least this (measured
# 6.2e-2 smallest on FJ1 at lmax 10).
NEG_MIN = 1e-4
# Cut against CK (a Layers with the boundary pair at the new radius), kernels e_l, on the
# full t_prem table: FULL, NO_SELF_GRAVITY and no slip (measured 6.6e-7 at lmax 10 and
# 9.5e-7 at lmax 30), and TALA off a density pair (measured 3.1e-6 and 6.6e-6). The two
# models differ by the rho0 of the two parts of the cut interval, so the difference grows
# with the interval width: 1.1e-4 and 2.0e-4 on the 25-node reduced table, which is why
# V7 runs on the full table (review r3 of the plan, issue 1).
CUT_CK_TOL = 5e-6
CUT_CK_TOL_TALA = 3e-5
# One-sided derivative jump of the geoid field over the derivative, incompressible
# (measured 1.6e-11 on FJ1 at lmax 10, 4.8e-10 on the full tables at lmax 50).
RHO_C1 = 1e-8
# TALA: the derivative jump at an input node must exceed this (measured 9.4e-4 at
# 5647.5 km on FJ1).
TALA_KINK_MIN = 1e-6
# TALA: the value jump at J as a displacement of the boundary must exceed this (measured
# 1.0e4 m).
TALA_JUMP_MIN_LENGTH = 1e3
# The value jump as a displacement where the map is continuous (incompressible
# everywhere, TALA at an input node that is not a pair): the rounding of the linear
# extrapolation (measured 3.2e-6 m on FJ1, 1.3e-5 m on the full tables).
C1_JUMP_MAX_LENGTH = 1e-4
# jvp/vjp dot product on the normalised map (CONVENTIONS 8).
DOT_RTOL = 1e-12
# vmap and eager evaluation against jit (chunk 4 constants).
VMAP_RTOL = 1e-12
JIT_EAGER_RTOL = 1e-12
# The derivative at r0 against the one-sided three-point differences, h = 10 m (chunk 4).
FD_ONE_SIDED_TOL = 1e-7
FD_STEP = 10.0
# The density map of a cut Layers is a static gather and scatter.
SHEET_TOL = 1e-15
# A Taylor base point under TALA is farther from every input radius than this many
# largest steps (chunk 4 CROSSING_CLEARANCE; the largest step is 1e-2 of the direction).
CROSSING_CLEARANCE = 20
RADIUS_DIRECTION = 5000.0

ETA_FJ1 = np.array([1e22, 1e21, 3e20, 1e22])
ETA_FJ2 = np.array([1e22, 3e20, 1e22])
B_FJ1 = np.array([3480, 5600, 5970, 6270, 6370]) * KM
B_FJ2 = np.array([3480, 5800, 6270, 6370]) * KM
B_FJ3 = np.array([3480, 5600, 5900, 6270, 6370]) * KM
B_FJ1X = np.array([3480, 4000, 5600, 5970, 6270, 6370]) * KM
ETA_FJ1X = np.array([3e22, 1e22, 1e21, 3e20, 1e22])
LAYOUTS = {
    # name: (reduced table, boundaries, movable, layer viscosities, cut boundary indices)
    "FJ1": (True, B_FJ1, {1: (5466.875e3, 5970e3)}, ETA_FJ1),
    "FJ2": (True, B_FJ2, {1: (5466.875e3, 6150e3)}, ETA_FJ2),
    "FJ3": (True, B_FJ3, {1: (5466.875e3, 5828.125e3), 2: (5828.125e3, 6150e3)}, ETA_FJ1),
    "FJ1x": (True, B_FJ1X, {2: (5466.875e3, 5970e3)}, ETA_FJ1X),
    "T3full": (False, np.array([3480, 5700, 5970, 6270, 6370]) * KM, {1: (5466.875e3, 5970e3)}, ETA_FJ1),
    "T4full": (False, B_FJ2, {1: (5466.875e3, 6150e3)}, ETA_FJ2),
}

FORMS = {
    "default": sj.Formulation(),
    "no_slip": sj.Formulation(surface=sj.NoSlip(), cmb=sj.NoSlip()),
    "nsg_uniform": sj.Formulation(gravity=sj.GravityModel.NO_SELF_GRAVITY, background=sj.Uniform(10.0)),
    "tala": sj.Formulation(compressible=True),
    "tala_nsg_uniform": sj.Formulation(compressible=True, gravity=sj.GravityModel.NO_SELF_GRAVITY,
                                       background=sj.Uniform(10.0)),
}


def physics(form_name: str):
    """The oracle ``Physics`` of a formulation of ``FORMS``."""
    import reference_mpmath as ro
    return {"default": ro.Physics(), "no_slip": ro.Physics(surface="no_slip", cmb="no_slip"),
            "nsg_uniform": ro.Physics(self_gravity="none", background="uniform", uniform_g="10.0"),
            "tala": ro.Physics(compressible=True),
            "tala_nsg_uniform": ro.Physics(compressible=True, self_gravity="none", background="uniform",
                                           uniform_g="10.0")}[form_name]


# ---------------------------------------------------------------------------
# Fixtures and helpers.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def input_profile(reduced: bool):
    """The input profile of a layout (reduced or full t_prem)."""
    return profile_from_table(cases.t_prem(reduced=reduced))


@functools.lru_cache(maxsize=None)
def layers(name: str, motion: str = "auto", r0: float | None = None):
    """``(Layers, B, eta)`` of a layout; ``r0`` replaces the construction radius of the
    first movable boundary (C12)."""
    reduced, B, movable, eta = LAYOUTS[name]
    B = B.copy()
    if r0 is not None:
        B[min(movable)] = r0
    return sj.Layers(input_profile(reduced), B, movable=movable, motion=motion), B, eta


def ref_input(name: str) -> rc.RefTable:
    """The input table of a layout as numpy arrays."""
    return rc.RefTable.from_profile(input_profile(LAYOUTS[name][0]))


def boundaries_at(name: str, x) -> np.ndarray:
    """The boundary vector of a layout with the movable boundaries at ``x`` (R,)."""
    L, B, _ = layers(name)
    b = B.copy()
    b[L.movable_index] = np.atleast_1d(np.asarray(x, dtype=np.float64))
    return b


def cut_index(name: str) -> tuple[int, ...]:
    return tuple(int(k) for k in layers(name)[0].movable_index)


@functools.lru_cache(maxsize=None)
def kernel_fn(name: str, form_name: str, lmax: int, motion: str = "auto"):
    """A jitted ``x -> input kernels`` of a layout at the movable radii ``x``: one compile
    per worker, layout, formulation and degree."""
    L, B, eta = layers(name, motion)
    cfg = sj.StokesConfig(lmax=lmax, formulation=FORMS[form_name])
    le = jnp.asarray(np.log(eta))

    def f(x):
        K = L.input_kernels(sj.stokes_kernels(L(L.full_boundaries(x), le), cfg))
        return {o: getattr(K, o) for o in SI_OUTPUTS}
    return jax.jit(f), cfg


def cut_kernels(name, form_name, lmax, x) -> dict:
    f, _ = kernel_fn(name, form_name, lmax)
    return {o: np.asarray(v) for o, v in f(jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))).items()}


def package_on(ref: rc.RefTable, form_name: str, lmax: int, keep) -> dict:
    """The package solver on a reference table, the rows of the input nodes."""
    K = sj.stokes_kernels(rc.as_profile(ref), sj.StokesConfig(lmax=lmax, formulation=FORMS[form_name]))
    return {o: np.asarray(getattr(K, o))[:, keep] for o in SI_OUTPUTS}


def worst_el(a: dict, b: dict) -> float:
    """The largest ``e_l`` of the four kernels (exact zeros must match exactly)."""
    out = 0.0
    for o in SI_OUTPUTS:
        e = row_errors(a[o], b[o])
        assert np.all(np.isfinite(e)), (o, e)
        out = max(out, float(np.max(e)))
    return out


def rows(K: dict, degrees, lmax_rows) -> dict:
    """The rows of the degrees ``degrees`` of a kernel dict whose first row is degree
    ``lmax_rows[0]``."""
    first = lmax_rows[0]
    return {o: np.asarray(K[o])[[l - first for l in degrees], :] for o in SI_OUTPUTS}


def degrees_of(lmax: int, form_name: str) -> tuple[int, ...]:
    """The degrees of the kernel rows (the frame rule may refuse degree 1)."""
    L, _, _ = layers("FJ1")
    plan = propagate.static_plan(L.profile.structure, sj.StokesConfig(lmax=lmax, formulation=FORMS[form_name]))
    return plan.degrees


def midpoint_above(name: str, x: float) -> float:
    """The middle of the input interval that starts at or above ``x``."""
    r = np.unique(ref_input(name).radius)
    j = int(np.flatnonzero(r >= x)[0])
    return 0.5 * (r[j] + r[j + 1])


def positions(name: str) -> dict:
    """The positions of plan section 9 for a one-cut layout (FJ1, FJ2): the name and the
    movable radius, and whether it lies on a node."""
    L, _, _ = layers(name)
    lo, hi = L.radius_bounds[0]
    out = {"J-10km": J - 10e3, "J-1m": J - 1.0, "J": J, "J+1m": J + 1.0, "J+10km": J + 10e3,
           "bound_lo": lo, "bound_hi": hi, "node": 5647.5e3, "r0": float(L.start_radii[0]),
           "mid": midpoint_above(name, J)}
    if name == "FJ2":
        out.update({"J2-1m": J2 - 1.0, "J2": J2, "J2+1m": J2 + 1.0, "node2": 6008.75e3})
    return out


def on_node(name: str, x) -> bool:
    return bool(np.any(ref_input(name).radius == x))


# ===========================================================================
# 9.1 Construction and the call (no solver derivative)
# ===========================================================================


def test_c1_motion_resolution():
    """C1: "auto" resolves to "cut" when a range holds a density-jump pair, to "affine"
    otherwise; ``MovableRange.density_jumps`` lists the pairs inside."""
    for name, jumps in (("FJ1", {1: (J,)}), ("FJ2", {1: (J, J2)}), ("FJ3", {1: (J,), 2: (J2,)})):
        L = layers(name)[0]
        assert L.motion == "cut"
        for k, v in jumps.items():
            assert L.ranges[k].motion == "cut" and L.ranges[k].density_jumps == v
    p = input_profile(True)
    for B, mov in ((tca.B_F1, tca.MOV_F1), (tca.B_F3, tca.MOV_F3),
                   (np.array([3480, 5286.25, 5700, 5970, 6270, 6370]) * KM,
                    {1: (5105.625e3, 5700e3), 4: (6189.375e3, 6370e3)})):
        L = sj.Layers(p, B, movable=mov)
        assert L.motion == "affine"
        assert all(g.motion == "affine" and g.density_jumps == () for g in L.ranges.values())
    assert sj.Layers(p, tca.B_F1).motion is None
    assert sj.Layers(p, tca.B_F1, movable=tca.MOV_F1, motion="cut").motion == "cut"


@pytest.mark.parametrize("name", ["FJ1", "FJ2", "FJ3", "FJ1x"])
def test_c2_construction_table(name):
    """C2: the construction table holds the input nodes and one pair per fixed boundary
    between nodes, no node for a cut; the cuts are the numpy range ends; no limits; the
    Pade plan is the plan of the same radius without cuts."""
    L, B, _ = layers(name)
    r_in = ref_input(name).radius
    fixed = [k for k in range(1, B.size - 1) if k not in L.ranges]
    between = sum(1 for k in fixed if not np.any(r_in == B[k]))
    assert L.n_nodes == r_in.size + 2 * between
    r = np.asarray(L.profile.radius)
    for k in L.movable_index:
        if not np.any(r_in == B[k]):
            assert not np.any(r == B[k])
        assert L.pairs[k - 1] == -1
    expect = []
    for k in L.movable_index:
        low, high = L.ranges[k].low, L.ranges[k].high
        expect.append((int(np.flatnonzero(r == low)[-1]), int(np.flatnonzero(r == high)[0])))
    assert L.profile.structure.cuts == tuple(expect)
    assert L.profile.structure.limits is None and L.width_factor == 1.0
    np.testing.assert_array_equal(np.asarray(L.profile.cut_radius), L.start_radii)
    for form in ("default", "tala"):
        cfg = sj.StokesConfig(lmax=50, formulation=FORMS[form])
        plain = NodeStructure.from_radius(r, consistent_core=L.profile.structure.consistent_core)
        assert L.pade_plan(cfg) == propagate.static_plan(plain, cfg).pade


def _c3_cases():
    """(layout, motion) of C3: the cut layouts, and the chunk 4 layouts F1 and F3 with
    ``motion="cut"`` (ranges without a density jump)."""
    return [("FJ1", "auto"), ("FJ3", "auto"), ("FJ1x", "auto"), ("F1", "cut"), ("F3", "cut")]


def _layers_c3(name, motion):
    if name in ("F1", "F3"):
        B, mov, eta = (tca.B_F1, tca.MOV_F1, tca.ETA_F1) if name == "F1" else (tca.B_F3, tca.MOV_F3, tca.ETA_F3)
        return sj.Layers(input_profile(True), B, movable=mov, motion=motion), B, eta
    return layers(name, motion)


@pytest.mark.parametrize("name,motion", _c3_cases())
def test_c3_c4_layer_map(name, motion):
    """C3, C4: the traced layer map at 200 random joint positions and at the tie positions
    equals the numpy rule of plan 9.0 (``reference_cut.layer_viscosity``) bit for bit;
    radius and rho0 are the construction arrays, and ``cut_radius`` is the clamped radius."""
    L, B, eta = _layers_c3(name, motion)
    r = np.asarray(L.profile.radius)
    r_in = ref_input("FJ1").radius
    rng = np.random.default_rng(3)
    lo, hi = L.radius_bounds[:, 0], L.radius_bounds[:, 1]
    X = [lo + (hi - lo) * rng.random(lo.size) for _ in range(200)]
    # Tie positions: on an input node inside each range, and on every density pair inside.
    for q, k in enumerate(L.movable_index):
        g = L.ranges[k]
        inside = r_in[(r_in > lo[q]) & (r_in < hi[q])]
        for tie in [inside[0], *g.density_jumps]:
            x = L.start_radii.copy(); x[q] = tie
            X.append(x)
    if name == "FJ3":
        X.append(np.array([J, J2]))              # both cuts on their pairs at the same time
    X = np.asarray(X)
    le = np.log(np.arange(1, eta.size + 1, dtype=np.float64) * 1e20)     # distinct layer viscosities

    def call(x, layer_ref):
        q = L(L.full_boundaries(x), jnp.asarray(le))
        return q.eta, jnp.exp(jnp.asarray(le))[layer_ref], q.radius, q.rho0, q.cut_radius
    f = jax.jit(call)
    for x in X:
        b = B.copy(); b[L.movable_index] = x
        layer_ref = np.searchsorted(np.exp(le), rc.layer_viscosity(r, b, L.movable_index, np.exp(le)))
        eta_q, eta_ref, rad, rho, cr = map(np.asarray, f(jnp.asarray(x), jnp.asarray(layer_ref)))
        np.testing.assert_array_equal(eta_q, eta_ref, err_msg=f"{name} at {x}")
        np.testing.assert_array_equal(rad, r)
        np.testing.assert_array_equal(rho, np.asarray(L.profile.rho0))
        np.testing.assert_array_equal(cr, np.clip(x, lo, hi))


def test_c5_clamp_derivative():
    """C5: the derivative of ``cut_radius`` with respect to each movable radius is 1
    inside the clamp range, 0 outside it and 1/2 at a bound, and 0 for every fixed entry,
    in forward and in reverse mode."""
    L, B, eta = layers("FJ3")
    le = jnp.asarray(np.log(eta))
    lo, hi = L.radius_bounds[:, 0], L.radius_bounds[:, 1]
    f = lambda b: L(b, le).cut_radius
    for x, diag in ((L.start_radii, [1.0, 1.0]), (np.array([lo[0] - 1e3, hi[1] + 1e3]), [0.0, 0.0]),
                    (np.array([lo[0], hi[1]]), [0.5, 0.5])):
        b = B.copy(); b[L.movable_index] = x
        expect = np.zeros((2, B.size))
        expect[0, 1], expect[1, 2] = diag
        for jac in (jax.jacfwd, jax.jacrev):
            np.testing.assert_array_equal(np.asarray(jax.jit(jac(f))(jnp.asarray(b))), expect)


def test_c6_refusals():
    """C6: the refusals of ``Layers`` under "cut" and the new refusals."""
    p = input_profile(True)
    with pytest.raises(ValueError, match="motion must be"):
        sj.Layers(p, B_FJ1, movable={1: (5466.875e3, 5970e3)}, motion="slide")
    with pytest.raises(ValueError, match="density jump pair"):
        sj.Layers(p, B_FJ1, movable={1: (5466.875e3, 5970e3)}, motion="affine")
    with pytest.raises(ValueError, match="lies inside the movable range"):
        sj.Layers(p, np.array([3480, 5600, 5750, 5970, 6270, 6370]) * KM, movable={1: (5466.875e3, 5970e3)})
    with pytest.raises(ValueError, match="overlap"):
        sj.Layers(p, B_FJ3, movable={1: (5466.875e3, 5828.125e3), 2: (5647.5e3, 6150e3)})
    with pytest.raises(ValueError, match="radius of the movable boundary"):
        sj.Layers(p, B_FJ3, movable={1: (5466.875e3, 5900e3), 2: (5828.125e3, 6150e3)})
    with pytest.raises(ValueError, match="not a node radius"):
        sj.Layers(p, B_FJ1, movable={1: (5466.875e3, 5950e3)})
    with pytest.raises(ValueError, match="strictly inside its movable range"):
        sj.Layers(p, B_FJ1, movable={1: (5647.5e3, 5970e3)}, motion="cut")
    with pytest.raises(ValueError, match="cannot be the input of another"):
        sj.Layers(layers("FJ1")[0].profile, B_FJ1)
    with pytest.raises(ValueError, match="cannot be the input of another"):
        sj.Layers(layers("FJ1")[0].profile, B_FJ1, movable={1: (5466.875e3, 5970e3)}, motion="cut")


def test_c7_c10_density_map():
    """C7: the density map of a cut ``Layers`` is the static gather (``input_kernels``) and
    scatter (``sheet_mass``); C10: ``fields.apply`` accepts a cut profile and equals
    ``radial_integral`` with the trapezoid weights."""
    L, B, eta = layers("FJ1x")
    lmax = 6
    f, cfg = kernel_fn("FJ1x", "default", lmax)
    d = densities.rough(ref_input("FJ1x").radius, lmax, 3)
    lo, hi = L.radius_bounds[0]
    for x in (lo, 5612.3e3, J, J + 1.0, hi):
        q = L(boundaries_at("FJ1x", x), np.log(eta))
        K = sj.stokes_kernels(q, cfg)
        Kin = L.input_kernels(K)
        for o in SI_OUTPUTS:
            np.testing.assert_array_equal(np.asarray(getattr(Kin, o)), np.asarray(getattr(K, o))[:, L.input_node])
        Fd = L.fields(K, d)
        Fs = fields.radial_integral(K, L.sheet_mass(d, K), jnp.ones(L.n_nodes))
        for n in fields.FIELD_NAMES:
            a, b = np.asarray(getattr(Fs, n)), np.asarray(getattr(Fd, n))
            assert np.max(np.abs(a - b)) <= SHEET_TOL * np.max(np.abs(b)), n
        dc = densities.rough(np.asarray(q.radius), lmax, 4)
        Fa = fields.apply(K, dc, q)
        Fr = fields.radial_integral(K, dc, fields.quadrature_weights(q))
        for n in fields.FIELD_NAMES:
            a, b = np.asarray(getattr(Fa, n)), np.asarray(getattr(Fr, n))
            assert np.max(np.abs(a - b)) <= SHEET_TOL * np.max(np.abs(b)), n


def test_c8_crossings():
    """C8: under "cut" the crossings are the input radii inside the clamp range: kind 4 at
    a density-jump pair, kind 2 elsewhere; no kind 1 or 3."""
    for name in ("FJ1", "FJ2", "FJ3"):
        L = layers(name)[0]
        r_in = ref_input(name).radius
        rho = ref_input(name).rho0
        dens = r_in[:-1][(r_in[1:] == r_in[:-1]) & (rho[1:] != rho[:-1])]
        for q, k in enumerate(L.movable_index):
            lo, hi = L.radius_bounds[q]
            radii, kinds = L.crossings(k)
            inside = np.unique(r_in[(r_in >= lo) & (r_in <= hi)])
            np.testing.assert_array_equal(radii, inside)
            np.testing.assert_array_equal(radii[kinds == 4], np.intersect1d(inside, dens))
            assert set(np.unique(kinds)) <= {2, 4}


def test_c9_structure_profile_plan_refusals():
    """C9: the refusals of ``NodeStructure``, ``RadialProfile`` and ``static_plan`` for cuts."""
    r = ref_input("FJ1").radius
    with pytest.raises(ValueError, match="0 <= lo < hi"):
        NodeStructure.from_radius(r, consistent_core=False, cuts=((5, 5),))
    with pytest.raises(ValueError, match="0 <= lo < hi"):
        NodeStructure.from_radius(r, consistent_core=False, cuts=((5, r.size),))
    with pytest.raises(ValueError, match="must ascend"):
        NodeStructure.from_radius(r, consistent_core=False, cuts=((5, 10), (8, 12)))
    # Nodes 13 and 14 are the pair at J: a range must end on its inner member.
    for cuts in (((11, 14),), ((13, 16),)):
        with pytest.raises(ValueError, match="outer member of a jump pair"):
            NodeStructure.from_radius(r, consistent_core=False, cuts=cuts)
    NodeStructure.from_radius(r, consistent_core=False, cuts=((11, 13), (14, 16)))
    L = layers("FJ1")[0]
    q = L.profile
    with pytest.raises(ValueError, match="pass a structure with cuts"):
        sj.RadialProfile(np.asarray(q.radius), np.asarray(q.rho0), np.asarray(q.eta),
                         core_density=float(q.core_density), core_mass=q.core_mass, cut_radius=np.array([5600e3]))
    plain = input_profile(True)
    with pytest.raises(ValueError, match="no viscosity cut"):
        plain.replace(cut_radius=jnp.array([5600e3]))
    with pytest.raises(ValueError, match="no cut_radius"):
        q.replace(cut_radius=None)
    with pytest.raises(ValueError, match="one radius in m per cut"):
        q.replace(cut_radius=jnp.array([5600e3, 5700e3]))
    # Under jit: a profile unflattened without its cut leaf.
    children = [getattr(q, k) for k in ("radius", "rho0", "eta", "core_density", "core_mass", "outer_density",
                                        "surface_density", "uniform_gravity")] + [None]
    cfg = sj.StokesConfig(lmax=4)

    def without_leaf(e):
        prof = _unflatten(q.structure, children[:2] + [e] + children[3:])
        return sj.stokes_kernels(prof, cfg).geoid
    with pytest.raises(ValueError, match="no cut_radius"):
        jax.jit(without_leaf)(q.eta)
    for value in (5400e3, 5970e3, np.nan):
        with pytest.raises(ValueError, match="outside its range|not finite"):
            sj.stokes_kernels(q.replace(cut_radius=jnp.array([value])), cfg)
    with pytest.raises(ValueError, match="material rule"):
        propagate.static_plan(q.structure, cfg, propagate.Discretisation(material="upper_node"))
    with pytest.raises(ValueError, match="TALA rule"):
        propagate.static_plan(q.structure, sj.StokesConfig(lmax=4, formulation=FORMS["tala"]),
                              propagate.Discretisation(kappa_rule="legacy_one_sided"))


def test_c9_tala_warning():
    """C9 (warning): under TALA with a density pair inside a cut range ``solve`` warns on
    every call, also when the plan is a cache hit; the plan field is ``True``; under FULL
    with the same structure no ``RuntimeWarning`` is issued and the field is ``False``."""
    L, B, eta = layers("FJ1")
    q = L(B, np.log(eta))
    cfg_t = sj.StokesConfig(lmax=4, formulation=FORMS["tala"])
    with pytest.warns(RuntimeWarning, match="density jump"):
        sj.stokes_kernels(q, cfg_t)
    with pytest.warns(RuntimeWarning, match="density jump") as rec:
        sj.stokes_kernels(q, cfg_t)
    assert sum(issubclass(w.category, RuntimeWarning) for w in rec) == 1
    assert propagate.static_plan(q.structure, cfg_t).tala_cut_across_pair is True
    cfg_f = sj.StokesConfig(lmax=4)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sj.stokes_kernels(q, cfg_f)
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert propagate.static_plan(q.structure, cfg_f).tala_cut_across_pair is False


def _primitive_names(jaxpr) -> list[str]:
    """Every primitive of a jaxpr, also inside the sub-jaxprs (scan, cond, pjit bodies)."""
    out = []
    for eqn in jaxpr.eqns:
        out.append(eqn.primitive.name)
        for v in eqn.params.values():
            for sub in (v if isinstance(v, (list, tuple)) else [v]):
                inner = getattr(sub, "jaxpr", None)
                if inner is not None and hasattr(inner, "eqns"):
                    out += _primitive_names(inner)
                elif hasattr(sub, "eqns"):
                    out += _primitive_names(sub)
    return out


def test_c11_loop_free():
    """C11: the call (its gradient) has no scan, while or cond; the core of a cut profile
    has as many scans as the core of the same table without cuts (the two sweeps)."""
    L, B, eta = layers("FJ3")

    def f(b, le):
        q = L(b, le)
        return sum(jnp.sum(getattr(q, k)) for k in ("radius", "rho0", "eta", "cut_radius"))
    names = _primitive_names(jax.make_jaxpr(jax.grad(f, argnums=(0, 1)))(jnp.asarray(B), jnp.log(eta)).jaxpr)
    assert not {"scan", "while", "cond"} & set(names), set(names)
    q = L(B, np.log(eta))
    plain = sj.RadialProfile(np.asarray(q.radius), np.asarray(q.rho0), np.asarray(q.eta),
                             core_density=float(q.core_density), core_mass=q.core_mass)
    cfg = sj.StokesConfig(lmax=4)
    counts = []
    for prof in (q, plain):
        plan = propagate.static_plan(prof.structure, cfg)
        jx = jax.make_jaxpr(lambda p: propagate._core(p, plan, cfg, "largest"))(prof)
        counts.append(_primitive_names(jx.jaxpr).count("scan"))
    assert counts == [2, 2], counts


@pytest.mark.parametrize("form_name", ["default", "tala"])
def test_c12_construction_radius_independent(form_name):
    """C12: two ``Layers`` of FJ1 with different construction radii (5600 and 5533.3 km)
    give the same kernels at the same position, bit for bit."""
    La = layers("FJ1", "auto", 5600e3)[0]
    Lb = layers("FJ1", "auto", 5533.3e3)[0]
    cfg = sj.StokesConfig(lmax=6, formulation=FORMS[form_name])
    for x in (5612.3e3, J, 5850.703e3):
        Ka = sj.stokes_kernels(La(La.full_boundaries(np.array([x])), np.log(ETA_FJ1)), cfg)
        Kb = sj.stokes_kernels(Lb(Lb.full_boundaries(np.array([x])), np.log(ETA_FJ1)), cfg)
        for o in SI_OUTPUTS:
            np.testing.assert_array_equal(np.asarray(getattr(Ka, o)), np.asarray(getattr(Kb, o)))


# ===========================================================================
# 9.2 Solver values
# ===========================================================================


def _v1_positions(name):
    """The positions of V1 (not on a node), as joint vectors."""
    L = layers(name)[0]
    if name == "FJ3":
        s = L.start_radii
        return [np.array([J - 1.0, s[1]]), np.array([J + 1.0, s[1]]), np.array([s[0], J2 - 1.0]),
                np.array([s[0], J2 + 1.0]), np.array([L.radius_bounds[0, 1], L.radius_bounds[1, 0]]),
                np.array([5612.3e3, 6100.0e3]), s]
    return [np.array([v]) for k, v in positions(name).items() if not on_node(name, v)]


@pytest.mark.parametrize("name", ["FJ1", "FJ2", "FJ3"])
@pytest.mark.parametrize("form_name", ["default", "nsg_uniform", "no_slip"])
def test_v1_swapped_pair(name, form_name):
    """V1: the cut kernels at lmax 20 against the package solver on the swapped-pair table
    (plan 9.0) at every position that is not on a node, incompressible formulations."""
    worst = 0.0
    for x in _v1_positions(name):
        K = cut_kernels(name, form_name, 20, x)
        ref, keep = rc.swapped_pair_table(ref_input(name), boundaries_at(name, x), LAYOUTS[name][3], cut_index(name))
        worst = max(worst, worst_el(K, package_on(ref, form_name, 20, keep)))
    print(f"V1 {name} {form_name}: largest e_l {worst:.2e}")
    assert worst <= CUT_SWAP_TOL


@pytest.mark.parametrize("name,node,pair", [("FJ1", 5647.5e3, (J,)), ("FJ2", 5828.125e3, (J, J2))])
@pytest.mark.parametrize("form_name", ["default", "tala"])
def test_v1b_node_and_pair(name, node, pair, form_name):
    """V1b: the cut on an input node against the node-split table (FULL and TALA), and on
    each density pair against the pair table (FULL)."""
    T, eta = ref_input(name), LAYOUTS[name][3]
    ref, keep = rc.node_split_table(T, boundaries_at(name, node), eta, cut_index(name))
    e = worst_el(cut_kernels(name, form_name, 20, node), package_on(ref, form_name, 20, keep))
    print(f"V1b {name} {form_name} node: {e:.2e}")
    assert e <= CUT_SWAP_TOL
    if form_name == "default":
        for x in pair:
            ref, keep = rc.pair_table(T, boundaries_at(name, x), eta, cut_index(name))
            e = worst_el(cut_kernels(name, form_name, 20, x), package_on(ref, form_name, 20, keep))
            print(f"V1b {name} pair {x / 1e3} km: {e:.2e}")
            assert e <= CUT_SWAP_TOL


@pytest.mark.parametrize("eta_cut", [(1e24, 1e18), (1e18, 1e24)])
def test_v1c_contrast(eta_cut):
    """V1c: a viscosity contrast of 1e6 across the cut, FULL, lmax 20, against the
    swapped-pair table."""
    L, B, _ = layers("FJ1")
    eta = np.array([eta_cut[0], eta_cut[1], 3e20, 1e22])
    cfg = sj.StokesConfig(lmax=20)
    f = jax.jit(lambda x: L.input_kernels(sj.stokes_kernels(L(L.full_boundaries(x), jnp.log(eta)), cfg)))
    worst = 0.0
    for x in (5612.3e3, J - 1.0, J + 1.0, 5903.3e3):
        Kc = f(jnp.array([x]))
        K = {o: np.asarray(getattr(Kc, o)) for o in SI_OUTPUTS}
        ref, keep = rc.swapped_pair_table(ref_input("FJ1"), boundaries_at("FJ1", x), eta)
        worst = max(worst, worst_el(K, package_on(ref, "default", 20, keep)))
    print(f"V1c eta {eta_cut}: largest e_l {worst:.2e}")
    assert worst <= CUT_SWAP_TOL_CONTRAST


ORACLE_DEGREES = (1, 2, 10)

# V2, V3, V3b (plan 9.2): position name -> the concrete reference tables with the public
# oracle (the builders of reference_cut), per layout and formulation. Every position is
# also compared with cut_oracle_kernels (the check of the helper).
V2_CASES = [
    ("FJ1", "default", "5612.3", 5612.3e3, ("swapped",)),
    ("FJ1", "default", "J-1m", J - 1.0, ("swapped",)),
    ("FJ1", "default", "J+1m", J + 1.0, ("swapped",)),
    ("FJ1", "default", "node", 5647.5e3, ("node_split",)),
    ("FJ1", "default", "J", J, ("pair", "triple_up")),
    ("FJ2", "default", "J2-1m", J2 - 1.0, ("swapped",)),
    ("FJ2", "default", "J2+1m", J2 + 1.0, ("swapped",)),
    ("FJ2", "default", "J2", J2, ("pair", "triple_up")),
    ("FJ1", "tala", "node", 5647.5e3, ("node_split",)),
    ("FJ1", "tala", "J", J, ("triple_up",)),
    ("FJ1", "tala", "5612.3", 5612.3e3, ()),
    ("FJ1", "tala", "J-1m", J - 1.0, ()),
    ("FJ1", "tala", "J+1m", J + 1.0, ()),
    ("FJ1", "tala_nsg_uniform", "5612.3", 5612.3e3, ("geometric",)),
    ("FJ1", "tala_nsg_uniform", "J-1m", J - 1.0, ("geometric",)),
    ("FJ1", "tala_nsg_uniform", "J+1m", J + 1.0, ("geometric",)),
    ("FJ1", "tala_nsg_uniform", "5711", 5711e3, ("geometric",)),
]

_BUILDERS = {"swapped": rc.swapped_pair_table, "node_split": rc.node_split_table, "pair": rc.pair_table,
             "geometric": rc.geometric_pair_table,
             "triple_up": functools.partial(rc.triple_table, side="up"),
             "triple_lo": functools.partial(rc.triple_table, side="lo")}


def oracle_on(name, form_name, x, builder, degrees=ORACLE_DEGREES) -> dict:
    """The public oracle on a concrete reference table, the rows of the input nodes."""
    import reference_mpmath as ro
    ref, keep = _BUILDERS[builder](ref_input(name), boundaries_at(name, x), LAYOUTS[name][3], cut_index(name))
    O = ro.kernels(rc.as_node_table(ref), physics(form_name), ro.Discretisation(), degrees)
    return {o: np.asarray(O[o])[:, keep] for o in SI_OUTPUTS}


def cut_oracle_on(name, form_name, x, degrees=ORACLE_DEGREES) -> dict:
    """``cut_oracle_kernels`` on the numpy table without a cut."""
    T0, cuts = rc.cut_table(ref_input(name), boundaries_at(name, x), LAYOUTS[name][3], cut_index(name))
    return rc.cut_oracle_kernels(rc.as_node_table(T0), physics(form_name), cuts, degrees)


@pytest.mark.parametrize("name,form_name,label,x,builders", V2_CASES,
                         ids=[f"{c[0]}-{c[1]}-{c[2]}" for c in V2_CASES])
def test_v2_v3_v3b_v4_oracle(name, form_name, label, x, builders):
    """V2 (FULL), V3 (TALA), V3b (TALA with NO_SELF_GRAVITY and uniform g) and the negative
    control V4: the cut kernels at lmax 10, l in {1, 2, 10}, every source, against the
    public oracle on every concrete table of the position, and against
    ``cut_oracle_kernels`` (the only reference of TALA with self-gravity between nodes).
    V3b also compares the package solver on the geometric-pair table. V4: at a position
    between nodes, the solve of the same profile without its cut misses the reference by
    at least ``NEG_MIN``."""
    K = rows(cut_kernels(name, form_name, 10, x), ORACLE_DEGREES, degrees_of(10, form_name))
    refs = {b: oracle_on(name, form_name, x, b) for b in builders}
    refs["cut_oracle"] = cut_oracle_on(name, form_name, x)
    for b, ref in refs.items():
        e = worst_el(K, ref)
        print(f"{name} {form_name} {label}: against the oracle on {b}: {e:.2e}")
        assert e <= ORACLE_TOL, b
    if "geometric" in builders:
        ref, keep = rc.geometric_pair_table(ref_input(name), boundaries_at(name, x), LAYOUTS[name][3])
        Kp = rows(package_on(ref, form_name, 10, keep), ORACLE_DEGREES, degrees_of(10, form_name))
        e = worst_el(K, Kp)
        print(f"  package on the geometric-pair table: {e:.2e}")
        assert e <= CUT_SWAP_TOL
    if not on_node(name, x):
        L, _, eta = layers(name)
        q = L(boundaries_at(name, x), np.log(eta))
        plain = sj.RadialProfile(np.asarray(q.radius), np.asarray(q.rho0), np.asarray(q.eta),
                                 core_density=float(q.core_density), core_mass=q.core_mass)
        Kn = sj.stokes_kernels(plain, sj.StokesConfig(lmax=10, formulation=FORMS[form_name]))
        Kn = rows({o: np.asarray(getattr(Kn, o)) for o in SI_OUTPUTS}, ORACLE_DEGREES, degrees_of(10, form_name))
        e = worst_el(Kn, refs["cut_oracle"])
        print(f"  V4 without the cut step: {e:.2e}")
        assert e >= NEG_MIN


V7_POSITIONS = (5612.3e3, 5690e3, 5712.3e3)


def ck_input_kernels(name, x, form_name, lmax) -> dict:
    """CK: a ``Layers`` with the movable boundaries as FIXED pairs at ``x`` (chunk 4
    reference), its kernels at the input radii."""
    L, B, eta = layers(name)
    b = boundaries_at(name, x)
    Lc = sj.Layers(L.input_profile, b)
    K = Lc.input_kernels(sj.stokes_kernels(Lc(b, np.log(eta)), sj.StokesConfig(lmax=lmax,
                                                                                 formulation=FORMS[form_name])))
    return {o: np.asarray(getattr(K, o)) for o in SI_OUTPUTS}


@pytest.mark.parametrize("form_name", ["default", "no_slip", "nsg_uniform", "tala"])
def test_v7_against_ck(form_name):
    """V7: the cut against CK on the full t_prem table (T3-full) at lmax 10, at 5612.3 km,
    5690 km (J - 10 km) and 5712.3 km, none on a node or a density pair. On the full table
    because the two models differ by the rho0 of the two parts of the cut interval, which
    grows with the interval width (review r3 of the plan, issue 1)."""
    tol = CUT_CK_TOL_TALA if form_name == "tala" else CUT_CK_TOL
    worst = 0.0
    for x in V7_POSITIONS:
        worst = max(worst, worst_el(cut_kernels("T3full", form_name, 10, x), ck_input_kernels("T3full", x, form_name, 10)))
    print(f"V7 {form_name}: largest e_l against CK {worst:.2e}")
    assert worst <= tol


def full_cut_layout(which: str):
    """F1-full and F3-full (the chunk 4 G15q layouts on the full table) with motion="cut"."""
    p = input_profile(False)
    r = np.asarray(p.radius)
    if which == "F1":
        return sj.Layers(p, tca.B_F1, movable={1: (float(r[70]), 5700e3)}, motion="cut"), tca.ETA_F1
    i0 = int(np.flatnonzero(r == 5700e3)[-1])
    sA = 0.5 * (r[np.searchsorted(r, 4665e3) - 1] + r[np.searchsorted(r, 4665e3)])
    sC = 0.5 * (r[i0 + 5] + r[i0 + 6])
    return sj.Layers(p, np.array([3480e3, sA, 5370e3, 5700e3, sC, 5970e3, 6270e3, 6370e3]),
                     movable={1: (float(r[36]), float(r[70])), 2: (float(r[70]), 5700e3), 4: (5700e3, 5970e3)},
                     motion="cut"), tca.ETA_F3


@pytest.mark.parametrize("which", ["F1", "F3"])
def test_v8_fields_against_ck(which):
    """V8: F1-full and F3-full with motion="cut" at the chunk 4 G15q positions, lmax 10,
    smooth and rough anomaly: the four fields of ``Layers.fields`` against CK (chunk 4
    affine motion: 4.7e-4 and 1.6e-3)."""
    L, eta = full_cut_layout(which)
    if which == "F1":
        g = L.ranges[1]
        pos = [np.array([g.low + g.margin + 50.0]), np.array([g.start - 50e3]), np.array([g.start + 60e3]),
               np.array([g.high - g.margin - 50.0])]
    else:
        pos = tca._corners(L) + [L.start_radii + np.array([-120e3, 60e3, -40e3])]
    cfg = sj.StokesConfig(lmax=10)
    r_in = np.asarray(L.input_profile.radius)
    f = jax.jit(lambda x, d: L.fields(sj.stokes_kernels(L(L.full_boundaries(x), jnp.log(eta)), cfg), d))
    worst = 0.0
    for d in (densities.smooth(r_in, 10, 11), densities.rough(r_in, 10, 12)):
        for x in pos:
            Fd = f(jnp.asarray(x), jnp.asarray(d))
            Lf = tca._fresh(L, x)
            Ff = Lf.fields(sj.stokes_kernels(Lf(Lf.boundaries, np.log(eta)), cfg), d)
            worst = max(worst, max(tca.rel_lge2(getattr(Fd, n), getattr(Ff, n)) for n in fields.FIELD_NAMES))
    print(f"V8 {which}-full: largest field error against CK {worst:.2e}")
    assert worst <= CUT_CK_TOL


def test_v9_tie_at_density_pair():
    """V9: the tie at the density pair J on FJ1, lmax 10 (plan 6.1): (a) FULL, cut(J)
    equals the pair table (M-mean, the fixed boundary on the pair); (b) TALA, cut(J)
    differs from M-mean; (c) TALA, cut(J) is the limit from above; (d) TALA, cut(J - 1 mm)
    equals the oracle on the triple-lo table (M-above) to the change over 1 mm; (e) TALA,
    cut(J - 1 mm) differs from cut(J)."""
    T, eta = ref_input("FJ1"), ETA_FJ1
    ref, keep = rc.pair_table(T, boundaries_at("FJ1", J), eta)
    a = worst_el(cut_kernels("FJ1", "default", 10, J), package_on(ref, "default", 10, keep))
    KJ = cut_kernels("FJ1", "tala", 10, J)
    b = worst_el(KJ, package_on(ref, "tala", 10, keep))
    c = worst_el(KJ, cut_kernels("FJ1", "tala", 10, J + 1e-3))
    Km = cut_kernels("FJ1", "tala", 10, J - 1e-3)
    d = worst_el(rows(Km, (2, 10), degrees_of(10, "tala")), oracle_on("FJ1", "tala", J, "triple_lo", (2, 10)))
    e = worst_el(Km, KJ)
    print(f"V9: (a) {a:.2e} (b) {b:.2e} (c) {c:.2e} (d) {d:.2e} (e) {e:.2e}")
    assert a <= CUT_SWAP_TOL and b >= TALA_TIE_MIN and c <= TIE_SIDE_TOL and d <= TIE_SIDE_TOL
    assert e >= TALA_TIE_MIN


# ===========================================================================
# 9.3 Continuity of the derivative
# ===========================================================================


@functools.lru_cache(maxsize=None)
def geoid_jvp(name: str, form_name: str, lmax: int):
    """A jitted ``vmap`` over cut radii of (geoid input kernels, their derivative with
    respect to the radius of the first movable boundary)."""
    L, B, eta = layers(name)
    cfg = sj.StokesConfig(lmax=lmax, formulation=FORMS[form_name])
    le = jnp.asarray(np.log(eta))
    q0 = int(np.flatnonzero(L.movable_index == min(L.movable_index))[0])

    def geo(c):
        x = jnp.asarray(L.start_radii).at[q0].set(c)
        return L.input_kernels(sj.stokes_kernels(L(L.full_boundaries(x), le), cfg)).geoid
    return jax.jit(jax.vmap(lambda c: jax.jvp(geo, (c,), (jnp.ones_like(c),))))


def one_sided(name, form_name, lmax, x, D=1.0) -> dict:
    """The one-sided limits of the geoid field (smooth and rough anomaly, l >= 2) and of its
    derivative at ``x``, by linear extrapolation from ``x -+ D`` and ``x -+ 2 D``: ``rho``
    (the derivative jump over the mean derivative) and the value jump as a displacement of
    the boundary, ``|[N]| / |dN/dc|`` in m."""
    L = layers(name)[0]
    w = L.input_weights
    Kv, dK = map(np.asarray, geoid_jvp(name, form_name, lmax)(jnp.asarray([x - 2 * D, x - D, x + D, x + 2 * D])))
    Km, Kp = 2 * Kv[1] - Kv[0], 2 * Kv[2] - Kv[3]
    dm, dp = 2 * dK[1] - dK[0], 2 * dK[2] - dK[3]
    r_in = ref_input(name).radius
    out = {}
    for dn, d in (("smooth", densities.smooth(r_in, lmax, 11)), ("rough", densities.rough(r_in, lmax, 12))):
        first = lmax + 1 - Kv.shape[1]
        F = lambda Kr: np.einsum("lj,jclm->clm", Kr * w, d[:, :, first:, :])[:, 2 - first:, :]
        nrm = np.linalg.norm(0.5 * (F(dm) + F(dp)))
        out[dn] = (float(np.linalg.norm(F(dp) - F(dm)) / nrm), float(np.linalg.norm(F(Kp) - F(Km)) / nrm))
    return out


def test_k1_continuous_derivative_full():
    """K1: FULL, FJ1, lmax 10: at J, at the input node 5647.5 km and at r0 the derivative
    of the geoid field has no jump (``rho <= RHO_C1``) and the value no jump."""
    for label, x in (("J", J), ("node", 5647.5e3), ("r0", 5600e3)):
        for dn, (rho, vl) in one_sided("FJ1", "default", 10, x).items():
            print(f"K1 {label} {dn}: rho {rho:.2e}, value jump {vl:.2e} m")
            assert rho <= RHO_C1 and vl <= C1_JUMP_MAX_LENGTH, (label, dn, rho, vl)


def test_k2_k2b_tala():
    """K2: TALA, FJ1, lmax 10: the value jumps at J (as a displacement at least
    ``TALA_JUMP_MIN_LENGTH``), the derivative has a kink at the input node 5647.5 km; the
    solve warns. K2b: the value is continuous at the input nodes 5647.5 and 5828.125 km."""
    L, B, eta = layers("FJ1")
    with pytest.warns(RuntimeWarning, match="density jump"):
        sj.stokes_kernels(L(B, np.log(eta)), sj.StokesConfig(lmax=10, formulation=FORMS["tala"]))
    for dn, (rho, vl) in one_sided("FJ1", "tala", 10, J).items():
        print(f"K2 J {dn}: value jump {vl:.3e} m")
        assert vl >= TALA_JUMP_MIN_LENGTH
    for dn, (rho, vl) in one_sided("FJ1", "tala", 10, 5647.5e3).items():
        print(f"K2 node {dn}: rho {rho:.2e}")
        assert rho >= TALA_KINK_MIN
    for x in (5647.5e3, 5828.125e3):
        for dn, (rho, vl) in one_sided("FJ1", "tala", 10, x).items():
            print(f"K2b {x / 1e3} km {dn}: value jump {vl:.2e} m")
            assert vl <= C1_JUMP_MAX_LENGTH


def test_k3_one_sided_differences_at_r0():
    """K3: at r0 = 5600 km of FJ1 (between nodes), FULL: the derivative in reverse and
    forward mode against the forward and the backward three-point one-sided differences,
    h = 10 m (chunk 4 G4 had a kink at r0; the cut has none)."""
    L, B, eta = layers("FJ1")
    cfg = sj.StokesConfig(lmax=10)
    d = jnp.asarray(densities.smooth(ref_input("FJ1").radius, 10, 4))
    w = jnp.asarray(np.random.default_rng(6).standard_normal((2, 11, 11)))

    def F(x):
        q = L(L.full_boundaries(x[None]), jnp.log(eta))
        return jnp.sum(w * L.fields(sj.stokes_kernels(q, cfg), d).geoid)
    r0 = float(L.start_radii[0])
    value = jax.jit(F)
    h = FD_STEP
    v = [float(value(jnp.asarray(r0 + t))) for t in (0.0, h, 2 * h, -h, -2 * h)]
    Dp = (-3 * v[0] + 4 * v[1] - v[2]) / (2 * h)
    Dm = (3 * v[0] - 4 * v[3] + v[4]) / (2 * h)
    for mode, D in (("reverse", float(jax.jit(jax.grad(F))(jnp.asarray(r0)))),
                    ("forward", float(jax.jit(lambda x: jax.jvp(F, (x,), (jnp.ones(()),))[1])(jnp.asarray(r0))))):
        print(f"K3 {mode}: |D - D+|/|D| {abs(D - Dp) / abs(D):.2e}, |D - D-|/|D| {abs(D - Dm) / abs(D):.2e}")
        assert abs(D - Dp) <= FD_ONE_SIDED_TOL * abs(D) and abs(D - Dm) <= FD_ONE_SIDED_TOL * abs(D)


# ===========================================================================
# 9.4 Taylor cases and dot products
# ===========================================================================


# The Taylor functional (plan 9.4; review r3 of the plan, issue 2): the geoid misfit of
# the field of a smooth density through ``Layers.fields``, with the ``spectral_misfit``
# weights of chunk 4 G16 (1 for l >= 2, normalised by the power of the target), and the
# target built at a radius far from every base point of the program. A target built at
# the base point leaves a residual of 1 % of the field there, and a 1 % bias of the
# derivative then hides below the second-order remainder at the judged steps (the 1.01
# control passes). The quadratic misfit is used in place of the linear functional of
# chunk 4 (``Program``) because the cut map is C1 and not C2 at a density pair (the
# second derivative of the kernels jumps with the interval density): for the linear
# functional that jump dominates the corrected remainder of a base point 20 m below J
# (T29), while the misfit's second derivative is dominated by the Gauss-Newton part
# ``|dN/dc|^2``, which is continuous (impl-evidence/taylor_functional.out).
TARGET_RADII = {"FJ1": (5520e3,), "FJ2": (5880e3,), "FJ3": (5520e3, 6080e3), "T3full": (5520e3,),
                "T4full": (5880e3,)}
TARGET_MIN_DISTANCE = 50e3


@functools.lru_cache(maxsize=None)
def misfit_program(name: str, form_name: str, lmax: int):
    """``(TaylorProgram, c0)`` of the geoid misfit of a layout (above); controls ``radii``,
    ``log_eta`` and ``rho0``. Cached per worker, so the cases of one program compile once."""
    L, B, eta = layers(name)
    cfg = sj.StokesConfig(lmax=lmax, formulation=FORMS[form_name])
    d = jnp.asarray(densities.smooth(ref_input(name).radius, lmax, 4))

    def geoid(c):
        q = L(c["radii"], c["log_eta"], rho0=c["rho0"])
        return L.fields(sj.stokes_kernels(q, cfg), d).geoid
    c0 = dict(radii=np.asarray(B, dtype=np.float64), log_eta=np.log(eta), rho0=np.asarray(L.profile.rho0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        obs = jax.jit(geoid)(base_at(c0, B, L, TARGET_RADII[name]))
    wl = np.where(np.arange(lmax + 1) >= 2, 1.0, 0.0)
    wl = jnp.asarray(0.5 * wl / float(jnp.sum(obs[:, 2:, :] ** 2)))

    def f(c):
        return fields.spectral_misfit(geoid(c), obs, wl, "real_4pi")
    return TaylorProgram(f), c0


def assert_taylor_controls(tp, x, dx, name):
    """``assert_taylor`` (both signs, both modes), then the four 1.01 controls must fail;
    the base point is at least ``TARGET_MIN_DISTANCE`` from the target radius."""
    L = layers(name)[0]
    dist = np.min(np.abs(np.asarray(x["radii"])[L.movable_index] - np.asarray(TARGET_RADII[name])))
    assert dist >= TARGET_MIN_DISTANCE, dist
    res = assert_taylor(tp, x, dx)
    verdicts = [assess(remeasure_taylor(r, 1.01 * r.derivative))[0] for r in res]
    assert all(v != "pass" for v in verdicts), verdicts
    rates = [float(np.nanmin(r.rates1[:3])) for r in res]
    print(f"Taylor pass, smallest corrected rate of the window {min(rates):.3f}; 1.01 controls {verdicts}")
    return res


@functools.lru_cache(maxsize=None)
def program(name: str, form_name: str, lmax: int):
    """The chunk 4 ``Program`` (a map of the four kernels and the four fields of
    ``Layers.fields``, with fixed random weights normalised at the construction point)
    of a layout, for the dot products: controls ``radii``, ``log_eta`` and ``rho0``. Cached
    per worker."""
    L, B, eta = layers(name)
    cfg = sj.StokesConfig(lmax=lmax, formulation=FORMS[form_name])
    d = densities.smooth(ref_input(name).radius, lmax, 4)

    def to_profile(c):
        return L(c["radii"], c["log_eta"], rho0=c["rho0"])
    c0 = dict(radii=np.asarray(B, dtype=np.float64), log_eta=np.log(eta), rho0=np.asarray(L.profile.rho0))
    return tca.Program(L, cfg, to_profile, c0, d), c0


def base_at(c0, B, L, x):
    b = np.asarray(B, dtype=np.float64).copy()
    b[L.movable_index] = np.atleast_1d(x)
    return dict(c0, radii=b)


def radius_direction(c0, L, amp=RADIUS_DIRECTION):
    dr = np.zeros_like(c0["radii"])
    dr[L.movable_index] = amp
    return tca.direction(c0, radii=dr)


TAYLOR_CASES = {
    # case: (layout, formulation, lmax, base radius (None: the construction radii))
    "T28": ("FJ1", "default", 20, J),
    "T29": ("FJ1", "default", 20, J - 20.0),
    "T30": ("FJ1", "default", 20, 5600e3),
    "T31": ("FJ1", "default", 20, 5647.5e3),
    "T33": ("FJ2", "default", 20, J2),
    "T34": ("FJ3", "default", 20, (J, J2)),
}


@pytest.mark.parametrize("case", list(TAYLOR_CASES))
def test_taylor_radius(case):
    """T28 to T31, T33, T34: second-order Taylor convergence along the movable radii
    (5000 m) for both signs in both modes, with the 1.01 controls failing; FULL, through
    and next to the density pairs, at r0 and at an input node."""
    name, form_name, lmax, x = TAYLOR_CASES[case]
    tp, c0 = misfit_program(name, form_name, lmax)
    L, B, _ = layers(name)
    assert_taylor_controls(tp, base_at(c0, B, L, x), radius_direction(c0, L), name)


def tala_base(name: str) -> float:
    """T35: the middle of the input interval above J, at least CROSSING_CLEARANCE x 50 m
    from every input radius."""
    x = midpoint_above(name, J)
    r = ref_input(name).radius
    assert np.min(np.abs(r - x)) >= CROSSING_CLEARANCE * 1e-2 * RADIUS_DIRECTION
    return x


@pytest.mark.parametrize("case", ["T32", "T35"])
def test_taylor_all_controls(case):
    """T32 (FULL, middle of an input interval) and T35 (TALA, 1 km clearance from every
    input radius): the radius direction alone, and the radius with ``log_eta`` (0.1 N(0, 1))
    and ``rho0`` (50 N(0, 1) kg/m^3 per node) together (review r3 of the plan, issue 6)."""
    form_name = "default" if case == "T32" else "tala"
    tp, c0 = misfit_program("FJ1", form_name, 20)
    L, B, _ = layers("FJ1")
    base = base_at(c0, B, L, tala_base("FJ1"))
    assert_taylor_controls(tp, base, radius_direction(c0, L), "FJ1")
    rng = np.random.default_rng(32)
    dx = radius_direction(c0, L)
    dx.update(log_eta=0.1 * rng.standard_normal(c0["log_eta"].size), rho0=50.0 * rng.standard_normal(L.n_nodes))
    assert_taylor_controls(tp, base, dx, "FJ1")


def test_t35n_tala_at_pair():
    """T35n: TALA at c = J: the value jumps there, so sign +1 (from the limit from above,
    the value at the tie) passes and sign -1 fails in both modes."""
    tp, c0 = misfit_program("FJ1", "tala", 20)
    L, B, _ = layers("FJ1")
    base = base_at(c0, B, L, J)
    dx = radius_direction(c0, L)
    assert_taylor(tp, base, dx, signs=(1.0,))
    for mode in ("reverse", "forward"):
        verdict, reason = assess(taylor_remainders(tp, base, dx, sign=-1.0, mode=mode))
        print(f"T35n sign -1 {mode}: {verdict} ({reason})")
        assert verdict == "fail"


@pytest.mark.parametrize("case", ["FJ1-J", "FJ3", "FJ1-tala"])
def test_dp4_dot_products(case):
    """DP4: the jvp/vjp dot product on the normalised map: FJ1 FULL at J, FJ3 FULL at the
    construction radii, FJ1 TALA at the T35 point."""
    name, form_name, x = {"FJ1-J": ("FJ1", "default", J), "FJ3": ("FJ3", "default", None),
                          "FJ1-tala": ("FJ1", "tala", None)}[case]
    prog, c0 = program(name, form_name, 20)
    L, B, _ = layers(name)
    if case == "FJ1-tala":
        x = tala_base("FJ1")
    base = c0 if x is None else base_at(c0, B, L, x)
    rng = np.random.default_rng(41)
    dr = np.zeros(B.size)
    dr[L.movable_index] = 1000.0 * rng.standard_normal(L.movable_index.size)
    dx = dict(radii=dr, log_eta=rng.standard_normal(c0["log_eta"].size), rho0=50.0 * rng.standard_normal(L.n_nodes))
    rel = assert_jvp_vjp(prog.dot, base, dx=dx, dy=prog.cotangent(42), rtol=DOT_RTOL)
    print(f"DP4 {case}: {rel:.2e}")


@functools.lru_cache(maxsize=None)
def control_vector_fn(name: str):
    """``(ControlVector, fn, density)`` over the movable radii and ``log_eta`` of a layout,
    FULL, lmax 20: a functional linear in the geoid field and the surface topography."""
    L, B, eta = layers(name)
    cfg = sj.StokesConfig(lmax=20)
    d = jnp.asarray(densities.smooth(ref_input(name).radius, 20, 4))
    cv = sj.ControlVector({"radii_movable": L.start_radii.copy(), "log_eta": np.log(eta)})
    rng = np.random.default_rng(9)
    w = jnp.asarray(rng.standard_normal((2, 21, 21)))

    def fn(c, d):
        q = L(L.full_boundaries(c["radii_movable"]), c["log_eta"])
        Fd = L.fields(sj.stokes_kernels(q, cfg), d)
        return jnp.sum(w * Fd.geoid) * 1e3 + jnp.sum(w * Fd.surface_topography)
    return cv, fn, d


def test_vmap_eager_jit():
    """vmap of the value and the gradient over 8 positions of FJ1 on both sides of J
    against single calls; the eager gradient against the jitted one at J and at a node."""
    cv, fn, d = control_vector_fn("FJ1")
    L, B, eta = layers("FJ1")
    xs = [J - 30e3, J - 1.0, J, J + 1.0, J + 7e3, J + 60e3, 5550e3, 5647.5e3]
    X = np.stack([np.asarray(cv.ravel({"radii_movable": jnp.array([x]), "log_eta": jnp.asarray(np.log(eta) + 0.05 * k)}))
                  for k, x in enumerate(xs)])
    single = cv.value_and_grad(fn)
    vals, grads = zip(*[(float(v), np.asarray(g)) for v, g in (single(jnp.asarray(x), d) for x in X)])
    vv, gv = jax.jit(jax.vmap(single, in_axes=(0, None)))(jnp.asarray(X), d)
    assert np.max(np.abs(np.asarray(vv) - vals)) <= VMAP_RTOL * np.max(np.abs(vals))
    G = np.stack(grads)
    assert np.max(np.abs(np.asarray(gv) - G)) <= VMAP_RTOL * np.max(np.abs(G))
    g_eager = jax.grad(lambda v: fn(cv.unravel(v), d))
    for x in (X[2], X[7]):
        ge = np.asarray(g_eager(jnp.asarray(x)))
        gj = np.asarray(single(jnp.asarray(x), d)[1])
        assert np.max(np.abs(ge - gj)) <= JIT_EAGER_RTOL * np.max(np.abs(gj))


def test_control_vector_fj3_one_trace():
    """``ControlVector`` with the two movable radii of FJ3: one jitted value-and-grad; the
    first call adds at most one trace of the core, a second call with new values adds
    none (CONVENTIONS 7, the trace-count rule)."""
    cv, fn, d = control_vector_fn("FJ3")
    L = layers("FJ3")[0]
    f = cv.value_and_grad(fn)
    x0 = np.asarray(cv.ravel({"radii_movable": L.start_radii, "log_eta": np.log(ETA_FJ1)}))
    n0 = propagate.trace_count()
    v0, g0 = f(jnp.asarray(x0), d)
    assert propagate.trace_count() <= n0 + 1
    n1 = propagate.trace_count()
    x1 = np.asarray(cv.ravel({"radii_movable": jnp.array([J + 1.0, J2 - 1.0]), "log_eta": jnp.log(ETA_FJ1)}))
    v1, g1 = f(jnp.asarray(x1), d)
    assert propagate.trace_count() == n1 and f._cache_size() == 1
    assert np.all(np.isfinite(np.asarray(g0))) and np.all(np.isfinite(np.asarray(g1)))
    radii_part = np.asarray(cv.unravel(jnp.asarray(g1))["radii_movable"])
    assert np.all(radii_part != 0.0)


# ===========================================================================
# Slow rows (the full tier: each repeats a named quick row on a full table or at a higher
# degree; CONVENTIONS 7).
# ===========================================================================


def _v5_cases():
    """(layout, formulation, position name, radius, builders) of V5: T3-full and T4-full,
    the position list of plan 9 (T4 around both jumps), five formulations."""
    out = []
    for name in ("T3full", "T4full"):
        jumps = (J,) if name == "T3full" else (J, J2)
        for form_name in ("default", "tala", "nsg_uniform", "tala_nsg_uniform", "no_slip"):
            pos = []
            for j in jumps:
                pos += [(f"{j / 1e3:.0f}-10km", j - 10e3), (f"{j / 1e3:.0f}-1m", j - 1.0),
                        (f"{j / 1e3:.0f}+1m", j + 1.0), (f"{j / 1e3:.0f}+10km", j + 10e3), (f"{j / 1e3:.0f}", j)]
            pos += [("node", 5647.5e3)]
            for label, x in pos:
                on_pair = x in jumps
                on_input_node = label == "node"
                if on_pair:
                    builders = ("triple_up",) if form_name.startswith("tala") else ("pair", "triple_up")
                elif on_input_node:
                    builders = ("node_split",)
                elif form_name == "tala":
                    builders = ()
                elif form_name == "tala_nsg_uniform":
                    builders = ("geometric",)
                else:
                    builders = ("swapped",)
                out.append((name, form_name, label, x, builders))
    return out


V5_CASES = _v5_cases()
V5_DEGREES = (1, 2, 10, 30, 50)


@pytest.mark.slow
@pytest.mark.parametrize("name,form_name,label,x,builders", V5_CASES,
                         ids=[f"{c[0]}-{c[1]}-{c[2]}" for c in V5_CASES])
def test_v5_oracle_full_tables(name, form_name, label, x, builders):
    """V5 (full tier of V2, V3, V3b): T3-full and T4-full, lmax 50, l in {1, 2, 10, 30, 50},
    every source, against the public oracle on every concrete table, and against
    ``cut_oracle_kernels`` for TALA with self-gravity between nodes."""
    # Degree 1 is refused under a no-slip boundary with a split core (the frame rule of
    # the package and of the oracle); the kernels then start at l = 2.
    have = degrees_of(50, form_name)
    degrees = tuple(l for l in V5_DEGREES if l in have)
    K = rows(cut_kernels(name, form_name, 50, x), degrees, have)
    refs = {b: oracle_on(name, form_name, x, b, degrees) for b in builders}
    if not builders:
        refs["cut_oracle"] = cut_oracle_on(name, form_name, x, degrees)
    for b, ref in refs.items():
        e = worst_el(K, ref)
        print(f"V5 {name} {form_name} {label}: against the oracle on {b}: {e:.2e}")
        assert e <= ORACLE_TOL, b


@pytest.mark.slow
@pytest.mark.parametrize("form_name", ["default", "no_slip"])
def test_v1_100_swapped_pair_lmax100(form_name):
    """V1-100 (full tier of V1): T3-full, lmax 100: the cut 1 mm from the input node above
    J, at J + 1 mm and in the middle of an input interval, and with 1e18 | 1e24 Pa s across
    the cut at one position, against the swapped-pair table, all degrees."""
    r = ref_input("T3full").radius
    above = float(r[r > J][0])
    worst = 0.0
    for x in (above - 1e-3, J + 1e-3, midpoint_above("T3full", J)):
        K = cut_kernels("T3full", form_name, 100, x)
        ref, keep = rc.swapped_pair_table(ref_input("T3full"), boundaries_at("T3full", x), ETA_FJ1)
        worst = max(worst, worst_el(K, package_on(ref, form_name, 100, keep)))
    print(f"V1-100 {form_name}: {worst:.2e}")
    assert worst <= CUT_SWAP_TOL
    L, B, _ = layers("T3full")
    eta = np.array([1e18, 1e24, 3e20, 1e22])
    cfg = sj.StokesConfig(lmax=100, formulation=FORMS[form_name])
    x = midpoint_above("T3full", J)
    Kc = L.input_kernels(sj.stokes_kernels(L(boundaries_at("T3full", x), np.log(eta)), cfg))
    ref, keep = rc.swapped_pair_table(ref_input("T3full"), boundaries_at("T3full", x), eta)
    e = worst_el({o: np.asarray(getattr(Kc, o)) for o in SI_OUTPUTS}, package_on(ref, form_name, 100, keep))
    print(f"V1-100 {form_name} contrast 1e6: {e:.2e}")
    assert e <= CUT_SWAP_TOL_CONTRAST


V6_POSITIONS = (5600e3, 5690e3, 5699e3, 5699.999e3, 5700.001e3, 5701e3, 5710e3, 5800e3, 5953e3)


@pytest.mark.slow
@pytest.mark.parametrize("form_name", ["default", "no_slip", "nsg_uniform", "tala"])
def test_v6_against_ck_lmax30(form_name):
    """V6 (full tier of V7): T3-full, lmax 30, the nine positions of the plan (none on a
    node or a density pair), against CK."""
    tol = CUT_CK_TOL_TALA if form_name == "tala" else CUT_CK_TOL
    worst = 0.0
    for x in V6_POSITIONS:
        worst = max(worst, worst_el(cut_kernels("T3full", form_name, 30, x), ck_input_kernels("T3full", x, form_name, 30)))
    print(f"V6 {form_name}: {worst:.2e}")
    assert worst <= tol


@pytest.mark.slow
@pytest.mark.parametrize("name", ["T3full", "T4full"])
def test_k4_continuous_derivative_full_tables(name):
    """K4 (full tier of K1): T3-full and T4-full, lmax 50, FULL: at the density pairs and at
    three input nodes inside the range."""
    r = ref_input(name).radius
    L = layers(name)[0]
    lo, hi = L.radius_bounds[0]
    singles = [x for x in np.unique(r[(r > lo) & (r < hi)]) if np.sum(r == x) == 1]
    points = [J] + ([J2] if name == "T4full" else []) + [singles[3], singles[len(singles) // 2], singles[-3]]
    for x in points:
        for dn, (rho, vl) in one_sided(name, "default", 50, x).items():
            print(f"K4 {name} {x / 1e3:.3f} km {dn}: rho {rho:.2e}, value jump {vl:.2e} m")
            assert rho <= RHO_C1 and vl <= C1_JUMP_MAX_LENGTH


SLOW_TAYLOR = {
    "S-T28": ("T3full", "default", J),
    "S-T33": ("T4full", "default", J2),
    "S-T35": ("T3full", "tala", None),
}


@pytest.mark.slow
@pytest.mark.parametrize("case", list(SLOW_TAYLOR))
def test_slow_taylor_full_tables(case):
    """S-T28, S-T33, S-T35 (full tier of T28, T33, T35): lmax 50 on the full tables."""
    name, form_name, x = SLOW_TAYLOR[case]
    tp, c0 = misfit_program(name, form_name, 50)
    L, B, _ = layers(name)
    if x is None:
        x = tala_base(name)
    assert_taylor_controls(tp, base_at(c0, B, L, x), radius_direction(c0, L), name)
