"""Controls without a solver derivative: ``Layers`` construction, the call and the clamp,
the density map, ``viscosity_by_node``, ``ControlVector``, ``spectral_misfit`` and the
uniform-gravity leaf (gates G1, G2, G8, G15 rows of section 5 of
NOTES/chunks/c4-controls-derivatives/01-plan.md).

Fixtures (plan section 5), all on the reduced PREM table ``cases.t_prem(reduced=True)``
(25 input nodes, split core):

- F1: boundaries 3480, 5370, 5700, 5970, 6270, 6370 km; 5370 km is inserted between
  nodes and movable in (5105.625, 5700) km; 5700, 5970, 6270 km are existing pairs.
- F3: boundaries 3480, 4650, 5370, 5700, 5900, 5970, 6270, 6370 km; three movable
  boundaries with ranges A (4202.5, 5105.625), B (5105.625, 5700) and C (5700, 5970) km.
  A and B share an end node; the lower end of C is the upper member of the 670 km
  density-jump pair, and the upper end of B its lower member.
- F-node: boundary 1 on the input node 5286.25 km, movable in (5105.625, 5700) km, and
  boundary 4 on the input viscosity-only pair 6270 km, movable in (6189.375, 6370) km.

The references loop over the ranges in numpy and read ``rho0`` from the INPUT profile
(not from the knot table of the code), so that an index error of the vectorised map of
several ranges shows (plan risk R10).

Every ``Layers`` with a movable boundary in this file passes ``motion="affine"``, because
the file tests the affine method (the moved nodes, the knot table, the static limits).
The tests of the default method, the viscosity cut, are in ``tests/test_controls_cut.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import cases  # noqa: E402
import densities  # noqa: E402
from adapters import SI_OUTPUTS, assert_rows_close, profile_from_table  # noqa: E402

import stokesjx as sj  # noqa: E402
from stokesjx import controls, fields, propagate  # noqa: E402

EPS = np.finfo(np.float64).eps
KM = 1e3
# Equality of a multiply-add between numpy and XLA (FMA contraction can change the last
# bit; plan risk R7): radii and rho0 against the numpy references.
FMA_TOL = 4 * EPS
# The kappa_h limit (a logarithm of a density ratio, dimensionless, at most 0.07 here)
# against the maximum recomputed from the input profile: absolute, a few roundings of the
# two densities and of the logarithm.
KAPPA_LIMIT_TOL = 4 * EPS
# exp(log(eta)) against eta: the logarithm of a viscosity of 1e19 to 1e23 Pa s is 44 to
# 53, its rounding is up to 4e-15 absolute, and exp turns that into a relative error of
# the same size (measured 3.3e-15 = 15 eps at 1e19 Pa s). lovejx states "a few units in
# the last place" for the same round trip; the plan's "2 ulp" is not reachable.
EXP_LOG_TOL = 64 * EPS
# vmap against single calls of the density map (the c2 constant VMAP_RTOL; the cause is
# the FMA contraction of the radii under vmap, amplified by the bracket weight).
VMAP_RTOL = 1e-12
# The inserted pair values against the D4 interpolation of the neighbours.
INSERT_TOL = 2 * EPS
# Kernels of the independent layering against the stored oracle (CONVENTIONS 10).
ORACLE_TOL = 1e-12
# The two forms of a uniform background (plan 5.4): the same arithmetic up to the fusion
# order. Under NO_SELF_GRAVITY only: under FULL the two forms differ by 1.5e-14 to
# 2.8e-14 (review r2 of the plan, extra_checks.out part 2), so this constant does not
# apply there.
UNIFORM_LEAF_TOL = 1e-15
# spectral_misfit against a numpy loop over l, m, C/S (rounding of a sum of a few
# hundred squares).
MISFIT_TOL = 1e-14
# Gradient of the misfit with respect to the field against 2 w_l d.
MISFIT_GRAD_TOL = 1e-15
# The transposed form of the density map against the kernel form (two summation orders).
SHEET_TOL = 1e-14
# The jvp of the density map at a tie against a forward difference over 1 mm: the
# truncation of a rational function of the radius over 1 mm and the rounding
# eps / 1e-3 m.
TIE_FD_TOL = 1e-6

ETA_F1 = np.array([3e22, 1e22, 1e21, 3e20, 1e22])
ETA_F3 = np.array([1e23, 3e22, 1e22, 1e21, 3e21, 3e20, 1e22])
ETA_FNODE = np.array([3e22, 1e22, 1e21, 3e20, 1e22])
B_F1 = np.array([3480, 5370, 5700, 5970, 6270, 6370]) * KM
B_F3 = np.array([3480, 4650, 5370, 5700, 5900, 5970, 6270, 6370]) * KM
B_FNODE = np.array([3480, 5286.25, 5700, 5970, 6270, 6370]) * KM
MOV_F1 = {1: (5105.625 * KM, 5700 * KM)}
MOV_F3 = {1: (4202.5 * KM, 5105.625 * KM), 2: (5105.625 * KM, 5700 * KM), 4: (5700 * KM, 5970 * KM)}
MOV_FNODE = {1: (5105.625 * KM, 5700 * KM), 4: (6189.375 * KM, 6370 * KM)}


@pytest.fixture(scope="module")
def p_red():
    return profile_from_table(cases.t_prem(reduced=True))


@pytest.fixture(scope="module")
def F1(p_red):
    return sj.Layers(p_red, B_F1, movable=MOV_F1, motion="affine")


@pytest.fixture(scope="module")
def F3(p_red):
    return sj.Layers(p_red, B_F3, movable=MOV_F3, motion="affine")


@pytest.fixture(scope="module")
def Fnode(p_red):
    return sj.Layers(p_red, B_FNODE, movable=MOV_FNODE, motion="affine")


@pytest.fixture(scope="module")
def S1():
    """The S1 benchmark layering (plan section 7): the 129-node golden table with the
    hager4 boundaries at 100, 400 and 670 km inserted as pairs (135 nodes), the 670 km
    boundary movable in (5466.875, 5970) km; the 400 km pair is an inserted FIXED pair
    and the upper range end (review r3 minor 9)."""
    # The node table of the golden run hager4_fsfs (129 LEBEDEV nodes, hager4
    # viscosity, rho_c = 9900, consistent core), as a profile with the default flags.
    p = profile_from_table(cases.golden_table("hager4_fsfs"))
    return sj.Layers(p, np.array([3480, 5700, 5970, 6270, 6370]) * KM, movable={1: (5466.875 * KM, 5970 * KM)},
                     motion="affine")


def _fixtures(request, names):
    return [request.getfixturevalue(n) for n in names]


# --------------------------------------------------------------------------
# numpy references that loop over the ranges.
# --------------------------------------------------------------------------


def reference_profile(L: sj.Layers, new_radii):
    """Moved radii by the two affine maps per range and rho0 from the INPUT profile
    (np.interp on the input radii of the range, the inner member of a pair at each end),
    with a Python loop over the ranges. Independent of the knot table of the code."""
    r_in = np.asarray(L.input_profile.radius)
    rho_in = np.asarray(L.input_profile.rho0)
    rad = np.asarray(L.profile.radius).copy()
    rho = np.asarray(L.profile.rho0).copy()
    for q, k in enumerate(L.movable_index):
        g = L.ranges[k]
        nb = float(np.clip(new_radii[q], g.low + g.margin, g.high - g.margin))
        rad[g.below] = g.low + (rad[g.below] - g.low) * ((nb - g.low) / (g.start - g.low))
        rad[g.above] = g.high - (g.high - rad[g.above]) * ((g.high - nb) / (g.high - g.start))
        rad[list(g.members)] = nb
        ilo = int(np.flatnonzero(r_in == g.low)[-1])
        ihi = int(np.flatnonzero(r_in == g.high)[0]) if np.any(r_in == g.high) else None
        idx = np.concatenate([g.below, list(g.members), g.above]).astype(int)
        if ihi is None:
            # A range end at a fixed inserted pair: the input function there is the
            # interpolation between its two input neighbours, which np.interp gives.
            xs, fs = r_in[ilo:], rho_in[ilo:]
        else:
            xs, fs = r_in[ilo:ihi + 1], rho_in[ilo:ihi + 1]
        rho[idx] = np.interp(rad[idx], xs, fs)
    return rad, rho


def reference_bracket(L: sj.Layers, radius):
    """Per range, ``a = lo + searchsorted(x[lo:hi + 1], r_i, "left") - 1`` and its weight."""
    radius = np.asarray(radius)
    r_in = np.asarray(L.input_profile.radius)
    a, t = [], []
    for q, k in enumerate(L.movable_index):
        g = L.ranges[k]
        xs = radius[g.lo_node:g.hi_node + 1]
        for ri in r_in[(r_in > g.low) & (r_in < g.high)]:
            j = g.lo_node + int(np.searchsorted(xs, ri, side="left")) - 1
            a.append(j)
            t.append((ri - radius[j]) / (radius[j + 1] - radius[j]))
    return np.asarray(a), np.asarray(t)


def joint_positions(L, n, seed):
    rng = np.random.default_rng(seed)
    lo, hi = L.radius_bounds[:, 0], L.radius_bounds[:, 1]
    return lo + (hi - lo) * rng.random((n, lo.size))


def primitives(jaxpr) -> set:
    """Every primitive name of a jaxpr, sub-jaxprs included."""
    out = set()
    for eqn in jaxpr.eqns:
        out.add(eqn.primitive.name)
        for v in eqn.params.values():
            for sub in (v if isinstance(v, (list, tuple)) else [v]):
                if hasattr(sub, "jaxpr"):
                    out |= primitives(sub.jaxpr if hasattr(sub.jaxpr, "eqns") else sub.jaxpr.jaxpr)
                elif hasattr(sub, "eqns"):
                    out |= primitives(sub)
    return out


def fake_kernels(L: sj.Layers, radius, seed=1, L_deg=5):
    """A ``Kernels`` record with random rows (degrees 1..5) on the construction table and
    the given radius, for the density map without a solver (plan 5.5)."""
    rng = np.random.default_rng(seed)
    arr = [jnp.asarray(rng.standard_normal((L_deg, L.n_nodes))) for _ in range(4)]
    return sj.Kernels(degrees=tuple(range(1, L_deg + 1)), degree_one_frame=sj.DegreeOneFrame.CENTRE_OF_MASS,
                      radius=jnp.asarray(radius), geoid=arr[0], surface_topography=arr[1], cmb_topography=arr[2],
                      surface_velocity=arr[3], degree_one_imbalance=jnp.asarray(rng.standard_normal(L.n_nodes)))


# ==========================================================================
# 5.1 Construction
# ==========================================================================


def test_node_count(p_red, F1):
    n_in = p_red.structure.n_nodes
    assert F1.n_nodes == n_in + 2                                      # one inserted pair
    assert sj.Layers(p_red, np.array([3480, 5700, 6370]) * KM).n_nodes == n_in       # an existing pair
    assert sj.Layers(p_red, np.array([3480, 5286.25, 6370]) * KM).n_nodes == n_in + 1  # a single node


def test_inserted_pair_values(p_red, F1):
    r_in, rho_in, eta_in = (np.asarray(getattr(p_red, k)) for k in ("radius", "rho0", "eta"))
    lower, upper = F1.ranges[1].members
    b = 5370 * KM
    j = int(np.searchsorted(r_in, b)) - 1
    for name, col in (("rho0", rho_in), ("eta", eta_in)):
        ref = col[j] + (b - r_in[j]) * ((col[j + 1] - col[j]) * (1.0 / (r_in[j + 1] - r_in[j])))
        got = np.asarray(getattr(F1.profile, name))
        for m in (lower, upper):
            assert abs(got[m] - ref) <= INSERT_TOL * abs(ref), (name, m)
    assert np.asarray(F1.profile.radius)[lower] == np.asarray(F1.profile.radius)[upper] == b


@pytest.mark.parametrize("fx", ["F1", "F3", "Fnode"])
def test_input_nodes_unchanged(request, p_red, fx):
    L = request.getfixturevalue(fx)
    for k in ("radius", "rho0", "eta"):
        np.testing.assert_array_equal(np.asarray(getattr(L.profile, k))[L.input_node], np.asarray(getattr(p_red, k)))


@pytest.mark.parametrize("fx", ["F1", "F3", "Fnode"])
def test_layer_of(request, fx):
    L = request.getfixturevalue(fx)
    r = np.asarray(L.profile.radius)
    assert L.layer_of.shape == (L.n_nodes,)
    assert L.layer_of[0] == 0 and L.layer_of[-1] == L.n_layers - 1
    assert np.all(np.diff(L.layer_of) >= 0) and set(L.layer_of.tolist()) == set(range(L.n_layers))
    for j, lower in enumerate(L.pairs):
        assert L.layer_of[lower] == j and L.layer_of[lower + 1] == j + 1
        assert r[lower] == r[lower + 1] == L.boundaries[j + 1]
    # Layer sizes: the nodes from the upper member of the boundary below through the lower
    # member of the boundary above.
    edges = np.concatenate([[-1], L.pairs, [L.n_nodes - 1]])
    np.testing.assert_array_equal(np.bincount(L.layer_of), np.diff(edges))


def test_knot_table_rule(p_red, F1, F3, Fnode, S1):
    """D1 item 6: the knots are the input nodes at their static radii and the members of
    the pairs that FIXED boundaries insert; the members of a pair that a movable boundary
    inserts are not knots; a boundary on one input node: the lower member is the knot;
    a boundary on an input viscosity-only pair: both members are knots at one radius with
    a zero slope weight."""
    for L in (F1, F3):
        np.testing.assert_array_equal(L.table_nodes, np.sort(L.input_node))
        for k, g in L.ranges.items():
            assert not np.isin(list(g.members), L.table_nodes).any()
    # F-node, boundary 1 on the input node 5286.25 km.
    lower, upper = Fnode.ranges[1].members
    assert lower in Fnode.input_node and lower in Fnode.table_nodes and upper not in Fnode.table_nodes
    # F-node, boundary 4 on the input pair 6270 km: both members are knots of one radius.
    lo4, up4 = Fnode.ranges[4].members
    assert lo4 in Fnode.table_nodes and up4 in Fnode.table_nodes
    q = int(np.flatnonzero(Fnode.table_nodes == lo4)[0])
    assert Fnode._knots[q] == Fnode._knots[q + 1] and Fnode._inv_width[q] == 0.0
    # S1 (review r3 minor 9): the fixed boundaries at 400 and 100 km insert pairs whose
    # members are knots; the movable 670 km pair is not.
    r = np.asarray(S1.profile.radius)
    for b in (5970 * KM, 6270 * KM):
        both = np.flatnonzero(r == b)
        assert both.size == 2 and np.isin(both, S1.table_nodes).all()
    assert not np.isin(list(S1.ranges[1].members), S1.table_nodes).any()
    assert S1.table_nodes.size == S1.input_node.size + 4


@pytest.mark.parametrize("fx", ["F1", "F3"])
def test_limits(request, fx):
    """D3: limits only with a movable boundary; widths at 200 joint positions below the
    limit (equality at a clamp end); the kappa_h limit is the maximum over the clamp
    bounds and the crossings and bounds the sampled values."""
    L = request.getfixturevalue(fx)
    st = L.profile.structure
    assert st.limits is not None
    r0 = np.asarray(L.profile.radius)
    mask = st.interval_mask
    h0 = np.log1p(np.diff(r0) / r0[:-1]) * mask
    assert np.all(st.width_bounds >= h0)
    lim_w = np.asarray(st.limits.width)
    lim_k = np.asarray(st.limits.kappa_h)
    call = jax.jit(lambda x: L(L.full_boundaries(x), np.log(ETA_F1 if fx == "F1" else ETA_F3)))
    worst, kworst = 0.0, np.zeros_like(lim_k)
    for x in joint_positions(L, 200, 7):
        q = call(jnp.asarray(x))
        r, rho = np.asarray(q.radius), np.asarray(q.rho0)
        h = np.log1p(np.diff(r) / r[:-1]) * mask
        ok = lim_w > 0
        worst = max(worst, float(np.max(h[ok] / lim_w[ok])))
        kworst = np.maximum(kworst, np.abs(np.log(rho[1:] / rho[:-1])) * mask)
    assert worst <= 1 + 4 * EPS
    assert np.all(kworst <= lim_k * (1 + 4 * EPS))
    # Equality at a clamp end of each range.
    for q_, k in enumerate(L.movable_index):
        for end in L.radius_bounds[q_]:
            x = L.start_radii.copy(); x[q_] = end
            r = np.asarray(call(jnp.asarray(x)).radius)
            h = np.log1p(np.diff(r) / r[:-1]) * mask
            assert np.any(np.abs(h - lim_w) <= 4 * EPS * lim_w)
    # kappa_h: the maximum over clamp bounds and crossings, recomputed here.
    ref = np.abs(np.log(np.asarray(L.profile.rho0)[1:] / np.asarray(L.profile.rho0)[:-1])) * mask
    for q_, k in enumerate(L.movable_index):
        g = L.ranges[k]
        for pos in np.concatenate([[g.low + g.margin, g.high - g.margin], L.crossings(k)[0]]):
            x = L.start_radii.copy(); x[q_] = pos
            _, rho = reference_profile(L, x)
            ref = np.maximum(ref, np.abs(np.log(rho[1:] / rho[:-1])) * mask)
    assert np.max(np.abs(ref - lim_k)) <= KAPPA_LIMIT_TOL
    # Without a movable boundary the structure has no limits.
    assert sj.Layers(L.input_profile, L.boundaries).profile.structure.limits is None


def test_radius_bounds(F3):
    assert F3.radius_bounds.shape == (3, 2)
    for q, (low, high) in enumerate([MOV_F3[1], MOV_F3[2], MOV_F3[4]]):
        m = 1e-3 * (high - low)
        assert F3.radius_bounds[q, 0] == low + m and F3.radius_bounds[q, 1] == high - m
    np.testing.assert_array_equal(F3.movable_index, [1, 2, 4])


def test_pade_plan(S1):
    cfg = sj.StokesConfig(lmax=50)
    assert S1.pade_plan(cfg) == propagate.static_plan(S1.profile.structure, cfg).pade
    assert (S1.pade_plan(cfg).order, S1.pade_plan(cfg).squarings) == (7, 1)
    # The construction table without the widened limits has the plan (9, 0) (D3 table).
    fresh = sj.RadialProfile(np.asarray(S1.profile.radius), np.asarray(S1.profile.rho0),
                             np.asarray(S1.profile.eta), core_density=float(S1.profile.core_density))
    plan0 = propagate.static_plan(fresh.structure, cfg).pade
    assert (plan0.order, plan0.squarings) == (9, 0)
    assert S1.width_factor > 2.0


def test_static_plan_refusals(F1):
    """The two refusals of propagate.static_plan for a table with limits
    (04-review-jax-r1.md minor 1): they are the only guard of the Pade plan when the
    radius is traced. On F1 the largest static |kappa_h| lies between 0.03 and 0.06, and
    the static rho0_max is above 5000 kg/m^3. Each refusal raises from a concrete
    call of the plan and, for the density limit, under jit(grad) with a traced radius
    (the check runs at trace time, before any compile). The accepted neighbours: a
    kappa_h_max above the limit under TALA, and rho0_max = 5000 under NO_SELF_GRAVITY,
    which has no density entry in the exponent."""
    assert 0.03 < max(F1.profile.structure.limits.kappa_h) < 0.06
    assert F1.profile.structure.limits.rho0_max > 5000.0
    tala = sj.Formulation(compressible=True)
    nsg = sj.Formulation(gravity=sj.GravityModel.NO_SELF_GRAVITY, background=sj.Uniform(9.81))
    with pytest.raises(ValueError, match="Numerics.kappa_h_max = 0.03"):
        F1.pade_plan(sj.StokesConfig(lmax=10, formulation=tala, numerics=sj.Numerics(kappa_h_max=0.03)))
    F1.pade_plan(sj.StokesConfig(lmax=10, formulation=tala, numerics=sj.Numerics(kappa_h_max=0.06)))
    cfg_rho = sj.StokesConfig(lmax=10, numerics=sj.Numerics(rho0_max=5000.0))
    with pytest.raises(ValueError, match="above Numerics.rho0_max = 5000.0"):
        F1.pade_plan(cfg_rho)
    eta = np.log(ETA_F1)
    def geoid_sum(x):
        return jnp.sum(sj.stokes_kernels(F1(F1.full_boundaries(x), eta), cfg_rho).geoid)
    with pytest.raises(ValueError, match="above Numerics.rho0_max = 5000.0"):
        jax.jit(jax.grad(geoid_sum))(jnp.asarray(F1.start_radii))
    F1.pade_plan(sj.StokesConfig(lmax=10, formulation=nsg, numerics=sj.Numerics(rho0_max=5000.0)))


def test_independent_layering(load_oracle):
    """Layers on the 129-node LEBEDEV table of t_hager() without its pairs, boundaries at
    the hager4 depths, hager4 layer viscosities: the table equals the oracle table of
    t_hager() (pairs and viscosities set by hand there) to rounding, and the kernels the
    stored oracle kernels of hager4_full_fsfs (review r1 of the plan, minor 3)."""
    th = cases.t_hager()
    full = profile_from_table(th)
    r = np.asarray(full.radius)
    keep = np.ones(r.size, bool)
    dup = np.flatnonzero(r[1:] == r[:-1])
    keep[dup] = keep[dup + 1] = False
    inp = sj.RadialProfile(r[keep], np.asarray(full.rho0)[keep], np.asarray(full.eta)[keep],
                           core_density=float(full.core_density), core_mass=float(full.core_mass))
    L = sj.Layers(inp, np.array([3480, 5700, 5970, 6270, 6370]) * KM)
    eta_layers = np.asarray(full.eta)[[0, dup[0] + 1, dup[1] + 1, dup[2] + 1]]
    prof = L(L.boundaries, np.log(eta_layers))
    np.testing.assert_array_equal(np.asarray(prof.radius), r)
    np.testing.assert_allclose(np.asarray(prof.rho0), np.asarray(full.rho0), rtol=4 * EPS, atol=0)
    np.testing.assert_allclose(np.asarray(prof.eta), np.asarray(full.eta), rtol=EXP_LOG_TOL, atol=0)
    K = sj.stokes_kernels(prof, sj.StokesConfig(lmax=10))
    O = load_oracle("hager4_full_fsfs")
    for l in (1, 2, 3, 10):
        frame = "centre_of_mass" if l == 1 else "raw"
        for o in SI_OUTPUTS:
            assert_rows_close(np.asarray(getattr(K, o))[K.row(l)], O[f"base/l{l}/{frame}/{o}"], ORACLE_TOL,
                              f"independent layering l={l} {o}")


def test_refusals(p_red):
    B = B_F1
    with pytest.raises(ValueError, match="concrete profile"):
        jax.jit(lambda rr: sj.Layers(p_red.replace(rho0=rr), B).n_nodes)(p_red.rho0)
    with pytest.raises(ValueError, match="cannot be the input of another"):
        sj.Layers(sj.Layers(p_red, B, movable=MOV_F1, motion="affine").profile, B)
    with pytest.raises(ValueError, match="at least two radii"):
        sj.Layers(p_red, [3480e3])
    with pytest.raises(ValueError, match="finite"):
        sj.Layers(p_red, [3480e3, np.nan, 6370e3])
    with pytest.raises(ValueError, match="ascend strictly"):
        sj.Layers(p_red, [3480e3, 5000e3, 4900e3, 6370e3])
    with pytest.raises(ValueError, match="CMB radius"):
        sj.Layers(p_red, [3500e3, 5000e3, 6370e3])
    with pytest.raises(ValueError, match="surface radius"):
        sj.Layers(p_red, [3480e3, 5000e3, 6300e3])
    for key in (0, 5, True, 1.0):
        with pytest.raises(ValueError, match="interior boundary"):
            sj.Layers(p_red, B, movable={key: (5105.625e3, 5700e3)}, motion="affine")
    with pytest.raises(ValueError, match="strictly inside its movable range"):
        sj.Layers(p_red, B, movable={1: (5400e3, 5700e3)}, motion="affine")
    with pytest.raises(ValueError, match="not a node radius"):
        sj.Layers(p_red, B, movable={1: (5100e3, 5700e3)}, motion="affine")
    with pytest.raises(ValueError, match="density jump pair"):
        sj.Layers(p_red, B, movable={1: (5105.625e3, 5828.125e3)}, motion="affine")
    # On the F3 layering: overlap, an end at another movable boundary, a density jump.
    with pytest.raises(ValueError, match="overlap"):
        sj.Layers(p_red, B_F3, movable={1: (4202.5e3, 5286.25e3), 2: (5105.625e3, 5700e3)}, motion="affine")
    with pytest.raises(ValueError, match="radius of the movable boundary"):
        sj.Layers(p_red, B_F3, movable={1: (4202.5e3, 5370e3), 2: (5105.625e3, 5700e3)}, motion="affine")
    with pytest.raises(ValueError, match="density jump pair"):
        sj.Layers(p_red, B_F3, movable={4: (5700e3, 6008.75e3)}, motion="affine")
    with pytest.raises(ValueError, match="lies inside the movable range"):
        sj.Layers(p_red, np.array([3480, 4650, 4700, 5700, 6370]) * KM, movable={1: (4202.5e3, 5105.625e3)},
                  motion="affine")
    # The positive cases: two ranges that share an end node, and two ranges that end on
    # the two members of one density-jump pair.
    sj.Layers(p_red, B_F3, movable={1: MOV_F3[1], 2: MOV_F3[2]}, motion="affine")
    sj.Layers(p_red, B_F3, movable={2: MOV_F3[2], 4: MOV_F3[4]}, motion="affine")


# ==========================================================================
# 5.2 The call
# ==========================================================================


@pytest.mark.parametrize("fx,eta", [("F1", ETA_F1), ("F3", ETA_F3), ("Fnode", ETA_FNODE)])
def test_construction_radius(request, fx, eta):
    L = request.getfixturevalue(fx)
    q = jax.jit(lambda b: L(b, np.log(eta)))(jnp.asarray(L.boundaries))
    np.testing.assert_array_equal(np.asarray(q.radius), np.asarray(L.profile.radius))
    rho, rho0 = np.asarray(q.rho0), np.asarray(L.profile.rho0)
    assert np.all(np.abs(rho - rho0) <= 2 * EPS * rho0)


# The construction radius under jit on WIDE ranges (04-review-jax-r1.md, B1). Under jit
# the static coefficients of the affine map are constants of the compiled program, and
# XLA folds a division by a constant into a multiplication by its reciprocal. Only a
# map whose factor (n - r0) is an exact zero at r0 returns the construction radius bit
# for bit for every range; the error of a folded form grows with the distance of a node
# from its range end, so the fixtures F1, F3 and F-node (ranges of at most 770 km) do not
# show it. These tests use single-boundary layerings on the reduced PREM table with the
# whole lower mantle (3480, 5700) km as the range.
WIDE_RANGE = (3480 * KM, 5700 * KM)
WIDE_BOUNDARIES = np.array([3480, np.nan, 5700, 5970, 6370]) * KM
# The named layering of B1: with the folded form, one input node sat one ulp below its
# own input radius under jit, and the jitted derivative at r0 missed D+ by 9.6 %.
B1_RADIUS = 3720276.3768699113
# Number of random wide layerings in the quick test (seed WIDE_SEED); the slow test
# runs the full sample of the review script chk_r0_wide.py.
WIDE_QUICK_COUNT = 24
WIDE_SEED = 2026


def _wide_layers(p, b, rng_=WIDE_RANGE):
    """A single-boundary layering with boundary ``b`` (m) movable in ``rng_`` (m)."""
    bounds = WIDE_BOUNDARIES.copy()
    bounds[1] = b
    return sj.Layers(p, bounds, movable={1: rng_}, motion="affine")


def _assert_construction_radius_jit(L, p):
    """Two compiled programs of one layering at r0, the whole profile and the radius
    alone (a map whose constant factors XLA folds can fail on different layerings
    depending on what the program returns): the radius is the construction radius bit for bit, every input
    node is on its own input radius, and rho0 is the construction rho0 to 2 eps."""
    le = np.full(L.n_layers, np.log(1e21))
    start = jnp.asarray(L.start_radii)
    q = jax.jit(lambda x: L(L.full_boundaries(x), le))(start)
    r_only = np.asarray(jax.jit(lambda x: L(L.full_boundaries(x), le).radius)(start))
    r_c = np.asarray(L.profile.radius)
    for r in (np.asarray(q.radius), r_only):
        np.testing.assert_array_equal(r, r_c, err_msg=f"boundary {L.start_radii[0]!r} m, range "
                                                      f"{L.ranges[1].low!r}..{L.ranges[1].high!r} m")
        np.testing.assert_array_equal(r[L.input_node], np.asarray(p.radius))
    rho, rho0 = np.asarray(q.rho0), np.asarray(L.profile.rho0)
    assert np.all(np.abs(rho - rho0) <= 2 * EPS * rho0)


def test_construction_radius_wide_ranges(p_red):
    """B1 named layering and WIDE_QUICK_COUNT random boundaries in 3700 to 5600 km, each
    movable in (3480, 5700) km: the construction radius under jit, bit for bit."""
    rng = np.random.default_rng(WIDE_SEED)
    r_in = np.asarray(p_red.radius)
    radii = [B1_RADIUS] + [b for b in rng.uniform(3700e3, 5600e3, size=WIDE_QUICK_COUNT) if not np.any(r_in == b)]
    for b in radii:
        _assert_construction_radius_jit(_wide_layers(p_red, float(b)), p_red)


@pytest.mark.slow
def test_construction_radius_wide_sample():
    """The sample of chk_r0_wide.py (04-review-jax-r1.md, B1): seed 3, 400 boundaries
    per table, uniform in 3700 to 5600 km, on the full and the reduced PREM table; every
    second range is the whole lower mantle (3480, 5700) km, the others span about 600 km
    around the boundary. The folded form put a node off its construction radius in 8 of
    these layerings."""
    rng = np.random.default_rng(3)
    for table in (cases.t_prem(), cases.t_prem(reduced=True)):
        p = profile_from_table(table)
        r_in = np.asarray(p.radius)
        for trial in range(400):
            b = float(rng.uniform(3700e3, 5600e3))
            if np.any(r_in == b):
                continue
            if trial % 2 == 0:
                rng_ = WIDE_RANGE
            else:
                lows = r_in[r_in < b - 250e3]
                highs = r_in[(r_in > b + 250e3) & (r_in <= 5700e3)]
                if lows.size == 0 or highs.size == 0:
                    continue
                rng_ = (float(lows[-1]), float(highs[0]))
            _assert_construction_radius_jit(_wide_layers(p, b, rng_), p)


def test_boundaries_as_list_with_tracer(F1):
    """A list or tuple of boundary radii that holds a tracer is stacked into one array
    (04-review-jax-r1.md minor 5); a concrete list is checked as an array."""
    b = [float(v) for v in F1.boundaries]
    eta = np.log(ETA_F1)
    def radius_of(x):
        return F1([b[0], x] + b[2:], eta).radius
    got = jax.jit(radius_of)(jnp.asarray(5400e3))
    want = jax.jit(lambda y: F1(F1.full_boundaries(y), eta).radius)(jnp.asarray([5400e3]))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_array_equal(np.asarray(F1(tuple(b), eta).radius), np.asarray(F1.profile.radius))
    with pytest.raises(ValueError, match="outside its movable range"):
        F1([b[0], 5800e3] + b[2:], eta)


@pytest.mark.parametrize("fx", ["F1", "F3", "Fnode"])
def test_affine_maps_and_rho0_source(request, fx):
    """The moved radii against the numpy loop over the ranges (<= 4 eps r), rho0 against
    the input profile (<= 4 eps), at 200 joint positions; nodes outside every range bit
    for bit; both members of a pair at one radius."""
    L = request.getfixturevalue(fx)
    eta = {"F1": ETA_F1, "F3": ETA_F3, "Fnode": ETA_FNODE}[fx]
    call = jax.jit(lambda x: L(L.full_boundaries(x), np.log(eta)))
    fixed = np.setdiff1d(np.arange(L.n_nodes), L.moved_nodes)
    er = eh = 0.0
    for x in joint_positions(L, 200, 3):
        q = call(jnp.asarray(x))
        rad, rho = reference_profile(L, x)
        r_q, rho_q = np.asarray(q.radius), np.asarray(q.rho0)
        er = max(er, float(np.max(np.abs(r_q - rad) / rad)))
        eh = max(eh, float(np.max(np.abs(rho_q - rho) / rho)))
        np.testing.assert_array_equal(r_q[fixed], np.asarray(L.profile.radius)[fixed])
        for g in L.ranges.values():
            assert r_q[g.members[0]] == r_q[g.members[1]]
    print(f"{fx}: radius {er / EPS:.2f} eps, rho0 {eh / EPS:.2f} eps")
    assert er <= FMA_TOL and eh <= FMA_TOL


def test_one_boundary_moves(F3):
    """Each boundary of F3 in turn at 20 positions, the other two at r0: every node outside
    the range of the moved boundary, the nodes of the other ranges included, is the
    construction table bit for bit."""
    call = jax.jit(lambda x: F3(F3.full_boundaries(x), np.log(ETA_F3)))
    r0 = np.asarray(F3.profile.radius)
    rho0 = np.asarray(call(jnp.asarray(F3.start_radii)).rho0)
    X = joint_positions(F3, 20, 5)
    for q, k in enumerate(F3.movable_index):
        own = F3.moved_nodes[F3._range_of == q]
        others = np.setdiff1d(np.arange(F3.n_nodes), own)
        for x in X:
            xx = F3.start_radii.copy(); xx[q] = x[q]
            p = call(jnp.asarray(xx))
            np.testing.assert_array_equal(np.asarray(p.radius)[others], r0[others])
            np.testing.assert_array_equal(np.asarray(p.rho0)[others], rho0[others])


def test_rho0_keyword(F1, Fnode):
    le = np.log(ETA_F1)
    x = jnp.asarray(F1.full_boundaries(np.array([5420e3])))
    call = jax.jit(lambda b, rho: F1(b, le, rho0=rho))
    a = F1(x, le)
    b = call(x, F1.profile.rho0)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda b: F1(b, le))(x).rho0), np.asarray(b.rho0))
    np.testing.assert_array_equal(np.asarray(a.radius), np.asarray(b.radius))
    # A perturbed rho0: the fixed nodes carry it, the moved nodes interpolate its knots.
    rng = np.random.default_rng(2)
    rho_p = np.asarray(F1.profile.rho0) * (1 + 0.01 * rng.standard_normal(F1.n_nodes))
    c = np.asarray(call(x, rho_p).rho0)
    fixed = np.setdiff1d(np.arange(F1.n_nodes), F1.moved_nodes)
    np.testing.assert_array_equal(c[fixed], rho_p[fixed])
    xs = np.asarray(F1.profile.radius)[F1.table_nodes]
    moved_r = np.asarray(b.radius)[F1.moved_nodes]
    ref = np.interp(moved_r, xs, rho_p[F1.table_nodes])
    assert np.max(np.abs(c[F1.moved_nodes] - ref) / ref) <= FMA_TOL
    # The members of the movable inserted pair are not read: zero derivative.
    g = jax.grad(lambda rho: jnp.sum(F1(x, le, rho0=rho).rho0 ** 2))(jnp.asarray(rho_p))
    assert np.all(np.asarray(g)[list(F1.ranges[1].members)] == 0.0)
    # Review r3 minor 8: different values on the two members of an input pair inside a
    # range (F-node, 6270 km) or of a duplicated node are refused when concrete.
    for lower in (Fnode.ranges[4].members[0], Fnode.ranges[1].members[0]):
        bad = np.asarray(Fnode.profile.rho0).copy()
        bad[lower + 1] += 1.0
        with pytest.raises(ValueError, match="one value"):
            Fnode(Fnode.boundaries, np.log(ETA_FNODE), rho0=bad)


def test_gravity_follows(F1):
    """g is integrated again on the moved nodes: below the range the hydrostatic gravity
    is the construction gravity bit for bit, above it differs (mass moved)."""
    units = sj.Units()
    g0 = np.asarray(F1.profile.hydrostatic_gravity(units))
    q = F1(F1.full_boundaries(np.array([5600e3])), np.log(ETA_F1))
    g1 = np.asarray(q.hydrostatic_gravity(units))
    lo = F1.ranges[1].lo_node
    np.testing.assert_array_equal(g1[:lo + 1], g0[:lo + 1])
    assert np.any(g1[F1.ranges[1].hi_node:] != g0[F1.ranges[1].hi_node:])


def test_clamp_traced(F1):
    """Traced radii outside the closed range give the profile of the clamp bound bit for
    bit (same jitted function), with an exactly zero derivative of the radius column."""
    le = np.log(ETA_F1)
    g = F1.ranges[1]
    call = jax.jit(lambda x: F1(F1.full_boundaries(x), le).radius)
    jac = jax.jit(jax.jacfwd(lambda x: F1(F1.full_boundaries(x), le).radius))
    jacr = jax.jit(jax.jacrev(lambda x: F1(F1.full_boundaries(x), le).radius))
    lo_b, hi_b = F1.radius_bounds[0]
    for value, bound in ((g.low - 11e3, lo_b), (g.low, lo_b), (g.low - 1e6, lo_b),
                         (g.high, hi_b), (g.high + 15e3, hi_b), (g.high + 1e6, hi_b)):
        np.testing.assert_array_equal(np.asarray(call(jnp.array([value]))), np.asarray(call(jnp.array([bound]))))
        assert np.all(np.asarray(jac(jnp.array([value]))) == 0.0)
        assert np.all(np.asarray(jacr(jnp.array([value]))) == 0.0)


def test_clamp_inside(F1):
    le = np.log(ETA_F1)
    g = F1.ranges[1]
    J = np.asarray(jax.jacfwd(lambda x: F1(F1.full_boundaries(x), le).radius)(jnp.array([5450e3])))[:, 0]
    assert J[g.members[0]] == 1.0 and J[g.members[1]] == 1.0
    f = (g.radii_below - g.low) / (g.start - g.low)
    np.testing.assert_allclose(J[g.below], f, rtol=4 * EPS, atol=0)
    f_up = (g.high - g.radii_above) / (g.high - g.start)
    np.testing.assert_allclose(J[g.above], f_up, rtol=4 * EPS, atol=0)


def test_pair_jacobian_and_clamp_per_boundary(F3):
    """The Jacobian of the six pair radii with respect to the three radii is [I; I]
    exactly in both modes; each boundary below or above its own range gives an exact zero
    column and the profile of its clamp bound bit for bit."""
    lowers = np.array([F3.ranges[k].members[0] for k in F3.movable_index])
    uppers = lowers + 1
    le = np.log(ETA_F3)

    def pairs(x):
        r = F3(F3.full_boundaries(x), le).radius
        return jnp.concatenate([r[lowers], r[uppers]])
    jf, jr = jax.jit(jax.jacfwd(pairs)), jax.jit(jax.jacrev(pairs))
    call = jax.jit(lambda x: F3(F3.full_boundaries(x), le).radius)
    xin = joint_positions(F3, 1, 9)[0]
    I2 = np.concatenate([np.eye(3), np.eye(3)])
    np.testing.assert_array_equal(np.asarray(jf(jnp.asarray(xin))), I2)
    np.testing.assert_array_equal(np.asarray(jr(jnp.asarray(xin))), I2)
    for q, k in enumerate(F3.movable_index):
        g = F3.ranges[k]
        for value, bound in ((g.low - 11e3, F3.radius_bounds[q, 0]), (g.high + 15e3, F3.radius_bounds[q, 1])):
            xx = xin.copy(); xx[q] = value
            xb = xin.copy(); xb[q] = bound
            Ik = I2.copy(); Ik[:, q] = 0.0
            np.testing.assert_array_equal(np.asarray(jr(jnp.asarray(xx))), Ik)
            np.testing.assert_array_equal(np.asarray(jf(jnp.asarray(xx))), Ik)
            np.testing.assert_array_equal(np.asarray(call(jnp.asarray(xx))), np.asarray(call(jnp.asarray(xb))))


def test_fixed_entries_and_concrete_checks(F1):
    le = np.log(ETA_F1)
    g = jax.jit(jax.grad(lambda b: jnp.sum(F1(b, le).radius)))(jnp.asarray(F1.boundaries))
    assert np.all(np.asarray(g)[[0, 2, 3, 4, 5]] == 0.0) and np.asarray(g)[1] != 0.0
    with pytest.raises(ValueError, match="outside its movable range"):
        F1(F1.boundaries + np.array([0, 400e3, 0, 0, 0, 0]), le)
    with pytest.raises(ValueError, match="fixed"):
        F1(F1.boundaries + np.array([0, 0, 1.0, 0, 0, 0]), le)
    with pytest.raises(ValueError, match="finite"):
        F1(F1.boundaries, np.array([np.nan, 1, 1, 1, 1.0]))
    with pytest.raises(ValueError, match="per layer"):
        F1(F1.boundaries, le[:4])
    with pytest.raises(ValueError, match="shape"):
        F1(F1.boundaries[:5], le)
    with pytest.raises(ValueError, match="rho0"):
        F1(F1.boundaries, le, rho0=np.ones(5))
    full = np.asarray(F1.full_boundaries(np.array([5400e3])))
    np.testing.assert_array_equal(full[[0, 2, 3, 4, 5]], F1.boundaries[[0, 2, 3, 4, 5]])
    assert full[1] == 5400e3


def test_no_loop_and_vmap(F3):
    le = np.log(ETA_F3)
    f = lambda x: jnp.sum(F3(F3.full_boundaries(x), le).rho0 * F3(F3.full_boundaries(x), le).radius)
    prims = primitives(jax.make_jaxpr(jax.grad(f))(jnp.asarray(F3.start_radii + 1e3)).jaxpr)
    assert not prims & {"scan", "while", "cond"}, prims
    # vmap against single calls: equal up to the FMA contraction of the affine map, which
    # XLA fuses differently under vmap (measured: one radius of 31 differs by one ulp).
    X = joint_positions(F3, 4, 13)
    one = jax.jit(lambda x: F3(F3.full_boundaries(x), le))
    many = jax.jit(jax.vmap(lambda x: F3(F3.full_boundaries(x), le)))(jnp.asarray(X))
    for b in range(4):
        p = one(jnp.asarray(X[b]))
        for k in ("radius", "rho0", "eta"):
            a, c = np.asarray(getattr(many, k)[b]), np.asarray(getattr(p, k))
            assert np.max(np.abs(a - c) / np.abs(c)) <= FMA_TOL, k


# ==========================================================================
# 5.3 viscosity_by_node, ControlVector, spectral_misfit
# ==========================================================================


def _five_node():
    r = np.array([3480e3, 4200e3, 5000e3, 5800e3, 6370e3])
    return sj.RadialProfile(r, np.array([5500.0, 5000.0, 4600.0, 3900.0, 3300.0]),
                            np.array([1e22, 3e21, 1e21, 1e20, 1e22]), core_density=9900.0)


def test_node_map():
    p = _five_node()
    m = sj.viscosity_by_node(p)
    assert m.size == 5
    np.testing.assert_allclose(np.asarray(m(m.initial).eta), np.asarray(p.eta), rtol=EXP_LOG_TOL, atol=0)
    with pytest.raises(ValueError, match=r"\(5,\)"):
        m(np.zeros(4))


def test_control_vector_round_trip_and_names():
    c = {"radii": np.array([5.0e6, 5.5e6]), "log_eta": np.array([1.0, 2.0, 3.0]), "g": np.array(9.81)}
    cv = sj.ControlVector(c)
    assert cv.size == 6
    back = cv.unravel(cv.ravel(c))
    for k in c:
        np.testing.assert_array_equal(np.asarray(back[k]), c[k])
    assert cv.names == ("g", "log_eta[0]", "log_eta[1]", "log_eta[2]", "radii[0]", "radii[1]")
    for bad in ({"a": np.ones(3, np.float32)}, {"a": np.ones(3, int)}):
        with pytest.raises(TypeError):
            sj.ControlVector(bad)


def test_numpy_front_and_scipy():
    """The NumPy front equals the JAX function bit for bit, refuses a wrong shape, and
    a second call with new values or new data of the same shapes neither traces the
    solver again nor compiles again (review r3 minor 7: the jitted function is cached per
    fn inside the ControlVector; ``with_data`` and a second ``numpy_value_and_grad`` reuse
    it). SciPy L-BFGS-B runs on it."""
    import scipy.optimize
    p = _five_node()
    m = sj.viscosity_by_node(p)
    cfg = sj.StokesConfig(lmax=3)
    calls = {"n": 0}

    def fn(c, w):
        calls["n"] += 1
        K = sj.stokes_kernels(m(c["log_eta"]), cfg)
        return jnp.sum(w * K.geoid) * 1e7 + 1e-3 * jnp.sum((c["log_eta"] - m.initial) ** 2)

    cv = sj.ControlVector({"log_eta": np.asarray(m.initial)})
    w1 = np.ones((3, 5))
    f = cv.numpy_value_and_grad(fn, w1)
    x0 = np.asarray(m.initial)
    v, g = f(x0)
    assert isinstance(v, float) and isinstance(g, np.ndarray)
    vj, gj = cv.value_and_grad(fn)(jnp.asarray(x0), jnp.asarray(w1))
    assert v == float(vj) and np.array_equal(g, np.asarray(gj))
    with pytest.raises(ValueError, match="shape"):
        f(x0[:4])
    n_trace, n_fn = propagate.trace_count(), calls["n"]
    f(x0 + 0.1)
    f.with_data(2 * w1)(x0 - 0.1)
    cv.numpy_value_and_grad(fn, 3 * w1)(x0)
    assert propagate.trace_count() == n_trace and calls["n"] == n_fn
    assert cv.value_and_grad(fn)._cache_size() == 1
    assert abs(cv.numpy_value(fn, w1)(x0) - v) <= 1e-14 * abs(v)
    res = scipy.optimize.minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 3})
    assert np.all(np.isfinite(res.x)) and np.isfinite(res.fun)


def _misfit_loop(field, obs, w):
    """numpy loop over l, m and C/S of a real_4pi misfit (batch axes summed)."""
    d = np.asarray(field) - np.asarray(obs)
    d = d.reshape((-1,) + d.shape[-3:])
    total = 0.0
    for b in range(d.shape[0]):
        for l in range(d.shape[-1]):
            for m_ in range(l + 1):
                for c in range(2):
                    total += w[l] * d[b, c, l, m_] ** 2
    return total


def test_spectral_misfit():
    rng = np.random.default_rng(3)
    L = 8
    tri = np.tril(np.ones((L + 1, L + 1)))
    field = rng.standard_normal((3, 2, L + 1, L + 1)) * tri
    obs = rng.standard_normal((3, 2, L + 1, L + 1)) * tri
    w = rng.random(L + 1)
    val = float(fields.spectral_misfit(field, obs, w, "real_4pi"))
    ref = _misfit_loop(field, obs, w)
    assert abs(val - ref) <= MISFIT_TOL * abs(ref)
    fc = sj.sh.convert(field, "real_4pi", "complex_4pi")
    oc = sj.sh.convert(obs, "real_4pi", "complex_4pi")
    assert abs(float(fields.spectral_misfit(fc, oc, w, "complex_4pi")) - val) <= MISFIT_TOL * abs(val)
    fo = sj.sh.convert(field, "real_4pi", "real_orthonormal")
    oo = sj.sh.convert(obs, "real_4pi", "real_orthonormal")
    v_o = float(fields.spectral_misfit(fo, oo, w, "real_orthonormal"))
    assert abs(v_o - 4 * np.pi * val) <= MISFIT_TOL * abs(v_o)
    # Zero weights remove degrees exactly.
    w0 = w.copy(); w0[:2] = 0.0
    f2 = field.copy(); f2[..., :2, :] += 100.0
    assert float(fields.spectral_misfit(f2, obs, w0, "real_4pi")) == float(fields.spectral_misfit(field, obs, w0, "real_4pi"))
    # Gradient with respect to the field: 2 w_l d.
    gr = np.asarray(jax.grad(lambda f: fields.spectral_misfit(f, obs, w, "real_4pi"))(jnp.asarray(field)))
    ref_g = 2 * w[None, None, :, None] * (field - obs)
    assert np.max(np.abs(gr - ref_g)) <= MISFIT_GRAD_TOL * np.max(np.abs(ref_g))
    with pytest.raises(ValueError, match="one weight per degree"):
        fields.spectral_misfit(field, obs, w[:-1], "real_4pi")
    with pytest.raises(ValueError):
        fields.spectral_misfit(field[..., :-1], obs[..., :-1], w, "real_4pi")
    with pytest.raises(ValueError):
        fields.spectral_misfit(field[:, 0], obs[:, 0], w, "real_4pi")


# ==========================================================================
# 5.4 The uniform gravity leaf
# ==========================================================================


def test_uniform_gravity_leaf():
    p = profile_from_table(cases.t_hager(reduced=True))
    nsg = sj.GravityModel.NO_SELF_GRAVITY
    cfg_a = sj.StokesConfig(lmax=10, formulation=sj.Formulation(gravity=nsg, background=sj.Uniform(9.81)))
    cfg_b = sj.StokesConfig(lmax=10, formulation=sj.Formulation(gravity=nsg, background=sj.Uniform()))
    Ka = sj.stokes_kernels(p, cfg_a)
    Kb = sj.stokes_kernels(p.replace(uniform_gravity=9.81), cfg_b)
    for o in SI_OUTPUTS:
        assert_rows_close(np.asarray(getattr(Kb, o)), np.asarray(getattr(Ka, o)), UNIFORM_LEAF_TOL, o)
    with pytest.raises(ValueError, match="uniform_gravity"):
        sj.stokes_kernels(p, cfg_b)
    with pytest.raises(ValueError, match="only the background Uniform"):
        sj.stokes_kernels(p.replace(uniform_gravity=9.81), cfg_a)
    with pytest.raises(ValueError, match="only the background Uniform"):
        sj.stokes_kernels(p.replace(uniform_gravity=9.81), sj.StokesConfig(lmax=10))
    for bad in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            sj.stokes_kernels(p.replace(uniform_gravity=bad), cfg_b)
    # The jit key: with and without the leaf the trees differ; without it the leaves are
    # the seven of the earlier layout.
    t0 = jax.tree_util.tree_structure(p)
    t1 = jax.tree_util.tree_structure(p.replace(uniform_gravity=9.81))
    assert t0 != t1
    assert len(jax.tree_util.tree_leaves(p)) == 6 and len(jax.tree_util.tree_leaves(p.replace(uniform_gravity=9.81))) == 7


# ==========================================================================
# 5.5 The density map
# ==========================================================================


@pytest.mark.parametrize("fx", ["F1", "F3"])
def test_density_map_construction_radius(request, fx):
    L = request.getfixturevalue(fx)
    K = fake_kernels(L, L.profile.radius)
    Kin = L.input_kernels(K)
    for o in SI_OUTPUTS:
        np.testing.assert_array_equal(np.asarray(getattr(Kin, o)), np.asarray(getattr(K, o))[:, L.input_node])
    np.testing.assert_array_equal(np.asarray(Kin.degree_one_imbalance), np.asarray(K.degree_one_imbalance)[L.input_node])
    a, t = L._bracket(jnp.asarray(L.profile.radius))
    assert np.all(np.asarray(t) == 1.0)


def test_density_map_linear_kernels(F1):
    """Kernel rows linear in r below the pair and with another slope above it (continuous
    at the pair): the map returns the linear values at the input radii."""
    le = np.log(ETA_F1)
    call = jax.jit(lambda x: F1(F1.full_boundaries(x), le).radius)
    r_in = np.asarray(F1.input_profile.radius)
    slopes = np.array([[1.0, -2.0], [0.3, 0.7], [-1.5, 0.2], [2.0, 2.0], [0.5, -0.5]])
    for xb in np.linspace(*F1.radius_bounds[0], 50):
        r = np.asarray(call(jnp.array([xb])))
        def lin(rr):
            return np.where(rr <= xb, 1.0 + slopes[:, :1] * (rr - xb) / 1e6, 1.0 + slopes[:, 1:] * (rr - xb) / 1e6)
        rows = lin(r[None, :])
        K = fake_kernels(F1, r)
        K = sj.Kernels(degrees=K.degrees, degree_one_frame=K.degree_one_frame, radius=jnp.asarray(r),
                       geoid=jnp.asarray(rows), surface_topography=jnp.asarray(rows),
                       cmb_topography=jnp.asarray(rows), surface_velocity=jnp.asarray(rows),
                       degree_one_imbalance=None)
        got = np.asarray(F1.input_kernels(K).geoid)
        ref = lin(r_in[None, :])
        # e_l of CONVENTIONS 10 (the error over the row maximum): the rows pass close to
        # zero, where a pointwise relative error measures the cancellation of 1 + slope dr.
        assert_rows_close(got, ref, FMA_TOL, f"linear kernels at {xb}")


@pytest.mark.parametrize("fx", ["F1", "F3"])
def test_bracket_per_range(request, fx):
    L = request.getfixturevalue(fx)
    eta = ETA_F1 if fx == "F1" else ETA_F3
    call = jax.jit(lambda x: F(L, x, eta))
    brk = jax.jit(L._bracket)
    for x in joint_positions(L, 200, 17):
        r = call(jnp.asarray(x))
        a, t = brk(r)
        a_ref, t_ref = reference_bracket(L, np.asarray(r))
        np.testing.assert_array_equal(np.asarray(a), a_ref)
        assert np.max(np.abs(np.asarray(t) - t_ref)) <= FMA_TOL


def F(L, x, eta):
    return L(L.full_boundaries(x), np.log(eta)).radius


def test_side_at_tie(F1):
    """At r_b = r0 (every dynamic input node tied with its own moved node) and with the
    pair on an input radius (5466.875 km), the jvp of the map equals the forward one-sided
    difference over 1 mm and differs from the backward one."""
    K = fake_kernels(F1, F1.profile.radius, seed=4)
    le = np.log(ETA_F1)

    def g(xb):
        r = F1(F1.full_boundaries(xb[None]), le).radius
        Kr = sj.Kernels(degrees=K.degrees, degree_one_frame=K.degree_one_frame, radius=r, geoid=K.geoid,
                        surface_topography=K.surface_topography, cmb_topography=K.cmb_topography,
                        surface_velocity=K.surface_velocity, degree_one_imbalance=None)
        return F1.input_kernels(Kr).geoid
    gj = jax.jit(g)
    jv = jax.jit(lambda xb: jax.jvp(g, (xb,), (jnp.ones(()),))[1])
    h = 1e-3
    for xb in (5370e3, 5466.875e3):
        x = jnp.asarray(xb)
        d = np.asarray(jv(x))
        fwd = (np.asarray(gj(x + h)) - np.asarray(gj(x))) / h
        bwd = (np.asarray(gj(x)) - np.asarray(gj(x - h))) / h
        scale = np.max(np.abs(d))
        assert np.max(np.abs(d - fwd)) <= TIE_FD_TOL * scale, xb
        assert np.max(np.abs(d - bwd)) > 1e3 * TIE_FD_TOL * scale, xb


def test_fixed_only_layers(p_red):
    L = sj.Layers(p_red, B_F1)
    K = fake_kernels(L, L.profile.radius)
    np.testing.assert_array_equal(np.asarray(L.input_kernels(K).geoid), np.asarray(K.geoid)[:, L.input_node])
    lower = L.pairs[0]
    _, vjp = jax.vjp(lambda k: L.input_kernels(sj.Kernels(K.degrees, K.degree_one_frame, K.radius, k, k, k, k,
                                                          None)).geoid, K.geoid)
    ct = np.asarray(vjp(jnp.ones((5, p_red.structure.n_nodes)))[0])
    assert np.all(ct[:, [lower, lower + 1]] == 0.0)
    # fields.apply accepts a fixed-only Layers profile.
    d = densities.smooth(np.asarray(L.profile.radius), 5, 1)
    fields.apply(K, d, L.profile)


def test_boundary_on_a_node(Fnode):
    """F-node: the map is finite at joint positions, and at the construction radius the
    input node 5286.25 km takes the lower member of the pair on it."""
    K = fake_kernels(Fnode, Fnode.profile.radius, seed=6)
    Kin = Fnode.input_kernels(K)
    i = int(np.flatnonzero(np.asarray(Fnode.input_profile.radius) == 5286.25e3)[0])
    lower = Fnode.ranges[1].members[0]
    assert np.asarray(Kin.geoid)[:, i].tolist() == np.asarray(K.geoid)[:, lower].tolist()
    call = jax.jit(lambda x: Fnode(Fnode.full_boundaries(x), np.log(ETA_FNODE)).radius)
    for x in joint_positions(Fnode, 20, 2):
        r = call(jnp.asarray(x))
        Kx = sj.Kernels(K.degrees, K.degree_one_frame, r, K.geoid, K.surface_topography, K.cmb_topography,
                        K.surface_velocity, None)
        assert np.all(np.isfinite(np.asarray(Fnode.input_kernels(Kx).geoid)))


@pytest.mark.parametrize("fx", ["F1", "F3"])
def test_transposed_form(request, fx):
    L = request.getfixturevalue(fx)
    eta = ETA_F1 if fx == "F1" else ETA_F3
    x = joint_positions(L, 1, 23)[0]
    r = jax.jit(lambda x: F(L, x, eta))(jnp.asarray(x))
    K = fake_kernels(L, r, seed=8)
    d = densities.rough(np.asarray(L.input_profile.radius), 5, 3)
    Fa = L.fields(K, d)
    sm = L.sheet_mass(d, K)
    assert sm.shape == (L.n_nodes,) + d.shape[1:]
    Fb = fields.radial_integral(K, sm, jnp.ones(L.n_nodes))
    for n in fields.FIELD_NAMES:
        a, b = np.asarray(getattr(Fa, n)), np.asarray(getattr(Fb, n))
        assert np.max(np.abs(a - b)) <= SHEET_TOL * np.max(np.abs(a)), n
    # The dot-product identity <K_in, w d> = <K, sheet_mass(d)> over (node, C/S, l, m).
    Kin = np.asarray(L.input_kernels(K).geoid)                      # (5, N_in), degrees 1..5
    w = L.input_weights
    lhs = float(np.sum(Kin.T[:, None, :, None] * w[:, None, None, None] * d[:, :, 1:6, :]))
    rhs = float(np.sum(np.asarray(K.geoid).T[:, None, :, None] * np.asarray(sm)[:, :, 1:6, :]))
    assert abs(lhs - rhs) <= SHEET_TOL * max(abs(lhs), abs(rhs))


def test_batch_axes_and_complex(F1):
    K = fake_kernels(F1, F1.profile.radius)
    rng = np.random.default_rng(1)
    d = rng.standard_normal((25, 3, 2, 6, 6))
    Fb = F1.fields(K, d)
    assert Fb.geoid.shape == (3, 2, 6, 6)
    for b in range(3):
        Fs = F1.fields(K, d[:, b])
        for n in fields.FIELD_NAMES:
            np.testing.assert_array_equal(np.asarray(getattr(Fb, n))[b], np.asarray(getattr(Fs, n)))
    dc = sj.sh.convert(d[:, 0], "real_4pi", "complex_4pi")
    Fc = F1.fields(K, dc)
    assert jnp.iscomplexobj(Fc.geoid)


def test_density_map_vmap_and_no_loop(F3):
    le = np.log(ETA_F3)
    K = fake_kernels(F3, F3.profile.radius, seed=2)

    def kin(x):
        r = F3(F3.full_boundaries(x), le).radius
        return F3.input_kernels(sj.Kernels(K.degrees, K.degree_one_frame, r, K.geoid, K.surface_topography,
                                           K.cmb_topography, K.surface_velocity, None)).geoid
    # The radii under vmap differ from the single calls by an ulp (FMA, above); the weight
    # t = (r_i - x_a) / (x_(a+1) - x_a) turns that into ~1e-14 of a kernel row.
    X = joint_positions(F3, 4, 29)
    many = np.asarray(jax.jit(jax.vmap(kin))(jnp.asarray(X)))
    one = jax.jit(kin)
    for b in range(4):
        assert_rows_close(many[b], np.asarray(one(jnp.asarray(X[b]))), VMAP_RTOL, f"vmap {b}")
    d = densities.smooth(np.asarray(F3.input_profile.radius), 5, 1)

    def scalar(x):
        r = F3(F3.full_boundaries(x), le).radius
        Kr = sj.Kernels(K.degrees, K.degree_one_frame, r, K.geoid, K.surface_topography, K.cmb_topography,
                        K.surface_velocity, None)
        return jnp.sum(F3.fields(Kr, d).geoid)
    prims = primitives(jax.make_jaxpr(jax.grad(scalar))(jnp.asarray(X[0])).jaxpr)
    assert not prims & {"scan", "while", "cond"}, prims


def test_density_map_refusals(F1):
    K = fake_kernels(F1, F1.profile.radius)
    d = densities.smooth(np.asarray(F1.input_profile.radius), 5, 1)
    with pytest.raises(ValueError, match="Layers.fields"):
        fields.apply(K, np.zeros((F1.n_nodes, 2, 6, 6)), F1(F1.boundaries, np.log(ETA_F1)))
    with pytest.raises(ValueError, match="N = 25"):
        F1.fields(K, d[:-1])
    with pytest.raises(ValueError, match="N_in = 25"):
        F1.sheet_mass(d[:-1], K)
    Kbad = sj.Kernels(K.degrees, K.degree_one_frame, K.radius[:-1], K.geoid[:, :-1], K.surface_topography[:, :-1],
                      K.cmb_topography[:, :-1], K.surface_velocity[:, :-1], None)
    with pytest.raises(ValueError, match="nodes"):
        F1.input_kernels(Kbad)
    # fields.apply with concrete radii that differ from the kernels' (c3 physics minor 8).
    p = F1.input_profile
    Kp = fake_kernels(sj.Layers(p, [3480e3, 6370e3]), p.radius)
    with pytest.raises(ValueError, match="other radii"):
        fields.apply(Kp, d, p.replace(radius=np.asarray(p.radius) * (1 + 1e-9)))
    # gravity_anomaly without degrees refuses a non-square pair of last axes (c3 JAX minor 2).
    with pytest.raises(ValueError, match="degrees=Kernels.degrees"):
        fields.gravity_anomaly(np.zeros((5, 25)), 9.8, 6.37e6)
