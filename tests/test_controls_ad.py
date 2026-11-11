"""Derivatives through the control maps (section 6 of
NOTES/chunks/c4-controls-derivatives/01-plan.md; gates G3, G3b, G4, G5, G6, G7, G9, G15,
G16).

Taylor policy of CONVENTIONS 8, unchanged: the ladder ``1e-2 * 2^-k``, ``k = 0..7``, the
window ``(0, 1, 2, 3)``, both direction signs, reverse and forward mode, the corrected
rate lower bound 1.9, the uncorrected band (0.8, 1.2), "unresolved" is not a pass, and
the derivative times 1.01 must fail for every (sign, mode). Each program (P1, P2, P3) is
one ``TaylorProgram`` and one ``DotProgram`` in a module fixture shared by its cases (plan
D9): every case is its own test, and a worker that runs several cases of one program
compiles it once.

Crossing clearance (plan section 6). The derivative with respect to a movable radius
jumps where a moved node or the pair crosses an input radius, and at the construction
radius. A Taylor base point and a one-sided difference must not straddle one: every test
with a movable radius computes the crossings of its table in numpy (``crossing_radii``
below, independent of ``Layers.crossings``) and asserts that its base point is farther
from every crossing than ``CROSSING_CLEARANCE`` times its largest step. T21 sits on the
crossing ``r0`` on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import cases  # noqa: E402
import densities  # noqa: E402
from adapters import SI_OUTPUTS, assert_rows_close, profile_from_table, row_errors  # noqa: E402
from taylor import (DotProgram, LinearTaylorProgram, assert_jvp_vjp, assert_taylor, assess,  # noqa: E402
                    remeasure_taylor, taylor_remainders)

import stokesjx as sj  # noqa: E402
from stokesjx import fields, propagate  # noqa: E402

KM = 1e3
FIELDS = fields.FIELD_NAMES
# CONVENTIONS 8: the dot product on the normalised map G.
DOT_RTOL = 1e-12
# A Taylor base point is farther from every crossing than this many largest steps
# (plan section 6; the largest Taylor step is 1e-2 times the direction amplitude, the
# one-sided difference uses 2h).
CROSSING_CLEARANCE = 20
# G4 (plan 6.1): the reverse-mode derivative at r0 against the forward one-sided
# difference D+ = (-3 F(r0) + 4 F(r0 + h) - F(r0 + 2h)) / (2h), h = 10 m. Truncation
# h^2 F''' / 3 and rounding eps |F| / (h |D|) about 1e-10; measured with the stand-in
# 3.4e-9 (smooth) and 4.6e-9 (rough).
FD_ONE_SIDED_TOL = 1e-7
FD_STEP = 10.0
# G4: the derivative at r0 differs from the backward one-sided difference by at least
# this fraction (the kink at r0; measured 1.8e-1 smooth, 3.2e-2 rough on the stand-in).
FD_KINK_MIN = 1e-3
# S-T26 (plan 6.5: "measured again on the full table and set by the implementer"): the
# kink of the P1-type functional at r0 on the full t_prem() table at lmax 50 is 1.59e-3
# of the derivative (impl-evidence/slow_controls_ad.log); a third of it.
FD_KINK_MIN_FULL = 5e-4
# c2 constants of the scaling invariants (eq. 11): the zero sum of the log_eta
# derivatives of the geoid and topography parts, and the velocity identity.
ZERO_SHIFT_TOL = 1e-10
SHIFT_TOL = 1e-10
# c3 constant: kernels under a common factor of the viscosities.
SCALE_TOL = 1e-12
# c3 constant: the linearity of the fields in the density (max-normalised).
APPLY_TOL = 1e-14
# I5: the vjp of the field map with respect to the density against the kernel row times
# the weights (the same products, relative to the row maximum).
SENSITIVITY_TOL = 1e-15
# c2 constant: vmap against single calls.
VMAP_RTOL = 1e-12
# I9: under NO_SELF_GRAVITY with a uniform g the kernels are exact powers of g (review
# r1 evidence uniform_g_scaling.out: 1e-14 to 6e-16).
G_SCALING_TOL = 1e-12
# The widened Pade plan against the construction plan of the same radii: both are exact
# steps to rounding (measured in impl-evidence/plan_widen.out, NOTES/TOLERANCES.md).
PLAN_WIDEN_TOL = 1e-13
# G15 and G16 (plan 6.4): the four fields of a moved profile and the derivative of a
# geoid misfit with respect to the movable radii against tables built at the new radius
# (reference CK). Three times the measured maximum of the stand-in, rounded up.
MOVED_FIELD_TOL_Q = 5e-3            # G15q, lmax 10, full t_prem() table (stand-in 1.6e-3)
RADIUS_DERIV_TOL_Q = 2.5e-2         # G16q, lmax 10 (stand-in 7.0e-3)
MOVED_FIELD_TOL = 2e-3              # G15-1, G15-1t, lmax 30 (stand-in 5.4e-4, 6.9e-4)
MOVED_FIELD_TOL_MULTI = 1e-2        # G15-3, G15-3t, three boundaries (stand-in 3.25e-3)
RADIUS_DERIV_TOL = 2e-2             # G16 slow rows (stand-in 6.2e-3, 7.2e-3, 2.6e-3)
# G15-node (slow): a movable boundary on an input node and one on the input
# viscosity-only pair at 6270 km whose range ends at the surface, full t_prem() table,
# lmax 30. The shallow range is the least accurate case of the linear density map
# (second order in the input spacing times the stretch; 04-review-physics-r1.md
# section 5): measured 8.9e-3 by the physics review, about four times the deep rows.
MOVED_FIELD_TOL_SHALLOW = 2e-2
# The node-attached anomaly (the negative control of G15 and G16) must miss each
# tolerance by at least this factor (largest over the points).
NEGATIVE_FACTOR = 10.0
# Slow rows: the reference with H/2 against H, over the derivative scale (stand-in at
# most 5e-4 on the full table).
REF_HALF_TOL = 5e-3
# S-G9: kernels at the clamp bounds against the mpmath oracle (CONVENTIONS 10).
ORACLE_TOL = 1e-12
# S-pair exact (review r2 minor 3): under NO_SELF_GRAVITY with a uniform g the interval
# exponent holds no density and the gravity does not depend on rho0, so an inserted pair
# leaves the kernels at the input nodes unchanged to rounding (review r2: 2.3e-14).
PAIR_EXACT_TOL = 1e-12
# S-pair (lovejx INSERTED_PAIR_BOUND): under the default flags an inserted pair with equal
# viscosity splits the interval mean of rho0, a change of second order in the width;
# three times the package value (impl-evidence/s_pair.out; NOTES/TOLERANCES.md).
INSERTED_PAIR_BOUND = 3.5e-7

ETA_F1 = np.array([3e22, 1e22, 1e21, 3e20, 1e22])
ETA_F3 = np.array([1e23, 3e22, 1e22, 1e21, 3e21, 3e20, 1e22])
B_F1 = np.array([3480, 5370, 5700, 5970, 6270, 6370]) * KM
B_F3 = np.array([3480, 4650, 5370, 5700, 5900, 5970, 6270, 6370]) * KM
MOV_F1 = {1: (5105.625 * KM, 5700 * KM)}
MOV_F3 = {1: (4202.5 * KM, 5105.625 * KM), 2: (5105.625 * KM, 5700 * KM), 4: (5700 * KM, 5970 * KM)}
TALA_NSNS = sj.Formulation(compressible=True, surface=sj.NoSlip(), cmb=sj.NoSlip())


# --------------------------------------------------------------------------
# Tables.
# --------------------------------------------------------------------------


def _p_red():
    return profile_from_table(cases.t_prem(reduced=True))


def _p_full():
    return profile_from_table(cases.t_prem())


def layers_f1(p=None):
    return sj.Layers(_p_red() if p is None else p, B_F1, movable=MOV_F1)


def layers_f3():
    return sj.Layers(_p_red(), B_F3, movable=MOV_F3)


def full_layout():
    """F1-full and F3-full (plan section 5): the layouts of F1 and F3 on the full
    ``t_prem()`` table (137 input nodes)."""
    p = _p_full()
    r = np.asarray(p.radius)
    i0 = int(np.flatnonzero(r == 5700e3)[-1])
    sA = 0.5 * (r[np.searchsorted(r, 4665e3) - 1] + r[np.searchsorted(r, 4665e3)])
    sC = 0.5 * (r[i0 + 5] + r[i0 + 6])
    f1 = sj.Layers(p, B_F1, movable={1: (float(r[70]), 5700e3)})
    f3 = sj.Layers(p, np.array([3480e3, sA, 5370e3, 5700e3, sC, 5970e3, 6270e3, 6370e3]),
                   movable={1: (float(r[36]), float(r[70])), 2: (float(r[70]), 5700e3), 4: (5700e3, 5970e3)})
    return p, f1, f3


# --------------------------------------------------------------------------
# Crossings (numpy, independent of Layers.crossings).
# --------------------------------------------------------------------------


def crossing_radii(L: sj.Layers, k: int) -> np.ndarray:
    """Every boundary radius of range ``k`` at which a moved node or the pair sits on an
    input radius strictly inside the range, and r0."""
    g = L.ranges[k]
    r_in = np.asarray(L.input_profile.radius)
    r_c = np.asarray(L.profile.radius)
    kn = r_in[(r_in > g.low) & (r_in < g.high)]
    out = [kn, [g.start]]
    for j in g.below:
        out.append(g.low + (kn - g.low) * (g.start - g.low) / (r_c[j] - g.low))
    for j in g.above:
        out.append(g.high - (g.high - kn) * (g.high - g.start) / (g.high - r_c[j]))
    return np.sort(np.concatenate([np.atleast_1d(np.asarray(o, float)) for o in out]))


def nearest_crossing(L, radii) -> np.ndarray:
    """Distance of each movable radius to the nearest crossing of its range, m."""
    return np.array([float(np.min(np.abs(crossing_radii(L, k) - x))) for k, x in zip(L.movable_index, radii)])


def assert_clearance(L, radii, largest_step):
    d = nearest_crossing(L, radii)
    assert np.all(d > CROSSING_CLEARANCE * largest_step), (d, largest_step)
    return d


def widest_gap_base(L, lo_off, hi_off):
    """Per movable boundary, the middle of the widest gap between consecutive crossings
    in ``r0 + [lo_off, hi_off]``."""
    out = []
    for k in L.movable_index:
        g = L.ranges[k]
        c = crossing_radii(L, k)
        lo, hi = g.start + lo_off, g.start + hi_off
        cc = np.sort(np.concatenate([c[(c > lo) & (c < hi)], [lo, hi]]))
        j = int(np.argmax(np.diff(cc)))
        out.append(0.5 * (cc[j] + cc[j + 1]))
    return np.array(out)


# --------------------------------------------------------------------------
# Programs: a functional linear in the outputs (the four kernels and the four fields).
# --------------------------------------------------------------------------


class Program:
    """The functional F (plan 6.1) and the map U of the eight outputs of a control map.

    ``U(c)`` is the tuple of the four kernels of ``stokes_kernels(to_profile(c), cfg)`` and
    the four fields of ``L.fields(K, density)``. ``F(c) = sum_o sum(w_o U_o / s_o)`` with
    fixed random weights ``w_o`` and the scales ``s_o = max |U_o(c0)|`` of the base point
    ``c0``: the normalised map ``G = U / s`` of CONVENTIONS 8 contracted with ``w``. An output
    that is an exact zero at ``c0`` (the velocity under a no-slip surface) has the weight
    zero, so it is left out of F and of the dot product (CONVENTIONS 8, chunk 3).

    One ``DotProgram`` of U serves everything: the scales come from the primal output of
    its jvp program at ``c0`` (no separate forward compile), the Taylor derivatives of F
    from its jvp and vjp (``LinearTaylorProgram``), and the dot product on G is the dot
    product on U with the cotangent ``r / s`` (``cotangent``), which equals
    ``<r, J_G dx> = <J_G^T r, dx>``."""

    def __init__(self, L, cfg, to_profile, c0, density, seed=4):
        self.L, self.cfg, self.to_profile, self.density = L, cfg, to_profile, jnp.asarray(density)
        self.dot = DotProgram(self.U)
        U0, _ = self.dot.jvp_both(c0, zeros_like(c0))
        self.scales = [float(np.max(np.abs(np.asarray(a)))) for a in U0]
        self.on = [s > 0 for s in self.scales]
        self.names = list(SI_OUTPUTS) + list(FIELDS)
        self.shapes = [np.shape(a) for a in U0]
        rng = np.random.default_rng(seed)
        raw = [rng.standard_normal(sh) for sh in self.shapes]
        self.weights = tuple(jnp.asarray(r / s if on else 0.0 * r) for r, s, on in zip(raw, self.scales, self.on))
        self.taylor = LinearTaylorProgram(self.F, self.dot, self.weights)

    def U(self, c):
        q = self.to_profile(c)
        K = sj.stokes_kernels(q, self.cfg)
        Fd = self.L.fields(K, self.density)
        return tuple([getattr(K, o) for o in SI_OUTPUTS] + [getattr(Fd, n) for n in FIELDS])

    def F(self, c):
        return sum(jnp.sum(w * u) for w, u in zip(self.weights, self.U(c)))

    def cotangent(self, seed, keep=lambda name: True):
        """A cotangent of U equal to a random cotangent ``r`` of the normalised map G:
        ``r / s`` on the outputs that are not exact zeros and for which ``keep(name)``."""
        rng = np.random.default_rng(seed)
        out = []
        for sh, s, on, n in zip(self.shapes, self.scales, self.on, self.names):
            r = rng.standard_normal(sh)
            out.append(jnp.asarray(r / s if (on and keep(n)) else 0.0 * r))
        return tuple(out)


def zeros_like(c):
    return jax.tree_util.tree_map(lambda a: np.zeros(np.shape(a)), c)


def direction(c, **leaves):
    d = zeros_like(c)
    d.update({k: np.asarray(v, dtype=np.float64) for k, v in leaves.items()})
    return d


def assert_taylor_with_controls(prog, x, dx, **kw):
    """assert_taylor on the program, then the four 1.01 controls must fail."""
    res = assert_taylor(prog.taylor, x, dx, **kw)
    verdicts = [assess(remeasure_taylor(r, 1.01 * r.derivative))[0] for r in res]
    assert all(v != "pass" for v in verdicts), verdicts
    rung3 = min(r.r1[3] / r.allowance for r in res)
    print(f"Taylor pass; 1.01 controls {verdicts}; smallest rung-3 corrected remainder {rung3:.0f} allowances")
    return res


# --------------------------------------------------------------------------
# P1: default flags, all controls but g (plan 6.1).
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def P1():
    L = layers_f1()
    cfg = sj.StokesConfig(lmax=20)
    d = densities.smooth(np.asarray(L.input_profile.radius), 20, 4)

    def to_profile(c):
        return L(c["radii"], c["log_eta"], rho0=c["rho0"]).replace(core_density=c["core_density"])
    c0 = dict(radii=np.asarray(L.boundaries), log_eta=np.log(ETA_F1), rho0=np.asarray(L.profile.rho0),
              core_density=np.asarray(float(L.profile.core_density)))
    c1 = dict(c0, radii=c0["radii"] + np.array([0, 31e3, 0, 0, 0, 0]))
    prog = Program(L, cfg, to_profile, c0, d)
    return dict(L=L, prog=prog, c0=c0, c1=c1, cfg=cfg, d=d)


# Every case of a program is its own test (NOTES/PLAN.md ruling 12: independent tests that
# pytest-xdist can distribute). The module fixture holds the compiled programs, so a
# worker that runs several cases of one program compiles it once; a case that lands on
# another worker compiles the program there. No case writes to its fixture.


P1_CASES = ("_t19_log_eta", "_t20_radius_away_from_r0", "_t21_g4_radius_at_r0", "_t22_rho0", "_t23_core_density",
            "_t24_all_controls", "_dp1_dot_product", "_i1_i2_shift_identities", "_i3_i4_i5_i6_values")


@pytest.mark.parametrize("case", P1_CASES, ids=[c[1:] for c in P1_CASES])
def test_p1(P1, case):
    """P1 (plan 6.1): T19 to T24, DP1 and the invariants I1 to I6, one test per case on
    the compiled programs of the fixture."""
    globals()[case](P1)


def _t19_log_eta(P1):
    x = P1["c1"]
    assert_clearance(P1["L"], [x["radii"][1]], 0.0)
    rng = np.random.default_rng(19)
    assert_taylor_with_controls(P1["prog"], x, direction(x, log_eta=0.1 * rng.standard_normal(5)))


def _t20_radius_away_from_r0(P1):
    x = P1["c1"]
    d = assert_clearance(P1["L"], [x["radii"][1]], 1e-2 * 5000.0)
    print(f"T20: nearest crossing {d[0] / 1e3:.3f} km from the base point")
    assert_taylor_with_controls(P1["prog"], x, direction(x, radii=np.array([0, 5000.0, 0, 0, 0, 0])))


def _t21_g4_radius_at_r0(P1):
    """T21 and G4: at r0 the derivative is the upper one-sided derivative. Sign +1 passes
    in both modes; sign -1 fails; the one-sided differences D+ and D- (h = 10 m) through
    the same compiled program."""
    L, prog, x = P1["L"], P1["prog"], P1["c0"]
    c = crossing_radii(L, 1)
    others = c[c != L.ranges[1].start]
    assert np.min(np.abs(others - L.ranges[1].start)) > CROSSING_CLEARANCE * 2 * FD_STEP
    dx = direction(x, radii=np.array([0, 1000.0, 0, 0, 0, 0]))
    assert_taylor(prog.taylor, x, dx, signs=(1.0,))
    for mode in ("reverse", "forward"):
        res = taylor_remainders(prog.taylor, x, dx, sign=-1.0, mode=mode)
        verdict, reason = assess(res)
        print(f"T21 sign -1 {mode}: {verdict} ({reason})")
        assert verdict == "fail"
    unit = direction(x, radii=np.array([0, 1.0, 0, 0, 0, 0]))
    h = FD_STEP
    v = prog.taylor.values(x, unit, [0.0, h, 2 * h, -h, -2 * h])
    Dp = (-3 * v[0] + 4 * v[1] - v[2]) / (2 * h)
    Dm = (3 * v[0] - 4 * v[3] + v[4]) / (2 * h)
    for mode in ("reverse", "forward"):
        D = prog.taylor.derivative(x, unit, mode)
        print(f"G4 {mode}: D = {D:.10e}, |D - D+|/|D| = {abs(D - Dp) / abs(D):.3e}, "
              f"|D - D-|/|D| = {abs(D - Dm) / abs(D):.3e}")
        assert abs(D - Dp) <= FD_ONE_SIDED_TOL * abs(D)
        assert abs(D - Dm) >= FD_KINK_MIN * abs(D)


def _t22_rho0(P1):
    x = P1["c1"]
    L = P1["L"]
    rng = np.random.default_rng(22)
    dx = direction(x, rho0=50.0 * rng.standard_normal(L.n_nodes))
    assert_taylor_with_controls(P1["prog"], x, dx)
    prog = P1["prog"]
    g = prog.dot.vjp(x, prog.taylor.weights)
    assert np.all(np.asarray(g["rho0"])[list(L.ranges[1].members)] == 0.0)


def _t23_core_density(P1):
    x = P1["c1"]
    assert_taylor_with_controls(P1["prog"], x, direction(x, core_density=0.05 * float(x["core_density"])))


def _t24_all_controls(P1):
    x = P1["c1"]
    L = P1["L"]
    rng = np.random.default_rng(24)
    dx = direction(x, log_eta=0.1 * rng.standard_normal(5), radii=np.array([0, 5000.0, 0, 0, 0, 0]),
                   rho0=50.0 * rng.standard_normal(L.n_nodes), core_density=0.05 * float(x["core_density"]))
    assert_taylor_with_controls(P1["prog"], x, dx)


def _dp1_dot_product(P1):
    x = P1["c1"]
    rng = np.random.default_rng(31)
    dx = dict(radii=1000.0 * rng.standard_normal(6), log_eta=rng.standard_normal(5),
              rho0=50.0 * rng.standard_normal(P1["L"].n_nodes), core_density=np.asarray(500.0 * rng.standard_normal()))
    rel = assert_jvp_vjp(P1["prog"].dot, x, dx=dx, dy=P1["prog"].cotangent(32), rtol=DOT_RTOL)
    print(f"DP1: {rel:.2e}")


def _i1_i2_shift_identities(P1):
    """I1: the log_eta part of the pulled-back cotangent of the geoid and topography
    outputs (kernels and fields) sums to zero. I2: with the velocity cotangent the sum is
    -<dy, U_vel> (the velocity scales with 1/eta)."""
    prog, x = P1["prog"], P1["c1"]
    y, _ = prog.dot.jvp_both(x, zeros_like(x))
    dy_gt = prog.cotangent(41, keep=lambda n: n != "surface_velocity")
    dy_v = prog.cotangent(41, keep=lambda n: n == "surface_velocity")
    g = np.asarray(prog.dot.vjp(x, dy_gt)["log_eta"])
    print(f"I1: sum {np.sum(g):.3e}, sum of moduli {np.sum(np.abs(g)):.3e}")
    assert abs(np.sum(g)) <= ZERO_SHIFT_TOL * np.sum(np.abs(g)), (np.sum(g), np.sum(np.abs(g)))
    gv = np.asarray(prog.dot.vjp(x, dy_v)["log_eta"])
    fv = sum(float(np.sum(np.asarray(a) * np.asarray(b))) for a, b in zip(dy_v, y))
    print(f"I2: sum {np.sum(gv):.6e}, -<dy, U_vel> {-fv:.6e}")
    assert abs(np.sum(gv) + fv) <= SHIFT_TOL * abs(fv), (np.sum(gv), fv)


def _i3_i4_i5_i6_values(P1):
    """I3: kernels at log_eta + 0.7. I4: linearity in the density. I5: the kernel is the
    sensitivity. I6: the density amplitude is an affine Taylor case."""
    L, x, cfg = P1["L"], P1["c1"], P1["cfg"]
    prog = P1["prog"]
    K = sj.stokes_kernels(prog.to_profile(x), cfg)
    Ks = sj.stokes_kernels(prog.to_profile(dict(x, log_eta=x["log_eta"] + 0.7)), cfg)
    for o in ("geoid", "surface_topography", "cmb_topography"):
        assert_rows_close(np.asarray(getattr(Ks, o)), np.asarray(getattr(K, o)), SCALE_TOL, o)
    assert_rows_close(np.exp(0.7) * np.asarray(Ks.surface_velocity), np.asarray(K.surface_velocity), SCALE_TOL, "vel")
    r_in = np.asarray(L.input_profile.radius)
    d1, d2 = densities.smooth(r_in, 20, 7), densities.rough(r_in, 20, 8)
    a, b = 0.3, -1.7
    Fab = L.fields(K, a * d1 + b * d2)
    F1_, F2_ = L.fields(K, d1), L.fields(K, d2)
    for n in FIELDS:
        lhs = np.asarray(getattr(Fab, n))
        rhs = a * np.asarray(getattr(F1_, n)) + b * np.asarray(getattr(F2_, n))
        assert np.max(np.abs(lhs - rhs)) <= APPLY_TOL * np.max(np.abs(lhs)), n
    Kin = L.input_kernels(K)
    _, vjp = jax.vjp(lambda d: L.fields(K, d).geoid, jnp.asarray(d1))
    for l, m in ((2, 0), (7, 3), (20, 20)):
        e = np.zeros((2, 21, 21)); e[0, l, m] = 1.0
        got = np.asarray(vjp(jnp.asarray(e))[0])[:, 0, l, m]
        ref = np.asarray(Kin.geoid)[Kin.row(l)] * L.input_weights
        assert np.max(np.abs(got - ref)) <= SENSITIVITY_TOL * np.max(np.abs(ref)), (l, m)
    v = np.random.default_rng(5).standard_normal((2, 21, 21))
    f = lambda s: jnp.sum(v * L.fields(K, jnp.asarray(d2) + s * jnp.asarray(d1)).geoid)
    assert_taylor(f, np.asarray(0.0), np.asarray(1.0), classification="affine")


def test_i7_i8_vmap_and_one_trace(P1):
    """I7: ControlVector({radii_movable, log_eta}) with the density as data, vmap of
    value_and_grad over 4 vectors against 4 calls. I8: the first single call through the
    NumPy front traces once, the further JAX calls with new values do not trace again, and
    the jit cache has one entry."""
    L, prog = P1["L"], P1["prog"]
    c_ex = {"radii_movable": np.array([5401e3]), "log_eta": np.log(ETA_F1)}
    cv = sj.ControlVector(c_ex)

    counter = {"n": 0}

    def fn(c, d):
        counter["n"] += 1
        q = L(L.full_boundaries(c["radii_movable"]), c["log_eta"])
        K = sj.stokes_kernels(q, prog.cfg)
        Fd = L.fields(K, d)
        U = [getattr(K, o) for o in SI_OUTPUTS] + [getattr(Fd, n) for n in FIELDS]
        return sum(jnp.sum(w * u) for w, u in zip(prog.weights, U))
    rng = np.random.default_rng(7)
    X = np.stack([cv.ravel(c_ex) for _ in range(4)])
    X = np.asarray(X).copy()
    X[:, 5] += 20e3 * (2 * rng.random(4) - 1)
    X[:, :5] += 0.2 * rng.standard_normal((4, 5))
    # The core of the solver is an inner jit: when another test of the worker traced it
    # with the same shapes, JAX reuses that trace, so the core counter grows by at most
    # one; the Python counter of fn grows by exactly one.
    n0, calls0 = propagate.trace_count(), counter["n"]
    f_np = cv.numpy_value_and_grad(fn, P1["d"])
    v0, g0 = f_np(X[0])
    assert counter["n"] == calls0 + 1 and propagate.trace_count() <= n0 + 1
    n1 = propagate.trace_count()
    single = cv.value_and_grad(fn)
    vals, grads = [v0], [g0]
    for b in range(1, 4):
        v, g = single(jnp.asarray(X[b]), jnp.asarray(P1["d"]))
        vals.append(float(v)); grads.append(np.asarray(g))
    assert counter["n"] == calls0 + 1 and propagate.trace_count() == n1 and single._cache_size() == 1
    vv, gv = jax.jit(jax.vmap(single, in_axes=(0, None)))(jnp.asarray(X), jnp.asarray(P1["d"]))
    assert np.max(np.abs(np.asarray(vv) - vals)) <= VMAP_RTOL * np.max(np.abs(vals))
    G = np.stack(grads)
    assert np.max(np.abs(np.asarray(gv) - G)) <= VMAP_RTOL * np.max(np.abs(G))


# --------------------------------------------------------------------------
# P2: uniform gravity (plan 6.2) and I9.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def P2():
    L = layers_f1()
    form = sj.Formulation(background=sj.Uniform())
    cfg = sj.StokesConfig(lmax=20, formulation=form)
    d = densities.smooth(np.asarray(L.input_profile.radius), 20, 4)

    def to_profile(c):
        return L(c["radii"], c["log_eta"]).replace(uniform_gravity=c["uniform_gravity"])
    c1 = dict(radii=np.asarray(L.boundaries) + np.array([0, 31e3, 0, 0, 0, 0]), log_eta=np.log(ETA_F1),
              uniform_gravity=np.asarray(9.81))
    return dict(L=L, prog=Program(L, cfg, to_profile, c1, d), c1=c1)


@pytest.mark.parametrize("case", ("_t25_uniform_g", "_dp2_dot_product"), ids=("t25_uniform_g", "dp2_dot_product"))
def test_p2(P2, case):
    """P2 (plan 6.2): T25 and DP2, one test per case on the compiled program of the
    fixture."""
    globals()[case](P2)


def _t25_uniform_g(P2):
    x = P2["c1"]
    assert_taylor_with_controls(P2["prog"], x, direction(x, uniform_gravity=0.05 * 9.81))


def _dp2_dot_product(P2):
    x = P2["c1"]
    rng = np.random.default_rng(32)
    dx = dict(radii=1000.0 * rng.standard_normal(6), log_eta=rng.standard_normal(5),
              uniform_gravity=np.asarray(rng.standard_normal()))
    print(f"DP2: {assert_jvp_vjp(P2['prog'].dot, x, dx=dx, dy=P2['prog'].cotangent(33), rtol=DOT_RTOL):.2e}")


def test_i9_uniform_g_scaling():
    """I9: NO_SELF_GRAVITY, Uniform(None), reduced layering, lmax 10: g dK_geoid/dg =
    -K_geoid, g dK_vel/dg = K_vel, dK_topo/dg = 0 (one jvp)."""
    L = layers_f1()
    form = sj.Formulation(gravity=sj.GravityModel.NO_SELF_GRAVITY, background=sj.Uniform())
    cfg = sj.StokesConfig(lmax=10, formulation=form)
    q = L(L.boundaries, np.log(ETA_F1))
    g0 = 9.81

    def kern(g):
        K = sj.stokes_kernels(q.replace(uniform_gravity=g), cfg)
        return tuple(getattr(K, o) for o in SI_OUTPUTS)
    K, dK = jax.jit(lambda g: jax.jvp(kern, (g,), (jnp.ones(()),)))(jnp.asarray(g0))
    K = dict(zip(SI_OUTPUTS, map(np.asarray, K)))
    dK = dict(zip(SI_OUTPUTS, map(np.asarray, dK)))
    assert_rows_close(g0 * dK["geoid"], -K["geoid"], G_SCALING_TOL, "geoid")
    assert_rows_close(g0 * dK["surface_velocity"], K["surface_velocity"], G_SCALING_TOL, "velocity")
    for o in ("surface_topography", "cmb_topography"):
        assert np.max(np.abs(g0 * dK[o])) <= G_SCALING_TOL * np.max(np.abs(K[o])), o


# --------------------------------------------------------------------------
# P3: three movable radii (plan 6.6).
# --------------------------------------------------------------------------


def _program_p3(kind):
    L = layers_f3()
    cfg = sj.StokesConfig(lmax=20)
    r_in = np.asarray(L.input_profile.radius)
    d = densities.smooth(r_in, 20, 4) if kind == "smooth" else densities.rough(r_in, 20, 5)

    def to_profile(c):
        return L(L.full_boundaries(c["movable_radii"]), c["log_eta"])
    base = widest_gap_base(L, 5e3, 60e3)
    c0 = dict(movable_radii=L.start_radii.copy(), log_eta=np.log(ETA_F3))
    c1 = dict(c0, movable_radii=base)
    return dict(L=L, prog=Program(L, cfg, to_profile, c0, d), c1=c1)


@pytest.fixture(scope="module")
def P3():
    return _program_p3("smooth")


@pytest.mark.parametrize("case", ("_t27", "_dp3_dot_product"), ids=("t27", "dp3_dot_product"))
def test_p3(P3, case):
    """P3 (plan 6.6): T27 and DP3, one test per case on the compiled program of the
    fixture."""
    globals()[case](P3)


def _t27(P):
    L, x = P["L"], P["c1"]
    dx = direction(x, movable_radii=np.array([5000.0, -3500.0, 2500.0]))
    d = assert_clearance(L, x["movable_radii"], 1e-2 * 5000.0)
    print(f"T27 base r_b - r0 = {(x['movable_radii'] - L.start_radii) / 1e3} km, nearest crossings {d / 1e3} km")
    assert_taylor_with_controls(P["prog"], x, dx)


def _dp3_dot_product(P3):
    x = P3["c1"]
    rng = np.random.default_rng(33)
    dx = dict(movable_radii=1000.0 * rng.standard_normal(3), log_eta=rng.standard_normal(7))
    print(f"DP3: {assert_jvp_vjp(P3['prog'].dot, x, dx=dx, dy=P3['prog'].cotangent(34), rtol=DOT_RTOL):.2e}")


def test_plan_widen():
    """The widened plan (Layers structure with limits) against the construction plan of
    the same concrete radii (a fresh structure): both are exact steps to rounding, and the
    two plans differ (review r3 minor 11)."""
    L = layers_f1()
    cfg = sj.StokesConfig(lmax=20)
    q = L(L.full_boundaries(np.array([5400e3])), np.log(ETA_F1))
    fresh = sj.RadialProfile(np.asarray(q.radius), np.asarray(q.rho0), np.asarray(q.eta),
                             core_density=float(q.core_density), core_mass=float(q.core_mass))
    pa = propagate.static_plan(q.structure, cfg).pade
    pb = propagate.static_plan(fresh.structure, cfg).pade
    print(f"widened plan {pa}, construction plan of the moved radii {pb}")
    assert (pa.order, pa.squarings) != (pb.order, pb.squarings)
    Ka, Kb = sj.stokes_kernels(q, cfg), sj.stokes_kernels(fresh, cfg)
    worst = max(assert_rows_close(np.asarray(getattr(Ka, o)), np.asarray(getattr(Kb, o)), PLAN_WIDEN_TOL, o)
                for o in SI_OUTPUTS)
    print(f"plan widen: largest e_l {worst:.2e}")


# --------------------------------------------------------------------------
# G4 on a wide range (04-review-jax-r1.md, B1).
# --------------------------------------------------------------------------

# The named layering of B1: reduced PREM, one boundary at this radius (m), movable in the
# whole lower mantle (3480, 5700) km. With an affine map whose constant factors XLA could
# fold into a value other than r_c at r0, one input node sat one ulp below its input
# radius under jit, and the jitted derivative at r0 was a mix of the two one-sided slopes
# (9.6 % from D+). The fields: geoid of the smooth anomaly at lmax 20.
B1_RADIUS = 3720276.3768699113
B1_LMAX = 20
B1_ETA = np.array([3e22, 1e22, 1e21, 1e22])
# Two steps of the one-sided difference: D+ must agree at both, which shows that the
# truncation of the difference is below the tolerance on this range (04-review-jax-r1.md
# minor 9; measured |D - D+| / |D| 4e-9 to 8e-9 at h = 5, 10, 20 m).
B1_FD_STEPS = (10.0, 20.0)
# The jitted derivative against the eager derivative of the same function: the same
# arithmetic up to the fusion order of XLA.
JIT_EAGER_RTOL = 1e-12


def test_g4_wide_range():
    """G4 on the B1 layering, jitted reverse mode (jit(grad)) and jitted forward mode
    (jit(jvp)): |D - D+| <= FD_ONE_SIDED_TOL |D| at both steps of B1_FD_STEPS, the kink at
    r0 resolved (|D - D-| >= FD_KINK_MIN |D|), and the eager reverse-mode derivative equal
    to the jitted one. The density, the observed field and the weights are arguments of
    the compiled functions."""
    p = _p_red()
    L = sj.Layers(p, np.array([3480e3, B1_RADIUS, 5700e3, 5970e3, 6370e3]), movable={1: (3480e3, 5700e3)})
    cfg = sj.StokesConfig(lmax=B1_LMAX)
    le = np.log(B1_ETA)
    r0 = float(L.start_radii[0])
    # No other crossing within the reach of the largest one-sided difference (2 h).
    others = crossing_radii(L, 1)
    others = others[others != r0]
    assert np.min(np.abs(others - r0)) > CROSSING_CLEARANCE * 2 * max(B1_FD_STEPS)

    def geoid(x, d):
        return L.fields(sj.stokes_kernels(L(L.full_boundaries(x[None]), le), cfg), d).geoid

    def misfit(x, d, obs, w):
        return fields.spectral_misfit(geoid(x, d), obs, w, "real_4pi")

    d = jnp.asarray(densities.smooth(np.asarray(p.radius), B1_LMAX, 4))
    g0 = jax.jit(geoid)(jnp.asarray(r0), d)
    obs = g0 * (1.0 + 0.3 * jnp.asarray(np.sin(np.arange(B1_LMAX + 1)))[None, :, None])
    w = jnp.asarray(np.where(np.arange(B1_LMAX + 1) >= 2, 1.0, 0.0) / float(jnp.sum(g0[:, 2:] ** 2)))
    args = (d, obs, w)
    value = jax.jit(misfit)
    D = {"reverse": float(jax.jit(jax.grad(misfit))(jnp.asarray(r0), *args)),
         "forward": float(jax.jit(lambda x, *a: jax.jvp(lambda y: misfit(y, *a), (x,), (jnp.ones(()),))[1])(
             jnp.asarray(r0), *args))}
    D_eager = float(jax.grad(misfit)(jnp.asarray(r0), *args))
    print(f"G4 wide: D reverse {D['reverse']:.10e}, forward {D['forward']:.10e}, eager {D_eager:.10e}")
    assert abs(D["reverse"] - D_eager) <= JIT_EAGER_RTOL * abs(D_eager)
    for h in B1_FD_STEPS:
        v = [float(value(jnp.asarray(r0 + t), *args)) for t in (0.0, h, 2 * h, -h, -2 * h)]
        Dp = (-3 * v[0] + 4 * v[1] - v[2]) / (2 * h)
        Dm = (3 * v[0] - 4 * v[3] + v[4]) / (2 * h)
        for mode, D_mode in D.items():
            print(f"G4 wide h = {h:.0f} m {mode}: |D - D+|/|D| = {abs(D_mode - Dp) / abs(D_mode):.3e}, "
                  f"|D - D-|/|D| = {abs(D_mode - Dm) / abs(D_mode):.3e}")
            assert abs(D_mode - Dp) <= FD_ONE_SIDED_TOL * abs(D_mode)
            assert abs(D_mode - Dm) >= FD_KINK_MIN * abs(D_mode)


# --------------------------------------------------------------------------
# G15 and G16: moved fields against tables built at the new radius (plan 6.4).
# --------------------------------------------------------------------------


def _fresh(L, x):
    """Reference CK: a Layers on the input profile with every movable boundary at ``x``
    and no movable boundary (its density map is a static gather)."""
    b = L.boundaries.copy()
    b[L.movable_index] = x
    return sj.Layers(L.input_profile, b)


def _interp_matrix(L):
    """The input anomaly to the construction table at the construction radii (the
    node-attached negative control interpolates the anomaly once with it)."""
    r_in = np.asarray(L.input_profile.radius)
    r_c = np.asarray(L.profile.radius)
    P = np.zeros((L.n_nodes, r_in.size))
    for i in range(r_in.size):
        e = np.zeros(r_in.size); e[i] = 1.0
        P[:, i] = np.interp(r_c, r_in, e)
    P[L.input_node, :] = np.eye(r_in.size)
    return P


class Gate:
    """The moved fields (K), the node-attached control (A) and the fresh reference (CK) of
    one layering, flags and lmax."""

    def __init__(self, L, eta, cfg):
        self.L, self.cfg, self.le = L, cfg, np.log(eta)
        self.P = jnp.asarray(_interp_matrix(L))
        self.fK = jax.jit(lambda x, d: self.moved(x, d))
        self.fA = jax.jit(lambda x, d: self.attached(x, d))

    def moved(self, x, d):
        q = self.L(self.L.full_boundaries(x), self.le)
        Fd = self.L.fields(sj.stokes_kernels(q, self.cfg), d)
        return [getattr(Fd, n) for n in FIELDS]

    def attached(self, x, d):
        q = self.L(self.L.full_boundaries(x), self.le)
        K = sj.stokes_kernels(q, self.cfg)
        dc = jnp.tensordot(self.P, d, axes=(1, 0))
        Fd = fields.radial_integral(K, dc, fields.quadrature_weights(q))
        return [getattr(Fd, n) for n in FIELDS]

    def fresh(self, x, d):
        Lf = _fresh(self.L, x)
        q = Lf(Lf.boundaries, self.le)
        Fd = Lf.fields(sj.stokes_kernels(q, self.cfg), d)
        return [np.asarray(getattr(Fd, n)) for n in FIELDS]


def rel_lge2(a, b):
    """Relative 2-norm over l >= 2; 0 when both are exact zeros, inf when only the
    reference is."""
    a, b = np.asarray(a)[..., 2:, :], np.asarray(b)[..., 2:, :]
    nb = np.linalg.norm(b)
    if nb == 0.0:
        return 0.0 if np.linalg.norm(a) == 0.0 else np.inf
    return float(np.linalg.norm(a - b) / nb)


def gate_fields(L, eta, cfg, positions, tol, label, zero_velocity=False):
    """G15: every field, both anomalies, every position within ``tol``; the node-attached
    control misses ``tol`` by NEGATIVE_FACTOR (largest over the points)."""
    gate = Gate(L, eta, cfg)
    r_in = np.asarray(L.input_profile.radius)
    worst, worst_a = 0.0, 0.0
    for name, d in (("smooth", densities.smooth(r_in, cfg.lmax, 11)), ("rough", densities.rough(r_in, cfg.lmax, 12))):
        for x in positions:
            ref = gate.fresh(x, d)
            K = gate.fK(jnp.asarray(x), jnp.asarray(d))
            A = gate.fA(jnp.asarray(x), jnp.asarray(d))
            eK, eA = [], []
            for k, n in enumerate(FIELDS):
                if zero_velocity and n == "surface_velocity":
                    # Review r3 minor 6: under a no-slip surface the velocity field is an
                    # exact zero for the moved table and for the reference.
                    assert np.all(np.asarray(K[k]) == 0.0) and np.all(ref[k] == 0.0)
                    continue
                eK.append(rel_lge2(K[k], ref[k]))
                eA.append(rel_lge2(A[k], ref[k]))
            worst, worst_a = max(worst, max(eK)), max(worst_a, max(eA))
            print(f"{label} {name:6s} r_b - r0 = {np.round((np.asarray(x) - L.start_radii) / 1e3, 3)} km: "
                  f"K vs CK {max(eK):.2e} (fields {', '.join(f'{e:.1e}' for e in eK)}), attached {max(eA):.2e}")
            assert max(eK) <= tol, (label, name, x, eK)
    print(f"{label}: largest K error {worst:.2e}, tolerance {tol:.1e}; largest attached error {worst_a:.2e} "
          f"= {worst_a / tol:.0f} x tolerance")
    assert worst_a >= NEGATIVE_FACTOR * tol


def mid_interval(L, x):
    """The middle of the input interval that holds ``x`` and a quarter of its width."""
    r = np.asarray(L.input_profile.radius)
    j = int(np.searchsorted(r, x)) - 1
    return 0.5 * (r[j] + r[j + 1]), 0.25 * (r[j + 1] - r[j])


def gate_derivative(L, eta, cfg, targets, tol, label, half=False):
    """G16: the derivative of a geoid misfit with respect to the movable radii against
    central differences of the fresh references at ``x +- H_k e_k`` (``H_k`` a quarter of
    the input interval that holds ``x_k``), error over the largest reference derivative;
    the node-attached control misses ``tol`` by NEGATIVE_FACTOR."""
    gate = Gate(L, eta, cfg)
    r_in = np.asarray(L.input_profile.radius)
    w = np.ones(cfg.lmax + 1); w[:2] = 0.0
    worst, worst_a = 0.0, 0.0
    def phi(N, obs, wl):
        return fields.spectral_misfit(N, obs, wl, "real_4pi")
    # One gradient program per treatment for both anomalies: the density, the observed
    # geoid and the weights are arguments.
    gK = jax.jit(jax.grad(lambda x, d, obs, wl: phi(gate.moved(x, d)[0], obs, wl)))
    gA = jax.jit(jax.grad(lambda x, d, obs, wl: phi(gate.attached(x, d)[0], obs, wl)))
    for name, d in (("smooth", densities.smooth(r_in, cfg.lmax, 11)), ("rough", densities.rough(r_in, cfg.lmax, 12))):
        N0 = gate.fresh(L.start_radii + 1.0, d)[0]
        obs = jnp.asarray(N0 * (1.0 + 0.2 * np.sin(np.arange(cfg.lmax + 1)))[None, :, None])
        wl = jnp.asarray(0.5 * w / float(np.sum(N0[:, 2:] ** 2)))
        args = (jnp.asarray(d), obs, wl)
        rows = []
        for t in targets:
            mids = [mid_interval(L, v) for v in t]
            x = np.array([m[0] for m in mids]); H = np.array([m[1] for m in mids])
            assert np.all(x != L.start_radii)               # review r3 minor 4
            # Plan 6.4: a base point never lies in the input interval that holds r0 of
            # its range (the interval whose moved nodes sit next to the pair at r0).
            assert np.all(np.searchsorted(r_in, x) != np.searchsorted(r_in, L.start_radii)), (label, x)
            dist = nearest_crossing(L, x)
            ref, ref2 = np.zeros(x.size), np.zeros(x.size)
            for k in range(x.size):
                e = np.zeros(x.size); e[k] = H[k]
                ref[k] = (float(phi(gate.fresh(x + e, d)[0], obs, wl))
                          - float(phi(gate.fresh(x - e, d)[0], obs, wl))) / (2 * H[k])
                if half:
                    ref2[k] = (float(phi(gate.fresh(x + e / 2, d)[0], obs, wl))
                               - float(phi(gate.fresh(x - e / 2, d)[0], obs, wl))) / H[k]
            rows.append((x, dist, np.asarray(gK(jnp.asarray(x), *args)), np.asarray(gA(jnp.asarray(x), *args)),
                         ref, ref2))
        scale = max(np.max(np.abs(r[4])) for r in rows)
        for x, dist, DK, DA, ref, ref2 in rows:
            eK = float(np.max(np.abs(DK - ref))) / scale
            eA = float(np.max(np.abs(DA - ref))) / scale
            eH = float(np.max(np.abs(ref2 - ref))) / scale if half else float("nan")
            worst, worst_a = max(worst, eK), max(worst_a, eA)
            print(f"{label} {name:6s} r_b - r0 = {np.round((x - L.start_radii) / 1e3, 2)} km (nearest crossing "
                  f"{np.round(dist, 3)} m): D {DK} D_CK {ref} error/scale {eK:.2e}; attached {eA:.2e}; H/2 {eH:.1e}")
            assert eK <= tol, (label, name, x, eK)
            if half:
                assert eH <= REF_HALF_TOL, (label, name, x, eH)
    print(f"{label}: largest derivative error {worst:.2e}, tolerance {tol:.1e}; largest attached error "
          f"{worst_a:.2e} = {worst_a / tol:.0f} x tolerance")
    assert worst_a >= NEGATIVE_FACTOR * tol


def _corners(L):
    lo, hi = L.radius_bounds[:, 0], L.radius_bounds[:, 1]
    alt = np.arange(L.movable_index.size) % 2 == 0
    return [lo + 50.0, hi - 50.0, np.where(alt, lo + 50.0, hi - 50.0), np.where(alt, hi - 50.0, lo + 50.0)]


@pytest.fixture(scope="module")
def full_tables():
    return full_layout()


def test_g15q_1(full_tables):
    _, f1, _ = full_tables
    g = f1.ranges[1]
    pos = [np.array([g.low + g.margin + 50.0]), np.array([g.start - 50e3]), np.array([g.start + 60e3]),
           np.array([g.high - g.margin - 50.0])]
    gate_fields(f1, ETA_F1, sj.StokesConfig(lmax=10), pos, MOVED_FIELD_TOL_Q, "G15q-1")


def test_g15q_3(full_tables):
    _, _, f3 = full_tables
    pos = _corners(f3) + [f3.start_radii + np.array([-120e3, 60e3, -40e3])]
    gate_fields(f3, ETA_F3, sj.StokesConfig(lmax=10), pos, MOVED_FIELD_TOL_Q, "G15q-3")


def test_g16q_1(full_tables):
    _, f1, _ = full_tables
    gate_derivative(f1, ETA_F1, sj.StokesConfig(lmax=10), [(5300e3,), (5500e3,)], RADIUS_DERIV_TOL_Q, "G16q-1")


def test_g16q_3(full_tables):
    _, _, f3 = full_tables
    s = f3.start_radii
    gate_derivative(f3, ETA_F3, sj.StokesConfig(lmax=10), [(s[0] - 150e3, s[1] + 100e3, s[2] + 30e3)],
                    RADIUS_DERIV_TOL_Q, "G16q-3")


# --------------------------------------------------------------------------
# The full tier of G15 and G16 (slow set, CONVENTIONS 7): each slow test below repeats
# the assertion of a named quick test (G15q, G16q, P1, I7) on the full table or at a
# higher degree. The other tests of this section are quick: they hold assertions that no
# quick test repeats (the S-pair tests, the G15 gate on the node-and-surface layout, the
# clamp bounds against the oracle, the flags and the rough anomaly of the Taylor rows).
# --------------------------------------------------------------------------

G15_1_OFFSETS = (-150e3, -50e3, -10e3, 1e3, 50e3, 150e3)


def _g15_1_positions(f1):
    g = f1.ranges[1]
    return ([np.array([g.low + g.margin + 50.0])] + [np.array([g.start + o]) for o in G15_1_OFFSETS]
            + [np.array([g.high - g.margin - 50.0])])


def _g16_1_targets(f1):
    """The G16-1 base points (through ``mid_interval``). The third one is the input
    interval directly below the one that holds r0: the nearest admissible interval to
    r0 from below (plan 6.4)."""
    g = f1.ranges[1]
    r_in = np.asarray(f1.input_profile.radius)
    j = int(np.searchsorted(r_in, g.start)) - 1          # r_in[j] < r0 <= r_in[j + 1]
    return [(g.low + 3 * g.margin,), (g.start - 150e3,), (r_in[j] - 1.0,), (g.start + 60e3,), (g.start + 200e3,),
            (g.high - 3 * g.margin,)]


@pytest.mark.slow
@pytest.mark.parametrize("flags", ["default", "tala_nsns"])
def test_g15_1(full_tables, flags):
    _, f1, _ = full_tables
    form = sj.Formulation() if flags == "default" else TALA_NSNS
    gate_fields(f1, ETA_F1, sj.StokesConfig(lmax=30, formulation=form), _g15_1_positions(f1), MOVED_FIELD_TOL,
                f"G15-1 {flags}", zero_velocity=flags != "default")


@pytest.mark.slow
@pytest.mark.parametrize("flags", ["default", "tala_nsns"])
def test_g15_3(full_tables, flags):
    _, _, f3 = full_tables
    form = sj.Formulation() if flags == "default" else TALA_NSNS
    pos = _corners(f3)
    if flags == "default":
        pos += [f3.start_radii + np.array(o) for o in ((-250e3, 250e3, 60e3), (300e3, -300e3, -80e3),
                                                  (-200e3, -80e3, 100e3), (120e3, 150e3, -100e3))]
    gate_fields(f3, ETA_F3, sj.StokesConfig(lmax=30, formulation=form), pos, MOVED_FIELD_TOL_MULTI,
                f"G15-3 {flags}", zero_velocity=flags != "default")


@pytest.mark.slow
@pytest.mark.parametrize("flags", ["default", "tala_nsns"])
def test_g16_1(full_tables, flags):
    _, f1, _ = full_tables
    form = sj.Formulation() if flags == "default" else TALA_NSNS
    gate_derivative(f1, ETA_F1, sj.StokesConfig(lmax=30, formulation=form), _g16_1_targets(f1), RADIUS_DERIV_TOL,
                    f"G16-1 {flags}", half=True)


def node_surface_layout(p):
    """G15-node: on the full table, boundary 1 on the input node nearest 5290 km, movable
    in (r_in[70], 5700) km, and boundary 4 on the input viscosity-only pair at 6270 km,
    movable in (6150, 6370) km, a range from a density-jump pair to the surface
    (04-review-physics-r1.md minors 1 and 3)."""
    r_in = np.asarray(p.radius)
    rn = float(r_in[int(np.argmin(np.abs(r_in - 5290e3)))])
    return sj.Layers(p, np.array([3480e3, rn, 5700e3, 5970e3, 6270e3, 6370e3]),
                     movable={1: (float(r_in[70]), 5700e3), 4: (6150e3, 6370e3)})


def test_g15_node_surface(full_tables):
    """G15 on the node-and-surface layout: the four moved fields against tables built
    at the new radii, smooth and rough anomaly, lmax 30, at three interior positions and
    the two opposite corners of the clamp box."""
    p, _, _ = full_tables
    L = node_surface_layout(p)
    g1, g4 = L.ranges[1], L.ranges[4]
    assert g1.members[0] == g1.members[1] - 1 and np.any(np.asarray(p.radius) == g1.start)
    s = L.start_radii
    pos = [s + np.array([40e3, 30e3]), s + np.array([-120e3, -80e3]), s + np.array([250e3, 80e3]),
           np.array([g1.low + g1.margin + 50.0, g4.high - g4.margin - 50.0]),
           np.array([g1.high - g1.margin - 50.0, g4.low + g4.margin + 50.0])]
    gate_fields(L, ETA_F1, sj.StokesConfig(lmax=30), pos, MOVED_FIELD_TOL_SHALLOW, "G15-node")


@pytest.mark.slow
def test_g16_3(full_tables):
    _, _, f3 = full_tables
    s = f3.start_radii
    targets = [(s[0] - 200e3, s[1] + 60e3, s[2] - 60e3), (s[0] + 150e3, s[1] - 150e3, s[2] + 80e3),
               (s[0] + 40e3, s[1] + 200e3, s[2] + 22.578125e3)]
    gate_derivative(f3, ETA_F3, sj.StokesConfig(lmax=30), targets, RADIUS_DERIV_TOL, "G16-3", half=True)


def test_s_t27_rough():
    _t27(_program_p3("rough"))


@pytest.mark.slow
def test_s_t26_full_table(full_tables):
    """T19 to T21 on the full t_prem() table at lmax 50, range (5060.47, 5700) km; the base
    point r1 is the middle of the widest gap between consecutive crossings in
    r0 + [5, 25] km; FD_KINK_MIN measured again here."""
    _, f1, _ = full_tables
    cfg = sj.StokesConfig(lmax=50)
    d = densities.smooth(np.asarray(f1.input_profile.radius), 50, 4)

    def to_profile(c):
        return f1(c["radii"], c["log_eta"])
    c0 = dict(radii=np.asarray(f1.boundaries), log_eta=np.log(ETA_F1))
    prog = Program(f1, cfg, to_profile, c0, d)
    r1 = widest_gap_base(f1, 5e3, 25e3)[0]
    c1 = dict(c0, radii=c0["radii"] + np.array([0, r1 - f1.start_radii[0], 0, 0, 0, 0]))
    rng = np.random.default_rng(19)
    assert_taylor_with_controls(prog, c1, direction(c1, log_eta=0.1 * rng.standard_normal(5)))
    gap = nearest_crossing(f1, [r1])[0]
    amp = min(5000.0, gap / (CROSSING_CLEARANCE * 1e-2) / 1.01)
    assert_clearance(f1, [r1], 1e-2 * amp)
    print(f"S-T26: r1 - r0 = {(r1 - f1.start_radii[0]) / 1e3:.3f} km, nearest crossing {gap:.1f} m, amplitude {amp:.0f} m")
    assert_taylor_with_controls(prog, c1, direction(c1, radii=np.array([0, amp, 0, 0, 0, 0])))
    # T21 at r0, sign +1: the plan's 1000 m leaves the corrected remainder of the full
    # table below ten allowances from the third rung on ("unresolved": r1 = 3.3e-12
    # against the allowance 7.8e-13 at rung 3); 5000 m, as T20, resolves it. The steps
    # stay on the upper side of r0: the nearest crossing above r0 is farther than the
    # clearance (NOTES/TOLERANCES.md).
    c_up = crossing_radii(f1, 1)
    c_up = c_up[c_up > f1.start_radii[0]]
    assert np.min(c_up - f1.start_radii[0]) > CROSSING_CLEARANCE * 1e-2 * 5000.0
    dx = direction(c0, radii=np.array([0, 5000.0, 0, 0, 0, 0]))
    assert_taylor(prog.taylor, c0, dx, signs=(1.0,))
    c_dn = crossing_radii(f1, 1)
    c_dn = c_dn[c_dn < f1.start_radii[0]]
    assert np.min(f1.start_radii[0] - c_dn) > CROSSING_CLEARANCE * 1e-2 * 5000.0
    for mode in ("reverse", "forward"):
        verdict, reason = assess(taylor_remainders(prog.taylor, c0, dx, sign=-1.0, mode=mode))
        print(f"S-T26 T21 sign -1 {mode}: {verdict} ({reason})")
        assert verdict == "fail"
    unit = direction(c0, radii=np.array([0, 1.0, 0, 0, 0, 0]))
    h = FD_STEP
    others = crossing_radii(f1, 1)
    others = others[others != f1.start_radii[0]]
    assert np.min(np.abs(others - f1.start_radii[0])) > CROSSING_CLEARANCE * 2 * h
    v = prog.taylor.values(c0, unit, [0.0, h, 2 * h, -h, -2 * h])
    Dp = (-3 * v[0] + 4 * v[1] - v[2]) / (2 * h)
    Dm = (3 * v[0] - 4 * v[3] + v[4]) / (2 * h)
    for mode in ("reverse", "forward"):
        D = prog.taylor.derivative(c0, unit, mode)
        print(f"S-T26 G4 {mode}: |D - D+|/|D| = {abs(D - Dp) / abs(D):.3e}, |D - D-|/|D| = {abs(D - Dm) / abs(D):.3e}")
        assert abs(D - Dp) <= FD_ONE_SIDED_TOL * abs(D)
        assert abs(D - Dm) >= FD_KINK_MIN_FULL * abs(D)


def test_s_g9_clamp_bounds_oracle():
    """Kernels of the reduced layering with the boundary at each clamp bound against the
    mpmath oracle of the same table (decimal strings from repr(float))."""
    import reference_mpmath as ro
    L = layers_f1()
    cfg = sj.StokesConfig(lmax=10)
    for bound in L.radius_bounds[0]:
        q = L(L.full_boundaries(np.array([bound])), np.log(ETA_F1))
        K = sj.stokes_kernels(q, cfg)
        table = ro.NodeTable(radius=tuple(repr(float(v)) for v in np.asarray(q.radius)),
                             rho0=tuple(repr(float(v)) for v in np.asarray(q.rho0)),
                             eta=tuple(repr(float(v)) for v in np.asarray(q.eta)),
                             core_density=repr(float(q.core_density)), core_mass=repr(float(q.core_mass)))
        ref = ro.kernels(table, ro.Physics(), ro.Discretisation(), (1, 2, 10))
        for k, l in enumerate((1, 2, 10)):
            for o in SI_OUTPUTS:
                e = assert_rows_close(np.asarray(getattr(K, o))[K.row(l)], ref[o][k], ORACLE_TOL, f"{bound} l={l} {o}")
                print(f"S-G9 bound {bound / 1e3:.3f} km l={l} {o}: e_l {e:.2e}")


@pytest.mark.slow
def test_s_g5_gradient_finite_lmax100(full_tables):
    _, f1, _ = full_tables
    cfg = sj.StokesConfig(lmax=100)
    d = densities.smooth(np.asarray(f1.input_profile.radius), 100, 4)

    def to_profile(c):
        return f1(c["radii"], c["log_eta"], rho0=c["rho0"]).replace(core_density=c["core_density"])
    c0 = dict(radii=np.asarray(f1.boundaries) + np.array([0, 31e3, 0, 0, 0, 0]), log_eta=np.log(ETA_F1),
              rho0=np.asarray(f1.profile.rho0), core_density=np.asarray(float(f1.profile.core_density)))
    prog = Program(f1, cfg, to_profile, c0, d)
    g = prog.dot.vjp(c0, prog.taylor.weights)
    for k, v in g.items():
        assert np.all(np.isfinite(np.asarray(v))), k
        assert np.any(np.asarray(v) != 0), k


@pytest.mark.slow
def test_s_vmap_16(full_tables):
    _, f1, _ = full_tables
    cfg = sj.StokesConfig(lmax=50)
    d = jnp.asarray(densities.smooth(np.asarray(f1.input_profile.radius), 50, 4))
    cv = sj.ControlVector({"radii_movable": np.array([5401e3]), "log_eta": np.log(ETA_F1)})

    def fn(c, d):
        q = f1(f1.full_boundaries(c["radii_movable"]), c["log_eta"])
        Fd = f1.fields(sj.stokes_kernels(q, cfg), d)
        return jnp.sum(Fd.geoid ** 2) * 1e6 + jnp.sum(Fd.surface_topography)
    rng = np.random.default_rng(8)
    X = np.tile(np.asarray(cv.ravel({"radii_movable": np.array([5401e3]), "log_eta": np.log(ETA_F1)})), (16, 1))
    X[:, 5] += 20e3 * (2 * rng.random(16) - 1)
    X[:, :5] += 0.2 * rng.standard_normal((16, 5))
    single = cv.value_and_grad(fn)
    vv, gv = jax.jit(jax.vmap(single, in_axes=(0, None)))(jnp.asarray(X), d)
    for b in range(16):
        v, g = single(jnp.asarray(X[b]), d)
        assert abs(float(vv[b]) - float(v)) <= VMAP_RTOL * abs(float(v))
        assert np.max(np.abs(np.asarray(gv[b]) - np.asarray(g))) <= VMAP_RTOL * np.max(np.abs(np.asarray(g)))


def test_s_flags():
    """T19 and T20 under TALA with no slip at both boundaries (the velocity is an exact
    zero and left out of F)."""
    L = layers_f1()
    cfg = sj.StokesConfig(lmax=20, formulation=TALA_NSNS)
    d = densities.smooth(np.asarray(L.input_profile.radius), 20, 4)

    def to_profile(c):
        return L(c["radii"], c["log_eta"])
    c1 = dict(radii=np.asarray(L.boundaries) + np.array([0, 31e3, 0, 0, 0, 0]), log_eta=np.log(ETA_F1))
    prog = Program(L, cfg, to_profile, c1, d)
    assert not prog.on[SI_OUTPUTS.index("surface_velocity")]
    rng = np.random.default_rng(19)
    assert_taylor_with_controls(prog, c1, direction(c1, log_eta=0.1 * rng.standard_normal(5)))
    assert_taylor_with_controls(prog, c1, direction(c1, radii=np.array([0, 5000.0, 0, 0, 0, 0])))


def _pair_change(form):
    """Largest e_l of the kernels at the input nodes of the full t_prem() table with a
    fixed pair inserted in the middle of the interval (r[40], r[41]) with equal viscosity
    on both sides, against the kernels of the input profile."""
    p = _p_full()
    r = np.asarray(p.radius)
    s = 0.5 * (r[40] + r[41])
    L = sj.Layers(p, np.array([3480e3, s, 5700e3, 5970e3, 6270e3, 6370e3]))
    eta_in = np.asarray(p.eta)
    lay = np.array([eta_in[0], eta_in[0], eta_in[int(np.flatnonzero(r == 5700e3)[-1])],
                    eta_in[int(np.flatnonzero(r == 5970e3)[-1])], eta_in[-1]])
    cfg = sj.StokesConfig(lmax=30, formulation=form)
    base = p.replace(eta=jnp.exp(jnp.log(jnp.asarray(eta_in))))
    Kin = L.input_kernels(sj.stokes_kernels(L(L.boundaries, np.log(lay)), cfg))
    K = sj.stokes_kernels(base, cfg)
    return max(float(np.max(row_errors(np.asarray(getattr(Kin, o)), np.asarray(getattr(K, o))))) for o in SI_OUTPUTS)


def test_s_pair():
    e = _pair_change(sj.Formulation())
    print(f"S-pair default flags: {e:.2e}")
    assert e <= INSERTED_PAIR_BOUND


def test_s_pair_exact():
    e = _pair_change(sj.Formulation(gravity=sj.GravityModel.NO_SELF_GRAVITY, background=sj.Uniform(9.81)))
    print(f"S-pair NO_SELF_GRAVITY, uniform g: {e:.2e}")
    assert e <= PAIR_EXACT_TOL
