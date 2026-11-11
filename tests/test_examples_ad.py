"""Taylor tests and the jvp/vjp dot product of the HMC potential of ``examples/
hmc_layers.py`` (NOTES/chunks/c5-integration/01-plan.md section 6.2).

The functional is the potential ``U(z)`` of the example (the quadratic geoid misfit plus
the prior) on the 25-node table at ``lmax = 6``, the c4b rule for the Taylor cases of a
cut range (CONVENTIONS 8). The problems are built through the example's own command line,
``hmc_layers.build(hmc_layers.parse_args([...]))``, so that an option that ``build``
ignores fails an assertion here.

- T36: the cut layering of the example (range across the 670 km density jump), base point
  20 m below the jump. Only the first three rungs of the sign +1 cross the jump, so T36
  tests the derivative of ``U`` at a base point near the jump; the crossing itself is the
  subject of c4b K1 and K4.
- T37: the affine layering of run C (range ending at the jump, start 5600 km), base point
  31 km above the start, clear of every crossing of ``Layers.crossings`` by at least 20
  times the largest step.
- DP5: the dot product of the map ``z -> N(z) / s`` (the geoid field over its largest
  value at the T36 base point).

Policy (CONVENTIONS 8): ladder ``1e-2 2^-k``, ``k = 0..7``, window of four rungs, both
signs, both modes, corrected rate at least 1.9, uncorrected band (0.8, 1.2), the 1.01
control fails, "unresolved" is not a pass.
"""

from __future__ import annotations

import functools
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from taylor import TaylorProgram, assert_jvp_vjp, assert_taylor, assess, remeasure_taylor  # noqa: E402

import stokesjx as sj  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
KM = 1e3
J = 5700.0 * KM            # the 670 km density-jump pair of the table

# The offset of the base point from the true log viscosities, ln(Pa s): the misfit has a
# residual of its own at the base point (CONVENTIONS 8, chunk 4: a target at the base point
# hides a 1 % bias of the derivative).
Y_OFFSET = np.array([0.1, -0.1, 0.05, 0.0])
# The radius of the data of T36 must be at least this far from the base point (c4b
# TARGET_MIN_DISTANCE).
TARGET_MIN_DISTANCE = 50e3
# The first rung (1e-2 of the direction) moves the radius by about this much, m.
RADIUS_FIRST_RUNG = 100.0
# The log-viscosity direction: a standard normal vector times this factor.
ETA_DIRECTION = 0.01
# T37: the base point is farther from every crossing than this many largest steps (c4
# CROSSING_CLEARANCE).
CROSSING_CLEARANCE = 20
# DP5: the dot product on the normalised map (CONVENTIONS 8).
DOT_RTOL = 1e-12
# T37: the clamp range against [low + m, high - m] (same arithmetic, measured 0).
BOUNDS_RTOL = 1e-15


@functools.lru_cache(maxsize=None)
def hmc_module():
    """``examples/hmc_layers.py`` under its own module name (no ``sys.path`` change)."""
    name = "stokesjx_examples_hmc_layers"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / "hmc_layers.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


T36_ARGS = ("--reduced", "--lmax", "6", "--true-radius", "5620")
T37_ARGS = ("--reduced", "--lmax", "6", "--motion", "affine", "--range", "5466.875", "5700",
            "--start-radius", "5600", "--true-radius", "5620")


@functools.lru_cache(maxsize=None)
def problem(args: tuple):
    """``(module, Problem, TaylorProgram of U)`` of a command line; cached per worker."""
    hmc = hmc_module()
    P = hmc.build(hmc.parse_args(list(args)))

    def U(z):
        return hmc.potential(z, P.density, P.data, P.weights, P)
    return hmc, P, TaylorProgram(U)


def base_point(hmc, P, radius):
    """``z`` at the true log viscosities plus ``Y_OFFSET`` and the boundary at ``radius``."""
    u = hmc.logit((radius - P.lo) / (P.hi - P.lo))
    return np.concatenate([hmc.Y_TRUE + Y_OFFSET, [u]])


def direction(P, z, seed):
    """A standard normal direction in ``y`` times ``ETA_DIRECTION``, and a ``u`` component
    scaled so that the first rung (1e-2) moves the radius by ``RADIUS_FIRST_RUNG``:
    ``dr/du = (hi - lo) s (1 - s)``."""
    rng = np.random.default_rng(seed)
    s = 1.0 / (1.0 + math.exp(-z[4]))
    du = RADIUS_FIRST_RUNG / (1e-2 * (P.hi - P.lo) * s * (1.0 - s))
    return np.concatenate([ETA_DIRECTION * rng.standard_normal(4), [du]])


def assert_taylor_controls(tp, z, dz):
    """Taylor test (both signs, both modes) and the four 1.01 controls must fail."""
    res = assert_taylor(tp, z, dz)
    verdicts = [assess(remeasure_taylor(r, 1.01 * r.derivative))[0] for r in res]
    assert all(v != "pass" for v in verdicts), verdicts
    rates = [float(np.nanmin(r.rates1[:3])) for r in res]
    print(f"Taylor pass, smallest corrected rate of the window {min(rates):.3f}; 1.01 controls {verdicts}")
    return res


def test_t36_cut_near_jump():
    """T36: the cut layering of the example, base point 20 m below the 670 km jump."""
    hmc, P, tp = problem(T36_ARGS)
    assert P.layers.motion == "cut"
    z = base_point(hmc, P, J - 20.0)
    r = float(hmc.radius_of(z[4], P))
    assert abs(r - (J - 20.0)) < 1e-6
    assert abs(float(P.args.true_radius) * KM - r) >= TARGET_MIN_DISTANCE
    assert_taylor_controls(tp, z, direction(P, z, 36))


def test_t37_affine():
    """T37: the affine layering of run C, base point 5631 km, clear of every crossing."""
    hmc, P, tp = problem(T37_ARGS)
    L = P.layers
    assert L.motion == "affine"
    g = L.ranges[1]
    assert g.low == 5466.875e3 and g.high == 5700e3
    m = sj.controls.CLAMP_FRACTION * (g.high - g.low)
    np.testing.assert_allclose(L.radius_bounds[0], [g.low + m, g.high - m], rtol=BOUNDS_RTOL, atol=0)
    assert L.start_radii[0] == 5600e3
    z = base_point(hmc, P, 5631e3)
    dz = direction(P, z, 37)
    radii, _ = L.crossings(1)
    largest_step = RADIUS_FIRST_RUNG          # the first rung moves the radius by about this
    clearance = float(np.min(np.abs(radii - float(hmc.radius_of(z[4], P)))))
    print(f"T37: clearance from the nearest crossing {clearance:.0f} m")
    assert clearance >= CROSSING_CLEARANCE * largest_step * 1.01
    assert_taylor_controls(tp, z, dz)


def test_dp5_dot_product():
    """DP5: ``<dy, J dz> = <J^T dy, dz>`` for ``z -> N(z) / s`` at the T36 base point."""
    hmc, P, _ = problem(T36_ARGS)
    z = base_point(hmc, P, J - 20.0)
    s = float(jnp.max(jnp.abs(jax.jit(lambda q: hmc.geoid(q, P.density, P))(jnp.asarray(z)))))

    def F(q):
        return hmc.geoid(q, P.density, P) / s
    rng = np.random.default_rng(5)
    dz = rng.standard_normal(5)
    dy = rng.standard_normal((2, P.config.lmax + 1, P.config.lmax + 1))
    rel = assert_jvp_vjp(F, z, dx=dz, dy=dy, rtol=DOT_RTOL)
    print(f"DP5: {rel:.2e}")
