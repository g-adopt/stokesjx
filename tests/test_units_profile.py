"""Units and the radial profile (plan 7.1, gate G9).

References: the oracle's legacy constants and its gravity integral at 60 digits
(``reference_mpmath``), the definitions of the pytree rules (plan 3.2).
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import cases  # noqa: E402
import reference_mpmath as ro  # noqa: E402
from adapters import profile_from_table  # noqa: E402

import stokesjx as sj  # noqa: E402
from stokesjx import profile as prof  # noqa: E402

# Gravity at the nodes against the oracle's gravity at 60 digits, relative per node.
# Cause: the float64 shell volumes r_k^3 - r_(k-1)^3 and their cumulative sum; the
# review prototype of the chunk 2 plan measured 3.6e-16 to 5.7e-16
# (plan-review-r1-evidence/rev_taylor.out).
GRAVITY_TOL = 1e-14

GRAVITY_TABLES = {
    "hager_split": lambda: cases.t_hager(),
    "hager_consistent": lambda: cases.t_hager(core="consistent"),
    "prem": lambda: cases.t_prem(),
    "layers": lambda: cases.t_layers(),
    "uniform": lambda: cases.t_uniform(),
}


def test_legacy_constants_equal_oracle():
    """CODE-A's float32 literals, bit for bit (F13)."""
    u = sj.Units(legacy_constants=True)
    assert mp.mpf(u.G) == ro.LEGACY_G
    assert mp.mpf(u.pi) == ro.LEGACY_PI
    assert mp.mpf(u.earth_mass) == ro.LEGACY_EMASS
    d = sj.Units()
    assert d.G == float(ro.Physics().G) and d.pi == float(mp.pi)


def test_units_refuse_nonpositive_scale():
    with pytest.raises(ValueError):
        sj.Units(density_scale=0.0)


@pytest.mark.parametrize("name", ["hager", "prem", "layers"])
def test_structure_pairs(name):
    """Pairs are detected from equal neighbouring radii, as the oracle does."""
    t = {"hager": cases.t_hager, "prem": cases.t_prem, "layers": cases.t_layers}[name](reduced=True)
    p = profile_from_table(t)
    expected = tuple(i for i in range(1, len(t.radius)) if t.radius[i] == t.radius[i - 1])
    assert p.structure.pairs == expected
    assert len(expected) == {"hager": 3, "prem": 4, "layers": 3}[name]
    # A pair has a static zero width and every other interval a positive one.
    h = p.structure.width_bounds
    assert np.all(h[np.asarray(expected) - 1] == 0) and np.sum(h > 0) == len(h) - len(expected)


def test_structure_refusals():
    r = np.array([3480e3, 4000e3, 5000e3, 6370e3])
    one = np.ones(4)
    with pytest.raises(ValueError):
        sj.RadialProfile(r[::-1], one, one, core_density=9900.0)          # decreasing
    with pytest.raises(ValueError):
        sj.RadialProfile(r, one[:3], one, core_density=9900.0)            # shape mismatch
    with pytest.raises(ValueError):
        sj.RadialProfile(r[:1], one[:1], one[:1], core_density=9900.0)    # one node
    with pytest.raises(ValueError):
        sj.RadialProfile(np.array([1.0, 2.0, 2.0, 2.0]), one, one, core_density=1.0)  # three equal radii

    def traced(x):
        return sj.RadialProfile(x, one, one, core_density=9900.0).radius
    with pytest.raises(ValueError, match="NodeStructure"):
        jax.jit(traced)(r)
    p = sj.RadialProfile(r, one, one, core_density=9900.0)
    with pytest.raises(ValueError):
        sj.RadialProfile(r, one, one, core_density=9900.0, core_mass=1e24, structure=p.structure)
    with pytest.raises(ValueError):
        p.replace(core_mass=1e24)


def test_flatten_unflatten_without_validation():
    """The unflatten builds the object without validation: placeholders pass."""
    p = profile_from_table(cases.t_hager(reduced=True))
    leaves, treedef = jax.tree_util.tree_flatten(p)
    assert len(leaves) == 6          # core_mass of a split core is a leaf; surface_density is None
    q = jax.tree_util.tree_unflatten(treedef, [object()] * len(leaves))
    assert q.structure == p.structure
    r = jax.tree_util.tree_unflatten(treedef, leaves)
    for a, b in zip(jax.tree_util.tree_leaves(r), leaves):
        assert a is b
    # A consistent core and the default surface density are empty subtrees.
    pc = profile_from_table(cases.t_hager(reduced=True, core="consistent"))
    assert len(jax.tree_util.tree_leaves(pc)) == 5
    # None placeholders, as some transformations pass them.
    s = jax.tree_util.tree_unflatten(treedef, [None] * len(leaves))
    assert s.radius is None


def test_replace_keeps_structure_and_rho_top_follows_rho0():
    """``replace`` keeps the structure (no new jit key), and the default topography
    density follows ``rho0`` under ``grad`` (plan 3.2 rule 4, risk R7)."""
    p = profile_from_table(cases.t_hager(reduced=True))
    q = p.replace(rho0=p.rho0 * 1.01)
    assert jax.tree_util.tree_structure(q) == jax.tree_util.tree_structure(p)
    assert float(q.rho_top) == float(p.rho0[-1] * 1.01)
    g = jax.grad(lambda rho: p.replace(rho0=rho).rho_top)(p.rho0)
    assert float(g[-1]) == 1.0 and float(jnp.sum(jnp.abs(g[:-1]))) == 0.0
    ps = profile_from_table(cases.t_hager(reduced=True), surface_density=2600.0)
    assert float(ps.rho_top) == 2600.0


@pytest.mark.parametrize("name", list(GRAVITY_TABLES))
def test_integrate_gravity_against_oracle(name):
    t = GRAVITY_TABLES[name]()
    p = profile_from_table(t)
    g = np.asarray(p.hydrostatic_gravity(sj.Units()))
    with mp.workdps(60):
        ref = np.array([float(x) for x in ro.gravity_at_nodes(t, ro.Physics())])
    assert np.max(np.abs(g - ref) / ref) <= GRAVITY_TOL


@pytest.mark.parametrize("name", list(GRAVITY_TABLES))
def test_mantle_mass_against_oracle(name):
    t = GRAVITY_TABLES[name]()
    p = profile_from_table(t)
    m = float(prof.mantle_mass(p.radius, p.rho0, sj.Units()))
    with mp.workdps(60):
        model, _ = ro._model(t, ro.Physics(), ro.Discretisation(), 2, 60)
        ref = float(model.mantle_mass)
    assert abs(m - ref) / ref <= GRAVITY_TOL


def test_dtypes_numpy_and_jax_inputs_agree():
    """NumPy and JAX inputs give the same strong float64 leaves (no weak types)."""
    t = cases.t_hager(reduced=True)
    p_np = profile_from_table(t)
    p_jx = sj.RadialProfile(jnp.asarray(p_np.radius), jnp.asarray(p_np.rho0), jnp.asarray(p_np.eta),
                            core_density=jnp.asarray(9900.0), core_mass=1.9355e24, outer_density=0)
    for a, b in zip(jax.tree_util.tree_leaves(p_np), jax.tree_util.tree_leaves(p_jx)):
        assert a.dtype == jnp.float64 and b.dtype == jnp.float64
        assert not a.weak_type and not b.weak_type
