"""The matrix, the interval steps, the jump, the boundary bases and rows, the static
frame rule and the refusals (plan 7.3, gate G9).

References: the oracle (``reference_mpmath``: ``matrix_a_rel``, ``interval_steps``,
``_Model.jump``, ``frame_options``) and the boundary conditions of
NOTES/docs/equations.md (eq. 8.1, 8.2, 8.5, 10.5).
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import mpmath as mp
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import cases  # noqa: E402
import reference_mpmath as ro  # noqa: E402
from adapters import profile_from_table  # noqa: E402

import stokesjx as sj  # noqa: E402
from stokesjx import boundary, propagate, system  # noqa: E402
from stokesjx.formulation import degree_one_options, resolve_degree_one  # noqa: E402

# Matrix entries against the oracle's matrix (60 digits), relative per entry: the
# float64 entries are integers times l and l^2 up to 6 L = 2.4e5 at l = 200, and the
# density entry is one division; the cause of a difference is the rounding of one
# float64 operation.
MATRIX_TOL = 1e-15
# Interval steps against the oracle's steps (rounded to float64), max-normalised per
# interval and degree in the balanced coordinates of the interval
# (``M -> S D^-1 M D S^-1``, as tests/test_primitives_values.py). Cause: the rounding
# of the float64 Pade pair and of its squarings (1e-15 to 4e-15) and of the width
# ``h`` (log1p form: about 1e-16 relative); the physics review r3 of chunk 1 measured
# 1.2e-15 at l = 100 on 129 nodes with log1p widths.
STEP_ORACLE_TOL = 1e-13
# The jump [y3] = r g / eta_s holds the integrated gravity of eq. 6.1, relative per
# node (same cause as GRAVITY_TOL of tests/test_units_profile.py).
GRAVITY_TOL = 1e-14
# The jump [y6] = -4 pi G r^2 rho_s / eta_s, relative per node. It holds no integrated
# quantity; measured 3.2e-16 to 3.3e-16 on seven tables, against 3.3e-16 to 7.2e-16 for
# [y3] (NOTES/chunks/c2-forward-core/plan-review-r4-evidence/r4_jump_tol.out). Cause:
# rounding of four products.
JUMP_TOL = 1e-15
# Boundary rows applied to the basis vectors: the residual divided by the largest
# absolute term of the row (review r5 of the plan, issue 7). A row is a sum of terms
# that cancel, for example (l - c_c rho_c / rho_s) - l + c_c rho_c / rho_s, so its
# absolute rounding grows with l (1.4e-14 at l = 200), while the residual relative to
# the largest term is a few units of the last place.
BOUNDARY_TOL = 1e-15


@pytest.mark.parametrize("l", [1, 2, 3, 50, 200])
def test_a_matrix_against_oracle(l):
    e, d = 0.37, 1.13
    A = np.asarray(sj.system.a_matrix(float(l), e, d, self_gravity=True))
    with mp.workdps(60):
        ref = np.array(ro.matrix_a_rel(l, mp.mpf(e), mp.mpf(d), True).tolist(), dtype=float)
    nz = ref != 0
    assert np.array_equal(A == 0, ~nz)
    assert np.max(np.abs(A[nz] - ref[nz]) / np.abs(ref[nz])) <= MATRIX_TOL
    # The static parts of the solver give the same balanced matrix: P + d Q.
    P, Q, _ = system._static_exponent_parts((l,), True)
    s = system.degree_scaling((l,))[0]
    bal = s[:, None] * np.asarray(sj.system.a_matrix(float(l), 1.0, d, self_gravity=True)) / s[None, :]
    assert np.max(np.abs(P[:, :, 0] + d * Q[:, :, 0] - bal)) <= MATRIX_TOL * np.max(np.abs(bal))


STEP_TABLES = {
    "hager_q": lambda: cases.t_hager(reduced=True),
    "hager": lambda: cases.t_hager(),
    "prem": lambda: cases.t_prem(),
}


@pytest.mark.parametrize("table_name", list(STEP_TABLES))
def test_interval_steps_against_oracle(table_name):
    """The steps of both sweeps against the oracle's steps and their inverses at l in
    {1, 2, 100}, with the solver's viscosity scale set to the oracle's ``eta_s = 1e19``
    (private hook), so that the ``y`` of both codes agree; a pair step is the identity
    exactly."""
    table = STEP_TABLES[table_name]()
    p = profile_from_table(table)
    cfg = sj.StokesConfig(lmax=100)
    plan = propagate.static_plan(p.structure, cfg)
    eta_s = 1e19
    Zu, Zd = (np.asarray(a) for a in propagate.interval_steps(p, jnp.asarray(eta_s), plan, cfg))
    eta = np.asarray(p.eta)
    e = 0.5 * (eta[1:] + eta[:-1]) / eta_s
    D = np.stack([np.ones_like(e), np.ones_like(e), e, e, e, e], axis=-1)       # (n, 6)
    for l in (1, 2, 100):
        k = plan.degrees.index(l)
        S = system.degree_scaling((l,))[0]
        model, dps = ro._model(table, ro.Physics(), ro.Discretisation(), l)
        for i in range(len(e)):
            # The oracle's step and its inverse, the inverse formed inside the oracle's
            # precision context (at least 60 digits; physics review r3 of chunk 1,
            # minor 1: at 15 digits mp.inverse is wrong by 1e-11 at l = 100).
            with mp.workdps(dps):
                ref = np.array(model.steps[i + 1].tolist(), dtype=float)
                ref_inv = np.array(mp.inverse(model.steps[i + 1]).tolist(), dtype=float)
            for Z, R in ((Zu, ref), (Zd, ref_inv)):
                # Balanced coordinates of the interval: S D^-1 M D S^-1 for the
                # oracle's y-step, D^-1 Z D for the solver's z-step (S and D commute).
                ref_b = (S / D[i])[:, None] * R * (D[i] / S)[None, :]
                mine_b = Z[i, k] / D[i][:, None] * D[i][None, :]
                err = np.max(np.abs(mine_b - ref_b)) / np.max(np.abs(ref_b))
                assert err <= STEP_ORACLE_TOL, (l, i, err)
    pairs = np.asarray(p.structure.pairs) - 1
    assert np.all(Zu[pairs] == np.eye(6)) and np.all(Zd[pairs] == np.eye(6))


@pytest.mark.parametrize("table_name", ["hager_q", "prem_q", "uniform_q"])
def test_jump_vectors_against_oracle(table_name):
    table = {"hager_q": lambda: cases.t_hager(reduced=True), "prem_q": lambda: cases.t_prem(reduced=True),
             "uniform_q": lambda: cases.t_uniform(reduced=True)}[table_name]()
    p = profile_from_table(table)
    units = sj.Units()
    g = p.hydrostatic_gravity(units)
    J = np.asarray(system.jump_vectors(p.radius, g, 1e19, units))
    m, d = ro._model(table, ro.Physics(), ro.Discretisation(), 2, 60)
    with mp.workdps(d):
        ref = np.array([[float(v) for v in m.jump(j)] for j in range(len(table.radius))])
    assert np.all(J[:, [0, 1, 3, 4]] == 0)
    assert np.max(np.abs(J[:, 2] - ref[:, 2]) / np.abs(ref[:, 2])) <= GRAVITY_TOL
    assert np.max(np.abs(J[:, 5] - ref[:, 5]) / np.abs(ref[:, 5])) <= JUMP_TOL


def _row_residual(row, vecs):
    """``row . v`` over the largest absolute term ``|row_k v_k|`` per vector."""
    terms = row[..., :] * vecs
    return np.abs(np.sum(terms, axis=-1)) / np.max(np.abs(terms), axis=-1)


DEGREES = tuple(range(1, 201))


def test_cmb_basis_conditions():
    """``y1 = 0``, ``y4 = 0`` and ``y6 = l y5 - c_c (y3 + rho_c / rho_s y5)`` (eq. 8.1,
    8.5) for every column; rank 3."""
    c_c, rc_rel = 3.37, 9900.0 / 4500.0
    Y = np.asarray(boundary.cmb_basis(DEGREES, c_c, rc_rel, formulation=sj.Formulation()))   # (L, 6, 3)
    l = np.asarray(DEGREES, float)
    assert np.all(Y[:, 0, :] == 0) and np.all(Y[:, 3, :] == 0)
    rows = np.zeros((len(l), 6))
    rows[:, 5] = 1.0
    rows[:, 4] = -(l - c_c * rc_rel)
    rows[:, 2] = c_c
    for k in range(3):
        v = Y[:, :, k]
        nonzero = np.max(np.abs(rows * v), axis=-1) > 0
        assert np.all(_row_residual(rows[nonzero], v[nonzero]) <= BOUNDARY_TOL)
    assert np.all(np.linalg.matrix_rank(Y) == 3)


@pytest.mark.parametrize("frame", [sj.DegreeOneFrame.RAW, sj.DegreeOneFrame.CENTRE_OF_MASS])
def test_surface_basis_conditions(frame):
    """The three surface rows annihilate the basis: ``y1``, ``y4`` and
    ``y6 + a_l y5 + c_s y3`` with the raw ``a_l = l + 1`` (eq. 8.2) and, at ``l = 1`` in
    the CM frame, ``a_1 = -1`` (eq. 10.5); rank 3."""
    c_s = 2.45
    a = boundary.surface_potential_coefficient(DEGREES, frame)
    l = np.asarray(DEGREES, float)
    expected = l + 1
    if frame is sj.DegreeOneFrame.CENTRE_OF_MASS:
        expected[0] = -1.0
    assert np.array_equal(a, expected)
    Y = np.asarray(boundary.surface_basis(DEGREES, c_s, a, formulation=sj.Formulation()))
    assert np.all(Y[:, 0, :] == 0) and np.all(Y[:, 3, :] == 0)
    rows = np.zeros((len(l), 6))
    rows[:, 5], rows[:, 4], rows[:, 2] = 1.0, a, c_s
    for k in range(3):
        v = Y[:, :, k]
        nonzero = np.max(np.abs(rows * v), axis=-1) > 0
        assert np.all(_row_residual(rows[nonzero], v[nonzero]) <= BOUNDARY_TOL)
    assert np.all(np.linalg.matrix_rank(Y) == 3)


_FLAG_COMBOS = list(itertools.product(("full", "none"), ("hydrostatic", "uniform"), (False, True),
                                      ("free_slip", "no_slip"), ("free_slip", "no_slip"), (True, False)))


@pytest.mark.parametrize("combo", _FLAG_COMBOS, ids=lambda c: "-".join(str(x) for x in c))
def test_degree_one_options_against_oracle(combo):
    """The static frame table for all 32 flag combinations and both core models
    against ``reference_mpmath.frame_options`` (reads flags only)."""
    sg, bg, comp, surf, cmb, consistent = combo
    form = sj.Formulation(gravity=sj.GravityModel.FULL if sg == "full" else sj.GravityModel.NO_SELF_GRAVITY,
                          background=sj.Hydrostatic() if bg == "hydrostatic" else sj.Uniform(9.81),
                          compressible=comp, surface=sj.Boundary(surf), cmb=sj.Boundary(cmb))
    table = cases.t_hager(reduced=True, core="consistent" if consistent else "split")
    ref = ro.frame_options(table, ro.Physics(self_gravity=sg, background=bg, compressible=comp,
                                              surface=surf, cmb=cmb))
    opts = degree_one_options(form, consistent)
    assert (opts.default.value if opts.default else "refused") == ref["default"]
    assert tuple(f.value for f in opts.offered) == ref["offered"]
    assert tuple(f.value for f in opts.explicit_only) == ref["explicit_only"]
    for req in ("raw", "centre_of_mass", "legacy_geoid_zero"):
        try:
            expected = ro.degree_one_rule(table, ro.Physics(self_gravity=sg, background=bg, compressible=comp,
                                                            surface=surf, cmb=cmb, degree_one=req))
        except ValueError:
            with pytest.raises(ValueError):
                resolve_degree_one(form, consistent, sj.DegreeOneFrame(req))
        else:
            assert resolve_degree_one(form, consistent, sj.DegreeOneFrame(req)).value == expected


def test_config_refuses_chunk3_values():
    """``StokesConfig`` refuses the flag values of chunk 3; ``Formulation`` accepts them."""
    for form in (sj.Formulation(gravity=sj.GravityModel.NO_SELF_GRAVITY), sj.Formulation(compressible=True),
                 sj.Formulation(surface=sj.Boundary.NO_SLIP), sj.Formulation(cmb=sj.Boundary.NO_SLIP)):
        with pytest.raises(NotImplementedError, match="c3-flags"):
            sj.StokesConfig(lmax=4, formulation=form)
    with pytest.raises(ValueError):
        sj.StokesConfig(lmax=4, degree_one=sj.DegreeOneFrame.LEGACY_GEOID_ZERO)
    with pytest.raises(ValueError):
        sj.StokesConfig(lmax=0)
    with pytest.raises(ValueError):
        sj.Numerics(node_solver="qr")


def test_frame_refusals():
    """A frame that the static rule does not offer raises ``ValueError`` before any
    solve (review r5 of the plan, issue 2). Raw on a consistent core under FULL with
    hydrostatic gravity and free slip matters most: the system is singular there
    (eq. 10(a), ratio 4.0e-63), and a float64 solve would return finite wrong numbers."""
    cc = profile_from_table(cases.t_hager(reduced=True, core="consistent"))
    split = profile_from_table(cases.t_hager(reduced=True))
    with pytest.raises(ValueError, match="not offered"):
        propagate.static_plan(cc.structure, sj.StokesConfig(lmax=2, degree_one=sj.DegreeOneFrame.RAW))
    uniform = sj.Formulation(background=sj.Uniform(9.81))
    with pytest.raises(ValueError, match="not offered"):
        propagate.static_plan(split.structure, sj.StokesConfig(lmax=2, formulation=uniform,
                                                               degree_one=sj.DegreeOneFrame.CENTRE_OF_MASS))
    with pytest.raises(ValueError, match="raw"):
        propagate.static_plan(cc.structure, sj.StokesConfig(lmax=2),
                              propagate.Discretisation(degree_one_geoid_zero=True))
    # Where raw is offered, CODE-A's frame resolves.
    plan = propagate.static_plan(split.structure, sj.StokesConfig(lmax=2, formulation=uniform),
                                 propagate.Discretisation(degree_one_geoid_zero=True))
    assert plan.frame is sj.DegreeOneFrame.LEGACY_GEOID_ZERO and plan.geoid_mask[0] == 0.0


def test_eager_value_checks_and_trace():
    """D5: a concrete profile with ``rho0`` above ``Numerics.rho0_max`` or a viscosity
    that is not positive raises ``ValueError``; the same call under ``jax.jit`` traces
    without the check (a tracer has no value)."""
    p = profile_from_table(cases.t_hager(reduced=True))
    cfg = sj.StokesConfig(lmax=2)
    with pytest.raises(ValueError, match="rho0_max"):
        sj.stokes_kernels(p.replace(rho0=p.rho0.at[3].set(14000.0)), cfg)
    with pytest.raises(ValueError, match="positive"):
        sj.stokes_kernels(p.replace(eta=p.eta.at[2].set(0.0)), cfg)
    with pytest.raises(ValueError, match="positive"):
        sj.stokes_kernels(p.replace(eta=p.eta.at[2].set(-1e21)), cfg)
    # Under jit the same values trace and run (the result is meaningless, not checked).
    out = jax.jit(lambda q: sj.stokes_kernels(q, cfg).geoid)(p.replace(rho0=p.rho0.at[3].set(14000.0)))
    assert out.shape == (2, p.structure.n_nodes)
