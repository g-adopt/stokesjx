"""The values of the candidate interval steps against the steps of the oracle.

tests/test_primitives_ad.py checks only derivatives of the candidates in
benchmarks/candidates.py: a wrong entry of the matrix A or K passes every test there.
Chunk 3 builds the solver from these functions, so this file connects the matrix of
the candidates to the matrix of the mpmath oracle (tests/reference_mpmath.py,
``interval_steps``), which the gates G1 to G19 check against CODE-A and against
independent physics.

The candidate step of an interval is ``T expm(B) T^-1`` (``step_balanced``) and the
step of a jump pair under TALA is ``I + K (1 - exp(-t))`` (``pair_step``). Both are
compared in the balanced coordinates ``T^-1 E T`` of the interval
(``T = D_eta S_l^-1``), where the entries carry no factor of the viscosity or of
``l`` and a max-normalised difference is a fair measure for every entry.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import candidates as c  # noqa: E402

import cases  # noqa: E402
import reference_mpmath as ro  # noqa: E402

# Agreement of the float64 candidate step with the oracle step rounded to float64,
# max-normalised in balanced coordinates. The reviewer measured at most 2.5e-14 on
# the full hager4, PREM TALA and wide5 tables up to l = 100 (cause: rounding of the
# Pade approximant and of its squarings, which grows with the squaring count, two
# at l = 100). A sign error or a wrong entry of A or K gives an error of order 1.
STEP_ORACLE_TOL = 1e-13

# Tables with jump pairs: the reduced hager4 table (incompressible, split core)
# and the reduced PREM table under TALA (nonzero kappa h and pair steps).
TABLES = {
    "hager4_q": (lambda: cases.t_hager(reduced=True), dict()),
    "prem_tala_q": (lambda: cases.t_prem(reduced=True), dict(compressible=True)),
}

# Step forms under test: the selected form (elimination without pivoting, order
# and squaring count from pade_plan) and the Pade (6, 6) LAPACK form.
FORMS = ("plan_ge", "pade6_lapack")


def _interval_data(table, physics):
    """Interval widths in ``ln r``, the interval-mean viscosity and density relative
    to the scales of ``physics`` (Discretisation "interval_mean", the stokesjx
    model), ``kappa h = ln(rho0_i / rho0_(i-1))`` under TALA (0 otherwise; a pair
    has ``h = 0``), and the mask of the jump pairs (two nodes at one radius)."""
    r = np.array([float(v) for v in table.radius])
    rho0 = np.array([float(v) for v in table.rho0])
    eta = np.array([float(v) for v in table.eta])
    pair = r[1:] == r[:-1]
    h = np.where(pair, 0.0, np.log(r[1:] / np.where(pair, 1.0, r[:-1])))
    eta_i = 0.5 * (eta[1:] + eta[:-1]) / float(physics.viscosity_scale)
    rho_i = 0.5 * (rho0[1:] + rho0[:-1]) / float(physics.density_scale)
    kh = np.log(rho0[1:] / rho0[:-1]) if physics.compressible else np.zeros(len(h))
    return h, eta_i, rho_i, kh, pair


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("l", [1, 2, 100])
@pytest.mark.parametrize("table_name", list(TABLES))
def test_step_equals_oracle_interval_step(table_name, l, form):
    """Every interval step and every pair step of the candidates equals the
    oracle's step ``P_i`` to ``STEP_ORACLE_TOL`` in balanced coordinates."""
    make_table, phys_kw = TABLES[table_name]
    table = make_table()
    physics = ro.Physics(**phys_kw)
    h, eta_i, rho_i, kh, pair = _interval_data(table, physics)
    assert pair.any()
    # Interval steps use kappa h of their own interval; a pair carries its density
    # jump in the pair step, so its interval exponent has kappa h = 0 and h = 0.
    kh_int = np.where(pair, 0.0, kh)
    bound = c.balanced_norm_bound_intervals(l, h, kh_int, rho_i)
    if form == "plan_ge":
        plan = c.pade_plan(bound, "elimination")
        method, sq = f"pade{plan['order']}_bl_ge", plan["squarings"]
    else:
        method, sq = "pade6_bl", c.pade_squarings(bound, 6, "lapack")
    E = np.asarray(c.step_balanced(float(l), eta_i, rho_i, h, kh_int, method=method, squarings=sq))
    P = np.asarray(c.pair_step(eta_i, kh))
    steps = ro.interval_steps(table, physics, ro.Discretisation(), l)
    t = np.asarray(c.viscosity_scaling(eta_i) / c.degree_scaling(float(l)))
    errors = []
    for i in range(len(h)):
        ref = np.array(steps[i + 1].tolist(), dtype=float)
        mine = P[i] if pair[i] else E[i]
        # Balanced coordinates of interval i: T^-1 E T.
        a = mine / t[i][:, None] * t[i][None, :]
        b = ref / t[i][:, None] * t[i][None, :]
        errors.append(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    assert max(errors) <= STEP_ORACLE_TOL, (int(np.argmax(errors)), max(errors))
