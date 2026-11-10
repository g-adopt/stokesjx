"""Gate G9: the oracle against the analytical delta-forcing solutions of Kramer,
Davies and Wilson (2021, GMD 14, 1899-1919), package ``assess``.

The ``assess`` problem is an isoviscous, incompressible shell with uniform gravity
and no self-gravitation: ``-div(tau) + grad(p) = -g rho' r_hat`` with
``rho' = delta(r - r') Y_lm``. That is the oracle's NO_SELF_GRAVITY, uniform-g
problem with a unit sheet (sigma = +1 kg/m^2 per unit harmonic coefficient) at
``r'``, the viscosity ``nu`` and the gravity ``g``. One global convention, no fitted
factor (NOTES/docs/equations.md section 7): the assess forcing is a positive sheet,
and both codes write the velocity as ``u_r Y r_hat + u_t dY/dtheta theta_hat``.
Non-dimensional values of the G-ADOPT benchmark: Rp = 2.22, Rm = 1.22, r' = 1.72,
nu = 1, g = 1.
"""

import mpmath as mp
import pytest

assess = pytest.importorskip("assess")

import reference_mpmath as ro  # noqa: E402

# assess evaluates closed-form solutions whose coefficients come from a 4x4 float64
# solve; the oracle is exact to 30 digits. Measured agreement: 1.2e-13 at worst
# (cause: the float64 coefficients of assess).
ASSESS_RTOL = 1e-10
RP, RM, RPRIME = "2.22", "1.22", "1.72"
THETA = 0.6          # colatitude at which assess is evaluated; any value with Y != 0
N_NODES = 51         # spacing 0.02: r' = 1.72 is node 25 and r = 2.0 is node 39


def _table():
    rad = tuple(str(round(1.22 + 0.02 * k, 2)) for k in range(N_NODES))
    assert rad[25] == RPRIME and rad[39] == "2.0" and rad[-1] == RP
    return ro.NodeTable(radius=rad, rho0=("1",) * N_NODES, eta=("1",) * N_NODES, core_density="0")


@pytest.mark.parametrize("bc", ["free_slip", "no_slip"])
@pytest.mark.parametrize("l", [2, 4, 8])
def test_delta_solution(bc, l):
    from assess import (SphericalStokesSolutionDeltaFreeSlip, SphericalStokesSolutionDeltaZeroSlip,
                        Y, dYdtheta)
    cls = SphericalStokesSolutionDeltaFreeSlip if bc == "free_slip" else SphericalStokesSolutionDeltaZeroSlip
    ph = ro.Physics(self_gravity="none", background="uniform", uniform_g="1", surface=bc, cmb=bc,
                    density_scale="1", viscosity_scale="1")
    st = ro.solve_state(_table(), ph, ro.Discretisation(), l, 25, nodes=[39])
    upper = cls(l, 0, +1, Rp=2.22, Rm=1.22, rp=1.72, nu=1.0, g=1.0)
    lower = cls(l, 0, -1, Rp=2.22, Rm=1.22, rp=1.72, nu=1.0, g=1.0)
    y, dy = Y(l, 0, THETA, 0), dYdtheta(l, 0, THETA, 0)
    with mp.workdps(40):
        yR, yc, y2 = st["yR"], st["yc"], st["y_nodes"][39]
        R, rc = mp.mpf(RP), mp.mpf(RM)
        ours = {
            "sigma_rr(R)": float(yR[2] / R),          # y3 = r sigma_rr / eta_s, eta_s = 1
            "sigma_rr(r_c)": float(yc[2] / rc),
            "u_r(2.0)": float(y2[0]),
            "u_t(2.0)": float(y2[1]),
        }
        if bc == "free_slip":
            ours["u_t(R)"] = float(yR[1])
        else:
            ours["sigma_rt(R)"] = float(yR[3] / R)
    ref = {
        "sigma_rr(R)": upper.radial_stress(2.22, THETA, 0) / y,
        "sigma_rr(r_c)": lower.radial_stress(1.22, THETA, 0) / y,
        "u_r(2.0)": upper.u_r(2.0, THETA, 0) / y,
        "u_t(2.0)": upper.u_theta(2.0, THETA, 0) / dy,
    }
    if bc == "free_slip":
        ref["u_t(R)"] = upper.u_theta(2.22, THETA, 0) / dy
    else:
        ref["sigma_rt(R)"] = upper.tau_rtheta(2.22, THETA, 0) / dy
    for k in ref:
        assert abs(ours[k] - ref[k]) <= ASSESS_RTOL * abs(ref[k]), (k, ours[k], ref[k])
