"""Gate G7: the matrix A of CODE-A follows from the vector equations (sympy, exact).

The test builds the instantaneous Stokes equations in spherical coordinates for an
axisymmetric spheroidal field of one degree ``l`` (``Y = P_l(cos theta)``):

    v = U(r) Y r_hat + V(r) dY/dtheta theta_hat
    div(rho0 v) = 0 (TALA) or div v = 0 (incompressible), rho0 = c r^kappa
    sigma = -p I + 2 eta e(v) - (2/3) eta (div v) I   (the last term on or off)
    div sigma + rho0 grad(phi) = 0,   laplacian(phi) = 0,   phi = Phi(r) Y

and derives ``dy/d ln r = A y`` for the state of NOTES/docs/equations.md section 3,

    y = (U, V, r S_rr / eta_s, r S / eta_s, rho_s r Phi / eta_s, rho_s r^2 Phi' / eta_s),

with ``sigma_rr = S_rr Y`` and ``sigma_rt = S dY/dtheta``. The result must equal
``aconstruct`` (driver.f:369-398) entry by entry, with and without the -2/3 term
(F2), and ``K = dA/dkappa`` must have the three entries of F8 with ``K^2 = -K``.
The angular dependence is removed by evaluating each projected equation at two
exact angles and requiring the two results to agree.
"""

import pytest

sp = pytest.importorskip("sympy")

r, eta, eta_s, rho_s, c0, kappa = sp.symbols("r eta eta_s rho_s c0 kappa", positive=True)
theta = sp.Symbol("theta")
U, V, Srr, S, Phi = (sp.Function(n)(r) for n in ("U", "V", "Srr", "S", "Phi"))


def _derive(l: int, compressible: bool, deviatoric_term: bool) -> sp.Matrix:
    """Matrix A(l) from the vector equations, as a sympy Matrix in eta/eta_s,
    rho0/rho_s and kappa."""
    L = l * (l + 1)
    Y = sp.legendre(l, sp.cos(theta))
    dY = sp.diff(Y, theta)
    rho0 = c0 * r ** kappa if compressible else c0
    kap = kappa if compressible else 0
    # Continuity: U' = (-(2 + kappa) U + L V) / r (div v = -kappa U / r).
    Up = (-(2 + kap) * U + L * V) / r
    divv = -kap * U / r
    flag = 1 if deviatoric_term else 0
    # Pressure from the definition of S_rr: S_rr = -P + 2 eta U' - (2/3) eta div v.
    P = -Srr + 2 * eta * Up - sp.Rational(2, 3) * eta * divv * flag
    vr = U * Y
    vt = V * dY
    e_tt = sp.diff(vt, theta) / r + vr / r
    e_pp = vr / r + vt * sp.cot(theta) / r
    iso = -P * Y - sp.Rational(2, 3) * eta * divv * Y * flag
    s_rr = Srr * Y
    s_rt = S * dY
    s_tt = iso + 2 * eta * e_tt
    s_pp = iso + 2 * eta * e_pp
    # Divergence of the symmetric stress tensor, axisymmetric, sigma_r_phi =
    # sigma_theta_phi = 0, plus the force of the perturbed potential on rho0.
    div_r = (sp.diff(s_rr, r) + sp.diff(s_rt, theta) / r
             + (2 * s_rr - s_tt - s_pp + s_rt * sp.cot(theta)) / r)
    div_t = (sp.diff(s_rt, r) + sp.diff(s_tt, theta) / r
             + ((s_tt - s_pp) * sp.cot(theta) + 3 * s_rt) / r)
    eq_r = div_r + rho0 * sp.diff(Phi, r) * Y
    eq_t = div_t + rho0 * Phi * dY / r

    def project(expr, shape):
        """The radial coefficient of ``expr = f(r) shape(theta)``, checked at two angles."""
        a1, a2 = sp.acos(sp.Rational(1, 3)), sp.acos(sp.Rational(-2, 7))
        f1 = sp.simplify(expr.subs(theta, a1) / shape.subs(theta, a1))
        f2 = sp.simplify(expr.subs(theta, a2) / shape.subs(theta, a2))
        assert sp.simplify(f1 - f2) == 0, "the equation is not proportional to its harmonic"
        return f1

    fr = project(sp.expand(eq_r), Y)
    ft = project(sp.expand(eq_t), dY)
    Srr_p = sp.solve(fr, sp.diff(Srr, r))[0]
    S_p = sp.solve(ft, sp.diff(S, r))[0]
    Vp = S / eta + V / r - U / r                       # definition of S
    Phi_pp = -2 * sp.diff(Phi, r) / r + L * Phi / r ** 2   # Laplace
    # d/d ln r of the state.
    dy = [r * Up,
          r * Vp,
          r * (Srr + r * Srr_p) / eta_s,
          r * (S + r * S_p) / eta_s,
          rho_s * r * (Phi + r * sp.diff(Phi, r)) / eta_s,
          rho_s * r * (2 * r * sp.diff(Phi, r) + r ** 2 * Phi_pp) / eta_s]
    ys = sp.symbols("y1:7")
    subs = {sp.diff(Phi, r): eta_s * ys[5] / (rho_s * r ** 2), U: ys[0], V: ys[1],
            Srr: eta_s * ys[2] / r, S: eta_s * ys[3] / r, Phi: eta_s * ys[4] / (rho_s * r)}
    A = sp.zeros(6, 6)
    for i, e in enumerate(dy):
        e = sp.expand(e.subs(subs))
        for j in range(6):
            A[i, j] = sp.simplify(sp.diff(e, ys[j]))
        assert sp.simplify(e - sum(A[i, j] * ys[j] for j in range(6))) == 0
    return A, rho0


def _aconstruct(l: int, eta_rel, rho_rel, kap) -> sp.Matrix:
    """aconstruct of CODE-A (driver.f:369-398) with kappa = kap."""
    L = l * (l + 1)
    a = sp.zeros(6, 6)
    a[0, 0] = -2 - kap
    a[0, 1] = L
    a[1, 0] = -1
    a[1, 1] = 1
    a[1, 3] = 1 / eta_rel
    a[2, 0] = 4 * (3 + kap) * eta_rel
    a[2, 1] = -6 * L * eta_rel
    a[2, 2] = 1
    a[2, 3] = L
    a[2, 5] = -rho_rel
    a[3, 0] = -2 * (3 + kap) * eta_rel
    a[3, 1] = 2 * (2 * L - 1) * eta_rel
    a[3, 2] = -1
    a[3, 3] = -2
    a[3, 4] = -rho_rel
    a[4, 4] = 1
    a[4, 5] = 1
    a[5, 4] = L
    return a


@pytest.mark.parametrize("l", [2, 3, 5])
@pytest.mark.parametrize("compressible", [False, True])
@pytest.mark.parametrize("deviatoric_term", [True, False])
def test_matrix_from_vector_equations(l, compressible, deviatoric_term):
    A, rho0 = _derive(l, compressible, deviatoric_term)
    ref = _aconstruct(l, eta / eta_s, rho0 / rho_s, kappa if compressible else 0)
    diff = sp.simplify(A - ref)
    assert diff == sp.zeros(6, 6), diff


@pytest.mark.parametrize("l", [2, 3, 5])
def test_k_matrix(l):
    """K = dA/dkappa has K(1,1) = -1, K(3,1) = 4 eta, K(4,1) = -2 eta and K^2 = -K."""
    A = _aconstruct(l, eta / eta_s, c0 / rho_s, kappa)
    K = sp.diff(A, kappa)
    expect = sp.zeros(6, 6)
    expect[0, 0] = -1
    expect[2, 0] = 4 * eta / eta_s
    expect[3, 0] = -2 * eta / eta_s
    assert sp.simplify(K - expect) == sp.zeros(6, 6)
    assert sp.simplify(K * K + K) == sp.zeros(6, 6)


def test_negative_control_wrong_kappa_entry_differs():
    """The comparison is not vacuous: an aconstruct without kappa in a(3,1) differs
    from the derived TALA matrix."""
    A, rho0 = _derive(2, True, True)
    ref = _aconstruct(2, eta / eta_s, rho0 / rho_s, kappa)
    ref[2, 0] = 12 * eta / eta_s
    assert sp.simplify(A - ref) != sp.zeros(6, 6)
