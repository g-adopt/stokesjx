"""Derivative jumps of the control map at crossings, and the HMC energy estimate
(NOTES/chunks/c4-controls-derivatives/01-plan.md section 8; gate G10).

Run from the repository root:

    python benchmarks/measure_kink.py [--tables T1 T2 T1s] [--json PATH] [--overwrite]

The map from a movable boundary radius ``r_b`` to the fields is continuous and piecewise
smooth. Its derivative jumps at "crossings": a moved node over an input radius (kind 1:
the knot table of ``rho0`` and the bracket of the density map change interval), the pair
over an input radius (kind 2: an input node changes side of the viscosity jump), and the
construction radius ``r0`` (kind 3). This script uses the package ``Layers`` and
``Layers.input_kernels`` and measures, on every crossing of the admissible range
``[low + m, high - m]``:

1. the crossings, grouped: neighbouring crossings closer than ``GROUP_GAP = 1 m`` form one
   group (crossings at rational stretch factors ``s = (r_b - low) / (r0 - low)`` with small
   denominators fall on one position up to the rounding of the tabulated radii). The
   script asserts that the widest group is below ``GROUP_GAP / 4`` and the smallest gap
   between groups above 4 mm;
2. per group, the one-sided derivatives of the input geoid kernel (forward mode), by
   linear extrapolation from ``D`` and ``2 D`` beyond the group's first and last crossing,
   ``D = min(1 m, gap to the neighbouring group / 4)`` per side; the total jump
   ``[dN/dr_b]_g`` of the geoid field and ``rho_g = |[dN]_g| / |dN|``;
3. a Gaussian log-likelihood ``Phi = (lam / 2) sum_(l >= 2, C/S, m <= l) (N(r_b) - N_obs)^2``
   with ``N_obs = N(r_true) + e`` (noise of variance ``1 / lam``, fixed seed), ``lam`` such
   that the Gauss-Newton posterior standard deviation of ``r_b`` at ``r_true`` is
   ``sigma_r``. For ``r_true`` on a grid over the whole range (``GRID_STEP = 2.5 km``) and
   ``sigma_r`` in {10, 30} km, over the groups within ``r_true +- 1.5 sigma_r``:
   ``J_g = lam <[dN]_g, N(r_g) - N_obs>``, ``E_kink = sqrt(sum_g (J_g sigma_r)^2)`` (one
   draw) and its root mean square over the noise,

       E_rms^2 = sigma_r^2 sum_g lam^2 (<[dN]_g, N(r_g) - N(r_true)>^2 + |[dN]_g|^2 / lam).

   ``E_rms`` uses the bound ``|dH| <= J |dx|`` of one leapfrog step of size
   ``dx = sigma_r`` across a jump ``J`` (plan 8.2), summed over the groups as a random
   walk: an estimate, not a bound. The root mean square of ``dH`` itself is about
   ``0.41 J sigma_r`` per group (review r3 of the plan, section 4.4, a simulation of 2e6
   leapfrog steps), so ``E_rms`` is about 2.5 times the typical energy error.

Decision (plan 8.4): ``E_TOL = 0.1`` nat. If ``E_rms <= E_TOL`` at every grid point, for both
``sigma_r``, on T1 and T2, for the decision set (the smooth anomaly of ``tests/densities.py``
and the LEBEDEV tomography density), linear interpolation stays. Otherwise the numbers
and the positions above the limit go to the owner; no remedy is implemented here. The
rough anomaly (an independent value per input node) is the stress case, recorded only.

Tables:

- T1: ``cases.t_prem()`` (137 input nodes), the pair at 5370 km movable in
  (5060.469, 5700) km, fixed pairs at 5700, 5970, 6270 km, layer viscosities 3e22, 1e22,
  1e21, 3e20, 1e22 Pa s, ``lmax = 50``;
- T2: the 257-node refined LEBEDEV table (the S2 table of ``bench_controls.py``), the
  hager4 boundaries 5700, 5970, 6270 km inserted as pairs, 5700 km movable in
  (5466.875, 5970) km, hager4 layer viscosities, ``lmax = 100``;
- T1s: T1 with the construction radius at 5395 km (52 % of its range), remedy 4 of plan
  8.5 measured with the package (review r3 of the plan, minor 2);
- T2s: T2 with the upper range end at the node below 5900 km (5895.859 km), so that the
  construction radius 5700 km sits at 54 % of its range: remedy 4 on T2 (run on request,
  ``--tables T2s --json ...``).

Also recorded: the crossing count per km in windows of 25, 50, 100 and 200 km around
``r0``, and the relative jumps per group of a kernel functional ``F_rand`` (a random
weighted sum of the four kernels on the construction table, no density map: the kinks of
the ``rho0`` interpolation alone). The JSON goes to ``NOTES/benchmarks/c4-kink.json``; the
script refuses to overwrite it without ``--overwrite``.

The viscosity cut (chunk c4b, NOTES/chunks/c4b-boundaries-across-jumps/01-plan.md section
11; ``measure_cut``). Under ``Layers(..., motion="cut")`` no node moves, and the only
crossings are the input radii inside the clamp range (``Layers.crossings``: kind 2 an input
node, kind 4 a density-jump pair; a pair is one radius, so no grouping is needed). At every
crossing ``c``: the one-sided derivatives and values by linear extrapolation from ``c -+ D``
and ``c -+ 2 D`` (``D = 1 m``); ``rho = |[dN]| / |dN|``; the value jump as a displacement of
the boundary, ``L = |[N]| / |dN|`` in m; ``E_rms`` as above (the derivative jumps only). At
every density-jump pair strictly inside the range, the energy error of one crossing of an
HMC trajectory from the jump of the VALUE (plan 6.3), for every ``r_true`` of the grid
within ``1.5 sigma_r`` of the pair,

    E_value(r_true) = sqrt(lam^2 ([N] . (N_mid - N(r_true)))^2 + lam |[N]|^2),

with ``[N] = N(J+) - N(J-)``, ``N_mid`` their mean and ``lam`` of the likelihood above; its
lower limit is ``sqrt(lam) |[N]| = L / sigma_r``. Tables (``--tables T1c T2c T3 T4 T3t``):

- T1c, T2c: T1 and T2 with ``motion="cut"`` (no density jump inside the range);
- T3: ``t_prem()``, boundaries 3480, 5700, 5970, 6270, 6370 km, the 5700 km boundary movable
  in (5466.875, 5970) km (it holds the 670 km density jump; ``r0 = J``), viscosities 1e22,
  1e21, 3e20, 1e22 Pa s, ``lmax = 50``;
- T4: ``t_prem()``, boundaries 3480, 5800, 6270, 6370 km, movable in (5466.875, 6150) km
  (it holds the 5700 and 5970 km jumps), 1e22, 3e20, 1e22 Pa s, ``lmax = 50``;
- T3t: T3 under TALA at the viscosity contrasts 1.1, 3, 10 and 100 across the boundary
  (lower-mantle viscosity ``contrast x 1e21`` Pa s; the contrast 10 is T3), names
  ``T3t-1.1`` and so on (``T3t`` runs all four).

Rule (gate G6 of the c4b plan): on T1c, T2c, T3 and T4, ``rho_max <= 1e-8``, ``E_rms <= 1e-3``
nat and ``E_value <= 1e-6`` nat for the three anomalies and both ``sigma_r``. T3t is recorded.
The JSON goes to ``NOTES/benchmarks/c4b-kink.json`` (``--json``).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO / "tests"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import stokesjx as sj  # noqa: E402

GROUP_GAP = 1.0          # m: crossings closer than this are one group
GRID_STEP = 2.5e3        # m: the grid of r_true over the admissible range
SIG_R = (10e3, 30e3)     # m: posterior standard deviations of r_b
E_TOL = 0.1              # nat: the energy estimate per trajectory segment (plan 8.4)
WINDOWS = (25e3, 50e3, 100e3, 200e3)
HAGER4 = np.array([1e20, 1e19, 0.03333333333333333e19, 1e20])     # CMB to surface, Pa s


def table(name: str):
    """``(Layers, log_eta, lmax)`` of a table of the measurement."""
    import cases
    from adapters import profile_from_table
    if name in ("T1", "T1s"):
        p = profile_from_table(cases.t_prem())
        r = np.asarray(p.radius)
        r0 = 5370e3 if name == "T1" else 5395e3
        L = sj.Layers(p, np.array([3480e3, r0, 5700e3, 5970e3, 6270e3, 6370e3]), movable={1: (float(r[70]), 5700e3)})
        return L, np.log(np.array([3e22, 1e22, 1e21, 3e20, 1e22])), 50
    if name in ("T2", "T2s"):
        from bench_forward import lebedev_refined
        p = lebedev_refined()
        r = np.asarray(p.radius)
        # T2s: the upper range end at the node below 5900 km, so that r0 = 5700 km sits at
        # 54 % of its range and the group s = 2 (2 r0 - low = 5933 km) lies outside it.
        high = 5970e3 if name == "T2" else float(r[r <= 5900e3][-1])
        L = sj.Layers(p, np.array([3480e3, 5700e3, 5970e3, 6270e3, 6370e3]), movable={1: (5466.875e3, high)})
        return L, np.log(HAGER4), 100
    raise ValueError(name)


def lebedev_on(r_in, lmax):
    """LEBEDEV_S4RTS.sph (129 radii, degrees up to 50), ``real_4pi`` ``(N, 2, L+1, L+1)``,
    linear in ``r`` to the input radii; degrees above 50 are zero (plan 8.4: the field above
    degree 50 is then zero for every ``r_b`` and adds noise to the likelihood and no jump)."""
    import cases
    from test_golden import _read_lebedev_coefficients
    path = REPO / "legacy" / "CODE-A" / "INPUT" / "DENSITY_INPUTS" / "LEBEDEV_S4RTS.sph"
    c = np.moveaxis(_read_lebedev_coefficients(path), -1, 1)
    lf = c.shape[-1] - 1
    k = min(lf, lmax)
    out = np.zeros((129, 2, lmax + 1, lmax + 1))
    out[:, :, : k + 1, : k + 1] = c[:, :, : k + 1, : k + 1]
    rl = np.array([float(x) for x in cases.LEBEDEV_RADIUS])
    flat = out.reshape(129, -1)
    return np.stack([np.interp(r_in, rl, flat[:, i]) for i in range(flat.shape[1])], axis=1).reshape(
        (np.size(r_in),) + out.shape[1:])


def measure(name: str) -> dict:
    """Every crossing, the groups, the jumps and the energy estimate of one table."""
    import densities
    L, log_eta, lmax = table(name)
    cfg = sj.StokesConfig(lmax=lmax)
    k_mov = int(L.movable_index[0])
    g = L.ranges[k_mov]
    LOW, START, HIGH, M = g.low, g.start, g.high, g.margin
    r_in = np.asarray(L.input_profile.radius)
    dens = {"smooth": densities.smooth(r_in, lmax, 11), "lebedev": lebedev_on(r_in, lmax),
            "rough": densities.rough(r_in, lmax, 12)}

    # ---- crossings and groups
    cross, kind = L.crossings(k_mov)
    splits = np.flatnonzero(np.diff(cross) > GROUP_GAP) + 1
    groups = np.split(np.arange(cross.size), splits)
    lo_g = np.array([cross[ix[0]] for ix in groups])
    hi_g = np.array([cross[ix[-1]] for ix in groups])
    cnt = np.array([ix.size for ix in groups])
    gaps = lo_g[1:] - hi_g[:-1]
    assert np.max(hi_g - lo_g) < GROUP_GAP / 4, np.max(hi_g - lo_g)
    assert np.min(gaps) > 4 * 1e-3, np.min(gaps)
    gap_left = np.concatenate([[np.inf], gaps])
    gap_right = np.concatenate([gaps, [np.inf]])
    D_left, D_right = np.minimum(1.0, gap_left / 4), np.minimum(1.0, gap_right / 4)
    centre = 0.5 * (lo_g + hi_g)
    s_of = lambda x: (x - LOW) / (START - LOW)
    span = HIGH - LOW - 2 * M
    ng = len(groups)
    print(f"{name}: {r_in.size} input nodes, {L.n_nodes} nodes, range ({LOW / 1e3:.3f}, {HIGH / 1e3:.3f}) km, "
          f"r0 {START / 1e3:.3f} km ({100 * (START - LOW) / (HIGH - LOW):.0f} % of the range), lmax {lmax}; "
          f"{cross.size} crossings ({int(np.sum(kind == 1))} kind 1, {int(np.sum(kind == 2))} kind 2, r0), "
          f"one per {span / cross.size / 1e3:.2f} km; {ng} groups, widest {np.max(hi_g - lo_g):.3f} m, "
          f"smallest gap {np.min(gaps):.3f} m", flush=True)
    windows = {f"{w / 1e3:.0f}": float(np.sum(np.abs(cross - START) <= w) / (2 * w / 1e3)) for w in WINDOWS}

    # ---- forward-mode derivatives of the input geoid kernel and of F_rand
    rng = np.random.default_rng(3)
    p0 = L(L.boundaries, log_eta)
    K0 = sj.stokes_kernels(p0, cfg)
    outs = ("geoid", "surface_topography", "cmb_topography", "surface_velocity")
    W = [jnp.asarray(rng.standard_normal(np.shape(getattr(K0, o))) / float(np.max(np.abs(getattr(K0, o)))))
         for o in outs]

    def kin(x):
        q = L(L.full_boundaries(x[None]), log_eta)
        K = sj.stokes_kernels(q, cfg)
        f_rand = sum(jnp.sum(w * getattr(K, o)) for w, o in zip(W, outs))
        return L.input_kernels(K).geoid, f_rand

    batch = 64 if lmax <= 50 else 24
    jv = jax.jit(jax.vmap(lambda x: jax.jvp(kin, (x,), (jnp.ones_like(x),))))

    def run(points):
        P, T, FP, FT = [], [], [], []
        for c0 in range(0, points.size, batch):
            chunk = points[c0:c0 + batch]
            pad = np.concatenate([chunk, np.full(batch - chunk.size, chunk[-1])])
            (a, fa), (b, fb) = jv(jnp.asarray(pad))
            P.append(np.asarray(a)[: chunk.size]); T.append(np.asarray(b)[: chunk.size])
            FP.append(np.asarray(fa)[: chunk.size]); FT.append(np.asarray(fb)[: chunk.size])
        return np.concatenate(P), np.concatenate(T), np.concatenate(FP), np.concatenate(FT)

    grid = np.arange(LOW + M + 1.0, HIGH - M, GRID_STEP)
    pts = np.concatenate([lo_g - 2 * D_left, lo_g - D_left, hi_g + D_right, hi_g + 2 * D_right, grid])
    t0 = time.perf_counter()
    Kp, dKp, Fp, dFp = run(pts)
    t_jvp = time.perf_counter() - t0
    print(f"   {pts.size} jvps in {t_jvp:.1f} s", flush=True)
    dK_minus = 2 * dKp[ng:2 * ng] - dKp[:ng]
    dK_plus = 2 * dKp[2 * ng:3 * ng] - dKp[3 * ng:4 * ng]
    K_at = 0.5 * (Kp[ng:2 * ng] + Kp[2 * ng:3 * ng])
    K_grid, dK_grid = Kp[4 * ng:], dKp[4 * ng:]
    dF_minus = 2 * dFp[ng:2 * ng] - dFp[:ng]
    dF_plus = 2 * dFp[2 * ng:3 * ng] - dFp[3 * ng:4 * ng]
    rho_F = np.abs(dF_plus - dF_minus) / np.abs(0.5 * (dF_plus + dF_minus))
    d0 = lmax + 1 - Kp.shape[1]
    w_in = L.input_weights
    mask = np.zeros((2, lmax + 1, lmax + 1), dtype=bool)
    mask[:, 2:, :] = np.tril(np.ones((lmax + 1, lmax + 1), dtype=bool))[2:]

    def fields_of(Karr, d_in):
        """``(n, L, N_in)`` input kernels to ``(n, 2, lmax+1, lmax+1)`` geoid fields, l >= 2."""
        out = np.zeros((Karr.shape[0], 2, lmax + 1, lmax + 1))
        out[:, :, d0:, :] = np.einsum("nlj,jclm->nclm", Karr * w_in, d_in[:, :, d0:, :])
        return out * mask

    rec = dict(table=name, lmax=lmax, n_input=int(r_in.size), n_nodes=int(L.n_nodes), low=LOW, high=HIGH, r0=START,
               margin=M, crossings=int(cross.size), kind1=int(np.sum(kind == 1)), kind2=int(np.sum(kind == 2)),
               groups=ng, groups_multi=int(np.sum(cnt > 1)), widest_group_m=float(np.max(hi_g - lo_g)),
               smallest_gap_m=float(np.min(gaps)), crossings_per_km_around_r0=windows, jvps=int(pts.size),
               jvp_seconds=t_jvp, grid_points=int(grid.size),
               largest_groups=[dict(count=int(cnt[i]), r_minus_r0_km=float((centre[i] - START) / 1e3),
                                    s=float(s_of(centre[i]))) for i in np.argsort(cnt, kind="stable")[::-1][:8]],
               F_rand=dict(rho_max=float(np.max(rho_F)), at_km=float((centre[int(np.argmax(rho_F))] - START) / 1e3),
                           rho_r0=float(rho_F[int(np.argmin(np.abs(centre - START)))]),
                           rho_median=float(np.median(rho_F))),
               anomalies={})
    print(f"   F_rand (rho0 kinks only): largest rho {rec['F_rand']['rho_max']:.2e} at "
          f"{rec['F_rand']['at_km']:+.1f} km, at r0 {rec['F_rand']['rho_r0']:.2e}, median {rec['F_rand']['rho_median']:.2e}")

    for aname, d_in in dens.items():
        Nm, Np_, Ng = fields_of(dK_minus, d_in), fields_of(dK_plus, d_in), fields_of(K_at, d_in)
        jump = Np_ - Nm
        rho_g = np.linalg.norm(jump.reshape(ng, -1), axis=1) / np.linalg.norm((0.5 * (Np_ + Nm)).reshape(ng, -1), axis=1)
        i_max = int(np.argmax(rho_g))
        i_r0 = int(np.argmin(np.abs(centre - START)))
        Nt_all, dNt_all = fields_of(K_grid, d_in), fields_of(dK_grid, d_in)
        nrng = np.random.default_rng(5)
        arec = dict(rho_max=float(rho_g[i_max]), rho_max_at_km=float((centre[i_max] - START) / 1e3),
                    rho_max_s=float(s_of(centre[i_max])), rho_max_count=int(cnt[i_max]),
                    rho_r0=float(rho_g[i_r0]), rho_median=float(np.median(rho_g)), sigma={})
        print(f"   {aname}: largest group jump rho {rho_g[i_max]:.2e} at {arec['rho_max_at_km']:+.1f} km "
              f"(s = {arec['rho_max_s']:.4f}, {cnt[i_max]} crossings); at r0 {rho_g[i_r0]:.2e}; "
              f"median {np.median(rho_g):.2e}")
        for sr in SIG_R:
            Es, Er, Jmax, where = [], [], [], []
            for it, rt in enumerate(grid):
                Nt, dNt = Nt_all[it], dNt_all[it]
                lam = 1.0 / (sr ** 2 * np.sum(dNt ** 2))
                e = nrng.standard_normal(Nt.shape) / np.sqrt(lam) * mask
                win = np.abs(centre - rt) <= 1.5 * sr
                res = Ng[win] - (Nt + e)[None]
                J = lam * np.abs(np.sum(jump[win] * res, axis=(1, 2, 3)))
                Es.append(np.sqrt(np.sum((J * sr) ** 2)))
                det = np.sum(jump[win] * (Ng[win] - Nt[None]), axis=(1, 2, 3))
                Jr2 = lam ** 2 * (det ** 2 + np.sum(jump[win] ** 2, axis=(1, 2, 3)) / lam)
                Er.append(np.sqrt(sr ** 2 * np.sum(Jr2)))
                k = int(np.argmax(Jr2)) if Jr2.size else 0
                Jmax.append(np.sqrt(Jr2[k]) * sr if Jr2.size else 0.0)
                where.append(centre[win][k] if Jr2.size else np.nan)
            Es, Er, Jmax = np.array(Es), np.array(Er), np.array(Jmax)
            ia, ib, ij = int(np.argmax(Es)), int(np.argmax(Er)), int(np.argmax(Jmax))
            over = grid[Er > E_TOL]
            srec = dict(E_rms_max=float(Er[ib]), E_rms_max_at_km=float((grid[ib] - START) / 1e3),
                        E_rms_median=float(np.median(Er)), E_kink_one_draw_max=float(Es[ia]),
                        E_kink_one_draw_at_km=float((grid[ia] - START) / 1e3),
                        J_sigma_single_group_max=float(Jmax[ij]),
                        J_sigma_single_group_at_km=float((where[ij] - START) / 1e3),
                        J_sigma_single_group_s=float(s_of(where[ij])),
                        grid_points_above_E_TOL=int(over.size),
                        r_true_above_E_TOL_km=[float((x - START) / 1e3) for x in over])
            arec["sigma"][f"{sr / 1e3:.0f}"] = srec
            print(f"      sigma_r {sr / 1e3:.0f} km: E_rms max {Er[ib]:.3f} nat at r_true - r0 "
                  f"{srec['E_rms_max_at_km']:+.1f} km, median {np.median(Er):.3f}; one-draw E_kink max {Es[ia]:.3f}; "
                  f"largest single-group J sigma_r {Jmax[ij]:.3f} at {srec['J_sigma_single_group_at_km']:+.1f} km "
                  f"(s = {srec['J_sigma_single_group_s']:.4f}); grid points above {E_TOL}: {over.size}"
                  + (f" (r_true - r0 from {srec['r_true_above_E_TOL_km'][0]:+.1f} to "
                     f"{srec['r_true_above_E_TOL_km'][-1]:+.1f} km)" if over.size else ""), flush=True)
        rec["anomalies"][aname] = arec
    return rec


CUT_TABLES = ("T1c", "T2c", "T3", "T4")
RHO_C1 = 1e-8            # c4b G6: the derivative jump of an incompressible cut table
E_RMS_CUT = 1e-3         # nat
E_VALUE_CUT = 1e-6       # nat
D_CUT = 1.0              # m: the extrapolation step at a crossing of a cut table


def cut_table(name: str):
    """``(Layers, log_eta, cfg)`` of a cut table (module docstring)."""
    import cases
    from adapters import profile_from_table
    if name == "T1c":
        p = profile_from_table(cases.t_prem())
        r = np.asarray(p.radius)
        L = sj.Layers(p, np.array([3480e3, 5370e3, 5700e3, 5970e3, 6270e3, 6370e3]),
                      movable={1: (float(r[70]), 5700e3)}, motion="cut")
        return L, np.log(np.array([3e22, 1e22, 1e21, 3e20, 1e22])), sj.StokesConfig(lmax=50)
    if name == "T2c":
        from bench_forward import lebedev_refined
        L = sj.Layers(lebedev_refined(), np.array([3480e3, 5700e3, 5970e3, 6270e3, 6370e3]),
                      movable={1: (5466.875e3, 5970e3)}, motion="cut")
        return L, np.log(HAGER4), sj.StokesConfig(lmax=100)
    p = profile_from_table(cases.t_prem())
    if name == "T3" or name.startswith("T3t"):
        L = sj.Layers(p, np.array([3480e3, 5700e3, 5970e3, 6270e3, 6370e3]), movable={1: (5466.875e3, 5970e3)})
        contrast = 10.0 if name == "T3" else float(name.split("-")[1])
        form = sj.Formulation(compressible=name.startswith("T3t"))
        return L, np.log(np.array([contrast * 1e21, 1e21, 3e20, 1e22])), sj.StokesConfig(lmax=50, formulation=form)
    if name == "T4":
        L = sj.Layers(p, np.array([3480e3, 5800e3, 6270e3, 6370e3]), movable={1: (5466.875e3, 6150e3)})
        return L, np.log(np.array([1e22, 3e20, 1e22])), sj.StokesConfig(lmax=50)
    raise ValueError(name)


def measure_cut(name: str) -> dict:
    """Crossings, derivative jumps, value jumps, ``E_rms`` and ``E_value`` of a cut table."""
    import warnings
    import densities
    warnings.filterwarnings("ignore", "under TALA the kernels jump")
    L, log_eta, cfg = cut_table(name)
    assert L.motion == "cut", (name, L.motion)
    lmax = cfg.lmax
    k_mov = int(L.movable_index[0])
    g = L.ranges[k_mov]
    LOW, HIGH, M = g.low, g.high, g.margin
    r_in = np.asarray(L.input_profile.radius)
    dens = {"smooth": densities.smooth(r_in, lmax, 11), "lebedev": lebedev_on(r_in, lmax),
            "rough": densities.rough(r_in, lmax, 12)}
    cross, kind = L.crossings(k_mov)
    nc = cross.size
    pairs = cross[kind == 4]
    print(f"{name}: {r_in.size} input nodes, range ({LOW / 1e3:.3f}, {HIGH / 1e3:.3f}) km, lmax {lmax}, "
          f"{cfg.formulation.compressible and 'TALA' or 'incompressible'}, {nc} crossings ({pairs.size} density "
          f"pairs at {[float(x / 1e3) for x in pairs]} km)", flush=True)

    def kin(x):
        q = L(L.full_boundaries(x[None]), log_eta)
        return L.input_kernels(sj.stokes_kernels(q, cfg)).geoid

    batch = 64 if lmax <= 50 else 24
    jv = jax.jit(jax.vmap(lambda x: jax.jvp(kin, (x,), (jnp.ones_like(x),))))

    def run(points):
        P, T = [], []
        for c0 in range(0, points.size, batch):
            chunk = points[c0:c0 + batch]
            pad = np.concatenate([chunk, np.full(batch - chunk.size, chunk[-1])])
            a, b = jv(jnp.asarray(pad))
            P.append(np.asarray(a)[: chunk.size]); T.append(np.asarray(b)[: chunk.size])
        return np.concatenate(P), np.concatenate(T)

    grid = np.arange(LOW + M + 1.0, HIGH - M, GRID_STEP)
    D = D_CUT
    pts = np.concatenate([cross - 2 * D, cross - D, cross + D, cross + 2 * D, grid])
    t0 = time.perf_counter()
    Kp, dKp = run(pts)
    t_jvp = time.perf_counter() - t0
    print(f"   {pts.size} jvps in {t_jvp:.1f} s", flush=True)
    dK_minus = 2 * dKp[nc:2 * nc] - dKp[:nc]
    dK_plus = 2 * dKp[2 * nc:3 * nc] - dKp[3 * nc:4 * nc]
    K_minus = 2 * Kp[nc:2 * nc] - Kp[:nc]
    K_plus = 2 * Kp[2 * nc:3 * nc] - Kp[3 * nc:4 * nc]
    K_grid, dK_grid = Kp[4 * nc:], dKp[4 * nc:]
    d0 = lmax + 1 - Kp.shape[1]
    w_in = L.input_weights
    mask = np.zeros((2, lmax + 1, lmax + 1), dtype=bool)
    mask[:, 2:, :] = np.tril(np.ones((lmax + 1, lmax + 1), dtype=bool))[2:]

    def fields_of(Karr, d_in):
        out = np.zeros((Karr.shape[0], 2, lmax + 1, lmax + 1))
        out[:, :, d0:, :] = np.einsum("nlj,jclm->nclm", Karr * w_in, d_in[:, :, d0:, :])
        return out * mask

    rec = dict(table=name, lmax=lmax, n_input=int(r_in.size), n_nodes=int(L.n_nodes), low=LOW, high=HIGH,
               r0=float(g.start), compressible=bool(cfg.formulation.compressible),
               eta=[float(v) for v in np.exp(log_eta)], crossings=int(nc), density_pairs=[float(x) for x in pairs],
               jvps=int(pts.size), jvp_seconds=t_jvp, anomalies={})
    for aname, d_in in dens.items():
        Nm, Np_ = fields_of(dK_minus, d_in), fields_of(dK_plus, d_in)
        Vm, Vp = fields_of(K_minus, d_in), fields_of(K_plus, d_in)
        jump = Np_ - Nm
        vjump = Vp - Vm
        Nmid = 0.5 * (Vp + Vm)
        nrm = np.linalg.norm((0.5 * (Np_ + Nm)).reshape(nc, -1), axis=1)
        rho = np.linalg.norm(jump.reshape(nc, -1), axis=1) / nrm
        length = np.linalg.norm(vjump.reshape(nc, -1), axis=1) / nrm
        on_pair = kind == 4
        i_max = int(np.argmax(rho))
        arec = dict(rho_max=float(rho[i_max]), rho_max_at_km=float(cross[i_max] / 1e3),
                    rho_median=float(np.median(rho)),
                    rho_at_density_pairs=[float(v) for v in rho[on_pair]],
                    value_jump_length_m=[float(v) for v in length],
                    value_jump_length_max_off_pairs_m=float(np.max(length[~on_pair])) if np.any(~on_pair) else 0.0,
                    value_jump_length_at_density_pairs_m=[float(v) for v in length[on_pair]], sigma={})
        Nt_all, dNt_all = fields_of(K_grid, d_in), fields_of(dK_grid, d_in)
        for sr in SIG_R:
            Er, pair_rec = [], []
            Ev = {float(x): [] for x in pairs}
            Lsig = {float(x): [] for x in pairs}
            Lm = {float(x): [] for x in pairs}
            for it, rt in enumerate(grid):
                Nt, dNt = Nt_all[it], dNt_all[it]
                lam = 1.0 / (sr ** 2 * np.sum(dNt ** 2))
                win = np.abs(cross - rt) <= 1.5 * sr
                det = np.sum(jump[win] * (Nmid[win] - Nt[None]), axis=(1, 2, 3))
                Jr2 = lam ** 2 * (det ** 2 + np.sum(jump[win] ** 2, axis=(1, 2, 3)) / lam)
                Er.append(np.sqrt(sr ** 2 * np.sum(Jr2)))
                for ic in np.flatnonzero(on_pair & win):
                    dv = vjump[ic]
                    ev = np.sqrt(lam ** 2 * np.sum(dv * (Nmid[ic] - Nt)) ** 2 + lam * np.sum(dv ** 2))
                    Ev[float(cross[ic])].append(ev)
                    Lsig[float(cross[ic])].append(np.sqrt(lam) * np.linalg.norm(dv))
                    Lm[float(cross[ic])].append(np.linalg.norm(dv) / np.linalg.norm(dNt))
            Er = np.array(Er)
            ib = int(np.argmax(Er))
            for x in Ev:
                v = np.array(Ev[x])
                pair_rec.append(dict(radius_km=x / 1e3, n_r_true=int(v.size),
                                     E_value_min=float(np.min(v)), E_value_median=float(np.median(v)),
                                     E_value_max=float(np.max(v)), L_over_sigma_min=float(np.min(Lsig[x])),
                                     L_median_m=float(np.median(Lm[x]))))
            arec["sigma"][f"{sr / 1e3:.0f}"] = dict(E_rms_max=float(Er[ib]), E_rms_max_at_km=float(grid[ib] / 1e3),
                                                    E_rms_median=float(np.median(Er)), E_value=pair_rec)
            ev_txt = "; ".join(f"E_value at {p['radius_km']:.0f} km {p['E_value_min']:.3g} to {p['E_value_max']:.3g} "
                               f"nat (L/sigma_r min {p['L_over_sigma_min']:.3g}, median L {p['L_median_m'] / 1e3:.2f} km)"
                               for p in pair_rec)
            print(f"   {aname} sigma_r {sr / 1e3:.0f} km: E_rms max {Er[ib]:.2e} nat at {grid[ib] / 1e3:.1f} km"
                  + (f"; {ev_txt}" if ev_txt else ""), flush=True)
        print(f"   {aname}: rho max {rho[i_max]:.2e} at {cross[i_max] / 1e3:.3f} km, median {np.median(rho):.2e}, "
              f"at density pairs {[f'{v:.2e}' for v in rho[on_pair]]}; value jump as a displacement: off the pairs "
              f"max {arec['value_jump_length_max_off_pairs_m']:.2e} m, at the pairs "
              f"{[f'{v:.3g}' for v in length[on_pair]]} m", flush=True)
        rec["anomalies"][aname] = arec
    return rec


def decision_cut(records) -> dict:
    """Gate G6 of the c4b plan on the incompressible cut tables."""
    rows, holds = {}, True
    for rec in records:
        if rec["table"] not in CUT_TABLES:
            continue
        for a, arec in rec["anomalies"].items():
            ev = max([p["E_value_max"] for s in arec["sigma"].values() for p in s["E_value"]], default=0.0)
            er = max(s["E_rms_max"] for s in arec["sigma"].values())
            ok = arec["rho_max"] <= RHO_C1 and er <= E_RMS_CUT and ev <= E_VALUE_CUT
            holds &= ok
            rows[f"{rec['table']} {a}"] = dict(rho_max=arec["rho_max"], E_rms_max=er, E_value_max=ev, ok=bool(ok))
    return dict(RHO_C1=RHO_C1, E_rms_limit=E_RMS_CUT, E_value_limit=E_VALUE_CUT, holds=bool(holds), rows=rows)


def decision(records) -> dict:
    """The rule of plan 8.4 on T1 and T2 for the decision set (smooth, LEBEDEV)."""
    worst = {}
    for rec in records:
        if rec["table"] not in ("T1", "T2"):
            continue
        for a in ("smooth", "lebedev"):
            for sr, srec in rec["anomalies"][a]["sigma"].items():
                worst[f"{rec['table']} {a} sigma {sr} km"] = srec["E_rms_max"]
    holds = all(v <= E_TOL for v in worst.values())
    return dict(E_TOL=E_TOL, holds=holds, E_rms_max=worst)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", nargs="+", default=["T1", "T2", "T1s"])
    ap.add_argument("--json", default=str(REPO / "NOTES" / "benchmarks" / "c4-kink.json"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    path = Path(a.json)
    if path.exists() and not a.overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    t0 = time.time()
    load0 = os.getloadavg()
    tables = [t for name in a.tables for t in (["T3t-1.1", "T3t-3", "T3t-10", "T3t-100"] if name == "T3t" else [name])]
    records = [measure_cut(t) if (t in CUT_TABLES or t.startswith("T3t")) else measure(t) for t in tables]
    dec = decision(records)
    if any(r["table"] in ("T1", "T2") for r in records):
        print(f"decision (plan 8.4): E_rms <= {E_TOL} nat everywhere on T1 and T2 for smooth and LEBEDEV: "
              f"{dec['holds']}")
        for k, v in dec["E_rms_max"].items():
            print(f"   {k}: {v:.3f}")
    dec_cut = decision_cut(records)
    if dec_cut["rows"]:
        print(f"c4b G6 on {', '.join(sorted({k.split()[0] for k in dec_cut['rows']}))}: rho <= {RHO_C1}, "
              f"E_rms <= {E_RMS_CUT} nat, E_value <= {E_VALUE_CUT} nat: {dec_cut['holds']}")
        for k, v in dec_cut["rows"].items():
            print(f"   {k}: rho max {v['rho_max']:.2e}, E_rms max {v['E_rms_max']:.2e}, E_value max {v['E_value_max']:.2e}")
    out = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), host=platform.node(),
               jax=jax.__version__, load_start=load0, load_end=os.getloadavg(), seconds=time.time() - t0,
               group_gap_m=GROUP_GAP, grid_step_m=GRID_STEP, sigma_r_m=list(SIG_R), records=records, decision=dec,
               decision_cut=dec_cut)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"written {path}")


if __name__ == "__main__":
    main()
