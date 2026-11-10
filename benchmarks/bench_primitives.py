"""Accuracy and timing of the candidate JAX primitives of the stokesjx solver.

Run from the repository root:

    python benchmarks/bench_primitives.py [--quick] [--json PATH] [--md PATH]

Parts (plan section 8):

1. Accuracy of every interval-step candidate against ``mpmath.expm`` at 50 digits
   on the grid ``l`` in {1, 2, 10, 50, 100}, the largest ``h`` of the 129- and the
   257-node tables, ``eta / eta_s`` in {1e-3, 1e-1, 1, 1e2, 1e4},
   ``kappa h`` in {0, -0.1, 0.1}, ``rho / rho_s`` in {0.71, 1.23}. A candidate is
   eligible when its error is below 1e-13 on the whole grid.
2. The NaN threshold of ``jax.scipy.linalg.expm`` on the unbalanced matrix.
3. Compile and run time, forward and reverse, of the batched interval steps, of the
   step-and-inverse pair from one set of Pade powers (Pade (6, 6) with the LAPACK
   solve in both product layouts; the order of ``pade_plan`` with the elimination
   solver, with and without its derivative rule, and with the LAPACK solve), of one
   QR and one Gram-Schmidt sweep and of the batched node solve at
   (L, N) = (51, 128) and (101, 256). L is the number of degrees (1..L) and N the
   number of intervals, so N + 1 nodes.
4. The composed stand-in of the solver (candidates.make_standin) at both sizes:
   the Pade order of ``pade_plan`` with elimination without pivoting and its
   derivative rule, batch-last products and Gram-Schmidt (the budget stand-in,
   ``pade_ge_gs``), the same with plain AD through the elimination, Pade (6, 6)
   with the LAPACK solve (batch-last and batch-first products), Taylor-12 +
   Gram-Schmidt and Taylor-12 + QR. The budget stand-in is timed a second time in
   a child process with one XLA thread (``--one-thread``), the regime of several
   sampler chains in parallel processes. The load average of the machine is
   recorded before and after the run.

Error metric of part 1: both the candidate and the reference are taken to the
balanced coordinates ``T^-1 E T`` (``T = D_eta S_l^-1``, benchmarks/candidates.py),
where the entries carry no factor of the viscosity or of ``l``, and the error is
``max |E - E_ref| / max |E_ref|`` there. The plain normwise error of the raw matrix
hides errors of the small entries when the viscosity spans many decades, which is
why it is reported but not used for eligibility.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidates as c  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
R, RC = 6370000.0, 3480000.0
H129 = math.log((RC + (R - RC) / 128) / RC)     # largest h of the 129-node table (at the CMB)
H257 = math.log((RC + (R - RC) / 256) / RC)
GRID_L = (1, 2, 10, 50, 100)
GRID_H = (H129, H257)
GRID_ETA = (1e-3, 1e-1, 1.0, 1e2, 1e4)
GRID_KH = (0.0, -0.1, 0.1)
GRID_RHO = (3198.72 / 4500, 5523.76 / 4500)
ELIGIBLE = 1e-13


def mp_reference(l, eta, rho, h, kh, dps=50):
    """``expm(A(eta) h + K(eta) kappa_h)`` in mpmath, returned in balanced coordinates."""
    with mp.workdps(dps):
        L = l * (l + 1)
        e = mp.mpf(eta)
        rows = [[-2, L, 0, 0, 0, 0], [-1, 1, 0, 1 / e, 0, 0],
                [12 * e, -6 * L * e, 1, L, 0, -mp.mpf(rho)],
                [-6 * e, 2 * (2 * L - 1) * e, -1, -2, -mp.mpf(rho), 0],
                [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, L, 0]]
        a = mp.matrix(rows) * mp.mpf(h)
        k = mp.zeros(6, 6)
        k[0, 0] = -1
        k[2, 0] = 4 * e
        k[3, 0] = -2 * e
        E = mp.expm(a + k * mp.mpf(kh), method="taylor")
        t = np.asarray(c.viscosity_scaling(eta) / c.degree_scaling(float(l)))
        out = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                out[i, j] = float(E[i, j] / mp.mpf(t[i]) * mp.mpf(t[j]))
        return out


def candidate_table(quick: bool = False) -> dict:
    """Part 1: the error of every candidate on the grid."""
    grid = [(l, h, eta, kh, rho) for l in GRID_L for h in GRID_H for eta in GRID_ETA
            for kh in GRID_KH for rho in GRID_RHO]
    if quick:
        grid = grid[::7]
    ref = np.stack([mp_reference(l, eta, rho, h, kh) for (l, h, eta, kh, rho) in grid])
    ls, hs, etas, khs, rhos = (np.array([g[k] for g in grid]) for k in range(5))
    t = np.asarray(c.viscosity_scaling(etas) / c.degree_scaling(ls))
    lmax, hmax = max(GRID_L), max(GRID_H)
    s_bal_t = c.taylor_squarings(c.balanced_norm_bound(lmax, hmax))
    s_bal_p6 = c.pade_squarings(c.balanced_norm_bound(lmax, hmax), 6, "lapack")
    s_bal_p13 = c.pade_squarings(c.balanced_norm_bound(lmax, hmax), 13, "lapack")
    # The elimination forms take their counts with the dominance condition N_m(x) < 2.
    s_bal_p7_ge = c.pade_squarings(c.balanced_norm_bound(lmax, hmax), 7, "elimination")
    s_bal_p9_ge = c.pade_squarings(c.balanced_norm_bound(lmax, hmax), 9, "elimination")
    # Norm bound of the raw matrix over the grid, for the unbalanced static squarings.
    raw = np.asarray(c.a_matrix(ls, etas, rhos)) * hs[:, None, None]
    raw_bound = float(np.max(np.abs(raw).sum(-2))) + 3 * max(GRID_ETA) * 0.1 * 4
    s_raw_t = c.taylor_squarings(raw_bound)
    s_raw_p6 = c.pade_squarings(raw_bound, 6, "lapack")

    def to_bal(E):
        return np.asarray(E) / t[:, :, None] * t[:, None, :]

    cands = {
        "a_jax_unbalanced": lambda: c.step_unbalanced(ls, etas, rhos, hs, khs, method="jax"),
        "b_taylor12_unbalanced": lambda: c.step_unbalanced(ls, etas, rhos, hs, khs, method="taylor",
                                                           terms=12, squarings=s_raw_t),
        "c_pade6_unbalanced": lambda: jnp.asarray(_pade_unbalanced(ls, etas, rhos, hs, khs, s_raw_p6)),
        "d_closed_balanced": lambda: c.step_balanced(ls, etas, rhos, hs, khs, method="closed"),
        "e_taylor12_balanced": lambda: c.step_balanced(ls, etas, rhos, hs, khs, method="taylor",
                                                       terms=12, squarings=s_bal_t),
        "e_pade6_balanced": lambda: c.step_balanced(ls, etas, rhos, hs, khs, method="pade6",
                                                    squarings=s_bal_p6),
        "e_pade6_balanced_batch_last": lambda: c.step_balanced(ls, etas, rhos, hs, khs, method="pade6_bl",
                                                               squarings=s_bal_p6),
        "e_pade13_balanced": lambda: c.step_balanced(ls, etas, rhos, hs, khs, method="pade13",
                                                     squarings=s_bal_p13),
        "e_pade7_balanced_batch_last_elimination": lambda: c.step_balanced(
            ls, etas, rhos, hs, khs, method="pade7_bl_ge", squarings=s_bal_p7_ge),
        "e_pade9_balanced_batch_last_elimination": lambda: c.step_balanced(
            ls, etas, rhos, hs, khs, method="pade9_bl_ge", squarings=s_bal_p9_ge),
        "e_jax_balanced": lambda: c.step_balanced(ls, etas, rhos, hs, khs, method="jax"),
    }
    out = {}
    for name, fn in cands.items():
        E = np.asarray(fn())
        finite = bool(np.all(np.isfinite(E)))
        Eb = to_bal(E)
        err_b = np.max(np.abs(Eb - ref), axis=(1, 2)) / np.max(np.abs(ref), axis=(1, 2))
        Eraw_ref = ref * t[:, :, None] / t[:, None, :]
        err_raw = np.max(np.abs(E - Eraw_ref), axis=(1, 2)) / np.max(np.abs(Eraw_ref), axis=(1, 2))
        mask = np.ones(len(grid), bool)
        if name.startswith("d_"):
            mask = khs == 0.0      # the closed form is for the incompressible matrix only
        worst = int(np.argmax(np.where(mask, err_b, -1)))
        out[name] = dict(
            finite=finite,
            max_err_balanced=float(np.max(err_b[mask])) if finite else float("nan"),
            max_err_raw_normwise=float(np.max(err_raw[mask])) if finite else float("nan"),
            worst_point=dict(zip(("l", "h", "eta", "kappa_h", "rho"), map(float, grid[worst]))),
            eligible=bool(finite and np.max(err_b[mask]) < ELIGIBLE),
            scope="kappa_h = 0 only, NaN at h = 0" if name.startswith("d_") else "full grid")
    # Zero-width interval (every jump pair has one): is each candidate finite at h = 0?
    # Evaluated at l = 2, eta = 1, rho = 1, kappa_h = 0.05 (and kappa_h = 0 for the
    # closed form, which has no kappa term).
    for name in cands:
        if name.startswith(("a_", "b_", "c_")):
            continue
        meth = {"d_closed_balanced": "closed", "e_taylor12_balanced": "taylor", "e_pade6_balanced": "pade6",
                "e_pade6_balanced_batch_last": "pade6_bl", "e_pade13_balanced": "pade13",
                "e_pade7_balanced_batch_last_elimination": "pade7_bl_ge",
                "e_pade9_balanced_batch_last_elimination": "pade9_bl_ge",
                "e_jax_balanced": "jax"}[name]
        kh0 = 0.0 if meth == "closed" else 0.05
        sq0 = {"taylor": s_bal_t, "pade13": s_bal_p13, "pade7_bl_ge": s_bal_p7_ge,
               "pade9_bl_ge": s_bal_p9_ge}.get(meth, s_bal_p6)
        E0 = np.asarray(c.step_balanced(2.0, 1.0, 1.0, 0.0, kh0, method=meth, squarings=sq0))
        out[name]["finite_at_h0"] = bool(np.all(np.isfinite(E0)))
    out["_squarings"] = dict(balanced_taylor12=s_bal_t, balanced_pade6=s_bal_p6, balanced_pade13=s_bal_p13,
                             balanced_pade7_elimination=s_bal_p7_ge, balanced_pade9_elimination=s_bal_p9_ge,
                             raw_taylor12=s_raw_t, raw_pade6=s_raw_p6,
                             balanced_norm_bound=c.balanced_norm_bound(lmax, hmax), raw_norm_bound=raw_bound,
                             grid_points=len(grid))
    return out


def _pade_unbalanced(ls, etas, rhos, hs, khs, s):
    """Candidate (c): Pade (6, 6) with ``s`` static squarings of the raw exponent
    ``A(eta) h + K(eta) kappa_h`` (no balancing), for the accuracy table only. The
    raw matrix has entries up to ``6 l (l + 1) eta h``, so ``s`` is 20 or more and
    every squaring doubles the relative error of the small entries."""
    M = (np.asarray(c.a_matrix(ls, etas, rhos)) * hs[:, None, None]
         + np.asarray(c.k_matrix(etas)) * khs[:, None, None])
    return c.expm_pade(jnp.asarray(M), 6, s, solver="lapack")


def nan_threshold() -> list:
    """Part 2: is ``jax.scipy.linalg.expm`` of the unbalanced matrix and its gradient
    finite at l = 100?"""
    rows = []
    for eta in (1.0, 1e2, 1e4):
        def f(log_eta):
            M = c.a_matrix(100.0, jnp.exp(log_eta), 1.2) * H129
            return jnp.sum(jax.scipy.linalg.expm(M))
        v = f(jnp.log(eta))
        g = jax.grad(f)(jnp.log(eta))
        rows.append(dict(l=100, eta=eta, h=H129, value_finite=bool(jnp.isfinite(v)),
                         grad_finite=bool(jnp.isfinite(g)), value=float(v), grad=float(g)))
    return rows


def time_fn(fn, args, repeats=20):
    """Compile time of ``jit(fn).lower(*args).compile()`` and the median run time."""
    t0 = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    tc = time.perf_counter() - t0
    jax.block_until_ready(compiled(*args))
    ts = []
    for _ in range(repeats):
        t1 = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        ts.append(time.perf_counter() - t1)
    return dict(compile_s=tc, run_ms=1e3 * statistics.median(ts))


def timings(sizes, one_thread: bool = False) -> dict:
    """Part 3 and 4. With ``one_thread`` only the budget stand-in of part 4 (the
    child process of ``one_thread_timings``)."""
    out = {}
    if one_thread:
        return {f"L{L}_N{N}": standin_timings(L, N, STANDINS[:1]) for (L, N) in sizes}
    rng = np.random.default_rng(0)
    for (L, N) in sizes:
        key = f"L{L}_N{N}"
        res = {}
        hmax = math.log((RC + (R - RC) / N) / RC)
        ls = jnp.arange(1, L + 1, dtype=jnp.float64)[:, None]
        rho = jnp.asarray(rng.uniform(0.7, 1.25, N))[None, :]
        h = jnp.full((1, N), hmax)
        kh = jnp.asarray(rng.uniform(-0.01, 0.01, N))[None, :]
        W = jnp.asarray(rng.standard_normal((L, N, 6, 6)))
        s_t = c.taylor_squarings(c.balanced_norm_bound(L, hmax))
        s_p = c.pade_squarings(c.balanced_norm_bound(L, hmax), 6, "lapack")
        # Order and squaring count of the selected pair (elimination solver).
        plan = c.pade_plan(c.balanced_norm_bound(L, hmax), "elimination")
        res["pade_plan"] = plan
        m_ge, s_ge = plan["order"], plan["squarings"]
        meth_ge = f"pade{m_ge}_bl_ge"
        steps = {
            "a_jax_unbalanced": lambda le: c.step_unbalanced(ls, jnp.exp(le), rho, h, kh, method="jax"),
            "e_taylor12_balanced": lambda le: c.step_balanced(ls, jnp.exp(le), rho, h, kh, method="taylor",
                                                              squarings=s_t),
            "e_pade6_balanced": lambda le: c.step_balanced(ls, jnp.exp(le), rho, h, kh, method="pade6",
                                                           squarings=s_p),
            "e_pade6_balanced_batch_last": lambda le: c.step_balanced(ls, jnp.exp(le), rho, h, kh,
                                                                      method="pade6_bl", squarings=s_p),
            "e_pade_plan_balanced_batch_last_elimination": lambda le: c.step_balanced(
                ls, jnp.exp(le), rho, h, kh, method=meth_ge, squarings=s_ge),
            "e_jax_balanced": lambda le: c.step_balanced(ls, jnp.exp(le), rho, h, kh, method="jax"),
        }
        le0 = jnp.asarray(rng.uniform(-1, 1, N))[None, :]
        for name, fn in steps.items():
            f = (lambda fn: lambda le: jnp.sum(fn(le) * W))(fn)
            res[f"step_{name}_forward"] = time_fn(f, (le0,))
            # The viscosity enters the balanced candidates only through the diagonal
            # similarity, so this gradient never passes through the exponential.
            res[f"step_{name}_gradient"] = time_fn(jax.grad(f), (le0,))
        # Gradient through the exponential: the density and the interval width are
        # controls (a movable boundary changes h). Balanced candidates only.
        for name in ("e_taylor12_balanced", "e_pade6_balanced", "e_pade6_balanced_batch_last",
                     "e_pade_plan_balanced_batch_last_elimination"):
            meth = {"e_taylor12_balanced": "taylor", "e_pade6_balanced": "pade6",
                    "e_pade6_balanced_batch_last": "pade6_bl",
                    "e_pade_plan_balanced_batch_last_elimination": meth_ge}[name]
            sq = {"taylor": s_t, meth_ge: s_ge}.get(meth, s_p)

            def g(le, rho_, h_, meth=meth, sq=sq):
                return jnp.sum(c.step_balanced(ls, jnp.exp(le), rho_, h_, kh, method=meth, squarings=sq) * W)
            res[f"step_{name}_gradient_all"] = time_fn(jax.grad(g, argnums=(0, 1, 2)), (le0, rho, h))
        # The step and its inverse (for the downward sweep) from one set of Pade
        # powers, r(-X) = r(X)^-1, in both product layouts; control: rho, h, kappa_h
        # (all three enter the exponent).
        Wi = jnp.asarray(rng.standard_normal((L, N, 6, 6)))
        # Key, order, squarings, product layout, solver of the denominators.
        pairs = (("pade6_pair_batch_first", 6, s_p, "batch_first", "lapack"),
                 ("pade6_pair_batch_last", 6, s_p, "batch_last", "lapack"),
                 ("pade_plan_pair_lapack", m_ge, s_ge, "batch_last", "lapack"),
                 ("pade_plan_pair_elimination_plain", m_ge, s_ge, "batch_last", "elimination_plain"),
                 ("pade_plan_pair_elimination", m_ge, s_ge, "batch_last", "elimination"))
        for key_, order, sq, layout, solver in pairs:
            def pair_fn(rho_, h_, kh_, order=order, sq=sq, layout=layout, solver=solver):
                Ef, Eb = c.expm_pade_pair(c.balanced_argument(ls, rho_, h_, kh_), order, sq,
                                          layout=layout, solver=solver)
                return jnp.sum(Ef * W) + jnp.sum(Eb * Wi)
            res[f"{key_}_forward"] = time_fn(pair_fn, (rho, h, kh))
            res[f"{key_}_gradient_all"] = time_fn(jax.grad(pair_fn, argnums=(0, 1, 2)), (rho, h, kh))
        # Sweep alone: N random well-scaled steps.
        P = jnp.asarray(rng.standard_normal((N, L, 6, 6)) * 0.3 + np.eye(6))
        start = jnp.asarray(rng.standard_normal((L, 6, 3)))
        Wq = jnp.asarray(rng.standard_normal((N, L, 6, 3)))
        for ortho in ("qr", "gs"):
            # One sweep per evaluation: both terms of the functional use the same call.
            def f(P, o=ortho):
                Qs, logs = c.sweep(P, start, o)
                return jnp.sum(Qs * Wq) + jnp.sum(logs)
            res[f"sweep_{ortho}_forward"] = time_fn(f, (P,))
            res[f"sweep_{ortho}_gradient"] = time_fn(jax.grad(f), (P,))
        # Batched node solve.
        M = jnp.asarray(rng.standard_normal((N + 1, L, 6, 6)) + 3 * np.eye(6))
        b = jnp.asarray(rng.standard_normal((N + 1, L, 6, 1)))
        Ws = jnp.asarray(rng.standard_normal((N + 1, L, 6, 1)))
        f = lambda M: jnp.sum(jnp.linalg.solve(M, b) * Ws)
        res["solve_forward"] = time_fn(f, (M,))
        res["solve_gradient"] = time_fn(jax.grad(f), (M,))
        # Composed stand-ins (STANDINS; the first is the budget stand-in).
        res.update(standin_timings(L, N, STANDINS))
        out[key] = res
    return out


# Composed stand-ins of part 4: key, method, orthonormalisation, product layout of
# the Pade stage, solver of the Pade denominators. The first entry is the budget
# stand-in of NOTES/CONVENTIONS.md section 7; ``pade6_gs`` is the LAPACK comparison
# form (Pade (6, 6), two squarings, LAPACK solve of the denominators).
STANDINS = (("standin_pade_ge_gs", "pade", "gs", "batch_last", "elimination"),
            ("standin_pade_ge_plain_gs", "pade", "gs", "batch_last", "elimination_plain"),
            ("standin_pade6_gs", "pade6", "gs", "batch_last", "lapack"),
            ("standin_pade6_gs_batch_first", "pade6", "gs", "batch_first", "lapack"),
            ("standin_taylor_gs", "taylor", "gs", "batch_first", "lapack"),
            ("standin_taylor_qr", "taylor", "qr", "batch_first", "lapack"))


def standin_timings(L, N, standins) -> dict:
    """Part 4: compile and run times of the composed stand-ins at one size."""
    res = {}
    hmax = math.log((RC + (R - RC) / N) / RC)
    for key_, method, ortho, layout, solver in standins:
        kw = dict(method=method, ortho=ortho, layout=layout, solver=solver)
        fwd, grd, args, info = c.make_standin(L, N, hmax, **kw)
        res[f"{key_}_forward"] = time_fn(fwd, args, repeats=20)
        res[f"{key_}_gradient"] = time_fn(grd, args, repeats=20)
        # Gradient with respect to every input (log_eta, rho, h, kappa_h): the
        # case of movable boundaries, which differentiates through expm.
        fwd_raw = c.make_standin(L, N, hmax, jit=False, **kw)[0]
        res[f"{key_}_gradient_all"] = time_fn(jax.grad(fwd_raw, argnums=(0, 1, 2, 3)), args, repeats=20)
        res[f"{key_}_info"] = info
    return res


# XLA flags of the one-thread run: no Eigen thread pool and one intra-op thread.
ONE_THREAD_FLAGS = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


def one_thread_timings(sizes) -> dict:
    """The budget stand-in in a child process with ``ONE_THREAD_FLAGS``. XLA reads
    its flags once per process, so the run needs its own interpreter."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "one_thread.json"
        env = dict(os.environ, XLA_FLAGS=ONE_THREAD_FLAGS)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--one-thread-child", str(path),
               "--sizes", ",".join(f"{L}x{N}" for L, N in sizes)]
        subprocess.run(cmd, env=env, check=True)
        return json.loads(path.read_text())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--json", default=str(REPO / "NOTES/benchmarks/c1-primitives.json"))
    p.add_argument("--md", default=str(REPO / "NOTES/benchmarks.md"))
    p.add_argument("--one-thread-child", default=None, help=argparse.SUPPRESS)
    p.add_argument("--sizes", default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    sizes = [(51, 128)] if a.quick else [(51, 128), (101, 256)]
    if a.one_thread_child:
        # Child process of one_thread_timings: XLA_FLAGS is set by the parent.
        if a.sizes:
            sizes = [tuple(int(v) for v in item.split("x")) for item in a.sizes.split(",")]
        child = dict(xla_flags=os.environ.get("XLA_FLAGS", ""), loadavg=list(os.getloadavg()),
                     timings=timings(sizes, one_thread=True))
        Path(a.one_thread_child).write_text(json.dumps(child, indent=1) + "\n")
        return
    t0 = time.time()
    record = dict(host=platform.node(), machine=platform.machine(), date=datetime.datetime.now().isoformat(
        timespec="seconds"), jax=jax.__version__, numpy=np.__version__, mpmath=mp.__version__,
        devices=[str(d) for d in jax.devices()], cpu_count=os.cpu_count(),
        xla_flags=os.environ.get("XLA_FLAGS", ""), loadavg_start=list(os.getloadavg()))
    record["candidates"] = candidate_table(a.quick)
    record["nan_threshold"] = nan_threshold()
    record["timings"] = timings(sizes)
    record["one_thread"] = one_thread_timings(sizes)
    record["loadavg_end"] = list(os.getloadavg())
    record["wall_s"] = time.time() - t0
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["candidates"], indent=1))
    print(json.dumps(record["nan_threshold"], indent=1))
    for label, block in (("", record["timings"]), ("1 thread ", record["one_thread"]["timings"])):
        for k, v in block.items():
            for kk, vv in v.items():
                if isinstance(vv, dict) and "run_ms" in vv:
                    print(f"{label}{k:12s} {kk:48s} compile {vv['compile_s']:7.2f} s  run {vv['run_ms']:9.3f} ms")
    print("load average start / end:", record["loadavg_start"], record["loadavg_end"])
    if not a.quick:
        md = Path(a.md)
        if not md.exists():
            md.write_text("# Benchmarks\n\nOne row per run of a benchmark script. Host, date and "
                          "versions per row; the JSON holds every number.\n\n"
                          "| date | host | script | JAX | stand-in | (51,128) forward ms | gradient log_eta ms | "
                          "gradient all ms | compile fwd/grad-all s | (101,256) forward ms | gradient log_eta ms | "
                          "gradient all ms | compile fwd/grad-all s | JSON |\n"
                          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        rows = []
        load = "load {:.1f} to {:.1f}".format(record["loadavg_start"][0], record["loadavg_end"][0])
        for t, best, label in ((record["timings"], "standin_pade_ge_gs", "pade_ge_gs"),
                               (record["one_thread"]["timings"], "standin_pade_ge_gs", "pade_ge_gs, one XLA thread"),
                               (record["timings"], "standin_pade_ge_plain_gs", "pade_ge_plain_gs"),
                               (record["timings"], "standin_pade6_gs", "pade6_gs"),
                               (record["timings"], "standin_pade6_gs_batch_first", "pade6_gs_batch_first"),
                               (record["timings"], "standin_taylor_gs", "taylor_gs")):
            cells = []
            for size in ("L51_N128", "L101_N256"):
                T = t[size]
                cells += [f"{T[best + '_forward']['run_ms']:.1f}", f"{T[best + '_gradient']['run_ms']:.1f}",
                          f"{T[best + '_gradient_all']['run_ms']:.1f}",
                          f"{T[best + '_forward']['compile_s']:.2f}/{T[best + '_gradient_all']['compile_s']:.2f}"]
            rows.append(f"| {record['date']} | {record['host']} | bench_primitives.py | {record['jax']} | "
                        f"{label} ({load}) | " + " | ".join(cells) + f" | {Path(a.json).name} |\n")
        row = "".join(rows)
        with open(md, "a") as f:
            f.write(row)


if __name__ == "__main__":
    main()
