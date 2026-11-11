"""Timing of the forward solver of stokesjx and the budget gate of CONVENTIONS 7.

Run from the repository root:

    python benchmarks/bench_forward.py [--quick] [--rounds 15] [--json PATH] [--md PATH]

Sizes (plan section 9 of chunk 2):

    S1  the node table of the golden run hager4_fsfs (129 LEBEDEV nodes), lmax = 50;
        budget row (51, 128)
    S2  the LEBEDEV table with its 128 mid-points (257 nodes, rho0 linear between the
        LEBEDEV nodes, hager4 viscosity; ``lebedev_refined``), lmax = 100; budget row
        (101, 256)
    S3  the S1 table at lmax = 100; the S2 row as an absolute upper limit, and the
        ratio against a stand-in of its own size, make_standin(100, 128, h_max)
    S4  the historical test_4_ishan table (257 nodes) at lmax = 100, when legacy/
        exists; recorded only

Quantities per size: the forward map from the controls to the four kernels; the
gradient of a scalar functional of the kernels with respect to ``log eta`` per node;
the gradient with respect to ``log eta``, ``rho0`` and the radius ("gradient all");
their compile times; the run time as median and minimum of 20 calls after one warm-up
call; the same in a child process with one XLA thread; ``vmap`` of the gradient over 4
chains; the split by stage (interval steps, sweeps, node solve and outputs) from
separately jitted stages.

Gate (CONVENTIONS 7): 15 rounds that alternate the solver and the stand-in
``benchmarks/candidates.py::make_standin`` at the budget size, median per quantity, the
load average at the start and at the end; the ratio solver / stand-in must be at most
2 for every run quantity, and the absolute times must be inside the budget table.

Compile times are measured in fresh child processes (CONVENTIONS 7), each after one
unrelated warm-up compile that also lowers a tiny LAPACK solve: the first LAPACK lowering
of a process imports ``scipy.linalg`` (``jaxlib/lapack.py``, ``prepare_lapack_call``,
0.12 s on reunion), which is a start-up cost of the process and not a compile cost of
either program. No solve of the package runs in the child before the timed section: the
scales of the functional come from the parent process, so the forward compile includes
the trace of the core. Two protocols, three processes each, median per quantity:
"first" (forward, then the gradient with respect to all controls: the first gradient
compile of the process) and "sequential" (forward, gradient ``log eta``, gradient all,
in this order in one process). Solver and stand-in are measured with the same
protocols.

Options measured at S1 and S2 (interleaved): the node solver (``"lapack"`` against
``"pivot_elimination"``) and block upper triangular products in the exponential
stage (``matmul_batch_last_block``, a variant of this script that is swapped into the
package for the measurement only).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO / "tests"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import stokesjx as sj  # noqa: E402
from stokesjx import linalg, propagate, system  # noqa: E402

R, RC = 6370e3, 3480e3
BUDGET = {"S1": dict(forward=17.0, gradient_log_eta=27.0, gradient_all=48.0),
          "S2": dict(forward=67.0, gradient_log_eta=106.0, gradient_all=181.0),
          "S3": dict(forward=67.0, gradient_log_eta=106.0, gradient_all=181.0)}
COMPILE_BUDGET = dict(forward=0.4, gradient_all=0.8)
# Stand-in of each size: (degrees L, intervals N) of make_standin.
STANDIN = {"S1": (51, 128), "S2": (101, 256), "S3": (100, 128)}
ONE_THREAD_FLAGS = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
QUANTITIES = ("forward", "gradient_log_eta", "gradient_all")


# --------------------------------------------------------------------------
# Tables.
# --------------------------------------------------------------------------


def golden_profile():
    """The node table of the golden run hager4_fsfs (129 LEBEDEV nodes, hager4
    viscosity, rho_c = 9900, consistent core), as a profile with the default flags."""
    import cases
    from adapters import profile_from_table
    return profile_from_table(cases.golden_table("hager4_fsfs"))


def lebedev_refined():
    """The LEBEDEV table with the 128 mid-points added (257 nodes, CMB first): ``rho0``
    linear between the LEBEDEV nodes, the hager4 viscosity at every node (Pa s,
    ``eta_rel x 1e19``), rho_c = 9900, consistent core. Defined here, so that
    tests/cases.py does not change after the oracle files recorded its sha256."""
    import cases
    r = [Decimal(x) for x in cases.LEBEDEV_RADIUS]
    rho = [Decimal(x) for x in cases.LEBEDEV_RHO0]
    radius, rho0 = [], []
    for k in range(len(r)):
        radius.append(r[k])
        rho0.append(rho[k])
        if k + 1 < len(r):
            radius.append((r[k] + r[k + 1]) / 2)
            rho0.append((rho[k] + rho[k + 1]) / 2)
    eta = [float(Decimal(cases.layered_eta_rel(str(x), cases.HAGER4_LAYERS)) * Decimal("1e19")) for x in radius]
    return sj.RadialProfile(np.array([float(x) for x in radius]), np.array([float(x) for x in rho0]),
                            np.array(eta), core_density=9900.0)


def ishan_profile():
    """The test_4_ishan table (257 nodes) when legacy/CODE-A exists, else None."""
    import cases
    from adapters import profile_from_table
    root = REPO / "legacy" / "CODE-A"
    if not (root / "test_4_ishan" / "Summary").exists():
        return None
    return profile_from_table(cases.ishan_table(root))


SIZES = {"S1": (golden_profile, 50), "S2": (lebedev_refined, 100), "S3": (golden_profile, 100),
         "S4": (ishan_profile, 100)}


# --------------------------------------------------------------------------
# The solver functions.
# --------------------------------------------------------------------------


def kernel_scales(p0, lmax, numerics=sj.Numerics(), formulation=None):
    """The largest absolute value of each of the four kernels at the base point (one
    solve; the parent process computes them for the compile children)."""
    cfg = sj.StokesConfig(lmax=lmax, numerics=numerics, formulation=formulation or sj.Formulation())
    K0 = sj.stokes_kernels(p0, cfg)
    ks = (K0.geoid, K0.surface_topography, K0.cmb_topography, K0.surface_velocity)
    return [float(jnp.max(jnp.abs(k))) or 1.0 for k in ks]


def solver_functions(p0, lmax, numerics=sj.Numerics(), scales=None, formulation=None):
    """``(forward, gradient_log_eta, gradient_all, x0)`` of the solver: a weighted sum of
    the four kernels, each divided by its largest value at the base point, as a
    function of ``x = (log_eta, rho0, radius)``; the forward function returns the four
    kernels (the forward map), the gradients the gradient of the sum. ``scales``: the
    four kernel scales (``kernel_scales``); without them one solve computes them here,
    which a compile child must avoid (CONVENTIONS 7)."""
    cfg = sj.StokesConfig(lmax=lmax, numerics=numerics, formulation=formulation or sj.Formulation())
    x0 = (jnp.log(p0.eta), p0.rho0, p0.radius)
    if scales is None:
        scales = kernel_scales(p0, lmax, numerics, formulation)
    plan = propagate.static_plan(p0.structure, cfg)
    shape = (len(plan.degrees), p0.structure.n_nodes)
    rng = np.random.default_rng(0)
    W = [jnp.asarray(rng.standard_normal(shape)) / s for s in scales]

    def kernels(log_eta, rho0, radius):
        K = sj.stokes_kernels(p0.replace(eta=jnp.exp(log_eta), rho0=rho0, radius=radius), cfg)
        return K.geoid, K.surface_topography, K.cmb_topography, K.surface_velocity

    def functional(log_eta, rho0, radius):
        return sum(jnp.sum(w * k) for w, k in zip(W, kernels(log_eta, rho0, radius)))

    return dict(forward=kernels, gradient_log_eta=jax.grad(functional, argnums=0),
                gradient_all=jax.grad(functional, argnums=(0, 1, 2)), functional=functional), x0


def standin_functions(L, N):
    """The budget stand-in of CONVENTIONS 7 at (L, N): forward, gradient with respect to
    ``log_eta``, gradient with respect to all four inputs."""
    import candidates as c
    hmax = math.log((RC + (R - RC) / N) / RC)
    fwd, _, args, info = c.make_standin(L, N, hmax, jit=False)
    return dict(forward=fwd, gradient_log_eta=jax.grad(fwd, argnums=0),
                gradient_all=jax.grad(fwd, argnums=(0, 1, 2, 3))), args, info


def warm_up():
    """One unrelated compile that also lowers a tiny LAPACK solve (module docstring)."""
    jax.jit(lambda a: jnp.linalg.solve(a, jnp.sin(a)[:, :1])).lower(jnp.eye(3)).compile()


def time_compiled(fn, args, repeats=20):
    """Compile time (lower and compile) and the median and minimum run time of
    ``repeats`` calls after one warm-up call."""
    t0 = time.perf_counter()
    comp = jax.jit(fn).lower(*args).compile()
    tc = time.perf_counter() - t0
    jax.block_until_ready(comp(*args))
    ts = []
    for _ in range(repeats):
        t1 = time.perf_counter()
        jax.block_until_ready(comp(*args))
        ts.append(time.perf_counter() - t1)
    return dict(compile_s=tc, run_ms=1e3 * statistics.median(ts), run_min_ms=1e3 * min(ts)), comp


# --------------------------------------------------------------------------
# Stage split.
# --------------------------------------------------------------------------


def stage_split(p0, lmax) -> dict:
    """Compile and run time of separately jitted stages of the forward map: the interval
    steps (exponential stage), the two sweeps, the node solve with the outputs, and the
    rest (gravity, boundary bases and rows, jumps, imbalance: the whole forward map
    minus the three stages)."""
    cfg = sj.StokesConfig(lmax=lmax)
    plan = propagate.static_plan(p0.structure, cfg)
    S = system.degree_scaling(plan.degrees)
    L = len(plan.degrees)
    rng = np.random.default_rng(1)
    start = jnp.asarray(np.linalg.qr(rng.standard_normal((L, 6, 3)))[0])
    rows1, rows4 = jnp.asarray(rng.standard_normal((L, 1, 6))), jnp.asarray(rng.standard_normal((L, 4, 6)))
    out = {}
    res, comp = time_compiled(lambda p: propagate.interval_steps(p, jnp.max(p.eta), plan, cfg), (p0,))
    out["steps"] = res
    Zu, Zd = comp(p0)
    res, comp = time_compiled(lambda a, b: propagate.sweeps(a, b, start * S[:, :, None], start * S[:, :, None],
                                                            rows1, rows4), (Zu, Zd))
    out["sweeps"] = res
    sw = comp(Zu, Zd)
    Jz = jnp.asarray(rng.standard_normal((p0.structure.n_nodes, L, 6)))
    out["node_solve_outputs"], _ = time_compiled(lambda s, j: propagate.solve_nodes_and_outputs(s, j, plan), (sw, Jz))
    out["whole_forward"], _ = time_compiled(lambda p: propagate._core(p, plan, cfg, "largest"), (p0,))
    out["rest"] = dict(run_ms=out["whole_forward"]["run_ms"] - sum(out[k]["run_ms"] for k in
                                                                    ("steps", "sweeps", "node_solve_outputs")))
    return out


# --------------------------------------------------------------------------
# Block upper triangular products (measured option, not package code).
# --------------------------------------------------------------------------


# The package product, kept before any swap (the block form calls it on the blocks).
_FULL_MATMUL = linalg.matmul_batch_last


def matmul_batch_last_block(a, b):
    """Product of two block upper triangular 6x6 stacks, batch-last.

    With the order (mechanics y1..y4, potential y5, y6) the exponent B, its powers and
    its exponential are block upper triangular (equations.md 4.4): rows 5 and 6 hold no
    mechanical unknown. ``[A11 A12; 0 A22] [B11 B12; 0 B22] = [A11 B11,
    A11 B12 + A12 B22; 0, A22 B22]``: 64 + 32 + 16 + 8 = 120 multiplications in place
    of 216. Valid only for block upper triangular inputs."""
    mm = _FULL_MATMUL
    a11, a12, a22 = a[:4, :4], a[:4, 4:], a[4:, 4:]
    b11, b12, b22 = b[:4, :4], b[:4, 4:], b[4:, 4:]
    c11 = mm(a11, b11)
    c12 = mm(a11, b12) + mm(a12, b22)
    c22 = mm(a22, b22)
    zero = jnp.zeros((2, 4) + c22.shape[2:], dtype=c22.dtype)
    return jnp.concatenate([jnp.concatenate([c11, c12], axis=1), jnp.concatenate([zero, c22], axis=1)], axis=0)


def block_variant_functions(p0, lmax):
    """The solver functions with the block products swapped into the exponential stage
    for the duration of every trace. The node solver is LAPACK, so every product that
    the swap reaches (Pade powers, squarings, the tangent ``dA X`` of the denominator
    solve) has block upper triangular factors. A fresh ``jax.jit`` of the core, so that
    no trace of the package with the full products is reused, and every trace of it
    (also the one for the scales of the functional) runs with the swap."""
    cfg = sj.StokesConfig(lmax=lmax, numerics=sj.Numerics(node_solver="lapack"))
    plan = propagate.static_plan(p0.structure, cfg)
    core = jax.jit(propagate._core, static_argnames=("plan", "config", "ref_rule"))

    def swapped(fn):
        def wrapped(*args):
            saved = linalg.matmul_batch_last
            linalg.matmul_batch_last = matmul_batch_last_block
            try:
                return fn(*args)
            finally:
                linalg.matmul_batch_last = saved
        return wrapped

    @swapped
    def kernels(log_eta, rho0, radius):
        out = core(p0.replace(eta=jnp.exp(log_eta), rho0=rho0, radius=radius), plan, cfg, "largest")
        return out["geoid"], out["surface_topography"], out["cmb_topography"], out["surface_velocity"]
    x0 = (jnp.log(p0.eta), p0.rho0, p0.radius)
    rng = np.random.default_rng(0)
    W = [jnp.asarray(rng.standard_normal(k.shape)) / float(jnp.max(jnp.abs(k))) for k in kernels(*x0)]

    def functional(log_eta, rho0, radius):
        return sum(jnp.sum(w * k) for w, k in zip(W, kernels(log_eta, rho0, radius)))
    return dict(forward=kernels, gradient_log_eta=swapped(jax.grad(functional)),
                gradient_all=swapped(jax.grad(functional, argnums=(0, 1, 2)))), x0


def _el(a, b):
    """Largest e_l over the rows of two kernel arrays."""
    a, b = np.asarray(a), np.asarray(b)
    scale = np.max(np.abs(b), axis=1)
    ok = scale > 0
    return float(np.max(np.max(np.abs(a - b), axis=1)[ok] / scale[ok]))


# --------------------------------------------------------------------------
# Interleaved measurements.
# --------------------------------------------------------------------------


def interleaved(groups: dict, rounds: int, calls: int = 5) -> dict:
    """``groups``: name -> (compiled function, args). Each round calls every function
    ``calls`` times in turn and records the median; returns the median and the minimum
    over the rounds per name, in ms."""
    t = {k: [] for k in groups}
    for _ in range(rounds):
        for k, (f, args) in groups.items():
            b = []
            for _ in range(calls):
                t0 = time.perf_counter()
                jax.block_until_ready(f(*args))
                b.append(time.perf_counter() - t0)
            t[k].append(statistics.median(b))
    return {k: dict(median_ms=1e3 * statistics.median(v), min_ms=1e3 * min(v)) for k, v in t.items()}


def compiled_set(fns, args):
    return {q: jax.jit(fns[q]).lower(*args).compile() for q in QUANTITIES}


def gate(size: str, rounds: int) -> dict:
    """Solver against the stand-in at the budget size, interleaved."""
    make, lmax = SIZES[size]
    p0 = make()
    fns, x0 = solver_functions(p0, lmax)
    L, N = STANDIN[size]
    sfns, sargs, info = standin_functions(L, N)
    groups = {}
    for q, comp in compiled_set(fns, x0).items():
        groups[f"solver_{q}"] = (comp, x0)
    for q, comp in compiled_set(sfns, sargs).items():
        groups[f"standin_{q}"] = (comp, sargs)
    load0 = list(os.getloadavg())
    res = interleaved(groups, rounds)
    res["loadavg_start"], res["loadavg_end"] = load0, list(os.getloadavg())
    res["ratio"] = {q: res[f"solver_{q}"]["median_ms"] / res[f"standin_{q}"]["median_ms"] for q in QUANTITIES}
    res["standin_info"] = info
    return res


def options(size: str, rounds: int) -> dict:
    """The node solver and the block products, interleaved, with value agreement."""
    make, lmax = SIZES[size]
    p0 = make()
    base, x0 = solver_functions(p0, lmax, sj.Numerics(node_solver="lapack"))
    piv, _ = solver_functions(p0, lmax, sj.Numerics(node_solver="pivot_elimination"))
    blk, _ = block_variant_functions(p0, lmax)
    groups = {}
    for name, fns in (("lapack", base), ("pivot_elimination", piv), ("block_products", blk)):
        for q in QUANTITIES:
            groups[f"{name}_{q}"] = (jax.jit(fns[q]).lower(*x0).compile(), x0)
    vals = {name: groups[f"{name}_forward"][0](*x0) for name in ("lapack", "pivot_elimination", "block_products")}
    agree = {name: max(_el(a, b) for a, b in zip(vals[name], vals["lapack"]))
             for name in ("pivot_elimination", "block_products")}
    load0 = list(os.getloadavg())
    res = interleaved(groups, rounds)
    res["loadavg_start"], res["loadavg_end"] = load0, list(os.getloadavg())
    res["agreement_e_l"] = agree
    return res


# --------------------------------------------------------------------------
# Child processes: compile times and one-thread timings.
# --------------------------------------------------------------------------


PROTOCOLS = {"first": ("forward", "gradient_all"), "sequential": QUANTITIES}


def child_compile(which: str, size: str, protocol: str = "sequential", scales: str = "") -> dict:
    """Compile times in this (fresh) process after the warm-up: solver or stand-in, the
    quantities of ``PROTOCOLS[protocol]`` in their order. The solver's kernel scales come
    from the parent (``scales``, comma separated), so no solve runs before the timed
    section."""
    warm_up()
    if which == "solver":
        make, lmax = SIZES[size]
        sc = [float(v) for v in scales.split(",")] if scales else None
        fns, args = solver_functions(make(), lmax, scales=sc)
    else:
        fns, args, _ = standin_functions(*STANDIN[size])
    out = {}
    for q in PROTOCOLS[protocol]:
        t0 = time.perf_counter()
        jax.jit(fns[q]).lower(*args).compile()
        out[q] = time.perf_counter() - t0
    return out


def child_one_thread(size: str) -> dict:
    """The run times of the solver in a process with one XLA thread."""
    warm_up()
    make, lmax = SIZES[size]
    fns, args = solver_functions(make(), lmax)
    return {q: time_compiled(fns[q], args)[0] for q in QUANTITIES}


def run_child(mode: str, *args, env=None) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", mode, *args]
    out = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def compile_times(sizes, repeats: int = 3) -> dict:
    """Median over ``repeats`` fresh processes of the compile times, solver and
    stand-in, under both protocols (``PROTOCOLS``). The key ``<size>_<which>`` holds the
    sequential protocol (the record of chunk 2), ``<size>_<which>_first`` the other."""
    out = {}
    for size in sizes:
        make, lmax = SIZES[size]
        scales = ",".join(repr(v) for v in kernel_scales(make(), lmax))
        for which in ("solver", "standin"):
            for protocol in ("sequential", "first"):
                runs = [run_child("compile", which, size, protocol, scales) for _ in range(repeats)]
                key = f"{size}_{which}" + ("" if protocol == "sequential" else "_first")
                out[key] = {q: statistics.median(r[q] for r in runs) for q in PROTOCOLS[protocol]}
                out[key]["runs"] = runs
    return out


# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="S1 only, 3 rounds")
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--json", default=str(REPO / "NOTES/benchmarks/c2-forward.json"))
    p.add_argument("--md", default=str(REPO / "NOTES/benchmarks.md"))
    p.add_argument("--child", nargs="+", default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.child:
        mode = a.child[0]
        res = child_compile(*a.child[1:]) if mode == "compile" else child_one_thread(a.child[1])
        print(json.dumps(res))
        return
    rounds = 3 if a.quick else a.rounds
    sizes = ["S1"] if a.quick else ["S1", "S2", "S3"]
    t0 = time.time()
    rec = dict(host=platform.node(), machine=platform.machine(),
               date=datetime.datetime.now().isoformat(timespec="seconds"), jax=jax.__version__,
               numpy=np.__version__, cpu_count=os.cpu_count(), xla_flags=os.environ.get("XLA_FLAGS", ""),
               loadavg_start=list(os.getloadavg()), rounds=rounds)
    warm_up()
    rec["sizes"] = {}
    for size in sizes + ([] if a.quick else ["S4"]):
        make, lmax = SIZES[size]
        p0 = make()
        if p0 is None:
            continue
        fns, x0 = solver_functions(p0, lmax)
        plan = propagate.static_plan(p0.structure, sj.StokesConfig(lmax=lmax))
        entry = dict(n_nodes=p0.structure.n_nodes, lmax=lmax, pade=plan.pade.__dict__)
        for q in QUANTITIES:
            entry[q], comp = time_compiled(fns[q], x0)
        chains = [tuple(jnp.asarray(v) for v in x0)]
        rng = np.random.default_rng(3)
        X = (jnp.stack([x0[0] + 0.1 * rng.standard_normal(x0[0].shape) for _ in range(4)]),
             jnp.stack([x0[1]] * 4), jnp.stack([x0[2]] * 4))
        entry["vmap_gradient_4_chains"], _ = time_compiled(jax.vmap(fns["gradient_log_eta"]), X)
        del chains
        if size != "S4":
            entry["stages"] = stage_split(p0, lmax)
        rec["sizes"][size] = entry
        print(size, json.dumps({k: v for k, v in entry.items() if k != "stages"}, default=str), flush=True)
    rec["gate"] = {size: gate(size, rounds) for size in sizes}
    for size, g in rec["gate"].items():
        print("gate", size, json.dumps(g["ratio"]), flush=True)
    rec["options"] = {size: options(size, rounds) for size in (["S1"] if a.quick else ["S1", "S2"])}
    rec["compile"] = compile_times(sizes, repeats=1 if a.quick else 3)
    env = dict(os.environ, XLA_FLAGS=ONE_THREAD_FLAGS)
    rec["one_thread"] = {size: run_child("one_thread", size, env=env) for size in sizes}
    rec["loadavg_end"] = list(os.getloadavg())
    rec["wall_s"] = time.time() - t0
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(rec, indent=1, default=str) + "\n")
    report(rec)
    if not a.quick:
        append_md(rec, Path(a.md), Path(a.json).name)


def report(rec):
    """Print the gate table."""
    print(f"load average start {rec['loadavg_start']}, end {rec['loadavg_end']}")
    for size, g in rec["gate"].items():
        for q in QUANTITIES:
            s, t = g[f"solver_{q}"]["median_ms"], g[f"standin_{q}"]["median_ms"]
            print(f"{size} {q:17s} solver {s:8.2f} ms  stand-in {t:8.2f} ms  ratio {s / t:5.2f}  "
                  f"budget {BUDGET[size][q]:6.1f} ms")
        c_s, c_t = rec["compile"][f"{size}_solver"], rec["compile"][f"{size}_standin"]
        for q in QUANTITIES:
            print(f"{size} compile {q:17s} solver {c_s[q]:.3f} s  stand-in {c_t[q]:.3f} s  ratio {c_s[q] / c_t[q]:.2f}")
        f_s, f_t = rec["compile"][f"{size}_solver_first"], rec["compile"][f"{size}_standin_first"]
        for q in PROTOCOLS["first"]:
            print(f"{size} compile (first) {q:17s} solver {f_s[q]:.3f} s  stand-in {f_t[q]:.3f} s")
    for size, o in rec["options"].items():
        for q in QUANTITIES:
            print(f"{size} options {q:17s} " + "  ".join(
                f"{n} {o[f'{n}_{q}']['median_ms']:.2f}" for n in ("lapack", "pivot_elimination", "block_products")))
        print(f"{size} options agreement {o['agreement_e_l']}")
    for size, o in rec["one_thread"].items():
        print(f"{size} one thread " + "  ".join(f"{q} {o[q]['run_ms']:.2f} ms" for q in QUANTITIES))


def append_md(rec, md: Path, json_name: str):
    """One row per size in NOTES/benchmarks.md (interleaved medians, compile medians)."""
    head = ("\n## bench_forward.py (chunk 2 solver)\n\n"
            "| date | host | JAX | size | load start/end | forward ms (stand-in) | gradient log_eta ms (stand-in) | "
            "gradient all ms (stand-in) | compile fwd / grad log_eta / grad all s (stand-in) | JSON |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    text = md.read_text() if md.exists() else "# Benchmarks\n"
    if "## bench_forward.py" not in text:
        text += head
    for size, g in rec["gate"].items():
        c_s, c_t = rec["compile"][f"{size}_solver"], rec["compile"][f"{size}_standin"]
        cells = [f"{g[f'solver_{q}']['median_ms']:.1f} ({g[f'standin_{q}']['median_ms']:.1f})" for q in QUANTITIES]
        comp = " / ".join(f"{c_s[q]:.2f}" for q in QUANTITIES) + " (" + " / ".join(
            f"{c_t[q]:.2f}" for q in QUANTITIES) + ")"
        text += (f"| {rec['date']} | {rec['host']} | {rec['jax']} | {size} | {g['loadavg_start'][0]:.1f}/"
                 f"{g['loadavg_end'][0]:.1f} | " + " | ".join(cells) + f" | {comp} | {json_name} |\n")
    md.write_text(text)


if __name__ == "__main__":
    main()
