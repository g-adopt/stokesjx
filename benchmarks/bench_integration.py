"""The performance table of stokesjx at lmax 50, 100 and 200 (forward map, fields, HMC
gradient, gradient of all profile leaves, vmap batches, compile time, memory), with the
budgets of NOTES/chunks/c5-integration/01-plan.md section 4.

Run from the repository root:

    python benchmarks/bench_integration.py [--sizes P50 P100 P200 P200c] [--rounds 7]
        [--calls 20] [--json NOTES/benchmarks/c5-integration.json] [--md NOTES/benchmarks.md]
        [--table NOTES/chunks/c5-integration/benchmark-table.md] [--no-compile]

Sizes. The node table grows with ``lmax``, so that the benchmark measures the problem
that a user runs at each degree: the radial scale of a kernel near the surface is about
``R / l`` (32 km at ``l = 200``), which the 22.6 km spacing of the 137-node table does not
resolve.

    P50    examples/earth_model.prem_profile(1), 137 nodes, lmax 50
    P100   prem_profile(2), 265 nodes (every LEBEDEV interval halved), lmax 100
    P200   prem_profile(4), 521 nodes (every interval divided by four), lmax 200
    P200c  prem_profile(1), 137 nodes, lmax 200 (recorded only: its Pade plan has two
           squarings)

Every size uses the layering of the HMC example (``examples/hmc_layers.py``): boundaries
at 3480, 5750, 5970, 6270 and 6370 km, boundary 1 movable in (5466.875, 5970) km across
the 670 km density jump, so ``Layers`` resolves ``motion="auto"`` to the viscosity cut.

Programs (each one ``jax.jit``):

``forward``
    The control vector ``z = (log eta of the four layers, logistic coordinate of the
    radius)`` to the four kernels.
``fields``
    ``z`` and a density on the input table to the four fields (``Layers.fields``).
``gradient``
    ``jax.value_and_grad`` of the potential ``U(z)`` of the HMC example (geoid misfit and
    prior; ``hmc_layers.potential``), five controls.
``gradient_all``
    The value and the gradient of the same geoid misfit with respect to the profile
    leaves ``log eta`` per node, ``rho0``, the radius and ``cut_radius`` (the ``direct``
    program of ``bench_controls.py``).
``vmap4``, ``vmap16``
    ``jax.vmap`` of ``gradient`` over 4 and 16 control vectors (P200 and P200c: ``vmap4``
    only).
``numpy``
    ``ControlVector.numpy_value_and_grad`` of the potential minus the JAX call of the same
    compiled function (P50 only): the overhead of the NumPy front.
``one_thread``
    ``gradient`` in a child process with one XLA thread (recorded only).

Run time: the programs are compiled once; then ``--rounds`` interleaved rounds, each
program one warm-up call and ``--calls`` timed calls per round. Recorded: the median and
the minimum over all calls per program, and for ``vmap4 / gradient`` the median of the
per-round ratios beside the ratio of the medians. The load average at the start and the
end of each size.

Compile time: protocol "first" of CONVENTIONS 7: a fresh child process per program and
size, one unrelated warm-up compile that lowers a tiny LAPACK solve, then the lowering and
compile of the program with the trace of the solver inside the timed section (no solve
runs before it: the data are zeros). ``forward`` and ``gradient``, three children each,
alternated; median, with the load average of each child.

Memory: the peak resident set (``resource.getrusage(RUSAGE_SELF).ru_maxrss``, converted
to bytes: the unit is bytes on macOS and kilobytes on Linux) of a fresh child that
compiles and runs ``gradient`` once, and of one that runs ``vmap4``.

Budgets (plan 4.4; checked and printed, recorded in the JSON):

    size   forward  fields  gradient  gradient_all  vmap4/gradient  compile fwd  compile grad  RSS grad
    P50    17 ms    19 ms   48 ms     48 ms         <= 4.0          <= 0.6 s     <= 1.2 s      <= 2 GB
    P100   67 ms    73 ms   181 ms    181 ms        <= 4.0          <= 0.6 s     <= 1.2 s      <= 4 GB
    P200   205 ms   225 ms  560 ms    560 ms        recorded        <= 0.6 s     <= 1.2 s      <= 8 GB

and the scaling P200 / P100 at most 4.5 for ``forward`` and ``gradient`` (4 for the work,
1.125 for memory traffic), the NumPy overhead at most 0.5 ms. P200c is recorded only.

Output: the JSON (every number, the Pade plan of each size, the host, the JAX version and
the load averages), one markdown row per size appended to ``--md`` under the heading
"bench_integration.py (chunk 5)", and the README block (the table with its date, host, JAX
version and load line) written to ``--table``.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import stokesjx as sj

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
KM = 1e3

#: ``size -> (refine of prem_profile, lmax)``.
SIZES = {"P50": (1, 50), "P100": (2, 100), "P200": (4, 200), "P200c": (1, 200)}
#: The run-time budgets in ms (plan 4.4).
BUDGET_MS = {
    "P50": dict(forward=17.0, fields=19.0, gradient=48.0, gradient_all=48.0),
    "P100": dict(forward=67.0, fields=73.0, gradient=181.0, gradient_all=181.0),
    "P200": dict(forward=205.0, fields=225.0, gradient=560.0, gradient_all=560.0),
}
#: The compile budgets in s (protocol "first").
COMPILE_BUDGET_S = dict(forward=0.6, gradient=1.2)
#: The peak resident set of the gradient child, bytes.
RSS_BUDGET = {"P50": 2e9, "P100": 4e9, "P200": 8e9}
#: ``vmap4`` over one ``gradient`` call (P50 and P100).
VMAP4_RATIO_MAX = 4.0
#: P200 over P100 for ``forward`` and ``gradient``.
SCALING_MAX = 4.5
#: The overhead of the NumPy front, ms (P50).
NUMPY_OVERHEAD_MAX_MS = 0.5
#: One XLA thread for the ``one_thread`` child.
ONE_THREAD_FLAGS = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

BOUNDARIES = np.array([3480.0, 5750.0, 5970.0, 6270.0, 6370.0]) * KM
RANGE = (5466.875 * KM, 5970.0 * KM)
Y_TRUE = np.log([3e22, 1e21, 3e20, 1e22])
R_TRUE = 5690.0 * KM
OUTS = ("geoid", "surface_topography", "cmb_topography", "surface_velocity")


def _example(name: str):
    """``examples/<name>.py`` under the module name of the examples (no ``sys.path``
    change)."""
    mod_name = f"stokesjx_examples_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def rss_bytes() -> int:
    """The peak resident set of this process in bytes (``ru_maxrss`` is in bytes on macOS
    and in kilobytes on Linux)."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(v if sys.platform == "darwin" else v * 1024)


def warm_up():
    """One unrelated compile that also lowers a tiny LAPACK solve (CONVENTIONS 7): the
    first LAPACK lowering of a process imports ``scipy.linalg``, a start-up cost of the
    process and not a compile cost of a program."""
    jax.jit(lambda a: jnp.linalg.solve(a, jnp.sin(a)[:, :1])).lower(jnp.eye(3)).compile()


def programs(size: str) -> dict:
    """``name -> (function, args)`` of the programs of a size (module docstring), and the
    entries ``_info`` (nodes, lmax, Pade plan, motion) and ``_cv`` (the ``ControlVector``
    front of ``numpy``). No solve runs here: the data are zeros and the weights ones, so
    a child can time a compile that includes the trace of the solver."""
    em = _example("earth_model")
    hmc = _example("hmc_layers")
    refine, lmax = SIZES[size]
    table = em.prem_profile(refine)
    L = sj.Layers(table, BOUNDARIES, movable={1: RANGE})
    cfg = sj.StokesConfig(lmax=lmax)
    lo, hi = (float(v) for v in L.radius_bounds[0])
    # The attributes of the example's Problem that ``hmc_layers.potential`` reads.
    P = SimpleNamespace(layers=L, config=cfg, m_true=float(np.mean(Y_TRUE)), lo=lo, hi=hi)
    density = jnp.asarray(em.smooth_density(np.asarray(table.radius), lmax, 7))
    data = jnp.zeros((2, lmax + 1, lmax + 1))
    weights = jnp.asarray(np.where(np.arange(lmax + 1) >= 2, 1.0, 0.0))
    z0 = jnp.asarray(np.concatenate([Y_TRUE, [hmc.logit((R_TRUE - lo) / (hi - lo))]]))

    def forward(z):
        c = hmc.to_controls(z, P)
        K = sj.stokes_kernels(L(L.full_boundaries(c["radius"]), c["log_eta"]), cfg)
        return tuple(getattr(K, o) for o in OUTS)

    def fields_fn(z, d):
        c = hmc.to_controls(z, P)
        K = sj.stokes_kernels(L(L.full_boundaries(c["radius"]), c["log_eta"]), cfg)
        F = L.fields(K, d)
        return tuple(getattr(F, o) for o in OUTS)

    def U(z, d, o, w):
        return hmc.potential(z, d, o, w, P)
    gradient = jax.value_and_grad(U)

    p0 = L(L.full_boundaries(jnp.asarray([R_TRUE])), jnp.asarray(Y_TRUE))

    def direct(le, rho0, radius, cut, d, o, w):
        p = p0.replace(eta=jnp.exp(le), rho0=rho0, radius=radius, cut_radius=cut)
        N = L.fields(sj.stokes_kernels(p, cfg), d).geoid
        return sj.spectral_misfit(N, o, w, "real_4pi")
    gradient_all = jax.value_and_grad(direct, argnums=(0, 1, 2, 3))
    leaves = (jnp.log(p0.eta), p0.rho0, p0.radius, p0.cut_radius)

    rng = np.random.default_rng(4)
    out = {
        "forward": (forward, (z0,)),
        "fields": (fields_fn, (z0, density)),
        "gradient": (gradient, (z0, density, data, weights)),
        "gradient_all": (gradient_all, leaves + (density, data, weights)),
    }
    for n in ((4,) if size in ("P200", "P200c") else (4, 16)):
        Z = z0[None, :] + jnp.asarray(0.05 * rng.standard_normal((n, 5)))
        out[f"vmap{n}"] = (jax.vmap(gradient, in_axes=(0, None, None, None)), (Z, density, data, weights))
    cv = sj.ControlVector({"z": np.asarray(z0)})

    def fn(c, d, o, w):
        return U(c["z"], d, o, w)
    out["_cv"] = (cv, fn, (density, data, weights), np.asarray(z0))
    plan = L.pade_plan(cfg)
    out["_info"] = dict(nodes=int(L.n_nodes), lmax=lmax, motion=L.motion,
                        pade=dict(order=int(plan.order), squarings=int(plan.squarings)))
    return out


def _timed_calls(compiled, args, calls: int) -> list[float]:
    """One warm-up call, then ``calls`` timed calls; seconds each."""
    jax.block_until_ready(compiled(*args))
    ts = []
    for _ in range(calls):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        ts.append(time.perf_counter() - t0)
    return ts


def run_times(size: str, rounds: int, calls: int) -> dict:
    """The run times of every program of a size (module docstring): median and minimum in
    ms, the per-round medians, and for P50 the NumPy overhead."""
    progs = programs(size)
    names = [k for k in progs if not k.startswith("_")]
    compiled = {k: jax.jit(progs[k][0]).lower(*progs[k][1]).compile() for k in names}
    numpy_front = None
    if size == "P50":
        cv, fn, data, z0 = progs["_cv"]
        nf = cv.numpy_value_and_grad(fn, *data)
        jf = cv.value_and_grad(fn)
        jd = tuple(jnp.asarray(a) for a in data)
        zj = jnp.asarray(z0)
        numpy_front = (nf, jf, jd, zj, z0)
        nf(z0)
    per_round = {k: [] for k in names}
    all_calls = {k: [] for k in names}
    np_calls, jx_calls = [], []
    load_start = os.getloadavg()
    for _ in range(rounds):
        for k in names:
            ts = _timed_calls(compiled[k], progs[k][1], calls)
            per_round[k].append(statistics.median(ts))
            all_calls[k] += ts
        if numpy_front is not None:
            nf, jf, jd, zj, z0 = numpy_front
            for _ in range(calls):
                t0 = time.perf_counter()
                nf(z0)
                np_calls.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                v, g = jf(zj, *jd)
                np.asarray(g)
                float(v)
                jx_calls.append(time.perf_counter() - t0)
    out = {"info": progs["_info"], "load_start": load_start, "load_end": os.getloadavg(), "programs": {}}
    for k in names:
        out["programs"][k] = dict(run_ms=1e3 * statistics.median(all_calls[k]), run_min_ms=1e3 * min(all_calls[k]),
                                  round_ms=[1e3 * v for v in per_round[k]])
    g = per_round["gradient"]
    for n in (4, 16):
        key = f"vmap{n}"
        if key in per_round:
            out[f"{key}_ratio"] = dict(
                median_of_round_ratios=statistics.median(a / b for a, b in zip(per_round[key], g)),
                ratio_of_medians=out["programs"][key]["run_ms"] / out["programs"]["gradient"]["run_ms"])
    if np_calls:
        out["numpy"] = dict(numpy_ms=1e3 * statistics.median(np_calls), jax_ms=1e3 * statistics.median(jx_calls),
                            overhead_ms=1e3 * (statistics.median(np_calls) - statistics.median(jx_calls)))
    return out


# ---------------------------------------------------------------------------
# Child processes.
# ---------------------------------------------------------------------------


def child(mode: str, size: str, name: str = "") -> dict:
    """The work of one fresh child: ``compile`` (the lowering and compile of one program
    after the warm-up), ``memory`` (compile and run one program, then the peak resident
    set) or ``one_thread`` (the run time of ``gradient``)."""
    warm_up()
    load = os.getloadavg()
    progs = programs(size)
    if mode == "compile":
        fn, args = progs[name]
        t0 = time.perf_counter()
        jax.jit(fn).lower(*args).compile()
        return dict(compile_s=time.perf_counter() - t0, load=load)
    if mode == "memory":
        fn, args = progs[name]
        jax.block_until_ready(jax.jit(fn)(*args))
        return dict(rss_bytes=rss_bytes(), load=load)
    if mode == "one_thread":
        fn, args = progs["gradient"]
        comp = jax.jit(fn).lower(*args).compile()
        ts = _timed_calls(comp, args, 10)
        return dict(run_ms=1e3 * statistics.median(ts), run_min_ms=1e3 * min(ts), load=load)
    raise ValueError(mode)


def run_child(*args, env=None) -> dict:
    """Run ``child(*args)`` in a fresh process and return its JSON record."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", *args]
    proc = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def compile_times(size: str, repeats: int = 3) -> dict:
    """Protocol "first": ``repeats`` fresh children per program, ``forward`` and
    ``gradient`` alternated; the median and every run with its load."""
    runs = {"forward": [], "gradient": []}
    for _ in range(repeats):
        for name in ("forward", "gradient"):
            runs[name].append(run_child("compile", size, name))
    return {name: dict(median_s=statistics.median(r["compile_s"] for r in rs), runs=rs) for name, rs in runs.items()}


def memory(size: str) -> dict:
    """The peak resident set of a child that runs ``gradient``, and of one that runs
    ``vmap4``, bytes."""
    return {name: run_child("memory", size, name) for name in ("gradient", "vmap4")}


def one_thread(size: str) -> dict:
    """``gradient`` in a child with one XLA thread."""
    env = dict(os.environ)
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " " + ONE_THREAD_FLAGS).strip()
    return run_child("one_thread", size, env=env)


# ---------------------------------------------------------------------------
# Gates and reports.
# ---------------------------------------------------------------------------


def gates(rec: dict) -> list[dict]:
    """The budgets of the module docstring on a record; each entry ``name, value, limit,
    holds``."""
    out = []

    def check(name, value, limit):
        out.append(dict(name=name, value=value, limit=limit, holds=bool(value <= limit)))
    sizes = rec["sizes"]
    for size, budget in BUDGET_MS.items():
        if size not in sizes:
            continue
        s = sizes[size]
        for prog, limit in budget.items():
            check(f"{size} {prog} ms", s["run"]["programs"][prog]["run_ms"], limit)
        if size in ("P50", "P100"):
            check(f"{size} vmap4 / gradient (median of round ratios)",
                  s["run"]["vmap4_ratio"]["median_of_round_ratios"], VMAP4_RATIO_MAX)
        if "compile" in s:
            for prog, limit in COMPILE_BUDGET_S.items():
                check(f"{size} compile {prog} s", s["compile"][prog]["median_s"], limit)
        if "memory" in s:
            check(f"{size} peak RSS gradient child bytes", s["memory"]["gradient"]["rss_bytes"], RSS_BUDGET[size])
    if "P100" in sizes and "P200" in sizes:
        for prog in ("forward", "gradient"):
            check(f"P200 / P100 {prog}", sizes["P200"]["run"]["programs"][prog]["run_ms"]
                  / sizes["P100"]["run"]["programs"][prog]["run_ms"], SCALING_MAX)
    if "P50" in sizes and "numpy" in sizes["P50"]["run"]:
        check("P50 numpy overhead ms", sizes["P50"]["run"]["numpy"]["overhead_ms"], NUMPY_OVERHEAD_MAX_MS)
    return out


def _row(size: str, s: dict) -> list[str]:
    """One table row of a size: the run times, the vmap ratio, the compile times, RSS."""
    p = s["run"]["programs"]
    info = s["run"]["info"]
    vr = s["run"].get("vmap4_ratio", {}).get("median_of_round_ratios")
    comp = s.get("compile", {})
    mem = s.get("memory", {})
    f = lambda k: f"{p[k]['run_ms']:.1f}"
    return [size, str(info["nodes"]), str(info["lmax"]), f"{info['pade']['order']}/{info['pade']['squarings']}",
            f("forward"), f("fields"), f("gradient"), f("gradient_all"),
            f"{vr:.2f}" if vr is not None else "-",
            f"{comp['forward']['median_s']:.2f}" if comp else "-",
            f"{comp['gradient']['median_s']:.2f}" if comp else "-",
            f"{mem['gradient']['rss_bytes'] / 1e9:.2f}" if mem else "-",
            f"{mem['vmap4']['rss_bytes'] / 1e9:.2f}" if mem else "-"]


HEADER = ["size", "nodes", "lmax", "Pade order/squarings", "forward ms", "fields ms", "gradient ms",
          "gradient_all ms", "vmap4 / gradient", "compile forward s", "compile gradient s", "RSS gradient GB",
          "RSS vmap4 GB"]


def table_block(rec: dict) -> str:
    """The README block: the table and one line with the date, host, JAX version and the
    load averages."""
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    for size, s in rec["sizes"].items():
        lines.append("| " + " | ".join(_row(size, s)) + " |")
    loads = [v for s in rec["sizes"].values() for v in (s["run"]["load_start"][0], s["run"]["load_end"][0])]
    lines.append("")
    lines.append(f"Measured {rec['date']} on {rec['host']} ({rec['machine']}), JAX {rec['jax']}, "
                 f"`benchmarks/bench_integration.py`; 1-minute load average {min(loads):.1f} to {max(loads):.1f}. "
                 "Run time: median of the calls of all rounds. Compile: protocol \"first\", median of three fresh "
                 "processes. RSS: peak resident set of a fresh process that compiles and runs the program once.")
    return "\n".join(lines) + "\n"


def append_md(rec: dict, md: Path) -> None:
    """Append the rows of a record to ``md`` under the heading of this script."""
    head = "## bench_integration.py (chunk 5)"
    text = md.read_text() if md.exists() else ""
    add = []
    if head not in text:
        add += ["", head, "", "Sizes, programs and budgets: the docstring of benchmarks/bench_integration.py.", "",
                "| date | load | " + " | ".join(HEADER) + " | gates |", "|" + "---|" * (len(HEADER) + 3)]
    ok = all(g["holds"] for g in rec["gates"])
    for size, s in rec["sizes"].items():
        load = f"{s['run']['load_start'][0]:.1f}-{s['run']['load_end'][0]:.1f}"
        add.append(f"| {rec['date']} | {load} | " + " | ".join(_row(size, s)) + f" | {'hold' if ok else 'FAIL'} |")
    md.write_text(text.rstrip("\n") + "\n" + "\n".join(add) + "\n")


def main(argv=None):
    """Parse the command line, measure, check the budgets, write the outputs."""
    p = argparse.ArgumentParser(description="The stokesjx performance table (chunk 5).")
    p.add_argument("--sizes", nargs="+", default=["P50", "P100", "P200", "P200c"], choices=list(SIZES))
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--json", type=Path, default=REPO / "NOTES" / "benchmarks" / "c5-integration.json")
    p.add_argument("--md", type=Path, default=REPO / "NOTES" / "benchmarks.md")
    p.add_argument("--table", type=Path, default=REPO / "NOTES" / "chunks" / "c5-integration" / "benchmark-table.md")
    p.add_argument("--no-compile", action="store_true", help="skip the compile, memory and one-thread children")
    p.add_argument("--child", nargs="+", default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.child:
        print(json.dumps(child(*a.child)))
        return
    rec = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), host=platform.node(),
               machine=platform.machine(), jax=jax.__version__, rounds=a.rounds, calls=a.calls, sizes={})
    for size in a.sizes:
        print(f"{size}: run times ({a.rounds} rounds of {a.calls} calls)", flush=True)
        s = {"run": run_times(size, a.rounds, a.calls)}
        if not a.no_compile:
            print(f"{size}: compile, memory, one thread", flush=True)
            s["compile"] = compile_times(size)
            s["memory"] = memory(size)
            s["one_thread"] = one_thread(size)
        rec["sizes"][size] = s
        print("  " + " | ".join(_row(size, s)), flush=True)
    rec["gates"] = gates(rec)
    for g in rec["gates"]:
        print(f"  {'hold' if g['holds'] else 'FAIL'}: {g['name']} {g['value']:.4g} (limit {g['limit']:.4g})")
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(rec, indent=1))
    append_md(rec, a.md)
    a.table.parent.mkdir(parents=True, exist_ok=True)
    a.table.write_text(table_block(rec))
    print(f"wrote {a.json}, {a.md}, {a.table}")


if __name__ == "__main__":
    main()
