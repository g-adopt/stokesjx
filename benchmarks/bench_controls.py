"""Timing of the control map (budget gate G11 of NOTES/chunks/c4-controls-derivatives/01-plan.md,
section 7).

Run from the repository root:

    python benchmarks/bench_controls.py [--quick] [--rounds 15] [--json PATH] [--md PATH] [--overwrite]

Sizes. S1: the 129-node golden table (``bench_forward.golden_profile``) with the hager4
boundaries at 100, 400 and 670 km inserted as pairs (135 nodes), the 670 km boundary
movable in (5466.875, 5970) km, ``lmax = 50``. S2: the 257-node refined LEBEDEV table
(``bench_forward.lebedev_refined``) with the same boundaries (263 nodes), ``lmax = 100``.
The layer viscosities are hager4 (1e20, 1e19, 3.33e17, 1e20 Pa s from the CMB up).

Functionals. The kernel functional of the tests (P1 of the plan): a fixed random weighted
sum of the four kernels, each divided by its largest value at the base point (the scales
come from the parent process, so a compile child runs no solve before its timed section).
The field configuration: the geoid misfit ``fields.spectral_misfit`` of
``layers.fields(K, density)`` with a fixed random ``real_4pi`` density on the input table
against a fixed observed field; the density and the observed field are arguments.

Quantities (15 interleaved rounds, medians, load average recorded):

- ``direct``: ``value_and_grad`` with respect to the profile leaves ``log eta`` (per node),
  ``rho0`` and the radius of the ``Layers`` construction profile (same structure, same
  Pade plan): the chunk 3 "gradient all" on this table;
- ``direct_construction``: the same on a structure without limits (the construction Pade
  plan); recorded, no gate: the cost of the widened plan;
- ``layers``: ``ControlVector.value_and_grad`` of the kernel functional with respect to
  ``(movable radius, log_eta)``;
- ``layers_field``: the same with the geoid field misfit through ``layers.fields``;
- ``layers_vmap4``, ``layers_vmap16``: ``vmap`` of ``layers`` over 4 and 16 vectors;
- ``numpy``: ``numpy_value_and_grad`` of ``layers``; its overhead is its time minus the
  time of the JAX call of the same compiled function;
- compile of ``layers``, ``layers_field``, ``layers_vmap4`` and ``direct``: protocol
  "first" of CONVENTIONS 7 (a fresh process, a warm-up compile that lowers a tiny LAPACK
  solve, no solve of the package, the forward compile, then the gradient compile), three
  processes each, the median.

Gates (section 7 of the plan):

- B1: ``layers`` run time at most 1.10 x ``direct`` at S1 and S2, and inside the budgets of
  CONVENTIONS 7 for "gradient run, all controls": 48 ms (S1), 181 ms (S2);
- B2: ``layers`` gradient compile at most 1.10 x ``direct`` gradient compile at S1 and S2;
- B3: ``layers_vmap4`` at most 4.0 x ``layers``;
- B4: ``numpy`` overhead at most 0.5 ms per call at S1;
- B5: ``layers_field`` minus ``layers`` at most + 2 ms (S1) and + 6 ms (S2) run time and
  + 0.10 s gradient compile.

Recorded without a gate: ``direct_construction``, ``layers`` with one XLA thread, the Pade
plans, and ``layers_two`` at S1 (the 670 km boundary in (5466.875, 5850.703) km and the
400 km boundary in (5850.703, 6189.375) km, sharing the end node 5850.703 km): run and
compile once, with its plan. The script refuses to overwrite an existing JSON record unless
``--overwrite`` is given, and appends rows to ``NOTES/benchmarks.md``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO / "tests"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import bench_forward as bf  # noqa: E402
import stokesjx as sj  # noqa: E402
from stokesjx import fields, propagate  # noqa: E402

SIZES = {"S1": (bf.golden_profile, 50), "S2": (bf.lebedev_refined, 100)}
BOUNDARIES = np.array([3480e3, 5700e3, 5970e3, 6270e3, 6370e3])
MOVABLE = {1: (5466.875e3, 5970e3)}
HAGER4 = np.array([1e20, 1e19, 0.03333333333333333e19, 1e20])
OUTS = ("geoid", "surface_topography", "cmb_topography", "surface_velocity")
RATIO_LIMIT = 1.10
BUDGET = {"S1": 48.0, "S2": 181.0}
VMAP4_LIMIT = 4.0
NUMPY_OVERHEAD_MS = 0.5
FIELD_EXTRA_MS = {"S1": 2.0, "S2": 6.0}
FIELD_COMPILE_EXTRA = 0.10
PROGRAMS = ("direct", "direct_construction", "layers", "layers_field", "layers_vmap4", "layers_vmap16")
COMPILED = ("layers", "layers_field", "layers_vmap4", "direct")


def make_layers(size: str, two: bool = False):
    """``(Layers, cfg)`` of a size; ``two`` gives the two-boundary layering of ``layers_two``."""
    make, lmax = SIZES[size]
    p = make()
    if two:
        r = np.asarray(p.radius)
        mid = float(r[np.argmin(np.abs(r - 5850.703e3))])
        L = sj.Layers(p, BOUNDARIES, movable={1: (5466.875e3, mid), 2: (mid, 6189.375e3)})
    else:
        L = sj.Layers(p, BOUNDARIES, movable=MOVABLE)
    return L, sj.StokesConfig(lmax=lmax)


def kernel_scales(size: str) -> list[float]:
    """The largest value of each kernel at the construction radius (one solve, parent)."""
    L, cfg = make_layers(size)
    K = sj.stokes_kernels(L(L.boundaries, np.log(HAGER4)), cfg)
    return [float(np.max(np.abs(np.asarray(getattr(K, o))))) for o in OUTS]


def functions(size: str, scales, two: bool = False) -> dict:
    """``name -> (function, args, forward)``: the programs of the benchmark. ``function`` is
    the gradient program, ``forward`` its value alone (for the compile protocol)."""
    L, cfg = make_layers(size, two)
    lmax = cfg.lmax
    n_deg = lmax                                  # degrees 1..lmax (CM frame)
    rng = np.random.default_rng(4)
    W = [jnp.asarray(rng.standard_normal((n_deg, L.n_nodes)) / s) for s in scales]

    def F_of(p):
        K = sj.stokes_kernels(p, cfg)
        return sum(jnp.sum(w * getattr(K, o)) for w, o in zip(W, OUTS))

    p0 = L(L.boundaries, np.log(HAGER4))
    out = {}

    def direct_f(le, rho0, radius, base=p0):
        return F_of(base.replace(eta=jnp.exp(le), rho0=rho0, radius=radius))
    xd = (jnp.log(p0.eta), p0.rho0, p0.radius)
    out["direct"] = (jax.value_and_grad(direct_f, argnums=(0, 1, 2)), xd, direct_f)
    fresh = sj.RadialProfile(np.asarray(p0.radius), np.asarray(p0.rho0), np.asarray(p0.eta),
                             core_density=float(p0.core_density))

    def direct_c(le, rho0, radius):
        return direct_f(le, rho0, radius, base=fresh)
    out["direct_construction"] = (jax.value_and_grad(direct_c, argnums=(0, 1, 2)), xd, direct_c)

    R = L.movable_index.size
    cv = sj.ControlVector({"radius": L.start_radii.copy(), "log_eta": np.log(HAGER4)})

    def fn(c):
        return F_of(L(L.full_boundaries(c["radius"]), c["log_eta"]))
    v0 = cv.ravel({"radius": L.start_radii + 10e3, "log_eta": np.log(HAGER4)})
    out["layers"] = (cv.value_and_grad(fn), (v0,), cv.value(fn))
    rng_d = np.random.default_rng(5)
    tri = np.tril(np.ones((lmax + 1, lmax + 1)))
    dens = jnp.asarray(10.0 * rng_d.standard_normal((L.input_profile.structure.n_nodes, 2, lmax + 1, lmax + 1)) * tri)
    obs = jnp.asarray(1e-2 * rng_d.standard_normal((2, lmax + 1, lmax + 1)) * tri)
    wl = jnp.asarray(np.where(np.arange(lmax + 1) >= 2, 1.0, 0.0))

    def fn_field(c, d, observed):
        K = sj.stokes_kernels(L(L.full_boundaries(c["radius"]), c["log_eta"]), cfg)
        return fields.spectral_misfit(L.fields(K, d).geoid, observed, wl, "real_4pi")
    out["layers_field"] = (cv.value_and_grad(fn_field), (v0, dens, obs), cv.value(fn_field))
    for n in (4, 16):
        V = jnp.stack([v0 + jnp.asarray(np.concatenate([0.1 * rng.standard_normal(4), 5e3 * rng.standard_normal(R)]))
                       for _ in range(n)])
        out[f"layers_vmap{n}"] = (jax.vmap(cv.value_and_grad(fn)), (V,), jax.vmap(cv.value(fn)))
    out["_cv"] = (cv, fn, v0)
    out["_plans"] = dict(widened=propagate.static_plan(L.profile.structure, cfg).pade.__dict__,
                         construction=propagate.static_plan(fresh.structure, cfg).pade.__dict__,
                         width_factor=L.width_factor)
    return out


def run_times(size: str, rounds: int) -> dict:
    """All programs of a size compiled once in this process and interleaved over the rounds;
    the NumPy front against the JAX call of the same compiled function."""
    fns = functions(size, kernel_scales(size))
    groups = {}
    for name in PROGRAMS:
        f, args, _ = fns[name]
        groups[name] = (jax.jit(f).lower(*args).compile(), args)
    cv, fn, v0 = fns["_cv"]
    single = cv.value_and_grad(fn)
    single(v0)                                                     # compile the cached jit
    f_np = cv.numpy_value_and_grad(fn)
    x_np = np.asarray(v0)
    groups["layers_jax_call"] = (single, (v0,))
    groups["numpy"] = (lambda x: f_np(x), (x_np,))
    load0 = list(os.getloadavg())
    res = bf.interleaved(groups, rounds)
    res["loadavg_start"], res["loadavg_end"] = load0, list(os.getloadavg())
    res["pade"] = fns["_plans"]
    return res


def child_compile(size: str, program: str, scales: str) -> dict:
    """Protocol "first" in this fresh process: the warm-up, no solve, the forward compile,
    then the gradient compile of ``program``."""
    bf.warm_up()
    fns = functions(size, [float(v) for v in scales.split(",")])
    f, args, fwd = fns[program]
    t0 = time.perf_counter()
    jax.jit(fwd).lower(*args).compile()
    t1 = time.perf_counter()
    jax.jit(f).lower(*args).compile()
    t2 = time.perf_counter()
    return dict(forward=t1 - t0, gradient=t2 - t1)


def child_one_thread(size: str, scales: str) -> dict:
    """Run time of ``layers`` in a process with one XLA thread."""
    bf.warm_up()
    fns = functions(size, [float(v) for v in scales.split(",")])
    f, args, _ = fns["layers"]
    return bf.time_compiled(f, args)[0]


def child_two(scales: str) -> dict:
    """``layers_two`` at S1: run and compile once, with its plan."""
    bf.warm_up()
    fns = functions("S1", [float(v) for v in scales.split(",")], two=True)
    f, args, _ = fns["layers"]
    t, _ = bf.time_compiled(f, args)
    t["pade"] = fns["_plans"]
    return t


def run_child(*args, env=None) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", *args]
    out = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def compile_times(size: str, repeats: int) -> dict:
    scales = ",".join(repr(v) for v in kernel_scales(size))
    out = {}
    for prog in COMPILED:
        runs = [run_child("compile", size, prog, scales) for _ in range(repeats)]
        out[prog] = dict(forward=statistics.median(r["forward"] for r in runs),
                         gradient=statistics.median(r["gradient"] for r in runs), runs=runs)
    return out


def gates(rec) -> list[str]:
    lines, ok = [], True
    for size, r in rec["run"].items():
        lay, dir_ = r["layers"]["median_ms"], r["direct"]["median_ms"]
        good = lay <= RATIO_LIMIT * dir_ and lay <= BUDGET[size]
        ok &= good
        lines.append(f"B1 {size} layers {lay:.2f} ms, direct {dir_:.2f} ms, ratio {lay / dir_:.3f} (limit 1.10), "
                     f"budget {BUDGET[size]:.0f} ms: {'ok' if good else 'FAIL'}")
        v4 = r["layers_vmap4"]["median_ms"]
        good = v4 <= VMAP4_LIMIT * lay
        ok &= good
        lines.append(f"B3 {size} layers_vmap4 {v4:.2f} ms = {v4 / lay:.2f} x layers (limit 4.0); vmap16 "
                     f"{r['layers_vmap16']['median_ms']:.2f} ms = {r['layers_vmap16']['median_ms'] / lay:.2f} x: "
                     f"{'ok' if good else 'FAIL'}")
        ov = r["numpy"]["median_ms"] - r["layers_jax_call"]["median_ms"]
        if size == "S1":
            good = ov <= NUMPY_OVERHEAD_MS
            ok &= good
            lines.append(f"B4 {size} numpy overhead {ov:.3f} ms (limit 0.5): {'ok' if good else 'FAIL'}")
        else:
            lines.append(f"B4 {size} numpy overhead {ov:.3f} ms (recorded)")
        extra = r["layers_field"]["median_ms"] - lay
        good = extra <= FIELD_EXTRA_MS[size]
        ok &= good
        lines.append(f"B5 {size} layers_field - layers = {extra:+.2f} ms (limit +{FIELD_EXTRA_MS[size]}): "
                     f"{'ok' if good else 'FAIL'}")
        lines.append(f"   {size} direct_construction {r['direct_construction']['median_ms']:.2f} ms (recorded; "
                     f"plans {r['pade']['construction']} against {r['pade']['widened']})")
    for size, c in rec["compile"].items():
        ratio = c["layers"]["gradient"] / c["direct"]["gradient"]
        good = ratio <= RATIO_LIMIT
        ok &= good
        lines.append(f"B2 {size} compile layers {c['layers']['gradient']:.3f} s, direct {c['direct']['gradient']:.3f} s, "
                     f"ratio {ratio:.3f} (limit 1.10): {'ok' if good else 'FAIL'}")
        extra = c["layers_field"]["gradient"] - c["layers"]["gradient"]
        good = extra <= FIELD_COMPILE_EXTRA
        ok &= good
        lines.append(f"B5 {size} compile layers_field - layers = {extra:+.3f} s (limit +0.10): {'ok' if good else 'FAIL'}")
    lines.append("G11 " + ("holds" if ok else "FAILS"))
    return lines


def append_md(rec, md: Path, json_name: str):
    head = ("\n## bench_controls.py (chunk 4: control map)\n\n"
            "| date | host | JAX | size | load start/end | direct ms | direct_construction ms | layers ms | "
            "layers_field ms | vmap4 ms | vmap16 ms | numpy overhead ms | compile direct / layers / layers_field / "
            "vmap4 s (gradient, first) | JSON |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    text = md.read_text() if md.exists() else "# Benchmarks\n"
    if "## bench_controls.py" not in text:
        text += head
    for size, r in rec["run"].items():
        c = rec["compile"].get(size)
        ctxt = (f"{c['direct']['gradient']:.2f} / {c['layers']['gradient']:.2f} / {c['layers_field']['gradient']:.2f} / "
                f"{c['layers_vmap4']['gradient']:.2f}") if c else ""
        ov = r["numpy"]["median_ms"] - r["layers_jax_call"]["median_ms"]
        text += (f"| {rec['date']} | {rec['host']} | {rec['jax']} | {size} | {r['loadavg_start'][0]:.1f}/"
                 f"{r['loadavg_end'][0]:.1f} | {r['direct']['median_ms']:.2f} | {r['direct_construction']['median_ms']:.2f} | "
                 f"{r['layers']['median_ms']:.2f} | {r['layers_field']['median_ms']:.2f} | "
                 f"{r['layers_vmap4']['median_ms']:.2f} | {r['layers_vmap16']['median_ms']:.2f} | {ov:.3f} | {ctxt} | "
                 f"{json_name} |\n")
    if "one_thread" in rec:
        text += "\n" + "; ".join(f"{s} layers with one XLA thread {v['run_ms']:.2f} ms" for s, v in rec["one_thread"].items())
        text += f" ({rec['date']}).\n"
    if "layers_two" in rec:
        t = rec["layers_two"]
        text += (f"\nS1 layers_two ({rec['date']}): two movable boundaries, run {t['run_ms']:.2f} ms, compile "
                 f"{t['compile_s']:.2f} s, plan {t['pade']['widened']}.\n")
    md.write_text(text)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="S1 only, 3 rounds, one compile process")
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--json", default=str(REPO / "NOTES" / "benchmarks" / "c4-controls.json"))
    p.add_argument("--md", default=str(REPO / "NOTES" / "benchmarks.md"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--child", nargs="+")
    a = p.parse_args(argv)
    if a.child:
        kind = a.child[0]
        if kind == "compile":
            res = child_compile(*a.child[1:])
        elif kind == "one_thread":
            res = child_one_thread(*a.child[1:])
        elif kind == "two":
            res = child_two(*a.child[1:])
        else:
            raise SystemExit(kind)
        print(json.dumps(res))
        return
    path = Path(a.json)
    if path.exists() and not a.overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    sizes = ["S1"] if a.quick else ["S1", "S2"]
    rounds = 3 if a.quick else a.rounds
    repeats = 1 if a.quick else 3
    rec = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), host=platform.node(), jax=jax.__version__,
               run={}, compile={}, one_thread={})
    for size in sizes:
        rec["run"][size] = run_times(size, rounds)
        print(f"{size} run: " + ", ".join(f"{k} {v['median_ms']:.2f}" for k, v in rec["run"][size].items()
                                        if isinstance(v, dict) and "median_ms" in v), flush=True)
    for size in sizes:
        rec["compile"][size] = compile_times(size, repeats)
        print(f"{size} compile: " + ", ".join(f"{k} {v['forward']:.3f}/{v['gradient']:.3f}"
                                            for k, v in rec["compile"][size].items()), flush=True)
    env = dict(os.environ, XLA_FLAGS=bf.ONE_THREAD_FLAGS)
    for size in sizes:
        rec["one_thread"][size] = run_child("one_thread", size, ",".join(repr(v) for v in kernel_scales(size)), env=env)
    rec["layers_two"] = run_child("two", ",".join(repr(v) for v in kernel_scales("S1")))
    rec["gates"] = gates(rec)
    for line in rec["gates"]:
        print(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, default=str))
    append_md(rec, Path(a.md), path.name)
    print(f"written {path}")


if __name__ == "__main__":
    main()
