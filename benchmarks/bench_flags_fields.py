"""Timing of the physics flags, the fields and the spherical-harmonic transforms
(budget gate G9 of NOTES/chunks/c3-flags-fields/01-plan.md, section 7).

Run from the repository root:

    python benchmarks/bench_flags_fields.py [--quick] [--rounds 15] [--json PATH] [--md PATH]

Sizes (the tables of ``benchmarks/bench_forward.py``): S1, the 129-node golden table at
``lmax = 50``; S2, the 257-node refined LEBEDEV table at ``lmax = 100``; S3, the 129-node
table at ``lmax = 100`` (recorded, default and TALA only).

Configurations: ``default``; ``tala`` (compressible); ``nsg_uniform``
(NO_SELF_GRAVITY with ``Uniform(9.81)``, the G-ADOPT level); ``nsns`` (no slip at both
boundaries); ``tala_nsns_nsg`` (the three together); ``default_fields`` and
``tala_fields``: the misfit of the four fields of ``fields.apply`` with a fixed random
``real_4pi`` density in place of the kernel misfit. The density is an argument of the
jitted functions (data, as the observed field of a misfit). Recorded without a gate:
``default_fields_const``, the same with the density closed over, which XLA embeds as a
constant of the program (42 MB at S2) and which adds about 0.3 s of compile at S2
(NOTES/chunks/c3-flags-fields/impl-evidence/density_constant.out).

Quantities per configuration: the forward map (the four kernels, or the four fields), the
gradient of a scalar functional with respect to ``log eta``, and with respect to
``log eta``, ``rho0`` and the radius ("gradient all"). The functional is a weighted sum
of the outputs, each divided by its largest value at the base point (a zero output, the
velocity under a no-slip surface, keeps the scale 1).

Gates (section 7 of the plan):

- B1: run times of every flag configuration, interleaved with ``default`` over the
  rounds (medians), ratio to ``default`` at most 1.10, and inside the budgets of
  CONVENTIONS 7;
- B2: compile time of the forward map and of the gradient all, in fresh processes
  under the protocol of CONVENTIONS 7 (warm-up, no solve before the timed section, the
  forward compile, then the first gradient compile of the process), median of three;
  ratio to ``default`` at most 1.10, ``default`` forward at most 0.4 s;
- B3: field misfit against the kernel misfit of the same flags: run time forward
  + 1 ms (S1) and + 3 ms (S2), gradient all + 2 ms (S1) and + 6 ms (S2), compile
  + 0.10 s;
- B4: ``sh.synthesis_grid`` on the 1-degree grid (179 x 359): 2 ms (``L = 50``), 5 ms
  (``L = 100``), compile 0.3 s;
- B5: Gauss-Legendre synthesis and analysis at ``L = 100``: 5 ms, compile 0.3 s.

Recorded without a limit: one XLA thread, ``vmap`` of the gradient over 4 chains (TALA),
the compile under the sequential protocol of chunk 2 (forward, gradient ``log eta``,
gradient all in one process).
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
from stokesjx import propagate, sh  # noqa: E402

CONFIGS = {
    "default": (sj.Formulation(), False),
    "tala": (sj.Formulation(compressible=True), False),
    "nsg_uniform": (sj.Formulation(gravity=sj.GravityModel.NO_SELF_GRAVITY, background=sj.Uniform(9.81)), False),
    "nsns": (sj.Formulation(surface=sj.NoSlip(), cmb=sj.NoSlip()), False),
    "tala_nsns_nsg": (sj.Formulation(gravity=sj.GravityModel.NO_SELF_GRAVITY, compressible=True,
                                     surface=sj.NoSlip(), cmb=sj.NoSlip()), False),
    "default_fields": (sj.Formulation(), "argument"),
    "tala_fields": (sj.Formulation(compressible=True), "argument"),
    "default_fields_const": (sj.Formulation(), "constant"),
}
#: Configurations of the gates (the constant-density variant is recorded only).
GATED = [c for c in CONFIGS if c != "default_fields_const"]
QUANTITIES = bf.QUANTITIES
RATIO_LIMIT = 1.10
FIELD_EXTRA = {"S1": dict(forward=1.0, gradient_all=2.0), "S2": dict(forward=3.0, gradient_all=6.0)}
FIELD_COMPILE_EXTRA = 0.10
SH_BUDGET = {50: 2.0, 100: 5.0}
GL_BUDGET = 5.0
SH_COMPILE_BUDGET = 0.3


def _density(p0, lmax):
    """A fixed random ``real_4pi`` density ``(N, 2, lmax+1, lmax+1)``, kg/m^3."""
    rng = np.random.default_rng(5)
    tri = np.tril(np.ones((lmax + 1, lmax + 1)))
    return jnp.asarray(10.0 * rng.standard_normal((p0.structure.n_nodes, 2, lmax + 1, lmax + 1)) * tri)


def output_scales(p0, lmax, config: str):
    """The four output scales at the base point (one solve; the parent computes them)."""
    form, with_fields = CONFIGS[config]
    cfg = sj.StokesConfig(lmax=lmax, formulation=form)
    K = sj.stokes_kernels(p0, cfg)
    if with_fields:
        F = sj.apply(K, _density(p0, lmax), p0)
        outs = [getattr(F, n) for n in sj.fields.FIELD_NAMES]
    else:
        outs = [K.geoid, K.surface_topography, K.cmb_topography, K.surface_velocity]
    return [float(jnp.max(jnp.abs(o))) or 1.0 for o in outs]


def functions(p0, lmax, config: str, scales):
    """``(fns, x0)``: the forward map and the two gradients of a configuration, as
    functions of ``(log_eta, rho0, radius)``, and of the density as a fourth argument for
    the field configurations with ``"argument"`` (the gradients are with respect to the
    first three)."""
    form, with_fields = CONFIGS[config]
    cfg = sj.StokesConfig(lmax=lmax, formulation=form)
    plan = propagate.static_plan(p0.structure, cfg)
    x0 = (jnp.log(p0.eta), p0.rho0, p0.radius)
    density = _density(p0, lmax) if with_fields else None
    if with_fields == "argument":
        x0 = x0 + (density,)
    rng = np.random.default_rng(0)
    if with_fields:
        shapes = [(2, lmax + 1, lmax + 1)] * 4
    else:
        shapes = [(len(plan.degrees), p0.structure.n_nodes)] * 4
    W = [jnp.asarray(rng.standard_normal(s)) / c for s, c in zip(shapes, scales)]

    def forward(log_eta, rho0, radius, d=density):
        p = p0.replace(eta=jnp.exp(log_eta), rho0=rho0, radius=radius)
        K = sj.stokes_kernels(p, cfg)
        if with_fields:
            F = sj.apply(K, d, p)
            return tuple(getattr(F, n) for n in sj.fields.FIELD_NAMES)
        return K.geoid, K.surface_topography, K.cmb_topography, K.surface_velocity

    def functional(*x):
        return sum(jnp.sum(w * o) for w, o in zip(W, forward(*x)))

    return dict(forward=forward, gradient_log_eta=jax.grad(functional, argnums=0),
                gradient_all=jax.grad(functional, argnums=(0, 1, 2))), x0


def run_times(size: str, configs, rounds: int) -> dict:
    """All configurations of a size, compiled once and interleaved over the rounds."""
    make, lmax = bf.SIZES[size]
    p0 = make()
    groups = {}
    for c in configs:
        fns, x0 = functions(p0, lmax, c, output_scales(p0, lmax, c))
        for q in QUANTITIES:
            groups[f"{c}_{q}"] = (jax.jit(fns[q]).lower(*x0).compile(), x0)
    load0 = list(os.getloadavg())
    res = bf.interleaved(groups, rounds)
    res["loadavg_start"], res["loadavg_end"] = load0, list(os.getloadavg())
    res["pade"] = {c: propagate.static_plan(p0.structure, sj.StokesConfig(lmax=lmax, formulation=CONFIGS[c][0])
                                            ).pade.__dict__ for c in configs}
    return res


def child_compile(size: str, config: str, protocol: str, scales: str) -> dict:
    """Compile times in a fresh process (protocol of CONVENTIONS 7): the warm-up, no solve,
    then the quantities of ``bf.PROTOCOLS[protocol]`` in their order."""
    bf.warm_up()
    make, lmax = bf.SIZES[size]
    fns, x0 = functions(make(), lmax, config, [float(v) for v in scales.split(",")])
    out = {}
    for q in bf.PROTOCOLS[protocol]:
        t0 = time.perf_counter()
        jax.jit(fns[q]).lower(*x0).compile()
        out[q] = time.perf_counter() - t0
    return out


def child_one_thread(size: str, config: str) -> dict:
    """Run times of a configuration in a process with one XLA thread."""
    bf.warm_up()
    make, lmax = bf.SIZES[size]
    p0 = make()
    fns, x0 = functions(p0, lmax, config, output_scales(p0, lmax, config))
    return {q: bf.time_compiled(fns[q], x0)[0] for q in QUANTITIES}


def run_child(*args, env=None) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", *args]
    out = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def compile_times(size: str, configs, repeats: int) -> dict:
    """Median over ``repeats`` fresh processes per configuration and protocol."""
    make, lmax = bf.SIZES[size]
    p0 = make()
    out = {}
    for c in configs:
        scales = ",".join(repr(v) for v in output_scales(p0, lmax, c))
        for protocol in ("first", "sequential"):
            runs = [run_child("compile", size, c, protocol, scales) for _ in range(repeats)]
            out[f"{c}_{protocol}"] = {q: statistics.median(r[q] for r in runs) for q in bf.PROTOCOLS[protocol]}
            out[f"{c}_{protocol}"]["runs"] = runs
    return out


def vmap_chains(size: str) -> dict:
    """``vmap`` of the TALA gradient with respect to ``log eta`` over 4 chains."""
    make, lmax = bf.SIZES[size]
    p0 = make()
    fns, x0 = functions(p0, lmax, "tala", output_scales(p0, lmax, "tala"))
    rng = np.random.default_rng(3)
    X = (jnp.stack([x0[0] + 0.1 * rng.standard_normal(x0[0].shape) for _ in range(4)]),
         jnp.stack([x0[1]] * 4), jnp.stack([x0[2]] * 4))
    return bf.time_compiled(jax.vmap(fns["gradient_log_eta"]), X)[0]


def sh_times() -> dict:
    """B4 and B5: the grid synthesis on the 1-degree grid and the Gauss-Legendre round
    trip, compile (lower and compile) and median run time."""
    lat = np.arange(89.0, -90.0, -1.0)
    lon = np.arange(-179.0, 180.0, 1.0)
    out = {}
    for L in (50, 100):
        c = jnp.asarray(np.random.default_rng(L).standard_normal((2, L + 1, L + 1)) * np.tril(np.ones((L + 1, L + 1))))
        sh._legendre_cached(L, tuple(float(v) for v in lat))           # the host table, outside the timing
        out[f"synthesis_grid_L{L}"] = bf.time_compiled(lambda q: sh.synthesis_grid(q, "real_4pi", lat, lon), (c,))[0]
    L = 100
    grid = sh.GaussLegendreGrid.build(L + 1, 2 * L + 3, L)
    U = sh.convert(jnp.asarray(np.random.default_rng(7).standard_normal((2, L + 1, L + 1))
                               * np.tril(np.ones((L + 1, L + 1)))), "real_4pi", "complex_4pi")
    out["gauss_legendre_round_trip_L100"] = bf.time_compiled(lambda g, u: sh.analysis(g, sh.synthesis(g, u)),
                                                              (grid, U))[0]
    return out


def gates(rec) -> list[str]:
    """The verdict lines of B1 to B5."""
    lines = []
    ok = True
    for size, r in rec["run"].items():
        if size not in ("S1", "S2"):
            continue
        for c in GATED:
            if c == "default" or c.endswith("_fields"):
                continue
            for q in QUANTITIES:
                ratio = r[f"{c}_{q}"]["median_ms"] / r[f"default_{q}"]["median_ms"]
                within = r[f"{c}_{q}"]["median_ms"] <= bf.BUDGET[size][q]
                good = ratio <= RATIO_LIMIT and within
                ok &= good
                lines.append(f"B1 {size} {c:14s} {q:17s} {r[f'{c}_{q}']['median_ms']:8.2f} ms ratio {ratio:5.3f} "
                             f"budget {bf.BUDGET[size][q]:.0f} ms {'ok' if good else 'FAIL'}")
        for base, fld in (("default", "default_fields"), ("tala", "tala_fields")):
            for q in ("forward", "gradient_all"):
                extra = r[f"{fld}_{q}"]["median_ms"] - r[f"{base}_{q}"]["median_ms"]
                good = extra <= FIELD_EXTRA[size][q]
                ok &= good
                lines.append(f"B3 {size} {fld:14s} {q:17s} + {extra:6.2f} ms (limit {FIELD_EXTRA[size][q]} ms) "
                             f"{'ok' if good else 'FAIL'}")
    for size, comp in rec["compile"].items():
        d = comp["default_first"]
        good = d["forward"] <= bf.COMPILE_BUDGET["forward"]
        ok &= good
        lines.append(f"B2 {size} default forward compile {d['forward']:.3f} s (limit 0.4 s) {'ok' if good else 'FAIL'}")
        for c in CONFIGS:
            if c == "default" or c not in GATED:
                continue
            for q in ("forward", "gradient_all"):
                v = comp[f"{c}_first"][q]
                if c.endswith("_fields"):
                    base = comp[f"{c[:-7]}_first"][q]
                    good = v - base <= FIELD_COMPILE_EXTRA
                    lines.append(f"B3 {size} {c:14s} compile {q:13s} {v:.3f} s (+ {v - base:.3f} s, limit 0.10 s) "
                                 f"{'ok' if good else 'FAIL'}")
                else:
                    ratio = v / d[q]
                    good = ratio <= RATIO_LIMIT
                    lines.append(f"B2 {size} {c:14s} compile {q:13s} {v:.3f} s ratio {ratio:5.3f} "
                                 f"{'ok' if good else 'FAIL'}")
                ok &= good
    for L in (50, 100):
        t = rec["sh"][f"synthesis_grid_L{L}"]
        good = t["run_ms"] <= SH_BUDGET[L] and t["compile_s"] <= SH_COMPILE_BUDGET
        ok &= good
        lines.append(f"B4 synthesis_grid L={L}: {t['run_ms']:.3f} ms (limit {SH_BUDGET[L]}), compile "
                     f"{t['compile_s']:.3f} s {'ok' if good else 'FAIL'}")
    t = rec["sh"]["gauss_legendre_round_trip_L100"]
    good = t["run_ms"] <= GL_BUDGET and t["compile_s"] <= SH_COMPILE_BUDGET
    ok &= good
    lines.append(f"B5 Gauss-Legendre round trip L=100: {t['run_ms']:.3f} ms (limit {GL_BUDGET}), compile "
                 f"{t['compile_s']:.3f} s {'ok' if good else 'FAIL'}")
    lines.append("G9 " + ("holds" if ok else "FAILS"))
    return lines


def append_md(rec, md: Path, json_name: str):
    """One row per size and configuration in NOTES/benchmarks.md."""
    head = ("\n## bench_flags_fields.py (chunk 3: flags, fields, SH)\n\n"
            "| date | host | JAX | size | configuration | load start/end | forward ms | gradient log_eta ms | "
            "gradient all ms | compile fwd / grad all s (first) | compile fwd / grad log_eta / grad all s "
            "(sequential) | JSON |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    text = md.read_text() if md.exists() else "# Benchmarks\n"
    if "## bench_flags_fields.py" not in text:
        text += head
    for size, r in rec["run"].items():
        comp = rec["compile"].get(size, {})
        for c in CONFIGS:
            if f"{c}_forward" not in r:
                continue
            cells = [f"{r[f'{c}_{q}']['median_ms']:.2f}" for q in QUANTITIES]
            first = comp.get(f"{c}_first")
            seq = comp.get(f"{c}_sequential")
            f_txt = f"{first['forward']:.2f} / {first['gradient_all']:.2f}" if first else ""
            s_txt = " / ".join(f"{seq[q]:.2f}" for q in QUANTITIES) if seq else ""
            text += (f"| {rec['date']} | {rec['host']} | {rec['jax']} | {size} | {c} | "
                     f"{r['loadavg_start'][0]:.1f}/{r['loadavg_end'][0]:.1f} | " + " | ".join(cells)
                     + f" | {f_txt} | {s_txt} | {json_name} |\n")
    for size, comp in rec["compile"].items():
        c = comp.get("default_fields_const_first")
        if c:
            text += (f"\n{size} ({rec['date']}), default_fields with the density closed over (a constant of the "
                     f"program; recorded, no gate): compile forward / gradient all {c['forward']:.2f} / "
                     f"{c['gradient_all']:.2f} s (first).\n")
    sht = rec["sh"]
    text += (f"\nSH ({rec['date']}): synthesis_grid 1-degree grid L=50 {sht['synthesis_grid_L50']['run_ms']:.3f} ms "
             f"(compile {sht['synthesis_grid_L50']['compile_s']:.3f} s), L=100 "
             f"{sht['synthesis_grid_L100']['run_ms']:.3f} ms (compile {sht['synthesis_grid_L100']['compile_s']:.3f} s); "
             f"Gauss-Legendre round trip L=100 {sht['gauss_legendre_round_trip_L100']['run_ms']:.3f} ms "
             f"(compile {sht['gauss_legendre_round_trip_L100']['compile_s']:.3f} s).\n")
    md.write_text(text)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="S1 only, 3 rounds, one compile process")
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--json", default=str(REPO / "NOTES/benchmarks/c3-flags-fields.json"))
    p.add_argument("--md", default=str(REPO / "NOTES/benchmarks.md"))
    p.add_argument("--child", nargs="+", default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.child:
        mode = a.child[0]
        res = child_compile(*a.child[1:]) if mode == "compile" else child_one_thread(*a.child[1:])
        print(json.dumps(res))
        return
    rounds = 3 if a.quick else a.rounds
    sizes = ["S1"] if a.quick else ["S1", "S2"]
    t0 = time.time()
    rec = dict(host=platform.node(), machine=platform.machine(),
               date=datetime.datetime.now().isoformat(timespec="seconds"), jax=jax.__version__,
               numpy=np.__version__, cpu_count=os.cpu_count(), loadavg_start=list(os.getloadavg()), rounds=rounds)
    bf.warm_up()
    rec["run"] = {size: run_times(size, GATED, rounds) for size in sizes}
    if not a.quick:
        rec["run"]["S3"] = run_times("S3", ["default", "tala"], rounds)
    print("run times done", flush=True)
    rec["compile"] = {size: compile_times(size, list(CONFIGS), 1 if a.quick else 3) for size in sizes}
    print("compile times done", flush=True)
    rec["vmap_tala_4_chains"] = {size: vmap_chains(size) for size in sizes}
    env = dict(os.environ, XLA_FLAGS=bf.ONE_THREAD_FLAGS)
    rec["one_thread"] = {size: {c: run_child("one_thread", size, c, env=env) for c in ("default", "tala")}
                         for size in sizes}
    rec["sh"] = sh_times()
    rec["loadavg_end"] = list(os.getloadavg())
    rec["wall_s"] = time.time() - t0
    rec["gates"] = gates(rec)
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"load average start {rec['loadavg_start']}, end {rec['loadavg_end']}, wall {rec['wall_s']:.0f} s")
    for line in rec["gates"]:
        print(line)
    for size, v in rec["vmap_tala_4_chains"].items():
        print(f"{size} vmap TALA gradient log eta, 4 chains: {v['run_ms']:.2f} ms")
    for size, d in rec["one_thread"].items():
        for c, o in d.items():
            print(f"{size} one thread {c}: " + "  ".join(f"{q} {o[q]['run_ms']:.2f} ms" for q in QUANTITIES))
    for size, comp in rec["compile"].items():
        for c in CONFIGS:
            if f"{c}_sequential" not in comp:
                continue
            s = comp[f"{c}_sequential"]
            print(f"{size} compile sequential {c}: " + " / ".join(f"{s[q]:.3f}" for q in QUANTITIES) + " s")
    if not a.quick:
        append_md(rec, Path(a.md), Path(a.json).name)


if __name__ == "__main__":
    main()
