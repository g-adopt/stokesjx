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

The viscosity cut (chunk c4b, NOTES/chunks/c4b-boundaries-across-jumps/01-plan.md section
10). With ``--c4b`` the script also times:

- at S1 and S2, interleaved with the programs above: ``layers_cut``, ``layers_cut_field`` and
  ``layers_cut_vmap4``, the same functionals with ``Layers(..., motion="cut")`` on the same
  range (the chunk 4 programs keep ``motion="auto"``, which is the affine motion there);
- SJ: the full ``t_prem()`` table (137 nodes, density pairs at 5700, 5970 and 6150 km) with
  the boundaries 3480, 5700, 5970, 6270, 6370 km and the 5700 km boundary movable in
  (5466.875, 5970) km, so that the range holds the 670 km density jump and ``"auto"`` is
  the cut; hager4 viscosities, ``lmax = 50``; SJ100: the same at ``lmax = 100``. Programs
  ``direct`` (``value_and_grad`` with respect to ``log eta`` per node, ``rho0``, the radius
  and ``cut_radius`` of the construction profile), ``layers``, ``layers_field`` and
  ``layers_vmap4``;
- the gradient compile (protocol "first", three processes) of every new program.

Gates of c4b (``gates_c4b``): B7 ``layers_cut`` at S1 and S2 over the HEAD ``layers`` (run and
compile, at most 1.10; the HEAD numbers come from ``--head-json`` records of this script
run with the HEAD tree); B8 SJ and SJ100 ``layers`` over ``direct`` (run and compile, at most
1.10) and inside the budgets 48 ms (SJ) and 181 ms (SJ100); B9 ``layers_cut_vmap4`` and SJ
``layers_vmap4`` at most 4.0 times one call; B10 ``layers_field - layers`` of the cut at most
+2 ms (S1, SJ) and +6 ms (S2) run and +0.10 s compile. B7 is the cost gate of the cut: it
compares the cut with the affine method at the same controls. B8 compares with a heavier
``direct`` program (it also differentiates the per-node ``log eta``, ``rho0`` and the
radius), so its ratio is not a measure of the cost of the cut. B6 (every chunk 4 program, run and
gradient compile, new tree over HEAD, at most 1.10) is ``--compare-head NEW.json ... --head-json
HEAD.json ...``: the medians over the records of each tree.
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

def _t_prem_profile():
    """The full t_prem() table of the tests (137 nodes, split core), for SJ and SJ100."""
    import cases
    from adapters import profile_from_table
    return profile_from_table(cases.t_prem())


SIZES = {"S1": (bf.golden_profile, 50), "S2": (bf.lebedev_refined, 100), "SJ": (_t_prem_profile, 50),
         "SJ100": (_t_prem_profile, 100)}
BOUNDARIES = np.array([3480e3, 5700e3, 5970e3, 6270e3, 6370e3])
MOVABLE = {1: (5466.875e3, 5970e3)}
HAGER4 = np.array([1e20, 1e19, 0.03333333333333333e19, 1e20])
OUTS = ("geoid", "surface_topography", "cmb_topography", "surface_velocity")
RATIO_LIMIT = 1.10
BUDGET = {"S1": 48.0, "S2": 181.0}
VMAP4_LIMIT = 4.0
NUMPY_OVERHEAD_MS = 0.5
FIELD_EXTRA_MS = {"S1": 2.0, "S2": 6.0, "SJ": 2.0, "SJ100": 6.0}
BUDGET.update({"SJ": 48.0, "SJ100": 181.0})
# c4b: the cut programs at S1 and S2 and the programs of SJ and SJ100.
CUT_PROGRAMS = ("layers_cut", "layers_cut_field", "layers_cut_vmap4")
SJ_PROGRAMS = ("direct", "layers", "layers_field", "layers_vmap4")
FIELD_COMPILE_EXTRA = 0.10
PROGRAMS = ("direct", "direct_construction", "layers", "layers_field", "layers_vmap4", "layers_vmap16")
COMPILED = ("layers", "layers_field", "layers_vmap4", "direct")


def make_layers(size: str, two: bool = False, motion: str = "auto"):
    """``(Layers, cfg)`` of a size; ``two`` gives the two-boundary layering of ``layers_two``;
    ``motion`` is the ``Layers`` keyword (``"auto"`` is the affine motion at S1 and S2 and the
    cut at SJ and SJ100)."""
    make, lmax = SIZES[size]
    p = make()
    if two:
        r = np.asarray(p.radius)
        mid = float(r[np.argmin(np.abs(r - 5850.703e3))])
        L = sj.Layers(p, BOUNDARIES, movable={1: (5466.875e3, mid), 2: (mid, 6189.375e3)}, motion=motion)
    else:
        L = sj.Layers(p, BOUNDARIES, movable=MOVABLE, motion=motion)
    return L, sj.StokesConfig(lmax=lmax)


def kernel_scales(size: str) -> list[float]:
    """The largest value of each kernel at the construction radius (one solve, parent)."""
    L, cfg = make_layers(size)
    K = sj.stokes_kernels(L(L.boundaries, np.log(HAGER4)), cfg)
    return [float(np.max(np.abs(np.asarray(getattr(K, o))))) for o in OUTS]


def functions(size: str, scales, two: bool = False, motion: str = "auto") -> dict:
    """``name -> (function, args, forward)``: the programs of the benchmark. ``function`` is
    the gradient program, ``forward`` its value alone (for the compile protocol). For a cut
    layering ``direct`` also differentiates ``cut_radius`` of the construction profile."""
    L, cfg = make_layers(size, two, motion)
    lmax = cfg.lmax
    n_deg = lmax                                  # degrees 1..lmax (CM frame)
    rng = np.random.default_rng(4)
    W = [jnp.asarray(rng.standard_normal((n_deg, L.n_nodes)) / s) for s in scales]

    def F_of(p):
        K = sj.stokes_kernels(p, cfg)
        return sum(jnp.sum(w * getattr(K, o)) for w, o in zip(W, OUTS))

    p0 = L(L.boundaries, np.log(HAGER4))
    out = {}
    if p0.cut_radius is None:
        def direct_f(le, rho0, radius, base=p0):
            return F_of(base.replace(eta=jnp.exp(le), rho0=rho0, radius=radius))
        xd = (jnp.log(p0.eta), p0.rho0, p0.radius)
        out["direct"] = (jax.value_and_grad(direct_f, argnums=(0, 1, 2)), xd, direct_f)
        fresh = sj.RadialProfile(np.asarray(p0.radius), np.asarray(p0.rho0), np.asarray(p0.eta),
                                 core_density=float(p0.core_density), core_mass=p0.core_mass)

        def direct_c(le, rho0, radius):
            return direct_f(le, rho0, radius, base=fresh)
        out["direct_construction"] = (jax.value_and_grad(direct_c, argnums=(0, 1, 2)), xd, direct_c)
    else:
        # A cut layering: the direct gradient also moves the traced cut radius.
        def direct_f(le, rho0, radius, cut):
            return F_of(p0.replace(eta=jnp.exp(le), rho0=rho0, radius=radius, cut_radius=cut))
        xd = (jnp.log(p0.eta), p0.rho0, p0.radius, p0.cut_radius)
        out["direct"] = (jax.value_and_grad(direct_f, argnums=(0, 1, 2, 3)), xd, direct_f)

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
    plain = sj.RadialProfile(np.asarray(p0.radius), np.asarray(p0.rho0), np.asarray(p0.eta),
                             core_density=float(p0.core_density), core_mass=p0.core_mass)
    out["_plans"] = dict(widened=propagate.static_plan(L.profile.structure, cfg).pade.__dict__,
                         construction=propagate.static_plan(plain.structure, cfg).pade.__dict__,
                         width_factor=L.width_factor, motion=L.motion)
    return out


def cut_functions(size: str, scales) -> dict:
    """The c4b programs of S1 and S2 under ``motion="cut"``: ``layers_cut``,
    ``layers_cut_field``, ``layers_cut_vmap4`` (the chunk 4 functionals on the cut layering)."""
    fns = functions(size, scales, motion="cut")
    return {"layers_cut": fns["layers"], "layers_cut_field": fns["layers_field"],
            "layers_cut_vmap4": fns["layers_vmap4"], "_plans": fns["_plans"]}


def run_times(size: str, rounds: int, c4b: bool = False) -> dict:
    """All programs of a size compiled once in this process and interleaved over the rounds;
    the NumPy front against the JAX call of the same compiled function. With ``c4b`` the cut
    programs join the interleaving (S1, S2); SJ and SJ100 time ``SJ_PROGRAMS``."""
    scales = kernel_scales(size)
    fns = functions(size, scales)
    names = SJ_PROGRAMS if size in ("SJ", "SJ100") else PROGRAMS
    if c4b and size in ("S1", "S2"):
        cfns = cut_functions(size, scales)
        fns.update({k: v for k, v in cfns.items() if not k.startswith("_")})
        names = names + CUT_PROGRAMS
    groups = {}
    for name in names:
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
    if c4b and size in ("S1", "S2"):
        res["pade_cut"] = cfns["_plans"]
    return res


def child_compile(size: str, program: str, scales: str) -> dict:
    """Protocol "first" in this fresh process: the warm-up, no solve, the forward compile,
    then the gradient compile of ``program``."""
    bf.warm_up()
    sc = [float(v) for v in scales.split(",")]
    fns = cut_functions(size, sc) if program in CUT_PROGRAMS else functions(size, sc)
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


def compile_times(size: str, repeats: int, c4b: bool = False) -> dict:
    scales = ",".join(repr(v) for v in kernel_scales(size))
    out = {}
    if size in ("SJ", "SJ100"):
        programs = SJ_PROGRAMS
    else:
        programs = COMPILED + (CUT_PROGRAMS if c4b else ())
    for prog in programs:
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


def gates_c4b(rec, head) -> list[str]:
    """The c4b gates B7 to B10 (module docstring). ``head``: the merged HEAD record of
    ``merge_records`` or ``None`` (then B7 is not evaluated)."""
    lines, ok = [], True

    def check(label, value, limit, text):
        nonlocal ok
        good = value <= limit
        ok &= good
        lines.append(f"{label} {text}: {'ok' if good else 'FAIL'}")
    for size in [s for s in ("S1", "S2") if s in rec["run"]]:
        r, c = rec["run"][size], rec["compile"][size]
        cut, aff = r["layers_cut"]["median_ms"], r["layers"]["median_ms"]
        lines.append(f"   {size} layers_cut {cut:.2f} ms against the new-tree layers (affine) {aff:.2f} ms in the same "
                     f"interleaving: {cut / aff:.3f}; plans cut {r['pade_cut']['widened']}, affine {r['pade']['widened']}")
        if head is not None:
            hr, hc = head["run"][size]["layers"], head["compile"][size]["layers"]["gradient"]
            check("B7", cut / hr, RATIO_LIMIT, f"{size} layers_cut {cut:.2f} ms over HEAD layers {hr:.2f} ms = "
                  f"{cut / hr:.3f} (limit 1.10)")
            cc = c["layers_cut"]["gradient"]
            check("B7", cc / hc, RATIO_LIMIT, f"{size} compile layers_cut {cc:.3f} s over HEAD layers {hc:.3f} s = "
                  f"{cc / hc:.3f} (limit 1.10)")
        v4 = r["layers_cut_vmap4"]["median_ms"]
        check("B9", v4 / cut, VMAP4_LIMIT, f"{size} layers_cut_vmap4 {v4:.2f} ms = {v4 / cut:.2f} x layers_cut (limit 4.0)")
        extra = r["layers_cut_field"]["median_ms"] - cut
        check("B10", extra, FIELD_EXTRA_MS[size], f"{size} layers_cut_field - layers_cut = {extra:+.2f} ms "
              f"(limit +{FIELD_EXTRA_MS[size]})")
        extra = c["layers_cut_field"]["gradient"] - c["layers_cut"]["gradient"]
        check("B10", extra, FIELD_COMPILE_EXTRA, f"{size} compile layers_cut_field - layers_cut = {extra:+.3f} s "
              "(limit +0.10)")
    for size in ("SJ", "SJ100"):
        r, c = rec["run"][size], rec["compile"][size]
        lay, dir_ = r["layers"]["median_ms"], r["direct"]["median_ms"]
        check("B8", lay / dir_, RATIO_LIMIT, f"{size} layers {lay:.2f} ms over direct {dir_:.2f} ms = {lay / dir_:.3f} "
              "(limit 1.10)")
        check("B8", lay, BUDGET[size], f"{size} layers {lay:.2f} ms (budget {BUDGET[size]:.0f} ms)")
        cl, cd = c["layers"]["gradient"], c["direct"]["gradient"]
        check("B8", cl / cd, RATIO_LIMIT, f"{size} compile layers {cl:.3f} s over direct {cd:.3f} s = {cl / cd:.3f} "
              "(limit 1.10)")
        v4 = r["layers_vmap4"]["median_ms"]
        check("B9", v4 / lay, VMAP4_LIMIT, f"{size} layers_vmap4 {v4:.2f} ms = {v4 / lay:.2f} x layers (limit 4.0)")
        if size == "SJ":
            extra = r["layers_field"]["median_ms"] - lay
            check("B10", extra, FIELD_EXTRA_MS[size], f"{size} layers_field - layers = {extra:+.2f} ms (limit +2)")
            extra = c["layers_field"]["gradient"] - c["layers"]["gradient"]
            check("B10", extra, FIELD_COMPILE_EXTRA, f"{size} compile layers_field - layers = {extra:+.3f} s "
                  "(limit +0.10)")
        lines.append(f"   {size} plan {r['pade']['widened']}, motion {r['pade']['motion']}")
    lines.append("c4b gates B7 to B10 " + ("hold" if ok else "FAIL"))
    return lines


def merge_records(paths) -> dict:
    """The medians over several JSON records of this script (one tree): run time (median of
    the per-record medians) and gradient compile per size and program."""
    recs = [json.loads(Path(p).read_text()) for p in paths]
    out = {"run": {}, "compile": {}, "n": len(recs)}
    for size in recs[0]["run"]:
        out["run"][size] = {k: statistics.median(r["run"][size][k]["median_ms"] for r in recs)
                            for k, v in recs[0]["run"][size].items() if isinstance(v, dict) and "median_ms" in v}
        out["compile"][size] = {k: dict(gradient=statistics.median(r["compile"][size][k]["gradient"] for r in recs))
                                for k in recs[0]["compile"][size]}
    return out


def compare_head(new_paths, head_paths) -> list[str]:
    """B6: every chunk 4 program (``direct``, ``layers``, ``layers_field``, ``layers_vmap4``,
    ``numpy``) at S1 and S2, run time and gradient compile, new tree over HEAD (medians over
    the records of each tree), at most 1.10."""
    new, head = merge_records(new_paths), merge_records(head_paths)
    lines, ok = [], True
    for size in ("S1", "S2"):
        for prog in ("direct", "layers", "layers_field", "layers_vmap4", "numpy"):
            a, b = new["run"][size][prog], head["run"][size][prog]
            good = a / b <= RATIO_LIMIT
            ok &= good
            lines.append(f"B6 {size} run {prog}: new {a:.2f} ms, HEAD {b:.2f} ms, ratio {a / b:.3f}: "
                         f"{'ok' if good else 'FAIL'}")
        for prog in COMPILED:
            a, b = new["compile"][size][prog]["gradient"], head["compile"][size][prog]["gradient"]
            good = a / b <= RATIO_LIMIT
            ok &= good
            lines.append(f"B6 {size} compile {prog}: new {a:.3f} s, HEAD {b:.3f} s, ratio {a / b:.3f}: "
                         f"{'ok' if good else 'FAIL'}")
    lines.append(f"B6 ({new['n']} new and {head['n']} HEAD records) " + ("holds" if ok else "FAILS"))
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
    p.add_argument("--c4b", action="store_true", help="also the cut programs at S1, S2 and the sizes SJ, SJ100")
    p.add_argument("--head-json", nargs="+", help="records of this script run with the HEAD tree (B6, B7)")
    p.add_argument("--compare-head", nargs="+", metavar="NEW_JSON", help="print B6 for these new-tree records")
    p.add_argument("--no-md", action="store_true", help="do not append to the Markdown table")
    a = p.parse_args(argv)
    if a.compare_head:
        for line in compare_head(a.compare_head, a.head_json):
            print(line)
        return
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
    all_sizes = sizes + (["SJ", "SJ100"] if a.c4b else [])
    for size in all_sizes:
        rec["run"][size] = run_times(size, rounds, a.c4b)
        print(f"{size} run: " + ", ".join(f"{k} {v['median_ms']:.2f}" for k, v in rec["run"][size].items()
                                        if isinstance(v, dict) and "median_ms" in v), flush=True)
    for size in all_sizes:
        rec["compile"][size] = compile_times(size, repeats, a.c4b)
        print(f"{size} compile: " + ", ".join(f"{k} {v['forward']:.3f}/{v['gradient']:.3f}"
                                            for k, v in rec["compile"][size].items()), flush=True)
    env = dict(os.environ, XLA_FLAGS=bf.ONE_THREAD_FLAGS)
    for size in sizes:
        rec["one_thread"][size] = run_child("one_thread", size, ",".join(repr(v) for v in kernel_scales(size)), env=env)
    rec["layers_two"] = run_child("two", ",".join(repr(v) for v in kernel_scales("S1")))
    c4_rec = dict(rec, run={k: v for k, v in rec["run"].items() if k in sizes},
                  compile={k: v for k, v in rec["compile"].items() if k in sizes})
    rec["gates"] = gates(c4_rec)
    if a.c4b:
        rec["gates_c4b"] = gates_c4b(rec, merge_records(a.head_json) if a.head_json else None)
    for line in rec["gates"] + rec.get("gates_c4b", []):
        print(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, default=str))
    if not a.no_md:
        append_md(c4_rec, Path(a.md), path.name)
        if a.c4b:
            append_md_c4b(rec, Path(a.md), path.name)
    print(f"written {path}")


def append_md_c4b(rec, md: Path, json_name: str):
    """Rows of the c4b programs (the cut at S1 and S2, the sizes SJ and SJ100)."""
    text = md.read_text()
    head = ("\n## bench_controls.py --c4b (chunk c4b: the viscosity cut)\n\n"
            "| date | host | JAX | size | load start/end | direct ms | layers ms | layers_field ms | vmap4 ms | "
            "layers_cut ms | layers_cut_field ms | layers_cut_vmap4 ms | compile layers / layers_cut / direct s "
            "(gradient, first) | JSON |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    if "## bench_controls.py --c4b" not in text:
        text += head
    for size, r in rec["run"].items():
        c = rec["compile"][size]
        g = lambda k: f"{r[k]['median_ms']:.2f}" if k in r else ""
        cc = lambda k: f"{c[k]['gradient']:.2f}" if k in c else "-"
        text += (f"| {rec['date']} | {rec['host']} | {rec['jax']} | {size} | {r['loadavg_start'][0]:.1f}/"
                 f"{r['loadavg_end'][0]:.1f} | {g('direct')} | {g('layers')} | {g('layers_field')} | "
                 f"{g('layers_vmap4')} | {g('layers_cut')} | {g('layers_cut_field')} | {g('layers_cut_vmap4')} | "
                 f"{cc('layers')} / {cc('layers_cut')} / {cc('direct')} | {json_name} |\n")
    md.write_text(text)


if __name__ == "__main__":
    main()
