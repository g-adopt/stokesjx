"""Gradient of a geoid misfit with respect to four layer viscosities and the radius of one
movable viscosity boundary, checked against central differences.

Command line::

    python examples/gradient_layers.py [--lmax 20] [--reduced] [--fd-steps 1e-4 50]
        [--out PATH]

Options: ``--lmax`` the largest degree (default 20); ``--reduced`` the 25-node table in
place of the 137-node PREM-like table of ``earth_model.prem_profile``; ``--fd-steps H_ETA
H_R`` the steps of the central differences, in ln(Pa s) for a log viscosity and in m for
the radius (default 1e-4 and 50 m); ``--out`` the JSON summary (default
``examples/output/gradient_summary.json`` beside this script).

Output: the JSON summary (strict JSON, no NaN) with the control names, the misfit, the
gradient, the central differences, the relative difference per component
``|g_i - fd_i| / |g_i|``, the ridge sum and the NumPy gradient. The script prints the same
numbers as a table.

The problem. The layering is that of the HMC example (``hmc_layers.py``): boundaries at
3480, ``r_b``, 5970, 6270 and 6370 km, so four layers (lower mantle, ``r_b`` to 400 km,
400 to 100 km, the lid). Boundary 1 is movable in (5466.875, 5970) km, a range that holds
the 670 km density jump of the table at 5700 km; ``Layers`` resolves ``motion="auto"`` to
the viscosity cut, so the boundary crosses the density jump freely
(``stokesjx.controls.Layers``). The density anomaly is the smooth anomaly of
``earth_model.smooth_density`` (seed 7) on the input table, and the field is
``Layers.fields`` (the density stays on the input table when the boundary moves). The
observed geoid is the geoid of the HMC truth (viscosities 3e22, 1e21, 3e20, 1e22 Pa s,
boundary at 5690 km) without noise, and the misfit is ``spectral_misfit`` with the weight
1 for every degree ``l >= 2`` (m^2).

The base point. The gradient is evaluated at ``Y_REF = ln(1e22, 1e21, 1e21, 1e21)`` Pa s
and the boundary at 5750 km: a lower mantle ten times stiffer than the three layers
above, so that there is a viscosity contrast of 10 across the movable boundary. When the
viscosities on both sides of a boundary are equal, the position of the boundary has no
effect on the kernels, and the radius derivative is exactly zero (measured at the
isoviscous point: -1.9e-15 on the full table, -3.3e-16 on the reduced table). At
``Y_REF`` the radius derivative is 1.82 m^-1 (full table, ``lmax = 20``) and 0.925 m^-1
(reduced, ``lmax = 6``). The base radius is 10.4 km (full) and 50 km (reduced) from the
nearest input node, so the central difference of 50 m does not cross a node.

The check. The components have different units (a log viscosity in ln(Pa s), the radius
in m), so the gradient is compared with the central differences per component, not as
one vector norm: at ``Y_REF`` the radius component is 3e-6 of the norm of the gradient.
The relative difference is the truncation error of the central difference (about 1e-9;
it falls by four when the steps are halved).

The ridge. The geoid and the topographies do not change when every viscosity is
multiplied by one factor (``NOTES/PLAN.md`` section 2), so the sum of the four
log-viscosity derivatives is zero to rounding. A sampler over all layer viscosities
therefore needs a prior on the mean log viscosity or a fixed layer; the HMC example uses
a narrow prior on the mean.

The NumPy front. ``ControlVector.numpy_value_and_grad`` gives the same value and
gradient from a NumPy vector, for samplers and optimisers outside JAX.

As a module: ``run(parse_args([...]))`` returns the dict of the summary;
``build(args)`` returns the ``Problem``; ``main(argv)`` also prints and writes the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

import stokesjx as sj

_HERE = Path(__file__).resolve().parent
#: The default JSON summary (``examples/output/`` is in ``.gitignore``).
DEFAULT_OUT = _HERE / "output" / "gradient_summary.json"

KM = 1e3
#: The layer boundaries at the construction, m: CMB, the movable boundary, 400 km,
#: 100 km and the surface.
BOUNDARIES = np.array([3480.0, 5750.0, 5970.0, 6270.0, 6370.0]) * KM
#: The movable range of boundary 1, m: from a LEBEDEV node at 903 km depth to the 400 km
#: pair; it holds the 670 km density jump at 5700 km.
RANGE = (5466.875 * KM, 5970.0 * KM)
#: The base point: ln of the layer viscosities in Pa s, lower mantle first. The lower
#: mantle is ten times stiffer than the layers above (contrast 10 across the boundary).
Y_REF = np.log([1e22, 1e21, 1e21, 1e21])
#: The viscosities of the truth that makes the observed geoid, ln(Pa s) (the HMC truth).
Y_TRUE = np.log([3e22, 1e21, 3e20, 1e22])
#: The radius of the boundary in the truth, m (680 km depth, 10 km below the density jump).
R_TRUE = 5690.0 * KM
#: The seed of the smooth density anomaly.
DENSITY_SEED = 7


def _earth_model():
    """The module ``examples/earth_model.py``, loaded by its path (no ``sys.path``
    change)."""
    import importlib.util
    name = "stokesjx_examples_earth_model"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _HERE / "earth_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None) -> argparse.Namespace:
    """The command line of the module docstring. ``argv=None`` reads ``sys.argv``."""
    p = argparse.ArgumentParser(description="Gradient of a geoid misfit through stokesjx Layers.")
    p.add_argument("--lmax", type=int, default=20, help="largest degree (default 20)")
    p.add_argument("--reduced", action="store_true", help="the 25-node table in place of the 137-node table")
    p.add_argument("--fd-steps", type=float, nargs=2, default=(1e-4, 50.0), metavar=("H_ETA", "H_R"),
                   help="central-difference steps: ln(Pa s) and m (default 1e-4 50)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON summary")
    return p.parse_args(argv)


@dataclass(eq=False)
class Problem:
    """The gradient problem.

    Attributes
    ----------
    layers
        ``sj.Layers`` of the four layers with boundary 1 movable.
    config
        ``sj.StokesConfig``.
    density
        ``(N_in, 2, lmax+1, lmax+1)`` kg/m^3, ``real_4pi``, on the input table.
    observed
        ``(2, lmax+1, lmax+1)`` m: the geoid of the truth, ``real_4pi``.
    weights
        ``(lmax+1,)``: 1 for ``l >= 2``, 0 otherwise.
    cv
        ``sj.ControlVector`` of ``{"radius": (1,), "log_eta": (4,)}``.
    fn
        ``fn(controls, density, observed)``: the misfit as the function that ``cv``
        compiles (one object, so that its compiled programs are cached).
    """

    layers: sj.Layers
    config: sj.StokesConfig
    density: jax.Array
    observed: jax.Array
    weights: jax.Array
    cv: sj.ControlVector
    fn: Callable | None = None


def geoid(controls, density, layers, config) -> jax.Array:
    """The geoid field of ``density`` (on the input table) for the controls
    ``{"radius": (1,) m, "log_eta": (4,) ln(Pa s)}``, ``(2, lmax+1, lmax+1)`` m,
    ``real_4pi``."""
    prof = layers(layers.full_boundaries(controls["radius"]), controls["log_eta"])
    return layers.fields(sj.stokes_kernels(prof, config), density).geoid


def misfit(controls, density, observed, problem) -> jax.Array:
    """The geoid misfit ``sum_l w_l S_dd(l)`` of the field minus the observed geoid, m^2
    (``fields.spectral_misfit`` in ``real_4pi``), without a prior."""
    N = geoid(controls, density, problem.layers, problem.config)
    return sj.spectral_misfit(N, observed, problem.weights, "real_4pi")


def build(args) -> Problem:
    """The ``Problem`` of the command line ``args`` (``parse_args``)."""
    em = _earth_model()
    table = em.prem_profile(reduced=bool(args.reduced))
    layers = sj.Layers(table, BOUNDARIES, movable={1: RANGE})
    config = sj.StokesConfig(lmax=int(args.lmax))
    density = jnp.asarray(em.smooth_density(np.asarray(table.radius), config.lmax, DENSITY_SEED))
    truth = {"radius": jnp.array([R_TRUE]), "log_eta": jnp.asarray(Y_TRUE)}
    observed = jax.jit(geoid, static_argnums=(2, 3))(truth, density, layers, config)
    weights = jnp.asarray(np.where(np.arange(config.lmax + 1) >= 2, 1.0, 0.0))
    cv = sj.ControlVector({"radius": np.asarray(layers.start_radii, dtype=np.float64),
                           "log_eta": np.asarray(Y_REF, dtype=np.float64)})
    problem = Problem(layers=layers, config=config, density=density, observed=observed, weights=weights, cv=cv)

    def fn(controls, density, observed):
        # The misfit as a function of the controls and the data, the form that
        # ControlVector compiles; the density and the observed geoid are arguments of
        # the compiled program, never closed-over constants.
        return misfit(controls, density, observed, problem)
    problem.fn = fn
    return problem


def run(args) -> dict:
    """The gradient at the base point, the central differences, the ridge and the NumPy
    front.

    Returns
    -------
    A dict of plain Python values (JSON-ready): ``names`` (five control names in the
    order of the vector), ``value`` (the misfit, m^2), ``gradient``, ``fd``,
    ``relative_difference`` (per component), ``numpy_gradient`` (five floats each),
    ``numpy_value``, ``ridge`` (``{"sum": ..., "norm": ...}``: the sum of the four
    log-viscosity derivatives and the norm of the log-viscosity gradient), ``steps`` (the
    step per component), ``base_radius_m``, ``lmax``, ``n_nodes``.
    """
    P = build(args)
    cv = P.cv
    x0 = cv.ravel({"radius": jnp.asarray(P.layers.start_radii), "log_eta": jnp.asarray(Y_REF)})
    value, grad = cv.value_and_grad(P.fn)(x0, P.density, P.observed)
    value, grad = float(value), np.asarray(grad)
    f = cv.value(P.fn)
    h_eta, h_r = (float(s) for s in args.fd_steps)
    steps = np.array([h_r if n.startswith("radius") else h_eta for n in cv.names])
    fd = np.empty(cv.size)
    for i in range(cv.size):
        # Central difference along component i: (f(x + h e_i) - f(x - h e_i)) / (2 h).
        e = np.zeros(cv.size)
        e[i] = steps[i]
        fp = float(f(x0 + e, P.density, P.observed))
        fm = float(f(x0 - e, P.density, P.observed))
        fd[i] = (fp - fm) / (2.0 * steps[i])
    rel = np.abs(grad - fd) / np.abs(grad)
    eta_idx = np.array([n.startswith("log_eta") for n in cv.names])
    ridge_sum = float(np.sum(grad[eta_idx]))
    ridge_norm = float(np.linalg.norm(grad[eta_idx]))
    nf = cv.numpy_value_and_grad(P.fn, P.density, P.observed)
    nv, ng = nf(np.asarray(x0))
    return {
        "names": list(cv.names), "value": value, "gradient": grad.tolist(), "fd": fd.tolist(),
        "relative_difference": rel.tolist(), "steps": steps.tolist(),
        "ridge": {"sum": ridge_sum, "norm": ridge_norm},
        "numpy_value": float(nv), "numpy_gradient": np.asarray(ng).tolist(),
        "base_radius_m": float(P.layers.start_radii[0]), "lmax": P.config.lmax,
        "n_nodes": int(P.layers.n_nodes),
    }


def main(argv=None) -> dict:
    """Parse, run, print the table and write the JSON summary (``--out``); returns the
    dict of ``run``."""
    args = parse_args(argv)
    out = run(args)
    print(f"gradient example: {out['n_nodes']} nodes, lmax {out['lmax']}, misfit {out['value']:.6e} m^2")
    print(f"{'control':>12s} {'gradient':>14s} {'central diff':>14s} {'rel. diff':>10s}")
    for n, g, d, r in zip(out["names"], out["gradient"], out["fd"], out["relative_difference"]):
        print(f"{n:>12s} {g:14.6e} {d:14.6e} {r:10.2e}")
    print(f"ridge: sum of the log-viscosity derivatives {out['ridge']['sum']:.3e}, "
          f"norm {out['ridge']['norm']:.3e}")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, allow_nan=False))
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    main()
