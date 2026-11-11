"""Hamiltonian Monte Carlo (HMC) over four layer viscosities and the radius of one movable
viscosity boundary, against a synthetic geoid, with a hand-written leapfrog.

Scope. The HMC here is an example only: the inversion that stokesjx serves is joint with
lovejx, and stokesjx supplies values and derivatives for it; the settings of this
example (noise 0.2, step 0.5, the continuation ladder, the prior) belong to this
synthetic problem and are not a recommendation for another problem.

Command line::

    python examples/hmc_layers.py [--lmax 20] [--reduced] [--samples 1000]
        [--leapfrog 10] [--step 0.5] [--seed 3] [--motion auto|cut|affine]
        [--range LO HI] [--start-radius KM] [--true-radius KM] [--noise 0.2]
        [--smoke] [--out PATH]

Options: ``--lmax`` the largest degree; ``--reduced`` the 25-node table in place of the
137-node PREM-like table of ``earth_model.prem_profile``; ``--samples`` the number of HMC
transitions; ``--leapfrog`` the leapfrog steps per trajectory; ``--step`` the leapfrog
step size in the whitened coordinates (each transition multiplies it by a uniform number
in [0.8, 1.2]); ``--seed`` the seed of the chain; ``--motion`` the method of the movable
boundary (``stokesjx.Layers``); ``--range LO HI`` the movable range in km (default
5466.875 5970, across the 670 km density jump at 5700 km); ``--start-radius`` the
construction radius of the boundary in km (default 5750, above the middle of the range);
``--true-radius`` the radius of the synthetic truth in km (default 5690); ``--noise`` the
noise as a fraction of the root mean square geoid coefficient per degree (default 0.2);
``--smoke`` a run of a few seconds (25-node table, ``lmax = 4``, step 0.2, 8 samples, 3
leapfrog steps; the same noise, MAP search and MAP gate); ``--out`` the JSON summary
(default ``examples/output/hmc_summary.json`` beside this script).

Output: the JSON summary (strict JSON: a non-finite number is written as ``null``),
with the settings (the resolved ``motion``, the range, the start and the true radius in
km), the MAP record (iterations per stage, the metric norm of the gradient, ``U`` at the
MAP and at the truth, the condition number of the metric), the samples in the
coordinates ``z`` and in physical units (Pa s, km), per transition the Metropolis
probability, the accept flag, ``dH``, the potential that the chain carries, the start
momentum and the smallest and largest radius of the path, and the summaries: mean
Metropolis probability with its standard error, accepted count, posterior mean and
standard deviation per control, the Laplace standard deviation of the metric, their
ratio, the effective sample size, the number of accepted trajectories that cross the
density jump at 5700 km, the wall time and the gradient counts.

The problem
-----------
The node table is ``earth_model.prem_profile()`` (137 nodes, PREM density with jumps at
220, 400 and 670 km depth). Four layers: the lower mantle up to the movable boundary
``r_b``, ``r_b`` to 400 km depth (5970 km), 400 to 100 km depth, and the lid. The truth
has the viscosities 3e22, 1e21, 3e20 and 1e22 Pa s and ``r_b = 5690`` km (680 km depth,
10 km below the density jump). The density anomaly is the smooth anomaly of
``earth_model.smooth_density`` (seed 7) on the input table. The data are the geoid
coefficients of the truth for ``2 <= l <= lmax`` plus Gaussian noise of standard
deviation ``sigma_l`` = noise times the root mean square coefficient of degree ``l``
(``np.random.default_rng(11)``). The likelihood is ``spectral_misfit`` with the weights
``1 / (2 sigma_l^2)``, the negative Gaussian log-likelihood.

The range holds the 670 km density jump, so ``motion="auto"`` resolves to the viscosity
cut: the boundary moves freely across the density jump, and under the default
formulation (incompressible) the map from the radius to the geoid is continuously
differentiable everywhere, so a trajectory that crosses 5700 km pays no energy error for
the crossing.

The controls ``z = (y_0, y_1, y_2, y_3, u)``: ``y_j`` the natural log of the viscosity of
layer ``j`` in Pa s, and ``u`` the logistic coordinate of the radius,
``r = lo + (hi - lo) s(u)`` with ``s`` the logistic function and ``(lo, hi) =
layers.radius_bounds[0]``. HMC then runs on an unbounded space and every proposal lies
inside the clamp range of ``Layers``, where the radius derivative is not the zero of a
clamped radius. The prior of the radius is uniform on that clamp range, 0.1 % of the
width inside each range end (5467.378 to 5969.497 km for the default range), not on the
whole range.

The potential (minus the log posterior, up to a constant)::

    U(z) = spectral_misfit(N(z), N_obs, w, "real_4pi")
           + 0.5 ((mean(y) - m_true) / 0.05)^2 + 0.5 sum_j ((y_j - ln 1e21) / 2)^2
           + softplus(-u) + softplus(u)

The geoid does not change when every viscosity is multiplied by one factor, so the data
constrain only the ratios of the viscosities. The narrow Gaussian on the mean log
viscosity (standard deviation 0.05 about the mean of the truth) stands for an absolute
viscosity from another observation, for example glacial isostatic adjustment; without it
the posterior has a flat direction (a wide prior of standard deviation 2 on each layer
alone leaves a standard deviation of 1 in the mean). The last two terms are the uniform
prior of the radius in ``u``: ``log(s (1 - s)) = -softplus(-u) - softplus(u)``.

The MAP search and the second mode
----------------------------------
The mass matrix is the Gauss-Newton Hessian of ``U`` at the maximum of the posterior
(MAP), so the sampler needs the MAP first. ``find_map`` is a Levenberg-Marquardt search
on the Gauss-Newton metric ``M = 2 J^T J`` plus the prior Hessian (``J`` the Jacobian of
the weighted residuals, one ``jax.jacfwd``).

The posterior of this problem has a second mode: a local minimum of ``U`` with the
boundary at 5940.9 km (noise 0.2) or 5959.8 km (noise 0.02), 10 to 30 km below the 400 km
boundary; the thin layer between them is 500 to 5400 times less viscous than the truth,
the 400 to 100 km layer 300 to 670 times more viscous and the lower mantle 14 to 31 times
more viscous; ``U`` is 331 (noise 0.2) and 8553 (noise 0.02) against 228.6 at the mode of
the truth, and every eigenvalue of the Hessian of ``U`` there is positive (planner
measurement, NOTES/chunks/c5-integration/plan-evidence/r1_mode2.out). The second mode is a
property of the posterior of this problem, not of the solver. A search or a chain from
the isoviscous model finds it. The search therefore uses a continuation in the likelihood
weights (``find_map_continuation``): it starts at the isoviscous model (1e21 Pa s in all
four layers, the boundary at its construction radius) and runs five stages with the
weights of the noise fractions 5, 2, 1 and 0.5 times larger than the target noise and
then the target weights, each stage from the end of the one before. With small weights
the prior dominates ``U``, so the first stage moves the viscosities away from the
isoviscous model before the data can pull the boundary into the second mode; each later
stage starts at the minimum of a slightly less smoothed potential. In every measured case
(29 of 29) this reached the mode of the truth. A start for another problem needs the
same care: a continuation, or a check of the misfit at the start point.

The MAP gate (``check_map``): the metric norm of the gradient ``sqrt(g^T M^-1 g)`` is at
most 1e-5, the last stage ended before its iteration limit, and ``U(MAP) <= U(truth)``
(a point with ``U`` above the truth is not the mode that holds the truth; the example
knows the truth because the data are synthetic). If one condition fails, the script
stops with exit status 1 and draws no sample. Then the metric at the MAP must pass a
Cholesky factorisation and have a condition number of at most 1e8.

The sampler
-----------
The mass matrix is ``M`` at the MAP with the target weights. A transition draws the
momentum ``p0 = C xi`` (``C`` the Cholesky factor of ``M``, ``xi`` standard normal, so
``p0 ~ N(0, M)``), runs ``n`` leapfrog steps of size ``eps`` (a half kick, ``n`` drifts
``z <- z + eps M^-1 p`` each followed by a full kick ``p <- p - eps grad U``, and half of
the last kick taken back), and accepts the end point if and only if ``dH`` is finite and
``log(v) < -dH`` with ``v`` uniform on [0, 1) and ``dH = H(z1, p1) - H(z, p0)``,
``H = U + p M^-1 p / 2``. After a rejection the chain keeps its point, its ``U`` and its
gradient unchanged. The leapfrog steps of one trajectory are one ``lax.scan`` inside one
jitted function; the samples are a host loop of jitted transitions. The random numbers
of transition ``i`` come from ``jax.random.fold_in(PRNGKey(seed), i)``, split into the
key of the transition (momentum and accept draw) and the key of the step jitter, so a
seed repeats its summary and the draws do not depend on the state of the chain.

Step size: with the metric at the MAP the posterior is close to a unit Gaussian in the
whitened coordinates, a leapfrog is stable to a step near 2, and step 0.5 with 10 steps
gave a mean Metropolis probability of 0.96 with the largest ``dH`` 0.35 on this problem
(200 samples); step 1.0 gave 0.84 with the largest ``dH`` 6.6.

The effective sample size uses Sokal's automatic window with ``c = 5``
(``effective_sample_size``).

As a module: ``build(parse_args([...]))`` returns the ``Problem``; ``main(argv)`` runs
everything and returns the summary dict.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

import stokesjx as sj

_HERE = Path(__file__).resolve().parent
#: The default JSON summary (``examples/output/`` is in ``.gitignore``).
DEFAULT_OUT = _HERE / "output" / "hmc_summary.json"

KM = 1e3
#: The radius of the 670 km density jump of the table, m: the crossings of the summary
#: are counted at this radius.
DENSITY_JUMP = 5700.0 * KM
#: The fixed boundaries at 400 km and 100 km depth, the CMB and the surface, m.
R_CMB, R_400, R_100, R_SURFACE = 3480.0 * KM, 5970.0 * KM, 6270.0 * KM, 6370.0 * KM
#: The viscosities of the truth, ln(Pa s), lower mantle first.
Y_TRUE = np.log([3e22, 1e21, 3e20, 1e22])
#: The isoviscous start of the MAP search, ln(Pa s).
Y_ISO = np.full(4, np.log(1e21))
#: The prior: standard deviation of the mean log viscosity about the mean of the truth,
#: and the centre and the standard deviation of the wide Gaussian on every layer.
PRIOR_MEAN_SD = 0.05
PRIOR_LAYER_CENTRE = math.log(1e21)
PRIOR_LAYER_SD = 2.0
#: Seeds of the density anomaly and of the noise draw.
DENSITY_SEED = 7
NOISE_SEED = 11

# Constants of the MAP search (Levenberg-Marquardt on the Gauss-Newton metric).
#: The smallest non-zero damping; an accepted step that would take the damping below it
#: sets it to zero (pure Gauss-Newton).
LAM_MIN = 1e-3
#: The damping above which the search stops with status "stalled".
LAM_MAX = 1e8
#: Floor of the damping matrix ``diag(M)`` relative to its largest entry. The ``u`` row of
#: the metric goes to zero when the logistic map saturates, and the floor keeps the
#: damping matrix positive definite there.
DIAG_FLOOR = 1e-8
#: Convergence: the metric norm of the gradient ``sqrt(g^T M^-1 g)``, in the units of
#: ``sqrt(U)`` (nat^(1/2)); 1e-6 is far below the posterior width (1 in these units).
GTOL = 1e-6
#: The MAP gate on the metric norm of the gradient (measured 4.6e-7 at the run-A MAP).
GTOL_GATE = 1e-5
#: The largest accepted condition number of the metric at the MAP (measured 50 for the
#: default run and 90 for the range that ends at the density jump).
METRIC_COND_MAX = 1e8
#: The continuation: the stages use the weights of these noise fractions over the target
#: noise (the prior dominates at first), then the target weights.
LADDER = (5.0, 2.0, 1.0, 0.5)


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


def logit(x):
    """The inverse of the logistic function, ``log(x / (1 - x))`` for ``0 < x < 1``."""
    return float(np.log(x / (1.0 - x)))


def parse_args(argv=None) -> argparse.Namespace:
    """The command line of the module docstring. ``argv=None`` reads ``sys.argv``.
    ``--smoke`` replaces the table, ``lmax``, the step, the sample count and the leapfrog
    count by the smoke settings."""
    p = argparse.ArgumentParser(description="HMC over layer viscosities and a boundary radius with stokesjx.")
    p.add_argument("--lmax", type=int, default=20)
    p.add_argument("--reduced", action="store_true")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--leapfrog", type=int, default=10)
    p.add_argument("--step", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--motion", choices=("auto", "cut", "affine"), default="auto")
    p.add_argument("--range", type=float, nargs=2, default=(5466.875, 5970.0), metavar=("LO", "HI"),
                   help="movable range of the boundary, km")
    p.add_argument("--start-radius", type=float, default=5750.0, help="construction radius, km")
    p.add_argument("--true-radius", type=float, default=5690.0, help="radius of the truth, km")
    p.add_argument("--noise", type=float, default=0.2, help="noise over the rms coefficient per degree")
    p.add_argument("--smoke", action="store_true", help="a run of a few seconds")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    if args.smoke:
        args.reduced, args.lmax, args.step, args.samples, args.leapfrog = True, 4, 0.2, 8, 3
    return args


@dataclass(eq=False)
class Problem:
    """The HMC problem of one command line.

    Attributes
    ----------
    args
        The namespace of ``parse_args``.
    layers, config
        ``sj.Layers`` (four layers, boundary 1 movable) and ``sj.StokesConfig``.
    density
        ``(N_in, 2, lmax+1, lmax+1)`` kg/m^3, ``real_4pi``, on the input table.
    data
        ``(2, lmax+1, lmax+1)`` m: the observed geoid (truth plus noise).
    sigma_l
        ``(lmax+1,)`` m: the noise standard deviation per coefficient of degree ``l``.
    weights
        ``(lmax+1,)`` m^-2: ``1 / (2 sigma_l^2)`` for ``l >= 2``, 0 otherwise.
    mask
        ``(2, lmax+1, lmax+1)``: 1 on the valid data entries (``l >= 2``, ``m <= l``, no
        sine at ``m = 0``).
    lo, hi
        The clamp range of the radius, m (``layers.radius_bounds[0]``).
    m_true
        The mean of the true log viscosities, ln(Pa s).
    z_true, z_start
        The truth and the start of the MAP search in ``z``.
    value_and_grad
        Jitted ``(z, density, data, weights) -> (U, grad U)``.
    jacobian
        Jitted ``(z, density, data, weights) -> J``, the ``(n_res, 5)`` Jacobian of the
        weighted residuals.
    counts
        Host counters of the calls of ``value_and_grad`` and ``jacobian`` by the MAP
        search.
    """

    args: argparse.Namespace
    layers: sj.Layers
    config: sj.StokesConfig
    density: jax.Array
    data: jax.Array
    sigma_l: np.ndarray
    weights: jax.Array
    mask: np.ndarray
    lo: float
    hi: float
    m_true: float
    z_true: np.ndarray
    z_start: np.ndarray
    value_and_grad: Callable | None = None
    jacobian: Callable | None = None
    counts: dict = field(default_factory=lambda: {"value_and_grad": 0, "jacobian": 0})


def radius_of(u, problem):
    """The boundary radius of the logistic coordinate ``u``, m: ``lo + (hi - lo) s(u)``."""
    return problem.lo + (problem.hi - problem.lo) * jax.nn.sigmoid(u)


def to_controls(z, problem) -> dict:
    """The controls of ``Layers`` of a point ``z``: ``{"radius": (1,) m, "log_eta": (4,)
    ln(Pa s)}``."""
    z = jnp.asarray(z)
    return {"radius": jnp.reshape(radius_of(z[4], problem), (1,)), "log_eta": z[:4]}


def geoid(z, density, problem) -> jax.Array:
    """The geoid field ``N(z)`` of the density, ``(2, lmax+1, lmax+1)`` m, ``real_4pi``."""
    c = to_controls(z, problem)
    prof = problem.layers(problem.layers.full_boundaries(c["radius"]), c["log_eta"])
    return problem.layers.fields(sj.stokes_kernels(prof, problem.config), density).geoid


def potential(z, density, data, weights, problem) -> jax.Array:
    """``U(z)``, minus the log posterior up to a constant (module docstring), nat."""
    z = jnp.asarray(z)
    y, u = z[:4], z[4]
    misfit = sj.spectral_misfit(geoid(z, density, problem), data, weights, "real_4pi")
    prior = (0.5 * ((jnp.mean(y) - problem.m_true) / PRIOR_MEAN_SD) ** 2
             + 0.5 * jnp.sum(((y - PRIOR_LAYER_CENTRE) / PRIOR_LAYER_SD) ** 2))
    # Uniform prior of the radius on the clamp range, written in u: the density of u is
    # s(u) (1 - s(u)), and -log(s (1 - s)) = softplus(-u) + softplus(u).
    return misfit + prior + jax.nn.softplus(-u) + jax.nn.softplus(u)


def residuals(z, density, data, weights, problem) -> jax.Array:
    """The weighted residuals ``sqrt(w_l) (N(z) - N_obs)`` on the valid entries, raveled:
    ``|residuals|^2`` is the misfit term of ``U``."""
    r = jnp.sqrt(weights)[None, :, None] * (geoid(z, density, problem) - data) * jnp.asarray(problem.mask)
    return r.ravel()


def build(args, *, noise_free: bool = False) -> Problem:
    """The ``Problem`` of the command line ``args``.

    Parameters
    ----------
    args
        The namespace of ``parse_args``.
    noise_free
        ``True`` keeps ``sigma_l`` and the weights of the noise level of ``args`` and sets
        the noise draw to zero, so that the data are the geoid of the truth.
    """
    em = _earth_model()
    table = em.prem_profile(reduced=bool(args.reduced))
    low, high = (float(v) * KM for v in args.range)
    start = float(args.start_radius) * KM
    boundaries = np.array([R_CMB, start, R_400, R_100, R_SURFACE])
    layers = sj.Layers(table, boundaries, movable={1: (low, high)}, motion=args.motion)
    config = sj.StokesConfig(lmax=int(args.lmax))
    L = config.lmax
    density = jnp.asarray(em.smooth_density(np.asarray(table.radius), L, DENSITY_SEED))
    lo, hi = (float(v) for v in layers.radius_bounds[0])
    mask = np.zeros((2, L + 1, L + 1))
    for l in range(2, L + 1):
        mask[0, l, : l + 1] = 1.0
        mask[1, l, 1: l + 1] = 1.0
    r_true = float(args.true_radius) * KM
    if not lo < r_true < hi:
        raise ValueError(f"the true radius {r_true / KM} km is outside the clamp range ({lo / KM}, {hi / KM}) km")
    z_true = np.concatenate([Y_TRUE, [logit((r_true - lo) / (hi - lo))]])
    z_start = np.concatenate([Y_ISO, [logit((start - lo) / (hi - lo))]])
    P = Problem(args=args, layers=layers, config=config, density=density, data=None, sigma_l=None,
                weights=None, mask=mask, lo=lo, hi=hi, m_true=float(np.mean(Y_TRUE)), z_true=z_true,
                z_start=z_start)
    # The data: the geoid of the truth plus noise of sigma_l per valid coefficient.
    N_true = np.asarray(jax.jit(lambda z, d: geoid(z, d, P))(jnp.asarray(z_true), density))
    count = np.maximum(mask.sum(axis=(0, 2)), 1.0)
    rms = np.sqrt(np.sum(N_true ** 2 * mask, axis=(0, 2)) / count)                 # (L+1,) m
    sigma = float(args.noise) * rms
    weights = np.where(np.arange(L + 1) >= 2, 1.0 / (2.0 * np.where(sigma > 0, sigma, 1.0) ** 2), 0.0)
    xi = np.random.default_rng(NOISE_SEED).standard_normal(N_true.shape) * mask
    noise = 0.0 if noise_free else sigma[None, :, None] * xi
    P.data = jnp.asarray(N_true + noise)
    P.sigma_l = sigma
    P.weights = jnp.asarray(weights)
    # One compiled value and gradient and one compiled Jacobian per problem. The density,
    # the data and the weights are arguments, never closed-over constants.
    P.value_and_grad = jax.jit(jax.value_and_grad(lambda z, d, o, w: potential(z, d, o, w, P)))
    P.jacobian = jax.jit(jax.jacfwd(lambda z, d, o, w: residuals(z, d, o, w, P)))
    return P


def _value_and_grad(z, problem, weights):
    """Host call of the compiled value and gradient with the weights of a stage."""
    problem.counts["value_and_grad"] += 1
    U, g = problem.value_and_grad(jnp.asarray(z), problem.density, problem.data, weights)
    return float(U), np.asarray(g)


def gauss_newton_metric(z, density, data, weights, problem) -> np.ndarray:
    """The Gauss-Newton Hessian of ``U`` at ``z``, ``(5, 5)``.

    ``2 J^T J`` of the weighted residuals (``J`` from one ``jax.jacfwd``, a batched jvp
    over the five controls) plus the prior Hessian in closed form: ``(1 / 0.05^2) (1 / 16)
    1 1^T + I / 4`` in ``y`` and ``2 s(u) (1 - s(u))`` in ``u``. The misfit part is linear
    in ``weights`` and the prior part does not depend on them. The ``y`` block is
    positive definite; the ``u`` entry goes to zero when the logistic map saturates, so
    the caller must not invert ``M`` without a guard."""
    problem.counts["jacobian"] += 1
    J = np.asarray(problem.jacobian(jnp.asarray(z), density, data, weights))
    M = 2.0 * J.T @ J
    n = 4
    M[:n, :n] += (1.0 / PRIOR_MEAN_SD ** 2) / n ** 2 * np.ones((n, n)) + np.eye(n) / PRIOR_LAYER_SD ** 2
    s = 1.0 / (1.0 + np.exp(-float(z[4])))
    M[n, n] += 2.0 * s * (1.0 - s)
    return M


def metric_norm(M, g) -> float:
    """``sqrt(g^T M^-1 g)`` by a Cholesky factorisation of ``M``; ``inf`` when ``M`` is not
    positive definite."""
    try:
        c = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return math.inf
    v = np.linalg.solve(c, g)
    return float(np.sqrt(v @ v))


@dataclass
class MapStage:
    """The result of one stage of the MAP search.

    Attributes: ``z`` the end point, ``U`` the potential there (with the weights of the
    stage), ``n_iter`` the iterations taken, ``status`` (``"converged"``, ``"stalled"``
    or ``"max_iter"``), ``gnorm`` the metric norm of the gradient at the end point,
    ``history`` a list of ``(iteration, U, gnorm, lam, radius in km)``."""

    z: np.ndarray
    U: float
    n_iter: int
    status: str
    gnorm: float
    history: list


@dataclass
class MapResult:
    """The result of the MAP search with the target weights.

    Attributes: ``z``, ``U`` (target weights), ``status`` and ``gnorm`` of the last
    stage, ``iterations`` per stage, ``U_truth`` the potential at the truth with the same
    data and weights, ``stages`` the ``MapStage`` records."""

    z: np.ndarray
    U: float
    status: str
    gnorm: float
    iterations: list
    U_truth: float
    stages: list = field(default_factory=list)


class MapGateError(RuntimeError):
    """The MAP gate failed: the search did not end at the mode of the truth."""


def find_map(z0, problem, weights, max_iter: int = 50) -> MapStage:
    """One stage of the MAP search: Levenberg-Marquardt on the Gauss-Newton metric.

    Each iteration: ``U``, ``g`` and ``M`` at ``z`` with ``weights``; stop with
    ``"converged"`` when ``sqrt(g^T M^-1 g) < GTOL``. Otherwise solve ``(M + lam D) dz =
    -g`` by Cholesky with ``D = diag(max(M_ii, DIAG_FLOOR max_j M_jj))``; accept the step
    if ``U(z + dz)`` is finite and below ``U(z)`` and divide ``lam`` by 10 (0 below
    ``LAM_MIN``), else set ``lam = max(10 lam, LAM_MIN)`` and stop with ``"stalled"``
    above ``LAM_MAX``. A failed factorisation counts as a rejected step. ``lam`` starts at
    0. Stop with ``"max_iter"`` after ``max_iter`` iterations. Each iteration is one call
    of the compiled value and gradient and one of the compiled Jacobian; the loop is on
    the host."""
    w = jnp.asarray(weights)
    z = np.array(z0, dtype=np.float64)
    lam = 0.0
    Uz, g = _value_and_grad(z, problem, w)
    history = []
    gn = math.inf
    for it in range(1, max_iter + 1):
        M = gauss_newton_metric(z, problem.density, problem.data, w, problem)
        gn = metric_norm(M, g)
        history.append((it, Uz, gn, lam, float(radius_of(z[4], problem)) / KM))
        if gn < GTOL:
            return MapStage(z, Uz, it - 1, "converged", gn, history)
        d = np.diag(M)
        damp = np.diag(np.maximum(d, DIAG_FLOOR * d.max()))
        try:
            c = np.linalg.cholesky(M + lam * damp)
            dz = -np.linalg.solve(c.T, np.linalg.solve(c, g))
        except np.linalg.LinAlgError:
            lam = max(10.0 * lam, LAM_MIN)
            if lam > LAM_MAX:
                return MapStage(z, Uz, it, "stalled", gn, history)
            continue
        Un, gnew = _value_and_grad(z + dz, problem, w)
        if np.isfinite(Un) and Un < Uz:
            z, Uz, g = z + dz, Un, gnew
            lam = lam / 10.0 if lam / 10.0 >= LAM_MIN else 0.0
        else:
            lam = max(10.0 * lam, LAM_MIN)
            if lam > LAM_MAX:
                return MapStage(z, Uz, it, "stalled", gn, history)
    M = gauss_newton_metric(z, problem.density, problem.data, w, problem)
    return MapStage(z, Uz, max_iter, "max_iter", metric_norm(M, g), history)


def find_map_continuation(problem, ladder=LADDER, max_iter: int = 50) -> MapResult:
    """The MAP search with the continuation in the likelihood weights (module docstring).

    Start at the isoviscous model and the construction radius. Stage ``k`` uses the
    weights ``w_l (noise / s_k)^2`` for ``s_k`` in ``ladder``: the likelihood of the noise
    fraction ``s_k`` in place of the target noise fraction ``noise`` (for the default
    noise 0.2 the first stage weights the data 625 times less). The last stage uses the
    target weights; each stage starts at the end of the one before. ``ladder=()`` is one
    stage with the target weights from the isoviscous start.

    Parameters
    ----------
    problem
        The ``Problem``.
    ladder
        The noise fractions of the stages before the last.
    max_iter
        The iteration limit of each stage.
    """
    noise = float(problem.args.noise)
    z = np.array(problem.z_start, dtype=np.float64)
    stages = []
    for s in tuple(ladder) + (None,):
        w = problem.weights if s is None else problem.weights * (noise / s) ** 2
        st = find_map(z, problem, w, max_iter=max_iter)
        stages.append(st)
        z = st.z
    U_truth, _ = _value_and_grad(problem.z_true, problem, problem.weights)
    last = stages[-1]
    return MapResult(z=last.z, U=last.U, status=last.status, gnorm=last.gnorm,
                     iterations=[s.n_iter for s in stages], U_truth=U_truth, stages=stages)


def check_map(result, problem) -> None:
    """The MAP gate. Raises ``MapGateError`` with the measured values unless all three
    hold: ``result.gnorm <= GTOL_GATE``, the last stage ended before its iteration limit
    (status ``"converged"`` or ``"stalled"``), and ``result.U <= result.U_truth``."""
    bad = []
    if not result.gnorm <= GTOL_GATE:
        bad.append(f"metric norm of the gradient {result.gnorm:.3e} > {GTOL_GATE:.0e}")
    if result.status not in ("converged", "stalled"):
        bad.append(f"the last stage ended with status {result.status!r}")
    if not result.U <= result.U_truth:
        bad.append(f"U(MAP) = {result.U:.6f} > U(truth) = {result.U_truth:.6f}: not the mode of the truth")
    if bad:
        raise MapGateError("MAP gate failed (radius at the MAP "
                           f"{float(radius_of(result.z[4], problem)) / KM:.3f} km): " + "; ".join(bad))


def make_trajectory(problem, metric, n_leapfrog: int, *, value_and_grad=None) -> Callable:
    """The jitted leapfrog trajectory for a mass matrix ``metric``.

    Returns ``trajectory(z, g, p, eps, density, data, weights) -> (z1, p1, U1, g1, r_min,
    r_max)``: from ``z`` with the gradient ``g`` and the momentum ``p``, ``n_leapfrog``
    steps of size ``eps`` (a half kick, then ``n`` times a drift ``z <- z + eps M^-1 p`` and
    a full kick ``p <- p - eps grad U``, then half of the last kick taken back); ``U1``,
    ``g1`` at the end point; ``r_min`` and ``r_max`` the smallest and largest radius of the
    path, the start included, m. The steps are one ``lax.scan`` of static length.
    ``value_and_grad`` (default ``jax.value_and_grad`` of the potential of the problem)
    replaces the potential and its gradient, for a test with a biased gradient. The
    function closes over ``M^-1`` only, never over the density, the data or the
    weights."""
    Minv = jnp.asarray(np.linalg.inv(np.asarray(metric)))
    n = int(n_leapfrog)
    vg = value_and_grad if value_and_grad is not None else jax.value_and_grad(
        lambda z, d, o, w: potential(z, d, o, w, problem))

    def trajectory(z, g, p, eps, density, data, weights):
        r0 = radius_of(z[4], problem)
        p = p - 0.5 * eps * g

        def step(carry, _):
            z, p, _, _ = carry
            z = z + eps * (Minv @ p)
            U, g = vg(z, density, data, weights)
            p = p - eps * g
            return (z, p, U, g), radius_of(z[4], problem)
        (z, p, U, g), rs = jax.lax.scan(step, (z, p, jnp.zeros((), z.dtype), g), None, length=n)
        p = p + 0.5 * eps * g
        return z, p, U, g, jnp.minimum(r0, jnp.min(rs)), jnp.maximum(r0, jnp.max(rs))
    return jax.jit(trajectory)


def make_transition(problem, metric, n_leapfrog: int) -> Callable:
    """The jitted HMC transition for a mass matrix ``metric``.

    Returns ``transition(z, U, g, key, eps, density, data, weights) -> ((z, U, g), info)``.
    The key is split into the momentum key and the accept key; ``p0 = C xi`` with ``C`` the
    Cholesky factor of ``M``; the trajectory of ``make_trajectory``; ``dH = (U1 + p1 M^-1
    p1 / 2) - (U + p0 M^-1 p0 / 2)``. The proposal is accepted if and only if ``dH`` is
    finite and ``log(v) < -dH``, ``v`` uniform on [0, 1): every finite ``dH <= 0`` is
    accepted, a proposal with ``dH > 0`` with probability ``exp(-dH)``. After a rejection
    the returned state is the input ``(z, U, g)`` unchanged; after an acceptance it is
    ``(z1, U1, g1)``. ``info`` holds ``prob = min(1, exp(-dH))`` (0 for a non-finite
    ``dH``), ``accept``, ``dH``, ``p0``, ``p1``, ``z1``, ``U1``, ``g1``, ``r_min`` and
    ``r_max``."""
    M = np.asarray(metric)
    C = jnp.asarray(np.linalg.cholesky(M))
    Minv = jnp.asarray(np.linalg.inv(M))
    trajectory = make_trajectory(problem, M, n_leapfrog)

    def transition(z, U, g, key, eps, density, data, weights):
        k1, k2 = jax.random.split(key)
        p0 = C @ jax.random.normal(k1, (z.shape[0],), dtype=z.dtype)
        z1, p1, U1, g1, r_min, r_max = trajectory(z, g, p0, eps, density, data, weights)
        dH = (U1 + 0.5 * p1 @ (Minv @ p1)) - (U + 0.5 * p0 @ (Minv @ p0))
        finite = jnp.isfinite(dH)
        accept = finite & (jnp.log(jax.random.uniform(k2, dtype=z.dtype)) < -dH)
        state = (jnp.where(accept, z1, z), jnp.where(accept, U1, U), jnp.where(accept, g1, g))
        prob = jnp.where(finite, jnp.minimum(1.0, jnp.exp(-jnp.where(finite, dH, 0.0))), 0.0)
        info = dict(prob=prob, accept=accept, dH=dH, p0=p0, p1=p1, z1=z1, U1=U1, g1=g1, r_min=r_min, r_max=r_max)
        return state, info
    return jax.jit(transition)


def hmc(problem, z0, metric, eps, n_leapfrog, n_samples, seed, *, progress: bool = True) -> dict:
    """Run the chain: a host loop of ``n_samples`` jitted transitions from ``z0``.

    The step of transition ``i`` is ``eps`` times a uniform number in [0.8, 1.2] (a traced
    argument, no recompile). The random numbers of transition ``i`` come from
    ``jax.random.fold_in(jax.random.PRNGKey(seed), i)``, split into the key of the
    transition and the key of the step jitter.

    Returns a dict of NumPy arrays per transition: ``z`` (the state after the transition,
    ``(n, 5)``), ``U`` (the potential that the chain carries), ``p0`` (the start momentum),
    ``prob``, ``accept``, ``dH``, ``eps``, ``r_min``, ``r_max`` (m), and ``wall_time`` (s).
    """
    transition = make_transition(problem, metric, n_leapfrog)
    z = jnp.asarray(z0, dtype=jnp.float64)
    U, g = problem.value_and_grad(z, problem.density, problem.data, problem.weights)
    base = jax.random.PRNGKey(int(seed))
    rec = {k: [] for k in ("z", "U", "p0", "prob", "accept", "dH", "eps", "r_min", "r_max")}
    t0 = time.perf_counter()
    for i in range(int(n_samples)):
        k_tr, k_eps = jax.random.split(jax.random.fold_in(base, i))
        e = eps * jax.random.uniform(k_eps, dtype=jnp.float64, minval=0.8, maxval=1.2)
        (z, U, g), info = transition(z, U, g, k_tr, e, problem.density, problem.data, problem.weights)
        rec["z"].append(np.asarray(z))
        rec["U"].append(float(U))
        rec["eps"].append(float(e))
        for k in ("p0", "prob", "accept", "dH", "r_min", "r_max"):
            rec[k].append(np.asarray(info[k]))
        if progress and (i + 1) % 100 == 0:
            acc = np.mean(np.asarray(rec["accept"], dtype=float))
            print(f"  transition {i + 1:5d} of {n_samples}: accepted {acc:.3f}, "
                  f"{time.perf_counter() - t0:.1f} s", flush=True)
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["accept"] = out["accept"].astype(bool)
    out["wall_time"] = time.perf_counter() - t0
    return out


def effective_sample_size(x: np.ndarray, c: float = 5.0) -> float:
    """The effective sample size ``n / tau`` of a one-dimensional series.

    ``tau(W) = 1 + 2 sum_{t=1}^{W} rho(t)`` with the sample autocorrelation ``rho`` (FFT,
    the mean removed, the biased estimator), and Sokal's automatic window: the smallest
    ``W`` with ``W >= c tau(W)``. A series of zero variance returns 1.0 (the information of
    one distinct value); the window is not evaluated then. On a short or anticorrelated
    series ``tau`` can be near zero or negative, so the result is capped at
    ``n max(1, log10 n)`` (the cap of ArviZ for antithetic chains); on a long positively
    correlated series the cap does not bind.

    Parameters
    ----------
    x
        ``(n,)`` samples.
    c
        The window factor (Sokal recommends 4 to 10; 5 here).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    d = x - x.mean()
    var = float(d @ d) / n
    if n < 2 or var == 0.0:
        return 1.0
    f = np.fft.rfft(d, 2 * n)
    acov = np.fft.irfft(f * np.conj(f), 2 * n)[:n] / n
    rho = acov / acov[0]
    tau = 1.0 + 2.0 * np.cumsum(rho[1:])                     # tau(W) for W = 1 .. n-1
    W = np.arange(1, n)
    ok = W >= c * tau
    t = tau[np.argmax(ok)] if np.any(ok) else tau[-1]
    cap = n * max(1.0, math.log10(n))
    return float(cap if t <= n / cap else n / t)


def _json_float(x):
    """A float for the strict JSON writer: ``None`` for a non-finite value."""
    x = float(x)
    return x if math.isfinite(x) else None


def main(argv=None) -> dict:
    """Run the example: build, MAP continuation, MAP gate, metric check, chain, summary.

    Returns the summary dict and writes it as JSON to ``--out``. A failed MAP gate or a
    metric that fails its check stops the script with the message and exit status 1."""
    args = parse_args(argv)
    t_start = time.perf_counter()
    P = build(args)
    L = P.layers
    print(f"HMC example: {L.n_nodes} nodes, lmax {P.config.lmax}, motion {L.motion}, range "
          f"({args.range[0]}, {args.range[1]}) km, noise {args.noise}, step {args.step}", flush=True)
    t0 = time.perf_counter()
    res = find_map_continuation(P)
    t_map = time.perf_counter() - t0
    try:
        check_map(res, P)
    except MapGateError as err:
        raise SystemExit(f"hmc_layers: {err}")
    M = gauss_newton_metric(res.z, P.density, P.data, P.weights, P)
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        raise SystemExit("hmc_layers: the metric at the MAP is not positive definite")
    cond = float(np.linalg.cond(M))
    if not cond <= METRIC_COND_MAX:
        raise SystemExit(f"hmc_layers: the condition number of the metric at the MAP is {cond:.3e} > "
                         f"{METRIC_COND_MAX:.0e}")
    r_map = float(radius_of(res.z[4], P))
    print(f"MAP: stages {res.iterations}, status {res.status}, |g|_M {res.gnorm:.2e}, U {res.U:.4f} "
          f"(truth {res.U_truth:.4f}), radius {r_map / KM:.3f} km, cond(M) {cond:.1f}, {t_map:.1f} s", flush=True)

    chain = hmc(P, res.z, M, float(args.step), int(args.leapfrog), int(args.samples), int(args.seed))
    z = chain["z"]
    n = z.shape[0]
    Minv = np.linalg.inv(M)
    laplace_sd = np.sqrt(np.diag(Minv))
    mean = z.mean(axis=0)
    sd = z.std(axis=0, ddof=1) if n > 1 else np.zeros(5)
    # A control has moved when one sample differs from the MAP, the start of the chain.
    moved = [bool(np.any(z[:, k] != res.z[k])) for k in range(5)]
    ess = [effective_sample_size(z[:, k]) for k in range(5)]
    prob, accept, dH = chain["prob"], chain["accept"], chain["dH"]
    finite = np.isfinite(dH)
    crossings = int(np.sum(accept & (chain["r_min"] < DENSITY_JUMP) & (chain["r_max"] > DENSITY_JUMP)))
    radius_km = np.asarray(radius_of(jnp.asarray(z[:, 4]), P)) / KM
    names = ["log_eta[0]", "log_eta[1]", "log_eta[2]", "log_eta[3]", "u"]
    summary = {
        "settings": {
            "motion": L.motion, "range_km": [float(v) for v in args.range],
            "clamp_range_km": [P.lo / KM, P.hi / KM], "start_radius_km": float(args.start_radius),
            "true_radius_km": float(args.true_radius), "lmax": P.config.lmax, "n_nodes": int(L.n_nodes),
            "noise": float(args.noise), "step": float(args.step), "leapfrog": int(args.leapfrog),
            "samples": int(args.samples), "seed": int(args.seed), "smoke": bool(args.smoke),
        },
        "map": {
            "iterations": [int(i) for i in res.iterations], "status": res.status, "gnorm": res.gnorm,
            "U_map": res.U, "U_truth": res.U_truth, "cond_metric": cond, "gate_passed": True,
            "z": res.z.tolist(), "radius_km": r_map / KM, "wall_time": t_map,
        },
        "names": names,
        "z_true": P.z_true.tolist(),
        "samples_z": z.tolist(),
        "samples_viscosity_pa_s": np.exp(z[:, :4]).tolist(),
        "samples_radius_km": radius_km.tolist(),
        "prob": [float(v) for v in prob],
        "accept": [bool(v) for v in accept],
        "dH": [_json_float(v) for v in dH],
        "U": [float(v) for v in chain["U"]],
        "p0": chain["p0"].tolist(),
        "eps": [float(v) for v in chain["eps"]],
        "r_min_km": [float(v) / KM for v in chain["r_min"]],
        "r_max_km": [float(v) / KM for v in chain["r_max"]],
        "n_nonfinite_dH": int(np.sum(~finite)),
        "mean_prob": float(np.mean(prob)),
        "se_prob": float(np.std(prob, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "accepted": int(np.sum(accept)),
        "accept_rate": float(np.mean(accept)),
        "max_dH": _json_float(np.max(dH[finite])) if np.any(finite) else None,
        "mean_abs_dH": _json_float(np.mean(np.abs(dH[finite]))) if np.any(finite) else None,
        "posterior_mean": mean.tolist(),
        "posterior_sd": sd.tolist(),
        "laplace_sd": laplace_sd.tolist(),
        "sd_over_laplace": (sd / laplace_sd).tolist(),
        "ess": ess,
        "moved": moved,
        "crossings_5700km": crossings,
        "posterior_radius_km": {"mean": float(np.mean(radius_km)),
                                "sd": float(np.std(radius_km, ddof=1)) if n > 1 else 0.0},
        "gradients": {"chain": int(n * int(args.leapfrog)), "map_value_and_grad": P.counts["value_and_grad"],
                      "map_jacobian": P.counts["jacobian"]},
        "wall_time": {"chain": float(chain["wall_time"]), "total": time.perf_counter() - t_start},
    }
    print(f"chain: {n} transitions, mean Metropolis probability {summary['mean_prob']:.3f} "
          f"(se {summary['se_prob']:.3f}), accepted {summary['accepted']}, largest dH {summary['max_dH']}, "
          f"crossings of 5700 km {crossings}, {chain['wall_time']:.1f} s")
    for k in range(5):
        print(f"  {names[k]:>10s}: mean {mean[k]:9.4f} sd {sd[k]:.4f} (Laplace {laplace_sd[k]:.4f}), "
              f"truth {P.z_true[k]:9.4f}, ESS {ess[k]:.0f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, allow_nan=False))
    print(f"wrote {out}")
    return summary


if __name__ == "__main__":
    main()
