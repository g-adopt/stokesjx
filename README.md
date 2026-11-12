# stokesjx

stokesjx computes the instantaneous viscous (Stokes) response of a spherically symmetric,
self-gravitating mantle to internal density anomalies. The outputs are the geoid, the
dynamic topography of the surface and of the core-mantle boundary (CMB), and the surface
velocity. For each spherical-harmonic degree `l` and each source radius, the package
computes a kernel. A field is the radial integral of a kernel times a density anomaly.

stokesjx is a JAX version of the Fortran propagator-matrix code CODE-A. The package
gives values and derivatives with respect to the radial viscosity and the radii of
viscosity jumps. The derivatives are for Hamiltonian Monte Carlo (HMC) over the radial
viscosity. The package supplies values and derivatives only. It does not choose a
sampler or a prior.

Documentation: <https://gadopt.org/stokesjx/>

## Status

The solver, the physics flags, the fields, the control maps and the derivatives are
complete and tested. This work is open:

- The benchmark against G-ADOPT is not done.
- The anelastic liquid approximation (ALA) and a partial-slip boundary are not implemented.
- The comparisons against published values are not done.

## Install

Install the package in editable mode with the test dependencies:

    pip install -e ".[test]"

The tests against the analytical solutions of Kramer, Davies and Wilson (2021) need the
package `assess`. To install it, add the extra `reference`:

    pip install -e ".[test,reference]"

The package sets `jax_enable_x64` at import. All arrays are float64.

## Quick start

One forward call:

```python
import numpy as np
import stokesjx as sj

radius = np.linspace(3480e3, 6370e3, 129)                  # m, CMB first
rho0 = np.linspace(5500.0, 3300.0, 129)                     # kg/m^3
eta = np.where(radius > 5700e3, 1e21, 1e22)                 # Pa s
profile = sj.RadialProfile(radius, rho0, eta, core_density=9900.0)
config = sj.StokesConfig(lmax=50)

K = sj.stokes_kernels(profile, config)                      # K.geoid: (50, 129), m per kg/m^2
density = np.zeros((129, 2, 51, 51))                        # kg/m^3, real_4pi, CMB first
density[:, 0, 2, 0] = 10.0
F = sj.apply(K, density, profile)                           # F.geoid: (2, 51, 51), m
```

One gradient through `Layers` and `ControlVector`. Boundary 1 moves inside the range
(5466.875, 5970) km. The range ends must be node radii of the table.

```python
import jax.numpy as jnp

boundaries = np.array([3480e3, 5750e3, 5970e3, 6270e3, 6370e3])
table = sj.RadialProfile(np.linspace(3480e3, 6370e3, 129), rho0, eta, core_density=9900.0)
layers = sj.Layers(table, boundaries, movable={1: (5466.875e3, 5970e3)})
cfg = sj.StokesConfig(lmax=20)
weights = np.where(np.arange(21) >= 2, 1.0, 0.0)

def misfit(c, density, observed):
    prof = layers(layers.full_boundaries(c["radius"]), c["log_eta"])
    N = layers.fields(sj.stokes_kernels(prof, cfg), density).geoid
    return sj.spectral_misfit(N, observed, weights, "real_4pi")

cv = sj.ControlVector({"radius": layers.start_radii, "log_eta": np.log([1e22, 1e21, 1e21, 1e21])})
x0 = cv.ravel({"radius": jnp.asarray(layers.start_radii), "log_eta": jnp.log(jnp.array([1e22, 1e21, 1e21, 1e21]))})
value, gradient = cv.value_and_grad(misfit)(x0, density[:, :, :21, :21], np.zeros((2, 21, 21)))
```

The density of `layers.fields` is given on the input table. It does not move when a
boundary moves.

## Examples

The directory `examples/` holds three scripts. Each script writes its output to `--out`
(default: `examples/output/`, which is not tracked).

- `examples/forward_legacy_density.py`: a forward run from a CODE-A density file
  (`.sph`). Outputs: the four fields in CODE-A's `SPHCO` format, an npz file with the
  kernels and the fields, and with `--grid` the fields on a 1-degree grid. The default
  density file `legacy/CODE-A/INPUT/DENSITY_INPUTS/LEBEDEV_S4RTS.sph` is not distributed
  with the repository. If you do not have it, give a file with `--sph PATH`.

      python examples/forward_legacy_density.py --sph PATH --out DIR

- `examples/gradient_layers.py`: the gradient of a geoid misfit with respect to four
  layer viscosities and one boundary radius, against central differences. Output: a JSON
  summary.

      python examples/gradient_layers.py --reduced --lmax 6

- `examples/hmc_layers.py`: HMC with a hand-written leapfrog over four layer viscosities
  and one movable boundary across the 670 km density jump, against a synthetic geoid.
  Output: a JSON summary with the samples, the energy errors and the effective sample
  size. The default run (1000 samples, `lmax = 20`) takes about 130 s on a Mac Studio.
  `--smoke` takes about 3 s.

      python examples/hmc_layers.py --smoke

`examples/earth_model.py` builds the PREM-like node table of the examples and the
benchmark.

## Test

Run every test:

    python -m pytest -n 8

The markers select tests by hand: `-m slow` (the tests above about 30 s), `-m "not slow"`,
`-m "not heavy"` (without the tests of `tests/heavy_tests.txt`) and `-m "not legacy"`.

Measured on reunion (Mac Studio, 32 CPUs) with `-n 8` on 2026-11-12: 1101 tests pass in
339 s (1-minute load average 6.0 at the start and 10.0 at the end).

The tests that need the directory `legacy/` (the CODE-A copy, not tracked) skip without
it. The 13 tests that compare with lovejx skip when lovejx is not installed (it is not on
PyPI). CI runs every test that needs neither `legacy/` nor lovejx, and skips 25 tests.

## Validation

The package is compared with these references. For each one, the test bound comes first
and the measured agreement second. The error metric is `e_l`: per degree, the largest
difference over the sources divided by the largest reference value.

| Reference | Bound | Measured |
|---|---|---|
| CODE-A golden output (Fortran, legacy switches: `g = 10`, upper-node material, CODE-A's weights), all 50 degrees | `1e-12` | 2.7e-14 (hager4), 1.8e-13 (smooth viscosity) |
| mpmath oracle at 50 or more digits, degrees 1 to 200 | `1e-12` | 8.5e-14 |
| Analytical solutions of Kramer, Davies and Wilson (2021) through `assess` | `1e-12` | 1.2e-14 |
| Viscosity cut against the mpmath oracle (`ORACLE_TOL` of `tests/test_controls_cut.py`) | `1e-12` | 1e-13 |
| Taylor tests of the derivatives, both direction signs, forward and reverse mode | corrected rate at least 1.9 | 2.00 |
| jvp/vjp dot products | `1e-12` | 1e-14 |

stokesjx fixes these errors and hacks of CODE-A:

| CODE-A | stokesjx |
|---|---|
| `grav()` returns 10 m/s^2 everywhere. | The gravity is integrated from `rho0` and the core mass. A uniform `g` is a flag. |
| The free-air kernel uses `G M / R^2`, everything else 10 m/s^2. | The gravity anomaly comes from the geoid and the surface `g` of the profile. |
| The viscosity and the density of a layer are those of its upper node. | The material of an interval is the mean of its two nodes. A jump is a node pair. |
| The radial integral gives every node the same weight. | The trapezoid weights of the node table. |
| The degree-1 geoid kernel is set to zero; the other degree-1 kernels are not. | Degree 1 is in the centre-of-mass frame for all outputs. |
| The Sylvester formula removes close eigenvalues; 128-bit reals for one product of propagators. | Exact interval exponentials and two orthonormalised sweeps per degree, stable in float64 at every degree. |
| The admittance kernel is a ratio of two kernels. | The admittance comes from the field coefficients. |
| The `SPHCO` header says "Cnm, Snm(m<0)", but each line holds C and then S. | The readers parse the data order. |
| The compressible term is a one-sided difference of `rho0`. | The TALA term comes from the same interval data as the rest. |

## Performance

| size | nodes | lmax | Pade order/squarings | forward ms | fields ms | gradient ms | gradient_all ms | vmap4 / gradient | compile forward s | compile gradient s | RSS gradient GB | RSS vmap4 GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P50 | 137 | 50 | 9/0 | 8.5 | 8.6 | 28.4 | 28.6 | 2.94 | 0.36 | 0.89 | 0.70 | 1.27 |
| P100 | 265 | 100 | 9/0 | 30.4 | 31.3 | 94.7 | 95.5 | 3.28 | 0.33 | 0.89 | 1.35 | 3.58 |
| P200 | 521 | 200 | 9/0 | 100.3 | 107.1 | 301.3 | 306.3 | 4.52 | 0.31 | 0.85 | 4.04 | 12.68 |
| P200c | 137 | 200 | 9/2 | 35.6 | 37.5 | 112.4 | 114.3 | 3.30 | 0.31 | 0.86 | 1.82 | 5.30 |

Measured 2026-11-11T23:26:46 on reunion (arm64), JAX 0.11.1, with the timing script of the working notes (not in the repository); 1-minute load average 6.5 to 17.6. Run time: median of the calls of all rounds. Compile: protocol "first", median of three fresh processes. RSS: peak resident set of a fresh process that compiles and runs the program once.

## Notes for samplers

The geoid and the topographies do not change when all viscosities are multiplied by one
factor. The data therefore constrain only the ratios of the viscosities. A sampler over
all layer viscosities needs a prior on the mean log viscosity or a fixed layer. A wide
prior on each layer alone leaves the uniform scaling as the widest direction of the
posterior. The HMC example uses a narrow Gaussian on the mean log viscosity.

`Layers` clamps a traced movable radius to `Layers.radius_bounds`, 0.1 % of the width
inside each range end. Outside these bounds the radius derivative is zero. The prior of
a radius must have its support inside these bounds. The HMC example samples the radius
through a logistic map onto the bounds.

Under `motion="cut"`, when the viscosities on both sides of a movable boundary are equal,
the position of the boundary has no effect on the kernels, and the radius derivative is
exactly zero. Under `motion="affine"` the derivative is small but not zero, because the
moved nodes change the density table. A start point or a base point of a test needs a
viscosity contrast across every movable boundary.

A movable boundary moves by one of two methods (`motion`). The default for every range
is `"cut"`. No node moves: the boundary is a viscosity cut inside one interval, and it can
cross the density jumps of the table. Under an incompressible formulation the cut has no
kinks. With `motion="affine"` the nodes of the range follow the boundary, and the radius
derivative has small jumps ("kinks") where a moved node crosses an input radius. For the
affine method, put the starting radius at or above the middle of its range.

The density jumps of `rho0` are phase changes. Under TALA, `rho0 u_r` is therefore
continuous across a density jump, and `u_r` jumps. Under TALA the kernels jump when a
viscosity cut crosses a density jump. This is a property of the discrete model.
`stokes_kernels` then gives a `RuntimeWarning` on every call. Usage rule: under TALA, when
the radius is sampled with HMC, end the movable range at the density jump.

The density of a field is given on the input table of `Layers` (`Layers.fields`). It
does not move with a boundary.

The posterior of the HMC example has a second mode. In it, the boundary is 10 to 30 km
below the 400 km boundary, the thin layer between them is 500 to 5400 times less viscous
than the truth, and the misfit is larger than at the mode of the truth (`U` = 331 against
228.6 at the noise level 0.2). This is a property of the posterior of this problem, not of
the solver. A search from the isoviscous model finds the second mode. The example
therefore uses a continuation in the likelihood weights from the isoviscous model, and it
stops when the end point has a larger `U` than the truth. A search or a chain for another
problem needs a start in the basin of the correct mode.

The HMC script is an example only: the inversion that the package serves is joint with
lovejx, and the settings of the example (noise 0.2, step 0.5, the continuation ladder)
belong to this synthetic problem.

## Credit

CODE-A was written and developed by Siavash Ghelichkhan at Ludwig-Maximilians-Universität
München. The source is the `READ.ME` of the repository that holds CODE-A ("CODE-A: is the
main code, written and developed by SiaG", signed Siavash Ghelichkhan, 26 September
2016). The wrapper scripts of CODE-A carry the same name and the LMU address (30 June
2017).

The method of CODE-A comes from Corrieu, Thoraval and Ricard (1995), Hager and Clayton
(1989), Forte and Peltier (1991) for the compressible terms, and Ricard, Fleitout and
Froidevaux (1984) and Richards and Hager (1984) for the propagator formulation.

The tests use the analytical solutions of Kramer, Davies and Wilson (2021, Geoscientific
Model Development) through `assess`, and the PREM density of Dziewonski and Anderson
(1981). The layer parameterisation of the controls, the gravity integration, the unit
scales, the spherical-harmonic conventions and the Taylor-test policy are ported from
lovejx. CODE-A uses the quad-precision LINPACK routines of John Burkardt. See the file
`NOTICE`.

## Agentic coding

Agentic coding tools wrote this software under the direction and review of the author.
[The documentation](https://gadopt.org/stokesjx/agentic-coding.html) states what the
tools wrote, what the author decided, and which checks the result must pass.

## Layout

- `stokesjx/`: the package.
  - `__init__.py`: the public names, float64 at import.
  - `units.py`: constants and the density scale.
  - `profile.py`: `RadialProfile`, the node structure, the gravity integral.
  - `formulation.py`: the physics flags and the boundary classes.
  - `system.py`: the 6x6 matrix in `ln r`, the interval exponents, the jump vectors.
  - `boundary.py`: the start bases and the output rows at the surface and the CMB.
  - `linalg.py`: the batched Pade exponentials and the node solvers.
  - `propagate.py`: the two sweeps and the node solve (the low layer).
  - `kernels.py`: `StokesConfig`, `Kernels`, `stokes_kernels`.
  - `fields.py`: the radial integral, the misfit, the gravity anomaly, the admittance.
  - `controls.py`: `Layers`, `viscosity_by_node`, `ControlVector`.
  - `sh.py`: the spherical-harmonic conventions and the grid synthesis.
  - `io/legacy.py`: the file formats of CODE-A.
- `examples/`: the three examples and the node table of the examples.
- `docs/`: the Quarto documentation site (`make docs`).
- `.github/`: the CI and documentation workflows.
- `tests/`: the test suite, with the golden reference data in `tests/golden/`.

The directories `NOTES/` (working notes) and `legacy/` (the CODE-A copy) are not tracked.

## Licence

Apache-2.0. See the files `LICENSE` and `NOTICE`.
