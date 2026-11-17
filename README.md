# stokesjx

stokesjx computes the instantaneous viscous (Stokes) response of a spherically symmetric,
self-gravitating mantle to internal density anomalies. The outputs are the geoid, the
dynamic topography of the surface and of the core-mantle boundary (CMB), and the surface
velocity. For each spherical-harmonic degree `l` and each source radius, the package
computes a kernel. A field is the radial integral of a kernel times a density anomaly.

stokesjx is a propagator-matrix code in JAX. It gives values and derivatives with respect
to the radial viscosity and the radii of viscosity jumps. The derivatives are for
Hamiltonian Monte Carlo (HMC) over the radial viscosity. The package supplies values and
derivatives only. It does not choose a sampler or a prior.

Documentation: <https://gadopt.org/stokesjx/>

## Install

Install the package from PyPI:

    pip install stokesjx

To change the code or to run the tests, clone the repository and install the package in
editable mode with the test dependencies:

    git clone https://github.com/g-adopt/stokesjx
    cd stokesjx
    pip install -e ".[test,reference]"

The extra `reference` installs `assess`, for the tests against the analytical solutions
of Kramer, Davies and Wilson (2021). The package sets `jax_enable_x64` at import. All
arrays are float64.

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

- `examples/forward_legacy_density.py`: a forward run from a `.sph` density file. Outputs:
  the four fields as `SPHCO` coefficient files, an npz file with the kernels and the
  fields, and with `--grid` the fields on a 1-degree grid.

      python examples/forward_legacy_density.py --sph PATH --out DIR

- `examples/gradient_layers.py`: the gradient of a geoid misfit with respect to four
  layer viscosities and one boundary radius, against central differences.

      python examples/gradient_layers.py --reduced --lmax 6

- `examples/hmc_layers.py`: HMC with a hand-written leapfrog over four layer viscosities
  and one movable boundary across the 670 km density jump, against a synthetic geoid.

      python examples/hmc_layers.py --smoke

`examples/earth_model.py` builds the PREM-like node table of the examples.

## Tests

From an editable install, run every test:

    python -m pytest -n 8

The marker `slow` marks the tests above about 30 s. Use `-m "not slow"` for a quick run.
The tests that need local reference data or lovejx skip when the reference is not
installed.

## Validation

stokesjx is benchmarked against the analytical solutions of Kramer, Davies and Wilson
(2021) through `assess` (bound `1e-12`, measured 1.2e-14), and it follows the
propagator-matrix formulation of Hager and O'Connell (1981). A second solver in mpmath at
50 to 250 digits and Taylor tests of every derivative check the rest. The validation page
of the documentation gives the detail: <https://gadopt.org/stokesjx/validation.html>

## Citation

If you use stokesjx, cite the software and the method papers. The file `CITATION.cff`
holds the citation of the software, and the citation page of the documentation lists the
papers: <https://gadopt.org/stokesjx/citation.html>

## Author

stokesjx is developed by Sia Ghelichkhan (<https://sia-g.com>, ORCID
[0000-0002-1316-3170](https://orcid.org/0000-0002-1316-3170)). The layer parameterisation
of the controls, the gravity integration, the unit scales, the spherical-harmonic
conventions and the Taylor-test policy are ported from lovejx (Sia Ghelichkhan, Jerry X.
Mitrovica, William Scott and Mark Hoggard). The tests use the analytical solutions of
Kramer, Davies and Wilson (2021) through `assess`, and the PREM density of Dziewonski and
Anderson (1981). See the file `NOTICE`.

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
  - `io/legacy.py`: the `.sph`, viscosity and `SPHCO` file formats.
- `examples/`: the three examples and the node table of the examples.
- `docs/`: the Quarto documentation site (`make docs`).
- `.github/`: the CI, documentation and release workflows.
- `tests/`: the test suite, with the golden reference data in `tests/golden/`.

## Licence

Apache-2.0. See the files `LICENSE` and `NOTICE`.
