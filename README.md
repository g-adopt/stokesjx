# stokesjx

stokesjx computes the instantaneous viscous (Stokes) response of a spherically
symmetric, self-gravitating mantle to internal density anomalies: the geoid, the
dynamic topography of the surface and of the core-mantle boundary, and the surface
velocity. It is a JAX version of the Fortran propagator-matrix code CODE-A, with
derivatives with respect to the radial viscosity and the radii of viscosity jumps.

Status: pre-alpha. The repository holds the reference data, the high-precision
reference solver and the test framework. The solver itself does not exist yet.

Install (editable, with the test dependencies):

    pip install -e '.[test]'

Run the quick test set:

    python -m pytest -n 8
