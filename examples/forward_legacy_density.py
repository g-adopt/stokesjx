"""Forward run of stokesjx from a CODE-A density file: kernels, the four fields and the
output files of CODE-A.

Command line::

    python examples/forward_legacy_density.py [--sph PATH] [--viscosity PATH]
        [--lmax L] [--out DIR] [--grid]

Options:

``--sph PATH``
    A ``.sph`` density file in CODE-A's format (``stokesjx.io.legacy.read_sph``). The
    default is ``legacy/CODE-A/INPUT/DENSITY_INPUTS/LEBEDEV_S4RTS.sph`` of the
    repository (the S4RTS tomography converted to density, ``LMAX`` 50, 129 radii). The
    ``legacy/`` directory is not tracked; if the default file is absent, the script stops
    with a message that names ``--sph``.
``--viscosity PATH``
    A CODE-A viscosity file (one relative viscosity per line, surface first, in units of
    CODE-A's ``visc0 = 1e19 Pa s``; ``io.legacy.read_viscosity``) with one value per node
    of the density file. The default is the hager4 layering by depth
    (``earth_model.HAGER4_LAYERS``): each node takes the viscosity of the layer that holds
    it, a node on a layer bottom the layer below.
``--lmax L``
    Largest degree (default: ``LMAX`` of the file; at most that value).
``--out DIR``
    Output directory (default ``examples/output/forward`` beside this script).
``--grid``
    Also evaluate the four fields on a 1-degree grid.

Outputs in ``--out``:

- ``SPH_geoid``, ``SPH_surftopo``, ``SPH_cmbtopo``, ``SPH_surfvel``: the geoid (m), the
  surface and the CMB dynamic topography (m, outward positive) and the surface velocity
  (m/s, the coefficients of the RMS horizontal speed) as ``real_orthonormal`` coefficients
  in CODE-A's ``SPHCO`` format (``io.legacy.write_sphco``, the Fortran ``e15.6``), the
  file names of CODE-A;
- ``kernels_fields.npz``: the degrees, the radius (m), the four kernels ``(L, N)`` per
  kg/m^2 and the four fields ``(2, L+1, L+1)`` in ``real_4pi``;
- with ``--grid``, ``grids.npz``: the latitudes (89.5 to -89.5 degrees north), the
  longitudes (0.5 to 359.5 degrees east) and the four fields on these cell centres,
  shape ``(180, 360)`` each (``sh.synthesis_grid``).

The script prints the root mean square over the sphere of each field
(``sqrt(sum_l S(l))`` with the degree power ``S(l)`` of ``sh.degree_power``) and the
geoid degree power for ``l = 2 .. 10``.

Physics. The run uses the SI physics of the high layer: the reference density
``rho0`` is the "Averages" column of the file (as in CODE-A), the gravity is integrated
from ``rho0`` and a consistent core of 9900 kg/m^3 (the core of the golden Fortran runs),
the material of an interval is the mean of its two nodes, and the radial integral is the
trapezoid rule (``fields.apply``). CODE-A's own choices (a uniform ``g = 10`` m/s^2, the
material of the upper node, CODE-A's quadrature weights) are test-only switches of the
low layer and are not offered here. The test that compares the package under these
switches with the Fortran output is ``tests/test_fields.py::test_fields_against_fortran``.
Because the physics differs, the fields of this script differ from CODE-A's; the
geoid coefficients of the LEBEDEV file correlate per degree with the Fortran geoid at
every degree from 2 to 50 (test ``tests/test_examples.py::test_forward_example_lebedev``).

As a module: ``run(parse_args([...]))`` returns ``{"profile", "kernels", "fields",
"files"}``; ``main(argv)`` also prints the summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import numpy as np

import stokesjx as sj
from stokesjx import sh
from stokesjx.io import legacy

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
#: The default density file: CODE-A's S4RTS density, not tracked (``legacy/``).
DEFAULT_SPH = _REPO / "legacy" / "CODE-A" / "INPUT" / "DENSITY_INPUTS" / "LEBEDEV_S4RTS.sph"
#: The default output directory (``examples/output/`` is in ``.gitignore``).
DEFAULT_OUT = _HERE / "output" / "forward"
#: The core density of the golden Fortran runs, kg/m^3, as a consistent core.
CORE_DENSITY = 9900.0
#: CODE-A's names of the field files, in the order of ``fields.FIELD_NAMES``.
SPHCO_NAMES = {"geoid": "SPH_geoid", "surface_topography": "SPH_surftopo",
               "cmb_topography": "SPH_cmbtopo", "surface_velocity": "SPH_surfvel"}


def _earth_model():
    """The module ``examples/earth_model.py``, loaded by its path, so that this script
    runs from any working directory and needs no ``sys.path`` change."""
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
    p = argparse.ArgumentParser(description="Forward run of stokesjx from a CODE-A density file.")
    p.add_argument("--sph", type=Path, default=DEFAULT_SPH, help="the .sph density file")
    p.add_argument("--viscosity", type=Path, default=None,
                   help="CODE-A viscosity file (surface first, units of 1e19 Pa s); default hager4 by depth")
    p.add_argument("--lmax", type=int, default=None, help="largest degree (default: LMAX of the file)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    p.add_argument("--grid", action="store_true", help="also write the fields on a 1-degree grid")
    return p.parse_args(argv)


def hager4_viscosity(radius) -> np.ndarray:
    """The hager4 viscosity of single nodes by depth, Pa s (``earth_model.HAGER4_LAYERS``;
    a node on a layer bottom belongs to the layer below)."""
    em = _earth_model()
    r = np.asarray(radius, dtype=np.float64)
    return em._layer_viscosity(r, np.zeros(r.shape, dtype=int), em.HAGER4_LAYERS)


def _kernels_and_fields_impl(profile, density, config):
    """The kernels of ``profile`` and the fields of ``density``. The density is an
    argument of the compiled program, not a closed-over constant (compile time,
    ``fields.apply``)."""
    K = sj.stokes_kernels(profile, config)
    return K, sj.apply(K, density, profile)


#: One compiled program of the profile and the density; ``config`` is static (a frozen,
#: hashable ``StokesConfig``).
_kernels_and_fields = jax.jit(_kernels_and_fields_impl, static_argnums=2)


def run(args) -> dict:
    """Read the files, compute the kernels and the fields, write the outputs.

    Parameters
    ----------
    args
        The namespace of ``parse_args``.

    Returns
    -------
    ``{"profile": RadialProfile, "kernels": Kernels, "fields": Fields, "files": list of
    Path}``; the fields are ``real_4pi``.

    Raises
    ------
    FileNotFoundError
        If the density file (or the viscosity file) does not exist.
    ValueError
        If ``--lmax`` exceeds ``LMAX`` of the file or is below 1.
    """
    sph_path = Path(args.sph)
    if not sph_path.is_file():
        raise FileNotFoundError(f"the density file {sph_path} does not exist; the default file is in the untracked "
                                f"legacy/ directory, so pass a CODE-A .sph file with --sph PATH")
    dens = legacy.read_sph(sph_path)
    lmax = dens.lmax if args.lmax is None else int(args.lmax)
    if not 1 <= lmax <= dens.lmax:
        raise ValueError(f"--lmax must be in 1 .. {dens.lmax} (LMAX of the file), got {lmax}")
    n = dens.radius.size
    if args.viscosity is not None:
        if not Path(args.viscosity).is_file():
            raise FileNotFoundError(f"the viscosity file {args.viscosity} does not exist")
        eta = legacy.read_viscosity(args.viscosity, n)                  # Pa s, CMB first
    else:
        eta = hager4_viscosity(dens.radius)
    profile = sj.RadialProfile(dens.radius, dens.average, eta, core_density=CORE_DENSITY)
    config = sj.StokesConfig(lmax=lmax)
    density = np.ascontiguousarray(dens.coefficients[..., : lmax + 1, : lmax + 1])
    K, F = _kernels_and_fields(profile, density, config)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for name, fname in SPHCO_NAMES.items():
        coef = np.asarray(sh.convert(getattr(F, name), "real_4pi", "real_orthonormal"))
        path = out / fname
        legacy.write_sphco(path, coef)
        files.append(path)
    npz = out / "kernels_fields.npz"
    np.savez(npz, degrees=np.asarray(K.degrees), radius=np.asarray(K.radius),
             **{f"kernel_{o}": np.asarray(getattr(K, o)) for o in SPHCO_NAMES},
             **{f"field_{o}": np.asarray(getattr(F, o)) for o in SPHCO_NAMES})
    files.append(npz)
    if args.grid:
        lat = np.linspace(89.5, -89.5, 180)
        lon = np.linspace(0.5, 359.5, 360)
        grids = {f"grid_{o}": np.asarray(sh.synthesis_grid(getattr(F, o), "real_4pi", lat, lon))
                 for o in SPHCO_NAMES}
        gpath = out / "grids.npz"
        np.savez(gpath, latitude=lat, longitude=lon, **grids)
        files.append(gpath)
    return {"profile": profile, "kernels": K, "fields": F, "files": files}


def main(argv=None) -> dict:
    """Parse the command line, run, print the summary; returns the dict of ``run``. A
    missing input file or a wrong ``--lmax`` stops the script with the message and exit
    status 1."""
    args = parse_args(argv)
    try:
        result = run(args)
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"forward_legacy_density: {err}")
    F = result["fields"]
    units = {"geoid": "m", "surface_topography": "m", "cmb_topography": "m", "surface_velocity": "m/s"}
    print(f"forward run: {Path(args.sph).name}, {result['profile'].radius.shape[0]} nodes, "
          f"lmax {result['kernels'].degrees[-1]}")
    for name in SPHCO_NAMES:
        power = np.asarray(sh.degree_power(getattr(F, name), getattr(F, name), "real_4pi"))
        print(f"  rms {name:20s} {np.sqrt(np.sum(power)):.4e} {units[name]}")
    power = np.asarray(sh.degree_power(F.geoid, F.geoid, "real_4pi"))
    print("  geoid degree power, m^2: " + ", ".join(f"l={l}: {power[l]:.3e}" for l in range(2, min(10, power.size - 1) + 1)))
    print(f"  wrote {len(result['files'])} files to {Path(args.out)}")
    return result


if __name__ == "__main__":
    main()
