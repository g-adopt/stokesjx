"""Shared fixtures of the stokesjx test suite.

Two rules for this file, as in lovejx:

1. It imports only numpy and pytest at module level, so that collection works in an
   environment without JAX or mpmath. Every other import happens inside a function.
2. It holds no assertions. A helper that fails is a broken fixture, not a failed
   gate.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden"
LEGACY = REPO / "legacy" / "CODE-A"
# The read-only original of CODE-A, used only by the regeneration test. The path
# comes from the environment variable STOKESJX_CODE_A_SOURCE; without it, the
# default is ~/Workplace/geoid_semianl/CODE-A. The test skips when the directory is
# absent.
CODE_A_SOURCE = Path(os.environ.get("STOKESJX_CODE_A_SOURCE",
                                    str(Path.home() / "Workplace" / "geoid_semianl" / "CODE-A")))

# --------------------------------------------------------------------------
# The `heavy` marker.
#
# tests/heavy_tests.txt lists the test files and test functions that take more than
# about 2 s each. They carry the `heavy` marker, and the fast tier
# `-m "not slow and not heavy"` leaves them out. An entry matches every
# parametrisation of a function, because the node id is compared without its
# `[...]` suffix.
# --------------------------------------------------------------------------

_HEAVY_LIST = Path(__file__).resolve().parent / "heavy_tests.txt"


def _heavy_entries() -> frozenset[str]:
    """The entries of tests/heavy_tests.txt: file paths and `file::function` ids."""
    if not _HEAVY_LIST.exists():
        return frozenset()
    lines = (line.strip() for line in _HEAVY_LIST.read_text().splitlines())
    return frozenset(line for line in lines if line and not line.startswith("#"))


def pytest_collection_modifyitems(config, items):
    """Add the `heavy` marker to every item whose file or function is listed."""
    heavy = _heavy_entries()
    if not heavy:
        return
    for item in items:
        function = item.nodeid.split("[", 1)[0]
        if function in heavy or function.split("::", 1)[0] in heavy:
            item.add_marker(pytest.mark.heavy)


# --------------------------------------------------------------------------
# Golden data.
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """tests/golden."""
    return GOLDEN


@pytest.fixture(scope="session")
def manifest() -> dict:
    """tests/golden/manifest.json."""
    return json.loads((GOLDEN / "manifest.json").read_text())


@pytest.fixture(scope="session")
def load_fortran():
    """Loader of tests/golden/fortran/<run>.npz as a dict of arrays."""
    cache = {}

    def load(run: str) -> dict:
        if run not in cache:
            with np.load(GOLDEN / "fortran" / f"{run}.npz") as d:
                cache[run] = {k: d[k] for k in d.files}
        return cache[run]
    return load


@pytest.fixture(scope="session")
def load_oracle():
    """Loader of tests/golden/oracle/<case>.npz as a dict of arrays."""
    cache = {}

    def load(case: str) -> dict:
        if case not in cache:
            with np.load(GOLDEN / "oracle" / f"{case}.npz") as d:
                cache[case] = {k: d[k] for k in d.files}
        return cache[case]
    return load


@pytest.fixture(scope="session")
def legacy_dir() -> Path:
    """legacy/CODE-A (the untracked copy of the Fortran and its historical output).
    Skips when absent."""
    if not (LEGACY / "test_4_ishan" / "KERNELS").is_dir():
        pytest.skip("legacy/CODE-A with test_4_ishan is not present")
    return LEGACY


@pytest.fixture(scope="session")
def legacy_toolchain(legacy_dir) -> Path:
    """The Fortran toolchain and the read-only CODE-A original. Skips when absent."""
    if shutil.which("mpif90") is None and not Path("/opt/homebrew/bin/mpif90").exists():
        pytest.skip("mpif90 is not installed")
    if not CODE_A_SOURCE.is_dir():
        pytest.skip("the CODE-A original is not present")
    return CODE_A_SOURCE


@pytest.fixture(scope="session")
def legacy_work() -> Path:
    """legacy/work/run-hager4_fsfs/out: the output of the golden Fortran run hager4_fsfs
    (SPHCO files with 16 digits and the GRD_* grids). Skips when absent."""
    out = REPO / "legacy" / "work" / "run-hager4_fsfs" / "out"
    if not (out / "GRD_geoid").is_file():
        pytest.skip("legacy/work/run-hager4_fsfs/out is not present")
    return out
