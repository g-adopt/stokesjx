"""The package imports, sets double precision, and holds a version."""

import pytest

jax = pytest.importorskip("jax")


def test_import_sets_x64():
    """Importing stokesjx enables 64-bit floats for every JAX array."""
    import jax.numpy as jnp

    import stokesjx

    assert jax.config.jax_enable_x64
    assert jnp.zeros(1).dtype == jnp.float64
    assert jnp.asarray(1.0).dtype == jnp.float64


def test_version():
    import stokesjx

    assert stokesjx.__version__ == "0.0.1.dev0"


def test_editable_install_points_at_this_checkout():
    """The installed package is this checkout (G13)."""
    from pathlib import Path

    import stokesjx

    assert Path(stokesjx.__file__).resolve().parent == Path(__file__).resolve().parent.parent / "stokesjx"
