"""Documentation tests (NOTES/chunks/c5-integration/01-plan.md section 6.3): complete
docstrings, the public API of ``stokesjx/__init__.py``, the README, the docstrings of the
examples, and the location of the TALA warning.

D1 ``test_docstrings_complete``
    Every public module, class, function, method and property of the 13 modules has a
    docstring, and every parameter name of a public function appears in its docstring
    (a private parameter with a leading underscore is exempt).
D2 ``test_public_api``
    Every name of ``stokesjx.__all__`` resolves, has a docstring and appears in the
    package docstring; every public class and function of the six high-layer modules is
    in ``__all__`` or in ``EXCLUDED`` with its reason.
D3 ``test_readme``
    The headings of README.md in order, every ``stokesjx.<name>`` and ``sj.<name>``
    resolves, no modal of the list ``should, may, might, could, would`` (ASD-STE100) as a
    whole lower-case word outside code blocks, and no absolute home path (``HOME_PREFIX``).
D4 ``test_examples_docstrings``
    Each example has a module docstring that names its command line and its outputs
    (read with ``ast``, without running the script).
D5 ``test_tala_warning_points_at_caller``
    The TALA cut warning of ``stokes_kernels`` names the file of its caller.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import stokesjx as sj  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

MODULES = ("stokesjx", "stokesjx.fields", "stokesjx.sh", "stokesjx.controls", "stokesjx.kernels", "stokesjx.profile",
           "stokesjx.formulation", "stokesjx.units", "stokesjx.io.legacy", "stokesjx.propagate", "stokesjx.system",
           "stokesjx.boundary", "stokesjx.linalg")
HIGH_LAYER = ("kernels", "fields", "controls", "profile", "formulation", "units")

#: Public names of the high-layer modules that are not in ``__all__``, with the reason.
EXCLUDED = {
    "radial_integral": "fields.apply with explicit weights; the caller takes the responsibility for the weights",
    "consistent_core_mass": "inside RadialProfile.effective_core_mass; a user sets core_mass=None",
    "mantle_mass": "internal to the degree-1 imbalance (eq. 10.6)",
    "check_cut_leaf": "a validation helper of RadialProfile and propagate.check_profile",
    "Terms": "the static term groups that the solver derives from a Formulation",
    "terms": "the function that derives Terms from a Formulation",
    "FrameOptions": "the static degree-1 frame table; StokesConfig.degree_one is the user's choice",
    "degree_one_options": "the static degree-1 frame rule, read by the solver",
    "resolve_degree_one": "the resolution of a degree-1 request, read by the solver",
}

README_HEADINGS = ("# stokesjx", "## Status", "## Install", "## Quick start", "## Examples", "## Test",
                   "## Validation", "## Performance", "## Notes for samplers", "## Credit", "## Agentic coding",
                   "## Layout", "## Licence")
BANNED_MODALS = ("should", "may", "might", "could", "would")
#: The start of an absolute home path on macOS. Written in two parts, so that this tracked
#: file itself holds no such path (the hygiene gate greps every tracked file for it).
HOME_PREFIX = "/" + "Users/"


def _public_items(mod):
    """``(qualified name, object)`` of every public class and function that ``mod``
    defines, and of the public methods and properties of those classes."""
    for name, obj in vars(mod).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != mod.__name__:
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        yield name, obj
        if inspect.isclass(obj):
            for k, v in vars(obj).items():
                if not k.startswith("_") and (inspect.isfunction(v) or isinstance(v, property)):
                    yield f"{name}.{k}", v


def test_docstrings_complete():
    """D1: zero items under the rule of the module docstring."""
    bad = []
    for mn in MODULES:
        m = importlib.import_module(mn)
        if not (m.__doc__ or "").strip():
            bad.append((mn, "module", "no docstring"))
        for qn, o in _public_items(m):
            f = o.fget if isinstance(o, property) else o
            doc = inspect.getdoc(f) or ""
            if not doc.strip():
                bad.append((mn, qn, "no docstring"))
                continue
            if inspect.isfunction(f):
                params = [p for p in inspect.signature(f).parameters
                          if p not in ("self", "cls") and not p.startswith("_")]
                missing = [p for p in params if p not in doc]
                if missing:
                    bad.append((mn, qn, "parameters not named: " + ", ".join(missing)))
    assert not bad, "\n".join(map(str, bad))


def test_public_api():
    """D2: ``__all__`` resolves, is documented and named in the package docstring, and
    holds every public name of the high-layer modules except ``EXCLUDED``."""
    doc = sj.__doc__ or ""
    for name in sj.__all__:
        obj = getattr(sj, name)
        assert (inspect.getdoc(obj) or "").strip(), name
        assert re.search(rf"\b{re.escape(name)}\b", doc), f"{name} is not named in the package docstring"
    for mn in HIGH_LAYER:
        mod = importlib.import_module("stokesjx." + mn)
        for name, obj in vars(mod).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != mod.__name__:
                continue
            if inspect.isclass(obj) or inspect.isfunction(obj):
                assert name in sj.__all__ or name in EXCLUDED, f"stokesjx.{mn}.{name}"
    assert all(n not in sj.__all__ for n in EXCLUDED)


def _readme_without_code(text: str) -> str:
    """README text with the fenced code blocks and the inline code spans removed."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"(?m)^(    |\t).*$", "", text)
    return re.sub(r"`[^`\n]*`", "", text)


def _resolve(path: str):
    """Resolve ``a.b.c`` below ``stokesjx``, importing a submodule when an attribute is
    missing (``stokesjx.io.legacy``)."""
    obj = sj
    parts = path.split(".")
    for k, p in enumerate(parts):
        if not hasattr(obj, p):
            importlib.import_module("stokesjx." + ".".join(parts[: k + 1]))
        obj = getattr(obj, p)
    return obj


def test_readme():
    """D3: README headings, names, modals and paths."""
    text = (REPO / "README.md").read_text()
    lines = [ln.rstrip() for ln in text.splitlines()]
    heads = [ln for ln in lines if re.match(r"^#{1,2} ", ln)]
    assert tuple(heads) == README_HEADINGS, heads
    for prefix in ("stokesjx.", "sj."):
        for m in re.finditer(rf"(?<![\w/.]){re.escape(prefix)}([A-Za-z_][\w.]*\w)", text):
            name = m.group(1)
            if prefix == "stokesjx." and name in ("py",):
                continue
            _resolve(name)
    prose = _readme_without_code(text)
    for word in BANNED_MODALS:
        hits = re.findall(rf"\b{word}\b", prose)
        assert not hits, f"README uses the modal {word!r} outside code"
    assert HOME_PREFIX not in text


def test_readme_quick_start():
    """The Python blocks of README "Quick start" run as they are, in order, and give a
    finite field and a finite gradient (not a gate of the plan; it keeps the README code
    in step with the package)."""
    text = (REPO / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert len(blocks) == 2
    ns: dict = {}
    for b in blocks:
        exec(compile(b, "README.md", "exec"), ns)
    assert ns["F"].geoid.shape == (2, 51, 51) and np.all(np.isfinite(np.asarray(ns["F"].geoid)))
    assert np.all(np.isfinite(np.asarray(ns["gradient"]))) and float(np.asarray(ns["gradient"])[-1]) != 0.0


def test_examples_docstrings():
    """D4: each example names its command line and its outputs in its module docstring;
    ``hmc_layers.py`` describes the second mode, ``gradient_layers.py`` the reason for the
    base point."""
    for name, phrase in (("forward_legacy_density", None), ("gradient_layers", "viscosity contrast"),
                         ("hmc_layers", "second mode")):
        path = REPO / "examples" / f"{name}.py"
        doc = ast.get_docstring(ast.parse(path.read_text())) or ""
        assert "Command line" in doc and f"{name}.py" in doc, name
        assert "Output" in doc, name
        if phrase:
            assert phrase in doc, (name, phrase)
    doc = ast.get_docstring(ast.parse((REPO / "examples" / "earth_model.py").read_text())) or ""
    assert "prem_profile" in doc


def test_tala_warning_points_at_caller():
    """D5: under TALA with a density pair inside a cut range, the warning of
    ``stokes_kernels`` names this file (the caller), not a file of the package."""
    import cases
    from adapters import profile_from_table
    p = profile_from_table(cases.t_prem(reduced=True))
    L = sj.Layers(p, np.array([3480, 5600, 5970, 6270, 6370]) * 1e3, movable={1: (5466.875e3, 5970e3)})
    q = L(L.boundaries, np.log([1e22, 1e21, 3e20, 1e22]))
    cfg = sj.StokesConfig(lmax=3, formulation=sj.Formulation(compressible=True))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sj.stokes_kernels(q, cfg)
    hits = [w for w in rec if issubclass(w.category, RuntimeWarning) and "density jump" in str(w.message)]
    assert len(hits) == 1
    assert Path(hits[0].filename).resolve() == Path(__file__).resolve(), hits[0].filename
