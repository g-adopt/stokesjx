"""Tests of the documentation site (``docs/``): its configuration, its pages, the API
reference list and the check script ``docs/_check_site.py``. The model is the docs test
of lovejx. The tests read files only; Quarto is not run, so the tests run in CI.

``test_quarto_config``
    ``docs/_quarto.yml`` parses; the site and repository URLs; the quartodoc package and
    parser; every page of the navbar and the sidebar exists; the set of pages is the
    eight pages of the site, and no page path names a benchmark.
``test_reference_lists_every_public_name``
    The entries of the quartodoc sections are exactly the public names (the rule of the
    function docstring), none twice, each resolves; the public constants of the modules
    in the rule are exactly ``PUBLIC_CONSTANTS``.
``test_pages_need_no_notes``
    No page points the reader at the working notes or at the CODE-A copy.
``test_pages_ste_modals``
    No page uses a modal of ``BANNED_MODALS`` (ASD-STE100) outside code.
``test_equation_numbers_resolve``
    Every equation number, section number and item that the docstrings of the reference
    cite is a tag, a heading or a bold item label of the physics page.
``test_equation_tags_unique``
    Every ``\\tag`` of the site occurs once.
``test_page_code_runs``
    The Python blocks of the usage and the derivatives page run in order and give finite
    results, with ``lmax`` at most 20.
``test_page_numbers_match_code``
    The numbers that the pages state and the code holds are equal (the table of the
    function docstring).
``test_open_points_and_readme_links``
    The open-points page says that the comparisons against published values come later;
    the README links the site and its agentic-coding page.
``test_check_site_detects_*``
    Each check function of ``docs/_check_site.py`` fails on a small broken input and
    passes on the repaired input.

Constants of other test files are read with ``ast`` (literal assignments at module
level), not by import, so that this file does not run their module code.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import math
import re

import numpy as np
import pytest

jax = pytest.importorskip("jax")
yaml = pytest.importorskip("yaml")

import stokesjx as sj  # noqa: E402
from test_docs import BANNED_MODALS, REPO, _readme_without_code  # noqa: E402

DOCS = REPO / "docs"
TESTS = REPO / "tests"
#: The eight pages of the site, in the order of the sidebar.
PAGES = ("index.qmd", "install.qmd", "physics.qmd", "usage.qmd", "gradients.qmd", "validation.qmd",
         "open-points.qmd", "agentic-coding.qmd")
#: The modules whose public classes and functions are in the reference besides the
#: non-module names of ``stokesjx.__all__``.
REFERENCE_MODULES = ("stokesjx.fields", "stokesjx.sh", "stokesjx.controls", "stokesjx.io.legacy")
#: The public constants of the modules of the reference. They are not reference entries;
#: a new constant fails the test and waits for a decision.
PUBLIC_CONSTANTS = {"stokesjx.sh.CONVENTIONS", "stokesjx.fields.FIELD_NAMES", "stokesjx.controls.CLAMP_FRACTION",
                    "stokesjx.io.legacy.VISC0", "stokesjx.io.legacy.DEFAULT_HEADER"}
#: Text that points the reader at the unpublished working notes or the CODE-A copy.
FORBIDDEN = ("NOTES/", "legacy/CODE-A", "PLAN.md", "CONVENTIONS.md", "HANDOVER", "TOLERANCES.md")
#: The number of Python blocks of each page that ``test_page_code_runs`` runs.
CODE_BLOCKS = {"usage.qmd": 5, "gradients.qmd": 2}


def _load_check_site():
    """``docs/_check_site.py`` as a module, imported by its path (``docs/`` is not a
    package)."""
    spec = importlib.util.spec_from_file_location("_check_site", DOCS / "_check_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quarto() -> dict:
    """The parsed ``docs/_quarto.yml``."""
    return yaml.safe_load((DOCS / "_quarto.yml").read_text())


def _page(name: str) -> str:
    """The text of one page of the site."""
    return (DOCS / name).read_text()


def _constant(path, name):
    """The value of a literal module-level assignment ``name = ...`` of a Python file,
    read with ``ast``."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} is not a literal assignment of {path}")


def _entries() -> list[str]:
    """The entries of every quartodoc section, as dotted paths below ``stokesjx``."""
    return [e for s in _quarto()["quartodoc"]["sections"] for e in (s.get("contents") or [])]


def _resolve(path: str):
    """The object of ``stokesjx.<path>``: import the longest module prefix, then follow
    the attributes."""
    parts = ("stokesjx." + path).split(".")
    for k in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:k]))
        except ImportError:
            continue
        for p in parts[k:]:
            obj = getattr(obj, p)
        return obj
    raise ImportError(path)


def _public_names() -> set[str]:
    """The public names of the package, by the rule of the reference: (1) every name of
    ``stokesjx.__all__`` that is not a module, as ``<module>.<name>`` from its
    ``__module__``; (2) every class and function without a leading underscore that a
    module of ``REFERENCE_MODULES`` defines (``__module__`` is that module)."""
    out = set()
    for name in sj.__all__:
        obj = getattr(sj, name)
        if inspect.ismodule(obj):
            continue
        out.add(obj.__module__.removeprefix("stokesjx.") + "." + name)
    for mn in REFERENCE_MODULES:
        mod = importlib.import_module(mn)
        for name, obj in vars(mod).items():
            if name.startswith("_") or not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            if obj.__module__ == mn:
                out.add(mn.removeprefix("stokesjx.") + "." + name)
    return out


def _public_constants() -> set[str]:
    """The public constants of the modules of ``REFERENCE_MODULES``: names without a
    leading underscore that the source of the module assigns at module level (an
    ``Assign`` or ``AnnAssign`` target of its ``ast``, so imported names do not count) and
    whose value is not a class or a function."""
    out = set()
    for mn in REFERENCE_MODULES:
        mod = importlib.import_module(mn)
        tree = ast.parse(inspect.getsource(mod))
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, ast.AnnAssign) else [])
            for t in targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    value = getattr(mod, t.id)
                    if not (inspect.isclass(value) or inspect.isfunction(value)):
                        out.add(f"{mn}.{t.id}")
    return out


def _documented_docstrings() -> dict[str, str]:
    """The docstrings of every reference entry and of the public methods and properties
    of its classes, by dotted name."""
    out = {}
    for e in _entries():
        obj = _resolve(e)
        out[e] = inspect.getdoc(obj) or ""
        if inspect.isclass(obj):
            for k, v in vars(obj).items():
                if k.startswith("_"):
                    continue
                if isinstance(v, property):
                    out[f"{e}.{k}"] = inspect.getdoc(v.fget) or ""
                elif inspect.isfunction(v) or isinstance(v, (classmethod, staticmethod)):
                    f = v.__func__ if isinstance(v, (classmethod, staticmethod)) else v
                    out[f"{e}.{k}"] = inspect.getdoc(f) or ""
    return out


def test_quarto_config():
    """The site configuration: URLs, quartodoc settings, the pages of the navigation."""
    cfg = _quarto()
    site = cfg["website"]
    assert site["site-url"] == "https://gadopt.org/stokesjx/"
    assert site["repo-url"] == "https://github.com/g-adopt/stokesjx"
    assert cfg["quartodoc"]["package"] == "stokesjx" and cfg["quartodoc"]["parser"] == "numpy"
    hrefs = [item["href"] for item in site["navbar"]["left"]]
    hrefs += [item for item in site["sidebar"]["contents"] if isinstance(item, str)]
    pages = {h for h in hrefs if h.endswith(".qmd")}
    for h in pages:
        assert (DOCS / h).is_file(), h
    assert pages == set(PAGES)
    assert tuple(p.name for p in sorted(DOCS.glob("*.qmd"))) == tuple(sorted(PAGES))
    assert not any("benchmark" in h for h in hrefs)


def test_reference_lists_every_public_name():
    """The reference entries equal the public names; every entry resolves; the public
    constants are the decided ones."""
    entries = _entries()
    assert len(entries) == len(set(entries)), "an entry is listed twice"
    assert set(entries) == _public_names(), (set(entries) ^ _public_names())
    assert len(entries) == 44
    for e in entries:
        _resolve(e)
    assert _public_constants() == PUBLIC_CONSTANTS


def test_pages_need_no_notes():
    """No page names the working notes or the CODE-A copy as a place to read."""
    for name in PAGES:
        text = _page(name)
        for word in FORBIDDEN:
            assert word not in text, (name, word)
        assert not re.search(r"CONVENTIONS [0-9]", text), name


def test_pages_ste_modals():
    """No banned modal as a whole lower-case word outside code blocks and inline code."""
    for name in PAGES:
        prose = _readme_without_code(_page(name))
        for word in BANNED_MODALS:
            assert not re.findall(rf"\b{word}\b", prose), (name, word)


def _physics_structure():
    """The tags, the section numbers of the headings and the bold item labels per
    section of the physics page."""
    text = _page("physics.qmd")
    tags = set(re.findall(r"\\tag\{([^}]*)\}", text))
    sections: dict[int, str] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"^## (\d+)\. ", line)
        if m:
            current = int(m.group(1))
            sections[current] = ""
        elif line.startswith("## "):
            current = None
        elif current is not None:
            sections[current] += line + "\n"
    labels = {n: set(re.findall(r"\*\*\(([a-z])\)\*\*", body)) for n, body in sections.items()}
    return tags, set(sections), labels


def _cited_equations(doc: str) -> set[str]:
    """The equation numbers that a docstring cites after "eq." or "eqs.", with lists
    ("8.4, 8.5"), "and" and ranges ("8.1 to 8.4", every number in between); a trailing
    row letter ("4.1a") is dropped."""
    out = set()
    seq = r"\d+\.\d+[a-z]?"
    for m in re.finditer(rf"\beqs?\.\s+({seq}(?:(?:,\s*|\s+to\s+|\s+and\s+){seq})*)", doc):
        text = m.group(1)
        for a, b in re.findall(r"(\d+\.\d+)[a-z]?\s+to\s+(\d+\.\d+)", text):
            sa, ia = a.split(".")
            sb, ib = b.split(".")
            assert sa == sb, (a, b)
            out.update(f"{sa}.{k}" for k in range(int(ia), int(ib) + 1))
        out.update(re.sub(r"[a-z]$", "", n) for n in re.findall(seq, text))
    return out


def test_equation_numbers_resolve():
    """Every equation, section and item citation of the reference docstrings exists on
    the physics page."""
    tags, sections, labels = _physics_structure()
    bad = []
    for name, doc in _documented_docstrings().items():
        flat = " ".join(doc.split())
        for n in _cited_equations(flat):
            if n not in tags:
                bad.append((name, "eq.", n))
        for m in re.finditer(r"equations\.md(?:\s+section)?\s+(\d+)(?:\.\d+)?", flat):
            if int(m.group(1)) not in sections:
                bad.append((name, "section", m.group(0)))
        for m in re.finditer(r"(?<![\w.])(\d+)\(([a-h])\)", flat):
            n, x = int(m.group(1)), m.group(2)
            if n not in sections or x not in labels.get(n, set()):
                bad.append((name, "item", m.group(0)))
    assert not bad, bad


def test_equation_tags_unique():
    """Every ``\\tag`` of the site occurs once."""
    seen = []
    for name in PAGES:
        seen += re.findall(r"\\tag\{([^}]*)\}", _page(name))
    assert len(seen) == len(set(seen)), sorted(t for t in set(seen) if seen.count(t) > 1)
    assert len(seen) >= 40           # the physics page holds the tags of the docstrings


@pytest.mark.parametrize("page", sorted(CODE_BLOCKS))
def test_page_code_runs(page, tmp_path, monkeypatch):
    """The Python blocks of a page run in order in one namespace, in a temporary working
    directory, and give finite results; no block uses ``lmax`` above 20."""
    blocks = re.findall(r"```python\n(.*?)```", _page(page), flags=re.S)
    assert len(blocks) == CODE_BLOCKS[page]
    for b in blocks:
        for value in re.findall(r"lmax=(\d+)", b):
            assert int(value) <= 20, (page, value)
    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    for b in blocks:
        exec(compile(b, page, "exec"), ns)
    if page == "usage.qmd":
        results = [ns["F"].geoid, ns["grid"], ns["Z"], ns["misfit"], ns["back"]]
    else:
        results = [ns["value"], ns["gradient"]]
    for r in results:
        assert np.all(np.isfinite(np.asarray(r)))


def _pade_backward_bound(order: int, x: float) -> float:
    """``|log(exp(-x) r_m(x))| / x`` at 50 digits: the bound of the relative backward error
    of the diagonal Pade approximant ``r_m = N_m(x) / N_m(-x)`` at the scaled 1-norm ``x``
    (Higham 2005), with ``c_k = (2m-k)! m! / ((2m)! k! (m-k)!)``."""
    import mpmath as mp
    with mp.workdps(50):
        xm = mp.mpf(x)
        c = [mp.factorial(2 * order - k) * mp.factorial(order)
             / (mp.factorial(2 * order) * mp.factorial(k) * mp.factorial(order - k)) for k in range(order + 1)]
        num = sum(ck * xm ** k for k, ck in enumerate(c))
        den = sum(ck * (-xm) ** k for k, ck in enumerate(c))
        return float(abs(mp.log(mp.exp(-xm) * num / den)) / xm)


def _lovejx_case_count() -> int:
    """The number of test cases that call lovejx, from the test files with ``ast``: every
    function of ``tests/test_sh.py`` with an argument ``sht`` and
    ``tests/test_reference_oracle.py::test_g19_gravity_against_lovejx``, each counted by
    the length of its ``parametrize`` list."""
    def cases(fn) -> int:
        n = 1
        for d in fn.decorator_list:
            if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "parametrize":
                n *= len(ast.literal_eval(d.args[1]))
        return n
    total = 0
    for node in ast.parse((TESTS / "test_sh.py").read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_") and \
                any(a.arg == "sht" for a in node.args.args):
            total += cases(node)
    for node in ast.parse((TESTS / "test_reference_oracle.py").read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == "test_g19_gravity_against_lovejx":
            total += cases(node)
    return total


def _fmt(x: float) -> str:
    """A tolerance as the pages write it, ``1e-12``."""
    return f"{x:.0e}".replace("e-0", "e-")


def test_page_numbers_match_code():
    """The numbers of the pages that the code holds: each is found on its page by a
    pattern built from the code value."""
    import taylor
    grad, val, phys = _page("gradients.qmd"), _page("validation.qmd"), _page("physics.qmd")
    install, usage = _page("install.qmd"), _page("usage.qmd")
    readme = (REPO / "README.md").read_text()
    flat = lambda t: " ".join(t.split())
    # The clamp margin of a movable range.
    assert f"{100 * sj.controls.CLAMP_FRACTION:g} %" in flat(grad)
    # The Taylor ladder: first rung, ratio and length.
    eps = taylor.DEFAULT_EPSILONS
    first = round(math.log10(eps[0]))
    ratio = eps[0] / eps[1]
    assert all(math.isclose(eps[k] / eps[k + 1], ratio) for k in range(len(eps) - 1))
    assert f"10^{{{first}}}\\cdot {ratio:g}^{{-k}}" in grad and f"k = 0..{len(eps) - 1}" in grad
    # The pass conditions of a smooth measurement.
    sig = inspect.signature(taylor.assess).parameters
    assert f"at least {sig['min_rate'].default:g}" in flat(grad)
    lo, hi = sig["first_order_band"].default
    assert f"band ({lo:g}, {hi:g})" in flat(grad)
    # The negative control and the dot products.
    assert f"multiplied by {_constant(TESTS / 'test_linalg.py', 'NEG_BIAS'):g}" in flat(grad)
    dot = _fmt(_constant(TESTS / "test_linalg.py", "DOT_RTOL"))
    assert f"`{dot}`" in grad and f"`{dot}`" in val
    assert f"`{_fmt(_constant(TESTS / 'test_kernels_ad.py', 'DOT_RTOL_CONTRAST'))}`" in grad
    # The default method of a movable boundary.
    motion = inspect.signature(sj.controls.Layers).parameters["motion"].default
    for text in (grad, usage):
        assert f'`motion="{motion}"`, the default of `Layers`' in flat(text)
    # The bounds of the references.
    sections = re.split(r"\n## ", val)
    sect = {s.split("\n", 1)[0]: s for s in sections}
    fortran = _fmt(_constant(TESTS / "test_kernels_legacy.py", "FORTRAN_TOL"))
    assert f"`{fortran}`" in sect["Fortran CODE-A"]
    for name in ("ORACLE_TOL", "CONTRAST_TOL"):
        assert f"`{_fmt(_constant(TESTS / 'test_kernels_oracle.py', name))}`" in sect["The mpmath reference"]
    assert f"Bound `{_fmt(_constant(TESTS / 'test_assess.py', 'ASSESS_TOL'))}`" in sect["Analytical solutions"]
    # The Pade thresholds and the backward error bound of order 6.
    theta = importlib.import_module("stokesjx.linalg").PADE_THETA
    for m in (6, 7, 9):
        assert f"{theta[m]:.3f}" in phys, m
    assert f"bound {_pade_backward_bound(6, theta[6]):.1e}" in flat(phys)
    for m in (7, 9):
        assert abs(_pade_backward_bound(m, theta[m]) / 2.0 ** -53 - 1.0) < 0.01, m
    # The test count and time of the one run: equal on README and the install page.
    run = re.compile(r"(\d+) tests pass in (\d+(?:\.\d+)?) s")
    assert run.findall(flat(readme)) and run.findall(flat(readme)) == run.findall(flat(install))
    # The CI skip count and the count of the lovejx tests: equal on the three texts.
    lovejx = _lovejx_case_count()
    assert lovejx == 13
    skips = re.compile(r"skips (\d+) tests")
    for text in (readme, install, val):
        assert skips.findall(flat(text)) == skips.findall(flat(readme)) != [], text[:40]
        assert f"{lovejx} tests that compare with lovejx" in flat(text)


def test_open_points_and_readme_links():
    """The open-points page states that the comparisons against published values come
    later; the README links the site and its agentic-coding page."""
    assert "comparisons against published values comes later" in " ".join(_page("open-points.qmd").split())
    readme = (REPO / "README.md").read_text()
    assert "https://gadopt.org/stokesjx/" in readme
    assert "https://gadopt.org/stokesjx/agentic-coding.html" in readme


# --------------------------------------------------------------------------
# The check functions of docs/_check_site.py on small inputs.
# --------------------------------------------------------------------------

_INTEGRATE_GRAVITY_PAGE = """# profile.integrate_gravity {{ #stokesjx.profile.integrate_gravity }}

```python
profile.integrate_gravity(radius, rho0, core_mass, units)
```

Hydrostatic gravity.

## Parameters {{.doc-section .doc-section-parameters}}

| Name      | Type   | Description   | Default    |
|-----------|--------|---------------|------------|
{rows}
"""


def _gravity_page(names) -> str:
    rows = "\n".join(f"| {n} | | text | _required_ |" for n in names)
    return _INTEGRATE_GRAVITY_PAGE.format(rows=rows)


def test_check_site_detects_bogus_parameters():
    """The table check fails on a bogus row ("Numerical") and on a missing parameter, and
    passes on the correct table."""
    cs = _load_check_site()
    good = ["radius", "rho0", "core_mass", "units"]
    assert cs.compare_tables(cs.parameter_tables(_gravity_page(good))) == []
    bogus = cs.compare_tables(cs.parameter_tables(_gravity_page(good + ["Numerical"])))
    assert len(bogus) == 1 and "Numerical" in bogus[0]
    missing = cs.compare_tables(cs.parameter_tables(_gravity_page(good[:3])))
    assert len(missing) == 1 and "units" in missing[0]
    # The numpy form "a, b" of two names on one line.
    assert cs.compare_tables(cs.parameter_tables(_gravity_page(["radius, rho0", "core_mass", "units"]))) == []


def test_check_site_detects_broken_links(tmp_path):
    """The link check fails on a link to a missing file and on a missing anchor, and passes
    on the same tree with both fixed."""
    cs = _load_check_site()
    site = tmp_path / "_site"
    (site / "reference").mkdir(parents=True)
    (site / "page.html").write_text('<html><body><h2 id="here">x</h2></body></html>')
    index = ('<html><body><a href="{a}">a</a><a href="page.html#{b}">b</a>'
             '<a href="https://example.org/x">ext</a><a href="#top" id="top">t</a></body></html>')
    (site / "index.html").write_text(index.format(a="reference/nothing.html", b="nope"))
    problems = cs.check_links(site)
    assert len(problems) == 2
    assert any("nothing.html" in p for p in problems) and any("'nope'" in p for p in problems)
    (site / "reference" / "index.html").write_text("<html></html>")
    (site / "index.html").write_text(index.format(a="reference/", b="here"))
    assert cs.check_links(site) == []


def test_check_site_detects_bad_math():
    """The tag check fails on a repeated tag; the log checks fail on a griffe line, a
    ``WARN`` line with colour codes and a pandoc ``[WARNING]`` line, and pass on a clean
    render log."""
    cs = _load_check_site()
    pages = {"a.qmd": "$$x \\tag{4.5}$$\n", "b.qmd": "$$y \\tag{4.5}$$\n$$z \\tag{4.6}$$\n"}
    problems = cs.check_tags(pages)
    assert len(problems) == 1 and "4.5" in problems[0]
    assert cs.check_tags({"a.qmd": pages["a.qmd"]}) == []
    griffe = "/x/stokesjx/profile.py:290: Parameter 'Numerical' does not appear in the function signature\n"
    assert len(cs.check_parser_log(griffe)) == 1
    assert cs.check_parser_log("") == []
    colour = "\x1b[33mWARN: unable to resolve link target missing.qmd\x1b[39m\n"
    assert len(cs.check_render_log(colour)) == 1
    pandoc = "[WARNING] Div at line 3 column 1 unclosed at line 45 column 1, closing implicitly.\n"
    assert len(cs.check_render_log(pandoc)) == 1
    clean = "\x1b[1m[1/2] index.qmd\x1b[22m\nOutput created: _site/index.html\n"
    assert cs.check_render_log(clean) == []
