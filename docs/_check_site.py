"""Build the documentation site of stokesjx and check it.

Usage, from the repository root::

    python docs/_check_site.py --build [--quarto PATH]   # build from scratch, then check
    python docs/_check_site.py [--quarto PATH]           # check an existing build

``--build`` removes ``docs/reference/``, ``docs/_site/`` and ``docs/.quarto/``, runs
``python -m quartodoc build --config _quarto.yml`` in ``docs/`` (the API reference from
the docstrings) and then ``quarto render`` (the site). The output of each step goes to
``docs/_logs/quartodoc.log`` and ``docs/_logs/render.log``, with the standard output and
the standard error in one stream, because griffe and Quarto write their messages to the
standard error. A nonzero exit of either step fails the script.

The checks run after the build, and every check runs even when an earlier one fails, so
that one run reports every problem. The script exits 1 if any check fails:

1. Parser log. ``quartodoc.log`` holds no griffe message (a line with ``.py:<line>: ``)
   and no line with "warn" or "error" in any case. A griffe message means that a
   docstring did not parse as a numpy docstring, and the rendered page is then wrong
   without an error of the build.
2. Render log. ``render.log``, with the colour codes removed, holds no line with
   ``WARN`` or ``ERROR`` in upper case at any position. This finds the Quarto forms
   ``WARN:`` and ``WARNING (...)``, the pandoc form ``[WARNING] ...`` (an unclosed
   ``:::`` block, a repeated heading identifier) and ``ERROR:``. ``quarto render`` exits
   0 on the pandoc warnings, so the exit code alone does not find them.
3. Parameter tables. For every object heading of ``docs/reference/*.qmd`` (the entry
   and each method that quartodoc renders), the names of its "Parameters" table equal
   the parameters of ``inspect.signature`` of the object, without ``self`` and ``cls``.
   An object without a table must have no parameters. An ``Enum`` class is not
   compared, because its constructor is not its interface.
4. Links. Every relative ``href`` and ``src`` of ``docs/_site/**/*.html`` resolves to a
   file, and every non-empty fragment is an ``id`` or a ``name`` of the target page.
5. Display math. Every ``\\tag{...}`` of the pages is unique on the site, and every
   page with display math converts to MathML through ``quarto pandoc`` without the
   message "Could not convert TeX math" (on a copy without the ``\\tag`` commands,
   which the MathML writer of pandoc does not accept).

The module imports the standard library only. The package is imported inside the
parameter check, and quartodoc and Quarto run as subprocesses, so the tests of
``tests/test_docs_site.py`` can import this file without quartodoc and test each check
function on small inputs. Quarto does not render or copy a file whose name starts with
``_``, so this script is not part of the site.
"""

from __future__ import annotations

import argparse
import html.parser
import importlib
import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: The directory of this script: the Quarto project.
DOCS = Path(__file__).resolve().parent
#: The repository root (the parent of ``docs/``).
REPO = DOCS.parent

#: A griffe message: ``<path>.py:<line>: <message>``.
GRIFFE_LINE = re.compile(r"\.py:[0-9]+: ")
#: Any mention of a warning or an error in the parser log, in any case.
PARSER_PROBLEM = re.compile(r"warn|error", re.IGNORECASE)
#: The ANSI colour codes that Quarto writes around its messages.
COLOUR_CODE = re.compile(r"\x1b\[[0-9;]*m")
#: A warning or an error in the render log, upper case, at any position.
RENDER_PROBLEM = re.compile(r"WARN|ERROR")
#: An object heading of a generated reference page: ``# name { #stokesjx.path }``.
OBJECT_HEADING = re.compile(r"^(#+) .*\{ #(stokesjx\.[\w.]+) \}\s*$")
#: A section heading of the "Parameters" table (any level).
PARAMETERS_HEADING = re.compile(r"^#+ Parameters\b")
#: Any Markdown heading.
ANY_HEADING = re.compile(r"^#+ ")
#: A display-math tag.
TAG = re.compile(r"\\tag\{([^}]*)\}")
#: A display-math block of a page.
DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.S)
#: The schemes of the links that the link check does not follow.
EXTERNAL = ("http:", "https:", "mailto:", "data:", "javascript:")


# --------------------------------------------------------------------------
# Checks 1 and 2: the logs.
# --------------------------------------------------------------------------


def check_parser_log(text: str) -> list[str]:
    """Problems of the quartodoc log: every griffe message and every line that names a
    warning or an error in any case.

    Parameters
    ----------
    text
        The content of ``quartodoc.log``.

    Returns
    -------
    One string per problem line, ``"quartodoc.log:<line>: <text>"``; empty if the log
    is clean.
    """
    out = []
    for k, line in enumerate(text.splitlines(), start=1):
        if GRIFFE_LINE.search(line) or PARSER_PROBLEM.search(line):
            out.append(f"quartodoc.log:{k}: {line}")
    return out


def check_render_log(text: str) -> list[str]:
    """Problems of the Quarto render log: every line that holds ``WARN`` or ``ERROR`` in
    upper case at any position, after the colour codes are removed.

    Parameters
    ----------
    text
        The content of ``render.log``.

    Returns
    -------
    One string per problem line, ``"render.log:<line>: <text>"``; empty if the log is
    clean.
    """
    out = []
    for k, line in enumerate(text.splitlines(), start=1):
        plain = COLOUR_CODE.sub("", line)
        if RENDER_PROBLEM.search(plain):
            out.append(f"render.log:{k}: {plain}")
    return out


# --------------------------------------------------------------------------
# Check 3: the parameter tables of the generated reference.
# --------------------------------------------------------------------------


def parameter_tables(text: str) -> dict[str, list[str] | None]:
    """The parameter names of every object heading of one generated reference page.

    Parameters
    ----------
    text
        The content of one ``docs/reference/<entry>.qmd`` page.

    Returns
    -------
    A map from the object path (``stokesjx.profile.integrate_gravity``,
    ``stokesjx.controls.Layers.fields``) to the names of the first column of its
    "Parameters" table, or ``None`` when the object has no such table. A cell
    ``a, b`` (the numpy form of two names on one line) gives two names, and a leading
    ``*`` or ``**`` is removed.
    """
    lines = text.splitlines()
    heads = [(k, m.group(2)) for k, line in enumerate(lines) if (m := OBJECT_HEADING.match(line))]
    out: dict[str, list[str] | None] = {}
    for n, (start, path) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        names = None
        k = start + 1
        while k < end:
            if PARAMETERS_HEADING.match(lines[k]):
                names = []
                k += 1
                # Skip to the table, then read its rows; the second row is the rule.
                while k < end and not lines[k].startswith("|"):
                    k += 1
                rows = []
                while k < end and lines[k].startswith("|"):
                    rows.append(lines[k])
                    k += 1
                for row in rows[2:]:
                    cell = row.strip().strip("|").split("|")[0].strip()
                    for name in cell.split(", "):
                        name = name.strip().lstrip("*")
                        if name:
                            names.append(name)
                break
            k += 1
        out[path] = names
    return out


def _resolve(path: str):
    """The object of a dotted path below ``stokesjx``: import the longest module prefix,
    then follow the attributes."""
    parts = path.split(".")
    for k in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:k]))
        except ImportError:
            continue
        for p in parts[k:]:
            obj = getattr(obj, p)
        return obj
    raise ImportError(path)


def signature_names(obj) -> list[str] | None:
    """The parameter names that the table of ``obj`` must hold: those of
    ``inspect.signature`` without ``self`` and ``cls``, or ``None`` for an ``Enum``
    class, which is not compared.

    Parameters
    ----------
    obj
        A class, a function or a method (a function of a class, a classmethod or a
        staticmethod).
    """
    import enum
    if inspect.isclass(obj) and issubclass(obj, enum.Enum):
        return None
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    return [p for p in sig.parameters if p not in ("self", "cls")]


def compare_tables(tables: dict[str, list[str] | None], resolve=_resolve) -> list[str]:
    """Compare the parameter tables of the generated reference with the signatures.

    Parameters
    ----------
    tables
        The output of ``parameter_tables`` of one or more pages.
    resolve
        The function from an object path to the object (default: import from the
        installed or the ``PYTHONPATH`` package). The tests pass a map of their own.

    Returns
    -------
    One string per object whose table differs from its signature, with the names that
    the table has in excess and the names that it lacks; empty if every table matches.
    """
    out = []
    for path, names in sorted(tables.items()):
        expected = signature_names(resolve(path))
        if expected is None:
            continue
        got = names or []
        extra = [n for n in got if n not in expected]
        missing = [n for n in expected if n not in got]
        if extra or missing:
            out.append(f"{path}: table has {extra} not in the signature, lacks {missing}")
    return out


def check_reference(reference_dir: Path) -> list[str]:
    """Check 3 on every page of ``reference_dir`` except its index."""
    tables: dict[str, list[str] | None] = {}
    for page in sorted(reference_dir.glob("*.qmd")):
        if page.name == "index.qmd":
            continue
        tables.update(parameter_tables(page.read_text()))
    if not tables:
        return [f"{reference_dir}: no reference page with an object heading"]
    return compare_tables(tables)


# --------------------------------------------------------------------------
# Check 4: the links of the rendered site.
# --------------------------------------------------------------------------


class _Links(html.parser.HTMLParser):
    """Collects the ``href`` and ``src`` values and the ``id`` and ``name`` values of one
    HTML page, with the line of each link."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[int, str]] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if value is None:
                continue
            if key in ("href", "src"):
                self.links.append((self.getpos()[0], value))
            elif key in ("id", "name"):
                self.anchors.add(value)


def _parse_html(path: Path) -> _Links:
    p = _Links()
    p.feed(path.read_text(errors="replace"))
    return p


def check_links(site: Path) -> list[str]:
    """Every relative ``href`` and ``src`` of ``site/**/*.html`` resolves to a file (a
    path that ends in ``/`` to its ``index.html``), and every non-empty fragment is an
    ``id`` or a ``name`` of the target page.

    Parameters
    ----------
    site
        The rendered site (``docs/_site``).

    Returns
    -------
    One string per broken link, ``"<page>:<line>: <link> (<reason>)"``; empty if every
    link resolves.
    """
    pages = sorted(site.rglob("*.html"))
    if not pages:
        return [f"{site}: no HTML page"]
    parsed = {p.resolve(): _parse_html(p) for p in pages}
    out = []
    for page, info in parsed.items():
        for line, link in info.links:
            if link.lower().startswith(EXTERNAL) or link.startswith("//"):
                continue
            target, _, fragment = link.partition("#")
            target = target.split("?", 1)[0]
            if target == "":
                path = page
            else:
                path = (page.parent / target).resolve()
                if target.endswith("/") or path.is_dir():
                    path = path / "index.html"
            rel = page.relative_to(site.resolve())
            if not path.is_file():
                out.append(f"{rel}:{line}: {link} (no file {path.relative_to(site.resolve()) if path.is_relative_to(site.resolve()) else path})")
                continue
            if fragment and path.suffix == ".html":
                anchors = parsed[path].anchors if path in parsed else _parse_html(path).anchors
                if fragment not in anchors:
                    out.append(f"{rel}:{line}: {link} (no anchor {fragment!r})")
    return out


# --------------------------------------------------------------------------
# Check 5: display math.
# --------------------------------------------------------------------------


def check_tags(pages: dict[str, str]) -> list[str]:
    """Every ``\\tag{...}`` of the pages occurs once on the site.

    Parameters
    ----------
    pages
        A map from the page name to its text.

    Returns
    -------
    One string per repeated tag with the pages that hold it; empty if every tag is
    unique.
    """
    seen: dict[str, list[str]] = {}
    for name, text in pages.items():
        for m in TAG.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            seen.setdefault(m.group(1), []).append(f"{name}:{line}")
    return [f"tag {tag} occurs {len(where)} times: {', '.join(where)}"
            for tag, where in sorted(seen.items()) if len(where) > 1]


def check_mathml(pages: dict[str, str], quarto: str) -> list[str]:
    """Every page with display math converts to MathML without the pandoc message "Could
    not convert TeX math". The page is copied without its ``\\tag`` commands first.

    Parameters
    ----------
    pages
        A map from the page name to its text.
    quarto
        The Quarto binary; ``quarto pandoc`` runs the pandoc that Quarto bundles.

    Returns
    -------
    One string per message line of pandoc; empty if every page converts.
    """
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in sorted(pages.items()):
            if not DISPLAY_MATH.search(text):
                continue
            copy = Path(tmp) / name
            copy.write_text(TAG.sub("", text))
            res = subprocess.run([quarto, "pandoc", str(copy), "-f", "markdown", "-t", "html", "--mathml",
                                  "-o", os.devnull], capture_output=True, text=True)
            for line in (res.stdout + res.stderr).splitlines():
                if "Could not convert TeX math" in line or res.returncode != 0:
                    out.append(f"{name}: {line}")
            if res.returncode != 0 and not out:
                out.append(f"{name}: quarto pandoc exited {res.returncode}")
    return out


# --------------------------------------------------------------------------
# The build and the command line.
# --------------------------------------------------------------------------


def _run_logged(cmd: list[str], log: Path, env: dict | None = None) -> int:
    """Run ``cmd`` in ``docs/`` with the standard output and the standard error in one
    stream, write the stream to ``log``, print it, and return the exit code."""
    res = subprocess.run(cmd, cwd=DOCS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write_text(res.stdout)
    print(f"$ {' '.join(cmd)}  (exit {res.returncode}, log {log.relative_to(REPO)})")
    print(res.stdout)
    return res.returncode


def build(quarto: str) -> list[str]:
    """Remove the old outputs, generate the reference and render the site. Returns one
    string per failed step."""
    for d in ("reference", "_site", ".quarto"):
        shutil.rmtree(DOCS / d, ignore_errors=True)
    logs = DOCS / "_logs"
    logs.mkdir(exist_ok=True)
    out = []
    code = _run_logged([sys.executable, "-m", "quartodoc", "build", "--config", "_quarto.yml"], logs / "quartodoc.log")
    if code:
        out.append(f"quartodoc build exited {code}")
    env = dict(os.environ, QUARTO_PYTHON=sys.executable)
    code = _run_logged([quarto, "render"], logs / "render.log", env=env)
    if code:
        out.append(f"quarto render exited {code}")
    return out


def main(argv=None) -> int:
    """Run the build (with ``--build``) and the five checks. Returns the exit code."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true", help="build the reference and the site from scratch first")
    ap.add_argument("--quarto", default="quarto", help="the Quarto binary (default: quarto)")
    args = ap.parse_args(argv)
    results: dict[str, list[str]] = {}
    if args.build:
        results["build"] = build(args.quarto)
    logs = DOCS / "_logs"
    for name, fn, log in (("parser log", check_parser_log, "quartodoc.log"),
                          ("render log", check_render_log, "render.log")):
        path = logs / log
        results[name] = fn(path.read_text()) if path.is_file() else [f"{path} is missing"]
    results["parameter tables"] = check_reference(DOCS / "reference")
    results["links"] = check_links(DOCS / "_site")
    pages = {p.name: p.read_text() for p in sorted(DOCS.glob("*.qmd"))}
    results["math tags"] = check_tags(pages)
    results["math conversion"] = check_mathml(pages, args.quarto)
    failed = False
    for name, problems in results.items():
        if problems:
            failed = True
            print(f"FAILED {name}: {len(problems)} problem(s)")
            for p in problems:
                print(f"  {p}")
        else:
            print(f"passed {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
