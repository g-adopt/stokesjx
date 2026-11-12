"""Check the skipped tests of a CI run against the two allowed groups.

Usage::

    python .github/check_skips.py artifacts/pytest.xml [--expected N]

The CI job does not have ``legacy/`` (the Fortran reference CODE-A, not distributed) and
does not install lovejx (not on PyPI). The tests that need one of them skip, and only
these. Any other skip (a pin with a wheel that does not load, a missing optional
package) turns a test that must run into a skip, and pytest still exits 0. This script
reads the JUnit XML file of pytest, prints every skipped node id with its reason, and
exits 1 when a reason is outside the two groups or when the number of skips differs from
``--expected``.

The two groups, as regular expressions of the skip message:

- ``legacy/... is not present`` (the fixtures of ``tests/conftest.py`` and the example
  test of the default density file);
- ``could not import 'lovejx...': No module named 'lovejx'`` (``pytest.importorskip``).

The script needs the standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET

#: The skip messages that the CI job allows.
ALLOWED = (re.compile(r"^legacy/\S+( with \S+)? is not present$"),
           re.compile(r"^could not import 'lovejx[\w.]*': No module named 'lovejx'$"))


def skipped_cases(path: str) -> list[tuple[str, str]]:
    """``(node id, message)`` of every skipped test case of a JUnit XML file of pytest.

    Parameters
    ----------
    path
        The file that ``pytest --junitxml`` wrote.
    """
    out = []
    for case in ET.parse(path).getroot().iter("testcase"):
        skip = case.find("skipped")
        if skip is None:
            continue
        node = f"{case.get('classname', '')}::{case.get('name', '')}"
        out.append((node, skip.get("message", "")))
    return out


def main(argv=None) -> int:
    """Print the skips and return 1 on a skip outside the allowed groups or a count that
    differs from ``--expected``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("junit_xml")
    ap.add_argument("--expected", type=int, default=None, help="the number of skips that the run must have")
    args = ap.parse_args(argv)
    cases = skipped_cases(args.junit_xml)
    bad = []
    for node, message in cases:
        ok = any(p.match(message) for p in ALLOWED)
        print(f"{'allowed' if ok else 'NOT ALLOWED'}: {node}: {message}")
        if not ok:
            bad.append(node)
    print(f"{len(cases)} skipped, {len(bad)} outside the allowed groups")
    if bad:
        return 1
    if args.expected is not None and len(cases) != args.expected:
        print(f"expected {args.expected} skipped tests, found {len(cases)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
