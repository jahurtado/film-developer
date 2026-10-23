#!/usr/bin/env python3
"""No module may define the same thing twice.

    uv run python tests/nodupes.py

Written because it happened, twice, and both times silently: a patch that
replaced a slice of `editor.py` with an end marker that came *before* the
start marker left the old copy in the file behind the new one. Python
keeps the **last** definition, so the file read as the new code and ran
the old -- `⌘L` still named the scan on screen as the flat field when the
source in front of you opened a file dialog. 1122 dead lines the first
time, 125 the second.

A duplicate definition is never intended in this codebase, so it can be
checked mechanically: walk the AST, and for every module, class and
function body, report any name defined twice at the same level. The one
legitimate pair is a property and its setter, which carry a decorator
naming the property, so those are excused.
"""

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def decorated(node):
    """The decorator names on a def, flattened to strings."""
    out = []
    for d in getattr(node, "decorator_list", []):
        t = d.func if isinstance(d, ast.Call) else d
        out.append(ast.unparse(t))
    return out


def dupes(body, where, path):
    """Names defined more than once directly in `body`."""
    seen, bad = {}, []
    for node in body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        name = node.name
        # `x.setter` (and `.getter`/`.deleter`) is the same name on purpose.
        if any(d.endswith((".setter", ".getter", ".deleter"))
               for d in decorated(node)):
            continue
        if name in seen:
            bad.append((where, name, seen[name], node.lineno))
        seen[name] = node.lineno
        kids = node.body
        bad += dupes(kids, f"{where}.{name}" if where else name, path)
    return bad


def main():
    bad = []
    for p in sorted(ROOT.glob("src/**/*.py")) + sorted(ROOT.glob("tools/*.py")) \
            + sorted(ROOT.glob("tests/*.py")):
        tree = ast.parse(p.read_text(), str(p))
        for where, name, first, again in dupes(tree.body, "", p):
            bad.append("%s:%d  %s is already defined at line %d"
                       % (p.relative_to(ROOT), again,
                          (where + "." if where else "") + name, first))
    for line in bad:
        print("  " + line)
    print("%d duplicate definition%s" % (len(bad), "" if len(bad) == 1 else "s"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
