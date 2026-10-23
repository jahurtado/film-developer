#!/usr/bin/env python3
"""One command, three subcommands, in the order they are meant to be run.

    film-developer                              # the editor, asking for a folder
    film-developer ROLL_DIR                     # the editor, on that folder
    film-developer detect ROLL_DIR
    film-developer apply  ROLL_DIR -o OUT_DIR
    film-developer develop ROLL_DIR
    film-developer flat NAME BLANK.NEF...       # the rig's light falloff, once
    film-developer roll ROLL_DIR [--json]       # what the document says, and change it
    film-developer process ROLL_DIR...          # what ⌘R writes, without the window

Naming no command opens the editor, because that is what somebody who typed
the name alone is after, and it is the only thing a double-clicked icon can
ask for.

They are subcommands and not flags of a single run because the separation is
the whole design: finding the frame is slow and fallible, writing the crops is
slow and irreversible, and a person reads the document in between.
"""

import argparse
import sys
from importlib import metadata

from . import apply as apply_cmd
from . import detect, develop, flatfield, gui, process, roll


def identity():
    """(author, site) read from the package, so they are written once.

    The version already comes from `__init__` and the author from the project
    metadata, so nothing that shows them -- the terminal, the About box, the
    window -- carries a copy of its own to fall out of date.
    """
    try:
        m = metadata.metadata("film-developer")
        who = m["Author"] or ", ".join(
            a.split("<")[0].strip() for a in m.get_all("Author-email") or [])
        site = ""
        for u in m.get_all("Project-URL") or []:
            if u.lower().startswith("homepage"):
                site = u.split(",", 1)[1].strip()
    except metadata.PackageNotFoundError:
        who, site = "", ""
    return who, site.replace("https://", "")


def credits():
    """The one-line form, for the terminal."""
    return " · ".join(x for x in identity() if x)


# A step is a module with `add_args(parser)` and `run(args)`, named here. That
# is the whole of the plumbing: there is no registry to update anywhere else.
COMMANDS = (
    ("detect", detect, "Find the exposed frame in every scan."),
    ("gui", gui, "Inspect and hand-correct what detect found."),
    ("apply", apply_cmd, "Write the crops out."),
    ("develop", develop, "Invert and balance every frame into a positive."),
    ("flat", flatfield, "Calibrate the copy stand's light falloff from blank frames."),
    ("roll", roll, "Read the roll's document, skip scans, set the roll's options."),
    ("process", process, "Everything ⌘R writes: pictures, raw DNGs, negatives."),
)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="film-developer",
        description="Find the picture inside a photographed negative: measure "
                    "where the exposed frame sits, correct by hand what the "
                    "detector missed, then crop and straighten -- and a raw "
                    "file stays raw.",
        epilog=credits(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version",
                    version="%(prog)s " + __import__("film_developer").__version__
                            + "\n" + credits())
    sub = ap.add_subparsers(dest="command", required=False, metavar="COMMAND")
    for name, mod, blurb in COMMANDS:
        p = sub.add_parser(
            name, help=blurb, description=mod.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
        mod.add_args(p)
        p.set_defaults(_run=mod.run)
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # No command means the editor. Opening a window is what somebody who typed
    # the name alone is after, and it is the only thing a double-clicked icon
    # can ask for -- a Dock has no subcommand to type. A path is a path and not
    # a misspelt command, so that goes to the editor too; anything starting
    # with a dash belongs to argparse, so --help and --version still answer.
    known = {name for name, _, _ in COMMANDS}
    if not argv or (not argv[0].startswith("-") and argv[0] not in known):
        argv = ["gui"] + argv
    args = build_parser().parse_args(argv)
    try:
        return args._run(args)
    except FileNotFoundError as e:
        # A missing document or missing scans is a normal thing to get wrong,
        # not a bug: say what is missing and stop, without a traceback.
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
