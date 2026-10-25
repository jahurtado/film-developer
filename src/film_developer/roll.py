"""Read and change a roll's document from the command line.

    film-developer roll ROLL_DIR                     # what the document says
    film-developer roll ROLL_DIR --json              # the same, for a program
    film-developer roll ROLL_DIR --skip DSC_0491 DSC_0492
    film-developer roll ROLL_DIR --keep DSC_0491
    film-developer roll ROLL_DIR --same-base --same-size --flat "D5600 55mm"
    film-developer roll ROLL_DIR --pick-base DSC_0419 300 120 900 260 --for-roll

Everything the window decides about a roll as a whole, and the one thing it
decides about a scan without measuring -- whether it is developed at all
(`del`) -- reachable without the window, so a script, or an agent driving
the program, can do what a person does there. The window and this write the same keys of
the same document; neither has a setting the other cannot reach.

Changing a setting here never measures anything. `same_size` and
`same_gap` are read when frames are found, so they change nothing until
`detect --again`; the base, the flat and the taste are read when frames are
developed, by `process`. The output says which.
"""

import json
import pathlib
import sys

from . import develop, document, flatfield


def add_args(ap):
    ap.add_argument("roll", metavar="ROLL_DIR",
                    help="folder of scans with its frames.json")
    ap.add_argument("--json", action="store_true",
                    help="print the summary as JSON, for a program to read")
    ap.add_argument("--skip", nargs="+", metavar="SCAN", default=[],
                    help="leave these scans out of the roll, as `del` does in "
                         "the window: not developed, not counted in the shot "
                         "numbers. The name with or without its extension")
    ap.add_argument("--keep", nargs="+", metavar="SCAN", default=[],
                    help="put skipped scans back")
    ap.add_argument("--one-roll", dest="one_roll", action="store_true",
                    default=None,
                    help="all the scans are from the same roll (the first "
                         "question when a folder is opened)")
    ap.add_argument("--not-one-roll", dest="one_roll", action="store_false",
                    help="a folder of scans from several rolls: every frame "
                         "measured and developed on its own")
    ap.add_argument("--same-base", dest="same_base", action="store_true",
                    default=None,
                    help="the film base is the same across the roll "
                         "(develop.base auto)")
    ap.add_argument("--own-base", dest="same_base", action="store_false",
                    help="every scan its own unexposed film (develop.base "
                         "frame)")
    ap.add_argument("--same-size", dest="same_size", action="store_true",
                    default=None,
                    help="every frame the same size; takes effect on "
                         "`detect --again`")
    ap.add_argument("--no-same-size", dest="same_size", action="store_false")
    ap.add_argument("--same-gap", dest="same_gap", action="store_true",
                    default=None,
                    help="every half-frame row the same gap; takes effect on "
                         "`detect --again`")
    ap.add_argument("--no-same-gap", dest="same_gap", action="store_false")
    ap.add_argument("--flat", metavar="NAME",
                    help="the copy stand calibration the roll was scanned on "
                         "(`flat --list` names them), or `none`")
    ap.add_argument("--numbered", dest="numbered", action="store_true",
                    default=None,
                    help="the shot number in front of every file name")
    ap.add_argument("--not-numbered", dest="numbered", action="store_false")
    ap.add_argument("--pick-base", nargs=5, metavar=("SCAN", "X0", "Y0", "X1", "Y1"),
                    help="`⇧B`: the film base read in this box of unexposed "
                         "film, in scan pixels; for that scan, or with "
                         "--for-roll its colour for every scan")
    ap.add_argument("--for-roll", action="store_true",
                    help="with --pick-base: for the whole roll")
    ap.add_argument("--no-picked-base", nargs="+", metavar="SCAN",
                    default=[], help="take a picked base away: a scan's, or "
                                     "`roll` for the roll's")


def _named(doc, names):
    """The document's scan names for `names`, given with or without the
    extension; an unknown one stops the run, so a typo is not a no-op."""
    items = doc["items"]
    stems = {}
    for n in items:
        stems.setdefault(n.rsplit(".", 1)[0], n)
    out, bad = [], []
    for q in names:
        hit = q if q in items else stems.get(q)
        (out if hit else bad).append(hit or q)
    if bad:
        raise SystemExit(f"not in the document: {', '.join(bad)}")
    return out


def summary(doc):
    """What a person reads off the window about the roll, as data."""
    r = doc["roll"]
    dev = dict(develop.TASTE) | dict(r.get("develop") or {})
    states = {}
    for it in doc["items"].values():
        states[it["state"]] = states.get(it["state"], 0) + 1
    numbers = document.shot_numbers(doc)
    scans = []
    for name in document.order_of(doc):
        it = doc["items"][name]
        scans.append({"scan": name, "shot": numbers.get(name),
                      "state": it["state"],
                      "parts": len(it.get("parts") or []),
                      "manual": list(it.get("manual") or [])})
    return {
        "scans": len(doc["items"]),
        "states": states,
        "one_roll": r.get("one_roll", True),
        "same_size": bool(r.get("same_size")),
        "same_gap": bool(r.get("same_gap")),
        "same_base": dev.get("base", "auto") != "frame",
        "flat": dev.get("flat"),
        "numbered": r.get("numbered", True),
        "picked_base": {"roll": (r.get("base_pick") or {}).get("scan"),
                        "scans": sorted(doc.get("bases") or {})},
        "measured": develop.measured_ready(r.get("measured"), dev),
        "taste": {k: dev[k] for k in develop.TASTE
                  if k not in ("base", "flat", "format", "order")},
        "apply": dict(r.get("apply") or {}),
        "list": scans,
    }


def print_summary(s):
    print(f"{s['scans']} scans: "
          + ", ".join(f"{v} {k}" for k, v in sorted(s["states"].items())))
    print(f"  one roll {s['one_roll']}, same size {s['same_size']}, "
          f"same gap {s['same_gap']}, same base {s['same_base']}, "
          f"flat {s['flat']}, numbered {s['numbered']}")
    print(f"  roll measured for develop: {s['measured']}")
    print("  taste: " + ", ".join(f"{k} {v}" for k, v in s["taste"].items()))
    for row in s["list"]:
        shot = f"{row['shot']:>3}" if row["shot"] else "  -"
        hand = f"  by hand: {', '.join(row['manual'])}" if row["manual"] else ""
        print(f"  {shot} {row['scan']:<28} {row['state']:<9}"
              f" {row['parts']} part{'s' if row['parts'] != 1 else ''}{hand}")


def run(args):
    doc = document.load(args.roll)
    r = doc["roll"]
    said = []
    for name in _named(doc, args.skip):
        if doc["items"][name]["state"] != "skipped":
            doc["items"][name]["state"] = "skipped"
            said.append(f"skipped {name}")
    for name in _named(doc, args.keep):
        if doc["items"][name]["state"] == "skipped":
            # What `del` pressed again does in the window.
            doc["items"][name]["state"] = "detected"
            said.append(f"kept {name}")
    if args.one_roll is not None:
        r["one_roll"] = args.one_roll
        said.append(f"one roll {args.one_roll}")
    for key in ("same_size", "same_gap"):
        val = getattr(args, key)
        if val is not None and bool(r.get(key)) != val:
            r[key] = val
            said.append(f"{key} {val} (frames move on `detect --again`)")
    dev = r.setdefault("develop", {})
    if args.same_base is not None:
        dev["base"] = "auto" if args.same_base else "frame"
        said.append(f"base {dev['base']}")
    if args.flat is not None:
        flat = None if args.flat.lower() in ("none", "off", "") else args.flat
        flatfield.of({"flat": flat})      # an unknown name stops here
        if dev.get("flat") != flat:
            dev["flat"] = flat
            # The base and the balance are read through it.
            r.pop("measured", None)
            said.append(f"flat {flat} (the roll is measured again on `process`)")
    if args.numbered is not None:
        r["numbered"] = args.numbered
        said.append(f"numbered {args.numbered}")
    for q in args.no_picked_base:
        if q == "roll":
            if r.pop("base_pick", None):
                said.append("the roll's picked base removed")
            continue
        for name in _named(doc, [q]):
            if (doc.get("bases") or {}).pop(name, None):
                said.append(f"{name}: picked base removed")
    if args.pick_base:
        # What `⇧B` does, from a box a person or a program already knows:
        # the median of unexposed film inside it, under the roll's falloff.
        name = _named(doc, [args.pick_base[0]])[0]
        box = [float(v) for v in args.pick_base[1:]]
        taste = dict(develop.TASTE) | dict(dev)
        value = develop.base_sample(pathlib.Path(args.roll) / name, box,
                                    flatfield.of(taste))
        own = {"box": box, "value": value, "flat": taste.get("flat")}
        g = max(value[1], 1e-6)
        read = f"R/G {value[0] / g:.3f}, B/G {value[2] / g:.3f}"
        if args.for_roll:
            r["base_pick"] = own | {"scan": name}
            said.append(f"the roll's base picked on {name}: {read} "
                        "(its colour, on every scan without its own)")
        else:
            doc.setdefault("bases", {})[name] = own
            said.append(f"{name}: base picked, {read}")
    if said:
        document.save(doc, args.roll)
    s = summary(doc)
    if args.json:
        json.dump(s | {"changed": said}, sys.stdout, indent=1)
        print()
    else:
        for line in said:
            print(line)
        print_summary(s)
    return 0
