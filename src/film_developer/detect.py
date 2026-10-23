"""Find the exposed frame in every scan, and write down what was found.

Reads the raw files in a folder and writes `frames.json` beside them. It does
not touch an image: the expensive, fallible pass runs once, and a person gets
to see what it decided before anything is written.

    film-developer detect ROLL_DIR
    film-developer detect ROLL_DIR --format half
    film-developer detect ROLL_DIR --aspect 3:2

A second run measures only the scans the document does not already have, and
never re-measures a hand correction -- that is what makes the document worth
keeping.
"""

import math
import pathlib
import sys

from . import detectors, document, image_io

FORMATS = ("auto", "full", "half", "6x6", "6x7", "6x4.5")

def add_args(ap):
    ap.add_argument("roll", metavar="ROLL_DIR",
                    help="folder of scans (NEF, DNG, CR2, CR3)")
    ap.add_argument("-o", "--output", metavar="FILE",
                    help="where to write the document "
                         "(default: ROLL_DIR/frames.json)")
    ap.add_argument("--format", default="auto", choices=FORMATS,
                    help="what is on the film. `auto` decides per scan from "
                         "the shape of what it finds, and says so in the "
                         "document (default: auto)")
    ap.add_argument("--aspect", default="native", metavar="W:H",
                    help="snap every crop to this ratio, or `native` to keep "
                         "what was measured, or `free` (default: native)")
    ap.add_argument("--crop", type=int, metavar="PX",
                    help="take this many scan pixels off every side of every "
                         "frame, or add them if negative. A frame edge is a "
                         "soft ramp a few pixels wide and where inside it the "
                         "cut goes is a matter of taste (default: 0)")
    ap.add_argument("--same-size", dest="same_size", action="store_true",
                    default=None,
                    help="one frame size for the whole roll: a frame whose "
                         "measured height or width disagrees with the roll's "
                         "has a side put right. On for a new roll; the "
                         "window asks when a roll is opened")
    ap.add_argument("--no-same-size", dest="same_size", action="store_false",
                    help="leave every frame the height it measured")
    ap.add_argument("--same-gap", dest="same_gap", action="store_true",
                    default=None,
                    help="give every half-frame row the roll's gap, about "
                         "the gap's centre, with no tolerance. Off by "
                         "default")
    ap.add_argument("--no-same-gap", dest="same_gap", action="store_false",
                    help="leave every row the gap it measured")
    ap.add_argument("--again", action="store_true",
                    help="measure the roll again under its settings now -- "
                         "after `roll --same-size`, say -- keeping what a "
                         "person did: skipped scans and scans placed by hand "
                         "stay as they are, a quarter turn given by hand is "
                         "carried onto the new frames, and the develop "
                         "settings stay")
    ap.add_argument("--force", action="store_true",
                    help="measure everything again, discarding the old "
                         "numbers -- hand corrections included, which it says "
                         "out loud before doing")
    ap.add_argument("--gui", action="store_true",
                    help="open the editor on the document it just wrote")


def scans(roll):
    d = pathlib.Path(roll)
    if not d.is_dir():
        raise FileNotFoundError(f"{d} is not a folder of scans")
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in image_io.RAW_SUFFIXES)


def measure(path, mirrored=True, settings=None, rolled=None):
    """Measure one scan and return a record for the document.

    A thin wrapper over `detectors.run`. There used to be a choice of
    detector, named in the roll's settings and written into every record;
    there is one now, so there is nothing to choose and nothing to pin a
    folder to the worse of them.

    `rolled` is what the roll decided when it was measured -- the format
    and the size clusters, kept in the document. Given, this measures the
    scan the way the roll pass would have: alone first, then again with
    the height, width, gap and row the roll says a frame of that size
    has. Without it a single scan is measured on its own, which is the
    worst case there is on a dark frame -- no edge to find and no roll to
    supply what is missing.
    """
    settings = settings or {}
    item = detectors.run(path, settings, rolled)
    return document.order_parts(item, mirrored)


# How far a scan's frames may sit from what the rest of the roll says a frame
# is, before it is measured again with the roll's own format imposed. Loose on
# purpose: a roll's frame size drifts by up to 8 % when the copy stand is
# moved partway through, and that drift is real. What this catches is the
# absurd -- a frame a fifth the width of its neighbours, which is half a
# picture read as a whole one.
DISAGREE = 0.30


def roll_says(items):
    """(format, median size) of the roll, over the frames it mostly has.

    The format the roll mostly is, and the median size within it. A median
    over every frame of a roll holding two formats is the size of neither.
    """
    seen = [(q["format"], q["size"]) for it in items.values()
            for q in it["parts"] if not it.get("manual")]
    if not seen:
        return None, None
    counts = {}
    for fmt, _ in seen:
        counts[fmt] = counts.get(fmt, 0) + 1
    fmt = max(counts, key=counts.get)
    sizes = sorted((s for f, s in seen if f == fmt), key=lambda s: s[0] * s[1])
    return fmt, sizes[len(sizes) // 2]


# A hand correction that says where a frame is. A quarter turn (`turn`)
# says which way up the picture is, and that survives the frame moving.
PLACED = ("placed", "frame", "moved", "dropped")


def again(doc):
    """`--again`: drop from the document every scan the roll may measure
    afresh, and return {scan: {part: turn}} for the turns given by hand.

    `⇧⌘D` and `--force` go back to a folder nobody has opened, hand
    corrections and develop settings with it, which is right for *start
    again* and wrong for *the roll's settings changed* -- `same_size` is
    read when frames are found, so turning it on for a roll with a sitting
    of corrections in it had no way through but to throw the sitting away.
    Kept whole: skipped scans, and scans with any part placed by hand.
    The roll's measurement goes: its base and balance were read inside the
    frames that are about to move."""
    turns = {}
    for name in list(doc["items"]):
        it = doc["items"][name]
        hand = it.get("manual") or []
        if it["state"] == "skipped" or any(m.split(".")[-1] in PLACED
                                           for m in hand):
            continue
        turns[name] = {p["name"]: p["turn"] for p in it.get("parts") or []
                       if p.get("turn") and f"parts.{p['name']}.turn" in hand}
        del doc["items"][name]
    doc["roll"].pop("measured", None)
    return turns


def carry_turns(doc, turns):
    """The hand-given turns back onto the frames measured again, by letter,
    and only where the scan still has as many frames as it had."""
    for name, by in turns.items():
        it = doc["items"].get(name)
        if not by or not it:
            continue
        letters = {p["name"] for p in it.get("parts") or []}
        if not set(by) <= letters:
            print(f"  {name}: its frames changed, so its turn by hand is lost")
            continue
        for p in it["parts"]:
            if p["name"] in by:
                p["turn"] = by[p["name"]]
                document.mark_manual(it, f"parts.{p['name']}.turn")


def run(args):
    files = scans(args.roll)
    if not files:
        print(f"no raw scans in {args.roll}", file=sys.stderr)
        return 1
    where = args.output or args.roll

    try:
        doc = document.load(where)
    except (FileNotFoundError, ValueError):
        doc = document.blank({"format": args.format, "aspect": args.aspect})

    if args.force:
        # **The same thing `⇧⌘D` does in the window**, and it says the same
        # things: back to a folder nobody has opened, keeping only what
        # nothing can measure -- whether the roll is mirrored, whether it is
        # one roll, and the crop, which is a way of showing. The two used to
        # differ, the key keeping the roll's settings and every record whose
        # scan had left the folder, so *start again* meant two things
        # depending on where it was asked from.
        hand = sum(1 for it in doc["items"].values() if it.get("manual"))
        if hand:
            print(f"--force discards {hand} hand-corrected "
                  f"scan{'s' if hand != 1 else ''}, which cannot be recovered")
        if doc["roll"].get("measured") or doc["roll"].get("develop"):
            print("--force discards the roll's measurements and its develop "
                  "settings; mirrored, one_roll and the crop are kept")
        keep = {k: doc["roll"][k] for k in ("mirrored", "one_roll")
                if k in doc["roll"]}
        crop = (doc["roll"].get("develop") or {}).get("crop")
        doc["roll"].pop("measured", None)
        doc["roll"].pop("develop", None)
        doc["roll"].pop("apply", None)
        doc["roll"].update(keep)
        if crop is not None:
            doc["roll"].setdefault("develop", {})["crop"] = crop
        doc["items"] = {}

    turns = {}
    if args.again and not args.force:
        turns = again(doc)
    todo = [p for p in files if p.name not in doc["items"]]
    already = len(files) - len(todo)
    w, h = image_io.scan_size(files[0])
    print(f"{len(files)} scans in {args.roll} ({w}x{h})")
    if already:
        print(f"  {already} already measured, {len(todo)} new")

    if args.crop is not None:
        doc["roll"]["crop"] = args.crop
    if args.format != "auto":
        doc["roll"]["format"] = args.format
    if args.same_size is not None:
        doc["roll"]["same_size"] = bool(args.same_size)
    if args.same_gap is not None:
        doc["roll"]["same_gap"] = bool(args.same_gap)
    mirrored = doc["roll"].get("mirrored", True)
    # The roll is the evidence: measured all together, twice -- see
    # `fitframe.measure_roll`. Hand-corrected scans are not in `todo`, so
    # they are not touched; they are also not consulted, which is a gap
    # worth closing when it costs a scan.
    def tick(i, n, scan):
        print(f"\r  {i + 1}/{n} {scan:<24}", end="", flush=True)
    # What the roll decided, kept in the document: `⌘D` on one scan reads
    # it so that scan comes out of the same roll it came out of the first
    # time. See `fitframe.measure_roll`.
    rolled = {}
    items = detectors.run_roll(todo, doc["roll"], tick, rolled)
    if rolled.get("clusters"):
        doc["roll"]["rolled"] = rolled
    if todo:
        print()
    for p in todo:
        doc["items"][p.name] = document.order_parts(items[p.name], mirrored)
    carry_turns(doc, turns)
    fixed = []

    items = doc["items"]
    parts = sum(len(it["parts"]) for it in items.values())
    weak = [n for n, it in items.items()
            if it["evidence"].get("angle_margin", 9) < detectors.WEAK_MARGIN]
    refused = [n for n, it in items.items() if it["state"] == "refused"]
    held = [n for n, it in items.items() if it["state"] == "held"]
    halves = sum(1 for it in items.values() if len(it["parts"]) > 1)
    tilts = sorted(abs(it["angle"]) for it in items.values() if it["parts"])

    print(f"  {len(items) - len(refused)} detected, {len(refused)} refused")
    print(f"  {halves} with two parts, {parts} parts in all")
    if tilts:
        print(f"  median tilt {tilts[len(tilts)//2]:.2f}°, "
              f"worst {tilts[-1]:.2f}°")
    if fixed:
        print(f"  {len(fixed)} disagreed with the roll and were measured "
              f"again: {', '.join(n[:-4] for n in fixed)}")
    if weak:
        print(f"  {len(weak)} with a weak angle peak: {', '.join(sorted(weak))}")
    if refused:
        print(f"  refused: {', '.join(sorted(refused))}")
    if held:
        print(f"  {len(held)} held -- the roll had to supply a side, or "
              f"contradicted one; look at these first: "
              f"{', '.join(n[:-4] for n in sorted(held))}")
    print(f"  written to {document.save(doc, where)}")
    return 0
