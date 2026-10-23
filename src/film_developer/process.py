"""`⌘R` without the window: everything that comes out of a roll, in one run.

    film-developer process ROLL_DIR                  # pictures and raw DNGs
    film-developer process ROLL_DIR --no-raws --format tiff16
    film-developer process ROLL_DIR --negatives      # the untouched crops too
    film-developer process A B C                     # several rolls, one after another

What the `⌘R` sheet asks is asked here as flags, with the sheet's
defaults: developed pictures and positive DNGs yes, the negatives no, and
the crop, the dust in the raws, the compression and the shot numbers
whatever the roll already says. It runs exactly what the window runs, in
the same order and on the same measurement: the roll's base, balance and
matrix are read once, kept in the document as `roll.measured` -- the
window reads them from there too -- and every output is made from them.
`develop` and `apply` each measure the roll for themselves and keep
nothing; they stay, for a run of one of the two.

How the pictures are developed is the roll's setting (`⌘K` in the window,
`develop --warmth ...` or `roll` here), not a flag of this.
"""

import os
import pathlib
import sys

from . import develop, document, flatfield, negative, positive


def add_args(ap):
    ap.add_argument("rolls", nargs="+", metavar="ROLL_DIR",
                    help="folders of scans, each with its frames.json")
    ap.add_argument("--no-positives", dest="positives", action="store_false",
                    help="no developed pictures in developed/")
    ap.add_argument("--no-raws", dest="raws", action="store_false",
                    help="no positive DNGs in frames/")
    ap.add_argument("--negatives", action="store_true",
                    help="the same frames not inverted, into negatives/ "
                         "(not kept: a question about this run)")
    ap.add_argument("--format", choices=("jpeg", "tiff16"),
                    help="of the developed pictures (kept in the document)")
    ap.add_argument("--crop", choices=positive.CROPS,
                    help="of the raws: outer, the tilted frame's bounding box, "
                         "or inner (kept in the document)")
    ap.add_argument("--dust", dest="dust", action="store_true", default=None,
                    help="the dust marks out of the raws too (kept)")
    ap.add_argument("--no-dust", dest="dust", action="store_false")
    ap.add_argument("--uncompressed", dest="compress", action="store_false",
                    default=None, help="DNGs without lossless JPEG (kept)")
    ap.add_argument("--compress", dest="compress", action="store_true")
    ap.add_argument("--numbered", dest="numbered", action="store_true",
                    default=None, help="shot number in front of every name (kept)")
    ap.add_argument("--not-numbered", dest="numbered", action="store_false")
    ap.add_argument("--workers", type=int, metavar="N",
                    help="processes (default: the machine's cores, up to 8)")


def _tick(what):
    def tick(i, total):
        print(f"\r  {what} {i + 1}/{total}", end="", flush=True)
    return tick


def measure(roll, doc, workers):
    """`progress.measure_base` without its window: the roll's statistics,
    measured once and kept as `roll.measured`, or the kept ones if they
    were measured under the settings now in force."""
    taste = dict(develop.TASTE) | dict(doc["roll"].get("develop") or {})
    if not develop.measured_ready(doc["roll"].get("measured"), taste):
        frames = develop.frames_of(roll, doc)
        if not frames:
            return None
        stats = develop.roll_statistics(frames, taste["gamma"], workers,
                                        _tick("measuring the roll"),
                                        flatfield.of(taste))
        print()
        kept = develop.measured_of(stats, taste)
        if not kept:
            return None
        doc["roll"]["measured"] = kept
        document.save(doc, roll)
    return develop.stats_of(doc["roll"]["measured"], taste, roll)


def one(roll, args, workers):
    doc = document.load(roll)
    n = sum(len(it["parts"]) for it in doc["items"].values()
            if it["state"] != "skipped")
    print(f"{roll}: {n} frames")
    if not n:
        print("  nothing to process: no frames yet", file=sys.stderr)
        return 1
    # What the sheet writes into the roll before it starts.
    taste = dict(develop.TASTE) | dict(doc["roll"].get("develop") or {})
    if args.format:
        taste["format"] = args.format
    doc["roll"]["develop"] = taste
    app = doc["roll"].setdefault("apply", {})
    for key in ("crop", "dust", "compress"):
        if getattr(args, key) is not None:
            app[key] = getattr(args, key)
    if args.numbered is not None:
        doc["roll"]["numbered"] = args.numbered
    flatfield.of(taste)                 # a missing calibration stops here
    document.save(doc, roll)

    stats = measure(roll, doc, workers)
    if stats is None:
        print("  the roll could not be measured", file=sys.stderr)
        return 1
    said = []
    if args.positives:
        out_dir = roll / "developed"
        results, ev = develop.develop_roll(roll, doc, out_dir, taste, workers,
                                           _tick("developing"), stats)
        print()
        develop.record(out_dir, taste, ev, results)
        said.append(f"{len(results)} pictures in developed/")
    if args.raws:
        settings = positive.settings_of(doc)
        out_dir = roll / "frames"
        results, ev = positive.write_roll(roll, doc, out_dir, settings, workers,
                                          _tick("raw DNGs"), stats)
        print()
        positive.record(out_dir, settings, ev, results)
        said.append(f"{len(results)} raw DNGs in frames/")
    if args.negatives and (args.positives or args.raws):
        out_dir = roll / negative.FOLDER
        crop = positive.settings_of(doc)["crop"]
        results = negative.write_roll(
            roll, doc, out_dir, taste, crop, workers, _tick("negatives"),
            stats, pictures=args.positives, raws=args.raws,
            compress=bool(app.get("compress", True)))
        print()
        negative.record(out_dir, taste, crop, results)
        said.append(f"{len(results)} negatives in {negative.FOLDER}/")
    print("  " + ("; ".join(said) if said else "nothing asked for"))
    return 0


def run(args):
    workers = args.workers or min(8, os.cpu_count() or 1)
    bad = 0
    for roll in map(pathlib.Path, args.rolls):
        try:
            bad += one(roll, args, workers) != 0
        except (FileNotFoundError, ValueError) as e:
            print(f"{roll}: {e}", file=sys.stderr)
            bad += 1
    return 1 if bad else 0
