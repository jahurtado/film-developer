"""Write every frame out as a positive DNG, exactly where the document says.

    film-developer apply ROLL_DIR
    film-developer apply ROLL_DIR -o somewhere/else

The step for whoever develops elsewhere: the negative inverted on its own
Bayer mosaic and written as a raw a digital camera could have made -- a
CFA DNG, never demosaiced, mirrored and turned on the mosaic, cropped by
translation to the rectangle inside the tilted frame, with the camera's
colour tags, the roll's balance as the neutral, and nothing the sensor
measured lost on the way into 16 bits (`positive.py` has the whole of
it). Lightroom, Capture One, darktable or RawTherapee open it as they
would a camera raw and do the rest.

It reads the document and re-reads nothing: whatever `detect` measured
and a person corrected is what gets cut. The inversion's gamma and the
gamma is the roll's `develop` setting, so the DNG and `develop`
agree on what the positive is.

**The originals are never written over.** Output lands in
`ROLL_DIR/frames/`, or wherever `-o` says, and there is no in-place mode
to ask for: the scan is the only copy of what the sensor saw, and a
positive can always be made again from the document. One scan makes one
file, or two when the film is half frame (`-a`, `-b`). The crop is the
tilted frame's bounding box, edges and all, unless `--crop inner` (or the
document's `apply.crop`) asks for the rectangle inside it.

The crop of the *negative*, untouched, is written by `⌘R` in the editor
when its sheet asks for it (`negative.py`), not here yet; the linear DNG
for a frame that has to be turned is not written at all.
"""

import json
import os
import pathlib
import sys

import numpy as np

from . import document, positive


def add_args(ap):
    ap.add_argument("roll", metavar="ROLL_DIR",
                    help="folder of scans with its frames.json")
    ap.add_argument("-o", "--output", metavar="DIR",
                    help="where the positives go (default: ROLL_DIR/frames/)")
    ap.add_argument("--crop", choices=positive.CROPS,
                    help="outer: the tilted frame's bounding box, its edges "
                         "with it, for the developer to straighten and crop "
                         "(default, kept in the document). inner: the "
                         "rectangle inside the frame, edges out")
    ap.add_argument("--dust", dest="dust", action="store_true", default=None,
                    help="fill the dust marked in the window (s) on the "
                         "mosaic too, before any demosaic (kept in the "
                         "document; off by default)")
    ap.add_argument("--no-dust", dest="dust", action="store_false",
                    help="leave the marked dust in the raws")
    ap.add_argument("--uncompressed", dest="compress", action="store_false",
                    default=None,
                    help="write the mosaic as it is, not as lossless JPEG "
                         "(kept in the document; compressed by default)")
    ap.add_argument("--compress", dest="compress", action="store_true",
                    help="lossless JPEG again")
    ap.add_argument("--workers", type=int, metavar="N",
                    help="processes (default: the machine's cores, up to 8)")


def run(args):
    roll = pathlib.Path(args.roll)
    doc = document.load(roll)
    settings = positive.settings_of(doc)
    if args.crop:
        settings["crop"] = args.crop
        doc["roll"].setdefault("apply", {})["crop"] = args.crop
        document.save(doc, roll)
    if args.dust is not None:
        settings["dust"] = args.dust
        doc["roll"].setdefault("apply", {})["dust"] = args.dust
        document.save(doc, roll)
    if args.compress is not None:
        settings["compress"] = args.compress
        doc["roll"].setdefault("apply", {})["compress"] = args.compress
        document.save(doc, roll)
    out_dir = pathlib.Path(args.output) if args.output else roll / "frames"
    workers = args.workers or min(8, os.cpu_count() or 1)
    n = sum(len(it["parts"]) for it in doc["items"].values()
            if it["state"] != "skipped")
    print(f"{n} frames in {roll} -> {out_dir}")

    def tick(i, total):
        print(f"\r  {i + 1}/{total}", end="", flush=True)

    results, roll_ev = positive.write_roll(roll, doc, out_dir, settings, workers, tick)
    print()
    if not results:
        print("  nothing to write: no frames in the document", file=sys.stderr)
        return 1
    blank = sorted(k for k, v in results.items() if v["blank"])
    base = [v["baseline_exposure"] for v in results.values()]
    print(f"  base colour R/G {roll_ev['base_colour'][0]:.3f}, "
          f"B/G {roll_ev['base_colour'][2]:.3f}; neutral from the roll balance "
          f"{roll_ev['balance_log2'][0]:+.2f} / {roll_ev['balance_log2'][2]:+.2f} EV; "
          f"gamma {roll_ev['gamma']}")
    print(f"  BaselineExposure median {np.median(base):+.2f} EV, "
          f"from {min(base):+.2f} to {max(base):+.2f}: what a developer that "
          f"honours it adds to show the median at 18 % grey")
    if blank:
        print(f"  {len(blank)} blank: {', '.join(blank)}")
    positive.record(out_dir, settings, roll_ev, results)
    print(f"  written {len(results)} DNG and positives.json")
    return 0
