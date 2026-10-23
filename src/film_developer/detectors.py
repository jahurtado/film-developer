"""Finding the frames, and the settings that are not the finding's business.

One detector, `fit`, and it is in `fitframe`. This is where a scan or a
folder is handed to it and where the settings that are nobody's detector --
the crop -- are applied afterwards.

    detectors.run(path, settings)              ->  an item record
    detectors.run_roll(paths, settings)        ->  {scan name: record}

`settings` is `roll` from the document: whatever the person chose, once, for
the whole folder.
"""

import numpy as np

from . import fitframe

# Below this the angle peak is not a measurement: on the sample set the one
# scan whose tilt disagreed with its neighbours scored 1.08 where the rest
# scored 1.29-1.65. It is not refused -- it is measured, marked, and shown.
WEAK_MARGIN = 1.15

# A frame does not fill the scan. Above this share of both sides it is the
# scan itself, which on a 6064x4042 sensor is 3:2 to four figures and so fits
# a film ratio better than any real frame does.


def fit(path, settings):
    """A rectangle whose sides lie on edges, whose inside is film, and whose
    size the roll knows -- see `fitframe`. On its own it measures a scan
    alone; `run_roll` is the form that lets the roll have its say, and is
    what `detect` uses. 97 of the 98 labelled scans, the one miss marked."""
    item, _ = fitframe.measure(path, settings)
    return item


def run(path, settings, rolled=None):
    """Measure one scan, then apply the settings that are not the detector's
    business.

    With `rolled` -- the format and size clusters the roll settled on -- the
    scan goes through the same two passes `run_roll` gives it: once alone to
    see what size it is, then again with the roll's height, width, gap and
    row for that size. That is what makes one scan measured again come out
    the way it came out the first time.
    """
    settings = settings or {}
    if rolled and rolled.get("clusters") and settings.get("one_roll", True):
        # **The format is the roll's, in both passes.** `measure` takes the
        # format from the settings and the sizes from the roll, so with
        # `format: auto` in the document -- which is the default, and what
        # every roll measured before the summary existed has -- a scan of a
        # half-frame roll was still free to come back as one full frame.
        # Two halves and the gap between them are 3:2, so that is exactly
        # what a strip looks like measured alone: SDIM0903 of roma-half
        # read 1 part, full, on its own and 2 parts, half, with the roll.
        want = dict(settings)
        if rolled.get("format") in ("full", "half"):
            want["format"] = rolled["format"]
        item, first = fitframe.measure(path, want)
        h_own = (float(np.median([q[3] for q in first["parts"]]))
                 if first and first.get("parts") else None)
        c = fitframe.cluster_for(rolled["clusters"], h_own,
                                  settings.get("same_size", False))
        if c:
            item, _ = fitframe.measure(path, want, {
                "format": rolled.get("format"), "h": c["h"], "w": c["w"],
                "gap": c["gap"], "y0": c["y0"],
                "same_size": bool(settings.get("same_size", False)),
                "same_gap": bool(settings.get("same_gap", False))})
    else:
        item = fit(path, settings)
    inset(item, settings.get("crop", 0))
    return item


def run_roll(paths, settings, progress=None, summary=None):
    """Measure a whole folder: {scan name: item}, the roll consulted, every
    item through the same settings.

    `fit` reads the roll twice -- heights to +-1 %, the gap to +-40 px, a
    side in dispute arbitrated by the roll -- so a folder is measured as one
    thing and not as a pile of scans.
    """
    settings = settings or {}
    items = fitframe.measure_roll(paths, settings, progress, summary=summary)
    for item in items.values():
        inset(item, settings.get("crop", 0))
    return items


def inset(item, pixels):
    """Take `pixels` off every side of every frame, or add them if negative.

    A setting rather than a detector's business, and applied after all of
    them, because it is not about finding the frame: it is about what to do
    once it is found. A frame edge is a soft ramp a few pixels wide, and where
    inside it a person wants the cut is a matter of taste and of the holder --
    a few pixels in to be sure of no rebate, or a few out to keep every last
    bit of the picture.

    In scan pixels, so it means the same thing whatever the scan's size, and
    on all four sides, so it does not move the frame.
    """
    if not pixels:
        return item
    for p in item.get("parts", []):
        w, h = p["size"]
        p["size"] = [round(max(1.0, w - 2 * pixels), 1),
                     round(max(1.0, h - 2 * pixels), 1)]
        p.setdefault("evidence", {})["crop"] = pixels
    return item
