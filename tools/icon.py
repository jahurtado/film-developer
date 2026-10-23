#!/usr/bin/env python3
"""Draw the application icon, one drawing per size.

    uv run python tools/icon.py

Generated rather than committed as an opaque blob, so it can be argued
with: the shapes are here, in numbers, and a change is a diff.

**The subject is the whole program, in one picture.** A colour negative,
orange with its mask and lying a few degrees off square, the way film
sits in a holder; four crop marks squared on it; and inside the marks
the same scene **developed** -- light sky, dark mountain, a sun -- where
outside them it is still the negative, dense sky and clear mountain.
Find the frame, straighten it, cut it, develop it.

**The accent is pink**, the colour the editor already uses for a frame
a person placed.

**Each size is drawn at its own size**, never reduced from the big one.
At 16 px the sun goes, because it is one pixel and reads as dirt; what is
left is an orange tilted card with a pale straight rectangle on it,
which is the idea with everything unaffordable removed.
"""

import os

from PIL import Image, ImageDraw

BG = (18, 18, 22)             # the window's background
ACCENT = (255, 106, 193)      # the editor's colour for a frame placed by hand
MASK = (214, 128, 66)         # the orange mask of colour negative film
NEG_SKY = (92, 44, 24)        # a bright sky is dense on the negative
NEG_SUN = (60, 28, 14)        # and the sun denser still
POS_SKY = (196, 214, 232)     # the same scene developed
POS_MTN = (44, 52, 64)
POS_SUN = (255, 240, 205)
ANGLE = 7                     # degrees off square: visibly tilted, not falling
SIZES = (16, 32, 64, 128, 256, 512)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "src", "film_developer", "icons")
SS = 4                        # supersample, then shrink: edges have to be clean


def scene(w, box, sky, mtn, sun):
    """A sky, a sun and a mountain inside `box`, on a transparent layer of
    `w` px; no sun when `sun` is None."""
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    im = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle(box, fill=sky)
    if sun is not None:
        r = H * 0.16
        cx, cy = x0 + W * 0.70, y0 + H * 0.32
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=sun)
    d.polygon([(x0, y1), (x0, y0 + H * 0.62), (x0 + W * 0.30, y0 + H * 0.38),
               (x0 + W * 0.52, y0 + H * 0.62), (x0 + W * 0.68, y0 + H * 0.50),
               (x1, y0 + H * 0.78), (x1, y1)], fill=mtn)
    return im


def icon(size):
    """One icon, drawn for `size`."""
    w = size * SS
    im = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, w - 1, w - 1],
                                         radius=int(w * 0.22), fill=BG)
    detail = size > 16
    fw = w * 0.66
    fh = fw / 1.5                 # 3:2, the real aspect
    cx = cy = w / 2

    # The negative: the mask's rebate around a frame, the scene in it with
    # its tones the wrong way round, all of it turned off square.
    neg = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    ImageDraw.Draw(neg).rectangle([cx - fw * 0.58, cy - fh * 0.62,
                                   cx + fw * 0.58, cy + fh * 0.62], fill=MASK)
    neg.alpha_composite(scene(w, [cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2],
                              NEG_SKY, MASK, NEG_SUN if detail else None))
    im.alpha_composite(neg.rotate(ANGLE, resample=Image.BICUBIC, center=(cx, cy)))

    # The crop: straight, inside the frame, and developed.
    box = [cx - fw * 0.40, cy - fh * 0.40, cx + fw * 0.40, cy + fh * 0.40]
    im.alpha_composite(scene(w, box, POS_SKY, POS_MTN, POS_SUN if detail else None))
    d = ImageDraw.Draw(im)
    stroke = max(2 * SS, int(w * (0.05 if size <= 32 else 0.03)))
    arm = fw * 0.16
    x0, y0, x1, y1 = box
    for x, y, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                         (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([(x, y + sy * arm), (x, y), (x + sx * arm, y)],
               fill=ACCENT, width=stroke, joint="curve")
    return im.resize((size, size), Image.LANCZOS)


def main():
    os.makedirs(OUT, exist_ok=True)
    for s in SIZES:
        p = os.path.join(OUT, "icon-%d.png" % s)
        icon(s).save(p)
        print("  %s" % os.path.relpath(p, HERE))
    icns = os.path.join(OUT, "icon.icns")
    icon(512).save(icns, format="ICNS",
                   sizes=[(s, s) for s in (16, 32, 128, 256, 512)])
    ico = os.path.join(OUT, "icon.ico")
    icon(256).save(ico, format="ICO", sizes=[(s, s) for s in SIZES if s <= 256])
    for p in (icns, ico):
        print("  %s" % os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()
