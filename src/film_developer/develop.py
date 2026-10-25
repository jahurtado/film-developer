"""Develop: the frames of a roll, inverted, balanced and written as positives.

    film-developer develop ROLL_DIR
    film-developer develop ROLL_DIR --warmth 0.1 --tint -0.05 --exposure 0.3
    film-developer develop ROLL_DIR --format tiff16 -o /somewhere

Reads `frames.json`, develops every frame at full resolution and writes one
image per frame into `ROLL_DIR/developed/`. Nothing here touches the
scans or the document beyond the `develop` settings it was given.

What is measured, and what is taste
-----------------------------------

The development splits into things that are **measured** -- on the scan,
on the frame, or on the whole roll -- and things that are **taste**, which
are global biases applied identically to every photo. A photo is never
adjusted on its own except by rules that read its own statistics within
fixed thresholds; there is no per-photo setting anywhere.

Measured:

- **The film's base**, per channel: the transmittance of unexposed film.
  It is what the orange mask *is*, and dividing by it is what removes the
  mask. It is read from the rebate of the scan where there is one, but
  with a holder that fits the frame there is none worth the name -- only
  its penumbra, which is darker -- so the base is also bounded below by
  the lightest 0.5 % of the frame itself: nothing on a negative is
  lighter than unexposed film. Its colour (R/G, B/G) is a property of the
  film, so it is the median over the roll. Measured before this rule was
  added: on 23 of 28 Cinestill scans and 18 of 23 Portra the rebate read
  darker than the frame, and everything lighter than it went to black.
- **The inversion**: scene light `(base / T) ** (1 / gamma) - 1`, the
  ratio floored at unexposed film. It is the film's own equation -- density
  `d = log10(base / T)` against that base, then `10^(d / gamma) - 1` with
  the negative's gamma -- with the log and the antilog cancelled, which
  they do exactly; see `invert` for what that saves and why the density
  range survives it. A linear inversion,
  `max - pixel`, neither removes the mask nor undoes the film's curve,
  and has to neutralise the mask afterwards with a grey world balance. This does both in one measured step, and a negative
  scanned at three exposures develops to the same picture.
- **No colour matrix** (`MATRIX`, the identity): camera RGB goes on as
  the picture's RGB. A camera's matrix undoes how its filters overlap
  under scene light, and is linear in that light; here it would act after
  the inversion, on `(base/T)**(1/gamma)`, which is not light the camera
  saw. Measured with and without it: the D5600's (diagonal 1.59 / 1.55 /
  1.40) multiplied saturation by 1.9 and chroma noise by 1.6-1.9 and
  pushed red, the fp's by 1.6 and 1.7; without either, the two cameras
  scanning one negative come out alike (saturation 0.43 and 0.40, chroma
  noise 0.049 and 0.045 at the same size), and both look
  more neutral. And the fp writes two different `ColorMatrix` with one
  firmware, so a matrix made one camera develop two ways.
- **The roll's balance**: for each photo, the gain that would make its
  *least coloured* mid-tones neutral (`WB_KEEP`: grey world is only
  evidence where there is grey, and half a beach frame is sky); the
  median over the roll is the film plus the light, and is applied whole. The photo keeps a residual, at `WB_STRENGTH` of
  what it asks and no more than `WB_LIMIT` -- a sunset is not made grey,
  a shot under tungsten is brought most of the way back.
- **Exposure**: the median of the luminance goes to middle grey, unless
  that would put the brightest half per cent more than `CLIP_MAX` above
  its target or a bright scene more than `DARK_MAX` below it.
- **Shadows**: the display curve's shadow slope drops, per photo and down
  to `SHADOW_AUTO_MIN`, until the 5th percentile reaches `SHADOW_FLOOR`.
  That is what opens a backlit interior without touching a normal scene.
- **Blank frames**: a frame whose density does not vary is an unexposed
  one; it is written dark rather than as stretched noise, and reported.

Taste, in the roll's `develop` settings and on the command line: `gamma`
(the negative's contrast), `exposure` (EV), `warmth` and `tint` (EV, red
against blue and green against magenta), `shadows` and `highlights` (the
display curve's slopes), `saturation`. Their defaults are what a first
round of looking chose, and nothing more.

**There was a flat field** -- the copy stand's falloff, a 25 % vignette
measured on the corpus's one blank scan, divided out of every scan before
anything was read. It is gone: one fewer thing between the negative and
the number, one fewer setting that could be measured under one value and
developed under another. What it corrected it no longer corrects.
"""

import json
import re
import math
import pathlib
import sys

import numpy as np
from scipy import ndimage

from . import detect, document, findframe, fitframe

# ------------------------------------------------------------- settings

# What a person may set, once per roll. Written into the document so a
# second run develops the same way, and overridden by the command line.
TASTE = {
    "gamma": 0.65,        # the negative's contrast: scene log-exposure = density / gamma
    "exposure": 0.0,      # EV on top of the automatic exposure
    "warmth": 0.08,       # EV added to red and taken from blue
    "tint": 0.0,          # EV added to green and taken from red and blue
    "shadows": 0.80,      # slope in log2 below middle grey; under 1 opens the shadows
    "highlights": 0.85,   # slope above middle grey
    "saturation": 1.05,
    "contrast": 1.0,      # both slopes at once, after each frame's own opening
    "pivot": 0.0,         # EV over middle grey where the curve bends
    "noise": 0.0,         # noise reduction, 0 none to 1 strong (`denoise`)
    "sharpen": 0.0,       # unsharp amount on the luminance (`sharpen`)
    "base": "auto",       # where the base comes from. auto: the roll's level, unless the
                          # folder is not one roll (`roll.one_roll`), where each
                          # frame is on its own. roll: the roll's level whatever
                          # the folder is. frame: each frame's own floor
    "crop": 0.0,          # per cent off every side of every frame, or onto it if
                          # negative. **Applied here and nowhere else.** It was
                          # a box under the canvas that drew a dashed rectangle
                          # and nothing more: the caption under it said "the
                          # positives only" and the develop never read it, so a
                          # crop set in the window moved a line and changed no
                          # picture. Taken off in `cut`, on top of `CUT_INSET`.
    "order": "after",     # after: demosaic, then invert. mosaic: invert every photosite, then demosaic
    "format": "jpeg",     # jpeg | tiff16
    "flat": None,         # the rig's light falloff: the name of a calibration
                          # in `flatfield.library()`, fitted from blank frames
                          # with `film-developer flat`, or None for none. Of the
                          # copy stand, not the roll, so it outlives rolls;
                          # the roll only says which one it was scanned under.
}

# What a photo may do on its own, within these. Constants, not settings:
# they are the rules, and the settings above are the taste.
MEDIAN_TARGET = 0.18      # linear luminance the median goes to
HIGHLIGHT_PCT = 99.5      # ... unless this percentile ends more than
HIGHLIGHT_TARGET = 0.92   #     CLIP_MAX EV above this target ...
CLIP_MAX = 0.5
DARK_MAX = 0.7            # ... or more than this below it (a bright scene stays bright)
FLOOR = 8                 # raw levels above black under which a photosite measured nothing
MEDIAN_DROP = 1.0        # EV the highlights may take the median under its target, at most
SHADOW_LEVEL = 0.10      # the near-black band of the finished picture
SHADOW_SHARE = 0.35      # of the frame, the most that may sit inside it
SHADOW_LIFT = 0.6        # EV the shadow rule may add, at most
SPAN_FLOOR = 3.0         # stops between the 5th and 95th percentiles the curve
                         # aims to leave the picture with, at least
SLOPE_MAX = 1.15         # the most the curve may open to get there
SHADOW_FLOOR = 0.012      # if the 5th percentile is under this, the shadow slope drops
SHADOW_AUTO_MIN = 0.62    # ... but never under this
SHOULDER_FROM = 0.6       # a soft roll-off from here to 1
BLACK_PCT = 0.2           # the percentile taken as black ...
BLACK_TARGET = 0.003      # ... where it goes ...
BLACK_MAX = 0.03          # ... and the most it may be moved
WB_KEEP = 50              # of the mid-tone band, the least coloured share the
                          # balance is read from -- see `wanted_balance`
WB_STRENGTH = 0.7         # share of a photo's own balance it keeps after the roll's
WB_LIMIT = 0.6            # EV per channel it may move
BLANK_RANGE = 0.12        # density range (p99 - p1) under which a frame is blank
BASE_TOP_PCT = 99.5       # the frame's lightest share that bounds the base
SATURATED = 0.9           # of the sensor's white: above this a photosite is not film
REBATE_VOTE = 0.5         # share of a roll's scans whose rebate must be film to use it
BASE_ROLL_KEEP = 75       # the share of a roll's own readings the roll's base
                          # is the median of, counted from the top.
                          #
                          # Every reading of the base is a **lower bound**. A
                          # frame's lightest pixels equal the base under a true
                          # black and fall below it where the scene has none; a
                          # rebate equals it in full light and falls below it in
                          # the holder's penumbra. So the base is at the top of
                          # the readings, and a middle statistic of them is
                          # wrong by however dark the typical scene is.
                          #
                          # roma-half is the only roll in the corpus with a
                          # **blank scan** -- bare film on the light table,
                          # SDIM0932 -- which is the base itself and the only
                          # anchor there is. Against it, per unit of
                          # exposure:
                          #
                          #   the rebate, median of the film ones   -0.227 stops
                          #   the frames' floors, median            -0.191
                          #   the top 40 %, median                  -0.077
                          #   the top 25 %, median                  -0.030
                          #   the top 20 %, median                  -0.023
                          #   the frames' floors, max               +0.031
                          #
                          # The top quarter's median, not the maximum (one
                          # light leak takes the roll with it) and not a single
                          # percentile (it is one reading; this is the middle of
                          # nineteen). It reads the same whether the rebate is
                          # consulted or not -- where the floors are highest the
                          # frame beats its own rebate anyway -- so one rule
                          # serves both kinds of roll. Only 5 of 74 floors come
                          # out over the base at all.
BASE_ROLL_MIN = 3         # readings the top share must hold, or it takes them all
BOX_AREA = 0.15           # of the biggest box: a smaller one may not stand for the base
STAT_INSET = 0.04         # of each side, the margin the roll's statistics keep
                          # off the frame's edge. The picture written is the
                          # whole frame, but a rectangle a little too large, or
                          # one whose angle is a little off, reaches into the
                          # holder -- denser than any film, so the inversion
                          # turns it into enormous values. DSC_0286 (Nikon) read
                          # its brightest 0.5 % at 27.5 instead of 12.6 and a
                          # maximum of 531 instead of 55, and the highlight rule
                          # developed it 0.88 EV dark, while the same negative
                          # placed by hand came out right.
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def add_args(ap):
    ap.add_argument("roll", metavar="ROLL_DIR",
                    help="folder of scans with its frames.json")
    ap.add_argument("-o", "--output", metavar="DIR",
                    help="where to write the positives (default: "
                         "ROLL_DIR/developed)")
    for key, val in TASTE.items():
        if key == "format":
            ap.add_argument("--format", choices=("jpeg", "tiff16"),
                            help="jpeg (8-bit sRGB) or tiff16 (16-bit sRGB) "
                                 "(default: what the document says, else jpeg)")
        elif key == "order":
            ap.add_argument("--order", choices=("after", "mosaic"),
                            help="after: demosaic the negative, then invert "
                                 "(default). mosaic: invert every photosite "
                                 "against its channel's base and demosaic the "
                                 "positive, which is what the demosaic was "
                                 "designed for; measured, 3-15 %% less false "
                                 "colour, at the cost of the positive passing "
                                 "through the raw's integer range")
        elif key == "flat":
            ap.add_argument("--flat", metavar="NAME",
                            help="divide by this rig's light falloff, a "
                                 "calibration made with `film-developer "
                                 "flat`; `none` for none (default: what the "
                                 "document says, else none)")
        elif key == "base":
            ap.add_argument("--base", choices=("auto", "roll", "frame"),
                            help="where the base comes from: `auto` is the "
                                 "roll's own rule -- its level where the "
                                 "rebate is film, each frame's own deepest "
                                 "shadows where it is not (default); `roll` "
                                 "forces the roll's level on every frame, "
                                 "which is right when the scans share one "
                                 "base and the roll cannot tell, and makes "
                                 "frames whose scenes have no true black "
                                 "come out lighter; `frame` forces each "
                                 "frame's own floor and does not consult the "
                                 "rebate at all")
        else:
            ap.add_argument(f"--{key}", type=float, metavar="X",
                            help=f"(default: what the document says, else {val})")
    ap.add_argument("--workers", type=int, metavar="N",
                    help="processes to develop with (default: the cores, up to 8)")


# --------------------------------------------------------------- planes

def linear(path, half, base_colour=None, flat=None):
    """Demosaiced camera RGB, linear, as transmittance in 0..1; and the
    scale from its pixels to the scan's.

    `base_colour` is the roll's (R/G, 1, B/G) of unexposed film. Given, it
    is divided out **before** the demosaic, as LibRaw's own multipliers:
    AHD chooses its interpolation direction from colour differences and
    assumes them smooth, which the orange mask breaks. Measured on three
    scans of three rolls: the high-frequency chroma of the blue channel
    (where false colour lives) drops 17-30 %, red is unchanged, and the
    positives differ by 0.16 % on average. Free, so it is done.

    `flat` is a calibration of the rig's light falloff (`flatfield`). Given,
    the demosaiced image is divided by it: smooth, so before or after the
    demosaic is the same thing, and after it is one map and not four.
    """
    import rawpy
    wb = [1, 1, 1, 1] if base_colour is None else \
        [1 / base_colour[0], 1.0, 1 / base_colour[2], 1.0]
    with rawpy.imread(str(path)) as r:
        # Before postprocess: with half_size LibRaw reports the half width
        # afterwards, and the scale comes out as 1 (see image_io.preview).
        full_h, full_w = r.raw_image_visible.shape
        rgb = r.postprocess(half_size=half, output_bps=16, gamma=(1, 1),
                            no_auto_bright=True, use_camera_wb=False,
                            user_wb=wb, output_color=rawpy.ColorSpace.raw,
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD)
    rgb = rgb.astype(np.float32) / 65535.0
    if flat:
        from . import flatfield
        flatfield.check(flat, path, full_w, full_h)
        flatfield.divide_rgb(rgb, flat, full_w, full_h)
    return rgb, full_w / rgb.shape[1]


# What the develop applies from camera RGB to the picture's: nothing (see the
# module docstring). Kept as a matrix in `roll.measured` so a document
# measured through a camera's can be told apart and measured again.
MATRIX = np.eye(3, dtype=np.float32)


def camera_matrix(path):
    """The 3x3 LibRaw applies from camera RGB to linear sRGB, fitted on the
    scan itself -- what the develop no longer applies (`MATRIX`); kept
    for comparison. Both outputs are linear, so away
    from clipping the fit is exact, and no ColorMatrix tag or illuminant has to be interpreted."""
    import rawpy
    raw, _ = linear(path, True)
    with rawpy.imread(str(path)) as r:
        srgb = r.postprocess(half_size=True, output_bps=16, gamma=(1, 1),
                             no_auto_bright=True, use_camera_wb=False,
                             user_wb=[1, 1, 1, 1], output_color=rawpy.ColorSpace.sRGB,
                             demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD
                             ).astype(np.float32) / 65535.0
    a, b = raw.reshape(-1, 3), srgb.reshape(-1, 3)
    sel = (a.max(1) < 0.9) & (a.min(1) > 0.002)
    idx = np.random.RandomState(0).choice(np.nonzero(sel)[0],
                                          min(200000, int(sel.sum())), replace=False)
    m, *_ = np.linalg.lstsq(a[idx], b[idx], rcond=None)
    return m.astype(np.float32)


CUT_INSET = 0.01          # of each side, what `cut` trims by default: the soft
                          # ramp of the exposure's edge, out of every statistic


def crop_inset(taste):
    """What to trim off each side: the ramp, plus whatever the roll's `crop`
    asks for. Negative asks for more of the negative, not less."""
    return CUT_INSET + float((taste or {}).get("crop", 0.0)) / 100.0


def frame_box(shape, taste):
    """Where the frame as found sits inside a cut made with `crop_inset`,
    as `(y0, y1, x0, x1)` -- the box `finish` may read.

    The cut is centred on the frame, so this is the middle of it: the crop
    grows or shrinks what came out, and the frame stayed where it was.
    """
    h, w = shape[:2]
    inset = crop_inset(taste)
    share = (1 - 2 * CUT_INSET) / max(1 - 2 * inset, 1e-6)
    if share >= 0.999:
        return None
    dy, dx = int(h * (1 - share) / 2), int(w * (1 - share) / 2)
    return (dy, h - dy, dx, w - dx)


def cut(rgb, scale, part, angle, inset=CUT_INSET):
    """The frame, straightened and cropped out of the linear image, trimmed
    by `inset` of each side so the soft edge of the exposure stays out."""
    cx, cy = (v / scale for v in part["centre"])
    w, h = (v / scale for v in part["size"])
    H, W = rgb.shape[:2]
    # **Room for a negative inset.** The pad used to be half the frame, so a
    # `crop` that asks for *more* than the frame -- the rebate, the sprocket
    # holes -- wanted a rectangle bigger than the square it was cut from:
    # the slice went negative, Python read it from the other end, and a
    # 874-pixel frame came out 32 pixels tall.
    pad = int(max(w, h) * max(1.0, 1 - 2 * inset) / 2 + 8)
    x0, y0 = int(cx) - pad, int(cy) - pad
    sub = np.zeros((2 * pad, 2 * pad, 3), np.float32)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + 2 * pad), min(H, y0 + 2 * pad)
    sub[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    rot = ndimage.rotate(sub, angle, reshape=False, order=1, mode="nearest")
    ccx, ccy = cx - x0, cy - y0
    w2, h2 = w * (1 - 2 * inset) / 2, h * (1 - 2 * inset) / 2
    y0i, y1i = max(0, int(ccy - h2)), max(1, int(ccy + h2))
    x0i, x1i = max(0, int(ccx - w2)), max(1, int(ccx + w2))
    return rot[y0i:y1i, x0i:x1i]


def base_mask(rgb, scale, item, part=None):
    """Which pixels of a scan are unexposed film: the rebate, and what the
    base is read from. The frames grown by 6 %, the holder and the light
    table (the detector's own classes, at a quarter) and a margin around
    them are left out.

    The margins are in scan pixels, whatever `scale` the image is at:
    read at half size with the full-size margins, the band of base between
    the sprocket holes of a Nikon scan was swallowed by the margin around
    the holes, and the rebate read 0.054 where at full size it read 0.116
    -- darker than the frame's own lightest 0.5 %, so the base fell back
    to the frame and the picture came out 17 % lighter in the base and
    crushed in the shadows.
    """
    H, W = rgb.shape[:2]
    s = max(1.0, float(scale))
    mask = np.ones((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]
    for p in item["parts"]:
        cx, cy = (v / scale for v in p["centre"])
        w, h = (v * 1.06 / scale for v in p["size"])
        r = math.radians(-item["angle"])
        co, si = math.cos(r), math.sin(r)
        dx, dy = xx - cx, yy - cy
        u, v = dx * co - dy * si, dx * si + dy * co
        mask &= ~((np.abs(u) < w / 2) & (np.abs(v) < h / 2))
    g = rgb[..., 1]
    g8 = g[:H // 4 * 4, :W // 4 * 4].reshape(H // 4, 4, W // 4, 4).mean((1, 3))
    direct, opaque, _ = fitframe.classify(-np.log10(np.clip(g8, 1e-4, None)))
    bad = ndimage.binary_dilation(np.kron(direct | opaque, np.ones((4, 4), bool)),
                                  iterations=max(1, round(12 / s)))
    full = np.zeros((H, W), bool)
    full[:bad.shape[0], :bad.shape[1]] = bad
    # Dark enough to be the holder at one end, bright enough to be the light
    # table at the other. A photosite above SATURATED of the sensor's white is
    # in the clipping shoulder and cannot be film: across the four rolls the
    # base has never read above 0.69, and on the one scan where the direct
    # light was not classified (DSC_0284, the brightest of a bracket) 6 % of
    # the sample sat at full scale and pulled the base 11 % high, which
    # developed that frame lighter than the same negative beside it. On every
    # other scan of the corpus not one sampled photosite is up there, so the
    # rule costs nothing where it is not needed.
    mask &= ~full & (g > 10 ** -2.2) & (g < SATURATED)
    # The frame cannot contain the light table, so its own lightest pixels
    # look like a scale that direct light has not touched, and over the four
    # rolls film sits at 0.5-1.6 times it and the light table at 3-6. Using
    # that as a ceiling was tried and **fails**, because the frame's
    # statistic is a lower bound and not a measurement: on a bright scene
    # with no deep shadow it sits far below the base, and twice it then
    # falls below real film. On roma it put one frame right (1.27 to 1.00
    # of the roll's base) and two wrong (0.93 to 0.57 and to 0.81). The
    # anchor has to track the base, and the frame's shadows do not.
    return ndimage.binary_erosion(mask, iterations=max(1, round(6 / s)))


def scan_exposure(path):
    """How much light this scan was given, in arbitrary but comparable
    units: shutter x sensitivity / aperture squared, from the file's own
    EXIF. None when the file does not say.

    It is the one exposure reference that owes nothing to the picture, the
    film or the base, and it is what lets a roll pool a quantity that is
    otherwise per scan.

    The aperture is in it **only where the file reports one**. On a copy
    stand it does not move, and a constant cancels: the roll's level is
    divided by this and multiplied back by it, so a factor shared by every
    scan of a roll changes nothing. Across the corpus the Sigma reports no
    aperture at all (88 scans of three rolls) and the Nikon reports f/8 on
    all 92, so today this is exactly the old shutter x ISO in both cases.
    It is here for the roll where somebody stops down halfway through --
    `roll_statistics` refuses to pool when some scans of a roll report an
    aperture and others do not, because then the factor does not cancel.

    **It is the nominal exposure, not the light.** One negative shot at
    1/125, 1/250 and 1/60 on the Nikon measured 1.000, 0.609 and 1.991
    times the light against a nominal 1.000, 0.500 and 2.083: the shutter
    is 18 % out at 1/250. Nothing in a file can fix that, and it is why a
    roll's level brought to a scan's exposure is exact on a roll shot at
    one setting and approximate on a bracketed one.
    """
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as t:
            for page in t.pages:
                tag = page.tags.get("ExifTag")
                v = tag.value if tag else None
                if isinstance(v, dict) and "ExposureTime" in v:
                    def num(x):
                        return x[0] / x[1] if isinstance(x, tuple) else float(x)
                    e = num(v["ExposureTime"])
                    iso = v.get("ISOSpeedRatings") or v.get("PhotographicSensitivity")
                    # Keep looking: a file can carry an Exif IFD on more
                    # than one page and only one of them complete. Bailing
                    # out on the first incomplete one made every Sigma scan
                    # read None.
                    if not (e > 0 and iso):
                        continue
                    light = float(e) * float(iso)
                    f = num(v["FNumber"]) if "FNumber" in v else None
                    return light / (f * f) if f and f > 0 else light
    except Exception:
        pass
    return None


def scan_aperture(path):
    """The f-number the file reports, or None where it reports none.

    **Zero counts as none.** The Sigma writes `FNumber` 0 on all 88 scans
    of its three rolls -- a manual lens with no electronics has no
    f-number to report -- and reading that as an aperture makes it look
    like a roll whose aperture is recorded on some scans and not others,
    which is the one case pooling has to refuse.

    Only for that consistency check in `roll_statistics`: a roll where
    some scans carry an aperture and some do not has `scan_exposure`
    meaning two different things inside it.
    """
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as t:
            for page in t.pages:
                tag = page.tags.get("ExifTag")
                v = tag.value if tag else None
                if isinstance(v, dict) and "FNumber" in v:
                    x = v["FNumber"]
                    f = x[0] / x[1] if isinstance(x, tuple) else float(x)
                    return f if f > 0 else None
    except Exception:
        pass
    return None


def rebate(rgb, scale, item, part=None):
    """Linear RGB of the film outside every frame of the scan, sampled."""
    mask = base_mask(rgb, scale, item, part)
    px = rgb[mask]
    if len(px) > 20000:
        px = px[np.random.RandomState(0).choice(len(px), 20000, replace=False)]
    return px


# ------------------------------------------------------------- the maths

def inside(fr, inset=STAT_INSET):
    """The middle of a cut frame, clear of its edge: what the statistics
    are read from. See STAT_INSET."""
    h, w = fr.shape[:2]
    my, mx = max(1, int(h * inset)), max(1, int(w * inset))
    return fr[my:h - my, mx:w - mx]


def frame_top(fr):
    """The frame's own lightest pixels, per channel: the bound that says the
    base cannot be darker than this, because nothing on a negative is more
    transparent than unexposed film.

    What must not be in it is anything that is not film. The light table
    seen through a sprocket hole is many times lighter than the base, and a
    rectangle placed a little wide -- or a scan where the gap between two
    frames falls inside one -- would put the light table in here and take
    the base with it. Measured across 48 scans of the four rolls not one
    such pixel gets in today, which is what a guard against a placement
    fault should look like when the placements are good.
    """
    g = fr[..., 1]
    ok = g < SATURATED
    if ok.sum() > 1000:
        # direct light is a plateau at the scan's own minimum density, and
        # `classify` is what the detector uses to find it
        try:
            small = g[::4, ::4]
            direct, _opaque, _ = fitframe.classify(
                -np.log10(np.clip(small / max(float(small.max()), 1e-6), 1e-4, None)))
            if direct.any():
                big = np.kron(direct, np.ones((4, 4), bool))
                lit = np.zeros(g.shape, bool)
                h, w = min(g.shape[0], big.shape[0]), min(g.shape[1], big.shape[1])
                lit[:h, :w] = big[:h, :w]
                if (ok & ~lit).sum() > 1000:
                    ok &= ~lit
        except Exception:
            pass
    px = fr[ok] if ok.sum() > 1000 else fr.reshape(-1, 3)
    return np.percentile(px, BASE_TOP_PCT, axis=0).astype(np.float32)


def measure(path, item, part, half, base_colour=None, inset=CUT_INSET,
            flat=None):
    """(frame, rebate base, frame top): the linear frame, and the two
    per-channel readings the base is taken from. With `base_colour` the
    frame comes out already balanced, and the base is grey. With `flat`,
    divided by the rig's light first (`linear`)."""
    rgb, scale = linear(path, half, base_colour, flat)
    reb = rebate(rgb, scale, item, part)
    fr = cut(rgb, scale, part, item["angle"], inset)
    del rgb
    top = frame_top(inside(fr))
    reb_base = (np.percentile(reb, 90, axis=0).astype(np.float32)
                if len(reb) > 500 else np.zeros(3, np.float32))
    return fr, reb_base, top


def inner_box(cx, cy, w, h, angle, inset=0.01):
    """The largest axis-aligned rectangle of the frame's aspect inside the
    tilted frame, about the same centre, less `inset` of each side, on
    even coordinates so a photosite keeps its colour. (x0, y0, w, h)."""
    w, h = w * (1 - 2 * inset), h * (1 - 2 * inset)
    a = abs(math.radians(angle))
    k = min(w / (w * math.cos(a) + h * math.sin(a)),
            h / (w * math.sin(a) + h * math.cos(a)))
    bw, bh = w * k, h * k
    return (int(cx - bw / 2) // 2 * 2, int(cy - bh / 2) // 2 * 2,
            int(bw) // 2 * 2, int(bh) // 2 * 2)


def invert_mosaic(path, item, part, roll_ratio, gamma,
                  trust_rebate=True, pooled=None, flat=None):
    """The inversion done on the Bayer mosaic, photosite by photosite.
    Returns (P, chan, measured, top, density range): the scene-linear
    positive over the whole scan, each photosite's channel (0 R, 1 G,
    2 B), which photosites the sensor measured at all, the frame's
    brightest measured value, and the frame's density range on green.

    The base's level is read **the same way as after the demosaic** --
    `base_of`, with the roll's vote and its pooled level -- but on the
    green photosites, in the mosaic's own units, because that is the only
    plane the mosaic has a whole grid of. It used to decide for itself,
    taking the lighter of the rebate and the frame on every scan, which is
    the rule from before the roll voted: the two orders then developed the
    same negative against two different bases.

    `pooled` is the roll's level already brought to this scan's exposure.
    It is measured on LibRaw's demosaiced output and used here on the raw
    green plane, so the two have to be the same scale: measured on four
    scans of two cameras, the frame's own floor reads 0.4-1.3 % lower in
    the mosaic than through LibRaw (0.3407 against 0.3419, 0.0696 against
    0.0705), which is the demosaic's own smoothing of a percentile and is
    well inside what the level is known to.

    A photosite standing no more than FLOOR levels above black measured
    nothing: the negative there is denser than the camera can read, and
    the positive is white without a number.
    """
    roll_ratio, trust_rebate, pooled = own_base(item, roll_ratio,
                                                 trust_rebate, pooled)
    import rawpy
    with rawpy.imread(str(path)) as r:
        v = r.raw_image_visible.astype(np.float32)
        black = float(np.mean(r.black_level_per_channel))
        white = float(r.white_level)
        colors = r.raw_colors_visible.copy()
    chan = np.where(colors == 3, 1, colors)
    T = np.clip(v - black, 1, None) / (white - black)
    H, W = T.shape
    if flat:
        # each photosite by its own channel's falloff (`flatfield`); what
        # the sensor measured at all is still read off `v`, below
        from . import flatfield
        flatfield.check(flat, path, W, H)
        flatfield.divide_mosaic(T, chan, flat)
    # the green photosites as a half-size plane, for the base's level
    g = T[1::2, 0::2] if chan[1, 0] == 1 else T[0::2, 1::2]
    scale = W / g.shape[1]
    reb = rebate(np.repeat(g[..., None], 3, -1), scale, item, part)
    fr_g = cut(np.repeat(g[..., None], 3, -1), scale, part, item["angle"])[..., 1]
    top = frame_top(np.repeat(fr_g[..., None], 3, -1))
    reb_base = (np.percentile(reb, 90, axis=0).astype(np.float32)
                if len(reb) > 500 else np.zeros(3, np.float32))
    base = base_of(reb_base, top, np.array(roll_ratio, np.float32),
                   trust_rebate, pooled)
    # The ratio, floored at unexposed film, and raised to 1/gamma: the
    # same arithmetic as `invert` with the log and the antilog cancelled,
    # and the same reason. See `invert`.
    R = np.maximum(base[chan] / T, 1.0)
    P = R ** (1.0 / gamma) - 1.0
    measured = (v - black) > FLOOR
    # the frame's top and range are read inside the tilted frame, clear of
    # its edge: an axis-aligned box the frame's size reaches into the
    # holder's penumbra at the corners, denser than anything in the
    # picture, and put the top 4-6 EV over the frame's on the Nikon roll
    x0, y0, bw, bh = inner_box(*part["centre"], *part["size"], item["angle"])
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x0 + bw), min(H, y0 + bh)
    box = P[y0:y1, x0:x1][measured[y0:y1, x0:x1]]
    top = max(float(box.max()) if box.size else 1.0, 1e-6)
    rg = R[y0:y1, x0:x1][chan[y0:y1, x0:x1] == 1]
    q = np.percentile(rg, (1, 99))
    drange = float(math.log10(max(q[1], 1e-9) / max(q[0], 1e-9)))
    return P, chan, measured, top, drange


def positive_mosaic(path, item, part, roll_ratio, gamma,
                    trust_rebate=True, pooled=None, flat=None):
    """The inversion on the mosaic (`invert_mosaic`), and LibRaw's demosaic
    run on the positive. Returns (frame, density range): the scene-linear
    positive cut out, after passing through the raw's buffer.

    Into that buffer without losing anything the sensor measured: the top
    is the frame's brightest measured photosite, and what is written is
    the **square root** of the positive over that top, on all 16 bits
    (LibRaw takes `user_sat`/`user_black` for the rewritten buffer, and
    the scale checks exact on a ramp to 0.1 %). The root, because a
    positive spans up to 15 EV from a night frame's median to its lamps,
    and 16 linear bits would leave the median on 2 levels; in root space
    the same frame keeps 400 levels at its median. Two rules measured and
    dropped: the brightest 0.1 % just under white (DSC_0305 on 30 levels,
    half of them zero) and a fixed 4 EV of headroom over the median (the
    window of SDIM0844, 9-28 times its median, clipped flat).
    """
    import rawpy
    P, chan, measured, top, drange = invert_mosaic(
        path, item, part, roll_ratio, gamma, trust_rebate, pooled, flat)
    W = P.shape[1]
    unit = 65535.0
    with rawpy.imread(str(path)) as r:
        vis = r.raw_image_visible
        code = np.sqrt(np.clip(P / top, 0, 1)) * unit
        code[~measured] = unit
        vis[...] = np.clip(code, 0, unit).astype(vis.dtype)
        # adjust_maximum_thr=0: LibRaw otherwise takes the *negative's* own
        # brightest value, read when the file was opened, as the white point
        # (a Nikon scan of 13361 on 16383), and clips the positive written
        # over it at 81 % of its top.
        rgb = r.postprocess(half_size=False, output_bps=16, gamma=(1, 1),
                            no_auto_bright=True, use_camera_wb=False,
                            user_wb=[1, 1, 1, 1], output_color=rawpy.ColorSpace.raw,
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                            adjust_maximum_thr=0.0, user_sat=65535, user_black=0)
    rgb = rgb.astype(np.float32) / 65535.0                  # LibRaw: value / user_sat
    fr = cut(rgb, W / rgb.shape[1], part, item["angle"]) ** 2 * top
    return fr, drange


def base_of(reb_base, top, roll_ratio, trust_rebate=True, pooled=None):
    """Not darker than the frame's own lightest pixels; coloured like the
    roll's film. On a roll whose rebate is not film the rebate is not
    consulted at all -- see `roll_statistics`.

    `pooled` is the roll's own level brought to this scan's exposure, and
    it wins where it is available, because it is measured over the whole
    roll rather than off one scan. The frame's lightest pixels still floor
    it: physics beats a pooled average, and a scan lighter than the roll
    says is a scan the roll was wrong about.
    """
    g = max(reb_base[1], top[1]) if trust_rebate else top[1]
    if pooled:
        g = max(float(pooled), top[1])
    return np.array([g * roll_ratio[0], g, g * roll_ratio[2]], np.float32)


def own_base(item, roll_ratio, trust_rebate, pooled):
    """(colour, trust, level) with the scan's own base, when a person picked
    one (`⇧B`, `doc["bases"]`, put on the item by `with_specks` as
    `base_sample`); with a base picked for the whole roll
    (`roll.base_pick`, put on as `base_colour`), its colour and the roll's
    level; the roll's three otherwise.

    **For the roll, only the colour.** The level is a scan's exposure of
    the copy, and one scan's green is every other scan's only when they
    were all copied alike; the roll's pooled level already says that, per
    unit of exposure. The colour is the film's, and the same on every frame.

    **A scan copied under another light needs its own base**: two scans
    of one negative, copied on another day, came into a roll and developed
    colder than the same negative scanned
    with it. Through the one negative, the earlier light passed +0.13 to
    +0.18 EV more red and 0.10 to 0.20 less blue against green -- a light
    of another colour, not another film. The roll divides by the base its
    own scans measured, and that is the wrong light for these two. Divided
    by unexposed film **on the scan itself**, the light's colour cancels:
    the scan is light x film, and the sample is light x base.

    The sample is transmittance per channel off `linear` (demosaiced, flat
    divided, no balance) in the box a person drew. Its colour replaces the
    roll's, and its green replaces the level, as the roll's pooled level
    does (`base_of`: still floored by the frame's own lightest pixels)."""
    smp = item.get("base_sample") if item else None
    if not smp:
        tint = item.get("base_colour") if item else None
        if tint:
            r, g, b = (float(v) for v in tint)
            g = max(g, 1e-6)
            return [r / g, 1.0, b / g], trust_rebate, pooled
        return roll_ratio, trust_rebate, pooled
    r, g, b = (float(v) for v in smp)
    g = max(g, 1e-6)
    return [r / g, 1.0, b / g], True, g


def base_sample(path, box, flat=None):
    """Per-channel median of the scan inside `box` (scan pixels, x0 y0 x1
    y1), off `linear` at half size with the rig's falloff divided out:
    what `own_base` takes for the scan's base."""
    rgb, scale = linear(path, True, None, flat)
    x0, y0, x1, y1 = (int(round(v / scale)) for v in box)
    x0, x1 = sorted((max(0, x0), min(rgb.shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(rgb.shape[0], y1)))
    sub = rgb[y0:y1, x0:x1].reshape(-1, 3)
    if len(sub) < 16:
        raise ValueError("the box is too small to read a base from")
    return [round(float(v), 6) for v in np.median(sub, 0)]


def base_source(reb_base, top, trust_rebate=True, pooled=None):
    """Which reading `base_of` took the level from: `roll` (the pooled
    level), `rebate`, or `frame` (the frame's own lightest pixels, which
    floor the other two)."""
    if pooled:
        return "roll" if float(pooled) >= top[1] else "frame"
    return "rebate" if trust_rebate and reb_base[1] >= top[1] else "frame"


def roll_base_of(per):
    """The roll's base level, from its frames' readings brought to one
    exposure: the median of the top `BASE_ROLL_KEEP` per cent of them.

    Every reading is a lower bound -- see `BASE_ROLL_KEEP`, which carries
    the measurement against the one blank scan in the corpus -- so the
    base is at the top of them, and the median of the top quarter is the
    top without being the maximum.
    """
    per = np.asarray(per, np.float64)
    per = per[np.isfinite(per) & (per > 0)]
    if not len(per):
        return None
    keep = per[per >= np.percentile(per, BASE_ROLL_KEEP)]
    if len(keep) < min(BASE_ROLL_MIN, len(per)):
        keep = per
    return float(np.median(keep))


def base_choice(stats, taste, one_roll=True):
    """(trust the rebate, the roll's level for this scan's exposure or None)
    -- the two arguments `base_of` takes, from the roll's measurements and
    the roll's `base` setting.

    One place, because three callers need the same answer and one of them
    getting it wrong is how `order` = mosaic came to develop against a
    different base from `order` = after.

    **`auto` is the roll's base.** Unexposed film is one thing and a roll
    is one piece of it, so the number that describes it is the roll's;
    each frame's own deepest shadows are a floor on that number, not a
    second opinion about it, and they are as far below it as that frame's
    scene is dark. The one case where the roll cannot answer is a folder
    that is not one roll -- `roll.one_roll` off, the start screen's
    question -- where the scans have no common film and each frame is on
    its own.
    """
    want = (taste or {}).get("base", "auto")
    if want == "frame":
        # **Each scan's own base: its own unexposed film**, where the scan
        # has some, floored by the frame's lightest pixels as always
        # (`base_of` takes the larger). It was the frame's lightest pixels
        # and nothing else, the rebate not consulted -- and those are a
        # bound, not a reading: on one frame (a
        # hazy backlit beach with no black in it) they read 0.186 where
        # the film round the frame on that same scan read 0.615 and the
        # roll 0.640, so the base sat 1.8 stops low, the shadows went
        # under it and the frame came out green. Unticking *the base is the same
        # across the roll* reads as "each frame its own base" -- which is
        # the film beside it, not its shadows.
        return True, None
    trust = stats.get("trust_rebate", True)
    if want == "roll" or one_roll:
        return trust, stats.get("base_roll_unit") or stats.get("base_unit")
    return trust, None


# What the roll measured, kept in the document under `roll.measured`. Not a
# setting -- nobody chose any of it, and measuring again overwrites it. It is
# in the document because it costs one pass over the roll, and because
# everything downstream of it has to agree with it: the read line's base
# label, `v`, `d` -- which renders the develop itself -- and the develop.
#
# All of it, not the base alone. The develop needs the camera's matrix and the
# roll's balance as well as the base, so keeping two of the three meant the one
# view whose whole claim is *this is what comes out* could not be built without
# measuring the roll again to draw one frame. With all of it kept, `⌘R`
# develops from the same numbers the window drew with, rather than measuring
# the roll a second time and hoping the two agree.
#
MEASURED = ("base_colour", "trust_rebate", "base_unit", "base_roll_unit",
            "matrix", "balance_log2", "base_method",
            "base_from", "base_rebate_won", "base_frames", "base_pooled",
            "frames_balanced", "under")


def measured_of(stats, taste=None):
    """`roll.measured` from a `roll_statistics` result: JSON, rounded, and in
    one function so the places that write it cannot drift apart."""
    if not stats or not stats.get("base_colour") or not stats.get("matrix"):
        return None
    return {
        # **Under which settings.** The base and the balance are measured
        # at a gamma, so a roll measured at one does not describe a roll
        # developed at another. `measured_ready` compares these and calls a
        # mismatch *not measured*, which is what it is.
        "under": {"gamma": (taste or {}).get("gamma", TASTE["gamma"]),
                  # by name: the calibration is the rig's, kept in the
                  # library, and the roll only says which one
                  "flat": (taste or {}).get("flat")},
        "base_colour": [round(float(v), 4) for v in stats["base_colour"]],
        "trust_rebate": bool(stats.get("trust_rebate", True)),
        "base_unit": stats.get("base_unit"),
        "base_roll_unit": stats.get("base_roll_unit"),
        "base_pooled": bool(stats.get("base_pooled")),
        "base_from": stats.get("base_from"),
        "base_rebate_won": stats.get("base_rebate_won"),
        "base_frames": stats.get("base_frames"),
        "matrix": [[round(float(v), 6) for v in row] for row in stats["matrix"]],
        "balance_log2": [round(float(v), 4) for v in stats["balance_log2"]],
        "frames_balanced": stats.get("frames_balanced"),
        "base_method": stats.get("base_method") or base_method_of(stats),
    }


def base_method_of(stats, taste=None):
    """Where the base came from, in words, for the document and the window."""
    want = (taste or {}).get("base", "auto")
    if want == "frame":
        return "each frame's own deepest shadows, forced"
    if want == "roll":
        return "the roll's own level, forced, brought to each scan's exposure"
    if stats.get("base_pooled"):
        return "the roll's rebate, brought to each scan's own exposure"
    return "the rebate" if stats.get("trust_rebate") else \
           "the frames' own deepest shadows"


def measured_ready(measured, taste=None):
    """Whether `roll.measured` holds enough to develop a frame from it, and
    was measured under the settings now in force.

    A document written before the matrix and the balance were kept holds the
    base and nothing else, which is enough for the read line and not enough
    for `d`. Saying so reads as a measurement that is missing, which is
    what it is, rather than as a document that is old.

    With `taste`, the gamma it was measured under has to be the one now
    set: the base and the balance are read through it, so a measurement
    made at another gamma describes another roll.
    """
    m = measured or {}
    if not (m.get("base_colour") and m.get("matrix")
            and m.get("balance_log2") is not None):
        return False
    # The balance is read through the matrix, so one measured through a
    # camera's (before `MATRIX` was the identity) is another roll's.
    if not np.allclose(np.asarray(m["matrix"], np.float64), MATRIX, atol=1e-6):
        return False
    if taste is None:
        return True
    under = m.get("under") or {}
    return (under.get("gamma", object()) == taste.get("gamma", TASTE["gamma"])
            and (under.get("flat") or None) == (taste.get("flat") or None))


def stats_of(measured, taste, roll_dir=None):
    """`roll.measured` back as the statistics the develop takes.

    One function, so the window and `⌘R` cannot each assemble the roll's
    numbers their own way and then disagree about what was developed.
    """
    if not measured_ready(measured, taste):
        return None
    from . import flatfield
    stats = {k: measured.get(k) for k in MEASURED}
    # the calibration itself, from the library: the document keeps its name
    try:
        stats["flat"] = flatfield.of(taste)
    except FileNotFoundError:
        # measured under a calibration since removed from the library: what
        # it measured cannot be developed as it was, so it is not measured
        return None
    return stats

def invert(fr, base, gamma):
    """(scene-linear positive, density range). Unexposed film is scene
    black; the range says whether there is a picture at all.

    **The density is not computed.** The film's response is defined as
    density against log exposure with slope `gamma`, and this was written
    as that definition transcribed -- `d = log10(base / T)` floored at
    zero, then `10 ** (d / gamma) - 1` -- from the day it was first
    written, when what had to be fixed about the previous program was that
    it inverted linearly (`max - pixel`, which removes neither the mask
    nor the curve). But the `10 **` undoes the `log10` exactly, so the
    pair cancels and what is left is a power of the ratio:

        P = (base / T) ** (1 / gamma) - 1,  the ratio floored at 1

    Measured on an array the size of a frame (3100x4700x3): 2.83 s against
    3.18 s, and the positive comes out with a median difference of exactly
    zero (worst 1.4e-04, which is float32 rounding on values in the tens).

    The **range** survives the same way, and that is the part that is not
    obvious: `log10` is monotonic, so percentiles commute with it, and the
    two percentiles can be taken on the ratio and logged as two scalars
    instead of logging every pixel of the frame. The floor commutes too --
    `max(x, 1)` then `log10` is `max(log10 x, 0)` -- so a frame with more
    than 1 % of itself against the floor still reads a 1st percentile of
    exactly 0 D. Checked on a frame-sized array: 1.142541 D both ways.
    """
    r = np.maximum(base[None, None, :] / np.clip(fr, 1e-5, None), 1.0)
    q = np.percentile(inside(r)[..., 1], (1, 99))
    drange = float(math.log10(max(q[1], 1e-9) / max(q[0], 1e-9)))
    return r ** (1.0 / gamma) - 1.0, drange


def wanted_balance(pos, keep=None):
    """log2 of the per-channel gain that would make the mid-tones neutral,
    or NaN where there are no mid-tones to speak of. Read clear of the
    frame's edge, like every other statistic (STAT_INSET).

    **Of the least coloured `WB_KEEP` per cent of them**, not of all. Grey
    world assumes a scene averages grey, and the mid-tone band of a
    photograph often is not a scene average at all: on a beach frame the
    30th to 85th percentile of luminance is 55 % of the picture and almost
    all of it is sky and sea. Asking that to be neutral asks for the blue
    of the sky to be taken out, and it comes back as orange sand.

    Trimming the coloured half leaves the pixels that could be grey
    driving the estimate and lets the sky sit where it is. What it does to
    the four rolls' own balance, which is the median of this over every
    frame:

        roll                    all         least coloured half
        roma-half           R +0.44 B +0.19   R +0.34 B +0.17
        cinestill-full      R +0.21 B +0.31   R +0.21 B +0.23
        portra400-full      R +0.30 B -0.46   R +0.10 B -0.26
        Negativos           R +0.09 B -0.22   R +0.12 B -0.07
        negativos-portra400 R +0.29 B -0.49   R +0.09 B -0.33

    Every roll moves toward neutral, and the two it moves most are the
    two Portra rolls, which are the outdoor ones with sky in most frames:
    it halves a correction that was taking the blue out of the sea.
    """
    pos = inside(pos)
    y = pos @ LUMA
    lo, hi = np.percentile(y, (30, 85))
    sel = (y >= lo) & (y <= hi)
    if sel.sum() < 100:
        return np.full(3, np.nan)
    px = pos[sel]
    g = np.maximum(px[:, 1], 1e-6)
    # how far from neutral each pixel is, in stops, on both ratios at once
    away = (np.abs(np.log2(np.maximum(px[:, 0], 1e-6) / g))
            + np.abs(np.log2(np.maximum(px[:, 2], 1e-6) / g)))
    px = px[away <= np.percentile(away, WB_KEEP if keep is None else keep)]
    if len(px) < 100:
        return np.full(3, np.nan)
    means = px.mean(0)
    if np.any(means <= 1e-6):
        return np.full(3, np.nan)
    return np.log2(means[1] / means)


NOISE_LUMA_PX = 1.5      # scan px: the grain the luminance shrinkage works at
NOISE_CHROMA_PX = 6.0    # scan px: how far the colour noise is smoothed
NOISE_K = 2.5            # the luminance threshold at full strength, in noise sigmas
NOISE_EPS = 1e-3         # linear: what keeps the logarithms finite in the shadows
SHARPEN_PX = 1.2         # scan px: the unsharp mask's radius
SHARPEN_THRESHOLD = 0.004  # display-referred luminance under which detail is grain, not edge


def denoise(pos, strength, px, look=None):
    """`pos` (scene-linear RGB) with its grain and colour noise reduced,
    `strength` 0..1, at `px` scan pixels per pixel.

    **In logarithms, luminance and colour apart**, because the noise of a
    scanned negative is multiplicative -- grain is a density -- and
    because the eye forgives grain in the luminance and not blotches in
    the colour. The colour (each channel's log minus their mean) is
    blurred over `NOISE_CHROMA_PX` and blended in by `strength`. The
    luminance is split at `NOISE_LUMA_PX` into a base and a detail; the
    detail is shrunk where it is the size of the noise -- `d^2 / (d^2 +
    (k s)^2)`, `s` the detail's robust spread inside the frame, `k`
    `NOISE_K` times the strength -- and kept where it is larger, which is
    an edge. **The sizes are scan pixels**: on the window's picture, four
    scan pixels to one, the grain is already averaged away and this does
    little, which is what that picture shows; on `⌘R`'s full size it is
    the grain itself."""
    if strength <= 0:
        return pos
    L = np.log(np.clip(pos, 0, None) + NOISE_EPS)
    lum = L.mean(axis=2)
    chroma = L - lum[..., None]
    sc = max(NOISE_CHROMA_PX / px, 0.5)
    blurred = ndimage.gaussian_filter(chroma, (sc, sc, 0))
    chroma = chroma + (blurred - chroma) * min(1.0, strength)
    sl = NOISE_LUMA_PX / px
    if sl >= 0.3:
        base = ndimage.gaussian_filter(lum, sl)
        d = lum - base
        probe = d if look is None else look(d[..., None])[..., 0]
        s = 1.4826 * float(np.median(np.abs(probe))) + 1e-9
        t = (NOISE_K * strength * s) ** 2
        lum = base + d * (d * d / (d * d + t))
    return np.clip(np.exp(lum[..., None] + chroma) - NOISE_EPS, 0, None).astype(pos.dtype)


def sharpen(out, amount, px):
    """`out` (display-referred linear sRGB) with its luminance sharpened by
    an unsharp mask of radius `SHARPEN_PX` scan pixels, `amount` the
    strength. On the encoded luminance (gamma 2.2), where an edge's
    overshoot is even on both sides, and through a ratio, so the colour of
    an edge is not touched -- no coloured halos. Detail under
    `SHARPEN_THRESHOLD` is left as it is: it is grain, and sharpening it is
    what makes a sharpened scan look noisy. Under a third of a pixel of
    radius (the window's picture, four scan pixels to one) there is
    nothing at that size left to sharpen, and nothing is done."""
    sr = SHARPEN_PX / px
    if amount <= 0 or sr < 0.33:
        return out
    y = np.clip(out @ LUMA, 1e-6, None)
    e = y ** (1 / 2.2)
    d = e - ndimage.gaussian_filter(e, sr)
    d = np.sign(d) * np.maximum(np.abs(d) - SHARPEN_THRESHOLD, 0.0)
    e2 = np.clip(e + amount * d, 0, None)
    return out * ((e2 ** 2.2) / y)[..., None]


def _render(pos, scale, taste, look, quick=False, applied=None, px=1.0):
    """`finish` from the exposure on: the curve, the black point, the
    saturation, the sharpening and the shoulder, at `scale`. `quick` is
    the shadow rule's search, on every fourth pixel and unsharpened.
    `applied` takes the slopes and the black from another call instead
    of measuring them."""
    out = (pos[::4, ::4] if quick else pos) * scale
    yin = look(out) @ LUMA
    p5 = max(float(np.percentile(yin, 5)), 1e-6)
    # **The curve compresses in proportion to what there is to
    # compress.** `shadows` and `highlights` are slopes under one:
    # they always squeeze, which is right for a negative holding more
    # range than a screen and wrong for one holding less. 022 of
    # negativos-portra400 arrives with 1.4 stops between its 5th and
    # 95th percentiles -- a bright hazy beach, where sand, sea and sky
    # sit within a couple of stops of each other -- and the curve took
    # it to 1.23, a picture with nowhere to go. Across that roll the
    # frames arrive holding anything from 1.5 to 6.8 stops and the
    # curve multiplies every one of them by 0.88.
    #
    # So the slopes are scaled to aim the output at `SPAN_FLOOR`. They
    # may only open, never close further, and never past `SLOPE_MAX`.
    # The shadow floor below still has the last word on the shadow
    # slope: opening the curve must not put the darkest twentieth of
    # the picture on the floor.
    q = np.percentile(yin, (5, 95))
    span = math.log2(max(q[1], 1e-6) / max(q[0], 1e-6))
    k = min(max(SPAN_FLOOR / max(span, 1e-3), 1.0),
            SLOPE_MAX / max(taste["shadows"], taste["highlights"], 1e-3))
    # **Contrast after the opening, pivot on the normalised picture**
    # (`⌘L`). Contrast multiplies both slopes once each
    # frame has opened its own: before it, the opening would undo a
    # lower contrast on every flat frame, which are the ones that
    # ask for it. The pivot is where the curve bends, in EV over
    # middle grey -- where the automatic exposure has just put every
    # frame's median, so one pivot means the same tone on a bright
    # frame and a dark one. With both slopes equal it is exactly an
    # exposure (the curve is p (y/p)^s, and p only scales it); with
    # them unequal it moves the knee between them.
    c = float(taste.get("contrast", 1.0))
    pv = 0.18 * 2.0 ** float(taste.get("pivot", 0.0))
    slope = taste["shadows"] * k * c
    high = taste["highlights"] * k * c
    if applied is not None:
        slope, high = applied["slope"], applied["high"]
    elif p5 < SHADOW_FLOOR and p5 < pv:
        need = math.log2(SHADOW_FLOOR / pv) / math.log2(p5 / pv)
        slope = max(SHADOW_AUTO_MIN, min(slope, need))
    y = out @ LUMA                   # the curve is applied to the whole frame
    e = np.log2(np.clip(y, 1e-6, None) / pv)
    e2 = np.where(e < 0, e * slope, e * high)
    out = out * (pv * np.exp2(e2) / np.clip(y, 1e-6, None))[..., None]

    sub = (applied["sub"] if applied is not None else
           min(max(float(np.percentile(look(out) @ LUMA, BLACK_PCT))
                   - BLACK_TARGET, 0.0), BLACK_MAX))
    out = np.clip(out - sub, 0, None) / max(1e-6, 1 - sub)
    # Saturation **before** the shoulder, so the shoulder catches it.
    # It used to be last, and it undid the one thing the shoulder is
    # for: it pushes a channel away from the luminance, which on a
    # saturated highlight takes it past white, and the final clip then
    # cut it flat. Measured over 29 frames of the four rolls, the share
    # of channel values landing on exactly 1.0 went from 0.5443 % at
    # worst (DSC_0358) to 0.0068 %, and 10 frames to 1, while the
    # picture itself moved 0.002 levels of 255 at the median.
    if taste["saturation"] != 1.0:
        yy = (out @ LUMA)[..., None]
        out = np.clip(yy + (out - yy) * taste["saturation"], 0, None)
    if not quick:
        out = sharpen(out, float(taste.get("sharpen", 0.0)), px)
    a = SHOULDER_FROM
    out = np.where(out > a, a + (1 - a) * (1 - np.exp(-(out - a) / (1 - a))),
                   out)
    return np.clip(out, 0, 1), slope, high, sub


def finish(pos, roll_wb, blank, taste, look_box=None, px=1.0, applied=None):
    """Balance, exposure, the display curve, black and shoulder: the
    per-photo automatics within their thresholds, then the taste.

    **`look_box` is where the frame is**, as `(y0, y1, x0, x1)` into `pos`,
    and it is the only place the automatics are allowed to read. Everything
    else in `pos` may be rebate, sprocket holes or the holder -- none of it
    a photograph, and all of it ruinous to read: the balance takes the
    orange mask for a mid-tone and the exposure takes unexposed film for a
    highlight. So **what comes out and what is measured are two different
    rectangles**: the crop decides the first and the frame as found, inset
    by `STAT_INSET`, decides every number. Without it a crop of -10 % moved
    the curve from 0.80/0.85 to 0.95/1.01 on a scan of roma-half.

    `px` is how many scan pixels one pixel of `pos` spans, for the sizes of
    the noise reduction and the sharpening. `applied` is another call's
    `ev["applied"]`: its decisions -- balance, exposure, slopes, black --
    are taken as they are instead of being measured here, so a crop of a
    frame at full size develops as the whole frame did (`⌘L`'s 100 %).

    **The order**, and why: the balance; the **noise reduction**, in
    scene-linear light, after the balance so the colour is smoothed about
    the right neutral and before the exposure and the curve so neither
    amplifies the grain and the automatics read clean numbers; the
    exposure, the curve, the black point, the saturation; the
    **sharpening**, on the finished luminance so it sharpens what is seen
    and the curve cannot raise its halo again, and before the shoulder so
    the shoulder rolls off what it pushes past white; the shoulder.
    """
    # **Kept as fractions, not pixels.** The shadow rule's search runs on
    # every fourth pixel -- sixteen times less of everything for the same
    # percentiles -- so `look` is handed arrays of two different sizes, and
    # a box in pixels of the big one selects nothing at all in the small
    # one. It did: an empty slice, and `np.percentile` on it.
    frac = None
    if look_box is not None:
        h0, w0 = pos.shape[:2]
        y0, y1, x0, x1 = look_box
        frac = (y0 / h0, y1 / h0, x0 / w0, x1 / w0)

    def look(x):
        """The part of `x` the automatics are allowed to read."""
        if frac is None:
            return inside(x)
        h, w = x.shape[:2]
        y0, y1 = int(frac[0] * h), max(int(frac[1] * h), int(frac[0] * h) + 1)
        x0, x1 = int(frac[2] * w), max(int(frac[3] * w), int(frac[2] * w) + 1)
        return inside(x[max(0, y0):y1, max(0, x0):x1])

    bias = np.array([taste["warmth"] - taste["tint"], taste["tint"],
                     -taste["warmth"] - taste["tint"]])
    if applied is not None:
        corr = np.array(applied["corr"], np.float32)
        residual = np.log2(corr) - roll_wb - bias
    else:
        want = wanted_balance(look(pos))
        if np.any(np.isnan(want)):
            want = roll_wb
        residual = np.clip((want - roll_wb) * WB_STRENGTH, -WB_LIMIT, WB_LIMIT)
        corr = np.exp2(roll_wb + residual + bias)
    pos = pos * corr[None, None, :]
    pos = denoise(pos, float(taste.get("noise", 0.0)), px,
                  None if applied is not None else look)
    if applied is not None:
        out, slope, high, sub = _render(pos, applied["scale"], taste, look,
                                        applied=applied, px=px)
        return out, {"applied": applied}

    # read clear of the frame's edge, where the holder can reach in
    y = look(pos) @ LUMA
    hp = max(float(np.percentile(y, HIGHLIGHT_PCT)), 1e-6)
    md = max(float(np.median(y)), 1e-6)
    scale_h = HIGHLIGHT_TARGET / hp
    scale = MEDIAN_TARGET / md
    scale = min(max(scale, scale_h * 2 ** -DARK_MAX), scale_h * 2 ** CLIP_MAX)
    # But the highlights never take the picture down more than MEDIAN_DROP
    # under its own rule: a night kitchen with lamps (DSC_0306, Nikon) has
    # 2 % of its frame beyond what the camera measured through the dense
    # negative, and protecting that took the picture 10 EV down.
    scale = max(scale, MEDIAN_TARGET / md * 2 ** -MEDIAN_DROP)
    if blank:
        scale = min(scale, 0.25 / hp)
    auto_ev = math.log2(max(scale, 1e-12))
    scale *= 2 ** taste["exposure"]
    base_scale = scale

    # The scale search below re-renders, and a full-size frame is 300 MB a
    # copy: doing it three times over kept every intermediate alive at
    # once and the develop stopped being able to finish a roll. The search
    # runs on every fourth pixel, which is 16 times less of everything and
    # the same percentiles to three decimals, and only the scale it
    # settles on is rendered whole.
    def render(scale, quick=False):
        return _render(pos, scale, taste, look, quick=quick, px=px)

    # **The shadow rule**, which is the highlight rule's other half. The
    # median can sit on its target while the picture is piled against
    # black: on 002 of negativos-portra400 the median is 0.182 and 44 % of
    # the frame is under 0.10. Labelled by eye, that frame and three others are
    # underexposed and six others right, and of six statistics tried
    # against those labels this is the only one that separates them --
    # the four sit at 36-44 % under 0.10, the six at 3.6-28 %.
    #
    # So: no more than SHADOW_SHARE of a frame may sit under SHADOW_LEVEL,
    # bought with at most SHADOW_LIFT of exposure and never past what the
    # highlights allow. Measured on the finished picture and not before
    # it, because the curve and the black point are what put it there, so
    # it takes a couple of passes to land.
    lift, ceiling = 1.0, max(1.0, (scale_h * 2 ** CLIP_MAX) / max(base_scale, 1e-9))
    if not blank:
        for _ in range(3):
            small, _s, _h, _b = render(base_scale * lift, quick=True)
            y = look(small) @ LUMA
            del small
            if float((y < SHADOW_LEVEL).mean()) <= SHADOW_SHARE:
                break
            q = max(float(np.percentile(y, 100 * SHADOW_SHARE)), 1e-5)
            grow = min(lift * (SHADOW_LEVEL / q), 2 ** SHADOW_LIFT, ceiling)
            if grow <= lift * 1.002:
                break
            lift = grow
    out, slope, high, sub = render(base_scale * lift)
    return out, {
        "balance": [round(float(v), 3) for v in corr],
        "exposure": round(math.log2(base_scale * lift), 2),
        "shadow_lift": round(math.log2(lift), 2),
        "shadow_slope": round(slope, 3),
        "highlight_slope": round(high, 3),
        "black": round(float(sub), 4),
        # The parts, for the window's "Development details": the balance
        # is the roll's, this frame's own pull off it, and the taste; the
        # exposure is the automatic one before the taste's and the shadow
        # rule's. `exposure` above is all three.
        "balance_ev": {"roll": [round(float(v), 3) for v in roll_wb],
                       "frame": [round(float(v), 3) for v in residual],
                       "taste": [round(float(v), 3) for v in bias]},
        "exposure_auto": round(auto_ev, 2),
        # unrounded, to develop a crop of this frame the same way
        "applied": {"corr": [float(v) for v in corr],
                    "scale": float(base_scale * lift), "slope": float(slope),
                    "high": float(high), "sub": float(sub)}}


def encode(lin, bits):
    x = np.clip(lin, 0, 1)
    enc = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    top = 65535 if bits == 16 else 255
    return (enc * top + 0.5).astype(np.uint16 if bits == 16 else np.uint8)


# ---------------------------------------------------------------- jobs

def _first(job):
    path, item, part, flat = job
    fr, reb_base, top = measure(path, item, part, True, None, flat=flat)
    base = np.maximum(reb_base, top)
    # Two percentiles of a density, without computing the density: the log
    # is monotonic, so it can be taken of the two numbers instead of the
    # frame. See `invert`.
    r = np.maximum(base[1], 1e-6) / np.clip(fr[..., 1], 1e-5, None)
    q = np.percentile(r, (1, 99))
    drange = float(math.log10(max(q[1], 1e-9) / max(q[0], 1e-9)))
    return reb_base.tolist(), top.tolist(), drange


GREY = [1.0, 1.0, 1.0]


def _second(job):
    path, item, part, matrix, roll_ratio, gamma, flat = job
    fr, reb_base, top = measure(path, item, part, True, roll_ratio, flat=flat)
    base = base_of(reb_base, np.array(top), np.array(GREY))
    pos, drange = invert(fr, base, gamma)
    pos = np.clip(pos @ np.array(matrix, np.float32), 0, None)
    return wanted_balance(pos).tolist(), drange


def _full_positive(path, item, part, matrix, roll_ratio, taste, trust, pooled,
                   flat):
    """`_full` up to the develop: (positive, density range, base, rebate
    reading, frame top), at full size, cut and straightened, dust out."""
    roll_ratio, trust, pooled = own_base(item, roll_ratio, trust, pooled)
    if taste.get("order") == "mosaic":
        pos, drange = positive_mosaic(path, item, part, roll_ratio,
                                      taste["gamma"], trust, pooled, flat)
        base = reb_base = top = np.array(GREY, np.float32)  # not read this way
    else:
        fr, reb_base, top = measure(path, item, part, False, roll_ratio,
                                    crop_inset(taste), flat)
        base = base_of(reb_base, top, np.array(GREY), trust, pooled)
        pos, drange = invert(fr, base, taste["gamma"])
        del fr
    pos = np.clip(pos @ np.array(matrix, np.float32), 0, None)
    pos = _cleaned(pos, item, part, taste)
    return pos, drange, base, reb_base, top


def full_positive(path, item, part, stats, taste, one_roll=True):
    """A frame's positive at full size, with the base chosen as
    `develop_frame` chooses it: what `⌘L`'s 100 % view crops."""
    trust, unit = base_choice(stats, taste, one_roll)
    pooled = (unit * (scan_exposure(path) or 0)) if unit else None
    pos, _d, _b, _r, _t = _full_positive(path, item, part, stats["matrix"],
                                         stats["base_colour"], taste, trust,
                                         pooled, stats.get("flat"))
    return pos


def _full(path, item, part, matrix, roll_ratio, roll_wb, taste, trust, pooled,
          flat):
    """One frame developed at full size: linear sRGB, cut and straightened
    but **not** mirrored or turned, and its evidence. What `⌘R` writes and
    what the window's dust marking looks at, so the two are one picture."""
    pos, drange, base, reb_base, top = _full_positive(
        path, item, part, matrix, roll_ratio, taste, trust, pooled, flat)
    blank = drange < BLANK_RANGE
    out, ev = finish(pos, np.array(roll_wb, np.float32), blank, taste,
                     look_box=frame_box(pos.shape, taste))
    ev.update(blank=bool(blank), density_range=round(drange, 3),
              order=taste.get("order", "after"),
              base=[round(float(v), 4) for v in base],
              base_source=("picked" if item.get("base_sample") else
                           base_source(reb_base, top, trust, pooled)
                           if taste.get("order") != "mosaic" else None),
              base_rebate=[round(float(v), 4) for v in reb_base],
              base_frame=[round(float(v), 4) for v in top])
    return out, ev


def _cleaned(pos, item, part, taste):
    """The frame with the dust marked on its scan taken out, before the
    automatics read it. `item["specks"]` is put there by whoever builds the
    job (`frames_of`, the window); the document keeps the marks beside the
    items, not in them. Before `finish`, so a speck whiter than the scene
    is not what the highlights are protected for."""
    marks = item.get("specks") or []
    if not marks:
        return pos
    from . import dust
    a = np.array(frame_to_scan(item, part, taste))
    inv = np.linalg.inv(a[:, :2])
    return dust.clean(pos, dust.points(
        marks, lambda x, y: tuple(inv @ (np.array([x, y]) - a[:, 2]))), 1.0)


def with_specks(item, doc, name):
    """`item` with what a person said about the scan on it, for a develop
    job: the dust -- the marks made by hand and the found ones a person
    accepted (`dust.used`) -- and the scan's own base, if one was picked
    (`own_base`), or else the colour of the one picked for the whole roll,
    under the light-falloff calibration now in force; one picked under
    another was read through other numbers and is left out until it is
    picked again."""
    from . import dust
    marks = [m for m in (doc.get("specks") or {}).get(name, {}).get("marks") or []
             if dust.used(m)]
    if marks:
        item = dict(item, specks=marks)
    own = (doc.get("bases") or {}).get(name)
    flat = (doc["roll"].get("develop") or {}).get("flat")
    if own and own.get("value") and own.get("flat") == flat:
        return dict(item, base_sample=own["value"])
    whole = doc["roll"].get("base_pick")
    if whole and whole.get("value") and whole.get("flat") == flat:
        item = dict(item, base_colour=whole["value"])
    return item


def old_marks(doc):
    """The scans whose dust has marks from before the brush: a hand mark
    that is a point and not a path (`dust.samples`)."""
    return [name for name, rec in (doc.get("specks") or {}).items()
            if name in doc.get("items", {})
            and any(not m.get("path") and m.get("by", "hand") == "hand"
                    for m in rec.get("marks") or [])]


def migrate_marks(roll, doc, name, stats, taste, one_roll=True):
    """The scan's marks from before the brush, made brush marks, in the
    document. How many were.

    Each is measured on its frame's positive at full size, before any
    dust is out -- the picture `clean` grew it on -- by `dust.migrate`,
    and written back as a dab, a stroke along a hair, or a soft dab. Its
    number stays. One outside every frame was taking nothing out and is
    left as it is."""
    import pathlib
    from . import dust
    item = doc["items"][name]
    marks = doc["specks"][name]["marks"]
    old = [i for i, m in enumerate(marks)
           if not m.get("path") and m.get("by", "hand") == "hand"]
    done, was_sized = 0, {}
    for part in item.get("parts") or []:
        if not old:
            break
        pos = full_positive(pathlib.Path(roll) / name, item, part, stats,
                            taste, one_roll)
        H, W = pos.shape[:2]
        a = np.array(frame_to_scan(item, part, taste))
        inv = np.linalg.inv(a[:, :2])
        to_scan = lambda x, y: [round(float(v), 1) for v in a[:, :2] @ (x, y) + a[:, 2]]
        for i in list(old):
            m = marks[i]
            x, y = inv @ (np.array([m["x"], m["y"]]) - a[:, 2])
            if not (0 <= x < W and 0 <= y < H):
                continue
            if m.get("r"):
                was_sized[i] = (m["x"], m["y"], m["r"])
            got = dust.migrate(pos, x, y, m.get("r"), 1.0)
            new = {k: m[k] for k in ("id",) if k in m}
            if got[0] == "stroke":
                path = [to_scan(*p) for p in got[1]]
                new.update(path=path, r=round(got[2], 1))
            else:
                path = [to_scan(got[1], got[2])]
                new.update(path=path, r=round(got[3], 1))
                if got[0] == "soft":
                    new["soft"] = True
            mid = path[len(path) // 2]
            marks[i] = {"x": mid[0], "y": mid[1], **new, "by": "hand"}
            old.remove(i)
            done += 1
    # a chain of sized discs along one hair is one stroke through their
    # middles, as wide as its widest disc; it keeps the first one's number
    sized = [i for i, m in enumerate(marks) if m.get("by", "hand") == "hand"
             and not m.get("soft") and was_sized.get(i)]
    gone = set()
    for chain in dust.chains([(was_sized[i][0], was_sized[i][1], was_sized[i][2])
                              for i in sized]):
        idx = [sized[c] for c in chain]
        path = [[was_sized[i][0], was_sized[i][1]] for i in idx]
        first = min(idx, key=lambda i: marks[i].get("id", 0))
        mid = path[len(path) // 2]
        marks[first] = {"x": mid[0], "y": mid[1],
                        **({"id": marks[first]["id"]} if "id" in marks[first] else {}),
                        "path": path, "r": max(was_sized[i][2] for i in idx),
                        "by": "hand"}
        gone |= set(idx) - {first}
    marks[:] = [m for i, m in enumerate(marks) if i not in gone]
    return done


def migrate_roll(roll, doc, progress=None):
    """`migrate_marks` over every scan that needs it, with the roll's
    measurement; how many marks were made brush marks. Opening the roll
    does it (`progress.migrate_dust`) and so does `process`.

    **Once, and automatically**, because a point grown into the speck under
    it meant *whatever is here* and only the picture says how big that
    was, so this is the one moment the old meaning can be written down.
    The document as it was is kept beside it the first time,
    `frames.before-brush.json`."""
    import pathlib
    import shutil
    names = old_marks(doc)
    if not names or not doc["roll"].get("measured"):
        return 0
    taste = dict(TASTE) | dict(doc["roll"].get("develop") or {})
    stats = stats_of(doc["roll"]["measured"], taste, roll)
    one = doc["roll"].get("one_roll", True)
    here = pathlib.Path(roll) / "frames.json"
    keep = pathlib.Path(roll) / "frames.before-brush.json"
    if here.exists() and not keep.exists():
        shutil.copy2(here, keep)
    n = 0
    for i, name in enumerate(names):
        if progress:
            progress(i, len(names))
        n += migrate_marks(roll, doc, name, stats, taste, one)
    return n


def develop_frame(path, item, part, stats, taste, one_roll=True):
    """`_full` for one frame, from the roll's statistics: the window's way
    in, with the base chosen exactly as `develop_roll` chooses it."""
    trust, unit = base_choice(stats, taste, one_roll)
    pooled = (unit * (scan_exposure(path) or 0)) if unit else None
    return _full(path, item, part, stats["matrix"], stats["base_colour"],
                 stats["balance_log2"], taste, trust, pooled, stats.get("flat"))


def frame_to_scan(item, part, taste, scale=8):
    """The affine map from a pixel of `_full`'s picture (before mirroring
    and turning) to scan pixels, as [[ax, bx, cx], [ay, by, cy]].

    Read off `cut` itself rather than worked out beside it: the scan's own
    coordinates, at 1/`scale`, are cut like a picture and a plane is
    fitted to what comes out. A second derivation of the cut's geometry
    could disagree with the first by the half pixel nobody would think to
    check, and a dust mark is a few pixels across. Against the full-size
    cut of the scan's own coordinates it is exact to 0.0003 px, on
    SDIM0911 of roma-half (1.74°, both halves) and on DSC_0239, whose
    negative crop reaches past the top of the scan.
    """
    w, h = item["scan"]
    ys, xs = np.mgrid[:h // scale, :w // scale].astype(np.float32) * scale
    # The third channel says "on the scan": a negative `crop` asks for more
    # than the scan holds, `cut` pads that with zeros, and zeros fitted as
    # coordinates bent the plane by 170 px.
    field = np.stack([xs, ys, np.ones_like(xs)], -1)
    out = cut(field, scale, part, item["angle"], crop_inset(taste))
    H, W = out.shape[:2]
    gy, gx = np.mgrid[2:H - 2:2, 2:W - 2:2]
    on = out[gy, gx, 2] > 0.999
    gy, gx = gy[on], gx[on]
    A = np.stack([gx * scale, gy * scale, np.ones(gx.size)], 1)
    B = np.stack([out[gy, gx, 0], out[gy, gx, 1]], 1)
    coef, *_ = np.linalg.lstsq(A, B, rcond=None)
    L = coef[:2].T
    # **The turn from the fit, the shift from `cut`'s own integers.** At
    # 1/8 the `int()`s in `cut` round to 8 scan pixels and the fitted shift
    # came out 3 px off the one at full size; the rotation does not round.
    # So the shift is `cut`'s arithmetic at scale 1: the padded square's
    # corner, the rotation about its centre, and the window into it.
    inset = crop_inset(taste)
    cx, cy = part["centre"]
    fw, fh = part["size"]
    pad = int(max(fw, fh) * max(1.0, 1 - 2 * inset) / 2 + 8)
    x0, y0 = int(cx) - pad, int(cy) - pad
    w2, h2 = fw * (1 - 2 * inset) / 2, fh * (1 - 2 * inset) / 2
    y0i, x0i = max(0, int(cy - y0 - h2)), max(0, int(cx - x0 - w2))
    c = (2 * pad - 1) / 2
    t = L @ np.array([x0i - c, y0i - c]) + np.array([c + x0, c + y0])
    return np.column_stack([L, t]).tolist()


def _final(job):
    (path, item, part, matrix, roll_ratio, roll_wb, taste, mirrored,
     out_path, trust, pooled) = job
    from . import flatfield
    flat = flatfield.of(taste)
    out, ev = _full(path, item, part, matrix, roll_ratio, roll_wb, taste,
                    trust, pooled, flat)
    return _write_final(out, ev, part, taste, mirrored, out_path)


def _write_final(out, ev, part, taste, mirrored, out_path):
    if mirrored:
        # Photographed through the base, so the picture arrives reversed.
        out = out[:, ::-1]
    turn = int(part.get("turn", 0) or 0) % 360
    if turn:
        out = np.rot90(out, k=-(turn // 90))
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if taste["format"] == "tiff16":
        import tifffile
        tifffile.imwrite(str(out_path), encode(out, 16), photometric="rgb")
    else:
        from PIL import Image
        Image.fromarray(encode(out, 8)).save(str(out_path), quality=92, subsampling=1)
    return out_path.name, ev


def _run(fn, jobs, workers, progress=None):
    out = []
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for i, r in enumerate(pool.imap(fn, jobs)):
                out.append(r)
                if progress:
                    progress(i, len(jobs))
    else:
        for i, job in enumerate(jobs):
            out.append(fn(job))
            if progress:
                progress(i, len(jobs))
    return out

def out_stem(doc, path, item, part):
    """The name of a frame's output files, without the extension: the
    scan's, `-a` / `-b` for the halves of a row, and **the shot number in
    front** while `roll.numbered` is on (the default): its place in the
    list's order among the scans not skipped (`document.shot_numbers`),
    zero-padded, so a folder sorted by
    name is the roll in the order it was shot -- `07_DSC_0339.jpg`. The
    scan order is not the shot order, and a file browser knows only names. One function for the pictures, the raw
    DNGs and the negatives, so the three folders line up."""
    suffix = f"-{part['name']}" if len(item["parts"]) > 1 else ""
    stem = pathlib.Path(path).stem + suffix
    if doc["roll"].get("numbered", True):
        shots = document.shot_numbers(doc)
        name = pathlib.Path(path).name
        if name in shots:
            width = max(2, len(str(len(shots))))
            stem = f"{shots[name]:0{width}d}_{stem}"
    return stem


def kept(out_dir, record_name, results, only=None):
    """The frames a record lists after this run, the old files it replaced
    removed (`sweep`). A run over the whole roll lists what it wrote. A run
    over a few scans (`only`) **adds to the record**: the rest of the roll's
    files are still there and still the roll's, so only the old files of
    those scans -- the same scan under another number -- go."""
    if only is None:
        sweep(out_dir, record_name, results)
        return results
    out_dir = pathlib.Path(out_dir)
    try:
        old = json.loads((out_dir / record_name).read_text()).get("frames") or {}
    except (OSError, ValueError):
        old = {}
    stems = {pathlib.Path(n).stem for n in only}

    def of_these(f):
        s = re.sub(r"^\d+_", "", pathlib.Path(f).stem)
        return re.sub(r"-[ab]$", "", s) in stems
    for k in old:
        f = out_dir / k
        if of_these(k) and k not in results and f.parent == out_dir and f.is_file():
            f.unlink()
    return {k: v for k, v in old.items() if not of_these(k)} | results


def sweep(out_dir, record_name, keep):
    """Remove the files the previous run recorded in `record_name` and this
    one did not write: a renumbered roll, or numbering switched on or off,
    renames every file, and the old names would sit beside the new ones as
    a second copy of the roll in the wrong order. Only names the record
    lists -- files this program wrote there -- are ever touched. Returns
    how many went."""
    out_dir = pathlib.Path(out_dir)
    try:
        old = json.loads((out_dir / record_name).read_text()).get("frames") or {}
    except (OSError, ValueError):
        return 0
    gone = 0
    for k in old:
        f = out_dir / k
        if k not in keep and f.parent == out_dir and f.is_file():
            f.unlink()
            gone += 1
    return gone


def frames_of(roll_dir, doc, only=None):
    """(path, item, part) for every frame the document says to develop, or
    for the scans named in `only`."""
    roll_dir = pathlib.Path(roll_dir)
    out = []
    for name, item in sorted(doc["items"].items()):
        path = roll_dir / name
        if item["state"] == "skipped" or not item["parts"] or not path.exists():
            continue
        if only is not None and name not in only:
            continue
        item = with_specks(item, doc, name)
        for part in item["parts"]:
            out.append((path, item, part))
    return out


def roll_statistics(frames, gamma, workers, progress=None, flat=None):
    """What the roll says, from every frame at half size: the base's
    colour and the balance, and the matrix they were read through. Two passes over the frames;
    `progress(i, n)` counts both. `flat`, the rig's light falloff, is
    divided out before anything is read: the base and the balance are
    measured through it."""
    n = len(frames)
    matrix = MATRIX.tolist()
    first = _run(_first, [(p, it, part, flat) for p, it, part in frames],
                 workers, (lambda i, m: progress(i, 2 * n)) if progress else None)
    # Two readings of the base, and which is worth having is a property of
    # the roll rather than of a scan. The rebate is film and beats
    # everything **when it is really film**: physics says unexposed film is
    # the most transparent thing on a negative, so a rebate that reads
    # darker than its own frame's lightest pixels is not film -- it is film
    # in the holder's shadow, or the scan's underlit edge. One scan like
    # that is noise; a roll of them means the rebate is not available at
    # all, and then trusting it anywhere lets a single spurious reading
    # poison the roll. So the roll votes.
    rebate_won = np.array([np.array(r)[1] >= np.array(t)[1] for r, t, dr in first])
    trust_rebate = bool(rebate_won.mean() >= REBATE_VOTE) if len(rebate_won) else False
    levels = np.array([np.maximum(r, t) if trust_rebate else np.array(t)
                       for r, t, dr in first])
    # The level is per scan only because the exposure is. The camera wrote
    # the exposure down, so the levels can be brought to a common one and
    # pooled -- and then the roll has a base level, measured over all of it,
    # instead of one noisy reading per scan.
    #
    # Only where the level is a measurement of the film. On a roll whose
    # rebate is film, every scan's level is the film itself and pooling
    # them is pooling one quantity measured many times. On a roll that
    # falls back to the frames' own shadows it is not: `frame_top` is a
    # **floor**, not an estimate. It equals the base where the scene has a
    # true black -- unexposed film under a black -- and falls below it
    # where the scene has none, so the roll's levels are a property of the
    # scenes. Pooling them was tried: it tightened the spread while pulling
    # three exposures of one negative from 0.1 levels of 255 apart to 0.9,
    # and moving single frames by up to 3.8 EV.
    #
    # What the floor costs is small, and measured on portra400-full, whose
    # 23 scans are all at one shutter speed, so its film base is a single
    # number and every bit of spread is ours. Against the roll's p90 the
    # frames' floors sit 0.17 stops low at the median and 0.32 at p75; one
    # frame, 007, sits 2.61 stops low, and developing it both ways moves
    # the picture 5 % in R/G and opens the shadow p5 from 0.039 to 0.051 --
    # the automatic exposure absorbs the rest, 4.0 stops of it, which is
    # 2.61 through the 1/0.65 of the inversion. (The x2.6 this comment used
    # to quote was the old `max(rebate, frame)` rule, from before the roll
    # voted; the rebate on that roll reads as low as 0.018 and was holding
    # the spread down by being wrong.)
    #
    # It is measured on every roll, because the editor offers it on every
    # roll: a roll can be told to use it (`base` = roll) even where the
    # vote said the rebate is not film, which is a thing a person can know
    # that the roll cannot -- a roll shot end to end at one exposure has
    # one base, whatever its scenes look like.
    #
    # `unit` is what `auto` uses and only exists where the rebate is film,
    # because that is the rule the develop's measurements were made
    # under. `base_roll_unit` is the roll's own level by
    # `roll_base_of`, always.
    unit = pooled_unit = None
    exp = [scan_exposure(p) for p, it, part in frames]
    # All or none: a roll where some files report an aperture and some do
    # not has `scan_exposure` meaning two different things inside it, and
    # the factor no longer cancels.
    ap = [scan_aperture(p) for p, it, part in frames]
    same = all(a is None for a in ap) or all(a and a > 0 for a in ap)
    if same and all(e and e > 0 for e in exp) and len(exp) == len(levels):
        per = levels[:, 1] / np.array(exp, np.float64)
        pooled_unit = roll_base_of(per)
        if trust_rebate:
            unit = float(np.median(per))
    ratios = levels / np.maximum(levels[:, 1:2], 1e-6)
    roll_ratio = np.median(ratios, axis=0).tolist()
    second = _run(_second, [(p, it, part, matrix, roll_ratio, gamma, flat)
                            for p, it, part in frames], workers,
                  (lambda i, m: progress(n + i, 2 * n)) if progress else None)
    wants = np.array([w for w, dr in second
                      if dr >= BLANK_RANGE and not np.any(np.isnan(w))])
    roll_wb = (np.median(wants, axis=0) if len(wants) else np.zeros(3)).tolist()
    return {"matrix": matrix, "flat": flat, "base_colour": roll_ratio,
            "base_from": len(ratios),
            "base_rebate_won": int(rebate_won.sum()), "base_frames": len(ratios),
            "trust_rebate": trust_rebate, "base_unit": unit,
            # The roll's level whether or not the default rule uses it, so
            # `base` = roll has something to force and the window has
            # something to show.
            "base_roll_unit": pooled_unit,
            "base_pooled": unit is not None,
            "balance_log2": roll_wb,
            "frames_balanced": int(len(wants))}


def sample(path, item, part, stats, taste=None, one_roll=True, shrink=4):
    """A frame small enough to redevelop as a slider moves: the linear frame
    at 1/(2*shrink) of the scan and the base to divide it by. Kept before
    the inversion, so `render_sample` is all a new setting costs.

    **The base comes through `base_choice`, like the develop's own.** It
    used to read `stats["base_unit"]` straight, which is the pooled level
    and only exists where the rebate is film; every roll whose rebate is
    not film therefore had its samples divided by each frame's own floor
    while the develop divided by the roll's level. On
    negativos-portra400 the two are 0.13 to 2.68 stops apart over the
    first eight frames, so the pictures in the window were not the
    pictures the roll would hold. There is one answer to where the base
    comes from and `base_choice` is it.
    """
    fr, reb_base, top = measure(path, item, part, True, stats["base_colour"],
                                crop_inset(taste), stats.get("flat"))
    if shrink > 1:
        fr = fr[::shrink, ::shrink]
    trust, unit = base_choice(stats, taste, one_roll)
    base = base_of(reb_base, top, np.array(GREY), trust,
                   (unit * (scan_exposure(path) or 0)) if unit else None)
    return {"frame": fr, "base": base,
            "matrix": np.array(stats["matrix"], np.float32)}


# Where the measured positive puts white. The base divides out the scan's
# exposure, so what is left is the negative's own scale, and the median of a
# well-exposed frame sat at 6 to 8 over the four rolls.
PLAIN_WHITE = 34.0


def whole_job(job):
    """**The whole scan developed, with one frame's automatics.**

    The window keeps this and cuts it, rather than keeping a cut and trying
    to put it back. A frame developed on its own comes out straightened and
    has to be scaled, turned and centred to sit on the scan again -- four
    chances to land in the wrong place, and it took all four. Developed in
    the scan's own coordinates there is nothing to align: the crop is the
    pixels inside a rectangle, and moving it costs a mask.

    The automatics still read the frame and only the frame -- `inner_box`
    of it, which is the largest upright rectangle inside the tilt, so no
    rebate creeps in at a corner. Everything outside is the rebate, the
    sprocket holes and the holder, developed with this frame's numbers
    because there are no others to develop it with; it is not a photograph
    and nothing claims it is.
    """
    path, item, part, stats, taste, one_roll, width = job
    w = whole_positive(job)
    out, ev = finish(w["pos"], w["roll_wb"], w["blank"], taste,
                     look_box=w["look"], px=w["px"])
    ev.update(w["ev"])
    return (path.name, part["name"]), encode(out, 8), ev


def whole_positive(job):
    """`whole_job` up to the develop: the scan inverted at `width`, and
    what `finish` takes besides the taste -- the roll's balance, whether
    the frame is blank, and the frame's box. A second's work, all before
    the taste is read but the gamma; `finish` after it is a tenth of
    that, which is what lets the look sheet (`look.py`) move a slider
    and show the frame at once."""
    path, item, part, stats, taste, one_roll, width = job
    trust, unit = base_choice(stats, taste, one_roll)
    exp = scan_exposure(path) if unit else None
    colour, trust, pooled = own_base(item, stats["base_colour"], trust,
                                     (unit * exp) if (unit and exp) else None)
    rgb, scale = linear(path, True, colour, stats.get("flat"))
    reb = rebate(rgb, scale, item, part)
    top = frame_top(inside(cut(rgb, scale, part, item["angle"])))
    lvl = (np.percentile(reb, 90, axis=0).astype(np.float32)
           if len(reb) > 500 else np.zeros(3, np.float32))
    base = base_of(lvl, top, np.array(GREY), trust, pooled)
    # Small before the inversion, not after: the develop then works on the
    # pixels the canvas will show and costs a fraction of the whole scan.
    k = max(1, round(rgb.shape[1] / width))
    if k > 1:
        rgb = rgb[:rgb.shape[0] // k * k:k, :rgb.shape[1] // k * k:k]
    small = scale * k
    x0, y0, bw, bh = inner_box(part["centre"][0] / small,
                               part["centre"][1] / small,
                               part["size"][0] / small,
                               part["size"][1] / small,
                               item["angle"], CUT_INSET)
    pos, _whole = invert(rgb, base, taste["gamma"])
    # **The range of the frame, not of the scan.** It was the range of the
    # whole scan -- holder, sprocket holes and all -- so a blank frame was
    # never blank here while `⌘R`, which inverts the cut, wrote it dark:
    # 002 of portra400-full read 2.80 D here against 0.77 in the file.
    _, drange = invert(rgb[max(0, y0):y0 + bh, max(0, x0):x0 + bw], base,
                       taste["gamma"])
    del rgb
    pos = np.clip(pos @ np.array(stats["matrix"], np.float32), 0, None)
    if item.get("specks"):
        # the scan's own coordinates, one pixel in `k` of the half-size
        # read, which bins 2x2: pixel j is centred on scan small*j + 0.5
        from . import dust
        c = (scale - 1) / 2
        pos = dust.clean(pos, dust.points(
            item["specks"], lambda x, y: ((x - c) / small, (y - c) / small)),
            small)
    blank = drange < BLANK_RANGE
    return {"pos": pos, "roll_wb": np.array(stats["balance_log2"], np.float32),
            "blank": bool(blank), "look": (y0, y0 + bh, x0, x0 + bw),
            "px": float(small),
            "ev": dict(blank=bool(blank), density_range=round(drange, 3),
                       base=[round(float(v), 4) for v in base],
                       base_source=("picked" if item.get("base_sample") else
                                    base_source(lvl, top, trust, pooled)))}


def render_sample(s, stats, taste, mirrored, turn=0):
    """The positive of a sample under this taste, 0..1 linear sRGB."""
    pos, drange = invert(s["frame"], s["base"], taste["gamma"])
    pos = np.clip(pos @ s["matrix"], 0, None)
    out, ev = finish(pos, np.array(stats["balance_log2"], np.float32),
                     drange < BLANK_RANGE, taste,
                     look_box=frame_box(pos.shape, taste))
    ev["blank"] = bool(drange < BLANK_RANGE)
    if mirrored:
        out = out[:, ::-1]
    if turn:
        out = np.rot90(out, k=-(int(turn) // 90))
    return out, ev


def develop_roll(roll_dir, doc, out_dir, taste, workers, progress=None,
                 stats=None, only=None):
    """Every frame of the document: the roll's statistics unless given,
    then each frame at full size. Returns ({output name: evidence}, roll).

    `previews`, if a dict is passed, is filled with `(scan, part) ->` a
    screen-sized copy of each finished picture, taken off the full-size one
    the job already has. It is for the window: a roll that has just been
    processed can then be walked frame by frame with the develop on screen
    and nothing left to compute.
    """
    frames = frames_of(roll_dir, doc, only)
    if not frames:
        return {}, {}
    if stats is None:
        from . import flatfield
        stats = roll_statistics(frames, taste["gamma"], workers,
                                flat=flatfield.of(taste))
    mirrored = doc["roll"].get("mirrored", True)
    ext = ".tif" if taste["format"] == "tiff16" else ".jpg"
    trust, unit = base_choice(stats, taste, doc["roll"].get("one_roll", True))
    jobs = []
    for path, item, part in frames:
        jobs.append((path, item, part, stats["matrix"], stats["base_colour"],
                     stats["balance_log2"], taste, mirrored,
                     str(pathlib.Path(out_dir) / (out_stem(doc, path, item, part)
                                                  + ext)),
                     trust,
                     (unit * (scan_exposure(path) or 0)) if unit else None))
    results = dict(_run(_final, jobs, workers, progress))
    won, total = stats.get("base_rebate_won", 0), stats.get("base_frames", 0)
    trust = stats.get("trust_rebate", True)
    want = (taste or {}).get("base", "auto")
    roll_evidence = {"base_colour": [round(v, 4) for v in stats["base_colour"]],
                     # The two the views need to read the base the same way
                     # this run did, rather than guess at it per scan.
                     "trust_rebate": trust,
                     "base_unit": stats.get("base_unit"),
                     "base_roll_unit": stats.get("base_roll_unit"),
                     "base_from": "%d scans" % (stats.get("base_from") or 0),
                     "base_rebate": "%d of %d scans show film lighter than their "
                                    "own frame" % (won, total),
                     "base_method": base_method_of(stats, taste),
                     "base_setting": want,
                     "balance_log2": [round(v, 3) for v in stats["balance_log2"]],
                     "frames_balanced": stats["frames_balanced"]}
    return results, roll_evidence


def record(out_dir, taste, roll_ev, results, only=None):
    """`developed.json` beside the positives: the settings, what the roll
    measured, and the evidence for every frame (`kept`)."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = kept(out_dir, "developed.json", results, only)
    (out_dir / "developed.json").write_text(json.dumps(
        {"settings": taste, "roll": roll_ev, "frames": results}, indent=1) + "\n")


def run(args):
    import os
    roll = pathlib.Path(args.roll)
    doc = document.load(roll)
    taste = dict(TASTE) | dict(doc["roll"].get("develop") or {})
    for key in TASTE:
        val = getattr(args, key, None)
        if val is not None:
            taste[key] = val
    if str(taste.get("flat")).lower() in ("none", "off", ""):
        taste["flat"] = None
    from . import flatfield
    flatfield.of(taste)                 # a missing calibration stops here, not in a worker
    doc["roll"]["develop"] = taste
    out_dir = pathlib.Path(args.output) if args.output else roll / "developed"
    workers = args.workers or min(8, os.cpu_count() or 1)
    n = sum(len(it["parts"]) for it in doc["items"].values()
            if it["state"] != "skipped")
    print(f"{n} frames in {roll} -> {out_dir}")

    def tick(i, total):
        print(f"\r  {i + 1}/{total}", end="", flush=True)

    results, roll_ev = develop_roll(roll, doc, out_dir, taste, workers, tick)
    print()
    blank = sorted(k for k, v in results.items() if v["blank"])
    evs = [v["exposure"] for v in results.values()]
    if "base_method" in roll_ev:
        print(f"  the base: {roll_ev['base_method']} — {roll_ev['base_rebate']}"
              + ("" if "rebate" in roll_ev["base_method"] else
                 ", so it is a floor and not a measurement: a shot of "
                 "unexposed film in the folder would settle it"))
    print(f"  base colour R/G {roll_ev['base_colour'][0]:.3f}, "
          f"B/G {roll_ev['base_colour'][2]:.3f}; roll balance "
          f"{roll_ev['balance_log2'][0]:+.2f} / {roll_ev['balance_log2'][2]:+.2f} EV")
    if evs:
        print(f"  exposure median {np.median(evs):+.2f} EV, "
              f"from {min(evs):+.2f} to {max(evs):+.2f}")
    if blank:
        print(f"  {len(blank)} blank, written dark: {', '.join(blank)}")
    record(out_dir, taste, roll_ev, results)
    document.save(doc, roll)
    print(f"  written {len(results)} files and developed.json; settings kept in "
          f"{document.FILENAME}")
    return 0
