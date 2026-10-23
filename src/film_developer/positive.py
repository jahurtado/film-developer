"""The positive DNG: the negative inverted on its own Bayer mosaic and
written as a raw a digital camera could have made.

What `apply` writes. Every photosite is inverted against its channel's
base (`develop.invert_mosaic`: the same inversion `develop` uses in its
`mosaic` order), so the file is still a CFA mosaic, never demosaiced, and
a raw developer opens it as it would a camera raw: its own demosaic, its
own colour, its own curves. The picture arrives reversed, photographed
through the base, and the mirror is done on the mosaic -- a mirrored
Bayer pattern is still a Bayer pattern, only named the other way round
-- as is a quarter turn. The tilt is not: a rotation takes the four
sub-planes off their common grid, so the crop is by
translation, on even coordinates, and the tilt of up to a degree is
left for the developer. By default it is the tilted frame's bounding
box with a small margin (`outer`): the frame's edges come with it,
tilted, so the file shows what was done to it and the developer
straightens and crops with everything still there. `inner` cuts the
axis-aligned rectangle inside the frame instead, edges out, a picture
that only looks straight.

Nothing the sensor measured is lost on the way into 16 bits. The top of
the file is the frame's brightest *measured* photosite -- more than
FLOOR levels above black; at the floor the negative is denser than the
camera reads, and those come out white, which is what they are -- and
the file holds the **square root** of the positive over that top, with a
LinearizationTable that squares it back. A developer that works in
float (Adobe) gets the positive exact; one that linearises into 16 bits
(LibRaw) gets it on 65535 linear levels, which is what a plain 16-bit
file would have given it anyway. Checked on LibRaw: the table is applied
on load, and the values read back linear to 0.002 %. A floating-point
DNG was measured too: LibRaw converts it to 16-bit integers on reading,
so it buys nothing there.

**The colour tags are the positive's own, not the camera's.** They were the camera's -- its ColorMatrix, ForwardMatrix and
CameraCalibration -- on the argument that the sensitivities are still the
camera's. But dividing by the base takes the copy light and the camera's
filters out as one gain per channel, and a daylight film's grey under
daylight is then three equal channels: on DSC_0315_alt-0239 the roll's
grey reads R 0.94, G 1, B 1.15. The D5600's matrix expects a daylight
grey at R 0.46, B 0.71, so a developer set to 5600 K doubled red and
put blue up 1.6x over green: pink, at any temperature (2.03 / 1.62 at
5600 K, 1.91 / 1.76 at 5000), and the camera's matrix had multiplied
saturation by 1.9 besides (`develop.MATRIX`). So the mosaic is divided by
the roll's balance, which puts the roll's grey at (1, 1, 1), AsShotNeutral
is (1, 1, 1), and ColorMatrix1 is linear sRGB's with its white moved to
5600 K (`POSITIVE_MATRIX`): R = G = B *is* 5600 K, so 5600 K and no tint
is the roll's own balance, and the temperature warms and cools from
there. Make, Model and UniqueCameraModel name the file as this program's
positive, because Camera Raw finds its own profile by UniqueCameraModel
and would put the camera's matrix back. The untouched negatives
(`negative.py`) keep the camera's tags: those are what the sensor saw.
BaselineExposure puts the frame's median at 18 % grey in a
developer that honours it -- one that ignores it shows the frame dark by
that much, which on a night frame with lamps 15 EV over its median is
8 EV.

The inversion's gamma is the roll's `develop`
settings: the DNG is the positive as this program understands it, and
the two steps have to agree.
"""

import json
import math
import pathlib

import numpy as np

from . import develop, document

FLOOR = develop.FLOOR
WHITE = 65535
TILE = 256       # px: the side of a lossless JPEG tile of the mosaic
INSET = 0.01          # inner crop: of each side, so the soft edge of the exposure stays out
MARGIN = 0.012        # outer crop: the least it reaches past the frame, of each side
BASELINE_MAX = 8.0    # EV; Camera Raw's exposure control does not go further anyway
PREVIEW_PX = 1400     # the preview's longest side
PREVIEW_TOP = 0.92    # where the preview's brightest 0.5 % lands, at most
CROPS = ("outer", "inner")
WHITE_K = 5600        # the temperature R = G = B stands for: the light a daylight film is made for


def _planck_xyz(T):
    """XYZ (Y = 1) of a black body at `T` kelvin (Kim et al.'s cubic fit to
    the Planckian locus, 1667-25000 K): the line a developer's temperature
    control walks and its tint is measured off. Daylight's own locus (CIE D)
    sits off it, and white put there read back in Apple's RAW engine as
    5598 K with tint +10."""
    if T <= 4000:
        x = -0.2661239e9 / T ** 3 - 0.2343589e6 / T ** 2 + 0.8776956e3 / T + 0.179910
    else:
        x = -3.0258469e9 / T ** 3 + 2.1070379e6 / T ** 2 + 0.2226347e3 / T + 0.240390
    if T <= 2222:
        y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    elif T <= 4000:
        y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483
    return np.array([x / y, 1.0, (1 - x - y) / y])


def positive_matrix(T=WHITE_K):
    """XYZ -> the positive's RGB, for ColorMatrix1: linear sRGB's primaries
    with the white moved from D65 to a black body at `T` (Bradford), scaled so
    its largest response to that white is 1, as DNG wants. Its neutral at
    `T` is (1, 1, 1); at 5000 K (1.10, 1, 0.87), at 6500 K (0.90, 1, 1.17)."""
    srgb = np.array([[3.2406, -1.5372, -0.4986],
                     [-0.9689, 1.8758, 0.0415],
                     [0.0557, -0.2040, 1.0570]])
    brad = np.array([[0.8951, 0.2664, -0.1614],
                     [-0.7502, 1.7135, 0.0367],
                     [0.0389, -0.0685, 1.0296]])
    d65 = np.array([0.95047, 1.0, 1.08883])
    white = _planck_xyz(T)
    cat = np.linalg.inv(brad) @ np.diag((brad @ d65) / (brad @ white)) @ brad
    m = srgb @ cat
    return m / (m @ white).max()


def positive_tags(path):
    """The colour and identity tags of a positive DNG (see the module's
    docstring): the matrix whose white is `WHITE_K`, under D55 -- the
    nearest standard illuminant, and with one matrix only it names it
    and nothing else -- and a camera name that is not the scanning
    camera's, carried in brackets so the file still says what scanned it."""
    src = source_tags(path)
    cam = str(src.get("UniqueCameraModel", src.get("Model", "camera"))).strip()
    name = f"Positive ({cam})"
    m = positive_matrix()
    return {"Make": "film-developer", "Model": name,
            "UniqueCameraModel": f"film-developer {name}",
            "ColorMatrix1": tuple(v for x in m.ravel()
                                  for v in (int(round(x * 10000)), 10000)),
            "CalibrationIlluminant1": 20}                 # D55


def inner_rect(cx, cy, w, h, angle, inset=INSET):
    """The largest axis-aligned rectangle of the frame's aspect inside the
    tilted frame (`develop.inner_box`): the `inner` crop, and where the
    frame's median is read for BaselineExposure."""
    return develop.inner_box(cx, cy, w, h, angle, inset)


def dng_rect(kind, part, angle, W, H, crop_pct=0.0):
    """Where a DNG of this frame is cut from the scan: (x0, y0, w, h, lost),
    on even coordinates and inside the scan; `lost` is how many pixels of
    the rectangle asked for each side (left, top, right, bottom) fell off
    the sensor and are not in the file.

    **The roll's crop, as the pictures take it** (`develop.crop_inset`),
    so the DNG shows what the window and the JPEG show:

    - `outer` reaches **at least** `MARGIN` past the frame on every side
      -- a raw that shows its own edges shows what was done to it, and the
      developer straightens -- and further if the roll's crop asks for more:
      at 0 % it is 1.2 % past, at -5 % it is the picture's own 4 % (5 %
      less the 1 % every picture trims off the exposure's soft edge). Then
      the box around that rectangle tilted, because a mosaic cannot turn.
    - `inner` is the largest upright rectangle inside the frame, trimmed by
      the roll's crop when it trims and never grown by it: outside the
      frame is not inside the frame.

    A picture past the sensor's edge is padded black by `develop.cut`; a
    raw has no photosites there, so it is cut and said (`lost`)."""
    from . import findframe
    cx, cy = part["centre"]
    w, h = part["size"]
    inset = develop.CUT_INSET + float(crop_pct or 0.0) / 100.0
    if kind == "inner":
        x0, y0, bw, bh = inner_rect(cx, cy, w, h, angle,
                                    max(inset, develop.CUT_INSET))
        x1, y1 = x0 + bw, y0 + bh
    else:
        inset = min(inset, -MARGIN)
        pts = findframe.corners(cx, cy, w * (1 - 2 * inset), h * (1 - 2 * inset), angle)
        x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
        x1, y1 = max(p[0] for p in pts), max(p[1] for p in pts)
    lost = [max(0, round(-x0)), max(0, round(-y0)),
            max(0, round(x1 - W)), max(0, round(y1 - H))]
    x0, y0 = int(max(0.0, x0)) // 2 * 2, int(max(0.0, y0)) // 2 * 2
    wc = min((int(math.ceil(min(x1, W))) - x0) // 2 * 2, (W - x0) // 2 * 2)
    hc = min((int(math.ceil(min(y1, H))) - y0) // 2 * 2, (H - y0) // 2 * 2)
    return x0, y0, wc, hc, lost


def source_tags(path):
    """The camera's colour and identity tags: from the DNG when the scan is
    one, from LibRaw's table for the camera otherwise."""
    import rawpy
    import tifffile
    tags = {}
    try:
        with tifffile.TiffFile(str(path)) as t:
            ifd0 = t.pages[0].tags
            for name in ("Make", "Model", "UniqueCameraModel", "ColorMatrix1",
                         "ColorMatrix2", "CalibrationIlluminant1",
                         "CalibrationIlluminant2", "ForwardMatrix1",
                         "ForwardMatrix2", "CameraCalibration1",
                         "CameraCalibration2"):
                tag = ifd0.get(name)
                if tag is not None:
                    tags[name] = tag.value
    except Exception:  # a raw that tifffile cannot walk: LibRaw still can
        pass
    if "ColorMatrix1" not in tags:
        with rawpy.imread(str(path)) as r:
            m = np.array(r.rgb_xyz_matrix[:3], np.float64)    # XYZ -> camera
        tags["ColorMatrix1"] = tuple(v for row in m
                                     for v in ((int(round(x * 10000)), 10000) for x in row))
        tags["CalibrationIlluminant1"] = 21                    # D65
    return tags


def _rational(values, den):
    return tuple(x for v in values for x in (int(round(float(v) * den)), den))


def _blocks(crop, cmap, longest=PREVIEW_PX):
    """Each 2 x 2 block of the mosaic as one RGB pixel (no interpolation),
    then every k-th. Returns (rgb, mosaic photosites per pixel)."""
    h, w = crop.shape[0] // 2 * 2, crop.shape[1] // 2 * 2
    blk = crop[:h, :w].reshape(h // 2, 2, w // 2, 2)
    cbl = cmap[:h, :w].reshape(h // 2, 2, w // 2, 2)
    rgb = np.zeros((h // 2, w // 2, 3), np.float32)
    for c in range(3):
        m = (cbl == c)
        rgb[..., c] = (blk * m).sum((1, 3)) / np.maximum(m.sum((1, 3)), 1)
    k = max(1, int(math.ceil(max(rgb.shape[:2]) / longest)))
    return rgb[::k, ::k], 2 * k


def preview_of(crop, cmap, neutral, baseline, matrix, longest=PREVIEW_PX):
    """An 8-bit sRGB-ish picture of a mosaic for the file's first IFD: the
    2 x 2 blocks, the neutral and BaselineExposure applied, the matrix, a
    2.2 gamma. What the Finder, Photos and a browser show of the
    *negatives* (`negative.py`); the positive's is `developed_preview`."""
    rgb, _ = _blocks(crop, cmap, longest)
    rgb = rgb / np.asarray(neutral, np.float32)[None, None, :]
    rgb = np.clip(rgb @ np.asarray(matrix, np.float32), 0, None)
    # BaselineExposure puts the median at 18 %, unless that clips the
    # brightest 0.5 %: a beach (021) went white. The preview is a picture,
    # not the file's exposure; the tag stays what it is.
    y = rgb @ develop.LUMA
    hh, ww = y.shape
    inside = y[hh // 10: -hh // 10 or None, ww // 10: -ww // 10 or None]   # not the holder, gone white
    scale = min(2.0 ** baseline, PREVIEW_TOP / max(float(np.percentile(inside, 99.5)), 1e-6))
    rgb = np.clip(rgb * scale, 0, 1)
    return (rgb ** (1 / 2.2) * 255).astype(np.uint8)


def developed_preview(crop, cmap, roll_wb, blank, taste, look_box,
                      longest=PREVIEW_PX):
    """The positive's preview: **the develop**, `develop.finish` on the
    2 x 2 blocks of the inverted mosaic with the roll's balance and taste,
    encoded as the JPEG is (`develop.encode`). `look_box` is the frame in
    the crop's photosites, `(y0, y1, x0, x1)`: what the automatics read.

    It was a minimal develop -- the neutral, BaselineExposure, a 2.2 gamma
    -- with no curve, no shadow rule, no black point, no saturation and no
    shoulder, so in the Finder and Photos the DNG looked flatter and duller
    than the JPEG the program writes of the same frame. A raw developer
    ignores the preview and renders the data with its own profile and
    curve, so this changes what the quick views show and nothing else."""
    rgb, step = _blocks(crop, cmap, longest)
    box = None
    if look_box is not None:
        y0, y1, x0, x1 = (int(round(v / step)) for v in look_box)
        h, w = rgb.shape[:2]
        y0, x0 = max(0, min(y0, h - 1)), max(0, min(x0, w - 1))
        box = (y0, max(y0 + 1, min(y1, h)), x0, max(x0 + 1, min(x1, w)))
    out, _ = develop.finish(rgb, np.asarray(roll_wb, np.float32), blank,
                            taste, look_box=box, px=step)
    return develop.encode(out, 8)


def write_frame(path, item, part, roll_ratio, balance_log2, gamma, out,
                mirrored=True, crop="outer", matrix=None,
                trust_rebate=True, pooled=None, flat=None, crop_pct=0.0,
                taste=None, dust_raw=False, compress=True):
    """One part of one scan as a positive DNG at `out`. Returns evidence.
    `crop` is `outer` (the tilted frame's bounding box, edges and all) or
    `inner` (the axis-aligned rectangle inside it). `matrix` is the
    camera's, for the preview only. `trust_rebate` and `pooled` are the
    roll's decision about the base, the same two the developed positives
    get -- without them this wrote its own base and the DNG and the JPEG
    of one negative disagreed. `flat` is the rig's light falloff
    (`flatfield`), divided out of the mosaic before the inversion.
    `taste` is the roll's develop settings, for the preview: with it the
    preview is the develop (`developed_preview`), without it the minimal
    one. `dust_raw` takes the scan's dust marks (`item["specks"]`, the hand
    marks and the accepted proposals) out of the mosaic itself
    (`dust.clean_mosaic`) -- off unless asked, because the file is then
    no longer only the negative inverted."""
    kind = crop if crop in CROPS else "outer"
    P, chan, measured, top, drange = develop.invert_mosaic(
        path, item, part, roll_ratio, gamma, trust_rebate, pooled, flat)
    # **The roll's balance goes into the photosites**, so the roll's grey is
    # R = G = B, which the tags call `WHITE_K` (`positive_tags`). Green
    # stays; the top moves with the channel raised most, so nothing that
    # fitted under it before is clipped now.
    bal = np.asarray(balance_log2, np.float64)
    gains = np.exp2(bal - bal[1])
    P = P * gains.astype(np.float32)[chan]
    top = top * float(gains.max())
    cx, cy = part["centre"]
    w, h = part["size"]
    ix0, iy0, iw, ih = inner_rect(cx, cy, w, h, item["angle"])
    # the frame's median, for BaselineExposure: read inside the frame, whatever
    # the crop -- the outer one brings the rebate (black) and the holder (white)
    median = float(np.median(P[iy0:iy0 + ih, ix0:ix0 + iw])) / top
    x0, y0, wc, hc, lost = dng_rect(kind, part, item["angle"], P.shape[1],
                                    P.shape[0], crop_pct)
    crop = P[y0:y0 + hc, x0:x0 + wc]
    cmap = chan[y0:y0 + hc, x0:x0 + wc]
    mcrop = measured[y0:y0 + hc, x0:x0 + wc]
    del P, chan, measured
    cleaned = 0
    if dust_raw and item.get("specks"):
        # **In the crop, before the mirror and the turn**: the marks are in
        # scan pixels and the crop is the scan shifted by (x0, y0) on even
        # coordinates, so a mark is just moved, and the 2 x 2 blocks the
        # sub-planes are cut from are the sensor's own. Before the preview
        # too, which is then the cleaned picture.
        from . import dust
        crop = dust.clean_mosaic(crop, dust.points(
            item["specks"], lambda x, y: (x - x0, y - y0)))
        cleaned = len(item["specks"])
    blank = bool(drange < develop.BLANK_RANGE)
    if taste is not None:
        # Developed before the mirror and the turn, where the frame's own
        # rectangle is still in the crop's coordinates; the 8-bit picture
        # is turned with the mosaic below.
        look = (iy0 - y0, iy0 - y0 + ih, ix0 - x0, ix0 - x0 + iw)
        # the balance is in the photosites now; what is left of it is
        # green's, which is exposure, so the develop comes out the same
        preview = developed_preview(crop, cmap, np.full(3, bal[1]), blank,
                                    taste, look)
        if mirrored:
            preview = preview[:, ::-1]
    if mirrored:
        crop, cmap, mcrop = crop[:, ::-1], cmap[:, ::-1], mcrop[:, ::-1]
    turn = int(part.get("turn", 0) or 0) % 360
    if turn:
        k = -(turn // 90)
        crop, cmap, mcrop = (np.rot90(crop, k=k), np.rot90(cmap, k=k),
                             np.rot90(mcrop, k=k))
        if taste is not None:
            preview = np.rot90(preview, k=k)
    # The top is the frame's brightest measured photosite (invert_mosaic
    # reads it inside the frame): with the outer crop the holder's penumbra
    # comes along, denser than anything in the picture, and goes white.
    code = np.sqrt(np.clip(crop / top, 0, 1)) * WHITE
    code[~mcrop] = WHITE
    raw = np.ascontiguousarray(np.clip(code, 0, WHITE).astype(np.uint16))
    table = np.round((np.arange(65536) / 65535.0) ** 2 * WHITE).astype(np.uint16)
    baseline = float(np.clip(math.log2(0.18 / max(median, 1e-9)), -6, BASELINE_MAX))
    neutral = np.ones(3)
    tags = positive_tags(path)
    if taste is None:
        preview = preview_of(crop, cmap, neutral, baseline,
                             np.eye(3) if matrix is None else matrix)
    preview = np.ascontiguousarray(preview)
    cfa = write_cfa(out, raw, cmap, tags, neutral, baseline, preview,
                    black=0, white=WHITE, table=table, compress=compress)
    return {"crop": [x0, y0, wc, hc], "crop_kind": kind,
            "crop_pct": crop_pct, "off_sensor_ltrb": lost,
            "size": [raw.shape[1], raw.shape[0]],
            "cfa": list(cfa), "baseline_exposure": round(baseline, 2),
            "median_of_top": round(median, 5), "top": round(top, 3),
            "unmeasured": round(float((~mcrop).mean()), 5),
            "blank": blank,
            "dust_marks": cleaned,
            "density_range": round(drange, 3)}


def write_cfa(out, raw, cmap, tags, neutral, baseline, preview,
              black=0, white=WHITE, table=None, compress=True):
    """A CFA DNG at `out`: `raw` the uint16 mosaic, `cmap` each photosite's
    channel (for the CFAPattern), `tags` the colour and identity tags: the
    camera's from `source_tags` for a negative, `positive_tags` for a positive.
    `black` is one level, or four for the 2 x 2 repeat in raster order.
    Shared by the positive and the untouched negative (`negative.py`),
    which differ in the values and the levels and in nothing else.
    `compress` writes the mosaic as lossless JPEG (`_lossless`).
    Returns the CFA pattern written."""
    import tifffile
    cfa = tuple(int(v) for v in cmap[0:2, 0:2].ravel())
    # Two IFDs, the layout Adobe writes and the one Apple's decoders want:
    # the first a JPEG preview (NewSubfileType 1) carrying the DNG and
    # colour tags, the mosaic in a SubIFD (NewSubfileType 0) with the CFA
    # tags. With the mosaic alone in IFD0, QuickLook made no thumbnail and
    # hung; with the preview first it thumbnails (measured with qlmanage),
    # and the preview has to be one JPEG strip -- in several, QuickLook
    # showed a 16-pixel band.
    ifd0 = [
        (50706, "B", 4, (1, 4, 0, 0), True),                    # DNGVersion
        (50707, "B", 4, (1, 1, 0, 0), True),                    # DNGBackwardVersion
        (50708, "s", 0, str(tags.get("UniqueCameraModel",
                                     tags.get("Model", "film-developer"))), True),
        (50728, "2I", 3, _rational(neutral, 1000000), True),    # AsShotNeutral
        (50730, "2i", 1, (int(round(baseline * 100)), 100), True),  # BaselineExposure
    ]
    for name, code_ in (("ColorMatrix1", 50721), ("ColorMatrix2", 50722),
                        ("ForwardMatrix1", 50964), ("ForwardMatrix2", 50965),
                        ("CameraCalibration1", 50723), ("CameraCalibration2", 50724)):
        if name in tags:
            v = tuple(int(x) for x in np.asarray(tags[name]).ravel())
            ifd0.append((code_, "2i", len(v) // 2, v, True))
    for name, code_ in (("CalibrationIlluminant1", 50778),
                        ("CalibrationIlluminant2", 50779)):
        if name in tags:
            ifd0.append((code_, "H", 1, int(tags[name]), True))
    for name, code_ in (("Make", 271), ("Model", 272)):
        if name in tags:
            ifd0.append((code_, "s", 0, str(tags[name]), True))
    sub = [
        (33421, "H", 2, (2, 2), True),                          # CFARepeatPatternDim
        (33422, "B", 4, cfa, True),                             # CFAPattern
        (50711, "H", 1, 1, True),                               # CFALayout
    ]
    if table is not None:
        sub.append((50712, "H", 65536, tuple(int(x) for x in table), True))  # LinearizationTable
    if isinstance(black, (tuple, list)):
        # one per photosite of the 2 x 2 repeat, for a sensor whose
        # channels sit on different blacks
        sub += [(50713, "H", 2, (2, 2), True),                  # BlackLevelRepeatDim
                (50714, "I", 4, tuple(int(b) for b in black), True)]
    else:
        sub.append((50714, "I", 1, int(black), True))           # BlackLevel
    sub += [
        (50717, "I", 1, int(white), True),                      # WhiteLevel
        (50719, "I", 2, (0, 0), True),                          # DefaultCropOrigin
        (50720, "I", 2, (raw.shape[1], raw.shape[0]), True),    # DefaultCropSize
        (50829, "I", 4, (0, 0, raw.shape[0], raw.shape[1]), True),  # ActiveArea
    ]
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(str(out)) as w:
        w.write(preview, photometric="rgb", compression="jpeg",
                compressionargs={"level": 90}, rowsperstrip=preview.shape[0],
                subifds=1, subfiletype=1, extratags=ifd0, metadata=None,
                description="", software="film-developer", resolution=(72, 72))
        if compress:
            w.write(_lossless(raw), shape=raw.shape, dtype=np.uint16,
                    tile=(TILE, TILE), compression=7, photometric=32803,
                    # said, or tifffile writes 12 for a JPEG of uint16:
                    # LibRaw and Apple read the precision off the JPEG
                    # and did not notice, DaVinci Resolve trusts the tag
                    # and showed the file burnt
                    bitspersample=16,
                    planarconfig=None, subfiletype=0, extratags=sub,
                    metadata=None, description="")
        else:
            w.write(raw, photometric=32803, compression=None, planarconfig=None,
                    subfiletype=0, extratags=sub, metadata=None, description="")
    return cfa


def _lossless(raw):
    """The mosaic's tiles, each one lossless JPEG (ITU T.81 process 14,
    Compression 7): what cameras and Adobe write into a raw DNG, and what
    LibRaw and Camera Raw read. Tiled because a reader can decode tiles
    in parallel; the edge tiles are padded with zeros to the full tile,
    as TIFF wants, and the reader crops them.

    **Two components, as Adobe writes it.** A tile row goes in as half as
    many samples of two components, interleaved, so the decoder's output
    is the row as it was and each photosite is predicted from the one two
    to its left -- its own colour -- instead of its neighbour of another.
    Measured on carrete_2 (DSC_0329), the left difference costs 13.8 bits
    a sample in the positive and 11.7 in the negative, the same-colour one
    11.9 and 8.5; on a tile of the negative, 102 kB in one component and
    74 kB in two. liblj92 (`ljpeg_encode`) only writes one, so libjpeg-
    turbo's lossless mode does it."""
    import imagecodecs
    H, W = raw.shape
    for y in range(0, H, TILE):
        for x in range(0, W, TILE):
            t = raw[y:y + TILE, x:x + TILE]
            if t.shape != (TILE, TILE):
                t = np.pad(t, ((0, TILE - t.shape[0]), (0, TILE - t.shape[1])))
            t = np.ascontiguousarray(t, np.uint16).reshape(TILE, TILE // 2, 2)
            yield imagecodecs.jpeg8_encode(t, lossless=True, bitspersample=16)


def _job(job):
    (path, item, part, roll_ratio, balance, gamma, out, mirrored, crop,
     matrix, trust, pooled, flat, crop_pct, taste, dust_raw, compress) = job
    return pathlib.Path(out).name, write_frame(path, item, part, roll_ratio, balance,
                                               gamma, out, mirrored, crop,
                                               matrix, trust, pooled, flat, crop_pct,
                                               taste, dust_raw, compress)


def write_roll(roll_dir, doc, out_dir, settings, workers, progress=None,
               stats=None):
    """Every frame of the document as a positive DNG. The roll's statistics
    (base colour, balance) unless given, then each frame.
    Returns ({file name: evidence}, roll evidence)."""
    roll_dir = pathlib.Path(roll_dir)
    frames = develop.frames_of(roll_dir, doc)
    if not frames:
        return {}, {}
    gamma = settings["gamma"]
    from . import flatfield
    flat = flatfield.load(settings.get("flat"))
    if stats is None:
        stats = develop.roll_statistics(frames, gamma, workers, progress, flat)
    mirrored = doc["roll"].get("mirrored", True)
    trust, unit = develop.base_choice(stats, settings,
                                      doc["roll"].get("one_roll", True))
    # the roll's develop settings, for the previews: the gamma is the
    # settings' own, which the inversion was made under
    taste = (dict(develop.TASTE) | dict(doc["roll"].get("develop") or {})
             | {"gamma": gamma})
    jobs = []
    for path, item, part in frames:
        jobs.append((path, item, part, stats["base_colour"], stats["balance_log2"],
                     gamma, str(pathlib.Path(out_dir)
                                / (develop.out_stem(doc, path, item, part) + ".dng")),
                     mirrored, settings.get("crop", "outer"),
                     stats.get("matrix"), trust,
                     (unit * (develop.scan_exposure(path) or 0)) if unit else None,
                     flat, settings.get("crop_pct", 0.0), taste,
                     bool(settings.get("dust", False)),
                     bool(settings.get("compress", True))))
    results = dict(develop._run(_job, jobs, workers, progress))
    roll_ev = {"base_colour": [round(v, 4) for v in stats["base_colour"]],
               "trust_rebate": stats.get("trust_rebate", True),
               "base_unit": stats.get("base_unit"),
               "base_roll_unit": stats.get("base_roll_unit"),
               "base_setting": settings.get("base", "auto"),
               "balance_log2": [round(v, 3) for v in stats["balance_log2"]],
               "gamma": gamma}
    return results, roll_ev


def record(out_dir, settings, roll_ev, results):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    develop.sweep(out_dir, "positives.json", results)
    (out_dir / "positives.json").write_text(json.dumps(
        {"settings": settings, "roll": roll_ev, "frames": results}, indent=1) + "\n")


def settings_of(doc):
    """What the DNG bakes in: the inversion's gamma, from the roll's
    develop settings so the two steps agree; and the crop, the roll's
    `apply` setting (`outer` unless it says `inner`)."""
    taste = dict(develop.TASTE) | dict(doc["roll"].get("develop") or {})
    own = dict(doc["roll"].get("apply") or {})
    crop = own.get("crop", "outer")
    return {"gamma": taste["gamma"], "base": taste.get("base", "auto"),
            "flat": taste.get("flat"),
            # the roll's crop, which the pictures take: the DNG frames
            # what they frame (`dng_rect`)
            "crop_pct": float(taste.get("crop", 0.0) or 0.0),
            "crop": crop if crop in CROPS else "outer",
            # the dust marks taken out of the mosaic too: the roll's
            # `apply` setting, off unless it says so
            "dust": bool(own.get("dust", False)),
            "compress": bool(own.get("compress", True))}
