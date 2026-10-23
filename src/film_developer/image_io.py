"""Raw in, previews out.

In: NEF, CR2, CR3, DNG through LibRaw (`rawpy`).

Two things are read out of a scan and they are read differently on purpose:

- **`density()`** is what the detector measures on: one green sub-plane of the
  mosaic, downsampled, as optical density. No demosaicing -- a green sub-plane
  is already a real image on its own grid, the frame edge is hundreds of pixels
  long, and an eighth of the resolution costs nothing while buying a 60x saving
  in the angle search.
- **`preview()`** is what a person looks at: demosaiced, and turned the right
  way up, because a crop cannot be judged on a negative. That inversion is a
  preview and nothing else -- `apply` writes what came in.

Density rather than raw counts because the black level differs between cameras
(1023 on the Sigma fp, 600 on the Nikon here) and every threshold downstream
would otherwise be per-camera. Density rather than transmittance because film
*is* a density medium: the step at a frame edge is the same size in density
whether the scan was made through a mask or on a bare light table, and it is
not in transmittance.
"""

import numpy as np
import rawpy

RAW_SUFFIXES = (".dng", ".nef", ".cr2", ".cr3")
# The working width for detection. 760 px of a 6064 px scan is a factor of 8,
# so an edge is located to within 8 px of the original before any refinement.
WORK_WIDTH = 760


def _levels(raw):
    return float(np.mean(raw.black_level_per_channel)), float(raw.white_level)


def _bin(a, k):
    """Average non-overlapping k x k blocks. Cheap, and it is a real filter:
    box-averaging before decimating is what keeps a soft edge in the profile
    instead of aliasing it."""
    if k <= 1:
        return a
    a = a[: a.shape[0] // k * k, : a.shape[1] // k * k]
    return a.reshape(a.shape[0] // k, k, a.shape[1] // k, k).mean((1, 3))


def scan_size(path):
    with rawpy.imread(str(path)) as r:
        v = r.raw_image_visible
        return v.shape[1], v.shape[0]


def density(path, width=WORK_WIDTH):
    """(density image, scale) -- scale being scan pixels per density pixel."""
    with rawpy.imread(str(path)) as r:
        v = r.raw_image_visible.astype(np.float32)
        black, white = _levels(r)
        full_w = v.shape[1]
    g = np.clip(v[1::2, 0::2] - black, 1.0, None) / (white - black)
    k = max(1, g.shape[1] // width)
    d = -np.log10(_bin(g, k))
    return d, full_w / d.shape[1]


def _base(y, radius, eps=2e-3):
    """The picture without its fine structure, with the edges left standing.

    A self-guided filter: four box filters and some arithmetic, milliseconds on
    a display-sized frame. A plain Gaussian would be simpler and wrong -- it
    blurs *across* the frame edge, and the frame edge is the one thing this
    window exists to find. `eps` is what counts as an edge.
    """
    from scipy.ndimage import uniform_filter
    r = max(3, int(radius))
    # Across the picture only, never across the channels. `size=r` on an
    # H x W x 3 array filters the third axis too: it blurs red into green into
    # blue, and it is four times the work for the privilege.
    size = (r, r, 1) if y.ndim == 3 else r
    mean = uniform_filter(y, size=size, mode="nearest")
    mean_sq = uniform_filter(y * y, size=size, mode="nearest")
    var = np.clip(mean_sq - mean * mean, 0, None)
    a = var / (var + eps)
    b = mean - a * mean
    return (uniform_filter(a, size=size, mode="nearest") * y
            + uniform_filter(b, size=size, mode="nearest"))


def tone(y, ev=0.0, detail=0.0, gamma=1.6):
    """A normalised preview, as a screen should show it.

    Two knobs, and both are **about the screen and not about the data**:
    nothing here reaches the document, the measurement or the output.

    **Exposure** is a gain before the curve, for a frame that sits four stops
    under its own rebate.

    **Local contrast** splits the picture into a *base* -- the slow falloff
    across the frame, which is most of the range and none of the information
    -- and the *detail* left over. It then flattens the base and lifts the
    detail. One knob moves both, because half the gesture is useless on its
    own: flattening alone gives a grey picture with no structure, lifting
    alone gives structure too dark to read.

    Here it earns its place on the **frame edge**, which is what a person is
    hunting: on bare film the step from rebate to picture is 0.24 in density
    against 2.01 on a masked holder, and at the low end it is a boundary you
    can measure and barely see.

    The split is done **after** the display curve and the clip to white
    **after that**, which is the one thing worth being careful about. Clipping first leaves the detail exactly
    zero wherever the frame is blown, so local contrast has nothing to lift in
    the very region the exposure was raised to see.

    At `ev=0, detail=0` this is exactly the curve the editor always had, so
    the plain view is not quietly a new picture.
    """
    if ev:
        y = y * (2.0 ** ev)
    y = np.clip(y, 0, None) ** (1 / gamma)
    if detail:
        r = max(3, int(min(y.shape[:2]) / 20))
        base = _base(y, r)
        fine = y - base
        anchor = float(np.median(base))
        compress = 0.85 * detail
        y = (anchor + (base - anchor) * (1.0 - 0.85 * compress)
             + fine * (1.0 + 4.0 * detail))
    return np.clip(y, 0, 1)


# What `preview` bins a scan down to, and so the width every way of looking
# at a scan shares. The canvas draws the frames from one scale, so two
# pictures of one scan at two sizes put every rectangle in the wrong place.
PREVIEW_WIDTH = 1600


def preview(path, width=PREVIEW_WIDTH, region=None):
    """(RGB float32 in 0..1, scale) -- the scan as a positive, for looking at.

    A plain per-channel stretch of the density, which is a preview and not a
    colour decision: brightness in the scene is density on the negative, so
    the positive is the density, stretched. No mask subtraction, no stock, no
    curve. Anything that claims to be an inversion belongs in a later step and
    says so.

    `region` is where the stretch is measured, in scan pixels, and the caller
    passes the frame it already found. Measuring it over the whole scan
    instead puts the holder and the bare light table in the percentiles, and
    on a scan of bare film that leaves the picture occupying a fifth of the
    range: pale, blue, and impossible to judge a crop on.
    """
    with rawpy.imread(str(path)) as r:
        # Before postprocess, and not after: `postprocess(half_size=True)`
        # leaves LibRaw reporting the *half* size from `raw_image_visible`,
        # so reading it afterwards silently halves the scale and everything
        # drawn on the canvas comes out at twice its size.
        full_w = r.raw_image_visible.shape[1]
        black, white = _levels(r)
        rgb = r.postprocess(half_size=True, output_bps=16, gamma=(1, 1),
                            no_auto_bright=True, use_camera_wb=False,
                            user_wb=[1, 1, 1, 1])
    t = np.clip(rgb.astype(np.float32) / 65535.0, 1e-4, None)
    k = max(1, round(t.shape[1] / width))
    # Per channel, so a colour negative's orange mask does not tint the whole
    # preview towards blue. The percentiles are of the whole scan including
    # the holder, which is what makes a masked scan look flat -- a known
    # limitation, and the reason `hold r` shows the scan as it came.
    chans = []
    for c in range(3):
        ch = -np.log10(_bin(t[..., c], k))
        if region:
            s = full_w / ch.shape[1]
            x0, y0, x1, y1 = (int(v / s) for v in region)
            sample = ch[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
        else:
            sample = ch
        lo, hi = np.percentile(sample, (2, 98))
        chans.append(np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1))
    # Returned **before** the display curve, so the two view knobs can put
    # their gain in front of it. `tone()` is what turns this into pixels.
    return np.stack(chans, -1).astype(np.float32), full_w / len(chans[0][0])


# A ceiling is a pile-up: at saturation thousands of photosites read the
# same top value, and an unclipped scan's maximum is a handful. On the
# Nikon the pile is at 16350 while the file declares 16383 as its white
# (nikon_55mm: 0.5-1.2 % of a channel at exactly 16350, none above); the
# Sigma fp piles at the declared 16383 (portra400 001: 0.02 %); a scan
# that clips nowhere tops out at one or two photosites (roma-half SDIM0903).
CLIP_PILE = 50            # photosites at the top value that make it a ceiling


def clip_level(v, white):
    """The level at which this scan's sensor saturated, or None if nothing
    in it did. `v` is the raw mosaic, `white` the file's declared white.

    The declared white is not where every camera clips -- the Nikon
    D5600's lossy NEF curve ends 33 levels under it -- so the scan says
    where its own ceiling is: its top value, if enough photosites sit on
    it to be a pile and not a peak, or the declared white if reached."""
    top = int(v.max())
    if top >= white:
        return int(white)
    if top >= 0.5 * white and int((v == top).sum()) >= CLIP_PILE:
        return top
    return None


def clipped_blocks(path, width, height):
    """Where the raw clipped, in any channel, on a picture of the scan
    `width` x `height` pixels -- the scan view's own size.

    Returns (mask, level, share): `mask` is True for each pixel whose block
    of photosites holds at least one at the ceiling -- a pixel of the scan
    view is every k-th of a half-size read, so it stands for a `2k x 2k`
    block -- `level` the ceiling (None if there is none) and `share` the
    fraction of the scan's photosites on it.

    Read off the mosaic and not the demosaiced picture: LibRaw's output
    scales and interpolates, and a photosite at the ceiling is the only
    statement about clipping the sensor makes."""
    with rawpy.imread(str(path)) as r:
        v = r.raw_image_visible
        white = int(r.white_level)
        level = clip_level(v, white)
        if level is None:
            return np.zeros((height, width), bool), None, 0.0
        b = 2 * max(1, round(v.shape[1] / 2 / width))
        h, w = v.shape[0] // b, v.shape[1] // b
        hit = v[:h * b, :w * b] >= level
        share = float(hit.mean())
    blocks = hit.reshape(h, b, w, b).any(axis=(1, 3))
    out = np.zeros((height, width), bool)
    out[:min(h, height), :min(w, width)] = blocks[:height, :width]
    return out, level, share
