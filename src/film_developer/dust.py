"""Dust marked by hand, taken out of the developed picture.

The marks are points, in scan pixels, kept in the document's `specks`
(`s` in the window puts them there). A point says *there is a speck here*
and not how big it is, so each one is grown into the speck it sits on
before anything is filled:

- **What a speck is, on the positive**: brighter than its surroundings in
  all three channels at once. Dust is opaque and neutral, so it adds the
  same density to every layer of the film and comes out white; grain and
  colour detail rise in one channel and not the others. The residual is
  the **minimum over the channels** of each one's rise above its local
  median, relative to that median -- a neutral cue, in the positive.
- **Seeded at the brightest such pixel near the click**, within `SNAP`
  scan pixels: a click at 100 % lands on the speck or next to it.
- **Grown** to the connected pixels above a fraction of the seed's height
  and above the window's own noise; capped at `AREA` so a click on a
  bright patch of the scene cannot take a wall with it.
- **Or a blur, not a speck.** Dust on the rig -- the sensor, the lens,
  the panel -- is out of focus: a soft disc 30-60 px across, a few per cent
  brighter than the sky round it (25 of the 58 marked on DSC_0238). The
  speck test cannot see it, its median is smaller than the disc. So when
  it finds nothing it looks again at a quarter of the size against a wider
  median, and a disc found that way is **divided out**, not filled: it is
  a shadow over the picture, a falloff on a small scale, and dividing by
  it leaves the grain and the detail under it where they were.
- **Nothing found under the click** fills a disc of `RADIUS`. Somebody
  looked at the frame at 100 % and said there was dust there; that is
  what is done.

A speck is filled by normalised convolution from the surroundings, then
given back grain drawn from a ring round the hole, so a filled speck does
not read as a smooth spot in a grainy sky. The grain is the ring's
residual from its own smooth level, clipped to its robust spread: next to
an edge of the scene the ring holds the edge, and drawn as grain that
came out as a dark checker over the fill.

**It is applied to the develop only**: the window's develop and `⌘R`'s
pictures. The scan, the DNG positive (which is still raw) and the
document's marks are left alone, so a mark taken off undoes the cleaning.
The view that marks dust is not cleaned either: it is where the specks
have to be seen.
"""

import math

import numpy as np
from scipy import ndimage

SNAP = 6.0       # scan px: how far from the click the speck's peak may be
RADIUS = 5.0     # scan px: the disc filled when no speck is found
WINDOW = 48.0    # scan px: half the side of what is looked at round a click
MEDIAN = 25.0    # scan px: the local median's size; bigger than any speck
AREA = 2500.0    # scan px²: the most one click can take
SEED_K = 4.0     # the seed's height, in the window's robust sigmas, to count
GROW = 0.20      # a pixel joins the speck above this fraction of the seed's height
GROW_K = 3.0     # and above this many robust sigmas, both on the channels' mean
BLUR_DOWN = 8.0  # scan px per pixel the soft disc is looked for at
BLUR_WINDOW = 96.0   # scan px: half the side looked at for one
BLUR_MEDIAN = 121.0  # scan px: the wider median, bigger than the disc
BLUR_SNAP = 16.0     # scan px: the disc's peak this far from the click
BLUR_K = 2.0         # its height, in robust sigmas of the blurred residual: low,
                     # because a hand mark is already the evidence; 2-4 % on DSC_0238
BLUR_SMOOTH = 6.0    # scan px: the blur that takes the grain out before looking
BLUR_AREA = 20000.0  # scan px²: the most a disc can be
SIZED_REACH = 2.0    # a sized mark's speck stays within this many of its radii
STROKE_R = 4.0       # scan px: a stroke's radius when none was given
CLOSE = 3.0          # scan px: a gap this narrow between parts of a hole is filled too
ISLAND = 400.0       # scan px²: the most picture a speck can close round and take


def samples(mark):
    """A mark as points in scan pixels: (x, y, r, painted).

    A click is one point. **A stroke** -- a scratch or a hair painted by
    dragging, `path` with a radius `r` -- is its path walked every half
    radius, so the discs overlap into the band that was painted."""
    path = mark.get("path")
    if not path:
        return [(mark["x"], mark["y"], mark.get("r"), False)]
    r = float(mark.get("r") or STROKE_R)
    if mark.get("soft"):
        return [(float(path[0][0]), float(path[0][1]), r, "soft")]
    # **Cloned from a place a person chose** (`from`, scan pixels from the
    # mark to its source): every point of the mark says so, and which mark
    # it is, so `clean` fills the mark from there as one piece
    paint = (("from", float(mark["from"][0]), float(mark["from"][1]),
              mark.get("id") or id(mark)) if mark.get("from") else True)
    step = max(1.0, r / 2)
    out = [(float(path[0][0]), float(path[0][1]), r, paint)]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / step)))
        out += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, r, paint)
                for i in range(1, n + 1)]
    return out


def points(marks, to_px):
    """The marks through `to_px(x, y) -> (x, y)`, as (x, y, r, painted):
    `r` is the radius a person gave the mark, in scan pixels, or None to
    find the speck's own extent; `painted` for a stroke's points, whose
    disc is filled as it is (`samples`)."""
    out = []
    for m in marks:
        for x, y, r, painted in samples(m):
            at = tuple(to_px(x, y))
            if isinstance(painted, tuple):
                # the way to the source, turned and mirrored as the point is
                to = to_px(x + painted[1], y + painted[2])
                painted = ("from", float(to[0] - at[0]), float(to[1] - at[1]),
                           painted[3])
            out.append(at + (r, painted))
    return out


def clean(img, pts, scan_px, seed=0):
    """`img` (H, W, 3, linear, any scale) with the specks under `pts`
    (image pixels) filled. `scan_px` is how many scan pixels one image
    pixel spans, so the sizes above mean the same on film at any scale.
    Returns a new array, or `img` itself when there is nothing to do."""
    H, W = img.shape[:2]
    pts = [tuple(p) + (None, False)[len(p) - 2:] for p in pts]
    # a painted disc counts if any of it is on the picture, not only its
    # middle: the window's live view cleans crops of the frame, and a dab
    # just outside one is still part of what the frame would fill
    reach = lambda p: 2 + (p[2] / scan_px if p[3] and p[2] else 0)
    pts = [p for p in pts if -reach(p) <= p[0] < W + reach(p)
           and -reach(p) <= p[1] < H + reach(p)]
    if not pts:
        return img
    out = img.copy()
    hole = np.zeros((H, W), bool)
    cloned = {}                 # mark -> (its mask, the way to its source)
    for x, y, r, painted in pts:
        if isinstance(painted, tuple):
            rr = max(1.0, r / scan_px)
            y0, y1 = max(0, int(y - rr - 1)), min(H, int(y + rr + 2))
            x0, x1 = max(0, int(x - rr - 1)), min(W, int(x + rr + 2))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            if painted[3] not in cloned:
                cloned[painted[3]] = (np.zeros((H, W), bool),
                                      painted[1] / scan_px, painted[2] / scan_px)
            cloned[painted[3]][0][y0:y1, x0:x1] |= (xx - x) ** 2 + (yy - y) ** 2 <= rr * rr
            continue
        if painted == "soft":
            # a soft disc of the rig's dust, brought over from a mark made
            # before the brush (`migrate`): divided out as it was then. Not
            # found -- a neighbour's division can have taken it, the four
            # top left of DSC_0238 overlap -- it is what an old mark was
            # then, a speck or a small disc, and never its whole radius
            # filled: on DSC_0238 that was 77 000 px of sky made up
            if not _blur(out, x, y, scan_px):
                _speck(out, hole, x, y, scan_px, True)
            continue
        if painted:
            # **A stroke is what was painted**, nothing grown: a scratch is
            # a line a person has just drawn along, and growing a speck
            # every half radius of it would follow whatever line of the
            # picture it crosses -- the wire of DSC_0427 again
            rr = max(1.0, r / scan_px)
            y0, y1 = max(0, int(y - rr - 1)), min(H, int(y + rr + 2))
            x0, x1 = max(0, int(x - rr - 1)), min(W, int(x + rr + 2))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            hole[y0:y1, x0:x1] |= (xx - x) ** 2 + (yy - y) ** 2 <= rr * rr
            continue
        if r:
            # **Sized by hand: the disc, and the speck under it too.** The
            # disc is what somebody drew round; the speck is grown as for
            # any mark and added, because a hair is not a disc -- on
            # DSC_0238, #89 and #90 were two small discs on a curved
            # fibre whose tip stayed out of both. No dividing: a disc
            # drawn by hand is to be filled.
            rr = max(1.0, r / scan_px)
            y0, y1 = max(0, int(y - rr - 1)), min(H, int(y + rr + 2))
            x0, x1 = max(0, int(x - rr - 1)), min(W, int(x + rr + 2))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            hole[y0:y1, x0:x1] |= (xx - x) ** 2 + (yy - y) ** 2 <= rr * rr
            _speck(out, hole, x, y, scan_px, False, SIZED_REACH * rr)
            continue
        if not _speck(out, hole, x, y, scan_px, False):
            if not _blur(out, x, y, scan_px):
                _speck(out, hole, x, y, scan_px, True)
    # **What a speck closes round is the speck's.** A hair that curls into
    # a loop leaves the pixels inside it out of every mark, and they carry
    # the hair's glow: on DSC_0427 of carrete_4 the loop at the tip of a
    # hair sat over a pair of wires and came out as a light spot on them,
    # with the hair round it gone. So a gap narrower than `CLOSE` between
    # two parts of the hole goes in (a closing), and then a gap the hole
    # encloses and no bigger than `ISLAND`: that loop was open on one side,
    # 2-3 px, and a closing alone left a diamond inside it.
    #
    # Round each speck in its own box, not over the frame: on 24 MP the two
    # took 0.7 s for a single 5 px dab. The box has `2k + 2` round it, so
    # the closing sees everything it would over the frame, and an island
    # is enclosed by the hole, so it is inside the box.
    k = max(1, int(round(CLOSE / scan_px)))
    pad = 2 * k + 2
    lab, _ = ndimage.label(hole)
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        sl = tuple(slice(max(0, s.start - pad), min(hole.shape[a], s.stop + pad))
                   for a, s in enumerate(sl))
        h = hole[sl]
        h = h | ndimage.binary_closing(h, iterations=k)
        shut = ndimage.binary_fill_holes(h) & ~h
        if shut.any():
            il, n = ndimage.label(shut)
            area = ndimage.sum(shut, il, np.arange(1, n + 1)) * scan_px * scan_px
            h = h | np.isin(il, 1 + np.flatnonzero(area <= ISLAND))
        hole[sl] |= h
    if not cloned:
        return fill(out, hole, np.random.default_rng(seed))
    # The marks with a chosen source go last, in the order they are in, each
    # copied from where it was told and healed to its edge (`clone`): what
    # the automatic did not get right is what they are for, so they are not
    # a place the automatic may copy from either.
    bad = np.zeros((H, W), bool)
    for m, _dx, _dy in cloned.values():
        bad |= m
    out = fill(out, hole & ~bad, np.random.default_rng(seed), bad)
    for m, dx, dy in cloned.values():
        clone(out, m, int(round(dy)), int(round(dx)))
    return out


def clone(img, h, dy, dx):
    """`img` under `h` copied from `(dy, dx)` away and healed to the edge
    round it (`_heal`), in place: the clone stamp with a source a person
    chose. The part of `h` whose source is off the picture is left."""
    H, W = img.shape[:2]
    ys, xs = np.nonzero(h)
    keep = (ys + dy >= 0) & (ys + dy < H) & (xs + dx >= 0) & (xs + dx < W)
    if not keep.any():
        return
    ys, xs = ys[keep], xs[keep]
    pad = 6
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    sub = img[y0:y1, x0:x1]
    hh = np.zeros(sub.shape[:2], bool)
    hh[ys - y0, xs - x0] = True
    ey, ex = np.nonzero(_edge(hh))
    ey_s = np.clip(ey + y0 + dy, 0, H - 1)
    ex_s = np.clip(ex + x0 + dx, 0, W - 1)
    hy, hx = np.nonzero(hh)
    patch = img[hy + y0 + dy, hx + x0 + dx].copy()
    sub[hy, hx] = _heal(sub, hh, patch, hy, hx, img[ey_s, ex_s],
                        (~hh).astype(np.float32))


def clean_mosaic(mos, pts, seed=0):
    """`clean` on a Bayer mosaic, before any demosaic: what the positive
    DNG gets when *the dust marks in the raws too* is ticked.

    `mos` is (H, W), one value per photosite; `pts` are (x, y, r) in its
    own photosite coordinates. The mosaic is split into its four
    sub-planes -- R, G, G, B, each a quarter of the photosites on a grid of
    its own -- and they go through `clean` together as one four-channel
    picture at half the size, so one plane pixel spans two photosites
    (`scan_px` 2). Together, not one by one, for two reasons: a speck is
    found by rising in every channel at once, which is what keeps a
    coloured detail of the scene out; and a hole is filled with **one**
    patch, copied at the same shift in all four planes, so the photosites
    put back still belong to one place on the film and the mosaic still
    demosaics as a mosaic. Filled plane by plane, each would pick its own
    patch and the four colours of one photosite block would come from four
    places.

    The mosaic is only ever shifted on its own grid, as the crop is:
    nothing is interpolated between colours. Returns a
    new array, or `mos` itself when there is nothing to do."""
    H, W = mos.shape
    H2, W2 = H - H % 2, W - W % 2
    pq = [((p[0] - 0.5) / 2, (p[1] - 0.5) / 2) + tuple(p[2:]) for p in pts]
    blocks = [(dy, dx) for dy in (0, 1) for dx in (0, 1)]
    Q = np.stack([mos[dy:H2:2, dx:W2:2] for dy, dx in blocks], -1)
    C = clean(Q, pq, 2.0, seed)
    if C is Q:
        return mos
    out = mos.copy()
    for i, (dy, dx) in enumerate(blocks):
        out[dy:H2:2, dx:W2:2] = C[..., i]
    return out


LONG = 2.5      # a grown speck this many times longer than wide is a hair
DAB_KEEP = 90   # %: of a grown speck inside the dab made from it


def migrate(img, x, y, r, scan_px):
    """An old mark -- a point grown into the speck under it -- as the brush
    would paint it: what `clean` would take out for it, measured once on
    `img` and written down. Image pixels in, image pixels out.

    Returns one of
      ("dab", x, y, r)       a round speck: the disc round what was grown;
      ("stroke", path, r)    a hair: its middle line, and half its width;
      ("soft", x, y, r)      a soft disc that `clean` divides out, not fills:
                             a dab marked `soft`, so it still is (`samples`).

    **Measured, not guessed from the mark**: a point without a size meant
    *whatever speck is under here*, and only the picture says how big that
    was. With a size it was that disc and the speck under it up to
    `SIZED_REACH` radii, so the same rule takes both."""
    H, W = img.shape[:2]
    hole = np.zeros((H, W), bool)
    if r:
        rr = max(1.0, r / scan_px)
        y0, y1 = max(0, int(y - rr - 1)), min(H, int(y + rr + 2))
        x0, x1 = max(0, int(x - rr - 1)), min(W, int(x + rr + 2))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        hole[y0:y1, x0:x1] |= (xx - x) ** 2 + (yy - y) ** 2 <= rr * rr
        _speck(img, hole, x, y, scan_px, False, SIZED_REACH * rr)
    elif not _speck(img, hole, x, y, scan_px, False):
        probe = img.copy()
        if _blur(probe, x, y, scan_px):
            # what the division moved by more than half a percent: its
            # gain is feathered and resampled, so it is nowhere exactly 1
            d = (np.abs(probe - img) / np.maximum(img, 1e-4)).max(axis=2) > 0.005
            ys, xs = np.nonzero(d)
            if len(ys):
                # kept at the click: `clean` looks for the disc from there
                # again, and from the disc's middle it can find another
                rad = np.sqrt(((ys - y) ** 2 + (xs - x) ** 2).max()) + 1
                return ("soft", float(x), float(y), float(rad))
            return ("soft", float(x), float(y), 5.0 / scan_px)
        _speck(img, hole, x, y, scan_px, True)
    ys, xs = np.nonzero(hole)
    if not len(ys):
        return ("dab", float(x), float(y), max(1.0, 5.0 / scan_px))
    pts = np.stack([xs, ys], 1).astype(np.float64) + 0.0
    c = pts.mean(0)
    w, v = np.linalg.eigh(np.cov((pts - c).T) + 1e-9 * np.eye(2))
    axis, across = v[:, 1], v[:, 0]
    if np.sqrt(w[1] / max(w[0], 1e-9)) < LONG:
        # round the speck's body, not its furthest pixel: a disc out to
        # the one stray pixel grown along a branch took the branch, and
        # copied it into the sky (#19 on DSC_0238)
        rad = np.percentile(np.sqrt(((pts - c) ** 2).sum(1)), DAB_KEEP) + 1.0
        return ("dab", float(c[0]), float(c[1]), float(rad))
    # a hair: its middle line, a point every few pixels along it, and the
    # band wide enough to take every pixel that was grown
    t = (pts - c) @ axis
    u = (pts - c) @ across
    step = max(2.0, np.sqrt(w[0]) * 2)
    edges = np.arange(t.min(), t.max() + step, step)
    path, half = [], 0.5
    for lo in edges:
        sel = (t >= lo) & (t < lo + step)
        if sel.any():
            mid = u[sel].mean()
            half = max(half, np.abs(u[sel] - mid).max() + 0.5)
            p = c + axis * t[sel].mean() + across * mid
            path.append((float(p[0]), float(p[1])))
    if len(path) < 2:
        rad = np.sqrt(((pts - c) ** 2).sum(1).max()) + 0.5
        return ("dab", float(c[0]), float(c[1]), float(rad))
    # the ends are the ends of the hair, not the middles of the last bins
    path[0] = tuple(c + axis * t.min() + across * u[t < t.min() + step].mean())
    path[-1] = tuple(c + axis * t.max() + across * u[t > t.max() - step].mean())
    return ("stroke", [(float(a), float(b)) for a, b in path], float(half))


def chains(pts):
    """Old marks drawn as a chain of discs along one hair, each chain in
    the order it runs: lists of indices into `pts`, (x, y, r) in scan
    pixels. Two discs are linked when they touch. Before the brush a hair
    was marked that way -- 12 discs along the one on DSC_0427 -- and made
    brush marks one by one it would stay a chain of dabs, where a person
    now paints one stroke."""
    n = len(pts)
    seen, out = set(), []
    for s in range(n):
        if s in seen:
            continue
        group, todo = [], [s]
        seen.add(s)
        while todo:
            i = todo.pop()
            group.append(i)
            for j in range(n):
                if j not in seen and math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) \
                        <= pts[i][2] + pts[j][2]:
                    seen.add(j)
                    todo.append(j)
        if len(group) < 2:
            continue
        # in order along it: from the disc furthest from the middle, then
        # always the nearest one not yet taken
        cx = sum(pts[i][0] for i in group) / len(group)
        cy = sum(pts[i][1] for i in group) / len(group)
        at = max(group, key=lambda i: math.hypot(pts[i][0] - cx, pts[i][1] - cy))
        order, left = [at], set(group) - {at}
        while left:
            at = min(left, key=lambda i: math.hypot(pts[i][0] - pts[order[-1]][0],
                                                    pts[i][1] - pts[order[-1]][1]))
            order.append(at)
            left.remove(at)
        out.append(order)
    return out


def _speck(img, hole, x, y, k, anyway, reach=None):
    """Grow the speck under (x, y) into `hole`; a disc if `anyway` and
    none is found. True if something went into `hole`.

    `reach` (image px) keeps it that close to the mark. **A sized mark
    needs it**: a hair is marked as a chain of discs, and where it crosses
    a line of the scene the line can stand out more than the hair -- on
    DSC_0427 of carrete_4 a white hair over a dark wire, and the two marks
    on the crossing grew as dark specks 96 px along the wire, which the
    fill then wiped out. Kept within two of the mark's radii they take 24."""
    H, W = img.shape[:2]
    half = max(4, int(math.ceil(WINDOW / k)))
    cx, cy = int(round(x)), int(round(y))
    y0, y1 = max(0, cy - half), min(H, cy + half + 1)
    x0, x1 = max(0, cx - half), min(W, cx + half + 1)
    if y1 - y0 < 3 or x1 - x0 < 3:
        return False
    sub = img[y0:y1, x0:x1].astype(np.float32)
    size = max(3, int(MEDIAN / k) | 1)
    if size >= 9:
        # a level to rise above, not a picture: taken at half size it is
        # the same level and costs a sixteenth -- at full size the median
        # was 9 s of the 9.2 s that 58 marks took
        h2 = ndimage.median_filter(sub[::2, ::2], size=(size // 2 | 1,) * 2 + (1,),
                                   mode="nearest")
        med = np.repeat(np.repeat(h2, 2, 0), 2, 1)[:sub.shape[0], :sub.shape[1]]
    else:
        med = ndimage.median_filter(sub, size=(size, size, 1), mode="nearest")
    rel = (sub - med) / np.maximum(med, 1e-4)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r2 = (xx - x) ** 2 + (yy - y) ** 2
    near = r2 <= max(1.5, SNAP / k) ** 2
    if not near.any():
        return False
    # Brighter in all three channels, or darker in all three: whichever
    # stands further out of the window's noise. Most marked dust is white on
    # the positive; some is dark -- a fibre on the emulsion side, a scratch
    # that let light through -- and a mark says only *here*.
    best = None
    for sign in (1, -1):
        rise = (sign * rel).min(axis=2)
        sigma = 1.4826 * np.median(np.abs(rise - np.median(rise))) + 1e-6
        peak = np.where(near, rise, -np.inf)
        sy, sx = np.unravel_index(np.argmax(peak), peak.shape)
        if best is None or rise[sy, sx] / sigma > best[0]:
            best = (rise[sy, sx] / sigma, rise, sigma, sy, sx, sign)
    _, rise, sigma, sy, sx, sign = best
    height = rise[sy, sx]
    if height >= SEED_K * sigma:
        # **Seeded on all three channels, grown on their mean.** The seed
        # has to rise in every channel, which is what keeps a coloured
        # detail of the scene from being taken for dust. But a fibre is
        # not grey all along: #89 on DSC_0238, an orange hair, rose 5.7 in
        # red and 1.8 in blue at its core, and grown on the weakest
        # channel it stopped at 36 px with its tip outside. On the mean,
        # down to a fifth of the peak, it is 49 px. A tenth took the whole
        # hair and also took #8 along the bright ledge it sat on, 50 px of
        # the picture: a hair longer than that is for a sized mark.
        g = (sign * rel).mean(axis=2)
        gs = 1.4826 * np.median(np.abs(g - np.median(g))) + 1e-6
        grown, n = ndimage.label(g > max(GROW * g[sy, sx], GROW_K * gs))
        speck = grown == grown[sy, sx]
        if reach is not None:
            speck &= r2 <= reach * reach
        if speck.sum() * k * k > AREA:
            if not anyway:
                return False
            speck = r2 <= max(1.0, RADIUS / k) ** 2
    elif anyway:
        speck = r2 <= max(1.0, RADIUS / k) ** 2
    else:
        return False
    # a pixel more all round: the speck's own soft edge, which the threshold
    # leaves outside and which would come back as a halo
    speck = ndimage.binary_dilation(speck, iterations=max(1, int(round(2 / k))))
    hole[y0:y1, x0:x1] |= speck
    return True


def _blur(img, x, y, k):
    """Divide out the soft disc under (x, y), in place. True if found."""
    H, W = img.shape[:2]
    half = max(8, int(math.ceil(BLUR_WINDOW / k)))
    cx, cy = int(round(x)), int(round(y))
    y0, y1 = max(0, cy - half), min(H, cy + half + 1)
    x0, x1 = max(0, cx - half), min(W, cx + half + 1)
    sub = img[y0:y1, x0:x1].astype(np.float32)
    f = max(1.0, BLUR_DOWN / k)          # this window's pixels per small pixel
    if min(sub.shape[:2]) / f < 8:
        return False
    # blurred well past the grain: the disc is tens of pixels, the grain one
    small = ndimage.zoom(ndimage.gaussian_filter(sub, (BLUR_SMOOTH / k,) * 2 + (0,)),
                         (1 / f, 1 / f, 1), order=1)
    size = max(3, int(BLUR_MEDIAN / (k * f)) | 1)
    bg = ndimage.median_filter(small, size=(size, size, 1), mode="nearest")
    ratio = small / np.maximum(bg, 1e-4)
    rise = (ratio - 1).min(axis=2)
    sigma = 1.4826 * np.median(np.abs(rise - np.median(rise))) + 1e-6
    sh, sw = rise.shape
    yy, xx = np.mgrid[:sh, :sw]
    lx, ly = (x - x0) / f, (y - y0) / f
    near = (xx - lx) ** 2 + (yy - ly) ** 2 <= (BLUR_SNAP / (k * f)) ** 2
    if not near.any():
        return False
    sy, sx = np.unravel_index(np.argmax(np.where(near, rise, -np.inf)), rise.shape)
    height = rise[sy, sx]
    if height < BLUR_K * sigma:
        return False
    lab, _ = ndimage.label(rise > max(0.5 * height, 2 * sigma))
    disc = lab == lab[sy, sx]
    if disc.sum() * (k * f) ** 2 > BLUR_AREA:
        return False
    # feathered, so the division fades out instead of drawing the disc's edge
    grow = ndimage.binary_dilation(disc, iterations=2)
    weight = ndimage.gaussian_filter(grow.astype(np.float32), 1.0)
    gain = 1 + (ratio - 1) * weight[..., None]
    gain = ndimage.zoom(gain, (sub.shape[0] / sh, sub.shape[1] / sw, 1), order=1)
    img[y0:y1, x0:x1] = sub / np.maximum(gain, 1e-3)
    return True


COPY_REACH = 3.0     # how far to look for a patch to copy, in the hole's sizes
COPY_MIN = 16        # px: and at least this far
HEAL_SWEEPS = 200    # Jacobi sweeps of the healing's harmonic fill


def _edge(h):
    """The pixels just outside the hole."""
    return ndimage.binary_dilation(h) & ~h


def _heal(sub, h, patch, ys, xs, patch_edge, valid):
    """The copied patch, its level bent to meet the hole's edge.

    **The healing brush.** A patch from beside the hole carries the right
    grain and the right detail and, between two different things, the
    wrong level: #89 on DSC_0238 sits between a dark window and a lit
    wall, and the best single patch came out a dark smudge on the wall's
    side. So the difference between the hole's edge and the patch's edge
    is spread smoothly over the hole -- a harmonic fill, Laplace's
    equation with the edge as its boundary, by Jacobi iteration -- and
    added. Where the patch already matched, the difference is grain and
    nearly zero; where it did not, the window's side is pulled dark and the
    wall's side light. Perez, Gangnet and Blake (2003)."""
    # worked on the hole's own box: the patch search wanted room round it,
    # this does not, and it was half the time of cleaning a frame
    y0, x0 = max(0, ys.min() - 4), max(0, xs.min() - 4)
    box = (slice(y0, ys.max() + 5), slice(x0, xs.max() + 5))
    # only the edge that is picture: beside a piece of a hole not yet
    # filled (`_fill_pieces`) the edge is still the speck
    edge = _edge(h)
    ey, ex = np.nonzero(edge)
    on = valid[ey, ex] > 0
    full_edge = (sub[ey, ex] - patch_edge)[on]
    ey, ex = ey[on], ex[on]
    edge = np.zeros_like(edge)
    edge[ey, ex] = True
    sub, h, edge = sub[box], h[box], edge[box]
    ys, xs = ys - y0, xs - x0
    D = np.zeros(sub.shape, np.float64)
    D[ey - y0, ex - x0] = full_edge
    # the edge's difference is itself grain: smoothed along the edge first,
    # or each pixel of it would be carried in as a streak
    e = edge.astype(np.float64)
    D = ndimage.gaussian_filter(D * e[..., None], (1.5, 1.5, 0)) / \
        np.maximum(ndimage.gaussian_filter(e, 1.5), 1e-9)[..., None] * e[..., None]
    # started from a blur of the edge, so a few hundred sweeps settle it
    # whatever the hole's size (a 60 px disc would need thousands from zero)
    w = ndimage.gaussian_filter(e, 4)
    D[h] = (ndimage.gaussian_filter(D, (4, 4, 0)) / np.maximum(w, 1e-9)[..., None])[h]
    for _ in range(HEAL_SWEEPS):
        avg = (np.roll(D, 1, 0) + np.roll(D, -1, 0) + np.roll(D, 1, 1) + np.roll(D, -1, 1)) / 4
        D[h] = avg[h]
    return patch + D[ys, xs].astype(sub.dtype)


def _copy(sub, h, valid):
    """The hole filled from the patch beside it that best continues it,
    healed to meet its edge (`_heal`).

    **The clone stamp, chosen by the pixels round the hole.** Every shift
    within reach is tried; the one whose surroundings best match the
    hole's own surroundings -- least squared difference over a ring three
    pixels wide -- is copied in, grain and all. Across an edge of the scene
    the best shift runs along the edge, so the edge goes on through the
    hole; the blur it replaces could only average the two sides of it, and
    a speck on the line of a roof came out as a grey smudge on it.

    Done for every shift at once by correlation (FFT): the ring's squared
    difference expands into two correlations and a constant. Returns the
    hole's pixels, or None when no shift has the hole and its ring all on
    valid pixels (a hole against the edge of the picture)."""
    from scipy.signal import fftconvolve
    ring = ndimage.binary_dilation(h, iterations=3) & (valid > 0)
    need = (ring | h).astype(np.float32)
    M = ring.astype(np.float32)
    H, W = h.shape

    def corr(a, k):
        # sum over p of k(p) a(p + o), for o from -(H-1, W-1) to (H-1, W-1)
        return fftconvolve(a, k[::-1, ::-1], mode="full")

    ok = corr(valid.astype(np.float32), need) > need.sum() - 0.5
    ssd = np.zeros(ok.shape, np.float64)
    for c in range(sub.shape[2]):
        a = sub[..., c].astype(np.float64)
        ssd += corr(a * a, M) - 2 * corr(a, M * a) + (M * a * a).sum()
    ys, xs = np.nonzero(h)
    reach = max(COPY_MIN, COPY_REACH * (max(np.ptp(ys), np.ptp(xs)) + 1))
    oy, ox = np.mgrid[-(H - 1):H, -(W - 1):W]
    ok &= (oy ** 2 + ox ** 2) <= reach ** 2
    if not ok.any():
        return None
    i = int(np.argmin(np.where(ok, ssd, np.inf)))
    dy, dx = int(oy.flat[i]), int(ox.flat[i])
    ey, ex = np.nonzero(_edge(h))
    return _heal(sub, h, sub[ys + dy, xs + dx], ys, xs, sub[ey + dy, ex + dx],
                 valid)


PIECE = 32           # px: a hole longer than this is filled in pieces this size


def _pieces(h):
    """The hole cut on a grid of `PIECE`, each cell's part of it a piece;
    the hole itself when it fits in one."""
    ys, xs = np.nonzero(h)
    if max(np.ptp(ys), np.ptp(xs)) < PIECE:
        return [h]
    # **On the hole's own grid**, from its top left corner, not the
    # window's: the window is the hole plus whatever room the picture
    # round it allows, so the same hole cut from another crop -- the
    # window's live view cleans only round what changed -- was cut in
    # other pieces and filled differently, up to 57 levels on DSC_0238
    cy = (np.arange(h.shape[0]) - ys.min()) // PIECE
    cx = (np.arange(h.shape[1]) - xs.min()) // PIECE
    cell = cy[:, None] * (h.shape[1] // PIECE + 2) + cx[None, :]
    out = []
    for c in np.unique(cell[h]):
        lab, n = ndimage.label(h & (cell == c))
        out += [lab == j for j in range(1, n + 1)]
    return out


def _fill_pieces(sub, h, valid):
    """`_copy` piece by piece (`_pieces`), in place. None when all of it
    went in; else what is left, for the fill that needs no patch."""
    pieces = _pieces(h)
    valid = valid.copy()
    # each piece's box and a pixel round it, which holds its whole edge:
    # taken over the hole's window, choosing the next piece dilated that
    # window once per piece left per piece filled, and a dab of 120 px
    # radius (58 pieces) took 39 s, 35 of them here
    H, W = h.shape
    boxes = []
    for p in pieces:
        ys, xs = np.nonzero(p)
        boxes.append((slice(max(0, ys.min() - 1), min(H, ys.max() + 2)),
                      slice(max(0, xs.min() - 1), min(W, xs.max() + 2))))
    while pieces:
        def edge_on(j):
            e = _edge(pieces[j][boxes[j]])
            return valid[boxes[j]][e].mean() if e.any() else 0.0
        k = max(range(len(pieces)), key=edge_on)
        p = pieces.pop(k)
        boxes.pop(k)
        # looked for round the piece, as a hole of its size is (`fill`):
        # over the whole hole's window each piece cost an FFT of it, and
        # the hair on DSC_0427 went from 3.6 s to 13.8 s
        ys, xs = np.nonzero(p)
        pad = int(max(24, COPY_MIN, COPY_REACH * (max(np.ptp(ys), np.ptp(xs)) + 1)) + 6)
        win = (slice(max(0, ys.min() - pad), ys.max() + pad + 1),
               slice(max(0, xs.min() - pad), xs.max() + pad + 1))
        copied = _copy(sub[win], p[win], valid[win])
        if copied is None:
            left = p.copy()
            for q in pieces:
                left |= q
            return left
        sub[win][p[win]] = copied
        valid[p] = 1.0
    return None


def fill(img, hole, rng, bad=None):
    """The best-matching neighbouring patch copied in (`_copy`); where
    there is none, normalised convolution from the surroundings, then
    grain from beside the hole.

    Each hole pixel takes the narrowest blur of the valid pixels that
    reaches it, so the fill keeps what detail the edge of the hole has. The
    grain is the high-pass of a neighbouring patch of the same shape, moved
    over whole: the grain of a film is a few pixels across, and drawn pixel
    by pixel from the ring it came out as white noise -- a checker -- that
    the eye found at once. Of the eight neighbours the one with the least
    high-pass goes, so it is grain and not an edge of the scene; clipped to
    its own robust spread all the same."""
    out = img
    lab, n = ndimage.label(hole)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        # room to look `COPY_REACH` sizes away on every side
        size = max(s.stop - s.start for s in sl)
        pad = int(max(24, COPY_MIN, COPY_REACH * size) + 6)
        sl = tuple(slice(max(0, s.start - pad), min(img.shape[a], s.stop + pad))
                   for a, s in enumerate(sl))
        h = lab[sl] == i
        sub = out[sl]
        valid = (~hole[sl] if bad is None else ~(hole[sl] | bad[sl])).astype(np.float32)
        # One patch, healed (`_copy`, `_heal`). Tried and dropped on
        # DSC_0238: the hole filled in layers from its edge (Criminisi),
        # which drew dark flecks of the trees into a light ledge (#8); and
        # each hole pixel from the locally best of eight shifts, which drew
        # the branch into the sky beside it (#19) and spotted #89.
        #
        # **A long hole goes in pieces** (`PIECE`), each its own patch, the
        # piece with most of its ring on the picture first, and each filled
        # piece then part of the picture for the next. One shift continues
        # one line of the scene: a hair on DSC_0427 of carrete_4, 167 px
        # long, crosses a sloping wire and then a pair of wires at another
        # slope, and the one patch that continued the first smudged the
        # pair where the hair's tip curls over them.
        left = _fill_pieces(sub, h, valid)
        if left is None:
            continue
        valid = np.maximum(valid, (h & ~left).astype(np.float32))
        h = left
        est = sub.copy()
        done = ~h
        for sigma in (1, 2, 4, 8, 16, 32):
            num = ndimage.gaussian_filter(sub * valid[..., None], (sigma, sigma, 0))
            den = ndimage.gaussian_filter(valid, sigma)
            now = ~done & (den > 0.2 if sigma < 32 else den > 0)
            est[now] = num[now] / np.maximum(den[now], 1e-6)[..., None]
            done |= now
            if done.all():
                break
        # the high-pass, from valid pixels only, so the hole's own speck
        # does not ring into it
        lo = (ndimage.gaussian_filter(sub * valid[..., None], (2, 2, 0))
              / np.maximum(ndimage.gaussian_filter(valid, 2), 1e-6)[..., None])
        hp = np.where(valid[..., None] > 0, sub - lo, 0)
        ys, xs = np.nonzero(h)
        span = max(np.ptp(ys), np.ptp(xs)) + 3
        best = None
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            ty, tx = ys + dy * span, xs + dx * span
            if (ty.min() < 0 or tx.min() < 0 or ty.max() >= h.shape[0]
                    or tx.max() >= h.shape[1] or not valid[ty, tx].all()):
                continue
            g = hp[ty, tx]
            v = float((g ** 2).mean())
            if best is None or v < best[0]:
                best = (v, g)
        if best is not None:
            g = best[1]
            mad = 1.4826 * np.median(np.abs(g - np.median(g, 0)), 0) + 1e-6
            est[h] = est[h] + np.clip(g, -2.5 * mad, 2.5 * mad)
        sub[h] = np.clip(est[h], 0, None)
    return out


# ---- finding dust: proposals for a person to accept or refuse
#
# What it finds is
# **proposed**, never used: the window shows it, a person accepts or
# refuses each one, and only the accepted ones are taken out of the
# develop (`used`). Measured on DSC_0238 against 89 hand marks: the rig's
# blanks give 24 specks, 6 of them marked (15 marks); of the other 74
# marks only 12 are among the film detector's 889 candidates at all --
# most of the rest are soft discs it does not look for -- and `PROPOSE`
# puts 38 forward, 9 of them marked. Tuned on that one frame. 8 s, 4 s
# once the blanks are in memory (46 s with the medians at full size).

FIND_MEDIAN = 15            # px of the Bayer plane: twice the largest dust it is meant for
FIND_SEED_K = 6.0           # seed threshold on the neutral residual, in robust sigmas
FIND_GROW_K = 2.5           # and the threshold a seed grows into
FIND_MAX_AREA = 2500        # plane px; larger is the picture, not a speck
NEUTRAL = 0.70         # min/max of the channels' rise for "neutral"
SHARP = 0.45           # steepest gradient / height, per plane px
RIG_SMOOTH = 40        # plane px: what is smooth light falloff, and what is a speck
RIG_AREA = 12         # plane px: the smallest rig speck kept
RIG_AREA_MAX = 600     # plane px: larger is the holder or the film edge, not dust
RIG_BORDER = 80        # plane px of the blanks' border that are not trusted
RIG_DIP = 0.98         # a blank darker than this against its surroundings has a speck
AREA_MAX = 60          # plane px: film dust on this frame is 4-30
ENCLOSED = 0.98        # of the ring round it below the speck's half height
ISOLATED = 8.0         # height over the ring's own spread of the neutral residual
EDGE_INSET = 0.01      # of the frame, kept off the frame's own edge
FIND_RING = 20              # plane px around a speck that say how busy its surroundings are
PROPOSE = dict(enclosed=0.98, isolated=8.0, area=60)  # neutrality left out:
# on DSC_0238 it let through 5 of 11 dust candidates and 285 of 796 others


def planes(path):
    """R, G, B Bayer planes as density against the frame's base later; here,
    transmittance in 0..1, one value per 2x2 block, no interpolation."""
    import rawpy
    with rawpy.imread(str(path)) as r:
        raw = r.raw_image_visible.astype(np.float32)
        cols = r.raw_colors_visible
        black = np.array(r.black_level_per_channel, np.float32)
        white = float(r.white_level)
        desc = r.color_desc.decode()
    H, W = raw.shape
    H, W = H - H % 2, W - W % 2
    out = {"R": [], "G": [], "B": []}
    for dy in (0, 1):
        for dx in (0, 1):
            c = int(cols[dy, dx])
            name = desc[c]
            name = "G" if name in "G" else name
            p = (raw[dy:H:2, dx:W:2] - black[c]) / (white - black[c])
            out[name].append(p)
    rgb = np.stack([out["R"][0], np.mean(out["G"], axis=0), out["B"][0]], -1)
    return np.clip(rgb, 1e-5, None)


def frame_mask(shape, part, angle, inset):
    """Inside the frame's rotated rectangle, in plane (half-scan) coordinates."""
    cx, cy = (v / 2 for v in part["centre"])
    w, h = (v / 2 * (1 - 2 * inset) for v in part["size"])
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    a = np.deg2rad(angle)
    u = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
    v = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
    return (np.abs(u) < w / 2) & (np.abs(v) < h / 2)


_BLANKS = {}


def _blank_median(blanks):
    """Each blank over its own smooth version, median over the blanks.
    Kept for the session, by file and modification time: it is the same
    for every frame of a roll and was 13 of the finder's 46 s. The smooth
    version is taken at a quarter of the size -- a 40 px blur is the same
    blur there, and costs a sixteenth."""
    import os
    key = tuple((b, os.path.getmtime(b)) for b in blanks)
    if key not in _BLANKS:
        maps = []
        for b in blanks:
            T = planes(b)
            maps.append(T / _smooth(T, RIG_SMOOTH))
        _BLANKS.clear()
        _BLANKS[key] = np.median(maps, 0)
    return _BLANKS[key]


def _smooth(a, sigma, down=4):
    """A wide Gaussian of an (H, W, C) array, worked at 1/`down`."""
    small = ndimage.gaussian_filter(a[::down, ::down], (sigma / down, sigma / down, 0))
    out = ndimage.zoom(small, (a.shape[0] / small.shape[0], a.shape[1] / small.shape[1], 1),
                       order=1)
    return out[:a.shape[0], :a.shape[1]]


def _median(a, size):
    """A median of each channel, worked at half size: the level a speck
    rises above, not a picture. Full size was 31 of the finder's 46 s."""
    h = ndimage.median_filter(a[::2, ::2], size=(size // 2 | 1, size // 2 | 1, 1),
                              mode="nearest")
    return np.repeat(np.repeat(h, 2, 0), 2, 1)[:a.shape[0], :a.shape[1]]


def rig_map(blanks, inside=None):
    """The rig's own specks, from the blank frames: per channel, each blank
    over its own smooth version, median over the blanks. Kept only where it
    dips (a speck), 1 elsewhere, so the blanks' grain is not divided into
    every photograph."""
    med = _blank_median(tuple(str(b) for b in blanks))
    m = ndimage.gaussian_filter(med, (1, 1, 0))
    # a speck is a dip that survives smoothing and has some size; single
    # dark pixels are the blanks' noise, and dividing by them adds it
    lab, n = ndimage.label(ndimage.gaussian_filter(med.mean(-1), 2) < RIG_DIP)
    size = ndimage.sum(np.ones(lab.shape), lab, range(1, n + 1))
    # and compact: what the edges of the blanks hold is the holder and the
    # film's edge, which sit elsewhere in every frame, not a speck
    ok = (size >= RIG_AREA) & (size <= RIG_AREA_MAX)
    H, W = lab.shape
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if ok[i] and (sl[0].start < RIG_BORDER or sl[1].start < RIG_BORDER
                      or sl[0].stop > H - RIG_BORDER or sl[1].stop > W - RIG_BORDER):
            ok[i] = False
    if inside is not None:
        ok &= ndimage.minimum(inside, lab, range(1, n + 1)).astype(bool)
    size = np.where(ok, size, 0)
    keep = np.isin(lab, 1 + np.flatnonzero(ok))
    m[~ndimage.binary_dilation(keep, iterations=4)] = 1.0
    return np.clip(m, 0.5, 1.0), int(ok.sum()), keep


def detect(path, item, part, rig=None):
    T = planes(path)
    if rig is not None:
        T = T / rig
    inside = frame_mask(T.shape, part, item["angle"], EDGE_INSET)
    D = -np.log10(T)
    # the base: the least dense film in the frame, per channel
    D -= np.percentile(D[inside], 0.5, axis=0)
    res = D - _median(D, FIND_MEDIAN)
    neutral = res.min(-1)
    sig = 1.4826 * np.median(np.abs(neutral[inside] - np.median(neutral[inside])))
    seeds = (neutral > FIND_SEED_K * sig) & inside
    grow = (neutral > FIND_GROW_K * sig) & inside
    lab, n = ndimage.label(grow)
    keep = np.unique(lab[seeds])
    keep = keep[keep > 0]
    # the frame's D-max, per channel, of the picture: specks are rare enough
    # that a high percentile does not see them
    dmax = np.percentile(D[inside], 99.95, axis=0)
    gy = np.stack([ndimage.sobel(D[..., c], 0) for c in range(3)], -1) / 8
    gx = np.stack([ndimage.sobel(D[..., c], 1) for c in range(3)], -1) / 8
    grad = np.hypot(gx, gy).max(-1)
    slices = ndimage.find_objects(lab)
    specks = []
    for k in keep:
        sl = slices[k - 1]
        # one pixel of margin, for the gradient on the speck's rim
        sl = tuple(slice(max(0, s.start - 2), s.stop + 2) for s in sl)
        m = lab[sl] == k
        area = int(m.sum())
        if area > FIND_MAX_AREA:
            continue
        r = res[sl][m]
        rise = r.max(0)                         # per channel
        height = float(rise.max())
        neu = float(rise.min() / max(rise.max(), 1e-6))
        rim = ndimage.binary_dilation(m, iterations=1)
        sharp = float(grad[sl][rim].max() / max(height, 1e-6))
        # how busy the neighbourhood is: the neutral residual's spread in a
        # ring around the speck, other candidates included
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        Y0, Y1 = max(0, y0 - FIND_RING), min(lab.shape[0], y1 + FIND_RING)
        X0, X1 = max(0, x0 - FIND_RING), min(lab.shape[1], x1 + FIND_RING)
        big = np.zeros((Y1 - Y0, X1 - X0), bool)
        big[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = m
        ring = ndimage.binary_dilation(big, iterations=FIND_RING) & \
            ~ndimage.binary_dilation(big, iterations=3) & inside[Y0:Y1, X0:X1]
        nr = neutral[Y0:Y1, X0:X1][ring]
        busy = float(1.4826 * np.median(np.abs(nr - np.median(nr)))) if len(nr) > 20 else sig
        iso = height / max(busy, sig * 0.5)
        # surrounded: a speck is denser than everything round it; a bright
        # patch of the scene against a dark edge continues on one side
        dm = D[sl].mean(-1)
        close = ndimage.binary_dilation(m, iterations=4) & ~ndimage.binary_dilation(m, iterations=1)
        top_m = float(dm[m].max())
        rise_m = top_m - float(np.median(dm[close])) if close.any() else 0.0
        enclosed = float((dm[close] < top_m - 0.5 * rise_m).mean()) if close.any() else 0.0
        # neutrality by regression over the speck's pixels: R and B against G
        g = r[:, 1]
        slopes = [float((r[:, c] * g).sum() / max((g * g).sum(), 1e-9)) for c in (0, 2)]
        peak = D[sl][m].max(0)
        above = bool((peak > dmax).sum() >= 2)
        ys, xs = np.nonzero(m)
        hgt, wid = np.ptp(ys) + 1, np.ptp(xs) + 1
        elong = float(max(hgt, wid) ** 2 / max(area, 1))
        specks.append({
            "id": int(k), "y": float(ys.mean() + sl[0].start),
            "x": float(xs.mean() + sl[1].start), "area": area,
            "height": round(height, 3), "rise": [round(float(v), 3) for v in rise],
            "neutral": round(neu, 2), "sharp": round(sharp, 3),
            "enclosed": round(enclosed, 2), "isolation": round(iso, 1),
            "busy": round(busy, 4), "slopes": [round(v, 2) for v in slopes],
            "above_dmax": above, "elongation": round(elong, 1),
            "bbox": [sl[0].start, sl[0].stop, sl[1].start, sl[1].stop],
        })
    for s in specks:
        s["cues"] = {"neutral": s["neutral"] >= NEUTRAL,
                     "sharp": s["sharp"] >= SHARP,
                     "dmax": s["above_dmax"],
                     "enclosed": s["enclosed"] >= ENCLOSED,
                     "isolated": s["isolation"] >= ISOLATED,
                     "small": s["area"] <= AREA_MAX}
        # sharp and D-max are measured and reported, and decide nothing: on
        # this frame neither separates dust from the scene (see the summary)
        c = s["cues"]
        s["dust"] = bool(c["neutral"] and c["enclosed"] and c["isolated"] and c["small"])
    return {"lab": lab, "specks": specks, "sigma": float(sig),
            "dmax": [round(float(v), 3) for v in dmax], "shape": T.shape[:2]}


def find(path, item, part, blanks=()):
    """Proposals for the frame: the rig's specks from `blanks` and the film
    specks `detect` puts forward, as marks in scan pixels.

    A Bayer-plane pixel i covers scan pixels 2i and 2i+1, so a plane
    coordinate p is scan 2p + 0.5."""
    out = []
    rig = None
    if blanks:
        shape = (item["scan"][1] // 2, item["scan"][0] // 2)
        rig, _n, where = rig_map(blanks, frame_mask(shape, part, item["angle"], 0.03))
        lab, n = ndimage.label(where)
        for (cy, cx), a in zip(ndimage.center_of_mass(where, lab, range(1, n + 1)),
                               ndimage.sum(where, lab, range(1, n + 1))):
            out.append({"x": round(2 * cx + 0.5, 1), "y": round(2 * cy + 0.5, 1),
                        "by": "found", "kind": "rig", "state": "proposed",
                        "area": int(4 * a)})
    for s in detect(path, item, part, rig)["specks"]:
        if (s["enclosed"] >= PROPOSE["enclosed"] and s["isolation"] >= PROPOSE["isolated"]
                and s["area"] <= PROPOSE["area"]):
            out.append({"x": round(2 * s["x"] + 0.5, 1), "y": round(2 * s["y"] + 0.5, 1),
                        "by": "found", "kind": "film", "state": "proposed",
                        "area": int(4 * s["area"])})
    return out


def used(mark):
    """Whether the develop takes this mark out: a hand mark, or a found
    one a person accepted. A proposal nobody has looked at is not
    evidence of anything, and a refused one is evidence it is not dust."""
    return mark.get("by", "hand") == "hand" or mark.get("state") == "accepted"
