"""The third detector, `fit`: a frame is a rectangle whose four sides lie on
edges, whose inside is film, and whose size the roll already knows.

Measured on 98 scans placed by hand: 97 right, and the one it gets wrong
-- two blank exposures with no edge to measure -- it marks `held`. Every scan
it leaves unmarked is right.

It is built on three things the other two detectors do not use:

- **Two classes of pixel that are never picture.** Direct light -- the table
  seen through a sprocket hole or past the end of the film -- is a plateau at
  the scan's minimum density with nothing for 0.35 D above it, because the
  film's base is that much denser. The holder is a plateau of dense pixels
  touching the scan's border. A frame may not contain the first at all and
  the second only as a band along one side (the picture running under the
  mask). That is what stops the strip -- sprocket rows above and below, mask
  left and right -- from being read as a 3:2 frame, which is the reading
  that beat both half frames on 36 of 37 scans of the half-frame roll.
- **Every side scored at once, saturated.** A candidate rectangle is scored
  by the coherent gradient integrated along each of its four sides, clipped
  per pixel first: a holder's edge is 27 times a faint frame edge, and unless
  it is capped it buys three wrong sides with one right one. After the cap
  the search is only asking whether there is an edge, not how loud it is.
- **The roll.** Every frame on a roll came through one gate: the heights
  agree to 0.5 % and the gap between the two exposures of a half-frame shot
  to 15 px. The first pass measures each scan alone; the roll's median size,
  gap and row position then constrain a second pass, which is what places a
  blank exposure beside a visible one, and what settles a side whose window
  holds more than one ramp.

The coarse search runs at a thirty-second, a sixteenth and an eighth of
the scan; each side
is then refined at half resolution against the hand-placed convention --
the innermost ramp with picture inside it, at its centre when it ends on
film and at its inner end when it runs into the holder, because a penumbra
is the picture being darkened and not picture.
"""

import math

import numpy as np
from scipy import ndimage

from . import findframe

# ---------------------------------------------------------------- classes

# Direct light: a plateau this far above the scan's robust minimum ...
DIRECT_BAND = 0.15
# ... with nothing between it and this far above, where the film's base
# starts. Measured on the half-frame roll: sprocket tips at 0.24-0.40 D,
# base at 0.83-0.97, the band between holding 0.11-0.36 % of the plane.
GAP_BAND = 0.35
# The holder: nothing on film is this dense (a transmittance of 0.3 %; a
# colour negative's densest highlight is under 2.5 D above the table it is
# measured against on every scan in the corpus), and it touches the border.
# It was first set relative to the scan's maximum, 0.4 D under it, which on
# the half-frame roll left half the mask unmarked -- harmless, since the
# rectangle constraint only needs the mask's core, but not what it claims.
OPAQUE_ABS = 2.5
# What counts as "into the holder" for a ramp: past this the profile is not
# film any more. Colour negative rebate reaches 1.9 on the Portra roll.
OPAQUE_END = 2.2


def classify(D):
    """(direct, opaque, info) -- boolean masks over a density image.

    Direct light is decided by a valley test rather than a level: on the
    Portra roll the frame's own shadows are the lowest plateau in the scan
    and a level test called them light table. A real light-table plateau has
    an empty band above it, because the base of the film is at least
    0.35 D denser than the table it lies on.

    The holder is decided by connectivity rather than density alone: a dense
    highlight in the picture crosses any level the mask does, but it does
    not touch the border of the scan.
    """
    dmin = float(np.percentile(D, 0.05))
    dmax = float(np.percentile(D, 99.5))
    direct_lvl = dmin + DIRECT_BAND
    m_direct = float(np.mean(D < direct_lvl))
    m_gap = float(np.mean((D >= direct_lvl) & (D < dmin + GAP_BAND)))
    has_direct = m_direct >= 0.001 and m_gap <= 0.005 and m_gap < m_direct
    direct = (D < direct_lvl) if has_direct else np.zeros(D.shape, bool)

    op_lvl = OPAQUE_ABS
    cand = ndimage.binary_opening(D >= op_lvl, np.ones((5, 5), bool))
    labels, n = ndimage.label(cand)
    opaque = np.zeros(D.shape, bool)
    if n:
        border = set(np.unique(labels[0])) | set(np.unique(labels[-1])) \
            | set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
        border.discard(0)
        if border:
            opaque = np.isin(labels, list(border))
    if opaque.mean() < 0.01:
        opaque[:] = False
    return direct, opaque, {
        "direct": bool(has_direct), "direct_level": round(direct_lvl, 3),
        "direct_share": round(m_direct, 4),
        "opaque": bool(opaque.any()), "opaque_level": round(op_lvl, 3),
        "opaque_share": round(float(opaque.mean()), 4)}


# ---------------------------------------------------------- coarse search

FORMATS = findframe.FORMATS
EPS = 3e-3
# Density per pixel, over the ramp, above which an edge is an edge. A faint
# frame edge on bare film is 0.1 D over three eighth-scale pixels, a
# holder's edge is 2 D; after the cap they score alike, which is the point.
CAP = 0.05
OFF = math.log(EPS * 3)       # a side beyond the sensor: unmeasured, not absent
PAD_FRAC = 0.12               # how far a frame may run off the sensor
RAMP = 3                      # pixels across the edge the gradient is summed over
MARGIN = 2                    # pixels inside the side the class test ignores
DIRECT_TOL = 0.0002           # share of the eroded area that may be light table
OPAQUE_TOL = 0.15             # share of the area that may be holder
ANGLES = np.arange(-3.0, 3.01, 0.5)
# A frame is never smaller than this share of the scan's height: the scan is
# made with some margin around the frame, not with the frame lost in it. On
# the corpus the lowest is 0.67 (a Cinestill scan at the smaller
# magnification); the rest sit at 0.83-0.88. It prunes the small heights,
# which are the ones with the most positions to try, and with them the
# small rectangles inside the picture that used to reach the top ten. The
# roll setting `min_fill` overrides it.
MIN_FILL = 0.5


def _bin(a, k):
    a = a[: a.shape[0] // k * k, : a.shape[1] // k * k]
    return a.reshape(a.shape[0] // k, k, a.shape[1] // k, k).mean((1, 3))


def _density(g):
    return -np.log10(np.clip(g.astype(np.float32), 1e-4, None))


def _rotate(D, angle):
    if not angle:
        return D
    return ndimage.rotate(D, angle, order=1, reshape=False, mode="nearest")


def _rotate_mask(m, angle):
    if m is None:
        return None
    if not angle:
        return m
    return ndimage.rotate(m.astype(np.float32), angle, order=1,
                          reshape=False, mode="constant", cval=0) > 0.5


def _integral(m):
    out = np.zeros((m.shape[0] + 1, m.shape[1] + 1), np.int64)
    out[1:, 1:] = np.cumsum(np.cumsum(m, 0), 1)
    return out


class _Tables:
    """Cumulative sums that make any axis-aligned rectangle O(1) to score."""

    def __init__(self, D, direct, opaque, pad):
        self.pad = pad
        self.H, self.W = D.shape
        Dp = np.pad(D, pad, mode="edge")
        gx = np.diff(Dp, axis=1)
        gy = np.diff(Dp, axis=0)
        gx = ndimage.uniform_filter1d(gx, RAMP, axis=1, mode="nearest")
        gy = ndimage.uniform_filter1d(gy, RAMP, axis=0, mode="nearest")
        # Saturate per pixel, *before* integrating along the side: a tilted
        # edge then still covers only part of the side and scores as such.
        gx = np.clip(gx, -CAP, CAP)
        gy = np.clip(gy, -CAP, CAP)
        z = np.zeros
        self.CX = np.vstack([z((1, gx.shape[1])), np.cumsum(gx, 0)])
        self.AX = np.vstack([z((1, gx.shape[1])), np.cumsum(np.abs(gx), 0)])
        self.CY = np.hstack([z((gy.shape[0], 1)), np.cumsum(gy, 1)])
        self.AY = np.hstack([z((gy.shape[0], 1)), np.cumsum(np.abs(gy), 1)])
        self.Hp, self.Wp = Dp.shape
        self.FD = _integral(np.pad(direct, pad)) if direct is not None else None
        self.FO = _integral(np.pad(opaque, pad)) if opaque is not None else None

    @staticmethod
    def _side(S, A, n):
        """log of (mean gradient x coherence^2): the coherence is what tells
        a frame edge from the picture -- see findframe.edge_strength."""
        m = np.abs(S) / n
        c = np.abs(S) / (A + 1e-9)
        return np.log(m * c * c + EPS)

    def _count(self, F, ny, nx, w, h, m):
        ya, yb = 1 + m, h - m
        xa, xb = 1 + m, w - m
        cnt = (F[yb:yb + ny, xb:xb + nx] - F[ya:ya + ny, xb:xb + nx]
               - F[yb:yb + ny, xa:xa + nx] + F[ya:ya + ny, xa:xa + nx])
        return cnt, max(1, (yb - ya) * (xb - xa))

    def rects(self, w, h):
        """(score, sides off the sensor) for every w x h rectangle, indexed
        by its top-left corner in padded coordinates."""
        ny, nx = self.Hp - 1 - h, self.Wp - 1 - w
        if ny <= 0 or nx <= 0:
            return None, None
        CX, AX, CY, AY = self.CX, self.AX, self.CY, self.AY
        L = self._side(CX[h:h + ny, :nx] - CX[:ny, :nx],
                       AX[h:h + ny, :nx] - AX[:ny, :nx], h)
        R = self._side(CX[h:h + ny, w:w + nx] - CX[:ny, w:w + nx],
                       AX[h:h + ny, w:w + nx] - AX[:ny, w:w + nx], h)
        T = self._side(CY[:ny, w:w + nx] - CY[:ny, :nx],
                       AY[:ny, w:w + nx] - AY[:ny, :nx], w)
        B = self._side(CY[h:h + ny, w:w + nx] - CY[h:h + ny, :nx],
                       AY[h:h + ny, w:w + nx] - AY[h:h + ny, :nx], w)
        p = self.pad
        x0 = np.arange(nx)[None, :]
        y0 = np.arange(ny)[:, None]
        offL, offR = x0 < p, x0 + w >= self.W + p
        offT, offB = y0 < p, y0 + h >= self.H + p
        S = (np.where(offL, OFF, L) + np.where(offR, OFF, R)
             + np.where(offT, OFF, T) + np.where(offB, OFF, B))
        noff = offL.astype(int) + offR + offT + offB
        if self.FD is not None:
            cnt, area = self._count(self.FD, ny, nx, w, h, MARGIN)
            S = np.where(cnt > DIRECT_TOL * area, -np.inf, S)
        if self.FO is not None:
            cnt, area = self._count(self.FO, ny, nx, w, h, 0)
            S = np.where(cnt > OPAQUE_TOL * area, -np.inf, S)
        return S, noff


def _window(S, ywin, pad):
    if ywin is None:
        return S
    ya = max(0, ywin[0] + pad)
    yb = min(S.shape[0], ywin[1] + pad + 1)
    M = np.full(S.shape, -np.inf)
    M[ya:yb] = S[ya:yb]
    return M


def _best_single(tab, w, h, ywin=None, max_off=1):
    S, noff = tab.rects(w, h)
    if S is None:
        return None
    S = _window(np.where(noff > max_off, -np.inf, S), ywin, tab.pad)
    i = int(np.argmax(S))
    y0, x0 = divmod(i, S.shape[1])
    if not np.isfinite(S[y0, x0]):
        return None
    return float(S[y0, x0]) / 4, [(x0 - tab.pad, y0 - tab.pad, w, h)], int(noff[y0, x0])


def _best_pair(tab, w, h, gap, ywin=None, max_off=2):
    """Two frames of one size on one row, `gap` = (min, max) px apart.

    The best partner of each left frame is a sliding maximum along the row,
    over the window of allowed distances, so the pair costs one pass and not
    one per gap value -- which was the whole cost of the half-frame search.

    A row of two measures its own width, so one side off the sensor is
    forgiven -- the rule findframe arrived at the hard way.
    """
    S, noff = tab.rects(w, h)
    if S is None:
        return None
    S = _window(np.where(noff > max_off, -np.inf, S), ywin, tab.pad)
    n = S.shape[1]
    dmin, dmax = w + gap[0], min(w + gap[1], n - 1)
    if dmin >= n:
        return None
    k = dmax - dmin + 1
    Sf = np.where(np.isfinite(S), S, -1e9)
    # right[y, x] = max of Sf[y, x : x + k]; the partner of a frame at xa
    # is the best of Sf[y, xa + dmin : xa + dmax + 1] = right[y, xa + dmin].
    right = ndimage.maximum_filter1d(Sf, size=k, axis=1, mode="constant",
                                     cval=-1e9, origin=-(k // 2))
    m = n - dmin
    tot = Sf[:, :m] + right[:, dmin:dmin + m]
    i = int(np.argmax(tot))
    y0, xa = divmod(i, tot.shape[1])
    if tot[y0, xa] < -1e8:
        return None
    seg = Sf[y0, xa + dmin: xa + dmax + 1]
    xb = xa + dmin + int(np.argmax(seg))
    p = tab.pad
    return float(tot[y0, xa]) / 8, [(xa - p, y0 - p, w, h), (xb - p, y0 - p, w, h)], \
        int(noff[y0, xa] + noff[y0, xb])


def _search(D, direct, opaque, fmt, angles, hs, pad, gap=None, ywin=None):
    best = None
    W = D.shape[1]
    for a in angles:
        tab = _Tables(_rotate(D, a), _rotate_mask(direct, a),
                      _rotate_mask(opaque, a), pad)
        for h in hs:
            w = int(round(FORMATS[fmt] * h))
            if w >= tab.Wp - 2 or h >= tab.Hp - 2 or h < 4:
                continue
            if fmt == "full":
                res = _best_single(tab, w, h, ywin)
            else:
                g = gap or (max(2, int(0.02 * W)), int(0.12 * W))
                res = _best_pair(tab, w, h, g, ywin)
            if res is not None and (best is None or res[0] > best[0]):
                best = (res[0], float(a), res[1], res[2])
    return best


def _hypothesis(D16, d16, o16, D8, d8, o8, fmt, hs16, gap16=None, y16=None):
    """The best set of frames of one format, coarse to fine.

    Every angle is swept at a thirty-second of the scan, where a half frame
    is still 110 px tall and the sweep costs a quarter of what it did at a
    sixteenth; then three angles at a sixteenth around the winner, then the
    fine steps at a sixteenth and an eighth. Measured on the 98 labelled
    scans against sweeping every angle at a sixteenth: the same 97 right,
    the same two held, at 2.6 s a scan and pass on one core instead of 12.
    Pruning by the classes -- the frame lies inside the window they leave --
    was tried first and bought nothing: on bare film the window is the scan.

    Returns dict(score, angle, rects at 1/8, off) or None.
    """
    H, W = D16.shape
    pad16 = int(PAD_FRAC * W)

    def half(t):
        return None if t is None else (t[0] // 2, (t[1] + 1) // 2)

    def twice(t):
        return None if t is None else (2 * t[0], 2 * t[1] + 1)

    D32 = _bin(D16, 2)
    d32 = _bin(d16.astype(np.float32), 2) > 0.5
    o32 = _bin(o16.astype(np.float32), 2) > 0.5
    pad32 = int(PAD_FRAC * D32.shape[1])
    hs32 = sorted({max(3, h // 2) for h in hs16})
    c = _search(D32, d32, o32, fmt, ANGLES, hs32, pad32, half(gap16), half(y16))
    if c is None:
        # Nothing at a thirty-second: the full sweep at a sixteenth is the
        # honest fallback, and it is what the search was before the pruning.
        c = _search(D16, d16, o16, fmt, ANGLES, hs16, pad16, gap16, y16)
        if c is None:
            return None
        sc, a, rects, noff = c
    else:
        sc, a, rects, noff = c
        h32 = rects[0][3]
        hs = [h for h in hs16 if 2 * h32 - 6 <= h <= 2 * h32 + 6] or list(hs16)
        c = _search(D16, d16, o16, fmt, [a - 0.5, a, a + 0.5], hs, pad16,
                    gap16, y16)
        if c is None:
            return None
        sc, a, rects, noff = c
    h16 = rects[0][3]
    c2 = _search(D16, d16, o16, fmt, np.arange(a - 0.4, a + 0.41, 0.1),
                 range(h16 - 3, h16 + 4), pad16, gap16, y16)
    if c2 is not None:
        sc, a, rects, noff = c2
    h16 = rects[0][3]
    c3 = _search(D8, d8, o8, fmt, np.arange(a - 0.12, a + 0.121, 0.04),
                 range(2 * h16 - 5, 2 * h16 + 6), 2 * pad16,
                 twice(gap16), twice(y16))
    if c3 is not None:
        sc, a, rects, noff = c3
    else:
        rects = [(2 * x, 2 * y, 2 * w, 2 * h) for x, y, w, h in rects]
    return dict(score=sc, angle=a, rects=rects, off=noff)


def _gap_evidence(D8, hyp):
    """Whether the band between two candidate frames is an unexposed gap:
    flat, lighter than both pictures, and at the film's base.

    A pair of half frames plus the gap between them is 3:2 within a few per
    cent, so the shape cannot say one frame or two; the gap can. A full
    frame would need a black band the height of the picture to fake it.
    """
    Dr = _rotate(D8, hyp["angle"])
    (xa, y0, w, h), (xb, _, _, _) = sorted(hyp["rects"])
    ys = slice(max(0, y0 + 3), y0 + h - 2)
    gap = Dr[ys, xa + w + 2: xb - 1]
    fa = Dr[ys, max(0, xa + 2): xa + w - 1]
    fb = Dr[ys, xb + 2: xb + w - 1]
    if gap.size < 10 or fa.size < 10 or fb.size < 10:
        return False, {}
    med, p90 = float(np.median(gap)), float(np.percentile(gap, 90))
    fa_med, fb_med = float(np.median(fa)), float(np.median(fb))
    p2 = float(min(np.percentile(fa, 2), np.percentile(fb, 2)))
    ok = (p90 - med < 0.12 and med < min(fa_med, fb_med) - 0.05
          and med < p2 + 0.08)
    return ok, {"gap_median": round(med, 3), "gap_p90": round(p90, 3),
                "frames_median": [round(fa_med, 3), round(fb_med, 3)]}


def _to_scan(rects, angle, shape, scale):
    """Rectangles in the squared-up eighth-scale image -> (cx, cy, w, h) in
    scan pixels, undoing the rotation about the image centre."""
    H, W = shape
    cx0, cy0 = (W - 1) / 2, (H - 1) / 2
    r = math.radians(angle)
    co, si = math.cos(r), math.sin(r)
    out = []
    for x0, y0, w, h in rects:
        cx, cy = x0 + 0.5 + w / 2, y0 + 0.5 + h / 2
        dx, dy = cx - cx0, cy - cy0
        sx = cx0 + dx * co - dy * si
        sy = cy0 + dx * si + dy * co
        out.append(((sx + 0.5) * scale - 0.5, (sy + 0.5) * scale - 0.5,
                    w * scale, h * scale))
    return out


# ---------------------------------------------------------------- planes

PITCH_MM = 4.75        # 135 film's perforation pitch (ISO 1007), long and short pitch alike to 0.5 %
GATE_H_MM = (23.5, 25.1)   # across the film, what a 135 exposure measures: 24 mm, full or half frame
PITCH_SURE = 0.35      # the normalised autocorrelation the pitch's peak must reach
PITCH_SPREAD = 0.02    # scans whose pitch differs more than this were copied at two magnifications
GATE_TOL = 0.01        # the tight clusters the gate is chosen among


def sprocket_pitch(g2):
    """(perforation pitch in scan px, how sure) off the green half plane,
    or (None, 0) where no row of holes is in the scan.

    **The one length on a strip of 135 film that is the same on every
    roll**: the holes are 4.75 mm apart, whatever the camera, the stock
    or the scene. With it the scan has a scale in millimetres, and a
    frame's height across the film -- 24 mm in full frame and half frame
    both -- stops being something only the frames themselves can vote
    on. Measured on one roll: 37 scans at 146.5 px/mm,
    +-0.3 %.

    The holes show the light table, so they are the brightest thing in
    the band along each long edge: a column profile of the band's 90th
    percentile, and its autocorrelation's first peak within the lags a
    35 mm frame filling 40-105 % of the scan's height allows."""
    H, W = g2.shape
    best = (None, 0.0)
    for band in (g2[:max(8, H // 10)], g2[-max(8, H // 10):]):
        prof = np.percentile(band, 90, axis=0).astype(np.float64)
        prof -= prof.mean()
        if not prof.any():
            continue
        ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
        ac /= max(ac[0], 1e-12)
        # half-plane px per mm for a 24 mm frame 40 % to 105 % of the height
        lo = int(PITCH_MM * 0.40 * H / 24.0)
        hi = min(len(ac) - 2, int(PITCH_MM * 1.05 * H / 24.0) + 1)
        if hi <= lo + 2:
            continue
        k = lo + int(np.argmax(ac[lo:hi]))
        if not (lo < k < hi - 1):
            continue
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        lag = k + 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c else float(k)
        # a row of holes repeats: the second multiple has to be there too
        k2 = int(round(2 * lag))
        sure = float(min(b, ac[max(0, k2 - 2):k2 + 3].max())) if k2 + 3 < len(ac) else float(b)
        if sure > best[1]:
            best = (2.0 * lag, sure)
    return best if best[1] >= PITCH_SURE else (None, best[1])


class Planes:
    """What one scan is measured on: the green sub-plane at a half, an
    eighth and a sixteenth of the scan, as transmittance."""

    def __init__(self, path):
        import rawpy
        with rawpy.imread(str(path)) as r:
            v = r.raw_image_visible.astype(np.float32)
            black = float(np.mean(r.black_level_per_channel))
            white = float(r.white_level)
            self.scan = (v.shape[1], v.shape[0])
        g = np.clip(v[1::2, 0::2] - black, 0, None) / (white - black)
        self.g2 = g
        self.g8 = _bin(g, 4)
        self.g16 = _bin(self.g8, 2)


def coarse(planes, fmts=("full", "half"), roll=None, min_fill=MIN_FILL):
    """The first stage: one or two frames, to within a few eighth-scale
    pixels. `roll` is None on the first pass; on the second it carries the
    roll's format, height, gap and row position in scan pixels."""
    D8, D16 = _density(planes.g8), _density(planes.g16)
    direct, opaque, classes = classify(D8)
    d16 = _bin(direct.astype(np.float32), 2) > 0.5
    o16 = _bin(opaque.astype(np.float32), 2) > 0.5
    H, W = D16.shape
    scale8 = planes.scan[0] / D8.shape[1]
    s16 = 2 * scale8
    hyps = {}
    for fmt in (fmts if roll is None else (roll["format"],)):
        if roll is None:
            hs16, gap16, y16 = range(int(min_fill * H), H), None, None
        else:
            h16 = roll["h"] / s16
            hs16 = range(int(round(h16 * 0.99)), int(round(h16 * 1.01)) + 1)
            gap16 = None if roll.get("gap") is None else \
                (int(roll["gap"] / s16) - 2, int(roll["gap"] / s16) + 3)
            y16 = None if roll.get("y0") is None else \
                (int(roll["y0"] / s16) - 8, int(roll["y0"] / s16) + 8)
        hy = _hypothesis(D16, d16, o16, D8, direct, opaque, fmt, hs16, gap16, y16)
        if hy is not None:
            hyps[fmt] = hy
    if not hyps:
        return None
    pair_ok, gap_ev = (False, {})
    if "half" in hyps:
        pair_ok, gap_ev = _gap_evidence(D8, hyps["half"])
    if roll is not None:
        fmt = roll["format"]
    elif pair_ok and "half" in hyps:
        fmt = "half"
    else:
        fmt = "full" if "full" in hyps else "half"
    hyp = hyps[fmt]
    return {"format": fmt, "angle": hyp["angle"], "score": hyp["score"],
            "off": hyp["off"], "pair": pair_ok,
            "parts": _to_scan(hyp["rects"], hyp["angle"], D8.shape, scale8),
            "classes": classes, "gap": gap_ev,
            "scores": {k: round(v["score"], 3) for k, v in hyps.items()}}


# ------------------------------------------------------------ refinement

R_ACROSS = 40       # half-res px either side of the coarse edge
WIDE = 0.06         # ... or this share of the width for left and right, which
#                     vary by up to 4 % across a roll where heights vary 0.5 %
K_SEGMENTS = 5      # segments along a side, each yielding one edge position
SMOOTH = 3
MIN_STEP = 0.04     # effective D; less is not an edge
REL_STEP = 0.05     # ... or less than this share of the window's largest ramp
MAX_RAMP = 40       # half-res px; wider is a gradient in the picture
MIN_PEAK = 0.006    # effective D per px at the ramp's steepest
INSIDE_MARGIN = 0.08
# **Nothing on film is less dense than unexposed film.** A ramp whose far
# side reads below the film's own base is not a frame edge: it is the edge
# of the film, where the light table shows past it. The frame edge and the
# film edge are both steps down going outward, so the step alone cannot
# tell them apart; what is on the far side can. It is the same test the
# holder already gets from the other direction (`into`, OPAQUE_END).
#
# Measured on roma-half, where the refinement was taking the film's edge
# for the picture's: SDIM0908's bottom ramp reads 0.782 outside against a
# base of 0.823, and the rebate outside a real frame edge reads base+0.06.
# The margin is what separates those two, with room for the ramp's own
# foot.
OFF_FILM = 0.02
# A frame edge runs the whole side; a line in the picture does not. Two
# candidate ramps count as the same line when their offsets agree to
# AGREE_TOL half-res px, and a line is one the side really has when at
# least AGREE_QUORUM of the K_SEGMENTS segments hold it. See `_agreed`.
AGREE_TOL = 4.0
AGREE_QUORUM = 4


def _ramps(p_eff, m):
    """Runs of one sign in the effective gradient, innermost first:
    (t_in, t_out, step, centre)."""
    n = len(p_eff)
    sign = np.sign(p_eff)
    out = []
    i = 0
    while i < n:
        s = sign[i]
        if s == 0:
            i += 1
            continue
        j = i
        while j < n and sign[j] == s:
            j += 1
        seg = p_eff[i:j]
        a = np.abs(seg)
        idx = np.nonzero(a >= 0.15 * a.max())[0]
        out.append((i + int(idx[0]), i + int(idx[-1]), float(seg.sum()),
                    float((np.arange(i, j) * a).sum() / a.sum())))
        i = j
    return out


def _side_profile(Dr, axis, pos, lo, hi, R):
    """(effective gradient, mean density) across a side, for t in [-R, R],
    averaged over [lo, hi) along it. The gradient is weighted by its own
    coherence squared, so a line that is only in part of the segment
    counts less than the frame edge that runs through all of it."""
    if pos - R < 0:
        return None, None
    band = Dr[lo:hi, pos - R: pos + R + 2] if axis == 1 \
        else Dr[pos - R: pos + R + 2, lo:hi].T
    if band.shape[1] < 2 * R + 2 or band.shape[0] < 8:
        return None, None
    g = ndimage.uniform_filter1d(np.diff(band, axis=1), SMOOTH, axis=1,
                                 mode="nearest")
    S = g.sum(0)
    c = np.abs(S) / (np.abs(g).sum(0) + 1e-9)
    return (S / g.shape[0]) * c * c, band.mean(0)[:-1]


def _find_edge(p_eff, m, outward, R, base):
    """(position in [-R, R], evidence, alternatives) or (None, 0, []).

    The rule a person's placements follow: the innermost ramp that is big
    and steep enough to be an edge and has picture inside it; its centre
    when it ends on film, its inner end when it runs into the holder.
    """
    if outward < 0:
        p_eff, m = p_eff[::-1].copy(), m[::-1].copy()
    rs = _ramps(p_eff, m)
    if not rs:
        return None, 0.0, []
    big = max(abs(r[2]) for r in rs)

    def is_edge(r):
        t_in, t_out, step, _ = r
        if abs(step) < max(MIN_STEP, REL_STEP * big):
            return False
        into = bool(m[min(len(m) - 1, t_out + 3)] >= OPAQUE_END)
        if t_out - t_in > MAX_RAMP and not into:
            return False
        if base is not None and not into:
            # Off the film: see OFF_FILM.
            beyond = m[min(len(m) - 1, t_out + 3): min(len(m), t_out + 12)]
            if beyond.size and float(np.median(beyond)) < base - OFF_FILM:
                return False
        return float(np.abs(p_eff[t_in:t_out + 1]).max()) >= MIN_PEAK

    def picture_inside(r):
        seg = m[max(0, r[0] - 12): max(0, r[0] - 3)]
        if seg.size == 0 or base is None:
            return True
        return base + INSIDE_MARGIN < float(np.median(seg)) < OPAQUE_END

    def position(r):
        t_in, t_out, _, centre = r
        into = bool(m[min(len(m) - 1, t_out + 3)] >= OPAQUE_END)
        t = (float(t_in) if into else centre) - R
        return -t if outward < 0 else t

    cands = sorted((r for r in rs if is_edge(r)), key=lambda r: r[0])
    if not cands:
        return None, 0.0, []
    passing = [c for c in cands if picture_inside(c)]
    r = passing[0] if passing else cands[0]
    alts = [(position(c), abs(c[2]), c in passing) for c in cands]
    return position(r), abs(r[2]), alts


def _agreed(cands, outward):
    """The candidate line the whole side has, innermost first, or None.

    `_find_edge` chooses **per segment**, and its rule -- the innermost ramp
    that is an edge with picture inside -- has a known failure: a line in
    the picture near the edge is such a ramp too, and it is more inside than
    the edge. The arbitration below exists because of it, but only fires on
    a side whose segments disagree or a frame off the roll's size.

    A person does not place an edge that way. A frame edge is a line that
    runs the **whole side**; a cornice, a horizon or a light leak runs
    through some of it. So the candidates of all the segments are pooled,
    grouped by offset, and only a group at least `AGREE_QUORUM` segments
    hold can be the edge; the innermost such group with picture inside it
    wins, and the segments' own positions within it are what the line is
    fitted through, so nothing is invented.

    Measured on SDIM0903 of roma-half, the worst-placed scan of that roll:
    on the top side the per-segment rule took ramps at +34.6, +26.1, +33.5,
    +30.0 and +3.1 (evidence 0.04-0.16, absent from one segment) while the
    frame edge sat at +3.0, +2.8, +3.3, +3.0, +3.1 -- five of five, within
    half a pixel, evidence 0.30-0.35. It cost 58 scan px of picture off the
    top, against a roll median of 8.
    """
    groups = []
    for s_k, alts in cands:
        for t, ev, ok in alts:
            for g in groups:
                if abs(g["t"] - t) <= AGREE_TOL:
                    g["pts"].append((s_k, t, ev))
                    g["segs"].add(s_k)
                    g["ok"].append(bool(ok))
                    g["t"] = float(np.mean([p[1] for p in g["pts"]]))
                    break
            else:
                groups.append(dict(t=float(t), pts=[(s_k, t, ev)],
                                   segs={s_k}, ok=[bool(ok)]))
    # Picture inside is read per segment and one segment can be wrong about
    # it -- a dark corner, the holder crossing one end -- so the line needs
    # most of its segments to say so, not all of them.
    full = [g for g in groups if len(g["segs"]) >= AGREE_QUORUM
            and sum(g["ok"]) * 2 >= len(g["ok"])]
    if not full:
        return None
    # Innermost, which is what the per-segment rule wanted all along; the
    # quorum is what stops the picture's own lines from standing for one.
    # Which way is in depends on the side: offsets run down and right, so
    # the top and left sides go in as the offset grows and the other two
    # the other way.
    best = (max if outward < 0 else min)(full, key=lambda g: g["t"])
    # One point per segment: the candidate of that segment nearest the line.
    out = {}
    for s_k, t, ev in best["pts"]:
        if s_k not in out or abs(t - best["t"]) < abs(out[s_k][1] - best["t"]):
            out[s_k] = (s_k, t, ev)
    return sorted(out.values())


def _at(side, own, slope=None):
    """Where a fitted side is at `own` along it, not where it was fitted.

    `_fit_line` returns the value **at the centre it was given** plus a
    slope, and for a side measured on its own that centre is the side's own
    middle, so the value is the answer. The top and bottom of a half-frame
    row are not: they are fitted once over both frames, about the row's
    centre, and each frame sits about half its width to one side of it.
    Reading the row's value at the row's centre for both put their top
    edges `slope * width` apart -- 43 px on SDIM0917 of roma-half, the
    scan with the largest angle correction of its roll, which is a visible
    step between two halves of one piece of film. For a side fitted about
    its own centre this is the identity, so it is used for every side.

    `slope` overrides the line's own, and the row's shared sides are read
    with the **tilt the frames are drawn at** -- the median over every
    measured side. A row is two rigid rectangles at one angle, and its top
    line was fitted with a tilt of its own; where the sides of a scan
    disagree those cannot both be obeyed. Reading the line at its own
    slope put the two top-edge midpoints on it and then drew the edges at
    a different angle, so they were not collinear after all: SDIM0917
    disagrees by 0.52 degrees and its halves came out 37 px out of line.
    Read at the angle they are drawn at, the row is consistent with
    itself, and the disagreement is already flagged as `angle-spread` for
    a person to look at. It also makes the two heights equal, which two
    halves of one piece of film have to be.
    """
    return side["offset"] + (side["slope"] if slope is None else slope) \
        * (own - side["centre"])


def _fit_line(pts, centre):
    """Theil-Sen through (position along the side, offset across it)."""
    s = np.array([p[0] for p in pts])
    t = np.array([p[1] for p in pts])
    ev = np.array([p[2] for p in pts])
    slopes = [(t[j] - t[i]) / (s[j] - s[i]) for i in range(len(s))
              for j in range(i + 1, len(s)) if s[j] != s[i]]
    slope = float(np.median(slopes)) if slopes else 0.0
    off = float(np.median(t - slope * (s - centre)))
    resid = float(np.median(np.abs(t - (off + slope * (s - centre)))))
    return dict(offset=off, slope=slope, resid=resid,
                evidence=float(np.median(ev)), n=len(pts))


def _refine_sides(Dr, x0, y0, w, h, base):
    """Each side of a coarse rectangle, measured on its own at half res.

    Returns {side: dict(offset, slope, resid, evidence, cands, centre)} with
    None for a side that could not be measured. Offsets are along the axis
    (positive = right or down), relative to the coarse side.
    """
    H, W = Dr.shape
    x1, y1 = x0 + w, y0 + h
    sides = {}
    spec = (("L", 1, x0, y0, y1, -1), ("R", 1, x1, y0, y1, +1),
            ("T", 0, y0, x0, x1, -1), ("B", 0, y1, x0, x1, +1))
    for name, axis, pos, lo, hi, outward in spec:
        origin = pos
        length = hi - lo
        a, b = lo + int(0.05 * length), hi - int(0.05 * length)
        seg = (b - a) / K_SEGMENTS
        limit = W if axis == 1 else H
        along = H if axis == 1 else W
        Rw = max(R_ACROSS, int(WIDE * w)) if axis == 1 else R_ACROSS
        # A side the coarse pass left inside the holder: walk inward until
        # there is film under it, and look for the edge from there.
        mprof = (Dr[max(0, lo):hi, :].mean(0) if axis == 1
                 else Dr[:, max(0, lo):hi].mean(1))
        moved = 0
        while 0 <= pos < limit and mprof[pos] >= OPAQUE_END \
                and moved < 0.1 * (w if axis == 1 else h):
            pos -= outward
            moved += 1
        if pos < 3 or pos >= limit - 3:
            sides[name] = None
            continue
        Rw = int(min(Rw, pos - 1, limit - pos - 3))
        if Rw < 8:
            sides[name] = None
            continue
        pts, cands = [], []
        for k in range(K_SEGMENTS):
            lo_k, hi_k = int(a + k * seg), int(a + (k + 1) * seg)
            p_eff, m = _side_profile(Dr, axis, pos, max(0, lo_k),
                                     min(along, hi_k), Rw)
            if p_eff is None:
                continue
            t, ev, alts = _find_edge(p_eff, m, outward, Rw, base)
            if t is None:
                continue
            shift = pos - origin
            pts.append(((lo_k + hi_k) / 2, t + shift, ev))
            cands.append(((lo_k + hi_k) / 2,
                          [(tt + shift, ee, ok) for tt, ee, ok in alts]))
        if len(pts) < 3:
            sides[name] = None
            continue
        # The line the whole side has, where there is one: see `_agreed`.
        # It is the same candidates and the same fit, chosen together
        # instead of one segment at a time.
        agreed = _agreed(cands, outward)
        if agreed is not None and len(agreed) >= 3 and (
                np.median([q[2] for q in agreed])
                >= np.median([q[2] for q in pts])):
            # **And not weaker than what it replaces.** The evidence of each
            # ramp is measured and was never used to choose; a line the
            # whole side has is the edge when it is also at least as strong
            # as the ramps picked one segment at a time. On SDIM0903 it was
            # two to eight times stronger (0.30-0.35 against 0.04-0.16).
            pts = agreed
        centre = (lo + hi) / 2
        sides[name] = _fit_line(pts, centre) | dict(cands=cands, centre=centre,
                                                    pts=pts)
    return sides


# What the roll may say about a frame, and when it is listened to.
W_WIDE = 1.03      # wider than the gate: a side is wrong
W_NARROW = 0.90    # narrower than this: the picture is cut, or a side is wrong
# Half-res px the gap may differ from the roll's before the scan is **held
# for a look**. It is not a tolerance on the correction any more: with the
# *same gap* box ticked the roll's gap is given to every row, full stop.
# A tolerance there could only ever mean "these rows are near enough to the
# roll's to be left alone", which is a statement about the camera, and on
# the one half-frame roll in the corpus it is false -- the placed gaps run
# from 166.4 px to 231.1. At 20 it also never fired on that roll, so the
# box looked broken: SDIM0916 and SDIM0917 came out 17.8 px apart with it
# ticked, both inside a window 40 scan px wide.
GAP_TOL = 20       # half-res px before the gap is worth saying out loud
H_TOL = 0.015
ANGLE_SPREAD = 0.15
RESID_TOL = 4.0    # half-res px of disagreement between one side's segments
WEAK = 0.03
ARBITRATE = 0.04   # share of the size within which the roll may choose a ramp
DISPUTE = 0.03     # a frame this far from the roll's size has a side in dispute
CLEAR_GAP = 0.15   # D over the base: a strip this close to it can be unexposed film
CLEAR_STEP = 0.08  # and the other strip this much denser, or the two do not tell


def _clearer_side(Dr, base, f, at, excess):
    """Of a frame too wide by `excess` half-res px, the side whose last
    `excess` px look like unexposed film -- the gap to the next frame --
    or None when the two strips do not tell.

    **What is too much is the gap, so look at it.** A frame measured wider
    than the gate has one side on the far edge of the gap between frames,
    and the strip that side added is film nobody exposed: the least dense
    thing on the negative. On the right side the same strip is picture.
    Neither the step's strength nor whether a ramp has picture inside says
    which: on one roll the wrong side was the stronger step
    and its ramp had picture inside, and both rules dropped the side that
    was right on six frames in seven (DSC_0417 lost 280 scan px of wall).
    Read as the median density over the base, in the middle 80 % of the
    frame's height so the corners stay out."""
    if base is None or excess < 4 or at.get("L") is None or at.get("R") is None:
        return None
    y0 = int(f["y0"] + 0.1 * f["h"])
    y1 = int(f["y0"] + 0.9 * f["h"])
    e = int(round(excess))
    over = {}
    for name, x0, x1 in (("L", at["L"], at["L"] + e), ("R", at["R"] - e, at["R"])):
        x0, x1 = max(0, int(round(x0))), min(Dr.shape[1], int(round(x1)))
        strip = Dr[max(0, y0):y1, x0:x1]
        if strip.size < 50:
            return None
        over[name] = float(np.median(strip)) - base
    lo, hi = sorted(over.values())
    # one strip near the base and the other well above it; not two
    # pictures. Near, not at: the gap on carrete_4 is fogged, 0.09-0.10 D
    # over the scan's clearest film, against 0.27 for the picture's strip
    # (DSC_0417)
    if lo > CLEAR_GAP or hi - lo < CLEAR_STEP:
        return None
    return min(over, key=over.get)


def refine(planes, found, roll=None):
    """Every side at half resolution, then the roll's say, then completion.

    `found` is what `coarse` returned; `roll` is None or dict(h, w, gap) in
    scan pixels. Returns dict(angle, parts, sides, flags, held) -- `held`
    being the flags that mean a person should look.
    """
    D2 = _density(planes.g2)
    classes = found["classes"]
    direct_lvl = classes["direct_level"] if classes["direct"] else None
    film = D2[(D2 < OPAQUE_END)
              & ((D2 >= direct_lvl) if direct_lvl is not None else True)]
    # The film's clearest level: what a gap between frames sits at.
    base = float(np.percentile(film, 1)) if film.size else None
    a = found["angle"]
    Dr = _rotate(D2, a)
    H, W = D2.shape
    sc = planes.scan[0] / W
    cx0, cy0 = (W - 1) / 2, (H - 1) / 2
    r = math.radians(a)
    co, si = math.cos(r), math.sin(r)
    flags, held = [], []
    frames = []
    for cx, cy, w, h in sorted(found["parts"]):
        x, y = (cx + 0.5) / sc - 0.5, (cy + 0.5) / sc - 0.5
        dx, dy = x - cx0, y - cy0
        rx = cx0 + dx * co + dy * si
        ry = cy0 - dx * si + dy * co
        ww, hh = w / sc, h / sc
        f = dict(x0=int(round(rx - ww / 2)), y0=int(round(ry - hh / 2)),
                 w=int(round(ww)), h=int(round(hh)))
        f["sides"] = _refine_sides(Dr, f["x0"], f["y0"], f["w"], f["h"], base)
        frames.append(f)

    # A row shares one top line and one bottom line: pool the segments, so
    # the frame whose edge is under the holder takes its neighbour's.
    #
    # **In the picture's coordinates, not each frame's.** Every segment's
    # position is measured against its own frame's coarse side, and the
    # coarse pass need not have given the two frames of a row the same one.
    # Pooled as they came, the fit mixed two origins and each frame then
    # read the result back against its own: a line called joint that put
    # the two tops exactly `y0_a - y0_b` apart. On SDIM0917 of roma-half
    # that is 53 px -- a visible step between two halves of one piece of
    # film, which cannot be. So they are shifted to absolute first, and the
    # side is marked as already absolute for whoever reads it.
    if len(frames) == 2:
        for name in "TB":
            base_of = {id(f): (f["y0"] if name == "T" else f["y0"] + f["h"])
                       for f in frames}
            pts = [(s_k, t + base_of[id(f)], ev)
                   for f in frames if f["sides"][name]
                   for s_k, t, ev in f["sides"][name]["pts"]]
            if len(pts) >= 3:
                centre = (frames[0]["x0"] + frames[1]["x0"] + frames[1]["w"]) / 2
                joint = _fit_line(pts, centre) | dict(cands=[], centre=centre,
                                                      pts=pts, absolute=True)
                for f in frames:
                    f["sides"][name] = joint

    # The angle: the median tilt of every measured side of every frame.
    angs = []
    for f in frames:
        for name in "LR":
            if f["sides"][name]:
                angs.append(math.degrees(math.atan(f["sides"][name]["slope"])))
        for name in "TB":
            if f["sides"][name]:
                angs.append(-math.degrees(math.atan(f["sides"][name]["slope"])))
    da = -float(np.median(angs)) if angs else 0.0
    if len(angs) >= 2:
        spread = float(np.median(np.abs(np.array(angs) - np.median(angs))))
        if spread > ANGLE_SPREAD:
            flags.append(f"angle-spread {spread:.2f}")

    h_roll = roll["h"] / sc if roll and roll.get("h") else None
    w_roll = roll["w"] / sc if roll and roll.get("w") else None
    gap_roll = roll["gap"] / sc if roll and roll.get("gap") else None

    # Arbitration. Where a window holds several ramps, innermost is the
    # wrong rule as often as not -- the picture's own lines near an edge
    # are ramps too. A side is in dispute when its segments disagree or the
    # frame is off the roll's size; then the opposite side plus that size
    # says where this one should be, and the nearest candidate with
    # picture inside it wins. Only disputed sides, and only from the
    # first measurements, so one wrong side cannot drag the other after it.
    for f in frames:
        sd = f["sides"]
        first = {k: (None if v is None else v["offset"]) for k, v in sd.items()}
        origin = {"L": f["x0"], "R": f["x0"] + f["w"],
                  "T": f["y0"], "B": f["y0"] + f["h"]}
        # Same care as below: a pooled side's offset is already where it
        # is, not how far it moved from the coarse edge.
        at = {k: (None if first[k] is None else
                  first[k] if (sd[k] or {}).get("absolute")
                  else origin[k] + first[k])
              for k in "LRTB"}
        w_off = ((at["R"] - at["L"]) / w_roll
                 if at["L"] is not None and at["R"] is not None and w_roll else 1.0)
        h_off = ((at["B"] - at["T"]) / h_roll
                 if at["T"] is not None and at["B"] is not None and h_roll else 1.0)
        # **Too wide for one size: try both sides, move one.** With *every
        # frame the same size* ticked the gate is known to the height's
        # tolerance, and a frame wider than that has one side wrong --
        # DSC_0313 of carrete_2 read 2.5 % wide, its left side on the
        # previous frame's edge across a pale, low-contrast gap, under the
        # 3 % that puts a side in dispute. Which side is the question, and
        # the evidence of the ramp chosen is no answer: the wrong one there
        # is the stronger step. So each side is put where the other side
        # and the roll's width say, and the one that lands on a ramp more
        # segments agree on is the one moved.
        if roll and roll.get("same_size") and w_roll and w_off > 1 + H_TOL \
                and len(frames) == 1:
            tries = []
            for name, opp, sign in (("L", "R", -1), ("R", "L", +1)):
                if not sd[name] or at[opp] is None or not sd[name].get("cands"):
                    continue
                expected = at[opp] + sign * w_roll - origin[name]
                pts = []
                for s_k, alts in sd[name]["cands"]:
                    pool = [x for x in alts if x[2]]
                    if not pool:
                        continue
                    t, ev, _ = min(pool, key=lambda x: abs(x[0] - expected))
                    if abs(t - expected) <= H_TOL * w_roll:
                        pts.append((s_k, t, ev))
                if len(pts) >= 3:
                    tries.append((len(pts), float(np.mean([q[2] for q in pts])),
                                  name, pts))
            if tries:
                _n, _ev, name, pts = max(tries)
                new = _fit_line(pts, sd[name]["centre"]) | dict(
                    cands=sd[name]["cands"], centre=sd[name]["centre"])
                flags.append(f"wide {name} "
                             f"{(new['offset'] - sd[name]['offset']) * sc:+.0f}")
                sd[name] = new
                first[name] = new["offset"]
                at[name] = (new["offset"] if new.get("absolute")
                            else origin[name] + new["offset"])
                w_off = (at["R"] - at["L"]) / w_roll
            elif _clearer_side(Dr, base, f, at, (w_off - 1.0) * w_roll):
                name = _clearer_side(Dr, base, f, at, (w_off - 1.0) * w_roll)
                flags.append(f"wide {name} dropped, clear")
                sd[name] = None
                first[name] = at[name] = None
                w_off = 1.0
            else:
                # No ramp where the width says: then the side whose own
                # ramp mostly has no picture inside it is not this frame's
                # edge but the next one's, seen across the gap (DSC_0313:
                # three of five segments, the coarse sweep 109 px further
                # in and right). It is dropped, and completed below from
                # the other side and the roll's width.
                def pic(nm):
                    got = [min(a, key=lambda x: abs(x[0] - first[nm]))[2]
                           for _k, a in sd[nm]["cands"] if a]
                    return sum(got) / len(got) if got else 1.0
                share = {nm: pic(nm) for nm in "LR"
                         if sd[nm] and sd[nm].get("cands")}
                if share:
                    name = min(share, key=share.get)
                    if share[name] < 0.5:
                        flags.append(f"wide {name} dropped")
                        sd[name] = None
                        first[name] = at[name] = None
                        w_off = 1.0
        for name, opp, size, sign, size_off in (
                ("L", "R", w_roll, -1, w_off), ("R", "L", w_roll, +1, w_off),
                ("T", "B", h_roll, -1, h_off), ("B", "T", h_roll, +1, h_off)):
            if not size or not sd[name] or first[opp] is None \
                    or not sd[name].get("cands"):
                continue
            if not (sd[name]["resid"] > RESID_TOL or abs(size_off - 1.0) > DISPUTE):
                continue
            expected = at[opp] + sign * size - origin[name]
            pts = []
            for s_k, alts in sd[name]["cands"]:
                pool = [x for x in alts if x[2]] or alts
                t, ev, _ = min(pool, key=lambda x: abs(x[0] - expected))
                if abs(t - expected) <= ARBITRATE * size:
                    pts.append((s_k, t, ev))
            if len(pts) >= 3:
                new = _fit_line(pts, sd[name]["centre"]) | dict(
                    cands=sd[name]["cands"], centre=sd[name]["centre"])
                if abs(new["offset"] - sd[name]["offset"]) > 1.0:
                    flags.append(f"arbitrated {name} "
                                 f"{(new['offset'] - sd[name]['offset']) * sc:+.0f}")
                sd[name] = new

    # Positions; then what the roll says about them; then completion.
    for f in frames:
        sd = f["sides"]
        f["L"] = f["x0"] + sd["L"]["offset"] if sd["L"] else None
        f["R"] = f["x0"] + f["w"] + sd["R"]["offset"] if sd["R"] else None
        # Read at **this frame's** middle, not at the middle of whatever
        # the line was fitted over (`_at`), and a pooled side is already in
        # the picture's coordinates, so adding the frame's own origin to it
        # again would split the row a second way.
        own = f["x0"] + f["w"] / 2
        tilt = math.tan(math.radians(da))
        f["T"] = (None if not sd["T"] else
                  _at(sd["T"], own, tilt) if sd["T"].get("absolute")
                  else f["y0"] + _at(sd["T"], own))
        f["B"] = (None if not sd["B"] else
                  _at(sd["B"], own, tilt) if sd["B"].get("absolute")
                  else f["y0"] + f["h"] + _at(sd["B"], own))
        for name in "LRTB":
            if sd[name] and sd[name]["resid"] > RESID_TOL:
                flags.append(f"segments {name} {sd[name]['resid']:.0f}")
            if sd[name] and sd[name]["evidence"] < WEAK:
                flags.append(f"weak {name} {sd[name]['evidence']:.3f}")
    for i, f in enumerate(frames):
        if f["L"] is not None and f["R"] is not None and w_roll:
            wm = f["R"] - f["L"]
            if wm > W_WIDE * w_roll:
                # A frame cannot be wider than the gate: one side is wrong.
                # In a row it is the outer one, which sits by the holder; on
                # its own, the one with less evidence.
                if len(frames) == 2:
                    outer = "L" if i == 0 else "R"
                else:
                    outer = _clearer_side(Dr, base, f, f, wm - w_roll) or (
                        "L" if (f["sides"]["L"]["evidence"]
                                < f["sides"]["R"]["evidence"]) else "R")
                held.append(f"wide {wm / w_roll:.3f}, dropped {outer}")
                f[outer] = None
            elif wm < W_NARROW * w_roll:
                held.append(f"narrow {wm / w_roll:.3f}")
    if len(frames) == 2 and gap_roll:
        a_, b_ = frames
        if a_["R"] is not None and b_["L"] is not None \
                and abs((b_["L"] - a_["R"]) - gap_roll) > GAP_TOL:
            held.append(f"gap {(b_['L'] - a_['R']) * sc:.0f}")
    # **A half-frame scan is one width.** Its two exposures are one
    # advance of the film past one gate, so they are the same size: on the
    # 37 rows of roma-half placed by hand the two widths differ by a median
    # of 0.0 px and never by more than 8.2. The two *heights* already come
    # out equal, because both frames read one pooled pair of lines; the two
    # widths are four separate sides, and measured they came out a median
    # 7.4 px apart and as much as 85 (SDIM0932) -- which is exactly what
    # one half laid over the other shows.
    #
    # So the two measured widths are two readings of one number, and the
    # row's width is their mean. It is the same move as pooling the top and
    # bottom segments of both halves into one line, on the other axis. A
    # half with a side it could not measure does **not** vote: the width it
    # is about to be given is the roll's upper bound, not a reading of this
    # row. On SDIM0924 that completed side took the roll's 2539.2 while its
    # sibling measured 2501.2 and the hand placed 2507.1.
    w_row = None
    if len(frames) == 2:
        read = [f["R"] - f["L"] for f in frames
                if f["L"] is not None and f["R"] is not None]
        # With neither half measured -- a blank exposure -- the roll is all
        # there is, and it is better than the coarse rectangle the last
        # resort would otherwise leave on one of them (SDIM0932: 2624
        # against the roll's 2539).
        w_row = float(np.mean(read)) if read else w_roll
    for i, f in enumerate(frames):
        other = frames[1 - i] if len(frames) == 2 else None
        missing = [n for n in "LRTB" if f[n] is None]
        if missing:
            f["completed"] = "".join(missing)
            tag = f" frame {'ab'[i]}" if len(frames) == 2 else ""
            (held if len(missing) >= 3 or (other is None and "L" in missing
                                           and "R" in missing)
             else flags).append("no-edge " + "".join(missing) + tag)
        if f["T"] is None and f["B"] is None:
            f["T"], f["B"] = f["y0"], f["y0"] + f["h"]
        elif f["T"] is None:
            f["T"] = f["B"] - (h_roll or f["h"])
        elif f["B"] is None:
            f["B"] = f["T"] + (h_roll or f["h"])
        wr = w_row or w_roll or f["w"]
        if f["L"] is None and f["R"] is None:
            if other is not None and gap_roll and i == 0 and other["L"] is not None:
                f["R"] = other["L"] - gap_roll
                f["L"] = f["R"] - wr
            elif other is not None and gap_roll and i == 1 and other["R"] is not None:
                f["L"] = other["R"] + gap_roll
                f["R"] = f["L"] + wr
            else:
                f["L"], f["R"] = f["x0"], f["x0"] + f["w"]
        # **A missing inner edge is completed from the neighbour and the
        # roll's gap, not from this frame's own other side and the width**,
        # and that order was measured rather than argued. The argument for
        # turning it round is good: the row's width is now a measurement
        # (the two halves a person placed are the same width to a median of
        # 0.0 px) while the roll's gap is **not a constant at all** -- over
        # the 37 placed rows of roma-half it runs from 166.4 to 231.1 px, a
        # median absolute deviation of 9.9 and a range of 65, because the
        # camera's advance varies. Swapped, the gap against the placed
        # frames gets **worse**: |error| p90 26.3 -> 34.9 px and the worst
        # 55.3 -> 59.0, on the same 37 rows. What it exposed is that on
        # SDIM0917 one of the two *measured* left edges is itself about
        # 47 px out, and the roll's gap had been covering part of it. A
        # completion cannot mend a bad measurement; leaning on the frame's
        # own bad edge only spends the cover.
        elif f["L"] is None:
            if other is not None and gap_roll and i == 1 and other["R"] is not None:
                f["L"] = other["R"] + gap_roll
            else:
                f["L"] = f["R"] - wr
        elif f["R"] is None:
            if other is not None and gap_roll and i == 0 and other["L"] is not None:
                f["R"] = other["L"] - gap_roll
            else:
                f["R"] = f["L"] + wr
    # The row's one width, given to both halves. About each frame's own
    # centre, both sides equally, for the reason the height is: the row
    # knows how far apart the two edges are and nothing about which of them
    # is the one sitting wrong.
    if w_row and len(frames) == 2:
        off = max(abs((f["R"] - f["L"]) - w_row) for f in frames)
        if off > 0.5:
            for f in frames:
                mid = (f["L"] + f["R"]) / 2
                f["L"], f["R"] = mid - w_row / 2, mid + w_row / 2
            flags.append(f"w-row {off * sc:.0f}")

    # **The height is the row's, not the frame's.** The two exposures of a
    # half-frame row are on one piece of film and cannot be different
    # heights; and the fitted top and bottom lines of a row are not exactly
    # parallel, so read at each frame's own middle their heights differ by
    # a few tenths of a per cent. Deciding per frame then fired on one half
    # and not the other and left them 26 px apart on SDIM0917 -- a rule for
    # keeping a roll one size making the two halves of one row two sizes.
    have = float(np.mean([q["B"] - q["T"] for q in frames])) if frames else 0.0
    if frames and h_roll and abs(have - h_roll) > H_TOL * h_roll:
        flags.append(f"h-off {have / h_roll:.3f}")
        if roll and roll.get("same_size"):
            # **One gate, one height.** Measured on the frames a person
            # placed, the heights within a roll's size cluster agree to
            # 0.14-0.45 % (median of |delta|) and are never more than
            # 1.81 % apart, on all four labelled rolls. A frame further
            # off than that is a side misplaced, not a frame of another
            # size, so it is set to the roll's height. About the centre,
            # both sides equally: which of the two is the wrong one is
            # exactly what is not known here -- the roll knows the
            # distance between them and nothing about where they sit.
            #
            # The **width is left alone**, and that is not an oversight:
            # a frame can be genuinely narrower because the exposure
            # ended before the gate did. 008 of portra400-full is 7.25 %
            # narrow with its height right, so a roll's width is an
            # upper bound and forcing it would invent 368 px of picture.
            for q in frames:
                mid = (q["T"] + q["B"]) / 2
                q["T"], q["B"] = mid - h_roll / 2, mid + h_roll / 2
            flags.append("h-set")

    # **One gap for the whole roll**, when the folder says so
    # (`roll.same_gap`, the *Half-frame rows all have the same gap* box).
    # The two exposures of a half-frame row are one advance of the film
    # past one gate, so the unexposed strip between them is the same on
    # every row. **Every** row is given it, about the gap's own centre:
    # each frame keeps the width it measured and moves half the
    # correction, because which of the two inner edges is the wrong one is
    # not known here. Only the inner edges are evidence about the gap; the
    # outer ones sit by the holder.
    #
    # **No tolerance.** It used to act only on a row more than `GAP_TOL`
    # (20 half-res px, 40 scan px) from the roll's, which meant the box
    # promised one gap and delivered a 40-px-wide window: on roma-half the
    # rows run from -34.1 to +25.1 against the roll's, so it fired on one
    # scan of 37 and two neighbours sat 17.8 px apart with it ticked. A
    # tolerance here can only mean "near enough to the roll's is the
    # roll's", which is a claim about the camera, not about this row. The
    # half-pixel below is a no-op guard, not a tolerance.
    if len(frames) == 2 and gap_roll and roll and roll.get("same_gap"):
        a_, b_ = frames
        if None not in (a_["L"], a_["R"], b_["L"], b_["R"]):
            have = b_["L"] - a_["R"]
            if abs(have - gap_roll) > 0.5:
                mid = (a_["R"] + b_["L"]) / 2
                wa, wb = a_["R"] - a_["L"], b_["R"] - b_["L"]
                a_["R"] = mid - gap_roll / 2
                a_["L"] = a_["R"] - wa
                b_["L"] = mid + gap_roll / 2
                b_["R"] = b_["L"] + wb
                flags.append(f"gap-set {(have - gap_roll) * sc:+.0f}")

    parts, sides = [], []
    for f in frames:
        ncx, ncy = (f["L"] + f["R"]) / 2 + 0.5, (f["T"] + f["B"]) / 2 + 0.5
        ddx, ddy = ncx - cx0, ncy - cy0
        sx = cx0 + ddx * co - ddy * si
        sy = cy0 + ddx * si + ddy * co
        parts.append(((sx + 0.5) * sc - 0.5, (sy + 0.5) * sc - 0.5,
                      (f["R"] - f["L"]) * sc, (f["B"] - f["T"]) * sc))
        sides.append({k: (None if v is None else round(v["evidence"], 3))
                      for k, v in f["sides"].items()}
                     | ({"completed": f["completed"]} if f.get("completed") else {}))

    # **A frame cannot be off the film.** A side nobody could measure is
    # filled in from the roll or, in the last resort, from the coarse
    # rectangle -- and the coarse search runs on the *rotated* plane, which
    # is padded, so a rectangle that is inside it can land outside the scan
    # once it is turned back. On SDIM0932 of roma-half, the blank exposure
    # where six of the eight sides have no edge to measure, one half came
    # out 720 px off the left of the scan: a quarter of its width written
    # as nothing.
    #
    # So the row is slid back on, by the least that does it, and only along
    # an axis whose sides were *completed*: moving a side somebody measured
    # would be the roll correcting evidence, which is the one thing it may
    # not do. Both frames move together, so the gap between them is kept.
    filled = "".join(f.get("completed") or "" for f in frames)
    if parts and filled:
        ang = a + da
        for axis, letters, span in ((0, "LR", planes.scan[0]),
                                    (1, "TB", planes.scan[1])):
            if not any(k in filled for k in letters):
                continue
            cs = [c for cx, cy, w, h in parts
                  for c in findframe.corners(cx, cy, w, h, ang)]
            lo = min(c[axis] for c in cs)
            hi = max(c[axis] for c in cs)
            if hi - lo > span:            # wider than the scan: nothing to do
                held.append(f"off-scan {letters} {hi - lo - span:.0f} too wide")
                continue
            shift = -lo if lo < 0 else (span - hi if hi > span else 0.0)
            if abs(shift) < 1:
                continue
            flags.append(f"on-scan {letters} {shift:+.0f}")
            parts = [(cx + (shift if axis == 0 else 0.0),
                      cy + (shift if axis == 1 else 0.0), w, h)
                     for cx, cy, w, h in parts]

    return dict(angle=a + da, parts=parts, sides=sides, flags=flags,
                held=held, base=None if base is None else round(base, 3))


# ---------------------------------------------------------------- the roll

def _clusters(vals, tol=0.03, min_n=3):
    """Groups of heights within `tol` of their group's median, biggest
    groups only. A roll whose copy stand moved partway holds two sizes --
    the Cinestill roll: 18 scans at 2722 px and 10 at 3148."""
    groups = []
    for v in sorted(vals):
        if groups and abs(v - np.median(groups[-1])) <= tol * np.median(groups[-1]):
            groups[-1].append(v)
        else:
            groups.append([v])
    big = [g for g in groups if len(g) >= min_n]
    return big or [max(groups, key=len)]


def perforations(g2, pitch):
    """Where the scan's perforations are, in scan px: the phase of the
    holes along the film (0..pitch), and the inner edge of the top and of the
    bottom row of holes across it. None where a row is not found.

    The holes are the brightest columns of each long edge's band (they
    show the light table); their centres give the phase, as a circular
    mean so a hole cut by the scan's edge does not matter, and the rows
    of those columns that are bright give the row."""
    from scipy import ndimage as ndi
    H, W = g2.shape
    out = {"phase": None, "top": None, "bottom": None}
    angles = []
    for name, y0, y1 in (("top", 0, H // 8), ("bottom", H - H // 8, H)):
        band = g2[y0:y1]
        colp = np.percentile(band, 90, axis=0)
        cols = colp > (np.percentile(colp, 95) + np.median(colp)) / 2
        lab, k = ndi.label(cols)
        cs = [float(np.mean(np.nonzero(lab == i)[0])) * 2.0
              for i in range(1, k + 1) if (lab == i).sum() > 10]
        if len(cs) >= 3:
            angles += list(np.exp(2j * np.pi * np.array(cs) / pitch))
        if cols.any():
            rowp = np.median(band[:, cols], axis=1)
            ys = np.nonzero(rowp > (rowp.max() + np.median(band)) / 2)[0]
            if len(ys):
                # the row's edge that faces the frame: the outer one is
                # where the scan cuts the holes, and how much it cuts
                # moves with the framing (carrete_4 cuts the top row in
                # half)
                out[name] = (y0 + (ys.max() if name == "top" else ys.min())) * 2.0
    if angles:
        out["phase"] = float(np.angle(np.mean(angles)) * pitch / (2 * np.pi) % pitch)
    return out


def roll_scale(results):
    """The roll's scale in scan px per mm, off its scans' perforation
    pitch (`sprocket_pitch`), or None: when fewer than half the scans
    show a row of holes (a holder that covers them -- cinestill and
    portra400 of the corpus, 0 of 51), or when the pitches disagree by
    more than `PITCH_SPREAD`, which is a copy stand moved partway."""
    have = [r for r in results.values() if r]
    pitches = [r["pitch"] for r in have if r.get("pitch")]
    if not have or len(pitches) < max(3, len(have) / 2):
        return None
    med = float(np.median(pitches))
    if (np.percentile(pitches, 90) - np.percentile(pitches, 10)) > PITCH_SPREAD * med:
        return None
    return med / PITCH_MM


def gate_cluster(heights, px_mm):
    """The roll's frame height, chosen with the film's own scale: among the
    measured heights that a 135 exposure can have (`GATE_H_MM`, 24 mm
    across the film), the **smallest** tight group of at least three.
    None when there is no such group.

    **Not the most frequent.** The frames mismeasured on a roll are not
    noise around the right size: they take in the same band of unexposed
    film every time -- the strip down to the perforations, the gap to the
    next frame -- so they are as tight a group as the right ones and can
    outnumber them. On one roll (D5600, 146.5 px/mm), alone,
    six scans read 3564-3586 px (24.4 mm, right), five 3627-3642 (55 px
    of rebate at the bottom), twelve 3689-3721 (25.3 mm) and the rest more;
    the 3 % clusters chained them into one at 3696 and every frame was set
    to it. On carrete_2 the wrong groups were 3715 and 3857 against 3580.
    The errors go outward, because the bands outside a frame are flat and
    their edges strong; inward the detector errs 4-8 px (the hand-placed
    corpus), which never makes a group of its own."""
    lo, hi = (v * px_mm for v in GATE_H_MM)
    hs = sorted(h for h in heights if lo <= h <= hi)
    groups = []
    for h in hs:
        if groups and h - groups[-1][0] <= GATE_TOL * groups[-1][0]:
            groups[-1].append(h)
        else:
            groups.append([h])
    groups = [g for g in groups if len(g) >= 3]
    return float(np.median(groups[0])) if groups else None


def roll_statistics(results, want="auto"):
    """What the first pass says about the roll: its format and, per size
    cluster, the median height, the 75th-percentile width (most frames are
    the gate's full width; a few are cut), the median row position and the
    median gap between the two frames of a row.

    `results` maps scan name -> the first-pass record (format, parts,
    pair). Returns (format, [cluster dicts]) or (None, []) when there is
    nothing to say.
    """
    have = {n: r for n, r in results.items() if r and r.get("parts")}
    if not have:
        return None, []
    if want in ("full", "half"):
        fmt = want
    else:
        # A tie goes to half frame, because the first pass only ever errs
        # one way. Alone, on the 88 labelled scans: 51 full-frame scans,
        # none read as half; 37 half-frame scans, 7 read as full -- a blank
        # half has no partner to pair with, and two halves and their gap
        # make a 3:2. A half reading needs a gap found between two
        # pictures; a full reading is what is left when none is.
        # It was `max(set(fmts), key=fmts.count)`, and on a tie a set's
        # order is the string hash, which Python seeds afresh per process:
        # nikon_55mm, two scans with a blank half and two without, came out
        # half or full depending on the run.
        fmts = [r["format"] for r in have.values()]
        fmt = max(sorted(set(fmts)),
                  key=lambda f: (fmts.count(f), f == "half"))
    good = {n: r for n, r in have.items() if r["format"] == fmt}
    if not good:
        return fmt, []
    stats = {}
    for n, r in good.items():
        ps = sorted(r["parts"])
        stats[n] = dict(
            h=float(np.median([p[3] for p in ps])),
            w=float(np.median([p[2] for p in ps])),
            y0=float(np.median([p[1] - p[3] / 2 for p in ps])),
            gap=((ps[1][0] - ps[1][2] / 2) - (ps[0][0] + ps[0][2] / 2))
            if len(ps) == 2 and r.get("pair") else None)
    clusters = []
    px_mm = roll_scale(good)
    gate = gate_cluster([s["h"] for s in stats.values()], px_mm) if px_mm else None
    if gate is not None:
        # one size, the film's: the members are the frames within the
        # tight tolerance of it, and nothing else speaks for the roll
        members = [n for n, s in stats.items() if abs(s["h"] - gate) <= 1.5 * GATE_TOL * gate]
        gaps = [stats[n]["gap"] for n in members if stats[n]["gap"] is not None]
        return fmt, [dict(
            format=fmt, h=gate,
            w=float(np.percentile([stats[n]["w"] for n in members], 75)),
            y0=float(np.median([stats[n]["y0"] for n in members])),
            gap=float(np.median(gaps)) if gaps else None,
            n=len(members), px_per_mm=round(px_mm, 3),
            mm=round(gate / px_mm, 2))]
    for g in _clusters([s["h"] for s in stats.values()]):
        hc = float(np.median(g))
        members = [n for n, s in stats.items() if abs(s["h"] - hc) <= 0.03 * hc]
        gaps = [stats[n]["gap"] for n in members if stats[n]["gap"] is not None]
        clusters.append(dict(
            format=fmt, h=hc,
            w=float(np.percentile([stats[n]["w"] for n in members], 75)),
            y0=float(np.median([stats[n]["y0"] for n in members])),
            gap=float(np.median(gaps)) if gaps else None,
            n=len(members)))
    return fmt, clusters


def roll_from_document(items, want="auto"):
    """What the roll says, read back off the records already in the
    document, in the shape `roll_statistics` returns.

    A document written before the roll's own summary was kept has no
    `roll.rolled`, and a scan measured again on its own is measured with
    none of what the roll knows -- which on half frame reads the whole
    strip as one full frame, because two halves and their gap make a 3:2.
    The document holds every frame's size and position, which is the same
    evidence the first pass produced, so it can be read back. It is better
    evidence, in fact: the frames a person corrected by hand are in it.
    """
    results = {}
    for name, it in items.items():
        parts = it.get("parts") or []
        if not parts or it.get("state") == "skipped":
            continue
        results[name] = {
            "format": parts[0].get("format", "full"),
            "parts": [(p["centre"][0], p["centre"][1], p["size"][0],
                       p["size"][1]) for p in parts],
            "pair": len(parts) > 1}
    fmt, clusters = roll_statistics(results, want)
    return {"format": fmt, "clusters": clusters} if clusters else None


def cluster_for(clusters, h_own, same_size=False):
    """The size cluster a scan belongs to, or the biggest one.

    **With `same_size` it is always the biggest one.** The box says *every
    frame is the same size*, which is one size, not one per cluster. Read
    per cluster, a frame mismeasured into the rebate found company: on
    one roll (D5600, stand untouched) the first pass made
    three clusters, 3580 px (21 frames), 3715 (12) and 3857 (8), and the
    box then set DSC_0312 to 3716 and arbitrated DSC_0314's right side out
    by 461 px -- the roll confirming the error it should have corrected.
    A roll whose stand moved (the Cinestill roll, 2722 and 3148 px) is
    exactly a roll where the box is false, so it is not ticked there."""
    if same_size:
        return max(clusters, key=lambda c: c["n"])
    if h_own is not None:
        near = [c for c in clusters if abs(c["h"] - h_own) <= 0.03 * c["h"]]
        if near:
            return near[0]
    return max(clusters, key=lambda c: c["n"])


# ---------------------------------------------------------------- records

def record(planes, found, refined, by="fit", roll=None):
    """An item record for the document, from one scan's coarse and refined
    measurements. `state` is `held` when the roll had to supply a side or
    contradicted one -- the cases a person should look at."""
    w, h = planes.scan
    if found is None:
        return {"scan": [w, h], "angle": 0.0, "state": "refused", "by": by,
                "evidence": {"reason": "no rectangle of a film's shape "
                                       "clear of light table and holder"},
                "parts": [], "manual": []}
    parts = []
    for i, ((cx, cy, pw, ph), sides) in enumerate(zip(refined["parts"],
                                                        refined["sides"])):
        parts.append({
            "name": "abcdefgh"[i],
            "centre": [round(cx, 1), round(cy, 1)],
            "size": [round(pw, 1), round(ph, 1)],
            "format": found["format"],
            "evidence": {"sides": sides}})
    ev = {"score": round(found["score"], 3),
          "scores": found["scores"],
          "classes": found["classes"],
          "pair": bool(found["pair"]),
          "base": refined["base"],
          "flags": refined["flags"],
          "held": refined["held"]}
    if found.get("gap"):
        ev["gap"] = found["gap"]
    if roll:
        ev["roll"] = {k: (round(v, 1) if isinstance(v, float) else v)
                      for k, v in roll.items() if k != "n"}
    return {"scan": [w, h],
            "angle": round(refined["angle"], 3),
            "state": "held" if refined["held"] else "detected",
            "by": by,
            "evidence": ev,
            "parts": parts,
            "manual": []}


def measure(path, settings=None, roll=None):
    """One scan, start to finish. Returns (item record, first-pass summary)
    -- the summary being what `roll_statistics` reads."""
    settings = settings or {}
    want = settings.get("format", "auto")
    fmts = (want,) if want in ("full", "half") else ("full", "half")
    planes = Planes(path)
    found = coarse(planes, fmts, roll, float(settings.get("min_fill", MIN_FILL)))
    if found is None:
        return record(planes, None, None), None
    refined = refine(planes, found, roll)
    summary = {"format": found["format"], "parts": refined["parts"],
               "pair": found["pair"],
               "pitch": sprocket_pitch(planes.g2)[0] if roll is None else None}
    if summary["pitch"]:
        summary["perf"] = perforations(planes.g2, summary["pitch"])
    return record(planes, found, refined, roll=roll), summary


def _job(args):
    path, settings, roll = args
    return path.name, measure(path, settings, roll)


def _run_pass(paths, settings, rolls, progress, offset, total, workers):
    """One pass over the folder, in parallel when there is more than a
    handful of scans: three seconds a scan on one core is still a couple of
    minutes for a roll of 37, and every core on the machine is otherwise
    idle."""
    jobs = [(p, settings, rolls[p.name]) for p in paths]
    out = {}
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for i, (name, res) in enumerate(pool.imap(_job, jobs)):
                out[name] = res
                if progress:
                    progress(offset + i, total, name)
    else:
        for i, job in enumerate(jobs):
            name, res = _job(job)
            out[name] = res
            if progress:
                progress(offset + i, total, name)
    return out


def measure_roll(paths, settings=None, progress=None, workers=None,
                 summary=None):
    """Every scan twice: alone, then with what the roll says.

    Returns {scan name: item record}. `progress(i, n, name)` is called as
    each scan finishes each pass, for the terminal.

    `summary`, if a dict is passed, is filled with what the first pass
    decided about the roll -- the format and the size clusters. **It is
    kept so one scan can be measured again later and come out the same.**
    Measuring a single scan on its own is the worst case there is: a dark
    frame has no edge to find, and the roll is what supplies the height,
    the gap and the row it should have had. Without this, `⌘D` on one scan
    gave an answer the roll would never have given.
    """
    import os
    settings = settings or {}
    paths = list(paths)
    if workers is None:
        workers = int(settings.get("workers") or min(8, os.cpu_count() or 1))
    n = len(paths)
    first_pass = _run_pass(paths, settings, {p.name: None for p in paths},
                           progress, 0, 2 * n, workers)
    items = {k: v[0] for k, v in first_pass.items()}
    first = {k: v[1] for k, v in first_pass.items()}
    fmt, clusters = roll_statistics(first, settings.get("format", "auto"))
    if summary is not None:
        summary.update(format=fmt, clusters=clusters)
    # `one_roll` off says the folder is not one roll -- different cameras,
    # different films -- so nothing a frame's size or format says about
    # another is evidence, and the first pass is the answer.
    if not clusters or not settings.get("one_roll", True):
        return items
    rolls = {}
    for p in paths:
        own = first[p.name]
        h_own = float(np.median([q[3] for q in own["parts"]])) \
            if own and own["parts"] else None
        c = cluster_for(clusters, h_own, settings.get("same_size", False))
        rolls[p.name] = {"format": fmt, "h": c["h"], "w": c["w"],
                         "gap": c["gap"], "y0": c["y0"],
                         "same_size": bool(settings.get("same_size", False)),
                         "same_gap": bool(settings.get("same_gap", False))}
    second = _run_pass(paths, settings, rolls, progress, n, 2 * n, workers)
    for k, v in second.items():
        items[k] = v[0]
    moved = align_to_perforations(items, first, clusters[0]["h"] if clusters else None,
                                  bool(settings.get("same_size", False)), summary)
    by_name = {p.name: p for p in paths}
    for name in moved:
        snap_to_edges(by_name[name], items[name], moved[name])
    return items


SNAP_REACH = 80.0     # scan px either way a frame placed by the perforations may slide to its edges
SNAP_REACH_Y = 30.0   # scan px: across the film, where the holes already say to ~10
SNAP_GAIN = 1.5       # and only if both edges there step this much more than where it is


def _colour_planes(path):
    """The mosaic's R, G and B sub-planes at half size, as density: what
    the snap reads, because an edge can be there in colour and not in the
    green the rest of the detector works on (carrete_4, DSC_0395: brown
    picture against a grey-blue gap, the same green)."""
    import rawpy
    with rawpy.imread(str(path)) as r:
        v = r.raw_image_visible.astype(np.float32)
        cols = r.raw_colors_visible
        black = float(np.mean(r.black_level_per_channel))
        white = float(r.white_level)
    H2, W2 = v.shape[0] // 2, v.shape[1] // 2
    planes = {}
    for dy in (0, 1):
        for dx in (0, 1):
            c = int(cols[dy, dx])
            c = 1 if c == 3 else c
            if c not in planes:
                planes[c] = _density(np.clip(v[dy:2 * H2:2, dx:2 * W2:2] - black, 1, None)
                                     / (white - black))
    return [planes[c] for c in sorted(planes)], v.shape[1] / W2


def _edge_fit(turned, sc, a, centre, size, axis, signed=False, reach_px=None):
    """(shift, score) along `axis` that puts a frame of `size` centred at
    `centre` (scan px) where both opposite sides sit on a step of the same
    sense, within `SNAP_REACH`; the shift in scan px along the frame's own
    axis. Each colour plane counts in units of its own profile's typical
    gradient. `signed` asks for the picture to be the denser side at both
    ends: the shift is the one where the summed steps say so most, and
    the score is that sum, negative where the gap is inside a side."""
    H, W = turned[0].shape
    cx0, cy0 = (W - 1) / 2, (H - 1) / 2
    r = math.radians(a)
    co, si = math.cos(r), math.sin(r)
    x, y = (centre[0] + 0.5) / sc - 0.5, (centre[1] + 0.5) / sc - 0.5
    dx, dy = x - cx0, y - cy0
    rx, ry = cx0 + dx * co + dy * si, cy0 - dx * si + dy * co
    w, h = size[0] / sc, size[1] / sc
    reach = int(round((reach_px or SNAP_REACH) / sc))
    shifts = np.arange(-reach, reach + 1)
    lo, hi = (rx - w / 2, rx + w / 2) if axis == "x" else (ry - h / 2, ry + h / 2)
    n = W if axis == "x" else H
    if lo - reach - 3 < 0 or hi + reach + 3 >= n:
        return 0.0, 0.0
    both = np.zeros(len(shifts))
    sense = np.zeros(len(shifts))
    for Dr in turned:
        prof = (np.median(Dr[int(ry - 0.35 * h):int(ry + 0.35 * h)], axis=0)
                if axis == "x" else
                np.median(Dr[:, int(rx - 0.35 * w):int(rx + 0.35 * w)], axis=1))
        dp = np.gradient(ndimage.gaussian_filter1d(prof.astype(np.float64), 1.5))
        noise = float(np.median(np.abs(dp))) + 1e-6
        at_lo = np.array([dp[int(round(lo + s_))] for s_ in shifts]) / noise
        at_hi = np.array([-dp[int(round(hi + s_))] for s_ in shifts]) / noise
        both += np.abs(at_lo + at_hi)
        sense += at_lo + at_hi
    if signed:
        k = int(np.argmax(sense))
        return float(shifts[k]) * sc, float(sense[k])
    k = int(np.argmax(both))
    if shifts[k] == 0 or both[k] < SNAP_GAIN * both[reach]:
        k = reach
    return float(shifts[k]) * sc, float(both[k])


def snap_to_edges(path, item, before=None):
    """A frame placed by the perforations slid, at its size, onto its own
    edges -- or, along the film, put back where it was measured if that is
    clearly a frame and the other is not. In place; flagged `snap x`,
    `snap y`, `perf undone`.

    **The perforations put a frame where the roll's frames typically are,
    not where this one is**: on carrete_4 the right ones sit within
    +-50 px of the phase, so a frame moved there lands near its edges and
    not on them: six of the eight it moved on one roll, with the edge plainly
    in sight. So, within `SNAP_REACH`, the frame is
    tried at every shift and kept where **both** opposite sides sit on a
    step of the same sense at once. A line in the picture is a step on one
    side only. Read on all three colour planes (`_colour_planes`), on the
    median across the middle 70 % of the frame, turned by the frame's
    angle. On carrete_4 the frames it was not asked about stay put (0 px
    on DSC_0393, 0402, 0408); DSC_0395 goes +50 px and DSC_0409 +46, where
    the edge is seen.

    **Along the film the holes only propose.** DSC_0429, the roll's last
    frame, sits 190 px off the phase every other frame keeps and was
    measured right, its left side on a black edge plain to see. So where
    it was measured is fitted too, and kept if its two sides are clearly
    the picture's -- denser inside than out at both ends, by `SNAP_GAIN`
    over the proposal. A frame slid by the gap fails that by construction:
    one of its sides has the gap inside it. Scored either way, the
    holder's and the holes' strong steps made the slid frames win. Across
    the film the holes are not questioned: the gate is fixed there."""
    q = item["parts"][0]
    planes, sc = _colour_planes(path)
    a = float(item.get("angle", 0.0))
    turned = [_rotate(P, a) for P in planes]
    r = math.radians(a)
    co, si = math.cos(r), math.sin(r)
    flags = item["evidence"].setdefault("flags", [])
    c = list(q["centre"])
    # **Only along the axes the holes moved.** A frame whose height alone
    # the roll set was measured along the film and is left there:
    # DSC_0427, right as measured, went 76 px off on a fogged frame.
    moved_x = before is None or abs(before[0] - c[0]) > 1.0
    moved_y = before is None or abs(before[1] - c[1]) > 1.0 \
        or any(f.startswith("perf h-set") for f in flags)
    # either way for the proposal's own fit: on the frames fogged by a
    # leak the gap is denser than the picture (DSC_0398, 0416)
    sx = _edge_fit(turned, sc, a, c, q["size"], "x")[0] if moved_x else 0.0
    if moved_x and before is not None:
        _sp, perf_score = _edge_fit(turned, sc, a, c, q["size"], "x", signed=True)
        mine = [before[0], c[1]]
        bx, before_score = _edge_fit(turned, sc, a, mine, q["size"], "x", signed=True)
        if before_score > 0 and before_score > SNAP_GAIN * max(perf_score, 0.0):
            c, sx = mine, bx
            flags.append("perf x undone")
    c = [c[0] + sx * co, c[1] + sx * si]
    # across the film the holes have it to ~10 px (carrete_2 against the
    # hand): only a short way to the edges
    sy = _edge_fit(turned, sc, a, c, q["size"], "y", reach_px=SNAP_REACH_Y)[0] if moved_y else 0.0
    q["centre"] = [round(c[0] - sy * si, 1), round(c[1] + sy * co, 1)]
    for axis, d in (("x", sx), ("y", sy)):
        if d:
            flags.append(f"snap {axis} {d:+.0f}")


PERF_X_TOL = 60.0     # scan px: a frame further than this off the roll's phase is misplaced
PERF_Y_TOL = 25.0     # scan px: and across the film
PERF_SPREAD = 25.0    # scan px: the roll's own frames must agree this well for either to apply
PERF_AGREE = 0.6      # and this share of them be within the tolerance


def _wrap(d, p):
    return (d + p / 2) % p - p / 2


def align_to_perforations(items, first, h_roll, same_size, summary=None):
    """Every full frame put where the roll's perforations say, when the
    roll's frames agree on it. In place; flagged `perf x`, `perf y`.

    **A camera advances the film by its perforations**, eight holes a
    frame, on a sprocket, so on one roll every frame starts at the same
    place among the holes; and its gate is fixed across the film, so
    every frame sits at the same height from the rows of holes. Both are
    lengths on the film, which the scan's framing and the scene do not
    move. On one roll, the frames that were right sat at
    466-564 px of the 696 px perforation phase and 2-20 px from the
    middle of the rows; the eight a person called wrong sat at 19-102 or
    287 (slid by the 260 px gap to the next frame, their edge on its far
    side) or 74-150 px across (the strip down to the holes taken in).
    Half of them are night frames whose picture reads within 0.04-0.2 D
    of the base: the edge a density step would need is not there, and
    the holes are.

    A frame off the phase slides to the nearest place on it (the next is
    a whole perforation away, 696 px, against 260 for the gap); one off
    across the film is centred where the rows say, and with *same size*
    given the roll's height there. Nothing is done unless the roll's own
    frames agree to `PERF_SPREAD` and `PERF_AGREE` of them are within
    the tolerance: a camera that slips, or a holder that hides a row,
    leaves the roll as measured."""
    moved = {}             # name -> the centre it was measured at
    rows = []
    for name, it in items.items():
        f = first.get(name) or {}
        perf, pitch = f.get("perf"), f.get("pitch")
        if not perf or not pitch or len(it.get("parts") or []) != 1:
            continue
        rows.append((name, it, perf, pitch))
    if len(rows) < 5:
        return moved
    pitch = float(np.median([r[3] for r in rows]))
    # a scan copied at another magnification is another set of lengths
    # (the two Alhambra rescans in carrete_2, from another session)
    rows = [r for r in rows if abs(r[3] - pitch) <= 0.01 * pitch]
    said = {}
    # along the film: the phase of the frame's centre among the holes
    rel = {n: (it["parts"][0]["centre"][0] - perf["phase"]) % pitch
           for n, it, perf, _ in rows if perf.get("phase") is not None}
    if len(rel) >= 5:
        z = np.exp(2j * np.pi * np.array(list(rel.values())) / pitch)
        phi = float(np.angle(np.mean(z)) * pitch / (2 * np.pi) % pitch)
        dev = {n: _wrap(v - phi, pitch) for n, v in rel.items()}
        near = [abs(d) for d in dev.values() if abs(d) <= PERF_X_TOL]
        if len(near) >= PERF_AGREE * len(dev) and np.median(near) <= PERF_SPREAD:
            phi = float((phi + np.median([d for d in dev.values() if abs(d) <= PERF_X_TOL])) % pitch)
            said["phase"] = round(phi, 1)
            for n, it, perf, _ in rows:
                if n not in rel:
                    continue
                d = _wrap(rel[n] - phi, pitch)
                if abs(d) > PERF_X_TOL:
                    moved.setdefault(n, list(it["parts"][0]["centre"]))
                    it["parts"][0]["centre"][0] = round(it["parts"][0]["centre"][0] - d, 1)
                    ev = it["evidence"]
                    ev.setdefault("flags", []).append(f"perf x {-d:+.0f}")
                    # "too wide, one side dropped" asked which side; the
                    # holes have answered it
                    held = ev.get("held") or []
                    if held and all(h.startswith("wide") for h in held):
                        ev["flags"] += held
                        ev["held"] = []
                        it["state"] = "detected"
    # across the film: the frame's centre from each row of holes
    for row, sign in (("top", 1), ("bottom", -1)):
        got = {n: perf[row] for n, it, perf, _ in rows if perf.get(row) is not None}
        if len(got) >= 5:
            mid = float(np.median(list(got.values())))
            got = {n: v for n, v in got.items() if abs(v - mid) <= PERF_Y_TOL}
        said[row] = got
    preds = {}
    for row, sign in (("top", 1), ("bottom", -1)):
        got = said.pop(row)
        if len(got) < 5:
            continue
        off = {n: sign * (items[n]["parts"][0]["centre"][1] - v) for n, v in got.items()}
        mo = float(np.median(list(off.values())))
        near = [abs(o - mo) for o in off.values() if abs(o - mo) <= PERF_Y_TOL]
        if len(near) < PERF_AGREE * len(off) or np.median(near) > PERF_SPREAD / 2:
            continue
        mo = float(np.median([o for o in off.values() if abs(o - mo) <= PERF_Y_TOL]))
        said[f"{row}_to_centre"] = round(mo, 1)
        for n, v in got.items():
            preds.setdefault(n, []).append(v + sign * mo)
    for n, ps in preds.items():
        it = items[n]
        q = it["parts"][0]
        want = float(np.mean(ps))
        d = want - q["centre"][1]
        if abs(d) > PERF_Y_TOL:
            moved.setdefault(n, list(q["centre"]))
            q["centre"][1] = round(want, 1)
            it["evidence"].setdefault("flags", []).append(f"perf y {d:+.0f}")
        if same_size and h_roll and abs(q["size"][1] - h_roll) > 0.005 * h_roll:
            # one gate, one height, now that where it sits is known
            moved.setdefault(n, list(q["centre"]))
            q["centre"][1] = round(want, 1)
            q["size"][1] = round(h_roll, 1)
            it["evidence"].setdefault("flags", []).append("perf h-set")
    if summary is not None and said:
        summary["perforations"] = said
    return moved
