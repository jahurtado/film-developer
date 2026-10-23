"""The estimators: where does the exposed frame sit in this scan?

Measured on ten sample scans.

The two facts the whole module is shaped by:

- **The polarity is not fixed.** With a masked holder the frame is the bright
  thing; with bare film on a light table the rebate is brighter and the frame
  is the dark thing. Both occur in ten scans from one photographer. So nothing
  here looks at the *sign* of a step, only at its magnitude, and which side the
  picture is on is never assumed.
- **The step at the frame edge varies by a factor of eight** between scans
  (2.01 to 0.24 in density). So no threshold is absolute: every one is a
  fraction of what the same scan shows at its own strongest edge.

What picks the frame out of the plateaus is therefore not brightness but
**shape**: a frame has one of a handful of known aspect ratios, and that is the
strongest prior available. Taking the two strongest steps instead was tried and
does not work -- it brackets both half-frames on one scan and one of them on
the next.
"""

import math

import numpy as np
from scipy import ndimage

# The aspect ratios of the image area on film, long side over short, as the
# frame sits on the film. Half frame is 18x24 mm, so it is *portrait* where the
# strip runs landscape, which is what makes it two parts of one scan.
# Long side over short, as the exposed area actually measures -- not as the
# format is named. Measured over the thirteen frames of the sample set, four of
# them placed by hand:
#
#     full frame   1.492 - 1.511   (mean 1.501)   nominal 36/24 = 1.500  ok
#     half frame   0.705 - 0.717   (mean 0.711)   nominal 18/24 = 0.750  no
#
# A gate is not the format it is sold as. Half frame nominally exposes 18x24
# and on this camera exposes about 17x24, which is 5 % narrower -- enough that
# searching for 0.75 fitted a rectangle 30 px too wide on every half-frame scan
# and beat the right one on ratio. The medium formats are still nominal and
# have never been measured; they are marked as such.
FORMATS = {
    "full": 1.501,        # measured
    "half": 0.711,        # measured
    "6x4.5": 56 / 42,     # nominal, unmeasured
    "6x6": 1.0,           # nominal, unmeasured
    "6x7": 70 / 56,       # nominal, unmeasured
}

ANGLES = np.arange(-2.5, 2.51, 0.05)
# A candidate must be at least this much of the scan in each direction. Below
# it, everything matches something: a 40x27 px plateau is a perfect 3:2.
MIN_SPAN = 0.15
# How far a measured ratio may sit from a real one, as |log| -- about 4 %.
# It was 13 % while the ratios themselves were nominal and wrong by 5 %. With
# them measured, the sample set's own spread is under 1 %, and 13 % is wide
# enough that almost any rectangle between 0.66 and 1.5 matches something.
ASPECT_TOL = 0.04

# What `auto` looks for. 35mm is what the program is for and what has been
# measured; the medium formats are nominal guesses, and letting them into the
# default search is what put a half-frame scan's frames 7 % too wide -- 6x4.5
# stood on end is 0.750, which is exactly the half-frame ratio that turned out
# not to be real.
DEFAULT_FORMATS = ("full", "half")


# A cut is only believed where the gradient points the same way down the whole
# length of it. Measured on the sample set: the spurious cut that split a frame
# in two scored 0.589, while every real frame edge scored 0.773 to 1.000 --
# including the ones with a fiftieth of the strength.
COHERENCE = 0.70
# No run shorter than this, in working pixels. The selection below rebuilds a
# frame out of several runs, so cutting too often is cheap and cutting too
# seldom is not -- but it was 6, which a sweep showed sits on a
# two-wide island with 7 and 8 failing either side of it. 9 to 16 all work, so
# it sits in the middle of that instead.
MIN_RUN = 12
# How far above its own stretch's median gradient a cut has to stand.
SIGNIFICANCE = 4.0


def edge_strength(band, axis):
    """Where the edges are along `axis`, as a coherence-gated gradient.

    The projection profile throws away the one thing that tells a frame edge
    from the picture inside it. Both make a step in the mean; but a frame edge
    has the gradient pointing **the same way in every row it crosses**, and the
    content of a photograph does not. So the gradient is summed two ways --
    with its sign and without -- and their ratio is how much of it agrees with
    itself.

    Evidence: on SDIM0897 a bright building against dark paving split a frame
    in two, scoring 0.589; the four real edges of that scan scored 0.847 to
    1.000, two of them at 2-3 % of the scan's strongest gradient.
    """
    g = np.diff(band, axis=axis)
    strength = np.abs(g.mean(1 - axis))
    coherence = np.abs(g.sum(1 - axis)) / (np.abs(g).sum(1 - axis) + 1e-9)
    strength = ndimage.uniform_filter1d(strength, 5)
    coherence = ndimage.uniform_filter1d(coherence, 5)
    return np.where(coherence > COHERENCE, strength, 0.0)


# One physical edge is a ramp several pixels wide, and the recursion happily
# cuts more than once inside it. Two cuts closer than this are the same edge.
MERGE = 5
# Every frame on a scan came through the same gate of the same camera, so they
# are the same size. Tight on purpose: on SDIM0898 a wrong left edge produced a
# frame 4.5 % narrower than its neighbour, and nothing else rejected it.
SAME_SIZE = 0.03
# A side has to be worth something, as a share of the scan's strongest cut.
# It was first placed at 0.02, in the gap between the weakest real edge in the
# sample set (SDIM0897, 2.3 % of that scan's strongest) and the strongest false
# one (SDIM0834, 1.9 %) -- a gap of four tenths of a percentage point, on one
# example of each. A sweep then showed the set is right for
# anything from under 0.001 up to 0.02, so 0.02 was sitting on the edge of what
# works with nothing above it. Moved to the middle of the range instead.
EVIDENCE_FLOOR = 0.006


# How far a span may sit from a size the roll already knows, once it knows
# one. Tight, because that is the point: neighbouring scans agree on the frame
# size to 0.18-0.43 % on the labelled corpus, while the aspect ratio cannot
# tell 1.482 from 1.490 -- which is the difference between a frame and a frame
# with 114 px of rebate down one side.
EXPECT_TOL = 0.04


def corners(cx, cy, w, h, angle):
    """The four corners of a part, in the space its centre is given in.

    A part is stored as centre, size and angle because those three cannot
    contradict each other and four corners can. This is the one place that
    turns them into corners, so nothing else has to know which way the
    rotation goes.
    """
    r = math.radians(angle)
    co, si = math.cos(r), math.sin(r)
    return [(cx + dx * co - dy * si, cy + dx * si + dy * co)
            for dx, dy in ((-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2))]


def fit_rectangles(quads, near=None):
    """The tilt shared by several placed frames, and each one's rectangle.

    Four points are eight numbers and a rectangle is five -- centre, size and
    angle -- so they almost never describe one exactly. This is the least
    squares fit: the rectangles whose corners are nearest, summed and squared,
    to where a person put theirs.

    The **angle is fitted once across every quad**, because the frames are on
    one piece of film and cannot be tilted differently. Fitting each frame its
    own angle and keeping the last one lets a slip on one vertex of one frame
    throw away the evidence of the other frame's four -- which it does: one
    corner moved 109 px swings a single-quad fit by 0.84 degrees.

    Each quad's points are in the order `corners()` produces them, which is the
    order the editor's handles are in, so which corner is which is known
    rather than guessed. The centre of a rectangle is the mean of its corners,
    so that part is exact; the angle is found by search and the sizes fall out
    of it.

    `near` is the angle to start from, so the search cannot wander to the
    rectangle turned ninety degrees, which fits any set of corners as well.
    """
    signs = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    prepared = []
    for quad in quads:
        pts = [(float(x), float(y)) for x, y in quad]
        cx = sum(p[0] for p in pts) / 4
        cy = sum(p[1] for p in pts) / 4
        prepared.append((cx, cy, [(p[0] - cx, p[1] - cy) for p in pts]))

    def at(angle):
        """(total residual, [(cx, cy, w, h)]) for the best rectangles at this
        angle."""
        r = math.radians(angle)
        co, si = math.cos(r), math.sin(r)
        total, out = 0.0, []
        for cx, cy, rel in prepared:
            q = [(dx * co + dy * si, -dx * si + dy * co) for dx, dy in rel]
            a = sum(s[0] * v[0] for s, v in zip(signs, q)) / 4
            b = sum(s[1] * v[1] for s, v in zip(signs, q)) / 4
            total += sum((v[0] - s[0] * a) ** 2 + (v[1] - s[1] * b) ** 2
                         for s, v in zip(signs, q))
            out.append((cx, cy, 2 * abs(a), 2 * abs(b)))
        return total, out

    base = 0.0 if near is None else float(near)
    best = min((at(base + d / 2)[0], base + d / 2) for d in range(-90, 91))
    lo, hi = best[1] - 0.5, best[1] + 0.5
    for _ in range(40):                       # ternary search, to 1e-4 degrees
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if at(m1)[0] < at(m2)[0]:
            hi = m2
        else:
            lo = m1
    angle = (lo + hi) / 2
    return angle, at(angle)[1]


def fit_rectangle(points, near=None):
    """The one-frame case of `fit_rectangles`: (cx, cy, w, h, angle)."""
    angle, rects = fit_rectangles([points], near)
    cx, cy, w, h = rects[0]
    return cx, cy, w, h, angle


def fit_row(quads, near=None):
    """The row of frames closest to several placed quadrilaterals.

    A camera's gate does not move. The film advances past it, so every frame
    on a strip is **the same size** and sits at **the same distance across**
    it -- anchored on one top line and one bottom line -- and the only thing
    that differs between them is how far along they are. Two frames out of
    square with each other is not something film does.

    So this fits, over all the quads at once: one angle, one width, one
    height, one across-strip position, and one along-strip position each. Each
    quad has eight numbers and contributes to five shared ones, which is why
    the answer is steadier than any frame on its own -- the same reason a
    rectangle through four placed corners beats the corners.

    Returns (angle, [(cx, cy, w, h)]) in the coordinates the quads came in.
    """
    signs = ((-1, -1), (1, -1), (1, 1), (-1, 1))

    def at(angle):
        r = math.radians(angle)
        co, si = math.cos(r), math.sin(r)
        frames = []
        for quad in quads:
            local = [(x * co + y * si, -x * si + y * co) for x, y in quad]
            uc = sum(p[0] for p in local) / 4
            vc = sum(p[1] for p in local) / 4
            a = sum(s[0] * (p[0] - uc) for s, p in zip(signs, local)) / 4
            b = sum(s[1] * (p[1] - vc) for s, p in zip(signs, local)) / 4
            frames.append((local, uc, vc, a, b))
        # What the frames share, averaged over them.
        A = sum(f[3] for f in frames) / len(frames)
        B = sum(f[4] for f in frames) / len(frames)
        V = sum(f[2] for f in frames) / len(frames)
        total, out = 0.0, []
        for local, uc, vc, _, _ in frames:
            total += sum((p[0] - (uc + s[0] * A)) ** 2
                         + (p[1] - (V + s[1] * B)) ** 2
                         for s, p in zip(signs, local))
            out.append((uc * co - V * si, uc * si + V * co,
                        2 * abs(A), 2 * abs(B)))
        return total, out

    base = 0.0 if near is None else float(near)
    best = min((at(base + d / 2)[0], base + d / 2) for d in range(-90, 91))
    lo, hi = best[1] - 0.5, best[1] + 0.5
    for _ in range(40):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if at(m1)[0] < at(m2)[0]:
            hi = m2
        else:
            lo = m1
    angle = (lo + hi) / 2
    return angle, at(angle)[1]
