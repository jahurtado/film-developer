#!/usr/bin/env python3
"""How sharp the copy stand is: the MTF from a slanted edge, per raw file.

    uv run python tools/mtf.py edge-centre.DNG edge-corner.DNG ...
    uv run python tools/mtf.py edge.DNG --region 2800,1900,400,400

Photograph a straight, opaque edge -- a razor blade, or black card with a
clean cut -- lying on the light table where the film would be, turned
about 5 degrees off the sensor's rows, once in the centre of the frame and
once in each corner, at the working aperture. Each file then gives the
edge spread function, its derivative and the MTF of the whole system --
lens, extension, focus, vibration, sensor -- at that spot, reported as
MTF50 in cycles per pixel of the green plane (ISO 12233's e-SFR, done
plainly).

What the numbers mean, roughly: MTF50 above 0.3 is a system limited by
its sensor, and a better lens buys nothing at that spot; 0.15-0.25 is a
soft spot; and the corner-to-centre ratio is the field curvature and the
alignment. A lens worth buying is one that lifts the corners to within
80 % of the centre; if the corners are already there, the money is better
spent elsewhere.

`--region` names the box to read the edge from, in scan pixels; without
it, the box with the strongest straight edge is used.
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def green(path):
    import rawpy
    with rawpy.imread(str(path)) as r:
        v = r.raw_image_visible.astype(np.float32)
        black = float(np.mean(r.black_level_per_channel))
    return np.clip(v[1::2, 0::2] - black, 0, None)


def find_edge(g, size=300):
    """The size x size box with the strongest single straight edge."""
    from scipy import ndimage
    gx = ndimage.sobel(g, axis=1)
    gy = ndimage.sobel(g, axis=0)
    mag = np.hypot(gx, gy)
    best = None
    for y in range(0, g.shape[0] - size, size // 2):
        for x in range(0, g.shape[1] - size, size // 2):
            m = mag[y:y + size, x:x + size]
            # one edge: strong gradient, and coherent in direction
            sx, sy = gx[y:y + size, x:x + size].sum(), gy[y:y + size, x:x + size].sum()
            coh = np.hypot(sx, sy) / (m.sum() + 1e-9)
            score = m.mean() * coh
            if best is None or score > best[0]:
                best = (score, x, y)
    return best[1], best[2], size, size


def esf_mtf(tile):
    """MTF from a slanted edge in a tile: fit the edge line, project every
    pixel onto the edge normal at 4x oversampling, differentiate, FFT."""
    from scipy import ndimage
    t = tile / max(tile.max(), 1e-6)
    gx = ndimage.sobel(t, axis=1)
    gy = ndimage.sobel(t, axis=0)
    ang = np.arctan2(gy.sum(), gx.sum())          # normal to the edge
    nx, ny = np.cos(ang), np.sin(ang)
    h, w = t.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = (xx - w / 2) * nx + (yy - h / 2) * ny      # distance along the normal
    # the edge's offset: where the projected profile crosses its mid level
    order = np.argsort(d.ravel())
    dd, vv = d.ravel()[order], t.ravel()[order]
    bins = np.round(dd * 4).astype(int)
    lo = bins.min()
    esf = np.bincount(bins - lo, weights=vv) / np.maximum(np.bincount(bins - lo), 1)
    # keep the central part, where every bin has samples
    n = len(esf)
    esf = esf[n // 4: 3 * n // 4]
    lsf = np.gradient(esf)
    lsf *= np.hanning(len(lsf))
    mtf = np.abs(np.fft.rfft(lsf))
    mtf /= max(mtf[0], 1e-9)
    freq = np.fft.rfftfreq(len(lsf), d=0.25)       # cycles per pixel (4x oversampled)
    return freq, mtf, np.degrees(ang)


def mtf50(freq, mtf):
    below = np.nonzero(mtf < 0.5)[0]
    if not len(below):
        return float("nan")
    i = below[0]
    if i == 0:
        return 0.0
    f0, f1, m0, m1 = freq[i - 1], freq[i], mtf[i - 1], mtf[i]
    return float(f0 + (0.5 - m0) * (f1 - f0) / (m1 - m0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--region", metavar="X,Y,W,H",
                    help="the box holding the edge, in scan pixels")
    args = ap.parse_args(argv)
    print(f"{'file':24s} {'box (scan px)':>22s} {'edge°':>6s} {'MTF50':>7s} {'MTF@0.25':>9s}")
    for f in args.files:
        g = green(f)
        if args.region:
            x, y, w, h = (int(v) // 2 for v in args.region.split(","))
        else:
            x, y, w, h = find_edge(g)
        tile = g[y:y + h, x:x + w]
        freq, mtf, ang = esf_mtf(tile)
        m50 = mtf50(freq, mtf)
        at25 = float(np.interp(0.25, freq, mtf))
        print(f"{pathlib.Path(f).name:24s} {f'{2*x},{2*y},{2*w},{2*h}':>22s} "
              f"{ang:6.1f} {m50:7.3f} {at25:9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
