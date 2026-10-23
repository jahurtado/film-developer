"""The copy stand's light falloff, fitted once from blank frames and kept
as a function, not as pixels.

A rig -- light table, lens, magnification, camera -- does not change from
roll to roll once it is set up, so its falloff is calibrated once and
named, and a roll says which calibration it was scanned under
(`roll.develop.flat`). The calibration lives in a library outside any
roll (`library()`), one JSON file per rig.

**The model**: per channel, the log2 of the light reaching the sensor is
a polynomial in the sensor's position, of total degree `DEGREE`, with
nothing at the centre -- a gain of one there, and the falloff is what the
rest of the sensor gets relative to it. Coordinates are the visible raw,
X from -1 to 1 across the width and Y on the same scale (so a circle is a
circle), centred on the sensor. Fitted on log2 because a falloff is a
ratio. Several blanks are fitted together, each with its own level --
the shutter and the film's base differ a little between them -- and only
the shape is shared.

Measured on four blanks of the Nikon D5600 rig (55 mm Micro, 1/100,
f/5.6), unexposed film over the light table with sprocket holes and holder
in frame: about 85 % of the frame usable once clipped, opaque and dusty
cells are out. Residual rms against degree, per channel, and the same
leaving each blank out of the fit and predicting it:

    degree 2   1.8 %   loo 1.8 %     5 terms
    degree 4   0.7 %   loo 0.8 %    14 terms
    degree 6   0.6 %   loo 0.7 %    27 terms
    degree 8   0.6 %   loo 0.7 %    44 terms

so 6, past which nothing more is explained. The rest is the film base's
own texture and dust, which move between blanks and are not the light.
The falloff is 0.7-0.9 EV at the sensor's corners and not symmetric; it
is nearly the same in the three channels (R/G and B/G vary 1-2 % across
the field). Outside the cells that had data the polynomial is
extrapolating -- the corners under the holder disagree by 0.1-0.2 EV
between degrees -- and `coverage` in the file says where the data was.
"""

import datetime
import json
import math
import os
import pathlib
import sys

import numpy as np

DEGREE = 6
BIN = 16            # mosaic plane cells per side, for the fit: 32 raw px
CLIP_SHARE = 0.95   # of the scan's ceiling: a cell this bright is direct light
DARK_SHARE = 0.15   # of the median: a cell this dark is the holder or a hole's edge
ERODE = 3           # cells off every masked region: the penumbra around it
DUST = 0.03         # log2 off a 9-cell median: a speck, not the light
REJECT = 4.0        # residual over this many rms after the first fit: refit without
CHANNELS = ("R", "G", "B")


def library():
    """Where calibrations live: outside any roll, because a rig outlives
    them."""
    if sys.platform == "darwin":
        root = pathlib.Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        root = pathlib.Path(os.environ.get("APPDATA",
                                           pathlib.Path.home() / "AppData" / "Roaming"))
    else:
        root = pathlib.Path(os.environ.get("XDG_DATA_HOME",
                                           pathlib.Path.home() / ".local" / "share"))
    return root / "film-developer" / "flats"


def path_of(name):
    """A calibration by name in the library, or a path to one."""
    p = pathlib.Path(name)
    if p.suffix == ".json" or os.sep in str(name):
        return p.expanduser()
    return library() / f"{name}.json"


def names():
    """The calibrations in the library, newest first: the stand being set
    up now is the one most likely wanted."""
    lib = library()
    if lib.is_dir():
        return [p.stem for p in sorted(lib.glob("*.json"),
                                       key=lambda q: q.stat().st_mtime,
                                       reverse=True)]
    return []


_CACHE = {}


def _legacy(name):
    """A value of the old flat field, which `roll.develop.flat` held before
    this: `auto` (a map from the roll's own scans), a percentage, or a
    blank scan's file. None of them names a calibration and that flat field
    is gone, so they mean none -- as they have since it went. Found on
    portra400-full, whose document says `auto`: `⌘R` and `apply` stopped
    on it with "no flat calibration 'auto'"."""
    s = str(name)
    if s.lower() == "auto" or "/" in s or "\\" in s:
        return True
    if pathlib.Path(s).suffix.lower() in RAWS:
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


def load(name):
    """The calibration called `name`, or None for none. Cached per process
    by the file's time, since every frame of a roll asks for it."""
    if not name or str(name).lower() in ("none", "off") or _legacy(name):
        return None
    p = path_of(name)
    if not p.exists():
        raise FileNotFoundError(
            f"no flat calibration {name!r} (looked for {p}); "
            f"known: {', '.join(names()) or 'none'}")
    key = (str(p), p.stat().st_mtime)
    if key not in _CACHE:
        _CACHE[key] = json.loads(p.read_text())
    return _CACHE[key]


def of(taste):
    """The calibration a roll's settings name."""
    return load((taste or {}).get("flat"))


def terms(degree):
    return [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)
            if i + j > 0]


def _coords(xs, ys, W, H):
    """Raw pixel centres to the model's coordinates."""
    return (np.asarray(xs, np.float64) / W * 2 - 1,
            (np.asarray(ys, np.float64) / H * 2 - 1) * H / W)


def log_gain(cal, channel, xs, ys):
    """log2 of the light at raw positions `xs` (columns) by `ys` (rows), on
    the outer grid they make: (len(ys), len(xs)). Separable, so a whole
    scan costs `degree` outer products and not a basis per pixel."""
    W, H = cal["sensor"]
    X, Y = _coords(xs, ys, W, H)
    c = dict(zip(map(tuple, cal["terms"]), cal["coef"][CHANNELS[channel]]))
    out = np.zeros((len(Y), len(X)), np.float32)
    deg = cal["degree"]
    for i in range(deg + 1):
        col = sum(c.get((i, j), 0.0) * Y ** j for j in range(deg + 1 - i))
        if np.isscalar(col) and col == 0:
            continue
        out += np.outer(col, X ** i).astype(np.float32)
    return out


def check(cal, path, W, H):
    """A calibration applies to the camera and sensor it was fitted on;
    anything else would divide by another rig's light."""
    model = model_of(path)
    if [int(W), int(H)] != list(cal["sensor"]):
        raise ValueError(f"flat {cal['name']!r} is for a {cal['sensor'][0]}x"
                         f"{cal['sensor'][1]} sensor, this scan is {W}x{H}")
    if cal.get("camera") and model and model != cal["camera"]:
        raise ValueError(f"flat {cal['name']!r} is for a {cal['camera']}, "
                         f"this scan is from a {model}")


def divide_rgb(rgb, cal, W, H):
    """A demosaiced image of the whole visible raw (any size), divided by
    the light, in place. Returns it."""
    h, w = rgb.shape[:2]
    xs = (np.arange(w) + 0.5) * W / w
    ys = (np.arange(h) + 0.5) * H / h
    for c in range(3):
        rgb[..., c] *= np.exp2(-log_gain(cal, c, xs, ys))
    return rgb


def divide_mosaic(T, chan, cal):
    """The mosaic (visible raw, any units, black out), divided by the light
    photosite by photosite, each by its own channel's falloff. In place."""
    H, W = T.shape
    for dy in (0, 1):
        for dx in (0, 1):
            xs = np.arange(dx, W, 2) + 0.5
            ys = np.arange(dy, H, 2) + 0.5
            T[dy::2, dx::2] *= np.exp2(-log_gain(cal, int(chan[dy, dx]), xs, ys))
    return T


# ---------------------------------------------------------------- fitting

def _cells(path):
    """One blank as (R, G, B) planes of BIN x BIN cell means, black out,
    their centres in raw pixels, the mask of cells that see the light and
    nothing else, and the scan's identity."""
    import rawpy
    from scipy.ndimage import binary_erosion, median_filter
    from . import image_io
    with rawpy.imread(str(path)) as r:
        v = r.raw_image_visible.astype(np.float32)
        colors = r.raw_colors_visible
        black = np.asarray(r.black_level_per_channel, np.float32)
        white = int(r.white_level)
        ceiling = image_io.clip_level(v, white) or white
    H, W = v.shape
    hh, ww = H // 2 * 2, W // 2 * 2
    planes, centres = {}, None
    sums = {0: [], 1: [], 2: []}
    for dy in (0, 1):
        for dx in (0, 1):
            k = int(colors[dy, dx])
            pl = v[dy:hh:2, dx:ww:2] - black[k]
            ph, pw = pl.shape[0] // BIN * BIN, pl.shape[1] // BIN * BIN
            cells = pl[:ph, :pw].reshape(ph // BIN, BIN, pw // BIN, BIN)
            # a cell with any photosite at the ceiling is clipped, whatever its mean
            hot = cells.max((1, 3)) >= ceiling - black[k] - 1
            sums[1 if k == 3 else k].append((cells.mean((1, 3)), hot))
            if centres is None:
                # a cell spans 2*BIN raw pixels from 2*BIN*i; its centre,
                # over the four photosites of the 2x2 it averages
                cy = np.arange(ph // BIN) * BIN * 2 + BIN
                cx = np.arange(pw // BIN) * BIN * 2 + BIN
                centres = (cx, cy)
    hot = np.zeros_like(sums[0][0][1])
    for c in range(3):
        planes[c] = np.mean([m for m, _ in sums[c]], 0)
        for _, h_ in sums[c]:
            hot |= h_
    stack = np.stack([planes[c] for c in range(3)])
    ok = ~hot
    ok &= stack.max(0) < CLIP_SHARE * (ceiling - black.mean())
    ok &= (stack > DARK_SHARE * np.median(stack, axis=(1, 2))[:, None, None]).all(0)
    ok = binary_erosion(ok, iterations=ERODE)
    for c in range(3):
        sm = median_filter(stack[c], 9)
        ok &= np.abs(np.log2(np.maximum(stack[c], 1) / np.maximum(sm, 1))) < DUST
    return stack, centres, ok, (W, H)


def _solve(blanks, channel, degree, keep=None):
    """Shared shape, one level per blank: (shape coefficients, levels)."""
    T = terms(degree)
    rows, ys = [], []
    for k, (stack, (cx, cy), ok, (W, H)) in enumerate(blanks):
        m = ok if keep is None else (ok & keep[k])
        yy, xx = np.nonzero(m)
        X, Y = _coords(cx[xx], cy[yy], W, H)
        A = np.stack([X ** i * Y ** j for i, j in T], -1)
        lev = np.zeros((len(X), len(blanks)))
        lev[:, k] = 1
        rows.append(np.hstack([A, lev]))
        ys.append(np.log2(stack[channel][m]))
    coef, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(ys), rcond=None)
    return coef[:len(T)], coef[len(T):]


def _residual(blank, channel, degree, coef, level=None):
    stack, (cx, cy), ok, (W, H) = blank
    yy, xx = np.nonzero(ok)
    X, Y = _coords(cx[xx], cy[yy], W, H)
    f = sum(c * X ** i * Y ** j for c, (i, j) in zip(coef, terms(degree)))
    r = np.log2(stack[channel][ok]) - f
    return r - (np.median(r) if level is None else level)


def fit(name, paths, degree=DEGREE, progress=None):
    """The calibration from these blanks, as the dict `save` writes.
    Raises if they are not one camera and one sensor. `progress(i, n)`
    counts the blanks read, which is most of the time."""
    paths = [pathlib.Path(p) for p in paths]
    if not paths:
        raise ValueError("no blanks to fit")
    blanks = []
    for k, p in enumerate(paths):
        blanks.append(_cells(p))
        if progress:
            progress(k, len(paths) + 1)
    sizes = {b[3] for b in blanks}
    if len(sizes) > 1:
        raise ValueError(f"the blanks are from different sensors: {sizes}")
    model = model_of(paths[0])
    if len({model_of(p) for p in paths}) > 1:
        raise ValueError("the blanks are from different cameras")
    W, H = blanks[0][3]
    coef, rms, loo, levels = {}, {}, {}, {}
    for c, nm in enumerate(CHANNELS):
        cf, lev = _solve(blanks, c, degree)
        # once more without the cells the first fit could not explain:
        # dust the median missed, a scratch, the edge of a sprocket hole
        res = [_residual(b, c, degree, cf, lev[k]) for k, b in enumerate(blanks)]
        s = float(np.concatenate(res).std())
        keep = []
        for k, b in enumerate(blanks):
            full = np.zeros_like(b[2])
            yy, xx = np.nonzero(b[2])
            full[yy, xx] = np.abs(res[k]) < REJECT * s
            keep.append(full)
        cf, lev = _solve(blanks, c, degree, keep)
        res = np.concatenate([_residual(b, c, degree, cf, lev[k])
                              for k, b in enumerate(blanks)])
        coef[nm] = [round(float(v), 7) for v in cf]
        rms[nm] = round(100 * float(res.std()), 3)
        levels[nm] = [round(float(v - lev[0]), 4) for v in lev]
        if len(blanks) > 1:
            out = []
            for k, b in enumerate(blanks):
                others = [o for i, o in enumerate(blanks) if i != k]
                ck, _ = _solve(others, c, degree,
                               [kk for i, kk in enumerate(keep) if i != k])
                out.append(_residual(b, c, degree, ck))
            loo[nm] = round(100 * float(np.concatenate(out).std()), 3)
    # where there was data: the box of the cells used in any blank
    used = np.zeros_like(blanks[0][2])
    for b in blanks:
        used |= b[2]
    yy, xx = np.nonzero(used)
    cx, cy = blanks[0][1]
    box = [float(cx[xx.min()]), float(cy[yy.min()]), float(cx[xx.max()]),
           float(cy[yy.max()])]
    cal = {"film-developer-flat": 1, "name": name, "camera": model,
           "sensor": [W, H], "degree": degree, "terms": terms(degree),
           "coef": coef,
           "created": datetime.date.today().isoformat(),
           "blanks": [str(p) for p in paths],
           "fit": {"rms_pct": rms, "loo_rms_pct": loo or None,
                   "levels_ev": levels,
                   "used_share": round(float(np.mean([b[2].mean() for b in blanks])), 3),
                   "coverage_px": [round(v) for v in box]}}
    corners = {}
    for c, nm in enumerate(CHANNELS):
        g = log_gain(cal, c, [box[0], box[2]], [box[1], box[3]])
        corners[nm] = [round(float(v), 3) for v in g.ravel()]
    cal["fit"]["falloff_ev_at_coverage_corners"] = corners
    cal["fit"]["falloff_ev_at_sensor_corners"] = {
        nm: [round(float(v), 3) for v in
             log_gain(cal, c, [0.5, W - 0.5], [0.5, H - 0.5]).ravel()]
        for c, nm in enumerate(CHANNELS)}
    return cal


def model_of(path):
    """The camera that wrote a scan, from its first IFD (a NEF and a DNG
    both carry Make and Model there), or None."""
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as t:
            tags = t.pages[0].tags
            make = tags.get("Make")
            model = tags.get("Model")
            if model is None:
                return None
            m = str(model.value).strip()
            mk = str(make.value).strip() if make is not None else ""
            return m if not mk or m.upper().startswith(mk.split()[0].upper()) \
                else f"{mk} {m}"
    except Exception:
        return None


def save(cal, force=False):
    p = path_of(cal["name"])
    if p.exists() and not force:
        raise FileExistsError(f"{p} exists; --force to replace it")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal, indent=1) + "\n")
    return p


# ---------------------------------------------------------------- command

def add_args(ap):
    ap.add_argument("name", nargs="?", metavar="NAME",
                    help="what to call the rig, e.g. nikon-55micro-table; "
                         "a roll names it with `develop --flat NAME`")
    ap.add_argument("blanks", nargs="*", metavar="BLANK",
                    help="raws of the light through unexposed film (or "
                         "nothing), shot on the rig as it scans; folders "
                         "are read for raws. Several: each is fitted with "
                         "its own level and the shape is shared")
    ap.add_argument("--degree", type=int, default=DEGREE,
                    help=f"of the polynomial (default {DEGREE})")
    ap.add_argument("--force", action="store_true",
                    help="replace a calibration of that name")
    ap.add_argument("--list", action="store_true",
                    help="the calibrations in the library, and where it is")


RAWS = {".nef", ".dng", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2", ".pef"}


def run(args):
    if args.list or not args.name:
        print(library())
        for n in names():
            c = load(n)
            f = c.get("fit") or {}
            print(f"  {n:28s} {c.get('camera') or '?'}  {len(c.get('blanks', []))} "
                  f"blanks  {c.get('created', '')}  rms "
                  + " ".join(f"{k} {v:.2f}%" for k, v in (f.get("rms_pct") or {}).items()))
        return 0
    paths = []
    for b in args.blanks:
        b = pathlib.Path(b)
        if b.is_dir():
            paths += sorted(q for q in b.iterdir() if q.suffix.lower() in RAWS)
        else:
            paths.append(b)
    if not paths:
        print("no blanks given", file=sys.stderr)
        return 1
    if path_of(args.name).exists() and not args.force:
        print(f"{path_of(args.name)} exists; --force to replace it", file=sys.stderr)
        return 1
    print(f"fitting {len(paths)} blanks, degree {args.degree}…")
    cal = fit(args.name, paths, args.degree)
    out = save(cal, force=args.force)
    f = cal["fit"]
    print(f"{out}\n  {cal['camera']}, {cal['sensor'][0]}x{cal['sensor'][1]}, "
          f"{len(cal['terms'])} terms a channel, {100 * f['used_share']:.0f} % of "
          f"each blank usable")
    for nm in CHANNELS:
        loo = (f.get("loo_rms_pct") or {}).get(nm)
        corners = f["falloff_ev_at_coverage_corners"][nm]
        print(f"  {nm}: residual {f['rms_pct'][nm]:.2f} %"
              + (f", predicting a blank left out {loo:.2f} %" if loo is not None else "")
              + f"; corners of the data {min(corners):+.2f}..{max(corners):+.2f} EV")
    if len(paths) == 1:
        print("  one blank: nothing to check the fit against; two or more say "
              "how well it predicts")
    return 0
