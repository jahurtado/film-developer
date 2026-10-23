"""The frames as negatives: cut out, not inverted.

What `⌘R` writes into `negatives/` when its sheet asks for it, beside the
positives and never instead of them. Two files per frame, each the twin of
a positive:

- **the picture** (`.jpg` or `.tif`, the format the positives take) is the
  cut the develop makes -- the same straightening, the same crop, the same
  mirror and quarter turns -- with nothing done to the values but the
  camera's as-shot white balance -- no colour matrix, as in the develop
  (`develop.MATRIX`). The
  orange mask, the base and the exposure of the copy are all still there.
  Nothing is measured off the frame, so two scans of one negative come out
  as different as they are, which is what the scan view of the editor
  says too.
- **the raw** (`.dng`) is the positive DNG's crop of the mosaic with the
  values the sensor wrote, untouched: the scan's own black and white
  levels, no linearisation table, the camera's colour tags, its as-shot
  neutral and no BaselineExposure. Mirrored and turned on the mosaic, which
  a Bayer pattern survives (`positive.py`); the tilt is left to the
  developer, as there.

The white balance is applied in float after the demosaic, not handed to
LibRaw: LibRaw scales into 16 bits and the base of a colour negative, red
times a camera's red gain of about two, is where that clips.
"""

import json
import pathlib

import numpy as np

from . import develop, positive

FOLDER = "negatives"


def as_shot(path):
    """The camera's as-shot white balance as (R, G, B) gains over green,
    or ones when the file carries none."""
    import rawpy
    with rawpy.imread(str(path)) as r:
        wb = [float(v) for v in r.camera_whitebalance[:3]]
    if not all(v > 0 for v in wb):
        return np.ones(3, np.float32)
    return np.array(wb, np.float32) / wb[1]


def write_picture(path, item, part, matrix, taste, mirrored, out):
    """One frame of one scan, cut out and not inverted, at `out`."""
    gain = as_shot(path)
    rgb, scale = develop.linear(path, False, None)
    fr = develop.cut(rgb, scale, part, item["angle"], develop.crop_inset(taste))
    del rgb
    fr = np.clip((fr * gain[None, None, :]) @ np.asarray(matrix, np.float32),
                 0, None)
    if mirrored:
        fr = fr[:, ::-1]
    turn = int(part.get("turn", 0) or 0) % 360
    if turn:
        fr = np.rot90(fr, k=-(turn // 90))
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if taste.get("format") == "tiff16":
        import tifffile
        tifffile.imwrite(str(out), develop.encode(fr, 16), photometric="rgb")
    else:
        from PIL import Image
        Image.fromarray(develop.encode(fr, 8)).save(str(out), quality=92,
                                                   subsampling=1)
    return {"size": [fr.shape[1], fr.shape[0]],
            "as_shot": [round(float(v), 4) for v in gain],
            # Of this picture, after the camera's balance, and
            # nothing else: a red gain of two on the orange base is where it
            # clips. It says nothing about the raw -- on nikon_55mm the
            # 1/100 frames clip 5-8 % here with no photosite at white -- and
            # it was called `clipped`, which another reader took for that.
            "picture_clipped": round(float((fr >= 1).any(-1).mean()), 5)}


def write_raw(path, item, part, crop, mirrored, matrix, out, crop_pct=0.0,
              compress=True):
    """One frame of one scan as a CFA DNG holding the sensor's own values,
    cut where the positive DNG is cut."""
    import rawpy
    with rawpy.imread(str(path)) as r:
        v = r.raw_image_visible.copy()
        colors = r.raw_colors_visible.copy()
        black_ch = np.asarray(r.black_level_per_channel, np.float64)
        white = int(r.white_level)
    from . import image_io
    ceiling = image_io.clip_level(v, white)
    chan = np.where(colors == 3, 1, colors)
    black = black_ch[colors]                    # per photosite, by its own channel
    kind = crop if crop in positive.CROPS else "outer"
    # the positive DNG's own rectangle, the roll's crop included
    x0, y0, wc, hc, lost = positive.dng_rect(kind, part, item["angle"],
                                             v.shape[1], v.shape[0], crop_pct)
    raw = v[y0:y0 + hc, x0:x0 + wc]
    cmap = chan[y0:y0 + hc, x0:x0 + wc]
    blk = black[y0:y0 + hc, x0:x0 + wc]
    del v, chan, black
    if mirrored:
        raw, cmap, blk = raw[:, ::-1], cmap[:, ::-1], blk[:, ::-1]
    turn = int(part.get("turn", 0) or 0) % 360
    if turn:
        k = -(turn // 90)
        raw, cmap, blk = np.rot90(raw, k=k), np.rot90(cmap, k=k), np.rot90(blk, k=k)
    raw = np.ascontiguousarray(raw.astype(np.uint16))
    neutral = 1.0 / as_shot(path).astype(np.float64)
    norm = np.clip((raw.astype(np.float32) - blk) / max(white - blk.mean(), 1), 0, None)
    preview = positive.preview_of(norm, cmap, neutral, 0.0,
                                  np.eye(3) if matrix is None else matrix)
    cfa = positive.write_cfa(out, raw, cmap, positive.source_tags(path),
                             neutral, 0.0, preview,
                             black=tuple(int(round(b)) for b in blk[0:2, 0:2].ravel()),
                             white=white, compress=compress)
    return {"crop": [x0, y0, wc, hc], "crop_kind": kind,
            "crop_pct": crop_pct, "off_sensor_ltrb": lost,
            "size": [raw.shape[1], raw.shape[0]], "cfa": list(cfa),
            "black": [round(float(b), 1) for b in blk[0:2, 0:2].ravel()],
            "white": white,
            # the share of the crop's photosites the sensor saturated: the
            # one clipping figure here that is about the scan. At the
            # scan's own ceiling, not the declared white, which the Nikon
            # never reaches (`image_io.clip_level`).
            "raw_clipped": round(float((raw >= ceiling).mean()), 6)
            if ceiling is not None else 0.0}


def _job(job):
    kind, path, item, part, matrix, taste, crop, mirrored, out, compress = job
    if kind == "raw":
        ev = write_raw(path, item, part, crop, mirrored, matrix, out,
                       float(taste.get("crop", 0.0) or 0.0), compress)
    else:
        ev = write_picture(path, item, part, matrix, taste, mirrored, out)
    return pathlib.Path(out).name, ev


def write_roll(roll_dir, doc, out_dir, taste, crop, workers, progress=None,
               stats=None, pictures=True, raws=True, compress=True):
    """Every frame of the document as a negative: the picture, the raw or
    both. `stats` is the roll's measurement, for its matrix.
    Returns {file name: evidence}."""
    frames = develop.frames_of(roll_dir, doc)
    if not frames or not (pictures or raws):
        return {}
    matrix = (stats or {}).get("matrix")
    if matrix is None:
        matrix = develop.MATRIX
    mirrored = doc["roll"].get("mirrored", True)
    ext = ".tif" if taste.get("format") == "tiff16" else ".jpg"
    jobs = []
    for path, item, part in frames:
        stem = pathlib.Path(out_dir) / develop.out_stem(doc, path, item, part)
        if pictures:
            jobs.append(("picture", path, item, part, matrix, taste, crop,
                         mirrored, str(stem) + ext, compress))
        if raws:
            jobs.append(("raw", path, item, part, matrix, taste, crop,
                         mirrored, str(stem) + ".dng", compress))
    return dict(develop._run(_job, jobs, workers, progress))


def record(out_dir, taste, crop, results):
    """`negatives.json` beside the files: what was cut and how."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    develop.sweep(out_dir, "negatives.json", results)
    (out_dir / "negatives.json").write_text(json.dumps(
        {"settings": {"format": taste.get("format", "jpeg"), "crop": crop,
                      "crop_inset": develop.crop_inset(taste)},
         "frames": results}, indent=1) + "\n")
