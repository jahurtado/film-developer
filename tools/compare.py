#!/usr/bin/env python3
"""Two developments of the same frames, side by side and in numbers.

    uv run python tools/compare.py DIR_A DIR_B [-o SHEETS_DIR] [--label-a after --label-b mosaic]

For every JPEG the two folders share: the mean absolute difference in
levels of 255, its 99th percentile, the same in the shadows (luminance
under 25), and the worst frame; and, with `-o`, one sheet per frame with
the two at full size, A on the left and B on the right, and under them
the 400 px box where they differ most from each, at 3x, outlined in
yellow on both. This is how the two orders of inversion were compared
over the four rolls (develop.md).
"""

import argparse
import pathlib
import sys

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
BOX, ZOOM = 400, 3


def sheet(A, B, out, la, lb):
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 44)
    except OSError:
        font = ImageFont.load_default()
    a = np.asarray(A, np.float32)
    b = np.asarray(B, np.float32)
    d = np.abs(a - b).sum(-1)
    m = ndimage.uniform_filter(d, BOX, mode="constant")
    cy, cx = np.unravel_index(np.argmax(m), m.shape)
    x0 = int(np.clip(cx - BOX // 2, 0, a.shape[1] - BOX))
    y0 = int(np.clip(cy - BOX // 2, 0, a.shape[0] - BOX))
    W, H = A.size
    gap, top, det = 24, 70, BOX * ZOOM
    canvas = Image.new("RGB", (2 * W + gap, top + H + gap + det + 60), (30, 30, 34))
    canvas.paste(A, (0, top))
    canvas.paste(B, (W + gap, top))
    za = A.crop((x0, y0, x0 + BOX, y0 + BOX)).resize((det, det), Image.NEAREST)
    zb = B.crop((x0, y0, x0 + BOX, y0 + BOX)).resize((det, det), Image.NEAREST)
    yz = top + H + gap
    canvas.paste(za, (W - det - gap // 2, yz))
    canvas.paste(zb, (W + gap + gap // 2, yz))
    dr = ImageDraw.Draw(canvas)
    dr.text((16, 14), la, fill=(230, 230, 232), font=font)
    dr.text((W + gap + 16, 14), lb, fill=(230, 230, 232), font=font)
    dr.text((16, yz + det + 8),
            f"below: the {BOX} px box where they differ most, at {ZOOM}x "
            f"(x {x0}, y {y0}); mean difference {np.abs(a - b).mean():.2f} of 255",
            fill=(160, 160, 165), font=font)
    for xx, yy in ((x0, y0 + top), (x0 + W + gap, y0 + top)):
        dr.rectangle((xx, yy, xx + BOX, yy + BOX), outline=(255, 212, 0), width=4)
    canvas.save(out, quality=90, subsampling=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("-o", "--output", metavar="DIR", help="write a sheet per frame here")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args(argv)
    from PIL import Image
    da, db = pathlib.Path(args.a), pathlib.Path(args.b)
    if args.output:
        pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    means, p99s, shad, worst, n = [], [], [], (0, ""), 0
    for f in sorted(da.glob("*.jpg")):
        g = db / f.name
        if not g.exists():
            continue
        A, B = Image.open(f).convert("RGB"), Image.open(g).convert("RGB")
        if A.size != B.size:
            print(f"  {f.name}: sizes differ, {A.size} and {B.size}", file=sys.stderr)
            continue
        a, b = np.asarray(A, np.float32), np.asarray(B, np.float32)
        d = np.abs(a - b)
        y = a @ LUMA
        m = float(d.mean())
        means.append(m)
        p99s.append(float(np.percentile(d, 99)))
        s = y < 25
        shad.append(float(d[s].mean()) if s.sum() > 1000 else np.nan)
        if m > worst[0]:
            worst = (m, f.name)
        n += 1
        if args.output:
            sheet(A, B, pathlib.Path(args.output) / f.name, args.label_a, args.label_b)
    if not n:
        print("no frames in common", file=sys.stderr)
        return 1
    print(f"{n} frames: mean |diff| {np.mean(means):.2f}, p99 {np.mean(p99s):.2f}, "
          f"shadows {np.nanmean(shad):.2f} (levels of 255); worst {worst[1]} {worst[0]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
