#!/usr/bin/env python3
"""Drive the editor with no screen, photograph it, and hold it against the files.

    uv run python tools/uishot.py ROLL -o SHOTS                   # look
    uv run python tools/uishot.py ROLL -o SHOTS --frames 022.DNG,002.DNG
    uv run python tools/uishot.py ROLL -o SHOTS --process         # run ⌘R first

The window is the program. Every measurement in this repository that was
made from the command line describes what `develop` writes, not what a
person sees, and the two have already been apart once: `d` drew the
develop without its transfer function and came out 72 levels of 255 dark,
with every decision of the automatics identical. Nothing on the command
line could have caught that.

So this opens the **real editor** on the `offscreen` platform -- the same
Qt, the same canvas, the same key handling, the same render thread -- and
presses real keys through `QTest`. `win.grab()` gives the pixels the
window would have shown. There is no server to run and no display to
find; `cocoa`, `minimal` and `offscreen` are the only platforms this Qt
was built with, and `offscreen` is the one that renders.

What it prints, per frame:

- what the develop decided, off the read line (`d`'s own evidence);
- how long the render took and the longest the event loop was blocked,
  because a window that stops answering is a bug of its own;
- and, where `developed/NAME.jpg` exists, **the canvas against the
  file**: the median of each channel and the mean per pixel. The medians
  are the real check -- the per-pixel figure carries the nearest-neighbour
  resample of a 5000 px picture down to 1250, which is 2.31 of 255 on its
  own.

`--process` runs `⌘R` first, for both outputs, so the files it compares
against are the ones this document and these settings produce rather than
whatever was in the folder from another afternoon. That is the
comparison worth making: same session, same measurement, same taste.
"""

import argparse
import os
import pathlib
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                            # noqa: E402

from film_developer import document, develop                  # noqa: E402


def settle(app, win, limit=120.0):
    """Wait for the render thread, timing the worst gap in the event loop.

    The gap is the measurement that says whether the window is still a
    window while it develops: press a key during a long render and this is
    how long it waits to be heard.
    """
    t0 = last = time.monotonic()
    worst = 0.0
    while win.dev_worker is not None and time.monotonic() - t0 < limit:
        app.processEvents()
        now = time.monotonic()
        worst, last = max(worst, now - last), now
        time.sleep(0.004)
    app.processEvents()
    return time.monotonic() - t0, worst


def canvas_pixels(win):
    """What the canvas is showing, as RGB bytes."""
    from PySide6.QtGui import QImage
    pm = win.canvas.current()
    if pm is None:
        return None
    im = pm.toImage().convertToFormat(QImage.Format.Format_RGB888)
    a = np.frombuffer(bytes(im.constBits()), np.uint8).reshape(
        im.height(), im.bytesPerLine())[:, :im.width() * 3]
    return a.reshape(im.height(), im.width(), 3).astype(np.int16)


def against_file(shown, path):
    """(per-channel median gap, mean per pixel) between the canvas and a
    developed file, both in 8-bit sRGB. The file is sampled down to the
    canvas by nearest neighbour, which is most of the per-pixel number."""
    from PIL import Image
    big = np.asarray(Image.open(path)).astype(np.int16)
    ys = np.linspace(0, big.shape[0] - 1, shown.shape[0]).astype(int)
    xs = np.linspace(0, big.shape[1] - 1, shown.shape[1]).astype(int)
    ref = big[ys][:, xs]
    med = [abs(float(np.median(big[..., i])) - float(np.median(shown[..., i])))
           for i in range(3)]
    return med, float(np.abs(ref - shown).mean()), big.shape


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roll", metavar="ROLL")
    ap.add_argument("-o", "--output", default="shots",
                    help="where the PNGs go (default: shots/)")
    ap.add_argument("--frames", default="",
                    help="scan names, comma separated (default: the first 6)")
    ap.add_argument("--process", action="store_true",
                    help="run ⌘R first, so the files compared against are "
                         "the ones these settings produce")
    ap.add_argument("--raws", action="store_true",
                    help="with --process, write the positive DNGs too")
    ap.add_argument("--size", default="1400x900")
    args = ap.parse_args(argv)

    roll = pathlib.Path(args.roll).expanduser()
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    w, h = (int(v) for v in args.size.lower().split("x"))

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from film_developer.editor import Editor, dark_palette
    from film_developer.progress import measure_base

    app = QApplication.instance() or QApplication([])
    dark_palette(app)
    doc = document.load(roll)
    taste = dict(develop.TASTE) | dict(doc["roll"].get("develop") or {})
    print(f"{roll.name}: {len(doc['items'])} scans, "
          f"base {taste.get('base')}, "
          f"gamma {taste.get('gamma')}, crop {taste.get('crop')} %")

    t = time.monotonic()
    fresh = not develop.measured_ready(doc["roll"].get("measured"))
    measure_base(roll, doc)
    if fresh:
        document.save(doc, roll)
        print(f"   the roll measured ({time.monotonic() - t:.0f} s)")
    m = doc["roll"].get("measured") or {}
    print(f"   measured under: gamma "
          f"{(m.get('under') or {}).get('gamma')}; "
          f"balance {m.get('balance_log2')}")

    win = Editor(roll, doc, ("film-developer", "elcacharrista.com"))
    win.resize(w, h)
    win.show()
    app.processEvents()
    here = win.canvas

    if args.process:
        # The sheet is a question, not logic: it is answered rather than
        # clicked, so the body that does the work is the real one.
        from film_developer import processing
        processing.ask = lambda d, parent=None: {
            "positives": True, "format": taste.get("format", "jpeg"),
            "raws": bool(args.raws), "crop": "outer"}
        t = time.monotonic()
        win._process()
        app.processEvents()
        print(f"   ⌘R ran ({time.monotonic() - t:.0f} s): {win.status.text()}")

    names = [q.strip() for q in args.frames.split(",") if q.strip()] \
        or win.names[:6]
    dev_dir = roll / "developed"
    rows = []
    for name in names:
        if name not in win.names:
            print(f"   {name}: not in the document")
            continue
        win.list.setCurrentRow(win.names.index(name))
        app.processEvents()
        for _ in range(4):
            if win.colour == "develop":
                break
            QTest.keyClick(here, Qt.Key_V)
            settle(app, win)
        if not win.cut:                  # the file is one frame, so cut to it
            QTest.keyClick(here, Qt.Key_Space)
            settle(app, win)
        secs, worst = settle(app, win)
        shown = canvas_pixels(win)
        png = out / f"{pathlib.Path(name).stem}.png"
        win.grab().save(str(png))
        line = (f"{name}: {secs:5.2f} s, loop gap {worst * 1000:3.0f} ms  "
                f"| {win.held.text()}")
        part = (win.part() or {}).get("name", "a")
        n_parts = len(win.item().get("parts") or [])
        suffix = f"-{part}" if n_parts > 1 else ""
        f = dev_dir / (develop.out_stem(win.doc, name, win.item(),
                                        win.part() or win.item()["parts"][0])
                       + ".jpg")
        if shown is None:
            line += "  | nothing on the canvas"
        elif f.exists():
            med, per_px, big = against_file(shown, f)
            line += (f"\n      against {f.name} ({big[1]}x{big[0]} against "
                     f"{shown.shape[1]}x{shown.shape[0]}): medians apart "
                     f"R {med[0]:.1f} G {med[1]:.1f} B {med[2]:.1f} of 255, "
                     f"{per_px:.2f} per pixel")
            rows.append(max(med))
        else:
            line += f"\n      no {f.name} to compare against"
        print(line)
    if rows:
        print(f"\nworst median gap between the canvas and the files: "
              f"{max(rows):.1f} of 255 over {len(rows)} frames")
    print(f"pictures of the window in {out}")
    win.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
