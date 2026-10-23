#!/usr/bin/env python3
"""One dust mark, before and after, at 100 % and enlarged.

    uv run python tools/dust_mark.py ROLL/DSC_0238.NEF 17 [18 ...] -o OUT

The number is the mark's own (`id` in `frames.json`), which the window shows
by the pointer in the dust mode. For talking about one mark: "17 looks
wrong" becomes a picture of 17 that both sides can look at. Writes
`OUT/mark-17.png`, the develop without and with the scan's dust taken out,
side by side, 3x, plus the hole the cleaning filled outlined in the middle.
Reads only.
"""

import argparse
import pathlib

import numpy as np

from film_developer import develop, document


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("raw")
    ap.add_argument("ids", nargs="+", type=int)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--half", type=int, default=48, help="px either side, at 100 %%")
    a = ap.parse_args(argv)
    from PIL import Image
    path = pathlib.Path(a.raw).expanduser()
    out = pathlib.Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    doc = document.load(path.parent)
    item = doc["items"][path.name]
    taste = dict(develop.TASTE) | dict(doc["roll"].get("develop") or {})
    stats = develop.stats_of(doc["roll"]["measured"], taste, path.parent)
    one = doc["roll"].get("one_roll", True)
    marks = {m.get("id"): m for m in
             (doc.get("specks") or {}).get(path.name, {}).get("marks", [])}
    for part in item["parts"]:
        before, _ = develop.develop_frame(path, item, part, stats, taste, one)
        after, _ = develop.develop_frame(path, develop.with_specks(item, doc, path.name),
                                         part, stats, taste, one)
        b8, a8 = develop.encode(before, 8), develop.encode(after, 8)
        aff = np.array(develop.frame_to_scan(item, part, taste))
        inv = np.linalg.inv(aff[:, :2])
        for i in a.ids:
            m = marks.get(i)
            if m is None:
                print(f"#{i}: no such mark on {path.name}")
                continue
            x, y = inv @ (np.array([m["x"], m["y"]]) - aff[:, 2])
            x, y, h = int(round(x)), int(round(y)), a.half
            if not (0 <= x < b8.shape[1] and 0 <= y < b8.shape[0]):
                continue
            sl = (slice(max(0, y - h), y + h), slice(max(0, x - h), x + h))
            changed = (np.abs(b8[sl].astype(int) - a8[sl]).max(-1) > 2)
            edge = changed & ~np.pad(changed, 1)[2:, 1:-1] | \
                changed & ~np.pad(changed, 1)[:-2, 1:-1] | \
                changed & ~np.pad(changed, 1)[1:-1, 2:] | \
                changed & ~np.pad(changed, 1)[1:-1, :-2]
            mid = a8[sl].copy()
            mid[edge] = (255, 40, 180)
            gap = np.full((b8[sl].shape[0], 4, 3), 255, np.uint8)
            row = np.concatenate([b8[sl], gap, mid, gap, a8[sl]], 1)
            Image.fromarray(row).resize((row.shape[1] * 3, row.shape[0] * 3),
                                        Image.NEAREST).save(out / f"mark-{i}.png")
            how = "hand" if m.get("by", "hand") == "hand" else m.get("state")
            print(f"#{i} ({how}{', r %.0f' % m['r'] if m.get('r') else ''}) at frame "
                  f"{part['name']} ({x}, {y}): {int(changed.sum())} px changed -> mark-{i}.png")


if __name__ == "__main__":
    main()
