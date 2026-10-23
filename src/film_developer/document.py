"""The format of what sits between the three steps: `frames.json`.

It lives in the folder of scans, so all three commands find it from the folder
alone. One folder, one document: the scans do not move, so a second opinion
about where the frames are in them is not worth keeping.

Two rules make it usable rather than merely present, and both are enforced
here rather than by the code that happens to write it:

- **Every decision carries the evidence behind it.** `angle` is never stored
  without `gain` and `margin`, because a peak that is 8% above what an
  obviously wrong angle scores is not a measurement, and on the sample set
  that is exactly how the one bad angle announced itself.
- **A hand correction survives a re-measure.** `manual` lists, by path, what a
  person set. `detect` recomputes everything except those.
"""

import json
import pathlib

FILENAME = "frames.json"
VERSION = 2

# What `state` a record can be in, and what each one claims. `refused` and
# `skipped` look alike and are not: one is the program saying it could not,
# the other is a person saying not this one.
STATES = ("detected", "held", "manual", "refused", "skipped")


def blank(roll=None):
    return {"film-developer": VERSION,
            "roll": {"crop": 0,
                     "format": "auto", "aspect": "native", "snap": "shrink",
                     "straighten": "auto", "mirrored": True, "stock": None,
                     "camera": None, "shot": None,
                     "one_roll": True, "same_size": True,
                     "same_gap": False} | (roll or {}),
            "items": {}}


def migrate(doc):
    """Bring an older document forward, rather than refusing it.

    v1 -> v2 moves `angle` from the part to the **item**. The two exposures of
    a half-frame shot are on one piece of film: they cannot have different
    tilts, and storing the tilt per part left it free to contradict itself --
    turning one part and not the other produced a scan whose two frames
    disagreed about how the film was lying.

    The evidence for the angle moves with it. A hand correction recorded
    against a part's angle becomes one against the item's.
    """
    # Before the version check, not inside it: a setting added after a
    # document was written is missing from it whatever version it claims, and
    # a document already at the current version would skip them entirely.
    roll = doc.setdefault("roll", {})
    # **No `detector` any more.** There was one, and a document that did not
    # name one was given `profiles` here -- the first detector, and the only
    # one when those documents were written. It was faithful and it was a
    # trap: `fit` arrived, a folder measured before it kept `profiles`
    # forever, the window offered no way to say otherwise, and a half-frame
    # roll went on being read at 31 of 37 where `fit` reads 36. There is one
    # detector now, so the key means nothing; an old document may still
    # carry it and nothing reads it.
    # **`same_size` and `same_gap` are both off by default.** They are the
    # two rules that **correct** a side somebody measured -- the height to
    # the roll's, the gap to the roll's -- and a measurement is the one
    # thing the roll may not overrule. `same_size` buys files that are all
    # one size and costs accuracy: measured against the placed frames it
    # changes nothing on two of the three labelled rolls and on the third
    # it costs frames off the top, because at the 1.5-2 % it fires on what
    # is usually wrong is where the frame sits and not how tall it is.
    # A document written before them says nothing, and nothing is what it
    # should be read as.
    #
    # **A new roll starts with `same_size` on**: `blank` writes it True, so a folder opened for the first
    # time is asked with the box ticked. Since it became one size for the
    # whole roll rather than one per cluster (`fitframe.cluster_for`), it
    # mended carrete_2's frames read into the rebate -- heights 3549-3594
    # and widths 5298-5367 against 3558-3876 and up to 5853. Off is still
    # right for a roll whose stand moved. An old document without the key
    # keeps reading it as off, here, for the reason above.
    for key, default in (("crop", 0), ("mirrored", True),
                         ("one_roll", True), ("same_size", False),
                         ("same_gap", False)):
        roll.setdefault(key, default)

    v = doc.get("film-developer")
    if v == VERSION:
        return doc
    if v != 1:
        raise ValueError(f"document version {v}, and this is version {VERSION}")
    for item in doc["items"].values():
        parts = item.get("parts", [])
        angles = [p.pop("angle", 0.0) for p in parts]
        item["angle"] = angles[0] if angles else 0.0
        item["evidence"] = {}
        for p in parts:
            for key in ("angle_gain", "angle_margin"):
                if key in p.get("evidence", {}):
                    item["evidence"][key] = p["evidence"].pop(key)
        item["manual"] = [
            "angle" if m.endswith(".angle") else m for m in item.get("manual", [])]
        # De-duplicated: two parts' angles both becoming the item's angle
        # would otherwise list it twice.
        item["manual"] = list(dict.fromkeys(item["manual"]))
        # And the same lettering rule as everything else: reading order as the
        # picture reads, not as the file is laid out. Only the letters move;
        # the rectangles and the hand corrections attached to them do not.
        order_parts(item, doc["roll"].get("mirrored", True))
    doc["film-developer"] = VERSION
    return doc


def path_for(where):
    """The document's path, given a folder or the document itself.

    Naming the folder is enough for all three commands, which is the point of
    keeping it there; naming the file works too, so `-o` has somewhere to put
    one and the other commands can still be pointed at it.
    """
    p = pathlib.Path(where)
    return p if p.suffix == ".json" else p / FILENAME


def load(where):
    p = path_for(where)
    if not p.exists():
        raise FileNotFoundError(f"no {FILENAME} in {p.parent} -- run detect first")
    doc = json.loads(p.read_text())
    v = doc.get("film-developer")
    if v is not None and v > VERSION:
        # A document from a *later* version is refused with a sentence naming
        # what wrote it, rather than read optimistically and misunderstood in
        # one field out of forty. An older one is brought forward instead:
        # refusing it would throw away hand corrections nothing can reproduce.
        raise ValueError(f"{p} was written by document version {v}, "
                         f"and this is version {VERSION}")
    return migrate(doc)


def save(doc, where):
    """Written beside and renamed over, so the document is the old one or
    the new one and never half of each: the editor writes it on every
    change, and the process can be stopped at any moment (Ctrl+C in the
    terminal ends it on the spot, `gui.keep_reachable`)."""
    p = path_for(where)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    tmp.replace(p)
    return p


def order_parts(item, mirrored=True):
    """Letter the parts in reading order, as the picture reads on screen.

    A negative photographed through its base comes out reversed, so the frame
    that reads first is the one on the *right* of the file. Lettering by the
    file's own coordinates puts `a` on the right of the screen and `b` on the
    left, and then `DSC_1234-a.dng` is the second photograph.

    The `manual` paths are rewritten with the letters, so a hand correction
    stays attached to the part it was made on rather than to whatever now
    holds its old letter.
    """
    parts = item.get("parts", [])
    if not parts:
        return item
    w = item["scan"][0]
    pos = [((w - p["centre"][0]) if mirrored else p["centre"][0], p["centre"][1])
           for p in parts]
    # Along whichever axis they are actually spread out: half frame is side by
    # side, but a strip photographed the other way round is not.
    spread_x = max(x for x, _ in pos) - min(x for x, _ in pos)
    spread_y = max(y for _, y in pos) - min(y for _, y in pos)
    key = (lambda i: (pos[i][0], pos[i][1])) if spread_x >= spread_y else \
          (lambda i: (pos[i][1], pos[i][0]))
    order = sorted(range(len(parts)), key=key)

    was = [p["name"] for p in parts]
    now = list("abcdefgh"[:len(parts)])
    rename = {}
    for slot, i in enumerate(order):
        rename[was[i]] = now[slot]
        parts[i]["name"] = now[slot]
    # Refilled in place, never swapped: the editor's canvas holds this very
    # list, and handing the item a new one leaves the canvas drawing something
    # nobody else can see.
    parts[:] = [parts[i] for i in order]
    item["manual"] = [_rename_path(m, rename) for m in item.get("manual", [])]
    return item


def _rename_path(path, rename):
    bits = path.split(".")
    if len(bits) >= 2 and bits[0] == "parts" and bits[1] in rename:
        bits[1] = rename[bits[1]]
    return ".".join(bits)


def is_manual(item, field):
    return field in item.get("manual", ())


def mark_manual(item, field):
    """Record that a person set this, so re-measuring will not undo it."""
    m = item.setdefault("manual", [])
    if field not in m:
        m.append(field)
    item["state"] = "manual"


def order_of(doc):
    """The scans in the order the list shows them: `roll.order`, the order
    a person put them in, and after it any scan it does not name, by file
    name.

    **The order the pictures were taken in is not always the order they
    were scanned in**: a strip can go through the holder
    either way round, and strips are scanned in whatever order they come
    out of the sleeve. So the list's order is a fact a person states, kept
    in the roll, and file name order is only where it starts. A scan that
    arrives later goes at the end, by name, rather than in a place nobody
    chose; a name the folder no longer has is passed over, not dropped,
    so it comes back to its place if the scan does."""
    items = doc["items"]
    given, seen = [], set()
    for n in doc["roll"].get("order") or []:
        if n in items and n not in seen:
            given.append(n)
            seen.add(n)
    return given + sorted(n for n in items if n not in seen)


def shot_numbers(doc):
    """{scan: its shot number}, counting in the list's order (`order_of`)
    only the scans that are developed: a skipped scan (`del`) has none
    and does not take one from the scans after it: the numbers are the
    roll's pictures, and a skipped scan is not one."""
    out, n = {}, 0
    for name in order_of(doc):
        if (doc["items"].get(name) or {}).get("state") == "skipped":
            continue
        n += 1
        out[name] = n
    return out
