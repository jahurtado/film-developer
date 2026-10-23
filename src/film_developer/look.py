"""The roll's look, set by eye: `⌘K` in the editor.

The develop has two halves. The automatics -- base, inversion, the roll's
balance, each frame's exposure and shadow rule, the curve opening for a
flat negative -- are rules with thresholds, the same for every picture.
The taste is a handful of biases on top of them, one set per roll
(`roll.develop`, `develop.TASTE`): exposure, warmth, tint, the two slopes
of the curve and saturation. They were only reachable from the command
line (`develop --warmth 0.1`), so changing the look of a roll meant
guessing a number, running the roll and opening the files.

This sheet moves them with the picture in front of you. It shows one frame
at a time, cut to the frame's upright inner box and turned and mirrored as
the roll shows it, and walks the roll with `‹` `›` (or `p` `n`) so a look
chosen on one frame can be checked on the others before it goes to all of
them. **Hold *before*** shows the frame under the look the roll has now.
Contrast and pivot sit on the curve of the frame already normalised: see
`develop.finish`.

It is cheap because the develop splits where the taste comes in: reading,
inverting and cleaning a scan (`develop.whole_positive`) is a second and
does not depend on any of these ten; `finish` after it is a tenth of a
second at the window's size. So a frame is inverted once when it is first
shown, and every slider move only runs `finish` again. What comes out is
the same picture the canvas draws and `⌘R` writes: it is `whole_job` cut to
the frame, with these numbers.

The gamma is not here: the base and the balance are measured through it
(`measured_ready`), so changing it measures the roll again. It stays a
roll-sheet decision.
"""

import copy

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QDialog, QGridLayout, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QVBoxLayout)

from . import develop, image_io

SUB = "#8a8a90"
PREVIEW = (960, 640)      # px: the largest the frame is drawn
KEEP = 8                  # frames whose inverted scan is kept while the sheet is open

# key, name, low, high, step, format, what it does
SLIDERS = (
    ("exposure", "Exposure", -1.5, 1.5, 0.05, "{:+.2f} EV",
     "On top of each frame's automatic exposure."),
    ("warmth", "Warmth", -0.4, 0.4, 0.01, "{:+.2f} EV",
     "Added to red and taken from blue, on top of the roll's balance and "
     "each frame's. The roll's default is +0.08."),
    ("tint", "Tint", -0.3, 0.3, 0.01, "{:+.2f} EV",
     "Added to green and taken from red and blue: up is greener, down is "
     "more magenta."),
    ("shadows", "Shadows", 0.5, 1.2, 0.01, "{:.2f}",
     "The curve's slope below middle grey. Under 1 opens the shadows and "
     "flattens them; over 1 deepens them. Each frame can still open it "
     "further when its negative holds little range."),
    ("highlights", "Highlights", 0.5, 1.2, 0.01, "{:.2f}",
     "The curve's slope above middle grey. Under 1 keeps more in the "
     "highlights and flattens them; over 1 brightens them."),
    ("contrast", "Contrast", 0.5, 1.6, 0.01, "{:.2f}",
     "Both slopes of the curve at once, after each frame has opened its "
     "own for a flat negative. Under 1 flatter, over 1 harder."),
    ("pivot", "Pivot", -2.0, 2.0, 0.05, "{:+.2f} EV",
     "Where the curve bends, over middle grey -- where each frame's "
     "automatic exposure has put its median. Up keeps more of the "
     "highlights at the shadows' slope; down the opposite. With the two "
     "slopes equal it acts as exposure."),
    ("saturation", "Saturation", 0.5, 1.6, 0.01, "{:.2f}",
     "1 leaves the colour as it is."),
    ("noise", "Noise reduction", 0.0, 1.0, 0.01, "{:.2f}",
     "Grain and colour noise, in the scene's light before the curve: the "
     "colour smoothed over 6 scan pixels, the grain shrunk where it is the "
     "size of the noise and kept where it is an edge. Judge it at 100 %: "
     "at the size of the window the grain is already averaged away."),
    ("sharpen", "Sharpening", 0.0, 2.0, 0.01, "{:.2f}",
     "An unsharp mask on the luminance, 1.2 scan pixels, after the curve; "
     "detail smaller than grain is left alone. Judge it at 100 %."),
)
KEYS = tuple(s[0] for s in SLIDERS)


class _Job(QThread):
    """One function off the window's thread, with the token it was asked
    for, so a result that arrives after the person moved on is dropped."""

    done = Signal(object, object)
    failed = Signal(object, str)

    def __init__(self, token, fn):
        super().__init__()
        self.token, self.fn = token, fn

    def run(self):
        try:
            out = self.fn()
        except Exception as e:               # shown, not lost in a thread
            return self.failed.emit(self.token, f"{type(e).__name__}: {e}")
        self.done.emit(self.token, out)


def frames_of(ed):
    """(scan, part) for every frame the window would develop, in the
    list's order."""
    out = []
    for name in ed.names:
        it = ed.doc["items"].get(name) or {}
        if it.get("state") == "skipped" or not ed.here(name):
            continue
        out += [(name, q["name"]) for q in it.get("parts", [])]
    return out


def _to_source(u, v, shape, mirrored, turn):
    """A point of the picture as shown -- (u, v), fractions of its width
    and height after the mirror and the turn -- as fractions of `shape`,
    the frame as developed. The same mirror and turn are done to a coarse
    grid of the frame's own coordinates and read back, so the two cannot
    disagree about which way a quarter turn goes."""
    gh, gw = max(2, shape[0] // 40), max(2, shape[1] // 40)
    yy, xx = np.mgrid[0:gh, 0:gw]
    grid = np.stack([(yy + 0.5) / gh, (xx + 0.5) / gw], -1)
    if mirrored:
        grid = grid[:, ::-1]
    if turn:
        grid = np.rot90(grid, k=-(int(turn) // 90))
    r = min(grid.shape[0] - 1, max(0, int(v * grid.shape[0])))
    c = min(grid.shape[1] - 1, max(0, int(u * grid.shape[1])))
    return float(grid[r, c, 1]), float(grid[r, c, 0])


def picture_full(w, full, taste, mirrored, turn, focus, size=PREVIEW):
    """A `size` piece of the frame at full size, round `focus` (fractions of
    the picture as shown), developed with the decisions the whole frame
    makes at the window's size (`finish(..., applied=)`): the piece cannot
    see the frame's median or its darkest twentieth, so it is told them.
    Measured on DSC_0329: such a piece and the same piece of the whole
    frame developed at full size differ by a median of 0 levels of 255."""
    _o, ev = develop.finish(w["pos"], w["roll_wb"], w["blank"], taste,
                            look_box=w["look"], px=w["px"])
    fx, fy = _to_source(focus[0], focus[1], full.shape, mirrored, turn)
    cw, ch = (size[1], size[0]) if (int(turn) // 90) % 2 else size
    H, W = full.shape[:2]
    cw, ch = min(cw, W), min(ch, H)
    x0 = int(min(max(fx * W - cw / 2, 0), W - cw))
    y0 = int(min(max(fy * H - ch / 2, 0), H - ch))
    out, _ = develop.finish(full[y0:y0 + ch, x0:x0 + cw], w["roll_wb"], w["blank"],
                            taste, px=1.0, applied=ev["applied"])
    out = develop.encode(out, 8)
    if mirrored:
        out = out[:, ::-1]
    if turn:
        out = np.rot90(out, k=-(int(turn) // 90))
    return np.ascontiguousarray(out), ev


def picture(w, taste, mirrored, turn):
    """`finish` on a frame's inverted scan, cut to the frame, as screen
    bytes the right way round. Returns (bytes, the develop's numbers)."""
    out, ev = develop.finish(w["pos"], w["roll_wb"], w["blank"], taste,
                             look_box=w["look"], px=w["px"])
    y0, y1, x0, x1 = w["look"]
    out = develop.encode(out[max(0, y0):y1, max(0, x0):x1], 8)
    if mirrored:
        out = out[:, ::-1]
    if turn:
        out = np.rot90(out, k=-(int(turn) // 90))
    return np.ascontiguousarray(out), ev


class LookSheet(QDialog):
    def __init__(self, ed):
        super().__init__(ed)
        self.ed, self.doc, self.roll = ed, ed.doc, ed.roll
        self.setWindowTitle("Develop look")
        self.setModal(True)
        self.base_taste = dict(develop.TASTE) | dict(self.doc["roll"].get("develop") or {})
        self.before = {k: float(self.base_taste[k]) for k in KEYS}
        self.frames = frames_of(ed)
        here = (ed.name(), (ed.part() or {}).get("name"))
        self.at = self.frames.index(here) if here in self.frames else 0
        self.positives = {}                 # frame -> `whole_positive`
        self.worker = None                  # the one inverting a scan
        self.finisher = None                # the one running `finish`
        self.again = False                  # a slider moved while it ran
        self.holding = False
        self.zoom = False                   # 100 %: a piece at full size
        self.focus = (0.5, 0.5)             # where, as fractions of the picture shown
        self.full = {}                      # frame -> its positive at full size (one kept)
        self.full_worker = None
        self.fitted = None                  # (x, y, w, h) of the picture in the view

        v = QVBoxLayout(self)
        v.setContentsMargins(22, 18, 22, 16)
        v.setSpacing(10)
        head = QLabel("The look of the whole roll")
        head.setStyleSheet("font-size: 15px; font-weight: 600;")
        v.addWidget(head)
        sub = QLabel("The same for every frame, on top of each frame's "
                     "automatics. Walk the roll before applying.")
        sub.setStyleSheet(f"color: {SUB};")
        v.addWidget(sub)

        self.view = QLabel()
        self.view.setFixedSize(*PREVIEW)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet("background: #141416; color: #8a8a90;")
        self.view.mousePressEvent = self._clicked
        self.view.setToolTip("Click a place to see it at 100 %")
        v.addWidget(self.view)

        nav = QHBoxLayout()
        self.prev_b = QPushButton("‹")
        self.next_b = QPushButton("›")
        for b in (self.prev_b, self.next_b):
            b.setFixedWidth(36)
            b.setFocusPolicy(Qt.NoFocus)
        self.prev_b.clicked.connect(lambda: self._walk(-1))
        self.next_b.clicked.connect(lambda: self._walk(1))
        self.where = QLabel("")
        self.where.setStyleSheet(f"color: {SUB};")
        self.hold_b = QPushButton("Hold to see before")
        self.hold_b.setFocusPolicy(Qt.NoFocus)
        self.hold_b.pressed.connect(lambda: self._hold(True))
        self.hold_b.released.connect(lambda: self._hold(False))
        nav.addWidget(self.prev_b)
        nav.addWidget(self.next_b)
        self.zoom_b = QPushButton("100 %")
        self.zoom_b.setCheckable(True)
        self.zoom_b.setFocusPolicy(Qt.NoFocus)
        self.zoom_b.setToolTip("The frame at full size round the place last "
                               "clicked (z): where grain, noise reduction and "
                               "sharpening can be seen")
        self.zoom_b.toggled.connect(self._set_zoom)
        nav.addWidget(self.where, 1)
        nav.addWidget(self.zoom_b)
        nav.addWidget(self.hold_b)
        v.addLayout(nav)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        self.sliders, self.values = {}, {}
        for i, (key, label, lo, hi, step, fmt, tip) in enumerate(SLIDERS):
            r, c = i % 5, (i // 5) * 3
            name = QLabel(label)
            name.setToolTip(tip)
            s = QSlider(Qt.Horizontal)
            s.setRange(0, int(round((hi - lo) / step)))
            s.setMinimumWidth(240)
            s.setToolTip(tip)
            val = QLabel("")
            val.setMinimumWidth(70)
            val.setStyleSheet(f"color: {SUB};")
            grid.addWidget(name, r, c)
            grid.addWidget(s, r, c + 1)
            grid.addWidget(val, r, c + 2)
            self.sliders[key], self.values[key] = s, val
            self._put(key, self.before[key])
            s.valueChanged.connect(lambda _v, k=key: self._moved(k))
        v.addLayout(grid)

        row = QHBoxLayout()
        defaults = QPushButton("Program defaults")
        defaults.setToolTip("Every slider back to what the program ships with.")
        defaults.clicked.connect(self._defaults)
        undo = QPushButton("As the roll has it")
        undo.setToolTip("Every slider back to the roll's look before this sheet.")
        undo.clicked.connect(lambda: self._set_all(self.before))
        row.addWidget(defaults)
        row.addWidget(undo)
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        go = QPushButton("Apply to the roll")
        go.setDefault(True)
        go.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(go)
        v.addLayout(row)

        for key, step in (("n", 1), ("p", -1), ("Right", 1), ("Left", -1)):
            QShortcut(QKeySequence(key), self, lambda s=step: self._walk(s))
        QShortcut(QKeySequence("z"), self, self.zoom_b.toggle)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._render)
        self._load()

    # ---- the numbers
    def _put(self, key, value):
        lo, step = next((s[2], s[4]) for s in SLIDERS if s[0] == key)
        s = self.sliders[key]
        s.blockSignals(True)
        s.setValue(int(round((float(value) - lo) / step)))
        s.blockSignals(False)
        self._say(key)

    def _say(self, key):
        fmt = next(s[5] for s in SLIDERS if s[0] == key)
        self.values[key].setText(fmt.format(self.value(key)))

    def value(self, key):
        lo, step = next((s[2], s[4]) for s in SLIDERS if s[0] == key)
        return round(lo + self.sliders[key].value() * step, 4)

    def chosen(self):
        return {k: self.value(k) for k in KEYS}

    def _moved(self, key):
        self._say(key)
        self.timer.start(40)

    def _set_all(self, values):
        for k in KEYS:
            self._put(k, values[k])
        self._render()

    def _defaults(self):
        self._set_all({k: develop.TASTE[k] for k in KEYS})

    # ---- the frame
    def _walk(self, step):
        if not self.frames:
            return
        self.at = (self.at + step) % len(self.frames)
        self._load()

    def _load(self):
        if not self.frames:
            self.view.setText("no frames found on this roll yet")
            return
        key = self.frames[self.at]
        name, pname = key
        self.where.setText(f"{name}"
                           + (f" · {pname}" if len(self._item(name)["parts"]) > 1 else "")
                           + f"   {self.at + 1} of {len(self.frames)}")
        if key in self.positives:
            return self._render()
        self.view.setText("developing…")
        if self.worker is not None:
            return                          # it starts the wanted one when done
        it = self._item(name)
        part = copy.deepcopy(next(q for q in it["parts"] if q["name"] == pname))
        item = copy.deepcopy(develop.with_specks(it, self.doc, name))
        stats = develop.stats_of(self.doc["roll"].get("measured"),
                                 self.base_taste, self.roll)
        if stats is None:
            self.view.setText("the roll's base and balance are not measured yet")
            return
        job = (self.roll / name, item, part, stats, self.base_taste,
               self.doc["roll"].get("one_roll", True), image_io.PREVIEW_WIDTH)
        self.worker = _Job(key, lambda: develop.whole_positive(job))
        self.worker.done.connect(self._inverted)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _item(self, name):
        return self.doc["items"].get(name) or {"parts": []}

    def _inverted(self, key, w):
        self.worker = None
        self.positives[key] = w
        while len(self.positives) > KEEP:
            self.positives.pop(next(iter(self.positives)))
        if key != self.frames[self.at]:
            return self._load()
        self._render()

    def _failed(self, key, why):
        self.worker = self.finisher = None
        self.view.setText(f"could not develop {key[0]}: {why}")

    # ---- 100 %
    def _set_zoom(self, on):
        self.zoom = bool(on)
        self._render()

    def _clicked(self, e):
        """In the whole frame, a click picks the place to see at 100 % and
        goes there; at 100 %, a click goes back to the whole frame."""
        if self.zoom:
            return self.zoom_b.setChecked(False)
        if not self.fitted:
            return
        x, y, w, h = self.fitted
        p = e.position()
        u, v = (p.x() - x) / max(w, 1), (p.y() - y) / max(h, 1)
        if 0 <= u <= 1 and 0 <= v <= 1:
            self.focus = (u, v)
            self.zoom_b.setChecked(True)

    def _want_full(self, key):
        """Start inverting `key` at full size, once; True if it is there."""
        if key in self.full:
            return True
        if self.full_worker is not None:
            return False
        name, pname = key
        it = self._item(name)
        part = copy.deepcopy(next(q for q in it["parts"] if q["name"] == pname))
        item = copy.deepcopy(develop.with_specks(it, self.doc, name))
        stats = develop.stats_of(self.doc["roll"].get("measured"),
                                 self.base_taste, self.roll)
        one = self.doc["roll"].get("one_roll", True)
        path, taste = self.roll / name, dict(self.base_taste)
        self.full_worker = _Job(key, lambda: develop.full_positive(
            path, item, part, stats, taste, one))
        self.full_worker.done.connect(self._full_ready)
        self.full_worker.failed.connect(self._failed)
        self.full_worker.start()
        self.where.setText(self.where.text().split("   ·")[0] + "   · developing at 100 %…")
        return False

    def _full_ready(self, key, full):
        self.full_worker = None
        self.full = {key: full}             # one frame at full size is 200 MB
        self.where.setText(self.where.text().split("   ·")[0])
        self._render()

    # ---- the picture
    def _hold(self, on):
        self.holding = on
        self._render()

    def _render(self):
        key = self.frames[self.at] if self.frames else None
        w = self.positives.get(key)
        if w is None:
            return
        if self.finisher is not None:
            self.again = True
            return
        taste = dict(self.base_taste) | (self.before if self.holding else self.chosen())
        it = self._item(key[0])
        part = next(q for q in it["parts"] if q["name"] == key[1])
        mirrored = self.doc["roll"].get("mirrored", True)
        turn = int(part.get("turn", 0) or 0) % 360
        zoom = self.zoom and self._want_full(key)
        if self.zoom and not zoom:
            return                          # `_full_ready` renders again
        if zoom:
            full, focus = self.full[key], self.focus
            fn = lambda: picture_full(w, full, taste, mirrored, turn, focus)
        else:
            fn = lambda: picture(w, taste, mirrored, turn)
        self.finisher = _Job((key, self.holding, zoom), fn)
        self.finisher.done.connect(self._shown)
        self.finisher.failed.connect(self._failed)
        self.finisher.start()

    def _shown(self, token, out):
        self.finisher = None
        key, held, zoom = token
        if self.again or key != (self.frames[self.at] if self.frames else None) \
                or held != self.holding or zoom != self.zoom:
            self.again = False
            return self._render()
        a, ev = out
        h, w = a.shape[:2]
        pix = QPixmap.fromImage(QImage(a.data, w, h, 3 * w,
                                       QImage.Format_RGB888).copy())
        if not zoom:
            pix = pix.scaled(self.view.size(), Qt.KeepAspectRatio,
                             Qt.SmoothTransformation)
            vw, vh = self.view.width(), self.view.height()
            self.fitted = ((vw - pix.width()) / 2, (vh - pix.height()) / 2,
                           pix.width(), pix.height())
        self.view.setPixmap(pix)
        if held:
            self.view.setToolTip("before")

    def done(self, r):
        # Nothing may outlive the sheet while it is still running.
        for t in (self.worker, self.finisher, self.full_worker):
            if t is not None:
                t.wait()
        super().done(r)


def ask(ed):
    """The ten numbers chosen, or None if the sheet was cancelled."""
    from .sheets import centre
    dlg = LookSheet(ed)
    dlg.adjustSize()
    centre(dlg)
    return dlg.chosen() if dlg.exec() == QDialog.Accepted else None
