"""The editor window.

    two bars corner to corner -- the read line at the top, the command sheet
    at the bottom -- with the canvas and the list of items between them.

It edits the document and nothing else: no image is written here, whatever is
pressed. What a person changes goes into `manual`, so measuring again cannot
undo it, and turns pink on the canvas and in the list, because `#ff6ac1` means
"a person set this" everywhere in the window.
"""

import copy
import json
import math
import os
import pathlib
import time

import numpy as np
from PySide6.QtCore import (QEvent, QPoint, QPointF, QRectF, QSize, Qt, QThread,
                            QTimer,
                            Signal)
from PySide6.QtGui import (QTransform, QAction, QColor, QFont, QFontMetrics, QImage,
                           QPainterPath,
                           QKeySequence, QPainter, QPalette, QPen, QPixmap,
                           QPolygonF, QTextDocument)
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QDoubleSpinBox,
                               QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMainWindow, QPushButton, QStackedWidget, QStyle,
                               QStyledItemDelegate, QVBoxLayout, QWidget)

from . import document, findframe, image_io, sheets

# ---------------------------------------------------------------- the palette
# A colour means one thing everywhere: these are about where a value came from
# and what it is worth.
WINDOW, BASE, BUTTON = "#1c1c1e", "#141416", "#2c2c2e"
TEXT, HILITE = "#e8e8ea", "#0a84ff"
BAR_BG, BAR_LINE = "#1a1b20", "#26282e"

DETECTED, HELD, MANUAL, GONE = "#5ac8fa", "#ffb340", "#ff6ac1", "#ff5f56"
SUN, FAST, REF = "#ffd400", "#32d74b", "#a78bfa"
# Where the measured positive puts white. The base divides out the
# scan's exposure, so what is left is the negative's own scale, and
# the median of a well-exposed frame sat at 6 to 8 over the four rolls.
PLAIN_WHITE = 34.0
# How close a pixel has to be to the base to be drawn as the base. Wide
# enough that the light's falloff across one scan -- 20 % on roma, end
# to end -- does not hide the film at the far side of the frame.
BASE_TOL = 0.08
# `d`, the develop, is rendered from the scan read at half size and then
# taken every other pixel: **four scan pixels a side, which is exactly what
# the ordinary canvas already has.** `image_io.preview` bins a 6064 px scan
# to 1516, and half size taken every other pixel is 1516 too, so holding or
# leaving `d` does not change the grain of what is on screen.
#
# It is a reduced render and it is the same develop. Measured against the
# full-size develop over 254 frames.
# The width every way of looking at a scan is drawn at, and the width the
# develop is kept at: `image_io.PREVIEW_WIDTH`. One number, because the
# canvas draws the frames from one scale and two pictures of one scan at two
# sizes put every rectangle in the wrong place.
CUT_INSET = 0.01

# **Three ways of looking at a scan, and they are the only three.** They are
# about colour and nothing else; cropping (`space`), mirroring (`m`) and
# turning (`r`) are separate axes and never change which of these is on.
#
#   the scan        the negative as the sensor saw it, no normalisation
#   the positive    the base divided out and the film's curve undone --
#                   **the pixels `apply` writes into the positive DNG**,
#                   the same two lines of arithmetic
#   the develop     all of it: balance, exposure, curve, black, shoulder
#
# There used to be a fourth, and it was the default: a per-channel stretch to
# the frame's 2nd and 98th percentiles. It is not a way of developing a
# negative -- it puts 2.00 % of the picture on pure black and 2.03 % on pure
# white *by construction* (measured on 023 of negativos-portra400), which
# makes every scan look equally well exposed whether it is or not -- and it
# was being read as a result. It still exists, because the frame edge on bare
# film is a 0.24 D step against 2.01 D on a masked holder and an autostretch
# is what makes that findable; but it belongs to **placing a frame**, so it
# is what the canvas draws inside the placement mode and what the two knobs
# under the canvas move, and it is no longer one of the ways of looking at a
# picture.
MODES = ("scan", "positive", "develop")
MODE_NAMES = {"scan": "the scan, as it came",
              "positive": "the positive — what the raw would hold",
              "develop": "developed"}

HOVER, PRIMARY, COMMAND = "#ffffff", "#e8e8ea", "#c8c8cc"
EXPLAIN, SMALL, SUB = "#8a8a90", "#70737c", "#5a5a60"
ASLEEP, NOKEY = "#4a4c53", "#3f4148"
KEY, MODIFIER = "#ffd400", "#9a7a12"

MONO = "Menlo"

# Below this the angle peak is not a measurement -- detect.WEAK_MARGIN, kept
# here too so the window can say so without importing the step.
WEAK = 1.15
# One press of a turn key, in degrees; shift makes it ten times finer. There is
# no constant for an arrow press: `Canvas.step()` works it out from the zoom,
# because placing a corner is done zoomed in and moving the whole film is done
# zoomed out, and a fixed number of scan pixels is the wrong step for one of
# the two whatever it is set to.
STEP_DEG = 0.1


def dark_palette(app):
    """Fusion with a dark palette set by hand."""
    app.setStyle("Fusion")
    p = QPalette()
    for role, col in ((QPalette.Window, WINDOW), (QPalette.Base, BASE),
                      (QPalette.Button, BUTTON), (QPalette.AlternateBase, BUTTON),
                      (QPalette.ToolTipBase, BUTTON), (QPalette.Text, TEXT),
                      (QPalette.WindowText, TEXT), (QPalette.ButtonText, TEXT),
                      (QPalette.ToolTipText, TEXT), (QPalette.Highlight, HILITE),
                      (QPalette.HighlightedText, "#ffffff")):
        p.setColor(role, QColor(col))
    app.setPalette(p)


# ----------------------------------------------------------------- the canvas
class ScanList(QListWidget):
    """The list of scans, which can now hold several at once (to drag them
    together). `QListWidget.setCurrentRow` does not touch the selection
    outside single selection, so walking with `n` left the scans picked
    before still picked and ⌥↑ moved them too; moving the current row
    here moves the selection with it, and only ⇧ and ⌘ with the pointer
    widen it."""

    def setCurrentRow(self, row, *flags):
        if flags:
            return super().setCurrentRow(row, *flags)
        from PySide6.QtCore import QItemSelectionModel
        super().setCurrentRow(row, QItemSelectionModel.ClearAndSelect)


class Canvas(QWidget):
    """The scan, with what was found drawn on it.

    A real view and not a scaled label: the wheel zooms about the pointer,
    dragging moves it, `f` fits. What is worth looking at here is the edge of
    an exposed frame, which is a ramp a few pixels wide, and at fit-to-window
    that is one pixel.

    It holds the part records themselves, not a copy of their corners, so an
    edit shows up by repainting and there is no second place for the geometry
    to be right in.
    """

    HANDLE = 9          # px on screen; how near a corner counts as grabbing it

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.pixmap = self.pixmap_base = None
        self.pixmap_dev = None            # the frame developed
        self.parts, self.manual = [], set()
        self.angle = 0.0                  # the item's tilt: one per scan
        self.mirrored = True              # shown the right way round
        self.scan_w = 0
        self.scale = 1.0                  # scan pixels per preview pixel
        self.sel = 0
        self.mode = False
        self.view = None                  # None, "raw", "base", "result" or "developed"
        self.crop_pct = 0.0               # drawn on the frame, live
        self.label = ""                   # which way of looking, on the canvas
        self.zoom, self.offset = 1.0, QPoint(0, 0)
        self._fitted_turn = 0             # the view turn the zoom was fitted to
        # `⇧B`: a box dragged over unexposed film, in scan pixels
        self.picking_base = False
        self.base_box = None              # (x0, y0, x1, y1), drawn while there
        self._base_from = None
        self.base_picked = lambda box: None
        # The cut and the develop are drawn fitted to the window rather than
        # in the scan's coordinates, so they carry their own zoom and pan.
        # Without them the one view a person judges a picture in was the one
        # view that could not be looked into.
        self.cut_zoom, self.cut_pan = 1.0, QPointF(0, 0)
        self._pan = None
        self._grab = None                 # (part, corner index) being dragged
        self.vertex = 0                   # which corner the arrows move
        self._shift_from = None           # where a ⇧/⌥ drag was last, in scan px
        self._shift_scope = "frame"       # what that drag is moving
        self._pan_from = None             # where a drag started, to tell a click
        self._pan_at = QPointF(0, 0)      # the same, for a fitted view
        self._moved = False               # whether this drag moved anything
        self._had_placed = False          # whether it already had vertices
        self.edited = lambda field: None  # the window's hook, set on build
        self.turned = lambda angle: None  # the window's hook for the item tilt
        self.picked = lambda i: None      # the window's hook for the selection
        self.dust_clicked = lambda pos: None  # the window's hook for a dust mark
        self.dust_resized = lambda pos, steps: None  # and for ⇧ + the wheel
        self.dust_hover = None            # the pointer, over the dust frame
        self.dust_named = lambda pos: None  # the window's: the mark under it, named
        self.dust_tip = None              # what that said, drawn by the pointer
        self.dust_brush = None            # picture px, or None: dust.STROKE_R
        self.dust_marks = []              # (x, y) on the full-size frame shown
        self.dust_stroked = lambda pts: None  # the window's hook for a painted stroke
        self._stroke = None               # the stroke being painted, screen points
        # the window's, for ⌘ + drag on a mark: pressed, moved, let go
        self.dust_clone_press = lambda pos: None
        self.dust_clone_drag = lambda pos: None
        self.dust_clone_release = lambda pos: None
        self._cloning = False
        # a picture laid over the frame, in picture px: what a mark would
        # look like cloned from under the pointer, while ⌘ is held
        self.dust_overlay = None          # (QPixmap, x, y) or None
        self.dust_source = None           # ((x, y) of the mark, (x, y) of its source)
        self.dragged = lambda scope: None  # the window's hook for a widened drag
        # Where the raw clipped, blinking over the scan as it came. The
        # window answers for a pixmap, so the marks go only on the picture
        # they were read off and not on a positive or a develop of it.
        self.clip_for = lambda pm: None
        self._blink = 0
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(450)
        self._blink_timer.timeout.connect(self._blink_step)

    # ---- what is on it
    def show_scan(self, pixmap, parts, manual, scale,
                  angle=0.0, mirrored=True, scan_w=0):
        self.pixmap = pixmap
        self.pixmap_base = None
        self.parts, self.manual, self.scale = parts, manual, scale
        self.angle, self.mirrored, self.scan_w = angle, mirrored, scan_w
        self.sel = 0
        self.fit()

    def current(self):
        if self.view in ("developed", "dust"):
            # No fallback. Until the render arrives there is nothing to
            # show, and the scan underneath is not a picture of the
            # develop: show nothing, not the previous one.
            return self.pixmap_dev
        if self.view == "base" and self.pixmap_base:
            return self.pixmap_base
        return self.pixmap

    def selected(self):
        return self.parts[self.sel] if 0 <= self.sel < len(self.parts) else None

    # ---- the view
    def cut_view(self):
        """Whether what is on screen is drawn fitted rather than in the
        scan's own coordinates: the cut, and the develop of a frame."""
        v = self.view or ""
        return v in ("developed", "dust") or v.startswith("result")

    def view_turn(self):
        """Degrees clockwise the whole scan is drawn turned: the frames'
        own turn (`r`), when they all share one.

        **The scan turns with its frames**: the
        turn was baked into the cut and the held view and nowhere else, so
        a portrait frame lay on its side in the view a roll is walked in.
        It is a turn of the drawing about the middle of the window, after
        the mirror -- the order the cut applies them in -- so the
        rectangles, the names and the pointer all go round with it and the
        scan's own coordinates do not change. Not while the corners are
        being placed: there the arrows move a corner along the scan's axes,
        and on a turned drawing *up* would move it sideways. Not either
        when the frames of a scan disagree -- two halves of one row turned
        two ways have no one way up between them."""
        if self.mode or self.cut_view() or self.view not in (None, "base") \
                or not self.parts:
            return 0
        turns = {int(p.get("turn", 0) or 0) % 360 for p in self.parts}
        return turns.pop() if len(turns) == 1 else 0

    def _view_xf(self):
        t = self.view_turn()
        if not t:
            return None
        c = QPointF(self.width() / 2, self.height() / 2)
        return QTransform().translate(c.x(), c.y()).rotate(t).translate(-c.x(), -c.y())

    def unturn(self, pos):
        """A point on screen, as it would be on the drawing before the
        view's turn: what every use of the pointer on the scan reads."""
        xf = self._view_xf()
        return xf.inverted()[0].map(QPointF(pos)) if xf is not None else QPointF(pos)

    def fit(self):
        self.cut_zoom, self.cut_pan = 1.0, QPointF(0, 0)
        pm = self.current()
        if not pm:
            return
        self._fitted_turn = self.view_turn()
        if self._fitted_turn % 180:
            self.zoom = min(self.width() / pm.height(), self.height() / pm.width())
        else:
            self.zoom = min(self.width() / pm.width(), self.height() / pm.height())
        self.offset = QPoint(int((self.width() - pm.width() * self.zoom) / 2),
                             int((self.height() - pm.height() * self.zoom) / 2))
        self.update()

    def resizeEvent(self, e):
        self.fit()

    def wheelEvent(self, e):
        if not self.current():
            return
        if self.view == "dust" and e.modifiers() & Qt.ShiftModifier:
            # A Mac turns a vertical wheel into a horizontal one while ⇧ is
            # down, so either axis is the same turn of the wheel.
            a = e.angleDelta()
            turn = a.y() if abs(a.y()) >= abs(a.x()) else a.x()
            return self.dust_resized(e.position(), turn / 120.0)
        f = 1.0015 ** e.angleDelta().y()
        p = e.position()
        if self.cut_view():
            # About the pointer here too, in the fitted picture's own
            # coordinates: the middle of the window is where the fit put
            # the middle of the frame.
            c = QPointF(self.width() / 2, self.height() / 2)
            self.cut_pan = (p - c) - (p - c - self.cut_pan) * f
            self.cut_zoom *= f
            return self.update()
        # Zoom about the pointer: what is under it stays under it.
        p = self.unturn(p)
        self.offset = QPoint(int(p.x() - (p.x() - self.offset.x()) * f),
                             int(p.y() - (p.y() - self.offset.y()) * f))
        self.zoom *= f
        self.update()

    # ---- coordinates
    def to_scan(self, pos):
        """A point on screen, in scan pixels.

        The canvas shows the picture the right way round, and the file is
        mirrored -- the negative was photographed through its base. So the
        display is a mirror of the scan and every coordinate crossing between
        them turns round here, in one place, rather than at each use.
        """
        pos = self.unturn(pos)
        x = (pos.x() - self.offset.x()) / self.zoom * self.scale
        y = (pos.y() - self.offset.y()) / self.zoom * self.scale
        return (self.scan_w - x if self.mirrored else x, y)

    def _screen(self, x, y):
        if self.mirrored:
            x = self.scan_w - x
        return QPointF(self.offset.x() + x / self.scale * self.zoom,
                       self.offset.y() + y / self.scale * self.zoom)

    def _corners(self, part):
        """The corners of the fitted rectangle, in scan pixels. The angle is
        the item's -- one piece of film, one tilt -- so it is not looked for
        on the part."""
        return findframe.corners(part["centre"][0], part["centre"][1],
                                 part["size"][0], part["size"][1], self.angle)

    def _handles(self, part):
        """Where the four handles are: the vertices a person placed if they
        placed any, and otherwise the corners of the rectangle.

        They are not the same thing once a vertex has been placed. Four points
        are eight numbers and a rectangle is five, so the rectangle cannot pass
        through all four, and pretending the handles sit on it would quietly
        move a vertex somebody put somewhere on purpose.
        """
        return part.get("placed") or self._corners(part)

    # ---- the pointer
    def mousePressEvent(self, e):
        if self.picking_base:
            self._base_from = self.to_scan(e.position())
            self.base_box = None
            return
        if self.mode and e.modifiers() & (Qt.ShiftModifier | Qt.AltModifier):
            # The pointer's version of the widened arrows: ⇧ slides this frame,
            # ⌥ slides every frame, so the film moves as one piece.
            self._shift_from = self.to_scan(e.position())
            self._shift_scope = ("all" if e.modifiers() & Qt.AltModifier
                                 else "frame")
            return
        if self.mode:
            # The nearest handle of **any** frame, not of the selected one.
            # On a half-frame scan both frames are on screen and both are
            # being placed; having to pick one first is a step that exists
            # only because the code needed it, and it is one the pointer has
            # already answered by being where it is.
            near = None
            for i, part in enumerate(self.parts):
                for k, (cx, cy) in enumerate(self._handles(part)):
                    d = (self._screen(cx, cy) - e.position()).manhattanLength()
                    if d < self.HANDLE * 2 and (near is None or d < near[0]):
                        near = (d, i, k)
            if near is not None:
                _, i, k = near
                part = self.parts[i]
                self._had_placed = "placed" in part
                part.setdefault("placed",
                                [list(q) for q in self._corners(part)])
                self._moved = False
                self.sel = i          # the read line follows the pointer
                self.vertex = k       # and so does which corner the keys move
                self._grab = (i, k)
                # The status names the corner ⇥ walks to; grabbing one with the
                # pointer moves the same marker, so it says so too. Otherwise
                # it keeps announcing wherever ⇥ left it, which is somewhere
                # else by then.
                self.picked(i)
                self.picked(i)
                self.update()
                return
        if (self.view == "dust" and e.button() == Qt.LeftButton
                and e.modifiers() & Qt.ControlModifier):
            # ⌘ + drag on a mark moves what it is cloned from; ⌘ + click,
            # back to the automatic fill
            self._cloning = True
            return self.dust_clone_press(e.position())
        if (self.view == "dust" and e.button() == Qt.LeftButton
                and not e.modifiers() & Qt.AltModifier):
            # **In the dust a drag paints** (a scratch, a hair) and ⌥ + drag
            # moves about; a press that does not move is still a click
            self._stroke = [e.position()]
            return
        self._pan = self.unturn(e.position()).toPoint() - self.offset
        self._pan_at = e.position() - self.cut_pan      # for the fitted views
        self._pan_from = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self.picking_base and self._base_from is not None:
            x, y = self.to_scan(e.position())
            self.base_box = (*self._base_from, x, y)
            return self.update()
        if self.view == "dust" and self._stroke is not None:
            p = e.position()
            if (p - self._stroke[-1]).manhattanLength() >= 2:
                self._stroke.append(p)
                r = self._dust_cursor_r() + 4
                a, b = self._stroke[-2], p
                self.update(int(min(a.x(), b.x()) - r), int(min(a.y(), b.y()) - r),
                            int(abs(a.x() - b.x()) + 2 * r) + 1,
                            int(abs(a.y() - b.y()) + 2 * r) + 1)
        if self.view == "dust" and self._cloning:
            self.dust_clone_drag(e.position())
        if self.view == "dust":
            # only the ring under the pointer, where it was and where it is:
            # repainting a 20 MP frame per pixel of movement was felt
            old, self.dust_hover = self.dust_hover, e.position()
            tip = self.dust_named(e.position())
            if tip != self.dust_tip:
                self.dust_tip = tip
                self.update()
            r = self._dust_cursor_r() + 4
            for p in (old, self.dust_hover):
                if p is not None:
                    self.update(int(p.x() - r), int(p.y() - r),
                                int(2 * r) + 1, int(2 * r) + 1)
        if self._shift_from is not None:
            x, y = self.to_scan(e.position())
            self.nudge(x - self._shift_from[0], y - self._shift_from[1],
                       scope=self._shift_scope)
            self._shift_from = (x, y)
            self._moved = True
            return
        if self._grab is not None:
            self._place(e)
        elif self._pan is not None:
            if self.cut_view():
                self.cut_pan = e.position() - QPointF(self._pan_at)
                return self.update()
            self.offset = self.unturn(e.position()).toPoint() - self._pan
            self.update()

    def mouseReleaseEvent(self, e):
        if self._cloning:
            self._cloning = False
            return self.dust_clone_release(e.position())
        if self._stroke is not None:
            pts, self._stroke = self._stroke, None
            span = max((p - pts[0]).manhattanLength() for p in pts)
            if span < 4:
                self.dust_clicked(e.position())
            else:
                self.dust_stroked(pts)
            return self.update()
        if self.picking_base and self._base_from is not None:
            self._base_from = None
            box = self.base_box
            if box and abs(box[2] - box[0]) > 8 and abs(box[3] - box[1]) > 8:
                x0, x1 = sorted((box[0], box[2]))
                y0, y1 = sorted((box[1], box[3]))
                self.base_picked((x0, y0, x1, y1))
            return
        if self._shift_from is not None:
            scope = self._shift_scope
            self._shift_from = None
            if self._moved:
                self.dragged(scope)
            return
        if self._grab is not None:
            i, _ = self._grab
            self._grab = None
            if self._moved:
                self.edited("placed")
            elif not self._had_placed:
                # A click that moved nothing is somebody choosing a frame, not
                # placing a corner. Leaving the vertices behind would mark the
                # frame as hand-placed for a click that changed no pixel of it.
                self.parts[i].pop("placed", None)
            self.update()
        elif (self._pan_from is not None
              and (e.position().toPoint() - self._pan_from).manhattanLength() < 4):
            # A click that did not drag is not a pan: it is somebody pointing
            # at a frame. Outside the mode that is the only way to choose one,
            # and it is why there is no key for it.
            if self.view == "dust":
                self.dust_clicked(e.position())
            else:
                self._choose(e.position())
        self._pan = self._pan_from = None

    def leaveEvent(self, e):
        if self.dust_hover is not None:
            self.dust_hover = self.dust_tip = None
            self.update()
        super().leaveEvent(e)

    def _dust_cursor_r(self, stroke=False):
        """The ring that follows the pointer, in screen px: the brush a
        click dabs and a drag paints with (`dust.STROKE_R` when no size was
        set). `stroke` is kept for the callers; the two are one brush."""
        from . import dust
        # never under 3 screen px: a 4 px brush over the whole frame is a
        # fifth of a pixel, and a ring that cannot be seen is no pointer
        return max(3.0, (self.dust_brush or dust.STROKE_R) * self.dust_scale()[0])

    def _choose(self, pos):
        pos = self.unturn(pos)
        for i, part in enumerate(self.parts):
            poly = QPolygonF([self._screen(x, y) for x, y in self._corners(part)])
            if poly.containsPoint(pos, Qt.OddEvenFill) and i != self.sel:
                self.sel = i
                self.picked(i)
                self.update()
                return

    def mouseDoubleClickEvent(self, e):
        # Not where a click places something: Qt sends the second click of
        # two quick ones here instead of to the press, and two specks dabbed
        # in a hurry fitted the frame instead of marking the second.
        if self.mode or self.view == "dust" or self.picking_base:
            return self.mousePressEvent(e)
        self.fit()

    def zoom_to(self, points, margin=1.25):
        """Put these scan points on screen, with room around them."""
        if not points:
            return
        pts = [self._to_display(x, y) for x, y in points]
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        w = max(max(xs) - min(xs), 1e-6) * margin
        h = max(max(ys) - min(ys), 1e-6) * margin
        self.zoom = min(self.width() / w, self.height() / h)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        self.offset = QPoint(int(self.width() / 2 - cx * self.zoom),
                             int(self.height() / 2 - cy * self.zoom))
        self.update()

    def _to_display(self, x, y):
        """Scan pixels to preview pixels, mirrored the way the canvas is."""
        return (((self.scan_w - x) if self.mirrored else x) / self.scale,
                y / self.scale)

    def zoom_corner(self):
        """In on the chosen corner, at a window sized to the frame it is on:
        a twentieth of its short side, so it is the same closeness on a full
        frame and on a half one."""
        p = self.selected()
        if not p:
            return
        x, y = self._handles(p)[self.vertex]
        r = min(p["size"]) / 20
        self.zoom_to([(x - r, y - r), (x + r, y + r)], margin=1.0)

    def zoom_frame(self):
        p = self.selected()
        if p:
            self.zoom_to(self._corners(p))

    def step(self):
        """One arrow press, in scan pixels: two pixels of the view.

        Not a fixed number of scan pixels. Placing a corner is done zoomed in
        and moving the whole film is done zoomed out, and a step that follows
        the zoom is the same gesture at both -- it moves what you can see it
        move. At fit-to-window on these scans it comes out at about 12 px,
        which is what the fixed step used to be.
        """
        return max(0.25, 2.0 / max(self.zoom, 1e-6) * self.scale)

    def tab_order(self):
        """Every corner of every frame, clockwise on screen, frames in order.

        Clockwise **as seen**: the fit numbers its corners clockwise in the
        scan's own coordinates, and the canvas mirrors the scan, so on screen
        that order runs backwards. It is reversed here rather than anywhere
        else, because this is the only place the order is about what a person
        is looking at.

        It runs across the frames as well as around each one, so tabbing past
        the last corner of the first frame lands on the first corner of the
        next. That is how you change frame in here: there is no separate key
        for it, because this one already goes there.
        """
        ring = (0, 3, 2, 1) if self.mirrored else (0, 1, 2, 3)
        return [(i, k) for i in range(len(self.parts)) for k in ring]

    def walk(self, step):
        """Move the chosen corner along the tab order, and say where it is.

        If the corner it lands on is not comfortably on screen, the view goes
        to it. Zoomed in on one corner, tabbing to the next would otherwise
        move a marker you cannot see -- and zoomed out it does nothing, since
        every corner is already in the middle of the window somewhere.
        """
        order = self.tab_order()
        if not order:
            return ""
        here = (self.sel, self.vertex)
        i = order.index(here) if here in order else 0
        self.sel, self.vertex = order[(i + step) % len(order)]
        self.picked(self.sel)
        p = self.selected()
        if p:
            s = self._screen(*self._handles(p)[self.vertex])
            m = self.rect().adjusted(int(self.width() * 0.2),
                                     int(self.height() * 0.2),
                                     -int(self.width() * 0.2),
                                     -int(self.height() * 0.2))
            if not m.contains(s.toPoint()):
                self.centre_on(*self._handles(p)[self.vertex])
        self.update()
        return self.corner_name()

    def centre_on(self, x, y):
        """Put this scan point in the middle of the window, at the same zoom."""
        dx, dy = self._to_display(x, y)
        self.offset = QPoint(int(self.width() / 2 - dx * self.zoom),
                             int(self.height() / 2 - dy * self.zoom))

    def corner_name(self):
        """Which corner the keys are on, said as where it sits on screen."""
        p = self.selected()
        if not p:
            return ""
        here = self._screen(*self._handles(p)[self.vertex])
        mid = self._screen(*p["centre"])
        where = ("top " if here.y() < mid.y() else "bottom ") + \
                ("left" if here.x() < mid.x() else "right")
        return f"{where} of {p['name']}" if len(self.parts) > 1 else where

    def nudge(self, dx, dy, scope="corner"):
        """Move one corner, one whole frame, or every frame.

        Three widths of the same command, and the modifiers widen it in order:
        nothing, `⇧`, `⌥`. Moving a whole frame translates its fit exactly --
        the size and the angle come out unchanged -- which is the difference
        between "this exposure sits a little further along the strip", which
        is common, and "the film slipped in the holder", which is not.
        """
        if scope == "corner":
            p = self.selected()
            if not p:
                return None
            p.setdefault("placed", [list(q) for q in self._corners(p)])
            x, y = p["placed"][self.vertex]
            p["placed"][self.vertex] = [x + dx, y + dy]
            self.refit()
            return "placed"

        moving = self.parts if scope == "all" else [self.selected()]
        for p in [q for q in moving if q]:
            p["centre"] = [round(p["centre"][0] + dx, 1),
                           round(p["centre"][1] + dy, 1)]
            if p.get("placed"):
                p["placed"] = [[x + dx, y + dy] for x, y in p["placed"]]
        self.update()
        return "all" if scope == "all" else "frame"

    def refit(self):
        """Re-fit every frame whose vertices have been placed, sharing one
        angle between them."""
        voting = [q for q in self.parts if q.get("placed")]
        if not voting:
            return
        angle, rects = findframe.fit_rectangles(
            [q["placed"] for q in voting], near=self.angle)
        for q, (cx, cy, w, h) in zip(voting, rects):
            q["centre"] = [round(cx, 1), round(cy, 1)]
            q["size"] = [round(w, 1), round(h, 1)]
        self.angle = round(angle, 3)
        self.turned(self.angle)
        self.update()

    def _place(self, e):
        """Move one vertex to the pointer and refit the rectangle to all four.

        The vertex goes exactly where it is put; the rectangle is the least
        squares fit through the four, which is why the other three handles do
        not move and the drawn rectangle does. Four placed points average into
        a better rectangle than any one of them: at 10 px of error per vertex
        the fitted centre is out by 4 px and the angle by 0.14 degrees.

        The angle it finds belongs to the **item**, because one piece of film
        has one tilt. Every frame whose vertices have been placed votes on it,
        so correcting the second frame of a pair refines the tilt rather than
        replacing what the first frame's four corners already said.
        """
        if self._grab is None:
            return
        i, k = self._grab
        p = self.parts[i]
        where = list(self.to_scan(e.position()))
        if where != p["placed"][k]:
            self._moved = True
        p["placed"][k] = where
        self.vertex = k
        self.refit()

    # ---- painting
    def _blink_step(self):
        self._blink ^= 1
        self.update()

    def paintEvent(self, e):
        q = QPainter(self)
        q.fillRect(self.rect(), QColor(BASE))
        pm = self.current()
        over = self.clip_for(pm) if (pm and self.view is None) else None
        if over and not self._blink_timer.isActive():
            self._blink_timer.start()
        elif not over and self._blink_timer.isActive():
            self._blink_timer.stop()
        if not pm:
            if self.view == "dust" and self.label:
                # the full-size develop takes seconds and there is nothing
                # to draw until it lands: an empty canvas read as a crash
                self._paint_waiting(q, self.label)
            return
        q.setRenderHint(QPainter.SmoothPixmapTransform)
        if self.view == "developed":
            self._paint_developed(q, pm)
            return self._paint_label(q)
        if self.view == "dust":
            self._paint_dust(q, pm)
            return self._paint_label(q)
        if self.view and self.view.startswith("result"):
            self._paint_result(q, pm)
            return self._paint_label(q)
        if self.view_turn() != self._fitted_turn:
            # turned or turned back: a portrait frame does not fit where a
            # landscape one did
            self.fit()
        xf = self._view_xf()
        if xf is not None:
            q.setTransform(xf)
        w, h = pm.width() * self.zoom, pm.height() * self.zoom
        for layer in (pm, over[self._blink] if over else None):
            if layer is None:
                continue
            # The clip marks are drawn exactly as the scan is, so they sit on
            # the photosites they were read off whatever the zoom; without
            # smoothing, so a single clipped block stays a hard square.
            q.setRenderHint(QPainter.SmoothPixmapTransform, layer is pm)
            if self.mirrored:
                # The picture the right way round. Mirroring is a property of
                # the roll and an axis of its own: it does not depend on which
                # of the three ways of looking is on.
                q.save()
                q.translate(self.offset.x() * 2 + w, 0)
                q.scale(-1, 1)
                q.drawPixmap(QRectF(self.offset.x(), self.offset.y(), w, h),
                             layer, QRectF(layer.rect()))
                q.restore()
            else:
                q.drawPixmap(QRectF(self.offset.x(), self.offset.y(), w, h),
                             layer, QRectF(layer.rect()))
        q.setRenderHint(QPainter.SmoothPixmapTransform)
        if self.view == "base":
            return
        for i, part in enumerate(self.parts):
            self._paint_part(q, i, part)
        self._paint_base_box(q)
        self._paint_label(q)

    def _paint_base_box(self, q):
        """The scan's own base (`⇧B`), or the box being dragged for it."""
        if not self.base_box:
            return
        x0, y0, x1, y1 = self.base_box
        poly = QPolygonF([self._screen(x, y) for x, y in
                          ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])
        pen = QPen(QColor(SUN), 2)
        pen.setStyle(Qt.DashLine)
        q.setPen(pen)
        q.setBrush(Qt.NoBrush)
        q.drawPolygon(poly)
        r = poly.boundingRect()
        q.drawText(QPointF(r.left() + 4, r.top() - 5), "base")

    def _paint_label(self, q):
        """**Which way of looking this is, on the picture itself.**

        It is on the read line too, and the house rule is that nothing is
        said twice -- but the read line is a strip of small text along the
        top of a 1400 px window and the picture is the thing being looked
        at. Three ways of drawing one scan, two of them photographs, is
        exactly the case where a person reads the picture and never the
        strip: this window was taken for "no automatics" by the person who
        asked for the mode. So it is said on the canvas as well, once,
        quietly, in the corner a frame is never placed by.
        """
        if not self.label:
            return
        q.resetTransform()
        q.setClipping(False)
        f = QFont()
        f.setPointSizeF(max(9.0, f.pointSizeF() - 1.0))
        f.setCapitalization(QFont.AllUppercase)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 1.4)
        q.setFont(f)
        m = QFontMetrics(f)
        box = QRectF(12, self.height() - m.height() - 20,
                     m.horizontalAdvance(self.label) + 18, m.height() + 8)
        q.setPen(Qt.NoPen)
        q.setBrush(QColor(20, 20, 22, 200))
        q.drawRoundedRect(box, 4, 4)
        q.setBrush(Qt.NoBrush)
        q.setPen(QPen(QColor(PRIMARY)))
        q.drawText(box, Qt.AlignCenter, self.label)

    def _paint_waiting(self, q, text):
        """A line in the middle of an empty canvas, for work under way."""
        f = QFont()
        f.setPointSizeF(f.pointSizeF() + 3)
        q.setFont(f)
        q.setPen(QPen(QColor(PRIMARY)))
        q.drawText(self.rect(), Qt.AlignCenter, text)

    def _paint_part(self, q, i, part):
        # Inside the mode every frame is live, so none of them is dimmed: what
        # the selection now marks is only which one the read line is talking
        # about, and dimming the others said they could not be touched.
        chosen = self.mode and i == self.sel
        # **Which one the read line is talking about, said on the rectangle
        # itself** -- by its weight, and nothing else. The colour still
        # means where the number came from, detected or a person's, and a
        # second hue for *selected* would be a colour that means two
        # things; turning the other one down was tried first and did not
        # read on screen, these being already at the top of their value.
        # With one frame there is nothing to tell apart and it stays thin.
        here = i == self.sel and len(self.parts) > 1
        colour = MANUAL if part["name"] in self.manual else DETECTED
        placed = part.get("placed") if self.mode else None
        # While the corners are being placed, what is drawn is **what you
        # placed** -- the four points, joined. The fitted rectangle cannot pass
        # through them and seeing it argue with them mid-edit reads as the
        # program moving your corner. It comes back when the mode is left,
        # which is when it is the answer rather than a running commentary.
        outline = placed or self._corners(part)
        poly = QPolygonF([self._screen(x, y) for x, y in outline])
        q.setPen(QPen(QColor(MANUAL if placed else colour),
                      3.0 if (chosen or here) else 1.6))
        q.drawPolygon(poly)
        # The crop, drawn on the frame it will be taken off: it is a develop
        # setting and nothing measures again for it, so the only honest place
        # to show it is here, at once, on the rectangle in front of you.
        if self.crop_pct:
            k = 1.0 - 2.0 * self.crop_pct / 100.0
            cx, cy = part["centre"]
            w, h = part["size"][0] * k, part["size"][1] * k
            inner = QPolygonF([self._screen(x, y) for x, y in
                               findframe.corners(cx, cy, w, h, self.angle)])
            pen = QPen(QColor(SUN), 2)
            pen.setStyle(Qt.DashLine)
            q.setPen(pen)
            q.drawPolygon(inner)
        q.setPen(QPen(QColor(SUN if chosen else colour), 2))
        turn = part.get("turn", 0)
        # the name reads upright on a turned drawing: placed at the corner
        # that is top-left on screen, written without the view's turn
        xf = self._view_xf()
        at = poly.at(0)
        if xf is not None:
            shown = [xf.map(poly.at(j)) for j in range(poly.size())]
            at = min(shown, key=lambda pt: pt.x() + pt.y())
            q.save()
            q.resetTransform()
        q.drawText(at + QPointF(6, 18),
                   part["name"] + (f" {turn}°" if turn else ""))
        if xf is not None:
            q.restore()
        if self.mode:
            handles = [self._screen(x, y) for x, y in self._handles(part)]
            for k, s in enumerate(handles):
                # The one the arrows and ⇥ are on is bigger and amber, because
                # a key that moves something invisible is a key you press
                # twice to find out what it did.
                on = chosen and k == self.vertex
                size = self.HANDLE + 4 if on else self.HANDLE
                q.setPen(QPen(QColor(SUN if on else
                                     (MANUAL if placed else colour)), 1))
                q.setBrush(QColor(SUN if on else
                                  (MANUAL if placed else colour)))
                q.drawRect(QRectF(s.x() - size / 2, s.y() - size / 2, size, size))
            q.setBrush(Qt.NoBrush)

    def _cut_corners(self, part):
        """The corners of what would actually come out: the fitted rectangle
        less the roll's crop. `_corners` is where the frame *is*; this is
        where the picture ends up."""
        k = 1.0 - 2.0 * self.crop_pct / 100.0
        return findframe.corners(part["centre"][0], part["centre"][1],
                                 part["size"][0] * k, part["size"][1] * k,
                                 self.angle)

    def _paint_developed(self, q, pm):
        """The developed picture, fitted to the window.

        It arrives already cut, straightened, mirrored and turned -- that is
        what the develop does to it -- so there is nothing to transform here
        and nothing to draw on top: an outline over a picture that no longer
        holds the frame's edge would be pointing at the wrong pixels.

        Fitted rather than zoomed, like the cut view, so the zoom and the pan
        of the scan underneath come back exactly as they were left.
        """
        k = min(self.width() / max(pm.width(), 1),
                self.height() / max(pm.height(), 1)) * 0.98 * self.cut_zoom
        w, h = pm.width() * k, pm.height() * k
        q.drawPixmap(QRectF((self.width() - w) / 2 + self.cut_pan.x(),
                            (self.height() - h) / 2 + self.cut_pan.y(),
                            w, h), pm, QRectF(pm.rect()))

    # ---- the full-size frame, for marking dust
    def dust_scale(self, pm=None):
        """Screen pixels per picture pixel, and the picture's top left."""
        pm = pm or self.current()
        if not pm:
            return 1.0, QPointF(0, 0)
        k = min(self.width() / max(pm.width(), 1),
                self.height() / max(pm.height(), 1)) * 0.98 * self.cut_zoom
        return k, QPointF((self.width() - pm.width() * k) / 2 + self.cut_pan.x(),
                          (self.height() - pm.height() * k) / 2 + self.cut_pan.y())

    def dust_at(self, pos):
        """A point on screen, on the frame shown: picture pixels."""
        k, o = self.dust_scale()
        return (pos.x() - o.x()) / k, (pos.y() - o.y()) / k

    def dust_screen(self, x, y):
        k, o = self.dust_scale()
        return QPointF(o.x() + x * k, o.y() + y * k)

    def dust_show(self, x, y, k):
        """Zoom to `k` screen pixels per picture pixel with the picture
        point (x, y) in the middle of the window."""
        pm = self.current()
        if not pm:
            return
        fit = min(self.width() / max(pm.width(), 1),
                  self.height() / max(pm.height(), 1)) * 0.98
        self.cut_zoom = k / fit
        self.cut_pan = QPointF(pm.width() * k / 2 - x * k,
                               pm.height() * k / 2 - y * k)
        self.update()

    def _paint_dust(self, q, pm):
        """The frame at full size, fitted or at 100 %, with the marks on it.

        Not smoothed once a picture pixel is bigger than a screen pixel: a
        speck is a few pixels across, and a blur over it is the one thing
        the person marking needs not to see."""
        k, o = self.dust_scale(pm)
        q.setRenderHint(QPainter.SmoothPixmapTransform, k < 1)
        q.drawPixmap(QRectF(o.x(), o.y(), pm.width() * k, pm.height() * k),
                     pm, QRectF(pm.rect()))
        if self.dust_overlay is not None:
            ov, ox, oy = self.dust_overlay
            q.drawPixmap(QRectF(o.x() + ox * k, o.y() + oy * k,
                                ov.width() * k, ov.height() * k), ov, QRectF(ov.rect()))
        q.setRenderHint(QPainter.SmoothPixmapTransform)
        q.setBrush(Qt.NoBrush)
        # round the speck, not on it: the mark must not hide what it marks.
        # Pink is a person's, blue the finder's -- the window's two colours
        # for who said so -- dashed while undecided, and red when refused.
        # A mark given a size is drawn at that size, just outside it: the
        # ring is then the edge of what will be filled. One without is
        # drawn round the speck, at a size that says nothing.
        # A mark cloned from a place a person chose: a thin line to a dashed
        # ring there, in the colour of a reference -- what it borrows from.
        lines = [(x, y, size, src) for x, y, style, focus, size, path, src
                 in self.dust_marks if src is not None]
        if self.dust_source is not None:
            (mx, my), (sx, sy) = self.dust_source
            lines.append((mx, my, self.dust_brush or 4.0, (sx, sy)))
        for x, y, size, (sx, sy) in lines:
            a = QPointF(o.x() + x * k, o.y() + y * k)
            b = QPointF(o.x() + sx * k, o.y() + sy * k)
            rr = max(3.0, (size or 4.0) * k)
            q.setPen(QPen(QColor(REF), 1.5))
            q.drawLine(a, b)
            q.setPen(QPen(QColor(REF), 1.5, Qt.DashLine))
            q.drawEllipse(b, rr, rr)
        for x, y, style, focus, size, path, _src in self.dust_marks:
            r = size * k + 1.5 if size else max(11.0, 8 * k)
            colour = {"hand": MANUAL, "refused": GONE}.get(style, DETECTED)
            if path:
                # a stroke: the band that will be filled, see-through, so
                # the scratch under it still shows
                pts = [QPointF(o.x() + px * k, o.y() + py * k) for px, py in path]
                band = QColor(colour)
                band.setAlpha(150 if focus else 90)
                q.setPen(QPen(band, max(6.0, 2 * size * k), Qt.SolidLine,
                              Qt.RoundCap, Qt.RoundJoin))
                if len(pts) == 1:
                    q.drawPoint(pts[0])     # a dab: the round cap is its disc
                else:
                    q.drawPolyline(pts)
                continue
            pen = QPen(QColor(colour), 3 if focus else 2)
            if style in ("proposed", "refused"):
                pen.setStyle(Qt.DashLine)
            q.setPen(pen)
            c = QPointF(o.x() + x * k, o.y() + y * k)
            q.drawEllipse(c, r, r)
            if focus:
                q.drawEllipse(c, r + 5, r + 5)
        if self._stroke is not None and len(self._stroke) > 1:
            band = QColor(MANUAL)
            band.setAlpha(110)
            q.setPen(QPen(band, 2 * self._dust_cursor_r(stroke=True), Qt.SolidLine,
                          Qt.RoundCap, Qt.RoundJoin))
            q.drawPolyline(self._stroke)
        if self.dust_hover is not None:
            # dotted and shaded, the brush and not a mark; dark under light,
            # so it reads on a white sky and on a black wall alike
            r = self._dust_cursor_r()
            if not self.dust_tip:
                # over a mark the ring would cover the mark's own; the
                # number says which, and ⇧ + the wheel resizes that one
                q.setPen(QPen(QColor(0, 0, 0, 150), 2))
                q.setBrush(QColor(255, 255, 255, 36))
                q.drawEllipse(self.dust_hover, r, r)
                q.setPen(QPen(QColor(255, 255, 255, 230), 1, Qt.DotLine))
                q.setBrush(Qt.NoBrush)
                q.drawEllipse(self.dust_hover, r, r)
            if self.dust_tip:
                # the mark's number by the pointer, to be able to say which
                f = QFont()
                q.setFont(f)
                m = QFontMetrics(f)
                box = QRectF(self.dust_hover.x() + r + 6, self.dust_hover.y() - m.height(),
                             m.horizontalAdvance(self.dust_tip) + 12, m.height() + 6)
                q.setPen(Qt.NoPen)
                q.setBrush(QColor(20, 20, 22, 220))
                q.drawRoundedRect(box, 4, 4)
                q.setPen(QPen(QColor(PRIMARY)))
                q.drawText(box, Qt.AlignCenter, self.dust_tip)
                q.setBrush(Qt.NoBrush)

    def _paint_result(self, q, pm):
        """What would come out: straightened, cropped, and the right way round.

        The frame asked for by name, held on its own letter. Drawn from the same numbers the outline is drawn from, so it cannot
        show one thing while the outline claims another. The clip is the
        **frame**, not the picture: clipping to the whole scan straightened it
        and centred it and cropped nothing, which is what it used to do.
        """
        want = self.view.split(":", 1)[1]
        parts = [q for q in self.parts if q["name"] == want]
        if not parts:
            return
        # How big the result is, measured in the film's own axes, where the
        # crop is a plain rectangle: that is what decides the scale.
        r = math.radians(self.angle)
        co, si = math.cos(r), math.sin(r)
        local = [(x * co + y * si, -x * si + y * co)
                 for p in parts for x, y in self._cut_corners(p)]
        w = (max(v[0] for v in local) - min(v[0] for v in local)) / self.scale
        h = (max(v[1] for v in local) - min(v[1] for v in local)) / self.scale
        # How the camera was held for *this* exposure, which is a fact about
        # the frame and not about the film: on half frame the two on one strip
        # can differ. Standing it up swaps what has to fit in the window.
        turn = parts[0].get("turn", 0)
        if turn % 180:
            w, h = h, w
        k = min(self.width() / max(w, 1e-6),
                self.height() / max(h, 1e-6)) * 0.98 * self.cut_zoom

        # Everything else is done in display coordinates -- the space the
        # pixmap is drawn in -- so the clip and the picture cannot disagree.
        poly = [self._to_display(x, y) for p in parts for x, y in self._cut_corners(p)]
        dcx = sum(v[0] for v in poly) / len(poly)
        dcy = sum(v[1] for v in poly) / len(poly)
        # Mirroring turns the sense of the tilt as well as the picture, so the
        # rotation that straightens the frame on screen is the other way round.
        shown = -self.angle if self.mirrored else self.angle

        q.translate(self.width() / 2 + self.cut_pan.x(),
                    self.height() / 2 + self.cut_pan.y())
        q.rotate(turn)
        q.scale(k, k)
        q.rotate(-shown)
        q.translate(-dcx, -dcy)
        # Clipped to the frames themselves, as a path, in these same
        # coordinates. A clip rectangle set before the last translate is in a
        # different space from the pixmap, and cropped nothing.
        path = QPainterPath()
        for p in parts:
            path.addPolygon(QPolygonF(
                [QPointF(*self._to_display(x, y)) for x, y in self._cut_corners(p)]))
        q.setClipPath(path)
        if self.mirrored:
            q.translate(pm.width(), 0)
            q.scale(-1, 1)
        q.drawPixmap(0, 0, pm)


# ------------------------------------------------------------------ the list
class RowDelegate(QStyledItemDelegate):
    """Paints a small rich-text document, so a row can carry a colour per
    label rather than one foreground for the whole line.

    A row carries several independent facts and a single
    foreground makes them queue up. The plain text stays in DisplayRole, so
    what the code compares to decide what changed is unchanged.
    """

    HTML = Qt.UserRole + 1

    def __init__(self):
        # Built once, and here rather than on the class: paint and sizeHint
        # run per row per repaint and QFontMetrics is not free, but a QFont
        # cannot be made before there is a QApplication.
        super().__init__()
        self.font = QFont(MONO, 11)
        self.row_h = QFontMetrics(self.font).height() + 8

    def paint(self, q, opt, index):
        q.save()
        if opt.state & QStyle.StateFlag.State_Selected:
            q.fillRect(opt.rect, QColor("#0a3a66"))
        q.setPen(QColor("#202024"))
        q.drawLine(opt.rect.bottomLeft(), opt.rect.bottomRight())
        doc = QTextDocument()
        doc.setDefaultFont(self.font)
        doc.setHtml(index.data(self.HTML) or index.data())
        q.translate(opt.rect.left() + 7, opt.rect.top() + 4)
        doc.drawContents(q)
        q.restore()

    def sizeHint(self, opt, index):
        return QSize(340, self.row_h)


# ------------------------------------------------------- developing to look
class _Render(QThread):
    """One frame developed off the window's thread.

    A scan's develop costs about a second, which is long enough to be felt:
    on the window's thread every press of `n` stopped the window for it. LibRaw and numpy both drop the interpreter lock for the length
    of their work, so the window goes on answering keys while this runs.

    It carries the key it was started for, because by the time it finishes
    the person may be two frames further down the roll, and a picture that
    arrives for a frame nobody is looking at is cached and not shown.
    """

    done = Signal(object, object, object, object)
    failed = Signal(object, str)

    def __init__(self, key, fn):
        super().__init__()
        self.key, self.fn = key, fn

    def run(self):
        try:
            sample, shown, evidence = self.fn()
        except Exception as e:               # shown, not lost in a thread
            return self.failed.emit(self.key, f"{type(e).__name__}: {e}")
        self.done.emit(self.key, sample, shown, evidence)


class _Prefetch(QThread):
    """The rest of the roll, developed on every core there is.

    One frame at a time on one thread took 40 s over a 37-frame roll, and a
    person walks faster than that: every picture said `developing…` and the
    machine was using one of its twelve cores to say it. A pool does the
    same work in a tenth of the time, and the frame actually on screen
    still goes through `_Render`, so jumping somewhere the pool has not
    reached yet is not a wait for its turn.

    It emits each frame as it lands rather than returning them all, so the
    roll fills in while it runs.
    """

    one = Signal(object, object, object)
    done = Signal()

    def __init__(self, jobs, workers):
        super().__init__()
        self.jobs, self.workers, self.stop = jobs, workers, False

    def run(self):
        """**Cancelled, never killed.**

        A `multiprocessing.Pool` has only `terminate()`, which closes the
        queue the workers are writing their results into: every worker
        mid-`put` then dies of a broken pipe and prints its own traceback
        from its own process, where no `except` in this one can reach it.
        Changing the base twice was enough to fill a terminal with them.

        An executor can cancel instead. `shutdown(cancel_futures=True)`
        drops the frames that have not started -- so nothing is killed
        holding a pipe open -- and waits only for the handful actually
        running. Measured over four interruptions of a 37-frame roll:
        0.0 to 3.7 s to stop, and not one line on stderr, against 24 broken
        pipes and a process that exited on SIGABRT before.
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from . import develop
        if not self.jobs:
            return self.done.emit()
        pool = None
        try:
            pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"))
            futures = [pool.submit(develop.whole_job, j) for j in self.jobs]
            cancelled = False
            for f in as_completed(futures):
                if self.stop and not cancelled:
                    # Cancel what has not started -- the rest of the roll --
                    # and go on draining the few that have. **Draining and
                    # not breaking**: abandoning the iterator leaves the
                    # results nobody is reading in the queue, and then
                    # `shutdown(wait=True)` waits for a queue that will
                    # never empty. The window hung on `⌘B` for exactly that.
                    for g in futures:
                        g.cancel()
                    cancelled = True
                if f.cancelled():
                    continue
                try:
                    got = f.result()
                except Exception:        # one frame that will not develop is
                    continue             # not a reason to stop the roll
                if self.stop:
                    continue             # drained, not shown
                self.one.emit(got[0], got[1], got[2])
        except Exception:                # a roll that cannot develop is not
            pass                         # a reason to take the window down
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        self.done.emit()


# --------------------------------------------------------------- the window
class Editor(QMainWindow):
    """One window, no tabs, no floating panels."""

    # Groups are columns, headed by what their entries reach, so an entry does
    # not repeat it. A command with nothing to act on is dimmed and keeps its
    # key: the key column is a picture of the keyboard, and the key does not
    # stop existing -- what changed is whether it has anything to work on.
    # (key, name, what it needs to be awake, what clicking it does). The last
    # is a method name, or None where the row is a gesture or a held key --
    # there is no way to press `hold r` or `← →` with a mouse, and a label
    # that lights up and then does nothing is a worse lie than a plain one.
    SHEET = (
        # Three ways of looking, on one key, because they are one question:
        # how far down the pipeline what you are looking at has come. The
        # sheet names them in that order and the read line says which is on.
        # **One row per action that can be clicked**: two keys on one row
        # leave the mouse no way to say which. Held keys and gestures, which
        # are not clicked, can share one.
        ("VIEW", (("n", "next", None, "_next"),
                  ("p", "previous", None, "_prev"),
                  ("v", "view: raw / positive / developed", None, "_cycle"),
                  ("⇧V", "view, backwards", None, "_cycle_back"),
                  # Its name is the mode it takes you to, and `_say_space`
                  # rewrites it: out here it offers the preview, in the
                  # preview the way back. This is the text for the state the
                  # window starts in.
                  ("space", "preview", "part", "_cut"),
                  ("f", "fit", None, "canvas.fit"),
                  # One row, because it is one thing asked of two keys and
                  # the sheet is read as a list of things you can do.
                  ("hold 1 / 2", "show frame a / b", "part", None),
                  # the develop's numbers for the frame in hand, under the
                  # canvas, with a link that copies them: something to look
                  # at, so with the other views
                  ("i", "development details", None, "_toggle_numbers"))),
        # **By area of the work, in its order**: the frames placed first,
        # then the colour checked, then the dust, then what is written.
        ("CROP", (("e", "edit crop frames", None, "_enter_mode"),
                  ("r", "rotate left", "part", "_turn_left"),
                  ("⇧R", "rotate right", "part", "_turn_right"),
                  # **Not a way of looking**: the roll's mirror is in every
                  # file written, and it orders the frames a and b
                  ("m", "mirror roll", None, "_mirror"),
                  ("⌘C", "copy crop", "part", "_copy_frames"),
                  ("⌘V", "paste crop", "copied", "_paste_frames"),
                  ("del", "skip scan", None, "_skip"))),
        ("COLOUR", (("hold b", "show base", None, None),
                    # this scan's own unexposed film, for a scan copied
                    # under another light, or for the whole roll
                    # (`develop.own_base`); `_say_base` rewrites it
                    ("⇧B", "pick base…", "part", "_pick_base"),
                    # the stand's light falloff the other way round while
                    # held: off where it is applied, on where it is not
                    ("hold l", "flip falloff", None, None),
                    ("⇧⌘L", "falloff calibration…", None, "_light_falloff"))),
        ("DUST", (("s", "edit dust marks", "part", "_enter_dust"),
                  # the develop with the dust left in, as inside `s`
                  ("hold c", "show original", None, None))),
        ("PROCESS", (# the roll's taste by eye, one look for every frame
                     ("⌘L", "edit roll look…", None, "_look"),
                     # the same sheet; with ⇧ it starts on this scan
                     ("⌘R", "process roll…", None, "_process"),
                     ("⇧⌘R", "process scan…", None, "_process_scan"),
                     # Work on the roll, as ⌘R is, so beside it; named
                     # apart from it because finding the frames and
                     # developing them are different jobs. The ellipsis
                     # says the key asks before it acts.
                     ("⌘D", "re-analyze scan…", None, "_redetect"),
                     ("⇧⌘D", "re-analyze roll…", None, "_redetect_roll"))),
    )

    MODE_SHEET = (
        ("MOVE", (("arrows", "corner", None, None),
                  ("⇧ arrows", "frame", None, None),
                  ("⌥ arrows", "all frames", "two", None),
                  ("⇧ drag", "move frame", None, None))),
        # A drag on a corner is not here: the corners are drawn, and taking
        # one is what a pointer does. The sheet is for what cannot be seen.
        ("FRAME", (("⇥", "next corner", None, "_tab_on"),
                   ("⇧⇥", "previous corner", None, "_tab_back"),
                   ("+", "add frame", None, "_add_part"),
                   ("-", "remove frame", "part", "_drop_part"),
                   ("⌫", "reset", "manual", "_reset"))),
        ("TURN", ((", .", "rotate", "part", None),
                  ("⇧ , .", "rotate fine", "part", None))),
        ("LOOK", (("z", "zoom to corner", "part", "canvas.zoom_corner"),
                  ("⇧Z", "zoom to frame", "part", "canvas.zoom_frame"),
                  ("f", "fit", None, "canvas.fit"))),
        ("", (("↩ e", "apply", None, "_apply_mode"),
              ("esc", "cancel", None, "_cancel_mode"))),
    )

    # Marking dust. **Every click is saved as it is made**, so there is no
    # cancel: an afternoon of marks is not something one key should be able
    # to throw away, and a wrong mark is taken back by clicking it.
    # A click and a drag paint, and are not here: the pointer is the brush's
    # ring. Clicking a mark is, because nothing on screen says it removes it.
    DUST_SHEET = (
        ("MARK", (("click", "remove mark", None, None),
                  ("⇧ wheel", "brush size", None, None),
                  ("⌘ drag", "set clone source", None, None),
                  ("⌘ click", "reset clone source", None, None))),
        ("LOOK", (("z", "100 %", None, "_dust_zoom"),
                  ("f", "fit", None, "_dust_fit"),
                  ("⌥ drag", "pan", None, None),
                  ("v", "show marks", None, "_dust_see"),
                  ("hold c", "show original", None, None))),
        ("", (("esc", "done", None, "_leave_dust"),)),
    )

    def __init__(self, roll, doc, identity=("", "")):
        super().__init__()
        self.roll, self.doc, self.identity = roll, doc, identity
        self.names = document.order_of(doc)
        self.cache, self.base_cache, self._dirty = {}, {}, False
        # One axis for colour, one for framing. `MODES` says why there are
        # three and no more.
        # **The develop, from the moment the roll opens.** It is what the
        # roll is for; the other two are for asking what the negative holds
        # and what the raw would.
        self.colour = "develop"
        self.mode_cache = {}               # (scan, mode) -> pixmap, for 1 and 2
        self.mode_arrays = {}              # the same as arrays, to composite on
        self.clip_masks = {}               # scan -> (mask, level, share) off the raw
        self._clip_over = None             # (scan, pixmap, (red, yellow)) on screen
        self.dev_plain = {}                # developed frames, unmirrored, unturned
        self.cut_before = None             # the framing the develop borrowed
        self.was_looking = None            # the view the placement mode borrowed
        self.cut = False                   # `space`: the frame, cut, and it stays
        # The develop keeps two caches of its own, because they cost
        # different things -- the cut frame is a raw read and a demosaic
        # (about a second), the picture is `finish` on it (about a fifth),
        # and only the second has to be redone when the taste changes.
        self.dev_plain, self.dev_ev = {}, {}   # the scan developed, whole
        self.dev_worker, self.dev_want = None, None
        # `hold l` and `hold c`: the light falloff the other way round, or
        # the dust left in, and the pictures drawn that way, kept apart from
        # the real ones (`_hold_other`). One at a time.
        self.held_other, self._sets = None, {}
        self.prefetch = None               # the roll developing itself
        self._threads = []                 # runs kept alive until they end
        self.mode = False
        self.before = None                 # the item as it was on entering
        self.dusting = False               # marking dust (`s`)
        self.dust = None                   # what that mode is marking, and how
        self.dust_worker = None            # the frame developing at full size
        self.dust_kept = {}                # its pictures, by what they depend on
        self.dust_kept_clean = {}          # and with the dust out, as left
        self.undo, self.redo = [], []
        self.setWindowTitle(f"film-developer — {roll.name}")
        self._build()
        # On the application, not on a widget: a filter on `self` never
        # sees a key that a focused child consumed, which is the whole
        # problem. See `eventFilter`.
        QApplication.instance().installEventFilter(self)
        if self.names:
            self.list.setCurrentRow(0)
        self.canvas.setFocus()

    # ---- the item on screen
    def name(self):
        r = self.list.currentRow()
        return self.names[r] if 0 <= r < len(self.names) else None

    def item(self):
        n = self.name()
        return self.doc["items"][n] if n else None

    def part(self):
        it = self.item()
        if it and 0 <= self.canvas.sel < len(it["parts"]):
            return it["parts"][self.canvas.sel]
        return None

    # ---- construction
    def _build(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._read_line())

        body = QWidget()
        row = QHBoxLayout(body)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(12)
        # The view is the view: no exposure and no local contrast knob any
        # more. They were about the screen and never about the data, and a
        # control that changes nothing that comes out earns its place only
        # while a frame edge is hard to see -- which the detector settled.
        self.ev, self.detail = 0.0, 0.0
        self.canvas = Canvas()
        self.canvas.edited = self._edited
        self.canvas.turned = self._turned
        self.canvas.picked = self._picked
        self.canvas.dust_clicked = self._dust_click
        self.canvas.dust_stroked = self._dust_stroke
        self.canvas.dust_clone_press = self._dust_clone_press
        self.canvas.dust_clone_drag = self._dust_clone_drag
        self.canvas.dust_clone_release = self._dust_clone_release
        self.canvas.dust_resized = self._dust_resize
        self.canvas.dust_named = self._dust_named
        self.canvas.clip_for = self._clip_for
        self.canvas.base_picked = self._base_picked
        self.canvas.dragged = lambda scope: (
            self._edited("moved", part=False) if scope == "all"
            else self._edited("frame"))
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)
        left.addWidget(self.canvas, 1)
        left.addWidget(self._controls())
        left.addWidget(self._numbers_panel())
        row.addLayout(left, 1)
        self.list = ScanList()
        self.list.setFont(QFont(MONO, 11))
        self.list.setItemDelegate(RowDelegate())
        # A fixed width, measured: the longest row in Menlo 11. A splitter is
        # one more thing on screen to have an opinion about, for a width that
        # is not a matter of opinion.
        self.list.setFixedWidth(360)
        # It never takes the keyboard: the arrows belong to the window, which
        # walks the list outside a mode and moves the frame inside one.
        self.list.setFocusPolicy(Qt.NoFocus)
        self.list.currentRowChanged.connect(self._select)
        # **The list is the order the pictures were taken in**, which a
        # person states: dragged (several at once with ⇧ or ⌘), ⌥↑ ⌥↓ for
        # the scan in hand, and by file name or turned round from the
        # menu on the right button. Kept as `roll.order` (`order_of`).
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        self.list.setDragDropMode(QListWidget.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.model().rowsMoved.connect(lambda *a: self._list_dropped())
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._list_menu)
        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(6)
        side.addWidget(self.list, 1)
        side.addWidget(self._order_row())
        row.addLayout(side)
        outer.addWidget(body, 1)

        outer.addWidget(self._sheets())
        self.setCentralWidget(central)
        self._fill_list()
        self._shortcuts()

    def _order_row(self):
        """Under the list, the three ways to put it in order at one click:
        by file name either way, and the order there is turned round.
        Buttons in sight: the menu on the right button was there and
        nobody could see it. Flat, label-like buttons:
        they act on the list, which is a document, but
        quietly, under what they act on. Moving one scan stays a drag or
        ⌥↑ ⌥↓, which a button cannot say where to."""
        w = QWidget()
        r = QHBoxLayout(w)
        r.setContentsMargins(2, 0, 2, 0)
        r.setSpacing(4)
        cap = QLabel("ORDER")
        cap.setStyleSheet(f"color: {SUB}; font-size: 9px; letter-spacing: 1.4px;")
        r.addWidget(cap)
        r.addSpacing(8)
        for text, tip, slot in (
                ("A → Z", "sort by file name, ascending", self._sort_names),
                ("Z → A", "sort by file name, descending", self._sort_names_back),
                ("⇅ reverse", "turn the present order round", self._reverse_order)):
            b = QPushButton(text)
            b.setFlat(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setFocusPolicy(Qt.NoFocus)
            b.setToolTip(_tip(tip + ". The list is the order the roll was "
                              "shot in, and the files are numbered by it; "
                              "drag a scan, or ⌥↑ ⌥↓, to move one. ⌘Z undoes"))
            b.setStyleSheet(
                "QPushButton { background: transparent; border: none; "
                "color: #8a8a90; padding: 2px 6px; }"
                "QPushButton:hover { color: #ff9f6b; }")
            b.clicked.connect(slot)
            r.addWidget(b)
        r.addStretch(1)
        hint = QLabel("drag to move")
        hint.setStyleSheet(f"color: {SUB};")
        r.addWidget(hint)
        return w

    def _controls(self):
        """The crop margin of the final image, under the canvas.

        The window puts what you *read* above the canvas and what you
        *operate* below it. The crop margin is the only control left here,
        and its name says what it changes: the final image.
        """
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 10, 0, 0)
        row.setSpacing(0)
        lab = QLabel("final image crop margin")
        lab.setStyleSheet(f"color: {COMMAND};")
        row.addWidget(lab)
        self.crop_box = QDoubleSpinBox()
        self.crop_box.setRange(-10.0, 10.0)
        self.crop_box.setSingleStep(0.5)
        self.crop_box.setDecimals(1)
        self.crop_box.setSuffix(" %")
        self.crop_box.setKeyboardTracking(False)
        self.crop_box.setValue(float(self.doc["roll"].get("develop", {}).get("crop", 0.0)))
        self.crop_box.setFixedWidth(92)
        self.crop_box.setToolTip(_tip(
            "cut from every side of every developed photo, as a percentage "
            "of that side, so the shape stays the same. A negative value "
            "adds a border. It applies to the whole roll, and not to the "
            "raw DNGs"))
        self.crop_box.valueChanged.connect(self._set_crop)
        row.addWidget(self.crop_box)
        row.addSpacing(14)
        self.crop_says = QLabel("")
        self.crop_says.setStyleSheet(f"color: {EXPLAIN};")
        row.addWidget(self.crop_says)
        self._say_crop()
        row.addStretch(1)
        return w

    def _numbers_panel(self):
        """`i`: the development details of the frame in hand, as numbers, under
        the controls -- the automatics apart from the taste, the window's
        develop beside the one `⌘R` wrote, and a link that copies it all.

        Comparing one frame across the window and `⌘R`, the read line said four numbers and nothing said
        which of them were the frame's own, which the roll's and which the
        taste's. Not a floating panel: a block in the column
        under the canvas, off until asked for. Monospace, because it is a
        list whose columns line up -- the one use the style gives it."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 10, 0, 0)
        lay.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(0)
        cap = QLabel("DEVELOPMENT DETAILS")
        cap.setStyleSheet(f"color: {SUB}; font-size: 9px; letter-spacing: 1.4px;")
        head.addWidget(cap)
        head.addSpacing(14)
        cp = QLabel(f'<a href="#copy" style="color:{EXPLAIN};'
                    f'text-decoration:none">⧉ copy</a>')
        cp.setTextFormat(Qt.RichText)
        cp.setToolTip(_tip("the numbers below, as text, to paste anywhere"))
        cp.linkActivated.connect(lambda _u: self._copy_numbers())
        head.addWidget(cp)
        head.addStretch(1)
        lay.addLayout(head)
        # Two blocks side by side, the exposure and the balance on the
        # left and the rest on the right: one column was 22 lines and took
        # the canvas down to 40 % of the window. The copy is one column.
        cols = QHBoxLayout()
        cols.setSpacing(30)
        self.numbers_cols = []
        for _ in range(2):
            lab = QLabel("")
            lab.setFont(QFont(MONO, 11))
            lab.setStyleSheet(f"color: {COMMAND};")
            # selectable with the mouse, but never the keyboard's: the
            # letters are the window's commands
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lab.setFocusPolicy(Qt.NoFocus)
            lab.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            cols.addWidget(lab, 0, Qt.AlignTop)
            self.numbers_cols.append(lab)
        cols.addStretch(1)
        lay.addLayout(cols)
        w.setVisible(False)
        self.numbers_box = w
        return w

    def _toggle_numbers(self):
        self.numbers_box.setVisible(not self.numbers_box.isVisible())
        self._say_numbers()

    def _copy_numbers(self):
        QApplication.clipboard().setText(self._numbers_text())
        self.status.setText("development details copied")

    def _file_evidence(self):
        """What `⌘R` wrote last, from `developed/developed.json`, re-read
        when the file changes."""
        import json
        f = self.roll / "developed" / "developed.json"
        try:
            t = f.stat().st_mtime
        except OSError:
            return None
        got = getattr(self, "_file_ev", None)
        if not got or got[0] != t:
            try:
                got = (t, json.loads(f.read_text()))
            except (OSError, ValueError):
                got = (t, None)
            self._file_ev = got
        return got[1]

    def _say_numbers(self):
        if not getattr(self, "numbers_box", None) or not self.numbers_box.isVisible():
            return
        got = self._numbers()
        if isinstance(got, str):
            self.numbers_cols[0].setText(got)
            self.numbers_cols[1].setText("")
            return
        head, rows, notes = got
        cut = next(i for i, r in enumerate(rows) if r[0].startswith("curve"))
        self.numbers_cols[0].setText(self._numbers_lines(head, rows[:cut], []))
        self.numbers_cols[1].setText(self._numbers_lines(
            [""] * len(head), rows[cut:], notes))

    def _numbers_text(self):
        """The develop of the frame in hand as aligned plain text, one
        column: what `copy` puts on the clipboard."""
        got = self._numbers()
        if isinstance(got, str):
            return got
        head, rows, notes = got
        return self._numbers_lines(head, rows, notes)

    @staticmethod
    def _numbers_lines(head, rows, notes):
        w0, w1 = 20, 24
        out = [head[0], f"{'':<{w0}}{head[1]:<{w1}}{head[2]}".rstrip()]
        out += [f"{a:<{w0}}{x:<{w1}}{y}".rstrip() for a, x, y in rows]
        return "\n".join(out + notes)

    def _numbers(self):
        """(heading, rows, notes) for the frame in hand: one row per
        number, the window's column and `⌘R`'s; or a sentence."""
        from . import develop, flatfield
        name, part, it = self.name(), self.part(), self.item()
        if not name or not part or not it:
            return "no frame here"
        taste = dict(develop.TASTE) | dict(self.doc["roll"].get("develop") or {})
        ev = self.dev_ev.get((name, part["name"])) or {}
        suffix = f"-{part['name']}" if len(it["parts"]) > 1 else ""
        stem = pathlib.Path(name).stem + suffix
        fj = self._file_evidence() or {}
        # numbered or not, and under whichever number the last `⌘R` gave it
        fe = next((v for k, v in (fj.get("frames") or {}).items()
                   if pathlib.Path(k).stem.split("_", 1)[-1] == stem
                   or pathlib.Path(k).stem == stem), {})
        if fe:
            other = [k for k in ("gamma", "exposure", "warmth", "tint", "shadows",
                                 "highlights", "saturation", "contrast", "pivot",
                                 "base", "flat")
                     if (fj.get("settings") or {}).get(k) != taste.get(k)]
        n = lambda v, f="+.2f": "—" if v is None else format(float(v), f)
        trio = lambda v: "—" if v is None else " ".join(
            format(float(x), "+.2f") for x in v)
        import math as _m
        lg = lambda g: None if not g else [_m.log2(max(float(x), 1e-9)) for x in g]
        b = lambda e, k: (e.get("balance_ev") or {}).get(k)
        rows = [
            ("exposure, EV", n(ev.get("exposure")), n(fe.get("exposure"))),
            ("  automatic", n(ev.get("exposure_auto")), n(fe.get("exposure_auto"))),
            ("  taste", n(taste["exposure"]), ""),
            ("  shadow rule", n(ev.get("shadow_lift")), n(fe.get("shadow_lift"))),
            ("balance, EV R G B", trio(lg(ev.get("balance"))), trio(lg(fe.get("balance")))),
            ("  roll", trio(b(ev, "roll")), trio(b(fe, "roll"))),
            ("  this frame", trio(b(ev, "frame")), trio(b(fe, "frame"))),
            ("  taste", trio(b(ev, "taste")), trio(b(fe, "taste"))),
            # beside the taste it started from: the curve opens on its own
            # when a frame holds less range than the screen (SPAN_FLOOR)
            ("curve, shadows", n(ev.get("shadow_slope"), ".3f")
             + f"  taste {taste['shadows']:.2f}", n(fe.get("shadow_slope"), ".3f")),
            ("curve, highlights", n(ev.get("highlight_slope"), ".3f")
             + f"  taste {taste['highlights']:.2f}",
             n(fe.get("highlight_slope"), ".3f")),
            ("black", n(ev.get("black"), ".4f"), n(fe.get("black"), ".4f")),
            ("saturation", n(taste["saturation"], ".2f"), ""),
            ("contrast, pivot", n(taste.get("contrast", 1.0), ".2f")
             + f"  {float(taste.get('pivot', 0.0)):+.2f} EV", ""),
            ("gamma", n(taste["gamma"], ".2f"), ""),
            ("base, R G B", " ".join(f"{float(x):.4f}" for x in ev["base"])
             if ev.get("base") else "—",
             " ".join(f"{float(x):.4f}" for x in fe["base"]) if fe.get("base") else "—"),
            ("  from", str(ev.get("base_source") or "—") + f" ({taste.get('base', 'auto')})",
             str(fe.get("base_source") or "—")),
            ("density range", n(ev.get("density_range"), ".3f"),
             n(fe.get("density_range"), ".3f")),
            ("light falloff", ("not corrected (hold l)" if self.flat_off else
                               (flatfield.of(taste) or {}).get("name") or "none"), ""),
        ]
        head = (stem + ("  blank frame" if ev.get("blank") else ""),
                "window, 1/4 size" if ev else "window: not developed yet",
                "⌘R file" if fe else "⌘R file: none")
        notes = []
        if fe and other:
            notes.append(f"⌘R ran under other settings: {', '.join(other)}")
        return head, rows, notes

    def _set_crop(self, pct):
        """A roll setting, so it is written down and not just remembered. It
        is kept with the develop settings because that is the only step it
        touches: the raw DNGs are written at the frame.

        **Nothing is developed again for it.** The window keeps each scan
        developed whole and uncropped, so the crop is where a mask is
        drawn, not what a picture was made with: moving the box redraws and
        that is all. It used to be an inset inside `develop.cut`, so every
        frame in hand had been cut to the old value and the roll had to be
        developed from scratch on every step of a spin box.
        """
        self.doc["roll"].setdefault("develop", {})["crop"] = round(float(pct), 2)
        self.dirty = True
        self.canvas.crop_pct = float(pct)
        self._say_crop()
        if self.colour == "develop":
            self._want_developed()
        else:
            self.canvas.update()

    def _say_crop(self):
        """One line under the setting: what it costs on the frame in front of
        you, in pixels, so a share has something to stand on."""
        pct = float(self.doc["roll"].get("develop", {}).get("crop", 0.0))
        p = self.part() if getattr(self, "list", None) else None
        if not p:
            self.crop_says.setText("")
            return
        if not pct:
            # Nothing said, because nothing is cut: a line about a zero
            # margin never changed and never meant anything.
            self.crop_says.setText("")
        else:
            dx, dy = p["size"][0] * pct / 100.0, p["size"][1] * pct / 100.0
            self.crop_says.setText(
                f"{abs(dx):.0f} px {'cut from' if pct > 0 else 'added to'} "
                f"left and right, {abs(dy):.0f} px top and bottom, "
                f"developed photos only")

    def _bar(self, name, top):
        f = QFrame()
        f.setObjectName(name)
        f.setStyleSheet(
            f"#{name} {{ background: {BAR_BG}; "
            f"border-{'top' if top else 'bottom'}: 1px solid {BAR_LINE}; }}")
        return f

    def _read_line(self):
        bar = self._bar("readline", top=False)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 5, 12, 5)
        self.reading = QLabel("")
        self.reading.setTextFormat(Qt.RichText)
        # It carries the stretch itself rather than sitting next to one: with
        # a stretch beside it the label is laid out at whatever it said first
        # and a longer reading is then clipped mid-number.
        lay.addWidget(self.reading, 1)
        self.held = QLabel("")
        self.held.setStyleSheet(f"color: {SUB};")
        lay.addWidget(self.held)
        return bar

    def _sheets(self):
        """A mode swaps the whole sheet for another one, in a stack fixed to
        the height of the first, so nothing below moves."""
        bar = self._bar("commands", top=True)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(30)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._sheet(self.SHEET))
        self.stack.addWidget(self._sheet(self.MODE_SHEET))
        self.stack.addWidget(self._sheet(self.DUST_SHEET))
        self.stack.setFixedHeight(self.stack.widget(0).sizeHint().height())
        lay.addWidget(self.stack, 0, Qt.AlignTop)
        lay.addStretch(1)

        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {EXPLAIN};")
        # A fixed width, right-aligned. A stretch instead looks right and is
        # not: Qt gives the stretch away and the message ends up three
        # letters wide.
        self.status.setFixedWidth(240)
        self.status.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        lay.addWidget(self.status, 0, Qt.AlignBottom)
        lay.addSpacing(18)
        lay.addWidget(self._signature(), 0, Qt.AlignBottom)
        return bar

    def _sheet(self, spec):
        page = QWidget()
        cols = QHBoxLayout(page)
        cols.setContentsMargins(0, 0, 0, 0)
        cols.setSpacing(30)
        self.names_by_need = getattr(self, "names_by_need", {})
        # Kept by key, for the rows whose name is not fixed: `space` names
        # the mode it takes you to, and which one that is depends on the
        # mode you are in. Only the main sheet, whose keys are unique.
        self.rows = getattr(self, "rows", {})
        main = spec is self.SHEET
        for heading, entries in spec:
            col = QWidget()
            grid = QGridLayout(col)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(9)
            grid.setVerticalSpacing(1)
            h = QLabel(heading)
            h.setStyleSheet(
                f"color: {SUB}; font-size: 9px; letter-spacing: 1.4px;")
            # **Over the names, not over the keys.** Spanning both columns
            # hung the heading out to the left of everything under it, past
            # the widest key in the column, and a column of text read as
            # two things indented differently. The keys are a right-aligned
            # picture of the keyboard and the names are the list; the
            # heading titles the list.
            grid.addWidget(h, 0, 1)
            for i, (key, label, need, does) in enumerate(entries, start=1):
                k = QLabel(self._key_html(key))
                k.setTextFormat(Qt.RichText)
                # Right-aligned, so the letter lands on one vertical whatever
                # hangs in front of it.
                k.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                grid.addWidget(k, i, 0)
                n = QLabel(label)
                n.clickable = self._bound(does)
                n.live = True
                self._dress(n)
                if n.clickable:
                    n.setCursor(Qt.PointingHandCursor)
                    n.mousePressEvent = (
                        lambda _e, w=n: w.clickable() if w.live else None)
                grid.addWidget(n, i, 1)
                if main:
                    self.rows[key] = n
                self.names_by_need.setdefault(need or "", []).append(n)
            cols.addWidget(col, 0, Qt.AlignTop)
        # the stack is as wide as the widest sheet; a shorter one keeps its
        # columns together on the left rather than spread over that width
        cols.addStretch(1)
        return page

    def _bound(self, does):
        """The method a sheet entry names, or None for a gesture."""
        if not does:
            return None
        obj = self
        for step in does.split("."):
            obj = getattr(obj, step)
        return obj

    @staticmethod
    def _dress(label):
        """A name is white under the pointer **when it can be clicked**, and a
        name with nothing to act on goes to sleep and stops answering."""
        if not label.live:
            label.setStyleSheet(f"color: {ASLEEP};")
        elif label.clickable:
            label.setStyleSheet(
                f"QLabel {{ color: {COMMAND}; }}"
                f"QLabel:hover {{ color: {HOVER}; }}")
        else:
            label.setStyleSheet(f"color: {COMMAND};")

    # Small wrappers so the sheet can name what a key does. A row that a
    # person can click has to reach the same code the key reaches, and some of
    # those take arguments a sheet entry has no way to carry.
    def _next(self):
        # The same step the key takes: with the cut on, the next picture.
        # A row on the sheet that means something other than its key is a
        # row that teaches the wrong thing.
        self._step_cut(+1) if self._walking() else self._step(+1)

    def _prev(self):
        self._step_cut(-1) if self._walking() else self._step(-1)

    def _tab_on(self):
        self.status.setText(self.canvas.walk(1))

    def _tab_back(self):
        self.status.setText(self.canvas.walk(-1))

    def _apply_mode(self):
        self._leave_mode(apply=True)

    def _cancel_mode(self):
        self._leave_mode(apply=False)

    def _key_html(self, key):
        """The letter is what the eye is hunting for, so it is amber; what
        qualifies it -- a modifier, `hold` -- takes the dim colour. A
        gesture or a plain key on its own is not a letter to hunt for, and
        is dim too."""
        if key in ("arrows", "drag", "click", "del", "esc"):
            return f'<span style="color:{MODIFIER}">{key}</span>'
        out = []
        for word in key.split(" "):
            if word == "hold":
                out.append(f'<span style="color:{MODIFIER}">hold</span>')
                continue
            out.append("".join(
                f'<span style="color:{MODIFIER if ch in "⌘⇧⌥" else KEY}">{ch}</span>'
                for ch in word))
        return " ".join(out)

    def _signature(self):
        from .about import signature
        return signature(self)

    def _about(self):
        from .about import about_box
        about_box(self)

    # Keys a text field is entitled to keep while it has the focus. The
    # crop is the only field in the window and it holds a number, so what
    # it needs is digits, a sign, a separator, the editing keys and the
    # arrows that step it. Everything else belongs to the window: the
    # letters are its commands and a field that swallows them is a field
    # that turns the keyboard off.
    TYPED = frozenset(
        list(range(Qt.Key_0, Qt.Key_9 + 1))
        + [Qt.Key_Period, Qt.Key_Comma, Qt.Key_Minus, Qt.Key_Plus,
           Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Left, Qt.Key_Right,
           Qt.Key_Up, Qt.Key_Down, Qt.Key_Home, Qt.Key_End,
           Qt.Key_Tab, Qt.Key_Backtab, Qt.Key_Return, Qt.Key_Enter,
           Qt.Key_Escape])
    # And the ones a text field owns with a modifier: select all, copy,
    # cut, paste, undo. `⌘C` and `⌘V` are the window's own commands too,
    # and in a field the field wins -- which is the one place where what
    # you are looking at says what the key means.
    TYPED_WITH_CTRL = frozenset([Qt.Key_A, Qt.Key_C, Qt.Key_V, Qt.Key_X,
                                 Qt.Key_Z])

    def eventFilter(self, obj, e):
        """Every key in this window reaches the window's own handler,
        whatever has the focus.

        `keyPressEvent` only sees what the focused widget does not take,
        and the crop box takes everything: with the cursor in it, `v`,
        `n`, `b` and the rest did nothing, which reads as the program
        having stopped listening. There is no second place to type in this
        window, so the rule is simple -- a field keeps the keys it needs
        to hold a number, and the window gets the rest.

        The ⌘ commands do not come through here at all: they are
        `QAction`s with an application shortcut context, which Qt delivers
        before any widget sees the key.
        """
        if e.type() not in (QEvent.KeyPress, QEvent.KeyRelease):
            return super().eventFilter(obj, e)
        if QApplication.activeModalWidget() is not None or not self.isActiveWindow():
            return super().eventFilter(obj, e)
        if not (isinstance(obj, QWidget) and self.isAncestorOf(obj)):
            return super().eventFilter(obj, e)
        # Tab walks the corners, and it has to be taken before the focused
        # widget sees it: Qt spends Tab on moving the focus, and the window
        # only got it in `event` while nothing else in it could take the
        # focus. Once the crop box could, Tab went canvas -> box -> canvas
        # and the chosen corner never moved.
        if self._tab(e):
            return True
        typing = isinstance(obj, (QLineEdit, QAbstractSpinBox)) or isinstance(
            obj.parent(), QAbstractSpinBox)
        if typing:
            if e.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
                if e.key() in self.TYPED_WITH_CTRL:
                    return super().eventFilter(obj, e)
            elif e.key() in self.TYPED:
                return super().eventFilter(obj, e)
        if obj is self or obj is self.canvas:
            return super().eventFilter(obj, e)     # it gets there on its own
        if e.type() == QEvent.KeyPress:
            self.keyPressEvent(e)
        else:
            self.keyReleaseEvent(e)
        return True

    def _shortcuts(self):
        for key, fn in (("Ctrl+Z", self._undo),
                        ("Ctrl+Shift+Z", self._redo), ("Ctrl+W", self.close),
                        ("Ctrl+D", self._redetect),
                        ("Ctrl+Shift+D", self._redetect_roll),
                        ("Ctrl+R", self._process),
                        ("Ctrl+Shift+R", self._process_scan),
                        ("Ctrl+L", self._look),
                        ("Ctrl+Shift+L", self._light_falloff),
                        ):
            a = QAction(self)
            a.setShortcut(QKeySequence(key))
            # Not the default `WindowShortcut`: with the cursor in the crop
            # box, ⌘B did nothing. An application context is delivered
            # before any widget sees the key, and a modal dialog blocks the
            # window it covers, so it cannot fire behind one.
            a.setShortcutContext(Qt.ApplicationShortcut)
            a.triggered.connect(fn)
            self.addAction(a)

    # ---- keys
    def event(self, e):
        """Tab is focus navigation to Qt, so it never reaches keyPressEvent.

        It is taken here instead, and only inside the mode: outside it there
        is no chosen vertex for it to walk.
        """
        if self._tab(e):
            return True
        return super().event(e)

    def _tab(self, e):
        """Walk the chosen corner on Tab / ⇧Tab inside the mode; say if
        the key was taken. A release is taken too, so that a Tab that
        walked a corner does not reach a widget as half a focus move."""
        if not (self.mode and e.type() in (QEvent.KeyPress, QEvent.KeyRelease)
                and e.key() in (Qt.Key_Tab, Qt.Key_Backtab)):
            return False
        if e.type() == QEvent.KeyPress:
            # Where it lands is said **as seen** -- top left, bottom right,
            # and which frame -- not by index. The scan is mirrored, so corner
            # 0 of the fit is the one on the right, and a number that disagrees
            # with what you are looking at is worse than no number.
            self.status.setText(
                self.canvas.walk(-1 if e.key() == Qt.Key_Backtab else 1))
        return True

    def keyPressEvent(self, e):
        k, mods = e.key(), e.modifiers()
        fine = bool(mods & Qt.ShiftModifier)
        if self.dusting:
            # First, before the held keys: the picture on the canvas is the
            # full-size frame, and none of the other ways of looking can be
            # laid over it.
            if k == Qt.Key_C and e.isAutoRepeat():
                return
            return self._dust_key(k, mods)

        # Held keys are for looking, pressed keys are for changing. A held
        # key is bare: `⌘V` is a paste and not a request to look at anything,
        # and testing the letter without testing the modifiers made the one
        # swallow the other.
        held = not (mods & (Qt.ControlModifier | Qt.AltModifier
                            | Qt.MetaModifier))
        if self.canvas.picking_base and k == Qt.Key_Escape:
            return self._stop_pick("no base picked")
        if k == Qt.Key_B and held and mods & Qt.ShiftModifier:
            return None if e.isAutoRepeat() else self._pick_base()
        if k == Qt.Key_B and held and not e.isAutoRepeat():
            return self._hold_base()
        if k == Qt.Key_L and held and not e.isAutoRepeat():
            return self._hold_light(True)
        if k == Qt.Key_C and held and not fine and not self.mode:
            # the original, dust and all, while held -- the same key as in
            # `s`; `⌘C` is a copy and never reaches here as held
            return None if e.isAutoRepeat() else self._hold_dust(True)
        if ((self.canvas.view or "").startswith("result:")
                and not self._walking()):
            # Turning while the frame is on screen, which is the only moment a
            # person can tell whether it is upright. It is why there is no key
            # for it anywhere else: from the editor you would be guessing.
            #
            # Left and right are turns; up and down are places to turn it to.
            # A quarter clockwise, a quarter back, upright, upside down --
            # four arrows, four meanings, and pressing up twice does nothing,
            # which is the honest difference between a turn and a destination.
            name = self.canvas.view.split(":", 1)[1]
            if k == Qt.Key_Right:
                return self._turn_part(90, name)
            if k == Qt.Key_Left:
                return self._turn_part(-90, name)
            if k == Qt.Key_Up:
                return self._turn_part(0, name, absolute=True)
            if k == Qt.Key_Down:
                return self._turn_part(180, name, absolute=True)
        if held and not e.isAutoRepeat() and Qt.Key_1 <= k <= Qt.Key_4:
            # A frame is asked for by position: hold 1 for the first, 2 for the
            # second. It asks for a frame rather than for "the selected one",
            # which is a thing you would have to select first -- and looking
            # should not need a change made before it.
            #
            # They were the frames' own letters, which read better on the sheet
            # and sit at opposite ends of the keyboard while the other hand is
            # on the arrows. Adjacent keys win.
            name = "abcd"[k - Qt.Key_1]
            parts = (self.item() or {}).get("parts", [])
            at = [i for i, q in enumerate(parts) if q["name"] == name]
            if at:
                # **Holding a frame picks it**, and that is one act and not
                # two. Everything that follows the selection -- the read
                # line, the crop caption, `space`, `r`, `⌘C` -- went on
                # answering about the other half while this one was on the
                # canvas, which is the same fault `_shown` fixed for
                # turning. Asking to see a frame is choosing it.
                if self.canvas.sel != at[0]:
                    self.canvas.sel = at[0]
                    self._refresh()
                return self._hold(f"result:{name}", f"frame {name}, cut")

        if self.mode:
            return self._mode_key(k, mods)

        if k == Qt.Key_Space:
            return self._cut()
        if mods & Qt.AltModifier and k in (Qt.Key_Up, Qt.Key_Down):
            return self._move_rows(-1 if k == Qt.Key_Up else +1)
        if k in (Qt.Key_Down, Qt.Key_Right, Qt.Key_N):
            return self._step_cut(+1) if self._walking() else self._step(+1)
        if k in (Qt.Key_Up, Qt.Key_Left, Qt.Key_P):
            return self._step_cut(-1) if self._walking() else self._step(-1)
        if k == Qt.Key_F:
            return self.canvas.fit()
        if k == Qt.Key_E:
            return self._enter_mode()
        if k == Qt.Key_S and not mods & Qt.ControlModifier:
            return self._enter_dust()
        # Here rather than in the mode, which is where they started. Copying a
        # frame is about carrying it from one scan to **another**, and moving
        # between scans is what you do out here; inside the mode you are
        # placing corners on one. Having to enter the mode to copy was a step
        # that existed only because the code put them there.
        if k == Qt.Key_C and mods & Qt.ControlModifier:
            return self._copy_frames()
        if k == Qt.Key_V and mods & Qt.ControlModifier:
            return self._paste_frames()
        if k == Qt.Key_M:
            return self._mirror()
        if k == Qt.Key_I:
            return self._toggle_numbers()
        # Turning, out here where a person is looking through the roll. It
        # used to need the frame held down, on the argument that upright is
        # only visible on the cut frame -- true, and it still is, but the
        # canvas already draws the cut, and a turn a hand cannot reach is a
        # turn nobody makes.
        if k == Qt.Key_V:
            return self._cycle(-1 if fine else 1)
        if k == Qt.Key_R:
            # Anticlockwise on the bare key, because a frame that comes up
            # on its side is almost always lying the same way, and the way
            # back from it is one press of the key without a modifier.
            return self._turn_right() if fine else self._turn_left()
        if k in (Qt.Key_Delete, Qt.Key_Backspace):
            return self._skip()
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if self.dusting:
            if e.key() == Qt.Key_C and not e.isAutoRepeat():
                self._dust_original(False)
            return
        if e.key() == Qt.Key_L and not e.isAutoRepeat() and self.flat_off:
            return self._hold_light(False)
        if e.key() == Qt.Key_C and not e.isAutoRepeat() and self.dust_off:
            return self._hold_dust(False)
        if e.key() == Qt.Key_B and e.modifiers() & Qt.ShiftModifier:
            return                       # `⇧B` is a press, not a hold
        if ((e.key() == Qt.Key_B or Qt.Key_1 <= e.key() <= Qt.Key_4)
                and not e.isAutoRepeat()):
            # Back to the view that was on -- the develop or the cut --
            # not to the whole scan: a held key is a look away from wherever
            # you were, and it has to put you back there.
            self._back_to_view()

    # ---- the three ways of looking
    def _cycle_back(self):
        """`⇧V`: the three ways of looking, the other way round."""
        self._cycle(-1)

    def _cycle(self, step=1):
        """`v`: the scan, the positive, the develop -- one key, in order;
        `⇧V` the other way round.

        They are one question with three answers, which is how far down the
        pipeline what you are looking at has come, so they are one axis and
        not three switches. Framing (`space`), mirroring (`m`) and turning
        (`r`) are other axes and never change which of these is on -- which
        is the fault this replaced: `space` drew from a preview
        normalisation, so cropping a frame silently changed what its
        colours meant, and it read as a result.
        """
        was = self.colour
        self.colour = MODES[(MODES.index(self.colour) + step) % len(MODES)]
        if self.colour == "develop" and not (self.item() or {}).get("parts"):
            self.colour = was
            return self.status.setText("no frame here to develop")
        # **It does not crop.** The develop is measured inside a frame, but
        # what it puts on screen is the whole scan with each frame developed
        # in its place -- the two axes are separate, and a colour key that
        # quietly reframed the picture is the fault this replaced.
        self.status.setText("")
        self._show_mode()

    def _show_mode(self):
        """Put the current way of looking on the canvas, and name it.

        The colour axis never reframes. With the cut off the develop is the
        whole scan with each frame developed in its place; with the cut on
        it is the frame, which is what `space` asked for.
        """
        if self.dusting:
            return                         # marking dust: see `_hold`
        name, it = self.name(), self.item()
        if not name or not it or not self.here(name):
            return self._hold(None, "")
        if self.colour == "develop":
            return self._want_developed()
        pm = self._mode_pixmap(name, it)
        if pm is None:
            return
        self.canvas.pixmap = pm
        self.canvas.pixmap_dev = None
        self.canvas.label = self._label()
        self._hold("result:" + (self.part() or {"name": "a"})["name"]
                   if self.cut else None, self._mode_says())
        self.canvas.update()

    @staticmethod
    def _canvas_scale(it):
        """Scan pixels per canvas pixel, worked out rather than read.

        Every way of looking is drawn at the width `image_io.preview` bins
        to, off a half-size read, so the number is arithmetic on the scan's
        own width -- and asking the file instead meant reading and
        demosaicing it to learn something the document already knows.
        """
        w = max(1, int(it["scan"][0]))
        half = w // 2
        k = max(1, round(half / image_io.PREVIEW_WIDTH))
        return w / max(1, half // k)

    def _mode_says(self):
        return MODE_NAMES[self.colour]

    def _label(self):
        """The short word for the canvas: the long one is the read line's.

        On the scan it also says how much of the raw clipped, beside the
        marks that show where -- the share, because a block marked on
        screen can stand for one photosite or for all of them."""
        word = {"scan": "the scan", "positive": "the positive",
                "develop": "developed"}[self.colour]
        got = self.clip_masks.get(self.name()) if self.colour == "scan" else None
        if got and got[1] is not None:
            word += f" · clipped {_share(got[2])}"
        if self.flat_off:
            word += (" · falloff corrected" if self.colour == "scan"
                     else " · falloff not corrected")
        if self.dust_off:
            word += " · original"
        return word

    CLIP_RED = (255, 40, 40)
    CLIP_YELLOW = (255, 225, 0)

    def _clip_for(self, pm):
        """The two blink states of the clip marks for the pixmap on the
        canvas, or None.

        **Only on the scan as it came.** Clipping is a fact about the raw,
        so it is drawn on the one way of looking that is the raw: a
        positive or a develop of a clipped photosite is a guess at a
        number the sensor never gave, and marking it there would say the
        develop clipped it. Any channel counts: one photosite of a block
        at the ceiling marks the block, red and yellow in turn.

        Read the first time the scan is looked at, not when the roll is
        opened: it is one more pass over the mosaic, and most scans are
        never looked at this way."""
        name = self.name()
        if (not pm or not name or self.colour != "scan"
                or self.mode_cache.get((name, "scan")) is not pm):
            return None
        if self._clip_over and self._clip_over[0] == name \
                and self._clip_over[1] is pm:
            return self._clip_over[2]
        if name not in self.clip_masks:
            try:
                self.clip_masks[name] = image_io.clipped_blocks(
                    self.roll / name, pm.width(), pm.height())
            except Exception:
                self.clip_masks[name] = (None, None, 0.0)
            self.canvas.label = self._label()
        mask = self.clip_masks[name][0]
        if mask is None or not mask.any():
            self._clip_over = (name, pm, None)
            return None
        pair = tuple(_mask_pix(mask, c) for c in (self.CLIP_RED, self.CLIP_YELLOW))
        self._clip_over = (name, pm, pair)
        return pair

    def _mode_pixmap(self, name, it, which=None):
        """The scan drawn in one of the ways, cached. `develop` is not here:
        it is per frame and goes through the render thread."""
        # **The develop draws on the scan.** Two reasons, and the second is
        # the one that matters: it is 0.39 s against 1.06 s for the positive,
        # which is most of what a scan change costs; and the surround of a
        # developed frame is the rebate and the holder, which are not a
        # photograph, so leaving them as the sensor saw them says that louder
        # than a positive of them would. Normalising the key here also stops
        # the background being computed twice a scan -- `_select` asked for
        # the mode and `_want_developed` asked again for its base, and in the
        # develop those were two different keys for one picture.
        which = {"develop": "scan"}.get(which or self.colour, which or self.colour)
        key = (name, which)
        # **Both or neither.** The pixmap and the array are two caches of one
        # picture and they used to be evicted on different counts, so a
        # pixmap could outlive its array -- and the uncropped develop, which
        # composites onto the array, then returned without drawing anything
        # and left the read line naming the view before it. One cache of
        # record, the array; the pixmap is made from it.
        if key in self.mode_arrays:
            if key not in self.mode_cache:
                self.mode_cache[key] = _pix(self.mode_arrays[key])
            return self.mode_cache[key]
        self.status.setText("reading…")
        QApplication.processEvents()
        try:
            shown = (self._scan_image(name) if which == "scan"
                     else self._positive_image(name, it))
        except Exception as e:
            self.status.setText(f"could not be read: {e}")
            return None
        # **The same pixels as every other way of looking.** The canvas draws
        # the frames from `scale`, which is scan pixels per canvas pixel, and
        # these come off a half-size read -- twice what `image_io.preview`
        # bins to. Left alone, every rectangle came out at half its size over
        # a picture at twice the size, and the zoom moved under the hand
        # every time `x` was pressed.
        # To the width `image_io.preview` bins to, so every way of looking
        # shares one scale and the rectangles land on the frame -- worked
        # out the same way it works it out, rather than by reading the scan
        # again to ask how big it came out.
        k = max(1, round(shown.shape[1] / image_io.PREVIEW_WIDTH))
        if k > 1:
            ys = np.arange(0, shown.shape[0] // k * k, k)
            xs = np.arange(0, shown.shape[1] // k * k, k)
            shown = shown[ys][:, xs]
        self.mode_arrays[key] = shown.astype(np.float32)
        self.mode_cache[key] = _pix(self.mode_arrays[key])
        self._trim_modes()
        self.status.setText("")
        return self.mode_cache[key]

    MODE_BUDGET = 250_000_000        # bytes of drawn scans, about 50 of them

    def _trim_modes(self):
        """**By bytes, not by a count of six.**

        Six was right when a drawn scan was something the canvas made on its
        way to showing one. The develop now sits on one, and the roll makes
        them all ahead of time, so six of thirty-seven meant every step but
        the last few read the raw again.
        """
        total = sum(a.nbytes for a in self.mode_arrays.values())
        while total > self.MODE_BUDGET and len(self.mode_arrays) > 2:
            k = next(iter(self.mode_arrays))
            total -= self.mode_arrays[k].nbytes
            self._forget_modes(key=k)

    def _forget_modes(self, name=None, key=None):
        """Drop the drawn scans -- **both caches, always**.

        The pixmap and the array are two caches of one picture, and they
        used to be emptied separately. That is how `⌘B` came to move the
        base while the positive on screen did not: the pixmaps went, the
        arrays survived, and the next draw rebuilt a pixmap from a picture
        of the old base. The check in `tests/walkthrough.py` that presses
        `⌘B` and compares the canvas caught it, which is why this is one
        method and not five call sites.
        """
        keys = ([key] if key is not None else
                [k for k in list(self.mode_arrays) + list(self.mode_cache)
                 if name is None or k[0] == name])
        for k in keys:
            self.mode_arrays.pop(k, None)
            self.mode_cache.pop(k, None)
            if k[1] == "scan":
                self.clip_masks.pop(k[0], None)
        self._clip_over = None
        self._sets = {}             # the held keys' pictures are of the same things

    def _flat(self):
        """The rig's light falloff the roll names (`flatfield`), or None.
        The positive and the base are read through it, as the develop
        reads them; the scan is not -- it is what the sensor saw."""
        if self.flat_off:
            return None
        return self._flat_named()

    def _flat_named(self):
        """The calibration the roll names, whatever `hold l` says."""
        from . import develop, flatfield
        try:
            return flatfield.of(dict(develop.TASTE)
                                | dict(self.doc["roll"].get("develop") or {}))
        except (OSError, ValueError):
            return None

    CACHES = ("mode_arrays", "mode_cache", "dev_plain", "dev_ev", "base_cache")

    @property
    def flat_off(self):
        """`hold l` is down: the falloff the other way round."""
        return self.held_other == "flat"

    @property
    def dust_off(self):
        """`hold c` is down: the develop with its dust left in."""
        return self.held_other == "dust"

    def _hold_light(self, on):
        """`hold l`: the stand's light falloff the other way round, for as
        long as the key is down -- **not corrected** on the positive and
        the develop, where it is, and **corrected** on the scan, where it
        is not. So holding it on a photo shows what the calibration does to
        it, and on a blank frame shows whether it comes out flat.

        Everything else is the roll's own: the base and the balance stay
        what the roll measured (under the calibration), so the difference
        on screen is the falloff and nothing re-measured around it."""
        if on and self._flat_named() is None:
            return self.status.setText("no light falloff calibration on this "
                                       "roll — ⇧⌘L chooses or makes one")
        self._hold_other("flat", on)

    def _hold_dust(self, on):
        """`hold c`: the original, the develop with the dust left in, for as
        long as the key is down. The same key as inside `s`, where it is
        the full-size frame; here it is the canvas's own develop."""
        from . import dust
        name = self.name()
        marks = [m for m in (self._dust_record(name).get("marks", []) if name else [])
                 if dust.used(m)]
        if on and not marks:
            return self.status.setText("no dust marked on this scan")
        if on and self.colour != "develop":
            return self.status.setText("the dust is taken out of the develop "
                                       "only: v to it")
        self._hold_other("dust", on)
        if on:
            n = len(marks)
            self.status.setText(f"original: {n} mark{'s' if n != 1 else ''} "
                                "left in")

    def _hold_other(self, kind, on):
        """Swap in the pictures drawn the other way, `kind` being `flat`
        (`hold l`) or `dust` (`hold c`), and back when the key is let go.

        They are kept apart, as another set of the view caches swapped in
        while the key is down: one cache with a flag in every key was the
        alternative, and every reader would have needed a test for it.
        The sets not on screen wait in `_sets` -- `real` for the real one
        while a key is down -- and are dropped whenever the real one is
        (`_forget_modes`). One key at a time: two would be a third set."""
        if on == (self.held_other == kind) or (on and self.held_other):
            return
        # Whatever is developing now is of the set on screen; the roll's own
        # background run would fill the wrong one, so it waits.
        self._stop_prefetch()
        fresh = lambda: {n: type(getattr(self, n))() for n in self.CACHES}
        here = {n: getattr(self, n) for n in self.CACHES}
        if on:
            self._sets["real"] = here
            other = self._sets.pop(kind, None) or fresh()
        else:
            self._sets[kind] = here
            other = self._sets.pop("real", None) or fresh()
        for n in self.CACHES:
            setattr(self, n, other[n])
        self.held_other = kind if on else None
        self._clip_over = None
        # drawn again from the swapped caches: `_back_to_view` only puts
        # back the view, and the picture in it is still the other set's
        self._show_mode()
        if self.cut and self.colour != "develop":
            self._show_cut()            # the cut is drawn off that picture
        self.canvas.label = self._label()
        self.canvas.update()
        if not on:
            self._prefetch_later()

    def _scan_image(self, name):
        """**The scan as it came**: the sensor's transmittance, nothing
        normalised, a 2.2 gamma so a screen can show it. It is dark because
        a negative on a light table is dark, and that is the point of it --
        nothing here is measured off this frame, so two scans of one
        negative come out as different as they really are.
        """
        from . import develop
        # corrected only while `hold l` asks for it: this is the scan
        rgb, _ = develop.linear(self.roll / name, True, None,
                                self._flat_named() if self.flat_off else None)
        return np.clip(rgb, 0, 1) ** (1 / 2.2)

    def _positive_image(self, name, it):
        """**The positive, with nothing automatic in it** -- and the pixels
        the positive DNG holds: `apply` writes `d = log10(base / T)` and
        `P = 10**(d / gamma) - 1`, which is these two lines. The DNG also
        has the roll's balance divided into them, so its colour tags can
        call that grey 5600 K (`positive.positive_tags`); its exposure goes
        in as a tag (`BaselineExposure`).

        One fixed display scale for the whole roll, not the frame's own, so
        three exposures of one negative come out at one brightness and two
        that really differ say so. `PLAIN_WHITE` is where the base's own
        scale puts a well-exposed mid-tone, read off the four rolls.

        The base is read exactly as the develop reads it, through
        `develop.base_of` and what the roll measured: its colour is the
        film's, and which of the two readings to trust is the roll's, so a
        per-scan answer carries one scan's noise. On DSC_0282/3/4 that cost
        0.088 in B/G against 0.017.
        """
        from . import develop
        if not it["parts"]:
            raise ValueError("no frame here, so no base to divide by")
        rgb, scale = develop.linear(self.roll / name, True, None, self._flat())
        part = it["parts"][0]
        reb = develop.rebate(rgb, scale, it)
        top = develop.frame_top(develop.inside(
            develop.cut(rgb, scale, part, it["angle"])))
        lvl = (np.percentile(reb, 90, axis=0).astype(np.float32)
               if len(reb) > 500 else np.zeros(3, np.float32))
        m = self.doc["roll"].get("measured") or {}
        colour = m.get("base_colour")
        if not colour:
            base = np.maximum(lvl, top)
        else:
            trust, unit = self._base_choice(name)
            colour, trust, unit = develop.own_base(
                develop.with_specks(it, self.doc, name), colour, trust, unit)
            base = develop.base_of(lvl, top, np.array(colour, np.float32),
                                   trust, unit)
        gamma = float(self.doc["roll"].get("develop", {}).get("gamma", 0.65))
        d = np.clip(np.log10(base[None, None, :] / np.clip(rgb, 1e-5, None)),
                    0, None)
        P = 10.0 ** (d / gamma) - 1.0
        return np.clip(P / PLAIN_WHITE, 0, 1) ** (1 / 2.2)

    # ---- the develop, which is per frame and renders off the thread
    def _walking(self):
        """Whether `n` and the arrows step pictures rather than scans. That
        is `space`'s doing and nobody else's: with one frame on the canvas
        the roll is a sequence of pictures, and on half frame the two
        exposures of a scan are two of them."""
        return self.cut

    def _back_to_view(self):
        """Back to whatever is on -- used by anything that was a look away
        from it, which is every held key."""
        if self.dusting:
            return                         # marking dust: see `_hold`
        if self.colour == "develop":
            self._want_developed()
        elif self.cut:
            self._show_cut()
        else:
            self._hold(None, self._mode_says())
        self.canvas.update()

    def _want_developed(self):
        """The develop on the canvas: the whole scan with each frame cut out
        of its own develop, and just that frame when `space` is on.

        **Everything here is a cut of a picture already developed.** The
        window keeps the scan developed, uncropped, in the scan's own
        coordinates; showing a crop is a mask, and moving the crop is a new
        mask. Nothing is developed again for it -- which is the point, and
        also why there is nothing left to misalign: the four faults this
        replaced were all a straightened frame being scaled, turned and
        centred back onto a scan it had been cut out of.
        """
        if self.dusting:
            return                         # marking dust: see `_hold`
        from . import develop
        name, part = self.name(), self.part()
        it = self.item()
        if not name or not part or not it or not self.here(name):
            self.colour = "positive"
            return self._hold(None, "")
        key = (name, part["name"])
        self.dev_want = key
        taste = dict(develop.TASTE) | dict(self.doc["roll"].get("develop") or {})
        inset = develop.crop_inset(taste)
        base = self.dev_plain.get(key)
        ready = base is not None
        self.canvas.label = self._label() if ready else "developing…"
        said = self._dev_says(key) if ready else "developing…"

        if self.cut:
            if ready:
                scale = it["scan"][0] / max(base.shape[1], 1)
                out = self._developed_cut(key, it, part, inset, scale)
                self.canvas.pixmap_dev = _pix8(out) if out is not None else None
            else:
                self.canvas.pixmap_dev = None
            self._hold("developed", said)
            return self._start_render()

        if not ready:
            # Nothing developed yet, and the scan underneath is not the
            # develop. The frames arrive one at a time and each redraws.
            self.canvas.pixmap = None
            self._hold(None, said)
            self.canvas.update()
            return self._start_render()

        scale = it["scan"][0] / max(base.shape[1], 1)
        shown = base.astype(np.float32) / 255.0
        # Each frame out of its own develop, inside its own crop. The
        # surround is the first frame's, because the rebate and the holder
        # have no automatics of their own and something has to draw them.
        for q in it["parts"]:
            k = (name, q["name"])
            other = self.dev_plain.get(k)
            if other is None or k == key and len(it["parts"]) == 1:
                continue
            # **Each frame's own develop fills its own rectangle**, not the
            # crop inside it. Masked at the crop, the ring between the crop
            # and the frame edge fell back to the backdrop -- which is the
            # *selected* frame's develop -- so the other half of a row wore
            # a band of somebody else's exposure just inside its outline,
            # and it read as the window shading the frame you had not
            # picked. Where the crop cuts is said by the dashed line on the
            # rectangle, which is a line and not a change of exposure.
            m = self._crop_mask(shown.shape[:2], q, it.get("angle", 0.0),
                                scale, 0.0)
            shown[m] = other[m].astype(np.float32) / 255.0
        self.canvas.pixmap = _pix(shown)
        self.canvas.pixmap_dev = None
        self._hold(None, said)
        self.canvas.update()
        # **And then the other half.** Uncropped, this draws every frame of
        # the scan, so every frame of the scan has to be developed -- and
        # only the selected one was ever asked for. The half that had not
        # been kept its rectangle filled with the backdrop, which is the
        # selected frame's develop, so a row showed one exposure twice and
        # clicking the other half appeared to change only itself. The
        # roll's background pass would have got there eventually; the frame
        # on screen should not wait for it.
        missing = [q["name"] for q in it["parts"]
                   if (name, q["name"]) not in self.dev_plain]
        if missing:
            self.dev_want = (name, missing[0])
        self._start_render()

    def _start_render(self):
        """One at a time. A second press of `n` while one is running moves
        what is wanted; it does not start a second develop, because the
        machine has one of these in it and the picture nobody is looking at
        any more is not worth the second.

        The rest of the roll is the pool's job (`_prefetch_roll`), not this
        thread's: this one exists so the frame being looked at jumps the
        queue, and there is nothing for it to do when that one is in hand.
        """
        if self.dev_worker is not None:
            return
        if not self.dev_want or self.dev_want in self.dev_plain:
            return
        from . import develop
        key = self.dev_want
        name, pname = key
        it = self.doc["items"].get(name) or {}
        parts = [q for q in it.get("parts", []) if q["name"] == pname]
        if not parts:
            return
        # Copies, because the worker reads them while the arrows can still
        # move the frame they came from.
        item = copy.deepcopy(develop.with_specks(it, self.doc, name))
        part = copy.deepcopy(parts[0])
        path, roll = self.roll / name, self.roll
        taste = dict(develop.TASTE) | dict(self.doc["roll"].get("develop") or {})
        measured = dict(self.doc["roll"].get("measured") or {})
        one = self.doc["roll"].get("one_roll", True)
        held = self.held_other
        if held == "dust":
            item.pop("specks", None)            # `hold c`

        def job():
            # **The same job the pool runs**, so the two cannot produce two
            # different pictures into one cache. They did: this thread made
            # the frame cut out and straightened while the pool made the
            # whole scan, and compositing one against the other failed on
            # the shapes -- 1010 rows against 980.
            #
            stats = develop.stats_of(measured, taste, roll)
            if stats is None:
                raise RuntimeError("the roll's base and balance are not "
                                   "measured yet")
            if held == "flat":
                stats["flat"] = None            # `hold l`
            _k, small, ev = develop.whole_job(
                (path, item, part, stats, taste, one, image_io.PREVIEW_WIDTH))
            return None, small, ev

        self.dev_worker = _Render(key, job)
        self.dev_worker.held = held
        self.dev_worker.done.connect(self._developed_ready)
        self.dev_worker.failed.connect(self._developed_failed)
        self.dev_worker.start()

    def _prefetch_roll(self):
        """Start the roll developing itself, if it is not already.

        Called when the roll is opened and whenever the measurement it
        stands on changes, because every picture it made is then of a
        different base.
        """
        import os
        from . import develop
        self._stop_prefetch()
        if self.colour != "develop" or self.mode or self.held_other:
            return
        if self.prefetch is not None and self.prefetch.isRunning():
            # The last one is still letting its running frames finish. Two
            # pools at once is twelve processes on a machine with twelve
            # cores, so wait for it rather than pile on.
            return self._prefetch_later(600)
        taste = dict(develop.TASTE) | dict(self.doc["roll"].get("develop") or {})
        stats = develop.stats_of(self.doc["roll"].get("measured"), taste,
                                 self.roll)
        if stats is None:
            return
        one = self.doc["roll"].get("one_roll", True)
        measured = dict(self.doc["roll"].get("measured") or {})
        jobs = []
        # **From where the person is, forward, wrapping round.** It used to
        # start at the first scan in the folder whatever was on screen, so
        # measuring the roll again from scan 22 left the next twenty presses
        # of `n` racing a pool that was busy at the other end of the roll.
        # `n` walks forward; so does this.
        at = max(0, self.list.currentRow())
        order = self.names[at:] + self.names[:at]
        for name in order:
            it = self.doc["items"].get(name) or {}
            if it.get("state") == "skipped" or not self.here(name):
                continue
            for q in it.get("parts", []):
                if (name, q["name"]) not in self.dev_plain:
                    jobs.append((self.roll / name,
                                 develop.with_specks(it, self.doc, name),
                                 q, stats, taste, one,
                                 image_io.PREVIEW_WIDTH))
        if not jobs:
            return
        # Not every core: the window is on one of them, and so is whatever
        # else the person is doing.
        workers = max(1, min(6, (os.cpu_count() or 2) - 2))
        self.prefetch = _Prefetch(jobs, workers)
        # Held until it says it has finished, whatever happens to
        # `self.prefetch` in the meantime: a QThread destroyed while it is
        # running takes the process down with it.
        self._threads.append(self.prefetch)
        self.prefetch.finished.connect(
            lambda w=self.prefetch: w in self._threads
            and self._threads.remove(w))
        self.prefetch.one.connect(self._prefetched)
        self.prefetch.done.connect(self._prefetch_done)
        self.prefetch.start()
        self.status.setText(f"developing {len(jobs)} frames…")

    def _prefetch_later(self, ms=1500):
        """Start the roll developing itself once the keys stop coming."""
        self._stop_prefetch()
        t = getattr(self, "_prefetch_timer", None)
        if t is None:
            from PySide6.QtCore import QTimer
            t = self._prefetch_timer = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(self._prefetch_roll)
        t.start(ms)

    def _stop_prefetch(self, wait=0):
        """Stop the roll developing itself: the timer that would start it,
        and the run that is going.

        **Asked to stop, not waited for.** Cancelling drops the frames that
        have not started and lets the handful running finish, which is
        0.0 to 3.7 s measured; blocking the window for it every time somebody
        presses `⌘B` would be trading one freeze for another. Only closing
        the window waits, and then only so nothing outlives it.
        """
        t = getattr(self, "_prefetch_timer", None)
        if t is not None:
            t.stop()
        w = self.prefetch
        self.prefetch = None
        if w is not None:
            w.stop = True
            if wait and not w.wait(wait):
                # It did not finish in time. **The reference is kept
                # anyway**: Qt aborts the process outright if a running
                # QThread is destroyed, and dropping the last reference to
                # one is how a window that was only being closed came to
                # exit on SIGABRT. `_threads` holds them until they say
                # they are done.
                pass
        w, self.prefetch = self.prefetch, None
        if w is not None:
            w.stop = True
            w.wait(5000)

    def _prefetched(self, key, small, evidence):
        if key in self.dev_plain or self.held_other:
            return
        self._keep_developed(key, small, evidence)
        if self.colour == "develop" and self.name() == key[0]:
            self._want_developed()

    def _prefetch_done(self):
        self.prefetch = None
        if self.colour == "develop" and "developing" in self.status.text():
            self.status.setText("")

    def _keep_developed(self, key, small, evidence):
        """One scan developed, in the scan's own coordinates, as it came off
        the pool: not mirrored, not turned, not cut. **Everything the window
        shows is a cut of this**, so a crop moved is a mask recomputed and
        nothing developed again."""
        self.dev_plain[key] = np.ascontiguousarray(small)
        self.dev_ev[key] = evidence
        self._trim_warm()
        if key == (self.name(), (self.part() or {}).get("name")):
            self._say_numbers()

    @staticmethod
    def _crop_mask(shape, part, angle, scale, inset):
        """Which pixels of a scan-sized picture fall inside this frame's
        crop -- the rectangle the cut will be made on, tilted with the
        film."""
        cx, cy = part["centre"][0] / scale, part["centre"][1] / scale
        w = part["size"][0] * (1 - 2 * inset) / scale / 2
        h = part["size"][1] * (1 - 2 * inset) / scale / 2
        t = math.radians(angle)
        co, si = math.cos(t), math.sin(t)
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        u = (xx - cx) * co + (yy - cy) * si
        v = -(xx - cx) * si + (yy - cy) * co
        return (np.abs(u) <= w) & (np.abs(v) <= h)

    def _developed_cut(self, key, it, part, inset, scale):
        """The frame on its own: the scan's develop straightened and cut to
        the crop. The same two lines `develop.cut` uses on the linear image,
        on a picture that is already developed."""
        from scipy import ndimage
        a = self.dev_plain.get(key)
        if a is None:
            return None
        angle = it.get("angle", 0.0)
        cx, cy = part["centre"][0] / scale, part["centre"][1] / scale
        w = part["size"][0] * (1 - 2 * inset) / scale
        h = part["size"][1] * (1 - 2 * inset) / scale
        pad = int(max(w, h) / 2 + 8)
        sub = np.zeros((2 * pad, 2 * pad, 3), np.float32)
        x0, y0 = int(cx) - pad, int(cy) - pad
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(a.shape[1], x0 + 2 * pad), min(a.shape[0], y0 + 2 * pad)
        if sx1 <= sx0 or sy1 <= sy0:
            return None
        sub[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
        if abs(angle) > 1e-3:
            sub = ndimage.rotate(sub, angle, reshape=False, order=1,
                                 mode="nearest")
        ccx, ccy = cx - x0, cy - y0
        out = sub[max(0, int(ccy - h / 2)):int(ccy + h / 2),
                  max(0, int(ccx - w / 2)):int(ccx + w / 2)]
        if out.size == 0:
            return None
        if self.doc["roll"].get("mirrored", True):
            out = out[:, ::-1]
        turn = int(part.get("turn", 0) or 0) % 360
        if turn:
            out = np.rot90(out, k=-(turn // 90))
        return np.ascontiguousarray(out).astype(np.uint8)

    def _developed_ready(self, key, _unused, shown, evidence):
        made = getattr(self.dev_worker, "held", None)
        self._drop_worker()
        if made != self.held_other:
            # developed for the other side of a held key, which was let go
            # (or pressed) meanwhile: into that set, and this one asks again
            into = self._sets.get(made or "real")
            if into is not None:
                into["dev_plain"][key] = np.ascontiguousarray(shown)
                into["dev_ev"][key] = evidence
            if self.colour == "develop" and self.name() == key[0]:
                self._want_developed()
            return
        self._keep_developed(key, shown, evidence)
        if self.colour == "develop" and self.name() == key[0]:
            # Only the scan on screen redraws; the rest are being made
            # ahead of time and must not move what is being looked at.
            self._want_developed()
        else:
            # Done with a frame nobody is on now. **What is wanted stays
            # wanted**: this cleared it, so a scan picked while the one
            # before it was still developing never developed at all --
            # the canvas said "developing…" for good, unless the roll's
            # pool happened to get there.
            if self.dev_want == key:
                self.dev_want = None
            self._start_render()

    def _developed_failed(self, key, why):
        self._drop_worker()
        if self.dev_want == key:
            # **Not again until it is asked for again.** It used to start
            # the same render straight back up, which fails the same way,
            # so a frame that cannot develop -- a roll whose base was never
            # measured -- looped as fast as LibRaw could read the scan. And
            # it said so only in the develop: each failure put the develop's
            # view back on the canvas, so `v` out of it lasted one failure.
            self.dev_want = None
            if self.colour == "develop":
                self._hold("developed", f"not developed: {why}")
        self._start_render()

    def _drop_worker(self):
        w, self.dev_worker = self.dev_worker, None
        if w is not None:
            w.wait()

    def _dev_says(self, key):
        """What the develop did to this frame, beside the picture of it: the
        exposure it applied, what the shadow rule bought, the slopes the
        curve came out at and the black it took off. The numbers, because a
        label naming a cause is a guess and a label naming a number is not.
        """
        ev = self.dev_ev.get(key) or {}
        if not ev:
            return MODE_NAMES["develop"]
        if ev.get("blank"):
            return "developed — blank frame, written dark"
        lift = float(ev.get("shadow_lift") or 0.0)
        return ("developed — {:+.2f} EV{}, curve {:.2f}/{:.2f}, black {:.3f}"
                .format(float(ev.get("exposure", 0.0)),
                        f", shadows +{lift:.2f}" if lift else "",
                        float(ev.get("shadow_slope", 0.0)),
                        float(ev.get("highlight_slope", 0.0)),
                        float(ev.get("black", 0.0))))

    def _forget_developed(self, name=None):
        """Drop the developed pictures, and the cut frames with them where
        the geometry is what changed. A taste change keeps the frames: they
        are read before the inversion, so `finish` is all a new setting
        costs."""
        if name is None:
            self.dev_plain.clear()
            self.dev_ev.clear()
            self._sets = {}
        else:
            for k in [k for k in self.dev_plain if k[0] == name]:
                self.dev_plain.pop(k)
                self.dev_ev.pop(k, None)
            self._forget_modes(name)
        # Only redraw if what was dropped is what is on screen: undo and
        # redo drop a scan before they move to it.
        if self.colour == "develop" and (name is None or name == self.name()):
            self._want_developed()

    def _hold_base(self):
        """`hold b`: every pixel of the scan whose tone is the base's.

        Not where the base was sampled -- what the base *is*, laid over the
        scan. A base that is right lights up unexposed film and nothing
        else; one that is too dark lights up inside the picture as well,
        which cannot happen, because nothing on a negative is more
        transparent than unexposed film. That is the check, and it needs no
        reference outside the scan.
        """
        import numpy as np
        from . import develop
        name = self.name()
        if not name or not self.here(name):
            return
        it = self.doc["items"][name]
        if not it["parts"]:
            return self.status.setText("no frame here, so no base to read")
        if name not in self.base_cache:
            self.status.setText("reading the base…")
            QApplication.processEvents()
            try:
                rgb, scale = develop.linear(self.roll / name, True, None,
                                            self._flat())
                part = it["parts"][0]
                reb = develop.rebate(rgb, scale, it, part)
                lvl = (np.percentile(reb, 90, axis=0).astype(np.float32)
                       if len(reb) > 500 else np.zeros(3, np.float32))
                top = develop.frame_top(develop.inside(
                    develop.cut(rgb, scale, part, it["angle"])))
                # Per channel, and not forced grey: nothing balanced this
                # scan on the way in, so the base still carries the orange
                # mask and a grey one would match nothing at all.
                #
                # **The base that will be used**, not the scan's own
                # reading of it. The check is still the same check -- a
                # base that is right lights up unexposed film and nothing
                # else -- and it is worth more against the number the
                # develop will actually divide by, which on a forced roll
                # is not this scan's. Both readings are still on the line
                # underneath, so a base that disagrees with its scan says
                # so in numbers.
                mm = self.doc["roll"].get("measured") or {}
                if mm.get("base_colour"):
                    trust, unit = self._base_choice(name)
                    colour, trust, unit = develop.own_base(
                        develop.with_specks(it, self.doc, name),
                        mm["base_colour"], trust, unit)
                    base = develop.base_of(
                        lvl, top, np.array(colour, np.float32), trust, unit)
                else:
                    base = np.maximum(lvl, top)
                m = np.ones(rgb.shape[:2], bool)
                for c in range(3):
                    m &= np.abs(rgb[..., c] - base[c]) <= BASE_TOL * max(base[c], 1e-4)
            except Exception as e:
                return self.status.setText(f"the base could not be read: {e}")
            self.base_cache[name] = (m, base, lvl, top)
            if len(self.base_cache) > 6:
                self.base_cache.pop(next(iter(self.base_cache)))
        m, base, lvl, top = self.base_cache[name]
        # A base lighter than anything the scan holds matches nothing, and
        # 0.0 % on its own reads as a fault. It is not one: it is what
        # forcing the roll onto a frame whose scene has no true black
        # looks like, and the two readings beside it are the evidence.
        nothing = base[1] > max(float(lvl[1]), float(top[1])) * (1 + BASE_TOL)
        self._hold("base", f"{self._base_says()}, {base[1]:.4f} green here "
                           f"(this scan reads rebate {lvl[1]:.4f}, frame "
                           f"{top[1]:.4f}) — ±{BASE_TOL:.0%} of it is on "
                           f"{m.mean() * 100:.1f} % of the scan"
                           + (", which holds nothing that light" if nothing else ""))
        self.canvas.pixmap_base = self._tinted(name, it, m)

    def _tinted(self, name, it, mask):
        """Whatever is on screen with `mask` painted over it, at the size of
        the view it is covering -- the plain one when that is showing, so
        holding a key does not swap the canvas for a picture of another size
        and move the zoom under the hand."""
        import numpy as np
        pm = self.canvas.pixmap or self._mode_pixmap(name, it)
        if pm is None:
            pos, _ = self._load(name, self._frame_box(it))
            shown = image_io.tone(pos, self.ev, self.detail)
        else:
            im = pm.toImage().convertToFormat(QImage.Format.Format_RGB888)
            a = np.frombuffer(im.constBits(), np.uint8).reshape(
                im.height(), im.bytesPerLine())[:, :im.width() * 3]
            shown = a.reshape(im.height(), im.width(), 3).astype(np.float32) / 255.0
        ys = np.linspace(0, mask.shape[0] - 1, shown.shape[0]).astype(int)
        xs = np.linspace(0, mask.shape[1] - 1, shown.shape[1]).astype(int)
        m = mask[ys][:, xs]
        out = shown.copy()
        out[m] = out[m] * 0.45 + np.array([1.0, 0.83, 0.0], np.float32) * 0.55
        return _pix(out.astype(np.float32))

    def _hold(self, which, name):
        if self.dusting:
            # marking dust: the full-size frame stays on the canvas until
            # `esc`, whatever finishes developing behind it
            return
        self.canvas.view = which
        self.held.setText(name)
        self.canvas.update()
        self._say_numbers()

    def _cut(self):
        """`space`: the frame cut, on and off.

        `hold 1` and `hold 2` are a look; this stays, because judging a
        crop means walking the roll looking at nothing but crops. What it
        shows is the selected frame straightened, cropped and the right
        way round -- what `apply` would write.

        It changes what `n` means, and that is the point: with the cut on,
        the roll is a sequence of **pictures**, so the next one is the
        next picture. On half frame that is the other half of the same
        scan before it is the next scan.
        """
        if not self.item() or not self.item()["parts"]:
            return self.status.setText("no frame here to cut")
        self.cut = not self.cut
        self._say_space()
        # **It never touches the processing.** Turning the cut off used to
        # drop the develop to the positive, from when there was no develop
        # of a whole scan to show; there is one now -- the frames on the
        # ground -- and the two axes are separate, so leaving a frame must
        # not quietly change what its colours mean.
        self._back_to_view()
        self.status.setText("")

    def _say_space(self):
        """`space`'s row on the sheet names the mode the press takes you
        to: the preview from out here, the frames from in there. It is the
        only row whose name moves, because it is the only key on the sheet
        that is a mode rather than an act."""
        w = (getattr(self, "rows", None) or {}).get("space")
        if w is not None:
            w.setText("whole scan" if self.cut else "preview")

    def _say_base(self, name=None):
        """`⇧B`'s row on the sheet says what the press will do to the scan
        in hand: pick a base, or take away the one picked. The key is a
        toggle and the sheet said only the first half of it."""
        w = (getattr(self, "rows", None) or {}).get("⇧B")
        if w is not None:
            name = name or self.name()
            own = bool(name and (self.doc.get("bases") or {}).get(name))
            w.setText("remove base" if own else "pick base…")

    def _show_cut(self):
        """Put the canvas on the selected part's cut, with the name of what
        it is showing. Used by `space` and by a held key letting go."""
        if self.dusting:
            return                         # marking dust: see `_hold`
        p = self.part()
        if not p:
            self.cut = False
            return self._hold(None, "")
        n = len(self.item()["parts"])
        self._hold("result:" + p["name"],
                   "cut" + (f" — frame {p['name']} of {n}" if n > 1 else ""))

    def _parts_of(self, row):
        return self.doc["items"][self.names[row]].get("parts") or []

    def _step_cut(self, d):
        """One picture on, not one scan: the other part of this scan first,
        then the next scan that has one. A scan with nothing found is not a
        picture and is stepped over -- with the cut on there is nothing to
        look at on it, and stopping there would show the last frame again.
        """
        row, sel = self.list.currentRow(), self.canvas.sel
        if 0 <= sel + d < len(self._parts_of(row)):
            self.canvas.sel = sel + d
            self._refresh()
            self._back_to_view()
            return
        r = row + d
        while 0 <= r < len(self.names):
            parts = self._parts_of(r)
            if parts and self.doc["items"][self.names[r]]["state"] != "skipped":
                self.list.setCurrentRow(r)
                self.canvas.sel = 0 if d > 0 else len(parts) - 1
                self._refresh()
                # **Back to whichever way of looking is on**, not to the
                # preview cut. It called `_show_cut` outright, from when the
                # cut had only one way of being drawn: in the develop that
                # put the canvas on `result:` with no pixmap behind it, so
                # every press of `n` left the frame blank until something
                # else redrew it. It reads as the window being slow, and the
                # 6 ms it takes are the 6 ms of drawing nothing.
                self._back_to_view()
                return
            r += d

    def _mode_key(self, k, mods):
        fine = bool(mods & Qt.ShiftModifier)
        wide = bool(mods & Qt.AltModifier)
        # Everything that changes the document lives in here: adding a frame,
        # dropping one, lining them up, putting the item back. Outside the mode
        # you are reading, and there is nothing to add a frame *for*.
        #
        # No key chooses a frame, in here or out there. The pointer says it --
        # a corner touched, a frame clicked -- and `⇥` walks from one frame to
        # the next of its own accord.
        if k == Qt.Key_Plus:
            return self._add_part()
        if k == Qt.Key_Minus:
            return self._drop_part()
        if k in (Qt.Key_Backspace, Qt.Key_Delete):
            # back to what was measured: what was done by hand, deleted
            return self._reset()
        if k == Qt.Key_Z:
            # The narrower twin and the wider one share a letter and differ by
            # shift, which is the house rule for exactly this pair.
            self.canvas.zoom_frame() if fine else self.canvas.zoom_corner()
            return
        if k == Qt.Key_F:
            return self.canvas.fit()
        p = self.part()
        if k == Qt.Key_Escape:
            return self._leave_mode(apply=False)
        if k in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_E):
            # `e` closes what `e` opened, as `s` does the dust
            return self._leave_mode(apply=True)
        if not p:
            return
        d = self.canvas.step()
        moved = {Qt.Key_Left: (-d, 0), Qt.Key_Right: (d, 0),
                 Qt.Key_Up: (0, -d), Qt.Key_Down: (0, d)}.get(k)
        if moved:
            # Right on screen is right on screen. The scan is mirrored, so in
            # its own coordinates that is leftwards, and the arrow that agrees
            # with the picture is the one that matters.
            sign = -1 if self.canvas.mirrored else 1
            scope = "all" if wide else ("frame" if fine else "corner")
            what = self.canvas.nudge(moved[0] * sign, moved[1], scope=scope)
            if what == "all":
                # Every frame moved, so the entry is about the item and not
                # about any one part.
                return self._edited("moved", part=False)
            return self._edited(what) if what else None
        # `,` and `.`, which carry `<` and `>` on every layout worth the
        # name. They used to be `[` and `]`, which on a Spanish Mac
        # keyboard need ⌥ and arrive as another key entirely, so they
        # could not be pressed at all.
        turn = {Qt.Key_Comma: -1, Qt.Key_Period: +1}.get(k)
        if turn:
            # One piece of film, one tilt: turning turns the scan, not the
            # part, so a half-frame pair cannot end up disagreeing about how
            # the film was lying.
            sign = -1 if self.canvas.mirrored else 1
            it = self.item()
            it["angle"] = round(
                it["angle"] + turn * sign * STEP_DEG / (10 if fine else 1), 3)
            self.canvas.angle = it["angle"]
            return self._edited("angle", part=False)

    # ---- editing
    def _snapshot(self):
        """Remember the item, so undo has something to put back."""
        n = self.name()
        if n:
            self.undo.append((n, copy.deepcopy(self.doc["items"][n])))
            self.redo.clear()
            del self.undo[:-60]

    def _edited(self, field, part=None):
        """Record that a person set this. Anything in `manual` is what
        measuring again must not undo.

        `part` names it explicitly, for the one case where the part the entry
        is about is not the one now selected: dropping it. `part=False` says
        the field belongs to the item rather than to any part, which is what
        the angle is.
        """
        it = self.item()
        if it is None:
            return
        if part is False:
            path = field
        else:
            name = part or (self.part() or {}).get("name")
            path = f"parts.{name}.{field}" if name else "parts"
        document.mark_manual(it, path)
        self.dirty = True
        # The develop is of a frame, so moving one makes the picture of it
        # out of date: a developed picture over the old rectangle is a
        # picture of pixels that are no longer in the frame.
        #
        # **A turn is not that.** What is kept is the scan developed in the
        # scan's own coordinates -- not mirrored, not turned, not cut (see
        # `_keep_developed`) -- and standing a frame up is done when it is
        # drawn, by `np.rot90` on the cut and by the canvas's own rotation.
        # The same pixels are in the frame afterwards. Dropping it made `r`
        # cost a whole develop, about a second, to come back to a picture
        # identical to the one thrown away.
        if field != "turn":
            self._forget_developed(self.name())
        self._refresh()
        if field == "turn" and (self.canvas.view or "").startswith("result:"):
            # **Held on `1` or `2`, stay there.** That view draws the frame
            # from the mode's own pixmap and turns it on the canvas
            # (`q.rotate(turn)` in `cut_view`), so repainting is the whole
            # job -- and going back to whatever `v` was on would drop the
            # frame the hand is holding, which reads as the key doing
            # nothing while the number underneath changes.
            self.canvas.update()
        elif field == "turn" and self.colour == "develop":
            # **Kept, but drawn again.** `_paint_developed` transforms
            # nothing: the picture reaches it already cut, straightened,
            # mirrored and turned, because `_developed_cut` bakes all four
            # in. So `update()` repaints the pixmap that was built with the
            # old turn and the frame does not move. The other two ways of
            # looking turn on the canvas (`q.rotate(turn)` in `cut_view`)
            # and need nothing. This is a cut of arrays already in memory,
            # milliseconds, not the develop that was being thrown away.
            self._back_to_view()

    def _drop_cut(self):
        """Leave the cut, and come back to a whole scan that can be placed
        on. The placement mode draws the **autostretch** and not one of the
        three ways of looking: a frame edge on bare film is a 0.24 D step
        against 2.01 D on a masked holder, and normalising each scan to its
        own percentiles is what makes that findable. It is the one thing
        that stretch is for, and the two knobs under the canvas move it.
        """
        # **Borrowed, and given back.** The placement mode needs the
        # autostretch and cannot have the develop, but which way of looking
        # a person was in is theirs: coming out of `e` into the positive,
        # having gone in from the develop, is the window forgetting where
        # you were. `_leave_mode` puts it back.
        self.was_looking = (self.colour, self.cut)
        self.cut = False
        if self.colour == "develop":
            self.colour = "positive"
        self._hold(None, "")
        name = self.name()
        if name and self.here(name):
            pm, scale = self._pixmaps(name, self._frame_box(self.item()))
            self.canvas.pixmap = pm

    def _enter_mode(self):
        # No frames is not a reason to refuse: it is the reason to come in.
        # A scan the detector could not read is exactly the one somebody has
        # to draw a frame on by hand, and `+` is in here.
        if not self.item():
            return
        self._drop_cut()
        if not self.item()["parts"]:
            self.status.setText("nothing found here — + draws one")
        self.before = copy.deepcopy(self.item())
        self.mode = True
        self.canvas.mode = True
        self.stack.setCurrentIndex(1)
        self._refresh()

    # ---- marking dust
    def _enter_dust(self):
        """`s`: the frame developed at full size, to mark its dust by hand.

        **What this is for is ground truth.** A dust finder is judged by
        what it calls dust and what it misses, and both need somebody to
        have looked at the frame and said where the specks are. The canvas
        cannot be where that is said: it is the develop at a quarter of the
        scan, where a speck is one to five pixels and a white pixel of the
        scene is the same size. So this develops the frame as `⌘R` would --
        `develop.develop_frame`, the same function -- and shows it at 100 %.

        **The marks are in scan pixels and live beside the items**, in the
        document's `specks`, by scan name. Scan pixels, because the frame's
        crop, turn and mirror are all things that can change and a speck
        on the film does not move when they do; beside the items, because
        measuring again (`⌘D`, `⇧⌘D`, `detect --force`) rewrites the items
        whole, and the dust on a negative is not something a detector
        found.
        """
        from . import develop
        if self.mode or self.dusting:
            return
        name, part = self.name(), self.part()
        if not name or not part or not self.here(name):
            return self.status.setText("no frame here to mark dust on")
        taste = dict(develop.TASTE) | dict(self.doc["roll"].get("develop") or {})
        stats = develop.stats_of(dict(self.doc["roll"].get("measured") or {}),
                                 taste, self.roll)
        if stats is None:
            return self.status.setText("the roll's base and balance are not "
                                       "measured yet")
        item, prt = copy.deepcopy(self.item()), copy.deepcopy(part)
        mirrored = self.doc["roll"].get("mirrored", True)
        turn = int(prt.get("turn", 0) or 0) % 360
        self.dust = {"name": name, "part": prt["name"], "mirrored": mirrored,
                     "turn": turn, "shape": None,
                     "changed": False, "clean": None, "marks_on": True,
                     "original": False, "cleaned": {}, "focus": None,
                     "aff": np.array(develop.frame_to_scan(item, prt, taste))}
        self.was_looking = (self.colour, self.cut)
        self.dusting = True
        self.stack.setCurrentIndex(2)
        self.canvas.view = "dust"
        self.canvas.pixmap_dev = None
        self.canvas.dust_marks = []
        path, one = self.roll / name, self.doc["roll"].get("one_roll", True)
        # **Kept, so `s` again is at once.** What is shown does not depend
        # on the marks -- they are drawn over it, and `v` cleans the
        # 8-bit picture -- only on everything the develop reads, which is
        # the key: the frame, the roll's settings and measurement.
        key = json.dumps([name, item, prt, taste, self.doc["roll"].get("measured"),
                          one, mirrored, turn], sort_keys=True, default=str)
        hit = self.dust_kept.get(key)
        if hit is not None:
            self.dust_kept[key] = self.dust_kept.pop(key)    # most recent last
            self.dust["key"] = key
            return self._dust_ready(("dust", name, prt["name"]), *hit)
        self.dust["key"] = key
        self.canvas.label = "Developing the frame at full size to mark dust…"
        self.canvas.update()
        self.status.setText("developing at full size…")

        def job():
            out, ev = develop.develop_frame(path, item, prt, stats, taste, one)
            shape = out.shape[:2]
            if mirrored:
                out = out[:, ::-1]
            if turn:
                out = np.rot90(out, k=-(turn // 90))
            return shape, develop.encode(out, 8), ev

        self.dust_worker = _Render(("dust", name, prt["name"]), job)
        self.dust_worker.done.connect(self._dust_ready)
        self.dust_worker.failed.connect(self._dust_failed)
        self.dust_worker.start()

    # Frames kept developed for `s`: 72 MB each at 24 MP, and going back
    # and forth between a few is what marking a roll is.
    DUST_KEEP = 4

    def _dust_ready(self, key, shape, shown, _evidence):
        w, self.dust_worker = self.dust_worker, None
        if w is not None:
            w.wait()
        if not self.dusting or key != ("dust", self.dust["name"], self.dust["part"]):
            return
        self.dust["shape"] = shape
        self.dust["shown"] = shown
        kept = self.dust_kept
        was = kept.pop(self.dust["key"], None)
        kept[self.dust["key"]] = (shape, shown, _evidence)
        while len(kept) > self.DUST_KEEP:
            kept.pop(next(iter(kept)))
        # and the frame with its dust out, as it was left: what changed
        # since is cleaned again round each mark (`_dust_clean_view`)
        out = self.dust_kept_clean.pop(self.dust["key"], None)
        if out is not None and was is not None:
            self.dust["clean"], self.dust["cleaned"] = out
        for k in list(self.dust_kept_clean):
            if k not in kept:
                del self.dust_kept_clean[k]
        rec = self._dust_record(self.dust["name"])
        if rec and self._dust_number(rec):
            self.dirty = True
        self.dust["before"] = copy.deepcopy(rec or None)
        self.canvas.pixmap_dev = _pix8(shown)
        self.canvas.label = "mark dust · frame %s" % self.dust["part"]
        self.canvas.cut_zoom, self.canvas.cut_pan = 1.0, QPointF(0, 0)
        self._dust_draw()

    def _dust_failed(self, key, why):
        w, self.dust_worker = self.dust_worker, None
        if w is not None:
            w.wait()
        if self.dusting:
            self.canvas.label = ""
            self.status.setText(f"not developed: {why}")

    def _leave_dust(self):
        if not self.dusting:
            return
        w, self.dust_worker = self.dust_worker, None
        if w is not None:
            # nothing to be told; let finish and dropped rather than
            # killed mid LibRaw read
            w.done.disconnect()
            w.failed.disconnect()
            w.wait()
        name = self.dust["name"]
        if self.dust.get("clean") is not None and self.dust.get("key"):
            self.dust_kept_clean[self.dust["key"]] = (self.dust["clean"],
                                                      self.dust["cleaned"])
        n = len(self._dust_record(name).get("marks", []))
        changed = self.dust["changed"]
        self.dusting = False
        self.dust = None
        self.stack.setCurrentIndex(0)
        self.canvas.view = None
        self.canvas.pixmap_dev = None
        self.canvas.dust_marks = []
        self.canvas.cut_zoom, self.canvas.cut_pan = 1.0, QPointF(0, 0)
        if self.was_looking is not None:
            self.colour, self.cut = self.was_looking
            self.was_looking = None
        if changed:
            # the develop takes the marked dust out, so what it drew of
            # this scan is of other marks now
            self._forget_developed(name)
        self._back_to_view()
        self.status.setText(f"{n} speck{'s' if n != 1 else ''} marked on this "
                            "scan, taken out of the develop")

    # The three coordinates a mark crosses: the picture on the canvas (cut,
    # straightened, mirrored, turned), the frame as `develop_frame` makes it
    # (cut and straightened only), and the scan.
    def _dust_frame(self, x, y):
        """Picture on the canvas -> frame before mirroring and turning."""
        h, w = self.dust["shape"]
        dims = [(w, h)]
        for _ in range(self.dust["turn"] // 90):
            dims.append((dims[-1][1], dims[-1][0]))
        for qw, qh in reversed(dims[:-1]):
            # undo one quarter clockwise, which took (x, y) to (qh - 1 - y, x)
            x, y = y, qh - 1 - x
        if self.dust["mirrored"]:
            x = w - 1 - x
        return x, y

    def _dust_shown(self, x, y):
        """Frame before mirroring and turning -> picture on the canvas."""
        h, w = self.dust["shape"]
        if self.dust["mirrored"]:
            x = w - 1 - x
        for _ in range(self.dust["turn"] // 90):
            x, y = h - 1 - y, x
            w, h = h, w
        return x, y

    def _dust_scan(self, x, y):
        a = self.dust["aff"]
        return (a[0, 0] * x + a[0, 1] * y + a[0, 2],
                a[1, 0] * x + a[1, 1] * y + a[1, 2])

    def _dust_from_scan(self, sx, sy):
        a = self.dust["aff"]
        m = np.linalg.inv(a[:, :2])
        x, y = m @ np.array([sx - a[0, 2], sy - a[1, 2]])
        return float(x), float(y)

    def _dust_record(self, name):
        return self.doc.get("specks", {}).get(name, {})

    def _dust_draw(self):
        """The scan's marks, on the picture; and how far the sweep has got."""
        d = self.dust
        if d["shape"] is None:
            return
        # **The dust goes as it is marked**: the picture is always the
        # frame with the marked dust taken out, and `hold c` the frame as
        # it was, to compare
        if d["original"]:
            if d.get("shown_pm") is None:
                d["shown_pm"] = _pix8(d["shown"])
            self.canvas.pixmap_dev = d["shown_pm"]
            self.canvas.label = "as it was, with its dust · frame %s" % d["part"]
        else:
            self.canvas.pixmap_dev = self._dust_clean_view()
            self.canvas.label = "mark dust · frame %s" % d["part"]
        pm = self.canvas.pixmap_dev
        marks = []
        count = {"hand": 0, "accepted": 0, "proposed": 0, "refused": 0}
        for m in self._dust_record(d["name"]).get("marks", []):
            style = "hand" if m.get("by", "hand") == "hand" else m.get("state", "proposed")
            count[style] = count.get(style, 0) + 1
            x, y = self._dust_shown(*self._dust_from_scan(m["x"], m["y"]))
            path = [self._dust_shown(*self._dust_from_scan(px, py))
                    for px, py in m["path"]] if m.get("path") else None
            src = (self._dust_shown(*self._dust_from_scan(
                m["x"] + m["from"][0], m["y"] + m["from"][1]))
                if m.get("from") else None)
            if path or -20 <= x <= pm.width() + 20 and -20 <= y <= pm.height() + 20:
                marks.append((x, y, style, m is d["focus"], m.get("r"), path, src))
        # `v` hides them, to judge what is left; the ring and the number
        # still say what is under the pointer, so marking goes on
        # nor while ⌘ is choosing where to clone from: the mark would
        # cover what it is about to become
        self.canvas.dust_marks = (marks if d["marks_on"] and not d.get("cloning")
                                  else [])
        n = count["hand"]
        said = f"{n} marked by hand"
        if count["accepted"] + count["proposed"] + count["refused"]:
            said += (f" · found: {count['accepted']} accepted, "
                     f"{count['proposed']} undecided, {count['refused']} refused")
        self.status.setText(said)
        self.canvas.update()

    def _dust_click(self, pos):
        d = self.dust
        if not self.dusting or d["shape"] is None:
            return
        x, y = self.canvas.dust_at(pos)
        pm = self.canvas.pixmap_dev
        if not (0 <= x < pm.width() and 0 <= y < pm.height()):
            return
        rec = self.doc.setdefault("specks", {}).setdefault(d["name"], {})
        marks = rec.setdefault("marks", [])
        # the nearest mark under the pointer goes, in screen pixels: what a
        # click can hit is what is drawn, whatever the zoom
        m = self._dust_hit(pos)
        hit = None if m is None else (0, next(i for i, t in enumerate(marks) if t is m))
        if hit is not None and marks[hit[1]].get("by", "hand") != "hand":
            # a found one goes round its three states; it is never removed
            # by a click, because a refusal is worth keeping: it is what
            # says the finder was wrong there
            m = marks[hit[1]]
            m["state"] = {"proposed": "accepted", "accepted": "refused"}.get(
                m.get("state"), "proposed")
            d["focus"] = m
        elif hit is not None:
            marks.pop(hit[1])
        else:
            # a dab of the brush: a stroke one point long, filled as it is
            # painted and resizable after. The old click -- a point grown
            # into the speck under it -- is still read from documents that
            # have it, and no longer made.
            from . import dust
            sx, sy = self._dust_scan(*self._dust_frame(x, y))
            sx, sy = round(float(sx), 1), round(float(sy), 1)
            r = float(self.canvas.dust_brush or dust.STROKE_R)
            marks.append({"x": sx, "y": sy, "path": [[sx, sy]],
                          "r": round(r, 1), "by": "hand"})
            self._dust_number(rec)
        d["changed"] = True
        self._dust_undoable()
        self.dirty = True
        self._dust_draw()

    def _dust_stroke(self, pts):
        """A drag in the mode: a stroke along a scratch or a hair, stored as
        its path in scan pixels and the brush's radius (`dust.samples`).

        Kept a point every half radius or so: a drag reports a point every
        couple of screen pixels, and at the whole frame that is tens of
        scan pixels apart, at 100 % one -- the document wants the line, not
        the mouse's sampling of it."""
        from . import dust
        d = self.dust
        if not self.dusting or d["shape"] is None:
            return
        pm = self.canvas.pixmap_dev
        r = float(self.canvas.dust_brush or dust.STROKE_R)
        path = []
        for pos in pts:
            x, y = self.canvas.dust_at(pos)
            if not (0 <= x < pm.width() and 0 <= y < pm.height()):
                continue
            sx, sy = self._dust_scan(*self._dust_frame(x, y))
            if not path or math.hypot(sx - path[-1][0], sy - path[-1][1]) >= max(1.0, r / 2):
                path.append([round(float(sx), 1), round(float(sy), 1)])
        if len(path) < 2:
            return
        rec = self.doc.setdefault("specks", {}).setdefault(d["name"], {})
        mid = path[len(path) // 2]
        rec.setdefault("marks", []).append(
            {"x": mid[0], "y": mid[1], "path": path, "r": round(r, 1), "by": "hand"})
        self._dust_number(rec)
        d["changed"] = True
        self._dust_undoable()
        self.dirty = True
        self._dust_draw()

    def _dust_undoable(self, resize=None):
        """After a change to the marks: what they were before it, onto the
        undo list, so `⌘Z` takes a mark back -- or puts back one taken away.

        The state before is kept from the last change (`dust["before"]`)
        rather than taken at each of the six places that change marks.
        Notches of the wheel on one mark, close together, are one change:
        undoing a resize a notch at a time would be twenty `⌘Z`."""
        d = self.dust
        now = time.monotonic()
        same = (resize is not None and d.get("resizing") is resize
                and now - d.get("resized", 0) < 1.5 and self.undo
                and self.undo[-1][0] == self.DUST)
        if not same:
            self.undo.append((self.DUST, (d["name"], d.get("before"))))
            del self.undo[:-60]
            self.redo.clear()
        d["resizing"], d["resized"] = resize, now
        d["before"] = copy.deepcopy(self._dust_record(d["name"]) or None)

    def _set_dust(self, name, rec, said):
        """`⌘Z` / `⇧⌘Z` on a dust entry: the scan's marks as `rec`."""
        specks = self.doc.setdefault("specks", {})
        if rec is None:
            specks.pop(name, None)
        else:
            specks[name] = copy.deepcopy(rec)
        self.dirty = True
        d = self.dust
        if self.dusting and d["name"] == name:
            d["before"] = copy.deepcopy(rec)
            d["focus"], d["changed"], d["resizing"] = None, True, None
            self._dust_draw()
            n = sum(1 for m in (rec or {}).get("marks", [])
                    if m.get("by", "hand") == "hand")
            self.status.setText(f"{said} · {n} marked by hand")
        else:
            self._forget_developed(name)
            self.status.setText(f"{said} on {name}")

    # How far a ⌘ + drag must go to be a drag and not a click: screen px.
    CLONE_MOVE = 4

    def _dust_clone_press(self, pos):
        """`⌘` + press on a mark: what it is cloned from is about to move
        with the pointer (`_dust_clone_drag`), and the marks hide so the
        picture under them shows. **For where the automatic fill gets it
        wrong**: it takes the patch whose ring best matches the hole's,
        which on an edge of the scene can be the wrong side of it, and a
        person sees the right place at once."""
        d = self.dust
        if not self.dusting or d["shape"] is None:
            return
        m = self._dust_hit(pos)
        if m is None or m.get("by", "hand") != "hand" or not m.get("path"):
            return self.status.setText("⌘ + drag on a mark to choose what it "
                                       "is cloned from")
        d["cloning"] = {"mark": m, "start": pos, "moved": False,
                        "from": list(m.get("from") or (0.0, 0.0)),
                        "at": self._dust_scan(*self._dust_frame(
                            *self.canvas.dust_at(pos)))}
        self._dust_draw()
        self._dust_clone_preview(m, d["cloning"]["from"])

    def _dust_clone_off(self, pos):
        """The way to the source for the pointer at `pos`: where it was at
        the press, less how far the pointer has gone -- **what is cloned
        follows the hand**, as if the picture were slid under the mark."""
        c = self.dust["cloning"]
        sx, sy = self._dust_scan(*self._dust_frame(*self.canvas.dust_at(pos)))
        return [round(float(c["from"][0] - (sx - c["at"][0])), 1),
                round(float(c["from"][1] - (sy - c["at"][1])), 1)]

    def _dust_clone_drag(self, pos):
        d = self.dust
        c = d.get("cloning") if self.dusting and d else None
        if not c:
            return
        if (pos - c["start"]).manhattanLength() >= self.CLONE_MOVE:
            c["moved"] = True
        self._dust_clone_preview(c["mark"], self._dust_clone_off(pos))

    def _dust_clone_cancel(self):
        d = self.dust
        if not self.dusting or not d.get("cloning"):
            return
        d["cloning"] = None
        self.canvas.dust_overlay = self.canvas.dust_source = None
        self._dust_draw()

    def _dust_clone_preview(self, m, off):
        """Mark `m` as it would come out cloned from `off` (scan pixels):
        copied and healed on the picture on screen, the same `dust.clone`
        the develop runs, and laid over it."""
        from . import dust
        d = self.dust
        to_shown = lambda x, y: self._dust_shown(*self._dust_from_scan(x, y))
        mx, my = to_shown(m["x"], m["y"])
        fx, fy = to_shown(m["x"] + off[0], m["y"] + off[1])
        self.canvas.dust_source = ((mx, my), (fx, fy))
        base = d["shown"] if d["original"] else d.get("clean")
        if base is None:
            return self.canvas.update()
        H, W = base.shape[:2]
        pts = [(*to_shown(x, y), r) for x, y, r, _ in dust.samples(dict(m, **{"from": None}))]
        rad = max(r for *_, r in pts) + 8
        x0 = max(0, int(min(p[0] for p in pts) - rad))
        y0 = max(0, int(min(p[1] for p in pts) - rad))
        x1 = min(W, int(max(p[0] for p in pts) + rad) + 1)
        y1 = min(H, int(max(p[1] for p in pts) + rad) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        h = np.zeros((H, W), bool)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        for x, y, r, *_ in pts:
            h[y0:y1, x0:x1] |= (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        # on a copy of the box and the source's box only: the whole frame
        # per movement of the mouse would be 70 MB a time. From the
        # picture without this mark's own fill, which is what it replaces.
        dx, dy = int(round(fx - mx)), int(round(fy - my))
        X0, Y0 = max(0, min(x0, x0 + dx) - 8), max(0, min(y0, y0 + dy) - 8)
        X1, Y1 = min(W, max(x1, x1 + dx) + 8), min(H, max(y1, y1 + dy) + 8)
        work = base[Y0:Y1, X0:X1].astype(np.float32) / 255.0
        dust.clone(work, h[Y0:Y1, X0:X1], dy, dx)
        out = np.clip(work[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] * 255.0 + 0.5,
                      0, 255).astype(np.uint8)
        self.canvas.dust_overlay = (_pix8(np.ascontiguousarray(out)), x0, y0)
        self.canvas.update()

    def _dust_clone_release(self, pos):
        """The drag let go: the mark keeps the source it shows. A click
        that did not move takes its source away, and the automatic fill
        is back."""
        d = self.dust
        c = d.get("cloning") if self.dusting and d else None
        if not c:
            return
        m = c["mark"]
        off = self._dust_clone_off(pos)
        self._dust_clone_cancel()
        if not c["moved"]:
            if not m.get("from"):
                return
            m.pop("from")
            said = "filled by the automatic again"
        else:
            m["from"] = off
            said = (f"cloned from {abs(off[0]):.0f}, {abs(off[1]):.0f} px away "
                    "· ⌘ + click it for the automatic fill again")
        d["changed"] = True
        self._dust_undoable()
        self.dirty = True
        self._dust_draw()
        self.status.setText(f"#{m.get('id', '?')} {said}")

    def _dust_see(self):
        """`v`: the marks drawn over the picture, or not. The picture has
        the dust taken out either way (`_dust_draw`), so with them hidden
        what is left of the dust is what shows -- and marking goes on."""
        d = self.dust
        if not self.dusting or d["shape"] is None:
            return
        d["marks_on"] = not d["marks_on"]
        self._dust_draw()

    def _dust_original(self, on):
        """`hold c`: the frame as it was, with all its dust, while held."""
        d = self.dust
        if not self.dusting or d["shape"] is None or d["original"] == on:
            return
        d["original"] = on
        self._dust_draw()

    # Round what is cleaned again, the picture it may read: as far as any
    # piece looks for its patch (`dust.COPY_REACH` pieces of `dust.PIECE`,
    # and the 6 px of room `dust` gives it) and a margin. With that, and
    # every mark it reaches taken whole, the box comes out as the whole
    # frame would (`_dust_clean_view`).
    SEE_PAD = int(3 * 32 + 6 + 8)

    def _dust_clean_view(self):
        """The 8-bit picture on screen with the dust that is used taken
        out, kept up to date with the marks. The develop does the same to
        its linear picture before the automatics (`develop._cleaned`); here
        it is done to what is being judged, which costs a fraction of a
        second and not a develop.

        **Only round what changed.** Each used mark is remembered as it was
        cleaned; a mark added, removed, moved or resized has the box round
        it, before and after, cleaned again from the picture with every
        mark that reaches into it, and pasted back. More than a few at once
        -- `⇧A`, finding again -- and the whole frame is done again."""
        import json
        from . import dust
        d = self.dust
        to_shown = lambda x, y: self._dust_shown(*self._dust_from_scan(x, y))
        used = [m for m in self._dust_record(d["name"]).get("marks", [])
                if dust.used(m)]
        now = {str(m.get("id", i)): json.dumps(m, sort_keys=True)
               for i, m in enumerate(used)}
        was = d["cleaned"]
        if d["clean"] is not None and now == was:
            if d.get("clean_pm") is None:
                d["clean_pm"] = _pix8(d["clean"])
            return d["clean_pm"]
        shown = d["shown"]
        H, W = shown.shape[:2]

        def box(mark):
            pts = [(*to_shown(x, y), r) for x, y, r, _ in dust.samples(mark)]
            if mark.get("from"):
                # and where it is cloned from, which is read
                fx, fy = mark["from"]
                pts += [(*to_shown(x + fx, y + fy), r)
                        for x, y, r, _ in dust.samples(mark)]
            rad = max((r or 40.0) for *_, r in pts) + 8
            return (min(p[0] for p in pts) - rad, min(p[1] for p in pts) - rad,
                    max(p[0] for p in pts) + rad, max(p[1] for p in pts) + rad)

        def clean(x0, y0, x1, y1):
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(W, int(math.ceil(x1))), min(H, int(math.ceil(y1)))
            img = shown[y0:y1, x0:x1].astype(np.float32) / 255.0
            pts = dust.points(used, lambda x, y: tuple(
                np.subtract(to_shown(x, y), (x0, y0))))
            return x0, y0, np.clip(dust.clean(img, pts, 1.0) * 255.0 + 0.5,
                                   0, 255).astype(np.uint8)

        touched = set(now) ^ set(was) | {k for k in now if k in was and now[k] != was[k]}
        if d["clean"] is None or len(touched) > 4:
            d["clean"] = clean(0, 0, W, H)[2]
        else:
            # **The same picture the whole frame gives, whatever order the
            # marks were made in.** What changed, then every mark whose box
            # touches it, then every mark touching those: marks that overlap
            # are one hole, filled together, and a box that cut one of them
            # filled a different hole -- 18 000 px, up to 57 levels, after a
            # long stroke. Marks further off only matter as where no patch
            # may come from, and the `SEE_PAD` round the box reads them.
            boxes = {str(m.get("id", i)): box(m) for i, m in enumerate(used)}
            seeds = [boxes[k] for k in touched if k in boxes]
            seeds += [box(json.loads(was[k])) for k in touched if k in was]
            P = self.SEE_PAD
            T = dust.CLOSE + 1           # the closing joins holes this close
            out = d["clean"].copy()
            while seeds:
                r = list(seeds.pop())
                grown = True
                while grown:
                    grown = False
                    for k, (bx0, by0, bx1, by1) in list(boxes.items()):
                        if (bx0 < r[2] + T and bx1 > r[0] - T
                                and by0 < r[3] + T and by1 > r[1] - T):
                            r = [min(r[0], bx0), min(r[1], by0),
                                 max(r[2], bx1), max(r[3], by1)]
                            del boxes[k]
                            grown = True
                # a seed this one swallowed is done with it
                seeds = [b for b in seeds if not (b[0] >= r[0] and b[1] >= r[1]
                                                  and b[2] <= r[2] and b[3] <= r[3])]
                x0, y0, part = clean(r[0] - P, r[1] - P, r[2] + P, r[3] + P)
                px0, py0 = max(0, int(r[0])), max(0, int(r[1]))
                px1, py1 = min(W, int(math.ceil(r[2]))), min(H, int(math.ceil(r[3])))
                if px1 > px0 and py1 > py0:
                    out[py0:py1, px0:px1] = part[py0 - y0:py1 - y0, px0 - x0:px1 - x0]
            d["clean"] = out
        d["cleaned"] = now
        d["clean_pm"] = _pix8(d["clean"])
        return d["clean_pm"]

    BRUSH_STEP = 1.15    # per notch of the wheel
    BRUSH_MIN, BRUSH_MAX = 2.0, 300.0   # picture px, which are scan px

    def _dust_resize(self, pos, steps):
        """`⇧` + the wheel: the size of the mark under the pointer, or, with
        none there, of the brush a click dabs and a drag paints with.

        **A size is a radius in scan pixels and it is exact**: the develop
        fills what was painted and does not look for the speck's own extent
        (`dust.clean`). A mark from before the brush, a point with no size,
        is grown from the click instead -- right for most and short for
        some, a white fleck on a brown wall on DSC_0238 kept a rim -- and
        shrunk below `BRUSH_MIN` it goes back to having none. A dab or a
        stroke always has a width, and so does the brush."""
        d = self.dust
        if not self.dusting or d["shape"] is None:
            return
        f = self.BRUSH_STEP ** steps
        hit = self._dust_hit(pos)
        if hit is not None:
            m = hit
            if not m.get("r") and steps < 0:
                return                  # already as small as it goes
            r = (m.get("r") or 8.0) * f
            if m.get("path") and r < self.BRUSH_MIN:
                return                  # a stroke always has a width
            if r < self.BRUSH_MIN:
                m.pop("r", None)
            else:
                m["r"] = round(min(r, self.BRUSH_MAX), 1)
            from . import dust
            if dust.used(m):
                d["changed"] = True
            self._dust_undoable(resize=m)
            self.dirty = True
            self._dust_draw()
            self.status.setText(self.status.text() + " · this one: "
                                + (f"{m['r']:.0f} px" if m.get("r") else "found under the click"))
        else:
            from . import dust as dust_
            r = (self.canvas.dust_brush or dust_.STROKE_R) * f
            self.canvas.dust_brush = min(max(r, self.BRUSH_MIN), self.BRUSH_MAX)
            self.status.setText(f"brush: {self.canvas.dust_brush:.0f} px radius")
            self.canvas.update()

    def _dust_number(self, rec):
        """Every mark on the scan gets a number, once, and keeps it.

        **To be able to say which one.** A mark is a point on a 20 MP frame
        and the list's order moves every time one is removed, so the only
        way to talk about one -- "number 17 looks wrong" -- is a number
        that is the mark's own: the next one up, never reused while the
        scan has marks. Given on entering the mode and to every new mark."""
        marks = rec.get("marks", [])
        top = max((m.get("id", 0) for m in marks), default=0)
        new = False
        for m in marks:
            if not m.get("id"):
                top += 1
                m["id"] = top
                new = True
        return new

    def _dust_named(self, pos):
        """`#id`, and what it is, for the mark under the screen point."""
        if not self.dusting or self.dust["shape"] is None:
            return None
        m = self._dust_hit(pos)
        if m is None:
            return None
        what = ("stroke" if len(m.get("path") or ()) > 1 else
                "dab" if m.get("path") else "hand") if m.get("by", "hand") == "hand" else \
            f"found {m.get('kind', '')}, {m.get('state', 'proposed')}"
        size = f", r {m['r']:.0f} px" if m.get("r") else ""
        if m.get("from"):
            size += ", cloned"
        return f"#{m.get('id', '?')} · {what}{size}"

    def _dust_hit(self, pos):
        """The mark under the screen point `pos`, by what is drawn."""
        best = None
        k = self.canvas.dust_scale()[0]
        for m in self._dust_record(self.dust["name"]).get("marks", []):
            if m.get("path"):
                ps = [self.canvas.dust_screen(*self._dust_shown(
                    *self._dust_from_scan(px, py))) for px, py in m["path"]]
                dd = (math.hypot(ps[0].x() - pos.x(), ps[0].y() - pos.y())
                      if len(ps) == 1 else
                      min(_to_segment(pos, a, b) for a, b in zip(ps, ps[1:])))
            else:
                p = self.canvas.dust_screen(*self._dust_shown(
                    *self._dust_from_scan(m["x"], m["y"])))
                dd = math.hypot(p.x() - pos.x(), p.y() - pos.y())
            reach = max(6.0, m["r"] * k + 1.5) if m.get("r") else max(11.0, 8 * k)
            if dd <= reach and (best is None or dd < best[0]):
                best = (dd, m)
        return best[1] if best else None

    def _dust_one_to_one(self):
        """Screen pixels per picture pixel at 100 %: one picture pixel to a
        pixel of the display, not to a point of the layout."""
        return 1.0 / max(self.canvas.devicePixelRatioF(), 1.0)

    def _dust_zoom(self):
        """100 % about the middle of the window, or back to the whole frame."""
        if not self.dusting or self.dust["shape"] is None:
            return
        k = self._dust_one_to_one()
        if abs(self.canvas.dust_scale()[0] - k) < 1e-6:
            return self._dust_fit()
        x, y = self.canvas.dust_at(QPointF(self.canvas.width() / 2,
                                           self.canvas.height() / 2))
        self.canvas.dust_show(x, y, k)

    def _dust_fit(self):
        self.canvas.cut_zoom, self.canvas.cut_pan = 1.0, QPointF(0, 0)
        self.canvas.update()

    def _dust_key(self, k, mods):
        if k in (Qt.Key_Escape, Qt.Key_Return, Qt.Key_Enter, Qt.Key_S):
            return self._leave_dust()
        if k == Qt.Key_V:
            return self._dust_see()
        if k == Qt.Key_C:
            return self._dust_original(True)
        if k == Qt.Key_Z:
            return self._dust_zoom()
        if k == Qt.Key_F:
            return self._dust_fit()

    def _square_up(self):
        """Put the frames back on the grid the film puts them on.

        The gate does not move and the film advances past it, so the frames of
        one scan share a size and a line across the strip; only how far along
        they sit differs. Inside the mode nothing is imposed -- what is on
        screen is where you put your corners -- and this is what happens on the
        way out, which is what the mode's `↩` means.

        It is the same regularisation as fitting a rectangle to four placed
        corners, one storey up: four points become a rectangle, several
        rectangles become a row. It also measures the tilt about 20 % better
        than fitting each frame on its own, because every frame's corners are
        evidence for the one angle.
        """
        it = self.item()
        if not it or len(it["parts"]) < 2:
            return
        quads = [q.get("placed") or findframe.corners(
            q["centre"][0], q["centre"][1], q["size"][0], q["size"][1],
            it["angle"]) for q in it["parts"]]
        angle, rects = findframe.fit_row(quads, near=it["angle"])
        worst = 0.0
        for q, (cx, cy, w, h) in zip(it["parts"], rects):
            worst = max(worst, abs(cx - q["centre"][0]), abs(cy - q["centre"][1]),
                        abs(w - q["size"][0]), abs(h - q["size"][1]))
            q["centre"] = [round(cx, 1), round(cy, 1)]
            q["size"] = [round(w, 1), round(h, 1)]
        it["angle"] = round(angle, 3)
        self.canvas.angle = it["angle"]
        if worst > 0.5:
            self.status.setText(f"squared to the strip, up to {worst:.0f} px")

    def _leave_mode(self, apply):
        if self.before is not None:
            if apply:
                self._square_up()
                # The state as it was on *entering* is what undo has to put
                # back. Snapshotting on the way out would push what the mode
                # had already changed, and the undo would be a no-op.
                self.undo.append((self.name(), self.before))
                self.redo.clear()
                del self.undo[:-60]
            else:
                # `esc` really cancels: it puts back the state as it was on
                # entering *and* leaves nothing on the undo stack, because an
                # undo that walks back into a cancelled session is putting
                # back something you just said you did not want.
                self.doc["items"][self.name()] = self.before
        self.before = None
        self.mode = self.canvas.mode = False
        self.stack.setCurrentIndex(0)
        # Back to the way of looking that was on when `e` was pressed,
        # whether it was applied or cancelled: the view is not part of the
        # edit.
        if self.was_looking is not None:
            self.colour, self.cut = self.was_looking
            self.was_looking = None
        # The autostretch belongs to placing; on the way out the canvas goes
        # back to whichever of the three ways of looking is on.
        self._forget_modes(self.name())
        # Only on the way out by `esc`: applying may have left a message of its
        # own about squaring the frames up, and clearing it here threw away the
        # one report of the one thing leaving the mode does on its own.
        if not apply:
            self.status.setText("cancelled")
        said = self.status.text()
        self._select(self.list.currentRow())
        # `_select` clears the status while it reads the file. Whatever leaving
        # the mode had to report is about what just happened, not about the
        # reading, so it goes back afterwards.
        self.status.setText(said)

    def _add_part(self):
        """A new part, beside the selected one, for a frame the detector
        missed. Its size and angle are the selected one's, because on a given
        scan the frames are the same size and sit on the same strip -- that is
        what is known about the missing one without measuring it.

        It goes on **whichever side has room for it**. Always offsetting the
        same way puts it off the edge of the scan half the time, and the case
        this command exists for is exactly the one where the found frame is
        the far one: SDIM0897 finds the right-hand half and the missing one is
        to its left.
        """
        it = self.item()
        if not it:
            return
        self._snapshot()
        src = self.part()
        if src:
            import math
            new = copy.deepcopy(src)
            gap = src["size"][0] * 1.06
            room_right = it["scan"][0] - (src["centre"][0] + src["size"][0] / 2)
            room_left = src["centre"][0] - src["size"][0] / 2
            side = 1 if room_right >= room_left else -1
            # Along the **film**, not along the scan. The strip runs at the
            # item's angle, so stepping in x and keeping y puts the twin off
            # the strip by `gap * tan(angle)` -- 49 px at 1.04 degrees on these
            # scans, which is exactly the misalignment that showed up on the
            # one frame added by hand and on none of the ones detected.
            r = math.radians(it.get("angle", 0.0))
            new["centre"] = [round(src["centre"][0] + side * gap * math.cos(r), 1),
                             round(src["centre"][1] + side * gap * math.sin(r), 1)]
        else:
            # From what the roll says a frame is, not from a fraction of the
            # scan. Every shot on a roll came off one copy stand and one
            # camera, so the median of the ones already found is the best
            # guess there is for one nobody has found yet.
            w, h = it["scan"]
            size, fmt = self._roll_frame()
            new = {"centre": [w / 2, h / 2],
                   "size": size or [w * 0.5, h * 0.5],
                   "format": fmt or "full", "evidence": {}}
            if not it.get("manual") and self._roll_angle() is not None:
                it["angle"] = self._roll_angle()
        new["evidence"] = {"set_by": "hand"}
        # The first letter not in use, rather than "however many there are
        # now": names are identifiers, and one that shifts when a neighbour
        # goes is one that yesterday's output filename no longer matches.
        used = {q["name"] for q in it["parts"]}
        new["name"] = next(c for c in "abcdefgh" if c not in used)
        it["parts"].append(new)
        it["parts"].sort(key=lambda q: q["name"])
        self.canvas.sel = it["parts"].index(new)
        self.canvas.vertex = 0
        self._edited("added", new["name"])

    def _copy_frames(self):
        """Take this scan's whole layout, to put on another.

        Every frame of it, not one: the sizes, where they sit, which way up
        each is, and the film's tilt with them. That is the unit because it is
        what a scan *is* -- a half-frame shot is a pair with a gap, and half a
        pair is not an answer to anything.

        On one roll it is nearly a complete answer for every other scan, which
        is the point: the copy stand did not move between shots and the gate
        did not move at all.
        """
        it = self.item()
        if not it or not it["parts"]:
            return self.status.setText("no frames here to copy")
        self.clipboard = {"angle": it["angle"],
                          "parts": copy.deepcopy(it["parts"])}
        n = len(it["parts"])
        self.status.setText(
            f"copied {n} frame{'s' if n > 1 else ''} from {self.name()}")

    def _paste_frames(self):
        """Put a copied layout on this scan, in place of whatever is here.

        It replaces rather than adds: what was copied is a whole answer, and
        merging it with a different whole answer would give a scan more frames
        than the film has. The tilt comes with the frames, because a layout
        without the angle it was measured at is not a layout.

        The list is emptied and refilled rather than swapped, so the canvas --
        which holds that list -- keeps pointing at the one everybody is
        looking at.
        """
        if not getattr(self, "clipboard", None):
            return self.status.setText("nothing copied")
        it = self.item()
        if not it:
            return
        self._snapshot()
        fresh = copy.deepcopy(self.clipboard["parts"])
        for q in fresh:
            q.pop("placed", None)          # somebody else's corners, not yours
            q["evidence"] = {"set_by": "hand"}
        it["parts"][:] = fresh
        it["angle"] = self.clipboard["angle"]
        it["state"] = "manual"
        document.order_parts(it, self.doc["roll"].get("mirrored", True))
        self.canvas.angle = it["angle"]
        self.canvas.sel = self.canvas.vertex = 0
        n = len(fresh)
        self.status.setText(f"pasted {n} frame{'s' if n > 1 else ''} here")
        self._edited("pasted", part=False)

    def _roll_frame(self):
        """(size, format) of the median frame on this roll, or (None, None).

        Half of a roll's scans are enough to say what a frame on it looks
        like: they went through one gate on one copy stand within minutes.
        """
        seen = [(q["format"], q["size"]) for it in self.doc["items"].values()
                for q in it["parts"]]
        if not seen:
            return None, None
        # The format the roll mostly is, and the median size **within it**. A
        # median over every frame of a roll that holds two formats is the size
        # of neither of them, and a roll is usually one film in one camera.
        counts = {}
        for fmt, _ in seen:
            counts[fmt] = counts.get(fmt, 0) + 1
        fmt = max(counts, key=counts.get)
        sizes = sorted((s for f, s in seen if f == fmt),
                       key=lambda s: s[0] * s[1])
        return list(sizes[len(sizes) // 2]), fmt

    def _roll_angle(self):
        """The median tilt of the roll: what the film was lying at that day."""
        tilts = sorted(it["angle"] for it in self.doc["items"].values()
                       if it["parts"])
        return tilts[len(tilts) // 2] if tilts else None

    def _drop_part(self):
        it, p = self.item(), self.part()
        if not it or not p:
            return
        self._snapshot()
        gone = p["name"]
        it["parts"].remove(p)
        # The survivors keep their names. Renaming them here would mean:
        # drop `a` and yesterday's
        # DSC_1234-b.dng becomes today's -a.dng, for a file nobody touched.
        self.canvas.sel = max(0, min(self.canvas.sel, len(it["parts"]) - 1))
        self.canvas.vertex = 0
        self._edited("dropped", gone)

    def _picked(self, i):
        """The pointer chose a frame by touching it or one of its corners."""
        if self.mode:
            self.status.setText(self.canvas.corner_name())
        self._refresh()
        # **And draw the picture the choice asks for.** `_refresh` repaints
        # the pixmap that is already there; it does not build a new one. In
        # the develop that left the canvas showing the composite made for
        # the previous frame -- the surround is the selected frame's develop
        # of the whole scan, so on half frame the rebate and the gap kept
        # the other half's exposure and curve (0.25 EV apart on SDIM0903 of
        # roma-half, 10 of 255 across the backdrop). `n` and the arrows
        # already did this; only the pointer did not. Not in placement mode,
        # where a click is a corner being dragged and the develop is not
        # what is on screen.
        if not self.mode:
            self._back_to_view()

    def _turned(self, angle):
        """The fit found a new tilt for the film. It is the item's, so it
        moves both frames of a half-frame pair together."""
        it = self.item()
        if it is None or it.get("angle") == angle:
            return
        it["angle"] = angle
        self.dirty = True

    def _shown(self):
        """The name of the frame the canvas is showing.

        **A turn acts on what you are looking at.** There were two answers
        to that and they disagreed: the arrows, handled inside the held
        `result:` view, took the frame whose letter was being held, while
        `r` and `u` fell through to the editor's own keys and took the
        *selected* part -- which on half frame is the other half as often
        as not. Held on `2`, `r` turned frame a. One answer, in one place,
        and everything that turns asks it.
        """
        view = self.canvas.view or ""
        if view.startswith("result:"):
            return view.split(":", 1)[1]
        return (self.part() or {}).get("name")

    def _turn_left(self):
        return self._turn_part(-90, self._shown())

    def _turn_right(self):
        return self._turn_part(90, self._shown())



    def _turn_part(self, by, name=None, absolute=False):
        """How the camera was held for this exposure.

        A fact about the **frame**, not about the film, and so per part: the
        two exposures of a half-frame shot were taken at different moments and
        the camera can have been on its side for one and not the other. It is
        the one thing about a frame that its rectangle cannot say, because a
        rectangle looks the same either way up.

        It changes no geometry: the crop is where it was, and this is how the
        crop is stood up afterwards. `[` and `]` turn it a quarter, `u` puts
        it upright and `⇧U` upside down, and what is written is turned with
        it -- on the mosaic for a raw, which costs nothing.
        """
        it = self.item()
        p = (next((q for q in it["parts"] if q["name"] == name), None)
             if name else self.part())
        if not p:
            return
        was = p.get("turn", 0)
        now = by % 360 if absolute else (was + by) % 360
        if now == was:
            return                      # a destination it is already at
        # The snapshot is taken *before* the change, which is the whole point
        # of one: taken after, undo put back what it had just done.
        self._snapshot()
        p = (next((q for q in self.item()["parts"] if q["name"] == p["name"])))
        p["turn"] = now
        self.status.setText(f"{p['name']} upright" if not p["turn"]
                            else f"{p['name']} turned {p['turn']}°")
        self._edited("turn", p["name"])
        self.canvas.update()

    def _rolled(self):
        """What the roll says, for measuring one scan the way the roll would.

        Kept in the document when the folder is measured; read back off the
        frames already in it when it is not, so a document written before
        the summary existed still gets the roll's help rather than none.
        """
        from . import fitframe
        rolled = self.doc["roll"].get("rolled")
        if rolled and rolled.get("clusters"):
            return rolled
        return fitframe.roll_from_document(
            self.doc["items"], self.doc["roll"].get("format", "auto"))

    def _redetect(self):
        """Measure this scan again, with whatever the settings now say.

        The way to see a changed `crop` -- or one day a changed detector -- on
        the scan in front of you. It discards whatever was done to this scan
        by hand, which is why it goes on the undo stack first.
        """
        n = self.name()
        if not n or not self.here(n):
            return self.status.setText("the scan is not here to measure again")
        # **Asked, not assumed.** By default this leans on everything the
        # roll knows, which is what a scan measured on its own most needs:
        # a dark frame has no edge to find, and a strip of two half frames
        # with the gap between them is a 3:2 that alone reads as one full
        # frame. But the reason to press this is sometimes that the roll
        # got *this* one wrong, and then the roll is the last thing to lean
        # on. The answers govern this press and are not written down: a
        # choice about one scan is not a fact about the folder.
        from . import opening
        rolled = self._rolled()
        said = opening.ask_one(n, self.doc,
                               hand=bool(self.doc["items"][n].get("manual")),
                               rolled=rolled, parent=self)
        if said is None:
            return
        self._snapshot()
        from . import detect
        self.status.setText("measuring…")
        QApplication.processEvents()
        settings = dict(self.doc["roll"]) | {
            "one_roll": said["one_roll"], "same_size": said["same_size"],
            "same_gap": said["same_gap"]}
        self.doc["items"][n] = detect.measure(
            self.roll / n, self.doc["roll"].get("mirrored", True),
            settings, rolled if said["one_roll"] else None)
        self.dirty = True
        self._forget_developed(n)
        self._select(self.list.currentRow())
        # The frames moved, so the pictures of them went; the roll has to
        # make the ones it dropped again or walking back to this scan is a
        # second of develop that nothing was doing in the meantime.
        self._prefetch_later(200)
        self.status.setText("measured again")

    def _redetect_roll(self):
        """Measure every scan again.

        **It always asks.** It used to ask only where there were hand
        corrections to lose, so on a roll with none -- which is most of a
        roll, most of the time -- it started a pass over every scan with no
        warning at all, and what appeared was a window counting scans: the
        same shape as the job at the end of a sitting, which is what it
        was mistaken for. Now it says what it is about to do, and says
        separately what it cannot give back.
        """
        from . import develop
        from .progress import measure_roll
        hand = [n for n, v in self.doc["items"].items() if v.get("manual")]
        gone = [n for n in self.doc["items"] if not self.here(n)]
        n = len(self.doc["items"])
        goes = [f"the frames found on all {n} scans"]
        if hand:
            goes.append(f"{len(hand)} corrected by hand, which nothing can "
                        f"reproduce")
        if gone:
            goes.append(f"{len(gone)} whose scan has left the folder")
        if self.doc["roll"].get("measured"):
            goes.append("the roll's measured base, balance and matrix")
        if self.doc["roll"].get("develop"):
            goes.append("the develop settings — the taste and the base")
        # **The same sheet the roll was opened with.** Measuring a whole
        # folder again is the same act, with the same four answers, and it
        # is the moment to change them -- before it there was a
        # confirmation that named what would be lost and then measured on
        # whatever the document happened to hold, so the only way to
        # answer them differently was to delete the document.
        from . import opening
        said = opening.ask(
            self.roll.name, n, self.doc,
            workers=int(self.doc["roll"].get("workers")
                        or min(8, os.cpu_count() or 1)),
            parent=self, action="Measure them all again",
            warning="It goes back to a folder nobody has opened: "
                    + "; ".join(goes) + ". Kept: whether the roll is "
                      "mirrored and the crop, which is a way of showing and "
                      "not something measured.")
        if said is None:
            return
        # **Everything the program worked out, and nothing a person told
        # it about the film.** It used to `update()` the items, so records
        # whose scan had left the folder survived and every roll setting
        # with them -- which made the key that means *start again* the one
        # command that kept the most. What is kept is what nothing can
        # measure: whether the roll was copied mirrored, whether the folder
        # is one roll, and the crop, which is a way of showing.
        keep = {k: self.doc["roll"][k] for k in ("mirrored",)
                if k in self.doc["roll"]}
        keep["one_roll"] = said["one_roll"]
        keep["same_size"] = said["same_size"]
        keep["same_gap"] = said["same_gap"]
        crop = (self.doc["roll"].get("develop") or {}).get("crop")
        self.doc["roll"].pop("measured", None)
        self.doc["roll"].pop("develop", None)
        self.doc["roll"].pop("apply", None)
        self.doc["roll"].update(keep)
        if crop is not None:
            self.doc["roll"].setdefault("develop", {})["crop"] = crop
        # The base is how a frame is developed rather than how it is found,
        # so it goes to the develop settings the sheet just emptied.
        self.doc["roll"].setdefault("develop", {})["base"] = (
            "auto" if said["same_base"] else "frame")
        self.doc["roll"]["develop"]["flat"] = said.get("flat")
        files = [self.roll / q for q in self.names if self.here(q)]
        rolled = {}
        found = measure_roll(self.roll, files,
                             self.doc["roll"].get("mirrored", True),
                             self.doc["roll"], rolled)
        if found is None:
            return
        if rolled.get("clusters"):
            self.doc["roll"]["rolled"] = rolled
        self.doc["items"] = found
        self.names = document.order_of(self.doc)
        self._fill_list()
        # The roll has no measurement now, and the read line, `v` and the
        # develop all need one.
        from .progress import measure_base
        measure_base(self.roll, self.doc, self)
        self.dirty = True
        self.cache.clear()
        # Every frame may have moved, so every picture of one is out of date.
        self.dev_plain.clear()
        self.dev_ev.clear()
        self._forget_modes()
        self.base_cache.clear()
        self._select(self.list.currentRow())
        # **And it develops itself again.** Everything above is a cache
        # being emptied, and nothing filled it back: after `⇧⌘D` the roll
        # was as cold as a roll nobody had opened, so every step of `n`
        # paid the second that opening the roll pays once -- and paid it in
        # front of the person, as a black canvas that fills in twice. The
        # roll starts developing on open for exactly this reason
        # (`gui.run`); measuring again is the other moment every picture it
        # had made stopped being of these frames.
        self._prefetch_later(200)
        self.status.setText(f"measured {len(found)} again")

    def _light_falloff(self):
        """`⇧⌘L`: which calibration of the copy stand's light falloff the
        roll is divided by, or a new one made from blank frames
        (`rigs.py`). Changing it measures the roll again -- the base and
        the balance are read through it (`measured_ready`) -- and nothing
        else: the frames were found without it and stay where they are."""
        from . import rigs
        from .progress import measure_base
        dev = self.doc["roll"].setdefault("develop", {})
        scan = next((self.roll / q for q in self.names if self.here(q)), None)
        said = rigs.choose(dev.get("flat"), scan, self, start=self.roll)
        if said is None:
            return
        flat = said[1]
        if flat == dev.get("flat"):
            return self.status.setText(f"light falloff: {flat or 'none'}, as it was")
        dev["flat"] = flat
        self.dirty = True
        measure_base(self.roll, self.doc, self)
        self._forget_base()
        self._refresh()
        self.status.setText(f"light falloff: {flat or 'none'} — the roll measured again")

    def _look(self):
        """`⌘L`: the roll's taste -- exposure, warmth, tint, the curve's two
        slopes, its contrast and pivot, saturation -- moved by eye on the roll's own frames
        (`look.py`), and kept in `roll.develop`, where the canvas and `⌘R`
        read it. One look for the roll, never one per frame."""
        from . import look
        if not self.doc["roll"].get("measured"):
            return self.status.setText("the roll is not measured yet")
        now = self._look_now()
        new = look.ask(self)
        if new is None or all(abs(new[k] - now[k]) < 1e-9 for k in new):
            return self.status.setText("look unchanged")
        self._set_look(new, "look: " + ", ".join(
            f"{k} {new[k]:+.2f}" for k in look.KEYS if abs(new[k] - now[k]) > 1e-9))

    def _look_now(self):
        from . import develop, look
        taste = dict(develop.TASTE) | dict(self.doc["roll"].get("develop") or {})
        return {k: float(taste[k]) for k in look.KEYS}

    def _set_look(self, new, said, undoable=True):
        """The roll's look, undoable; every developed picture is of the old
        one, so all of them go and the roll develops itself again."""
        if undoable:
            self.undo.append((self.LOOK, self._look_now()))
            self.redo.clear()
        self.doc["roll"].setdefault("develop", {}).update(new)
        self.dirty = True
        self._forget_developed(None)
        self._prefetch_later(200)
        self.status.setText(said)

    def _process_scan(self):
        """`⇧⌘R`: the same sheet, on *only this photo* -- marking its dust
        or setting the look on it, and wanting the file, is the usual
        reason to process one."""
        self._process(alone=True)

    def _process(self, alone=False):
        """`⌘R`: the files to write for this roll. One sheet for both
        outputs, because they are one question, and the work then runs in
        the progress window."""
        import os
        from . import develop, positive, processing
        from .progress import Running
        chosen = [self.names[r] for r in self._chosen_rows()]
        want = processing.ask(self.doc, self, chosen, alone)
        if not want:
            return
        only = want.get("only")
        # The roll may still be developing itself in the background. The
        # job wants every core there is, and the pictures it makes are
        # better than the ones the prefetch would: they come off the
        # full-size render it is about to do anyway.
        self._stop_prefetch()
        doc, roll = self.doc, self.roll
        n = sum(len(it["parts"]) for name, it in doc["items"].items()
                if it["state"] != "skipped" and (only is None or name in only))
        if not n:
            return self.status.setText("nothing to process: no frames yet")
        workers = min(8, os.cpu_count() or 1)
        taste = dict(develop.TASTE) | dict(doc["roll"].get("develop") or {})
        taste["format"] = want["format"]
        doc["roll"]["develop"] = taste
        doc["roll"].setdefault("apply", {})["crop"] = want["crop"]
        doc["roll"]["apply"]["dust"] = bool(want.get("dust"))
        doc["roll"]["apply"]["compress"] = bool(want.get("compress", True))
        doc["roll"]["numbered"] = bool(want.get("numbered", True))
        self.dirty = True
        said = []

        # **The roll is measured once.** Its base, balance and matrix were
        # read when the roll was opened and kept in the document;
        # `d` draws with them, and now so does this. It used to measure the
        # roll again here, which cost a second pass over every frame and --
        # worse -- left the window and the files on two measurements that
        # nothing checked against each other.
        from .progress import measure_base
        measure_base(self.roll, doc, self)
        stats = develop.stats_of(doc["roll"].get("measured"), taste, self.roll)

        if want["positives"]:
            def job(progress):
                out_dir = roll / "developed"
                results, ev = develop.develop_roll(roll, doc, out_dir, taste,
                                                   workers, progress, stats, only)
                develop.record(out_dir, taste, ev, results, only)
                return results, ev
            dlg = Running("Developing", f"{n} frame{'s' if n != 1 else ''} of {roll.name} into "
                          f"developed/, as {want['format']}.", job, self)
            dlg.exec()
            if dlg.result is not None:
                results, ev = dlg.result
                said.append(f"{len(results)} positives")
                # **The roll develops itself again rather than keeping what
                # the job made.** `⌘R` used to hand its full-size renders
                # back subsampled, which saved the pool a pass -- but those
                # are frames cut out, and the canvas now keeps whole scans.
                # Two shapes in one cache is what broke compositing; the
                # pool does the roll in seconds and there is only one shape.
                self._prefetch_later(200)
            elif dlg.error:
                return self.status.setText(f"not developed: {dlg.error}")
            else:
                return self.status.setText("stopped")

        if want["raws"]:
            settings = positive.settings_of(doc)

            def job2(progress):
                out_dir = roll / "frames"
                results, ev = positive.write_roll(roll, doc, out_dir, settings,
                                                  workers, progress, stats, only)
                positive.record(out_dir, settings, ev, results, only)
                return results, ev
            dlg = Running("Writing raws", f"{n} frame{'s' if n != 1 else ''} of {roll.name} as "
                          f"positive DNGs into frames/, {want['crop']} crop"
                          + (", dust marks filled on the mosaic."
                             if settings["dust"] else "."),
                          job2, self)
            dlg.exec()
            if dlg.result is not None:
                results, ev = dlg.result
                said.append(f"{len(results)} raw DNGs")
            elif dlg.error:
                return self.status.setText(f"raws not written: {dlg.error}")

        if want.get("negatives"):
            from . import negative
            what = " and ".join(x for x, on in (("pictures", want["positives"]),
                                                ("raw DNGs", want["raws"])) if on)

            def job3(progress):
                out_dir = roll / negative.FOLDER
                results = negative.write_roll(
                    roll, doc, out_dir, taste, want["crop"], workers, progress,
                    stats, pictures=want["positives"], raws=want["raws"],
                    compress=bool(want.get("compress", True)), only=only)
                negative.record(out_dir, taste, want["crop"], results, only)
                return results
            dlg = Running("Writing negatives", f"{n} frame{'s' if n != 1 else ''} of {roll.name}, "
                          f"not inverted, as {what} into {negative.FOLDER}/.",
                          job3, self)
            dlg.exec()
            if dlg.result is not None:
                said.append(f"{len(dlg.result)} negatives")
            elif dlg.error:
                return self.status.setText(f"negatives not written: {dlg.error}")

        self.status.setText(" and ".join(said) + " written" if said else "")


    WARM_BUDGET = 400_000_000        # bytes of kept develops, about 130 frames

    def _trim_warm(self):
        total = sum(a.nbytes for a in self.dev_plain.values())
        while total > self.WARM_BUDGET and len(self.dev_plain) > 2:
            k = next(iter(self.dev_plain))
            total -= self.dev_plain.pop(k).nbytes
            self.dev_ev.pop(k, None)

    def _base_choice(self, name):
        """(trust the rebate, the roll's level for this scan) -- the two
        arguments `develop.base_of` takes, worked out exactly as the
        develop works them out, so a view cannot show a base the develop
        will not use."""
        from . import develop
        m = self.doc["roll"].get("measured") or {}
        taste = self.doc["roll"].get("develop") or {}
        trust, unit = develop.base_choice(
            m, taste, self.doc["roll"].get("one_roll", True))
        if not unit:
            return trust, None
        exp = develop.scan_exposure(self.roll / name)
        return trust, (unit * exp if exp else None)

    def _base_label(self):
        """The read line's label for where the base comes from: the roll or
        this frame. `_base_says` is the same fact in a sentence.

        It is read off `base_choice`, not off the setting, so it cannot
        disagree with what the develop will do -- `auto` on a folder that
        is not one roll says `THIS SCAN`, because that is what happens.
        """
        from . import develop
        name = self.name()
        if name and develop.with_specks(self.item() or {}, self.doc,
                                        name).get("base_sample"):
            return f'<span style="color:{SUN}">BASE: PICKED (⇧B)</span>'
        if name and develop.with_specks(self.item() or {}, self.doc,
                                        name).get("base_colour"):
            return f'<span style="color:{SUN}">BASE: PICKED FOR THE ROLL (⇧B)</span>'
        trust, unit = self._base_choice(name)
        if unit:
            return f'<span style="color:{REF}">BASE: THE ROLL</span>'
        want = (self.doc["roll"].get("develop") or {}).get("base", "auto")
        if want == "auto" and not self.doc["roll"].get("measured"):
            return f'<span style="color:{HELD}">BASE: NOT MEASURED</span>'
        return f'<span style="color:{REF}">BASE: THIS SCAN</span>'

    def _base_says(self, name=None):
        """One line: where the base of the scan on screen comes from, and
        what it is worth here."""
        want = (self.doc["roll"].get("develop") or {}).get("base", "auto")
        one = self.doc["roll"].get("one_roll", True)
        m = self.doc["roll"].get("measured") or {}
        trust, g = self._base_choice(name) if name else (True, None)
        if want == "frame":
            what = ("this scan's own unexposed film, or its frame's lightest "
                    "pixels where it has none")
        elif want == "roll":
            what = "the roll's, forced"
        elif not m:
            return "base: not measured yet — ⌘R"
        elif not one:
            what = ("this frame's own deepest shadows — the folder is not "
                    "one roll")
        else:
            what = "the roll's"
        return f"base: {what}" + (f", {g:.4f} here" if g else "")

    def _forget_base(self):
        """Every cached view is a picture of a base, so a changed base
        invalidates all of them -- the develop included, and its cut frames
        with it: the base is divided out before the frame is cut."""
        self._forget_modes()
        self.base_cache.clear()
        self.dev_plain.clear()
        self.dev_ev.clear()
        # Every picture the roll had made of itself was of the old base.
        # **Debounced.** `⌘B` walks three positions and `⇧⌘L` three answers,
        # and starting a pool of six processes on each of them -- and
        # killing it a keypress later -- is how a window that should have
        # been idle came to leave broken pipes behind. One press of a key
        # that changes the base is nearly always followed by another.
        self._prefetch_later()

    BASES = ("auto", "roll", "frame")
    BASE_NAMES = {"auto": "the roll's own rule",
                  "roll": "the roll's level, on every frame",
                  "frame": "each frame's own"}



    def _mirror(self):
        """Whether the scan is a mirror of the picture.

        A negative photographed through its base comes out reversed, which is
        the usual way of doing it and so the default. It is a property of how
        the roll was copied, not of one scan, so it lives in `roll` and one
        key changes it for all of them.
        """
        r = self.doc["roll"]
        r["mirrored"] = not r.get("mirrored", True)
        self.canvas.mirrored = r["mirrored"]
        # The letters go in reading order, and turning the picture round
        # changes which frame reads first. The `manual` paths are rewritten
        # with them, so a correction stays on the part it was made on.
        for it in self.doc["items"].values():
            document.order_parts(it, r["mirrored"])
        self.dirty = True
        self.status.setText("mirrored" if r["mirrored"] else "not mirrored")
        self._refresh()
        if self.colour == "develop" and self.cut:
            # The cut develop has the mirror baked into its pixmap
            # (`_developed_cut`), as it has the turn; the other views mirror
            # as they draw. Without this the picture stayed as it was until
            # the next scan.
            self._want_developed()

    def _reset(self):
        """Put this item back as it was measured: forget every hand
        correction on it, and re-measure it from the file."""
        n = self.name()
        if not n or not self.item().get("manual"):
            return self.status.setText("nothing was changed on this scan")
        if not self.here(n):
            return self.status.setText("the scan is not here to measure again")
        # `placed` goes with the rest: it is a hand correction, and putting the
        # item back as measured means putting back a rectangle nobody placed.
        self._snapshot()
        if self.mode:
            # It replaces the whole record, and the canvas holds the old list
            # of parts. Leaving first is what rebuilds it; staying in would
            # edit a list nothing else can see any more.
            self.mode = self.canvas.mode = False
            self.before = None
            self.stack.setCurrentIndex(0)
        from . import detect
        self.status.setText("measuring again…")
        QApplication.processEvents()
        self.doc["items"][n] = detect.measure(
            self.roll / n, self.doc["roll"].get("mirrored", True))
        self.dirty = True
        self.status.setText("put back as measured")
        self._select(self.list.currentRow())

    def _skip(self):
        it = self.item()
        if not it:
            return
        self._snapshot()
        it["state"] = "detected" if it["state"] == "skipped" else "skipped"
        self.dirty = True
        self._refresh()
        self._fill_list()          # every number after this one moved

    def _undo(self):
        self._dust_clone_cancel()
        if not self.undo:
            return self.status.setText("nothing to undo")
        n, snap = self.undo.pop()
        if n == self.LOOK:
            self.redo.append((n, self._look_now()))
            return self._set_look(snap, "look put back", undoable=False)
        if n == self.DUST:
            name, old = snap
            self.redo.append((n, (name, copy.deepcopy(
                (self.doc.get("specks") or {}).get(name)))))
            return self._set_dust(name, old, "dust marks put back")
        if n == self.BASE:
            name, old = snap
            self.redo.append((n, (name, self._base_of(name))))
            return self._set_base(name, old, "base put back", undoable=False)
        if n == self.ORDER:
            here = self.name()
            self.redo.append((n, list(self.names)))
            self._apply_order(snap)
            return self._goto(here) if here in self.names else None
        self.redo.append((n, copy.deepcopy(self.doc["items"][n])))
        skipped = self.doc["items"][n].get("state") == "skipped"
        self.doc["items"][n] = snap
        self.dirty = True
        if skipped != (snap.get("state") == "skipped"):
            self._fill_list()      # a skip undone renumbers the roll
        # The frame moved back, so the picture of it developed is of pixels
        # that are no longer in it.
        self._forget_developed(n)
        self._goto(n)

    def _redo(self):
        self._dust_clone_cancel()
        if not self.redo:
            return self.status.setText("nothing to redo")
        n, snap = self.redo.pop()
        if n == self.LOOK:
            self.undo.append((n, self._look_now()))
            return self._set_look(snap, "look redone", undoable=False)
        if n == self.DUST:
            name, new = snap
            self.undo.append((n, (name, copy.deepcopy(
                (self.doc.get("specks") or {}).get(name)))))
            return self._set_dust(name, new, "dust marks redone")
        if n == self.BASE:
            name, new = snap
            self.undo.append((n, (name, self._base_of(name))))
            return self._set_base(name, new, "base redone", undoable=False)
        if n == self.ORDER:
            here = self.name()
            self.undo.append((n, list(self.names)))
            self._apply_order(snap)
            return self._goto(here) if here in self.names else None
        self.undo.append((n, copy.deepcopy(self.doc["items"][n])))
        skipped = self.doc["items"][n].get("state") == "skipped"
        self.doc["items"][n] = snap
        self.dirty = True
        if skipped != (snap.get("state") == "skipped"):
            self._fill_list()
        self._forget_developed(n)
        self._goto(n)

    def _goto(self, name):
        i = self.names.index(name)
        if self.list.currentRow() == i:
            self._select(i)
        else:
            self.list.setCurrentRow(i)

    # ---- saving, which is not a thing a person does any more
    #
    # There is no `save`. Every change writes the document, and `⌘Z` takes it
    # back -- which is the pair that makes autosave safe. A key that has to be
    # pressed for the work to survive is a key that will one day not be
    # pressed, and the corrections in this window are the expensive half of
    # the program: measuring them again is cheap, placing them by hand is not.
    @property
    def dirty(self):
        return self._dirty

    @dirty.setter
    def dirty(self, value):
        self._dirty = False
        if value:
            self._save()

    def _save(self):
        try:
            document.save(self.doc, self.roll)
        except OSError as e:                      # a folder gone, a disk full
            self.status.setText(f"not saved: {e}")

    def closeEvent(self, e):
        self._stop_prefetch(wait=15000)
        for w in list(self._threads):
            w.stop = True
            w.wait(15000)
        self._drop_worker()
        super().closeEvent(e)

    # ---- the list
    def _row_html(self, i, name):
        # the shot number, not the row's: a skipped scan has none and the
        # ones after it close up (`document.shot_numbers`)
        num = document.shot_numbers(self.doc).get(name)
        lab = f"{num:>3}" if num else "   "
        # the name column fits the roll's longest name: at a fixed 14 a
        # renamed scan (DSC_0315_alt-0239.NEF, 21) ran into its letter
        w = max(14, 1 + max(map(len, self.doc["items"]), default=0))
        it = self.doc["items"][name]
        parts, hand = it["parts"], bool(it.get("manual"))
        if not self.here(name):
            return (f'<span style="color:{ASLEEP}">{lab} {name:<{w}} '
                    f'not in the folder</span>')
        if it["state"] == "skipped":
            # A row that is switched off is the exception: that fact is about
            # the whole row, so the whole row goes dim.
            return (f'<span style="color:{ASLEEP}">{lab} {name:<{w}} '
                    f'skipped</span>')
        if not parts:
            return (f'<span style="color:{SMALL}">{lab}</span> '
                    f'<span style="color:{PRIMARY}">{name:<{w}}</span>'
                    f'<span style="color:{GONE}">nothing found</span>')
        weak = it.get("evidence", {}).get("angle_margin", 9) < WEAK
        held = it["state"] == "held"
        return (f'<span style="color:{SMALL}">{lab}</span> '
                f'<span style="color:{PRIMARY}">{name:<{w}}</span>'
                f'<span style="color:{COMMAND}">'
                f'{" ".join(p["name"] for p in parts):<4}</span>'
                f'<span style="color:{SMALL}">'
                f'{self._format_label(parts):<5}</span>'
                f'<span style="color:{HELD if weak else DETECTED}">'
                f'{it.get("angle", 0.0):+.2f}°</span>'
                + (f' <span style="color:{MANUAL}">yours</span>' if hand else
                   f' <span style="color:{HELD}">held</span>' if held else ""))

    def _fill_list(self):
        """Build the list from `names`, replacing whatever is there.

        It only ever appended, which was invisible while it ran once at
        build time. `⇧⌘D` calls it again on a roll that just lost a record,
        and nine rows appeared for four scans. It rebuilds in place --
        reusing the rows it can, never `clear()` -- because clearing throws
        away the scroll position, so the row under the pointer stops being
        the row you clicked.
        """
        # The 360 fits a 14-character name; a longer one widens the name
        # column (`_row_html`), so the list widens by as much, or the angle
        # is what falls off the right.
        extra = max(0, max(map(len, self.doc["items"]), default=0) + 1 - 14)
        self.list.setFixedWidth(
            360 + extra * QFontMetrics(self.list.font()).horizontalAdvance("0"))
        while self.list.count() > len(self.names):
            self.list.takeItem(self.list.count() - 1)
        for i, name in enumerate(self.names):
            row = self.list.item(i)
            if row is None:
                row = QListWidgetItem()
                self.list.addItem(row)
            num = document.shot_numbers(self.doc).get(name)
            row.setText(f"{num or ''} {name}".strip())
            row.setData(Qt.UserRole, name)
            row.setData(RowDelegate.HTML, self._row_html(i, name))
            row.setToolTip(self._tooltip(name))

    def _tooltip(self, name):
        it = self.doc["items"][name]
        parts = it["parts"]
        if not parts:
            return ('<qt><div style="white-space:normal" width="380">'
                    f'{name}: nothing with a believable shape. '
                    f'+ adds a part by hand.</div></qt>')
        m = it.get("evidence", {}).get("angle_margin")
        held = it.get("evidence", {}).get("held") or []
        return ('<qt><div style="white-space:normal" width="380">'
                f'{name} · {len(parts)} part{"s" if len(parts) > 1 else ""} '
                f'· {self._format_label(parts)} '
                f'· turned {it.get("angle", 0.0):+.2f}°.'
                + (f' The angle peak scored {m:.2f} against the best angle half '
                   f'a degree away; below {WEAK} that is not a measurement.'
                   if m else "" if it.get("by") == "fit" else " Set by hand.")
                + (f' Held: {"; ".join(held)} -- the roll supplied what the '
                   f'scan could not, or contradicted it.' if held else "")
                + (f' Corrected by hand: {", ".join(it["manual"])}.'
                   if it.get("manual") else "")
                + '</div></qt>')

    # ---- the scan's own base (`⇧B`)
    BASE = "#base"       # an undo entry that is one scan's picked base
    DUST = "#dust"       # an undo entry that is one scan's dust marks
    LOOK = "#look"       # an undo entry that is the roll's look (`⌘L`)

    ROLL_BASE = ""       # the name a base picked for the whole roll goes by

    def _base_of(self, name):
        """The base picked for this scan, or for the roll (`ROLL_BASE`)."""
        if name == self.ROLL_BASE:
            return self.doc["roll"].get("base_pick")
        return (self.doc.get("bases") or {}).get(name)

    def _base_box(self, name):
        """The box drawn on this scan: its own pick, or the roll's if it
        was picked here."""
        own = self._base_of(name) if name else None
        whole = self._base_of(self.ROLL_BASE)
        if not own and whole and whole.get("scan") == name:
            own = whole
        return tuple(own["box"]) if own else None

    def _pick_base(self):
        """`⇧B`: drag a box over unexposed film on this scan, and the scan
        -- or the whole roll -- is developed against that film rather than
        the base the roll measured.

        For a scan copied under another light -- another session, another
        panel -- whose colour the roll's base, measured on the rest, cannot
        take out (`develop.own_base`); or for a roll whose measured base
        came out the wrong colour, which is the same correction on every
        frame. On the whole scan, where the film between the frames and
        along the edge is in sight. On a scan that has one already, it
        takes it away; with one for the roll, it asks."""
        from . import sheets
        name = self.name()
        if not name or not (self.item() or {}).get("parts"):
            return self.status.setText("no frame here to pick a base for")
        if self._base_of(name):
            return self._set_base(name, None, "the roll's base again on this "
                                  "scan (⇧B to pick one, ⌘Z to undo)")
        whole = self._base_of(self.ROLL_BASE)
        if whole:
            got = sheets.ask(
                "Film base", "The roll uses a base picked on "
                + whole.get("scan", "a scan").rsplit(".", 1)[0] + ".",
                "Remove it to go back to the base the roll measured, or "
                "pick another.",
                (("cancel", "Cancel", "cancel"), ("remove", "Remove it", "quiet"),
                 ("pick", "Pick another…", "go")), self)
            if got == "remove":
                return self._set_base(self.ROLL_BASE, None, "the measured base "
                                      "again on the roll (⌘Z to undo)")
            if got != "pick":
                return
        if self.cut:
            self._cut()
        self.canvas.picking_base = True
        self.canvas.base_box = None
        self.canvas.setCursor(Qt.CrossCursor)
        self.status.setText("drag a box over unexposed film on this scan -- "
                            "between two frames or along the edge, clear of "
                            "the holder -- esc to leave it")

    def _stop_pick(self, said):
        self.canvas.picking_base = False
        self.canvas._base_from = None
        self.canvas.unsetCursor()
        self.canvas.base_box = self._base_box(self.name())
        self.canvas.update()
        if said:
            self.status.setText(said)

    def _base_picked(self, box):
        from . import develop, sheets
        name = self.name()
        self.canvas.picking_base = False
        self.canvas.unsetCursor()
        self.status.setText("reading the base…")
        QApplication.processEvents()
        try:
            value = develop.base_sample(self.roll / name, box, self._flat_named())
        except Exception as e:
            return self._stop_pick(f"no base read: {e}")
        roll = (self.doc["roll"].get("measured") or {}).get("base_colour")
        g = max(value[1], 1e-6)
        read = (f"R/G {value[0] / g:.3f}, B/G {value[2] / g:.3f}"
                + (f" -- the roll measured {roll[0]:.3f}, {roll[2]:.3f}"
                   if roll else ""))
        # **This scan, or the roll.** A scan copied under another light is
        # the first; a roll whose measured base is off is the second, and
        # picking it scan by scan would be the same box drawn on every one.
        got = sheets.ask(
            "Film base", "Use this base for this scan, or for the whole roll?",
            read + ". For the roll, only its colour is used: each scan keeps "
            "the level of its own exposure.",
            (("cancel", "Cancel", "cancel"), ("roll", "The whole roll", "quiet"),
             ("scan", "This scan", "go")), self)
        if got not in ("scan", "roll"):
            return self._stop_pick("no base picked")
        flat = (self.doc["roll"].get("develop") or {}).get("flat")
        own = {"box": [round(v, 1) for v in box], "value": value, "flat": flat}
        if got == "roll":
            own["scan"] = name
            return self._set_base(self.ROLL_BASE, dict(own),
                                  f"the roll's base, picked here: {read}  (⌘Z to undo)")
        self._set_base(name, own, f"this scan's base: {read}  (⌘Z to undo)")

    def _set_base(self, name, own, said, undoable=True):
        if undoable:
            self.undo.append((self.BASE, (name, self._base_of(name))))
            self.redo.clear()
        if name == self.ROLL_BASE:
            if own:
                self.doc["roll"]["base_pick"] = own
            else:
                self.doc["roll"].pop("base_pick", None)
        else:
            bases = self.doc.setdefault("bases", {})
            if own:
                bases[name] = own
            else:
                bases.pop(name, None)
        self.dirty = True
        here = self.name()
        if name == self.ROLL_BASE:
            self._stop_prefetch()
            self.base_cache.clear()
            self._forget_developed()
            self._forget_modes()
            self._prefetch_later(200)
        else:
            self.base_cache.pop(name, None)
            self._forget_developed(name)
        if name in (here, self.ROLL_BASE):
            self.canvas.base_box = self._base_box(here)
            self._say_base(here)
            # the positive and the scan views are drawn through the base
            # too, and were dropped with it: drawn again from scratch
            self._select(self.list.currentRow())
        self.status.setText(said)

    # ---- the order of the list
    ORDER = "#order"     # an undo entry that is the list's order, not an item

    def _set_order(self, new, said):
        """The list in this order, kept in the roll, undoable; the scan in
        hand stays the one in hand, wherever it went."""
        new = list(new)
        if new == self.names:
            return
        here = self.name()
        self.undo.append((self.ORDER, list(self.names)))
        self.redo.clear()
        self._apply_order(new)
        if here in self.names:
            self._goto(here)
        self.status.setText(said)

    def _apply_order(self, new):
        self.names = list(new)
        self.doc["roll"]["order"] = list(new)
        self.dirty = True
        self._fill_list()

    def _list_dropped(self):
        """After a drag: the order is the rows' own, read off their names."""
        new = [self.list.item(i).data(Qt.UserRole)
               for i in range(self.list.count())]
        if None in new or sorted(new) != sorted(self.names):
            return self._fill_list()
        moved = len(self.list.selectedItems()) or 1
        self._set_order(new, f"moved {moved} scan{'s' if moved != 1 else ''}"
                        " in the list")

    def _chosen_rows(self):
        rows = sorted(self.list.row(r) for r in self.list.selectedItems())
        return rows or ([self.list.currentRow()]
                        if self.list.currentRow() >= 0 else [])

    def _move_rows(self, d):
        """⌥↑ ⌥↓: the chosen scans one place up or down, as a block."""
        rows = self._chosen_rows()
        if not rows or rows[0] + d < 0 or rows[-1] + d >= len(self.names):
            return
        chosen = [self.names[r] for r in rows]
        rest = [n for n in self.names if n not in chosen]
        at = rows[0] + d
        new = rest[:at] + chosen + rest[at:]
        self._set_order(new, f"{len(chosen)} scan{'s' if len(chosen) > 1 else ''}"
                        f" moved {'up' if d < 0 else 'down'}")
        self.list.blockSignals(True)
        for r in range(len(chosen)):
            self.list.item(at + r).setSelected(True)
        self.list.blockSignals(False)

    def _sort_names(self):
        self._set_order(sorted(self.names), "sorted by file name, A to Z")

    def _sort_names_back(self):
        self._set_order(sorted(self.names, reverse=True),
                        "sorted by file name, Z to A")

    def _reverse_order(self):
        self._set_order(self.names[::-1], "the list turned round")

    def _list_menu(self, at):
        from PySide6.QtWidgets import QMenu
        m = QMenu(self)
        m.addAction("Sort by file name, A to Z", self._sort_names)
        m.addAction("Sort by file name, Z to A", self._sort_names_back)
        m.addAction("Reverse the order", self._reverse_order)
        m.addSeparator()
        n = len(self._chosen_rows())
        what = f"{n} scans" if n > 1 else "this scan"
        m.addAction(f"Move {what} up\t⌥↑", lambda: self._move_rows(-1))
        m.addAction(f"Move {what} down\t⌥↓", lambda: self._move_rows(+1))
        m.exec(self.list.mapToGlobal(at))

    # ---- selection and refresh
    def _step(self, d):
        r = self.list.currentRow() + d
        if 0 <= r < len(self.names):
            self.list.setCurrentRow(r)


    def here(self, name):
        """Whether the scan this record is about is still in the folder."""
        return (self.roll / name).exists()

    def _select(self, row):
        if not (0 <= row < len(self.names)):
            return
        if self.dusting and self.names[row] != self.dust["name"]:
            # the marks are of that scan, and they are already saved
            self._leave_dust()
        name = self.names[row]
        it = self.doc["items"][name]
        if not self.here(name):
            # The record stays: it is somebody's hand-placed frames and the
            # only copy of them. What is gone is the picture, so there is
            # nothing to draw and nothing to measure -- and saying so beats
            # a traceback from LibRaw about a file that is not there.
            self.canvas.show_scan(None, [], set(), 1.0)
            self._read(row, name)
            return
        self.status.setText("reading…")
        QApplication.processEvents()
        # Inside the placement mode the canvas is the autostretch, which is
        # the tool for finding an edge; outside it, it is whichever of the
        # three ways of looking is on.
        #
        # **The scale is arithmetic, not a raw read.** It used to come from
        # `_load`, which reads and demosaics the scan -- 0.40 s of the 2.2 s
        # a scan change cost in the develop, for a number that is the scan's
        # own width over the width of the picture drawn from it.
        scale = self._canvas_scale(it)
        if self.mode:
            pm, scale = self._pixmaps(name, self._frame_box(it))
        elif self.colour == "develop" and it["parts"]:
            # **Nothing to read.** Cropped, the canvas draws the developed
            # frame and the scan behind it is never painted; building it
            # cost a raw read a scan -- 0.43 s -- for pixels nobody sees,
            # which is what kept a roll from being walked at speed after
            # `⌘R` had already developed every frame of it.
            pm = None
        else:
            pm = self._mode_pixmap(name, it)
            if pm is None:
                pm, scale = self._pixmaps(name, self._frame_box(it))
        sel = self.canvas.sel
        self.canvas.crop_pct = float(
            self.doc["roll"].get("develop", {}).get("crop", 0.0))
        self.canvas.show_scan(pm, it["parts"], self._manual_parts(it), scale,
                              angle=it.get("angle", 0.0),
                              mirrored=self.doc["roll"].get("mirrored", True),
                              scan_w=it["scan"][0])
        self.canvas.sel = min(sel, max(0, len(it["parts"]) - 1))
        self.canvas.base_box = self._base_box(name)
        self._say_base(name)
        if self.canvas.picking_base:
            self._stop_pick("")
        # **From the moment the roll opens.** The indicator used to be set
        # only by pressing `v`, so a window that had just opened showed a
        # picture and said nothing about which of the three it was -- which
        # is the one moment a person has no way of knowing.
        self.canvas.label = self._label()
        self.status.setText("")
        self._refresh()
        # Both axes outlive moving between scans -- they are ways of reading
        # the roll, not a look at one item. A scan with nothing found has no
        # frame to cut or develop and comes up whole.
        if self.colour == "develop":
            self._want_developed() if it["parts"] else self._hold(None, "")
        elif self.cut:
            self._show_cut() if it["parts"] else self._hold(None, "")
        else:
            self._hold(None, self._mode_says())

    def _refresh(self):
        """Redraw the canvas, the read line, this row, and the sheet.

        The sheet is recomputed here rather than once at build time because
        dimming is per item: a scan with one part has nothing for `2` to pick
        and nothing for `-` to drop, and the sheet is meant to be the one
        thing on screen that answers honestly about the item in front of you.
        """
        self.canvas.update()
        row = self.list.currentRow()
        if not (0 <= row < len(self.names)):
            return
        name = self.names[row]
        self.list.item(row).setData(RowDelegate.HTML, self._row_html(row, name))
        self.list.item(row).setToolTip(self._tooltip(name))
        self._read(row, name)
        self._say_crop()
        self._say_space()
        it = self.doc["items"][name]
        have = {"": True, "part": bool(it["parts"]),
                "two": len(it["parts"]) > 1,
                "manual": bool(it.get("manual")), "undo": bool(self.undo),
                "redo": bool(self.redo),
                "copied": bool(getattr(self, "clipboard", None))}
        for need, labels in getattr(self, "names_by_need", {}).items():
            for lab in labels:
                lab.live = have.get(need, True)
                self._dress(lab)

    @staticmethod
    def _format_label(parts):
        """What this scan holds, said the same way everywhere.

        Two parts is half frame whatever the per-part ratio came out as --
        the count is the stronger evidence, and the list saying `full` while
        the read line says `HALF FRAME` about the same scan is the kind of
        thing the house rule about one word per thing exists to stop."""
        if not parts:
            return ""
        return "half" if len(parts) > 1 else parts[0]["format"]

    def _manual_parts(self, it):
        return {m.split(".")[1] for m in it.get("manual", []) if "." in m}

    def _read(self, row, name):
        """Position and name first, then labels in the colour of what they
        claim, then after a separator what the job will do with this item and
        the numbers measured off what is on screen. It never lists keys."""
        it = self.doc["items"][name]
        parts = it["parts"]
        head = (f'<span style="color:{SMALL}">{row+1}/{len(self.names)}</span>'
                f'&nbsp;&nbsp;<span style="color:{PRIMARY}">{name}</span>')
        if not self.here(name):
            return self.reading.setText(
                head + f'&nbsp;&nbsp;<span style="color:{GONE}">NOT IN THE '
                       f'FOLDER</span>&nbsp;&nbsp;'
                       f'<span style="color:{EXPLAIN}">its frames are still '
                       f'here, and are the only copy</span>')
        if it["state"] == "skipped":
            return self.reading.setText(
                head + f'&nbsp;&nbsp;<span style="color:{GONE}">SKIPPED</span>')
        if not parts:
            return self.reading.setText(
                head + f'&nbsp;&nbsp;<span style="color:{GONE}">'
                       f'NOTHING FOUND</span>')
        labels = [f'<span style="color:{DETECTED}">'
                  f'{self._format_label(parts).upper()}</span>']
        if it.get("evidence", {}).get("angle_margin", 9) < WEAK:
            labels.append(f'<span style="color:{HELD}">WEAK ANGLE</span>')
        if it["state"] == "held":
            labels.append(f'<span style="color:{HELD}">HELD</span>')
        # **Always**, because the difference is what the picture is divided
        # by: the film, measured over the whole roll, or this one frame's
        # deepest shadow. Neither is a fault -- which one a roll gets is a
        # property of its rebate -- so both are in the reference colour and
        # only *not measured yet* is marked as something to do.
        labels.append(self._base_label())
        if it.get("manual"):
            labels.append(f'<span style="color:{MANUAL}">YOURS</span>')
        elif it.get("by") and it["by"] not in ("fit", "hand"):
            # **What wrote this, when it was not the detector there is.**
            # It used to compare against the detector the roll had chosen,
            # and there is no choosing any more -- so this is the one thing
            # left worth saying: a record measured by `profiles`, which is
            # gone, and which read 31 of 37 on half frame where `fit` reads
            # 36. It is the label that would have said out loud what a roll
            # could be quietly doing. `⇧⌘D` is the way out.
            labels.append(f'<span style="color:{HELD}">{it["by"].upper()}</span>')
        p = parts[min(self.canvas.sel, len(parts) - 1)]
        w, h = p["size"]
        self.reading.setText(
            head + f'&nbsp;&nbsp;{"&nbsp;".join(labels)}'
            f'&nbsp;&nbsp;<span style="color:#3a3a42">│</span>&nbsp;&nbsp;'
            f'<span style="color:{EXPLAIN}">'
            + (f'part {p["name"]} of {len(parts)}, ' if len(parts) > 1 else "")
            + (f'turned {p["turn"]}°, ' if p.get("turn") else "")
            + f'{w:.0f}×{h:.0f}, {w/max(h,1):.3f}:1, '
              f'film turned {it.get("angle", 0.0):+.2f}°'
            # Said from the scan's side, because the scan is what is on
            # disk and what the numbers beside it are measured in. The
            # canvas flips it so the picture reads the right way round,
            # and that flip is the thing worth naming.
            + (", shown mirrored" if self.doc["roll"].get("mirrored", True)
               else ", shown as scanned")
            + '</span>')

    # ---- pixels
    @staticmethod
    def _frame_box(item):
        """The bounding box of every part, in scan pixels -- where the
        preview's stretch is measured, so it is measured on the picture."""
        pts = [c for p in item["parts"] for c in findframe.corners(
            p["centre"][0], p["centre"][1], p["size"][0], p["size"][1],
            item.get("angle", 0.0))]
        if not pts:
            return None
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def _load(self, name, region=None):
        """The scan as a positive, before the display curve.

        Cached like this rather than as pixels because the two view knobs put
        their gain in *front* of the curve: caching what is on screen would
        mean re-reading a 30 MB raw every time a slider moved.
        """
        if name not in self.cache:
            self.cache[name] = image_io.preview(self.roll / name, region=region)
            if len(self.cache) > 6:            # a hundred-odd MB, no more
                self.cache.pop(next(iter(self.cache)))
        return self.cache[name]

    def _pixmaps(self, name, region=None):
        """The autostretch: the frame's own 2nd and 98th percentiles, per
        channel, then the display curve and the two knobs.

        **This is the tool for placing a frame, not a way of looking at a
        picture.** It puts 2 % of the picture on pure black and 2 % on pure
        white by construction -- measured on 023 of negativos-portra400,
        2.00 % and 2.03 % -- which is what makes an edge findable and what
        made it read as a result when it was the canvas's default. The
        canvas draws it inside the placement mode, and nowhere else.
        """
        pos, scale = self._load(name, region)
        return _pix(image_io.tone(pos, self.ev, self.detail)), scale

    def _retone(self):
        """New knobs, same scan: rebuild the pixels and leave everything else,
        including where the view is pointing."""
        name = self.name()
        if not name:
            return
        pm, _ = self._pixmaps(name, self._frame_box(self.doc["items"][name]))
        self.canvas.pixmap = pm
        self.canvas.update()


def _to_segment(p, a, b):
    """Screen distance from the point `p` to the segment `a`-`b`."""
    dx, dy = b.x() - a.x(), b.y() - a.y()
    t = ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / max(dx * dx + dy * dy, 1e-9)
    t = min(1.0, max(0.0, t))
    return math.hypot(p.x() - a.x() - t * dx, p.y() - a.y() - t * dy)


def _tip(text):
    """Rich text always: a plain tooltip is laid out on one line however long
    it is, and these run to three sentences."""
    return f'<qt><div style="white-space:normal" width="380">{text}</div></qt>'


def _pix(arr):
    """A float picture in 0..1 as a pixmap.

    A plain scaling by 255, because everything that reaches it is already
    display-referred: `image_io.tone` ends on its own curve and the
    measured positive on a 2.2 gamma. The develop does **not** -- `finish`
    returns scene-linear sRGB and the transfer is `develop.encode` -- so
    that one goes through `_pix8`.
    """
    a = np.ascontiguousarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    h, w, _ = a.shape
    return QPixmap.fromImage(QImage(a.data, w, h, 3 * w,
                                    QImage.Format_RGB888).copy())


def _mask_pix(mask, rgb):
    """A boolean picture as a pixmap: `rgb` where True, transparent
    elsewhere."""
    h, w = mask.shape
    a = np.zeros((h, w, 4), np.uint8)
    a[mask] = (*rgb, 255)
    return QPixmap.fromImage(QImage(np.ascontiguousarray(a).data, w, h, 4 * w,
                                    QImage.Format_RGBA8888).copy())


def _share(x):
    """A share of a scan, said so a small one is not rounded to nothing."""
    if x <= 0:
        return "0 %"
    if x < 0.0001:
        return "< 0.01 %"
    return f"{x * 100:.2f} %" if x < 0.1 else f"{x * 100:.1f} %"


def _pix8(a):
    """Bytes already encoded for a screen, as a pixmap."""
    a = np.ascontiguousarray(a)
    h, w, _ = a.shape
    return QPixmap.fromImage(QImage(a.data, w, h, 3 * w,
                                    QImage.Format_RGB888).copy())
