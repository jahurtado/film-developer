"""A window for anything a person would otherwise wait through.

A job of minutes gets a window, not a spinner, and it says what
it is doing in countable units. An icon in the Dock has no terminal, so the
longest thing the program does was the one thing with nothing to look at.

Closing the window abandons the run. Measuring writes nothing until it
finishes, so there is nothing to leave half-done.
"""

import pathlib
import time

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (QDialog, QLabel, QProgressBar, QVBoxLayout)

from . import detect, detectors, document


class _Abandoned(Exception):
    pass


class _Worker(QThread):
    measured = Signal(str, object)
    ticked = Signal(str, int, int)
    rolled = Signal(object)
    done = Signal()

    def __init__(self, files, mirrored=True, settings=None):
        super().__init__()
        self.files, self.mirrored = files, mirrored
        self.settings = settings or {}
        self.stop = False

    def run(self):
        # `fit` measures the folder as one thing, twice; the bar counts both
        # passes, and the records only exist at the end.
        def tick(i, n, scan):
            if self.stop:
                raise _Abandoned()
            self.ticked.emit(scan, i + 1, n)
        summary = {}
        try:
            items = detectors.run_roll(self.files, self.settings, tick, summary)
        except _Abandoned:
            return
        if summary.get("clusters"):
            self.rolled.emit(summary)
        for p in self.files:
            self.measured.emit(
                p.name, document.order_parts(items[p.name], self.mirrored))
        self.done.emit()


class Measuring(QDialog):

    def __init__(self, roll, files, mirrored=True, settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Measuring")
        self.setModal(True)
        self.found, self.rolled = {}, None
        self.files, self.started, self.n = files, time.monotonic(), 0
        self._shown = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(30, 26, 30, 26)
        lay.setSpacing(10)
        head = QLabel("Measuring…")
        head.setStyleSheet("font-size: 15px;")
        lay.addWidget(head)
        self.what = QLabel(f"Finding the frames in {len(files)} scans "
                           f"in {roll.name}")
        self.what.setStyleSheet("color: #8a8a90;")
        lay.addWidget(self.what)
        self.bar = QProgressBar()
        self.bar.setRange(0, len(files))
        self.bar.setTextVisible(False)
        lay.addWidget(self.bar)
        self.left = QLabel("")
        self.left.setStyleSheet("color: #70737c;")
        lay.addWidget(self.left)
        self.setFixedWidth(460)

        self.worker = _Worker(files, mirrored, settings)
        self.worker.measured.connect(self._one)
        self.worker.rolled.connect(self._rolled)
        self.worker.ticked.connect(self._tick)
        self.worker.done.connect(self.accept)
        self.worker.start()

    def _tick(self, name, i, n):
        """Progress from a detector that reports as it goes and delivers its
        records at the end: the bar follows the ticks, the records follow."""
        self.bar.setRange(0, n)
        self.n = i
        self.bar.setValue(i)
        self.what.setText(f"{name} — {i} of {n}"
                          + (", second pass" if i > n // 2 else ""))
        self._estimate(n)

    def _rolled(self, summary):
        """What the first pass decided about the roll, so a single scan can
        be measured again against the same numbers."""
        self.rolled = summary

    def _one(self, name, record):
        self.found[name] = record
        if self.bar.maximum() != len(self.files):
            return                            # ticks already drew the bar
        self.n += 1
        self.bar.setValue(self.n)
        self.what.setText(f"{name} — {self.n} of {len(self.files)}")
        self._estimate(len(self.files))

    def _estimate(self, total):
        # An estimate that keeps changing is not one: recompute it from work
        # done, and only change what it says when the answer moves by a fifth.
        if not self.n:
            return
        per = (time.monotonic() - self.started) / self.n
        secs = per * (total - self.n)
        if self._shown is None or abs(secs - self._shown) > self._shown * 0.2:
            self._shown = secs
        self.left.setText("about {:.0f} s left".format(self._shown)
                          if self._shown > 1 else "")

    def closeEvent(self, e):
        self.worker.stop = True
        self.worker.wait(2000)
        super().closeEvent(e)


class _Runner(QThread):
    """A callable that takes `progress(i, n)`, run off the window's thread."""
    ticked = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn, self.stop = fn, False

    def run(self):
        def tick(i, n):
            if self.stop:
                raise _Abandoned()
            self.ticked.emit(i + 1, n)
        try:
            self.done.emit(self.fn(tick))
        except _Abandoned:
            return
        except Exception as e:  # shown in the window, not lost in a thread
            self.failed.emit(f"{type(e).__name__}: {e}")


def measure_base(roll, doc, parent=None):
    """The roll's base, balance and matrix, measured once and kept in
    the document as `roll.measured`. True if it is there afterwards.

    It is measured **when the roll is opened**, not when something is
    developed. Which base a picture is divided by is on the read line of
    every scan, `v` divides by it and `hold b` paints it, and all three
    were lying -- or saying `NOT MEASURED` -- on a roll nobody had
    processed yet. It is a property of the roll, it costs one pass over
    the scans at half size, and the document keeps it, so it is paid once
    per roll and never again.
    """
    import os
    from . import develop
    taste = dict(develop.TASTE) | dict(doc["roll"].get("develop") or {})
    # **Under the settings now in force.** The base and the balance are
    # read at a gamma, so a measurement taken at another one describes a
    # different roll.
    if develop.measured_ready(doc["roll"].get("measured"), taste):
        return True
    frames = develop.frames_of(roll, doc)
    if not frames:
        return False
    workers = min(8, os.cpu_count() or 1)

    def job(progress):
        from . import flatfield
        return develop.roll_statistics(frames, taste["gamma"], workers,
                                       progress, flatfield.of(taste))

    dlg = Running("Measuring the roll",
                  f"The base and the balance of {pathlib.Path(roll).name}, "
                  f"over {len(frames)} frames at half size. Kept in the "
                  f"document, so this happens once.", job, parent)
    dlg.exec()
    kept = develop.measured_of(dlg.result, taste)
    if not kept:
        return False
    doc["roll"]["measured"] = kept
    return True


class Running(QDialog):
    """The same window for any job that counts: a title, what it is doing,
    the bar, and the time left. `fn(progress)` runs in a thread and its
    result is `self.result` when the window closes on its own."""

    def __init__(self, title, what, fn, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.result, self.error = None, None
        self.started, self.n, self._shown = time.monotonic(), 0, None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(30, 26, 30, 26)
        lay.setSpacing(10)
        head = QLabel(title + "…")
        head.setStyleSheet("font-size: 15px;")
        lay.addWidget(head)
        self.what = QLabel(what)
        self.what.setStyleSheet("color: #8a8a90;")
        self.what.setWordWrap(True)
        lay.addWidget(self.what)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(False)
        lay.addWidget(self.bar)
        self.left = QLabel("")
        self.left.setStyleSheet("color: #70737c;")
        lay.addWidget(self.left)
        self.setFixedWidth(460)
        self.worker = _Runner(fn)
        self.worker.ticked.connect(self._tick)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _tick(self, i, n):
        self.bar.setRange(0, n)
        self.bar.setValue(i)
        self.n = i
        if i:
            per = (time.monotonic() - self.started) / i
            secs = per * (n - i)
            if self._shown is None or abs(secs - self._shown) > self._shown * 0.2:
                self._shown = secs
            self.left.setText("about {:.0f} s left".format(self._shown)
                              if self._shown > 1 else "")

    def _done(self, result):
        self.result = result
        self.accept()

    def _failed(self, text):
        self.error = text
        self.what.setText(text)
        self.what.setStyleSheet("color:#ff6b60;")
        self.left.setText("close this window")
        self.bar.setRange(0, 1)

    def closeEvent(self, e):
        self.worker.stop = True
        self.worker.wait(2000)
        super().closeEvent(e)


def measure_roll(roll, files, mirrored=True, settings=None, summary=None):
    """The records for these scans, or None if the window was closed.

    Only the scans it is given, never the folder: it is used both for a roll
    nobody has opened and for the handful somebody has just dropped into one
    that is half labelled, and in the second case everything already in the
    document has to come through untouched.
    """
    dlg = Measuring(roll, files, mirrored, settings)
    ok = dlg.exec() == QDialog.Accepted
    if summary is not None and dlg.rolled:
        summary.update(dlg.rolled)
    return dlg.found if ok else None
