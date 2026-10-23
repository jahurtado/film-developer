"""The start screen: what opens when there is nothing to open.

An icon in the Dock arrives with nothing to go on, and so does the bare
command. The screen takes a dropped folder, offers a chooser, lists the
last eight rolls.

What the measurement cannot find out for itself -- whether the folder is
one roll, and the two things that follow from it -- is asked by
`opening.ask`, after a folder is chosen and before anything is measured.
It used to be a checkbox here, above the chooser: answered about no
folder in particular, and never asked at all when the folder came from
the command line.
"""

import json
import os
import pathlib

from . import detect

RECENT = 8


def _store():
    """`recent.json` in the application's own folder, which is named for
    the application now that it has a name (`gui.APP_NAME`).

    It had none, so Qt named it after `argv[0]` -- under `python -m` that
    is `__main__.py` -- and the list lived in
    `Application Support/__main__.py/film-developer/`. Brought over once,
    so naming the program does not empty the start screen."""
    import shutil
    from PySide6.QtCore import QStandardPaths
    d = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    here = os.path.join(d, "recent.json")
    if not os.path.exists(here):
        up = os.path.dirname(d)
        for old in (os.path.join(up, "__main__.py", "film-developer", "recent.json"),
                    os.path.join(up, "python", "film-developer", "recent.json")):
            if os.path.exists(old):
                try:
                    os.makedirs(d, exist_ok=True)
                    shutil.copy2(old, here)
                except OSError:
                    pass
                break
    return here


def recent():
    try:
        with open(_store()) as fh:
            return [p for p in json.load(fh) if os.path.isdir(p)][:RECENT]
    except (OSError, ValueError):
        return []


def remember(path):
    path = os.path.abspath(path)
    paths = [p for p in recent() if p != path]
    paths.insert(0, path)
    try:
        os.makedirs(os.path.dirname(_store()), exist_ok=True)
        with open(_store(), "w") as fh:
            json.dump(paths[:RECENT], fh, indent=1)
    except OSError:
        pass


def forget(path):
    """Off the list, and nothing else touched: the list is a convenience."""
    path = os.path.abspath(path)
    keep = [p for p in recent() if p != path]
    try:
        with open(_store(), "w") as fh:
            json.dump(keep, fh, indent=1)
    except OSError:
        pass


def describe(path):
    """One line under a folder's name: what is in it, as far as is known
    without opening a file."""
    from . import document
    try:
        n = len(detect.scans(path))
    except FileNotFoundError:
        return "not a folder"
    try:
        doc = document.load(path)
        items = doc["items"]
        held = sum(1 for it in items.values() if it["state"] == "held")
        hand = sum(1 for it in items.values() if it.get("manual"))
        done = (pathlib.Path(path) / "developed").is_dir()
        bits = [f"{n} scans", f"{len(items)} measured"]
        if held:
            bits.append(f"{held} held")
        if hand:
            bits.append(f"{hand} by hand")
        if done:
            bits.append("developed")
        return " · ".join(bits)
    except (FileNotFoundError, ValueError):
        return f"{n} scans · not measured yet" if n else "no scans here"


def ask(identity=("", ""), signature=None):
    """The folder chosen on the start screen, or None.

    It used to answer *one roll* here too, with a checkbox above the
    chooser -- which asked about no folder in particular, and which the
    command line never asked at all. That question is now `opening.ask`,
    put once to the folder in hand, from either way in.
    """
    from PySide6.QtCore import QSize, Qt
    from PySide6.QtWidgets import (QApplication, QFileDialog, QFrame,
                                   QHBoxLayout, QLabel, QPushButton,
                                   QVBoxLayout, QWidget)
    from . import __version__
    app = QApplication.instance()
    chosen = []

    class Drop(QFrame):
        def __init__(self):
            super().__init__()
            self.setAcceptDrops(True)
            self.setObjectName("drop")
            self.setMinimumHeight(150)

        @staticmethod
        def _folder(e):
            for u in e.mimeData().urls():
                if u.isLocalFile() and os.path.isdir(u.toLocalFile()):
                    return u.toLocalFile()
            return None

        def dragEnterEvent(self, e):
            if self._folder(e):
                self.setProperty("over", True)
                self.setStyleSheet("")
                e.acceptProposedAction()

        def dragLeaveEvent(self, e):
            self.setProperty("over", False)
            self.setStyleSheet("")

        def dropEvent(self, e):
            self.setProperty("over", False)
            self.setStyleSheet("")
            if self._folder(e):
                take(self._folder(e))

    def take(path):
        path = path.rstrip(os.sep) or os.sep
        try:
            n = len(detect.scans(path))
        except FileNotFoundError:
            n = 0
        if not n:
            problem.setText(f"No raw scans in {os.path.basename(path)} -- the "
                            f"folder should hold the raws themselves "
                            f"(NEF, DNG, CR2, CR3), one level, no subfolders.")
            problem.show()
            return
        chosen.append(path)
        win.close()

    win = QWidget()
    win.setWindowTitle("film-developer")
    win.setMinimumWidth(560)
    win.resize(640, 600)
    v = QVBoxLayout(win)
    # The bottom margin is small because the signature sits on it; the air
    # it needs is its own top margin, below.
    v.setContentsMargins(30, 26, 30, 10)
    v.setSpacing(16)
    head = QLabel(
        f'<div style="font-size:20px"><b>film-developer</b>'
        f'<span style="color:#5a5a60"> {__version__}</span></div>'
        f'<div style="color:#8a8a90;margin-top:4px">Find the frames in a roll '
        f'of photographed negatives, correct them, then develop.</div>')
    head.setTextFormat(Qt.RichText)
    v.addWidget(head)

    drop = Drop()
    dv = QVBoxLayout(drop)
    dv.addStretch(1)
    for text, style in (("Drop a roll here", "color:#c8c8cc;font-size:15px;"),
                        ("a folder of raw scans, one shutter press per file, "
                         "no subfolders", "color:#5a5a60;")):
        lab = QLabel(text)
        lab.setAlignment(Qt.AlignCenter)
        lab.setStyleSheet(style)
        dv.addWidget(lab)
    pick = QPushButton("Choose a folder…")
    pick.clicked.connect(lambda: take(QFileDialog.getExistingDirectory(
        win, "Open a roll -- the folder of raw scans") or ""))
    row = QHBoxLayout()
    row.addStretch(1)
    row.addWidget(pick)
    row.addStretch(1)
    dv.addLayout(row)
    dv.addStretch(1)
    v.addWidget(drop)

    problem = QLabel()
    problem.setWordWrap(True)
    problem.setStyleSheet("color:#ff6b60;")
    problem.hide()
    v.addWidget(problem)

    seen = recent()
    if seen:
        lab = QLabel("RECENT")
        lab.setStyleSheet("color:#5a5a60;font-size:9px;letter-spacing:1.4px;")
        v.addWidget(lab)
    for p in seen:
        row = QFrame()
        row.setObjectName("recent")
        row.setCursor(Qt.PointingHandCursor)
        rh = QHBoxLayout(row)
        rh.setContentsMargins(10, 5, 10, 6)
        rv = QVBoxLayout()
        rv.setSpacing(1)
        name = QLabel(os.path.basename(p))
        name.setStyleSheet("color:#e8e8ea;font-weight:600;")
        where = QLabel(f"{describe(p)}  ·  {os.path.dirname(p)}")
        where.setStyleSheet("color:#70737c;font-size:11px;")
        rv.addWidget(name)
        rv.addWidget(where)
        rh.addLayout(rv, 1)
        off = QPushButton("remove")
        off.setCursor(Qt.PointingHandCursor)
        off.setToolTip('<qt><div style="white-space:normal" width="380">Take '
                       'this folder off the list. Nothing on disk is touched: '
                       'not the scans, not the document, not the positives.'
                       '</div></qt>')
        off.setStyleSheet(
            "QPushButton{background:transparent;color:#5a5a60;border:0;"
            "padding:2px 6px;font-size:11px;}"
            "QPushButton:hover{color:#ff9f6b;}"
            "QPushButton:disabled{color:#4a4a50;}")
        off.clicked.connect(lambda _c=False, q=p, r=row: (forget(q), r.hide()))
        rh.addWidget(off, 0, Qt.AlignTop)
        for w in (name, where, row):
            w.mousePressEvent = lambda _e, q=p: take(q)
        v.addWidget(row)

    foot = QHBoxLayout()
    foot.setContentsMargins(0, 14, 0, 0)
    foot.addStretch(1)
    if signature is not None:
        foot.addWidget(signature)
    v.addStretch(1)
    v.addLayout(foot)

    win.setStyleSheet(
        "#drop{border:1.5px dashed #3a3d45;border-radius:6px;background:#17181c;}"
        "#drop[over=\"true\"]{border-color:#ffd400;background:#1d1e23;}"
        "#recent{border-radius:4px;} #recent:hover{background:#22242a;}")
    from .gui import _centre
    _centre(win)
    win.show()
    app.exec()
    return chosen[0] if chosen else None
