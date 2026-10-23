"""The questions the window asks, in the window's own clothes.

These were `QMessageBox`, which on a Mac draws the system alert: a folder
icon, a blue capsule button, a typeface and a spacing that belong to the
operating system and not to this program. The rest of the program
has a look of its own -- the process sheet, the roll sheet, the progress window
are all plain `QDialog`s with the dark palette, a 15 px head, a grey line
of consequence under it and the choice last. An alert that looks like a
different program in the middle of this one is the one dialog nobody
reads.

So: the same shape for every question. A head that asks it in one line,
the grey text that says what it costs, the buttons at the bottom right,
and the safe one first. And on the screen the pointer is on, in the
middle, like every other window this program opens -- a modal sheet
inheriting its parent's position lands wherever the editor happens to be,
which on two displays is not where the person is looking.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout)

SUB = "#8a8a90"
DANGER = "#ff5f56"


def centre(win):
    """In the middle of the screen the pointer is on, which on two
    displays is the one the person is working on and which the primary
    screen need not be."""
    screen = (QGuiApplication.screenAt(QCursor.pos())
              or QGuiApplication.primaryScreen())
    g = screen.availableGeometry()
    win.move(g.center().x() - win.width() // 2,
             g.center().y() - win.height() // 2)


def ask(title, question, detail="", choices=(("ok", "OK", "go"),),
        parent=None, width=540):
    """Put a question and return the key of what was chosen, or None.

    `choices` is (key, label, kind) in the order they should be read, and
    `kind` is what the button is: `go` for the one that does the thing,
    `danger` for one that loses something, `quiet` for an alternative,
    `cancel` for the way out. The dialog is laid out with the buttons at
    the bottom right and the cancel first, so the safe one is where the
    hand goes and the destructive one is not under the return key.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setModal(True)
    dlg.setMinimumWidth(width)
    v = QVBoxLayout(dlg)
    v.setContentsMargins(26, 22, 26, 18)
    v.setSpacing(12)

    head = QLabel(question)
    head.setWordWrap(True)
    head.setStyleSheet("font-size: 15px;")
    v.addWidget(head)
    if detail:
        body = QLabel(detail)
        body.setWordWrap(True)
        body.setTextFormat(Qt.PlainText)
        body.setStyleSheet(f"color: {SUB};")
        v.addWidget(body)

    picked = {}
    row = QHBoxLayout()
    row.addStretch(1)
    for key, label, kind in choices:
        b = QPushButton(label)
        if kind == "danger":
            b.setStyleSheet(f"color: {DANGER};")
        if kind == "go":
            b.setDefault(True)
        b.clicked.connect(
            lambda _c=False, k=key: (picked.update(key=k), dlg.accept()))
        row.addWidget(b)
    v.addLayout(row)

    dlg.adjustSize()
    centre(dlg)
    dlg.exec()
    return picked.get("key")


def say(title, what, detail="", parent=None):
    """Something with nothing to decide: one way out, and it is the
    return key."""
    return ask(title, what, detail, (("ok", "OK", "go"),), parent, width=460)
