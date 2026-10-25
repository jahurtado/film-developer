"""What to write, asked once: the `⌘R` sheet.

`develop` and `apply` used to be two commands in the editor, and they are
one question -- *what do you want out of this roll?* -- with two answers
that are usually both yes. This asks it, and the editor does the work.

There is nothing about **how** the pictures are developed in here. That
is a property of the roll and not of one run, it is the same for every
frame, and it belongs in the develop settings, not in a sheet that
appears when you are already finished deciding.

The one choice that does belong here is the raw crop, because it is the
only one that can lose something: `outer` keeps the whole frame and its
tilted edges and lets the other developer straighten, `inner` cuts the
rectangle inside the tilt and throws the corners away.

The dust marks are a third: the developed pictures always have them
taken out, and the raw DNGs only when *the dust marks out of the raws
too* is ticked -- filled in the mosaic, before any demosaic, so the file
is then no longer only the negative inverted. Kept in the roll's `apply`
settings, off until somebody ticks it, and greyed with no marks to use.

And one more, off every time the sheet opens: the same frames **not
inverted**, into `negatives/` -- a picture beside each positive picture, a
raw beside each positive DNG. It is not kept in the document, because it
is a question about this run and not a fact about the roll.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout,
                               QLabel, QPushButton, QRadioButton, QVBoxLayout)

SUB = "#8a8a90"
# measured on carrete_2: positives 70-76 % of the uncompressed size,
# negatives 56-60 %, every photosite read back the same by LibRaw and Apple
SQUEEZE_SAY = ("bit for bit the same, at about 3/4 of the size (the negatives' "
               "DNGs about 3/5)")
WARN = "#ffb340"


def ask(doc, parent=None, chosen=(), alone=False):
    """The choices, or None if the sheet was closed.

    {"positives": bool, "format": "jpeg"|"tiff16",
     "raws": bool, "crop": "outer"|"inner", "dust": bool, "negatives": bool,
     "numbered": bool, "compress": bool, "only": None | [scan names]}

    `chosen` are the scans chosen in the list, or the one on screen: the
    sheet offers them alone as well as the whole roll. **A photo can be
    processed alone** -- marking its dust, or setting the roll's look on
    it, and wanting to see the file -- without paying the roll for it.
    `alone` (`⇧⌘R`) opens the sheet on that choice.
    """
    from . import dust as dust_
    marks = [len([m for m in (v.get("marks") or []) if dust_.used(m)])
             for v in (doc.get("specks") or {}).values()]
    n_marks, n_scans = sum(marks), sum(1 for m in marks if m)
    dev = dict(doc["roll"].get("develop") or {})
    app = dict(doc["roll"].get("apply") or {})
    out = {}

    dlg = QDialog(parent)
    dlg.setWindowTitle("Process the roll")
    dlg.setModal(True)
    dlg.setMinimumWidth(520)
    v = QVBoxLayout(dlg)
    v.setContentsMargins(26, 22, 26, 18)
    v.setSpacing(12)

    head = QLabel("What should come out of this roll?")
    head.setStyleSheet("font-size: 15px;")
    v.addWidget(head)

    chosen = [n for n in chosen if n in (doc.get("items") or {})
              and doc["items"][n].get("state") != "skipped"
              and doc["items"][n].get("parts")]
    whole = QRadioButton("The whole roll")
    whole.setChecked(True)
    few = None
    if chosen:
        name = (chosen[0].rsplit(".", 1)[0] if len(chosen) == 1
                else f"the {len(chosen)} chosen in the list")
        few = QRadioButton("Only this photo, " + name if len(chosen) == 1
                           else "Only " + name)
        few.setToolTip('<qt><div style="white-space:normal" width="400">The '
                       'rest of the roll\'s files are left as they are, and '
                       'the folder\'s record keeps them.</div></qt>')
        few.setChecked(alone)
        which = QHBoxLayout()
        which.addWidget(whole)
        which.addWidget(few)
        which.addStretch(1)
        v.addLayout(which)

    pos = QCheckBox("Positives, developed")
    pos.setChecked(True)
    v.addWidget(pos)
    fmt_row = QHBoxLayout()
    fmt_row.setContentsMargins(24, 0, 0, 0)
    fmt_lab = QLabel("as")
    fmt_lab.setStyleSheet(f"color: {SUB};")
    fmt = QComboBox()
    fmt.addItems(["jpeg", "tiff16"])
    fmt.setCurrentText(dev.get("format", "jpeg"))
    into = QLabel("into developed/")
    into.setStyleSheet(f"color: {SUB};")
    fmt_row.addWidget(fmt_lab)
    fmt_row.addWidget(fmt)
    fmt_row.addWidget(into)
    fmt_row.addStretch(1)
    v.addLayout(fmt_row)

    raws = QCheckBox("Raw DNGs, the negative inverted on its own mosaic")
    raws.setChecked(True)
    raws.setToolTip('<qt><div style="white-space:normal" width="400">A CFA '
                    'file another raw developer opens as it would a camera '
                    'raw: its own demosaic, its own colour, its own curves. '
                    'Written into frames/.</div></qt>')
    v.addWidget(raws)

    crop_row = QVBoxLayout()
    crop_row.setContentsMargins(24, 0, 0, 0)
    crop_row.setSpacing(4)
    outer = QRadioButton("the whole frame, edges and all (the developer straightens)")
    inner = QRadioButton("the rectangle inside the tilt")
    (inner if app.get("crop") == "inner" else outer).setChecked(True)
    crop_row.addWidget(outer)
    crop_row.addWidget(inner)
    warn = QLabel("")
    warn.setWordWrap(True)
    warn.setStyleSheet(f"color: {WARN};")
    crop_row.addWidget(warn)
    v.addLayout(crop_row)

    def say():
        warn.setText(
            "The frame is tilted, so the corners of that rectangle fall "
            "inside it: part of the picture is thrown away, and no developer "
            "can put it back." if inner.isChecked() else "")
    outer.toggled.connect(say)
    say()

    clean = QCheckBox("The dust marks out of the raws too")
    clean.setChecked(bool(app.get("dust")) and n_marks > 0)
    clean.setToolTip('<qt><div style="white-space:normal" width="400">The '
                     'specks you marked (s) are always taken out of the '
                     'developed pictures. Ticked, they are also filled in '
                     'the raw DNG: on the Bayer mosaic itself, before any '
                     'demosaic, the four colour planes from one patch '
                     'beside the speck. The DNG is then no longer only the '
                     'negative inverted.</div></qt>')
    crop_row.addWidget(clean)
    clean_say = QLabel(
        f"{n_marks} mark{'s' if n_marks != 1 else ''} on {n_scans} "
        f"scan{'s' if n_scans != 1 else ''}, filled on the mosaic before any "
        "demosaic" if n_marks else
        "no dust marked on this roll yet: s in the window marks it")
    clean_say.setContentsMargins(24, 0, 0, 0)
    clean_say.setStyleSheet(f"color: {SUB};")
    crop_row.addWidget(clean_say)

    squeeze = QCheckBox("Lossless compression in the DNGs")
    squeeze.setChecked(bool(app.get("compress", True)))
    squeeze.setToolTip('<qt><div style="white-space:normal" width="400">The '
                       'mosaic written as lossless JPEG, the compression '
                       'cameras and Adobe use in a raw DNG: every photosite '
                       'reads back the same, bit for bit, in a smaller file. '
                       'Unticked, the mosaic goes in as it is.</div></qt>')
    crop_row.addWidget(squeeze)
    squeeze_say = QLabel(SQUEEZE_SAY)
    squeeze_say.setContentsMargins(24, 0, 0, 0)
    squeeze_say.setStyleSheet(f"color: {SUB};")
    crop_row.addWidget(squeeze_say)

    neg = QCheckBox("The negatives too, not inverted, into negatives/")
    neg.setChecked(False)
    neg.setToolTip('<qt><div style="white-space:normal" width="400">Each '
                   'frame cut out the same way, as the sensor saw it: the '
                   'orange mask, the base and the exposure of the copy all '
                   'still there. A picture beside each positive picture '
                   '(the camera\'s white balance and colour, nothing '
                   'else), and a raw DNG beside each positive DNG (the '
                   'sensor\'s own values, untouched).</div></qt>')
    v.addWidget(neg)
    neg_say = QLabel("")
    neg_say.setContentsMargins(24, 0, 0, 0)
    neg_say.setStyleSheet(f"color: {SUB};")
    v.addWidget(neg_say)

    def enable():
        for w in (fmt, fmt_lab, into):
            w.setEnabled(pos.isChecked())
        for w in (outer, inner, warn):
            w.setEnabled(raws.isChecked())
        clean.setEnabled(raws.isChecked() and n_marks > 0)
        clean_say.setEnabled(raws.isChecked())
        squeeze.setEnabled(raws.isChecked())
        squeeze_say.setEnabled(raws.isChecked())
        neg.setEnabled(pos.isChecked() or raws.isChecked())
        what = [x for x, on in (("pictures", pos.isChecked()),
                                ("raw DNGs", raws.isChecked())) if on]
        neg_say.setText(" and ".join(what) + ", the same cut as the positives"
                        if neg.isChecked() and what else "")
        go.setEnabled(pos.isChecked() or raws.isChecked())
    pos.toggled.connect(enable)
    raws.toggled.connect(enable)
    neg.toggled.connect(enable)

    from . import document as document_
    first = (document_.order_of(doc) or ["DSC_0001.NEF"])[0]
    width = max(2, len(str(len(doc.get("items") or {}))))
    numbered = QCheckBox("Shot number in front of every name")
    numbered.setChecked(bool(doc["roll"].get("numbered", True)))
    numbered.setToolTip('<qt><div style="white-space:normal" width="400">Each '
                        'file named by its place in the list, which is the '
                        'order the roll was shot in, so a folder sorted by '
                        'name is the roll in order. Files an earlier run '
                        'wrote under other names are removed.</div></qt>')
    v.addWidget(numbered)
    num_say = QLabel(f"{1:0{width}d}_{first.rsplit('.', 1)[0]}.jpg, in the "
                     "list's order")
    num_say.setContentsMargins(24, 0, 0, 0)
    num_say.setStyleSheet(f"color: {SUB};")
    v.addWidget(num_say)
    numbered.toggled.connect(num_say.setVisible)
    num_say.setVisible(numbered.isChecked())

    note = QLabel("How they are developed is the roll's own setting, the same "
                  "for every frame, and is not asked here.")
    note.setWordWrap(True)
    note.setStyleSheet(f"color: {SUB};")
    v.addWidget(note)

    row = QHBoxLayout()
    row.addStretch(1)
    cancel = QPushButton("Cancel")
    cancel.clicked.connect(dlg.reject)
    go = QPushButton("Process")
    go.setDefault(True)
    row.addWidget(cancel)
    row.addWidget(go)
    v.addLayout(row)
    enable()

    def take():
        out.update(positives=pos.isChecked(), format=fmt.currentText(),
                   raws=raws.isChecked(),
                   crop="inner" if inner.isChecked() else "outer",
                   dust=clean.isEnabled() and clean.isChecked(),
                   negatives=neg.isEnabled() and neg.isChecked(),
                   numbered=numbered.isChecked(),
                   compress=squeeze.isChecked(),
                   only=chosen if few is not None and few.isChecked() else None)
        dlg.accept()
    go.clicked.connect(take)
    dlg.exec()
    return out or None
