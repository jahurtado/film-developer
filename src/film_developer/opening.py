"""What the roll is, asked once, before anything is measured.

Choosing a folder used to start the measurement on the spot -- two
minutes of every core, on settings nobody had been shown. The one
question that was asked, *one roll*, sat on the start screen **before** a
folder was chosen, so it was answered about no folder in particular, and
the command line never asked it at all. This sheet is the same question
asked about the folder in hand, from either way in, and it is the last
thing before the work starts.

There is one fact here and the rest follow from it: **whether the scans
are all from one roll**. One roll is one camera, one gate, one film, so
every frame is the same size, they vote on each other's, and the
unexposed film between them is one colour at one density. A folder that
mixes rolls has none of that, and each scan is measured and developed on
its own.

The three under it are the places where that fact is spent, and they are
separate because each can be wrong on its own: a gate that was not moved
but a film changed mid-roll, or a camera that advances unevenly on a roll
whose frames are all one size. They are the
roll's, not this run's -- they go into the document and the editor keeps
using them.

Nothing about how the pictures are *developed* is here. That is
`⌘R` and the roll's develop settings; this sheet is about what the
folder **is** -- and that includes **what it was scanned on**: the copy
stand's light falloff, calibrated once from blank frames
(`film-developer flat`, `flatfield.py`) and named, is a fact about how
these scans were taken, like the roll being one roll. Not under the roll
box: a folder of mixed rolls shot on one stand has one stand.
"""

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout,
                               QLabel, QPushButton, QVBoxLayout)

SUB = "#8a8a90"
WARN = "#ff5f56"

# What a scan costs to measure: 2.6 s per core per pass, two passes.
# Said out loud because this sheet is the button that
# starts a job of minutes, and a dialog that starts one says how long it
# takes and what it writes.
SECONDS_PER_SCAN_PASS = 2.6


def how_long(n, workers):
    """A plain-words estimate of the measurement, or "" when it is quick."""
    secs = 2 * n * SECONDS_PER_SCAN_PASS / max(1, workers)
    if secs < 45:
        return "under a minute"
    if secs < 90:
        return "about a minute"
    return f"about {round(secs / 60)} minutes"


def ask_one(scan, doc, hand=False, rolled=None, parent=None):
    """`⌘D` on one scan: how much of the roll it may lean on.

    {"one_roll": bool, "same_size": bool, "same_gap": bool} or None.

    Measuring one scan alone is the worst case the detector has -- a dark
    frame has no edge to find, and a strip of two half frames with the gap
    between them is a 3:2 that reads as one full frame. So by default it
    is measured the way the roll's own pass measured it. But the reason to
    press `⌘D` is sometimes that the roll got this one wrong, and then
    leaning on the roll again is the last thing wanted; so it is asked
    rather than assumed, with the same three answers the roll was opened
    with and remembered from it.

    The answers are **this press only**: they are not written to the
    document, because a choice made about one scan is not a fact about the
    folder. The film base is not here at all -- it is how a frame is
    developed, not how it is found.
    """
    roll = doc.get("roll") or {}
    out = {}

    dlg = QDialog(parent)
    dlg.setWindowTitle("Measure this scan again")
    dlg.setModal(True)
    dlg.setMinimumWidth(520)
    v = QVBoxLayout(dlg)
    v.setContentsMargins(26, 22, 26, 18)
    v.setSpacing(12)

    head = QLabel(f"Measure {scan} again?")
    head.setStyleSheet("font-size: 15px;")
    v.addWidget(head)

    said = ("What you placed on this scan by hand goes, and it cannot be "
            "recovered -- \u2318Z puts it back." if hand else
            "Nothing else in the roll is touched.")
    if not (rolled or {}).get("clusters"):
        said += (" There is nothing measured about this roll yet, so the "
                 "scan is read on its own whatever is ticked.")
    note = QLabel(said)
    note.setWordWrap(True)
    note.setStyleSheet(f"color: {SUB};")
    v.addWidget(note)

    same = QCheckBox("Measure it the way the roll was measured")
    same.setChecked(bool(roll.get("one_roll", True)))
    same.setToolTip(
        '<qt><div style="white-space:normal" width="400">The roll\'s '
        'format, the height of a frame, the gap and where the row sits, '
        'as the pass over the folder settled them. On its own a scan has '
        'none of that, which is what a dark frame most needs -- and a '
        'strip of two half frames plus its gap is a 3:2, so alone it can '
        'come back as one full frame. Off, this scan is read from nothing '
        'but itself, which is what to do when the roll is what got it '
        'wrong.</div></qt>')
    v.addWidget(same)

    subs = QVBoxLayout()
    subs.setContentsMargins(24, 0, 0, 0)
    subs.setSpacing(6)
    size = QCheckBox("Give it the roll's frame size")
    size.setChecked(bool(roll.get("same_size", True)))
    gap = QCheckBox("Give it the roll's gap, if it is half frame")
    gap.setChecked(bool(roll.get("same_gap", False)))
    for w in (size, gap):
        subs.addWidget(w)
    # **What these two do not govern**, which is what makes them look
    # broken on the scan that needs the roll most. A side with no edge to
    # measure is *completed* from the roll -- the opposite side plus the
    # width, the neighbour plus the gap -- and that happens whenever the
    # box above is on, because a side nobody could measure has to come
    # from somewhere. These two only correct a side that **was** measured
    # and disagrees. On SDIM0932 of roma-half, the blank exposure, both
    # frames have three sides with no edge at all: it comes out at the
    # roll's height and the roll's gap with both of these off, and ticking
    # them changes nothing.
    fine = QLabel("A side with no edge to measure is taken from the roll "
                  "whatever these two say — that is how a blank exposure "
                  "gets placed at all. These only correct a side that was "
                  "measured and disagrees.")
    fine.setWordWrap(True)
    fine.setStyleSheet(f"color: {SUB};")
    subs.addWidget(fine)
    v.addLayout(subs)

    def enable():
        on = same.isChecked()
        for w in (size, gap):
            if not on and w.isChecked():
                w.setProperty("was", True)
            w.setChecked(bool(w.property("was")) if on else False)
            w.setEnabled(on)
    same.toggled.connect(enable)
    for w in (size, gap):
        w.setProperty("was", w.isChecked())
    enable()

    row = QHBoxLayout()
    row.addStretch(1)
    cancel = QPushButton("Cancel")
    cancel.clicked.connect(dlg.reject)
    go = QPushButton("Measure")
    go.setDefault(True)
    row.addWidget(cancel)
    row.addWidget(go)
    v.addLayout(row)

    def take():
        on = same.isChecked()
        out.update(one_roll=on,
                   same_size=bool(on and size.isChecked()),
                   same_gap=bool(on and gap.isChecked()))
        dlg.accept()
    go.clicked.connect(take)
    from .sheets import centre
    dlg.adjustSize()
    centre(dlg)
    dlg.exec()
    return out or None


def ask(name, n, doc, workers=8, parent=None, warning="", action="Measure"):
    """The roll's own facts, or None if the sheet was closed.

    {"one_roll": bool, "same_size": bool, "same_gap": bool,
     "same_base": bool, "flat": name or None}

    `name` is the folder's name and `n` how many scans are about to be
    measured -- not how many are in the folder: a roll grows between
    sittings and the ones already in the document are not touched.
    Defaults come from the document, so a roll answered once is not asked
    the same thing differently the second time.
    """
    roll = doc.get("roll") or {}
    out = {}

    dlg = QDialog(parent)
    dlg.setWindowTitle("Open a roll")
    dlg.setModal(True)
    dlg.setMinimumWidth(540)
    v = QVBoxLayout(dlg)
    v.setContentsMargins(26, 22, 26, 18)
    v.setSpacing(12)

    head = QLabel(f"Measure {name}?")
    head.setStyleSheet("font-size: 15px;")
    v.addWidget(head)

    says = QLabel(f"{n} scan{'s' if n != 1 else ''} to measure, "
                  f"{how_long(n, workers)}. It writes frames.json in the "
                  f"folder and reads everything else.")
    says.setWordWrap(True)
    says.setStyleSheet(f"color: {SUB};")
    v.addWidget(says)
    if warning:
        # **What it cannot give back**, said apart from what it costs.
        # `⇧⌘D` asks this same sheet, because measuring a whole folder
        # again is the same act with the same four answers -- it was a
        # separate confirmation that named the loss and then measured on
        # whatever the document happened to hold.
        lost = QLabel(warning)
        lost.setWordWrap(True)
        lost.setStyleSheet(f"color: {WARN};")
        v.addWidget(lost)

    same = QCheckBox("All the scans are from the same roll")
    same.setChecked(bool(roll.get("one_roll", True)))
    same.setToolTip(
        '<qt><div style="white-space:normal" width="400">One camera, one '
        'gate, one film. Then the frames are evidence about each other: a '
        'second pass measures every scan again with what the roll says, '
        'which is what places a blank exposure beside a visible one. Off, '
        'each scan is measured on its own -- for a folder that mixes '
        'rolls, where nothing one frame says is evidence about '
        'another.</div></qt>')
    v.addWidget(same)

    subs = QVBoxLayout()
    subs.setContentsMargins(24, 0, 0, 0)
    subs.setSpacing(6)

    size = QCheckBox("Every frame is the same size")
    size.setChecked(bool(roll.get("same_size", True)))
    size.setToolTip(
        '<qt><div style="white-space:normal" width="400">Every crop of the '
        'roll the same height, so the files come out one size. A frame '
        'whose measured height disagrees with the roll\'s is set to it, '
        'keeping its centre, so neither edge is preferred.<br><br>'
        'It buys uniformity and not accuracy: measured against the frames '
        'a person placed, it changes nothing on two of the three labelled '
        'rolls and on the third it costs three frames more than 25 px off '
        'the top, where none lost that much. At the 1.5-2 % it fires on, '
        'what is usually wrong is where the frame sits rather than how '
        'tall it is, and moving both edges mends one end and spoils the '
        'other.<br><br>One size for the whole roll, the size most of its '
        'frames have: a frame mismeasured into the rebate cannot pull the '
        'roll its way. A frame wider than that has one side wrong, and the '
        'side the roll\'s width finds an edge for is moved. A narrower '
        'frame is left alone: the exposure can end before the gate does '
        '-- one Portra frame is 7 % narrow with its height right -- so '
        'forcing it would invent picture that is not there.<br><br>Untick '
        'it for a roll whose copy stand moved partway.</div></qt>')
    subs.addWidget(size)

    gap = QCheckBox("Half-frame rows all have the same gap")
    gap.setChecked(bool(roll.get("same_gap", False)))
    gap.setToolTip(
        '<qt><div style="white-space:normal" width="400">The two exposures '
        'of a half-frame row are one advance of the film past one gate, so '
        'the unexposed strip between them is the same on every row. '
        '<b>Every</b> row is set to the roll\'s gap, about the gap\'s own '
        'centre -- each frame keeps the width it measured and moves half '
        'the correction, because which of the two inner edges is the wrong '
        'one is not known. There is no tolerance: a window of "near enough '
        'to the roll\'s" would be a claim about the camera, and it left '
        'two neighbouring scans 17.8 px apart with this ticked.<br><br>'
        'It costs accuracy where the advance really does vary. On the one '
        'half-frame roll measured against frames placed by hand, the gaps '
        'run from 166 to 231 px, and forcing them all to the roll\'s takes '
        'the p90 of the gap error from 26 to 43 px and the p90 of the left '
        'and right edges from 11 and 14 to 21 and 22. The score does not '
        'move.<br><br>Nothing to do on a '
        'full-frame roll, where there is one frame per scan and no '
        'gap.</div></qt>')
    subs.addWidget(gap)

    base = QCheckBox("The film base is the same across the roll")
    base.setChecked((roll.get("develop") or {}).get("base", "auto") != "frame")
    base.setToolTip(
        '<qt><div style="white-space:normal" width="400">Unexposed film is '
        'one thing and a roll is a piece of it, so the base is read from '
        'the whole roll and every frame is developed against that one '
        'number. Off, each scan uses its own unexposed film, round the '
        'frame -- for scans copied under different lights -- and the '
        'frame\'s lightest pixels only where the scan shows no film. '
        '⇧⌘D in the editor asks this again.</div></qt>')
    subs.addWidget(base)
    v.addLayout(subs)

    from . import flatfield
    have = flatfield.names()
    now = (roll.get("develop") or {}).get("flat")
    if now and flatfield._legacy(now):
        now = None              # the old flat field's `auto` and kin: none
    rig_row = QHBoxLayout()
    rig_lab = QLabel("Scanned on")
    rig = QComboBox()
    NONE = "no light falloff calibration"
    rig.addItems([NONE] + have)
    if now in have:
        rig.setCurrentText(now)
    elif now:
        # named in the document and not in the library: say so rather than
        # quietly offering "none", which would drop it on Measure
        rig.addItem(f"{now} (missing)")
        rig.setCurrentIndex(rig.count() - 1)
    make = QPushButton("New…")
    rig_row.addWidget(rig_lab)
    rig_row.addWidget(rig, 1)
    rig_row.addWidget(make)
    v.addLayout(rig_row)
    rig_say = QLabel(
        "The copy stand's light falloff, divided out before the base and "
        "the balance are read. Made once per setup from blank frames, and "
        "offered to every roll from then on; ⌘L changes it later.")

    def made():
        from . import rigs
        nm = rigs.new(dlg)
        if nm:
            if rig.findText(nm) < 0:
                rig.insertItem(1, nm)
            rig.setCurrentText(nm)
    make.clicked.connect(made)
    rig_say.setWordWrap(True)
    rig_say.setStyleSheet(f"color: {SUB};")
    v.addWidget(rig_say)

    def enable():
        # The two below are how the one fact above is spent: with the
        # scans not from one roll there is no roll to take a size or a
        # base from, and offering the choice would be offering nothing.
        #
        # **Cleared as well as greyed.** Disabled alone they stay ticked,
        # and a ticked grey box reads as a thing that is still going to
        # happen. What they were is put back if the roll comes back on, so
        # unticking the top one and changing your mind does not quietly
        # answer the other two differently.
        on = same.isChecked()
        for w in (size, gap, base):
            if not on and w.isChecked():
                w.setProperty("was", True)
            w.setChecked(bool(w.property("was")) if on else False)
            w.setEnabled(on)
    same.toggled.connect(enable)
    for w in (size, gap, base):
        w.setProperty("was", w.isChecked())
    enable()

    row = QHBoxLayout()
    row.addStretch(1)
    cancel = QPushButton("Cancel")
    cancel.clicked.connect(dlg.reject)
    go = QPushButton(action)
    go.setDefault(True)
    if warning:
        go.setStyleSheet(f"color: {WARN};")
    row.addWidget(cancel)
    row.addWidget(go)
    v.addLayout(row)

    def take():
        on = same.isChecked()
        out.update(one_roll=on,
                   same_size=bool(on and size.isChecked()),
                   same_gap=bool(on and gap.isChecked()),
                   same_base=bool(on and base.isChecked()),
                   flat=None if rig.currentText() == NONE
                   else rig.currentText().removesuffix(" (missing)"))
        dlg.accept()
    go.clicked.connect(take)
    from .sheets import centre
    dlg.adjustSize()
    centre(dlg)
    dlg.exec()
    return out or None
