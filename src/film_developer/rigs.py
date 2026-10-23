"""The copy stand's light falloff, from the window: make a calibration,
and choose which one a roll was scanned under.

Everything `flatfield` does is reachable here; the command `flat` is the
same thing for a script. Two sheets:

- **New calibration**: pick the blank frames, name the stand, and it is
  fitted, shown -- how well it fits, how well it predicts a blank it was
  not fitted on, how much the light falls at the corners -- and kept only
  if you say so. A calibration nobody looked at is one nobody can trust.
- **Light falloff** (`⌘L` in the editor): which calibration this roll
  divides out, with what each one is -- its camera, its blanks, its fit --
  and a warning where it is for another camera or sensor, because then it
  is another stand's light.
"""

import pathlib

from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QVBoxLayout)

from . import flatfield

SUB = "#8a8a90"
WARN = "#ffb340"
NONE = "no light falloff calibration"


def _sheet(title, parent, width=540):
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setModal(True)
    dlg.setMinimumWidth(width)
    v = QVBoxLayout(dlg)
    v.setContentsMargins(26, 22, 26, 18)
    v.setSpacing(12)
    return dlg, v


def _grey(text=""):
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(f"color: {SUB};")
    return lab


# How even a corrected blank has to come out to be called good, or usable.
# Judgement, anchored on one measurement: the D5600 stand's four blanks
# predict each other to 0.7 %, and what is left there is the film base's
# own texture and dust, not light. Past these the blanks themselves are
# the likelier fault -- dust, a holder that moved, light leaking in.
EVEN_GOOD = 1.0
EVEN_FAIR = 2.5


def verdict(cal):
    """(head, detail): what a calibration does, conclusion first.

    It said the numbers first -- camera, sensor, two rows of three
    percentages -- and the one thing that mattered, that the light comes
    out even, was the last figure on the last line. Read that way a good
    calibration looked like a report of what went wrong."""
    f = cal.get("fit") or {}
    rms = f.get("rms_pct") or {}
    loo = f.get("loo_rms_pct") or {}
    g = (f.get("falloff_ev_at_coverage_corners") or {}).get("G") or []
    n = len(cal.get("blanks") or [])
    even = max((loo or rms).values()) if (loo or rms) else None
    fall = -min(g) if g else None
    if even is None:
        head = "Calibrated."
    elif even < EVEN_GOOD:
        head = f"Calibrated: corrected, the light is even to within {even:.1f} %."
    elif even < EVEN_FAIR:
        head = f"Calibrated, usable: the light comes out even to within {even:.1f} %."
    else:
        head = f"The light still varies {even:.1f} % after correcting."
    lines = []
    if fall is not None:
        lines.append(f"The stand loses up to {fall:.1f} EV towards the corners "
                     f"({100 * (1 - 2 ** -fall):.0f} % less light than the "
                     f"centre); that is what it takes out.")
    if even is not None and even >= EVEN_GOOD:
        lines.append("More blanks, or blanks with less dust, would tighten it."
                     if even < EVEN_FAIR else
                     "Look at the blanks: dust, a holder that moved between "
                     "them, or light leaking in.")
    if loo:
        lines.append(f"Checked by fitting on all but one blank and predicting "
                     f"the one left out, each in turn ({n} blanks).")
    elif n == 1:
        lines.append("From one blank, so nothing to check it against; "
                     "two or more say how well it predicts.")
    lines.append(f"{cal.get('camera') or 'Unknown camera'}, sensor "
                 f"{cal['sensor'][0]}×{cal['sensor'][1]}, made {cal.get('created', '')}.")
    return head, "\n".join(lines)


def describe(cal):
    """The verdict and its detail as one text, for the Light falloff sheet."""
    head, detail = verdict(cal)
    return head + "\n" + detail


def mismatch(cal, scan):
    """Why this calibration is not for this scan, or ""."""
    if not scan:
        return ""
    import rawpy
    try:
        with rawpy.imread(str(scan)) as r:
            H, W = r.raw_image_visible.shape
        flatfield.check(cal, scan, W, H)
    except ValueError as e:
        return str(e)
    except Exception:
        return ""
    return ""


def new(parent=None, start=None):
    """Make a calibration from blank frames. Returns its name, or None."""
    from .progress import Running
    from .sheets import ask, centre

    files, _ = QFileDialog.getOpenFileNames(
        parent, "The blank frames: light through unexposed film, on the stand "
                "as it scans", str(start or pathlib.Path.home()),
        "Raw files (" + " ".join(f"*{e} *{e.upper()}" for e in sorted(flatfield.RAWS))
        + ")")
    if not files:
        return None
    paths = [pathlib.Path(f) for f in files]
    model = flatfield.model_of(paths[0]) or "camera"
    suggest = f"{model.split()[-1]} {paths[0].parent.parent.name if paths[0].parent.name.lower() in ('blank', 'blanks', 'flat', 'flats') else paths[0].parent.name}"
    suggest = suggest.replace("/", "-").strip()

    dlg, v = _sheet("New calibration", parent)
    head = QLabel("Name this copy stand")
    head.setStyleSheet("font-size: 15px;")
    v.addWidget(head)
    v.addWidget(_grey(
        f"{len(paths)} blank frame{'s' if len(paths) != 1 else ''} from "
        f"{paths[0].parent}. The falloff is the stand's — light, lens, "
        f"magnification, camera — so it is named for the stand and offered "
        f"to every roll from now on. Several blanks are better than one: "
        f"each is fitted with its own level and the fit says how well it "
        f"predicts the others."))
    name = QLineEdit(suggest)
    name.selectAll()
    v.addWidget(name)
    clash = QLabel("")
    clash.setStyleSheet(f"color: {WARN};")
    v.addWidget(clash)
    row = QHBoxLayout()
    row.addStretch(1)
    cancel = QPushButton("Cancel")
    go = QPushButton("Calibrate")
    go.setDefault(True)
    row.addWidget(cancel)
    row.addWidget(go)
    v.addLayout(row)

    def check():
        n = name.text().strip()
        bad = (not n) or any(c in n for c in "/\\:")
        go.setEnabled(not bad)
        clash.setText("There is a calibration of that name; it will be replaced."
                      if n and not bad and flatfield.path_of(n).exists() else "")
    name.textChanged.connect(check)
    check()
    cancel.clicked.connect(dlg.reject)
    go.clicked.connect(dlg.accept)
    dlg.adjustSize()
    centre(dlg)
    if dlg.exec() != QDialog.Accepted:
        return None
    nm = name.text().strip()

    def job(progress):
        return flatfield.fit(nm, paths, progress=progress)
    run = Running("Calibrating the light",
                  f"Fitting the falloff of {len(paths)} blank frames.", job, parent)
    run.exec()
    if run.error is not None or run.result is None:
        ask("New calibration", "The blanks could not be fitted.",
            str(run.error or "It was stopped."), (("ok", "OK", "go"),), parent)
        return None
    cal = run.result
    warn = ""
    if (cal["fit"].get("used_share") or 0) < 0.5:
        warn = ("\n\nLess than half of each blank could be used — most of "
                "it is clipped, holder or dust — so the corners are guessed "
                "more than measured.")
    head, detail = verdict(cal)
    keep = ask("New calibration", head,
               detail + warn + f"\n\nKeep it as “{nm}”?",
               (("cancel", "Discard", "cancel"), ("keep", "Keep", "go")), parent)
    if keep != "keep":
        return None
    flatfield.save(cal, force=True)
    return nm


def choose(current, scan=None, parent=None, start=None):
    """Which calibration a roll is divided by. Returns ("set", name or
    None), or None if the sheet was closed."""
    from .sheets import centre
    if current and flatfield._legacy(current):
        current = None          # the old flat field's `auto` and kin: none
    dlg, v = _sheet("Light falloff", parent)
    head = QLabel("What was this roll scanned on?")
    head.setStyleSheet("font-size: 15px;")
    v.addWidget(head)
    v.addWidget(_grey(
        "The copy stand's light falloff, divided out of every scan before "
        "the base and the balance are read. Changing it measures the roll "
        "again, which takes a moment; the frames stay where they are."))
    row = QHBoxLayout()
    combo = QComboBox()
    make = QPushButton("New calibration…")
    row.addWidget(combo, 1)
    row.addWidget(make)
    v.addLayout(row)
    info = _grey()
    v.addWidget(info)
    warn = QLabel("")
    warn.setWordWrap(True)
    warn.setStyleSheet(f"color: {WARN};")
    v.addWidget(warn)

    def fill(select):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(NONE)
        have = flatfield.names()
        combo.addItems(have)
        if select and select not in have:
            combo.addItem(f"{select} (missing)")
        combo.setCurrentText(select if select in have
                             else (f"{select} (missing)" if select else NONE))
        combo.blockSignals(False)
        show()

    def picked():
        t = combo.currentText()
        return None if t == NONE else t.removesuffix(" (missing)")

    def show():
        nm = picked()
        if nm is None:
            info.setText("The scans are developed as the light fell on them.")
            warn.setText("")
            use.setEnabled(True)
            return
        try:
            cal = flatfield.load(nm)
        except (OSError, ValueError) as e:
            info.setText("")
            warn.setText(str(e))
            use.setEnabled(False)
            return
        info.setText(describe(cal))
        bad = mismatch(cal, scan)
        # another stand's light: dividing by it would invent a falloff
        warn.setText(f"Not for this roll: {bad}." if bad else "")
        use.setEnabled(not bad)

    def made():
        nm = new(dlg, start)
        if nm:
            fill(nm)
    combo.currentIndexChanged.connect(lambda _i: show())
    make.clicked.connect(made)

    out = {}
    buttons = QHBoxLayout()
    buttons.addStretch(1)
    cancel = QPushButton("Cancel")
    use = QPushButton("Use")
    use.setDefault(True)
    buttons.addWidget(cancel)
    buttons.addWidget(use)
    v.addLayout(buttons)
    cancel.clicked.connect(dlg.reject)
    fill(current)

    def take():
        out["flat"] = picked()
        dlg.accept()
    use.clicked.connect(take)
    dlg.adjustSize()
    centre(dlg)
    dlg.exec()
    return ("set", out["flat"]) if out else None
