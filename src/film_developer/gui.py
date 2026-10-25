"""Inspect what `detect` found.

The middle step, and the only one a person drives. It edits the document and
nothing else: no image is written here, whatever is pressed.

    film-developer gui ROLL_DIR
    film-developer                       # asks which folder

**This first version reads.** It shows every scan with what the detector found
drawn on it; it does not yet let anything be moved. The command sheet lists
what works and nothing else, because a sheet advertising a key that does
nothing is worse than a short sheet.

Opening a folder with no document in it measures it first, in a window rather
than in the terminal: an icon in the Dock has no terminal, so "it reports in
the terminal" is an answer to nobody.
"""

import os
import pathlib
import sys

from . import detect, document


def add_args(ap):
    ap.add_argument("roll", metavar="ROLL_DIR", nargs="?",
                    help="folder of scans, or the document itself. Left out, "
                         "it asks for one")
    ap.add_argument("--serve", nargs="?", type=int, const=8765, default=None,
                    metavar="PORT",
                    help="Open a control socket on 127.0.0.1 (default port "
                         "8765) so the window can be driven and photographed "
                         "from outside. For reproducing a problem with "
                         "somebody, not for daily use.")


APP_NAME = "film-developer"


def app_icon():
    """The icon, every size in one object, so Qt picks the drawing made for
    the size it needs rather than shrinking the big one -- the 16 px icon is
    a different drawing, without the sprocket holes.

    On macOS the *Dock* icon comes from an .app bundle and not from here, so
    run as a script it stays Python's. This governs the window, the dialogs
    and the start screen.
    """
    import os
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QIcon
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
    ic = QIcon()
    for s in (16, 32, 64, 128, 256, 512):
        p = os.path.join(here, "icon-%d.png" % s)
        if os.path.exists(p):
            ic.addFile(p, QSize(s, s))
    return ic


def keep_reachable(app):
    """Two ways the program could not be left, both on a Mac.

    **A modal sheet behind its window.** A dialog opened without a parent
    -- the roll sheet and the progress windows before the editor exists,
    a file chooser -- is a window of its own, and clicking the window it
    blocks brings that one to the front and buries the sheet: the editor
    then takes no input, ⌘Q is refused because a modal is open, and the
    sheet is nowhere to be seen. So whenever another window of the program
    is activated, or the program itself is, while a modal is open, the
    modal is brought back on top.

    **Ctrl+C in the terminal did nothing.** Python runs a signal handler
    only between its own bytecodes, and Qt's event loop is C++ that may
    not hand control back for as long as nothing happens in the window --
    so the interrupt waited forever. A timer gives the interpreter a turn
    four times a second, and the handler ends the process at once: the
    document is written on every change (`Editor.dirty`), atomically
    (`document.save`), so there is nothing to save, and asking to close
    windows politely is what a hidden modal refuses. The worker processes
    are in the same process group and get the same interrupt."""
    import os
    import signal
    from PySide6.QtCore import QEvent, QObject, QTimer
    from PySide6.QtWidgets import QApplication

    def stop(*_):
        print("\ninterrupted", file=sys.stderr)
        os._exit(130)
    signal.signal(signal.SIGINT, stop)
    tick = QTimer(app)
    tick.timeout.connect(lambda: None)
    tick.start(250)

    from PySide6.QtWidgets import QWidget

    class Front(QObject):
        def eventFilter(self, obj, e):
            # Activations only, not clicks: a click lands in the sheet's own
            # popups (a combo's list) and raising the sheet would close them.
            if e.type() in (QEvent.WindowActivate, QEvent.ApplicationActivate):
                m = QApplication.activeModalWidget()
                behind = (e.type() == QEvent.ApplicationActivate or (
                    isinstance(obj, QWidget) and obj.isWindow()
                    and obj is not m and not m.isAncestorOf(obj))) if m else False
                if behind:
                    QTimer.singleShot(0, lambda: (m.raise_(), m.activateWindow()))
            return False

    app._front = Front(app)            # kept alive with the app
    app.installEventFilter(app._front)
    app._tick = tick


def run(args):
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("the editor needs Qt: uv pip install -e '.[gui]'", file=sys.stderr)
        return 1

    from .cli import identity
    from .editor import Editor, dark_palette
    from .progress import measure_base, measure_roll

    app = QApplication.instance()
    if app is None:
        # **Named before it exists.** macOS builds the application menu
        # inside QApplication's own constructor, from `argv[0]` -- which
        # under `python -m film_developer` is Python's -- so a name set
        # afterwards is too late and the menu bar said "Python". A bundle's CFBundleName wins over this
        # anyway; running from a clone is where it showed.
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.setApplicationName(APP_NAME)
        app = QApplication([APP_NAME])
    app.setApplicationName(APP_NAME)
    keep_reachable(app)
    app.setWindowIcon(app_icon())
    dark_palette(app)

    where = args.roll
    if not where:
        from .start import ask, remember
        from .about import signature
        where = ask(identity(), signature())
        if not where:
            return 0
        remember(where)
    roll = pathlib.Path(where)
    roll = roll.parent if roll.suffix == ".json" else roll

    try:
        doc = document.load(roll)
    except FileNotFoundError:
        doc = document.blank()

    # The folder is looked at every time, not only the first. A roll grows:
    # more scans get dropped into it between sittings, and they have to appear
    # without anybody remembering to run `detect` from a terminal. Only the
    # ones the document does not already have are measured, so nothing that
    # has been through somebody's hands is touched.
    files = detect.scans(roll)
    if not files and not doc["items"]:
        print(f"no raw scans in {roll}", file=sys.stderr)
        return 1
    new = [p for p in files if p.name not in doc["items"]]
    if new:
        # **Asked before the cores start.** Measuring a roll is two minutes
        # of every core, and choosing a folder used to begin it on the spot,
        # on settings nobody had been shown -- and, from the command line, on
        # settings nobody had ever been asked. The sheet is put once, about
        # the folder in hand, and only when there is something to measure:
        # re-opening a roll that is already done asks nothing.
        from . import opening
        said = opening.ask(roll.name, len(new), doc,
                           workers=int(doc["roll"].get("workers")
                                       or min(8, os.cpu_count() or 1)))
        if said is None:
            return 0
        doc["roll"]["one_roll"] = said["one_roll"]
        doc["roll"]["same_size"] = said["same_size"]
        doc["roll"]["same_gap"] = said["same_gap"]
        dev = doc["roll"].setdefault("develop", {})
        dev["base"] = "auto" if said["same_base"] else "frame"
        dev["flat"] = said.get("flat")
        rolled = {}
        found = measure_roll(roll, new, doc["roll"].get("mirrored", True),
                             doc["roll"], rolled)
        if found is None:                    # closing the window abandons it
            return 0
        doc["items"].update(found)
        if rolled.get("clusters"):
            doc["roll"]["rolled"] = rolled
        document.save(doc, roll)

    # The roll's base, once. Every view that shows a base -- the read
    # line's label, `v`, `hold b` -- needs it, and a roll nobody has
    # processed yet had none, so all three said `NOT MEASURED` or fell
    # back to a per-scan reading that is not what the develop uses. It is
    # kept in the document, so this is paid once per roll.
    if measure_base(roll, doc):
        document.save(doc, roll)

    # Dust marked before the brush, made brush marks: once, and needs the
    # roll's measurement to develop the frames it is measured on.
    from .progress import migrate_dust
    if doc["roll"].get("measured") and migrate_dust(roll, doc):
        document.save(doc, roll)

    win = Editor(roll, doc, identity())
    win.resize(1400, 900)
    _centre(win)
    win.show()
    # The roll starts developing itself as soon as it is on screen: a frame
    # takes about a second, and the machine is idle between keypresses.
    win._prefetch_roll()
    if getattr(args, "serve", None):
        # After `show()`, so `/shot` has a window to photograph and
        # `/state` a canvas with a scan on it.
        from . import remote
        remote.serve(win, args.serve)
    return app.exec()


def _centre(win):
    """Every window opens in the middle of the screen -- the one the pointer
    is on, which on two displays is the one the person is working on, and
    which the primary screen need not be."""
    from PySide6.QtGui import QCursor, QGuiApplication
    screen = (QGuiApplication.screenAt(QCursor.pos())
              or QGuiApplication.primaryScreen())
    g = screen.availableGeometry()
    win.move(g.center().x() - win.width() // 2,
             g.center().y() - win.height() // 2)
