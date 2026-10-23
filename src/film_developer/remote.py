#!/usr/bin/env python3
"""A control socket for the editor, so a problem can be reproduced instead of
described.

    film-developer gui ROLL --serve

Off unless asked for, bound to 127.0.0.1 and to nothing else. It exists so that
whoever is helping with a bug can drive the same window the reporter is looking
at: press the keys, read the state, take the picture. A screenshot answers in
one round trip what a paragraph cannot -- what the window actually shows.

    curl localhost:8765/state
    curl -X POST localhost:8765/key  -d '{"key": "d"}'
    curl -X POST localhost:8765/key  -d '{"key": "r", "mods": ["ctrl"]}'
    curl -X POST localhost:8765/call -d '{"name": "_goto", "args": ["022.DNG"]}'
    curl -o shot.png localhost:8765/shot

Three things in it are particular to this program:

- **`d` renders off the window's thread**, so a shot taken the instant after
  the key is a shot of a canvas with nothing on it yet. `/state` carries
  `developing`, and `/shot?wait=1` waits for the render before it draws.
- **The `⌘` commands are `QAction`s, not `QShortcut`s.** Qt matches a shortcut
  on the way in from the platform, before anything is delivered as an event,
  so a synthesised key event reaches the window's filters and misses every
  shortcut there is -- which would make this socket quietly useless for
  exactly the keys the sheet advertises. They are looked up by their own key
  sequence and triggered.
- **The state says what the develop is standing on**: which base, measured
  and whether that still matches the settings. A
  window drawing from a stale measurement is the failure this socket was
  opened to chase.

Qt objects belong to the thread that made them, so nothing here touches a
widget. Every request drops a callable on a queue and waits; a timer inside the
GUI thread drains it, runs it there, and hands the answer back. The wait has a
timeout, so a hung window returns an error rather than a hung client.
"""

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# What `/call` may reach. A whitelist and not getattr on anything: this is a
# debugging aid on a socket, and the smallest useful surface is the right one.
# Everything here is a command the sheet already offers, or a way of getting
# to the scan a report names.
CALLABLE = (
    # looking
    "_cut", "_cycle", "_show_mode", "_want_developed", "_next", "_prev",
    "_step", "_step_cut", "_goto", "_refresh", "_select",
    # the frame
    "_turn_left", "_turn_right", "_mirror",
    "_copy_frames", "_paste_frames", "_skip", "_reset",
    "_enter_mode", "_leave_mode", "_add_part", "_drop_part",
    # the roll
    "_process", "_redetect", "_redetect_roll", "_look",
    "_undo", "_redo", "_save", "_forget_base", "_forget_developed",
    # the canvas
    "_retone",
)

KEYS = {}


def _qt():
    from PySide6 import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


def _key_table():
    """Name to key code: "down", "space", "d", and `mods` for the rest.

    Built from `Qt.Key.__members__` and not from `dir(Qt)`: the names live on
    the enum, not loose on the namespace, and scanning the namespace finds none
    of them.
    """
    if KEYS:
        return KEYS
    QtCore, _, _ = _qt()
    for name, value in QtCore.Qt.Key.__members__.items():
        KEYS[name[4:].lower()] = value
    return KEYS


class _Bridge:
    """Runs callables on the GUI thread and hands back what they return."""

    def __init__(self, win):
        QtCore, _, _ = _qt()
        self.win = win
        self.jobs = queue.Queue()
        self.timer = QtCore.QTimer(win)
        self.timer.timeout.connect(self._drain)
        self.timer.start(20)

    def _drain(self):
        while True:
            try:
                fn, box, done = self.jobs.get_nowait()
            except queue.Empty:
                return
            try:
                box.append(("ok", fn()))
            except Exception as e:                          # noqa: BLE001
                box.append(("error", "%s: %s" % (type(e).__name__, e)))
            done.set()

    def run(self, fn, timeout=120.0):
        _, _, QtWidgets = _qt()
        modal = QtWidgets.QApplication.activeModalWidget()
        if modal is not None:
            # Not a failure to report as one: a sheet is open and waiting for
            # a person, and saying which is the useful answer.
            return "error", ("a dialog is waiting: %r"
                             % (modal.windowTitle() or type(modal).__name__))
        box, done = [], threading.Event()
        self.jobs.put((fn, box, done))
        if not done.wait(timeout):
            return "error", "timed out waiting for the window"
        return box[0]


def _plain(html):
    """A read line without its colours, for a client that wants the words."""
    import re
    return re.sub(r"\s+", " ", re.sub("<[^>]+>", "", html or "")
                  .replace("&nbsp;", " ").replace("&amp;", "&")).strip()


def _state(win):
    """Everything the window knows, so a helper can see it without a screen."""
    from . import develop
    doc = win.doc
    name = win.name()
    it = win.item() or {}
    taste = dict(develop.TASTE) | dict(doc["roll"].get("develop") or {})
    m = doc["roll"].get("measured") or {}
    under = m.get("under") or {}
    key = (name, (win.part() or {}).get("name")) if name else None
    return {
        "roll": str(win.roll),
        "scan": name,
        "row": win.list.currentRow() + 1,
        "scans": len(win.names),
        "state": it.get("state"),
        "angle": it.get("angle"),
        "parts": [{"name": p["name"], "centre": p["centre"], "size": p["size"],
                   "format": p.get("format"), "turn": p.get("turn", 0)}
                  for p in it.get("parts", [])],
        "part": (win.part() or {}).get("name"),
        "manual": it.get("manual", []),
        # What is on the canvas, which is the thing a report is usually about.
        "view": win.canvas.view,
        "colour": win.colour, "cut": win.cut,
        "mode": win.mode,
        "developing": win.dev_worker is not None,
        "develop_evidence": win.dev_ev.get(key) if key else None,
        "read_line": _plain(win.reading.text()),
        "held": win.held.text(),
        "status": win.status.text(),
        # What the develop is standing on. A window drawing from a
        # measurement taken under a different gamma is the failure this
        # socket exists to chase, so it is on the state and not inferred.
        "settings": {k: taste.get(k) for k in
                     ("base", "gamma", "exposure", "warmth", "tint",
                      "shadows", "highlights", "saturation", "contrast",
                      "pivot", "order", "format")},
        "one_roll": doc["roll"].get("one_roll", True),
        "mirrored": doc["roll"].get("mirrored", True),
        "measured": {
            "ready": develop.measured_ready(m),
            "matches_settings": develop.measured_ready(m, taste),
            "under": under,
            "base_method": m.get("base_method"),
            "base_colour": m.get("base_colour"),
            "base_roll_unit": m.get("base_roll_unit"),
            "balance_log2": m.get("balance_log2"),
        },
        # Where the window is, so a recording can be cropped to it rather
        # than to a rectangle somebody guessed.
        "window": [win.x(), win.y(), win.width(), win.height()],
        "dirty": win.dirty,
    }


def _shot(win):
    QtCore, _, _ = _qt()
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.WriteOnly)
    win.grab().save(buf, "PNG")
    return bytes(buf.data())


def _press(win, key, mods):
    """Press a key the way the keyboard would, shortcuts included.

    A synthesised event is not enough on its own. Qt matches shortcuts on the
    way in from the platform, before anything is delivered as an event, so
    posting a key event reaches the window's filters (the held views, the
    digits) and misses every shortcut there is -- which would make this socket
    quietly useless for testing exactly the keys the sheet advertises. Here
    the `⌘` commands are `QAction`s with an application context, so they are
    looked up by their own key sequence and triggered; the event is sent as
    well for whatever reads it directly.
    """
    QtCore, QtGui, QtWidgets = _qt()
    code = _key_table().get(str(key).lower())
    if code is None:
        raise ValueError("unknown key %r" % key)
    flags = QtCore.Qt.NoModifier
    for m in mods or []:
        flags |= {"shift": QtCore.Qt.ShiftModifier,
                  "ctrl": QtCore.Qt.ControlModifier,
                  "cmd": QtCore.Qt.ControlModifier,
                  "alt": QtCore.Qt.AltModifier,
                  "meta": QtCore.Qt.MetaModifier}[m.lower()]
    want = QtGui.QKeySequence(QtCore.QKeyCombination(flags, code))
    fired = False
    for a in win.findChildren(QtGui.QAction):
        if (a.isEnabled() and not a.shortcut().isEmpty()
                and a.shortcut().matches(want) == QtGui.QKeySequence.ExactMatch):
            a.trigger()
            fired = True
    if not fired:
        target = QtWidgets.QApplication.focusWidget() or win.canvas
        for kind in (QtCore.QEvent.KeyPress, QtCore.QEvent.KeyRelease):
            QtWidgets.QApplication.sendEvent(
                target, QtGui.QKeyEvent(kind, code, flags))
    return _state(win)


def _hold(win, key, mods, ms):
    """Press and hold, for the views that are held rather than switched:
    `o`, `b`, `1`, `2`. The picture is taken while it is down."""
    QtCore, QtGui, QtWidgets = _qt()
    code = _key_table().get(str(key).lower())
    if code is None:
        raise ValueError("unknown key %r" % key)
    flags = QtCore.Qt.NoModifier
    for m in mods or []:
        flags |= {"shift": QtCore.Qt.ShiftModifier}[m.lower()]
    target = QtWidgets.QApplication.focusWidget() or win.canvas
    QtWidgets.QApplication.sendEvent(
        target, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, code, flags))
    shot = _shot(win)
    QtWidgets.QApplication.sendEvent(
        target, QtGui.QKeyEvent(QtCore.QEvent.KeyRelease, code, flags))
    return shot


def _jsonable(v, depth=0):
    """Whatever came back, in something `json` will take.

    Arrays are summarised rather than sent: a preview is six million
    numbers and nobody asked for them down a socket. Anything else that
    will not serialise comes back as its `repr`, which is the point --
    this is for looking, and a truthful `repr` beats an error.
    """
    import numpy as np
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, np.ndarray):
        out = {"__array__": list(v.shape), "dtype": str(v.dtype)}
        if v.size:
            out |= {"min": float(v.min()), "max": float(v.max()),
                    "mean": float(v.mean())}
        return out
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if depth > 4:
        return repr(v)[:400]
    if isinstance(v, dict):
        return {str(k): _jsonable(x, depth + 1) for k, x in list(v.items())[:200]}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x, depth + 1) for x in list(v)[:200]]
    return repr(v)[:400]


def _eval(win, expr):
    """Evaluate one expression against the live window, on its own thread.

    `/state` says what the window is; this says anything else about it --
    which frame the canvas thinks is selected, what is in a cache, what a
    develop decided -- without restarting the program to add a print. It
    is what makes a session of "why is it doing that?" one question long
    instead of one build long.

    `win` is the editor, and `doc`, `np` and the program's own modules are
    in scope. It runs on the GUI thread, like everything else here, so it
    sees the widgets as they are.

    **It runs arbitrary code in the program**, which is what it is for and
    is worth saying out loud: the socket is opt-in (`--serve`), bound to
    127.0.0.1, and is for reproducing a problem with somebody, not for
    leaving open.
    """
    import math
    import pathlib as _pathlib
    import numpy as np
    from . import develop, document, fitframe, findframe, image_io
    scope = {"win": win, "doc": win.doc, "roll": win.roll, "np": np,
             "math": math, "pathlib": _pathlib, "develop": develop,
             "document": document, "fitframe": fitframe,
             "findframe": findframe, "image_io": image_io}
    return {"expr": expr, "value": _jsonable(eval(expr, scope))}


def _call(win, name, args):
    if name not in CALLABLE:
        raise ValueError("%s is not on the list of things this may call" % name)
    getattr(win, name)(*(args or []))
    return _state(win)


def serve(win, port=8765):
    """Start the control socket. Returns the port it is listening on."""
    bridge = _Bridge(win)

    def wait_for_render(limit=120.0):
        """`d` renders off the window's thread, so a shot taken the instant
        after the key is a shot of nothing. This waits from the *serving*
        thread, which leaves the GUI thread free to do the render."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < limit:
            how, what = bridge.run(lambda: bridge.win.dev_worker is not None)
            if how == "error" or not what:
                return
            time.sleep(0.05)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):        # the terminal belongs to the user
            pass

        def _send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(
                body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _reply(self, fn, kind="application/json"):
            how, what = bridge.run(fn)
            if how == "error":
                return self._send(500, {"error": what})
            self._send(200, what, kind)

        def do_GET(self):
            if self.path.startswith("/shot"):
                if "wait" in self.path:
                    wait_for_render()
                return self._reply(lambda: _shot(bridge.win), "image/png")
            if self.path.startswith("/state"):
                return self._reply(lambda: _state(bridge.win))
            if self.path.startswith("/document"):
                return self._reply(lambda: bridge.win.doc)
            self._send(404, {"error": "try /state, /document, /shot, "
                                      "/key, /hold, /call, /eval"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError as e:
                return self._send(400, {"error": "bad JSON: %s" % e})
            win = bridge.win
            if self.path.startswith("/key"):
                how, what = bridge.run(
                    lambda: _press(win, body.get("key"), body.get("mods")))
                if how == "error":
                    return self._send(500, {"error": what})
                if body.get("wait"):
                    wait_for_render()
                    how, what = bridge.run(lambda: _state(win))
                return self._send(200, what)
            if self.path.startswith("/hold"):
                return self._reply(
                    lambda: _hold(win, body.get("key"), body.get("mods"),
                                  body.get("ms", 0)), "image/png")
            if self.path.startswith("/eval"):
                how, what = bridge.run(
                    lambda: _eval(win, body.get("expr") or "None"))
                if how == "error":
                    return self._send(500, {"error": what})
                return self._send(200, what)
            if self.path.startswith("/call"):
                how, what = bridge.run(
                    lambda: _call(win, body.get("name"), body.get("args")))
                if how == "error":
                    return self._send(500, {"error": what})
                if body.get("wait"):
                    wait_for_render()
                    how, what = bridge.run(lambda: _state(win))
                return self._send(200, what)
            self._send(404, {"error": "try /key, /hold, /call or /eval"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("control socket on http://127.0.0.1:%d  "
          "(/state /document /shot /key /hold /call /eval)" % port)
    return port
