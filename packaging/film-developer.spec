# -*- mode: python ; coding: utf-8 -*-
"""How the macOS bundle is built.

The whole cost here is size. A stock PySide6 install is 1.2 GB because it
ships every Qt module; this editor imports three -- QtCore, QtGui, QtWidgets --
so everything else is excluded by name. The scientific wheels bring their own
bulk and cannot be pruned the same way: imagecodecs and scipy load extensions
by path, and dropping one only shows up as a crash on the frame that needs it.

`rawpy` carries LibRaw as a dylib beside its extension (`libraw_r.*.dylib`,
and its own `.dylibs`); `collect_dynamic_libs` brings them along.

Windows does not go through here. PyInstaller's bootloader trips a Defender
false positive (`Program:Win32/Vigram.A`) that silently truncates the compiled
installer; the Windows build instead ships an embeddable CPython with the wheels
laid beside it and no frozen binary at all. See packaging/make-win.ps1.
"""

import os
import sys

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

# The version, from the one place that has it. Without it PyInstaller writes
# `CFBundleShortVersionString: 0.0.0`, which is what the Finder shows under
# Get Info -- a build that says 0.0.0 is a build nobody can tell apart from
# the last one.
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from film_developer import __version__ as VERSION

# `collect_submodules` returns the package itself first, and excluding that
# takes Qt out altogether -- which builds, and dies the moment the window is
# asked for. Keep the package and the three modules used.
QT_KEEP = {"PySide6", "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"}
QT_DROP = [m for m in collect_submodules("PySide6")
           if m not in QT_KEEP and not m.startswith("PySide6.support")]

a = Analysis(
    ["app.py"],
    pathex=[".."],
    binaries=collect_dynamic_libs("imagecodecs") + collect_dynamic_libs("rawpy"),
    datas=[("../src/film_developer/icons", "film_developer/icons")],
    # imagecodecs reaches its extensions with a computed __import__, so the
    # graph sees none of them; the whole package goes in rather than a guessed
    # list. The film_developer modules are imported by name in the workers a
    # spawn pool starts, so they are all named too.
    hiddenimports=collect_submodules("imagecodecs")
                  + collect_submodules("film_developer"),
    excludes=QT_DROP + [
        "matplotlib", "tkinter", "PyQt5", "PyQt6", "IPython", "pytest",
        "pandas", "notebook", "sphinx", "setuptools", "pkg_resources",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name="film-developer", console=False,
          disable_windowed_traceback=False, argv_emulation=True,
          target_arch=None, codesign_identity=None, entitlements_file=None)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name="film-developer")

app = BUNDLE(coll, name="film-developer.app",
             icon="../src/film_developer/icons/icon.icns",
             bundle_identifier="com.elcacharrista.film-developer",
             info_plist={
                 "CFBundleName": "film-developer",
                 "CFBundleDisplayName": "film-developer",
                 "CFBundleShortVersionString": VERSION,
                 "CFBundleVersion": VERSION,
                 "NSHighResolutionCapable": True,
                 # A roll is a folder of scans, so the app says it takes one:
                 # that lets a folder be dropped on the icon, and `open -a`
                 # pass one through.
                 "CFBundleDocumentTypes": [{
                     "CFBundleTypeName": "Roll folder",
                     "CFBundleTypeRole": "Editor",
                     "LSItemContentTypes": ["public.folder"],
                     "CFBundleTypeOSTypes": ["fold"],
                 }],
             })
