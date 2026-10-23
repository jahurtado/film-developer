# Third-party software in the desktop builds

Running from source, this program is the code in `src/` and nothing else: uv
fetches the libraries and their licences come with them.

The **desktop builds are different**. The macOS `.app` and the Windows folder
carry their own Python and their own Qt, so what you download is one file
containing work by a lot of other people under their own terms. This is the
list, and it is shipped inside the builds as well —
`Contents/Resources/licenses/` on macOS, `licenses\` on Windows — because a
notice that only exists in a source repository is not next to the thing it
describes.

**No GPL software is bundled here.** This one carries LGPL libraries and
permissive ones, and nothing else; the GPL-3 on this program is a choice, not
something inherited.

## film-developer itself

GPL-3.0-or-later. The full text is in [LICENSE](LICENSE); the source is
published at <https://github.com/jahurtado/film-developer>.

## Qt

**LGPL-3.0**, through PySide6 — the editor window, and nothing else in the
program. Qt is an unmodified upstream binary, used as a shared library and
dynamically linked, which is what the LGPL asks for. Relinking against a
different build of Qt means replacing the libraries inside the bundle; the
source is published at the address below.

- Qt: <https://www.qt.io/> — source at <https://download.qt.io/archive/qt/>
- PySide6 / shiboken6: LGPL-3.0-only, <https://wiki.qt.io/Qt_for_Python>

## LibRaw

**LGPL-2.1 or CDDL-1.0**, reached through `rawpy` (MIT). LibRaw is what reads
the scans — NEF, DNG, CR2, CR3 — and demosaics them. The wheel carries an
unmodified build.

- LibRaw: <https://www.libraw.org/> — source at <https://github.com/LibRaw/LibRaw>
- rawpy: <https://github.com/letmaik/rawpy>

## Python

**PSF-2.0.** The builds carry their own interpreter — the one PyInstaller
freezes on macOS, the embeddable distribution on Windows — so nothing has to be
installed. <https://docs.python.org/3/license.html>

## The rest

Permissive, and none of them modified:

| package | licence |
|---|---|
| numpy | BSD-3-Clause (with 0BSD, MIT, Zlib and CC0-1.0 parts) |
| scipy | BSD-3-Clause |
| tifffile | BSD-3-Clause |
| imagecodecs | BSD-3-Clause |
| Pillow | MIT-CMU |

tifffile writes the DNG files and imagecodecs gives it the codecs; Pillow
writes the JPEG positives and draws the icon.

PyInstaller (GPL-2.0 with its bootloader exception) and the Inno Setup compiler
build the packages and are not part of them.

## Asking

<https://elcacharrista.com>
