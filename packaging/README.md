# Building the desktop bundles

Two things are particular to this program: the worker pools need
`freeze_support` (`app.py`), and LibRaw comes in with `rawpy` (the spec).

The tool runs from source everywhere (see the [README](../README.md)).
This folder is only for the two download-and-run bundles: the **macOS app** and
its `.dmg`, and the **Windows** portable zip and installer. Neither bundle is
kept in the repository; both are built here into `dist/`, which is gitignored.

Both carry their own Python and every library, so the machine needs nothing. Both
are **unsigned** — a code-signing certificate (Apple's, or a Windows one) is a
yearly cost this project does not carry — so each warns once on first launch and
the person allows it on purpose. The two platforms reach that result by opposite
routes, and that is most of what this folder is about.

## What is here

| File | Role |
|---|---|
| `make-dmg.sh` | macOS: build the `.app` and wrap it in a `.dmg` |
| `film-developer.spec` | macOS: the PyInstaller recipe (`make-dmg.sh` drives it) |
| `app.py` | macOS: the bundle's entry point (strips the `-psn` launch arg) |
| `make-win.ps1` | Windows: build the embeddable-Python payload and the `.zip` |
| `film-developer.iss` | Windows: wrap that payload in an Inno Setup installer |

The version is read from the package (`src/film_developer/__init__.py`) in both
builds, so the file names and the metadata cannot drift from `--version`.

## macOS

```sh
uv venv
uv pip install -e '.[mac]'      # PySide6 + pyinstaller
sh packaging/make-dmg.sh
```

PyInstaller freezes a self-contained `.app`; `make-dmg.sh` wraps it in the
ordinary drag-to-Applications disk image, first-launch note included. **Apple
Silicon only** — PySide6 stopped shipping universal2 wheels, so an Intel Mac
cannot run this build, and the file name says `arm64`.

The whole cost is size: a stock PySide6 is 1.2 GB, so the spec excludes every Qt
module but the three the editor imports. The scientific wheels cannot be pruned
the same way — imagecodecs and scipy load extensions by path, and dropping one
only surfaces as a crash on the frame that needs it — so they go in whole. The
reasoning lives in the spec's docstring.

## Windows

```powershell
uv venv
uv pip install -e ".[win]"      # PySide6 only
powershell -File packaging\make-win.ps1
```

That produces `dist\film-developer\` (the payload) and
`dist\film-developer-VERSION-x64.zip`. The zip is the whole deliverable: unpack
and run `film-developer.cmd`. **x64 only.**

For the installer, install [Inno Setup](https://jrsoftware.org/isinfo.php) once
and compile the payload `make-win.ps1` left behind:

```powershell
winget install JRSoftware.InnoSetup
iscc packaging\film-developer.iss
```

That writes `dist\film-developer-VERSION-x64-setup.exe` — a per-user install
(no admin prompt) with a Start-menu shortcut and an uninstaller.

## Checksums

Each script writes a `.sha256` beside the file it just made, in the format
`shasum -c` and `sha256sum -c` both read back — lower-case hex, two spaces, the
bare name. It is done by the build and not by hand at release time, because a
hash worked out afterwards is a hash of whatever happened to be left in
`dist/`.

`iscc` runs after `make-win.ps1`, so the installer is the one artefact neither
script has hashed. Same helper, one line:

```powershell
$f = "dist\<name>-<version>-x64-setup.exe"
[IO.File]::WriteAllText("$f.sha256",
    "$((Get-FileHash -Algorithm SHA256 $f).Hash.ToLower())  $(Split-Path -Leaf $f)`n")
```

At release time the three come from two machines, so gather them into one file
rather than uploading three:

```sh
cat *.sha256 > SHA256SUMS      # and verify with:  shasum -c SHA256SUMS
```

### Why Windows does not use PyInstaller

It was tried first and abandoned. PyInstaller's bootloader trips a Windows
Defender false positive (`Program:Win32/Vigram.A`); Defender then silently
**truncates the compiled `setup.exe`**, and it fails at install with "The setup
files are corrupted." Signing would not help — Defender is matching a byte
pattern, not the absence of a signature.

So Windows ships an **embeddable CPython** instead: `python.exe`, its stdlib
zip, and a folder of ordinary wheels. There is no frozen binary, so Defender has
nothing to object to. `make-win.ps1` downloads the embeddable Python (pinned to
the venv's exact version, because the wheels are for that ABI), installs the
project into it with `uv` (a uv-made venv has no `pip`, so `python -m pip` is not
an option — `uv pip install --target` is), and lays a `.cmd` launcher and the
icon beside it. The editor is launched with `pythonw.exe -m film_developer`;
`detect`, `develop` and `apply` run through `python\python.exe -m film_developer`.

### Things that bit, and are handled

- **PySide6-Essentials, not the full PySide6.** The full wheel drags in
  pyside6-addons — Qt WebEngine (a 198 MB DLL by itself), Quick, 3D, Charts —
  none of which a widgets editor touches. Essentials supplies the same `PySide6`
  package, only the parts used, and saves ~370 MB.
- **The QML tree is pruned.** Even Essentials ships `PySide6\qml`, whose build
  objects sit at exactly 260 characters — Windows' path limit — and the Inno
  compiler cannot read past it ("cannot find the path specified").
- **The `.pyi` stubs are kept.** A lazy loader such as scikit-image's can
  read its own `__init__.pyi` at runtime; nothing here is known to, but they
  are small and the rule is cheaper than finding out.
- **The uninstaller removes the whole folder.** Python writes `.pyc` caches
  beside the code on first run; Inno only tracks what it installed, so without an
  explicit `[UninstallDelete]` the folder lingers full of bytecode.

### If `iscc` produces a "source file is corrupted" installer

Rarely, a freshly compiled `setup.exe` fails at install with "the source file is
corrupted" (Inno's `lzmadecomp: Compressed data is corrupted`), on a different
file each time. The compile itself reports success — the bad block is baked into
the `setup.exe` and only surfaces on extraction, so it would fail on any machine.

It is **not** an LZMA2 bug and **not** fixed by switching to `Compression=lzma`:
measured 11 clean compiles across both codecs, none reproduced it. The cause is
real-time Defender scanning `packaging\dist` aggressively the *first* time those
exact files appear on the system — right after `make-win.ps1` writes them with
`uv` — and an occasional locked read makes `iscc` copy a bad block. Once Defender
has scanned and cached those bytes as clean, recompiles come out fine, which is
why it is intermittent and usually gone on the next try.

So the first fix is simply to **recompile** — the second pass almost always
succeeds. If it keeps happening on a clean machine, exclude the build output from
Defender once (elevated PowerShell) and recompile:

```powershell
Add-MpPreference -ExclusionPath "$PWD\packaging\dist"
```

This is a build-machine step only; it stamps nothing into the installer, and
whoever runs the finished `setup.exe` needs no exclusion. The zip is never
affected — it is stored, not LZMA-compressed by Inno. Either way, **verify a
build by installing it once** (`setup.exe /VERYSILENT /DIR=...` then check the
files are there): a corrupt block is silent at compile but obvious at install.
