# Build the Windows bundle: an embeddable CPython with the wheels laid beside
# it, wrapped in a portable zip.
#
#     pwsh packaging/make-win.ps1     (or:  powershell -File packaging\make-win.ps1)
#
# There is no frozen binary here, and that is the whole point. PyInstaller's
# bootloader trips a Windows Defender false positive (`Program:Win32/Vigram.A`)
# that silently truncates whatever carries it -- the compiled installer came out
# corrupt every time. An embeddable Python is just python.exe, its stdlib zip,
# and a folder of ordinary wheels; Defender has nothing to object to, so nothing
# gets eaten.
#
# The result is dist\film-developer\ (the payload the installer also takes) and
# dist\film-developer-VERSION-x64.zip. It carries its own Python and needs
# nothing on the machine. **Not signed** -- an unsigned download still trips
# SmartScreen the first time; the note inside the zip says so.
#
# x64 only, and the embeddable Python is pinned to the exact version of the venv
# doing the build, because the wheels pip resolves are for that ABI.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here
Set-Location $here

$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No .venv. Run:  uv venv; uv pip install -e `".[win]`"" }

# Version and Python version both from the venv, the one place each lives, so
# the zip's name and the embed's ABI cannot drift from what built them.
$version = & $py -c "import sys; sys.path.insert(0,'src'); import film_developer as e; print(e.__version__)"
$pyver   = & $py -c "import sys; print('%d.%d.%d' % sys.version_info[:3])"
$pytag   = & $py -c "import sys; print('%d%d' % sys.version_info[:2])"   # e.g. 313

$build = Join-Path $here "build"
$dist  = Join-Path $here "dist"
$payload = Join-Path $dist "film-developer"          # the folder both outputs share
$embed = Join-Path $payload "python"
$sp    = Join-Path $embed "Lib\site-packages"
New-Item -ItemType Directory -Force -Path $build, $dist | Out-Null

# 1. The embeddable CPython, cached between builds -- it is a fixed download.
$zipname = "python-$pyver-embed-amd64.zip"
$cached  = Join-Path $build $zipname
if (-not (Test-Path $cached)) {
    $url = "https://www.python.org/ftp/python/$pyver/$zipname"
    Write-Output "downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $cached
}

# 2. Lay it down fresh under the payload folder.
if (Test-Path $payload) { Remove-Item -Recurse -Force $payload }
New-Item -ItemType Directory -Force -Path $embed | Out-Null
Expand-Archive -Path $cached -DestinationPath $embed -Force

# 3. Turn on site so Lib\site-packages is searched. The embeddable ships with
#    it off and that folder unlisted; without both edits the wheels are invisible
#    and every import fails.
$pth = Join-Path $embed "python$pytag._pth"
@("python$pytag.zip", ".", "Lib\site-packages", "import site") |
    Set-Content -Path $pth -Encoding ascii

# 4. The wheels, resolved from this project so the versions match the venv's.
#    --target drops them flat into site-packages. uv, not pip: a uv-made venv
#    ships no pip, and uv installs into --target for the pinned interpreter's ABI
#    just the same.
#
#    PySide6-Essentials, not the full PySide6. The full meta-package pulls in
#    pyside6-addons -- Qt WebEngine (a 198 MB DLL on its own), Quick, Qml, 3D,
#    Charts, Multimedia -- none of which this editor touches; it imports only
#    QtCore, QtGui and QtWidgets, all of which live in Essentials. That one
#    swap takes the payload from ~815 MB to ~250 MB. The project itself is
#    installed without its `gui` extra so it does not drag the full PySide6 back
#    in; Essentials supplies the same `PySide6` package, only lighter.
uv pip install --python "$py" --target "$sp" "$repo" "PySide6-Essentials>=6.6"
if ($LASTEXITCODE -ne 0) { throw "install into the embed failed" }

# 4b. Drop the QML tree, the C++ headers and the bytecode caches -- all dead
#     weight for a widgets-only editor. QML matters most: even Essentials ships
#     it, and its build objects sit at 260 chars exactly, which the Inno compiler
#     cannot read past ("cannot find the path"). The .pyi stubs stay (a lazy
#     loader can read one at runtime; here nothing is known to, and they
#     are small). The scientific extensions are left whole -- imagecodecs loads them by
#     path and a missing one only surfaces as a decode failing on some frame.
$pyside = Join-Path $sp "PySide6"
foreach ($d in "qml", "include", "typesystems", "glue") {
    $p = Join-Path $pyside $d
    if (Test-Path $p) { Remove-Item -Recurse -Force $p }
}
Get-ChildItem $sp -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# 4d. The licences, in the payload, which is also what the installer lays down.
#     Nothing GPL travels here, but Qt and LibRaw are LGPL and their notice is
#     owed just the same.
$lic = Join-Path $payload "licenses"
New-Item -ItemType Directory -Force -Path $lic | Out-Null
Copy-Item -Force (Join-Path $repo "LICENSE") (Join-Path $lic "LICENSE-GPL-3.0.txt")
Copy-Item -Force (Join-Path $repo "THIRD-PARTY-LICENSES.md") $lic

# The same notice as plain text, for the installer's InfoAfter page: Inno shows
# it verbatim and does not read Markdown, so the .md's headings, bold and table
# would show their raw syntax there. md_to_text renders it readable; the .md
# stays for GitHub, and both travel in the payload.
& $py (Join-Path $here "md_to_text.py") `
      (Join-Path $repo "THIRD-PARTY-LICENSES.md") `
      (Join-Path $lic "THIRD-PARTY-LICENSES.txt")
if ($LASTEXITCODE -ne 0) { throw "md_to_text failed" }

# 4e. Build leftovers the payload has no use for. direct_url.json is the one
#     that matters: an editable install records where it came from, so it
#     carries the absolute path of the build checkout. METADATA stays -- the
#     About box reads the author and homepage out of it at runtime.
foreach ($junk in @("direct_url.json","uv_cache.json","uv_build.json","REQUESTED","INSTALLER")) {
    Get-ChildItem -Path $sp -Recurse -Filter $junk -ErrorAction SilentlyContinue |
        Where-Object { $_.DirectoryName -like "*.dist-info" } |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

#     The console-script trampolines are the other copy of that path, and a
#     worse one: `uv pip install --target` drops a `bin\` of .exe launchers
#     (film-developer, f2py, numpy-config, the pyside6-* tools) and each one
#     embeds the build venv's python.exe absolute path as its UV_PYTHON_PATH
#     launch target -- so the private-string check catches them even though
#     they are binaries. The payload never runs them; it launches
#     python\pythonw.exe -m film_developer. Drop the folder: dead weight that
#     also leaks the build checkout.
$scripts = Join-Path $sp "bin"
if (Test-Path $scripts) { Remove-Item -Recurse -Force $scripts }

# 5. The launcher. pythonw.exe is the windowed interpreter -- no console -- and
#    `start` returns at once so nothing lingers. %~dp0 keeps it working wherever
#    the folder is unzipped. For the command-line steps (detect, develop, apply) call the
#    console interpreter directly:  python\python.exe -m film_developer detect ...
$cmd = @'
@echo off
start "" "%~dp0python\pythonw.exe" -m film_developer %*
'@
Set-Content -Path (Join-Path $payload "film-developer.cmd") -Value $cmd -Encoding ascii

$note = @'
film-developer
===============

This folder is the whole program. Nothing to install: run
film-developer.cmd to open the editor. Put the folder wherever you like --
Documents, the Desktop, a USB stick -- and make a shortcut to the .cmd if you
want it handy.

For the command line -- detect, develop and apply -- call the interpreter inside:

    python\python.exe -m film_developer detect  my-roll
    python\python.exe -m film_developer develop my-roll
    python\python.exe -m film_developer apply   my-roll

The first time you run it, Windows may say "Windows protected your PC" and
show a blue box. That is expected: this program is not signed with a
code-signing certificate, which is a yearly cost, so Windows does not
recognise the publisher. Click "More info", then "Run anyway". Once is enough.

It carries everything it needs. Nothing has to be installed beside it.

x64 Windows only. To run from source instead, see:
https://www.elcacharrista.com
'@
Set-Content -Path (Join-Path $payload "First launch.txt") -Value $note -Encoding utf8

# The icon travels in the payload so the installer's shortcut has one to point
# at -- there is no versioned exe to carry it any more.
Copy-Item (Join-Path $repo "src\film_developer\icons\icon.ico") `
          (Join-Path $payload "film-developer.ico")

# The version, written beside dist for the installer to read -- the one place it
# lives is the package, and this is how the .iss reaches it without an exe to
# stamp it into.
Set-Content -Path (Join-Path $dist "version.txt") -Value $version -Encoding ascii -NoNewline

# 6. Zip the payload folder itself, so it unpacks into a named directory rather
#    than scattering files wherever it was opened.
$zip = Join-Path $dist "film-developer-$version-x64.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $payload -DestinationPath $zip

# The checksum, beside the file and in the format `sha256sum -c` reads back --
# lower-case hex, two spaces, the bare name -- so it verifies the same way on
# either platform. Get-FileHash returns upper case, which that format does not
# expect. The installer is built afterwards by iscc, so its own .sha256 is
# written by the same helper; see packaging/README.md.
function Write-Sha256($path) {
    $h = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
    $n = Split-Path -Leaf $path
    # WriteAllText and not Set-Content: Set-Content ends the line with CRLF,
    # and `shasum -c` on macOS or Linux then reads the filename with a trailing
    # carriage return and cannot find the file. The hash is right and the check
    # fails anyway, which is the worst way for it to fail. LF verifies on all
    # three platforms.
    [IO.File]::WriteAllText("$path.sha256", "$h  $n`n")
    Write-Output "$h  $n"
}
Write-Sha256 $zip

$mb = "{0:N0}" -f ((Get-Item $zip).Length / 1MB)
Write-Output ""
Write-Output "payload: $payload"
Write-Output "zip:     $zip  ($mb MB)"
