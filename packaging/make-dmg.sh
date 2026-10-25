#!/bin/sh
# Build the macOS bundle and wrap it in a disk image.
#
#     sh packaging/make-dmg.sh
#
# The image is the ordinary kind: the app on the left, a link to /Applications
# on the right, drag one onto the other. It also carries a note about the
# first launch, because there is one.
#
# **Nothing here is signed with a Developer ID.** PyInstaller signs the binary
# ad-hoc, which is what lets it run on Apple Silicon at all, but Gatekeeper
# wants more than that from something downloaded. Notarising needs Apple's
# Developer Program at 99 USD a year; without it the first launch is blocked
# once and the person has to say so on purpose; the disk image carries a note
# saying how.
#
# arm64 only. PySide6 stopped shipping universal2 wheels, so an Intel Mac
# cannot run this build and the file name says which is which.
set -e
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"

version=$(cd .. && .venv/bin/python -c \
    "import sys; sys.path.insert(0,'src'); import film_developer as e; print(e.__version__)")
arch=$(uname -m)
dmg="dist/film-developer-$version-$arch.dmg"

../.venv/bin/pyinstaller --noconfirm --clean film-developer.spec

# Build leftovers that the bundle has no use for. `direct_url.json` is the
# worst of them: an editable install records where it was installed from, so it
# carries the absolute path of this checkout -- somebody's home directory,
# shipped in every DMG. The uv files are build timestamps. METADATA stays,
# because `cli.identity()` reads the author and the homepage out of it at
# runtime, and so does the About box.
# The licences, inside the bundle. Nothing GPL travels in here -- Qt and LibRaw
# are LGPL and the rest is BSD or MIT -- but the LGPL asks for its notice just
# as plainly, and a notice that only exists in a source repository is not next
# to the thing it describes.
app=dist/film-developer.app

for junk in direct_url.json uv_cache.json uv_build.json REQUESTED INSTALLER; do
  find "$app" -name "$junk" -path "*film_developer-*.dist-info/*" -delete
done

mkdir -p "$app/Contents/Resources/licenses"
cp ../LICENSE                 "$app/Contents/Resources/licenses/LICENSE-GPL-3.0.txt"
cp ../THIRD-PARTY-LICENSES.md "$app/Contents/Resources/licenses/"
# PyInstaller signs the bundle ad-hoc as it builds it, and the files above went
# in after that, so the seal is renewed over what is now there. Verified rather
# than assumed: a bundle that signs but does not verify is one Gatekeeper stops
# later, on somebody else's machine.
codesign --force --deep --sign - "$app"
codesign --verify --deep --strict "$app"

rm -rf dist/dmg "$dmg"
mkdir -p dist/dmg
cp -R dist/film-developer.app dist/dmg/
ln -s /Applications dist/dmg/Applications
cp ../THIRD-PARTY-LICENSES.md dist/dmg/
cp ../LICENSE "dist/dmg/LICENSE-GPL-3.0.txt"
cat > "dist/dmg/First launch.txt" <<'TXT'
film-developer
===============

Drag the app onto the Applications folder beside it.

The first time you open it, macOS will say it cannot be opened because
Apple cannot check it for malicious software. That is expected: this app
is not signed with an Apple Developer ID, which costs 99 USD a year.

To open it anyway, either

  * open  System Settings -> Privacy & Security,  scroll to the bottom,
    and press "Open Anyway" next to the message about film-developer.
    Then open the app again and confirm. Once is enough, for ever.

or, if you prefer the one line,

    xattr -dr com.apple.quarantine /Applications/film-developer.app

Control-clicking the icon and choosing Open used to work; Apple removed
that shortcut in macOS 15.

Apple Silicon only. On an Intel Mac, install from source instead:
https://github.com/jahurtado/film-developer#run-from-source
TXT
hdiutil create -volname "film-developer $version" -srcfolder dist/dmg \
    -ov -format UDZO "$dmg" >/dev/null
rm -rf dist/dmg
# The checksum, beside the file and in the format `shasum -c` reads back, so
# whoever downloads it can check it with one command and no arguments. Written
# by the build rather than typed at release time: a hash worked out later is a
# hash of whatever happened to be in dist/.
( cd "$(dirname "$dmg")" && shasum -a 256 "$(basename "$dmg")" \
    > "$(basename "$dmg").sha256" )

echo "$dmg  ($(du -h "$dmg" | cut -f1))"
cat "$dmg.sha256"
