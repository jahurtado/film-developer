# film-developer

A desktop application for digitising photographic film negatives with a
digital camera.

You photograph each negative on a light table, with the camera on a copy
stand, and get a raw file that holds rather more than the picture: the
unexposed film around it, the sprocket holes, the holder, the light table
behind, and a degree or two of tilt. film-developer takes a folder of those
scans — a roll — finds every exposed frame in it, one per scan or two on
half frame, straightens and crops it, and turns the orange negative into a
finished, print-ready positive. You check the roll in the window, correct
what needs correcting, and write it out.

<p align="center">
  <img src="docs/images/walkthrough.gif" width="720"
       alt="Opening a roll, checking it, placing a frame, marking dust, and writing it out">
</p>

## What comes out

- **Developed pictures**, JPEG or 16-bit TIFF, inverted against the film's
  own base and balanced across the roll.
- **Positive raw files (DNG)**: the negative inverted while it is still a
  raw, for another raw developer to open as if a camera had shot the
  positive (tried in DaVinci Resolve and in the one built into macOS). A daylight white
  balance — 5600 K, no tint — is the roll's own neutral.
- Optionally **the negatives**, cropped and untouched.

Files are numbered in the order the roll was shot. **Your original scans are
never written over.**

## How you use it

1. **Open a folder of scans.** Drag it onto the window, or choose it. Camera
   raw (NEF, CR2, CR3, DNG), TIFF and JPEG are read.
2. **Say what the folder is.** A sheet asks once, before anything starts:
   are these scans one roll, is every frame the same size, does the roll
   share one film base, and on which copy stand was it scanned. Then the
   roll is measured, on every core of the machine, in a minute or two.
3. **Look through it.** The window opens on the first picture, already
   developed. `n` and `p` walk the roll, `v` switches between the scan as
   the camera saw it, the plain positive and the developed picture, and
   `space` fills the window with the frame. The few frames the program is
   not sure about are marked in the list.
4. **Correct what needs it.** `e` puts the frame's corners under your
   hands; `r` turns a picture, `del` leaves a scan out, and dragging in the
   list puts the roll in the order it was shot. Anything placed by hand
   stays placed, even if the roll is measured again.

5. **Mark dust**, if you want it gone. `s` shows the frame developed at full
   size; a click marks a speck, and the develop fills it in.

6. **Set the look of the roll**, once, with a frame in front of you (`⌘K`):
   exposure, warmth, tint, contrast, saturation, noise reduction and
   sharpening, the last two judged at 100 %.
7. **Write it out** (`⌘R`): pick the pictures, the raws, or both.

Every key the window answers to is listed along its bottom edge, for the
mode you are in.

<p align="center">
  <img src="docs/images/frames.png" width="400"
       alt="Placing the frame by hand: its corners, and the film base picked beside it">
  &nbsp;
  <img src="docs/images/dust.png" width="400"
       alt="Marking dust by hand, on the frame developed at full size">
  <br>
  <sub>Placing a frame by hand (<code>e</code>) and marking dust (<code>s</code>).</sub>
</p>

## Getting it

A macOS app (Apple Silicon) carrying its own Python is on its way to the
[releases page](https://github.com/jahurtado/film-developer/releases). Until
it is there, run it from source.

## From source

You need [uv](https://docs.astral.sh/uv/) and Python 3.10 or newer. There is
nothing to build and nothing to activate:

```sh
git clone https://github.com/jahurtado/film-developer.git
cd film-developer
./film-developer
```

The launcher hands the work to `uv run`, which builds the environment on the
first call, and opens the window. Building the desktop bundles yourself is
described in [packaging/README.md](packaging/README.md).

## Under the hood

**Finding the frame.** A search over rectangles, scored against two kinds of
pixel that are never picture — light shining straight through the holder,
and the opaque holder itself — then refined side by side to the innermost
edge that runs the whole length of the side. A roll is one camera, so its
frames are measured twice, the second time knowing their size. Where the
sprocket holes are in the scan they give the roll a scale and a place: the
holes are 4.75 mm apart on all 135 film, and a camera advances the film by
them, so every frame of a roll starts at the same point among them. On 98
scans placed by hand, from four rolls and two cameras, it gets 97 right and
flags the one it gets wrong.

**Developing.** Each frame is divided by the film base — unexposed film,
read from the rebate, bounded by the frame's own shadows, or picked by hand
— and the film's curve is undone. Base and colour balance are measured over
the whole roll; each frame's exposure and curve follow rules; the look on
top is the roll's, not the picture's. The light falloff of a copy stand can
be calibrated once from a few blank shots and applies to every roll scanned
on it.

What was decided for each scan, and the evidence for it, is kept in a
readable `frames.json` in the folder of scans.

[![Read the article on elcacharrista.com](docs/images/article.png)](https://www.elcacharrista.com/en/articles/digitizing-film-negatives-with-a-camera/)

## Licence

GPL-3.0-or-later; the full text is in [LICENSE](LICENSE). Nothing GPL is
bundled — the dependencies are LGPL (Qt, LibRaw) and BSD/MIT — so this is a
choice rather than an inherited obligation. What the desktop build carries
inside it, and where its source is, is in
[THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md).
