# Changelog

## 0.1.0

The first release.

- Finds one frame per scan, or the two of a half-frame pair, with its angle,
  and measures a roll twice so its frames share a height and a gap.
- A window to check every scan, correct a frame by hand, and see the scan,
  the positive or the develop.
- Develops positives against the film base measured over the roll, with
  taste set once per roll.
- Writes positive DNGs, inverted on the Bayer mosaic, for another raw
  developer; and optionally the cropped negatives.
- Light falloff calibrated once per copy stand, from blank shots.
- Dust marked by hand, taken out of the develop; an experimental finder
  proposes marks to accept or refuse.
- The perforations, where they show, set the roll's frame size in
  millimetres and where each frame sits along and across the film.
- The roll's look set by eye on its own frames: exposure, warmth, tint,
  shadow and highlight slopes, contrast, pivot, saturation, noise reduction
  and sharpening, with a 100 % view.
- A scan's own film base picked by hand, for scans copied under another
  light.
- Positive DNGs losslessly compressed, tagged so 5600 K and no tint is the
  roll's neutral, and optionally cleaned of the marked dust.
- The list in shot order, set by hand, and files numbered by it.
- The whole roll from the command line as well as the window: `roll` reads
  the document (as JSON too), skips scans and sets the roll's options,
  `detect --again` measures again keeping what was done by hand, and
  `process` writes what the window's ⌘R writes, roll after roll.
- A desktop build for Apple Silicon, carrying its own Python.
