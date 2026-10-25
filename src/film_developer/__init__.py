"""Find the picture inside a photographed negative.

A negative digitised with a camera arrives as a raw file holding rather more
than the picture: the unexposed rebate around it, the sprocket holes, the light
table behind, and a degree or two of tilt from the film never sitting square in
the holder. The package finds the exposed frame -- one of them, or two when the
film is half frame -- and writes it out cropped, straightened and still raw.

Three steps kept deliberately apart -- `detect` measures, `gui` corrects by
hand, `apply` writes the files -- over the support modules: `findframe` (the
estimators), `image_io` (raw in, DNG out) and `document` (the format of what
sits between the three).
"""

__version__ = "0.2.0"
