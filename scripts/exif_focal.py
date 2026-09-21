"""
EXIF-derived focal length in pixels, for seeding/constraining the `f`
parameter instead of letting the optimizer discover it from scratch.

Why this matters here: `f` trades off against camera position along the
optical axis, so an unconstrained per-image fit can match a horizon curve
well while landing on a badly wrong focal length. Measured on this dataset
(2026-09-11), the independently-fit `f` values sat at 1.07-1.18x their EXIF
value for images 2213/2234/2216 -- and at **0.52x** for 2210, the one image
whose antenna sightline never converged with the others. Seeding (or
fixing) `f` from EXIF removes that degenerate direction.

Convention notes (verified against this dataset's iPhone 13 mini files):
  - `FocalLengthIn35mmFilm` is the focal length that gives the same field
    of view on a 36x24mm frame, so f_px = FL35 / 36 * (long image side).
    The long side is used because 36mm is the *long* dimension of the 35mm
    frame; this is orientation-independent (portrait shots included).
  - Apple's FL35 value already accounts for digital zoom. Confirmed by
    cross-check: 2216 reports FL35=21 on the 13mm-equivalent ultrawide
    (21/13 = 1.62) against DigitalZoomRatio = 1.624; 2234 reports FL35=32
    on the 26mm-equivalent wide (32/26 = 1.23) against DigitalZoomRatio =
    1.258. So do NOT multiply by DigitalZoomRatio again.
"""
import os

from PIL import Image
from PIL.ExifTags import TAGS

FRAME_35MM_LONG_SIDE_MM = 36.0


def exif_tags(img_path):
    with Image.open(img_path) as im:
        raw = im._getexif() or {}
        size = im.size
    return {TAGS.get(t, t): v for t, v in raw.items()}, size


def focal_px(img_path):
    """Return EXIF-derived focal length in pixels, or None if the file has
    no FocalLengthIn35mmFilm tag."""
    tags, size = exif_tags(img_path)
    fl35 = tags.get("FocalLengthIn35mmFilm")
    if fl35 is None:
        return None
    return float(fl35) / FRAME_35MM_LONG_SIDE_MM * float(max(size))


def focal_px_for_key(key, img_dir=".", template="IMG_{key}.jpg"):
    path = os.path.join(img_dir, template.format(key=key))
    if not os.path.exists(path):
        return None
    return focal_px(path)
