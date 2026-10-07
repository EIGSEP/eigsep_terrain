# eigsep_terrain

Tools for EIGSEP Digital Elevation Models.



Run `pip install .` then `jupyter lab` to get started.

### DEM coordinates and azimuths

GeoTIFF DEMs use local **UTM grid metres**, with `(0, 0)` at the
southwest pixel centre of the loaded mosaic. `latlon_to_raster` and
`raster_to_latlon` use the geographic datum recorded in the TIFF (Marjum:
NAD83(2011), EPSG:6341). Heights are passed through in the DEM vertical
datum; GPS ellipsoidal heights require a separate geoid correction.
WGS84 GPS lat/lon may differ from NAD83(2011) by roughly a metre here.
`latlon_to_enu` / `enu_to_latlon` are compatibility names for these grid
conversions; `frame="tangent"` retains the historical tangent-plane code
for reproducing earlier results, not for indexing the UTM raster.

`calc_horizon` and `DEM.ray_trace` now use true azimuth by default, applying
UTM convergence at the observer. Use `azimuth_frame="grid"` for grid
azimuths. Low-level ray tracers and camera poses still use raster grid
axes: convert true bearings before passing rays directly to them. UTM
metres retain the projection scale factor; they are not exact ground
metres. Observer convergence approximates bearings across the local DEM.

Horizon bins hold conservative maxima over pixels touching each bin,
not point samples. Prefer `n_az >= 1440` when interpolating a horizon,
and record `azimuth_frame` alongside saved horizon/ray products.
Previously saved horizons, camera fits and raster positions need
recomputation. Pre-UTM DEM caches raise an error, including with
`clear_cache=True`, so opening a pinned product cannot silently replace it.
Build the UTM cache at a new path from the original GeoTIFF tile list. For
Marjum `dem/v0001`, that list is 3 east by 4 north tiles; the package's
default `get_tif_files()` selects a different 5-by-3 footprint. Verify the
new cache's shape and elevation values against the old product before using
it, and publish it as a new version with its own checksum.

Marjum's default survey offset is now `[0, 0, 3]` metres. The previous
horizontal correction `[-11, 36]` was established with the old coordinate
conversion. The new zero horizontal default is provisional pending
validation against independently identified survey features. The vertical
correction is retained. `LEGACY_SURVEY_OFFSET` preserves the previous value for
explicit reproduction of old results. No new benchmark calibration is
claimed by this change.

The executed [UTM Frame Survey Review notebook](notebooks/UTM%20Frame%20Survey%20Review.ipynb)
compares terrain placement, local translations, survey elevations and
azimuths. Its editable reference tables support field calibration and
held-out checks; empty tables explicitly mark validation as pending.

EXIF focal conversion uses the full image diagonal relative to a 36 × 24 mm
frame, so portrait rotation preserves the inferred focal length in pixels.
The supplied dimensions must retain the EXIF field of view; cropping needs
a separate intrinsics transform.

## Recent changes

- 2026-10-07: Aligned the executed UTM survey review with the published 3-by-4 Marjum DEM footprint and its source checksums, so the horizon preview uses the same terrain extent as v0002.
- 2026-10-07: Reject pre-UTM caches before any rebuild, preserving pinned DEM products and requiring an explicit source footprint for replacements.
- 2026-10-04: Corrected EXIF focal conversion for portrait images and non-3:2
  image shapes; added shared conversion and rotation/resize checks.
