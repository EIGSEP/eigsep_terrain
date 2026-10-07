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
recomputation; old DEM caches are ignored and Marjum rebuilds them from
GeoTIFFs. Generic `DEM` callers must reload TIFFs after a legacy cache.

Marjum's default survey offset is now `[0, 0, 3]` metres. The previous
horizontal correction `[-11, 36]` absorbed the faulty coordinate frame
and must be recalibrated from GPS benchmarks. The vertical correction
is retained. `LEGACY_SURVEY_OFFSET` preserves the previous value for
explicit reproduction of old results. No new benchmark calibration is
claimed by this change.
