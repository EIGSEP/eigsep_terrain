# Marjum geometry model sources (frozen at v0004)

These five files are the camera, bundle and MCMC model code behind the
Marjum 2026-07 joint geometry fit. Geometry posterior v0004
(`marjum-2026-07/derived/geometry_posterior/v0004/`) pins them by SHA-256
in its `input_manifest.json`, under their old paths in the retired
`terrain/` workspace checkout. They were copied here byte for byte so that
release stays reproducible after `terrain/` is removed. `PINS.json` maps
each recorded `terrain/` path to its file here and its hash.

| File | Role |
|---|---|
| `marjum_bundle.py` | DEM terrain wrapper, camera parameter order, rotations, projection |
| `marjum_camera.py` | camera projection and ray helpers |
| `marjum_fitio.py` | reading and writing fitted camera states |
| `marjum_mcmc.py` | single-camera likelihood pieces, focal priors, input digests |
| `marjum_mcmc_b21.py` | the joint 29-camera sampler used for v0003 and v0004 |

They are scripts, not package modules: they import one another as
top-level modules (`from marjum_bundle import ...`), and some read paths
relative to the working directory. There is no `__init__.py`, so they are
not importable as `eigsep_terrain.marjum_geometry` and are not shipped in
a wheel. Run code puts this directory on `sys.path`, as the v0004 drivers in
`data-analysis/scripts/marjum-2026-07/geometry/` do. `marjum_bundle.py`
needs `opencv-python` (`cv2`).

Do not edit these files: any change breaks v0004's source check. A later
release will refactor them into proper modules with tests and pin a single
`eigsep_terrain` commit instead.
