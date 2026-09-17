# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`eigsep_terrain` is a Python package providing Digital Elevation Model (DEM) tools for the EIGSEP (Experiment to Investigate the Global Signal Epoch of reionization with Parents) radio telescope. The primary application is localizing camera positions and antenna positions by fitting ray-traced horizon images against DEM terrain data using Bayesian MCMC inference.

## Installation & Setup

```bash
pip install .                    # standard install
pip install ".[numba]"           # with numba acceleration
jupyter lab                      # to explore notebooks
```

## Common Commands

```bash
# Run all tests
pytest eigsep_terrain/tests/

# Run a single test file
pytest eigsep_terrain/tests/test_ray.py

# Run a single test by name
pytest eigsep_terrain/tests/test_ray.py::test_ray_trace_basic_flat_ground_downward_matches_analytic_within_step

# Run MCMC solver (installed as script after pip install)
eigsep_terrain_pymc.py --cache-file marjum_dem.npz --img-glob "/path/to/IMG_08*.jpg"

# Run MCMC solver directly
python scripts/eigsep_terrain_pymc.py --help
```

## Architecture

### Core pipeline

1. **DEM loading** (`dem.py`, `marjum_dem.py`): `DEM` (dict subclass) loads GeoTIFF elevation tiles + XML metadata from USGS, converts lat/lon to ENU (East-North-Up) coordinates in meters, and caches as `.npz`. `MarjumDEM` is a site-specific subclass that auto-downloads Marjum Pass USGS tiles.

2. **Ray tracing** (`ray.py`, `ray_numba.py`): `ray_trace_basic()` marches rays from a 3D start point through the DEM grid in steps of `delta_r_m` meters, returning intersection distance or `NaN` for misses. `ray_trace_basic_numba()` is a parallel Numba-accelerated drop-in replacement. Rays are represented as `(3, N)` float32 arrays in ENU coordinates.

3. **Image segmentation** (`seg.py`): `TiledSkyProbSegFormer` runs a HuggingFace SegFormer model (`nvidia/segformer-b0-finetuned-ade-512-512`) on overlapping tiles with Hann-window blending to produce `P(sky)` probability maps at full image resolution.

4. **Horizon image fitting** (`img.py`): `HorizonImage` wraps a camera image. It stores camera pose parameters `(e, n, u, th, ph, ti, f)` = (ENU position in m, tilt angles in rad, focal length in px), projects image pixels to 3D rays via `pixels_to_rays()`, and computes log-likelihood by comparing ray-traced terrain intersections against the segmented sky probability map. Segmentation results are cached in `img_seg_*.npz` files alongside the images. `PositionSolver` aggregates multiple `HorizonImage` objects for joint MCMC fitting.

5. **MCMC inference** (`scripts/eigsep_terrain_pymc.py`): Uses PyMC with `DEMetropolisZ` sampler and a custom `as_op` wrapping `PositionSolver.total_logL`. Output is an ArviZ NetCDF trace file (`trace_seed*.nc`).

### Key conventions

- All spatial coordinates are **ENU in meters**, origin at `map_crd['southbc'], map_crd['westbc']` minus `survey_offset`.
- `DEM.data` is indexed as `[n_px, e_px]` (row = north, col = east), with `(0,0)` at the south-west corner.
- Camera parameters `PRM_ORDER = ('e', 'n', 'u', 'th', 'ph', 'ti', 'f')`: tilt `ti` is in-plane rotation around z-axis applied first, then elevation `th` around y-axis, then azimuth `ph` around z-axis.
- `ray_trace_basic` accepts rays as `(3, N)` arrays; it does **not** accept HealPix nside directly — call `healpix_rays(nside)` first.
- Numba JIT compilation happens on first call; subsequent calls are fast. The numba kernel is `cache=True` so it persists across runs.
- Image pixels are stored **flipped** (`np.flipud`) on load to match the ENU convention (y increases upward).
- Line length: 79 characters (Black formatter, see `pyproject.toml`).
