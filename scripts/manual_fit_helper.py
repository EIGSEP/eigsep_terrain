#!/usr/bin/env python
"""
run from img folder (import as a module, not a CLI)

Headless harness that drives the exact same underlying functions as
pick_points.py (point-correspondence least-squares) and tune_image.py
(ray-traced overlay + horizon_ray_logL) so an operator who can't drive the
matplotlib GUIs directly can still alternate between the two: pick
correspondence points by inspecting rendered reference images, solve, then
inspect a rendered verification overlay and hand-adjust params, repeat.

Not a CLI -- import the functions from a Python shell / script.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "../scripts")
from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, dtype_r
from eigsep_terrain.img_defaults import load_defaults
import pick_points as pp

IMG_GLOB, CACHE_FILE, DEFAULT_META, DEFAULT_PRMS_U_BY_KEY, IMG_KEYS = load_defaults(
    '/Users/komalkaur/Desktop/eigsep_stuff/eigsep_terrain/eigsep_terrain/defaults.json'
)
NAMES = ["e", "n", "u", "th", "ph", "ti", "f"]

_dem = None
def get_dem():
    global _dem
    if _dem is None:
        _dem = DEM(cache_file=CACHE_FILE)
    return _dem


def load_img(which):
    key = IMG_KEYS[which]
    meta = {k: dict(v) for k, v in DEFAULT_META.items()}
    img_file = pp.find_img_file(which, IMG_GLOB)
    img = HorizonImage(img_file, meta, px_smooth=150, px_dist=30)
    init = list(DEFAULT_PRMS_U_BY_KEY[key])
    return key, img, init


def hillshade(U, res, azdeg=315, altdeg=45):
    """Standard hillshade so ridgelines/peaks are visible -- a flat elevation
    colormap alone doesn't show terrain shape."""
    gy, gx = np.gradient(U.astype(np.float64), res)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.deg2rad(azdeg)
    alt = np.deg2rad(altdeg)
    shaded = (np.sin(alt) * np.sin(slope) +
              np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip(shaded, 0, 1)


def render_reference(which, out_path, erng=(200, 3200), nrng=(550, 3550), decimate=2):
    """Photo (with actual sky-mask contour + pixel gridlines) alongside a
    hillshaded terrain elevation map (with E/N gridlines) -- for reading off
    correspondence points by eye, same information pick_points.py's GUI
    shows (plus hillshading, since the flat elevation colormap alone makes
    ridgelines/peaks very hard to make out)."""
    key, img, init = load_img(which)
    dem = get_dem()

    fig = plt.figure(figsize=(16, 8))
    ax_img = fig.add_axes([0.04, 0.08, 0.44, 0.86])
    ax_ter = fig.add_axes([0.54, 0.08, 0.44, 0.86])

    ax_img.imshow(img.img, origin="lower")
    ax_img.contour(img.sky_mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.2)
    ax_img.set_xticks(np.arange(0, img.npix_x, 200))
    ax_img.set_yticks(np.arange(0, img.npix_y, 200))
    ax_img.grid(color="yellow", alpha=0.3, linewidth=0.5)
    ax_img.tick_params(labelsize=6)
    ax_img.set_title(f"{key}: photo (row,col) -- cyan = actual mask", fontsize=10)

    E, N, U = dem.get_tile(erng_m=erng, nrng_m=nrng, mesh=False, decimate=decimate)
    extent = (E[0], E[-1], N[0], N[-1])
    shade = hillshade(U, dem.res * decimate)
    ax_ter.imshow(shade, extent=extent, cmap="gray", origin="lower", interpolation="nearest")
    im = ax_ter.imshow(U, extent=extent, cmap="terrain", origin="lower",
                       interpolation="nearest", alpha=0.45)
    plt.colorbar(im, ax=ax_ter, label="Elevation [m]")
    ax_ter.plot(init[0], init[1], marker="*", color="red", markeredgecolor="black", markersize=14)
    ax_ter.set_xticks(np.arange(erng[0], erng[1] + 1, 200))
    ax_ter.set_yticks(np.arange(nrng[0], nrng[1] + 1, 200))
    ax_ter.grid(color="white", alpha=0.3, linewidth=0.5)
    ax_ter.tick_params(labelsize=7)
    ax_ter.set_xlabel("East [m]")
    ax_ter.set_ylabel("North [m]")
    ax_ter.set_title(f"terrain (red star = seed cam pos {init[0]:.0f},{init[1]:.0f})", fontsize=10)

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return key, init


def solve(which, points, init=None):
    """points: list of ((img_row, img_col), (E, N)) pairs, exactly what a
    human would click in pick_points.py. Returns (solved_prms, rmse)."""
    key, img, default_init = load_img(which)
    if init is None:
        init = default_init
    dem = get_dem()
    img_pts = [(c, r) for (r, c), (E, N) in points]   # pick_points stores (xdata=col, ydata=row)
    world_pts = []
    for (r, c), (E, N) in points:
        U = float(dem.interp_alt(E, N))
        world_pts.append((E, N, U))
    solved, res = pp.solve_camera(world_pts, img_pts, img.npix_y, img.npix_x, init, dem)
    rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    return list(solved), rmse


def solve_fixed(which, points, fix, init=None):
    """Like solve(), but holds the params named in `fix` (subset of NAMES)
    at their init value and only solves the rest -- much better conditioned
    when position is already known (e.g. GPS) and only orientation/focal
    length need fitting from a few noisy manually-picked points."""
    from scipy.optimize import least_squares
    key, img, default_init = load_img(which)
    if init is None:
        init = default_init
    init = list(init)
    dem = get_dem()
    img_pts = np.array([(c, r) for (r, c), (E, N) in points], dtype=np.double)
    world_pts = np.array([(E, N, float(dem.interp_alt(E, N))) for (r, c), (E, N) in points],
                         dtype=np.double)

    ground = float(dem.interp_alt(init[0], init[1]))
    lo_all = [-np.inf, -np.inf, ground + 1e-2, -np.pi, -2 * np.pi, -np.pi, 1.0]
    hi_all = [np.inf, np.inf, np.inf, 2 * np.pi, 2 * np.pi, 2 * np.pi, 1e5]
    free_idx = [i for i, name in enumerate(NAMES) if name not in fix]
    x0 = [init[i] for i in free_idx]
    lo = [lo_all[i] for i in free_idx]
    hi = [hi_all[i] for i in free_idx]

    def full_params(x_free):
        p = list(init)
        for i, v in zip(free_idx, x_free):
            p[i] = v
        return p

    def resid(x_free):
        p = full_params(x_free)
        pred = pp.project(p, world_pts, img.npix_y, img.npix_x)
        return (pred - img_pts).ravel()

    res = least_squares(resid, x0=np.array(x0, dtype=np.double), bounds=(lo, hi), method="trf")
    solved = full_params(res.x)
    rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    return solved, rmse


def render_verify(which, prms, out_path, stride=8, n_rays=3000, fine_delta=0.25):
    """tune_image.py's exact verification view: red = ray-traced model
    contour at `prms`, cyan = actual sky-mask contour, title = logL."""
    key, img, _ = load_img(which)
    dem = get_dem()
    img.set_prms(tuple(prms))

    ys = np.arange(0, img.npix_y, stride)
    xs = np.arange(0, img.npix_x, stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    x_px, y_px = yy.ravel(), xx.ravel()
    actual_sky = img.sky_mask[np.ix_(ys, xs)] > 0.5

    rays = img.get_rays(pixels=(x_px, y_px), dtype=dtype_r)
    r = img.ray_distance(dem, rays, dtype=dtype_r, fine_delta=fine_delta)
    model_sky = np.isnan(r).reshape(yy.shape)

    logL = img.horizon_ray_logL(dem, n_rays=n_rays, eps=1e-2, fine_delta=fine_delta)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(img.img, origin="lower")
    ax.contour(xx, yy, actual_sky.astype(float), levels=[0.5], colors="cyan", linewidths=1.5)
    ax.contour(xx, yy, model_sky.astype(float), levels=[0.5], colors="red", linewidths=1.5)
    ax.set_title(f"{key}  logL={logL:.1f}  prms=({prms[0]:.1f},{prms[1]:.1f},{prms[2]:.1f},"
                f"{prms[3]:.3f},{prms[4]:.3f},{prms[5]:.3f},{prms[6]:.1f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return logL
