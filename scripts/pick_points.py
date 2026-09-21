#!/usr/bin/env python
"""
run from img folder

Interactive point-correspondence camera solver.

Click a feature on the image (left), then click the same feature on the
terrain map (right) — DEM supplies elevation via dem.interp_alt. Repeat for
~6-10 well-spread points (favor ridgelines/peaks near the horizon). Hit
"Solve" to run a nonlinear least-squares fit of (e, n, u, th, ph, ti, f)
that minimizes reprojection error using the *exact* same ray/rotation model
as img.py (pixels_to_rays + rm_ph @ rm_th @ rm_tilt), so there's no PnP
convention mismatch to worry about.

This replaces "does the simulated horizon overlap" (which has many
degenerate/plateaued logL solutions) with "do these N concrete tie-points
reproject correctly" (well-conditioned, and you pick points you trust).

Usage:
  python pick_points.py --which 0
"""
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.optimize import least_squares

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, dtype_r, PRM_ORDER
from eigsep_terrain.img_defaults import load_defaults
from eigsep_terrain.utils import rot_m

IMG_GLOB, CACHE_FILE, DEFAULT_META, DEFAULT_PRMS_U_BY_KEY, IMG_KEYS = load_defaults(
    '/Users/komalkaur/Desktop/eigsep_stuff/eigsep_terrain/eigsep_terrain/defaults.json'
)

NAMES = ["e", "n", "u", "th", "ph", "ti", "f"]


def build_rm(th, ph, ti):
    rm_tilt = rot_m(ti, np.array([0, 0, 1], dtype=np.double))
    rm_th = rot_m(th, np.array([0, 1, 0], dtype=np.double))
    rm_ph = rot_m(ph, np.array([0, 0, 1], dtype=np.double))
    return rm_ph @ (rm_th @ rm_tilt)


def project(params, world_pts, npix_y, npix_x):
    """Project world (E,N,U) points to (u_px, v_px) using the same model as
    img.pixels_to_rays / HorizonImage.get_rays, inverted."""
    e, n, u, th, ph, ti, f = params
    cam = np.array([e, n, u])
    rm = build_rm(th, ph, ti)
    r = world_pts - cam[None, :]          # (N,3) world-frame vectors
    d = r @ rm                            # rm.T @ r, i.e. camera-frame rays (N,3)
    dz = d[:, 2]
    dz = np.where(np.abs(dz) < 1e-9, 1e-9, dz)
    u_px = npix_y // 2 - d[:, 0] * f / dz
    v_px = npix_x // 2 - d[:, 1] * f / dz
    return np.stack([u_px, v_px], axis=1)


def residuals(params, world_pts, img_pts, npix_y, npix_x):
    pred = project(params, world_pts, npix_y, npix_x)
    return (pred - img_pts).ravel()


def solve_camera(world_pts, img_pts, npix_y, npix_x, init, dem):
    world_pts = np.asarray(world_pts, dtype=np.double)
    img_pts = np.asarray(img_pts, dtype=np.double)
    ground = float(dem.interp_alt(init[0], init[1]))
    lo = [-np.inf, -np.inf, ground + 1e-2, -np.pi, -2 * np.pi, -np.pi, 1.0]
    hi = [np.inf, np.inf, np.inf, 2 * np.pi, 2 * np.pi, 2 * np.pi, 1e5]
    x0 = np.array(init, dtype=np.double)
    x0[2] = max(x0[2], lo[2])  # init u may sit exactly on the DEM ground surface
    res = least_squares(
        residuals, x0=x0,
        args=(world_pts, img_pts, npix_y, npix_x),
        bounds=(lo, hi), method="trf",
    )
    return res.x, res


def find_img_file(which, img_glob):
    key = IMG_KEYS[which]
    files = sorted(glob.glob(img_glob))
    matches = [f for f in files if os.path.basename(f).split("_")[-1].split(".")[0] == key]
    if not matches:
        raise FileNotFoundError(f"No file matching key {key!r} via {img_glob!r}")
    return matches[0]


def terrain_plot(dem, ax, erng_m=None, nrng_m=None, decimate=1, cmap="terrain"):
    E, N, U = dem.get_tile(erng_m=erng_m, nrng_m=nrng_m, mesh=False, decimate=decimate)
    extent = (E[0], E[-1], N[0], N[-1])
    im = ax.imshow(U, extent=extent, cmap=cmap, origin="lower", interpolation="nearest")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Elevation [m]")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    return im


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=int, required=True, choices=list(range(len(IMG_KEYS))))
    ap.add_argument("--img-glob", default=IMG_GLOB)
    ap.add_argument("--cache-file", default=CACHE_FILE)
    ap.add_argument("--n-rays", type=int, default=2000)
    ap.add_argument("--fine-delta", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--erng", type=float, nargs=2, default=(200, 3200))
    ap.add_argument("--nrng", type=float, nargs=2, default=(550, 3550))
    ap.add_argument("--decimate", type=int, default=2)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    key = IMG_KEYS[args.which]

    dem = DEM(cache_file=args.cache_file)
    meta = {k: dict(v) for k, v in DEFAULT_META.items()}
    img_file = find_img_file(args.which, args.img_glob)
    img = HorizonImage(img_file, meta, px_smooth=150, px_dist=30)

    init = DEFAULT_PRMS_U_BY_KEY.get(key)
    if init is None:
        raise ValueError(f"No default prms for key {key!r}; add one to defaults.json.")
    init = list(init)

    fig = plt.figure(figsize=(15, 8))
    ax_img = fig.add_axes([0.05, 0.15, 0.42, 0.78])
    ax_terrain = fig.add_axes([0.54, 0.15, 0.42, 0.78])

    ax_img.imshow(img.img, origin="lower")
    ax_img.set_title(f"Image {key} — click a feature", fontsize=10)

    terrain_plot(dem, ax_terrain, erng_m=args.erng, nrng_m=args.nrng, decimate=args.decimate)
    ax_terrain.set_title("click same feature here", fontsize=10)

    fig.suptitle("Point-correspondence camera solve", fontsize=13)

    img_pts = []      # list of (u_px, v_px) i.e. (x=col, y=row) in plotted coords
    terrain_pts = []  # list of (E, N)
    img_markers = []
    terrain_markers = []
    overlay = {"cs": None}

    status_ax = fig.add_axes([0.05, 0.02, 0.6, 0.05])
    status_ax.axis("off")
    status_txt = status_ax.text(0, 0.5, "", va="center", fontsize=9)

    def set_status(msg):
        status_txt.set_text(msg)
        fig.canvas.draw_idle()

    def refresh_status():
        n_pairs = min(len(img_pts), len(terrain_pts))
        pending = "image" if len(img_pts) == len(terrain_pts) else "terrain"
        set_status(f"{n_pairs} complete pair(s). Next click should be on the {pending} panel.")

    def on_click(event):
        if event.inaxes == ax_img and len(img_pts) == len(terrain_pts):
            if event.xdata is None:
                return
            img_pts.append((event.xdata, event.ydata))
            idx = len(img_pts)
            m = ax_img.plot(event.xdata, event.ydata, "o", color="yellow",
                             markeredgecolor="black", markersize=8, zorder=5)[0]
            t = ax_img.annotate(str(idx), (event.xdata, event.ydata),
                                 color="yellow", fontsize=9, fontweight="bold",
                                 xytext=(5, 5), textcoords="offset points")
            img_markers.append((m, t))
            fig.canvas.draw_idle()
        elif event.inaxes == ax_terrain and len(terrain_pts) < len(img_pts):
            if event.xdata is None:
                return
            terrain_pts.append((event.xdata, event.ydata))
            idx = len(terrain_pts)
            m = ax_terrain.plot(event.xdata, event.ydata, "o", color="red",
                                 markeredgecolor="black", markersize=8, zorder=5)[0]
            t = ax_terrain.annotate(str(idx), (event.xdata, event.ydata),
                                     color="red", fontsize=9, fontweight="bold",
                                     xytext=(5, 5), textcoords="offset points")
            terrain_markers.append((m, t))
            fig.canvas.draw_idle()
        else:
            set_status("Click ignored — wrong panel for the next point (see status).")
            return
        refresh_status()

    fig.canvas.mpl_connect("button_press_event", on_click)

    def on_undo(_):
        if len(img_pts) > len(terrain_pts):
            img_pts.pop()
            m, t = img_markers.pop()
            m.remove(); t.remove()
        elif terrain_pts:
            terrain_pts.pop()
            m, t = terrain_markers.pop()
            m.remove(); t.remove()
        fig.canvas.draw_idle()
        refresh_status()

    def on_clear(_):
        img_pts.clear(); terrain_pts.clear()
        for m, t in img_markers: m.remove(); t.remove()
        for m, t in terrain_markers: m.remove(); t.remove()
        img_markers.clear(); terrain_markers.clear()
        if overlay["cs"] is not None:
            overlay["cs"].remove(); overlay["cs"] = None
        fig.canvas.draw_idle()
        refresh_status()

    def on_solve(_):
        n_pairs = min(len(img_pts), len(terrain_pts))
        if n_pairs < 4:
            set_status(f"Need >=4 complete pairs to solve (have {n_pairs}).")
            return
        world_pts = []
        for (E, N) in terrain_pts[:n_pairs]:
            U = float(dem.interp_alt(E, N))
            world_pts.append((E, N, U))
        solved, res = solve_camera(
            world_pts, img_pts[:n_pairs], img.npix_y, img.npix_x, init, dem
        )
        rmse = np.sqrt(np.mean(res.fun ** 2))
        img.set_prms(tuple(solved))
        prms_u = tuple(round(float(v), 4) for v in solved)
        print("prms_u =", prms_u, f" reprojection RMSE={rmse:.2f}px")
        set_status(f"Solved with {n_pairs} pairs. Reprojection RMSE={rmse:.2f}px "
                   f"(see console for prms_u). Fill defaults.json with these values.")

        # draw predicted horizon overlay for a visual sanity check
        stride = 10
        ys = np.arange(0, img.npix_y, stride)
        xs = np.arange(0, img.npix_x, stride)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        x_px, y_px = yy.ravel(), xx.ravel()
        rays = img.get_rays(pixels=(x_px, y_px), dtype=dtype_r)
        r = img.ray_distance(dem, rays, dtype=dtype_r, fine_delta=args.fine_delta)
        model_sky = np.isnan(r).reshape(yy.shape)
        if overlay["cs"] is not None:
            overlay["cs"].remove()
        overlay["cs"] = ax_img.contour(
            xx, yy, model_sky.astype(float), levels=[0.5], colors="red", linewidths=1.5,
        )
        fig.canvas.draw_idle()

    undo_ax = fig.add_axes([0.70, 0.02, 0.08, 0.05])
    clear_ax = fig.add_axes([0.80, 0.02, 0.08, 0.05])
    solve_ax = fig.add_axes([0.90, 0.02, 0.08, 0.05])
    undo_btn = Button(undo_ax, "Undo")
    clear_btn = Button(clear_ax, "Clear")
    solve_btn = Button(solve_ax, "Solve")
    undo_btn.on_clicked(on_undo)
    clear_btn.on_clicked(on_clear)
    solve_btn.on_clicked(on_solve)

    refresh_status()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())