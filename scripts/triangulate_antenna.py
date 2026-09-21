#!/usr/bin/env python
"""
run from img folder (2026_imgs/), same convention as fit_horizon_exact.py

Take the "three good" polished camera fits (fit_horizon_exact_polished.json:
2213, 2234, 2210), build the antenna sight-line (world-frame ray from each
camera's position through its ant_px) for each, and triangulate their
closest-approach point in ENU. Then project that single triangulated 3D
point back into each of the three images and compare it against the actual
ant_px -- if the fits and the antenna picks are all self-consistent, the
reprojected point should land close to ant_px in every image.

Uses the exact same pixel/ray model as img.py's get_rays / ant_logL and
pick_points.py's project() (its exact inverse) -- no HorizonImage instance
needed (avoids pulling in torch/transformers/pymc just to read pixels).

IMPORTANT pixel-convention fix: defaults.json's ant_px = [x, y] stores y in
standard top-down image coordinates (row 0 = top, as read directly off the
raw photo), but HorizonImage flips every image vertically on load
(`np.flipud`) and every ray/sky-mask computation in img.py operates in that
flipped, row-0-at-bottom frame. Verified against the segmentation sky mask
across all 34 images: treating ant_px's row as already flipped puts the
antenna pixel on the *ground* in 32/33 cases; converting it via
`row_flipped = npix_y - 1 - row_y` puts it in the sky in 26/33 (the other
~6 plausibly have the antenna against a ridge, not open sky). So this
script converts ant_px's row before building rays / after reprojecting.

Usage:
  python triangulate_antenna.py
  python triangulate_antenna.py --keys 2213 2234 2210 --out-dir reproj_plots
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.utils import rot_m

DEFAULTS_PATH = "/Users/komalkaur/Desktop/eigsep_stuff/eigsep_terrain/eigsep_terrain/defaults.json"
POLISHED_PATH = "fit_horizon_exact_polished.json"
IMG_GLOB_TEMPLATE = "IMG_{key}.jpg"
GOOD_KEYS = ("2213", "2234", "2216", "2210")


def build_rm(th, ph, ti):
    rm_tilt = rot_m(ti, np.array([0, 0, 1], dtype=np.double))
    rm_th = rot_m(th, np.array([0, 1, 0], dtype=np.double))
    rm_ph = rot_m(ph, np.array([0, 0, 1], dtype=np.double))
    return rm_ph @ (rm_th @ rm_tilt)


def pixel_to_ray(row, col, npix_y, npix_x, f, rm):
    """World-frame unit vector for pixel (row, col). Exact match to
    img.pixels_to_rays + HorizonImage.get_rays."""
    cam_ray = np.array([npix_y // 2 - row, npix_x // 2 - col, f], dtype=np.double)
    cam_ray /= np.linalg.norm(cam_ray)
    return rm @ cam_ray


def project_point(world_pt, cam, rm, f, npix_y, npix_x):
    """World (E,N,U) -> pixel (col, row) = (x, y). Exact inverse of
    pixel_to_ray / toy_problem.project_ant_to_pixel."""
    d = np.asarray(world_pt, dtype=np.double) - cam
    d_cam = rm.T @ d
    row = npix_y // 2 - d_cam[0] / d_cam[2] * f
    col = npix_x // 2 - d_cam[1] / d_cam[2] * f
    return col, row, d_cam[2]  # d_cam[2] > 0 means in front of camera


def triangulate(cams, dirs):
    """Least-squares closest point to a set of 3D lines (cam_i + t*dir_i).
    Minimizes sum_i || (I - d_i d_i^T) (X - cam_i) ||^2."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for cam, d in zip(cams, dirs):
        d = d / np.linalg.norm(d)
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ cam
    X = np.linalg.solve(A, b)
    return X


def ray_to_ray_closest_points(cam_a, d_a, cam_b, d_b):
    """Closest-approach points/params on two 3D lines cam_i + t*d_i.
    Returns (t_a, t_b, point_a, point_b, gap). t_i > 0 means the closest
    point is in front of camera i (along its actual viewing direction,
    not just anywhere on the infinite line)."""
    d_a = d_a / np.linalg.norm(d_a)
    d_b = d_b / np.linalg.norm(d_b)
    A = np.array([[np.dot(d_a, d_a), -np.dot(d_a, d_b)],
                  [np.dot(d_a, d_b), -np.dot(d_b, d_b)]])
    r = cam_b - cam_a
    rhs = np.array([np.dot(d_a, r), np.dot(d_b, r)])
    t_a, t_b = np.linalg.solve(A, rhs)
    p_a = cam_a + t_a * d_a
    p_b = cam_b + t_b * d_b
    return t_a, t_b, p_a, p_b, np.linalg.norm(p_a - p_b)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="+", default=list(GOOD_KEYS))
    ap.add_argument("--polished", default=POLISHED_PATH)
    ap.add_argument("--defaults", default=DEFAULTS_PATH)
    ap.add_argument("--cache-file", default="marjum_dem.npz")
    ap.add_argument("--out-dir", default="reproj_plots")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    with open(args.polished) as f:
        polished = json.load(f)
    with open(args.defaults) as f:
        defaults = json.load(f)["images"]

    dem = DEM(cache_file=args.cache_file)

    cams, dirs, rms, prms_by_key, dims_by_key, ant_px_by_key = {}, {}, {}, {}, {}, {}
    for key in args.keys:
        e, n, u, th, ph, ti, f = polished[key]["prms_u"]
        cam = np.array([e, n, u], dtype=np.double)
        rm = build_rm(th, ph, ti)

        img_path = IMG_GLOB_TEMPLATE.format(key=key)
        img_shape = imread(img_path).shape  # (rows, cols, 3), NOT y-flipped -- shape unaffected by flip
        npix_y, npix_x = img_shape[0], img_shape[1]

        col_x, row_y_top = defaults[key]["ant_px"]  # (x, y) = (col, row-from-TOP, as picked)
        row_y = npix_y - 1 - row_y_top  # convert to img.py's flipped, row-from-bottom frame
        ray = pixel_to_ray(row_y, col_x, npix_y, npix_x, f, rm)

        cams[key] = cam
        dirs[key] = ray
        rms[key] = rm
        prms_by_key[key] = (e, n, u, th, ph, ti, f)
        dims_by_key[key] = (npix_y, npix_x)
        ant_px_by_key[key] = (col_x, row_y)

        print(f"[{key}] cam=({e:.2f}, {n:.2f}, {u:.2f})  ant_px=({col_x}, {row_y})  "
              f"ray_dir=({ray[0]:.4f}, {ray[1]:.4f}, {ray[2]:.4f})")

    # pairwise ray-to-ray closest approach (sanity check before trusting the
    # combined least-squares triangulation): if a pair's t_a, t_b are both
    # positive, that pair's sight-lines cross in front of both cameras --
    # a geometrically valid candidate intersection. Negative t means the
    # closest point on that camera's line is actually *behind* it, i.e.
    # the two rays don't really point at a shared point in front of both.
    print("\nPairwise closest-approach between antenna sight-lines:")
    for i, ka in enumerate(args.keys):
        for kb in args.keys[i + 1:]:
            t_a, t_b, p_a, p_b, gap = ray_to_ray_closest_points(
                cams[ka], dirs[ka], cams[kb], dirs[kb])
            valid = "OK (both in front)" if (t_a > 0 and t_b > 0) else "** inconsistent (behind one/both cameras) **"
            print(f"  {ka} <-> {kb}: gap={gap:.2f} m  "
                  f"t_{ka}={t_a:.1f}m  t_{kb}={t_b:.1f}m  {valid}")

    X = triangulate([cams[k] for k in args.keys], [dirs[k] for k in args.keys])
    ground_u = float(dem.interp_alt(X[0], X[1]))
    print(f"\nTriangulated antenna position (E, N, U) = "
          f"({X[0]:.3f}, {X[1]:.3f}, {X[2]:.3f})")
    print(f"  ground elevation at that (E,N) = {ground_u:.3f} m  "
          f"(height above ground = {X[2] - ground_u:.3f} m)")
    for key in args.keys:
        r = X - cams[key]
        print(f"  distance from {key}'s camera = {np.linalg.norm(r):.2f} m")

    os.makedirs(args.out_dir, exist_ok=True)
    print("\nReprojection of triangulated point back into each image:")
    for key in args.keys:
        e, n, u, th, ph, ti, f = prms_by_key[key]
        npix_y, npix_x = dims_by_key[key]
        col_x, row_y = ant_px_by_key[key]
        col_pred, row_pred, depth = project_point(X, cams[key], rms[key], f, npix_y, npix_x)
        dx, dy = col_pred - col_x, row_pred - row_y
        err_px = np.hypot(dx, dy)
        behind = "  ** BEHIND CAMERA **" if depth <= 0 else ""
        print(f"  [{key}] actual ant_px=({col_x}, {row_y})  "
              f"reprojected=({col_pred:.1f}, {row_pred:.1f})  "
              f"error={err_px:.1f} px{behind}")

        img = np.flipud(imread(IMG_GLOB_TEMPLATE.format(key=key)))
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(img, origin="lower")
        ax.plot(col_x, row_y, "y*", ms=18, markeredgecolor="k",
                markeredgewidth=0.8, label="actual ant_px")
        ax.plot(col_pred, row_pred, "rx", ms=14, mew=3,
                label="reprojected triangulated point")
        ax.plot([col_x, col_pred], [row_y, row_pred], "r--", lw=1, alpha=0.7)
        ax.set_title(f"{key}: reprojection error = {err_px:.1f} px", fontsize=11)
        ax.set_xlabel("pixel x (col)")
        ax.set_ylabel("pixel y (row, 0=bottom)")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"reproj_{key}.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"    saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
