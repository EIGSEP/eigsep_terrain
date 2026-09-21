#!/usr/bin/env python
"""
run from img folder (2026_imgs/)

Decisive test of whether radial lens distortion (unmodeled by the pinhole
camera used everywhere else in this repo) explains part of the ~20-35px
horizon-fit floor and/or the focal length drifting 7-18% above EXIF.

docs/fitting_improvement_plan.md section 7.1 tested this once already by
checking whether |residual| correlates with distance from image center --
found nothing, and concluded distortion wasn't the cause. That's a weak
test: distortion is quadratic in radius and partly absorbable by `f`, so a
flat correlation doesn't rule it out. This script instead actually fits a
Brown-Conrady radial term k1 (optionally k2) and asks whether it earns its
keep.

Design: holds each image's (e, n, u, th, ph, ti) FIXED at its current
best-fit pose (from fit_horizon_exact_polished.json) and refits only (f,
k1[, k2]) against that image's own horizon curve. This isolates the
distortion question cleanly -- no joint re-optimization, no confound from
position/orientation moving to compensate. Two of the four images
(2213, 2234) share the iPhone 13 mini's main "wide" lens (5.1mm / 26mm-
equiv per EXIF); the other two (2216, 2210) share its "ultrawide" lens
(1.54mm / 13mm-equiv) -- physically distinct optics, so distortion is
fit per-image first, then checked for within-lens-group agreement as a
sanity check (if it's real distortion, not overfitting, same-lens images
should land on similar k1).

Usage:
  python fit_distortion.py --keys 2213 2234 2216 2210
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eigsep_terrain.marjum_dem import MarjumDEM as DEM
import fit_horizon_exact as fhe
import triangulate_antenna as ta
from exif_focal import focal_px_for_key

LENS_GROUPS = {"2213": "wide", "2234": "wide", "2216": "ultrawide", "2210": "ultrawide"}


def pixel_to_ray_k(row, col, npix_y, npix_x, f, rm, k1, k2=0.0, n_iter=8):
    """Vectorized pixel(s) -> world ray unit vector(s), undoing radial
    distortion first. Degenerates exactly to triangulate_antenna.pixel_to_ray
    when k1=k2=0 (same [Nu//2-row, Nv//2-col, f] convention)."""
    xd = (npix_y // 2 - row) / f
    yd = (npix_x // 2 - col) / f
    xn, yn = np.array(xd, dtype=np.float64), np.array(yd, dtype=np.float64)
    for _ in range(n_iter):
        r2 = xn ** 2 + yn ** 2
        factor = 1 + k1 * r2 + k2 * r2 ** 2
        xn = xd / factor
        yn = yd / factor
    cam_ray = np.stack([xn, yn, np.ones_like(xn)], axis=0)
    cam_ray /= np.linalg.norm(cam_ray, axis=0, keepdims=True)
    return np.einsum("ij,j...->i...", rm, cam_ray)


def project_point_k(world_pt, cam, rm, f, npix_y, npix_x, k1, k2=0.0):
    """World point -> pixel, applying forward radial distortion. Degenerates
    exactly to triangulate_antenna.project_point when k1=k2=0."""
    d = np.asarray(world_pt, dtype=np.double) - cam
    d_cam = rm.T @ d
    xn, yn = d_cam[0] / d_cam[2], d_cam[1] / d_cam[2]
    r2 = xn ** 2 + yn ** 2
    factor = 1 + k1 * r2 + k2 * r2 ** 2
    row = npix_y // 2 - (xn * factor) * f
    col = npix_x // 2 - (yn * factor) * f
    return col, row, d_cam[2]


def predicted_horizon_rows_k(npix_y, npix_x, hangles, cols, f, rm, k1, k2, n_iter=14):
    lo = np.zeros(len(cols), dtype=np.float64)
    hi = np.full(len(cols), npix_y - 1, dtype=np.float64)
    cols_f = cols.astype(np.float64)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        rays = pixel_to_ray_k(mid, cols_f, npix_y, npix_x, f, rm, k1, k2)
        el, az = fhe.elevation_azimuth(rays)
        hz = fhe.interp_hangles(hangles, az)
        model_sky = el > hz
        hi = np.where(model_sky, mid, hi)
        lo = np.where(model_sky, lo, mid)
    return (lo + hi) / 2


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="+", default=["2213", "2234", "2216", "2210"])
    ap.add_argument("--seed", default="fit_horizon_exact_polished.json")
    ap.add_argument("--cache-file", default="marjum_dem.npz")
    ap.add_argument("--n-az", type=int, default=2048)
    ap.add_argument("--ncols", type=int, default=500)
    ap.add_argument("--bisect-iters", type=int, default=14)
    ap.add_argument("--fit-k2", action="store_true")
    ap.add_argument("--out", default="fit_distortion_results.json")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    dem = DEM(cache_file=args.cache_file)
    seed_json = json.load(open(args.seed))

    class LoadArgs:
        img_glob = fhe.IMG_GLOB
        ncols = args.ncols

    out = {}
    print(f"{'key':6} {'lens':10} {'f_exif':>8} {'f_old':>8} {'f_old/exif':>10} | "
          f"{'RMSE base':>10} {'RMSE f-only':>12} {'RMSE f+k1':>10} {'f+k1/exif':>10} "
          f"{'k1':>10}  (+k2 RMSE, k2 if requested)")
    for key in args.keys:
        which = fhe.IMG_KEYS.index(key)
        _, img, _, cols, y_actual = fhe.load_image(which, LoadArgs(), dem)
        e, n, u, th, ph, ti, f_old = seed_json[key]["prms_u"]
        rm = ta.build_rm(th, ph, ti)
        hangles, _ = dem.calc_horizon(e, n, u, n_az=args.n_az)
        f_exif = focal_px_for_key(key)

        def rmse_of(f, k1, k2=0.0):
            yp = predicted_horizon_rows_k(img.npix_y, img.npix_x, hangles, cols,
                                          f, rm, k1, k2, n_iter=args.bisect_iters)
            return float(np.sqrt(np.mean((yp - y_actual) ** 2)))

        rmse_base = rmse_of(f_old, 0.0)  # sanity check vs fhe's own number

        def resid_f(x):
            return predicted_horizon_rows_k(img.npix_y, img.npix_x, hangles, cols,
                                            x[0], rm, 0.0, 0.0, n_iter=args.bisect_iters) - y_actual
        res_f = least_squares(resid_f, x0=[f_old], bounds=([1.0], [1e5]),
                              diff_step=1e-3, loss="soft_l1", f_scale=5.0, x_scale="jac")
        rmse_f_only = float(np.sqrt(np.mean(res_f.fun ** 2)))

        def resid_fk1(x):
            return predicted_horizon_rows_k(img.npix_y, img.npix_x, hangles, cols,
                                            x[0], rm, x[1], 0.0, n_iter=args.bisect_iters) - y_actual
        # k1 seeded away from exactly 0 -- with x0=0 the parameter's initial
        # Jacobian column can be degenerate/tiny relative to f's, and even
        # x_scale='jac' can't rescale a column it hasn't sampled yet.
        res_fk1 = least_squares(resid_fk1, x0=[f_old, 0.01], bounds=([1.0, -5.0], [1e5, 5.0]),
                                diff_step=1e-3, loss="soft_l1", f_scale=5.0, x_scale="jac")
        rmse_fk1 = float(np.sqrt(np.mean(res_fk1.fun ** 2)))
        f_new, k1_new = res_fk1.x

        extra = ""
        k2_new = None
        if args.fit_k2:
            def resid_fk1k2(x):
                return predicted_horizon_rows_k(img.npix_y, img.npix_x, hangles, cols,
                                                x[0], rm, x[1], x[2], n_iter=args.bisect_iters) - y_actual
            k2_seed = 0.01 if abs(k1_new) < 1e-6 else 0.0
            res3 = least_squares(resid_fk1k2, x0=[f_new, k1_new, k2_seed],
                                 bounds=([1.0, -5.0, -5.0], [1e5, 5.0, 5.0]),
                                 diff_step=1e-3, loss="soft_l1", f_scale=5.0, x_scale="jac")
            rmse_fk1k2 = float(np.sqrt(np.mean(res3.fun ** 2)))
            k2_new = res3.x[2]
            extra = f"  (+k2: RMSE={rmse_fk1k2:.2f}px k2={k2_new:.4g})"

        lens = LENS_GROUPS.get(key, "?")
        print(f"{key:6} {lens:10} {f_exif:8.0f} {f_old:8.0f} {f_old/f_exif:9.2f}x | "
              f"{rmse_base:9.2f}px {rmse_f_only:11.2f}px {rmse_fk1:9.2f}px "
              f"{f_new/f_exif:9.2f}x {k1_new:10.4g}{extra}")

        out[key] = {
            "lens": lens, "f_exif": f_exif, "f_old": f_old,
            "rmse_base": rmse_base, "rmse_f_only": rmse_f_only,
            "f_refit_only": float(res_f.x[0]),
            "rmse_f_k1": rmse_fk1, "f_with_k1": float(f_new), "k1": float(k1_new),
            "k2": (float(k2_new) if k2_new is not None else None),
        }

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
