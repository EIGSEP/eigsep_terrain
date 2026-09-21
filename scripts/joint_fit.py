#!/usr/bin/env python
"""
run from img folder (2026_imgs/)

Joint multi-image bundle adjustment: solves for several cameras' full poses
(e, n, u, th, ph, ti, f) *simultaneously*, tied together by cross-image
correspondences (tiepoints.json, from pick_tiepoints.py) instead of fitting
each image's horizon curve independently the way fit_horizon_exact.py does.

Why: fit_horizon_exact.py's independent per-image fits can trade camera
*position* against orientation/focal length while still matching the
horizon curve well -- confirmed directly in this project (image 2210: 36px
horizon RMSE, visually near-perfect, yet its antenna sightline misses
2213/2234's by 100+ meters; see docs/fitting_improvement_plan.md and the
triangulate_antenna.py results). A horizon silhouette alone constrains
orientation tightly but position only weakly. Cross-image tie points fix
this the way stereo/multi-view geometry always does: the *same* physical
point, seen from two-plus cameras, pins down both cameras' positions
relative to each other, not just their individual horizon match.

Residuals combined in one least_squares call (all in pixel units, so they
combine on a comparable scale without needing hand-tuned relative weights,
though --horizon-weight/--tiepoint-weight are exposed if you disagree):
  1. Per-image horizon-curve residual (predicted_row - actual_row per
     sampled column) -- exactly fit_horizon_exact.py's cost, reused
     directly via that module's HorizonCache/predicted_horizon_rows.
  2. Per-tiepoint-per-image reprojection residual (predicted_pixel -
     observed_pixel) -- each tie point's 3D position (E, N, U) is a free
     unknown too, solved jointly with the camera poses (classic bundle
     adjustment), initialized by triangulating from the seed camera poses
     via triangulate_antenna.triangulate.

The antenna position itself is auto-injected as a tie point (id "antenna")
using each image's current ant_px from defaults.json, unless
--no-auto-antenna -- you already have that correspondence for every image,
no reason to waste it.

Usage:
  python joint_fit.py --keys 2213 2234 2216 2210 --out joint_fit_results.json
  python joint_fit.py --keys 2213 2234 2216 2210 --tiepoints tiepoints.json \
      --fix f   # hold focal length at its seed value per image
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

NAMES = ["e", "n", "u", "th", "ph", "ti", "f"]
DEFAULTS_PATH = "/Users/komalkaur/Desktop/eigsep_stuff/eigsep_terrain/eigsep_terrain/defaults.json"


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def cam_from_prms(prms_u):
    e, n, u, th, ph, ti, f = prms_u
    cam = np.array([e, n, u], dtype=np.double)
    rm = ta.build_rm(th, ph, ti)
    return cam, rm, f


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="+", required=True,
                    help="Image keys to jointly fit.")
    ap.add_argument("--defaults", default=DEFAULTS_PATH)
    ap.add_argument("--seed", default="fit_horizon_exact_polished.json",
                    help="JSON with per-key prms_u to seed camera poses "
                         "from; falls back to defaults.json's prms_u for "
                         "any key not present in this file.")
    ap.add_argument("--tiepoints", default="tiepoints.json")
    ap.add_argument("--no-auto-antenna", action="store_true",
                    help="Don't auto-inject each image's ant_px as a tie point.")
    ap.add_argument("--cache-file", default="marjum_dem.npz")
    ap.add_argument("--n-az", type=int, default=1024)
    ap.add_argument("--ncols", type=int, default=300)
    ap.add_argument("--bisect-iters", type=int, default=14)
    ap.add_argument("--diff-step", type=float, default=1e-2)
    ap.add_argument("--fix", nargs="+", default=[], choices=NAMES,
                    help="Hold these params fixed at their seed value for "
                         "every image (e.g. --fix f if focal length is "
                         "EXIF-known and trusted).")
    ap.add_argument("--exif-focal", action="store_true",
                    help="Seed each image's focal length from its EXIF "
                         "FocalLengthIn35mmFilm instead of the seed file's "
                         "fitted value. `f` trades off against position "
                         "along the optical axis, so an unconstrained fit "
                         "can match the horizon while landing on a badly "
                         "wrong focal length (image 2210 fitted to 0.52x "
                         "its EXIF value). Combine with --fix f to hold it "
                         "there, or --f-prior-sigma to allow limited drift.")
    ap.add_argument("--f-prior-sigma", type=float, default=None,
                    help="Soft Gaussian prior (in pixels) pulling each "
                         "image's f toward its seed value -- use with "
                         "--exif-focal to let f move a little off EXIF "
                         "without letting it run away.")
    ap.add_argument("--horizon-weight", type=float, default=1.0)
    ap.add_argument("--tiepoint-weight", type=float, default=1.0)
    ap.add_argument("--free-points", nargs="+", default=["antenna"],
                    help="Tie-point IDs whose 3D position is a fully free "
                         "(E,N,U) unknown -- for points that aren't on the "
                         "terrain surface (e.g. the suspended antenna). "
                         "Every other point is constrained to lie exactly "
                         "on the DEM surface (U = dem.interp_alt(E,N)), "
                         "using the terrain model itself -- not just "
                         "cross-image reprojection -- to pin down the "
                         "point, since a real rock corner's height isn't a "
                         "free unknown once its (E,N) is known.")
    ap.add_argument("--no-robust", action="store_true")
    ap.add_argument("--out", default="joint_fit_results.json")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    dem = DEM(cache_file=args.cache_file)
    defaults = load_json(args.defaults)
    seed_json = load_json(args.seed, {}) or {}
    keys = args.keys

    class LoadArgs:
        img_glob = fhe.IMG_GLOB
        ncols = args.ncols

    imgs, cols_y, inits = {}, {}, {}
    for key in keys:
        which = fhe.IMG_KEYS.index(key)
        _, img, init_default, cols, y_actual = fhe.load_image(which, LoadArgs(), dem)
        seed = list(seed_json.get(key, {}).get("prms_u", init_default))
        src = 'polished/seed file' if key in seed_json else 'defaults.json'
        if args.exif_focal:
            f_exif = focal_px_for_key(key)
            if f_exif is None:
                print(f"[{key}] WARNING: --exif-focal requested but no EXIF "
                      f"FocalLengthIn35mmFilm found; keeping seed f={seed[6]:.1f}")
            else:
                print(f"[{key}] f: {seed[6]:.1f} -> {f_exif:.1f} (EXIF, "
                      f"seed was {seed[6]/f_exif:.2f}x)")
                seed[6] = f_exif
                src += ' + EXIF f'
        imgs[key] = img
        cols_y[key] = (cols, y_actual)
        inits[key] = seed
        print(f"[{key}] seeded prms_u = {[round(v,3) for v in seed]}  ({src})")

    tiepoints = dict(load_json(args.tiepoints, {}) or {})
    if not args.no_auto_antenna:
        ant = {k: list(defaults["images"][k]["ant_px"]) for k in keys if k in defaults["images"]}
        if len(ant) >= 2:
            tiepoints["antenna"] = ant

    pts = {}
    for pid, obs in tiepoints.items():
        obs2 = {k: v for k, v in obs.items() if k in keys}
        if len(obs2) >= 2:
            pts[pid] = obs2
    point_ids = sorted(pts.keys())
    print(f"\nUsing {len(point_ids)} tie point(s) with >=2 observations: {point_ids}")
    if not point_ids:
        print("WARNING: no cross-image tie points -- this reduces to independent "
              "per-image horizon fits (same as fit_horizon_exact.py). Pick some "
              "points with pick_tiepoints.py first.")

    free_point_ids = set(args.free_points)
    is_terrain = {pid: pid not in free_point_ids for pid in point_ids}
    n_terrain = sum(is_terrain.values())
    print(f"  {n_terrain} constrained to the DEM surface (U = dem.interp_alt(E,N)), "
          f"{len(point_ids) - n_terrain} free 3D: "
          f"{[pid for pid in point_ids if not is_terrain[pid]]}")

    X0 = {}
    for pid in point_ids:
        obs = pts[pid]
        cams, rays = [], []
        for k, (x_obs, y_top) in obs.items():
            img = imgs[k]
            cam, rm, f = cam_from_prms(inits[k])
            row_flip = img.npix_y - 1 - y_top
            rays.append(ta.pixel_to_ray(row_flip, x_obs, img.npix_y, img.npix_x, f, rm))
            cams.append(cam)
        X = ta.triangulate(cams, rays)
        if is_terrain[pid]:
            X[2] = dem.interp_alt(X[0], X[1])  # snap init height onto the DEM surface
        X0[pid] = X

    fix = set(args.fix)
    free_idx = [i for i, nm in enumerate(NAMES) if nm not in fix]

    x0, lo, hi, slices = [], [], [], {}
    for k in keys:
        start = len(x0)
        ground = float(dem.interp_alt(inits[k][0], inits[k][1]))
        lo_all = [-np.inf, -np.inf, ground + 1e-2, -np.pi, -2 * np.pi, -np.pi, 1.0]
        hi_all = [np.inf, np.inf, np.inf, 2 * np.pi, 2 * np.pi, 2 * np.pi, 1e5]
        for i in free_idx:
            x0.append(inits[k][i])
            lo.append(lo_all[i])
            hi.append(hi_all[i])
        slices[k] = start

    e_edges, n_edges = dem.get_en()
    e_lo, e_hi = float(e_edges.min()) + 1.0, float(e_edges.max()) - 1.0
    n_lo, n_hi = float(n_edges.min()) + 1.0, float(n_edges.max()) - 1.0

    pt_slices = {}
    for pid in point_ids:
        pt_slices[pid] = len(x0)
        if is_terrain[pid]:
            # (E, N) only -- U comes from the DEM. Bounded to the DEM's own
            # extent: dem.interp_alt has no bounds-checking of its own, and
            # an unbounded (E,N) can wander off the grid mid-solve.
            x0.extend(X0[pid][:2].tolist())
            lo.extend([e_lo, n_lo])
            hi.extend([e_hi, n_hi])
        else:
            x0.extend(X0[pid].tolist())       # free (E, N, U)
            lo.extend([-np.inf] * 3)
            hi.extend([np.inf] * 3)

    def unpack(x):
        prms = {}
        for k in keys:
            start = slices[k]
            p = list(inits[k])
            for j, i in enumerate(free_idx):
                p[i] = x[start + j]
            prms[k] = p
        Xs = {}
        for pid in point_ids:
            start = pt_slices[pid]
            if is_terrain[pid]:
                e, n = x[start], x[start + 1]
                Xs[pid] = np.array([e, n, dem.interp_alt(e, n)])
            else:
                Xs[pid] = np.asarray(x[start:start + 3])
        return prms, Xs

    cache = fhe.HorizonCache(dem, args.n_az)
    n_horizon_res = sum(len(cols_y[k][0]) for k in keys)

    def residuals(x):
        prms, Xs = unpack(x)
        res = []
        for k in keys:
            img = imgs[k]
            img.set_prms(tuple(prms[k]))
            hangles = cache.get(*prms[k][:3])
            cols, y_actual = cols_y[k]
            y_pred = fhe.predicted_horizon_rows(img, hangles, cols, n_iter=args.bisect_iters)
            res.append(args.horizon_weight * (y_pred - y_actual))
        for pid in point_ids:
            Xp = Xs[pid]
            for k, (x_obs, y_obs_top) in pts[pid].items():
                cam, rm, f = cam_from_prms(prms[k])
                img = imgs[k]
                col_pred, row_pred_flip, _depth = ta.project_point(
                    Xp, cam, rm, f, img.npix_y, img.npix_x)
                row_pred_top = img.npix_y - 1 - row_pred_flip
                res.append(args.tiepoint_weight *
                          np.array([col_pred - x_obs, row_pred_top - y_obs_top]))
        if args.f_prior_sigma:
            res.append(np.array([(prms[k][6] - inits[k][6]) / args.f_prior_sigma
                                for k in keys]))
        return np.concatenate(res)

    # tie-point residuals occupy [n_horizon_res : n_horizon_res + n_tie_res];
    # any f-prior residuals come after, and must be excluded from both RMSEs.
    n_tie_res = 2 * sum(len(pts[pid]) for pid in point_ids)

    def report_rmse(r):
        h = np.sqrt(np.mean(r[:n_horizon_res] ** 2))
        t = (np.sqrt(np.mean(r[n_horizon_res:n_horizon_res + n_tie_res] ** 2))
             if n_tie_res else float("nan"))
        return h, t

    x0 = np.array(x0, dtype=np.double)
    h0, t0 = report_rmse(residuals(x0))
    print(f"\nInitial: horizon RMSE={h0:.2f}px  tiepoint RMSE={t0:.2f}px")

    print("Solving (stage 1: plain least-squares)...")
    result = least_squares(residuals, x0=x0, bounds=(lo, hi), method="trf",
                           diff_step=args.diff_step, x_scale="jac")
    if not args.no_robust:
        print("Solving (stage 2: robust refine)...")
        result = least_squares(residuals, x0=result.x, bounds=(lo, hi), method="trf",
                               diff_step=args.diff_step, x_scale="jac",
                               loss="soft_l1", f_scale=5.0)

    horizon_rmse, tie_rmse = report_rmse(residuals(result.x))
    print(f"\nFinal: horizon RMSE={horizon_rmse:.2f}px  tiepoint RMSE={tie_rmse:.2f}px  "
          f"status={result.status}")

    prms_final, Xs_final = unpack(result.x)
    out = {"cameras": {}, "points": {}}
    for k in keys:
        entry = {"prms_u": [round(float(v), 4) for v in prms_final[k]]}
        f_exif = focal_px_for_key(k)
        if f_exif:
            entry["f_exif_px"] = round(float(f_exif), 1)
            entry["f_over_exif"] = round(float(prms_final[k][6] / f_exif), 3)
        out["cameras"][k] = entry
    for pid in point_ids:
        X = Xs_final[pid]
        ground = float(dem.interp_alt(X[0], X[1]))
        out["points"][pid] = {
            "E": round(float(X[0]), 3), "N": round(float(X[1]), 3), "U": round(float(X[2]), 3),
            "height_above_ground_m": round(float(X[2] - ground), 3),
            "terrain_constrained": is_terrain[pid],
            "n_obs": len(pts[pid]),
        }
        tag = "on DEM surface" if is_terrain[pid] else "free 3D"
        print(f"  point {pid} [{tag}]: (E,N,U)=({X[0]:.2f},{X[1]:.2f},{X[2]:.2f})  "
              f"height above ground={X[2]-ground:.2f}m  (from {len(pts[pid])} images)")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
