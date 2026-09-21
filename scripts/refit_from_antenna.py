#!/usr/bin/env python
"""
run from img folder (2026_imgs/)

Rescue an image whose independently-fit pose landed in a grossly wrong
basin, by re-seeding it from the *known* antenna position and refitting.

Motivating case (2026-09-11): image 2210's horizon-only fit converged to a
camera 198m horizontally from the antenna and 91m ABOVE it, which implies
the antenna should appear 25 degrees BELOW the horizon -- but 2210's photo
plainly shows it ~77 degrees UP, near the top of frame. A 102-degree
contradiction that no focal-length or refinement tweak can fix, because the
optimizer is in the wrong basin entirely, not merely imprecise. (The three
trusted cameras agree with their own antenna rays to 0.1-0.2 degrees.)

Method: treat the antenna's 3D position (solved by the trusted cameras) as
a KNOWN point, and do a single-image resection seeded from the geometry
that the antenna's observed elevation angle implies -- a camera on the
ground at radius r from the antenna's footprint, where r follows from the
observed elevation. Multi-start over a ring of candidate footprints and
azimuths, cheap-score each seed, then fully optimize only the most
promising ones.

Residuals (all pixels): the same horizon-curve residual every other fitter
here uses, plus the antenna's reprojection error against its known 3D
position, weighted by --antenna-weight so one point can hold its own
against a few hundred horizon samples.

Usage:
  python refit_from_antenna.py --key 2210
  python refit_from_antenna.py --key 2210 --antenna 1668.467 2037.109 1769.035
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

DEFAULTS_PATH = "/Users/komalkaur/Desktop/eigsep_stuff/eigsep_terrain/eigsep_terrain/defaults.json"
ANTENNA_DEFAULT = (1668.467, 2037.109, 1769.035)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--antenna", nargs=3, type=float, default=list(ANTENNA_DEFAULT),
                    help="Known antenna (E N U), from the trusted cameras.")
    ap.add_argument("--defaults", default=DEFAULTS_PATH)
    ap.add_argument("--cache-file", default="marjum_dem.npz")
    ap.add_argument("--cam-height", type=float, default=1.6,
                    help="Assumed camera height above ground for seeds [m].")
    ap.add_argument("--radii", nargs="+", type=float, default=[10, 15, 20, 28],
                    help="Candidate horizontal distances from the antenna footprint [m].")
    ap.add_argument("--n-az-seeds", type=int, default=12,
                    help="Candidate azimuths around the footprint.")
    ap.add_argument("--n-best", type=int, default=4,
                    help="How many cheap-scored seeds to fully optimize.")
    ap.add_argument("--antenna-weight", type=float, default=30.0)
    ap.add_argument("--search-n-az", type=int, default=512)
    ap.add_argument("--search-ncols", type=int, default=150)
    ap.add_argument("--final-n-az", type=int, default=2048)
    ap.add_argument("--final-ncols", type=int, default=400)
    ap.add_argument("--bisect-iters", type=int, default=14)
    ap.add_argument("--diff-step", type=float, default=5e-3)
    ap.add_argument("--fix-f", action="store_true", default=True,
                    help="Hold f at its EXIF value (default on).")
    ap.add_argument("--free-f", dest="fix_f", action="store_false")
    ap.add_argument("--out", default=None)
    return ap


def optical_axis_angles(cam, target, img, ant_px_top, f):
    """Seed (th, ph) so the optical axis points such that `target` lands at
    the observed antenna pixel. img.py's model gives optical-axis world
    direction [cos(ph)sin(th), sin(ph)sin(th), cos(th)], i.e.
    elevation = 90deg - th and azimuth(E from N) = 90deg - ph."""
    d = np.asarray(target) - np.asarray(cam)
    az_target = np.arctan2(d[0], d[1])                    # E from N
    el_target = np.arcsin(d[2] / np.linalg.norm(d))

    # where the antenna sits in the frame relative to centre, in angle
    col_x, row_top = ant_px_top
    row_flip = img.npix_y - 1 - row_top
    dy = row_flip - img.npix_y // 2      # +ve = above centre
    dx = col_x - img.npix_x // 2         # +ve = right of centre
    el_off = np.arctan2(dy, f)
    az_off = np.arctan2(dx, f)

    el_axis = el_target - el_off
    az_axis = az_target - az_off
    th = np.pi / 2 - el_axis
    ph = np.pi / 2 - az_axis
    return float(th), float(ph)


def main(argv=None):
    args = build_argparser().parse_args(argv)
    key = args.key
    ANT = np.array(args.antenna, dtype=float)
    dem = DEM(cache_file=args.cache_file)
    defaults = json.load(open(args.defaults))["images"]
    ant_px_top = defaults[key]["ant_px"]

    f_exif = focal_px_for_key(key)
    if f_exif is None:
        raise SystemExit(f"No EXIF focal length for {key}")
    print(f"[{key}] EXIF f = {f_exif:.1f}px   known antenna = {ANT}")

    class LoadArgs:
        img_glob = fhe.IMG_GLOB
        ncols = args.search_ncols

    which = fhe.IMG_KEYS.index(key)
    _, img, seed_default, cols_s, y_s = fhe.load_image(which, LoadArgs(), dem)

    class LoadArgsFinal:
        img_glob = fhe.IMG_GLOB
        ncols = args.final_ncols

    _, _, _, cols_f, y_f = fhe.load_image(which, LoadArgsFinal(), dem)

    def make_resid(cache, cols, y_actual):
        def resid(p):
            img.set_prms(tuple(p))
            hangles = cache.get(p[0], p[1], p[2])
            y_pred = fhe.predicted_horizon_rows(img, hangles, cols,
                                                n_iter=args.bisect_iters)
            r_h = y_pred - y_actual
            cam, rm, f = np.array(p[:3]), ta.build_rm(p[3], p[4], p[5]), p[6]
            col_pred, row_pred_flip, depth = ta.project_point(
                ANT, cam, rm, f, img.npix_y, img.npix_x)
            row_pred_top = img.npix_y - 1 - row_pred_flip
            r_a = args.antenna_weight * np.array(
                [col_pred - ant_px_top[0], row_pred_top - ant_px_top[1]])
            if depth <= 0:      # antenna behind camera -- strongly penalised
                r_a = r_a + 1e4
            return np.concatenate([r_h, r_a])
        return resid

    # ---- build candidate seeds on a ring around the antenna footprint ----
    seeds = []
    for r in args.radii:
        for adeg in np.linspace(0, 360, args.n_az_seeds, endpoint=False):
            a = np.radians(adeg)
            e = ANT[0] + r * np.sin(a)
            n = ANT[1] + r * np.cos(a)
            u = float(dem.interp_alt(e, n)) + args.cam_height
            cam = np.array([e, n, u])
            th, ph = optical_axis_angles(cam, ANT, img, ant_px_top, f_exif)
            seeds.append([e, n, u, th, ph, 0.0, f_exif])

    cache_s = fhe.HorizonCache(dem, args.search_n_az)
    resid_s = make_resid(cache_s, cols_s, y_s)
    scored = []
    for s in seeds:
        try:
            c = float(np.sum(resid_s(s) ** 2))
        except Exception:
            c = np.inf
        scored.append((c, s))
    scored.sort(key=lambda t: t[0])
    print(f"\nScored {len(scored)} seeds; optimizing best {args.n_best}:")
    for c, s in scored[:args.n_best]:
        print(f"   seed cost={c:.4g} at (E,N,U)=({s[0]:.1f},{s[1]:.1f},{s[2]:.1f})")

    NAMES = fhe.NAMES
    results = []
    for c0, s in scored[:args.n_best]:
        ground = float(dem.interp_alt(s[0], s[1]))
        lo = [s[0] - 120, s[1] - 120, ground + 1e-2, -np.pi, -2 * np.pi, -np.pi,
              f_exif if args.fix_f else 1.0]
        hi = [s[0] + 120, s[1] + 120, ground + 200, 2 * np.pi, 2 * np.pi, 2 * np.pi,
              f_exif if args.fix_f else 1e5]
        if args.fix_f:
            hi[6] = f_exif + 1e-6      # scipy needs lo < hi
        x0 = np.clip(np.array(s, dtype=float), lo, hi)
        try:
            res = least_squares(resid_s, x0=x0, bounds=(lo, hi), method="trf",
                                diff_step=args.diff_step, x_scale="jac",
                                loss="soft_l1", f_scale=5.0, max_nfev=300)
        except Exception as ex:
            print(f"   (seed failed: {ex})")
            continue
        results.append((float(np.sum(res.fun ** 2)), list(res.x)))
        print(f"   -> cost {c0:.4g} -> {np.sum(res.fun**2):.4g}")

    if not results:
        raise SystemExit("No seed converged.")
    results.sort(key=lambda t: t[0])
    best = results[0][1]

    # ---- final polish at full resolution ----
    cache_f = fhe.HorizonCache(dem, args.final_n_az)
    resid_f = make_resid(cache_f, cols_f, y_f)
    ground = float(dem.interp_alt(best[0], best[1]))
    lo = [best[0] - 60, best[1] - 60, ground + 1e-2, -np.pi, -2 * np.pi, -np.pi,
          f_exif if args.fix_f else 1.0]
    hi = [best[0] + 60, best[1] + 60, ground + 200, 2 * np.pi, 2 * np.pi, 2 * np.pi,
          (f_exif + 1e-6) if args.fix_f else 1e5]
    res = least_squares(resid_f, x0=np.clip(best, lo, hi), bounds=(lo, hi),
                        method="trf", diff_step=args.diff_step, x_scale="jac",
                        loss="soft_l1", f_scale=5.0)
    solved = list(res.x)

    n_h = len(cols_f)
    rf = resid_f(solved)
    horizon_rmse = float(np.sqrt(np.mean(rf[:n_h] ** 2)))
    ant_err = float(np.linalg.norm(rf[n_h:] / args.antenna_weight))

    cam = np.array(solved[:3])
    rm = ta.build_rm(solved[3], solved[4], solved[5])
    ray = ta.pixel_to_ray(img.npix_y - 1 - ant_px_top[1], ant_px_top[0],
                          img.npix_y, img.npix_x, solved[6], rm)
    el_ray = np.degrees(np.arcsin(np.clip(ray[2], -1, 1)))
    d = ANT - cam
    el_req = np.degrees(np.arcsin(d[2] / np.linalg.norm(d)))

    print(f"\n[{key}] solved prms_u = {[round(float(v),4) for v in solved]}")
    print(f"   horizon RMSE = {horizon_rmse:.2f}px")
    print(f"   antenna reprojection error = {ant_err:.1f}px")
    print(f"   antenna ray elevation {el_ray:+.1f}deg vs required {el_req:+.1f}deg "
          f"(mismatch {abs(el_ray-el_req):.2f}deg)")
    print(f"   camera is {np.hypot(d[0],d[1]):.1f}m horizontally from the antenna, "
          f"{-d[2]:+.1f}m in elevation relative to it")

    out_path = args.out or f"refit_{key}_from_antenna.json"
    json.dump({key: {"prms_u": [round(float(v), 4) for v in solved],
                     "horizon_rmse_px": round(horizon_rmse, 3),
                     "antenna_reproj_px": round(ant_err, 2),
                     "antenna_elev_mismatch_deg": round(float(abs(el_ray - el_req)), 3)}},
              open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
