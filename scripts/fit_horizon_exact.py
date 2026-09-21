#!/usr/bin/env python
"""
run from img folder

Fits camera parameters against the DEM's *exact* analytic horizon
(DEM.calc_horizon) instead of the noisy per-ray marching used by every other
fitting script in this repo, and never models an antenna term at all.

Why:
  - `img.horizon_ray_logL` / `img.ray_distance` decide sky-vs-ground per ray
    by marching a fixed-step ray through the DEM -- quantized (fine_delta),
    randomly subsampled per call, and flat once a ray is already on the
    correct side of the mask. `fit_horizon_curve.py` works around the flat
    part locally (bisection -> continuous per-column residual) but still
    pays for marching's quantization noise on every evaluation.
  - `DEM.calc_horizon` instead computes the *true* horizon elevation angle
    at every azimuth around a camera position exactly, via a deterministic
    coarse-to-fine max-pool search over the whole DEM -- no ray marching, no
    per-call resampling noise. It depends only on (e, n, u), not on the
    camera's orientation/focal length, so it's computed once per residual
    evaluation and reused across every sampled column.
  - `map_estimate.py` / `eigsep_terrain_pymc.py` couple every image to a
    *fake* antenna anchor by default (this dataset has no real tracked
    antenna position), which biases their joint fits. This script fits each
    image's pose from its own horizon curve alone -- there is no antenna
    term anywhere in this file to disable.

Method: for each image, extract the actual ground/sky transition row per
sampled column from the segmentation mask (as in fit_horizon_curve.py), find
the *predicted* transition row by bisecting each column against
`calc_horizon`'s exact elevation-vs-azimuth curve (no ray marching at all),
and minimize (predicted_row - actual_row) in pixels via a two-stage
`scipy.optimize.least_squares` (plain LSQ, then a `soft_l1` robust refine).

Usage:
  python fit_horizon_exact.py --which 0            # interactive, one image
  python fit_horizon_exact.py --all --out fit.json  # headless, every image
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.optimize import least_squares

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, dtype_r
from eigsep_terrain.img_defaults import load_defaults

IMG_GLOB, CACHE_FILE, DEFAULT_META, DEFAULT_PRMS_U_BY_KEY, IMG_KEYS = load_defaults(
    '/Users/komalkaur/Desktop/eigsep_stuff/eigsep_terrain/eigsep_terrain/defaults.json'
)

NAMES = ["e", "n", "u", "th", "ph", "ti", "f"]


def extract_actual_horizon(mask, cols):
    """For each column, the row of the first sky pixel scanning from the
    bottom of the frame (row 0 = bottom, since img.py flips the image).
    Columns that are entirely sky or entirely ground (no visible
    transition) are dropped -- they carry no horizon information."""
    y = np.full(len(cols), np.nan)
    for i, x in enumerate(cols):
        col = mask[:, x]
        if not col.any() or col.all():
            continue
        y[i] = np.argmax(col)
    valid = ~np.isnan(y)
    return cols[valid], y[valid]


def elevation_azimuth(rays):
    """rays: (3, N) unit vectors in (E, N, U) world components (the same
    frame img.get_rays / pick_points.project use). Returns (elevation,
    azimuth) in radians, azimuth wrapped to [0, 2*pi) matching
    eigsep_terrain.utils.az_bin's convention (0 = north, +E is positive)."""
    el = np.arctan2(rays[2], np.hypot(rays[0], rays[1]))
    az = np.arctan2(rays[0], rays[1])
    az = np.where(az < 0, az + 2 * np.pi, az)
    return el, az


def interp_hangles(hangles, az):
    """Linear interpolation (with azimuthal wraparound) of calc_horizon's
    per-bin elevation-angle curve at continuous azimuth `az`. Keeps the
    predicted horizon a smooth function of azimuth instead of a staircase,
    so small parameter changes move the residual smoothly across bin
    boundaries."""
    n_az = hangles.shape[0]
    bin_w = 2 * np.pi / n_az
    pos = az / bin_w
    i0 = np.floor(pos).astype(np.int64) % n_az
    i1 = (i0 + 1) % n_az
    frac = pos - np.floor(pos)
    return hangles[i0] * (1 - frac) + hangles[i1] * frac


class HorizonCache:
    """calc_horizon depends only on (e, n, u) -- not on orientation/focal
    length -- so within one solve, the ~4 of 7 finite-difference
    evaluations that only perturb (th, ph, ti, f) can reuse the exact same
    horizon curve instead of paying for another calc_horizon call."""
    def __init__(self, dem, n_az):
        self.dem = dem
        self.n_az = n_az
        self._cache = {}
        self.n_calls = 0
        self.n_hits = 0

    def get(self, e, n, u):
        key = (float(e), float(n), float(u))
        self.n_calls += 1
        hangles = self._cache.get(key)
        if hangles is None:
            hangles, _ = self.dem.calc_horizon(e, n, u, n_az=self.n_az)
            self._cache[key] = hangles
        else:
            self.n_hits += 1
        return hangles


def predicted_horizon_rows(img, hangles, cols, n_iter=14):
    """Bisect each column between row 0 (assumed ground) and row H-1
    (assumed sky) to find where the ray's elevation angle crosses the
    DEM's exact horizon elevation at that ray's azimuth. Vectorized across
    columns; cheap even for hundreds of columns since it's just
    interpolation lookups into an already-computed `hangles` curve."""
    H = img.npix_y
    lo = np.zeros(len(cols), dtype=np.float64)
    hi = np.full(len(cols), H - 1, dtype=np.float64)
    cols_f = cols.astype(dtype_r)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        rays = img.get_rays(pixels=(mid.astype(dtype_r), cols_f))
        el, az = elevation_azimuth(rays)
        hz = interp_hangles(hangles, az)
        model_sky = el > hz
        hi = np.where(model_sky, mid, hi)
        lo = np.where(model_sky, lo, mid)
    return (lo + hi) / 2


def solve_camera(img, dem, cache, cols, y_actual, init, robust=True,
                  fix=None, prior_sigma=None, n_iter=14, diff_step=1e-2):
    fix = set(fix or [])
    prior_sigma = prior_sigma or {}
    ground = float(dem.interp_alt(init[0], init[1]))
    lo_all = [-np.inf, -np.inf, ground + 1e-2, -np.pi, -2 * np.pi, -np.pi, 1.0]
    hi_all = [np.inf, np.inf, np.inf, 2 * np.pi, 2 * np.pi, 2 * np.pi, 1e5]

    free_idx = [i for i, name in enumerate(NAMES) if name not in fix]
    if not free_idx:
        raise ValueError("Cannot fix every parameter -- nothing left to solve for.")

    init = list(init)
    init[2] = max(init[2], lo_all[2])  # seed u may sit exactly on DEM ground
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
        img.set_prms(tuple(p))
        hangles = cache.get(p[0], p[1], p[2])
        y_pred = predicted_horizon_rows(img, hangles, cols, n_iter=n_iter)
        r = y_pred - y_actual
        extra = [
            (p[i] - init[i]) / prior_sigma[NAMES[i]]
            for i in free_idx if NAMES[i] in prior_sigma
        ]
        if extra:
            r = np.concatenate([r, extra])
        return r

    def run(x0, use_robust):
        kwargs = dict(loss="soft_l1", f_scale=5.0) if use_robust else {}
        return least_squares(resid, x0=np.array(x0, dtype=np.double),
                             bounds=(lo, hi), method="trf",
                             diff_step=diff_step, **kwargs)

    # Stage 1: plain least-squares. A robust loss suppresses the gradient
    # almost entirely when residuals start out large (bad initial guess),
    # so fitting robustly from a bad start can silently fail to move at all.
    res = run(x0, use_robust=False)
    stage1_status = (res.cost, res.status)
    if robust:
        # Stage 2: robust refine from the now-close starting point, to
        # reduce the influence of columns that are still bad (rugged
        # terrain, mis-segmented mask, etc).
        res = run(res.x, use_robust=True)
    solved = np.array(full_params(res.x))
    res.fun = res.fun[:len(cols)]  # trim prior residuals before reporting RMSE
    return solved, res, stage1_status


def find_img_file(which, img_glob):
    key = IMG_KEYS[which]
    files = sorted(glob.glob(img_glob))
    matches = [f for f in files if os.path.basename(f).split("_")[-1].split(".")[0] == key]
    if not matches:
        raise FileNotFoundError(f"No file matching key {key!r} via {img_glob!r}")
    return matches[0]


def load_image(which, args, dem):
    key = IMG_KEYS[which]
    meta = {k: dict(v) for k, v in DEFAULT_META.items()}
    img_file = find_img_file(which, args.img_glob)
    img = HorizonImage(img_file, meta, px_smooth=150, px_dist=30)
    init = DEFAULT_PRMS_U_BY_KEY.get(key)
    if init is None:
        raise ValueError(f"No default prms for key {key!r}; add one to defaults.json.")
    cols = np.unique(np.linspace(0, img.npix_x - 1, args.ncols).astype(int))
    cols, y_actual = extract_actual_horizon(img.sky_mask, cols)
    return key, img, list(init), cols, y_actual


def build_argparser():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--which", type=int, choices=list(range(len(IMG_KEYS))),
                       help="Interactively fit a single image (opens a GUI).")
    group.add_argument("--all", action="store_true",
                       help="Headlessly fit every image in defaults.json and "
                            "write results to --out.")
    group.add_argument("--keys", nargs="+", default=None, choices=IMG_KEYS,
                       help="Headlessly fit only these image keys (e.g. a "
                            "polish pass on a few candidates) and write "
                            "results to --out.")
    ap.add_argument("--out", default="fit_horizon_exact_results.json",
                    help="Where to write results in --all mode.")
    ap.add_argument("--img-glob", default=IMG_GLOB)
    ap.add_argument("--cache-file", default=CACHE_FILE)
    ap.add_argument("--n-az", type=int, default=1024,
                    help="Azimuth bins for DEM.calc_horizon. Higher = finer "
                         "horizon resolution but slower (~0.4s @256, "
                         "~0.8s @1024, ~1.4s @4096 per call on this DEM).")
    ap.add_argument("--ncols", type=int, default=300,
                    help="Number of columns to sample across the image width.")
    ap.add_argument("--bisect-iters", type=int, default=14)
    ap.add_argument("--diff-step", type=float, default=1e-2,
                    help="Relative finite-difference step for the optimizer's "
                         "numerical Jacobian. calc_horizon's own max-over-"
                         "azimuth-bin selection is a discrete argmax, so it "
                         "has a small quantization floor too (~0.1 deg "
                         "hangle jitter even at ~mm position steps) -- this "
                         "must be big enough to move past that floor.")
    ap.add_argument("--no-robust", action="store_true",
                    help="Disable soft_l1 robust loss (use plain least-squares).")
    ap.add_argument("--fix", nargs="+", default=[], choices=NAMES,
                    help="Hold these params at their defaults.json value "
                         "(e.g. --fix e n u if camera position is known).")
    ap.add_argument("--prior-sigma", nargs="+", default=[],
                    help="Soft Gaussian priors as name:sigma pairs, e.g. "
                         "--prior-sigma f:500 ti:0.2, pulling weakly-"
                         "constrained free params toward their default.")
    return ap


def parse_prior_sigma(tokens):
    prior_sigma = {}
    for tok in tokens:
        name, val = tok.split(":")
        prior_sigma[name] = float(val)
    return prior_sigma


def fit_one(which, args, dem, prior_sigma, verbose=True):
    key, img, init, cols, y_actual = load_image(which, args, dem)
    cache = HorizonCache(dem, args.n_az)
    img.set_prms(tuple(init))
    y_pred_before = predicted_horizon_rows(
        img, cache.get(*init[:3]), cols, n_iter=args.bisect_iters
    )
    rmse_before = np.sqrt(np.mean((y_pred_before - y_actual) ** 2))
    solved, res, stage1 = solve_camera(
        img, dem, cache, cols, y_actual, init,
        robust=not args.no_robust, fix=args.fix, prior_sigma=prior_sigma,
        n_iter=args.bisect_iters, diff_step=args.diff_step,
    )
    rmse = np.sqrt(np.mean(res.fun ** 2))
    if verbose:
        print(f"[{key}] stage1 cost={stage1[0]:.3g} status={stage1[1]} | "
              f"final status={res.status} | calc_horizon calls={cache.n_calls} "
              f"(cache hits={cache.n_hits}) | RMSE {rmse_before:.2f}px -> {rmse:.2f}px")
    return key, solved, rmse_before, rmse


def main(argv=None):
    args = build_argparser().parse_args(argv)
    prior_sigma = parse_prior_sigma(args.prior_sigma)
    if args.fix:
        print(f"Holding fixed: {args.fix} (from defaults.json)")
    if prior_sigma:
        print(f"Gaussian priors: {prior_sigma}")

    dem = DEM(cache_file=args.cache_file)

    if args.all or args.keys:
        whichs = (range(len(IMG_KEYS)) if args.all
                 else [IMG_KEYS.index(k) for k in args.keys])
        results = {}
        for which in whichs:
            key, solved, rmse_before, rmse = fit_one(which, args, dem, prior_sigma)
            results[key] = {
                "prms_u": [round(float(v), 4) for v in solved],
                "rmse_before_px": round(float(rmse_before), 3),
                "rmse_after_px": round(float(rmse), 3),
            }
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {len(results)} fits to {args.out}")
        return 0

    # --which: interactive single-image mode
    key, img, init, cols, y_actual = load_image(args.which, args, dem)
    cache = HorizonCache(dem, args.n_az)

    fig = plt.figure(figsize=(10, 8))
    ax_img = fig.add_axes([0.08, 0.15, 0.87, 0.78])
    ax_img.imshow(img.img, origin="lower")
    ax_img.plot(cols, y_actual, ".", color="cyan", markersize=3, label="actual horizon")
    (pred_line,) = ax_img.plot([], [], ".", color="red", markersize=3,
                               label="predicted horizon (calc_horizon)")
    ax_img.legend(loc="upper right")
    ax_img.set_title(f"Image {key} — click Fit", fontsize=11)

    status_ax = fig.add_axes([0.08, 0.02, 0.6, 0.05])
    status_ax.axis("off")
    status_txt = status_ax.text(0, 0.5, "Ready.", va="center", fontsize=9)

    state = {"init": init}

    def on_fit(_):
        status_txt.set_text("Fitting...")
        fig.canvas.draw()
        fig.canvas.flush_events()
        img.set_prms(tuple(state["init"]))
        y_pred_before = predicted_horizon_rows(
            img, cache.get(*state["init"][:3]), cols, n_iter=args.bisect_iters
        )
        rmse_before = np.sqrt(np.mean((y_pred_before - y_actual) ** 2))
        solved, res, stage1 = solve_camera(
            img, dem, cache, cols, y_actual, state["init"],
            robust=not args.no_robust, fix=args.fix, prior_sigma=prior_sigma,
            n_iter=args.bisect_iters, diff_step=args.diff_step,
        )
        rmse = np.sqrt(np.mean(res.fun ** 2))
        img.set_prms(tuple(solved))
        y_pred = predicted_horizon_rows(img, cache.get(*solved[:3]), cols,
                                        n_iter=args.bisect_iters)
        pred_line.set_data(cols, y_pred)
        state["init"] = list(solved)  # allow iterative re-fitting from here
        prms_u = tuple(round(float(v), 4) for v in solved)
        print(f"stage1 cost={stage1[0]:.3g} status={stage1[1]} | "
              f"calc_horizon calls={cache.n_calls} (cache hits={cache.n_hits})")
        print(f"RMSE before={rmse_before:.2f}px -> after={rmse:.2f}px")
        print("prms_u =", prms_u)
        status_txt.set_text(f"RMSE {rmse_before:.1f}px -> {rmse:.1f}px "
                            f"(see console for prms_u). Click Fit to refine further.")
        fig.canvas.draw_idle()

    fit_ax = fig.add_axes([0.78, 0.02, 0.17, 0.05])
    fit_btn = Button(fit_ax, "Fit")
    fit_btn.on_clicked(on_fit)

    img.set_prms(tuple(state["init"]))
    y_pred0 = predicted_horizon_rows(img, cache.get(*state["init"][:3]), cols,
                                     n_iter=args.bisect_iters)
    pred_line.set_data(cols, y_pred0)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
