#!/usr/bin/env python
"""
run from img folder

Fits camera parameters by matching the *entire* horizon curve extracted
from the segmentation mask, instead of a handful of hand-clicked
correspondences.

Why: a few clicks (especially clustered, as your images require) leave the
7-parameter fit underdetermined. logL-on-full-mask (the old MCMC approach)
is also a poor cost -- it's flat once a ray lands on the correct side of
the mask, so very different-looking horizons can tie. This script instead:

  1. For each image column, extracts the actual ground/sky transition row
     from img.sky_mask (the existing filled binary segmentation).
  2. For candidate camera params, finds the *predicted* transition row in
     that same column via bisection with the existing ray tracer.
  3. Minimizes (predicted_row - actual_row) in pixels, across every column
     with a valid transition -- thousands of residuals from one image,
     smooth in camera params, and far better conditioned than clicks.

Usage:
  python fit_horizon_curve.py --which 0
"""
import argparse
import glob
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


def predicted_horizon_rows(img, dem, cols, fine_delta, n_iter=14):
    """Bisect each column between row 0 (assumed ground) and row H-1
    (assumed sky) to find the model's ground/sky transition row. Fully
    vectorized across columns -- cheap even for hundreds of columns."""
    H = img.npix_y
    lo = np.zeros(len(cols), dtype=np.float64)
    hi = np.full(len(cols), H - 1, dtype=np.float64)
    cols_f = cols.astype(dtype_r)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        rays = img.get_rays(pixels=(mid.astype(dtype_r), cols_f))
        r = img.ray_distance(dem, rays, dtype=dtype_r, fine_delta=fine_delta)
        model_sky = np.isnan(r)
        hi = np.where(model_sky, mid, hi)
        lo = np.where(model_sky, lo, mid)
    return (lo + hi) / 2


def solve_camera(img, dem, cols, y_actual, init, fine_delta, robust=True,
                  fix=None, prior_sigma=None, n_iter=14, diff_step=1e-2):
    fix = set(fix or [])
    prior_sigma = prior_sigma or {}
    ground = float(dem.interp_alt(init[0], init[1]))
    lo_all = [-np.inf, -np.inf, ground + 1e-2, -np.pi, -2 * np.pi, -np.pi, 1.0]
    hi_all = [np.inf, np.inf, np.inf, 2 * np.pi, 2 * np.pi, 2 * np.pi, 1e5]

    free_idx = [i for i, name in enumerate(NAMES) if name not in fix]
    if not free_idx:
        raise ValueError("Cannot fix every parameter -- nothing left to solve for.")
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
        y_pred = predicted_horizon_rows(img, dem, cols, fine_delta, n_iter=n_iter)
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
    print(f"  stage 1 (plain LSQ): cost={res.cost:.3g}, status={res.status}")
    if robust:
        # Stage 2: robust refine from the now-close starting point, to
        # reduce the influence of any columns that are still bad (rugged
        # terrain, mis-segmented mask, etc).
        res = run(res.x, use_robust=True)
        print(f"  stage 2 (robust refine): cost={res.cost:.3g}, status={res.status}")
    solved = np.array(full_params(res.x))
    res.fun = res.fun[:len(cols)]  # trim prior residuals before reporting RMSE
    return solved, res


def find_img_file(which, img_glob):
    key = IMG_KEYS[which]
    files = sorted(glob.glob(img_glob))
    matches = [f for f in files if os.path.basename(f).split("_")[-1].split(".")[0] == key]
    if not matches:
        raise FileNotFoundError(f"No file matching key {key!r} via {img_glob!r}")
    return matches[0]


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=int, required=True, choices=list(range(len(IMG_KEYS))))
    ap.add_argument("--img-glob", default=IMG_GLOB)
    ap.add_argument("--cache-file", default=CACHE_FILE)
    ap.add_argument("--fine-delta", type=float, default=0.25)
    ap.add_argument("--ncols", type=int, default=300,
                    help="Number of columns to sample across the image width.")
    ap.add_argument("--bisect-iters", type=int, default=14)
    ap.add_argument("--diff-step", type=float, default=1e-2,
                    help="Relative finite-difference step for the optimizer's "
                         "numerical Jacobian. The ray tracer is quantized "
                         "(fine_delta), so scipy's default tiny step sees a "
                         "flat Jacobian and never moves -- this must be big "
                         "enough to flip at least one ray's hit/miss outcome. "
                         "Raise it if the fit still doesn't move; lower it "
                         "for a more precise final polish.")
    ap.add_argument("--no-robust", action="store_true",
                    help="Disable soft_l1 robust loss (use plain least-squares).")
    ap.add_argument("--fix", nargs="+", default=[], choices=NAMES,
                    help="Hold these params at their defaults.json value "
                         "(e.g. --fix e n u if camera position is known).")
    ap.add_argument("--prior-sigma", nargs="+", default=[],
                    help="Soft Gaussian priors as name:sigma pairs, e.g. "
                         "--prior-sigma f:500 ti:0.2, pulling weakly-"
                         "constrained free params toward their default.")
    ap.add_argument("--e", type=float, default=None)
    ap.add_argument("--n", type=float, default=None)
    ap.add_argument("--u", type=float, default=None)
    ap.add_argument("--th", type=float, default=None)
    ap.add_argument("--ph", type=float, default=None)
    ap.add_argument("--ti", type=float, default=None)
    ap.add_argument("--f", type=float, default=None)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    key = IMG_KEYS[args.which]

    dem = DEM(cache_file=args.cache_file)
    meta = {k: dict(v) for k, v in DEFAULT_META.items()}
    img_file = find_img_file(args.which, args.img_glob)
    img = HorizonImage(img_file, meta, px_smooth=150, px_dist=30)

    default = DEFAULT_PRMS_U_BY_KEY.get(key)
    cli_overrides = [args.e, args.n, args.u, args.th, args.ph, args.ti, args.f]
    if default is None:
        if any(v is None for v in cli_overrides):
            raise ValueError(
                f"No default prms for key {key!r} and not all of "
                f"--e/--n/--u/--th/--ph/--ti/--f were passed."
            )
        init = list(cli_overrides)
    else:
        init = [c if c is not None else d for c, d in zip(cli_overrides, default)]

    prior_sigma = {}
    for tok in args.prior_sigma:
        name, val = tok.split(":")
        prior_sigma[name] = float(val)
    if args.fix:
        print(f"Holding fixed: {args.fix} (from defaults.json / CLI values)")
    if prior_sigma:
        print(f"Gaussian priors: {prior_sigma}")

    cols = np.unique(np.linspace(0, img.npix_x - 1, args.ncols).astype(int))
    cols, y_actual = extract_actual_horizon(img.sky_mask, cols)
    print(f"Extracted horizon at {len(cols)} / {args.ncols} sampled columns "
          f"(rest had no visible transition in-frame).")

    fig = plt.figure(figsize=(10, 8))
    ax_img = fig.add_axes([0.08, 0.15, 0.87, 0.78])
    ax_img.imshow(img.img, origin="lower")
    ax_img.plot(cols, y_actual, ".", color="cyan", markersize=3, label="actual horizon")
    (pred_line,) = ax_img.plot([], [], ".", color="red", markersize=3, label="predicted horizon")
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
        y_pred_before = predicted_horizon_rows(img, dem, cols, args.fine_delta,
                                               n_iter=args.bisect_iters)
        rmse_before = np.sqrt(np.mean((y_pred_before - y_actual) ** 2))
        solved, res = solve_camera(
            img, dem, cols, y_actual, state["init"], args.fine_delta,
            robust=not args.no_robust, fix=args.fix, prior_sigma=prior_sigma,
            n_iter=args.bisect_iters, diff_step=args.diff_step,
        )
        rmse = np.sqrt(np.mean(res.fun ** 2))
        img.set_prms(tuple(solved))
        y_pred = predicted_horizon_rows(img, dem, cols, args.fine_delta,
                                        n_iter=args.bisect_iters)
        pred_line.set_data(cols, y_pred)
        state["init"] = list(solved)  # allow iterative re-fitting from here
        prms_u = tuple(round(float(v), 4) for v in solved)
        print(f"RMSE before={rmse_before:.2f}px -> after={rmse:.2f}px")
        print("prms_u =", prms_u)
        status_txt.set_text(f"RMSE {rmse_before:.1f}px -> {rmse:.1f}px "
                            f"(see console for prms_u). Click Fit to refine further.")
        fig.canvas.draw_idle()

    fit_ax = fig.add_axes([0.78, 0.02, 0.17, 0.05])
    fit_btn = Button(fit_ax, "Fit")
    fit_btn.on_clicked(on_fit)

    img.set_prms(tuple(state["init"]))
    y_pred0 = predicted_horizon_rows(img, dem, cols, args.fine_delta,
                                     n_iter=args.bisect_iters)
    pred_line.set_data(cols, y_pred0)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())