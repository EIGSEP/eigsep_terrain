#!/usr/bin/env python
"""
Load a trace_img{KEY}_seed{NNN}.nc + _meta.json produced by fit_image.py and
produce:
  {stem}_trace.png    ArviZ trace plot (chains + marginals)
  {stem}_overlay.png  Original image with actual-sky-mask boundary vs
                       predicted (ray-traced) horizon boundary, using the
                       posterior mean params.
"""
import argparse
import glob
import json
import os

import numpy as np
import arviz as az
import matplotlib.pyplot as plt

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER, dtype_r
from eigsep_terrain.img_defaults import load_defaults

# Per-image defaults now loaded from defaults.json so this stays in sync
# with tune_image.py / fit_image.py. Edit defaults.json, not this script.
IMG_GLOB, DEFAULT_CACHE_FILE, DEFAULT_META, DEFAULT_PRMS_U_BY_KEY, IMG_KEYS = \
    load_defaults()


def find_img_file(which: int, img_glob: str) -> str:
    key = IMG_KEYS[which]
    files = sorted(glob.glob(img_glob))
    matches = [f for f in files if os.path.basename(f).split("_")[-1].split(".")[0] == key]
    if not matches:
        raise FileNotFoundError(f"No file matching key {key!r} found via glob {img_glob!r}")
    return matches[0]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=None, help="trace_img{KEY}_seed{NNN}.nc")
    ap.add_argument("--params", type=float, nargs=7, default=None,
                    metavar=("E", "N", "U", "TH", "PH", "TI", "F"),
                    help="Skip the trace/posterior and overlay these params directly.")
    ap.add_argument("--meta", default=None,
                    help="Defaults to {trace_stem}_meta.json")
    ap.add_argument("--which", type=int, required=True,
                    choices=list(range(len(IMG_KEYS))),
                    help="Which image was fit, by index into IMG_KEYS "
                         "(matches tune_image.py's --which). "
                         "0=%s ... %d=%s" % (IMG_KEYS[0], len(IMG_KEYS) - 1, IMG_KEYS[-1]))
    ap.add_argument("--img-glob", default=IMG_GLOB)
    ap.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
    ap.add_argument("--stride", type=int, default=8,
                    help="Pixel stride for the overlay ray grid.")
    ap.add_argument("--n-fine-delta", type=float, default=0.25)
    ap.add_argument("--out-prefix", default=None)
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    dem = DEM(cache_file=args.cache_file)

    if args.params is not None:
        key = IMG_KEYS[args.which]
        e, n, u, th, ph, ti, f_ = args.params
        out_dir = "plots_manual"
        os.makedirs(out_dir, exist_ok=True)
        tag = np.random.randint(1000)
        out_prefix = os.path.join(
            out_dir, args.out_prefix if args.out_prefix is not None else f"img{key}_manual{tag:03d}"
        )
    else:
        if args.trace is None:
            raise ValueError("Must pass --trace or --params.")
        stem = os.path.splitext(os.path.basename(args.trace))[0]
        metafile = args.meta if args.meta is not None else f"{os.path.splitext(args.trace)[0]}_meta.json"

        idata = az.from_netcdf(args.trace)
        with open(metafile) as f:
            run_meta = json.load(f)

        key = run_meta["img_key"]
        param_names = run_meta["param_names"]

        out_dir = f"plots_seed{run_meta['seed']:03d}"
        os.makedirs(out_dir, exist_ok=True)
        out_prefix = os.path.join(out_dir, args.out_prefix if args.out_prefix is not None else stem)

        # ── trace plot ───────────────────────────────────────────────────
        axes = az.plot_trace(idata, var_names=param_names, compact=False)
        fig = axes.ravel()[0].figure
        fig.suptitle(f"Trace: image {key} (seed {run_meta['seed']})")
        fig.tight_layout()
        trace_png = f"{out_prefix}_trace.png"
        fig.savefig(trace_png, dpi=150)
        plt.close(fig)
        print(f"Wrote {trace_png}")

        # ── posterior mean params -> absolute u ─────────────────────────
        means_h = {}
        for name in param_names:
            means_h[name] = float(idata.posterior[name].values.mean())

        e = means_h[f"{key}_e"]
        n = means_h[f"{key}_n"]
        log_h = means_h[f"{key}_log_h"]
        th = means_h[f"{key}_th"]
        ph = means_h[f"{key}_ph"]
        ti = means_h[f"{key}_ti"]
        f_ = means_h[f"{key}_f"]

        u0 = float(dem.interp_alt(e, n))
        ant_dummy = (e, n, u0 + 1.0)
        ps = PositionSolver(ant_dummy, fit_imgs=[None], static_imgs=[],
                            n_rays=1, dem=dem, box_size=0.3)
        theta_h_padded = np.array([e, n, log_h, th, ph, ti, f_, e, n, 0.0], dtype=dtype_r)
        u = ps._convert_uh_prms(theta_h_padded, sign=1)[2]

    img_file = find_img_file(args.which, args.img_glob)
    meta = {k: dict(v) for k, v in DEFAULT_META.items()}
    img = HorizonImage(img_file, meta, px_smooth=150, px_dist=30)
    if img.key != key:
        raise ValueError(f"--which key {img.key!r} does not match trace key {key!r}")
    img.set_prms((e, n, u, th, ph, ti, f_))

    # ── build a downsampled pixel grid and ray-trace it ─────────────────
    ys = np.arange(0, img.npix_y, args.stride)
    xs = np.arange(0, img.npix_x, args.stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    x_px = yy.ravel()  # row index (Nu)
    y_px = xx.ravel()  # col index (Nv)

    rays = img.get_rays(pixels=(x_px, y_px), dtype=dtype_r)
    r = img.ray_distance(dem, rays, dtype=dtype_r, fine_delta=args.n_fine_delta)
    model_sky = np.isnan(r).reshape(yy.shape)

    actual_sky = img.sky_mask[np.ix_(ys, xs)] > 0.5

    # ── default-params horizon, for comparison ───────────────────────────
    img.set_prms(DEFAULT_PRMS_U_BY_KEY[key])
    rays_d = img.get_rays(pixels=(x_px, y_px), dtype=dtype_r)
    r_d = img.ray_distance(dem, rays_d, dtype=dtype_r, fine_delta=args.n_fine_delta)
    default_model_sky = np.isnan(r_d).reshape(yy.shape)
    img.set_prms((e, n, u, th, ph, ti, f_))

    # ── overlay figure ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img.img, origin="lower")
    ax.contour(xx, yy, actual_sky.astype(float), levels=[0.5],
               colors="cyan", linewidths=1.5)
    ax.contour(xx, yy, model_sky.astype(float), levels=[0.5],
               colors="red", linewidths=1.5)
    ax.contour(xx, yy, default_model_sky.astype(float), levels=[0.5],
               colors="lime", linewidths=1.5, linestyles="dashed")
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="cyan", lw=1.5, label="actual sky boundary"),
        Line2D([0], [0], color="red", lw=1.5, label="fit horizon"),
        Line2D([0], [0], color="lime", lw=1.5, ls="dashed", label="default horizon"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", framealpha=0.8)

    priors_txt = (
        f"priors: pos_err={run_meta['priors']['pos_err']:.1f}  "
        f"ang_err_deg={run_meta['priors']['ang_err_deg']:.1f}  "
        f"f_err={run_meta['priors']['f_err']:.2f}  "
        f"log_h_sigma={run_meta['priors']['log_h_sigma']:.2f}"
        if args.params is None else "manual params (no priors)"
    )
    fit_txt = (f"fit: e={e:.1f} n={n:.1f} u={u:.1f} th={th:.3f} "
               f"ph={ph:.3f} ti={ti:.3f} f={f_:.1f}")
    fig_txt = fit_txt if args.params is not None else f"{fit_txt}\n{priors_txt}"
    ax.text(0.01, -0.08, fig_txt, transform=ax.transAxes,
            fontsize=9, va="top", ha="left")
    ax.set_title(f"Image {key}")
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    overlay_png = f"{out_prefix}_overlay.png"
    fig.tight_layout()
    fig.savefig(overlay_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {overlay_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())