#!/usr/bin/env python
"""
Horizon overlay visualization — Test 1 for solution validity.

For each camera, renders three panels side by side:
  1. Raw image with predicted horizon line
  2. P(sky) segmentation mask with predicted horizon overlay
  3. Pixel-wise agreement map (where model agrees/disagrees with psky)

If the MAP solution is physically correct, the predicted horizon line
should align with the visible sky/ground boundary in the raw image.

Usage
-----
  viz_horizon.py --map-file map_seed398.json [options]
  viz_horizon.py --map-file map_seed398.json --trace-file trace_seed634.nc
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER, dtype_r
from eigsep_terrain.img_defaults import load_defaults

BOX_SIZE = 0.3

# Generic, arbitrary-image-count defaults loaded from defaults.json (same
# file used by tune_image.py / fit_image.py / plot_image_fit.py). Edit
# defaults.json, not this script, when a starting value changes.
DEFAULT_IMG_GLOB, DEFAULT_CACHE_FILE, DEFAULT_META, DEFAULT_PRMS_U_BY_KEY, IMG_KEYS = \
    load_defaults()


def _build_prms_u(img_keys, dem, set_cam_height=False, cam_height=1.6):
    """Concatenate per-key (e, n, u, th, ph, ti, f) starting params, in
    img_keys order, from DEFAULT_PRMS_U_BY_KEY. Optionally override each
    image's u to dem.interp_alt(e, n) + cam_height."""
    chunks = []
    for key in img_keys:
        p = np.asarray(DEFAULT_PRMS_U_BY_KEY[key], dtype=dtype_r)
        if set_cam_height:
            p = p.copy()
            p[2] = float(dem.interp_alt(p[0], p[1])) + cam_height
        chunks.append(p)
    return np.concatenate(chunks).astype(dtype_r)


def _dummy_ant_pos_prior(prms_u, dem):
    """Antenna position isn't tracked in defaults.json (2026 dataset fits
    images independently, antenna term disabled by default). Build a dummy
    prior at the mean camera E/N, 1m above ground, just so PositionSolver's
    u<->log_h conversion has a valid DEM location."""
    n_imgs = prms_u.size // len(PRM_ORDER)
    es = prms_u[0::len(PRM_ORDER)][:n_imgs]
    ns = prms_u[1::len(PRM_ORDER)][:n_imgs]
    e0, n0 = float(es.mean()), float(ns.mean())
    return np.array([e0, n0, float(dem.interp_alt(e0, n0)) + 1.0], dtype=dtype_r)



# ── parameter source modes ────────────────────────────────────────────────────
# --mode map          : MAP values from --map-file  (default when map-file given)
# --mode post_mean    : posterior mean from --trace-file
# --mode post_median  : posterior median from --trace-file
# --mode post_sample  : single random posterior draw from --trace-file
# --mode post_last  : last step in posterior from --trace-file

def _load_theta(args, prms_h, param_names):
    """
    Load a parameter vector according to args.mode, args.map_file,
    and args.trace_file.  Returns (theta, param_source_str).

    Rules
    -----
    - map-file only            -> mode defaults to "map"
    - trace-file only          -> mode defaults to "post_mean"
    - both                     -> mode selects which to use
    - neither                  -> returns prms_h (DEFAULT_PRMS baseline)
    """
    import arviz as _az

    mode = args.mode

    # Auto-default mode when not explicitly set
    if mode is None:
        if args.trace_file is not None and args.map_file is None:
            mode = "post_mean"
        elif args.map_file is not None:
            mode = "map"
        else:
            mode = "default"

    theta = prms_h.copy()

    if mode == "map":
        if args.map_file is None:
            raise ValueError("--mode map requires --map-file")
        with open(args.map_file) as f:
            mj = json.load(f)
        for i, name in enumerate(param_names):
            if name in mj.get("map_params_h", {}):
                theta[i] = dtype_r(mj["map_params_h"][name])
        src = (f"MAP  logL={mj['map_logL']:.1f}  "
               f"method={mj['method']}  converged={mj['converged']}  "
               f"seed={mj['seed']}")
        map_json = mj

    elif mode in ("post_mean", "post_median", "post_sample", "post_last"):
        if args.trace_file is None:
            raise ValueError(f"--mode {mode} requires --trace-file")
        trace = _az.from_netcdf(args.trace_file)
        for i, name in enumerate(param_names):
            if name in trace.posterior:
                vals = trace.posterior[name].values.flatten()
                if mode == "post_mean":
                    theta[i] = dtype_r(float(vals.mean()))
                elif mode == "post_median":
                    theta[i] = dtype_r(float(np.median(vals)))
                elif mode == "post_last":
                    theta[i] = dtype_r(float(vals[-1]))
                else:  # post_sample
                    theta[i] = dtype_r(float(
                        np.random.choice(vals)
                    ))
        mode_label = {"post_mean": "posterior mean",
                      "post_median": "posterior median",
                      "post_sample": "posterior sample"}[mode]
        src = f"{mode_label}  ({os.path.basename(args.trace_file)})"
        map_json = None
        # try to load map_json from sidecar for provenance
        if args.map_file is not None:
            with open(args.map_file) as f:
                map_json = json.load(f)
            src += f"  |  MAP seed={map_json['seed']}  logL={map_json['map_logL']:.1f}"

    else:  # default — just prms_h
        src = "DEFAULT_PRMS (no map-file or trace-file)"
        map_json = None

    return theta, src, map_json

def _apply_prms(dem, meta, img_keys, prms, prm_len):
    dem["platform"] = prms[-3:].astype(dtype_r)
    off = 0
    for key in img_keys:
        chunk = prms[off: off + prm_len]
        off += prm_len
        meta[key]["prms"] = tuple(float(x) for x in chunk)
        dem[key] = np.asarray(chunk[:3], dtype=dtype_r)


def _predicted_sky_grid(img, dem, decimate=8, fine_delta=0.25):
    """Ray-trace a decimated pixel grid. Returns (row_coords, col_coords, model_sky bool array)."""
    Ny, Nx = img.npix_y, img.npix_x
    xs = np.arange(0, Ny, decimate)
    ys = np.arange(0, Nx, decimate)
    yy, xx = np.meshgrid(ys, xs)
    rays = img.get_rays(pixels=(xx.ravel(), yy.ravel()), dtype=dtype_r)
    r = img.ray_distance(dem, rays, dtype=dtype_r, fine_delta=fine_delta)
    model_sky = np.isnan(r).reshape(xx.shape)
    return xs, ys, model_sky


def _horizon_scatter(xs, ys, model_sky):
    """Return full-pixel (row, col) coords of the sky/ground boundary."""
    horiz = np.zeros_like(model_sky, dtype=bool)
    horiz[:, :-1] |= model_sky[:, :-1] != model_sky[:, 1:]
    horiz[:-1, :] |= model_sky[:-1, :] != model_sky[1:, :]
    hx, hy = np.where(horiz)
    return xs[hx], ys[hy]


def _upsample_sky(model_sky, target_shape, xs, ys):
    """Nearest-neighbour upsample of decimated model_sky to full image shape."""
    Ny, Nx = target_shape
    sky_full = np.zeros((Ny, Nx), dtype=bool)
    # For each decimated row/col, fill the block
    row_edges = np.append(xs, Ny)
    col_edges = np.append(ys, Nx)
    for i, r0 in enumerate(xs):
        r1 = row_edges[i + 1]
        for j, c0 in enumerate(ys):
            c1 = col_edges[j + 1]
            sky_full[r0:r1, c0:c1] = model_sky[i, j]
    return sky_full


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map-file",   default=None,
                    help="Path to map_seed{NNN}.json (optional if --trace-file given)")
    ap.add_argument("--trace-file", default=None,
                    help="Path to ArviZ .nc trace file (optional if --map-file given)")
    ap.add_argument("--mode", default=None,
                    choices=["map", "post_mean", "post_median", "post_sample", "post_last"],
                    help="Which parameter estimate to use. Auto-detected if not set: "
                         "map-file only -> map; trace-file only -> post_mean; "
                         "both -> map unless overridden.")
    ap.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
    ap.add_argument("--img-glob", default=DEFAULT_IMG_GLOB)
    ap.add_argument("--px-dist",    type=int, default=30)
    ap.add_argument("--px-smooth",  type=int, default=150)
    ap.add_argument("--decimate",   type=int, default=8,
                    help="Pixel stride for ray tracing (default 8; use 4 for finer horizon)")
    ap.add_argument("--set-cam-height", action="store_true", default=True)
    ap.add_argument("--cam-height", type=float, default=1.6)
    ap.add_argument("--fine-delta", type=float, default=0.25,
                    help="Ray trace fine step size [m] (default 0.25).")
    ap.add_argument("--outdir", default=None)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    _src_file = args.map_file or args.trace_file
    if _src_file is None:
        raise ValueError("Must provide at least one of --map-file or --trace-file")
    stem   = os.path.splitext(os.path.basename(_src_file))[0]
    outdir = args.outdir or f"{stem}_horizon_viz"
    os.makedirs(outdir, exist_ok=True)

    # ── setup ─────────────────────────────────────────────────────────────────
    dem   = DEM(cache_file=args.cache_file)
    files = sorted(glob.glob(args.img_glob))
    if not files:
        raise FileNotFoundError(f"No images matched: {args.img_glob}")

    meta  = {k: dict(v) for k, v in DEFAULT_META.items()}
    imgs  = [HorizonImage(f, meta, px_smooth=args.px_smooth, px_dist=args.px_dist)
             for f in files]
    imgs  = [img for img in imgs if img.key in meta]
    img_keys = [img.key for img in imgs]

    cam_prms_u = _build_prms_u(img_keys, dem, set_cam_height=args.set_cam_height,
                               cam_height=args.cam_height)
    ant_pos_prior = _dummy_ant_pos_prior(cam_prms_u, dem)
    prms_u = np.concatenate([cam_prms_u, ant_pos_prior]).astype(dtype_r)

    _apply_prms(dem, meta, img_keys, prms_u, len(PRM_ORDER))

    ps = PositionSolver(dem["platform"], imgs, [], 100, dem, box_size=BOX_SIZE)
    prms_h = ps.prms_u_to_h(prms_u)
    param_names = [
        f"{img.key}_log_h" if k == "u" else f"{img.key}_{k}"
        for img in imgs for k in PRM_ORDER
    ] + ["ant_e", "ant_n", "ant_log_h"]

    # ── load params ───────────────────────────────────────────────────────────
    theta, param_source, map_json = _load_theta(args, prms_h, param_names)
    ps.set_mcmc_prms(theta)
    print(f"Parameters: {param_source}")

    # ── per-camera plots ──────────────────────────────────────────────────────
    for img in imgs:
        key = img.key
        print(f"\nCamera {key}: ray-tracing {img.npix_y}x{img.npix_x} "
              f"at 1/{args.decimate} resolution...")

        xs, ys, model_sky = _predicted_sky_grid(img, dem, decimate=args.decimate,
                                                fine_delta=args.fine_delta)
        hx_full, hy_full  = _horizon_scatter(xs, ys, model_sky)
        sky_up = _upsample_sky(model_sky, (img.npix_y, img.npix_x), xs, ys)

        # agreement map
        obs_sky      = img.psky > 0.5
        agree_sky    =  sky_up &  obs_sky
        agree_gnd    = ~sky_up & ~obs_sky
        wrong_ground = ~sky_up &  obs_sky   # model says ground, image says sky
        wrong_sky    =  sky_up & ~obs_sky   # model says sky,    image says ground

        rgb = np.ones((*img.psky.shape, 3))
        rgb[agree_sky]    = [0.20, 0.78, 0.20]
        rgb[agree_gnd]    = [0.90, 0.90, 0.90]
        rgb[wrong_ground] = [0.88, 0.18, 0.18]
        rgb[wrong_sky]    = [0.18, 0.18, 0.88]

        pct_agree = 100 * (agree_sky | agree_gnd).sum() / img.psky.size

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        _h_str = (f"h={np.exp(map_json['map_params_h'].get(f'{key}_log_h', 0)):.2f}m  "
                  if map_json else "")
        title = (f"Camera {key}  —  Horizon overlay\n"
                 f"{param_source}\n"
                 f"E={img.prms['e']:.1f}  N={img.prms['n']:.1f}  "
                 + _h_str
                 + f"θ={img.prms['th']:.4f}  φ={img.prms['ph']:.4f}")
        fig.suptitle(title, fontsize=8, y=1.01)

        # Panel 1: raw image
        ax = axes[0]
        ax.imshow(img.img, origin="lower", aspect="auto")
        ax.scatter(hy_full, hx_full, s=2, c="red", linewidths=0,
                   label="predicted horizon", rasterized=True)
        if "ant_px" in img.meta:
            apx = img.meta["ant_px"]
            ax.plot(apx[0], apx[1], "y*", ms=14, label="antenna pixel",
                    markeredgecolor="k", markeredgewidth=0.5)
        ax.set_title("Raw image  +  predicted horizon", fontsize=8)
        ax.set_xlabel("pixel x");  ax.set_ylabel("pixel y (0=bottom)")
        ax.legend(fontsize=7)

        # Panel 2: psky
        ax = axes[1]
        im2 = ax.imshow(img.psky, origin="lower", aspect="auto",
                        cmap="RdYlGn", vmin=0, vmax=1)
        ax.scatter(hy_full, hx_full, s=2, c="blue", linewidths=0,
                   label="predicted horizon", rasterized=True)
        plt.colorbar(im2, ax=ax, fraction=0.03, pad=0.02, label="P(sky)")
        ax.set_title("P(sky) mask  +  predicted horizon", fontsize=8)
        ax.set_xlabel("pixel x")
        ax.legend(fontsize=7)

        # Panel 3: agreement
        ax = axes[2]
        ax.imshow(rgb, origin="lower", aspect="auto")
        ax.scatter(hy_full, hx_full, s=2, c="black", linewidths=0,
                   label="predicted horizon", rasterized=True)
        legend_elements = [
            Line2D([0],[0], marker="s", color="w",
                   markerfacecolor="#33c733", ms=10,
                   label=f"agree sky    ({agree_sky.sum():,})"),
            Line2D([0],[0], marker="s", color="w",
                   markerfacecolor="#e5e5e5", ms=10,
                   label=f"agree ground ({agree_gnd.sum():,})"),
            Line2D([0],[0], marker="s", color="w",
                   markerfacecolor="#e02e2e", ms=10,
                   label=f"wrong ground ({wrong_ground.sum():,})"),
            Line2D([0],[0], marker="s", color="w",
                   markerfacecolor="#2e2ee0", ms=10,
                   label=f"wrong sky    ({wrong_sky.sum():,})"),
            Line2D([0],[0], marker="s", color="w",
                   markerfacecolor="black", ms=10,
                   label="predicted horizon"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="upper right")
        ax.set_title(f"Agreement map  ({pct_agree:.1f}% pixels agree)", fontsize=8)
        ax.set_xlabel("pixel x")

        plt.tight_layout()
        outpath = os.path.join(outdir, f"horizon_{key}.png")
        fig.savefig(outpath, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {outpath}")
        print(f"  pixel agreement: {pct_agree:.1f}%  "
              f"(wrong_ground={wrong_ground.sum():,}  wrong_sky={wrong_sky.sum():,})")

    print(f"\nDone. Output in: {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())