#!/usr/bin/env python
"""
Re-evaluate total_logL at every posterior draw in a trace file.

This is expensive (one full likelihood evaluation per draw per chain)
but produces the actual logL timeseries that DEMetropolisZ does not store.

Outputs
-------
  <stem>_logL.npz    logL array shape (chains, draws), plus ray/ant components
  <stem>_logL.png    timeseries plot with rolling statistics

Usage
-----
  eval_trace_logL.py trace_seed634.nc --map-file map_seed792.json [options]
"""
import argparse
import glob
import json
import os

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

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


def _apply_prms(dem, meta, img_keys, prms, prm_len):
    dem["platform"] = prms[-3:].astype(dtype_r)
    off = 0
    for key in img_keys:
        chunk = prms[off: off + prm_len]
        off += prm_len
        meta[key]["prms"] = tuple(float(x) for x in chunk)
        dem[key] = np.asarray(chunk[:3], dtype=dtype_r)


def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trace-file", help="ArviZ .nc trace file")
    ap.add_argument("--map-file", default=None,
                    help="map_seed{NNN}.json — used for setup context only "
                         "(param names, img_keys). Not required if meta sidecar present.")
    ap.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
    ap.add_argument("--img-glob", default=DEFAULT_IMG_GLOB)
    ap.add_argument("--px-dist",    type=int,   default=30)
    ap.add_argument("--px-smooth",  type=int,   default=150)
    ap.add_argument("--n-rays",     type=int,   default=500,
                    help="Rays per logL eval (default 500 — fewer than training "
                         "for speed; increase for accuracy)")
    ap.add_argument("--eps",        type=float, default=1e-2)
    ap.add_argument("--fine-delta", type=float, default=0.25)
    ap.add_argument("--ant-weight", type=float, default=1.0)
    ap.add_argument("--disable-ant", action="store_true")
    ap.add_argument("--set-cam-height", action="store_true", default=True)
    ap.add_argument("--cam-height", type=float, default=1.6)
    ap.add_argument("--pos-err",     type=float, default=10.0)
    ap.add_argument("--ang-err-deg", type=float, default=1.0)
    ap.add_argument("--f-err",       type=float, default=0.1)
    ap.add_argument("--log-h-sigma", type=float, default=1.0)
    ap.add_argument("--stride",  type=int, default=1,
                    help="Evaluate every Nth draw (default 1 = all draws). "
                         "Use e.g. --stride 10 for a quick overview.")
    ap.add_argument("--outdir",  default=None)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    stem   = os.path.splitext(os.path.basename(args.trace_file))[0]
    outdir = args.outdir or f"{stem}_logL"
    os.makedirs(outdir, exist_ok=True)

    # ── load trace ────────────────────────────────────────────────────────────
    trace = az.from_netcdf(args.trace_file)
    n_chains = trace.posterior.dims["chain"]
    n_draws  = trace.posterior.dims["draw"]

    # ── load meta sidecar if present ──────────────────────────────────────────
    meta_path = args.trace_file.replace(".nc", "_meta.json")
    run_meta  = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            run_meta = json.load(f)
        print(f"Loaded sidecar: {meta_path}")

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

    ps = PositionSolver(dem["platform"], imgs, [], args.n_rays, dem,
                        box_size=BOX_SIZE)
    prms_h = ps.prms_u_to_h(prms_u)
    ps.set_mcmc_prms(prms_h)
    ps.set_mcmc_sigmas(
        pos_err=args.pos_err,
        ang_err=np.deg2rad(args.ang_err_deg),
        f_err=args.f_err,
        log_h_sigma=args.log_h_sigma,
    )

    # Fix a pixel sample for consistency across all evaluations
    for img in imgs:
        img.choose_pixels(N=args.n_rays, reset=True)

    param_names = run_meta.get("param_names") or [
        f"{img.key}_log_h" if k == "u" else f"{img.key}_{k}"
        for img in imgs for k in PRM_ORDER
    ] + ["ant_e", "ant_n", "ant_log_h"]

    eps = dtype_r(args.eps)

    # ── build draw index ──────────────────────────────────────────────────────
    draw_indices = np.arange(0, n_draws, args.stride)
    n_eval       = len(draw_indices)

    print(f"\nTrace: {args.trace_file}")
    print(f"  chains={n_chains}  draws={n_draws}  "
          f"evaluating every {args.stride} draw ({n_eval} per chain)")
    print(f"  n_rays={args.n_rays}  fine_delta={args.fine_delta}  eps={args.eps}")
    print(f"  total evaluations: {n_chains * n_eval}")
    print(f"  output: {outdir}/\n")

    # ── evaluate logL at every draw ───────────────────────────────────────────
    # shape: (chains, eval_draws)
    logL_total = np.full((n_chains, n_eval), np.nan)
    logL_rays  = np.full((n_chains, n_eval), np.nan)
    logL_ant   = np.full((n_chains, n_eval), np.nan)

    for c in range(n_chains):
        print(f"Chain {c+1}/{n_chains}:")
        for ei, d in enumerate(tqdm(draw_indices, unit="draw")):
            # Extract theta for this draw
            theta = np.array([
                float(trace.posterior[name].values[c, d])
                for name in param_names
            ], dtype=dtype_r)

            # Evaluate components separately
            try:
                ps.set_mcmc_prms(theta)
                lr = 0.0
                for img in ps.fit_imgs:
                    lr += img.horizon_ray_logL(
                        ps.dem, n_rays=args.n_rays, eps=eps,
                        fine_delta=args.fine_delta,
                    )
                la = 0.0
                if not args.disable_ant:
                    for img in ps.imgs:
                        la += img.ant_logL(ps.ant_pos, ps.box_size)
                logL_rays[c, ei] = lr
                logL_ant[c, ei]  = la
                logL_total[c, ei] = lr + args.ant_weight * la
            except Exception as exc:
                print(f"  draw {d} failed: {exc}")

    # ── save npz ──────────────────────────────────────────────────────────────
    npz_path = os.path.join(outdir, f"{stem}_logL.npz")
    np.savez(npz_path,
             logL_total=logL_total,
             logL_rays=logL_rays,
             logL_ant=logL_ant,
             draw_indices=draw_indices,
             n_chains=n_chains,
             n_draws=n_draws)
    print(f"\nSaved: {npz_path}")

    # ── plot ──────────────────────────────────────────────────────────────────
    draws_x = draw_indices  # x-axis = draw number

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"logL timeseries  —  {os.path.basename(args.trace_file)}\n"
        f"n_rays={args.n_rays}  fine_delta={args.fine_delta}  "
        f"stride={args.stride}  "
        f"accept={run_meta.get('accepted_mean', float('nan')):.3f}",
        fontsize=8, y=1.01
    )

    colors = plt.cm.tab10.colors
    window = max(1, n_eval // 20)  # rolling window ~5% of draws

    for panel, (data, title, ylabel) in enumerate([
        (logL_total, "Total logL  (rays + ant)", "logL"),
        (logL_rays,  "Ray likelihood only",       "logL_rays"),
        (logL_ant,   "Antenna likelihood only",   "logL_ant"),
    ]):
        ax = axes[panel]
        for c in range(n_chains):
            col   = colors[c % len(colors)]
            valid = np.isfinite(data[c])
            if not valid.any():
                continue
            # raw (faint)
            ax.plot(draws_x[valid], data[c][valid],
                    color=col, lw=0.4, alpha=0.3)
            # rolling mean
            roll = np.convolve(
                np.where(valid, data[c], np.nan),
                np.ones(window) / window, mode="same"
            )
            ax.plot(draws_x, roll, color=col, lw=1.2,
                    label=f"chain {c} (roll {window})")

        # reference lines
        finite_all = data[np.isfinite(data)]
        if finite_all.size > 0:
            lo = np.percentile(finite_all, 2)
            hi = np.percentile(finite_all, 98)
            span = hi - lo if hi != lo else 1.0
            ax.set_ylim(lo - 0.05 * span, hi + 0.05 * span)
            ax.axhline(np.nanmedian(data), color="k", lw=0.8,
                       ls="--", alpha=0.5, label="median")

        ax.set_title(title, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        ax.legend(fontsize=7, ncol=n_chains + 1, loc="upper right")

    axes[-1].set_xlabel("draw index")
    plt.tight_layout()
    plot_path = os.path.join(outdir, f"{stem}_logL.png")
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  logL SUMMARY  (across all chains and draws)")
    print(f"{'='*55}")
    for label, data in [("total", logL_total),
                         ("rays",  logL_rays),
                         ("ant",   logL_ant)]:
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            print(f"  {label:6s}: all NaN")
            continue
        print(f"  {label:6s}:  mean={finite.mean():.2f}  "
              f"std={finite.std():.2f}  "
              f"min={finite.min():.2f}  max={finite.max():.2f}")

    # Per-chain stationarity check: compare first/second half means
    print(f"\n  Stationarity (first half vs second half mean per chain):")
    mid = n_eval // 2
    for c in range(n_chains):
        d = logL_total[c]
        f1 = np.nanmean(d[:mid])
        f2 = np.nanmean(d[mid:])
        std = np.nanstd(d)
        drift = abs(f2 - f1)
        flag  = "WARN" if drift > std else "ok"
        print(f"  chain {c}: first={f1:.2f}  second={f2:.2f}  "
              f"drift={drift:.2f}  std={std:.2f}  [{flag}]")
    print(f"{'='*55}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())