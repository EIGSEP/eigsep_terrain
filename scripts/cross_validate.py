#!/usr/bin/env python
"""
Held-out pixel cross-validation — Test 5 for solution validity.

Splits horizon pixels into two halves (fit / held-out), evaluates logL
on the held-out half at the MAP solution and at random prior draws, and
reports whether the MAP generalizes or is overfitting the fit pixels.

A solution that genuinely fits the horizon should score well on held-out
pixels. If MAP logL on held-out pixels is much worse than on fit pixels,
the solution is overfitting noise in the pixel sample.

Also evaluates logL on the held-out set at n_prior_samples random prior
draws to establish a baseline for what "random" looks like — the MAP
should be substantially better than this baseline.

Outputs
-------
  crossval_<map_stem>.json   Numerical results
  crossval_<map_stem>.png    Visualization of fit vs held-out logL distributions

Usage
-----
  cross_validate.py --map-file map_seed398.json [options]
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER, dtype_r
from eigsep_terrain.img_defaults import load_defaults

BOX_SIZE = 0.3

# Generic, arbitrary-image-count defaults loaded from defaults.json (same file
# used by tune_image.py / fit_image.py / plot_image_fit.py). Edit
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
    images independently, antenna term disabled). Build a dummy prior at
    the mean camera E/N, 1m above ground, just so PositionSolver's
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
                      "post_sample": "posterior sample",
                      "post_last": "posterior last"}[mode]
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


def _eval_logL_on_pixels(img, dem, x_px, y_px, eps=1e-2, fine_delta=0.25, dtype=dtype_r):
    """
    Evaluate horizon_ray_logL on a specific fixed set of pixels.
    Returns per-pixel logL contributions (not summed) so we can inspect
    the distribution.
    """
    psky = img.psky[x_px, y_px].clip(eps, 1 - eps)
    rays = img.get_rays(pixels=(x_px, y_px), dtype=dtype)
    r    = img.ray_distance(dem, rays, dtype=dtype, fine_delta=fine_delta)
    model_sky  = np.isnan(r)
    logp_sky   = np.log(psky)
    logp_gnd   = np.log1p(-psky)
    per_pixel  = np.where(model_sky, logp_sky, logp_gnd)
    return per_pixel


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
    ap.add_argument("--px-dist",    type=int,   default=30)
    ap.add_argument("--px-smooth",  type=int,   default=150)
    ap.add_argument("--n-rays",     type=int,   default=4000,
                    help="Total horizon pixels to draw (split 50/50 fit/held-out)")
    ap.add_argument("--eps",        type=float, default=1e-2)
    ap.add_argument("--fine-delta", type=float, default=0.25,
                    help="Ray trace fine step size [m] (default 0.25).")
    ap.add_argument("--n-prior-samples", type=int, default=50,
                    help="Random prior draws for baseline comparison")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--set-cam-height", action="store_true", default=True)
    ap.add_argument("--cam-height", type=float, default=1.6)
    ap.add_argument("--pos-err",     type=float, default=10.0)
    ap.add_argument("--ang-err-deg", type=float, default=1.0)
    ap.add_argument("--f-err",       type=float, default=0.1)
    ap.add_argument("--log-h-sigma", type=float, default=1.0)
    ap.add_argument("--outdir",     default=None)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    rng  = np.random.default_rng(args.seed)

    _src_file = args.map_file or args.trace_file
    if _src_file is None:
        raise ValueError("Must provide at least one of --map-file or --trace-file")
    stem   = os.path.splitext(os.path.basename(_src_file))[0]
    outdir = args.outdir or f"{stem}_crossval"
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

    prms_u = _build_prms_u(img_keys, dem, set_cam_height=args.set_cam_height,
                           cam_height=args.cam_height)
    ant_pos_prior = _dummy_ant_pos_prior(prms_u, dem)

    _apply_prms(dem, meta, img_keys, np.concatenate([prms_u, ant_pos_prior]),
               len(PRM_ORDER))

    ps = PositionSolver(dem["platform"], imgs, [], args.n_rays, dem, box_size=BOX_SIZE)
    prms_h = ps.prms_u_to_h(prms_u)
    ps.set_mcmc_prms(prms_h)
    ps.set_mcmc_sigmas(
        pos_err=args.pos_err,
        ang_err=np.deg2rad(args.ang_err_deg),
        f_err=args.f_err,
        log_h_sigma=args.log_h_sigma,
    )
    sigmas = np.asarray(ps.sigmas)

    param_names = [
        f"{img.key}_log_h" if k == "u" else f"{img.key}_{k}"
        for img in imgs for k in PRM_ORDER
    ] + ["ant_e", "ant_n", "ant_log_h"]

    # ── load params ───────────────────────────────────────────────────────────
    map_theta, param_source, map_json = _load_theta(args, prms_h, param_names)
    ps.set_mcmc_prms(map_theta)

    logL_str = f"logL={map_json['map_logL']:.2f}" if map_json else ""
    print(f"\n{'='*55}")
    print(f"  CROSS-VALIDATION")
    print(f"  {param_source}  {logL_str}")
    print(f"  n_rays={args.n_rays}  split=50/50  "
          f"n_prior_samples={args.n_prior_samples}")
    print(f"{'='*55}")

    results = {}

    fig, axes = plt.subplots(len(imgs), 3,
                             figsize=(15, 4 * len(imgs)))
    if len(imgs) == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        f"Cross-validation  —  {os.path.basename(args.map_file)}\n"
        f"MAP logL={map_json['map_logL']:.1f}  "
        f"n_rays={args.n_rays}  seed={args.seed}",
        fontsize=9, y=1.01
    )

    for row, img in enumerate(imgs):
        key = img.key
        print(f"\nCamera {key}:")

        # ── draw all pixels then split 50/50 ──────────────────────────────────
        # Draw 2x n_rays so each half has n_rays pixels
        all_px = img.choose_pixels(N=args.n_rays * 2, reset=True)
        x_all, y_all = all_px
        n_half = len(x_all) // 2
        # shuffle before splitting so spatial distribution is similar
        perm = rng.permutation(len(x_all))
        fit_idx  = perm[:n_half]
        held_idx = perm[n_half: 2 * n_half]
        x_fit,  y_fit  = x_all[fit_idx],  y_all[fit_idx]
        x_held, y_held = x_all[held_idx], y_all[held_idx]

        # ── MAP logL on each split ─────────────────────────────────────────────
        pp_fit  = _eval_logL_on_pixels(img, dem, x_fit,  y_fit,  eps=args.eps, fine_delta=args.fine_delta)
        pp_held = _eval_logL_on_pixels(img, dem, x_held, y_held, eps=args.eps, fine_delta=args.fine_delta)

        logL_fit  = float(pp_fit.sum())
        logL_held = float(pp_held.sum())
        # Per-pixel means for fair comparison across different n
        ppl_fit  = float(pp_fit.mean())
        ppl_held = float(pp_held.mean())
        gap      = ppl_held - ppl_fit   # negative = held-out worse

        print(f"  MAP logL/pixel   fit={ppl_fit:.4f}   held-out={ppl_held:.4f}   "
              f"gap={gap:+.4f}")

        # ── prior baseline on held-out pixels ─────────────────────────────────
        prior_ppl_held = []
        for _ in range(args.n_prior_samples):
            theta_prior = prms_h + rng.normal(0.0, sigmas)
            ps.set_mcmc_prms(theta_prior)
            try:
                pp = _eval_logL_on_pixels(img, dem, x_held, y_held, eps=args.eps, fine_delta=args.fine_delta)
                if np.isfinite(pp.sum()):
                    prior_ppl_held.append(float(pp.mean()))
            except Exception:
                pass
        # Restore MAP params
        ps.set_mcmc_prms(map_theta)

        prior_mean = float(np.mean(prior_ppl_held)) if prior_ppl_held else float("nan")
        prior_std  = float(np.std(prior_ppl_held))  if prior_ppl_held else float("nan")
        z_vs_prior = ((ppl_held - prior_mean) / prior_std
                      if prior_std > 0 else float("nan"))

        print(f"  Prior baseline (held-out):  mean={prior_mean:.4f}  "
              f"std={prior_std:.4f}  z={z_vs_prior:+.2f}")

        # ── interpretation ────────────────────────────────────────────────────
        # A good solution should:
        #   1. Have small |gap| — fit and held-out logL/pixel are similar
        #   2. Be much better than the prior baseline (large positive z)
        gap_warn  = abs(gap) > 0.05 * abs(ppl_fit)   # >5% relative gap
        z_warn    = z_vs_prior < 2.0                   # less than 2σ above prior

        status = "OK"
        if gap_warn and z_warn:
            status = "FAIL — overfitting AND not better than prior"
        elif gap_warn:
            status = "WARN — gap between fit/held-out suggests overfitting"
        elif z_warn:
            status = "WARN — not clearly better than random prior draw"

        print(f"  Status: {status}")

        results[key] = {
            "logL_fit":       logL_fit,
            "logL_held":      logL_held,
            "ppl_fit":        ppl_fit,
            "ppl_held":       ppl_held,
            "gap":            gap,
            "prior_ppl_mean": prior_mean,
            "prior_ppl_std":  prior_std,
            "z_vs_prior":     z_vs_prior,
            "n_fit":          int(n_half),
            "n_held":         int(n_half),
            "n_prior_finite": len(prior_ppl_held),
            "status":         status,
        }

        # ── plots ─────────────────────────────────────────────────────────────
        # Panel 1: per-pixel logL distribution fit vs held-out
        ax = axes[row, 0]
        bins = np.linspace(
            min(pp_fit.min(), pp_held.min()),
            max(pp_fit.max(), pp_held.max()), 50
        )
        ax.hist(pp_fit,  bins=bins, alpha=0.6, label=f"fit  μ={ppl_fit:.3f}",
                color="steelblue", density=True)
        ax.hist(pp_held, bins=bins, alpha=0.6, label=f"held-out  μ={ppl_held:.3f}",
                color="coral", density=True)
        ax.axvline(ppl_fit,  color="steelblue", lw=1.5, ls="--")
        ax.axvline(ppl_held, color="coral",      lw=1.5, ls="--")
        ax.set_title(f"Cam {key}: per-pixel logL  (gap={gap:+.4f})", fontsize=8)
        ax.set_xlabel("logL per pixel");  ax.set_ylabel("density")
        ax.legend(fontsize=7)

        # Panel 2: MAP ppl vs prior distribution on held-out
        ax = axes[row, 1]
        if prior_ppl_held:
            ax.hist(prior_ppl_held, bins=25, alpha=0.7, color="gray",
                    label=f"prior draws  μ={prior_mean:.3f}", density=True)
        ax.axvline(ppl_held, color="red", lw=2,
                   label=f"MAP held-out  {ppl_held:.3f}  (z={z_vs_prior:+.1f})")
        ax.set_title(f"Cam {key}: MAP vs prior baseline (held-out)", fontsize=8)
        ax.set_xlabel("logL/pixel");  ax.set_ylabel("density")
        ax.legend(fontsize=7)

        # Panel 3: spatial map of fit vs held-out pixels coloured by logL
        ax = axes[row, 2]
        ax.imshow(img.psky, origin="lower", aspect="auto",
                  cmap="gray", alpha=0.4)
        sc_fit  = ax.scatter(y_fit,  x_fit,  c=pp_fit,  s=2,
                             cmap="RdYlGn", vmin=bins[0], vmax=bins[-1],
                             label="fit pixels", rasterized=True)
        ax.scatter(y_held, x_held, c=pp_held, s=2, marker="x",
                   cmap="RdYlGn", vmin=bins[0], vmax=bins[-1],
                   label="held-out pixels", rasterized=True)
        plt.colorbar(sc_fit, ax=ax, fraction=0.03, label="logL/pixel")
        ax.set_title(f"Cam {key}: pixel locations coloured by logL", fontsize=8)
        ax.set_xlabel("pixel x");  ax.legend(fontsize=7)

    plt.tight_layout()
    plot_path = os.path.join(outdir, f"crossval_{stem}.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  CROSS-VALIDATION SUMMARY")
    print(f"{'='*55}")
    for key, r in results.items():
        print(f"  {key}:  gap={r['gap']:+.4f}  "
              f"z_vs_prior={r['z_vs_prior']:+.2f}  "
              f"status={r['status']}")
    print(f"{'='*55}")

    # ── write JSON ─────────────────────────────────────────────────────────────
    out = {
        "param_source":    param_source,
        "map_file":        args.map_file,
        "map_logL":        map_json["map_logL"] if map_json else None,
        "seed":            args.seed,
        "n_rays_per_half": args.n_rays,
        "n_prior_samples": args.n_prior_samples,
        "cameras":         results,
    }
    json_path = os.path.join(outdir, f"crossval_{stem}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nPlot: {plot_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())