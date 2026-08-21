#!/usr/bin/env python
"""
DEM profile diagnostic for missing horizon chunks.

For each camera, identifies the blue (wrong-sky) regions from the
agreement map, picks representative rays through those pixels, and
plots the ray path altitude vs DEM altitude along each ray.

This distinguishes two failure modes:
  A. DEM doesn't have the terrain: the DEM surface never rises above
     the ray altitude along the full path — the canyon wall simply
     isn't represented in the data. No amount of fine_delta will fix this.

  B. Ray step too coarse: the DEM surface does rise above the ray
     but the step lands on the other side of the wall — fine_delta
     needs to be smaller, or the DEM resolution is insufficient.

Usage
-----
  viz_dem_profile.py --map-file map_seed792.json [options]
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER, dtype_r
from eigsep_terrain.ray_numba import ray_distance_coarse_to_fine_numba
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


def _load_theta(args, prms_h, param_names):
    """Same as viz_horizon — load params from map or trace file."""
    theta = prms_h.copy()
    map_json = None
    if args.map_file is not None:
        with open(args.map_file) as f:
            map_json = json.load(f)
        for i, name in enumerate(param_names):
            if name in map_json.get("map_params_h", {}):
                theta[i] = dtype_r(map_json["map_params_h"][name])
    elif args.trace_file is not None:
        import arviz as az
        trace = az.from_netcdf(args.trace_file)
        for i, name in enumerate(param_names):
            if name in trace.posterior:
                theta[i] = dtype_r(float(trace.posterior[name].values.mean()))
    return theta, map_json


def _ray_dem_profile(img, dem, px_row, px_col, fine_delta=0.25,
                     r_max=None, n_sample=500):
    """
    For a single pixel (px_row, px_col), trace the ray from the camera
    and sample both:
      - ray_alt(r): camera_u + r * ray_z  (altitude of the ray at distance r)
      - dem_alt(r): DEM elevation directly below the ray at distance r

    Returns:
      r_vals      : distances along ray [m]
      ray_alts    : altitude of ray point at each r [m]
      dem_alts    : DEM altitude below ray at each r [m]
      r_hit       : distance of first terrain intersection (NaN if missed)
      clearance   : ray_alt - dem_alt at each r (negative = ray below terrain)
      min_clearance: minimum clearance along the ray (most dangerous miss)
      min_clearance_r: r at which minimum clearance occurs
    """
    E_grid, N_grid = dem.get_en()
    U = dem.data
    dE = float(E_grid[1] - E_grid[0])
    dN = float(N_grid[1] - N_grid[0])
    E0, Emax = float(E_grid[0]), float(E_grid[-1])
    N0, Nmax = float(N_grid[0]), float(N_grid[-1])
    Ne, Nn   = len(E_grid), len(N_grid)

    # Camera start point
    sp = np.array([img.prms[k] for k in 'enu'], dtype=np.float64)

    # Ray direction for this pixel
    ray = img.get_rays(pixels=(np.array([px_row]), np.array([px_col]))).flatten()
    ray = ray.astype(np.float64)

    # Max range: to DEM boundary or r_max
    if r_max is None:
        r_max = 3000.0  # generous upper bound

    r_vals = np.linspace(0.0, r_max, n_sample)
    ray_alts = np.full(n_sample, np.nan)
    dem_alts  = np.full(n_sample, np.nan)

    for i, r in enumerate(r_vals):
        pt = sp + r * ray
        px_e = pt[0];  px_n = pt[1];  px_u = pt[2]

        # Out of DEM bounds
        if not (E0 <= px_e <= Emax and N0 <= px_n <= Nmax):
            break

        e_idx = int(np.clip((px_e - E0) / dE, 0, Ne - 2))
        n_idx = int(np.clip((px_n - N0) / dN, 0, Nn - 2))
        dem_u = float(U[n_idx, e_idx])

        ray_alts[i] = px_u
        dem_alts[i]  = dem_u

    # Trim to valid range
    valid = np.isfinite(ray_alts) & np.isfinite(dem_alts)
    r_vals    = r_vals[valid]
    ray_alts  = ray_alts[valid]
    dem_alts   = dem_alts[valid]

    clearance = ray_alts - dem_alts

    # Find terrain hit with fine_delta stepping
    rays_2d  = ray.reshape(3, 1).astype(dtype_r)
    sp_f32   = sp.astype(dtype_r)
    r_hit_arr = ray_distance_coarse_to_fine_numba(
        E_grid, N_grid, U, sp_f32, rays_2d, fine_delta=fine_delta
    )
    r_hit = float(r_hit_arr[0])  # NaN if missed

    # Minimum clearance (most dangerous near-miss)
    if clearance.size > 0:
        min_idx = int(np.argmin(clearance))
        min_clearance   = float(clearance[min_idx])
        min_clearance_r = float(r_vals[min_idx])
    else:
        min_clearance   = np.nan
        min_clearance_r = np.nan

    return (r_vals, ray_alts, dem_alts, r_hit,
            clearance, min_clearance, min_clearance_r)


def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--map-file",   default=None)
    ap.add_argument("--trace-file", default=None)
    ap.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
    ap.add_argument("--img-glob", default=DEFAULT_IMG_GLOB)
    ap.add_argument("--px-dist",    type=int,   default=30)
    ap.add_argument("--px-smooth",  type=int,   default=150)
    ap.add_argument("--fine-delta", type=float, default=0.25)
    ap.add_argument("--decimate",   type=int,   default=8)
    ap.add_argument("--n-rays-profile", type=int, default=8,
                    help="Number of representative wrong-sky rays to profile per camera")
    ap.add_argument("--r-max",      type=float, default=None,
                    help="Max ray distance to profile [m] (default: auto)")
    ap.add_argument("--n-sample",   type=int,   default=800,
                    help="Points sampled along each ray profile")
    ap.add_argument("--set-cam-height", action="store_true", default=True)
    ap.add_argument("--cam-height", type=float, default=1.6)
    ap.add_argument("--outdir",     default=None)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    _src = args.map_file or args.trace_file
    if _src is None:
        raise ValueError("Provide --map-file or --trace-file")
    stem   = os.path.splitext(os.path.basename(_src))[0]
    outdir = args.outdir or f"{stem}_dem_profiles"
    os.makedirs(outdir, exist_ok=True)

    # ── setup ──────────────────────────────────────────────────────────────────
    dem   = DEM(cache_file=args.cache_file)
    E_grid, N_grid = dem.get_en()
    print(f"DEM: {len(E_grid)}x{len(N_grid)}  "
          f"dE={E_grid[1]-E_grid[0]:.2f}m  dN={N_grid[1]-N_grid[0]:.2f}m")

    files = sorted(glob.glob(args.img_glob))
    if not files:
        raise FileNotFoundError(f"No images: {args.img_glob}")

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

    theta, map_json = _load_theta(args, prms_h, param_names)
    ps.set_mcmc_prms(theta)

    src_str = (f"MAP logL={map_json['map_logL']:.1f}" if map_json
               else "posterior mean")

    # ── per-camera analysis ────────────────────────────────────────────────────
    for img in imgs:
        key = img.key
        print(f"\n{'='*55}")
        print(f"  Camera {key}  E={img.prms['e']:.1f}  N={img.prms['n']:.1f}  "
              f"u={img.prms['u']:.1f}")

        # ── find wrong-sky pixels (model=sky, obs=ground) ──────────────────────
        dec = args.decimate
        Ny, Nx = img.npix_y, img.npix_x
        xs = np.arange(0, Ny, dec)
        ys = np.arange(0, Nx, dec)
        yy, xx = np.meshgrid(ys, xs)
        rays_all = img.get_rays(pixels=(xx.ravel(), yy.ravel()), dtype=dtype_r)
        r_all = img.ray_distance(dem, rays_all, dtype=dtype_r,
                                  fine_delta=args.fine_delta)
        model_sky_dec = np.isnan(r_all).reshape(len(xs), len(ys))

        # Identify wrong-sky in decimated grid: model=sky, psky<0.5
        psky_dec = img.psky[xs[:, None], ys[None, :]]
        wrong_sky_dec = model_sky_dec & (psky_dec < 0.5)

        n_wrong = wrong_sky_dec.sum()
        print(f"  Wrong-sky pixels (decimated): {n_wrong}")

        if n_wrong == 0:
            print("  No wrong-sky pixels found — skipping profile analysis")
            continue

        # Pick representative wrong-sky pixels spread across x-range
        ri, ci = np.where(wrong_sky_dec)
        # Sort by column (x position) and sample evenly
        sort_idx = np.argsort(ci)
        ri, ci   = ri[sort_idx], ci[sort_idx]
        if len(ri) > args.n_rays_profile:
            sample_idx = np.round(
                np.linspace(0, len(ri) - 1, args.n_rays_profile)
            ).astype(int)
            ri, ci = ri[sample_idx], ci[sample_idx]

        px_rows = xs[ri]
        px_cols = ys[ci]

        # ── profile each selected ray ──────────────────────────────────────────
        profiles = []
        for px_row, px_col in zip(px_rows, px_cols):
            result = _ray_dem_profile(
                img, dem, px_row, px_col,
                fine_delta=args.fine_delta,
                r_max=args.r_max,
                n_sample=args.n_sample,
            )
            profiles.append({
                "px_row": int(px_row), "px_col": int(px_col),
                "r_vals":          result[0],
                "ray_alts":        result[1],
                "dem_alts":        result[2],
                "r_hit":           result[3],
                "clearance":       result[4],
                "min_clearance":   result[5],
                "min_clearance_r": result[6],
            })

        # ── diagnosis per ray ──────────────────────────────────────────────────
        print(f"\n  Ray profiles (fine_delta={args.fine_delta}m):")
        print(f"  {'px_col':>7}  {'r_hit':>8}  {'min_clear':>10}  "
              f"{'min_clear_r':>12}  diagnosis")
        print(f"  {'-'*65}")

        diagnoses = []
        for p in profiles:
            mc  = p["min_clearance"]
            mcr = p["min_clearance_r"]
            hit = p["r_hit"]

            if np.isnan(hit):
                # Ray missed — why?
                if np.isfinite(mc) and mc < 0:
                    # DEM surface crossed the ray path — fine_delta too coarse
                    diag = f"STEP_MISS  (DEM rose {-mc:.1f}m above ray at r={mcr:.0f}m)"
                elif np.isfinite(mc) and mc >= 0:
                    # Ray always above DEM — terrain not in DEM
                    diag = f"NO_TERRAIN (min clearance={mc:.1f}m — canyon not in DEM)"
                else:
                    diag = "OUT_OF_BOUNDS"
            else:
                diag = f"HIT at r={hit:.0f}m (should not be wrong-sky)"

            diagnoses.append(diag)
            print(f"  {p['px_col']:>7}  "
                  f"{hit if np.isfinite(hit) else 'nan':>8}  "
                  f"{mc:>10.2f}  {mcr:>12.1f}  {diag}")

        # Summarise
        n_step_miss  = sum("STEP_MISS"  in d for d in diagnoses)
        n_no_terrain = sum("NO_TERRAIN" in d for d in diagnoses)
        n_oob        = sum("OUT_OF_BOUNDS" in d for d in diagnoses)
        print(f"\n  Summary for {key}:")
        print(f"    STEP_MISS  (fine_delta too coarse): {n_step_miss}/{len(diagnoses)}")
        print(f"    NO_TERRAIN (DEM missing terrain):   {n_no_terrain}/{len(diagnoses)}")
        print(f"    OUT_OF_BOUNDS:                      {n_oob}/{len(diagnoses)}")

        if n_no_terrain > n_step_miss:
            print(f"  → PRIMARY CAUSE: DEM does not contain the terrain in these notches.")
            print(f"    Decreasing fine_delta will NOT fix this.")
        elif n_step_miss > 0:
            print(f"  → PRIMARY CAUSE: Ray stepping too coarse.")
            print(f"    Try smaller fine_delta (current: {args.fine_delta}m).")
        else:
            print(f"  → Cause unclear — check out-of-bounds rays.")

        # ── plots ──────────────────────────────────────────────────────────────
        n_profiles = len(profiles)
        fig = plt.figure(figsize=(6 * min(n_profiles, 4),
                                  4 * int(np.ceil(n_profiles / 4)) + 3))
        gs  = gridspec.GridSpec(
            int(np.ceil(n_profiles / 4)) + 1, min(n_profiles, 4),
            figure=fig, hspace=0.5
        )
        fig.suptitle(
            f"Camera {key} — DEM profile along wrong-sky rays\n"
            f"{src_str}  fine_delta={args.fine_delta}m\n"
            f"STEP_MISS={n_step_miss}  NO_TERRAIN={n_no_terrain}",
            fontsize=8, y=1.01
        )

        # Top row: agreement map with selected rays marked
        ax_map = fig.add_subplot(gs[0, :])
        # Agreement map background
        obs_sky = img.psky > 0.5
        # Recompute full-res sky_up from decimated
        from eigsep_terrain.utils import mask_near_horizon
        sky_full = np.zeros((Ny, Nx), dtype=bool)
        row_edges = np.append(xs, Ny)
        col_edges = np.append(ys, Nx)
        for i_r, r0 in enumerate(xs):
            for i_c, c0 in enumerate(ys):
                sky_full[r0:row_edges[i_r+1], c0:col_edges[i_c+1]] = \
                    model_sky_dec[i_r, i_c]

        wrong_sky_full = sky_full & ~obs_sky
        rgb = np.ones((*img.psky.shape, 3))
        rgb[ sky_full &  obs_sky] = [0.20, 0.78, 0.20]
        rgb[~sky_full & ~obs_sky] = [0.90, 0.90, 0.90]
        rgb[~sky_full &  obs_sky] = [0.88, 0.18, 0.18]
        rgb[ sky_full & ~obs_sky] = [0.18, 0.18, 0.88]
        ax_map.imshow(rgb, origin="lower", aspect="auto")

        # Mark selected rays
        colors_p = plt.cm.plasma(np.linspace(0.1, 0.9, len(profiles)))
        for pi, p in enumerate(profiles):
            ax_map.axvline(p["px_col"], color=colors_p[pi], lw=1.5,
                           alpha=0.8, label=f"ray {pi+1} col={p['px_col']}")
        ax_map.set_title(f"Agreement map — selected rays marked", fontsize=8)
        ax_map.legend(fontsize=6, ncol=len(profiles), loc="upper right")

        # Profile rows
        for pi, p in enumerate(profiles):
            row = pi // 4 + 1
            col = pi % 4
            ax = fig.add_subplot(gs[row, col])

            r   = p["r_vals"]
            ra  = p["ray_alts"]
            da  = p["dem_alts"]
            cl  = p["clearance"]
            hit = p["r_hit"]
            diag = diagnoses[pi]

            ax.plot(r, ra, color="steelblue", lw=1.5, label="ray altitude")
            ax.plot(r, da, color="saddlebrown", lw=1.5, label="DEM terrain")
            ax.fill_between(r, da, ra,
                            where=(ra < da), color="red", alpha=0.4,
                            label="ray below terrain")
            ax.fill_between(r, da, ra,
                            where=(ra >= da), color="lightblue", alpha=0.2)

            if np.isfinite(hit):
                ax.axvline(hit, color="green", lw=1.5, ls="--",
                           label=f"hit r={hit:.0f}m")

            if np.isfinite(p["min_clearance_r"]):
                ax.axvline(p["min_clearance_r"], color="orange", lw=1,
                           ls=":", label=f"min_clear={p['min_clearance']:.1f}m")

            ax.set_xlabel("distance along ray [m]", fontsize=7)
            ax.set_ylabel("altitude [m]", fontsize=7)
            ax.set_title(
                f"ray {pi+1}  col={p['px_col']}  row={p['px_row']}\n"
                f"{diag[:45]}",
                fontsize=6
            )
            ax.legend(fontsize=5, loc="upper right")
            ax.tick_params(labelsize=6)

        plt.tight_layout()
        outpath = os.path.join(outdir, f"dem_profile_{key}.png")
        fig.savefig(outpath, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: {outpath}")

    print(f"\nDone. Output in: {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())