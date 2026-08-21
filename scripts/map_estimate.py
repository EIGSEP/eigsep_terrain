#!/usr/bin/env python
"""
MAP estimation via Powell optimization.

Finds the maximum a posteriori parameter vector by minimizing -total_logL
using scipy.optimize.minimize (Powell method, gradient-free).

Outputs
-------
  map_seed{NNN}.json   MAP result + all metadata needed to initialize MCMC

The JSON is designed to be piped directly into eigsep_terrain_pymc.py via
--map-file, which uses the MAP values as prior centres and optionally
tightens the prior sigmas to a fraction of the optimizer's estimated
uncertainty (Hessian diagonal via finite differences).

Usage
-----
  map_estimate.py [options]
  eigsep_terrain_pymc.py --map-file map_seed042.json [other options]
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os as _os

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER, dtype_r
from eigsep_terrain.img_defaults import load_defaults

BOX_SIZE = 0.3  # m

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


# ── helpers ───────────────────────────────────────────────────────────────────

def _apply_prms_to_dem_and_meta(dem, meta, img_keys, prms, prm_len):
    nimgs = len(img_keys)
    expected = nimgs * prm_len + 3
    if prms.size != expected:
        raise ValueError(
            f"prms has {prms.size} values; expected {expected}."
        )
    dem["platform"] = prms[-3:].astype(dtype_r)
    off = 0
    for key in img_keys:
        chunk = prms[off: off + prm_len]
        off += prm_len
        meta[key]["prms"] = tuple(float(x) for x in chunk)
        dem[key] = np.asarray(chunk[:3], dtype=dtype_r)


def _json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _finite_diff_hessian_diag(f, x, eps_fd=1e-3):
    """
    Estimate the diagonal of the Hessian of f at x via central finite differences.
    Returns an array of second derivatives — positive values indicate a local minimum
    (the function curves upward), which is what we want at the MAP.
    Only the diagonal is computed (O(n) evaluations) to keep cost manageable.
    """
    n = len(x)
    f0 = f(x)
    diag = np.zeros(n, dtype=np.float64)
    for i in range(n):
        xp = x.copy(); xp[i] += eps_fd
        xm = x.copy(); xm[i] -= eps_fd
        fp = f(xp)
        fm = f(xm)
        diag[i] = (fp - 2 * f0 + fm) / eps_fd**2
    return diag


def _uncertainty_from_hessian_diag(hess_diag):
    """
    Approximate marginal std for each parameter from the diagonal of the
    Hessian of -logL (i.e. the observed Fisher information diagonal).
    std_i = 1 / sqrt(H_ii)  where H is the Hessian of -logL at the MAP.
    Returns NaN for entries where H_ii <= 0 (non-convex direction).
    """
    stds = np.where(hess_diag > 0, 1.0 / np.sqrt(hess_diag), np.nan)
    return stds


# ── argparse ──────────────────────────────────────────────────────────────────

def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
    ap.add_argument("--img-glob", default=DEFAULT_IMG_GLOB)
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for jitter (default: random [0,999])")

    # Image params
    ap.add_argument("--px-dist",   type=int, default=30)
    ap.add_argument("--px-smooth", type=int, default=150)

    # Likelihood params
    ap.add_argument("--n-rays", type=int,   default=4000)
    ap.add_argument("--eps",    type=float, default=1e-2)
    ap.add_argument("--fine-delta", type=float, default=0.25,
                    help="Ray trace fine step size [m] (default 0.25). "
                         "Should be <= DEM grid spacing (0.5m).")

    # Prior sigmas — used to build the log-prior term added to logL
    ap.add_argument("--pos-err",     type=float, default=30.0,
                    help="Position prior sigma [m]")
    ap.add_argument("--ang-err-deg", type=float, default=5.0,
                    help="Angle prior sigma [deg]")
    ap.add_argument("--f-err",       type=float, default=0.1,
                    help="Focal-length prior sigma as fraction of f")
    ap.add_argument("--log-h-sigma", type=float, default=1.0,
                    help="log-height prior sigma")

    # Camera height correction (per-image E/N now comes from defaults.json;
    # must match eigsep_terrain_pymc.py)
    ap.add_argument("--set-cam-height", action="store_true", default=True,
                    help="Override each image's u with DEM(e, n) + cam_height")
    ap.add_argument("--cam-height", type=float, default=1.6,
                    help="Camera height above ground [m] (default: 1.6)")

    # Optimizer params
    ap.add_argument("--method", default="Powell",
                    choices=["Powell", "Nelder-Mead", "COBYLA"],
                    help="Optimization method (default: Powell)")
    ap.add_argument("--maxiter", type=int, default=100000,
                    help="Max optimizer iterations (default: 100000)")
    ap.add_argument("--ftol", type=float, default=1e-6,
                    help="Function value tolerance (default: 1e-6)")
    ap.add_argument("--jitter-scaling", type=float, default=0.0,
                    help="Jitter init by this fraction of prior sigmas "
                         "(0 = start from per-key defaults.json values exactly)")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel worker processes (default: min(n_restarts, cpu_count)). "
                         "Set to 1 to disable parallelism.")
    ap.add_argument("--n-restarts", type=int, default=1,
                    help="Number of random restarts; best result is kept "
                         "(default: 1, use 3-5 to check for multiple modes)")

    # Hessian diagonal estimation for uncertainty
    ap.add_argument("--hessian", action="store_true",
                    help="Estimate parameter uncertainties from Hessian diagonal "
                         "at MAP (adds ~2*n_params extra evaluations)")
    ap.add_argument("--hessian-eps", type=float, default=1e-3,
                    help="Finite-difference step for Hessian estimation")

    ap.add_argument("--outfile", default=None,
                    help="Output JSON path (default: map_seed{NNN}.json)")
    return ap



def _run_one_restart(ra):
    """
    Top-level picklable worker: rebuilds full setup in subprocess and runs
    one Powell restart. ra is a plain dict of serialisable config values.
    """
    import numpy as _np
    import time as _time
    from scipy.optimize import minimize as _minimize
    from eigsep_terrain.marjum_dem import MarjumDEM as _DEM
    from eigsep_terrain.img import (HorizonImage as _HI,
                                    PositionSolver as _PS,
                                    PRM_ORDER as _PO, dtype_r as _dr)

    rng_w  = _np.random.default_rng(ra["seed_w"])
    dem_w  = _DEM(cache_file=ra["cache_file"])
    meta_w = {k: dict(v) for k, v in ra["default_meta"].items()}
    imgs_w = [_HI(f, meta_w, px_smooth=ra["px_smooth"], px_dist=ra["px_dist"])
              for f in ra["files"]]
    imgs_w = [img for img in imgs_w if img.key in meta_w]

    prms_u_w = _np.asarray(ra["prms_u"], dtype=_dr)
    dem_w["platform"] = prms_u_w[-3:].astype(_dr)
    off = 0
    for key in ra["img_keys"]:
        chunk = prms_u_w[off: off + ra["prm_len"]]
        off  += ra["prm_len"]
        meta_w[key]["prms"] = tuple(float(x) for x in chunk)
        dem_w[key] = _np.asarray(chunk[:3], dtype=_dr)

    ps_w = _PS(dem_w["platform"], imgs_w, [], ra["n_rays"], dem_w,
               box_size=ra["box_size"])
    prms_h_w = ps_w.prms_u_to_h(prms_u_w)
    ps_w.set_mcmc_prms(prms_h_w)
    ps_w.set_mcmc_sigmas(pos_err=ra["pos_err"], ang_err=ra["ang_err"],
                         f_err=ra["f_err"], log_h_sigma=ra["log_h_sigma"])
    for img_w in imgs_w:
        img_w.choose_pixels(N=ra["n_rays"], reset=True)

    eps_w    = _dr(ra["eps"])
    sigmas_w = _np.asarray(ps_w.sigmas, dtype=_np.float64)
    prms_h_f = prms_h_w.astype(_np.float64)

    def _nlp(theta):
        try:
            logL = float(ps_w.total_logL(
                _np.asarray(theta, dtype=_dr), eps=eps_w,
                fine_delta=ra["fine_delta"]))
        except Exception:
            return _np.inf
        if not _np.isfinite(logL):
            return _np.inf
        lp = -0.5 * _np.sum(((theta - prms_h_f) / sigmas_w) ** 2)
        return -(logL + lp)

    idx = ra["restart_idx"]
    if ra["jitter_scaling"] > 0 or idx > 0:
        scale  = max(ra["jitter_scaling"], 0.3) if idx > 0 else ra["jitter_scaling"]
        x0     = prms_h_f + rng_w.normal(0.0, sigmas_w * scale)
    else:
        x0 = prms_h_f.copy()

    init_logL = -_nlp(x0)
    t0     = _time.time()
    result = _minimize(_nlp, x0.astype(_np.float64), method=ra["method"],
                       options={"maxiter": ra["maxiter"],
                                "ftol":    ra["ftol"], "disp": False})
    elapsed   = _time.time() - t0
    map_logL  = -_nlp(result.x)

    return {"restart_idx": idx, "result": result,
            "init_logL": init_logL, "map_logL": map_logL,
            "elapsed": elapsed, "nfev": result.nfev,
            "success": result.success, "message": result.message}


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = build_argparser().parse_args(argv)

    seed = args.seed if args.seed is not None else int(np.random.randint(1000))
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    outfile = args.outfile or f"map_seed{seed:03d}.json"
    print(f"{'='*55}")
    print(f"  MAP ESTIMATION")
    print(f"{'='*55}")
    print(f"  seed        = {seed}")
    print(f"  method      = {args.method}")
    print(f"  n_restarts  = {args.n_restarts}")
    print(f"  n_rays      = {args.n_rays}")
    print(f"  eps         = {args.eps}")
    print(f"  pos_err     = {args.pos_err} m")
    print(f"  ang_err     = {args.ang_err_deg} deg")
    print(f"  log_h_sigma = {args.log_h_sigma}")
    print(f"  cam_height  = {args.cam_height}m")
    print(f"  fine_delta  = {args.fine_delta}m")
    print(f"  outfile     = {outfile}")
    print()

    # ── setup ─────────────────────────────────────────────────────────────────
    dem = DEM(cache_file=args.cache_file)

    files = sorted(glob.glob(args.img_glob))
    if not files:
        raise FileNotFoundError(f"No images matched: {args.img_glob}")

    meta = {k: dict(v) for k, v in DEFAULT_META.items()}
    imgs = [HorizonImage(f, meta, px_smooth=args.px_smooth, px_dist=args.px_dist)
            for f in files]
    imgs = [img for img in imgs if img.key in meta]
    if not imgs:
        raise RuntimeError("No images matched keys in meta.")

    fit_imgs, static_imgs = imgs, []
    img_keys = [img.key for img in fit_imgs]

    cam_prms_u = _build_prms_u(img_keys, dem, set_cam_height=args.set_cam_height,
                               cam_height=args.cam_height)
    if args.set_cam_height:
        print(f"Camera heights set to {args.cam_height}m above terrain.")
    ant_pos_prior = _dummy_ant_pos_prior(cam_prms_u, dem)
    prms_u = np.concatenate([cam_prms_u, ant_pos_prior]).astype(dtype_r)

    _apply_prms_to_dem_and_meta(dem, meta, img_keys, prms_u, len(PRM_ORDER))

    ps = PositionSolver(
        dem["platform"], fit_imgs, static_imgs,
        args.n_rays, dem, box_size=BOX_SIZE,
    )
    prms_h = ps.prms_u_to_h(prms_u)
    ps.set_mcmc_prms(prms_h)
    ps.set_mcmc_sigmas(
        pos_err=args.pos_err,
        ang_err=np.deg2rad(args.ang_err_deg),
        f_err=args.f_err,
        log_h_sigma=args.log_h_sigma,
    )

    # Fix pixel sample for deterministic likelihood across optimizer calls
    for img in fit_imgs:
        img.choose_pixels(N=args.n_rays, reset=True)

    eps = dtype_r(args.eps)
    sigmas = np.asarray(ps.sigmas, dtype=np.float64)
    param_names = [
        f"{img.key}_log_h" if k == "u" else f"{img.key}_{k}"
        for img in fit_imgs for k in PRM_ORDER
    ] + ["ant_e", "ant_n", "ant_log_h"]

    # ── objective: -logL (includes prior via ps.total_logL which uses
    #    the Normal priors implicitly through get_mcmc_prms / set_mcmc_prms)
    #    We add the log-prior term explicitly here so the optimizer minimizes
    #    the full MAP objective = -logL - log_prior
    # ─────────────────────────────────────────────────────────────────────────
    n_evals = [0]

    def neg_log_posterior(theta):
        n_evals[0] += 1
        try:
            logL = float(ps.total_logL(
                np.asarray(theta, dtype=dtype_r), eps=eps,
                fine_delta=args.fine_delta,
            ))
        except Exception:
            return np.inf
        if not np.isfinite(logL):
            return np.inf
        # Gaussian log-prior: -0.5 * sum((theta - mu)^2 / sigma^2)
        # mu = prms_h (the prior centres set by set_mcmc_sigmas)
        log_prior = -0.5 * np.sum(((theta - prms_h) / sigmas) ** 2)
        return -(logL + log_prior)

    # ── run optimizer (parallel restarts) ────────────────────────────────────
    n_workers = args.workers if args.workers is not None else min(
        args.n_restarts, _os.cpu_count() or 1
    )
    print(f"  Running {args.n_restarts} restarts across {n_workers} workers...")

    restart_seeds = [int(rng.integers(0, 2**31)) for _ in range(args.n_restarts)]
    cfgs = [
        dict(restart_idx=i, seed_w=restart_seeds[i],
             cache_file=args.cache_file, default_meta=DEFAULT_META,
             files=files, img_keys=img_keys,
             prms_u=prms_u.tolist(), prm_len=len(PRM_ORDER),
             box_size=BOX_SIZE, n_rays=args.n_rays,
             px_smooth=args.px_smooth, px_dist=args.px_dist,
             pos_err=args.pos_err, ang_err=float(np.deg2rad(args.ang_err_deg)),
             f_err=args.f_err, log_h_sigma=args.log_h_sigma,
             eps=float(args.eps), fine_delta=float(args.fine_delta),
             jitter_scaling=args.jitter_scaling, method=args.method,
             maxiter=args.maxiter, ftol=args.ftol)
        for i in range(args.n_restarts)
    ]

    best_result: OptimizeResult | None = None
    best_val   = np.inf
    total_nfev = 0

    def _report(out):
        print(f"  Restart {out['restart_idx']+1}/{args.n_restarts}  "
              f"init_logL={out['init_logL']:.2f}  "
              f"converged={out['success']}  nfev={out['nfev']}  "
              f"map_logL={out['map_logL']:.2f}  time={out['elapsed']:.1f}s",
              flush=True)
        if not out["success"]:
            print(f"    message: {out['message']}")

    if n_workers == 1:
        for cfg in cfgs:
            out = _run_one_restart(cfg)
            total_nfev += out["nfev"]
            _report(out)
            if out["result"].fun < best_val:
                best_val, best_result = out["result"].fun, out["result"]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_run_one_restart, cfg): cfg["restart_idx"]
                    for cfg in cfgs}
            for fut in as_completed(futs):
                try:
                    out = fut.result()
                except Exception as exc:
                    print(f"  Restart {futs[fut]+1} FAILED: {exc}", flush=True)
                    continue
                total_nfev += out["nfev"]
                _report(out)
                if out["result"].fun < best_val:
                    best_val, best_result = out["result"].fun, out["result"]

    print(f"\n  Best MAP logL = {-best_val:.2f}  "
          f"(total evals across restarts: {total_nfev})")

    map_theta = best_result.x

    # ── convert MAP back to absolute-u representation for readability ─────────
    ps.set_mcmc_prms(map_theta)
    map_theta_u = ps._convert_uh_prms(map_theta.copy(), sign=1)

    # Per-param MAP values in both representations
    map_params_h = {name: float(map_theta[i])
                    for i, name in enumerate(param_names)}
    map_params_u = {}
    for i, img in enumerate(fit_imgs):
        for j, k in enumerate(PRM_ORDER):
            name = f"{img.key}_log_h" if k == "u" else f"{img.key}_{k}"
            map_params_u[name] = float(map_theta_u[i * len(PRM_ORDER) + j])
    map_params_u["ant_e"] = float(map_theta_u[-3])
    map_params_u["ant_n"] = float(map_theta_u[-2])
    map_params_u["ant_u"] = float(map_theta_u[-1])

    # ── optional Hessian diagonal ─────────────────────────────────────────────
    hess_stds = None
    hess_diag = None
    if args.hessian:
        print(f"\n  Estimating Hessian diagonal ({2 * len(map_theta)} evals)...")
        hess_diag = _finite_diff_hessian_diag(
            neg_log_posterior, map_theta, eps_fd=args.hessian_eps
        )
        hess_stds = _uncertainty_from_hessian_diag(hess_diag)
        n_nonconvex = int(np.sum(~np.isfinite(hess_stds)))
        print(f"  Non-convex directions: {n_nonconvex}/{len(map_theta)}")
        if n_nonconvex > 0:
            nc_names = [param_names[i] for i in range(len(map_theta))
                        if not np.isfinite(hess_stds[i])]
            print(f"  Non-convex params: {nc_names}")

    # ── print MAP summary ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  MAP PARAMETER SUMMARY")
    print(f"{'='*55}")
    col_w = max(len(n) for n in param_names) + 2
    print(f"  {'param':<{col_w}}  {'MAP (log_h repr)':>18}  {'prior_mu':>10}  {'prior_sigma':>12}", end="")
    if hess_stds is not None:
        print(f"  {'hess_std':>10}", end="")
    print()
    for i, name in enumerate(param_names):
        val   = map_theta[i]
        mu    = prms_h[i]
        sigma = sigmas[i]
        line  = f"  {name:<{col_w}}  {val:>18.4f}  {mu:>10.4f}  {sigma:>12.4f}"
        if hess_stds is not None:
            hs = hess_stds[i]
            line += f"  {hs:>10.4f}" if np.isfinite(hs) else f"  {'(non-cvx)':>10}"
        print(line)
    print(f"{'='*55}")

    # ── write JSON ────────────────────────────────────────────────────────────
    # The JSON is structured so eigsep_terrain_pymc.py can load it directly:
    #   --map-file map_seed042.json
    # The MCMC script will use map_params_h as prior centres (mu) and
    # hess_stds (if present) as prior sigmas, falling back to the original
    # sigmas if Hessian estimation failed for a given parameter.

    out = {
        "seed": seed,
        "outfile": outfile,
        "method": args.method,
        "converged": bool(best_result.success),
        "n_evals": n_evals[0],
        "n_restarts": args.n_restarts,
        "map_logL": float(-best_val),
        "init_logL": float(-neg_log_posterior(prms_h)),
        "logL_improvement": float(-best_val - (-neg_log_posterior(prms_h))),
        "param_names": param_names,
        "prm_order": list(PRM_ORDER),
        "img_keys": img_keys,

        # Prior used during optimization — MCMC should use same values
        "fine_delta": args.fine_delta,
        "priors": {
            "pos_err": args.pos_err,
            "ang_err_deg": args.ang_err_deg,
            "f_err": args.f_err,
            "log_h_sigma": args.log_h_sigma,
        },

        # MAP values in log_h representation — use as MCMC prior centres
        "map_params_h": map_params_h,

        # MAP values in absolute-u representation — for human inspection
        "map_params_u": map_params_u,

        # Prior centres and sigmas used during optimization
        "prior_mu": {name: float(prms_h[i])
                     for i, name in enumerate(param_names)},
        "prior_sigma": {name: float(sigmas[i])
                        for i, name in enumerate(param_names)},

        # Hessian-derived uncertainties (if computed) — use as MCMC sigmas
        # NaN entries mean that direction was non-convex at the MAP;
        # fall back to prior_sigma for those parameters in the MCMC
        "hess_stds": ({name: float(hess_stds[i])
                       for i, name in enumerate(param_names)}
                      if hess_stds is not None else None),

        # Run config
        "args": vars(args),
    }

    with open(outfile, "w") as f:
        json.dump(out, f, indent=2, default=_json_safe)

    print(f"\n  MAP result written to: {outfile}")
    print(f"  Pass to MCMC with:  eigsep_terrain_pymc.py --map-file {outfile}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())