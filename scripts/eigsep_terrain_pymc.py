#!/usr/bin/env python
"""
MCMC runner — sampling only, no plotting.

Outputs for each run:
  trace_seed{NNN}.nc        ArviZ InferenceData (posterior + sample_stats)
  trace_seed{NNN}_meta.json Sampling metadata (args, seed, acceptance, param names)

Use diagnostic_plot.py to load both files and produce figures.
Use map_estimate.py first to find the MAP, then pipe in via --map-file.
"""
import argparse
import glob
import json
import os

import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER, dtype_r
from eigsep_terrain.img_defaults import load_defaults

BOX_SIZE = 0.3  # m

# Fallback (used only if --meta-file is not given) — loaded from
# defaults.json so this stays in sync with tune_image.py / fit_image.py /
# plot_image_fit.py. Edit defaults.json, not this script, when a starting
# value changes.
DEFAULT_IMG_GLOB, DEFAULT_CACHE_FILE, DEFAULT_META, DEFAULT_PRMS_U_BY_KEY, IMG_KEYS = \
    load_defaults()


def _dummy_platform(img_keys, dem):
    """defaults.json has no tracked platform/antenna position. Build a
    dummy prior at the mean camera E/N, 1m above ground, so
    PositionSolver's u<->log_h conversion has a valid DEM location."""
    es = [DEFAULT_PRMS_U_BY_KEY[k][0] for k in img_keys]
    ns = [DEFAULT_PRMS_U_BY_KEY[k][1] for k in img_keys]
    e0, n0 = float(np.mean(es)), float(np.mean(ns))
    return np.array([e0, n0, float(dem.interp_alt(e0, n0)) + 1.0], dtype=dtype_r)


def load_meta_file(path: str):
    """Load a JSON meta file describing an arbitrary number of images.

    Expected format:
    {
      "images": {
        "<key>": {"ant_px": [x, y], "e":.., "n":.., "u":.., "th":.., "ph":.., "ti":.., "f":..},
        ...
      },
      "platform": [e, n, u]   # antenna prior position, optional
    }

    Returns (meta, prms, platform) where:
      meta   : {key: {"ant_px": (x, y)}}  (extra fields ignored downstream)
      prms   : flat np.array of len(images)*len(PRM_ORDER) in file/img_keys order
      platform: np.array of 3 platform (antenna) values
    """
    with open(path) as f:
        raw = json.load(f)

    images = raw["images"]
    keys = list(images.keys())

    meta = {k: {"ant_px": tuple(images[k]["ant_px"])} for k in keys}

    prms = []
    for k in keys:
        entry = images[k]
        prms.extend(float(entry[p]) for p in PRM_ORDER)

    platform = raw.get("platform", [0.0, 0.0, 0.0])

    return meta, keys, np.asarray(prms, dtype=dtype_r), np.asarray(platform, dtype=dtype_r)


def _apply_prms_to_dem_and_meta(
    dem: DEM,
    meta: dict,
    img_keys_in_fit_order: list[str],
    prms: np.ndarray,
    prm_len: int,
) -> None:
    nimgs = len(img_keys_in_fit_order)
    expected = nimgs * prm_len + 3
    if prms.size != expected:
        raise ValueError(
            f"prms has {prms.size} values; expected {expected} "
            f"({nimgs} images * {prm_len} params + 3 platform)."
        )
    platform = prms[-3:].astype(dtype_r)
    dem["platform"] = platform
    off = 0
    for key in img_keys_in_fit_order:
        chunk = prms[off: off + prm_len]
        off += prm_len
        meta[key]["prms"] = tuple(float(x) for x in chunk)
        dem[key] = np.asarray(chunk[:3], dtype=dtype_r)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
    ap.add_argument("--img-glob", default=DEFAULT_IMG_GLOB)
    ap.add_argument("--seed", type=int, default=None,
                    help="Defaults to random [0,999]")
    ap.add_argument("--meta-file", default=None,
                    help="JSON file describing an arbitrary number of images "
                         "(key, ant_px, e/n/u/th/ph/ti/f, platform prior). "
                         "If omitted, falls back to the hardcoded 3-image "
                         "DEFAULT_META/DEFAULT_PRMS.")

    # HorizonImage params
    ap.add_argument("--px-dist",   type=int, default=30)
    ap.add_argument("--px-smooth", type=int, default=150)

    # PositionSolver / ray tracing params
    ap.add_argument("--n-rays",      type=int,   default=4000)
    ap.add_argument("--ant-weight",  type=float, default=1.0,
                    help="Weight on ant_logL term (1.0=full, 0.0=disabled)")
    ap.add_argument("--disable-ant", action="store_true",
                    help="Completely disable ant_logL (overrides --ant-weight)")

    # logL op params
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--fine-delta", type=float, default=0.25,
                    help="Ray trace fine step size [m] (default 0.25). "
                         "Should be <= DEM grid spacing (0.5m).")

    # Camera height correction (applied per-image using each image's e/n
    # from --meta-file, or from DEFAULT_PRMS if --meta-file is omitted)
    ap.add_argument("--set-cam-height", action="store_true", default=True,
                    help="Override u from DEFAULT_PRMS with DEM + cam_height")
    ap.add_argument("--cam-height", type=float, default=1.6,
                    help="Camera height above ground [m] (default: 1.6)")

    # Prior sigmas
    ap.add_argument("--pos-err",     type=float, default=30.0,
                    help="Position prior sigma [m] (default: 30.0)")
    ap.add_argument("--ang-err-deg", type=float, default=5.0,
                    help="Angle prior sigma [deg] (default: 5.0)")
    ap.add_argument("--f-err",       type=float, default=0.1,
                    help="Focal-length prior sigma as fraction of f (default: 0.1)")
    ap.add_argument("--log-h-sigma", type=float, default=1.0,
                    help="log-height prior sigma (default: 1.0)")

    # MAP file — overrides prior centres (and optionally sigmas)
    ap.add_argument("--map-file", default=None,
                    help="Path to map_seed{NNN}.json from map_estimate.py. "
                         "Uses MAP values as prior centres. If the MAP JSON "
                         "contains hess_stds, those replace prior sigmas "
                         "(NaN entries fall back to --pos-err / --ang-err-deg etc).")

    # Step method params
    ap.add_argument("--scaling",        type=float, default=1e-2)
    ap.add_argument("--tune-interval",  type=int,   default=50)
    ap.add_argument("--jitter-scaling", type=float, default=1.0)

    # Sampling params
    ap.add_argument("--draws",  type=int, default=4500)
    ap.add_argument("--tune",   type=int, default=500)
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--cores",  type=int, default=1)

    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    # ── seed / outfile ────────────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else int(np.random.randint(1000))
    np.random.seed(seed)
    stem     = f"trace_seed{seed:03d}"
    outfile  = f"{stem}.nc"
    metafile = f"{stem}_meta.json"
    print(f"RANDOM SEED: {seed}")
    print(f"OUTFILE:     {outfile}")
    print(f"METAFILE:    {metafile}")
    assert not os.path.exists(outfile), \
        f"{outfile} already exists; choose a different seed or move the file."

    # ── load DEM ──────────────────────────────────────────────────────────────
    dem = DEM(cache_file=args.cache_file)

    # ── load images ───────────────────────────────────────────────────────────
    files = sorted(glob.glob(args.img_glob))
    if not files:
        raise FileNotFoundError(f"No images matched --img-glob: {args.img_glob}")

    prm_len = len(PRM_ORDER)

    if args.meta_file is not None:
        print(f"Loading meta/prms from: {args.meta_file}")
        meta, meta_img_keys, prms_imgs, platform = load_meta_file(args.meta_file)
        prms_u = np.concatenate([prms_imgs, platform]).astype(dtype_r)
    else:
        meta = {k: dict(v) for k, v in DEFAULT_META.items()}
        meta_img_keys = list(DEFAULT_META.keys())
        cam_prms = np.concatenate(
            [np.asarray(DEFAULT_PRMS_U_BY_KEY[k], dtype=dtype_r) for k in meta_img_keys]
        )
        platform = _dummy_platform(meta_img_keys, dem)
        prms_u = np.concatenate([cam_prms, platform]).astype(dtype_r)

    imgs = [HorizonImage(f, meta, px_smooth=args.px_smooth, px_dist=args.px_dist)
            for f in files]
    imgs = [img for img in imgs if img.key in meta]
    if not imgs:
        raise RuntimeError("No images matched keys in meta after loading HorizonImage objects.")

    fit_imgs, static_imgs = imgs, []
    img_keys = [img.key for img in fit_imgs]

    # Any loaded image not represented in prms_u/meta_img_keys can't be placed
    missing = [k for k in img_keys if k not in meta_img_keys]
    if missing:
        raise ValueError(
            f"Images found with keys not present in meta file: {missing}. "
            "Add entries for these keys to --meta-file."
        )
    # Order fit_imgs/img_keys to match prms_u ordering (meta_img_keys order),
    # restricted to images actually present on disk.
    img_keys = [k for k in meta_img_keys if k in img_keys]
    key_to_img = {img.key: img for img in fit_imgs}
    fit_imgs = [key_to_img[k] for k in img_keys]

    # If some meta entries have no corresponding file, drop their prms rows
    # so prms_u stays aligned with img_keys.
    if len(img_keys) != len(meta_img_keys):
        keep_idx = [meta_img_keys.index(k) for k in img_keys]
        rows = [prms_u[i*prm_len:(i+1)*prm_len] for i in keep_idx]
        platform_tail = prms_u[-3:]
        prms_u = np.concatenate(rows + [platform_tail]).astype(dtype_r)

    # Report original heights, then optionally overwrite u with DEM + cam_height
    for i, k in enumerate(img_keys):
        e_i = float(prms_u[i*prm_len + 0])
        n_i = float(prms_u[i*prm_len + 1])
        u_orig = float(prms_u[i*prm_len + 2])
        h_orig = u_orig - float(dem.interp_alt(e_i, n_i))
        u_new  = float(dem.interp_alt(e_i, n_i)) + args.cam_height
        print(f"{k}: orig u={u_orig:.2f}  orig h={h_orig:.2f}m  "
              f"-> new u={u_new:.2f}  new h={args.cam_height:.2f}m")
        if args.set_cam_height:
            prms_u[i*prm_len + 2] = u_new

    if args.set_cam_height:
        print(f"Camera heights set to {args.cam_height}m above terrain.")

    _apply_prms_to_dem_and_meta(
        dem=dem,
        meta=meta,
        img_keys_in_fit_order=img_keys,
        prms=prms_u,
        prm_len=len(PRM_ORDER),
    )

    # ── build solver ──────────────────────────────────────────────────────────
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

    # ── load MAP file if provided ─────────────────────────────────────────────
    map_file_meta = None
    if args.map_file is not None:
        print(f"\nLoading MAP from: {args.map_file}")
        with open(args.map_file) as _f:
            _map = json.load(_f)

        map_file_meta = {
            "map_file":         args.map_file,
            "map_seed":         _map.get("seed"),
            "map_method":       _map.get("method"),
            "map_logL":         _map.get("map_logL"),
            "map_converged":    _map.get("converged"),
            "map_n_restarts":   _map.get("n_restarts"),
            "logL_improvement": _map.get("logL_improvement"),
        }

        _param_names = _map["param_names"]
        _map_h       = _map["map_params_h"]

        # Override prms_h with MAP values as new prior centres
        for i, name in enumerate(_param_names):
            if name in _map_h:
                prms_h[i] = dtype_r(_map_h[name])
        ps.set_mcmc_prms(prms_h)

        # Override sigmas with Hessian stds where available and finite
        if _map.get("hess_stds") is not None:
            _hess      = _map["hess_stds"]
            new_sigmas = list(ps.sigmas)
            for i, name in enumerate(_param_names):
                v = _hess.get(name)
                if v is not None and np.isfinite(v) and v > 0:
                    new_sigmas[i] = dtype_r(v)
            ps.sigmas = new_sigmas
            print(f"  Prior centres : MAP values  "
                  f"(logL={_map['map_logL']:.2f}, method={_map['method']})")
            print(f"  Prior sigmas  : Hessian-derived "
                  f"(non-convex params fall back to CLI sigmas)")
        else:
            print(f"  Prior centres : MAP values  "
                  f"(logL={_map['map_logL']:.2f}, method={_map['method']})")
            print(f"  Prior sigmas  : CLI values (no Hessian in MAP file)")

    # ── likelihood op ─────────────────────────────────────────────────────────
    eps = dtype_r(args.eps)

    @as_op(itypes=[pt.fvector], otypes=[pt.fscalar])
    def total_logp_op(theta):
        try:
            return np.asarray(
                ps.total_logL(
                    theta=np.asarray(theta, dtype=dtype_r),
                    n_rays=args.n_rays,
                    eps=args.eps,
                    ant_weight=args.ant_weight,
                    disable_ant=args.disable_ant,
                    fine_delta=args.fine_delta,
                ),
                dtype=dtype_r,
            )
        except (ValueError, FloatingPointError):
            return np.asarray(-np.inf, dtype=dtype_r)

    # ── PyMC model ────────────────────────────────────────────────────────────
    with pm.Model() as model:
        mcmc_prms = ps.get_mcmc_prms()

        rng_pm = np.random.default_rng(seed)

        initvals = []
        for c in range(args.chains):
            jitter   = rng_pm.normal(0.0,
                                     np.asarray(ps.sigmas) * args.jitter_scaling,
                                     size=prms_h.size)
            jittered = prms_h + jitter
            ps.set_mcmc_prms(jittered)
            start_c  = ps.eval_cur_prms()
            initvals.append({p.name: v for p, v in zip(mcmc_prms, start_c)})

        theta = pt.cast(pt.stack(mcmc_prms), "float32")
        logL  = total_logp_op(theta)
        pm.Potential("lik", logL)

        step = pm.DEMetropolisZ(
            S=np.asarray(ps.sigmas, dtype=dtype_r),
            scaling=args.scaling,
            tune="scaling",
            tune_interval=args.tune_interval,
        )

        trace = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            step=step,
            initvals=initvals,
            cores=args.cores,
            random_seed=seed,
            progressbar=True,
        )

    # ── save trace ────────────────────────────────────────────────────────────
    az.to_netcdf(trace, outfile)

    # ── summary stats ─────────────────────────────────────────────────────────
    accepted = float(trace.sample_stats.accepted.mean())
    param_names = [p.name for p in mcmc_prms]

    try:
        tuned_scaling = float(step.scaling)
    except Exception:
        tuned_scaling = args.scaling

    param_summary = {}
    for i, name in enumerate(param_names):
        arr         = trace.posterior[name].values.flatten()
        prior_sigma = float(ps.sigmas[i])
        param_summary[name] = {
            "mean":           float(arr.mean()),
            "std":            float(arr.std()),
            "prior_mu":       float(prms_h[i]),
            "prior_sigma":    prior_sigma,
            "effective_step": float(tuned_scaling * prior_sigma),
        }

    # ── write metadata sidecar ────────────────────────────────────────────────
    run_meta = {
        "seed":          seed,
        "outfile":       outfile,
        "img_keys":      img_keys,
        "param_names":   param_names,
        "prm_order":     list(PRM_ORDER),
        "accepted_mean": accepted,
        "map_file":      map_file_meta,
        "sampling": {
            "draws":  args.draws,
            "tune":   args.tune,
            "chains": args.chains,
            "cores":  args.cores,
        },
        "step": {
            "scaling":        args.scaling,
            "tuned_scaling":  tuned_scaling,
            "tune_interval":  args.tune_interval,
            "jitter_scaling": args.jitter_scaling,
        },
        "priors": {
            "pos_err":     args.pos_err,
            "ang_err_deg": args.ang_err_deg,
            "f_err":       args.f_err,
            "log_h_sigma": args.log_h_sigma,
        },
        "likelihood": {
            "eps":         args.eps,
            "n_rays":      args.n_rays,
            "ant_weight":  args.ant_weight,
            "disable_ant": args.disable_ant,
            "fine_delta":  args.fine_delta,
        },
        "image": {
            "px_dist":        args.px_dist,
            "px_smooth":      args.px_smooth,
            "img_glob":       args.img_glob,
            "cam_height":     args.cam_height,
            "set_cam_height": args.set_cam_height,
        },
        "param_summary": param_summary,
    }

    with open(metafile, "w") as f:
        json.dump(run_meta, f, indent=2)

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Accepted step fraction = {accepted:.3f}")
    print(f"Tuned scaling          = {tuned_scaling:.6f}")
    print(f"Trace written to:        {outfile}")
    print(f"Metadata written to:     {metafile}")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())