#!/usr/bin/env python
"""
Fit the 7 geometric params (e, n, u, th, ph, ti, f) of a SINGLE HorizonImage
independently, using only the horizon-ray likelihood (no antenna term).

Reuses PositionSolver (img.py) for all u<->log_h conversion and likelihood
logic instead of reimplementing it: fit_imgs=[img], disable_ant=True.

Supports an arbitrary number of images via --meta-file (same JSON format
used by eigsep_terrain_pymc.py):
{
  "images": {
    "<key>": {"ant_px": [x, y], "e":.., "n":.., "u":.., "th":.., "ph":.., "ti":.., "f":..},
    ...
  },
  "platform": [e, n, u]   # optional, unused here (ant term disabled)
}
If --meta-file is omitted, falls back to the hardcoded 3-image
DEFAULT_META/DEFAULT_PRMS_U_BY_KEY below.

Outputs:
  trace_img{KEY}_seed{NNN}.nc         ArviZ InferenceData
  trace_img{KEY}_seed{NNN}_meta.json  metadata (args, seed, param names, priors)

Use plot_image_fit.py to load both files + the source image and produce
trace plots and a horizon-overlay figure.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER, dtype_r
from eigsep_terrain.img_defaults import load_defaults

BOX_SIZE = 0.3  # m, unused (ant term disabled) but required by PositionSolver

# Fallback (used only if --meta-file is not given) — loaded from defaults.json
# so this stays in sync with tune_image.py / plot_image_fit.py. Edit
# defaults.json, not this script, when a starting value changes.
DEFAULT_IMG_GLOB, DEFAULT_CACHE_FILE, DEFAULT_META, DEFAULT_PRMS_U_BY_KEY, IMG_KEYS = \
    load_defaults()


def load_meta_file(path: str):
    """Load a JSON meta file describing an arbitrary number of images.

    Returns (meta, keys, prms_u_by_key) where:
      meta          : {key: {"ant_px": (x, y)}}
      keys          : list of keys, in file order
      prms_u_by_key : {key: (e, n, u, th, ph, ti, f)}
    """
    with open(path) as f:
        raw = json.load(f)

    images = raw["images"]
    keys = list(images.keys())
    meta = {k: {"ant_px": tuple(images[k]["ant_px"])} for k in keys}
    prms_u_by_key = {
        k: tuple(float(images[k][p]) for p in PRM_ORDER) for k in keys
    }
    return meta, keys, prms_u_by_key


def find_img_file(key: str, img_glob: str) -> str:
    files = sorted(glob.glob(img_glob))
    matches = [f for f in files if os.path.basename(f).split("_")[-1].split(".")[0] == key]
    if not matches:
        raise FileNotFoundError(f"No file matching key {key!r} found via glob {img_glob!r}")
    if len(matches) > 1:
        print(f"WARNING: multiple files match key {key!r}, using first: {matches[0]}")
    return matches[0]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-file", default=None,
                    help="JSON file describing an arbitrary number of images "
                         "(key, ant_px, e/n/u/th/ph/ti/f). If omitted, falls "
                         "back to the hardcoded 3-image DEFAULT_META/"
                         "DEFAULT_PRMS_U_BY_KEY.")
    ap.add_argument("--which", type=int, required=True, nargs="+",
                    help="Which image(s) to fit, by index into the ordered "
                         "list of meta keys (0-based; matches tune_image.py's "
                         "--which). Pass multiple (e.g. --which 0 1 2) to fit "
                         "them in parallel, independent subprocesses.")
    ap.add_argument("--img-glob", default=DEFAULT_IMG_GLOB)
    ap.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--px-dist", type=int, default=30)
    ap.add_argument("--px-smooth", type=int, default=150)

    ap.add_argument("--n-rays", type=int, default=4000)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--fine-delta", type=float, default=0.25)

    ap.add_argument("--e", type=float, default=None)
    ap.add_argument("--n", type=float, default=None)
    ap.add_argument("--set-cam-height", action="store_true", default=True)
    ap.add_argument("--cam-height", type=float, default=1.6)

    ap.add_argument("--pos-err", type=float, default=30.0)
    ap.add_argument("--ang-err-deg", type=float, default=5.0)
    ap.add_argument("--f-err", type=float, default=0.1)
    ap.add_argument("--log-h-sigma", type=float, default=1.0)

    ap.add_argument("--map-file", default=None,
                    help="Optional map_estimate.py-style JSON with "
                         "param_names/map_params_h[/hess_stds] to seed priors.")

    ap.add_argument("--scaling", type=float, default=1e-2)
    ap.add_argument("--tune-interval", type=int, default=50)
    ap.add_argument("--jitter-scaling", type=float, default=1.0)

    ap.add_argument("--draws", type=int, default=4500)
    ap.add_argument("--tune", type=int, default=500)
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--cores", type=int, default=1)

    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    if args.meta_file is not None:
        meta_all, valid_keys, prms_u_by_key = load_meta_file(args.meta_file)
    else:
        meta_all = {k: dict(v) for k, v in DEFAULT_META.items()}
        valid_keys = IMG_KEYS
        prms_u_by_key = DEFAULT_PRMS_U_BY_KEY

    bad_idx = [w for w in args.which if w < 0 or w >= len(valid_keys)]
    if bad_idx:
        raise ValueError(
            f"--which index(es) {bad_idx} out of range "
            f"(0..{len(valid_keys)-1}, {len(valid_keys)} images available)."
        )
    which_keys = [valid_keys[w] for w in args.which]

    if len(which_keys) > 1:
        raw = argv if argv is not None else sys.argv[1:]
        procs = []
        for w in args.which:
            sub_argv = [a for a in raw if a not in ("--which", *map(str, args.which))]
            cmd = [sys.executable, __file__, "--which", str(w)] + sub_argv
            print(f"Launching: {' '.join(cmd)}")
            procs.append(subprocess.Popen(cmd))
        rc = 0
        for p in procs:
            p.wait()
            rc = rc or p.returncode
        return rc

    key = which_keys[0]

    seed = args.seed if args.seed is not None else int(np.random.randint(1000))
    np.random.seed(seed)

    img_file = find_img_file(key, args.img_glob)
    dem = DEM(cache_file=args.cache_file)

    meta = {k: dict(v) for k, v in meta_all.items()}
    img = HorizonImage(img_file, meta, px_smooth=args.px_smooth, px_dist=args.px_dist)
    if img.key != key:
        raise ValueError(f"Loaded image key {img.key!r} does not match requested {key!r}")

    stem = f"trace_img{key}_seed{seed:03d}"
    outfile = f"{stem}.nc"
    metafile = f"{stem}_meta.json"
    print(f"IMAGE FILE:  {img_file}")
    print(f"IMAGE KEY:   {key}")
    print(f"RANDOM SEED: {seed}")
    print(f"OUTFILE:     {outfile}")
    assert not os.path.exists(outfile), \
        f"{outfile} already exists; choose a different seed."

    prms_u = np.asarray(prms_u_by_key[key], dtype=dtype_r)
    e0 = args.e if args.e is not None else prms_u[0]
    n0 = args.n if args.n is not None else prms_u[1]
    prms_u[0], prms_u[1] = e0, n0
    if args.set_cam_height:
        prms_u[2] = float(dem.interp_alt(e0, n0)) + args.cam_height
        print(f"Camera height set to {args.cam_height}m above terrain "
              f"(u={prms_u[2]:.2f}).")
    img.set_prms(tuple(prms_u))

    # Dummy antenna prior: unused (ant term disabled below), just needs to be
    # a valid DEM location so PositionSolver's u<->log_h conversion doesn't error.
    ant_pos_prior = (e0, n0, float(dem.interp_alt(e0, n0)) + 1.0)

    ps = PositionSolver(ant_pos_prior, fit_imgs=[img], static_imgs=[],
                        n_rays=args.n_rays, dem=dem, box_size=BOX_SIZE)
    prms_h_padded = ps.prms_u_to_h(np.concatenate([prms_u, ant_pos_prior]))
    prms_h, ant_h_dummy = prms_h_padded[:7], prms_h_padded[-3:]

    sigmas = np.array([
        args.pos_err, args.pos_err, args.log_h_sigma,
        np.deg2rad(args.ang_err_deg), np.deg2rad(args.ang_err_deg),
        np.deg2rad(args.ang_err_deg), args.f_err * prms_u[6],
    ], dtype=dtype_r)

    map_file_meta = None
    if args.map_file is not None:
        print(f"\nLoading MAP from: {args.map_file}")
        with open(args.map_file) as _f:
            _map = json.load(_f)
        map_file_meta = {
            "map_file": args.map_file,
            "map_seed": _map.get("seed"),
            "map_method": _map.get("method"),
            "map_logL": _map.get("map_logL"),
        }
        _param_names = _map["param_names"]
        _map_h = _map["map_params_h"]
        for i, name in enumerate(_param_names):
            if name in _map_h:
                prms_h[i] = dtype_r(_map_h[name])
        if _map.get("hess_stds") is not None:
            _hess = _map["hess_stds"]
            for i, name in enumerate(_param_names):
                v = _hess.get(name)
                if v is not None and np.isfinite(v) and v > 0:
                    sigmas[i] = dtype_r(v)

    param_names = [f"{key}_e", f"{key}_n", f"{key}_log_h",
                   f"{key}_th", f"{key}_ph", f"{key}_ti", f"{key}_f"]

    @as_op(itypes=[pt.fvector], otypes=[pt.fscalar])
    def logp_op(theta_h_img):
        try:
            theta_h = np.concatenate([theta_h_img, ant_h_dummy]).astype(dtype_r)
            logL = ps.total_logL(
                theta_h, n_rays=args.n_rays, eps=args.eps,
                ant_weight=0.0, disable_ant=True, fine_delta=args.fine_delta,
            )
            return np.asarray(logL, dtype=dtype_r)
        except (ValueError, FloatingPointError):
            return np.asarray(-np.inf, dtype=dtype_r)

    with pm.Model() as model:
        mcmc_prms = [
            pm.Normal(name, mu=float(mu), sigma=float(sig))
            for name, mu, sig in zip(param_names, prms_h, sigmas)
        ]

        rng_pm = np.random.default_rng(seed)
        initvals = []
        for c in range(args.chains):
            jitter = rng_pm.normal(0.0, sigmas * args.jitter_scaling, size=prms_h.size)
            jittered = prms_h + jitter
            initvals.append({name: float(v) for name, v in zip(param_names, jittered)})

        theta = pt.cast(pt.stack(mcmc_prms), "float32")
        logL = logp_op(theta)
        pm.Potential("lik", logL)

        step = pm.DEMetropolisZ(
            S=sigmas, scaling=args.scaling, tune="scaling",
            tune_interval=args.tune_interval,
        )

        trace = pm.sample(
            draws=args.draws, tune=args.tune, chains=args.chains,
            step=step, initvals=initvals, cores=args.cores,
            random_seed=seed, progressbar=True,
        )

    az.to_netcdf(trace, outfile)

    accepted = float(trace.sample_stats.accepted.mean())
    try:
        tuned_scaling = float(step.scaling)
    except Exception:
        tuned_scaling = args.scaling

    param_summary = {}
    for i, name in enumerate(param_names):
        arr = trace.posterior[name].values.flatten()
        param_summary[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "prior_mu": float(prms_h[i]),
            "prior_sigma": float(sigmas[i]),
            "effective_step": float(tuned_scaling * sigmas[i]),
        }

    run_meta = {
        "img_key": key,
        "img_file": img_file,
        "seed": seed,
        "outfile": outfile,
        "param_names": param_names,
        "prm_order": list(PRM_ORDER),
        "accepted_mean": accepted,
        "map_file": map_file_meta,
        "sampling": {"draws": args.draws, "tune": args.tune,
                     "chains": args.chains, "cores": args.cores},
        "step": {"scaling": args.scaling, "tuned_scaling": tuned_scaling,
                 "tune_interval": args.tune_interval,
                 "jitter_scaling": args.jitter_scaling},
        "priors": {"pos_err": args.pos_err, "ang_err_deg": args.ang_err_deg,
                   "f_err": args.f_err, "log_h_sigma": args.log_h_sigma},
        "likelihood": {"eps": args.eps, "n_rays": args.n_rays,
                       "fine_delta": args.fine_delta},
        "image": {"px_dist": args.px_dist, "px_smooth": args.px_smooth,
                  "cam_height": args.cam_height,
                  "set_cam_height": args.set_cam_height},
        "param_summary": param_summary,
    }
    with open(metafile, "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Accepted step fraction = {accepted:.3f}")
    print(f"Tuned scaling          = {tuned_scaling:.6f}")
    print(f"Trace written to:        {outfile}")
    print(f"Metadata written to:     {metafile}")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())