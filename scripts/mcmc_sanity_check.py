#!/usr/bin/env python
"""
Pre-flight sanity checks for MCMC runs produced by eigsep_terrain_pymc.py.

Runs five fast checks without doing any MCMC sampling, prints a
pass/warn/fail report, and writes a JSON sidecar so results are
machine-readable.

Checks
------
  1. logL at init          Is the starting logL finite and reasonable?
  2. DEM bounds            Is each camera and the antenna inside the DEM?
  3. Prior predictive      Does the init sit in a plausible region of prior
                           logL, or is it implausibly good/bad?
  4. Scaling probe         Estimate acceptance rate from a tiny MH probe
                           (~200 proposals) to see if --scaling is in range.
  5. Pixel stability       How much does logL vary across independent pixel
                           draws at the same parameters? (stochasticity check)

Usage
-----
  mcmc_sanity_check [options]

Output
------
  sanity_<timestamp>.json   Machine-readable results (pass/warn/fail + values)
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER, dtype_r
from eigsep_terrain.img_defaults import load_defaults

BOX_SIZE = 0.3  # m

# Generic, arbitrary-image-count defaults loaded from defaults.json (same
# file used by tune_image.py / fit_image.py / plot_image_fit.py). Edit
# defaults.json, not this script, when a starting value changes.
DEFAULT_IMG_GLOB, DEFAULT_CACHE_FILE, DEFAULT_META, DEFAULT_PRMS_U_BY_KEY, IMG_KEYS = \
    load_defaults()


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

# ── thresholds ────────────────────────────────────────────────────────────────
ACCEPT_WARN_LOW  = 0.10   # below this → scaling too large
ACCEPT_WARN_HIGH = 0.60   # above this → scaling too small
ACCEPT_OK_LOW    = 0.20   # ideal lower bound
ACCEPT_OK_HIGH   = 0.40   # ideal upper bound
PRIOR_PRED_WARN_SIGMA = 2.0  # init logL more than this many σ above prior mean → suspicious


# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

def _status(condition_pass, condition_warn=True):
    """Return PASS / WARN / FAIL based on two boolean conditions."""
    if condition_pass:
        return PASS
    if condition_warn:
        return WARN
    return FAIL


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


def _safe_logL(ps, theta, eps, n_rays=None):
    """Evaluate total_logL, returning -inf on any exception."""
    try:
        return float(ps.total_logL(
            np.asarray(theta, dtype=dtype_r),
            n_rays=n_rays,
            eps=eps,
        ))
    except Exception:
        return -np.inf


# ── checks ────────────────────────────────────────────────────────────────────

def check_logL_at_init(ps, prms_h, eps, n_rays):
    """Check 1: is the logL at the initial parameters finite?
    
    Only claim: finite logL is a necessary condition for the chain to move.
    We do not threshold on magnitude — that would require knowing the posterior
    scale, which we don't have at this stage.
    """
    print("\n[1/5] logL at init ...")
    logL = _safe_logL(ps, prms_h, eps, n_rays=n_rays)
    finite = bool(np.isfinite(logL))
    status = PASS if finite else FAIL
    result = {
        "status": status,
        "logL_init": logL,
        "finite": finite,
    }
    _print_result(status, f"logL = {logL:.2f}",
                  fail_msg="logL is -inf — init is outside the support. Check DEM bounds and params.")
    return result


def check_dem_bounds(ps, dem):
    """Check 2: are all cameras and the antenna inside the DEM extent?

    Checks three things per point:
      - E within DEM grid (horizontal)
      - N within DEM grid (horizontal)
      - u (altitude) is above the terrain surface and within a plausible
        height above ground (0–200 m). A camera with u set below the DEM
        surface or implausibly high will cause interp_alt to return wrong
        values silently.

    Trustworthiness: high. This is pure geometry with no probabilistic
    reasoning. The altitude check uses dem.interp_alt which is the same
    function used during ray tracing, so any discrepancy found here will
    also affect logL.
    """
    print("\n[2/5] DEM bounds ...")
    E, N = dem.get_en()
    e_range = (float(E[0]), float(E[-1]))
    n_range = (float(N[0]), float(N[-1]))
    MAX_HEIGHT_ABOVE_GROUND = 10.0  # m — above this is implausible for a camera

    def _check_point(e, n, u, label):
        in_e = e_range[0] <= e <= e_range[1]
        in_n = n_range[0] <= n <= n_range[1]
        horiz_ok = in_e and in_n
        height_ok = False
        terrain_u = None
        height_above = None
        if horiz_ok:
            try:
                terrain_u = float(dem.interp_alt(e, n))
                height_above = u - terrain_u
                height_ok = 0 < height_above < MAX_HEIGHT_ABOVE_GROUND
            except Exception:
                height_ok = False
        ok = horiz_ok and height_ok
        marker = "  ok" if ok else "FAIL" if not horiz_ok else "WARN"
        h_str = f"  h={height_above:.1f}m above terrain" if height_above is not None else "  (terrain lookup failed)"
        print(f"    [{marker}] {label}: E={e:.1f}  N={n:.1f}  u={u:.1f}{h_str}")
        if not in_e or not in_n:
            print(f"           Outside DEM footprint "
                  f"E={e_range[0]:.0f}..{e_range[1]:.0f}  N={n_range[0]:.0f}..{n_range[1]:.0f}")
        if horiz_ok and not height_ok:
            if height_above is not None and height_above <= 0:
                print(f"           u={u:.1f} is at or below terrain surface ({terrain_u:.1f}m) — check init params.")
            elif height_above is not None:
                print(f"           h={height_above:.1f}m above ground exceeds {MAX_HEIGHT_ABOVE_GROUND}m — check init params.")
        return {"e": e, "n": n, "u": u, "in_bounds": ok,
                "horiz_ok": horiz_ok, "height_ok": height_ok,
                "terrain_u": terrain_u, "height_above": height_above}

    results = {}
    all_ok = True
    for img in ps.fit_imgs:
        r = _check_point(img.prms["e"], img.prms["n"], img.prms["u"],
                         f"camera {img.key}")
        results[img.key] = r
        all_ok = all_ok and r["in_bounds"]

    ae, an, au = ps.ant_pos[0], ps.ant_pos[1], ps.ant_pos[2]
    r = _check_point(ae, an, au, "antenna     ")
    results["antenna"] = r
    all_ok = all_ok and r["in_bounds"]

    status = PASS if all_ok else FAIL
    return {"status": status, "points": results,
            "dem_e_range": e_range, "dem_n_range": n_range}


def check_prior_predictive(ps, prms_h, eps, n_rays, n_samples=80):
    """
    Check 3: sample from the prior and evaluate logL.

    Prior draws are N(prms_h, sigmas) — centred on the init because the init
    IS the prior mean (DEFAULT_PRMS). The z-score of the init relative to
    prior draws is therefore not informative (it will always be near 0).

    What IS informative:
      - frac_finite: if many prior draws give -inf, the priors are too wide
        and most of parameter space is degenerate.
      - logL_std: a huge spread (as seen in your run: std ~6M) means the
        prior is very uninformative — the sampler will wander a vast space.
        Consider tightening pos_err / ang_err in set_mcmc_sigmas.
      - logL_std / |init_logL|: how large is prior logL variation relative
        to the signal at the init? If this ratio >> 1, prior draws are wildly
        off and the sampler has almost no gradient to follow.
    """
    print(f"\n[3/5] Prior predictive ({n_samples} samples) ...")
    sigmas = np.asarray(ps.sigmas)
    rng = np.random.default_rng(0)
    prior_logLs = []
    for _ in range(n_samples):
        theta = prms_h + rng.normal(0.0, sigmas)
        lL = _safe_logL(ps, theta, eps, n_rays=min(n_rays, 500))
        if np.isfinite(lL):
            prior_logLs.append(lL)

    init_logL = _safe_logL(ps, prms_h, eps, n_rays=min(n_rays, 500))
    n_finite = len(prior_logLs)
    frac_finite = n_finite / n_samples

    if n_finite < 5:
        status = FAIL
        print(f"    [FAIL] only {n_finite}/{n_samples} prior samples had finite logL.")
        print(f"           Priors are too wide — most of parameter space is degenerate.")
        return {"status": status, "n_finite": n_finite, "n_samples": n_samples,
                "init_logL": init_logL, "frac_finite": frac_finite}

    mu    = float(np.mean(prior_logLs))
    std   = float(np.std(prior_logLs))
    # Relative spread: how many times larger is the prior logL std than the
    # init logL magnitude? >>1 means priors are extremely uninformative.
    rel_spread = std / abs(init_logL) if abs(init_logL) > 0 else float("inf")

    # Thresholds: rel_spread > 10 is a strong warning (prior std is 10x the
    # signal), > 100 is a fail (prior is essentially flat).
    ok   = frac_finite > 0.8 and rel_spread < 10.0
    warn = frac_finite > 0.5 and rel_spread < 100.0
    status = _status(ok, warn)

    print(f"    prior logL: mean={mu:.1f}  std={std:.1f}  "
          f"finite={n_finite}/{n_samples} ({100*frac_finite:.0f}%)")
    print(f"    init logL:  {init_logL:.1f}")
    print(f"    prior std / |init logL| = {rel_spread:.1f}x")

    if frac_finite < 0.8:
        print(f"    [{status}] {100*(1-frac_finite):.0f}% of prior draws are degenerate (-inf).")
        print(f"           Consider tightening pos_err or ang_err in set_mcmc_sigmas.")
    if rel_spread > 10:
        print(f"    [{status}] Prior logL std is {rel_spread:.0f}x the init logL magnitude.")
        print(f"           Priors are very uninformative — sampler will wander a vast space.")
        print(f"           Consider tightening set_mcmc_sigmas (pos_err, ang_err, log_h_sigma).")
    if ok:
        print(f"    [PASS]")

    return {
        "status": status,
        "prior_logL_mean": mu,
        "prior_logL_std": std,
        "prior_logL_rel_spread": rel_spread,
        "init_logL": init_logL,
        "frac_finite": frac_finite,
        "n_finite": n_finite,
        "n_samples": n_samples,
    }


def check_scaling_probe(ps, prms_h, eps, n_rays, scaling, n_proposals=200):
    """
    Check 4: run a tiny vanilla MH probe (no tuning) and measure acceptance.

    NOTE: DEMetropolisZ uses differential evolution proposals from chain
    history, not the diagonal Gaussian used here. This probe is therefore
    a proxy — it tells you if step sizes are grossly wrong, but the true
    DEMetropolisZ acceptance rate will differ. Treat PASS as "not obviously
    broken", not as a guarantee of good mixing.
    """
    print(f"\n[4/5] Scaling probe ({n_proposals} proposals, scaling={scaling}) ...")
    sigmas = np.asarray(ps.sigmas, dtype=dtype_r)
    rng = np.random.default_rng(1)

    current = prms_h.copy()
    current_logL = _safe_logL(ps, current, eps, n_rays=n_rays)
    if not np.isfinite(current_logL):
        status = FAIL
        print(f"    [FAIL] init logL is not finite — cannot probe scaling.")
        return {"status": status, "accepted": None, "scaling": scaling}

    n_accepted = 0
    for _ in range(n_proposals):
        proposal = current + rng.normal(0.0, sigmas * scaling)
        prop_logL = _safe_logL(ps, proposal, eps, n_rays=n_rays)
        if not np.isfinite(prop_logL):
            continue  # reject -inf or +inf proposals outright
        log_alpha = prop_logL - current_logL
        if np.log(rng.uniform()) < log_alpha:
            current = proposal
            current_logL = prop_logL
            n_accepted += 1

    accept_rate = n_accepted / n_proposals
    ok   = ACCEPT_OK_LOW  <= accept_rate <= ACCEPT_OK_HIGH
    warn = ACCEPT_WARN_LOW <= accept_rate <= ACCEPT_WARN_HIGH

    status = _status(ok, warn)
    msg = f"acceptance = {accept_rate:.3f}"
    if accept_rate < ACCEPT_WARN_LOW:
        _print_result(status, msg,
            fail_msg=f"Too low (<{ACCEPT_WARN_LOW}). Decrease --scaling.")
    elif accept_rate > ACCEPT_WARN_HIGH:
        _print_result(status, msg,
            warn_msg=f"Too high (>{ACCEPT_WARN_HIGH}). Increase --scaling.")
    else:
        _print_result(status, msg,
            warn_msg=f"Outside ideal {ACCEPT_OK_LOW}–{ACCEPT_OK_HIGH} range but acceptable.")

    return {
        "status": status,
        "accept_rate": accept_rate,
        "n_proposals": n_proposals,
        "scaling": scaling,
        "ideal_range": [ACCEPT_OK_LOW, ACCEPT_OK_HIGH],
    }


def check_pixel_stability(ps, prms_h, eps, n_rays, n_repeats=10):
    """
    Check 5: evaluate logL multiple times at the same parameters with
    different pixel draws, and measure the std of logL across draws.

    The relevant quantity for MH correctness is std in logL units: the
    acceptance ratio exp(logL_prop - logL_cur) fluctuates by exp(std) due
    to pixel noise alone. std > 10 means the acceptance ratio is off by up
    to exp(10) ~ 22000x from pixel noise, which severely corrupts sampling.

    Implementation note: pixel refresh is done by setting img._px_choice = None
    before each call, which forces choose_pixels() to redraw. This relies on a
    private attribute and would break if the internal caching logic changes.
    """
    print(f"\n[5/5] Pixel stability ({n_repeats} draws at same params) ...")
    logLs = []
    for _ in range(n_repeats):
        # Force a fresh pixel draw each time
        for img in ps.fit_imgs:
            img._px_choice = None
        lL = _safe_logL(ps, prms_h, eps, n_rays=n_rays)
        if np.isfinite(lL):
            logLs.append(lL)

    # Restore a fixed pixel draw for the rest of the script
    for img in ps.fit_imgs:
        img._px_choice = None
        img.choose_pixels(N=n_rays, reset=True)

    if len(logLs) < 3:
        print("    [FAIL] fewer than 3 finite logL values — cannot assess stability.")
        return {"status": FAIL, "logLs": logLs}

    mu  = float(np.mean(logLs))
    std = float(np.std(logLs))
    # Use absolute std in logL units, not CV. CV (std/|mean|) is misleading
    # because logL magnitude has no natural scale — a std of 20 on logL=-200
    # and logL=-200000 would give very different CVs but the same effect on
    # MH acceptance. What matters for MH correctness is std in logL units:
    # if std >> 1, the acceptance ratio exp(logL_prop - logL_cur) fluctuates
    # by exp(std) due to pixel noise alone, invalidating the criterion.
    # Warn at std > 10 (exp(10) ~ 22000x noise on acceptance ratio).
    # Fail at std > 50.
    WARN_STD = 10.0
    FAIL_STD = 50.0
    ok   = std < WARN_STD
    warn = std < FAIL_STD
    status = _status(ok, warn)

    print(f"    logL across {len(logLs)} pixel draws:  "
          f"mean={mu:.2f}  std={std:.2f} (threshold: warn>{WARN_STD}, fail>{FAIL_STD})")
    if not ok:
        advice = (f"Pixel noise std={std:.1f} logL units. "
                  f"Increase --n-rays to reduce stochasticity, or fix the pixel "
                  f"sample before sampling (img.choose_pixels + reset=False).")
        print(f"    [{status}] {advice}")
    else:
        print(f"    [PASS]")

    return {
        "status": status,
        "logL_mean": mu,
        "logL_std": std,
        "warn_std_threshold": WARN_STD,
        "fail_std_threshold": FAIL_STD,
        "n_repeats": len(logLs),
        "logLs": logLs,
    }


# ── formatting ────────────────────────────────────────────────────────────────

def _print_result(status, value_str, warn_msg="", fail_msg=""):
    print(f"    [{status}] {value_str}")
    if status == WARN and warn_msg:
        print(f"           {warn_msg}")
    if status == FAIL and fail_msg:
        print(f"           {fail_msg}")


def _print_summary(results):
    statuses = [r["status"] for r in results.values()]
    n_pass = statuses.count(PASS)
    n_warn = statuses.count(WARN)
    n_fail = statuses.count(FAIL)
    print("\n" + "=" * 55)
    print(f"  SANITY CHECK SUMMARY: "
          f"{n_pass} PASS  {n_warn} WARN  {n_fail} FAIL")
    print("=" * 55)
    for name, r in results.items():
        print(f"  {r['status']:4s}  {name}")
    print("=" * 55)
    if n_fail > 0:
        print("  ✗ Fix FAIL items before running MCMC.")
    elif n_warn > 0:
        print("  ⚠ Address WARN items for best mixing.")
    else:
        print("  ✓ All checks passed — good to run MCMC.")
    print()


# ── argparse / main ───────────────────────────────────────────────────────────

def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
    ap.add_argument("--img-glob", default=DEFAULT_IMG_GLOB)

    # Match the MCMC script args so you can paste the same flags
    ap.add_argument("--px-dist",   type=int,   default=30)
    ap.add_argument("--px-smooth", type=int,   default=150)
    ap.add_argument("--n-rays",    type=int,   default=4000)
    ap.add_argument("--eps",       type=float, default=1e-2)
    ap.add_argument("--scaling",   type=float, default=1e-2,
                    help="DEMetropolisZ scaling to probe (same as MCMC --scaling)")
    ap.add_argument("--set-cam-height", action="store_true", default=True,
                    help="Override each image's u with DEM(e, n) + cam_height")
    ap.add_argument("--cam-height", type=float, default=1.6,
                    help="Camera height above ground [m] (default: 1.6)")

    # Prior sigmas — use the same values as the MCMC run
    ap.add_argument("--pos-err",      type=float, default=30.0,
                    help="Position prior sigma in metres (default: 30.0)")
    ap.add_argument("--ang-err-deg",  type=float, default=5.0,
                    help="Angle prior sigma in degrees (default: 5.0)")
    ap.add_argument("--f-err",        type=float, default=0.1,
                    help="Focal-length prior sigma as fraction of f (default: 0.1)")
    ap.add_argument("--log-h-sigma",  type=float, default=1.0,
                    help="log-height prior sigma (default: 1.0)")

    # Probe settings
    ap.add_argument("--prior-samples", type=int, default=80,
                    help="Number of prior draws for prior predictive check (default: 80)")
    ap.add_argument("--probe-proposals", type=int, default=200,
                    help="Number of MH proposals for scaling probe (default: 200)")
    ap.add_argument("--pixel-repeats", type=int, default=10,
                    help="Number of pixel re-draws for stability check (default: 10)")

    ap.add_argument("--outfile", default=None,
                    help="Path for JSON output (default: sanity_<timestamp>.json)")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    outfile = args.outfile or f"sanity_{int(time.time())}.json"
    eps = dtype_r(args.eps)

    print("=" * 55)
    print("  MCMC SANITY CHECK")
    print("=" * 55)
    print(f"  scaling     = {args.scaling}")
    print(f"  n_rays      = {args.n_rays}")
    print(f"  eps         = {args.eps}")
    print(f"  img_glob    = {args.img_glob}")
    print(f"  pos_err     = {args.pos_err} m")
    print(f"  ang_err     = {args.ang_err_deg} deg")
    print(f"  f_err       = {args.f_err}")
    print(f"  log_h_sigma = {args.log_h_sigma}")
    print(f"  output      = {outfile}")

    # ── setup (mirrors eigsep_terrain_pymc.py) ────────────────────────────────
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

    cam_chunks = []
    for key in img_keys:
        p = np.asarray(DEFAULT_PRMS_U_BY_KEY[key], dtype=dtype_r)
        if args.set_cam_height:
            p = p.copy()
            p[2] = float(dem.interp_alt(p[0], p[1])) + args.cam_height
        cam_chunks.append(p)
    cam_prms_u = np.concatenate(cam_chunks).astype(dtype_r)
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

    # ── run checks ────────────────────────────────────────────────────────────
    results = {}

    results["1_logL_at_init"] = check_logL_at_init(
        ps, prms_h, eps, args.n_rays)

    results["2_dem_bounds"] = check_dem_bounds(ps, dem)

    results["3_prior_predictive"] = check_prior_predictive(
        ps, prms_h, eps, args.n_rays, n_samples=args.prior_samples)

    results["4_scaling_probe"] = check_scaling_probe(
        ps, prms_h, eps, args.n_rays,
        scaling=args.scaling, n_proposals=args.probe_proposals)

    results["5_pixel_stability"] = check_pixel_stability(
        ps, prms_h, eps, args.n_rays, n_repeats=args.pixel_repeats)

    # ── summary ───────────────────────────────────────────────────────────────
    _print_summary(results)

    # ── write JSON ────────────────────────────────────────────────────────────
    def _json_safe(obj):
        """Convert numpy scalars and bools to plain Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    out = {
        "args": vars(args),
        "checks": results,
    }
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2, default=_json_safe)
    print(f"Results written to: {outfile}\n")

    # Exit code: 1 if any FAIL, 0 otherwise (useful for CI / scripting)
    any_fail = any(r["status"] == FAIL for r in results.values())
    return int(any_fail)


if __name__ == "__main__":
    raise SystemExit(main())