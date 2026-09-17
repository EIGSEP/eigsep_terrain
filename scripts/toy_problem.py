#!/usr/bin/env python
"""
toy_problem.py — Synthetic end-to-end test of the eigsep MCMC pipeline.

Stages:
  1. Load DEM, place 3 cameras and 1 antenna at realistic positions.
  2. Ray-trace every pixel for each camera with TRUE params → binary sky/ground.
  3. Build soft psky map (noise-blurred) as synthetic "segmentation" output.
  4. Run MCMC exactly as eigsep_terrain_pymc.py does.
  5. Compare posterior to true params.

Usage:
  python toy_problem.py --cache-file marjum_dem.npz [--stage 1|2|3|4|all]
"""

import argparse
import json
import os
import sys
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

# ── eigsep imports ────────────────────────────────────────────────────────────
from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import (
    HorizonImage, PositionSolver, PRM_ORDER, pixels_to_rays, dtype_r
)
from healjax.coord import rot_m
from eigsep_terrain.utils import mask_near_horizon

BOX_SIZE = 0.3  # m
IMG_W = 640     # synthetic image width  (pixels) — keep small for speed
IMG_H = 480     # synthetic image height (pixels)
FOCAL = 500.0   # focal length (pixels)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 helpers — camera / antenna placement
# ══════════════════════════════════════════════════════════════════════════════

def find_valid_position(dem, e_range, n_range, cam_height=1.6, seed=0):
    """Return (e, n, u) with u = terrain + cam_height, avoiding map edges."""
    rng = np.random.default_rng(seed)
    e_m, n_m = dem.get_en()
    border = 50  # m from edge
    for _ in range(10000):
        e = rng.uniform(e_range[0] + border, e_range[1] - border)
        n = rng.uniform(n_range[0] + border, n_range[1] - border)
        u0 = float(dem.interp_alt(e, n))
        if np.isfinite(u0) and u0 > 0:
            return np.array([e, n, u0 + cam_height], dtype=dtype_r)
    raise RuntimeError("Could not find valid camera position in given range")


def place_cameras_and_antenna(dem):
    """
    Manually pick three camera positions spread around the Marjum canyon area
    and one antenna suspended across the canyon.

    The DEM coordinate system is local ENU in metres.
    Approximate extents: E ~ [0, 1500], N ~ [0, 2000] (depends on tile layout).
    We target positions near canyon walls so the horizon is interesting.
    """
    e_m, n_m = dem.get_en()
    E_max, N_max = float(e_m[-1]), float(n_m[-1])
    print(f"DEM extent:  E=[0, {E_max:.0f} m]  N=[0, {N_max:.0f} m]")

    cam_height = 1.6  # m above ground

    # Three camera positions (spread across the scene)
    cam_seeds = [
        (E_max * 0.30, E_max * 0.45, N_max * 0.40, N_max * 0.55),  # cam0 west-center
        (E_max * 0.45, E_max * 0.65, N_max * 0.55, N_max * 0.75),  # cam1 center-north
        (E_max * 0.60, E_max * 0.80, N_max * 0.30, N_max * 0.50),  # cam2 east-south
    ]
    cameras = []
    for i, (e0, e1, n0, n1) in enumerate(cam_seeds):
        pos = find_valid_position(dem, (e0, e1), (n0, n1),
                                  cam_height=cam_height, seed=i * 17)
        cameras.append(pos)
        print(f"  cam{i}: e={pos[0]:.1f}  n={pos[1]:.1f}  u={pos[2]:.1f}  "
              f"(terrain={pos[2]-cam_height:.1f})")

    # Antenna: midpoint between cam0 and cam2, elevated 10 m above terrain
    ant_e = (cameras[0][0] + cameras[2][0]) / 2.0
    ant_n = (cameras[0][1] + cameras[2][1]) / 2.0
    ant_u0 = float(dem.interp_alt(ant_e, ant_n))
    ant_pos = np.array([ant_e, ant_n, ant_u0 + 10.0], dtype=dtype_r)
    print(f"  ant  : e={ant_pos[0]:.1f}  n={ant_pos[1]:.1f}  u={ant_pos[2]:.1f}  "
          f"(h={ant_pos[2]-ant_u0:.1f} m above terrain)")

    return cameras, ant_pos


def default_camera_angles():
    """
    Return (th, ph, ti) orientation angles for each camera.
    th  = tilt from vertical (camera pointing outward ~horizontal)
    ph  = azimuth (pan direction)
    ti  = in-plane roll (small)
    """
    return [
        (np.deg2rad(85), np.deg2rad(45),  np.deg2rad(0)),   # cam0 NE
        (np.deg2rad(80), np.deg2rad(200), np.deg2rad(1)),   # cam1 SW
        (np.deg2rad(88), np.deg2rad(310), np.deg2rad(-1)),  # cam2 NW
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Synthetic psky generation
# ══════════════════════════════════════════════════════════════════════════════

def compute_synthetic_psky(dem, e, n, u, th, ph, ti, f,
                           img_h=IMG_H, img_w=IMG_W,
                           noise_sigma=0.05, fine_delta=0.25):
    """
    Ray-trace all pixels, return soft psky map (H x W float32).
    Pixels where ray misses terrain → psky ~ 1 (sky).
    Pixels where ray hits terrain  → psky ~ 0 (ground).
    Gaussian noise added for realism.
    """
    from eigsep_terrain.ray_numba import ray_distance_coarse_to_fine_numba

    # Build all ray directions
    z_rays = pixels_to_rays(img_h, img_w, f=f, dtype=dtype_r)   # (3, H, W)
    rm_ti = rot_m(ti,  np.array([0, 0, 1], dtype=dtype_r))
    rm_th = rot_m(th,  np.array([0, 1, 0], dtype=dtype_r))
    rm_ph = rot_m(ph,  np.array([0, 0, 1], dtype=dtype_r))
    rm = rm_ph @ (rm_th @ rm_ti)
    rays = np.einsum('ij,j...->i...', rm, z_rays)               # (3, H, W)

    start_point = np.array([e, n, u], dtype=dtype_r)
    (E, N), U_dem = dem.get_en(), dem.data
    rays_2d = rays.reshape(3, -1)

    r = ray_distance_coarse_to_fine_numba(
        E, N, U_dem, start_point, rays_2d, fine_delta=fine_delta
    )
    model_sky = np.isnan(r).reshape(img_h, img_w)

    # Soft psky: binary + Gaussian noise, clamped
    psky = model_sky.astype(np.float32)
    rng = np.random.default_rng(42)
    psky += rng.normal(0, noise_sigma, psky.shape).astype(np.float32)
    psky = np.clip(psky, 0.02, 0.98)
    return psky, model_sky


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Synthetic HorizonImage
# ══════════════════════════════════════════════════════════════════════════════

class SyntheticHorizonImage:
    """
    Drop-in replacement for HorizonImage that uses a pre-computed psky map
    instead of a real image file + SegFormer segmentation.

    Exposes the same interface used by PositionSolver and total_logL:
      - key, prms, meta
      - set_prms / get_prms
      - choose_pixels
      - get_rays
      - ray_distance
      - horizon_ray_logL
      - ant_logL
    """

    def __init__(self, key, psky, ant_px, prms_dict,
                 px_dist=30, px_smooth=50):
        self.key = key
        self.psky = psky.astype(np.float32)    # (H, W) in [0,1]
        self.meta = {'ant_px': ant_px}         # (col, row) = (x, y)
        self.px_dist = px_dist
        self.px_smooth = px_smooth
        self._px_choice = None

        # Build a synthetic sky_mask from psky threshold
        self.sky_mask = (psky > 0.5)
        # Horizon mask = strip near the sky/ground boundary
        self.horizon_mask, self.horizon_dist = mask_near_horizon(
            self.sky_mask, px_dist
        )

        self.set_prms([prms_dict[k] for k in PRM_ORDER])

    @property
    def npix_y(self):
        return self.psky.shape[0]

    @property
    def npix_x(self):
        return self.psky.shape[1]

    def set_prms(self, prms):
        self.prms = dict(zip(PRM_ORDER, prms))

    def get_prms(self):
        return (self.prms[k] for k in PRM_ORDER)

    @property
    def prms_str(self):
        return (f"{self.prms['e']: 7.2f}, {self.prms['n']: 7.2f}, "
                f"{self.prms['u']: 7.2f}, {self.prms['th']: 6.4f}, "
                f"{self.prms['ph']: 6.4f}, {self.prms['ti']: 5.4f}, "
                f"{self.prms['f']: 7.2f}")

    def choose_pixels(self, N=1000, mask=None, reset=False):
        if reset:
            self._px_choice = None
        if self._px_choice is None:
            if mask is None:
                mask = self.horizon_mask
            x, y = np.where(mask)
            if x.size == 0:
                raise RuntimeError(
                    f"[{self.key}] horizon_mask is empty — "
                    "no pixels near sky/ground boundary. "
                    "Check camera orientation or psky map."
                )
            if x.size < N:
                print(f"  [{self.key}] WARNING: only {x.size} horizon pixels; "
                      f"requested {N}. Using all.")
                N = x.size
            w = np.exp(
                -0.5 * self.horizon_dist[x, y] ** 2 / (self.px_dist / 2) ** 2
            )
            w = w / w.sum()
            rng = np.random.default_rng()
            inds = rng.choice(x.size, size=N, replace=False, p=w)
            self._px_choice = (x[inds], y[inds])
        return self._px_choice

    def get_rays(self, pixels=None, dtype=dtype_r):
        z_rays = pixels_to_rays(
            self.npix_y, self.npix_x, f=self.prms['f'], uv=pixels, dtype=dtype
        )
        rm_ti = rot_m(self.prms['ti'], np.array([0, 0, 1], dtype=dtype))
        rm_th = rot_m(self.prms['th'], np.array([0, 1, 0], dtype=dtype))
        rm_ph = rot_m(self.prms['ph'], np.array([0, 0, 1], dtype=dtype))
        rm = rm_ph @ (rm_th @ rm_ti)
        return np.einsum('ij,j...->i...', rm, z_rays)

    def ray_distance(self, dem, rays, dtype=dtype_r, fine_delta=0.25):
        from eigsep_terrain.ray_numba import ray_distance_coarse_to_fine_numba
        rays_2d = rays.reshape(rays.shape[0], -1)
        (E, N), U = dem.get_en(), dem.data
        start_point = np.array(
            [self.prms[k] for k in ('e', 'n', 'u')], dtype=dtype
        )
        r = ray_distance_coarse_to_fine_numba(
            E, N, U, start_point, rays_2d, fine_delta=fine_delta
        )
        r.shape = rays.shape[1:]
        return r

    def horizon_ray_logL(self, dem, n_rays=1000, dtype=dtype_r,
                         eps=1e-3, fine_delta=0.25):
        x_px, y_px = self.choose_pixels(N=n_rays)
        psky = self.psky[x_px, y_px].clip(eps, 1 - eps)
        rays = self.get_rays(pixels=(x_px, y_px), dtype=dtype)
        r = self.ray_distance(dem, rays, dtype=dtype, fine_delta=fine_delta)
        model_sky = np.isnan(r)
        logL = np.sum(
            np.where(model_sky, np.log(psky), np.log1p(-psky))
        )
        return logL

    def ant_logL(self, ant_pos, box_size):
        ant_px_rc = np.array(self.meta['ant_px'][::-1])   # (row, col) → (y, x)
        ant_ray = self.get_rays(ant_px_rc)
        r_ant = ant_pos - np.array(
            [self.prms['e'], self.prms['n'], self.prms['u']]
        )
        cos_pred = (np.dot(ant_ray, r_ant)
                    / (np.linalg.norm(ant_ray) * np.linalg.norm(r_ant)))
        delta_theta = np.arccos(cos_pred.clip(-1, 1))
        sigma_theta = box_size / np.linalg.norm(r_ant)
        logL = (np.log(1.0 / np.sqrt(2 * np.pi * sigma_theta ** 2))
                - 0.5 * delta_theta ** 2 / sigma_theta ** 2)
        return logL


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def project_ant_to_pixel(cam_e, cam_n, cam_u, ant_pos,
                         th, ph, ti, f, img_h, img_w):
    """
    Project antenna world-position into camera pixel coordinates.
    Returns (col, row) = (x, y) tuple, or (img_w//2, img_h//2) if behind.

    Must be the exact inverse of get_rays / pixels_to_rays:
      pixels_to_rays builds:
        rays = [Nu//2 - u,  Nv//2 - v,  f]   (u=row index, v=col index)
      then applies  rm = rm_ph @ rm_th @ rm_ti
      so world_ray = rm @ cam_ray

    Inverse: cam_ray = rm.T @ world_dir
      row = Nu//2 - cam_ray[0]/cam_ray[2] * f
      col = Nv//2 - cam_ray[1]/cam_ray[2] * f
    """
    rm_ti = rot_m(ti, np.array([0, 0, 1], dtype=np.float64))
    rm_th = rot_m(th, np.array([0, 1, 0], dtype=np.float64))
    rm_ph = rot_m(ph, np.array([0, 0, 1], dtype=np.float64))
    rm = rm_ph @ (rm_th @ rm_ti)

    d = np.array(ant_pos, dtype=np.float64) - np.array([cam_e, cam_n, cam_u])
    d_cam = rm.T @ d   # camera frame: [0]=row axis, [1]=col axis, [2]=optical axis

    if d_cam[2] <= 0:
        print("  WARNING: antenna is behind camera — using image centre")
        return (img_w // 2, img_h // 2)

    # Invert pixels_to_rays exactly
    row = int(round(img_h // 2 - d_cam[0] / d_cam[2] * f))
    col = int(round(img_w // 2 - d_cam[1] / d_cam[2] * f))
    row = int(np.clip(row, 0, img_h - 1))
    col = int(np.clip(col, 0, img_w - 1))
    return (col, row)   # meta['ant_px'] = (col, row); ant_logL reverses to (row, col)


def build_true_prms_vector(cameras, angles, ant_pos, focal=FOCAL):
    """Flat float32 array [cam0_e, cam0_n, cam0_u, cam0_th, cam0_ph, cam0_ti, cam0_f, cam1_..., ant_e, ant_n, ant_u]"""
    vec = []
    for (e, n, u), (th, ph, ti) in zip(cameras, angles):
        vec += [e, n, u, th, ph, ti, focal]
    vec += list(ant_pos)
    return np.array(vec, dtype=dtype_r)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", default="marjum_dem.npz")
    ap.add_argument("--stage", default="all",
                    help="1|2|3|4|all  (default: all)")
    ap.add_argument("--seed",  type=int, default=7)
    ap.add_argument("--n-rays", type=int, default=500)
    ap.add_argument("--draws",         type=int, default=2000)
    ap.add_argument("--tune",          type=int, default=500)
    ap.add_argument("--chains",        type=int, default=1)
    ap.add_argument("--cores",         type=int, default=1)
    ap.add_argument("--tune-interval", type=int, default=50)
    ap.add_argument("--pos-err",     type=float, default=30.0)
    ap.add_argument("--ang-err-deg", type=float, default=5.0)
    ap.add_argument("--f-err",       type=float, default=0.1)
    ap.add_argument("--fine-delta",  type=float, default=0.25)
    ap.add_argument("--eps",         type=float, default=1e-2)
    ap.add_argument("--ant-weight",  type=float, default=1.0)
    ap.add_argument("--disable-ant", action="store_true")
    ap.add_argument("--scaling",     type=float, default=1e-2)
    ap.add_argument("--outfile",     default="toy_trace.nc")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    np.random.seed(args.seed)

    run_all = (args.stage == "all")
    stages = set(args.stage.split(",")) if not run_all else set()

    # ── Stage 1: Load DEM, place cameras & antenna ───────────────────────────
    print("\n" + "="*60)
    print("STAGE 1 — DEM + placement")
    print("="*60)

    dem = DEM(cache_file=args.cache_file)
    cameras, ant_pos = place_cameras_and_antenna(dem)
    angles = default_camera_angles()

    # Print true parameters
    true_prms = build_true_prms_vector(cameras, angles, ant_pos, focal=FOCAL)
    print("\nTrue parameter vector (u, not log_h):")
    names = [f"cam{i}_{k}" for i in range(3) for k in PRM_ORDER]
    names += ["ant_e", "ant_n", "ant_u"]
    for name, val in zip(names, true_prms):
        print(f"  {name:20s} = {val:.4f}")

    # Save true params for toy_plot.py
    prms_json = "toy_true_prms.json"
    true_json = {"n_cams": len(cameras), "ant": list(map(float, ant_pos))}
    for i, ((e, n, u), (th, ph, ti)) in enumerate(zip(cameras, angles)):
        true_json[f"cam{i}"]    = [float(e), float(n), float(u)]
        true_json[f"angles{i}"] = [float(th), float(ph), float(ti)]
    with open(prms_json, "w") as f:
        json.dump(true_json, f, indent=2)
    print(f"True params saved to {prms_json}")

    if args.stage == "1":
        return 0

    # ── Stage 2: Generate synthetic psky maps ────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2 — Synthetic psky generation")
    print("="*60)

    synth_data = []   # list of (psky, sky_mask, ant_px_xy) per camera
    for i, ((e, n, u), (th, ph, ti)) in enumerate(zip(cameras, angles)):
        print(f"\n  Computing psky for cam{i}  (e={e:.1f}, n={n:.1f}, u={u:.1f}) ...")
        psky, sky_mask = compute_synthetic_psky(
            dem, e, n, u, th, ph, ti, FOCAL,
            img_h=IMG_H, img_w=IMG_W, fine_delta=args.fine_delta
        )
        sky_frac = sky_mask.mean()
        print(f"    sky fraction = {sky_frac:.2%}   "
              f"psky range [{psky.min():.3f}, {psky.max():.3f}]")

        # Project antenna into this camera's pixel space
        ant_px_xy = project_ant_to_pixel(
            e, n, u, ant_pos, th, ph, ti, FOCAL, IMG_H, IMG_W
        )
        print(f"    antenna pixel (col, row) = {ant_px_xy}")

        synth_data.append((psky, sky_mask, ant_px_xy))
        np.savez(f"toy_synth_cam{i}.npz",
                 psky=psky, sky_mask=sky_mask, ant_px=np.array(ant_px_xy))
        print(f"    Saved toy_synth_cam{i}.npz")

    if args.stage == "2":
        return 0

    # ── Stage 3: Build SyntheticHorizonImage objects ─────────────────────────
    print("\n" + "="*60)
    print("STAGE 3 — Build SyntheticHorizonImage objects")
    print("="*60)

    fit_imgs = []
    for i, ((e, n, u), (th, ph, ti), (psky, sky_mask, ant_px_xy)) in enumerate(
        zip(cameras, angles, synth_data)
    ):
        prms_dict = dict(zip(PRM_ORDER, [e, n, u, th, ph, ti, FOCAL]))
        img = SyntheticHorizonImage(
            key=f"cam{i}",
            psky=psky,
            ant_px=ant_px_xy,
            prms_dict=prms_dict,
            px_dist=30,
        )
        n_hor = img.horizon_mask.sum()
        print(f"  cam{i}: horizon pixels = {n_hor}")
        if n_hor < 100:
            print(f"  WARNING: very few horizon pixels for cam{i}. "
                  "Consider adjusting camera angles.")
        fit_imgs.append(img)

    # Antenna projection diagnostic
    print("\n  Antenna projection check:")
    for i, (img, (e, n, u), (th, ph, ti)) in enumerate(
        zip(fit_imgs, cameras, angles)
    ):
        col, row = img.meta['ant_px']
        ray_from_px = img.get_rays(np.array([row, col]))
        r_ant = ant_pos - np.array([e, n, u], dtype=np.float64)
        r_ant_hat = r_ant / np.linalg.norm(r_ant)
        cos_sim = np.dot(ray_from_px.astype(np.float64), r_ant_hat)
        angle_err_deg = np.rad2deg(np.arccos(np.clip(cos_sim, -1, 1)))
        status = "\u2713" if angle_err_deg < 1.0 else "\u2717 BAD"
        print(f"    cam{i}: ant_px=({col},{row})  angle_error={angle_err_deg:.3f} deg  {status}")

    # Build PositionSolver with TRUE ant_pos as prior centre
    ps = PositionSolver(
        ant_pos_prior=ant_pos,
        fit_imgs=fit_imgs,
        static_imgs=[],
        n_rays=args.n_rays,
        dem=dem,
        box_size=BOX_SIZE,
    )

    # Convert true prms to h-space and set
    prms_h = ps.prms_u_to_h(true_prms)
    ps.set_mcmc_prms(prms_h)
    ps.set_mcmc_sigmas(
        pos_err=args.pos_err,
        ang_err=np.deg2rad(args.ang_err_deg),
        f_err=args.f_err,
        log_h_sigma=1.0,
    )

    # Sanity-check logL at true params
    logL_true = ps.total_logL(
        prms_h, n_rays=args.n_rays, eps=args.eps,
        ant_weight=args.ant_weight, disable_ant=args.disable_ant,
        fine_delta=args.fine_delta,
    )
    print(f"\n  logL at TRUE params = {logL_true:.2f}")

    # Sanity-check logL at perturbed params
    perturbed = prms_h.copy()
    perturbed[:3] += np.array([10.0, 10.0, 0.5], dtype=dtype_r)
    logL_pert = ps.total_logL(
        perturbed, n_rays=args.n_rays, eps=args.eps,
        ant_weight=args.ant_weight, disable_ant=args.disable_ant,
        fine_delta=args.fine_delta,
    )
    print(f"  logL at PERTURBED   = {logL_pert:.2f}  "
          f"(should be < true logL for a well-defined problem)")

    # Restore true params
    ps.set_mcmc_prms(prms_h)

    if args.stage == "3":
        return 0

    # ── Stage 4: Run MCMC ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 4 — MCMC")
    print("="*60)

    assert not os.path.exists(args.outfile), \
        f"{args.outfile} already exists — delete it or use --outfile."

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

    with pm.Model() as model:
        mcmc_prms = ps.get_mcmc_prms()

        rng_pm = np.random.default_rng(args.seed)
        initvals = []
        for c in range(args.chains):
            jitter = rng_pm.normal(0, np.asarray(ps.sigmas) * 0.1, prms_h.size)
            jittered = prms_h + jitter
            ps.set_mcmc_prms(jittered)
            start = ps.eval_cur_prms()
            initvals.append({p.name: v for p, v in zip(mcmc_prms, start)})

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
            random_seed=args.seed,
            progressbar=True,
        )

    az.to_netcdf(trace, args.outfile)
    print(f"\nTrace saved to {args.outfile}")

    # ── Stage 5: Compare posterior to true params ────────────────────────────
    print("\n" + "="*60)
    print("STAGE 5 — Posterior vs True")
    print("="*60)

    param_names = [p.name for p in mcmc_prms]
    accepted = float(trace.sample_stats.accepted.mean())
    print(f"Acceptance rate = {accepted:.3f}")
    print()

    # True values in h-space
    true_h = ps.prms_u_to_h(true_prms)

    results = {}
    print(f"{'param':25s}  {'true_h':>10s}  {'post_mean':>10s}  {'post_std':>10s}  {'z-score':>8s}")
    print("-" * 70)
    for i, name in enumerate(param_names):
        arr = trace.posterior[name].values.flatten()
        mu, std = arr.mean(), arr.std()
        truth = float(true_h[i])
        z = (mu - truth) / std if std > 0 else float('nan')
        results[name] = dict(true=truth, mean=mu, std=std, z=z)
        flag = " ✓" if abs(z) < 2 else " ✗"
        print(f"  {name:23s}  {truth:10.4f}  {mu:10.4f}  {std:10.4f}  {z:8.2f}{flag}")

    # Save comparison JSON
    out_json = args.outfile.replace(".nc", "_comparison.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison saved to {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())