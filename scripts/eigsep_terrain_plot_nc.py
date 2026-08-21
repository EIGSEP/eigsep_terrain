#!/usr/bin/env python
"""Plot MCMC traces from eigsep_terrain_pymc.py output .nc files.

Usage:
    eigsep_terrain_plot_nc.py [--cache-file F] [--img-glob G] trace_seed*.nc
"""
import argparse
import glob
import numpy as np
import matplotlib.pylab as plt
from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER
from eigsep_terrain.img_defaults import load_defaults
from eigsep_data.plot import terrain_plot
import arviz
import corner

# Generic, arbitrary-image-count defaults loaded from defaults.json (same
# file used by tune_image.py / fit_image.py / plot_image_fit.py). Edit
# defaults.json, not this script, when a starting value changes.
DEFAULT_IMG_GLOB, DEFAULT_CACHE_FILE, DEFAULT_META, DEFAULT_PRMS_U_BY_KEY, IMG_KEYS = \
    load_defaults()

ap = argparse.ArgumentParser()
ap.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
ap.add_argument("--img-glob", default=DEFAULT_IMG_GLOB)
ap.add_argument("nc_files", nargs="*")
args = ap.parse_args()

np.random.seed(42)

dem = DEM(cache_file=args.cache_file)

meta = {k: dict(v) for k, v in DEFAULT_META.items()}
BOX_SIZE = 0.3  # m

files = sorted(glob.glob(args.img_glob))
print(files)
imgs = [HorizonImage(f, meta, px_smooth=150, px_dist=30) for f in files]
imgs = [img for img in imgs if img.key in meta]
fit_imgs, static_imgs = imgs, []
n_rays = 4000
img_keys = [img.key for img in fit_imgs]

# Default parameter values (e, n, u, th, ph, ti, f) per camera, in fit_imgs
# order, from defaults.json's per-key prms_u — followed by a dummy
# (ant_e, ant_n, ant_u). defaults.json has no tracked platform/antenna
# position, so the antenna is seeded at the mean camera E/N, 1m above
# ground, purely so PositionSolver's u<->log_h conversion has a valid
# DEM location.
default_prms = np.concatenate(
    [np.asarray(DEFAULT_PRMS_U_BY_KEY[k], dtype=np.float32) for k in img_keys]
    + [np.zeros(3, dtype=np.float32)]  # placeholder, filled in below
)

# Initialise images and dem markers from default_prms
for i, img in enumerate(fit_imgs):
    base = i * len(PRM_ORDER)
    img.set_prms(default_prms[base:base + len(PRM_ORDER)])
    dem[img.key] = np.asarray(
        default_prms[base:base + 3], dtype=np.float32
    )
_e0 = float(np.mean([img.prms['e'] for img in fit_imgs]))
_n0 = float(np.mean([img.prms['n'] for img in fit_imgs]))
ant_pos_prior = np.array(
    [_e0, _n0, float(dem.interp_alt(_e0, _n0)) + 1.0], dtype=np.float32
)
default_prms[-3:] = ant_pos_prior
dem['platform'] = ant_pos_prior

ps = PositionSolver(
    ant_pos_prior, fit_imgs, static_imgs, n_rays, dem, box_size=BOX_SIZE
)

trace_files = args.nc_files or sorted(glob.glob("*.nc"))
print(trace_files)

idata = [arviz.from_netcdf(filename) for filename in trace_files]
trc = arviz.concat(*idata, dim="chain")

# Acceptance fraction per chain and overall
if hasattr(trc, 'sample_stats') and hasattr(trc.sample_stats, 'accepted'):
    acc = np.asarray(trc.sample_stats.accepted)  # (chain, draw)
    for c, frac in enumerate(acc.mean(axis=1)):
        print(f"chain {c}: acceptance fraction = {frac:.3f}")
    print(f"overall acceptance fraction = {acc.mean():.3f}")

# Built dynamically from whichever images were actually loaded (previously
# hardcoded to the fixed 0817/0833/0860 3-camera set).
ordered_names = [
    f"{key}_{('log_h' if p == 'u' else p)}"
    for key in img_keys for p in PRM_ORDER
] + ["ant_e", "ant_n", "ant_log_h"]

# Update solver and dem markers from trace posterior means (log_h -> u)
trace_means = np.array(
    [float(np.mean(trc.posterior[k])) for k in ordered_names]
)
ps.set_mcmc_prms(trace_means)
for img in ps.fit_imgs:
    dem[img.key] = np.asarray(
        [img.prms[k] for k in 'enu'], dtype=np.float32
    )
dem['platform'] = ps.ant_pos.astype(np.float32)

ps.set_mcmc_sigmas()

fig, axes = plt.subplots(
    nrows=len(trc.posterior), sharex=True, figsize=(8, 12)
)
for i, k in enumerate(trc.posterior.keys()):
    v = np.asarray(trc.posterior[k])
    print(k, np.std(v) / ps.sigmas[i], np.mean(v))
    axes[i].plot(v.T)

for t in range(trc.posterior['ant_e'].shape[0]):
    iprms = [trc.posterior[k][t, 0] for k in ordered_names]
    fprms = [trc.posterior[k][t, -1] for k in ordered_names]
    print(
        t,
        ps.total_logL(np.asarray(iprms)),
        ps.total_logL(np.asarray(fprms)),
    )

vars_to_plot = ["ant_e", "ant_n", "ant_log_h"]

posterior = trc.posterior[vars_to_plot].stack(sample=("chain", "draw"))
samples = np.column_stack(
    [posterior[v].values for v in vars_to_plot]
)
corner.corner(samples, labels=vars_to_plot, show_titles=True)

fig, ax = plt.subplots()
e0, n0, u0 = dem['platform']
rng = 750
alpha = 0.02
terrain_plot(
    dem, ax=ax,
    vmin=u0 - 300, vmax=u0 + 200,
    erng_m=(e0 - rng, e0 + rng),
    nrng_m=(n0 - rng, n0 + rng),
)
ax.plot(
    np.asarray(trc.posterior['ant_e']).flatten(),
    np.asarray(trc.posterior['ant_n']).flatten(),
    'k.', alpha=alpha,
)
for img in imgs:
    try:
        ax.plot(
            np.asarray(trc.posterior[f'{img.key}_e']).flatten(),
            np.asarray(trc.posterior[f'{img.key}_n']).flatten(),
            '.', alpha=alpha,
        )
    except KeyError:
        ax.plot(
            np.asarray(trc.posterior['e']).flatten(),
            np.asarray(trc.posterior['n']).flatten(),
            '.', alpha=alpha,
        )

plt.show()