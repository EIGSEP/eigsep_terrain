#!/usr/bin/env python
"""
toy_plot.py — Visualise toy_problem.py results on the terrain.

Loads:
  - marjum_dem.npz          (DEM)
  - toy_synth_cam{0,1,2}.npz (true camera positions stored by toy_problem.py)
  - toy_trace.nc            (ArviZ trace from MCMC)

Produces two figures:
  Fig 1 — Overview: terrain + true camera & antenna positions.
  Fig 2 — Posterior: terrain + true positions + MCMC posterior scatter/KDE
           for each camera and the antenna.

Usage:
  python toy_plot.py [--cache-file marjum_dem.npz] [--trace toy_trace.nc]
                     [--outdir .] [--no-show]
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import arviz as az

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import PRM_ORDER, dtype_r
from eigsep_terrain.img_defaults import load_defaults

_, DEFAULT_CACHE_FILE, _, _, _ = load_defaults()


# ── terrain helper (as provided) ─────────────────────────────────────────────

def terrain_plot(dem, ax=None, xlabel=True, ylabel=True,
                 colorbar=True, cmap='terrain', erng_m=None, nrng_m=None,
                 decimate=1, **kw):
    E, N, U = dem.get_tile(erng_m=erng_m, nrng_m=nrng_m,
                           mesh=False, decimate=decimate)
    extent = (E[0], E[-1], N[0], N[-1])
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(U, extent=extent, cmap=cmap, origin='lower',
                   interpolation='nearest', **kw)
    if colorbar:
        cb = plt.colorbar(im, ax=ax)
        cb.set_label('Elevation [m]')
    if xlabel:
        ax.set_xlabel('East [m]')
    if ylabel:
        ax.set_ylabel('North [m]')
    return im


# ── colours / markers ─────────────────────────────────────────────────────────

CAM_COLORS  = ['#e63946', '#2a9d8f', '#f4a261']   # red, teal, orange
ANT_COLOR   = '#a855f7'                             # purple
POST_ALPHA  = 0.25
TRUE_MARKER = '*'
TRUE_SIZE   = 200
POST_MARKER = '.'
POST_SIZE   = 6


def _arrow_kwargs(color):
    return dict(color=color, width=0.5, head_width=4,
                head_length=3, length_includes_head=True)


def _pointing_arrow(ax, e, n, th, ph, length=40, color='k'):
    """Draw a short arrow showing where the camera is pointing (azimuth ph)."""
    de = length * np.sin(ph)
    dn = length * np.cos(ph)
    ax.annotate('', xy=(e + de, n + dn), xytext=(e, n),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))


# ── load true positions from npz files ───────────────────────────────────────

def load_true_positions(n_cams=3):
    """
    toy_problem.py stores per-camera truth in toy_synth_cam{i}.npz.
    We also re-derive (e, n, u, th, ph) from the toy_problem defaults
    so we don't need a separate JSON.  The npz doesn't store prms, so
    we read them from a companion JSON if present, otherwise ask the user.
    """
    # Try companion JSON first
    if os.path.exists("toy_true_prms.json"):
        with open("toy_true_prms.json") as f:
            d = json.load(f)
        cameras = [np.array(d[f"cam{i}"]) for i in range(n_cams)]
        ant_pos = np.array(d["ant"])
        angles  = [np.array(d[f"angles{i}"]) for i in range(n_cams)]
        return cameras, ant_pos, angles

    # Fall back: import toy_problem and recompute (requires DEM already loaded)
    return None, None, None


# ── figure 1: overview ────────────────────────────────────────────────────────

def fig_overview(dem, cameras, ant_pos, angles, outdir='.'):
    fig, ax = plt.subplots(figsize=(8, 7))
    terrain_plot(dem, ax=ax, decimate=2, alpha=0.85)
    ax.set_title("True camera & antenna positions", fontsize=13)

    for i, ((e, n, u), (th, ph, ti)) in enumerate(zip(cameras, angles)):
        ax.scatter(e, n, s=TRUE_SIZE, marker=TRUE_MARKER,
                   color=CAM_COLORS[i], zorder=5,
                   label=f'cam{i}  (u={u:.1f} m)')
        _pointing_arrow(ax, e, n, th, ph, color=CAM_COLORS[i])

    ax.scatter(*ant_pos[:2], s=TRUE_SIZE, marker='D',
               color=ANT_COLOR, zorder=5,
               label=f'antenna  (u={ant_pos[2]:.1f} m)')

    # Draw line from each camera to antenna
    for i, (e, n, u) in enumerate(cameras):
        ax.plot([e, ant_pos[0]], [n, ant_pos[1]],
                '--', color=CAM_COLORS[i], lw=0.8, alpha=0.5)

    ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
    fig.tight_layout()
    out = os.path.join(outdir, "toy_overview.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return fig


# ── figure 2: posterior scatter on terrain ───────────────────────────────────

def fig_posterior(dem, cameras, ant_pos, angles, trace, outdir='.'):
    fig, ax = plt.subplots(figsize=(9, 8))
    terrain_plot(dem, ax=ax, decimate=2, alpha=0.70)
    ax.set_title("Posterior samples on terrain", fontsize=13)

    post = trace.posterior

    for i, ((e_true, n_true, u_true), (th, ph, ti)) in enumerate(
        zip(cameras, angles)
    ):
        color = CAM_COLORS[i]

        # Posterior e/n samples
        e_name = f"cam{i}_e"
        n_name = f"cam{i}_n"
        if e_name in post and n_name in post:
            e_samp = post[e_name].values.flatten()
            n_samp = post[n_name].values.flatten()
            ax.scatter(e_samp, n_samp, s=POST_SIZE, marker=POST_MARKER,
                       color=color, alpha=POST_ALPHA, zorder=3)
            # posterior mean
            ax.scatter(e_samp.mean(), n_samp.mean(),
                       s=80, marker='o', color=color,
                       edgecolors='k', linewidths=0.5, zorder=6)

        # True position
        ax.scatter(e_true, n_true, s=TRUE_SIZE, marker=TRUE_MARKER,
                   color=color, edgecolors='k', linewidths=0.5, zorder=7)
        _pointing_arrow(ax, e_true, n_true, th, ph, color=color)
        ax.annotate(f'cam{i}', (e_true, n_true),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=8, color=color, fontweight='bold')

    # Antenna posterior
    if 'ant_e' in post and 'ant_n' in post:
        ae = post['ant_e'].values.flatten()
        an = post['ant_n'].values.flatten()
        ax.scatter(ae, an, s=POST_SIZE, marker=POST_MARKER,
                   color=ANT_COLOR, alpha=POST_ALPHA, zorder=3)
        ax.scatter(ae.mean(), an.mean(),
                   s=80, marker='o', color=ANT_COLOR,
                   edgecolors='k', linewidths=0.5, zorder=6)

    # True antenna
    ax.scatter(*ant_pos[:2], s=TRUE_SIZE, marker='D',
               color=ANT_COLOR, edgecolors='k', linewidths=0.5, zorder=7)
    ax.annotate('ant', ant_pos[:2],
                textcoords='offset points', xytext=(6, 4),
                fontsize=8, color=ANT_COLOR, fontweight='bold')

    # Legend
    legend_elements = (
        [Line2D([0], [0], marker=TRUE_MARKER, color='w',
                markerfacecolor=c, markersize=11,
                markeredgecolor='k', label=f'cam{i} true')
         for i, c in enumerate(CAM_COLORS)]
        + [Line2D([0], [0], marker='D', color='w',
                  markerfacecolor=ANT_COLOR, markersize=9,
                  markeredgecolor='k', label='ant true'),
           Line2D([0], [0], marker='o', color='w',
                  markerfacecolor='grey', markersize=8,
                  markeredgecolor='k', label='post. mean'),
           Line2D([0], [0], marker=POST_MARKER, color='grey',
                  markersize=5, alpha=0.6, label='post. samples')]
    )
    ax.legend(handles=legend_elements, loc='upper left',
              fontsize=8, framealpha=0.85)

    fig.tight_layout()
    out = os.path.join(outdir, "toy_posterior.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return fig


# ── figure 3: per-parameter marginals ────────────────────────────────────────

def fig_marginals(cameras, ant_pos, angles, trace, outdir='.'):
    """1-D posterior histograms with true value marked."""
    post = trace.posterior

    # Build true-value dict in the same h-space the trace uses
    # (we only have u, not log_h, so we skip the log_h params here
    #  and just mark the raw e/n/angles/f values directly)
    truth = {}
    for i, ((e, n, u), (th, ph, ti)) in enumerate(zip(cameras, angles)):
        truth[f"cam{i}_e"]  = e
        truth[f"cam{i}_n"]  = n
        # cam_log_h: we can't compute without dem here, skip marking
        truth[f"cam{i}_th"] = th
        truth[f"cam{i}_ph"] = ph
        truth[f"cam{i}_ti"] = ti
        truth[f"cam{i}_f"]  = 500.0   # FOCAL from toy_problem
    truth["ant_e"] = ant_pos[0]
    truth["ant_n"] = ant_pos[1]

    param_names = [v for v in post.data_vars]
    ncols = 4
    nrows = int(np.ceil(len(param_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 2.5))
    axes = axes.flatten()

    for ax, name in zip(axes, param_names):
        samples = post[name].values.flatten()
        # pick color by cam index
        color = 'steelblue'
        for i, c in enumerate(CAM_COLORS):
            if name.startswith(f"cam{i}"):
                color = c
                break
        if name.startswith("ant"):
            color = ANT_COLOR

        ax.hist(samples, bins=40, color=color, alpha=0.7, density=True)
        if name in truth:
            ax.axvline(truth[name], color='k', lw=1.5, ls='--', label='true')
        ax.axvline(samples.mean(), color='k', lw=1.0, ls=':', label='mean')
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=7)

    # hide unused axes
    for ax in axes[len(param_names):]:
        ax.set_visible(False)

    fig.suptitle("Posterior marginals  (-- true,  ··· mean)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(outdir, "toy_marginals.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
    ap.add_argument("--trace",      default="toy_trace.nc")
    ap.add_argument("--prms-json",  default="toy_true_prms.json",
                    help="JSON with true params written by toy_problem.py")
    ap.add_argument("--outdir",     default=".")
    ap.add_argument("--no-show",    action="store_true")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    # ── load DEM ──────────────────────────────────────────────────────────────
    dem = DEM(cache_file=args.cache_file)

    # ── load true params ──────────────────────────────────────────────────────
    if not os.path.exists(args.prms_json):
        raise FileNotFoundError(
            f"{args.prms_json} not found. "
            "Run toy_problem.py first (it saves this file after Stage 1)."
        )
    with open(args.prms_json) as f:
        d = json.load(f)

    n_cams   = d["n_cams"]
    cameras  = [np.array(d[f"cam{i}"]) for i in range(n_cams)]
    ant_pos  = np.array(d["ant"])
    angles   = [np.array(d[f"angles{i}"]) for i in range(n_cams)]

    # ── load trace (optional) ─────────────────────────────────────────────────
    trace = None
    if os.path.exists(args.trace):
        trace = az.from_netcdf(args.trace)
        print(f"Loaded trace: {args.trace}")
    else:
        print(f"No trace found at {args.trace} — skipping posterior plots.")

    os.makedirs(args.outdir, exist_ok=True)

    # ── figures ───────────────────────────────────────────────────────────────
    fig_overview(dem, cameras, ant_pos, angles, outdir=args.outdir)

    if trace is not None:
        fig_posterior(dem, cameras, ant_pos, angles, trace, outdir=args.outdir)
        fig_marginals(cameras, ant_pos, angles, trace, outdir=args.outdir)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()