#!/usr/bin/env python
"""
Plotting script for MCMC runs produced by eigsep_terrain_pymc.py.

Loads a .nc trace and its _meta.json sidecar, then produces any
combination of diagnostic plots controlled by boolean flags.

Usage
-----
  python plot_trace.py trace_seed042.nc [options]

All plots are saved as <stem>_<plot>.png unless --show is given.

Available plots (each a boolean flag, all off by default)
----------------------------------------------------------
  --trace       Chain timeseries + marginal KDE (az.plot_trace)
  --rank        Rank plots for multi-chain mixing (az.plot_rank)
  --autocorr    Autocorrelation per param (az.plot_autocorr)
  --posterior   Marginal posteriors with prior overlay
  --shrinkage   Bar chart: 1 - post_std / prior_sigma per param
  --pair        Bivariate scatter/KDE for a param group
  --acceptance  Rolling acceptance rate over the chain
  --logp        Log-likelihood (lik potential) timeseries
  --prior-predictive   Prior predictive vs posterior histogram per param
  --step-size          Effective step size vs post_std and prior_sigma
  --prior-sensitivity  Z-score and contraction ratio per param
  --all         Enable every plot above
"""
import argparse
import json
import os
import re

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import glob

from eigsep_terrain.img import HorizonImage
from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img_defaults import load_defaults

# Generic, arbitrary-image-count defaults loaded from defaults.json (same
# file used by tune_image.py / fit_image.py / plot_image_fit.py). Edit
# defaults.json, not this script, when a starting value changes.
DEFAULT_IMG_GLOB, DEFAULT_CACHE_FILE, DEFAULT_META, DEFAULT_PRMS_U_BY_KEY, IMG_KEYS = \
    load_defaults()

# ── figure style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.constrained_layout.use": True,
})

PARAM_LABELS = {
    "e": "E [m]", "n": "N [m]", "u": "U [m]",
    "log_h": "log h", "th": "θ [rad]", "ph": "φ [rad]",
    "ti": "tilt [rad]", "f": "f [px]",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(nc_path):
    """Load trace and sidecar metadata. Returns (trace, meta, stem, outdir)."""
    stem = re.sub(r"\.nc$", "", nc_path)
    metafile = f"{stem}_meta.json"
    trace = az.from_netcdf(nc_path)
    if os.path.exists(metafile):
        with open(metafile) as f:
            meta = json.load(f)
    else:
        print(f"WARNING: {metafile} not found — prior overlay and shrinkage unavailable.")
        meta = {}

    seed = meta.get("seed", re.search(r"seed(\d+)", stem).group(1)
                    if re.search(r"seed(\d+)", stem) else "unknown")
    outdir = os.path.join(os.path.dirname(os.path.abspath(nc_path)),
                          f"{seed}_dgnstc_plots")
    os.makedirs(outdir, exist_ok=True)
    print(f"  output folder: {outdir}")

    return trace, meta, stem, outdir


def _suptitle(fig, meta, plot_name="", extra=""):
    """Attach plot name + compact run-summary as the figure suptitle.

    Line 1 (bold): plot name
    Line 2: core sampling info
    Line 3: likelihood config (eps, ant_weight, disable_ant)
    Line 4: MAP provenance (if run was initialised from a MAP file)
    """
    s  = meta.get("sampling", {})
    st = meta.get("step", {})
    lk = meta.get("likelihood", {})
    mf = meta.get("map_file") or {}

    # ── line 2: sampling ──────────────────────────────────────────────────────
    accept = meta.get("accepted_mean", float("nan"))
    accept_str = f"{accept:.3f}" if isinstance(accept, float) else "?"
    line2_parts = [
        f"seed={meta.get('seed', '?')}",
        f"draws={s.get('draws', '?')}",
        f"tune={s.get('tune', '?')}",
        f"chains={s.get('chains', '?')}",
        f"scaling={st.get('scaling', '?')}",
        f"tuned_scaling={st.get('tuned_scaling', '?')}",
        f"accept={accept_str}",
    ]
    if extra:
        line2_parts.append(extra)

    # ── line 3: likelihood config ─────────────────────────────────────────────
    ant_weight   = lk.get("ant_weight", 1.0)
    disable_ant  = lk.get("disable_ant", False)
    if disable_ant:
        ant_str = "ant=DISABLED"
    else:
        ant_str = f"ant_weight={ant_weight}"
    line3_parts = [
        f"eps={lk.get('eps', '?')}",
        f"n_rays={lk.get('n_rays', '?')}",
        ant_str,
    ]

    # ── line 4: MAP provenance ─────────────────────────────────────────────────
    line4_parts = []
    if mf:
        map_logL    = mf.get("map_logL")
        map_method  = mf.get("map_method", "?")
        map_conv    = mf.get("map_converged", "?")
        map_seed    = mf.get("map_seed", "?")
        map_nrest   = mf.get("map_n_restarts", "?")
        logL_imp    = mf.get("logL_improvement")
        logL_imp_str = f"{logL_imp:+.1f}" if isinstance(logL_imp, float) else "?"
        line4_parts = [
            f"MAP seed={map_seed}",
            f"method={map_method}",
            f"converged={map_conv}",
            f"restarts={map_nrest}",
            f"map_logL={map_logL:.1f}" if isinstance(map_logL, float) else "map_logL=?",
            f"logL_improvement={logL_imp_str}",
        ]

    lines = ["  |  ".join(line2_parts), "  |  ".join(line3_parts)]
    if line4_parts:
        lines.append("MAP:  " + "  |  ".join(line4_parts))

    run_str = "\n".join(lines)

    if plot_name:
        fig.suptitle(plot_name + "\n" + run_str, fontsize=7, y=1.03,
                     fontweight="bold")
    else:
        fig.suptitle(run_str, fontsize=7, y=1.02)


def _param_label(name):
    """Turn raw param name (e.g. '0817_log_h') into a readable label."""
    parts = name.split("_", 1)
    if len(parts) == 2:
        key_part = parts[1]
        suffix = PARAM_LABELS.get(key_part, key_part)
        return f"{parts[0]} {suffix}"
    return PARAM_LABELS.get(name, name)


def _save(fig, outdir, tag):
    path = os.path.join(outdir, f"{tag}.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved: {path}")
    plt.close(fig)


def _posterior_array(trace, name):
    """Return flattened posterior samples for a param name."""
    return trace.posterior[name].values.flatten()


def _autoscale_y(ax, data, margin=0.05, symm=False):
    """Set y-limits to [p1, p99] of data with a fractional margin.
    If symm=True, make the range symmetric around zero (for z-score bars).
    Ignores NaN values."""
    data = np.asarray(data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    lo, hi = np.percentile(finite, 1), np.percentile(finite, 99)
    if symm:
        bound = max(abs(lo), abs(hi))
        lo, hi = -bound, bound
    span = hi - lo if hi != lo else 1.0
    ax.set_ylim(lo - margin * span, hi + margin * span)


def _autoscale_x(ax, samples, plo=0.5, phi=99.5, margin=0.05):
    """Set x-limits to [plo, phi] percentile of samples with a margin.
    Useful for histogram axes where prior tails would compress the posterior."""
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return
    lo, hi = np.percentile(finite, plo), np.percentile(finite, phi)
    span = hi - lo if hi != lo else 1.0
    ax.set_xlim(lo - margin * span, hi + margin * span)


def _terrain_plot(dem, ax=None, xlabel=True, ylabel=True,
             colorbar=True, cmap='terrain', erng_m=None, nrng_m=None,
             decimate=1, **kw):

    E, N, U = dem.get_tile(erng_m=erng_m, nrng_m=nrng_m, mesh=False, decimate=decimate)
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


# ── individual plot functions ─────────────────────────────────────────────────

def plot_trace(trace, meta, stem, outdir):
    """Chain timeseries + marginal KDE via ArviZ."""
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    axes = az.plot_trace(trace, var_names=param_names, compact=False,
                         figsize=(12, max(3, len(param_names) * 1.2)))
    fig = axes.ravel()[0].get_figure()
    # relabel y-axes with human-readable names
    for row, name in zip(axes, param_names):
        for ax in row:
            ax.set_title(_param_label(name), fontsize=8)
    _suptitle(fig, meta, plot_name="Trace plot")
    _save(fig, outdir, "trace")


def plot_rank(trace, meta, stem, outdir):
    """Rank plots — better than trace for diagnosing multi-chain mixing."""
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    axes = az.plot_rank(trace, var_names=param_names,
                        figsize=(10, max(3, len(param_names) * 0.9)))
    fig = axes.ravel()[0].get_figure()
    _suptitle(fig, meta, plot_name="Rank plot")
    _save(fig, outdir, "rank")


def plot_autocorr(trace, meta, stem, outdir):
    """Autocorrelation per param — slow decay = poor ESS."""
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    axes = az.plot_autocorr(trace, var_names=param_names, max_lag=200,
                            figsize=(12, max(3, len(param_names) * 1.0)))
    fig = axes.ravel()[0].get_figure()
    _suptitle(fig, meta, plot_name="Autocorrelation")
    _save(fig, outdir, "autocorr")


def plot_posterior(trace, meta, stem, outdir):
    """
    Marginal posterior KDE per param with prior Normal overlay.
    Prior sigma comes from meta['param_summary'][name]['prior_sigma'].
    """
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    param_summary = meta.get("param_summary", {})

    ncols = 4
    nrows = int(np.ceil(len(param_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.4))
    axes = np.array(axes).flatten()

    for ax, name in zip(axes, param_names):
        samples = _posterior_array(trace, name)
        ax.hist(samples, bins=50, density=True, color="steelblue",
                alpha=0.6, label="posterior")

        # Prior overlay if available
        if name in param_summary:
            ps = param_summary[name]
            mu = ps["mean"]   # use posterior mean as proxy for prior centre
            # better: use the initial param value — but we don't store it in meta yet,
            # so fall back to posterior mean (conservative; shows width comparison)
            prior_mu = mu
            prior_sigma = ps["prior_sigma"]
            x = np.linspace(samples.min(), samples.max(), 300)
            prior_pdf = (1 / (prior_sigma * np.sqrt(2 * np.pi)) *
                         np.exp(-0.5 * ((x - prior_mu) / prior_sigma) ** 2))
            ax.plot(x, prior_pdf, "r--", lw=1.2, label=f"prior σ={prior_sigma:.3g}")

        ax.set_title(_param_label(name), fontsize=8)
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mticker.NullFormatter())
        _autoscale_x(ax, samples)
        ax.legend(fontsize=7)

    for ax in axes[len(param_names):]:
        ax.set_visible(False)

    _suptitle(fig, meta, plot_name="Posterior marginals")
    _save(fig, outdir, "posterior")


def plot_shrinkage(trace, meta, stem, outdir):
    """
    Shrinkage = 1 - posterior_std / prior_sigma per param.
    Near 1 → data-dominated. Near 0 → prior not updated (weak likelihood or prior too tight).
    """
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    param_summary = meta.get("param_summary", {})
    if not param_summary:
        print("  shrinkage: no param_summary in metadata, skipping.")
        return

    shrinkages = []
    labels = []
    for name in param_names:
        if name not in param_summary:
            continue
        post_std = _posterior_array(trace, name).std()
        prior_sigma = param_summary[name]["prior_sigma"]
        shrinkages.append(1.0 - post_std / prior_sigma)
        labels.append(_param_label(name))

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.55), 4))
    colors = ["#d9534f" if s < 0.1 else "#5cb85c" if s > 0.5 else "#f0ad4e"
              for s in shrinkages]
    ax.bar(range(len(labels)), shrinkages, color=colors, width=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", label="0.5  (data-prior boundary)")
    ax.axhline(1.0, color="#2ecc71", lw=0.8, ls=":", alpha=0.7, label="1.0  (fully data-dominated)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("shrinkage  (1 − post_std / prior_σ)")
    # dynamic ylim: always show 0 and 1 as anchors, but expand if any value
    # is outside that range (e.g. shrinkage > 1 if posterior wider than prior)
    s_arr = np.array(shrinkages)
    lo = min(-0.15, float(s_arr.min()) - 0.05)
    hi = max(1.10,  float(s_arr.max()) + 0.05)
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=8)
    _suptitle(fig, meta, plot_name="Shrinkage")
    _save(fig, outdir, "shrinkage")


def plot_pair(trace, meta, stem, outdir, group="position"):
    """
    Bivariate scatter/KDE for a param subset.
    group: 'position'  → E, N, log_h params for each camera + antenna
           'angles'    → th, ph, ti params for each camera
           'all'       → all params (can be slow for many params)
    """
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)

    if group == "position":
        selected = [n for n in param_names
                    if any(n.endswith(s) for s in ("_e", "_n", "_log_h"))
                    or n in ("ant_e", "ant_n", "ant_log_h")]
    elif group == "angles":
        selected = [n for n in param_names
                    if any(n.endswith(s) for s in ("_th", "_ph", "_ti"))]
    else:
        selected = param_names

    if len(selected) < 2:
        print(f"  pair ({group}): fewer than 2 params matched, skipping.")
        return

    axes = az.plot_pair(trace, var_names=selected, kind="scatter",
                        marginals=True, divergences=False,
                        figsize=(max(6, len(selected) * 1.8),
                                 max(6, len(selected) * 1.8)))
    fig = axes.ravel()[0].get_figure()
    # relabel axes
    for ax in axes.flatten():
        xl = ax.get_xlabel()
        yl = ax.get_ylabel()
        if xl:
            ax.set_xlabel(_param_label(xl), fontsize=8)
        if yl:
            ax.set_ylabel(_param_label(yl), fontsize=8)
    _suptitle(fig, meta, plot_name=f"Pair plot — {group}", extra=f"pair={group}")
    _save(fig, outdir, f"pair_{group}")


def plot_acceptance(trace, meta, stem, outdir, window=100):
    """Rolling acceptance rate over the chain draw index."""
    # sample_stats.accepted is bool (chain, draw)
    accepted = trace.sample_stats.accepted.values  # (chains, draws)
    n_chains, n_draws = accepted.shape

    fig, ax = plt.subplots(figsize=(10, 3))
    x = np.arange(n_draws)
    for c in range(n_chains):
        roll = np.convolve(accepted[c].astype(float),
                           np.ones(window) / window, mode="valid")
        ax.plot(np.arange(len(roll)) + window // 2, roll,
                lw=1.0, label=f"chain {c}")

    overall = float(accepted.mean())
    ax.axhline(overall, color="k", ls="--", lw=0.8,
               label=f"overall {overall:.3f}")
    ax.axhspan(0.20, 0.40, color="green", alpha=0.08, label="ideal 0.20–0.40")
    ax.axhline(0.20, color="green", lw=0.7, ls="--", alpha=0.5)
    ax.axhline(0.40, color="green", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel(f"draw  (rolling window={window})")
    ax.set_ylabel("acceptance rate")
    # collect all rolling values to set a tight but complete y range
    all_rolls = []
    for c in range(n_chains):
        roll = np.convolve(accepted[c].astype(float),
                           np.ones(window) / window, mode="valid")
        all_rolls.extend(roll.tolist())
    lo = max(0.0,  float(np.min(all_rolls)) - 0.05)
    hi = min(1.0,  float(np.max(all_rolls)) + 0.05)
    ax.set_ylim(lo, hi)
    ax.legend()
    _suptitle(fig, meta, plot_name="Rolling acceptance rate")
    _save(fig, outdir, "acceptance")


def plot_sampler_stats(trace, meta, stem, outdir):
    """
    DEMetropolisZ sampler diagnostics: scaling and lambda timeseries.

    scaling: the adaptive step-size multiplier — should stabilise during
             tuning and stay flat during sampling.
    lambda:  the DE jump size (distance between two history points) —
             gives a sense of how far proposals are jumping in parameter space.
    Both are from sample_stats, which for DEMetropolisZ contains:
    ['accept', 'accepted', 'lambda', 'scaling'].
    """
    ss = trace.sample_stats
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    for attr, ax, ylabel in [
        ("scaling", axes[0], "scaling"),
        ("lambda",  axes[1], "lambda (DE jump size)"),
    ]:
        if not hasattr(ss, attr):
            ax.set_visible(False)
            continue
        vals = getattr(ss, attr).values  # (chains, draws)
        if vals.ndim == 1:
            vals = vals[None, :]
        for c in range(vals.shape[0]):
            ax.plot(vals[c], lw=0.6, alpha=0.8, label=f"chain {c}")
        ax.set_ylabel(ylabel)
        _autoscale_y(ax, vals.flatten())
        ax.legend(fontsize=7)

    axes[-1].set_xlabel("draw")
    _suptitle(fig, meta, plot_name="Sampler stats (scaling & lambda)")
    _save(fig, outdir, "sampler_stats")


def plot_prior_predictive(trace, meta, stem, outdir, n_samples=2000):
    """
    Prior predictive distribution vs posterior for every param.

    Draws n_samples from Normal(prior_mu, prior_sigma) for each param and
    overlays the resulting prior predictive histogram against the posterior
    KDE.  This makes it visually clear which params are data-dominated
    (prior and posterior differ markedly) vs prior-dominated (they overlap).

    Requires prior_mu in param_summary (stored by eigsep_terrain_pymc.py).
    Falls back to the posterior mean as prior centre if absent.
    """
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    param_summary = meta.get("param_summary", {})
    if not param_summary:
        print("  prior_predictive: no param_summary in metadata, skipping.")
        return

    rng = np.random.default_rng(meta.get("seed", 0))

    ncols = 4
    nrows = int(np.ceil(len(param_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.6))
    axes = np.array(axes).flatten()

    for ax, name in zip(axes, param_names):
        post = _posterior_array(trace, name)
        ps = param_summary.get(name, {})
        prior_mu    = ps.get("prior_mu", float(post.mean()))
        prior_sigma = ps.get("prior_sigma", None)

        if prior_sigma is None:
            ax.set_title(_param_label(name) + "\n(no prior info)", fontsize=8)
            ax.hist(post, bins=40, density=True, color="steelblue", alpha=0.7)
            continue

        prior_samples = rng.normal(prior_mu, prior_sigma, size=n_samples)

        # shared x range covering both distributions
        lo = min(post.min(), prior_samples.min())
        hi = max(post.max(), prior_samples.max())
        bins = np.linspace(lo, hi, 50)

        ax.hist(prior_samples, bins=bins, density=True, color="#e07b54",
                alpha=0.45, label="prior")
        ax.hist(post, bins=bins, density=True, color="steelblue",
                alpha=0.65, label="posterior")
        ax.axvline(prior_mu, color="#c0392b", lw=1.0, ls="--", label="prior μ")
        ax.axvline(float(post.mean()), color="#1a5276", lw=1.0, ls="--",
                   label="post μ")

        ax.set_title(_param_label(name), fontsize=8)
        ax.yaxis.set_major_formatter(mticker.NullFormatter())
        # zoom x to posterior bulk — prior tails can be orders of magnitude wider
        _autoscale_x(ax, post)
        ax.legend(fontsize=6, ncol=2)

    for ax in axes[len(param_names):]:
        ax.set_visible(False)

    _suptitle(fig, meta, plot_name="Prior predictive vs posterior")
    _save(fig, outdir, "prior_predictive")


def plot_step_size(trace, meta, stem, outdir):
    """
    Proposal step size diagnostics — three panels per param group:

      1. Effective step size vs posterior std
         Bar chart of  step / post_std  per param.
         Well-tuned MH target ≈ 0.3–0.6 (grey band).
         Too small → slow diffusion.  Too large → high rejection.

      2. Absolute effective step size vs prior sigma
         Shows how the proposal compares to the prior width.
         step ≫ prior_sigma means proposals jump outside the prior bulk.

      3. Tuned scaling value (single number from metadata, shown as text
         annotation on panel 1 for reference).

    Requires effective_step and prior_sigma in param_summary.
    """
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    param_summary = meta.get("param_summary", {})
    if not param_summary:
        print("  step_size: no param_summary in metadata, skipping.")
        return

    names, labels, step_over_post, step_over_prior = [], [], [], []
    for name in param_names:
        ps = param_summary.get(name, {})
        eff_step    = ps.get("effective_step")
        prior_sigma = ps.get("prior_sigma")
        post_std    = _posterior_array(trace, name).std()
        if eff_step is None or prior_sigma is None:
            continue
        names.append(name)
        labels.append(_param_label(name))
        step_over_post.append(eff_step / post_std  if post_std  > 0 else np.nan)
        step_over_prior.append(eff_step / prior_sigma if prior_sigma > 0 else np.nan)

    if not names:
        print("  step_size: effective_step missing from all params, skipping.")
        return

    x = np.arange(len(names))
    tuned_scaling = meta.get("step", {}).get("tuned_scaling",
                    meta.get("step", {}).get("scaling", "?"))

    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=(max(7, len(names) * 0.55), 7),
                                   sharex=True)

    # Panel 1: step / post_std
    colors1 = []
    for v in step_over_post:
        if np.isnan(v):
            colors1.append("gray")
        elif v < 0.1:
            colors1.append("#d9534f")   # too small — red
        elif v > 1.5:
            colors1.append("#e67e22")   # too large — orange
        else:
            colors1.append("#5cb85c")   # roughly in range — green
    ax1.bar(x, step_over_post, color=colors1, width=0.7)
    ax1.axhspan(0.3, 0.6, color="gray", alpha=0.15, label="target range 0.3–0.6")
    ax1.axhline(0.3, color="gray", lw=0.8, ls="--")
    ax1.axhline(0.6, color="gray", lw=0.8, ls="--")
    ax1.set_ylabel("step / post_std")
    finite1 = [v for v in step_over_post if not np.isnan(v)]
    ax1.set_ylim(bottom=0, top=max(max(finite1) * 1.15, 0.65) if finite1 else 1.0)
    ax1.legend(fontsize=8)
    ax1.text(0.99, 0.97, f"tuned scaling = {tuned_scaling}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=8,
             color="dimgray")

    # Panel 2: step / prior_sigma
    colors2 = ["#d9534f" if v > 0.5 else "#5cb85c" if v < 0.15 else "#5bc0de"
               for v in step_over_prior]
    ax2.bar(x, step_over_prior, color=colors2, width=0.7)
    ax2.axhline(0.15, color="gray", lw=0.8, ls="--", label="0.15 reference")
    ax2.set_ylabel("step / prior_σ")
    finite2 = [v for v in step_over_prior if not np.isnan(v)]
    ax2.set_ylim(bottom=0, top=max(max(finite2) * 1.15, 0.20) if finite2 else 1.0)
    ax2.legend(fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    fig.align_ylabels([ax1, ax2])
    _suptitle(fig, meta, plot_name="Step size vs posterior & prior widths")
    _save(fig, outdir, "step_size")


def plot_prior_sensitivity(trace, meta, stem, outdir):
    """
    Prior sensitivity summary — two panels:

      1. Z-score of posterior mean relative to prior:
            z = (post_mean - prior_mu) / prior_sigma
         Large |z| means the posterior has been pulled far from the prior
         centre, indicating that the likelihood is informative on that param.
         |z| < 0.5 (grey band) suggests the prior is dominating or the
         likelihood has little information there.

      2. Posterior contraction ratio:
            ratio = post_std / prior_sigma
         Near 1 → no contraction (prior = posterior, likelihood uninformative).
         Near 0 → strong contraction (data-dominated).
         Plotted on a log scale so both extremes are visible.

    Together these two panels identify four regimes:
      high |z|, low ratio  → data-dominated, well-identified param
      low  |z|, low ratio  → data tightens posterior near prior (consistent)
      high |z|, high ratio → data shifts but doesn't tighten (weak signal)
      low  |z|, high ratio → prior dominates entirely
    """
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    param_summary = meta.get("param_summary", {})
    if not param_summary:
        print("  prior_sensitivity: no param_summary in metadata, skipping.")
        return

    names, labels, zscores, ratios = [], [], [], []
    for name in param_names:
        ps = param_summary.get(name, {})
        prior_mu    = ps.get("prior_mu")
        prior_sigma = ps.get("prior_sigma")
        post_mean   = ps.get("mean")
        post_std    = _posterior_array(trace, name).std()
        if prior_mu is None or prior_sigma is None:
            continue
        names.append(name)
        labels.append(_param_label(name))
        zscores.append((post_mean - prior_mu) / prior_sigma)
        ratios.append(post_std / prior_sigma if prior_sigma > 0 else np.nan)

    if not names:
        print("  prior_sensitivity: prior_mu missing from all params, skipping.")
        return

    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=(max(7, len(names) * 0.55), 7),
                                   sharex=True)

    # Panel 1: z-score
    colors1 = ["#d9534f" if abs(z) > 2 else "#f0ad4e" if abs(z) > 0.5 else "#5cb85c"
               for z in zscores]
    ax1.bar(x, zscores, color=colors1, width=0.7)
    ax1.axhspan(-0.5, 0.5, color="gray", alpha=0.12, label="|z| < 0.5 (prior-dominated)")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.axhline( 2, color="#c0392b", lw=0.8, ls="--", alpha=0.6, label="|z| = 2")
    ax1.axhline(-2, color="#c0392b", lw=0.8, ls="--", alpha=0.6)
    ax1.set_ylabel("z-score  (post_mean − prior_μ) / prior_σ")
    _autoscale_y(ax1, zscores, margin=0.1, symm=True)
    ax1.legend(fontsize=8)

    # Panel 2: contraction ratio (log scale)
    colors2 = ["#5cb85c" if r < 0.5 else "#f0ad4e" if r < 0.85 else "#d9534f"
               for r in ratios]
    ax2.bar(x, ratios, color=colors2, width=0.7)
    ax2.axhline(1.0, color="k", lw=0.8, ls="--", label="ratio = 1  (no contraction)")
    ax2.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6, label="ratio = 0.5")
    ax2.set_ylabel("contraction  post_std / prior_σ")
    finite_r = [r for r in ratios if not np.isnan(r)]
    ax2.set_ylim(0, max(1.1, max(finite_r) * 1.15) if finite_r else 1.2)
    ax2.legend(fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    fig.align_ylabels([ax1, ax2])
    _suptitle(fig, meta, plot_name="Prior sensitivity (z-score & contraction)")
    _save(fig, outdir, "prior_sensitivity")

def plot_canyon_overlay(trace, meta, stem, outdir, img_glob=None, cache_file=None):
    fig, ax = plt.subplots()

    cache_file = cache_file or DEFAULT_CACHE_FILE
    img_glob = img_glob or DEFAULT_IMG_GLOB
    dem = DEM(cache_file=cache_file)

    imgmeta = {k: dict(v) for k, v in DEFAULT_META.items()}

    files = sorted(glob.glob(img_glob))
    imgs = [HorizonImage(f, imgmeta, px_dist=30) for f in files]
    imgs = [img for img in imgs if img.key in imgmeta]

    alpha = 0.02
    _terrain_plot(dem, ax=ax)
    plt.plot(np.asarray(trace.posterior['ant_e']).flatten(), 
             np.asarray(trace.posterior['ant_n']).flatten(), 
             'k.', alpha=alpha, label=f'antenna')
    colors = plt.cm.tab10.colors
    for i, img in enumerate(imgs):
        try:
            plt.plot(np.asarray(trace.posterior[f'{img.key}_e']).flatten(), 
                     np.asarray(trace.posterior[f'{img.key}_n']).flatten(), 
                     '.', alpha=alpha, label=f'img {i}', color=colors[i % len(colors)]);
        except(KeyError):
            plt.plot(np.asarray(trace.posterior['e']).flatten(), 
                     np.asarray(trace.posterior['n']).flatten(), 
                     '.', alpha=alpha, label=f'img {i}');
    leg = plt.legend()
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    ax.set_title('MCMC steps in the canyon')

    _suptitle(fig, meta, plot_name="Canyon Overlay")
    _save(fig, outdir, "canyon_overlay")

# ── main ──────────────────────────────────────────────────────────────────────

def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("nc_file", help="Path to the ArviZ .nc trace file")

    # plot toggles
    ap.add_argument("--trace",      action="store_true", help="Chain timeseries + marginal KDE")
    ap.add_argument("--rank",       action="store_true", help="Rank plots (multi-chain mixing)")
    ap.add_argument("--autocorr",   action="store_true", help="Autocorrelation per param")
    ap.add_argument("--posterior",  action="store_true", help="Marginals with prior overlay")
    ap.add_argument("--shrinkage",  action="store_true", help="Shrinkage bar chart")
    ap.add_argument("--pair",       action="store_true", help="Bivariate scatter (position group)")
    ap.add_argument("--pair-angles",action="store_true", help="Bivariate scatter (angles group)")
    ap.add_argument("--pair-all",   action="store_true", help="Bivariate scatter (all params)")
    ap.add_argument("--acceptance", action="store_true", help="Rolling acceptance rate")
    ap.add_argument("--sampler-stats", action="store_true", help="DEMetropolisZ scaling and lambda timeseries")
    ap.add_argument("--prior-predictive",  action="store_true",
                    help="Prior predictive vs posterior histogram per param")
    ap.add_argument("--step-size",         action="store_true",
                    help="Effective step size vs post_std and prior_sigma")
    ap.add_argument("--prior-sensitivity", action="store_true",
                    help="Z-score and contraction ratio per param")
    ap.add_argument("--canyon-overlay", action="store_true",
                    help="Terrain plot with posterior E/N scatter per image + antenna")
    ap.add_argument("--all",        action="store_true", help="Enable every plot")

    # options
    ap.add_argument("--show", action="store_true",
                    help="Display figures interactively instead of saving")
    ap.add_argument("--window", type=int, default=100,
                    help="Rolling window size for acceptance plot (default: 100)")
    ap.add_argument("--img-glob", default=DEFAULT_IMG_GLOB,
                    help="Used only by --canyon-overlay, to locate per-image keys.")
    ap.add_argument("--cache-file", default=DEFAULT_CACHE_FILE,
                    help="Used only by --canyon-overlay, for the DEM terrain plot.")

    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    trace, meta, stem, outdir = _load(args.nc_file)

    do_all = args.all
    show   = args.show

    print(f"Loaded: {args.nc_file}")
    if meta:
        print(f"  seed={meta.get('seed')}  "
              f"chains={meta.get('sampling',{}).get('chains')}  "
              f"draws={meta.get('sampling',{}).get('draws')}  "
              f"accept={meta.get('accepted_mean', float('nan')):.3f}")

    plots_run = 0

    if do_all or args.trace:
        print("Plotting: trace")
        plot_trace(trace, meta, stem, outdir)
        plots_run += 1

    if do_all or args.rank:
        print("Plotting: rank")
        plot_rank(trace, meta, stem, outdir)
        plots_run += 1

    if do_all or args.autocorr:
        print("Plotting: autocorr")
        plot_autocorr(trace, meta, stem, outdir)
        plots_run += 1

    if do_all or args.posterior:
        print("Plotting: posterior")
        plot_posterior(trace, meta, stem, outdir)
        plots_run += 1

    if do_all or args.shrinkage:
        print("Plotting: shrinkage")
        plot_shrinkage(trace, meta, stem, outdir)
        plots_run += 1

    if do_all or args.pair:
        print("Plotting: pair (position)")
        plot_pair(trace, meta, stem, outdir, group="position")
        plots_run += 1

    if do_all or args.pair_angles:
        print("Plotting: pair (angles)")
        plot_pair(trace, meta, stem, outdir, group="angles")
        plots_run += 1

    if args.pair_all:
        print("Plotting: pair (all) — may be slow")
        plot_pair(trace, meta, stem, outdir, group="all")
        plots_run += 1

    if do_all or args.acceptance:
        print("Plotting: acceptance")
        plot_acceptance(trace, meta, stem, outdir, window=args.window)
        plots_run += 1

    if do_all or args.sampler_stats:
        print("Plotting: sampler_stats")
        plot_sampler_stats(trace, meta, stem, outdir)
        plots_run += 1

    if do_all or args.prior_predictive:
        print("Plotting: prior_predictive")
        plot_prior_predictive(trace, meta, stem, outdir)
        plots_run += 1

    if do_all or args.step_size:
        print("Plotting: step_size")
        plot_step_size(trace, meta, stem, outdir)
        plots_run += 1

    if do_all or args.prior_sensitivity:
        print("Plotting: prior_sensitivity")
        plot_prior_sensitivity(trace, meta, stem, outdir)
        plots_run += 1

    if do_all or args.canyon_overlay:
        print("Plotting: canyon_overlay")
        plot_canyon_overlay(trace, meta, stem, outdir,
                            img_glob=args.img_glob, cache_file=args.cache_file)
        plots_run += 1

    if plots_run == 0:
        print("No plots selected. Pass --all or one or more plot flags. See --help.")
        return 1

    if show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())