#!/usr/bin/env python
"""
run from img folder

Interactive matplotlib sliders for e, n, u, th, ph, ti, f — live-updates the
ray-traced horizon overlay on the image (left) alongside an E/N terrain plot
(right) with a marker showing the current camera position. Use this to
hand-tune a starting point, then feed the printed params into fit_image.py's
--e/--n or a --map-file, or just eyeball the fit directly.

Each slider has a paired "range" textbox (lo, hi) so you can widen or
narrow its span on the fly during the session. The "u" slider's lower bound
is clamped live to dem.interp_alt(e, n) — you can never drag the camera
below the ground at its current E/N.

Fill in DEFAULT_META / IMG_GLOB / DEFAULT_PRMS_U_BY_KEY below before running.

Usage:
  python tune_image.py --which 0
"""
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, dtype_r
from eigsep_terrain.img_defaults import load_defaults


# ── per-image defaults now live in defaults.json (edit that file, not this
#    script, when a starting value changes — fit_image.py / plot_image_fit.py
#    load the same file) ──────────────────────────────────────────────────
IMG_GLOB, CACHE_FILE, DEFAULT_META, DEFAULT_PRMS_U_BY_KEY, IMG_KEYS = load_defaults('/Users/komalkaur/Desktop/eigsep_stuff/eigsep_terrain/eigsep_terrain/defaults.json')

# Absolute (min, max) bounds for each slider — same across every image,
# independent of that image's starting value. Fill these in. You can also
# edit lo/hi live via the "range" textboxes in the UI. "u"'s lo is
# overridden live by the DEM regardless of what's set here.
SLIDER_RANGES = {
    "e": (1500, 1900), 
    "n": (2000, 2300), 
    "u": (1600, 1900),
    "th": (0, np.pi), 
    "ph":(0, 2*np.pi), 
    "ti": (-np.pi, np.pi),
    "f": (500, 15000),
}


def terrain_plot(dem=DEM(cache_file='marjum_dem2.npz'), ax=None, xlabel=True, ylabel=True,
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


def find_img_file(which, img_glob):
    key = IMG_KEYS[which]
    files = sorted(glob.glob(img_glob))
    matches = [f for f in files if os.path.basename(f).split("_")[-1].split(".")[0] == key]
    if not matches:
        raise FileNotFoundError(f"No file matching key {key!r} via {img_glob!r}")
    return matches[0]


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=int, required=True,
                    choices=list(range(len(IMG_KEYS))))
    ap.add_argument("--img-glob", default=IMG_GLOB)
    ap.add_argument("--cache-file", default=CACHE_FILE)
    ap.add_argument("--stride", type=int, default=10,
                    help="Pixel stride for the live ray grid (bigger = faster).")
    ap.add_argument("--fine-delta", type=float, default=0.25)
    ap.add_argument("--n-rays", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--e", type=float, default=None)
    ap.add_argument("--n", type=float, default=None)
    ap.add_argument("--u", type=float, default=None)
    ap.add_argument("--th", type=float, default=None)
    ap.add_argument("--ph", type=float, default=None)
    ap.add_argument("--ti", type=float, default=None)
    ap.add_argument("--f", type=float, default=None)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    key = IMG_KEYS[args.which]

    dem = DEM(cache_file=args.cache_file)
    meta = {k: dict(v) for k, v in DEFAULT_META.items()}
    img_file = find_img_file(args.which, args.img_glob)
    img = HorizonImage(img_file, meta, px_smooth=150, px_dist=30)

    default = DEFAULT_PRMS_U_BY_KEY.get(key)
    cli_overrides = [args.e, args.n, args.u, args.th, args.ph, args.ti, args.f]
    names = ["e", "n", "u", "th", "ph", "ti", "f"]
    if default is None:
        if any(v is None for v in cli_overrides):
            raise ValueError(
                f"No default prms set for key {key!r} in "
                f"DEFAULT_PRMS_U_BY_KEY, and not all of --e/--n/--u/--th/"
                f"--ph/--ti/--f were passed. Fill in one or the other."
            )
        default = tuple(cli_overrides)
    else:
        default = tuple(c if c is not None else d
                        for c, d in zip(cli_overrides, default))
    start = dict(zip(names, default))

    if any(v is None for v in SLIDER_RANGES.values()):
        missing = [k for k, v in SLIDER_RANGES.items() if v is None]
        raise ValueError(
            f"SLIDER_RANGES not filled in for: {missing}. "
            f"Set each to a (delta_minus, delta_plus) tuple, e.g. (-30, 30)."
        )

    ys = np.arange(0, img.npix_y, args.stride)
    xs = np.arange(0, img.npix_x, args.stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    x_px, y_px = yy.ravel(), xx.ravel()
    actual_sky = img.sky_mask[np.ix_(ys, xs)] > 0.5

    # ── layout: image | terrain, sliders below ──────────────────────────
    fig = plt.figure(figsize=(15, 8))
    ax_img = fig.add_axes([0.05, 0.48, 0.42, 0.48])
    ax_terrain = fig.add_axes([0.54, 0.48, 0.42, 0.48])

    ax_img.imshow(img.img, origin="lower")
    ax_img.contour(xx, yy, actual_sky.astype(float), levels=[0.5],
                   colors="cyan", linewidths=1.5)
    ax_img.set_title("horizon fit", fontsize=10)

    # terrain_im = terrain_plot(ax=ax_terrain, erng_m=SLIDER_RANGES["e"], nrng_m=SLIDER_RANGES["n"])
    terrain_im = terrain_plot(ax=ax_terrain, erng_m=(1000, 2500), nrng_m=(1500, 3000))

    # ax_terrain.set_xlim(1000, 2500)
    # ax_terrain.set_ylim(1500, 3000)
    ax_terrain.set_title("E / N position", fontsize=10)

    fig.suptitle(f"Image {key}", fontsize=13)

    contour_holder = {"cs": None}
    marker_holder = {"pt": None}
    arrow_holder = {"ann": None}
    colorbar_marker_holder = {"line": None, "star": None}

    def compute_and_draw(prms):
        img.set_prms(tuple(prms[k] for k in names))
        rays = img.get_rays(pixels=(x_px, y_px), dtype=dtype_r)
        r = img.ray_distance(dem, rays, dtype=dtype_r, fine_delta=args.fine_delta)
        model_sky = np.isnan(r).reshape(yy.shape)
        if contour_holder["cs"] is not None:
            contour_holder["cs"].remove()
        contour_holder["cs"] = ax_img.contour(
            xx, yy, model_sky.astype(float), levels=[0.5],
            colors="red", linewidths=1.5,
        )

    cb_ax = terrain_im.colorbar.ax  # cache once; avoid re-fetching per update

    def update_marker():
        e, n = sliders["e"].val, sliders["n"].val
        if marker_holder["pt"] is not None:
            marker_holder["pt"].remove()
        marker_holder["pt"] = ax_terrain.plot(
            e, n, marker="*", color="red", markeredgecolor="black",
            markersize=16, zorder=5,
        )[0]

        # boresight direction (depends on th, ph; independent of roll ti)
        center_ray = img.get_rays(
            pixels=(np.array([img.npix_y // 2]), np.array([img.npix_x // 2])),
            dtype=dtype_r,
        )
        dE, dN = float(center_ray[0][0]), float(center_ray[1][0])
        norm = np.hypot(dE, dN)
        if norm > 1e-6:
            dE, dN = dE / norm, dN / norm
        arrow_len = 0.15 * (SLIDER_RANGES["e"][1] - SLIDER_RANGES["e"][0])
        if arrow_holder["ann"] is not None:
            arrow_holder["ann"].remove()
        arrow_holder["ann"] = ax_terrain.annotate(
            "", xy=(e + arrow_len * dE, n + arrow_len * dN), xytext=(e, n),
            arrowprops=dict(arrowstyle="->", color="red", lw=2), zorder=5,
        )

        ground = float(dem.interp_alt(e, n))
        cam_u = float(sliders["u"].val)
        if colorbar_marker_holder["line"] is not None:
            colorbar_marker_holder["line"].remove()
        if colorbar_marker_holder["star"] is not None:
            colorbar_marker_holder["star"].remove()
        colorbar_marker_holder["line"] = cb_ax.axhline(
            ground, color="red", lw=2.5, zorder=5,
        )
        xmid = sum(cb_ax.get_xlim()) / 2
        colorbar_marker_holder["star"] = cb_ax.plot(
            xmid, cam_u, marker="*", color="red", markeredgecolor="black",
            markersize=16, zorder=6, clip_on=False,
        )[0]
        print(f"[colorbar marker] ground={ground:.2f}  cam_u={cam_u:.2f}")
        fig.canvas.draw()  # force full redraw (not just draw_idle) as a safety net

    # ── sliders panel ─────────────────────────────────────────────────
    sliders = {}
    reset_btns = {}
    textboxes = {}
    lo_boxes = {}
    hi_boxes = {}

    row_h = 0.045
    top_y = 0.44
    col_slider = [0.08, 0.36]
    col_value  = [0.46, 0.08]
    col_lo     = [0.56, 0.08]
    col_hi     = [0.66, 0.08]
    col_reset  = [0.76, 0.07]

    fig.text(col_slider[0], top_y + row_h * 0.6, "parameter", fontsize=9, weight="bold")
    fig.text(col_value[0],  top_y + row_h * 0.6, "value",     fontsize=9, weight="bold")
    fig.text(col_lo[0],     top_y + row_h * 0.6, "lo",        fontsize=9, weight="bold")
    fig.text(col_hi[0],     top_y + row_h * 0.6, "hi",        fontsize=9, weight="bold")

    for i, (name, (lo, hi)) in enumerate(SLIDER_RANGES.items()):
        row_y = top_y - (i + 1) * row_h
        s0 = min(max(start[name], lo), hi)
        if s0 != start[name]:
            print(f"WARNING: default {name}={start[name]} outside "
                  f"SLIDER_RANGES[{name!r}]=({lo}, {hi}); clamping.")

        s_ax = fig.add_axes([col_slider[0], row_y, col_slider[1], row_h * 0.5])
        sliders[name] = Slider(s_ax, name, lo, hi, valinit=s0)
        sliders[name].valtext.set_visible(False)  # avoid overlap with our value TextBox

        t_ax = fig.add_axes([col_value[0], row_y, col_value[1], row_h * 0.5])
        textboxes[name] = TextBox(t_ax, "", initial=f"{s0:.4f}")

        lo_ax = fig.add_axes([col_lo[0], row_y, col_lo[1], row_h * 0.5])
        lo_boxes[name] = TextBox(lo_ax, "", initial=f"{lo:.4f}")

        hi_ax = fig.add_axes([col_hi[0], row_y, col_hi[1], row_h * 0.5])
        hi_boxes[name] = TextBox(hi_ax, "", initial=f"{hi:.4f}")

        r_ax = fig.add_axes([col_reset[0], row_y, col_reset[1], row_h * 0.5])
        reset_btns[name] = Button(r_ax, "reset")

    def _apply_range(name, new_lo, new_hi, redraw=True):
        """Update a slider's valmin/valmax, re-clamp its value, refresh
        its axis limits + the paired lo/hi textboxes."""
        s = sliders[name]
        if new_lo >= new_hi:
            lo_boxes[name].set_val(f"{s.valmin:.4f}")
            hi_boxes[name].set_val(f"{s.valmax:.4f}")
            return
        clamped_val = min(max(s.val, new_lo), new_hi)
        s.valmin = new_lo
        s.valmax = new_hi
        s.ax.set_xlim(new_lo, new_hi)
        lo_boxes[name].set_val(f"{new_lo:.4f}")
        hi_boxes[name].set_val(f"{new_hi:.4f}")
        if clamped_val != s.val or redraw:
            s.set_val(clamped_val)  # fires on_changed
        else:
            fig.canvas.draw_idle()

    def refresh_u_min():
        ground = float(dem.interp_alt(sliders["e"].val, sliders["n"].val))
        _apply_range("u", ground, sliders["u"].valmax)

    def on_change(_):
        prms = {name: s.val for name, s in sliders.items()}
        compute_and_draw(prms)
        update_marker()
        logL = img.horizon_ray_logL(dem, n_rays=args.n_rays, eps=args.eps,
                                    fine_delta=args.fine_delta)
        ax_img.set_title(f"horizon fit — logL={logL:.2f}", fontsize=10)
        fig.canvas.draw_idle()
        print("prms_u =", tuple(round(prms[k], 4) for k in names),
              f" logL={logL:.2f}")

    _updating = {"flag": False}

    def on_slider_change(name):
        def cb(val):
            if _updating["flag"]:
                return
            _updating["flag"] = True
            textboxes[name].set_val(f"{val:.4f}")
            _updating["flag"] = False
            if name in ("e", "n"):
                refresh_u_min()  # may itself trigger on_change via 'u'
            on_change(val)
        return cb

    for name, s in sliders.items():
        s.on_changed(on_slider_change(name))

    def on_text_submit(name):
        def cb(text):
            if _updating["flag"]:
                return
            try:
                v = float(text)
            except ValueError:
                return
            v = min(max(v, sliders[name].valmin), sliders[name].valmax)
            _updating["flag"] = True
            sliders[name].set_val(v)
            _updating["flag"] = False
            textboxes[name].set_val(f"{v:.4f}")
            if name in ("e", "n"):
                refresh_u_min()
            on_change(v)
        return cb

    for name, t in textboxes.items():
        t.on_submit(on_text_submit(name))

    def on_lo_submit(name):
        def cb(text):
            try:
                v = float(text)
            except ValueError:
                lo_boxes[name].set_val(f"{sliders[name].valmin:.4f}")
                return
            _apply_range(name, v, sliders[name].valmax)
        return cb

    def on_hi_submit(name):
        def cb(text):
            try:
                v = float(text)
            except ValueError:
                hi_boxes[name].set_val(f"{sliders[name].valmax:.4f}")
                return
            _apply_range(name, sliders[name].valmin, v)
        return cb

    for name in SLIDER_RANGES:
        lo_boxes[name].on_submit(on_lo_submit(name))
        hi_boxes[name].on_submit(on_hi_submit(name))

    for name, b in reset_btns.items():
        b.on_clicked(lambda event, n=name: sliders[n].reset())

    # ── top-right utility buttons ────────────────────────────────────
    reset_ax = fig.add_axes([0.87, 0.075, 0.10, 0.035])
    reset_btn = Button(reset_ax, "Reset all")

    def on_reset(_):
        for s in sliders.values():
            s.reset()

    reset_btn.on_clicked(on_reset)

    img_xlim0, img_ylim0 = ax_img.get_xlim(), ax_img.get_ylim()
    reset_zoom_ax = fig.add_axes([0.87, 0.03, 0.10, 0.035])
    reset_zoom_btn = Button(reset_zoom_ax, "Reset zoom")

    def on_reset_zoom(_):
        ax_img.set_xlim(img_xlim0)
        ax_img.set_ylim(img_ylim0)
        fig.canvas.draw_idle()

    reset_zoom_btn.on_clicked(on_reset_zoom)

    # ── click-to-focus + arrow-key stepping ──────────────────────────
    active = {"name": None}
    for name, s in sliders.items():
        fig.canvas.mpl_connect(
            "button_press_event",
            lambda event, n=name, a=s.ax: active.__setitem__("name", n)
            if event.inaxes == a else None,
        )

    def on_key(event):
        n = active["name"]
        if n is None:
            return
        s = sliders[n]
        step = (s.valmax - s.valmin) / 100
        if event.key == "right":
            s.set_val(min(s.val + step, s.valmax))
        elif event.key == "left":
            s.set_val(max(s.val - step, s.valmin))

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ── scroll-to-zoom on the image axis, centered on cursor ─────────
    def on_scroll(event):
        if event.inaxes != ax_img or event.xdata is None:
            return
        factor = 0.9 if event.button == "up" else 1.1
        xlo, xhi = ax_img.get_xlim()
        ylo, yhi = ax_img.get_ylim()
        xd, yd = event.xdata, event.ydata
        ax_img.set_xlim(xd - (xd - xlo) * factor, xd + (xhi - xd) * factor)
        ax_img.set_ylim(yd - (yd - ylo) * factor, yd + (yhi - yd) * factor)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    refresh_u_min()
    compute_and_draw(start)
    update_marker()
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())