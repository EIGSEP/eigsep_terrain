#!/usr/bin/env python
"""
run from img folder (2026_imgs/)

Interactive antenna-pixel picker. Click the antenna in each photo; the pixel
you clicked is saved straight into defaults.json's per-image ant_px.

Convention: ant_px is stored and used here exactly as clicked on the
*original, unflipped* photo (row 0 = top) -- confirmed against known-good
picks (e.g. 2210, 2245) and does NOT need any vertical flip. This tool never
touches HorizonImage/segmentation, so it starts fast and has no torch/
transformers/pymc dependency.

Controls:
  - Click on the image to place a pending pick (red star). Click again to
    move it.
  - "Save & Next": writes the pending pick to defaults.json for the current
    image, then advances. No-op (just advances) if you haven't clicked --
    use "Skip" instead to be explicit about leaving an image unchanged.
  - "Skip": leave this image's ant_px untouched, advance to the next.
  - "Prev" / "Next": navigate without saving.
  - "Full image" / "Reset zoom": toggle between the full photo and a crop
    centered on the currently-stored ant_px (useful once you know roughly
    where to look).
  - Scroll to zoom in/out (centered on the cursor); right-click-drag to
    pan. No need to touch the matplotlib toolbar -- left-click stays free
    for picking the whole time.
  - The blue circle (if shown) is the ant_px currently in defaults.json,
    for reference -- it's what you're about to overwrite.

Usage:
  python pick_antenna_pixel.py                  # all images, in order
  python pick_antenna_pixel.py --which 2213      # start at a specific key
  python pick_antenna_pixel.py --keys 2213 2234 2210
"""
import argparse
import glob
import json
import os
import re
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.widgets import Button

from zoom_pan import enable_zoom_pan

DEFAULTS_PATH = "/Users/komalkaur/Desktop/eigsep_stuff/eigsep_terrain/eigsep_terrain/defaults.json"
IMG_GLOB = "*.jpg"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def atomic_write(text, path):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=d, prefix=".defaults_", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def update_ant_px_in_file(path, key, new_ant_px):
    """Surgically replace one image's ant_px array in the JSON *text*,
    leaving every other line byte-identical -- avoids reformatting the
    whole file (which a plain json.dump would do) into a noisy diff."""
    with open(path) as f:
        text = f.read()
    pattern = re.compile(r'("' + re.escape(key) + r'":\s*\{"ant_px":\s*)\[[^\]]*\]')
    replacement = r"\g<1>" + f"[{int(round(new_ant_px[0]))}, {int(round(new_ant_px[1]))}]"
    new_text, n = pattern.subn(replacement, text, count=1)
    if n != 1:
        raise ValueError(
            f"Expected exactly 1 ant_px entry for key {key!r} in {path}, found {n}. "
            "File may have been reformatted -- refusing to write."
        )
    json.loads(new_text)  # validate before touching the real file
    atomic_write(new_text, path)


def find_img_file(key, img_glob):
    files = sorted(glob.glob(img_glob))
    matches = [f for f in files if os.path.basename(f).split("_")[-1].split(".")[0] == key]
    if not matches:
        raise FileNotFoundError(f"No file matching key {key!r} via {img_glob!r}")
    return matches[0]


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defaults", default=DEFAULTS_PATH)
    ap.add_argument("--img-glob", default=IMG_GLOB)
    ap.add_argument("--keys", nargs="+", default=None,
                    help="Only cycle through these keys (default: all in defaults.json).")
    ap.add_argument("--which", default=None,
                    help="Start at this key instead of the first one.")
    ap.add_argument("--win-frac", type=float, default=0.28,
                    help="Half-crop-window as a fraction of max(npix_x, npix_y) for the initial zoomed view.")
    ap.add_argument("--backup", action="store_true", default=True)
    ap.add_argument("--no-backup", dest="backup", action="store_false")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    data = load_json(args.defaults)
    images = data["images"]
    keys = args.keys or list(images.keys())
    if args.which is not None:
        if args.which not in keys:
            raise ValueError(f"--which {args.which!r} not in key list {keys}")
        start = keys.index(args.which)
        keys = keys[start:] + keys[:start]

    if args.backup:
        bak_path = args.defaults + ".bak"
        if not os.path.exists(bak_path):
            shutil.copyfile(args.defaults, bak_path)
            print(f"Backup written to {bak_path}")

    state = {
        "idx": 0,
        "pending": None,      # (x, y) in raw/unflipped pixel coords, or None
        "pending_marker": None,
        "current_marker": None,
        "zoomed": True,
        "img_cache": {},
    }

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_axes([0.06, 0.14, 0.90, 0.80])
    status_ax = fig.add_axes([0.06, 0.02, 0.62, 0.07])
    status_ax.axis("off")
    status_txt = status_ax.text(0, 0.5, "", va="center", fontsize=9)

    enable_zoom_pan(fig, ax, lambda: (state["npix_x"], state["npix_y"]))

    def set_status(msg):
        status_txt.set_text(msg)
        fig.canvas.draw_idle()

    def get_img(key):
        if key not in state["img_cache"]:
            path = find_img_file(key, args.img_glob)
            state["img_cache"][key] = imread(path)
        return state["img_cache"][key]

    def current_key():
        return keys[state["idx"]]

    def zoomed_window(npix_x, npix_y, cx, cy):
        half = int(args.win_frac * max(npix_x, npix_y))
        x0, x1 = max(cx - half, 0), min(cx + half, npix_x)
        y0, y1 = max(cy - half, 0), min(cy + half, npix_y)
        return x0, x1, y0, y1

    def load_image(reset_zoom=True):
        key = current_key()
        img = get_img(key)
        npix_y, npix_x = img.shape[0], img.shape[1]
        state["npix_x"], state["npix_y"] = npix_x, npix_y
        ax.clear()
        ax.imshow(img, origin="upper")
        ax.set_xlabel("pixel x (col)")
        ax.set_ylabel("pixel y (row, 0=top)")

        cur = images.get(key, {}).get("ant_px")
        state["current_marker"] = None
        if cur is not None:
            ax.plot(cur[0], cur[1], "o", ms=14, mfc="none", mec="dodgerblue",
                    mew=2.5, label="current (defaults.json)")
        state["pending"] = None
        state["pending_marker"] = None

        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"[{state['idx']+1}/{len(keys)}] {key}  —  click the antenna", fontsize=11)

        if reset_zoom and state["zoomed"] and cur is not None:
            x0, x1, y0, y1 = zoomed_window(npix_x, npix_y, cur[0], cur[1])
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)  # inverted y since origin="upper"
        else:
            ax.set_xlim(0, npix_x)
            ax.set_ylim(npix_y, 0)

        set_status(f"{key}: current ant_px={cur}. Click to pick a new one, or Skip/Prev/Next.")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button != 1:
            return  # right-click is reserved for pan (see zoom_pan.py)
        if fig.canvas.toolbar is not None and fig.canvas.toolbar.mode != "":
            return  # pan/zoom tool active, not a pick
        if event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata
        state["pending"] = (x, y)
        if state["pending_marker"] is not None:
            state["pending_marker"].remove()
        state["pending_marker"] = ax.plot(x, y, "r*", ms=20, mec="k", mew=0.8,
                                          label="pending pick", zorder=6)[0]
        fig.canvas.draw_idle()
        set_status(f"{current_key()}: pending pick = ({x:.0f}, {y:.0f}). Click Save & Next to commit.")

    fig.canvas.mpl_connect("button_press_event", on_click)

    def advance(delta):
        state["idx"] = (state["idx"] + delta) % len(keys)
        load_image(reset_zoom=True)

    def on_save_next(_):
        key = current_key()
        if state["pending"] is not None:
            x, y = state["pending"]
            new_ant_px = [int(round(x)), int(round(y))]
            try:
                update_ant_px_in_file(args.defaults, key, new_ant_px)
            except ValueError as e:
                set_status(f"SAVE FAILED for {key}: {e}")
                print(f"[{key}] SAVE FAILED: {e}")
                return
            images[key]["ant_px"] = new_ant_px  # keep in-memory copy in sync
            print(f"[{key}] saved ant_px = {new_ant_px}")
        advance(1)

    def on_skip(_):
        advance(1)

    def on_prev(_):
        advance(-1)

    def on_toggle_zoom(_):
        state["zoomed"] = not state["zoomed"]
        load_image(reset_zoom=True)

    btn_specs = [
        ("Prev", 0.06, on_prev),
        ("Skip", 0.20, on_skip),
        ("Save & Next", 0.34, on_save_next),
        ("Full image / Reset zoom", 0.54, on_toggle_zoom),
    ]
    btn_objs = []
    for label, xpos, cb in btn_specs:
        w = 0.12 if label != "Full image / Reset zoom" else 0.22
        bax = fig.add_axes([xpos, 0.09, w, 0.05])
        b = Button(bax, label)
        b.on_clicked(cb)
        btn_objs.append(b)

    load_image(reset_zoom=True)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
