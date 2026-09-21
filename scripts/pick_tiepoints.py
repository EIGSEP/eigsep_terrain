#!/usr/bin/env python
"""
run from img folder (2026_imgs/)

Interactive cross-image tie-point picker. Click the same physical terrain
feature (a sharp corner, an isolated rock, a distinctive ledge) in two or
more photos; each feature gets a point ID, and its pixel location in every
image you click it in is saved to a JSON file. Unlike pick_points.py (which
needs you to *also* identify the matching point on a hillshaded terrain
map -- tried in this project already, see docs/fitting_improvement_plan.md
section 6, and found unreliable for anything but the sharpest features),
this tool never asks you to know the terrain (E, N) of anything -- that's
solved for jointly by joint_fit.py from the correspondences alone, exactly
like triangulate_antenna.py already does for the single antenna point.

Pixel convention: stored exactly as clicked on the original (unflipped)
photo -- row 0 = top -- same convention as defaults.json's ant_px.

Controls:
  - Click the image to set/move the pending pick for the *current point* in
    the *current image*.
  - "Save pick": commit the pending click for (current point, current
    image). Does not advance -- so you can place the same point in several
    images before moving on.
  - "New point": start a fresh point (gets the next free numeric ID).
  - "Next point" / "Prev point": cycle through existing points (e.g. to
    place point 3 in another image after already placing 1 and 2 there).
  - "Delete from this image": remove the current point's pick in the
    current image only (e.g. you misclicked, or it's actually occluded
    here).
  - "Next image" / "Prev image": navigate the photo set. All points already
    placed in the image you land on are drawn (small numbered markers);
    the current point is highlighted (large red star).
  - "Full image / Reset zoom": toggle between the full photo and a
    zoomed default view.
  - Scroll to zoom in/out (centered on the cursor); right-click-drag to
    pan. No need to touch the matplotlib toolbar -- left-click stays free
    for picking the whole time.

Usage:
  python pick_tiepoints.py                         # all images in defaults.json
  python pick_tiepoints.py --keys 2213 2234 2216 2210
  python pick_tiepoints.py --out tiepoints.json
"""
import argparse
import glob
import json
import os
import tempfile

import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.widgets import Button

from zoom_pan import enable_zoom_pan

DEFAULTS_PATH = "/Users/komalkaur/Desktop/eigsep_stuff/eigsep_terrain/eigsep_terrain/defaults.json"
IMG_GLOB = "*.jpg"
OUT_PATH = "tiepoints.json"


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def atomic_write_json(data, path):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=d, prefix=".tiepoints_", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def find_img_file(key, img_glob):
    files = sorted(glob.glob(img_glob))
    matches = [f for f in files if os.path.basename(f).split("_")[-1].split(".")[0] == key]
    if not matches:
        raise FileNotFoundError(f"No file matching key {key!r} via {img_glob!r}")
    return matches[0]


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defaults", default=DEFAULTS_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--img-glob", default=IMG_GLOB)
    ap.add_argument("--keys", nargs="+", default=None,
                    help="Only cycle through these image keys (default: all in defaults.json).")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    with open(args.defaults) as f:
        defaults = json.load(f)
    keys = args.keys or list(defaults["images"].keys())

    points = load_json(args.out, {})  # {point_id: {img_key: [x, y]}}

    def point_ids_sorted():
        ids = sorted(points.keys(), key=lambda s: int(s) if s.isdigit() else s)
        return ids

    def next_new_id():
        existing = [int(k) for k in points if k.isdigit()]
        return str(max(existing) + 1) if existing else "1"

    state = {
        "img_idx": 0,
        "point_id": None,
        "pending": None,
        "pending_marker": None,
        "other_markers": [],
        "zoomed": False,
        "img_cache": {},
    }

    if points:
        state["point_id"] = point_ids_sorted()[0]

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_axes([0.06, 0.16, 0.90, 0.78])
    status_ax = fig.add_axes([0.06, 0.02, 0.88, 0.07])
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
        return keys[state["img_idx"]]

    def ensure_point():
        if state["point_id"] is None:
            state["point_id"] = next_new_id()
            points[state["point_id"]] = {}

    def redraw():
        key = current_key()
        img = get_img(key)
        npix_y, npix_x = img.shape[0], img.shape[1]
        state["npix_x"], state["npix_y"] = npix_x, npix_y
        ax.clear()
        ax.imshow(img, origin="upper")
        ax.set_xlabel("pixel x (col)")
        ax.set_ylabel("pixel y (row, 0=top)")

        state["pending"] = None
        state["pending_marker"] = None
        state["other_markers"] = []

        for pid in point_ids_sorted():
            if pid == state["point_id"]:
                continue
            xy = points.get(pid, {}).get(key)
            if xy is not None:
                m = ax.plot(xy[0], xy[1], "o", ms=9, mfc="none", mec="dodgerblue", mew=2)[0]
                t = ax.annotate(pid, xy, color="dodgerblue", fontsize=9, fontweight="bold",
                                xytext=(6, 6), textcoords="offset points")
                state["other_markers"].append((m, t))

        cur_xy = points.get(state["point_id"], {}).get(key) if state["point_id"] else None
        if cur_xy is not None:
            ax.plot(cur_xy[0], cur_xy[1], "r*", ms=20, mec="k", mew=0.8,
                    label=f"point {state['point_id']} (saved)")
            ax.legend(loc="upper right", fontsize=8)

        n_here = sum(1 for pid in points if key in points.get(pid, {}))
        ax.set_title(
            f"[{state['img_idx']+1}/{len(keys)}] {key}  —  point {state['point_id']}  "
            f"({n_here} point(s) placed in this image)", fontsize=11
        )

        if state["zoomed"] and cur_xy is not None:
            half = int(0.15 * max(npix_x, npix_y))
            ax.set_xlim(max(cur_xy[0] - half, 0), min(cur_xy[0] + half, npix_x))
            ax.set_ylim(min(cur_xy[1] + half, npix_y), max(cur_xy[1] - half, 0))
        else:
            ax.set_xlim(0, npix_x)
            ax.set_ylim(npix_y, 0)

        set_status(
            f"Points defined so far: {', '.join(point_ids_sorted()) or '(none)'}. "
            f"Click to pick point {state['point_id']} here, then Save pick."
        )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button != 1:
            return  # right-click is reserved for pan (see zoom_pan.py)
        if fig.canvas.toolbar is not None and fig.canvas.toolbar.mode != "":
            return
        if event.xdata is None or event.ydata is None:
            return
        ensure_point()
        x, y = event.xdata, event.ydata
        state["pending"] = (x, y)
        if state["pending_marker"] is not None:
            state["pending_marker"].remove()
        state["pending_marker"] = ax.plot(x, y, "r*", ms=20, mec="k", mew=0.8, zorder=6)[0]
        fig.canvas.draw_idle()
        set_status(f"Pending pick for point {state['point_id']} at ({x:.0f}, {y:.0f}). Click Save pick to commit.")

    fig.canvas.mpl_connect("button_press_event", on_click)

    def on_save(_):
        ensure_point()
        if state["pending"] is None:
            set_status("Nothing pending -- click the image first.")
            return
        key = current_key()
        x, y = state["pending"]
        points.setdefault(state["point_id"], {})[key] = [round(float(x), 1), round(float(y), 1)]
        atomic_write_json(points, args.out)
        print(f"point {state['point_id']} @ {key} = {points[state['point_id']][key]}")
        redraw()

    def on_delete_here(_):
        key = current_key()
        if state["point_id"] and key in points.get(state["point_id"], {}):
            del points[state["point_id"]][key]
            atomic_write_json(points, args.out)
            print(f"deleted point {state['point_id']} @ {key}")
        redraw()

    def on_new_point(_):
        state["point_id"] = next_new_id()
        points[state["point_id"]] = {}
        redraw()

    def cycle_point(delta):
        ids = point_ids_sorted()
        if not ids:
            on_new_point(None)
            return
        if state["point_id"] not in ids:
            state["point_id"] = ids[0]
        else:
            i = ids.index(state["point_id"])
            state["point_id"] = ids[(i + delta) % len(ids)]
        redraw()

    def on_next_point(_):
        cycle_point(1)

    def on_prev_point(_):
        cycle_point(-1)

    def on_next_img(_):
        state["img_idx"] = (state["img_idx"] + 1) % len(keys)
        redraw()

    def on_prev_img(_):
        state["img_idx"] = (state["img_idx"] - 1) % len(keys)
        redraw()

    def on_toggle_zoom(_):
        state["zoomed"] = not state["zoomed"]
        redraw()

    specs = [
        ("Prev img", 0.06, 0.10, on_prev_img),
        ("Next img", 0.18, 0.10, on_next_img),
        ("Prev point", 0.30, 0.10, on_prev_point),
        ("Next point", 0.42, 0.10, on_next_point),
        ("New point", 0.54, 0.10, on_new_point),
        ("Save pick", 0.66, 0.10, on_save),
        ("Delete from image", 0.78, 0.10, on_delete_here),
        ("Zoom", 0.90, 0.10, on_toggle_zoom),
    ]
    btns = []
    for label, xpos, w, cb in specs:
        bax = fig.add_axes([xpos, 0.095, w - 0.01, 0.045])
        b = Button(bax, label)
        b.on_clicked(cb)
        btns.append(b)

    redraw()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
