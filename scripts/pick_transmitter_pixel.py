#!/usr/bin/env python
"""
run from img folder (2026_imgs/)

Interactive transmitter-pixel picker -- same tool as pick_antenna_pixel.py,
but for a second, separate physical transmitter, stored alongside ant_px in
the *same* defaults.json entries as tx_px. Unlike the antenna, the
transmitter is not expected to be visible in every photo: images with no
transmitter pixel simply have no "tx_px" key, and that's a valid, expected
end state -- not every image needs one, and this tool never forces one in.

Convention: tx_px is stored exactly as clicked on the original, unflipped
photo (row 0 = top) -- the same convention as ant_px (confirmed correct
against known-good antenna picks; see pick_antenna_pixel.py). No vertical
flip, no HorizonImage/segmentation dependency, so it starts fast.

Controls:
  - Click on the image to place a pending pick (red star). Click again to
    move it.
  - "Save & Next": writes the pending pick to defaults.json's tx_px for the
    current image (adding the key if it wasn't there yet), then advances.
  - "Skip": leave this image untouched (no transmitter here, or not sure
    yet), advance to the next. This is the normal way to move past an
    image with no visible transmitter -- it does NOT write a null/empty
    marker, it just leaves tx_px absent.
  - "Clear tx_px here": remove this image's tx_px entirely (e.g. you
    mis-picked earlier, or decided on review that what you clicked isn't
    actually the transmitter).
  - "Prev" / "Next": navigate without saving.
  - "Full image" / "Reset zoom": toggle between the full photo and a crop
    centered on the currently-stored tx_px (if this image has one).
  - Scroll to zoom in/out (centered on the cursor); right-click-drag to
    pan. No need to touch the matplotlib toolbar -- left-click stays free
    for picking the whole time.
  - The blue circle (if shown) is the tx_px currently in defaults.json,
    for reference -- it's what you're about to overwrite.

Usage:
  python pick_transmitter_pixel.py                  # all images, in order
  python pick_transmitter_pixel.py --which 2213      # start at a specific key
  python pick_transmitter_pixel.py --keys 2213 2234 2210
"""
import argparse
import glob
import json
import os
import re
import shutil
import tempfile

import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.widgets import Button

from zoom_pan import enable_zoom_pan

DEFAULTS_PATH = "/Users/komalkaur/Desktop/eigsep_stuff/eigsep_terrain/eigsep_terrain/defaults.json"
IMG_GLOB = "*.jpg"
FIELD = "tx_px"


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


def _find_key_line(lines, key):
    prefix = f'"{key}":'
    for i, line in enumerate(lines):
        if line.strip().startswith(prefix):
            return i
    return None


def update_tx_px_in_file(path, key, new_tx_px):
    """Surgically set (inserting if absent, replacing if present) one
    image's tx_px in the JSON *text*, leaving every other line
    byte-identical -- avoids reformatting the whole file into a noisy
    diff. Relies on defaults.json's one-image-per-line layout (already the
    case; every prior surgical-edit tool in this repo assumes the same)."""
    with open(path) as f:
        lines = f.readlines()
    idx = _find_key_line(lines, key)
    if idx is None:
        raise ValueError(f"Could not find a line for key {key!r} in {path}")
    line = lines[idx]
    val_str = f"[{int(round(new_tx_px[0]))}, {int(round(new_tx_px[1]))}]"
    if f'"{FIELD}"' in line:
        new_line, n = re.subn(rf'"{FIELD}":\s*\[[^\]]*\]', f'"{FIELD}": {val_str}', line, count=1)
    else:
        new_line, n = re.subn(r'("ant_px":\s*\[[^\]]*\])', r"\1, " + f'"{FIELD}": {val_str}', line, count=1)
    if n != 1:
        raise ValueError(f"Could not locate an anchor to write {FIELD} on key {key!r}'s line")
    lines[idx] = new_line
    new_text = "".join(lines)
    json.loads(new_text)  # validate before touching the real file
    atomic_write(new_text, path)


def clear_tx_px_in_file(path, key):
    """Remove tx_px for `key` if present. Returns True if something was
    removed, False if the image had no tx_px to begin with."""
    with open(path) as f:
        lines = f.readlines()
    idx = _find_key_line(lines, key)
    if idx is None or f'"{FIELD}"' not in lines[idx]:
        return False
    new_line, n = re.subn(rf',\s*"{FIELD}":\s*\[[^\]]*\]', "", lines[idx], count=1)
    if n != 1:
        raise ValueError(f"Found {FIELD!r} on key {key!r}'s line but couldn't remove it cleanly")
    lines[idx] = new_line
    new_text = "".join(lines)
    json.loads(new_text)
    atomic_write(new_text, path)
    return True


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
        "zoomed": True,
        "img_cache": {},
    }

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

        cur = images.get(key, {}).get(FIELD)
        if cur is not None:
            ax.plot(cur[0], cur[1], "o", ms=14, mfc="none", mec="dodgerblue",
                    mew=2.5, label="current tx_px (defaults.json)")
            ax.legend(loc="upper right", fontsize=8)
        state["pending"] = None
        state["pending_marker"] = None

        has_str = "HAS tx_px" if cur is not None else "no tx_px yet"
        ax.set_title(
            f"[{state['idx']+1}/{len(keys)}] {key}  ({has_str})  —  "
            f"click the transmitter, or Skip if not visible here", fontsize=10
        )

        if reset_zoom and state["zoomed"] and cur is not None:
            x0, x1, y0, y1 = zoomed_window(npix_x, npix_y, cur[0], cur[1])
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)  # inverted y since origin="upper"
        else:
            ax.set_xlim(0, npix_x)
            ax.set_ylim(npix_y, 0)

        set_status(f"{key}: current {FIELD}={cur}. Click to pick, or Skip/Clear/Prev/Next.")
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
            new_tx_px = [int(round(x)), int(round(y))]
            try:
                update_tx_px_in_file(args.defaults, key, new_tx_px)
            except ValueError as e:
                set_status(f"SAVE FAILED for {key}: {e}")
                print(f"[{key}] SAVE FAILED: {e}")
                return
            images.setdefault(key, {})[FIELD] = new_tx_px  # keep in-memory copy in sync
            print(f"[{key}] saved {FIELD} = {new_tx_px}")
        advance(1)

    def on_skip(_):
        advance(1)

    def on_prev(_):
        advance(-1)

    def on_clear(_):
        key = current_key()
        try:
            removed = clear_tx_px_in_file(args.defaults, key)
        except ValueError as e:
            set_status(f"CLEAR FAILED for {key}: {e}")
            print(f"[{key}] CLEAR FAILED: {e}")
            return
        if removed:
            images[key].pop(FIELD, None)
            print(f"[{key}] cleared {FIELD}")
        load_image(reset_zoom=True)

    def on_toggle_zoom(_):
        state["zoomed"] = not state["zoomed"]
        load_image(reset_zoom=True)

    btn_specs = [
        ("Prev", 0.04, 0.10, on_prev),
        ("Skip", 0.16, 0.10, on_skip),
        ("Save & Next", 0.28, 0.14, on_save_next),
        ("Clear tx_px here", 0.44, 0.16, on_clear),
        ("Full image / Reset zoom", 0.62, 0.22, on_toggle_zoom),
    ]
    btn_objs = []
    for label, xpos, w, cb in btn_specs:
        bax = fig.add_axes([xpos, 0.09, w, 0.05])
        b = Button(bax, label)
        b.on_clicked(cb)
        btn_objs.append(b)

    load_image(reset_zoom=True)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
