#!/usr/bin/env python
"""
run from img folder (2026_imgs/)

Sanity-check every ant_px in defaults.json against its segmentation mask and
(when the antenna sits against open sky) an automated candidate detector, so
bad picks like 2213's (star landed in empty cloud, ~1000px from the actual
visible antenna rig) can be found across the whole dataset, not just the 3
"good fit" images.

Convention: defaults.json stores ant_px=[x,y] with y measured top-down (row
0 = top), but img.py's ray math / sky masks use a vertically-flipped,
row-0-at-bottom frame (see triangulate_antenna.py's docstring). This script
works entirely in that flipped frame for consistency with the masks.

Candidate detection heuristic: when a crop around the current pick is mostly
"sky" per the segmentation mask, the physical antenna rig is a small dark
object that the segmenter often classifies as *not* sky -- so an isolated
non-sky connected component surrounded by sky, not touching the crop edge,
is a strong candidate for the true antenna pixel. Only fires when the crop
is majority-sky; skipped (no candidate) when the pick sits over terrain,
since rig-vs-rock contrast is unreliable for this heuristic.

Usage:
  python check_antenna_pixels.py
  python check_antenna_pixels.py --keys 2213 2234 2210
"""
import argparse
import json
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

DEFAULTS_PATH = "/Users/komalkaur/Desktop/eigsep_stuff/eigsep_terrain/eigsep_terrain/defaults.json"
IMG_TEMPLATE = "IMG_{key}.jpg"
SEG_TEMPLATE = "img_seg_IMG_{key}.npz"


def find_candidate(skymask_crop, star_x_local, star_y_local):
    """skymask_crop: bool array, True=sky. Look for a small non-sky blob
    surrounded by sky, not touching the crop border. Returns (cx, cy, area)
    in crop-local coords, or None."""
    h, w = skymask_crop.shape
    frac_sky = skymask_crop.mean()
    if frac_sky < 0.6:
        return None  # not predominantly sky; heuristic unreliable here
    not_sky = (~skymask_crop).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(not_sky, connectivity=8)
    best = None
    best_d = None
    for lbl in range(1, n):
        x, y, bw, bh, area = stats[lbl]
        if area < 15 or area > 0.05 * h * w:
            continue
        if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
            continue  # touches border -> likely terrain/frame edge, not an isolated rig
        cx, cy = centroids[lbl]
        d = np.hypot(cx - star_x_local, cy - star_y_local)
        if best_d is None or d < best_d:
            best_d = d
            best = (cx, cy, area)
    return best


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defaults", default=DEFAULTS_PATH)
    ap.add_argument("--keys", nargs="+", default=None)
    ap.add_argument("--out-dir", default="antenna_px_check")
    ap.add_argument("--win-frac", type=float, default=0.22,
                    help="Half-crop-window as a fraction of max(npix_x, npix_y).")
    ap.add_argument("--grid-cols", type=int, default=6)
    ap.add_argument("--contact-sheet", default="antenna_px_contact_sheet.png")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    with open(args.defaults) as f:
        defaults = json.load(f)["images"]
    keys = args.keys or list(defaults.keys())
    os.makedirs(args.out_dir, exist_ok=True)

    findings = {}
    tiles = []
    for key in keys:
        img_path = IMG_TEMPLATE.format(key=key)
        seg_path = SEG_TEMPLATE.format(key=key)
        if not (os.path.exists(img_path) and os.path.exists(seg_path)):
            print(f"[{key}] SKIP (missing image or seg npz)")
            continue

        img = np.flipud(imread(img_path))
        skymask = np.flipud(np.load(seg_path)["skymask"]).astype(bool)
        npix_y, npix_x = img.shape[0], img.shape[1]

        col_x, row_y_top = defaults[key]["ant_px"]
        row_y = npix_y - 1 - row_y_top  # convert to flipped frame

        half = int(args.win_frac * max(npix_x, npix_y))
        x0, x1 = max(col_x - half, 0), min(col_x + half, npix_x)
        y0, y1 = max(row_y - half, 0), min(row_y + half, npix_y)

        crop = img[y0:y1, x0:x1]
        sky_crop = skymask[y0:y1, x0:x1]
        star_local = (col_x - x0, row_y - y0)
        cand = find_candidate(sky_crop, *star_local)

        on_sky = bool(skymask[row_y, col_x]) if 0 <= row_y < npix_y and 0 <= col_x < npix_x else None

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(crop, origin="lower", extent=[x0, x1, y0, y1])
        ax.plot(col_x, row_y, "y*", ms=16, markeredgecolor="k", markeredgewidth=0.6,
                label="current ant_px")
        note = ""
        if cand is not None:
            cx, cy, area = cand
            cx_full, cy_full = cx + x0, cy + y0
            ax.plot(cx_full, cy_full, "r+", ms=18, mew=3, label="candidate (isolated non-sky blob)")
            d = np.hypot(cx_full - col_x, cy_full - row_y)
            note = f"  candidate {d:.0f}px away"
            findings[key] = {
                "current_ant_px": [int(col_x), int(row_y_top)],
                "on_sky": on_sky,
                "candidate_flipped_xy": [round(float(cx_full), 1), round(float(cy_full), 1)],
                "candidate_top_down_xy": [round(float(cx_full), 1),
                                          round(float(npix_y - 1 - cy_full), 1)],
                "candidate_dist_px": round(float(d), 1),
            }
        else:
            findings[key] = {
                "current_ant_px": [int(col_x), int(row_y_top)],
                "on_sky": on_sky,
                "candidate_flipped_xy": None,
            }
        ax.set_title(f"{key}  on_sky={on_sky}{note}", fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
        ax.axis("off")
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"check_{key}.png")
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        tiles.append((key, out_path))
        print(f"[{key}] on_sky={on_sky}{note}")

    with open(os.path.join(args.out_dir, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)

    n = len(tiles)
    ncols = args.grid_cols
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.array(axes).reshape(nrows, ncols)
    for i, (key, path) in enumerate(tiles):
        r, c = divmod(i, ncols)
        axes[r, c].imshow(plt.imread(path))
        axes[r, c].axis("off")
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(args.contact_sheet, dpi=130)
    plt.close(fig)
    print(f"\ncontact sheet -> {args.contact_sheet}")
    print(f"findings -> {os.path.join(args.out_dir, 'findings.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
