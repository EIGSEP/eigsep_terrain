#!/usr/bin/env python
"""
run from img folder

Visualize scripts/fit_horizon_exact.py's fits: for each image, overlay the
actual (segmentation-mask-derived) horizon against the predicted
(calc_horizon-based, post-fit) horizon curve. Saves one PNG per image plus a
single contact-sheet PNG of all of them.

Usage:
  python plot_horizon_exact_fits.py --results fit_horizon_exact_results.json
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
import fit_horizon_exact as fhe


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="fit_horizon_exact_results.json")
    ap.add_argument("--img-glob", default=fhe.IMG_GLOB)
    ap.add_argument("--cache-file", default=fhe.CACHE_FILE)
    ap.add_argument("--n-az", type=int, default=1024)
    ap.add_argument("--ncols", type=int, default=300)
    ap.add_argument("--bisect-iters", type=int, default=14)
    ap.add_argument("--out-dir", default="horizon_fit_plots")
    ap.add_argument("--contact-sheet", default="horizon_fit_contact_sheet.png")
    ap.add_argument("--grid-cols", type=int, default=6)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    with open(args.results) as f:
        results = json.load(f)

    dem = DEM(cache_file=args.cache_file)
    os.makedirs(args.out_dir, exist_ok=True)

    class LoadArgs:
        img_glob = args.img_glob
        ncols = args.ncols

    keys_sorted = [k for k in fhe.IMG_KEYS if k in results]
    thumbs = []
    for key in keys_sorted:
        which = fhe.IMG_KEYS.index(key)
        _, img, init, cols, y_actual = fhe.load_image(which, LoadArgs(), dem)
        cache = fhe.HorizonCache(dem, args.n_az)

        img.set_prms(tuple(init))
        hangles_before = cache.get(*init[:3])
        y_pred_before = fhe.predicted_horizon_rows(img, hangles_before, cols, n_iter=args.bisect_iters)

        prms_u = results[key]["prms_u"]
        img.set_prms(tuple(prms_u))
        hangles_after = cache.get(*prms_u[:3])
        y_pred = fhe.predicted_horizon_rows(img, hangles_after, cols, n_iter=args.bisect_iters)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img.img, origin="lower")
        ax.plot(cols, y_actual, ".", color="cyan", markersize=2, label="actual (mask)")
        ax.plot(cols, y_pred_before, ".", color="yellow", markersize=2, label="default (before)")
        ax.plot(cols, y_pred, ".", color="red", markersize=2, label="fit (after)")
        ax.legend(loc="upper right", fontsize=7)
        rmse_before = results[key].get("rmse_before_px")
        rmse = results[key]["rmse_after_px"]
        title = (f"{key}  RMSE {rmse_before:.0f}px -> {rmse:.1f}px" if rmse_before is not None
                else f"{key}  RMSE (fit) {rmse:.1f}px")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"horizon_fit_{key}.png")
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        thumbs.append((key, out_path, rmse))
        print(f"saved {out_path}")

    n = len(thumbs)
    ncols = args.grid_cols
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.4))
    axes = np.array(axes).reshape(nrows, ncols)
    for i, (key, path, rmse) in enumerate(thumbs):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.imshow(plt.imread(path))
        ax.axis("off")
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(args.contact_sheet, dpi=130)
    plt.close(fig)
    print(f"contact sheet -> {args.contact_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
