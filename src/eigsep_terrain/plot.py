'''Module for generating DEM plots.'''

import numpy as np
import matplotlib.pyplot as plt

def terrain_plot(dem, ax=None, xlabel=True, ylabel=True,
             colorbar=False, cmap='terrain', erng_m=None, nrng_m=None,
             decimate=1, **kw):
    '''Generate standard terrain plot.'''
    E, N, U = dem.get_tile(erng_m=erng_m, nrng_m=nrng_m, mesh=False, decimate=decimate)
    extent = (E[0], E[-1], N[0], N[-1])
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(U, extent=extent, cmap=cmap, origin='lower',
                   interpolation='nearest', **kw)
    if colorbar:
        plt.colorbar(im)
    if xlabel:
        ax.set_xlabel('East [m]')
    if ylabel:
        ax.set_ylabel('North [m]')
    return im


def overlay_horizon_prediction(img, dem, axes=None, decimate=8, alpha=0.35,
                                cmap='cool_r', figsize=(16, 6), title_prefix=None):
    '''Side-by-side comparison of a HorizonImage's actual sky segmentation
    (left) against the sky/ground boundary predicted by ray-tracing its
    current pose (img.prms) against dem (right, computed on a decimated
    pixel grid for speed). Blue/cyan = sky in both panels. Returns
    (axes, r_map) where r_map is the decimated ray-distance map (NaN=sky).'''
    if axes is None:
        _, axes = plt.subplots(ncols=2, figsize=figsize)
    prefix = f'{img.key}: ' if title_prefix is None else title_prefix

    axes[0].imshow(img.img, origin='lower')
    axes[0].imshow(img.sky_mask, cmap=cmap, origin='lower', alpha=alpha)
    axes[0].set_title(f'{prefix}actual segmentation (blue=sky)')

    sl = slice(None, None, decimate)
    rays = img.get_rays()[..., sl, sl]
    r_map = img.ray_distance(dem, rays)
    axes[1].imshow(img.img[sl, sl], origin='lower')
    axes[1].imshow(np.isnan(r_map), cmap=cmap, origin='lower', alpha=alpha)
    axes[1].set_title(f'{prefix}predicted horizon (blue=sky)\n{img.prms_str}')

    return axes, r_map
