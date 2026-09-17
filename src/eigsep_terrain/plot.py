'''Module for generating DEM plots.'''

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
