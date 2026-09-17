'''Persist and load per-image metadata -- currently just the hand-picked
antenna pixel, ant_px -- across notebook sessions. Replaces the old pattern
of hardcoding a `meta = {...}` dict directly in a notebook, which meant
antenna picks were lost/had to be redone whenever that notebook wasn't the
one in use.'''

import json
import os


def load_meta(filename='meta.json'):
    '''Load the meta dict from filename, or return {} if it doesn't exist
    yet. Keys are image keys (e.g. '2224'); values are dicts that may
    include 'ant_px': (x_px, y_px), matching the shape HorizonImage's
    constructor and ant_logL/export_jax already expect.'''
    if not os.path.exists(filename):
        return {}
    with open(filename) as f:
        raw = json.load(f)
    for v in raw.values():
        if 'ant_px' in v:
            v['ant_px'] = tuple(v['ant_px'])
    return raw


def save_meta(meta, filename='meta.json'):
    '''Write meta (see load_meta) to filename as pretty-printed JSON.'''
    with open(filename, 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def set_ant_px(meta, key, ant_px, filename='meta.json'):
    '''Set meta[key]['ant_px'] = ant_px (x_px, y_px) and immediately persist
    meta to filename, so a single click/pick isn't lost if the notebook
    session ends before an explicit save.'''
    meta.setdefault(key, {})['ant_px'] = tuple(float(c) for c in ant_px)
    save_meta(meta, filename)
    return meta


def pick_ant_px(img, meta, filename='meta.json', ax=None, zoom=None):
    '''Interactively pick the antenna pixel in img (a HorizonImage) by
    clicking on a plot of img.img. The click handler immediately persists
    the pick to meta[img.key]['ant_px'] in filename via set_ant_px, and
    also updates img.meta in place, so nothing is lost if the notebook
    session ends right after clicking and later cells (e.g. ant_logL) see
    the pick right away.

    zoom, if given, is (xmin, xmax, ymin, ymax) in pixel coordinates to set
    as the initial axis limits -- helpful once you have a rough location
    from a previous pick or a wide-view look at the image.

    Click again to move the pick; a magenta '+' marks the current pick.
    Returns the Axes so the caller can keep tweaking it (e.g. re-zooming)
    before or between clicks.'''
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(img.img, origin='lower')
    if zoom is not None:
        xmin, xmax, ymin, ymax = zoom
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_title(f'{img.key}: click the antenna')

    marker = [None]
    existing = meta.get(img.key, {}).get('ant_px')
    if existing is not None:
        marker[0] = ax.plot(*existing, 'm+', ms=16, mew=2)[0]

    def onclick(event):
        if event.inaxes != ax:
            return
        x, y = event.xdata, event.ydata
        set_ant_px(meta, img.key, (x, y), filename=filename)
        img.meta = meta[img.key]
        if marker[0] is not None:
            marker[0].remove()
        marker[0] = ax.plot(x, y, 'm+', ms=16, mew=2)[0]
        ax.set_title(f'{img.key}: ant_px = ({x:.1f}, {y:.1f}) -- saved to {filename}')
        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect('button_press_event', onclick)
    return ax
