"""
Shared scroll-to-zoom + right-drag-to-pan for the pick_*.py picking tools.

matplotlib's own toolbar zoom/pan works, but requires switching tool modes
(the magnifying-glass / cross-arrows buttons) to use, and while a pan/zoom
mode is active, left-click doesn't reach the pick handler -- you have to
toggle the toolbar off again before you can click a point. This gives
scroll-wheel zoom and right-click-drag pan directly on the image axes, with
left-click always free for picking, no mode-switching needed.
"""


def enable_zoom_pan(fig, ax, dims_getter, zoom_factor=1.3, max_zoom_out=1.5):
    """Attach scroll-to-zoom and right-drag-to-pan handlers to `ax`.

    dims_getter: callable returning (npix_x, npix_y) for whatever image is
    currently displayed on `ax` -- queried fresh on every zoom, so this
    keeps working correctly as the tool cycles between images of different
    sizes/orientations.

    max_zoom_out: won't zoom out past this multiple of the image's native
    size, so scrolling can't wander off into empty space indefinitely.
    """
    state = {"panning": False, "press_xy": None, "press_xlim": None, "press_ylim": None}

    def on_scroll(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        npix_x, npix_y = dims_getter()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        factor = 1 / zoom_factor if event.button == "up" else zoom_factor
        new_w = (xlim[1] - xlim[0]) * factor
        new_h = (ylim[1] - ylim[0]) * factor
        if abs(new_w) > npix_x * max_zoom_out or abs(new_h) > npix_y * max_zoom_out:
            return  # already at (or past) the zoomed-out cap
        x, y = event.xdata, event.ydata
        left_frac = (x - xlim[0]) / (xlim[1] - xlim[0])
        bottom_frac = (y - ylim[0]) / (ylim[1] - ylim[0])
        ax.set_xlim(x - left_frac * new_w, x + (1 - left_frac) * new_w)
        ax.set_ylim(y - bottom_frac * new_h, y + (1 - bottom_frac) * new_h)
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax or event.button != 3:
            return
        state["panning"] = True
        state["press_xy"] = (event.x, event.y)
        state["press_xlim"] = ax.get_xlim()
        state["press_ylim"] = ax.get_ylim()

    def on_release(event):
        if event.button == 3:
            state["panning"] = False

    def on_motion(event):
        if not state["panning"] or event.x is None or event.y is None:
            return
        inv = ax.transData.inverted()
        x0, y0 = inv.transform(state["press_xy"])
        x1, y1 = inv.transform((event.x, event.y))
        ddx, ddy = x1 - x0, y1 - y0
        xlim, ylim = state["press_xlim"], state["press_ylim"]
        ax.set_xlim(xlim[0] - ddx, xlim[1] - ddx)
        ax.set_ylim(ylim[0] - ddy, ylim[1] - ddy)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
