# ray_numba.py (or inside ray.py guarded by try/except ImportError)

import numpy as np
from .ray import calc_maxiter

dtype_r = np.float32

try:
    import numba as nb
except ImportError:
    nb = None

if nb is not None:
    @nb.njit(cache=True, fastmath=True, parallel=True)
    def _ray_trace_basic_numba(E, N, U, start_point, rays, delta_r_m, r_start, max_iter):
        # E: (Ne,), N: (Nn,), U: (Nn,Ne)
        # start_point: (3,)
        # rays: (3,Nr)
        Nr = rays.shape[1]
        out = np.empty(Nr, dtype=dtype_r)

        E0 = E[0]
        N0 = N[0]
        Emax = E[E.shape[0] - 1]
        Nmax = N[N.shape[0] - 1]

        dE = E[1] - E[0]
        dN = N[1] - N[0]
        inv_dE = 1.0 / dE
        inv_dN = 1.0 / dN

        Ne = E.shape[0]
        Nn = N.shape[0]

        for j in nb.prange(Nr):
            # init r
            if r_start is None:
                r = delta_r_m
            else:
                r = r_start[j]
                if np.isnan(r):
                    out[j] = np.nan
                    continue

            rx = rays[0, j]
            ry = rays[1, j]
            rz = rays[2, j]

            hit_or_oob = False

            # step loop
            for _ in range(max_iter):
                px = start_point[0] + r * rx
                py = start_point[1] + r * ry
                pz = start_point[2] + r * rz

                # out of bounds => NaN
                if (px < E0) or (px > Emax) or (py < N0) or (py > Nmax):
                    out[j] = np.nan
                    hit_or_oob = True
                    break

                e_px = int(np.floor((px - E0) * inv_dE))
                n_px = int(np.floor((py - N0) * inv_dN))

                # safety clamp (should already be in bounds)
                if e_px < 0:
                    e_px = 0
                elif e_px >= Ne:
                    e_px = Ne - 1
                if n_px < 0:
                    n_px = 0
                elif n_px >= Nn:
                    n_px = Nn - 1

                u_m = U[n_px, e_px]

                # if we are not above ground, we "hit": return current r
                if not (pz > u_m):
                    out[j] = r
                    hit_or_oob = True
                    break

                r = r + delta_r_m

            if not hit_or_oob:
                # max_iter exhausted: treat as no-intersection
                out[j] = np.nan

        return out


def ray_trace_basic_numba(E, N, U, start_point, rays, delta_r_m=1.0,
                         r_start=None, max_iter=4096, dtype=dtype_r):
    """
    Drop-in replacement for ray_trace_basic() in ray.py, but faster.

    Returns (Nr,) array of distances, with non-intersecting rays set to NaN.
    """
    if nb is None:
        raise ImportError("numba not installed")

    E = np.asarray(E, dtype=dtype_r)
    N = np.asarray(N, dtype=dtype_r)
    U = np.asarray(U, dtype=dtype_r, order="C")
    start_point = np.asarray(start_point, dtype=dtype_r)
    rays = np.asarray(rays, dtype=dtype_r)

    if r_start is None:
        rs = None
    else:
        rs = np.asarray(r_start, dtype=dtype_r).reshape(-1)

    delta = dtype_r(delta_r_m)
    return _ray_trace_basic_numba(E, N, U, start_point, rays, delta, rs, int(max_iter))


def ray_distance_coarse_to_fine_numba(E, N, U, start_point, rays,
                                       coarse_delta=5.0, fine_delta=1.0,
                                       dtype=dtype_r):
    '''Two-pass coarse-to-fine ray trace using the Numba backend. First pass
    uses coarse_delta step size; the second pass refines from just before the
    coarse hit using fine_delta. Returns distances in rays.shape[1:], NaN for
    misses.'''
    if nb is None:
        raise ImportError("numba not installed")
    max_iter_coarse = calc_maxiter(E, N, U, start_point, delta_r_m=coarse_delta)
    r_coarse = ray_trace_basic_numba(E, N, U, start_point, rays,
                                     delta_r_m=coarse_delta,
                                     max_iter=max_iter_coarse, dtype=dtype)
    r_start = np.where(
        np.isnan(r_coarse),
        np.nan,
        np.maximum(r_coarse - dtype(coarse_delta), dtype(fine_delta)),
    ).astype(dtype)
    max_iter_fine = int(np.ceil(2 * coarse_delta / fine_delta))
    return ray_trace_basic_numba(E, N, U, start_point, rays,
                                 delta_r_m=fine_delta, r_start=r_start,
                                 max_iter=max_iter_fine, dtype=dtype)
