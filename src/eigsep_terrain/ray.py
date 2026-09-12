import numpy as np
import healpy

dtype_r = np.float32
dtype_i = np.int32

def healpix_rays(nside, dtype=dtype_r):
    npix = healpy.nside2npix(nside)
    px = np.arange(npix)
    rays = np.array(healpy.pix2vec(nside, px), dtype=dtype)
    return rays

def calc_maxiter(E, N, U, start_point, delta_r_m=1, r_max=None, dtype=dtype_r):
    '''Return the maximum iterations needed to resolve distances for ray tracing.'''
    if r_max is None:
        corners = np.array([[E[0],  N[0],  U[0, 0]],
                            [E[0],  N[-1], U[-1, 0]],
                            [E[-1], N[0],  U[0, -1]],
                            [E[-1], N[-1], U[-1, -1]]], dtype=dtype)
        r_max = np.linalg.norm(corners - start_point[None, :], axis=1).max()
    if np.isnan(r_max):
        raise ValueError(
            f"r_max is NaN; start_point={start_point} may be outside DEM."
        )
    return int(np.ceil(r_max / delta_r_m))

def ray_trace_basic(E, N, U, start_point, rays, delta_r_m=1,
                    r_start=None, max_iter=4096, dtype=dtype_r):
    '''Return the distance along a ray grid from a ENU start_point until a
    ray intersects the terrain, in steps of delta_r_m [m]. Returns distance
    [m] in ray order, with non-intersecting rays set to NaN.'''
    E = np.asarray(E, dtype=dtype)
    N = np.asarray(N, dtype=dtype)
    U = np.asarray(U, dtype=dtype, order='C')
    start_point = np.asarray(start_point, dtype=dtype)
    rays = np.asarray(rays, dtype=dtype)
    delta_r_m = dtype(delta_r_m)
    dE = E[1] - E[0]
    dN = N[1] - N[0]
    inv_dE = dtype(1.0) / dE
    inv_dN = dtype(1.0) / dN
    Nn, Ne = U.shape
    assert rays.shape[0] == 3
    dr_vec = delta_r_m * rays
    if r_start is None:
        r = np.full(rays.shape[1:], delta_r_m, dtype=dtype)
    else:
        r = np.asarray(r_start, dtype=dtype)
    active = ~np.isnan(r)
    inds = np.flatnonzero(active)
    points_m = start_point[:, None] + r[None, inds] * rays[:, inds]
    for i in range(2, max_iter):
        e_px = np.floor((points_m[0] - E[0]) * inv_dE).astype(dtype_i).clip(0, Ne - 1)
        n_px = np.floor((points_m[1] - N[0]) * inv_dN).astype(dtype_i).clip(0, Nn - 1)
        u_m = U[n_px, e_px]
        # prune to points that are above ground
        active = (points_m[2] > u_m)
        inds = inds[active]
        if inds.size == 0:
            break
        # take a step along the ray
        r[inds] += delta_r_m
        points_m = points_m[:, active] + dr_vec[:, inds]
        # check if out of bounds
        active = (E[0] <= points_m[0]) & (points_m[0] <= E[-1])
        active &= (N[0] <= points_m[1]) & (points_m[1] <= N[-1])
        r[inds[~active]] = np.nan  # set newly out-of-bounds rays to nan
        # prune to points that are in bounds
        inds = inds[active]
        points_m = points_m[:, active]
        if inds.size == 0:
            break
    # Any remaining active pixels should be set to nan
    r[inds] = np.nan
    return r


def ray_distance_coarse_to_fine(E, N, U, start_point, rays,
                                 coarse_delta=5.0, fine_delta=1.0,
                                 dtype=dtype_r):
    '''Two-pass coarse-to-fine ray trace. First pass uses coarse_delta step
    size; the second pass refines from just before the coarse hit using
    fine_delta. Returns distances in rays.shape[1:], NaN for misses.'''
    max_iter_coarse = calc_maxiter(E, N, U, start_point, delta_r_m=coarse_delta)
    r_coarse = ray_trace_basic(E, N, U, start_point, rays,
                               delta_r_m=dtype(coarse_delta),
                               max_iter=max_iter_coarse, dtype=dtype)
    r_start = np.where(
        np.isnan(r_coarse),
        np.nan,
        np.maximum(r_coarse - dtype(coarse_delta), dtype(fine_delta)),
    ).astype(dtype)
    max_iter_fine = int(np.ceil(2 * coarse_delta / fine_delta))
    return ray_trace_basic(E, N, U, start_point, rays,
                           delta_r_m=dtype(fine_delta), r_start=r_start,
                           max_iter=max_iter_fine, dtype=dtype)
