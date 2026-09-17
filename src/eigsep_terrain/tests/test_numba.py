import numpy as np
import pytest

from eigsep_terrain.ray_numba import (ray_trace_basic_numba,
                                       ray_distance_coarse_to_fine_numba)

numba = pytest.importorskip("numba")


def _flat_dem(npnts=256, extent=10.0, dtype=np.float32):
    E = np.linspace(-extent / 2, extent / 2, npnts).astype(dtype)
    N = np.linspace(-extent / 2, extent / 2, npnts).astype(dtype)
    U = np.zeros((npnts, npnts), dtype=dtype)
    return E, N, U


def _sloped_dem(npnts=256, extent=10.0, a=0.1, b=-0.05, dtype=np.float32):
    E = np.linspace(-extent / 2, extent / 2, npnts).astype(dtype)
    N = np.linspace(-extent / 2, extent / 2, npnts).astype(dtype)
    EE, NN = np.meshgrid(E, N)
    U = (a * EE + b * NN).astype(dtype)
    return E, N, U, dtype(a), dtype(b)


def _normalize_rays(rays):
    return rays / np.linalg.norm(rays, axis=0, keepdims=True)


def test_ray_trace_basic_numba_flat_ground_downward_matches_analytic_within_step():
    """Flat ground U=0. Downward rays should hit at r=z0/(-ray_z) within ~1 step."""
    dtype = np.float32
    E, N, U = _flat_dem(dtype=dtype)

    start = np.array([0.0, 0.0, 1.0], dtype=dtype)

    rays = np.array(
        [
            [0.0, 0.1, 0.0],    # x
            [0.0, 0.0, 0.1],    # y
            [-1.0, -0.5, -0.2], # z (down)
        ],
        dtype=dtype,
    )
    rays = _normalize_rays(rays)

    delta = dtype(0.02)
    r = ray_trace_basic_numba(E, N, U, start, rays, delta_r_m=delta, max_iter=200000)
    assert r.shape == (3,)
    assert np.all(np.isfinite(r))

    # analytic hit distance for plane z=0
    r_true = start[2] / (-rays[2])
    assert np.all(np.abs(r - r_true) <= delta + dtype(1e-6))


def test_ray_trace_basic_numba_sloped_plane_vertical_ray():
    """Sloped plane U(E,N)=aE+bN; a vertical ray should hit at analytic distance."""
    dtype = np.float32
    E, N, U, a, b = _sloped_dem(npnts=256, extent=10.0, a=0.1, b=-0.05, dtype=dtype)

    # Use exact grid points to match nearest-neighbor index behavior.
    x0 = E[100]
    y0 = N[120]
    z0 = dtype(5.0)
    start = np.array([x0, y0, z0], dtype=dtype)

    rays = np.array([[0.0], [0.0], [-1.0]], dtype=dtype)
    delta = dtype(0.01)

    r = ray_trace_basic_numba(E, N, U, start, rays, delta_r_m=delta, max_iter=200000)
    assert r.shape == (1,)
    assert np.isfinite(r[0])

    u0 = a * x0 + b * y0
    r_true = z0 - u0
    assert abs(r[0] - r_true) <= delta + dtype(1e-6)


def test_ray_trace_basic_numba_out_of_bounds_returns_nan():
    """Rays that leave the DEM bounds before intersecting should return NaN."""
    dtype = np.float32
    E, N, U = _flat_dem(npnts=64, extent=2.0, dtype=dtype)

    start = np.array([0.0, 0.0, 1.0], dtype=dtype)

    # Point mostly sideways so it exits bounds quickly while staying above ground
    rays = np.array([[1.0], [0.0], [0.0]], dtype=dtype)
    rays = _normalize_rays(rays)

    r = ray_trace_basic_numba(E, N, U, start, rays, delta_r_m=dtype(0.1), max_iter=10000)
    assert np.isnan(r[0])


def test_ray_trace_basic_numba_r_start_nan_propagates():
    """If r_start is NaN, output should be NaN for that ray."""
    dtype = np.float32
    E, N, U = _flat_dem(dtype=dtype)

    start = np.array([0.0, 0.0, 1.0], dtype=dtype)
    rays = np.array([[0.0], [0.0], [-1.0]], dtype=dtype)

    r = ray_trace_basic_numba(
        E, N, U, start, rays,
        delta_r_m=dtype(0.01),
        r_start=np.array([np.nan], dtype=dtype),
        max_iter=1000,
    )
    assert np.isnan(r[0])


def test_ray_distance_coarse_to_fine_numba_flat_ground():
    """Coarse-to-fine numba result agrees with analytic hit on flat ground."""
    dtype = np.float32
    E = np.linspace(-10, 10, 256, dtype=dtype)
    N = np.linspace(-10, 10, 256, dtype=dtype)
    U = np.zeros((256, 256), dtype=dtype)
    start = np.array([0., 0., 2.], dtype=dtype)

    rays = np.array([[0., 0.1, 0.], [0., 0., 0.1], [-1., -1., -0.5]],
                    dtype=dtype)
    rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)

    r = ray_distance_coarse_to_fine_numba(E, N, U, start, rays,
                                          coarse_delta=0.5, fine_delta=0.1)

    assert r.shape == (3,)
    assert np.all(np.isfinite(r))
    expected = start[2] / np.abs(rays[2])
    np.testing.assert_allclose(r, expected, atol=0.5)


def test_ray_distance_coarse_to_fine_numba_sky_rays_are_nan():
    """Upward rays have no intersection and must return NaN."""
    dtype = np.float32
    E = np.linspace(-10, 10, 128, dtype=dtype)
    N = np.linspace(-10, 10, 128, dtype=dtype)
    U = np.zeros((128, 128), dtype=dtype)
    start = np.array([0., 0., 1.], dtype=dtype)

    rays = np.array([[0., 0.1], [0., 0.], [1., 0.5]], dtype=dtype)
    rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)

    r = ray_distance_coarse_to_fine_numba(E, N, U, start, rays,
                                          coarse_delta=1.0, fine_delta=0.2)
    assert np.all(np.isnan(r))
