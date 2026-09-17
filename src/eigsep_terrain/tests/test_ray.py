"""Tests for eigsep_corr.io"""
import copy
import os
import pytest
import numpy as np
import healpy
from eigsep_terrain.ray import (ray_trace_basic, healpix_rays, calc_maxiter,
                                  ray_distance_coarse_to_fine)


class TestUtils:

    def test_ray_trace_basic(self):
        '''Unit test for calc_az_bin_range'''
        npnts = 512
        E = np.linspace(-1000, 1000, npnts)
        N = np.linspace(-1000, 1000, npnts)
        U = np.ones((npnts, npnts), dtype=float)
        start_point = np.array([0, 0, 2], dtype=float)
        nside = 128
        rays = healpix_rays(nside)
        rays_2d = rays.reshape(rays.shape[0], -1) 
        r = ray_trace_basic(E, N, U, start_point, rays_2d)
        npix = healpy.nside2npix(nside)
        px = np.arange(npix)
        th, phi = healpy.pix2ang(nside, ipix=px)
        assert np.all(np.isnan(r[th < np.pi/2]))
        assert np.all(r[th > 1.01 * np.pi/2] >= 1)


def _flat_dem(npnts=128, extent=10.0, dtype=np.float32):
    E = np.linspace(-extent, extent, npnts, dtype=dtype)
    N = np.linspace(-extent, extent, npnts, dtype=dtype)
    U = np.zeros((npnts, npnts), dtype=dtype)
    return E, N, U


def _sloped_dem(npnts=128, extent=10.0, a=0.1, b=-0.05, dtype=np.float32):
    E = np.linspace(-extent, extent, npnts, dtype=dtype)
    N = np.linspace(-extent, extent, npnts, dtype=dtype)
    EE, NN = np.meshgrid(E, N)
    U = (a * EE + b * NN).astype(dtype)
    return E, N, U, a, b


def _normalize(rays, eps=1e-12):
    rays = np.asarray(rays, dtype=np.float32)
    n = np.linalg.norm(rays, axis=0, keepdims=True)
    return rays / (n + eps)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ray_trace_basic_flat_ground_downward_matches_analytic_within_step(dtype):
    """Flat ground U=0. Downward rays should hit at r=z0/(-ray_z) within ~1 step."""
    E, N, U = _flat_dem(dtype=dtype)
    start = np.array([0.0, 0.0, 1.0], dtype=dtype)

    rays = np.array(
        [
            [0.0, 0.1, 0.0],   # x
            [0.0, 0.0, 0.1],   # y
            [-1.0, -0.5, -0.2] # z (down)
        ],
        dtype=dtype,
    )
    rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)

    delta = dtype(0.02)
    r = ray_trace_basic(E, N, U, start, rays, delta_r_m=delta, max_iter=20000)
    assert r.shape == (3,)
    assert np.all(np.isfinite(r))

    expected = start[2] / (-rays[2])
    assert np.all(np.abs(r - expected) <= (delta * 1.05))


@pytest.mark.parametrize("dtype", [np.float32])
def test_ray_trace_basic_upward_and_horizontal_return_nan(dtype):
    """Flat ground; upward or horizontal rays should return NaN (no hit)."""
    E, N, U = _flat_dem(dtype=dtype)
    start = np.array([0.0, 0.0, 1.0], dtype=dtype)

    rays_up = np.array([[0.0, 0.1], [0.0, 0.0], [1.0, 0.5]], dtype=dtype)
    rays_up = rays_up / np.linalg.norm(rays_up, axis=0, keepdims=True)

    # avoid div-by-zero in normalization; resulting z=0 rays are still horizontal
    rays_h = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    rays_h = rays_h / np.linalg.norm(rays_h + 1e-12, axis=0, keepdims=True)

    r_up = ray_trace_basic(E, N, U, start, rays_up, delta_r_m=dtype(0.05), max_iter=4000)
    r_h = ray_trace_basic(E, N, U, start, rays_h, delta_r_m=dtype(0.05), max_iter=4000)

    assert np.all(np.isnan(r_up))
    assert np.all(np.isnan(r_h))


@pytest.mark.parametrize("dtype", [np.float32])
def test_ray_trace_basic_leaves_dem_returns_nan(dtype):
    """Small DEM, strong horizontal component: leave bounds before hit -> NaN."""
    E, N, U = _flat_dem(npnts=128, extent=1.0, dtype=dtype)
    start = np.array([0.0, 0.0, 1.0], dtype=dtype)

    rays = np.array([[10.0], [0.0], [-1.0]], dtype=dtype)
    rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)

    r = ray_trace_basic(E, N, U, start, rays, delta_r_m=dtype(0.02), max_iter=2000)
    assert r.shape == (1,)
    assert np.isnan(r[0])


@pytest.mark.parametrize("dtype", [np.float32])
def test_ray_trace_basic_start_below_ground_returns_first_step(dtype):
    """Current behavior: algorithm starts at r=delta_r_m, so below-ground returns delta."""
    E, N, U = _flat_dem(dtype=dtype)
    start = np.array([0.0, 0.0, -1.0], dtype=dtype)

    rays = np.array([[0.0], [0.0], [-1.0]], dtype=dtype)
    rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)

    delta = dtype(0.05)
    r = ray_trace_basic(E, N, U, start, rays, delta_r_m=delta, max_iter=100)
    assert r.shape == (1,)
    assert np.isfinite(r[0])
    assert r[0] == pytest.approx(float(delta), rel=0, abs=0)


@pytest.mark.parametrize("dtype", [np.float32])
def test_ray_trace_basic_sloped_plane_vertical_ray(dtype):
    """Sloped plane U(E,N)=aE+bN; a vertical ray should hit at analytic distance."""
    E, N, U, a, b = _sloped_dem(npnts=256, extent=10.0, a=0.1, b=-0.05, dtype=dtype)

    # Use exact grid points to match U lookup (nearest-neighbor index) behavior.
    x0 = E[100]
    y0 = N[120]
    z0 = dtype(5.0)
    start = np.array([x0, y0, z0], dtype=dtype)

    rays = np.array([[0.0], [0.0], [-1.0]], dtype=dtype)
    delta = dtype(0.01)

    r = ray_trace_basic(E, N, U, start, rays, delta_r_m=delta, max_iter=200000)
    assert r.shape == (1,)
    assert np.isfinite(r[0])

    ground = a * x0 + b * y0
    expected = (z0 - ground)
    assert abs(r[0] - expected) <= float(delta) * 1.05


def test_ray_distance_coarse_to_fine_flat_ground():
    """Coarse-to-fine result agrees with analytic hit on flat ground."""
    dtype = np.float32
    E = np.linspace(-10, 10, 256, dtype=dtype)
    N = np.linspace(-10, 10, 256, dtype=dtype)
    U = np.zeros((256, 256), dtype=dtype)
    start = np.array([0., 0., 2.], dtype=dtype)

    rays = np.array([[0., 0.1, 0.], [0., 0., 0.1], [-1., -1., -0.5]],
                    dtype=dtype)
    rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)

    r = ray_distance_coarse_to_fine(E, N, U, start, rays,
                                    coarse_delta=0.5, fine_delta=0.1)

    assert r.shape == (3,)
    assert np.all(np.isfinite(r))
    # analytic: downward ray hits U=0 at r = z0 / |ray_z|
    expected = start[2] / np.abs(rays[2])
    np.testing.assert_allclose(r, expected, atol=0.5)


def test_ray_distance_coarse_to_fine_sky_rays_are_nan():
    """Upward rays have no intersection and must return NaN."""
    dtype = np.float32
    E = np.linspace(-10, 10, 128, dtype=dtype)
    N = np.linspace(-10, 10, 128, dtype=dtype)
    U = np.zeros((128, 128), dtype=dtype)
    start = np.array([0., 0., 1.], dtype=dtype)

    rays = np.array([[0., 0.1], [0., 0.], [1., 0.5]], dtype=dtype)
    rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)

    r = ray_distance_coarse_to_fine(E, N, U, start, rays,
                                    coarse_delta=1.0, fine_delta=0.2)
    assert np.all(np.isnan(r))
