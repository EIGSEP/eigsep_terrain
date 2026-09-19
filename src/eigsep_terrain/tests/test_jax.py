import os
import sys
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from eigsep_terrain.ray import ray_trace_basic
from eigsep_terrain.ray_jax import ray_trace_basic_jax, ray_trace_basic_jax_jit, ray_distance_coarse_to_fine
from eigsep_terrain.img_jax import (horizon_ray_logL_jax, ant_logL_jax,
                                    get_rays_jax)
from eigsep_terrain.solver_jax import logL_from_problem_jit, stack_problem


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_ray_trace_basic_jax_flat_ground_hits(dtype):
    """
    Flat ground U=0. Start at z=1.
    A ray that points downward should hit at r = z0 / (-ray_z).
    """
    npnts = 128
    E = jnp.linspace(-10.0, 10.0, npnts, dtype=dtype)
    N = jnp.linspace(-10.0, 10.0, npnts, dtype=dtype)
    U = jnp.zeros((npnts, npnts), dtype=dtype)

    start_point = jnp.array([0.0, 0.0, 1.0], dtype=dtype)

    rays = jnp.array(
        [
            [0.0, 0.1, 0.0],   # Ex
            [0.0, 0.0, 0.1],   # Ny
            [-1.0, -0.5, -0.2] # Uz (downward)
        ],
        dtype=dtype,
    )
    # normalize
    rays = rays / jnp.linalg.norm(rays, axis=0, keepdims=True)

    r = ray_trace_basic_jax(E, N, U, start_point, rays, delta_r_m=dtype(0.01), max_iter=10000)
    assert r.shape == (3,)
    assert jnp.all(jnp.isfinite(r))

    expected = start_point[2] / (-rays[2])
    # stepping discretizes; allow small tolerance
    np.testing.assert_allclose(np.array(r), np.array(expected), rtol=1e-2, atol=5e-2)


def test_ray_trace_basic_jax_returns_nan_when_never_hits():
    """
    Flat ground U=0. Start at z=1.
    A ray pointing upward never intersects; should return NaN after max_iter.
    """
    npnts = 64
    E = jnp.linspace(-10.0, 10.0, npnts, dtype=jnp.float32)
    N = jnp.linspace(-10.0, 10.0, npnts, dtype=jnp.float32)
    U = jnp.zeros((npnts, npnts), dtype=jnp.float32)

    start_point = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    rays = jnp.array(
        [
            [0.0],
            [0.0],
            [1.0],  # up
        ],
        dtype=jnp.float32,
    )

    r = ray_trace_basic_jax(E, N, U, start_point, rays, delta_r_m=1.0, max_iter=200)
    assert r.shape == (1,)
    assert bool(jnp.isnan(r[0]))


def test_ray_trace_basic_jax_jit_compiles_and_runs():
    npnts = 64
    E = jnp.linspace(-10.0, 10.0, npnts, dtype=jnp.float32)
    N = jnp.linspace(-10.0, 10.0, npnts, dtype=jnp.float32)
    U = jnp.zeros((npnts, npnts), dtype=jnp.float32)
    start_point = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    rays = jnp.array([[0.0, 0.1], [0.0, 0.0], [-1.0, -1.0]], dtype=jnp.float32)
    rays = rays / jnp.linalg.norm(rays, axis=0, keepdims=True)

    r1 = ray_trace_basic_jax_jit(E, N, U, start_point, rays, delta_r_m=0.05, max_iter=2000)
    r2 = ray_trace_basic_jax_jit(E, N, U, start_point, rays, delta_r_m=0.05, max_iter=2000)

    assert r1.shape == (2,)
    assert jnp.all(jnp.isfinite(r1))
    np.testing.assert_allclose(np.array(r1), np.array(r2), rtol=0, atol=0)


def test_horizon_ray_logL_jax_smoke_and_expected_value_simple():
    """
    Build a tiny synthetic case where rays always hit ground (so model_sky=False)
    and psky is constant. Then logL should be sum(log(1-psky)).
    """
    npnts = 128
    E = jnp.linspace(-10.0, 10.0, npnts, dtype=jnp.float32)
    N = jnp.linspace(-10.0, 10.0, npnts, dtype=jnp.float32)
    U = jnp.zeros((npnts, npnts), dtype=jnp.float32)

    # Small "image grid"
    Nu, Nv = 16, 16
    # pick 8 pixels
    x_px = jnp.array([2, 4, 6, 8, 10, 12, 14, 15], dtype=jnp.int32)
    y_px = jnp.array([1, 3, 5, 7, 9, 11, 13, 15], dtype=jnp.int32)

    p0 = 0.25
    psky = jnp.full((x_px.shape[0],), p0, dtype=jnp.float32)

    # th=pi flips the camera's +z boresight to point straight down
    e, n, u = 0.0, 0.0, 1.0
    th, ph, ti, f = np.pi, 0.0, 0.0, 50.0

    logL = horizon_ray_logL_jax(
        E, N, U,
        Nu, Nv,
        e, n, u, th, ph, ti, f,
        x_px, y_px, psky,
        eps=1e-6,
        max_iters=(4096, 4096),
    )

    assert logL.shape == ()
    assert jnp.isfinite(logL)

    expected = x_px.shape[0] * np.log1p(-p0)
    np.testing.assert_allclose(float(logL), expected, rtol=5e-2, atol=5e-2)


def test_solver_logL_from_problem_jit_smoke():
    """
    Minimal stacked-problem smoke test for solver_jax.logL_from_problem_jit.
    """
    npnts = 128
    E = jnp.linspace(-10.0, 10.0, npnts, dtype=jnp.float32)
    N = jnp.linspace(-10.0, 10.0, npnts, dtype=jnp.float32)
    U = jnp.zeros((npnts, npnts), dtype=jnp.float32)

    # 1 fit image
    n_fit = 1
    Nu = jnp.array([16], dtype=jnp.int32)
    Nv = jnp.array([16], dtype=jnp.int32)
    x_px = jnp.stack([jnp.array([2, 4, 6, 8], dtype=jnp.int32)])
    y_px = jnp.stack([jnp.array([1, 3, 5, 7], dtype=jnp.int32)])
    psky = jnp.stack([jnp.full((4,), 0.2, dtype=jnp.float32)])
    correlation_px = jnp.array([4.0], dtype=jnp.float32)

    # 1 "all" image (same one), and treat it as fit-indexed
    all_Nu = jnp.array([16], dtype=jnp.int32)
    all_Nv = jnp.array([16], dtype=jnp.int32)
    ant_uv_u = jnp.array([8], dtype=jnp.int32)
    ant_uv_v = jnp.array([8], dtype=jnp.int32)
    fixed_prms = jnp.zeros((1, 7), dtype=jnp.float32)
    is_fit = jnp.array([True], dtype=jnp.bool_)
    fit_index = jnp.array([0], dtype=jnp.int32)

    problem = {
        "dem": {"E": E, "N": N, "U": U},
        "fit": {"Nu": Nu, "Nv": Nv, "x_px": x_px, "y_px": y_px, "psky": psky,
                "correlation_px": correlation_px},
        "all": {
            "Nu": all_Nu,
            "Nv": all_Nv,
            "ant_u_px": ant_uv_u,
            "ant_v_px": ant_uv_v,
            "has_ant": jnp.array([True]),
            "fixed_prms": fixed_prms,
            "is_fit": is_fit,
            "fit_index": fit_index,
        },
        "box_size": jnp.asarray(1.0, dtype=jnp.float32),
    }

    # theta = [fit_prms (7), ant_e,ant_n,ant_u]
    fit_prms = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 50.0], dtype=jnp.float32)
    ant = jnp.array([1.0, 0.0, 1.0], dtype=jnp.float32)
    theta = jnp.concatenate([fit_prms, ant], axis=0)

    logL = logL_from_problem_jit(theta, problem, eps=1e-6, max_iters=(4096, 4096))
    assert logL.shape == ()
    assert jnp.isfinite(logL)


def test_logL_jittable_and_repeatable():
    """
    Ensures the full logL function is jit-callable and stable across repeated calls.
    """
    npnts = 64
    E = jnp.linspace(-5.0, 5.0, npnts, dtype=jnp.float32)
    N = jnp.linspace(-5.0, 5.0, npnts, dtype=jnp.float32)
    U = jnp.zeros((npnts, npnts), dtype=jnp.float32)

    problem = {
        "dem": {"E": E, "N": N, "U": U},
        "fit": {
            "Nu": jnp.array([16], dtype=jnp.int32),
            "Nv": jnp.array([16], dtype=jnp.int32),
            "x_px": jnp.stack([jnp.array([2, 6], dtype=jnp.int32)]),
            "y_px": jnp.stack([jnp.array([3, 7], dtype=jnp.int32)]),
            "psky": jnp.stack([jnp.array([0.2, 0.8], dtype=jnp.float32)]),
            "correlation_px": jnp.array([4.0], dtype=jnp.float32),
        },
        "all": {
            "Nu": jnp.array([16], dtype=jnp.int32),
            "Nv": jnp.array([16], dtype=jnp.int32),
            "ant_u_px": jnp.array([8], dtype=jnp.float32),
            "ant_v_px": jnp.array([8], dtype=jnp.float32),
            "has_ant": jnp.array([True]),
            "fixed_prms": jnp.zeros((1, 7), dtype=jnp.float32),
            "is_fit": jnp.array([True], dtype=jnp.bool_),
            "fit_index": jnp.array([0], dtype=jnp.int32),
        },
        "box_size": jnp.asarray(1.0, dtype=jnp.float32),
    }

    theta = jnp.array([0, 0, 1, 0, 0, 0, 50, 1, 0, 1], dtype=jnp.float32)

    f = jax.jit(lambda th: logL_from_problem_jit(th, problem, eps=1e-6, max_iters=(4096, 4096)))
    v1 = f(theta)
    v2 = f(theta)
    assert jnp.isfinite(v1)
    np.testing.assert_allclose(float(v1), float(v2), rtol=0, atol=0)


def _flat_dem(npnts=128, extent=10.0, dtype=jnp.float32):
    E = jnp.linspace(-extent, extent, npnts, dtype=dtype)
    N = jnp.linspace(-extent, extent, npnts, dtype=dtype)
    U = jnp.zeros((npnts, npnts), dtype=dtype)
    return E, N, U


def _sloped_dem(npnts=128, extent=10.0, slope_e=0.1, slope_n=-0.05, dtype=jnp.float32):
    """
    Plane: U(E,N) = slope_e * E + slope_n * N
    """
    E = jnp.linspace(-extent, extent, npnts, dtype=dtype)
    N = jnp.linspace(-extent, extent, npnts, dtype=dtype)
    EE, NN = jnp.meshgrid(E, N, indexing="ij")
    U = slope_e * EE + slope_n * NN
    return E, N, U


def _normalize(rays):
    return rays / jnp.linalg.norm(rays, axis=0, keepdims=True)


@pytest.mark.parametrize("dtype", [jnp.float32])
def test_ray_trace_basic_jax_upward_rays_return_nan(dtype):
    """
    Flat ground at U=0, start above ground. Rays pointing upward should never hit -> NaN.
    """
    E, N, U = _flat_dem(dtype=dtype)
    start = jnp.array([0.0, 0.0, 1.0], dtype=dtype)

    rays = jnp.array(
        [
            [0.0, 0.1, 0.0],  # x
            [0.0, 0.0, 0.1],  # y
            [1.0, 0.5, 0.2],  # z (up)
        ],
        dtype=dtype,
    )
    rays = _normalize(rays)

    r = ray_trace_basic_jax(E, N, U, start, rays, delta_r_m=dtype(0.05), max_iter=4000)
    assert r.shape == (3,)
    assert jnp.all(jnp.isnan(r))


@pytest.mark.parametrize("dtype", [jnp.float32])
def test_ray_trace_basic_jax_horizontal_rays_return_nan(dtype):
    """
    Flat ground; purely horizontal rays should not intersect -> NaN.
    """
    E, N, U = _flat_dem(dtype=dtype)
    start = jnp.array([0.0, 0.0, 1.0], dtype=dtype)

    rays = jnp.array(
        [
            [1.0, 0.0],  # x
            [0.0, 1.0],  # y
            [0.0, 0.0],  # z
        ],
        dtype=dtype,
    )
    rays = _normalize(rays)

    r = ray_trace_basic_jax(E, N, U, start, rays, delta_r_m=dtype(0.05), max_iter=4000)
    assert r.shape == (2,)
    assert jnp.all(jnp.isnan(r))


@pytest.mark.parametrize("dtype", [jnp.float32])
def test_ray_trace_basic_jax_start_below_ground_hits_immediately(dtype):
    """
    If start is already below ground, some implementations return r=0.
    This test encodes that expectation; if your intended behavior is NaN,
    flip the assertion accordingly.
    """
    E, N, U = _flat_dem(dtype=dtype)
    start = jnp.array([0.0, 0.0, -1.0], dtype=dtype)  # below U=0

    rays = jnp.array([[0.0], [0.0], [-1.0]], dtype=dtype)
    rays = _normalize(rays)

    r = ray_trace_basic_jax(E, N, U, start, rays, delta_r_m=dtype(0.05), max_iter=100)
    assert r.shape == (1,)
    # choose ONE behavior for consistency:
    assert jnp.isfinite(r[0]) and (r[0] == dtype(0.0))


@pytest.mark.parametrize("dtype", [jnp.float32])
def test_ray_trace_basic_jax_leaves_dem_returns_nan(dtype):
    """
    Ray points down but with strong horizontal component; it may leave DEM bounds
    before hitting ground (depending on DEM extent and max_iter).
    Expect NaN to signal 'no hit in domain'.
    """
    E, N, U = _flat_dem(npnts=128, extent=1.0, dtype=dtype)  # small domain
    start = jnp.array([0.0, 0.0, 1.0], dtype=dtype)

    rays = jnp.array(
        [
            [10.0],   # x big
            [0.0],
            [-1.0],   # down
        ],
        dtype=dtype,
    )
    rays = _normalize(rays)

    r = ray_trace_basic_jax(E, N, U, start, rays, delta_r_m=dtype(0.02), max_iter=2000)
    assert r.shape == (1,)
    assert jnp.isnan(r[0])


@pytest.mark.parametrize("dtype", [jnp.float32])
def test_ray_trace_basic_jax_step_size_convergence(dtype):
    """
    Same ray traced with smaller delta_r should be close to larger-delta result,
    but typically not identical. Checks stability to discretization.
    """
    E, N, U = _flat_dem(dtype=dtype)
    start = jnp.array([0.0, 0.0, 1.0], dtype=dtype)

    ray = jnp.array([[0.0], [0.0], [-1.0]], dtype=dtype)  # straight down => r=1
    ray = _normalize(ray)

    r_coarse = ray_trace_basic_jax(E, N, U, start, ray, delta_r_m=dtype(0.1), max_iter=2000)[0]
    r_fine = ray_trace_basic_jax(E, N, U, start, ray, delta_r_m=dtype(0.01), max_iter=20000)[0]

    assert jnp.isfinite(r_coarse) and jnp.isfinite(r_fine)
    assert jnp.abs(r_fine - dtype(1.0)) < dtype(0.05)
    assert jnp.abs(r_coarse - r_fine) < dtype(0.2)


@pytest.mark.parametrize("dtype", [jnp.float32])
def test_ray_trace_basic_jax_vmap_many_rays(dtype):
    """
    Smoke-test vectorization: trace many rays at once and check finite for down-going.
    """
    E, N, U = _flat_dem(dtype=dtype)
    start = jnp.array([0.0, 0.0, 2.0], dtype=dtype)

    # 64 rays: random small tilts but all down-going
    key = jax.random.PRNGKey(0)
    xy = 0.05 * jax.random.normal(key, (2, 64), dtype=dtype)
    z = -jnp.ones((1, 64), dtype=dtype)
    rays = jnp.vstack([xy, z])
    rays = _normalize(rays)

    r = ray_trace_basic_jax(E, N, U, start, rays, delta_r_m=dtype(0.02), max_iter=20000)
    assert r.shape == (64,)
    assert jnp.all(jnp.isfinite(r))


def test_ray_trace_basic_jax_jit_matches_eager():
    """
    JIT and eager results should agree (within tolerance).
    """
    dtype = jnp.float32
    E, N, U = _flat_dem(dtype=dtype)
    start = jnp.array([0.0, 0.0, 1.0], dtype=dtype)

    rays = jnp.array([[0.02, 0.0], [0.0, 0.02], [-1.0, -1.0]], dtype=dtype)
    rays = _normalize(rays)

    r_eager = ray_trace_basic_jax(E, N, U, start, rays, delta_r_m=dtype(0.02), max_iter=20000)
    r_jit = ray_trace_basic_jax_jit(E, N, U, start, rays, delta_r_m=dtype(0.02), max_iter=20000)

    assert jnp.all(jnp.isfinite(r_eager))
    assert jnp.all(jnp.isfinite(r_jit))
    assert jnp.allclose(r_eager, r_jit, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.float32])
def test_ray_distance_coarse_to_fine_consistent_with_basic(dtype):
    """
    Compare coarse-to-fine distance with a single fine basic trace on flat ground.
    """
    E, N, U = _flat_dem(dtype=dtype)
    start = jnp.array([0.0, 0.0, 1.0], dtype=dtype)

    rays = jnp.array([[0.0, 0.03], [0.0, 0.0], [-1.0, -1.0]], dtype=dtype)
    rays = _normalize(rays)

    r_basic = ray_trace_basic_jax(E, N, U, start, rays, delta_r_m=dtype(0.01), max_iter=20000)
    r_ctf = ray_distance_coarse_to_fine(
        E, N, U, start, rays,
        delta_steps=(0.2, 0.02),
        max_iters=(5000, 20000),
    )

    assert jnp.all(jnp.isfinite(r_basic))
    assert jnp.all(jnp.isfinite(r_ctf))
    assert jnp.allclose(r_basic, r_ctf, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [jnp.float32])
def test_ray_trace_on_sloped_plane_sanity(dtype):
    """
    Sloped plane: at least verify that down-going rays hit at finite positive distances.
    (Closed form exists for a plane, but this keeps the test robust to conventions.)
    """
    E, N, U = _sloped_dem(dtype=dtype, slope_e=0.05, slope_n=-0.02)
    start = jnp.array([0.0, 0.0, 2.0], dtype=dtype)

    rays = jnp.array(
        [
            [0.01, -0.02, 0.0],
            [0.00,  0.01, 0.02],
            [-1.0, -1.0, -1.0],
        ],
        dtype=dtype,
    )
    rays = _normalize(rays)

    r = ray_trace_basic_jax(E, N, U, start, rays, delta_r_m=dtype(0.01), max_iter=40000)
    assert r.shape == (3,)
    assert jnp.all(jnp.isfinite(r))
    assert jnp.all(r > 0)


@pytest.mark.skipif("jax" not in globals(), reason="JAX not available")
def test_ray_trace_basic_jax_matches_numpy_on_random_rays():
    """Regression test: JAX and NumPy implementations should agree (incl NaN pattern)."""
    rng = np.random.default_rng(0)
    E, N, U = _flat_dem(npnts=128, extent=10.0, dtype=np.float32)
    start = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    rays = rng.normal(size=(3, 64)).astype(np.float32)
    rays[2] -= 1.0  # bias downward
    rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)

    delta = np.float32(0.02)
    r_np = ray_trace_basic(E, N, U, start, rays, delta_r_m=delta, max_iter=20000)

    r_j = np.array(
        ray_trace_basic_jax(
            jnp.asarray(E), jnp.asarray(N), jnp.asarray(U),
            jnp.asarray(start), jnp.asarray(rays),
            delta_r_m=delta, max_iter=20000,
        )
    )

    assert np.array_equal(np.isnan(r_np), np.isnan(r_j))
    mask = np.isfinite(r_np)
    assert np.all(np.abs(r_np[mask] - r_j[mask]) <= float(delta) * 1.05)


def _numpy_rays(Nu, Nv, f, th, ph, ti, uv):
    from eigsep_terrain.img import pixels_to_rays
    from eigsep_terrain.utils import rot_m
    z_rays = pixels_to_rays(Nu, Nv, f, uv=uv)
    rm = rot_m(ph, np.array([0, 0, 1])) @ (
        rot_m(th, np.array([0, 1, 0])) @ rot_m(ti, np.array([0, 0, 1])))
    return np.einsum('ij,j...->i...', rm, z_rays)


def test_get_rays_jax_matches_numpy():
    """get_rays_jax must follow the same camera convention as
    HorizonImage.get_rays (boresight along +z before rotation)."""
    Nu, Nv, f = 400, 600, 350.0
    th, ph, ti = 1.2, -0.7, 0.05
    u = np.array([0, 17, 200, 399])
    v = np.array([599, 3, 300, 150])
    expected = _numpy_rays(Nu, Nv, f, th, ph, ti, (u, v))
    got = get_rays_jax(Nu, Nv, f, th, ph, ti, u, v)
    np.testing.assert_allclose(np.asarray(got), expected, atol=1e-5)


def _numpy_horizon_logL(E, N, U, Nu, Nv, prm, x_px, y_px, psky, corr_px):
    """Same computation as HorizonImage.horizon_ray_logL, without an
    image on disk."""
    e, n, u, th, ph, ti, f = prm
    rays = _numpy_rays(Nu, Nv, f, th, ph, ti, (x_px, y_px))
    r = ray_trace_basic(np.asarray(E), np.asarray(N), np.asarray(U),
                        np.array([e, n, u], dtype=np.float32),
                        rays.astype(np.float32))
    per_pixel = np.where(np.isnan(r), np.log(psky), np.log1p(-psky))
    col_span = max(y_px.max() - y_px.min(), 1)
    n_eff = np.clip(col_span / corr_px, 1, len(x_px))
    return np.mean(per_pixel) * n_eff


def test_horizon_ray_logL_jax_matches_numpy_on_hill():
    """Pixels straddling a hill horizon: some rays hit, some miss."""
    E, N, U = _sloped_dem(npnts=200, extent=100.0, slope_e=0.3,
                          slope_n=0.0, dtype=jnp.float32)
    U = U.T  # _sloped_dem is indexed [e, n]; DEM data is [n, e]
    Nu, Nv = 64, 64
    # looking east (ph=0), boresight tipped to the horizon (th=pi/2)
    prm = (0.0, 0.0, 5.0, np.pi / 2, 0.0, 0.0, 40.0)
    rng = np.random.default_rng(1)
    x_px = rng.integers(0, Nu, 50)
    y_px = rng.integers(0, Nv, 50)
    psky = rng.uniform(0.1, 0.9, 50).astype(np.float32)
    corr_px = 8.0
    expected = _numpy_horizon_logL(E, N, U, Nu, Nv, prm, x_px, y_px,
                                   psky, corr_px)
    got = horizon_ray_logL_jax(E, N, U, Nu, Nv, *prm,
                               jnp.asarray(x_px), jnp.asarray(y_px),
                               jnp.asarray(psky), eps=1e-6,
                               correlation_px=corr_px)
    np.testing.assert_allclose(float(got), expected, rtol=1e-4)


def test_ant_logL_jax_matches_numpy_and_has_finite_grad():
    Nu, Nv = 400, 600
    prm = (1.0, 2.0, 3.0, 1.4, 0.3, 0.02, 350.0)
    ant_uv = (123.0, 456.0)
    ant = np.array([40.0, 25.0, 2.0])
    box_size = 0.3
    ray = _numpy_rays(Nu, Nv, prm[-1], *prm[3:6], np.array(ant_uv))
    r_ant = ant - np.array(prm[:3])
    cos = np.dot(ray, r_ant) / np.linalg.norm(ray) / np.linalg.norm(r_ant)
    dth = np.arccos(cos.clip(-1, 1))
    sig = box_size / np.linalg.norm(r_ant)
    expected = np.log(1 / np.sqrt(2 * np.pi * sig**2)) - 0.5 * dth**2 / sig**2

    def f(ant_pos):
        return ant_logL_jax(Nu, Nv, *prm, *ant_pos, *ant_uv, box_size)

    np.testing.assert_allclose(float(f(jnp.asarray(ant))), expected,
                               rtol=1e-3)
    # antenna placed exactly on the pixel's ray: delta_theta = 0
    on_ray = jnp.asarray(np.array(prm[:3]) + 30.0 * ray)
    assert np.all(np.isfinite(np.asarray(jax.grad(f)(on_ray))))


def test_stack_problem_from_export():
    """stack_problem consumes PositionSolver.export_jax() output,
    including static images and images lacking an ant_px."""
    E, N, U = _flat_dem(npnts=64, extent=50.0)

    def img(prms, has_ant):
        return dict(npix_y=np.int32(16), npix_x=np.int32(16),
                    x_px=np.array([2, 6, 9], np.int32),
                    y_px=np.array([3, 7, 11], np.int32),
                    psky=np.array([0.2, 0.5, 0.8], np.float32),
                    ant_px=np.array([8.0, 8.0], np.float32),
                    has_ant=has_ant,
                    prms=np.array(prms, np.float32),
                    px_smooth=np.float32(4))

    fit_img = img([0, 0, 1, np.pi, 0, 0, 50], True)
    static_ant = img([5, 5, 1, np.pi / 2, 0, 0, 50], True)
    static_noant = img([0, 0, 0, 0, 0, 0, 0], False)
    export = dict(dem=dict(E=E, N=N, U=U), fit=[fit_img],
                  all=[fit_img, static_ant, static_noant],
                  box_size=np.float32(0.3))
    problem = stack_problem(export)
    assert problem['all']['is_fit'].tolist() == [True, False, False]
    assert problem['all']['has_ant'].tolist() == [True, True, False]

    theta = jnp.concatenate([jnp.asarray(fit_img['prms']),
                             jnp.array([20.0, 5.0, 1.0])])
    logL = logL_from_problem_jit(theta, problem, eps=1e-6)
    assert np.isfinite(float(logL))

    # the no-ant image contributes nothing; the others match per-term calls
    expected = horizon_ray_logL_jax(
        E, N, U, 16, 16, *fit_img['prms'], fit_img['x_px'],
        fit_img['y_px'], fit_img['psky'], eps=1e-6, correlation_px=4.0)
    for d in (fit_img, static_ant):
        expected += ant_logL_jax(16, 16, *d['prms'], 20.0, 5.0, 1.0,
                                 8.0, 8.0, 0.3)
    np.testing.assert_allclose(float(logL), float(expected), rtol=1e-5)
